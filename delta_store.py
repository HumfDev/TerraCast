"""
Unity Catalog Delta access for TerraCast via Databricks SQL Warehouse.

No PySpark on the request path — all reads/writes use the SQL Statement API.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional


def _cached_token(host: str) -> Optional[str]:
    try:
        cache = Path.home() / ".databricks" / "token-cache.json"
        data = json.loads(cache.read_text())
        return data.get("tokens", {}).get(host, {}).get("access_token")
    except Exception:
        return None

_TABLE_RE = re.compile(r"^[a-zA-Z0-9_.]+$")


@dataclass(frozen=True)
class LakehouseConfig:
    warehouse_id: str
    merged_table: str
    predictions_table: Optional[str]
    enabled: bool

    @classmethod
    def from_env(cls) -> "LakehouseConfig":
        warehouse_id = (os.environ.get("SQL_WAREHOUSE_ID") or "").strip()
        merged = (os.environ.get("DELTA_TABLE_MERGED") or "workspace.default.merged").strip()
        predictions = (os.environ.get("DELTA_TABLE_PREDICTIONS") or "").strip() or None
        enabled = os.environ.get("LAKEHOUSE_ENABLED", "true").lower() not in ("0", "false", "no")
        return cls(
            warehouse_id=warehouse_id,
            merged_table=merged,
            predictions_table=predictions,
            enabled=enabled and bool(warehouse_id),
        )


def _validate_table(name: str) -> str:
    if not _TABLE_RE.match(name):
        raise ValueError(f"Invalid table identifier: {name!r}")
    return name


def _irrigation_label(is_irrigated: int) -> str:
    return "Irrigated" if is_irrigated else "Non-Irrigated"


def _commodity_label(crop_type: str) -> str:
    c = crop_type.strip().lower()
    if c == "corn":
        return "Corn"
    if c in ("soybeans", "soybean", "soy"):
        return "Soybeans"
    raise ValueError(f"Unsupported crop_type: {crop_type!r}")


class DeltaStore:
    """Thin SQL client for merged yield+weather and predictions log tables."""

    def __init__(self, config: Optional[LakehouseConfig] = None):
        self.config = config or LakehouseConfig.from_env()
        self._client = None

    @property
    def is_configured(self) -> bool:
        return self.config.enabled

    def _workspace(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient
            host = os.environ.get("DATABRICKS_HOST", "")
            token = _cached_token(host)
            if token:
                self._client = WorkspaceClient(host=host, token=token)
            else:
                self._client = WorkspaceClient(host=host, auth_type="external-browser")
        return self._client

    def _execute(self, sql: str, wait_timeout: str = "50s") -> list[dict[str, Any]]:
        if not self.config.enabled:
            raise RuntimeError(
                "Lakehouse is not configured. Set SQL_WAREHOUSE_ID and DELTA_TABLE_MERGED."
            )

        w = self._workspace()
        resp = w.statement_execution.execute_statement(
            warehouse_id=self.config.warehouse_id,
            statement=sql,
            wait_timeout=wait_timeout,
        )

        statement_id = resp.statement_id
        status = resp.status
        deadline = time.time() + 55
        while True:
            s = w.statement_execution.get_statement(statement_id)
            state = s.status.state.value if hasattr(s.status.state, "value") else str(s.status.state)
            if state not in ("PENDING", "RUNNING"):
                break
            if time.time() > deadline:
                raise TimeoutError("Databricks SQL statement timed out")
            time.sleep(0.35)

        if state != "SUCCEEDED":
            raise RuntimeError(f"SQL failed [{state}]: {s.status}")

        if not s.result or not s.result.data_array:
            return []

        columns = [c.name for c in s.manifest.schema.columns]
        rows = []
        for raw in s.result.data_array:
            row = {col: raw[i] if i < len(raw) else None for i, col in enumerate(columns)}
            rows.append(row)
        return rows

    def ping_merged(self) -> bool:
        if not self.config.enabled:
            return False
        table = _validate_table(self.config.merged_table)
        try:
            self._execute(f"SELECT 1 AS ok FROM {table} LIMIT 1", wait_timeout="30s")
            return True
        except Exception:
            return False

    def fetch_features(
        self,
        state_abbr: str,
        county_fips: int,
        year: int,
        crop_type: str,
        is_irrigated: int,
    ) -> Optional[dict[str, Any]]:
        table = _validate_table(self.config.merged_table)
        commodity = _commodity_label(crop_type)
        irrigation = _irrigation_label(is_irrigated)
        st = state_abbr.strip().upper().replace("'", "")

        if county_fips and county_fips > 0:
            sql = f"""
            SELECT
              state_fips, county_fips, state_abbr, year, commodity, irrigation,
              avg_temp_max_c, avg_temp_min_c, avg_temp_mean_c, peak_temp_max_c,
              heat_stress_days, avg_precip_mm, precip_std_mm, dry_days, heavy_rain_days,
              gdd, frost_days, winter_snowfall_mm, snow_days, yield_amount
            FROM {table}
            WHERE state_abbr = '{st}'
              AND county_fips = {int(county_fips)}
              AND year = {int(year)}
              AND commodity = '{commodity}'
              AND TRIM(irrigation) = '{irrigation}'
            LIMIT 1
            """
        else:
            sql = f"""
            SELECT
              MAX(state_fips) AS state_fips, 0 AS county_fips,
              '{st}' AS state_abbr, {int(year)} AS year,
              '{commodity}' AS commodity, '{irrigation}' AS irrigation,
              AVG(avg_temp_max_c) AS avg_temp_max_c,
              AVG(avg_temp_min_c) AS avg_temp_min_c,
              AVG(avg_temp_mean_c) AS avg_temp_mean_c,
              AVG(peak_temp_max_c) AS peak_temp_max_c,
              AVG(heat_stress_days) AS heat_stress_days,
              AVG(avg_precip_mm) AS avg_precip_mm,
              AVG(precip_std_mm) AS precip_std_mm,
              AVG(dry_days) AS dry_days,
              AVG(heavy_rain_days) AS heavy_rain_days,
              AVG(gdd) AS gdd,
              AVG(frost_days) AS frost_days,
              AVG(winter_snowfall_mm) AS winter_snowfall_mm,
              AVG(snow_days) AS snow_days,
              AVG(yield_amount) AS yield_amount
            FROM {table}
            WHERE state_abbr = '{st}' AND year = {int(year)}
              AND commodity = '{commodity}' AND TRIM(irrigation) = '{irrigation}'
            """

        rows = self._execute(sql)
        if not rows:
            return None
        return _row_to_features(rows[0], is_irrigated)

    def log_prediction(
        self,
        inp: dict[str, Any],
        result: dict[str, Any],
        model_version: str = "1",
        region: str = "",
    ) -> bool:
        if not self.config.predictions_table:
            return False
        table = _validate_table(self.config.predictions_table)
        payload = json.dumps(inp, default=str).replace("'", "''")
        sql = f"""
        INSERT INTO {table} (
          predicted_at, crop_type, state_abbr, state_fips, county_fips, year,
          is_irrigated, predicted_yield, yield_category, model_version, input_json, region
        ) VALUES (
          current_timestamp(),
          '{str(result.get("crop_type", "")).replace("'", "''")}',
          '{str(inp.get("state_abbr", "")).replace("'", "''")}',
          {int(inp.get("state_fips") or 0)},
          {int(inp.get("county_fips") or 0)},
          {int(inp.get("year") or 0)},
          {int(inp.get("is_irrigated") or 0)},
          {float(result.get("predicted_yield") or 0)},
          '{str(result.get("yield_category", "")).replace("'", "''")}',
          '{model_version.replace("'", "''")}',
          '{payload}',
          '{region.replace("'", "''")}'
        )
        """
        try:
            self._execute(sql)
            return True
        except Exception:
            return False


def _as_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _as_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _row_to_features(row: dict[str, Any], is_irrigated: int) -> dict[str, Any]:
    actual = row.get("yield_amount")
    return {
        "state_fips": _as_int(row.get("state_fips")),
        "county_fips": _as_int(row.get("county_fips")),
        "state_abbr": str(row.get("state_abbr") or "").strip(),
        "year": _as_int(row.get("year")),
        "crop_type": str(row.get("commodity") or "Corn").strip(),
        "is_irrigated": is_irrigated,
        "avg_temp_max_c": _as_float(row.get("avg_temp_max_c")),
        "avg_temp_min_c": _as_float(row.get("avg_temp_min_c")),
        "avg_temp_mean_c": _as_float(row.get("avg_temp_mean_c")),
        "peak_temp_max_c": _as_float(row.get("peak_temp_max_c")),
        "heat_stress_days": _as_float(row.get("heat_stress_days")),
        "avg_precip_mm": _as_float(row.get("avg_precip_mm")),
        "precip_std_mm": _as_float(row.get("precip_std_mm")),
        "dry_days": _as_float(row.get("dry_days")),
        "heavy_rain_days": _as_float(row.get("heavy_rain_days")),
        "gdd": _as_float(row.get("gdd")),
        "frost_days": _as_float(row.get("frost_days")),
        "winter_snowfall_mm": _as_float(row.get("winter_snowfall_mm")),
        "snow_days": _as_float(row.get("snow_days")),
        "actual_yield": _as_float(actual) if actual is not None else None,
        "source": "lakehouse",
    }


_store: Optional[DeltaStore] = None


def get_store() -> DeltaStore:
    global _store
    if _store is None:
        _store = DeltaStore()
    return _store
