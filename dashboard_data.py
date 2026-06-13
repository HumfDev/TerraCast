"""Dashboard data queries for TerraCast.

Loads KPIs, trends, and breakdowns from the three Unity Catalog tables that
power the live demo dashboard:

  - workspace.default.df_final_lightgbm       (historical anomaly, 2010-2023)
  - workspace.default.predictions_2024_partial (2024 prediction-only records)
  - workspace.default.stats_2024_pred_summary  (2024 summary statistics)

All reads go through delta_store.DeltaStore (SQL Warehouse via databricks-sdk).
Results are cached in-process for `_CACHE_TTL_SECONDS` to keep page loads snappy
and avoid hammering the warehouse.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from delta_store import get_store


DF_HISTORICAL = "workspace.default.df_final_lightgbm"
DF_2024 = "workspace.default.predictions_2024_partial"
DF_2024_STATS = "workspace.default.stats_2024_pred_summary"

# Documented model performance from training notebook (not in the table).
MODEL_CV_R2 = 0.4220

_CACHE_TTL_SECONDS = 600  # 10 min
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, fn: Callable[[], Any]) -> Any:
    now = time.time()
    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]
    value = fn()
    _cache[key] = (now + _CACHE_TTL_SECONDS, value)
    return value


def _query(sql: str) -> list[dict[str, Any]]:
    store = get_store()
    if not store.is_configured:
        raise RuntimeError(
            "Lakehouse is not configured. Set SQL_WAREHOUSE_ID env var."
        )
    return store._execute(sql)


def _f(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        x = float(v)
        return default if x != x else x
    except (TypeError, ValueError):
        return default


def _i(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _irrigation_to_int(v: Any) -> int | None:
    """Normalize the (mixed-type) irrigation column to 0/1/None.

    The Unity Catalog tables store this as STRING with messy values such as
    "0", "1", " Non-Irrigated", " Irrigated", etc. (note the leading spaces).
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s == "none" or s == "null":
        return None
    if s in ("1", "1.0", "irrigated", "yes", "y", "true", "t"):
        return 1
    if s in ("0", "0.0", "non-irrigated", "non irrigated", "no", "n", "false", "f"):
        return 0
    try:
        return 1 if int(float(s)) >= 1 else 0
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Historical (df_final_lightgbm) — full-row data
#
# Mirrors the Streamlit live-dashboard's `load_historical_data()`: fetches the
# full table once, lets the client filter (~6,823 rows). The outcome_category
# is computed server-side using the same logic as the Streamlit app.
# ---------------------------------------------------------------------------

_HISTORICAL_COLS = (
    "state_fips, county_fips, county_name, state_abbr, year, commodity, "
    "irrigation, yield_amount, pred_yield, signed_residual, residual_zscore, "
    "anomaly_flag, weather_anomaly_flag, avg_precip_mm, heat_stress_days, gdd"
)

_OUTCOME_CASE_SQL = (
    "CASE "
    "WHEN anomaly_flag = 0 THEN 'Normal' "
    "WHEN weather_anomaly_flag = 1 AND ABS(residual_zscore) < 2 THEN 'Extreme Weather Model Handled' "
    "WHEN weather_anomaly_flag = 0 AND ABS(residual_zscore) >= 2 THEN 'Yield Mismatch / Unexplained' "
    "ELSE 'Extreme Weather Yield Impact' "
    "END AS outcome_category"
)


def historical_full() -> dict[str, Any]:
    """Load the full historical anomaly table + filter options."""

    def _load() -> dict[str, Any]:
        rows = _query(f"""
            SELECT {_HISTORICAL_COLS}, {_OUTCOME_CASE_SQL}
            FROM {DF_HISTORICAL}
        """)
        cleaned: list[dict[str, Any]] = []
        for r in rows:
            cleaned.append({
                "state_fips": _i(r.get("state_fips")),
                "county_fips": _i(r.get("county_fips")),
                "county_name": (str(r.get("county_name") or "")).strip(),
                "state_abbr": str(r.get("state_abbr") or ""),
                "year": _i(r.get("year")),
                "commodity": str(r.get("commodity") or ""),
                "irrigation": _irrigation_to_int(r.get("irrigation")),
                "yield_amount": _f(r.get("yield_amount")) if r.get("yield_amount") is not None else None,
                "pred_yield": _f(r.get("pred_yield")) if r.get("pred_yield") is not None else None,
                "signed_residual": _f(r.get("signed_residual")) if r.get("signed_residual") is not None else None,
                "residual_zscore": _f(r.get("residual_zscore")) if r.get("residual_zscore") is not None else None,
                "anomaly_flag": _i(r.get("anomaly_flag")),
                "weather_anomaly_flag": _i(r.get("weather_anomaly_flag")),
                "outcome_category": str(r.get("outcome_category") or ""),
                "avg_precip_mm": _f(r.get("avg_precip_mm")) if r.get("avg_precip_mm") is not None else None,
                "heat_stress_days": _f(r.get("heat_stress_days")) if r.get("heat_stress_days") is not None else None,
                "gdd": _f(r.get("gdd")) if r.get("gdd") is not None else None,
            })

        years = sorted({r["year"] for r in cleaned if r["year"]}, reverse=True)
        commodities = sorted({r["commodity"] for r in cleaned if r["commodity"]})
        states = sorted({r["state_abbr"] for r in cleaned if r["state_abbr"]})
        outcome_categories = sorted({r["outcome_category"] for r in cleaned if r["outcome_category"]})

        return {
            "rows": cleaned,
            "filter_options": {
                "years": years,
                "commodities": commodities,
                "states": states,
                "outcome_categories": outcome_categories,
            },
            "model_r2": MODEL_CV_R2,
        }

    return _cached("hist_full", _load)


# ---------------------------------------------------------------------------
# 2024 predictions (predictions_2024_partial + stats_2024_pred_summary)
# ---------------------------------------------------------------------------

_PRED2024_COLS = (
    "state_fips, county_fips, county_name, state_abbr, year, commodity, "
    "irrigation, pred_yield, noaa_window_start, noaa_window_end, "
    "avg_precip_mm, heat_stress_days, gdd"
)

_SUMMARY_COLS = (
    "group_key, n_counties, n_rows, pred_mean, pred_median, pred_std, "
    "pred_min, pred_max, pred_p25, pred_p75, avg_precip_mm_mean, "
    "heat_stress_days_mean, gdd_mean, group_type, commodity, region, "
    "state_abbr, year, noaa_window_start, noaa_window_end"
)


def predictions_2024_full() -> dict[str, Any]:
    """Load the full 2024 prediction table + summary stats + filter options."""

    def _load() -> dict[str, Any]:
        partial_rows = _query(f"SELECT {_PRED2024_COLS} FROM {DF_2024}")
        cleaned: list[dict[str, Any]] = []
        for r in partial_rows:
            cleaned.append({
                "state_fips": _i(r.get("state_fips")),
                "county_fips": _i(r.get("county_fips")),
                "county_name": (str(r.get("county_name") or "")).strip(),
                "state_abbr": str(r.get("state_abbr") or ""),
                "year": _i(r.get("year")),
                "commodity": str(r.get("commodity") or ""),
                "irrigation": _irrigation_to_int(r.get("irrigation")),
                "pred_yield": _f(r.get("pred_yield")) if r.get("pred_yield") is not None else None,
                "avg_precip_mm": _f(r.get("avg_precip_mm")) if r.get("avg_precip_mm") is not None else None,
                "heat_stress_days": _f(r.get("heat_stress_days")) if r.get("heat_stress_days") is not None else None,
                "gdd": _f(r.get("gdd")) if r.get("gdd") is not None else None,
                "noaa_window_start": str(r.get("noaa_window_start") or ""),
                "noaa_window_end": str(r.get("noaa_window_end") or ""),
            })

        summary_raw = _query(f"SELECT {_SUMMARY_COLS} FROM {DF_2024_STATS}")
        summary_rows: list[dict[str, Any]] = []
        for r in summary_raw:
            summary_rows.append({
                "group_key": str(r.get("group_key") or ""),
                "group_type": str(r.get("group_type") or ""),
                "commodity": (str(r.get("commodity")) if r.get("commodity") is not None else None),
                "state_abbr": (str(r.get("state_abbr")) if r.get("state_abbr") is not None else None),
                "region": (str(r.get("region")) if r.get("region") is not None else None),
                "year": _i(r.get("year")) if r.get("year") is not None else None,
                "n_counties": _i(r.get("n_counties")),
                "n_rows": _i(r.get("n_rows")),
                "pred_mean": _f(r.get("pred_mean")) if r.get("pred_mean") is not None else None,
                "pred_median": _f(r.get("pred_median")) if r.get("pred_median") is not None else None,
                "pred_std": _f(r.get("pred_std")) if r.get("pred_std") is not None else None,
                "pred_min": _f(r.get("pred_min")) if r.get("pred_min") is not None else None,
                "pred_max": _f(r.get("pred_max")) if r.get("pred_max") is not None else None,
                "pred_p25": _f(r.get("pred_p25")) if r.get("pred_p25") is not None else None,
                "pred_p75": _f(r.get("pred_p75")) if r.get("pred_p75") is not None else None,
                "noaa_window_start": str(r.get("noaa_window_start") or ""),
                "noaa_window_end": str(r.get("noaa_window_end") or ""),
            })

        commodities = sorted({r["commodity"] for r in cleaned if r["commodity"]})
        states = sorted({r["state_abbr"] for r in cleaned if r["state_abbr"]})

        return {
            "rows": cleaned,
            "summary_rows": summary_rows,
            "filter_options": {
                "commodities": commodities,
                "states": states,
            },
        }

    return _cached("pred2024_full", _load)


def clear_cache() -> None:
    _cache.clear()


# ---------------------------------------------------------------------------
# Report (Executive Summary)
# ---------------------------------------------------------------------------
#
# Reuses cached `historical_full()` + `predictions_2024_full()` payloads and
# computes:
#   - hero KPIs (records, year range, states/counties/commodities, anomalies, R^2)
#   - auto-generated "headline insights" sentences (worst year / drought-driven /
#     anomaly hotspot state / 2024 NOAA window)
#   - top-10 historical anomalies sorted by |signed_residual|
#   - 2024 lowest predicted-yield counties (worst-hit prediction-only regions)
#
# All math is done in Python over already-loaded rows — no extra SQL.
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def report_summary() -> dict[str, Any]:
    """Build the Executive Summary report payload."""

    def _load() -> dict[str, Any]:
        hist = historical_full()
        pred = predictions_2024_full()
        hist_rows: list[dict[str, Any]] = hist.get("rows", [])
        pred_rows: list[dict[str, Any]] = pred.get("rows", [])

        # ---------------- Hero KPIs ----------------
        years = [r["year"] for r in hist_rows if r.get("year")]
        states_set = {r["state_abbr"] for r in hist_rows if r.get("state_abbr")}
        commodities_set = {r["commodity"] for r in hist_rows if r.get("commodity")}
        counties_set = {(r["state_abbr"], r["county_fips"]) for r in hist_rows if r.get("state_abbr")}
        anomaly_count = sum(1 for r in hist_rows if (r.get("anomaly_flag") or 0) == 1)
        weather_anomaly_count = sum(1 for r in hist_rows if (r.get("weather_anomaly_flag") or 0) == 1)

        hero = {
            "total_records": len(hist_rows),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "n_states": len(states_set),
            "n_counties": len(counties_set),
            "n_commodities": len(commodities_set),
            "anomaly_count": anomaly_count,
            "anomaly_rate": _safe_div(anomaly_count, len(hist_rows)),
            "weather_anomaly_count": weather_anomaly_count,
            "model_r2": MODEL_CV_R2,
            "n_pred_2024_rows": len(pred_rows),
        }

        # ---------------- Headline insights ----------------
        insights: list[str] = []

        # 1) Worst yield-shock year by mean(signed_residual) where actual is below pred
        by_year: dict[int, list[float]] = {}
        for r in hist_rows:
            sr = r.get("signed_residual")
            yr = r.get("year")
            if sr is None or not yr:
                continue
            by_year.setdefault(int(yr), []).append(float(sr))
        if by_year:
            year_means = {y: (sum(v) / len(v)) for y, v in by_year.items()}
            worst_year = min(year_means, key=year_means.get)
            best_year = max(year_means, key=year_means.get)
            worst_avg = year_means[worst_year]
            best_avg = year_means[best_year]
            insights.append(
                f"{worst_year} was the largest yield-shock year — actuals fell {abs(worst_avg):.1f} "
                f"bu/ac below the model on average across {len(by_year[worst_year])} county-records."
            )
            insights.append(
                f"{best_year} was the strongest yield year — actuals exceeded the model by "
                f"{best_avg:+.1f} bu/ac on average."
            )

        # 2) State with most anomalies
        state_anom: dict[str, int] = {}
        for r in hist_rows:
            if (r.get("anomaly_flag") or 0) == 1 and r.get("state_abbr"):
                state_anom[r["state_abbr"]] = state_anom.get(r["state_abbr"], 0) + 1
        if state_anom:
            top_state, top_n = max(state_anom.items(), key=lambda kv: kv[1])
            insights.append(
                f"{top_state} recorded the most yield anomalies historically "
                f"({top_n} county-year events flagged)."
            )

        # 3) Anomaly rate
        if hero["total_records"] > 0:
            insights.append(
                f"{anomaly_count:,} of {hero['total_records']:,} county-year records were flagged "
                f"as anomalies ({hero['anomaly_rate'] * 100:.1f}% rate). The model "
                f"(LightGBM, R² = {MODEL_CV_R2:.4f}) explains the bulk of normal seasons."
            )

        # 4) 2024 NOAA window
        noaa_starts = sorted({r["noaa_window_start"] for r in pred_rows if r.get("noaa_window_start")})
        noaa_ends = sorted({r["noaa_window_end"] for r in pred_rows if r.get("noaa_window_end")})
        if noaa_starts and noaa_ends:
            insights.append(
                f"2024 predictions cover {len(pred_rows):,} county-records using NOAA weather "
                f"from {noaa_starts[0]} to {noaa_ends[-1]} (prediction-only — no observed yield)."
            )

        # ---------------- Top-10 historical anomalies (|signed_residual|) ----------------
        rated = [
            r for r in hist_rows
            if r.get("signed_residual") is not None and (r.get("anomaly_flag") or 0) == 1
        ]
        rated.sort(key=lambda r: abs(float(r["signed_residual"])), reverse=True)
        top_anoms: list[dict[str, Any]] = []
        for r in rated[:10]:
            top_anoms.append({
                "year": r.get("year"),
                "state_abbr": r.get("state_abbr"),
                "county_name": r.get("county_name"),
                "commodity": r.get("commodity"),
                "yield_amount": r.get("yield_amount"),
                "pred_yield": r.get("pred_yield"),
                "signed_residual": r.get("signed_residual"),
                "residual_zscore": r.get("residual_zscore"),
                "outcome_category": r.get("outcome_category"),
                "weather_anomaly_flag": r.get("weather_anomaly_flag"),
                "avg_precip_mm": r.get("avg_precip_mm"),
                "heat_stress_days": r.get("heat_stress_days"),
                "gdd": r.get("gdd"),
            })

        # ---------------- Top-10 lowest 2024 predicted yields ----------------
        pred_sorted = [r for r in pred_rows if r.get("pred_yield") is not None]
        pred_sorted.sort(key=lambda r: float(r["pred_yield"]))
        worst_2024: list[dict[str, Any]] = []
        for r in pred_sorted[:10]:
            worst_2024.append({
                "year": r.get("year"),
                "state_abbr": r.get("state_abbr"),
                "county_name": r.get("county_name"),
                "commodity": r.get("commodity"),
                "pred_yield": r.get("pred_yield"),
                "irrigation": r.get("irrigation"),
                "avg_precip_mm": r.get("avg_precip_mm"),
                "heat_stress_days": r.get("heat_stress_days"),
                "gdd": r.get("gdd"),
            })

        # ---------------- By-state anomaly bar (top 12) ----------------
        state_anom_sorted = sorted(state_anom.items(), key=lambda kv: kv[1], reverse=True)[:12]
        anomaly_by_state = [{"state_abbr": s, "anomalies": n} for s, n in state_anom_sorted]

        # ---------------- Anomaly trend by year ----------------
        year_anom: dict[int, dict[str, int]] = {}
        for r in hist_rows:
            yr = r.get("year")
            if not yr:
                continue
            d = year_anom.setdefault(int(yr), {"total": 0, "anomalies": 0})
            d["total"] += 1
            d["anomalies"] += int(r.get("anomaly_flag") or 0)
        anomaly_trend = [
            {"year": y, "total": d["total"], "anomalies": d["anomalies"]}
            for y, d in sorted(year_anom.items())
        ]

        return {
            "hero": hero,
            "insights": insights,
            "top_anomalies": top_anoms,
            "worst_2024_predicted": worst_2024,
            "anomaly_by_state": anomaly_by_state,
            "anomaly_trend": anomaly_trend,
            "noaa_window": {
                "start": noaa_starts[0] if noaa_starts else None,
                "end": noaa_ends[-1] if noaa_ends else None,
            },
        }

    return _cached("report_summary", _load)


# ---------------------------------------------------------------------------
# Predictor (lookup-based, see PREDICTOR_DESIGN below)
# ---------------------------------------------------------------------------
#
# PREDICTOR_DESIGN
# -----------------
# This is *lookup-based prediction* — we filter the same Unity Catalog tables
# that power the Dashboard and aggregate their existing `pred_yield` column.
# The numeric weather inputs (avg_precip_mm, heat_stress_days, gdd, flags) are
# accepted by the API but ARE NOT used as filter criteria — instead they are
# returned alongside results so the UI can show "what weather did matching
# counties experience?" context.
#
# To switch to real LightGBM inference later, implement `predict_yield_real()`
# below — it has the same input shape as the lookup filters and is wired to
# return whatever `predict_yield(...)` from app.py would produce.

PREDICTION_ONLY_YEAR = 2024  # year >= this uses predictions_2024_partial


def _filter_helper(
    year: int,
    commodity: str,
    state: str | None,
    county: str | None,
    irrigation: int | None,
) -> tuple[str, str]:
    """Build (table, where_clause) for predictor lookup. `state` may be None
    (meaning "all states") — county is also ignored in that case."""
    is_2024 = year >= PREDICTION_ONLY_YEAR
    table = DF_2024 if is_2024 else DF_HISTORICAL

    where: list[str] = [f"year = {int(year)}", "pred_yield IS NOT NULL"]
    safe_commodity = str(commodity).replace("'", "''")
    where.append(f"UPPER(commodity) = UPPER('{safe_commodity}')")
    if state:
        safe_state = str(state).replace("'", "''")
        where.append(f"state_abbr = '{safe_state}'")
    if state and county:
        safe_county = str(county).strip().replace("'", "''")
        where.append(f"TRIM(county_name) = '{safe_county}'")
    if irrigation is not None and irrigation in (0, 1):
        # `irrigation` column is STRING with mixed values
        # ("0"/"1"/" Non-Irrigated"/" Irrigated" etc.). Normalize before compare.
        if irrigation == 1:
            where.append(
                "LOWER(TRIM(CAST(irrigation AS STRING))) IN "
                "('1','1.0','irrigated','yes','y','true','t')"
            )
        else:
            where.append(
                "LOWER(TRIM(CAST(irrigation AS STRING))) IN "
                "('0','0.0','non-irrigated','non irrigated','no','n','false','f')"
            )

    return table, " AND ".join(where)


def predictor_options() -> dict[str, Any]:
    """Dropdown options for the Predictor form (years, commodities, states, counties)."""

    def _load() -> dict[str, Any]:
        years = _query(f"""
            SELECT year FROM (
                SELECT DISTINCT year FROM {DF_HISTORICAL} WHERE year IS NOT NULL
                UNION
                SELECT DISTINCT year FROM {DF_2024} WHERE year IS NOT NULL
            )
            ORDER BY year
        """)
        commodities = _query(f"""
            SELECT commodity FROM (
                SELECT DISTINCT commodity FROM {DF_HISTORICAL} WHERE commodity IS NOT NULL
                UNION
                SELECT DISTINCT commodity FROM {DF_2024} WHERE commodity IS NOT NULL
            )
            ORDER BY commodity
        """)
        states_counties = _query(f"""
            SELECT DISTINCT state_abbr, county_name FROM (
                SELECT state_abbr, TRIM(county_name) AS county_name FROM {DF_HISTORICAL}
                UNION
                SELECT state_abbr, TRIM(county_name) AS county_name FROM {DF_2024}
            )
            WHERE state_abbr IS NOT NULL AND county_name IS NOT NULL AND county_name <> ''
            ORDER BY state_abbr, county_name
        """)

        counties_by_state: dict[str, list[str]] = {}
        for r in states_counties:
            st = str(r.get("state_abbr") or "")
            cy = str(r.get("county_name") or "")
            if not st or not cy:
                continue
            counties_by_state.setdefault(st, []).append(cy)

        return {
            "years": [_i(r.get("year")) for r in years if r.get("year") is not None],
            "commodities": [str(r.get("commodity") or "") for r in commodities],
            "states": sorted(counties_by_state.keys()),
            "counties_by_state": counties_by_state,
            "prediction_only_year": PREDICTION_ONLY_YEAR,
        }

    return _cached("predictor_options", _load)


def _aggregate_pred_yield(values: list[float]) -> dict[str, Any]:
    """Compute basic distribution stats over a list of pred_yield numbers."""
    n = len(values)
    if n == 0:
        return {
            "count": 0, "mean": None, "median": None,
            "min": None, "max": None, "std": None,
            "p25": None, "p75": None,
        }
    sv = sorted(values)

    def _q(p: float) -> float:
        if n == 1:
            return sv[0]
        idx = (n - 1) * p
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return sv[lo] + (sv[hi] - sv[lo]) * frac

    mean = sum(sv) / n
    var = sum((x - mean) ** 2 for x in sv) / n if n > 1 else 0.0
    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(_q(0.5), 2),
        "min": round(sv[0], 2),
        "max": round(sv[-1], 2),
        "std": round(var ** 0.5, 2),
        "p25": round(_q(0.25), 2),
        "p75": round(_q(0.75), 2),
    }


def predictor_lookup(
    year: int,
    commodity: str,
    state: str | None = None,
    county: str | None = None,
    irrigation: int | None = None,
    limit: int = 10000,
) -> dict[str, Any]:
    """Look up matching prediction records from the historical or 2024 table.

    Returns aggregated stats, per-county breakdown, weather context, and the
    raw distribution of pred_yield values for histogram rendering.
    """
    table, where = _filter_helper(year, commodity, state, county, irrigation)
    is_2024 = table == DF_2024

    select_extra = (
        ", noaa_window_start, noaa_window_end, NULL AS yield_amount, "
        "NULL AS signed_residual, NULL AS residual_zscore, NULL AS anomaly_flag"
        if is_2024
        else (
            ", CAST(NULL AS STRING) AS noaa_window_start, "
            "CAST(NULL AS STRING) AS noaa_window_end, "
            "yield_amount, signed_residual, residual_zscore, anomaly_flag"
        )
    )

    rows = _query(f"""
        SELECT state_abbr,
               TRIM(county_name) AS county_name,
               year, commodity, irrigation, pred_yield,
               avg_precip_mm, heat_stress_days, gdd,
               drought_flag, flood_flag, extreme_heat_flag, extreme_events
               {select_extra}
        FROM {table}
        WHERE {where}
        ORDER BY pred_yield DESC
        LIMIT {int(limit)}
    """)

    pred_values = [_f(r.get("pred_yield")) for r in rows if r.get("pred_yield") is not None]
    stats = _aggregate_pred_yield(pred_values)

    by_county_map: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        cy = str(r.get("county_name") or "")
        st = str(r.get("state_abbr") or "")
        if not cy:
            continue
        key = (st, cy)
        bucket = by_county_map.setdefault(key, {"county": cy, "state": st, "values": []})
        bucket["values"].append(_f(r.get("pred_yield")))

    by_county = [
        {
            "county": v["county"],
            "state": v["state"],
            "count": len(v["values"]),
            "avg_pred_yield": round(sum(v["values"]) / len(v["values"]), 2),
            "min_pred_yield": round(min(v["values"]), 2),
            "max_pred_yield": round(max(v["values"]), 2),
        }
        for v in by_county_map.values()
    ]
    by_county.sort(key=lambda x: x["avg_pred_yield"], reverse=True)

    def _avg(field: str) -> float | None:
        nums = [_f(r.get(field)) for r in rows if r.get(field) is not None]
        return round(sum(nums) / len(nums), 2) if nums else None

    weather_summary = {
        "avg_precip_mm": _avg("avg_precip_mm"),
        "heat_stress_days": _avg("heat_stress_days"),
        "gdd": _avg("gdd"),
        "drought_rate": _avg("drought_flag"),
        "flood_rate": _avg("flood_flag"),
        "extreme_heat_rate": _avg("extreme_heat_flag"),
    }

    noaa_window = None
    if is_2024 and rows:
        starts = [r.get("noaa_window_start") for r in rows if r.get("noaa_window_start")]
        ends = [r.get("noaa_window_end") for r in rows if r.get("noaa_window_end")]
        if starts and ends:
            noaa_window = {"start": str(min(starts)), "end": str(max(ends))}

    sample_rows = [
        {
            "county": str(r.get("county_name") or ""),
            "state": str(r.get("state_abbr") or ""),
            "year": _i(r.get("year")),
            "commodity": str(r.get("commodity") or ""),
            "irrigation": _irrigation_to_int(r.get("irrigation")),
            "pred_yield": _f(r.get("pred_yield")),
            "yield_amount": _f(r.get("yield_amount")) if r.get("yield_amount") is not None else None,
            "signed_residual": _f(r.get("signed_residual")) if r.get("signed_residual") is not None else None,
            "residual_zscore": _f(r.get("residual_zscore")) if r.get("residual_zscore") is not None else None,
            "anomaly_flag": _i(r.get("anomaly_flag")) if r.get("anomaly_flag") is not None else None,
        }
        for r in rows[:50]
    ]

    return {
        "mode": "lookup",
        "source_table": table,
        "is_prediction_only": is_2024,
        "filters": {
            "year": int(year),
            "commodity": str(commodity),
            "state": (str(state) if state else None),
            "county": county,
            "irrigation": irrigation,
        },
        "stats": stats,
        "distribution": [round(v, 2) for v in pred_values],
        "by_county": by_county,
        "weather_summary": weather_summary,
        "noaa_window": noaa_window,
        "sample_rows": sample_rows,
    }


def predict_yield_real(features: dict[str, Any]) -> dict[str, Any]:
    """Hook for future real LightGBM inference.

    The TerraCast app already exposes `POST /predict` which runs the loaded
    regional LightGBM models against a `YieldInput`. To wire the Predictor
    page to real inference, call this function from the API endpoint —
    today it raises NotImplementedError so the UI can fall back to lookup.
    """
    raise NotImplementedError(
        "Real-model inference is not wired into the Predictor page yet. "
        "Use POST /predict directly, or implement this helper to bridge."
    )
