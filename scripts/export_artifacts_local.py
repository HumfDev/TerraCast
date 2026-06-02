"""
Run locally to export trend + baseline JSON artifacts into models/.
Requires DATABRICKS_HOST and SQL_WAREHOUSE_ID in .env (or env vars).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.polynomial import polynomial as P

load_dotenv()

WAREHOUSE_ID = os.environ.get("SQL_WAREHOUSE_ID", "").strip()
TABLE        = os.environ.get("DELTA_TABLE_MERGED", "workspace.default.merged_yield_weather").strip()
HOST         = os.environ.get("DATABRICKS_HOST", "").strip()
OUT_DIR      = Path(__file__).parent.parent / "models"


def get_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient(host=HOST, auth_type="external-browser")


def run_sql(w, sql: str) -> list[dict]:
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="50s",
    )
    sid = resp.statement_id
    deadline = time.time() + 120
    while True:
        s = w.statement_execution.get_statement(sid)
        state = s.status.state.value if hasattr(s.status.state, "value") else str(s.status.state)
        if state not in ("PENDING", "RUNNING"):
            break
        if time.time() > deadline:
            raise TimeoutError("SQL timed out")
        time.sleep(0.5)

    if state != "SUCCEEDED":
        raise RuntimeError(f"SQL failed [{state}]: {s.status}")

    if not s.result or not s.result.data_array:
        return []
    cols = [c.name for c in s.manifest.schema.columns]
    return [{cols[i]: row[i] for i in range(len(cols))} for row in s.result.data_array]


def build_trends(df: pd.DataFrame, commodity: str) -> dict:
    sub = df[df["commodity"] == commodity]
    trends = {}
    for state, grp in sub.groupby("state_abbr"):
        g = grp.groupby("year")["yield_amount"].mean().dropna()
        if len(g) < 3:
            continue
        coeffs = P.polyfit(g.index.astype(float), g.values.astype(float), 1)
        trends[state] = [float(coeffs[0]), float(coeffs[1])]
    return trends


def build_baselines(df: pd.DataFrame, commodity: str) -> dict:
    sub = df[df["commodity"] == commodity]
    state_bl = sub.groupby("state_abbr")["yield_amount"].median().to_dict()
    county_bl = {}
    for _, row in sub.groupby(["state_fips", "county_fips"])["yield_amount"].median().reset_index().iterrows():
        key = f"{int(row['state_fips'])}-{int(row['county_fips'])}"
        county_bl[key] = float(row["yield_amount"])
    return {"state": {k: float(v) for k, v in state_bl.items()}, "county": county_bl}


def main():
    if not WAREHOUSE_ID:
        sys.exit("SQL_WAREHOUSE_ID not set in .env")
    if not HOST:
        sys.exit("DATABRICKS_HOST not set in .env")

    print(f"Connecting to {HOST} ...")
    w = get_client()

    print(f"Fetching {TABLE} ...")
    sql = f"""
    SELECT state_fips, county_fips, state_abbr, year, commodity, yield_amount
    FROM {TABLE}
    WHERE TRIM(commodity) IN ('Corn', 'Soybeans') AND yield_amount IS NOT NULL
    """
    rows = run_sql(w, sql)
    if not rows:
        sys.exit(f"No rows returned from {TABLE}")

    df = pd.DataFrame(rows)
    df["state_abbr"]   = df["state_abbr"].str.strip()
    df["commodity"]    = df["commodity"].str.strip()
    df["year"]         = pd.to_numeric(df["year"])
    df["yield_amount"] = pd.to_numeric(df["yield_amount"])
    df["state_fips"]   = pd.to_numeric(df["state_fips"])
    df["county_fips"]  = pd.to_numeric(df["county_fips"])
    print(f"  {len(df)} rows loaded")

    print("Computing trends and baselines ...")
    corn_trend = build_trends(df, "Corn")
    soy_trend  = build_trends(df, "Soybeans")
    corn_bl    = build_baselines(df, "Corn")
    soy_bl     = build_baselines(df, "Soybeans")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "corn_trend.json").write_text(json.dumps(corn_trend, indent=2))
    (OUT_DIR / "soybean_trend.json").write_text(json.dumps(soy_trend, indent=2))
    (OUT_DIR / "corn_baselines.json").write_text(json.dumps(corn_bl, indent=2))
    (OUT_DIR / "soybean_baselines.json").write_text(json.dumps(soy_bl, indent=2))
    print(f"Wrote 4 JSON artifacts to {OUT_DIR}")
    print(f"  corn_trend:      {len(corn_trend)} states")
    print(f"  soybean_trend:   {len(soy_trend)} states")


if __name__ == "__main__":
    main()
