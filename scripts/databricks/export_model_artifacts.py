# Databricks notebook source
# MAGIC %md
# MAGIC # Export trend + baseline JSON for TerraCast App
# MAGIC Run on a cluster with `workspace.default.merged` available.
# MAGIC Copy output files to `MODEL_VOLUME_PATH` (e.g. `/Volumes/workspace/terract/models/`).

# COMMAND ----------

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P

TABLE = "workspace.default.merged"
OUT_DIR = "/Volumes/workspace/terract/models"  # adjust

# COMMAND ----------

pdf = spark.table(TABLE).toPandas()
pdf.columns = [c.strip() for c in pdf.columns]
pdf["commodity"] = pdf["commodity"].str.strip()
pdf["state_abbr"] = pdf["state_abbr"].str.strip()

# COMMAND ----------

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


corn_trend = build_trends(pdf, "Corn")
soy_trend = build_trends(pdf, "Soybeans")
corn_bl = build_baselines(pdf, "Corn")
soy_bl = build_baselines(pdf, "Soybeans")

# COMMAND ----------

out = Path(OUT_DIR)
out.mkdir(parents=True, exist_ok=True)
(out / "corn_trend.json").write_text(json.dumps(corn_trend, indent=2))
(out / "soybean_trend.json").write_text(json.dumps(soy_trend, indent=2))
(out / "corn_baselines.json").write_text(json.dumps(corn_bl, indent=2))
(out / "soybean_baselines.json").write_text(json.dumps(soy_bl, indent=2))
print("Wrote JSON artifacts to", OUT_DIR)
