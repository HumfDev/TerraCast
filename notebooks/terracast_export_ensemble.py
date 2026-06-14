# Databricks notebook source
# MAGIC %md
# MAGIC # TerraCast — Export ensemble predictions to Delta
# MAGIC
# MAGIC Runs the LightGBM notebook (LightGBM 60% + RF 40% per `(crop, region)`)
# MAGIC and writes the test-set predictions + summary stats to two Delta tables
# MAGIC consumed by the TerraCast app's Model page:
# MAGIC
# MAGIC * `workspace.default.df_ensemble_test_2021_2023`
# MAGIC * `workspace.default.model_ensemble_stats`

# COMMAND ----------

# MAGIC %run "/Workspace/Users/dlee23@uw.edu/databricks_hackathon/LightGBM"

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Re-affirm ensemble weights (already set by the LightGBM notebook).
w_lgbm, w_rf = (0.6, 0.4)

# ── Build per-row test predictions ────────────────────────────────────
records = []
for (crop, region), (lgbm_m, rf_m) in models.items():
    X_te, y_te = test_Xy[(crop, region)]
    pred = w_lgbm * lgbm_m.predict(X_te) + w_rf * rf_m.predict(X_te)
    dc_test = (
        df[(df["commodity"] == crop) & (df["region"] == region) & df["is_test"]]
        .reset_index(drop=True)
    )
    for i in range(len(dc_test)):
        actual = float(y_te.values[i])
        p = float(pred[i])
        records.append({
            "state_abbr":      str(dc_test.loc[i, "state_abbr"]),
            "county_name":     str(dc_test.loc[i, "county_name"]),
            "year":            int(dc_test.loc[i, "year"]),
            "commodity":       str(crop),
            "region":          str(region),
            "yield_amount":    actual,
            "pred_yield":      p,
            "signed_residual": actual - p,
            "abs_residual":    abs(actual - p),
        })

out = pd.DataFrame(records)
print(f"Built {len(out):,} test-row predictions across "
      f"{out[['commodity', 'region']].drop_duplicates().shape[0]} crop×region groups")

# ── Compute test metrics (weighted per-crop R² to avoid scale inflation) ────
crop_r2 = {}
crop_n  = {}
for crop in ["Corn", "Soybeans"]:
    m = out["commodity"] == crop
    if m.sum() > 0:
        crop_r2[crop] = float(r2_score(out.loc[m, "yield_amount"], out.loc[m, "pred_yield"]))
        crop_n[crop]  = int(m.sum())
total_n   = sum(crop_n.values())
test_r2   = sum(crop_r2[c] * crop_n[c] / total_n for c in crop_r2)
test_mae  = float(mean_absolute_error(out["yield_amount"], out["pred_yield"]))
test_rmse = float(np.sqrt(mean_squared_error(out["yield_amount"], out["pred_yield"])))
n_train   = int(sum(len(v[1]) for v in train_Xy.values()))
print(f"Test R²={test_r2:.4f}  MAE={test_mae:.2f}  RMSE={test_rmse:.2f}  "
      f"Train={n_train}  Test={len(out)}")

# ── Write predictions table ────────────────────────────────────────────
TARGET_PRED = "workspace.default.df_ensemble_test_2021_2023"
(spark.createDataFrame(out)
      .write.format("delta")
      .mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(TARGET_PRED))
print(f"✅ Wrote {len(out):,} rows to {TARGET_PRED}")

# ── Write stats sidecar ────────────────────────────────────────────────
stats = [{
    "model_name":        "LightGBM + Random Forest (ensemble)",
    "ensemble_weights":  "LightGBM 60% + RF 40%",
    "train_rows":        n_train,
    "test_rows":         int(len(out)),
    "test_year_min":     int(out["year"].min()),
    "test_year_max":     int(out["year"].max()),
    "test_r2":           test_r2,
    "test_mae":          test_mae,
    "test_rmse":         test_rmse,
}]
schema = StructType([
    StructField("model_name",       StringType()),
    StructField("ensemble_weights", StringType()),
    StructField("train_rows",       IntegerType()),
    StructField("test_rows",        IntegerType()),
    StructField("test_year_min",    IntegerType()),
    StructField("test_year_max",    IntegerType()),
    StructField("test_r2",          DoubleType()),
    StructField("test_mae",         DoubleType()),
    StructField("test_rmse",        DoubleType()),
])
TARGET_STATS = "workspace.default.model_ensemble_stats"
(spark.createDataFrame(stats, schema=schema)
      .write.format("delta")
      .mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(TARGET_STATS))
print(f"✅ Wrote 1 stats row to {TARGET_STATS}")

# ── Persist trained ensemble artifacts for real-time app inference ─────
# Each (crop, region) pair → pickle file with the LGBM booster + RF
# regressor + ensemble weights + feature spec. The TerraCast app loads
# these from /Volumes/workspace/terract/models/ to power the Predictor
# page (`predict_yield_real`).
import os, pickle, json

ARTIFACT_DIR = "/Volumes/workspace/default/models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Recover the feature name list from the first training matrix.
_first_X = next(iter(train_Xy.values()))[0]
FEATURE_NAMES = list(_first_X.columns)
print(f"Feature schema ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

manifest = {
    "ensemble_weights": {"lgbm": w_lgbm, "rf": w_rf},
    "features": FEATURE_NAMES,
    "regions": sorted({r for (_, r) in models.keys()}),
    "crops": sorted({c for (c, _) in models.keys()}),
    "test_r2": test_r2,
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "train_rows": n_train,
    "test_rows": int(len(out)),
}

for (crop, region), (lgbm_m, rf_m) in models.items():
    fname = f"ensemble_{crop.lower()}_{region.lower()}.pkl"
    fpath = os.path.join(ARTIFACT_DIR, fname)
    payload = {
        "crop": crop,
        "region": region,
        "lgbm": lgbm_m,
        "rf": rf_m,
        "weights": {"lgbm": w_lgbm, "rf": w_rf},
        "features": FEATURE_NAMES,
    }
    with open(fpath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved {fpath}  (LGBM + RF, {len(FEATURE_NAMES)} features)")

manifest_path = os.path.join(ARTIFACT_DIR, "ensemble_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"✅ Saved {manifest_path}")

# ── Grant SELECT to the TerraCast app's service principal ──────────────
APP_SP = "60ef545d-c68d-4cf8-acc8-9a3c2d9f85a9"
try:
    spark.sql(f"GRANT SELECT ON TABLE {TARGET_PRED}  TO `{APP_SP}`")
    spark.sql(f"GRANT SELECT ON TABLE {TARGET_STATS} TO `{APP_SP}`")
    print(f"✅ Granted SELECT on both tables to '{APP_SP}'")
except Exception as exc:
    print(f"⚠️ Grant skipped — service principal may differ: {exc}")
