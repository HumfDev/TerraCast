# Databricks notebook source
MERGED_TABLE = "workspace.default.merged"  # or spark.table("catalog.schema.merged_yield_weather")

# Option B: If you have the merge in a temp view from another notebook, use that name
# MERGED_TABLE = "final_df"  # after: final_df.createOrReplaceTempView("final_df")

WEATHER_FEATURES = [
    "avg_precip_mm", "dry_days", "heavy_rain_days", "precip_std_mm",
    "avg_temp_max_c", "avg_temp_min_c", "avg_temp_mean_c", "peak_temp_max_c",
    "heat_stress_days", "gdd", "frost_days", "winter_snowfall_mm",
    "drought_flag", "flood_flag", "extreme_heat_flag",
]
# Stress-combination interactions (heat x drought, precip x drought, etc.)
INTERACTION_PAIRS = [
    ("heat_stress_days", "dry_days"), ("avg_precip_mm", "drought_flag"), ("dry_days", "drought_flag"),
    ("heat_stress_days", "extreme_heat_flag"), ("heavy_rain_days", "flood_flag"), ("peak_temp_max_c", "dry_days"),
]
TARGET = "yield_amount"
COMMODITY_FILTER = None  # "Corn" or "Soybeans"; None = all

# COMMAND ----------

# DBTITLE 1,Cell 1
from pathlib import Path
import numpy as np
import pandas as pd

# Read merged yield+weather from Spark (adjust table name if needed)
try:
    sdf = spark.table(MERGED_TABLE)
except Exception:
    # Fallback: read from SQL if table is in a catalog
    sdf = spark.sql(f"SELECT * FROM {MERGED_TABLE}")

# Convert to pandas for sklearn/correlation (or keep Spark for large data and use MLlib)
df = sdf.toPandas()

# Normalize column names (spaces/casing)
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
if COMMODITY_FILTER:
    df = df[df["commodity"].astype(str).str.strip().str.lower() == COMMODITY_FILTER.strip().lower()]

def add_interaction_features(df):
    """Add stress-combination interaction columns (e.g. heat x drought). Returns (df, list of added col names)."""
    df = df.copy()
    added = []
    for a, b in INTERACTION_PAIRS:
        if a not in df.columns or b not in df.columns:
            continue
        col_a = pd.to_numeric(df[a], errors="coerce")
        col_b = pd.to_numeric(df[b], errors="coerce")
        if col_a.notna().sum() < 100 or col_b.notna().sum() < 100:
            continue
        name = f"{a}_x_{b}"
        df[name] = col_a * col_b
        added.append(name)
    return df, added

def get_available_features(df, extra_cols=None):
    out = []
    for c in WEATHER_FEATURES:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 100:
            continue
        out.append(c)
    for c in extra_cols or []:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 100:
                out.append(c)
    return out

def county_wise_split(df, test_size=0.2, random_state=42):
    """Split by county: some counties 100% train, others 100% test. Adds column _split: 'train' | 'test'."""
    df = df.copy()
    if "state_fips" in df.columns and "county_fips" in df.columns:
        county_id = df["state_fips"].astype(str) + "_" + df["county_fips"].astype(str)
    elif "county_fips" in df.columns:
        county_id = df["county_fips"].astype(str)
    else:
        df["_split"] = "train"
        return df
    unique_counties = county_id.unique()
    n_test = max(1, int(len(unique_counties) * test_size))
    rng = np.random.default_rng(random_state)
    test_counties = set(rng.choice(unique_counties, size=n_test, replace=False))
    df["_split"] = county_id.map(lambda c: "test" if c in test_counties else "train")
    return df

def feature_importance_correlation(df, features):
    corrs = []
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        r = df[TARGET].corr(s)
        corrs.append({"feature": f, "correlation": round(r, 4), "abs_correlation": abs(r)})
    corrs.sort(key=lambda x: -x["abs_correlation"])
    return pd.DataFrame(corrs)

def feature_importance_model(df, features):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].values
    mask = np.isfinite(y)
    X, y = X.loc[mask], y[mask]
    if len(X) < 50:
        return pd.DataFrame([{"feature": f, "importance": 0.0} for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)

df, interaction_cols = add_interaction_features(df)
df = county_wise_split(df, test_size=0.2, random_state=42)
features = get_available_features(df, extra_cols=interaction_cols)
n_train = (df["_split"] == "train").sum()
n_test = (df["_split"] == "test").sum()
print(f"Rows: {len(df)}, train: {n_train}, test: {n_test}, weather features: {len(features)}, interactions: {len([c for c in interaction_cols if c in features])}")
display(df.head(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature importance: correlation with yield

# COMMAND ----------

def feature_importance_correlation(df, features):
    corrs = []
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        r = df[TARGET].corr(s)
        corrs.append({"feature": f, "correlation": round(r, 4), "abs_correlation": abs(r)})
    corrs.sort(key=lambda x: -x["abs_correlation"])
    return pd.DataFrame(corrs)

corr_rank = feature_importance_correlation(df, features)
display(corr_rank)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature importance: Random Forest model

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

def _fit_ensemble(X, y):
    models = []
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        m.fit(X, y)
        models.append(m)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X, y)
        models.append(m)
    except ImportError:
        pass
    try:
        import catboost as cb
        m = cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=0)
        m.fit(X, y)
        models.append(m)
    except ImportError:
        pass
    if not models:
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        m.fit(X, y)
        models.append(m)
    return models

def _predict_ensemble(models, X):
    preds = np.array([m.predict(X) for m in models])
    return np.mean(preds, axis=0)

def _importance_ensemble(models, feature_names):
    imp_list = []
    for m in models:
        imp = getattr(m, "feature_importances_", None)
        if imp is not None:
            imp = np.asarray(imp)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            imp_list.append(imp)
    if not imp_list:
        return np.ones(len(feature_names)) / len(feature_names)
    return np.mean(imp_list, axis=0)

def feature_importance_model(df, features, stratify_col="commodity", train_mask=None):
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].values
    mask = np.isfinite(y)
    if train_mask is not None:
        mask = mask & np.asarray(train_mask)
    X, y = X.loc[mask], y[mask]
    if len(X) < 50:
        return pd.DataFrame([{"feature": f, "importance": 0.0} for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if stratify_col in df.columns and mask.sum() > 0:
        grp = df.loc[mask, stratify_col].astype(str)
        groups = grp.unique()
        if len(groups) > 1 and len(groups) <= 20:
            imp_agg = np.zeros(len(features))
            for g in groups:
                gmask = grp == g
                if gmask.sum() < 30:
                    continue
                models = _fit_ensemble(X_scaled[gmask], y[gmask])
                imp_agg += _importance_ensemble(models, features)
            imp_agg = imp_agg / max(imp_agg.sum(), 1e-9)
            return pd.DataFrame({"feature": features, "importance": imp_agg}).sort_values("importance", ascending=False).reset_index(drop=True)
    models = _fit_ensemble(X_scaled, y)
    imp = pd.DataFrame({"feature": features, "importance": _importance_ensemble(models, features)})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)

train_mask = (df["_split"] == "train") if "_split" in df.columns else None
model_rank = feature_importance_model(df, features, train_mask=train_mask)
display(model_rank)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Anomaly detection (yield vs weather mismatch)

# COMMAND ----------

def flag_anomalies(df, features, stratify_col="commodity", train_mask=None):
    """Fit on train counties only, predict on all. Threshold from train residuals (county-wise split)."""
    from sklearn.preprocessing import StandardScaler
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].values
    valid = np.isfinite(y)
    if valid.sum() < 50:
        df = df.copy()
        df["yield_predicted"] = np.nan
        df["yield_residual"] = np.nan
        df["anomaly_flag"] = ""
        return df
    train_valid = (np.asarray(train_mask) & valid) if train_mask is not None else valid
    if train_valid.sum() < 30:
        train_valid = valid
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X.loc[train_valid])
    y_fit = y[train_valid]
    X_all = scaler.transform(X.loc[valid])
    y_all = y[valid]
    n_valid = valid.sum()
    idx_train = np.where(train_valid)[0]
    if stratify_col in df.columns:
        grp = df.loc[valid, stratify_col].astype(str).values
        grp_train = grp[idx_train]
        groups = pd.Series(grp_train).unique()
        if len(groups) > 1 and len(groups) <= 20:
            pred_all = np.full(n_valid, np.nan, dtype=float)
            for g in groups:
                gmask_train = (grp_train == g)
                if gmask_train.sum() < 30:
                    continue
                models = _fit_ensemble(X_fit[gmask_train], y_fit[gmask_train])
                gmask_all = (grp == g)
                pred_all[gmask_all] = _predict_ensemble(models, X_all[gmask_all])
            missing = np.isnan(pred_all)
            if missing.any():
                models = _fit_ensemble(X_fit, y_fit)
                pred_all[missing] = _predict_ensemble(models, X_all[missing])
        else:
            models = _fit_ensemble(X_fit, y_fit)
            pred_all = _predict_ensemble(models, X_all)
    else:
        models = _fit_ensemble(X_fit, y_fit)
        pred_all = _predict_ensemble(models, X_all)
    residual_all = y_all - pred_all
    is_train_in_valid = train_valid[valid] if train_mask is not None else np.ones(n_valid, dtype=bool)
    res_train = residual_all[is_train_in_valid]
    thresh = 2.0 * np.nanstd(res_train) if np.nanstd(res_train) > 0 else 0
    df = df.copy()
    df["yield_predicted"] = np.nan
    df["yield_residual"] = np.nan
    df.loc[valid, "yield_predicted"] = pred_all
    df.loc[valid, "yield_residual"] = residual_all
    df["anomaly_flag"] = ""
    df.loc[valid, "anomaly_flag"] = np.where(
        df.loc[valid, "yield_residual"] > thresh, "high_yield_vs_weather",
        np.where(df.loc[valid, "yield_residual"] < -thresh, "low_yield_vs_weather", ""),
    )
    return df

def add_zscore_anomaly(df, group_cols=None):
    """Aux: Z-score-based anomaly (|z| > 2 for yield within same county·commodity)."""
    if group_cols is None:
        group_cols = [c for c in ["state_fips", "county_fips", "commodity"] if c in df.columns]
    if not group_cols:
        df["yield_zscore"] = np.nan
        df["anomaly_flag_zscore"] = ""
        return df
    df = df.copy()
    grp = df.groupby(group_cols)[TARGET].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan)
    df["yield_zscore"] = grp
    df["anomaly_flag_zscore"] = ""
    z = df["yield_zscore"]
    df.loc[z > 2, "anomaly_flag_zscore"] = "high_z"
    df.loc[z < -2, "anomaly_flag_zscore"] = "low_z"
    return df

train_mask = (df["_split"] == "train") if "_split" in df.columns else None
df = flag_anomalies(df, features, train_mask=train_mask)
df = add_zscore_anomaly(df)

# R² by commodity (Corn / Soybeans) and overall (county-wise split)
from sklearn.metrics import r2_score
mask_r2 = df["yield_predicted"].notna() & df[TARGET].notna()
r2 = r2_score(df.loc[mask_r2, TARGET], df.loc[mask_r2, "yield_predicted"]) if mask_r2.sum() >= 2 else None
r2_train = r2_test = None
if "_split" in df.columns and mask_r2.sum() >= 2:
    m_train = mask_r2 & (df["_split"] == "train")
    m_test = mask_r2 & (df["_split"] == "test")
    if m_train.sum() >= 2:
        r2_train = r2_score(df.loc[m_train, TARGET], df.loc[m_train, "yield_predicted"])
    if m_test.sum() >= 2:
        r2_test = r2_score(df.loc[m_test, TARGET], df.loc[m_test, "yield_predicted"])

def _r2_row(sub, name):
    m = sub["yield_predicted"].notna() & sub[TARGET].notna()
    if m.sum() < 2:
        return {"commodity": name, "R² (all)": None, "R² train": None, "R² test": None, "n": m.sum()}
    r2_all = r2_score(sub.loc[m, TARGET], sub.loc[m, "yield_predicted"])
    r2_tr = r2_te = None
    if "_split" in sub.columns:
        mt = m & (sub["_split"] == "train")
        me = m & (sub["_split"] == "test")
        if mt.sum() >= 2:
            r2_tr = r2_score(sub.loc[mt, TARGET], sub.loc[mt, "yield_predicted"])
        if me.sum() >= 2:
            r2_te = r2_score(sub.loc[me, TARGET], sub.loc[me, "yield_predicted"])
    return {"commodity": name, "R² (all)": r2_all, "R² train": r2_tr, "R² test": r2_te, "n": m.sum()}

r2_rows = []
for comm in ["Corn", "Soybeans"]:
    if "commodity" in df.columns:
        sub = df[df["commodity"].astype(str).str.strip().str.lower() == comm.lower()]
    else:
        sub = pd.DataFrame()
    r2_rows.append(_r2_row(sub, comm) if len(sub) >= 1 else {"commodity": comm, "R² (all)": None, "R² train": None, "R² test": None, "n": 0})
r2_rows.append({"commodity": "Overall", "R² (all)": r2, "R² train": r2_train, "R² test": r2_test, "n": int(mask_r2.sum()) if mask_r2 is not None else 0})
r2_df = pd.DataFrame(r2_rows)
cols = ["commodity", "R² (all)", "R² train", "R² test", "n"]
r2_df = r2_df[[c for c in cols if c in r2_df.columns]]
for c in ["R² (all)", "R² train", "R² test"]:
    if c in r2_df.columns:
        r2_df[c] = r2_df[c].apply(lambda x: round(x, 4) if x is not None and not np.isnan(x) else None)
print("R² by commodity (county-wise train/test):")
display(r2_df)
print("Overall: R² (all) =", f"{r2:.4f}" if r2 is not None else "—", ", R² train =", f"{r2_train:.4f}" if r2_train is not None else "—", ", R² test =", f"{r2_test:.4f}" if r2_test is not None else "—")

anomalies_res = df[df["anomaly_flag"] != ""].copy()
anomalies_z = df[df["anomaly_flag_zscore"] != ""].copy()
print("Main (Residual):", len(anomalies_res), "— high:", (anomalies_res["anomaly_flag"] == "high_yield_vs_weather").sum(), ", low:", (anomalies_res["anomaly_flag"] == "low_yield_vs_weather").sum())
print("Aux (Z-score):", len(anomalies_z), "— high_z:", (anomalies_z["anomaly_flag_zscore"] == "high_z").sum(), ", low_z:", (anomalies_z["anomaly_flag_zscore"] == "low_z").sum())
display(df[[c for c in ["state_fips", "county_fips", "year", "commodity", "yield_amount", "yield_predicted", "yield_residual", "anomaly_flag", "yield_zscore", "anomaly_flag_zscore"] if c in df.columns]].head(500))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register anomalies as Spark table (for SQL / Dashboards)
# MAGIC
# MAGIC Save the full result with predictions so you can query by state, year, or plot on a map.

# COMMAND ----------

# Register as temp view for SQL and Databricks SQL / Dashboards
anomalies_spark = spark.createDataFrame(df)
anomalies_spark.createOrReplaceTempView("weather_yield_anomalies")
print("Temp view created: weather_yield_anomalies")
display(spark.sql("SELECT state_abbr, year, anomaly_flag, anomaly_flag_zscore, COUNT(*) AS n FROM weather_yield_anomalies WHERE anomaly_flag != '' OR anomaly_flag_zscore != '' GROUP BY state_abbr, year, anomaly_flag, anomaly_flag_zscore ORDER BY state_abbr, year"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature importance plot

# COMMAND ----------

# Recompute if previous cells were skipped
if 'corr_rank' not in globals():
    corr_rank = feature_importance_correlation(df, features)
if 'model_rank' not in globals():
    model_rank = feature_importance_model(df, features)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
top = corr_rank.head(10)
ax1.barh(range(len(top)), top["abs_correlation"], color="steelblue")
ax1.set_yticks(range(len(top)))
ax1.set_yticklabels(top["feature"], fontsize=9)
ax1.set_xlabel("|Correlation with yield|")
ax1.set_title("Weather–Yield Correlation")
ax1.invert_yaxis()
topm = model_rank.head(10)
ax2.barh(range(len(topm)), topm["importance"], color="darkgreen", alpha=0.8)
ax2.set_yticks(range(len(topm)))
ax2.set_yticklabels(topm["feature"], fontsize=9)
ax2.set_xlabel("Importance (XGB+LGB+CatBoost ensemble)")
ax2.set_title("Feature Importance")
ax2.invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. "What Drives Yield?" explainer

# COMMAND ----------

# Recompute if previous cells were skipped
if 'corr_rank' not in globals():
    corr_rank = feature_importance_correlation(df, features)
if 'model_rank' not in globals():
    model_rank = feature_importance_model(df, features)

def write_explainer(corr_rank, model_rank, anomalies_df, r2=None, r2_test=None, out_path=None):
    lines = [
        "What Drives Yield? (Weather–Yield Signal)",
        "=" * 50,
        "", "0. Model fit (per Corn / Soybeans, county-wise split):",
        "   Overall R² test = " + (f"{r2_test:.4f}" if r2_test is not None else "—"),
        "   (See Section 4 table for Corn vs Soybeans R² train/test.)",
        "", "1. Top weather drivers (correlation):",
    ]
    for _, row in corr_rank.head(5).iterrows():
        lines.append(f"   - {row['feature']}: {row['correlation']:.3f}")
    lines.extend(["", "2. Top weather drivers (model importance):"])
    for _, row in model_rank.head(5).iterrows():
        lines.append(f"   - {row['feature']}: {row['importance']:.3f}")
    n_high = (anomalies_df["anomaly_flag"] == "high_yield_vs_weather").sum()
    n_low = (anomalies_df["anomaly_flag"] == "low_yield_vs_weather").sum()
    n_high_z = (anomalies_df["anomaly_flag_zscore"] == "high_z").sum() if "anomaly_flag_zscore" in anomalies_df.columns else 0
    n_low_z = (anomalies_df["anomaly_flag_zscore"] == "low_z").sum() if "anomaly_flag_zscore" in anomalies_df.columns else 0
    lines.extend([
        "", "3. Anomalies:",
        "   Main (Residual): higher than weather suggests: " + str(n_high) + ", lower: " + str(n_low),
        "   Aux (Z-score): high_z: " + str(n_high_z) + ", low_z: " + str(n_low_z),
        "", "   Use drought_flag, flood_flag, extreme_heat_flag for extreme events (e.g. 2012/2021 drought).",
    ])
    return "\n".join(lines)

r2 = globals().get("r2", None)
r2_test = globals().get("r2_test", None)
print(write_explainer(corr_rank, model_rank, df, r2=r2, r2_test=r2_test))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export for map & extreme events

# COMMAND ----------

# Map-ready dataframe (use with Databricks map or display)
cols = ["state_fips", "county_fips", "year", "county_name", "state_abbr", "commodity",
        "yield_amount", "yield_predicted", "yield_residual", "anomaly_flag", "yield_zscore", "anomaly_flag_zscore"]
available = [c for c in cols if c in df.columns]
anomalies_for_map = df[available]

# Extreme events summary
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["drought_flag"] = pd.to_numeric(df.get("drought_flag", 0), errors="coerce").fillna(0)
df["flood_flag"] = pd.to_numeric(df.get("flood_flag", 0), errors="coerce").fillna(0)
df["extreme_heat_flag"] = pd.to_numeric(df.get("extreme_heat_flag", 0), errors="coerce").fillna(0)
rows = []
for year in [2012, 2019, 2021]:
    sub = df[df["year"] == year]
    if len(sub) < 10:
        continue
    rows.append({
        "year": year, "n_counties": len(sub), "avg_yield": round(sub["yield_amount"].mean(), 2),
        "drought_pct": round(100 * sub["drought_flag"].mean(), 1),
        "flood_pct": round(100 * sub["flood_flag"].mean(), 1),
        "extreme_heat_pct": round(100 * sub["extreme_heat_flag"].mean(), 1),
    })
if rows:
    ext_df = pd.DataFrame(rows)
    display(ext_df)

display(anomalies_for_map)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export for map & extreme events

# COMMAND ----------

# Map-ready dataframe (use with Databricks map or display)
cols = ["state_fips", "county_fips", "year", "county_name", "state_abbr", "commodity",
        "yield_amount", "yield_predicted", "yield_residual", "anomaly_flag"]
available = [c for c in cols if c in df.columns]
anomalies_for_map = df[available]

# Extreme events summary
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["drought_flag"] = pd.to_numeric(df.get("drought_flag", 0), errors="coerce").fillna(0)
df["flood_flag"] = pd.to_numeric(df.get("flood_flag", 0), errors="coerce").fillna(0)
df["extreme_heat_flag"] = pd.to_numeric(df.get("extreme_heat_flag", 0), errors="coerce").fillna(0)
rows = []
for year in [2012, 2019, 2021]:
    sub = df[df["year"] == year]
    if len(sub) < 10:
        continue
    rows.append({
        "year": year, "n_counties": len(sub), "avg_yield": round(sub["yield_amount"].mean(), 2),
        "drought_pct": round(100 * sub["drought_flag"].mean(), 1),
        "flood_pct": round(100 * sub["flood_flag"].mean(), 1),
        "extreme_heat_pct": round(100 * sub["extreme_heat_flag"].mean(), 1),
    })
if rows:
    ext_df = pd.DataFrame(rows)
    display(ext_df)

display(anomalies_for_map)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key message (1 line) — All graphs support this message
# MAGIC
# MAGIC **"Yield fluctuations are more influenced by weather stress that has occurred 'when', 'how long' and 'how extreme' than the average temperature."**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Slide 1 — Problem & Why it matters
# MAGIC
# MAGIC - **"Yield is confirmed once a year, but the cause is daily accumulated weather stress"**
# MAGIC - **"We summarize the weather on a country-year basis with 'agricultural mechanism' based features, accounting for yields and detecting outliers"**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Slide 2 — Turning Daily Weather into Agronomic Signals 
# MAGIC
# MAGIC *This is where I feel like a professional.*.
# MAGIC
# MAGIC - It's not just a simple average:
# MAGIC   - **GDD** (Cumulative Growth Temperature)
# MAGIC   - **Heatwave days** (Tmax>95°F)
# MAGIC   - **Longest dry spell**
# MAGIC   - **Heavy-rain bursts** (max 5-day rain)
# MAGIC - **"We didn't remove Extreme, we structured it with a stress indicator"**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Slide 3 — What Drives Yield? (Feature importance)
# MAGIC
# MAGIC - Dashboard: SHAP/Permutation Criticality Top10
# MAGIC - **"Top drivers are organized into three categories"**
# MAGIC   1. **Heat stress** — heat_days, peak_tmax
# MAGIC   2. **Water availability** — total precip, dry spell
# MAGIC   3. **Timing/variability** — rain volatility, bursts
# MAGIC - ✅ Point: Speak with **driver 'group'**, not "variable list".
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Slide 4 — Where does yield not match the weather? (Anomaly map)
# MAGIC
# MAGIC - **"Normal: Weather stress → reduced yield"**
# MAGIC - **"Above: Excessive reduction/increase not accounted for by weather"**
# MAGIC - **residual (actual-predicted) heatmap on the map**
# MAGIC - **Shows the top 10 random country-year cards** together
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Slide 5 — Explain 2–3 case studies
# MAGIC
# MAGIC *What the judge remembers is here.*.
# MAGIC
# MAGIC ### Case A: "Yield is high even though it is a heat wave/drought"
# MAGIC → **Estimated protection factors such as irrigation/soil/variety/management** (clearly called **inference**)
# MAGIC
# MAGIC ### Case B: "The weather is fine, but the yield plunges"
# MAGIC → **Infectious disease, market/insurance reporting delays, missing data, regional disasters** potential
# MAGIC
# MAGIC ### Case C: 2021 drag/heatwave year (best if any)
# MAGIC → "This year, heat_days forecast a **+X days below normal, model forecast a yield decline, and actual declines—but **some regions mismatch**"