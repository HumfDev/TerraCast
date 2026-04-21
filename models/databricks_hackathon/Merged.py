# Databricks notebook source
df = spark.sql("SELECT * FROM rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily")

df2 = spark.sql("SELECT * FROM rma_county_yields_report_399_1")

# COMMAND ----------

print("df rows:", df.count())
print("df2 rows:", df2.count())

# COMMAND ----------

import pandas as pd
import geopandas as gpd
from pyspark.sql import functions as F
from pyspark.sql.functions import col, sum as spark_sum

# ── STEP 1: Preparing weather_df ──
weather_df = df.withColumn("year", F.year("date")) \
               .withColumn("month", F.month("date"))

# ── STEP 2: Feature Engineering ──
growing_season = weather_df.filter(F.col("month").between(4, 9))
winter_season = weather_df.filter((F.col("month") >= 10) | (F.col("month") <= 3))

growing_agg = growing_season.groupBy("station", "year", "latitude", "longitude").agg(
    F.sum("precipitation").alias("total_precip"),
    F.stddev("precipitation").alias("precip_std"),
    F.sum(F.when(F.col("precipitation") == 0, 1).otherwise(0)).alias("dry_days"),
    F.sum(F.when(F.col("precipitation") > 500, 1).otherwise(0)).alias("heavy_rain_days"),
    F.avg("temp_max").alias("avg_temp_max"),
    F.avg("temp_min").alias("avg_temp_min"),
    F.max("temp_max").alias("peak_temp_max"),
    F.avg((F.col("temp_max") + F.col("temp_min")) / 2).alias("avg_temp_mean"),
    F.sum(F.when(F.col("temp_max") > 350, 1).otherwise(0)).alias("heat_stress_days"),
    F.sum(
        F.greatest(
            F.least((F.col("temp_max") + F.col("temp_min")) / 2, F.lit(300)) - F.lit(100),
            F.lit(0)
        )
    ).alias("gdd_raw"),
    F.sum(F.when(F.col("temp_min") < 0, 1).otherwise(0)).alias("frost_days"),
    F.count("*").alias("obs_count_growing")
)

snow_agg = winter_season.groupBy("station", "year").agg(
    F.sum("snowfall").alias("winter_snowfall"),
    F.avg("snow_depth").alias("avg_snow_depth"),
    F.max("snow_depth").alias("max_snow_depth"),
    F.sum(F.when(F.col("snowfall") > 0, 1).otherwise(0)).alias("snow_days")
)

weather_agg = growing_agg.join(snow_agg, on=["station", "year"], how="left")

weather_agg = weather_agg \
    .withColumn("total_precip_mm", F.col("total_precip") / 10) \
    .withColumn("precip_std_mm", F.col("precip_std") / 10) \
    .withColumn("avg_temp_max_c", F.col("avg_temp_max") / 10) \
    .withColumn("avg_temp_min_c", F.col("avg_temp_min") / 10) \
    .withColumn("avg_temp_mean_c", F.col("avg_temp_mean") / 10) \
    .withColumn("peak_temp_max_c", F.col("peak_temp_max") / 10) \
    .withColumn("gdd", F.col("gdd_raw") / 10) \
    .withColumn("winter_snowfall_mm", F.col("winter_snowfall") / 10) \
    .withColumn("avg_snow_depth_mm", F.col("avg_snow_depth") / 10) \
    .withColumn("max_snow_depth_mm", F.col("max_snow_depth") / 10) \
    .drop("total_precip", "precip_std", "avg_temp_max", "avg_temp_min",
          "avg_temp_mean", "peak_temp_max", "gdd_raw",
          "winter_snowfall", "avg_snow_depth", "max_snow_depth") \
    .withColumn("drought_flag", F.when(
        (F.col("total_precip_mm") < 200) & (F.col("heat_stress_days") > 10), 1).otherwise(0)) \
    .withColumn("flood_flag", F.when(F.col("heavy_rain_days") > 5, 1).otherwise(0)) \
    .withColumn("extreme_heat_flag", F.when(F.col("heat_stress_days") > 20, 1).otherwise(0))

print("weather_agg feature #s:", len(weather_agg.columns))

# ── STEP 3: Station → County mapping ──
counties_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_5m.zip"
counties_gdf = gpd.read_file(counties_url)
counties_gdf["state_fips"] = counties_gdf["STATEFP"].astype(int)
counties_gdf["county_fips"] = counties_gdf["COUNTYFP"].astype(int)
counties_gdf = counties_gdf[["state_fips", "county_fips", "NAME", "STUSPS", "geometry"]].to_crs("EPSG:4326")

stations_pd = (
    df.select("station", "latitude", "longitude", "group_")
    .distinct()
    .filter(F.col("group_").startswith("US"))
    .toPandas()
)
print(f"US stations: {len(stations_pd)}")

stations_gdf = gpd.GeoDataFrame(
    stations_pd,
    geometry=gpd.points_from_xy(stations_pd["longitude"], stations_pd["latitude"]),
    crs="EPSG:4326"
)

stations_mapped = gpd.sjoin(stations_gdf, counties_gdf, how="left", predicate="within")
stations_mapped = stations_mapped[["station", "state_fips", "county_fips", "STUSPS"]].dropna()
print(f"Mapped stations: {len(stations_mapped)}")

station_county_spark = spark.createDataFrame(stations_mapped)

# ── STEP 4: Weather + County mapping join ──
weather_with_county = weather_agg.join(station_county_spark, on="station", how="inner")

# ── STEP 5: County (Means of stations) ──
weather_county_agg = weather_with_county.groupBy("state_fips", "county_fips", "STUSPS", "year").agg(
    F.avg("total_precip_mm").alias("avg_precip_mm"),
    F.avg("precip_std_mm").alias("precip_std_mm"),
    F.avg("dry_days").alias("dry_days"),
    F.avg("heavy_rain_days").alias("heavy_rain_days"),
    F.avg("avg_temp_max_c").alias("avg_temp_max_c"),
    F.avg("avg_temp_min_c").alias("avg_temp_min_c"),
    F.avg("avg_temp_mean_c").alias("avg_temp_mean_c"),
    F.avg("peak_temp_max_c").alias("peak_temp_max_c"),
    F.avg("heat_stress_days").alias("heat_stress_days"),
    F.avg("gdd").alias("gdd"),
    F.avg("frost_days").alias("frost_days"),
    F.avg("winter_snowfall_mm").alias("winter_snowfall_mm"),
    F.avg("avg_snow_depth_mm").alias("avg_snow_depth_mm"),
    F.avg("max_snow_depth_mm").alias("max_snow_depth_mm"),
    F.avg("snow_days").alias("snow_days"),
    F.max("drought_flag").alias("drought_flag"),
    F.max("flood_flag").alias("flood_flag"),
    F.max("extreme_heat_flag").alias("extreme_heat_flag")
).filter(F.col("avg_temp_max_c").isNotNull())

print("weather_county_agg 확인:")
weather_county_agg.show(5)

# ── STEP 6: Yields ──
yield_clean = df2.select(
    F.col("County Code").alias("county_fips_raw"),
    F.col("State Code").alias("state_fips_raw"),
    F.col("County Name").alias("county_name"),
    F.col("State Abbreviation").alias("state_abbr"),
    F.col("Commodity Name").alias("commodity"),
    F.col("Yield Year").cast("int").alias("year"),
    F.col("Yield Amount").alias("yield_amount"),
    F.col("Irrigation Practice Name").alias("irrigation")
) \
.withColumn("state_fips", F.col("state_fips_raw").cast("int")) \
.withColumn("county_fips", F.col("county_fips_raw").cast("int")) \
.filter(F.col("yield_amount").isNotNull())

# ── STEP 7: Final join ──
final_df = yield_clean.join(
    weather_county_agg,
    on=["state_fips", "county_fips", "year"],
    how="inner"
)

print(f"\nFinal joined rows: {final_df.count()}")
final_df.show(5)
final_df.printSchema()

# COMMAND ----------

# %pip install geopandas

# COMMAND ----------

import pandas as pd
import numpy as np

# ============================================================
# Feature Engineering for Yield Prediction
# ============================================================

df = final_df.toPandas()
df.columns = [c.strip() for c in df.columns]
df["commodity"] = df["commodity"].str.strip()

# ── 1. Missing value imputation ──────────────────────────────
# Temperature columns (2041 missing) → impute with median grouped by commodity x state
temp_cols = ["avg_temp_max_c", "avg_temp_min_c", "avg_temp_mean_c", "peak_temp_max_c"]
for c in temp_cols:
    df[c] = df[c].fillna(
        df.groupby(["commodity", "state_abbr"])[c].transform("median")
    )

# Snow-related → 0 (interpreted as no snow in that region/season)
for c in ["winter_snowfall_mm", "avg_snow_depth_mm", "max_snow_depth_mm"]:
    df[c] = df[c].fillna(0)

df["avg_precip_mm"] = df["avg_precip_mm"].fillna(df["avg_precip_mm"].median())
df["precip_std_mm"] = df["precip_std_mm"].fillna(df["precip_std_mm"].median())
df["snow_days"] = df["snow_days"].fillna(0)

# ── 2. Categorical encoding ─────────────────────────────────
df["is_corn"] = (df["commodity"] == "Corn").astype(int)  # corr=0.805 ★

# ── 3. Temperature features ─────────────────────────────────
# Diurnal temperature range: larger range favors photosynthesis but extreme range causes stress
df["temp_range"] = df["avg_temp_max_c"] - df["avg_temp_min_c"]

# Heat wave intensity: how much peak temp deviates from the mean
df["peak_vs_mean_temp"] = df["peak_temp_max_c"] - df["avg_temp_mean_c"]

# Cumulative heat stress: peak temp x number of heat stress days (Corn corr=-0.41)
df["temp_stress_index"] = df["peak_temp_max_c"] * df["heat_stress_days"]

# ── 4. Precipitation features ───────────────────────────────
# Precipitation variability (CV): captures unstable rainfall patterns
df["precip_cv"] = df["precip_std_mm"] / (df["avg_precip_mm"] + 1)

# ── 5. Compound stress features ─────────────────────────────
# Combined heat + dry stress (moisture deficit x high temperature)
df["heat_dry_combo"] = df["heat_stress_days"] * df["dry_days"]

# Total extreme events (drought + extreme_heat, flood excluded)
df["extreme_events"] = df["drought_flag"] + df["extreme_heat_flag"]

# ── 6. GDD features ─────────────────────────────────────────
# GDD x precipitation: GDD is only meaningful when growing conditions are met
df["gdd_precip"] = df["gdd"] * df["avg_precip_mm"]

# Precipitation efficiency relative to GDD
df["gdd_per_precip"] = df["gdd"] / (df["avg_precip_mm"] + 1)

# ── 7. Temporal trend ───────────────────────────────────────
# Year-based technology/practice improvement trend (corr=+0.25 for Corn)
df["year_trend"] = df["year"] - 2010

# ── 8. Weather x Weather interactions ───────────────────────
# Diurnal range x heat stress days (Corn -0.41, Soy -0.36)
df["temprange_x_heat"]  = (df["avg_temp_max_c"] - df["avg_temp_min_c"]) * df["heat_stress_days"]

# Peak temp x heat stress days (Corn -0.41, Soy -0.35)
df["peak_x_heat"]       = df["peak_temp_max_c"] * df["heat_stress_days"]

# Extreme heat flag x precipitation: dangerous when heat wave coincides with low rainfall (Corn -0.41, Soy -0.37)
df["extreme_x_precip"]  = df["extreme_heat_flag"] * df["avg_precip_mm"]

# GDD x heat stress: high growth energy combined with heat stress (Corn -0.37)
df["gdd_x_heat"]        = df["gdd"] * df["heat_stress_days"]

# Precipitation x snow days: winter moisture supply (positive correlation)
df["precip_x_snow"]     = df["avg_precip_mm"] * df["snow_days"]

# GDD x precipitation: combined growth conditions (Corn +0.22)
df["gdd_x_precip"]      = df["gdd"] * df["avg_precip_mm"]

# ── 9. Nonlinear transformations ────────────────────────────
# Peak temp squared: captures nonlinear acceleration of heat damage (Corn -0.55, Soy -0.41)
df["peak_temp_sq"]      = df["peak_temp_max_c"] ** 2

# Log of heat stress days: captures early sensitivity to heat stress (Corn -0.48, Soy -0.40)
df["log_heat_days"]     = np.log1p(df["heat_stress_days"])

# ── 10. Compound extreme stress ─────────────────────────────
# Extreme events x dry days (Corn -0.48, Soy -0.40)
df["extreme_x_dry"]     = (df["drought_flag"] + df["extreme_heat_flag"]) * df["dry_days"]

# GDD x extreme events: impact of extreme events during high-growth periods (Corn -0.48, Soy -0.40)
df["gdd_x_extreme"]     = df["gdd"] * (df["drought_flag"] + df["extreme_heat_flag"])

# ── 11. Growth condition indices ────────────────────────────
# Heat stress ratio relative to GDD: proportion of growth energy lost to stress (Corn -0.42, Soy -0.35)
df["stress_gdd_ratio"]  = df["heat_stress_days"] * df["peak_temp_max_c"] / (df["gdd"] + 1)

# Optimal temperature score: approaches 0 as mean temp nears 22.5°C (ideal growing temp)
df["optimal_temp_score"] = -abs(df["avg_temp_mean_c"] - 22.5)

# Cool nights x moisture: cool overnight temps + precipitation -> favorable growing conditions (Corn +0.31)
df["cool_moist"]        = (50 - df["avg_temp_min_c"]) * df["avg_precip_mm"]

# GDD x precipitation / heat stress: growth potential adjusted for stress (Corn +0.38)
df["gdd_precip_heat"]   = df["gdd"] * df["avg_precip_mm"] / (df["heat_stress_days"] + 1)

# ── 12. Year x Weather interactions (climate change trend) ──
year_trend = df["year"] - 2010
# Recent extreme events have larger impact — captures intensifying climate risk (Corn -0.38, Soy -0.33)
df["year_x_extreme"]    = year_trend * (df["drought_flag"] + df["extreme_heat_flag"])
# Annual precipitation trend (Corn +0.35)
df["year_x_precip"]     = year_trend * df["avg_precip_mm"]
# Annual heat stress trend
df["year_x_heat"]       = year_trend * df["heat_stress_days"]

# ── 13. Commodity x Weather interactions ────────────────────
# Corn and Soybeans respond differently to weather -> captured via interaction terms
df["corn_x_gdd"]     = df["is_corn"] * df["gdd"]           # corr=0.767
df["corn_x_precip"]  = df["is_corn"] * df["avg_precip_mm"]  # corr=0.792
df["corn_x_heat"]    = df["is_corn"] * df["heat_stress_days"]
df["corn_x_drought"] = df["is_corn"] * df["drought_flag"]
df["corn_x_extreme"] = df["is_corn"] * df["extreme_events"]

# ── 14. Final feature set ───────────────────────────────────
FEATURES = [
    # Core
    "is_corn",              # corr=0.805 ★
    "corn_x_precip",        # corr=0.792 ★
    "corn_x_gdd",           # corr=0.767 ★
    # Temperature
    "avg_temp_max_c",
    "avg_temp_mean_c",
    "peak_temp_max_c",
    "temp_range",           # diurnal temperature range
    "peak_vs_mean_temp",    # heat wave intensity
    "temp_stress_index",    # cumulative heat stress
    "heat_stress_days",
    "peak_temp_sq",         # peak temp squared (Corn -0.55)
    "log_heat_days",        # log(heat stress days) (Corn -0.48)
    "optimal_temp_score",   # distance from optimal growing temp (22.5C)
    # Precipitation
    "avg_precip_mm",
    "dry_days",
    "heavy_rain_days",
    "precip_cv",            # precipitation variability (CV)
    # Compound stress
    "heat_dry_combo",       # heat x dry days
    "extreme_events",       # drought + extreme_heat
    "drought_flag",
    "extreme_heat_flag",
    "extreme_x_dry",        # extreme events x dry days (Corn -0.48)
    "gdd_x_extreme",        # GDD x extreme events (Corn -0.48)
    "stress_gdd_ratio",     # heat stress / GDD ratio (Corn -0.42)
    # Growth
    "gdd",
    "gdd_precip",           # GDD x precipitation
    "gdd_per_precip",
    "gdd_precip_heat",      # GDD x precip / heat stress (Corn +0.38)
    "cool_moist",           # cool nights x precipitation (Corn +0.31)
    # Winter / Snow
    "frost_days",
    "winter_snowfall_mm",
    "snow_days",
    # Weather x Weather interactions
    "temprange_x_heat",     # diurnal range x heat stress
    "peak_x_heat",          # peak temp x heat stress
    "extreme_x_precip",     # extreme heat x precipitation
    "gdd_x_heat",           # GDD x heat stress
    "gdd_x_precip",         # GDD x precipitation
    "precip_x_snow",        # precipitation x snow days
    # Temporal x Weather
    "year_trend",
    "year_x_extreme",       # year x extreme events (Corn -0.38)
    "year_x_precip",        # year x precipitation (Corn +0.35)
    "year_x_heat",          # year x heat stress
    # Commodity x Weather interactions
    "corn_x_heat",
    "corn_x_drought",
    "corn_x_extreme",
]

TARGET = "yield_amount"

# ── 15. Correlation summary ─────────────────────────────────
corr = (
    df[FEATURES + [TARGET]]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)
print("=== Feature-Yield Correlations ===")
print(corr.round(4).to_string())
print(f"\nTotal features: {len(FEATURES)}")
print(f"|corr| > 0.5: {(corr.abs() > 0.5).sum()}")
print(f"|corr| > 0.3: {(corr.abs() > 0.3).sum()}")

# ── 16. Save engineered dataset ─────────────────────────────
out_cols = ["state_fips", "county_fips", "year", "county_name", "state_abbr", "commodity"] + FEATURES + [TARGET]
df_out = df[[c for c in out_cols if c in df.columns]].dropna(subset=[TARGET])
df_out.to_csv("features_engineered.csv", index=False)
print(f"\nSaved: features_engineered.csv {df_out.shape}")

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 1. Load ──────────────────────────────────────────────────
df = final_df.toPandas()
df.columns = [c.strip() for c in df.columns]

TARGET = "yield_amount"
FEATURES = [
    "is_corn", "corn_x_precip", "corn_x_gdd",
    "avg_temp_max_c", "avg_temp_mean_c", "peak_temp_max_c",
    "temp_range", "peak_vs_mean_temp", "temp_stress_index", "heat_stress_days",
    "peak_temp_sq", "log_heat_days", "optimal_temp_score",
    "avg_precip_mm", "dry_days", "heavy_rain_days", "precip_cv",
    "heat_dry_combo", "extreme_events", "drought_flag", "extreme_heat_flag",
    "extreme_x_dry", "gdd_x_extreme", "stress_gdd_ratio",
    "gdd", "gdd_precip", "gdd_per_precip", "gdd_precip_heat", "cool_moist",
    "frost_days", "winter_snowfall_mm", "snow_days",
    "temprange_x_heat", "peak_x_heat", "extreme_x_precip",
    "gdd_x_heat", "gdd_x_precip", "precip_x_snow",
    "year_trend", "year_x_extreme", "year_x_precip", "year_x_heat",
    "corn_x_heat", "corn_x_drought", "corn_x_extreme",
]
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df[TARGET]

# ── 2. 8:2 county-wise split (data leakage 방지) ────────────
counties = df["county_fips"].unique()
np.random.seed(42)
test_counties = np.random.choice(counties, size=int(len(counties) * 0.2), replace=False)
test_mask = df["county_fips"].isin(test_counties)

X_train, X_test = X[~test_mask], X[test_mask]
y_train, y_test = y[~test_mask], y[test_mask]
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 3. Model ─────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=5,
    max_features=0.4,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
)
rf.fit(X_train, y_train)

# ── 4. Evaluation ────────────────────────────────────────────
y_pred = rf.predict(X_test)

print(f"R²  Train : {r2_score(y_train, rf.predict(X_train)):.4f}")
print(f"R²  Test  : {r2_score(y_test, y_pred):.4f}")
print(f"OOB Score : {rf.oob_score_:.4f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ── 5. Feature importance ────────────────────────────────────
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

display(imp_df.head(20))