# TerraCast lakehouse data

Production TerraCast reads Unity Catalog Delta tables via **SQL Warehouse** (see [SETUP_DATABRICKS.md](SETUP_DATABRICKS.md)).

## `workspace.default.merged` (default)

Canonical merged yield + weather dataset. Same table Genie should use.

| Column | Type | Notes |
|--------|------|--------|
| `state_fips` | int | US state FIPS |
| `county_fips` | int | County FIPS (within state) |
| `state_abbr` | string | e.g. `IL` |
| `year` | int | Yield year |
| `commodity` | string | `Corn` or `Soybeans` |
| `irrigation` | string | `Non-Irrigated` or `Irrigated` |
| `avg_temp_max_c` … `snow_days` | double | Growing-season / winter weather features |
| `yield_amount` | double | Actual yield (bu/acre) |
| `drought_flag`, `flood_flag`, `extreme_heat_flag` | int | Optional flags |

**Grain:** one row per `(state_fips, county_fips, year, commodity, irrigation)`.

**State average (`county_fips = 0` in the App):** `GET /features` returns `AVG(...)` of weather columns across counties in that state/year/crop/irrigation.

### Persist from notebook

```python
final_df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.merged")
```

## `workspace.default.predictions_log`

Append-only log written by `POST /predict` when `LOG_PREDICTIONS=true`.

See [sql/predictions_log.sql](../sql/predictions_log.sql).

## Model artifacts (Volume, not Delta)

| File | Purpose |
|------|---------|
| `corn_model.txt` / `soybean_model.txt` | LightGBM boosters |
| `corn_trend.json` / `soybean_trend.json` | State detrend polynomials |
| `corn_baselines.json` / `soybean_baselines.json` | State/county yield baselines |

Path: `MODEL_VOLUME_PATH` (e.g. `/Volumes/workspace/terract/models`).

## Environment variables

| Variable | Default |
|----------|---------|
| `SQL_WAREHOUSE_ID` | (required for lakehouse) |
| `DELTA_TABLE_MERGED` | `workspace.default.merged` |
| `DELTA_TABLE_PREDICTIONS` | `workspace.default.predictions_log` |
| `MODEL_VOLUME_PATH` | `models` (app bundle) |
| `LAKEHOUSE_ENABLED` | `true` |
| `LOG_PREDICTIONS` | `true` when predictions table set |
