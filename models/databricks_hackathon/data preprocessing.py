# Databricks notebook source
# DBTITLE 1,Open weather table as DataFrame
df = spark.sql("SELECT * FROM rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily")
display(df)

# COMMAND ----------

# DBTITLE 1,Cell 1
df_station = spark.sql("SELECT COUNT(DISTINCT station) AS distinct_station_count FROM rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily")
display(df_station)

# COMMAND ----------

df = spark.sql("SELECT * FROM rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily")
display(df)

# COMMAND ----------

# =============================================================================
# WEATHER DATA PROCESSING PIPELINE FOR HACKATHON OBJECTIVE 1
# Weather-to-Yield Signal Detection Challenge
# =============================================================================
# Focus: Top 5 Corn/Soybean states (IA, IL, IN, NE, MN)
# Period: 2010-2024
# Source: NOAA GHCN Daily Weather Observations
# =============================================================================

# ---- STEP 1: Load raw data and inspect ----
df_raw = spark.sql("SELECT * FROM rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily")
display(df_raw)

# ---- STEP 2: Filter to US stations in target states and date range (2010-2024) ----
# NOAA GHCN station IDs starting with 'US' are US stations.
# Station IDs encode state via FIPS codes in the station name or we filter by state abbreviation in the `name` column.
# We also filter to the growing season relevant period and convert units.

spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_us_filtered AS
SELECT
    station,
    date,
    latitude,
    longitude,
    elevation,
    name,
    
    -- Convert precipitation from tenths of mm to mm
    CASE WHEN precipitation >= 0 THEN precipitation / 10.0 ELSE NULL END AS precip_mm,
    
    -- Convert snowfall from mm to mm (already in mm in GHCN)
    CASE WHEN snowfall >= 0 THEN snowfall ELSE NULL END AS snowfall_mm,
    
    -- Convert snow_depth from mm to mm
    CASE WHEN snow_depth >= 0 THEN snow_depth ELSE NULL END AS snow_depth_mm,
    
    -- Convert temp_max from tenths of °C to °C
    CASE WHEN temp_max IS NOT NULL AND temp_max_attrs NOT LIKE '%S%' 
         THEN temp_max / 10.0 ELSE NULL END AS temp_max_c,
    
    -- Convert temp_min from tenths of °C to °C
    CASE WHEN temp_min IS NOT NULL AND temp_min_attrs NOT LIKE '%S%' 
         THEN temp_min / 10.0 ELSE NULL END AS temp_min_c,
    
    -- Derived: average daily temperature
    CASE 
        WHEN temp_max IS NOT NULL AND temp_min IS NOT NULL 
             AND temp_max_attrs NOT LIKE '%S%' AND temp_min_attrs NOT LIKE '%S%'
        THEN (temp_max / 10.0 + temp_min / 10.0) / 2.0 
        ELSE NULL 
    END AS temp_avg_c,
    
    -- Derived: daily temperature range (diurnal range)
    CASE 
        WHEN temp_max IS NOT NULL AND temp_min IS NOT NULL 
             AND temp_max_attrs NOT LIKE '%S%' AND temp_min_attrs NOT LIKE '%S%'
        THEN (temp_max - temp_min) / 10.0 
        ELSE NULL 
    END AS temp_range_c,
    
    -- Convert temps to Fahrenheit for agronomic context
    CASE WHEN temp_max IS NOT NULL AND temp_max_attrs NOT LIKE '%S%' 
         THEN (temp_max / 10.0) * 9.0/5.0 + 32.0 ELSE NULL END AS temp_max_f,
    CASE WHEN temp_min IS NOT NULL AND temp_min_attrs NOT LIKE '%S%' 
         THEN (temp_min / 10.0) * 9.0/5.0 + 32.0 ELSE NULL END AS temp_min_f,
    CASE 
        WHEN temp_max IS NOT NULL AND temp_min IS NOT NULL 
             AND temp_max_attrs NOT LIKE '%S%' AND temp_min_attrs NOT LIKE '%S%'
        THEN ((temp_max / 10.0 + temp_min / 10.0) / 2.0) * 9.0/5.0 + 32.0
        ELSE NULL 
    END AS temp_avg_f,
    
    -- Extract year, month, day for aggregation
    YEAR(date) AS year,
    MONTH(date) AS month,
    DAYOFYEAR(date) AS day_of_year,
    
    -- Assign growing season flag (April - October for Corn Belt)
    CASE WHEN MONTH(date) BETWEEN 4 AND 10 THEN 1 ELSE 0 END AS is_growing_season,
    
    -- Assign crop growth phases for corn
    CASE 
        WHEN MONTH(date) IN (4, 5) THEN 'planting'
        WHEN MONTH(date) IN (6) THEN 'vegetative_early'
        WHEN MONTH(date) IN (7) THEN 'tasseling_silking'   -- critical period
        WHEN MONTH(date) IN (8) THEN 'grain_fill'          -- critical period
        WHEN MONTH(date) IN (9, 10) THEN 'maturity_harvest'
        ELSE 'dormant'
    END AS corn_growth_phase,
    
    -- Assign crop growth phases for soybean
    CASE 
        WHEN MONTH(date) IN (5) THEN 'planting'
        WHEN MONTH(date) IN (6) THEN 'vegetative'
        WHEN MONTH(date) IN (7) THEN 'flowering'           -- critical period
        WHEN MONTH(date) IN (8) THEN 'pod_fill'            -- critical period
        WHEN MONTH(date) IN (9, 10) THEN 'maturity_harvest'
        ELSE 'dormant'
    END AS soy_growth_phase,
    
    -- Growing Degree Days (GDD) base 50°F / 10°C for corn
    -- GDD = max(0, avg_temp_F - 50), capped at 86°F max
    CASE 
        WHEN temp_max IS NOT NULL AND temp_min IS NOT NULL 
             AND temp_max_attrs NOT LIKE '%S%' AND temp_min_attrs NOT LIKE '%S%'
        THEN GREATEST(0, 
            (LEAST((temp_max / 10.0) * 9.0/5.0 + 32.0, 86.0) 
             + GREATEST((temp_min / 10.0) * 9.0/5.0 + 32.0, 50.0)) / 2.0 - 50.0
        )
        ELSE NULL 
    END AS gdd_corn,
    
    -- GDD for soybean (base 50°F, no upper cap typically used)
    CASE 
        WHEN temp_max IS NOT NULL AND temp_min IS NOT NULL 
             AND temp_max_attrs NOT LIKE '%S%' AND temp_min_attrs NOT LIKE '%S%'
        THEN GREATEST(0, 
            ((temp_max / 10.0 + temp_min / 10.0) / 2.0) * 9.0/5.0 + 32.0 - 50.0
        )
        ELSE NULL 
    END AS gdd_soy,
    
    -- Heat stress flag: temp_max > 95°F (35°C) — harmful to corn during tasseling
    CASE WHEN temp_max IS NOT NULL AND temp_max / 10.0 > 35.0 THEN 1 ELSE 0 END AS heat_stress_flag,
    
    -- Frost flag: temp_min <= 0°C (32°F)
    CASE WHEN temp_min IS NOT NULL AND temp_min / 10.0 <= 0.0 THEN 1 ELSE 0 END AS frost_flag,
    
    -- Extreme cold flag: temp_min < -10°C
    CASE WHEN temp_min IS NOT NULL AND temp_min / 10.0 < -10.0 THEN 1 ELSE 0 END AS extreme_cold_flag,
    
    -- Heavy rain flag: > 25mm (1 inch) in a day
    CASE WHEN precipitation >= 0 AND precipitation / 10.0 > 25.0 THEN 1 ELSE 0 END AS heavy_rain_flag,
    
    -- Drought indicator: no precip day
    CASE WHEN precipitation IS NOT NULL AND precipitation = 0 THEN 1 ELSE 0 END AS dry_day_flag

FROM rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily
WHERE 
    -- Filter to US stations
    station LIKE 'US%'
    -- Filter to 2010-2024
    AND YEAR(date) BETWEEN 2010 AND 2024
    -- Filter to approximate lat/lon bounding box for IA, IL, IN, NE, MN
    AND latitude BETWEEN 36.0 AND 49.5
    AND longitude BETWEEN -104.5 AND -84.5
""")

display(spark.sql("SELECT * FROM weather_us_filtered LIMIT 100"))


# ---- STEP 3: Map stations to states using lat/lon bounding boxes ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_with_state AS
SELECT 
    *,
    CASE
        WHEN latitude BETWEEN 40.375 AND 43.501 AND longitude BETWEEN -96.639 AND -90.140 THEN 'IA'
        WHEN latitude BETWEEN 36.970 AND 42.508 AND longitude BETWEEN -91.513 AND -87.020 THEN 'IL'
        WHEN latitude BETWEEN 37.772 AND 41.760 AND longitude BETWEEN -88.098 AND -84.784 THEN 'IN'
        WHEN latitude BETWEEN 40.002 AND 43.001 AND longitude BETWEEN -104.053 AND -95.308 THEN 'NE'
        WHEN latitude BETWEEN 43.499 AND 49.384 AND longitude BETWEEN -97.239 AND -89.489 THEN 'MN'
        ELSE 'OTHER'
    END AS state_abbr
FROM weather_us_filtered
""")

-- Keep only target states
spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_target_states AS
SELECT * FROM weather_with_state
WHERE state_abbr IN ('IA', 'IL', 'IN', 'NE', 'MN')
""")

display(spark.sql("SELECT state_abbr, COUNT(*) as record_count, COUNT(DISTINCT station) as station_count FROM weather_target_states GROUP BY state_abbr ORDER BY state_abbr"))


# ---- STEP 4: Monthly aggregation per station ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_monthly_station AS
SELECT
    station,
    state_abbr,
    latitude,
    longitude,
    year,
    month,
    corn_growth_phase,
    soy_growth_phase,
    
    -- Temperature metrics
    AVG(temp_avg_c) AS avg_temp_c,
    AVG(temp_avg_f) AS avg_temp_f,
    MAX(temp_max_c) AS max_temp_c,
    MIN(temp_min_c) AS min_temp_c,
    AVG(temp_range_c) AS avg_diurnal_range_c,
    STDDEV(temp_avg_c) AS temp_variability,
    
    -- Precipitation metrics
    SUM(precip_mm) AS total_precip_mm,
    AVG(precip_mm) AS avg_daily_precip_mm,
    MAX(precip_mm) AS max_daily_precip_mm,
    COUNT(CASE WHEN precip_mm > 0 THEN 1 END) AS rainy_days,
    COUNT(CASE WHEN precip_mm = 0 OR precip_mm IS NULL THEN 1 END) AS dry_days,
    
    -- Growing Degree Days
    SUM(gdd_corn) AS total_gdd_corn,
    SUM(gdd_soy) AS total_gdd_soy,
    
    -- Stress indicators
    SUM(heat_stress_flag) AS heat_stress_days,
    SUM(frost_flag) AS frost_days,
    SUM(heavy_rain_flag) AS heavy_rain_days,
    SUM(dry_day_flag) AS consecutive_dry_days_approx,
    
    -- Humidity proxy: days with precipitation
    COUNT(*) AS total_obs_days

FROM weather_target_states
GROUP BY station, state_abbr, latitude, longitude, year, month, corn_growth_phase, soy_growth_phase
""")

display(spark.sql("SELECT * FROM weather_monthly_station LIMIT 50"))


# ---- STEP 5: Seasonal / annual aggregation per station ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_growing_season_station AS
SELECT
    station,
    state_abbr,
    latitude,
    longitude,
    year,
    
    -- Full growing season temperature
    AVG(CASE WHEN is_growing_season = 1 THEN temp_avg_c END) AS gs_avg_temp_c,
    AVG(CASE WHEN is_growing_season = 1 THEN temp_avg_f END) AS gs_avg_temp_f,
    MAX(CASE WHEN is_growing_season = 1 THEN temp_max_c END) AS gs_max_temp_c,
    MIN(CASE WHEN is_growing_season = 1 THEN temp_min_c END) AS gs_min_temp_c,
    
    -- Full growing season precipitation
    SUM(CASE WHEN is_growing_season = 1 THEN precip_mm ELSE 0 END) AS gs_total_precip_mm,
    AVG(CASE WHEN is_growing_season = 1 THEN precip_mm END) AS gs_avg_daily_precip_mm,
    
    -- Cumulative GDD over growing season
    SUM(CASE WHEN is_growing_season = 1 THEN gdd_corn ELSE 0 END) AS gs_total_gdd_corn,
    SUM(CASE WHEN is_growing_season = 1 THEN gdd_soy ELSE 0 END) AS gs_total_gdd_soy,
    
    -- Stress counts during growing season
    SUM(CASE WHEN is_growing_season = 1 THEN heat_stress_flag ELSE 0 END) AS gs_heat_stress_days,
    SUM(CASE WHEN is_growing_season = 1 THEN frost_flag ELSE 0 END) AS gs_frost_days,
    SUM(CASE WHEN is_growing_season = 1 THEN heavy_rain_flag ELSE 0 END) AS gs_heavy_rain_days,
    
    -- Critical period metrics (July-Aug for corn: tasseling/grain fill)
    AVG(CASE WHEN month IN (7, 8) THEN temp_avg_c END) AS critical_corn_avg_temp_c,
    SUM(CASE WHEN month IN (7, 8) THEN precip_mm ELSE 0 END) AS critical_corn_total_precip_mm,
    SUM(CASE WHEN month IN (7, 8) THEN heat_stress_flag ELSE 0 END) AS critical_corn_heat_days,
    SUM(CASE WHEN month IN (7, 8) THEN gdd_corn ELSE 0 END) AS critical_corn_gdd,
    
    -- Critical period metrics (July-Aug for soy: flowering/pod fill)
    AVG(CASE WHEN month IN (7, 8) THEN temp_avg_c END) AS critical_soy_avg_temp_c,
    SUM(CASE WHEN month IN (7, 8) THEN precip_mm ELSE 0 END) AS critical_soy_total_precip_mm,
    SUM(CASE WHEN month IN (7, 8) THEN heat_stress_flag ELSE 0 END) AS critical_soy_heat_days,
    
    -- Planting season conditions (April-May)
    AVG(CASE WHEN month IN (4, 5) THEN temp_avg_c END) AS planting_avg_temp_c,
    SUM(CASE WHEN month IN (4, 5) THEN precip_mm ELSE 0 END) AS planting_total_precip_mm,
    SUM(CASE WHEN month IN (4, 5) THEN frost_flag ELSE 0 END) AS planting_frost_days,
    
    -- Late season (Sept-Oct)
    AVG(CASE WHEN month IN (9, 10) THEN temp_avg_c END) AS harvest_avg_temp_c,
    SUM(CASE WHEN month IN (9, 10) THEN precip_mm ELSE 0 END) AS harvest_total_precip_mm,
    
    -- Annual totals for context
    SUM(precip_mm) AS annual_total_precip_mm,
    AVG(temp_avg_c) AS annual_avg_temp_c,
    SUM(gdd_corn) AS annual_total_gdd_corn,
    
    -- Observation quality
    COUNT(*) AS total_obs_days,
    COUNT(temp_avg_c) AS days_with_temp,
    COUNT(precip_mm) AS days_with_precip

FROM weather_target_states
GROUP BY station, state_abbr, latitude, longitude, year
""")

display(spark.sql("SELECT * FROM weather_growing_season_station LIMIT 50"))


# ---- STEP 6: State-level annual aggregation (for joining with yield data) ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_state_annual AS
SELECT
    state_abbr,
    year,
    
    -- Station counts
    COUNT(DISTINCT station) AS num_stations,
    
    -- Growing season averages across all stations in the state
    AVG(gs_avg_temp_c) AS state_gs_avg_temp_c,
    AVG(gs_avg_temp_f) AS state_gs_avg_temp_f,
    AVG(gs_max_temp_c) AS state_gs_max_temp_c,
    AVG(gs_min_temp_c) AS state_gs_min_temp_c,
    AVG(gs_total_precip_mm) AS state_gs_avg_precip_mm,
    STDDEV(gs_total_precip_mm) AS state_gs_precip_stddev,
    
    -- GDD
    AVG(gs_total_gdd_corn) AS state_gs_avg_gdd_corn,
    AVG(gs_total_gdd_soy) AS state_gs_avg_gdd_soy,
    
    -- Stress
    AVG(gs_heat_stress_days) AS state_avg_heat_stress_days,
    MAX(gs_heat_stress_days) AS state_max_heat_stress_days,
    AVG(gs_frost_days) AS state_avg_frost_days,
    AVG(gs_heavy_rain_days) AS state_avg_heavy_rain_days,
    
    -- Critical period (July-Aug)
    AVG(critical_corn_avg_temp_c) AS state_critical_corn_temp_c,
    AVG(critical_corn_total_precip_mm) AS state_critical_corn_precip_mm,
    AVG(critical_corn_heat_days) AS state_critical_corn_heat_days,
    AVG(critical_corn_gdd) AS state_critical_corn_gdd,
    AVG(critical_soy_avg_temp_c) AS state_critical_soy_temp_c,
    AVG(critical_soy_total_precip_mm) AS state_critical_soy_precip_mm,
    
    -- Planting conditions
    AVG(planting_avg_temp_c) AS state_planting_avg_temp_c,
    AVG(planting_total_precip_mm) AS state_planting_precip_mm,
    AVG(planting_frost_days) AS state_planting_frost_days,
    
    -- Harvest conditions
    AVG(harvest_avg_temp_c) AS state_harvest_avg_temp_c,
    AVG(harvest_total_precip_mm) AS state_harvest_precip_mm,
    
    -- Annual
    AVG(annual_total_precip_mm) AS state_annual_precip_mm,
    AVG(annual_avg_temp_c) AS state_annual_avg_temp_c

FROM weather_growing_season_station
GROUP BY state_abbr, year
ORDER BY state_abbr, year
""")

display(spark.sql("SELECT * FROM weather_state_annual"))


# ---- STEP 7: Compute weather anomalies (deviation from station long-term average) ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW station_climatology AS
SELECT
    station,
    state_abbr,
    month,
    AVG(temp_avg_c) AS clim_avg_temp_c,
    STDDEV(temp_avg_c) AS clim_std_temp_c,
    AVG(precip_mm) AS clim_avg_precip_mm,
    STDDEV(precip_mm) AS clim_std_precip_mm,
    AVG(gdd_corn) AS clim_avg_gdd_corn
FROM weather_target_states
GROUP BY station, state_abbr, month
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_monthly_anomalies AS
SELECT
    m.station,
    m.state_abbr,
    m.year,
    m.month,
    m.avg_temp_c,
    m.total_precip_mm,
    m.total_gdd_corn,
    m.heat_stress_days,
    m.frost_days,
    
    -- Temperature anomaly (z-score)
    CASE 
        WHEN c.clim_std_temp_c > 0 
        THEN (m.avg_temp_c - c.clim_avg_temp_c) / c.clim_std_temp_c 
        ELSE 0 
    END AS temp_anomaly_zscore,
    
    -- Precipitation anomaly (z-score)
    CASE 
        WHEN c.clim_std_precip_mm > 0 
        THEN (m.total_precip_mm - c.clim_avg_precip_mm * m.total_obs_days) / (c.clim_std_precip_mm * SQRT(m.total_obs_days))
        ELSE 0 
    END AS precip_anomaly_zscore,
    
    -- Absolute departures
    m.avg_temp_c - c.clim_avg_temp_c AS temp_departure_c,
    m.total_precip_mm - (c.clim_avg_precip_mm * m.total_obs_days) AS precip_departure_mm

FROM weather_monthly_station m
JOIN station_climatology c 
    ON m.station = c.station AND m.month = c.month AND m.state_abbr = c.state_abbr
""")

display(spark.sql("SELECT * FROM weather_monthly_anomalies ORDER BY ABS(temp_anomaly_zscore) DESC LIMIT 50"))


# ---- STEP 8: Identify extreme weather events/seasons ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW extreme_weather_events AS
SELECT
    station,
    state_abbr,
    year,
    month,
    avg_temp_c,
    total_precip_mm,
    temp_anomaly_zscore,
    precip_anomaly_zscore,
    
    -- Classify extreme events
    CASE
        WHEN temp_anomaly_zscore > 2.0 AND precip_anomaly_zscore < -1.5 THEN 'HOT_DRY_EXTREME'
        WHEN temp_anomaly_zscore > 2.0 THEN 'HEAT_WAVE'
        WHEN temp_anomaly_zscore < -2.0 THEN 'COLD_EXTREME'
        WHEN precip_anomaly_zscore > 2.0 THEN 'FLOOD_RISK'
        WHEN precip_anomaly_zscore < -2.0 THEN 'DROUGHT'
        WHEN temp_anomaly_zscore > 1.5 AND precip_anomaly_zscore < -1.0 THEN 'HOT_DRY'
        WHEN precip_anomaly_zscore > 1.5 AND temp_anomaly_zscore < -0.5 THEN 'COOL_WET'
        ELSE 'NORMAL'
    END AS weather_event_type

FROM weather_monthly_anomalies
WHERE ABS(temp_anomaly_zscore) > 1.5 OR ABS(precip_anomaly_zscore) > 1.5
""")

display(spark.sql("""
    SELECT state_abbr, year, weather_event_type, COUNT(*) as event_months
    FROM extreme_weather_events 
    GROUP BY state_abbr, year, weather_event_type 
    ORDER BY year, state_abbr
"""))


# ---- STEP 9: Create consecutive dry days metric (drought proxy) ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW dry_spells AS
SELECT 
    station,
    state_abbr,
    year,
    month,
    -- Count max consecutive dry days per station-month using a running sum approach
    -- Approximate: ratio of dry days to total days gives drought intensity
    SUM(dry_day_flag) AS dry_days_in_month,
    COUNT(*) AS days_in_month,
    ROUND(SUM(dry_day_flag) * 100.0 / COUNT(*), 1) AS pct_dry_days,
    SUM(CASE WHEN is_growing_season = 1 THEN dry_day_flag ELSE 0 END) AS gs_dry_days
FROM weather_target_states
GROUP BY station, state_abbr, year, month
""")

display(spark.sql("SELECT * FROM dry_spells WHERE pct_dry_days > 80 ORDER BY year, state_abbr LIMIT 50"))


# ---- STEP 10: Final comprehensive weather features table (ready for ML / join with yields) ----
spark.sql("""
CREATE OR REPLACE TEMP VIEW weather_features_final AS
SELECT
    s.state_abbr,
    s.year,
    s.num_stations,
    
    -- Growing season temperature features
    ROUND(s.state_gs_avg_temp_c, 2) AS gs_avg_temp_c,
    ROUND(s.state_gs_avg_temp_f, 2) AS gs_avg_temp_f,
    ROUND(s.state_gs_max_temp_c, 2) AS gs_max_temp_c,
    ROUND(s.state_gs_min_temp_c, 2) AS gs_min_temp_c,
    
    -- Growing season precipitation features
    ROUND(s.state_gs_avg_precip_mm, 1) AS gs_total_precip_mm,
    ROUND(s.state_gs_precip_stddev, 1) AS gs_precip_spatial_variability,
    
    -- GDD accumulation
    ROUND(s.state_gs_avg_gdd_corn, 0) AS gs_gdd_corn,
    ROUND(s.state_gs_avg_gdd_soy, 0) AS gs_gdd_soy,
    
    -- Stress indicators
    ROUND(s.state_avg_heat_stress_days, 1) AS avg_heat_stress_days,
    ROUND(s.state_max_heat_stress_days, 0) AS max_heat_stress_days,
    ROUND(s.state_avg_frost_days, 1) AS avg_gs_frost_days,
    ROUND(s.state_avg_heavy_rain_days, 1) AS avg_heavy_rain_days,
    
    -- Critical reproductive period (July-August)
    ROUND(s.state_critical_corn_temp_c, 2) AS jul_aug_avg_temp_c,
    ROUND(s.state_critical_corn_precip_mm, 1) AS jul_aug_total_precip_mm,
    ROUND(s.state_critical_corn_heat_days, 1) AS jul_aug_heat_stress_days,
    ROUND(s.state_critical_corn_gdd, 0) AS jul_aug_gdd_corn,
    
    -- Planting period
    ROUND(s.state_planting_avg_temp_c, 2) AS apr_may_avg_temp_c,
    ROUND(s.state_planting_precip_mm, 1) AS apr_may_precip_mm,
    ROUND(s.state_planting_frost_days, 1) AS apr_may_frost_days,
    
    -- Harvest period
    ROUND(s.state_harvest_avg_temp_c, 2) AS sep_oct_avg_temp_c,
    ROUND(s.state_harvest_precip_mm, 1) AS sep_oct_precip_mm,
    
    -- Annual context
    ROUND(s.state_annual_precip_mm, 1) AS annual_precip_mm,
    ROUND(s.state_annual_avg_temp_c, 2) AS annual_avg_temp_c

FROM weather_state_annual s
ORDER BY s.state_abbr, s.year
""")

display(spark.sql("SELECT * FROM weather_features_final"))


# ---- STEP 11: Quick validation & summary stats ----
print("=== DATA QUALITY CHECK ===")
display(spark.sql("""
    SELECT 
        state_abbr,
        COUNT(DISTINCT year) AS years_covered,
        MIN(year) AS first_year,
        MAX(year) AS last_year,
        ROUND(AVG(gs_gdd_corn), 0) AS avg_gdd_corn,
        ROUND(AVG(gs_total_precip_mm), 0) AS avg_gs_precip,
        ROUND(AVG(avg_heat_stress_days), 1) AS avg_heat_days
    FROM weather_features_final
    GROUP BY state_abbr
    ORDER BY state_abbr
"""))

# ---- STEP 12: Verify known events (2012 drought, 2019 flooding) ----
print("=== KNOWN EVENT VALIDATION: 2012 Drought & 2019 Flooding ===")
display(spark.sql("""
    SELECT 
        state_abbr,
        year,
        gs_avg_temp_f,
        gs_total_precip_mm,
        avg_heat_stress_days,
        jul_aug_avg_temp_c,
        jul_aug_total_precip_mm
    FROM weather_features_final
    WHERE year IN (2011, 2012, 2013, 2018, 2019, 2020)
    ORDER BY state_abbr, year
"""))