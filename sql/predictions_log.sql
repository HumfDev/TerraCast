-- Run once in Databricks SQL or a notebook (%sql).
-- Adjust catalog.schema to match your workspace.

CREATE TABLE IF NOT EXISTS workspace.default.predictions_log (
  predicted_at TIMESTAMP,
  crop_type STRING,
  state_abbr STRING,
  state_fips INT,
  county_fips INT,
  year INT,
  is_irrigated INT,
  predicted_yield DOUBLE,
  yield_category STRING,
  model_version STRING,
  input_json STRING
)
USING DELTA
COMMENT 'TerraCast API prediction audit log';

-- GRANT SELECT, MODIFY ON TABLE workspace.default.predictions_log TO `your-app-principal`;
