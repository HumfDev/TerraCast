"""Creates the predictions_log Delta table in Databricks."""
import json, os, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

cache = Path.home() / ".databricks" / "token-cache.json"
token = json.loads(cache.read_text())["tokens"][os.environ["DATABRICKS_HOST"]]["access_token"]
os.environ["DATABRICKS_TOKEN"] = token

from databricks.sdk import WorkspaceClient
w = WorkspaceClient(host=os.environ["DATABRICKS_HOST"], auth_type="external-browser")

sql = (
    "CREATE TABLE IF NOT EXISTS workspace.default.predictions_log ("
    "  predicted_at TIMESTAMP,"
    "  crop_type STRING,"
    "  state_abbr STRING,"
    "  state_fips INT,"
    "  county_fips INT,"
    "  year INT,"
    "  is_irrigated INT,"
    "  predicted_yield DOUBLE,"
    "  yield_category STRING,"
    "  model_version STRING,"
    "  input_json STRING"
    ") USING DELTA COMMENT 'TerraCast API prediction audit log'"
)

resp = w.statement_execution.execute_statement(
    warehouse_id=os.environ["SQL_WAREHOUSE_ID"],
    statement=sql,
    wait_timeout="30s",
)
sid = resp.statement_id
while True:
    s = w.statement_execution.get_statement(sid)
    state = s.status.state.value
    if state not in ("PENDING", "RUNNING"):
        break
    time.sleep(0.5)

print("Result:", state)
if state == "SUCCEEDED":
    print("Table workspace.default.predictions_log created successfully.")
