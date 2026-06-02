"""
Downloads all 6 regional LightGBM models from MLflow and saves them
as .pkl files in the models/ directory.

Run once: python scripts/download_models.py
"""

import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HOST = os.environ.get("DATABRICKS_HOST", "").strip()

# Run names logged by the training notebook
RUN_NAME_TO_FILE = {
    "ensemble_corn_midwest":    "lgbm_corn_midwest.pkl",
    "ensemble_corn_plains":     "lgbm_corn_plains.pkl",
    "ensemble_corn_south":      "lgbm_corn_south.pkl",
    "ensemble_soybeans_midwest": "lgbm_soybeans_midwest.pkl",
    "ensemble_soybeans_plains":  "lgbm_soybeans_plains.pkl",
    "ensemble_soybeans_south":   "lgbm_soybeans_south.pkl",
}

EXPERIMENT_PATH = "/Users/dlee23@uw.edu/terra_ml"
OUT_DIR = Path(__file__).parent.parent / "models"


def _get_cached_token(host: str) -> str:
    cache = Path.home() / ".databricks" / "token-cache.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        entry = data.get("tokens", {}).get(host, {})
        token = entry.get("access_token", "")
        if token:
            return token
    raise SystemExit(
        f"No cached token for {host}.\n"
        f"Run: databricks auth login --host {host}"
    )


def main():
    if not HOST:
        raise SystemExit("DATABRICKS_HOST not set in .env")

    import mlflow
    from mlflow.tracking import MlflowClient

    token = _get_cached_token(HOST)
    os.environ["DATABRICKS_HOST"] = HOST
    os.environ["DATABRICKS_TOKEN"] = token
    mlflow.set_tracking_uri("databricks")

    client = MlflowClient()

    print(f"Finding experiment: {EXPERIMENT_PATH}")
    exp = client.get_experiment_by_name(EXPERIMENT_PATH)
    if exp is None:
        raise SystemExit(f"Experiment not found: {EXPERIMENT_PATH}")
    print(f"  Experiment ID: {exp.experiment_id}")

    print("Searching for logged models ...")
    logged_models = client.search_logged_models(
        experiment_ids=[exp.experiment_id],
        max_results=50,
    )

    # Map model name → model_uri
    uri_map = {}
    for lm in logged_models:
        print(f"  Found: {lm.name}  uri={lm.model_uri}")
        uri_map[lm.name] = lm.model_uri

    OUT_DIR.mkdir(exist_ok=True)

    for dest_file in RUN_NAME_TO_FILE.values():
        model_name = dest_file.replace(".pkl", "")
        dest = OUT_DIR / dest_file
        model_uri = uri_map.get(model_name)
        print(f"Downloading {model_name} ...", end=" ", flush=True)
        if not model_uri:
            print("SKIP — not found in experiment")
            continue
        try:
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            with open(dest, "wb") as f:
                pickle.dump(pyfunc_model, f)
            print(f"saved → {dest.name}")
        except Exception as e:
            print(f"FAILED — {e}")

    print("\nDone. Files in models/:")
    for f in sorted(OUT_DIR.glob("*.pkl")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
