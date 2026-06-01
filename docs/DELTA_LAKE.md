# Delta Lake

[Delta Lake](https://delta.io/) is an open-source storage layer for data lakes. It adds ACID transactions, scalable metadata, streaming/batch unification, schema enforcement, time travel, and merge/update/delete on top of object stores (S3, ADLS, GCS, HDFS, or local paths).

TerraCast uses the **official examples** from [delta-io/delta/examples](https://github.com/delta-io/delta/tree/master/examples) (Apache 2.0). Do not fork custom tutorials here—run and learn from those scripts.

## Main features (from Delta Lake docs)

| Feature | Example script |
|---------|----------------|
| Quickstart (create, read, merge, overwrite, update, delete, time travel) | `examples/python/quickstart.py` |
| SQL quickstart | `examples/python/quickstart_sql.py`, `quickstart_sql_on_paths.py` |
| Structured streaming | `examples/python/streaming.py` |
| pip + `configure_spark_with_delta_pip` | `examples/python/using_with_pip.py` |
| Convert Parquet, vacuum, history, manifest | `examples/python/utilities.py` |
| Change data feed | `examples/python/change_data_feed.py` |
| Table exists | `examples/python/table_exists.py` |
| Image storage | `examples/python/image_storage.py` |
| Delta Connect | `examples/python/delta_connect.py` |

See [examples/README.md](../examples/README.md) for the upstream run instructions.

## Prerequisites

1. **Java 17** — Spark 3.5 does not work on Java 25.
   ```bash
   brew install openjdk@17
   source scripts/use-java17.sh
   ```
2. **Python venv** with PySpark + delta-spark (versions must match; see [releases](https://docs.delta.io/releases/)):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-delta.txt
   ```

## Run examples (official pattern)

Per [delta-io/delta examples](https://github.com/delta-io/delta/tree/master/examples):

```bash
source scripts/use-java17.sh
source .venv/bin/activate
./scripts/spark-submit-delta.sh examples/python/quickstart.py
```

Or set `DELTA_VERSION` if you change `requirements-delta.txt`:

```bash
spark-submit --packages io.delta:delta-spark_2.12:3.2.1 \
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
  examples/python/quickstart.py
```

**pip-only** (no `--packages` on the command line): run `examples/python/using_with_pip.py` with plain `python` after `pip install delta-spark`.

**Interactive shell:**

```bash
./scripts/pyspark-delta.sh
```

Examples write under `/tmp/delta-table` by default (as in upstream).

## Databricks

Notebooks under `models/databricks_hackathon/` use the cluster `spark` session on Databricks; they are separate from local `examples/python/`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `getSubject is not supported` | `source scripts/use-java17.sh` |
| `PackageNotFoundError: delta_spark` | `pip install -r requirements-delta.txt` |
| Class not found / Delta extensions | Use `spark-submit-delta.sh` or `--packages io.delta:delta-spark_2.12:3.2.1` |
