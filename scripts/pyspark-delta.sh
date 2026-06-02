#!/usr/bin/env bash
# Interactive PySpark shell with Delta Lake (matches official docs pattern).
# Requires: pip install -r requirements-delta.txt and Java 11+ on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/use-java17.sh"

if ! command -v pyspark &>/dev/null; then
  echo "pyspark not found. Run: pip install -r requirements-delta.txt" >&2
  exit 1
fi

# delta-spark 3.2.x on Spark 3.5.x (Scala 2.12 artifact name is historical)
DELTA_PKG="io.delta:delta-spark_2.12:3.2.1"

exec pyspark \
  --packages "$DELTA_PKG" \
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
  "$@"
