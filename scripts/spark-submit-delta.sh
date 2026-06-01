#!/usr/bin/env bash
# Run an official Delta Lake Python example (see examples/README.md).
# Usage: ./scripts/spark-submit-delta.sh examples/python/quickstart.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/use-java17.sh"

DELTA_VERSION="${DELTA_VERSION:-3.2.1}"
DELTA_PKG="io.delta:delta-spark_2.12:${DELTA_VERSION}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PATH/TO/EXAMPLE.py [spark-submit args...]" >&2
  exit 1
fi

EXAMPLE="$1"
shift

if [[ ! -f "$EXAMPLE" ]]; then
  echo "Example not found: $EXAMPLE" >&2
  exit 1
fi

exec spark-submit \
  --packages "$DELTA_PKG" \
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
  "$EXAMPLE" "$@"
