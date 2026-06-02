#!/usr/bin/env bash
# Source before running Spark/Delta (Spark 3.5 does not support Java 25+).
#   source scripts/use-java17.sh
_candidates=(
  "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
  "/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home"
)
for _j in "${_candidates[@]}"; do
  if [[ -x "$_j/bin/java" ]]; then
    export JAVA_HOME="$_j"
    export PATH="$JAVA_HOME/bin:$PATH"
  fi
done
unset _candidates _j
if [[ -z "${JAVA_HOME:-}" ]] || ! command -v java &>/dev/null; then
  echo "Java 17 not found. Install with: brew install openjdk@17" >&2
  return 1 2>/dev/null || exit 1
fi
