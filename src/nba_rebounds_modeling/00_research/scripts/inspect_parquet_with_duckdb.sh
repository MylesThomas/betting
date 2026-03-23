#!/usr/bin/env bash
set -euo pipefail

# Inspect candidate modeling parquet files with DuckDB.
# Defaults target the known three-point modeling universe/cache artifacts.
#
# Usage:
#   bash src/nba_rebounds_modeling/00_research/scripts/inspect_parquet_with_duckdb.sh
#   bash src/nba_rebounds_modeling/00_research/scripts/inspect_parquet_with_duckdb.sh /path/to/file.parquet
#   bash src/nba_rebounds_modeling/00_research/scripts/inspect_parquet_with_duckdb.sh /path/a.parquet /path/b.parquet

if ! command -v duckdb >/dev/null 2>&1; then
  echo "ERROR: duckdb is not installed or not on PATH."
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  PARQUET_FILES=("$@")
else
  PARQUET_FILES=(
    "$HOME/Downloads/tmp/v6_spread_universe.parquet"
    "$HOME/Downloads/tmp/v5_eval_universe.parquet"
    "$HOME/Downloads/tmp/v5_logs_2025-26.parquet"
    "$HOME/Downloads/tmp/v5_props_2025-26.parquet"
  )
fi

run_duckdb_sql() {
  local file_path="$1"
  local sql="$2"
  duckdb -csv -c "$sql" 2>/dev/null
}

echo "Inspecting ${#PARQUET_FILES[@]} parquet file(s) with DuckDB"
echo

for parquet_path in "${PARQUET_FILES[@]}"; do
  echo "==================================================================="
  echo "FILE: $parquet_path"

  if [[ -f "$parquet_path" ]]; then
    ls -lh "$parquet_path"
  else
    echo "MISSING: $parquet_path"
    echo
    continue
  fi

  echo
  echo "ROW COUNT"
  run_duckdb_sql "$parquet_path" "SELECT COUNT(*) AS row_count FROM read_parquet('$parquet_path');"

  echo
  echo "COLUMN TYPES"
  run_duckdb_sql "$parquet_path" "DESCRIBE SELECT * FROM read_parquet('$parquet_path');"

  echo
  echo "SEASON COVERAGE (if season column exists)"
  if ! run_duckdb_sql "$parquet_path" "SELECT season, COUNT(*) AS rows FROM read_parquet('$parquet_path') GROUP BY 1 ORDER BY 1;"; then
    echo "season column not present"
  fi

  echo
  echo "SAMPLE ROWS"
  run_duckdb_sql "$parquet_path" "SELECT * FROM read_parquet('$parquet_path') LIMIT 5;"
  echo
done
