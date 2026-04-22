#!/usr/bin/env bash
# Query one player's rebound ML model inputs (B_MIN_MAX_FEATS) from the latest
# rebounds_scored_<slate>.parquet on S3 for a given slate date.
#
# Usage:
#   ./scripts/query_rebounds_player_ml_features.sh [player_normalized]
# Env (optional):
#   REBOUNDS_SLATE_DATE=2026-04-20
#   REBOUNDS_S3_BUCKET=nba-betting-mt
#   REBOUNDS_RUNS_PREFIX=rebounds/daily_runs
#   REBOUNDS_PLAYER=...   (same as first arg)
#
# Requires: aws, duckdb (CLI), network for INSTALL httpfs if not cached.
#
# Player matching is loose: hyphens vs spaces and extra whitespace are ignored
# (e.g. "karl anthony towns" matches "Karl-Anthony Towns").

set -euo pipefail

SLATE_DATE="${REBOUNDS_SLATE_DATE:-2026-04-20}"
BUCKET="${REBOUNDS_S3_BUCKET:-nba-betting-mt}"
PREFIX="${REBOUNDS_RUNS_PREFIX:-rebounds/daily_runs}"
PLAYER="${1:-${REBOUNDS_PLAYER:-}}"

die() {
  echo "error: $*" >&2
  exit 1
}

command -v aws >/dev/null 2>&1 || die "aws CLI not found"
command -v duckdb >/dev/null 2>&1 || die "duckdb CLI not found"

# Refresh env-style creds for DuckDB (avoid literal $(aws ...) inside SQL).
if aws configure export-credentials --format env >/dev/null 2>&1; then
  # shellcheck disable=SC2046
  eval "$(aws configure export-credentials --format env)"
fi
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || die "missing AWS_ACCESS_KEY_ID (configure aws or export-credentials)"
[[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] || die "missing AWS_SECRET_ACCESS_KEY"

RUN_PREFIX="s3://${BUCKET}/${PREFIX}/${SLATE_DATE}/"
# Latest run_id = lexicographically max folder name under the slate prefix.
RUN_ID="$(
  aws s3 ls "${RUN_PREFIX}" 2>/dev/null | awk '{gsub(/\/$/, "", $2); print $2}' | grep -E '^[0-9A-Za-z._-]+$' | sort | tail -n 1
)"
[[ -n "${RUN_ID}" ]] || die "no run folders under ${RUN_PREFIX} (wrong date or permissions?)"

SCORED_URI="s3://${BUCKET}/${PREFIX}/${SLATE_DATE}/${RUN_ID}/rebounds_scored_${SLATE_DATE}.parquet"
echo "using scored parquet: ${SCORED_URI}" >&2

# Escape single quotes for embedding in SQL literals
sql_escape() { printf "%s" "$1" | sed "s/'/''/g"; }

TOKEN_SQL=""
if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
  TOKEN_SQL="SET s3_session_token='$(sql_escape "${AWS_SESSION_TOKEN}")';"
fi

if [[ -z "${PLAYER}" ]]; then
  echo "No player passed. Distinct player_normalized on this slate (max 40):" >&2
  duckdb -batch -c "
INSTALL httpfs; LOAD httpfs;
SET s3_region='us-east-2';
SET s3_access_key_id='$(sql_escape "${AWS_ACCESS_KEY_ID}")';
SET s3_secret_access_key='$(sql_escape "${AWS_SECRET_ACCESS_KEY}")';
${TOKEN_SQL}
SELECT DISTINCT player_normalized
FROM read_parquet('$(sql_escape "${SCORED_URI}")')
WHERE CAST(date AS DATE) = DATE '$(sql_escape "${SLATE_DATE}")'
ORDER BY 1
LIMIT 40;
"
  echo >&2
  die "pass player_normalized as arg1 or set REBOUNDS_PLAYER, e.g. $0 'nikola jokic'"
fi

P_ESC="$(sql_escape "${PLAYER}")"

duckdb -batch -c "
INSTALL httpfs; LOAD httpfs;
SET s3_region='us-east-2';
SET s3_access_key_id='$(sql_escape "${AWS_ACCESS_KEY_ID}")';
SET s3_secret_access_key='$(sql_escape "${AWS_SECRET_ACCESS_KEY}")';
${TOKEN_SQL}
SELECT
  --season, CAST(date AS DATE) AS date, game_id,
  player_normalized,
  bookmaker,
  round(line, 2) AS line,
  round(consensus_reb_line, 2) AS consensus_reb_line,
  round(over_odds, 2) AS over_odds,
  round(under_odds, 2) AS under_odds,
  round(min_line, 2) AS min_line,
  round(max_line, 2) AS max_line,
  round(spread_signed, 2) AS spread_signed,
  round(roll_reb_mean_60, 2) AS roll_reb_mean_60,
  round(roll_fg3a_mean_20, 2) AS roll_fg3a_mean_20,
  round(roll_reb_std_5, 2) AS roll_reb_std_5,
  round(yhat_ols, 2) AS yhat_ols,
  round(yhat_xgb::DOUBLE, 2) AS yhat_xgb
FROM read_parquet('$(sql_escape "${SCORED_URI}")')
WHERE trim(regexp_replace(regexp_replace(lower(player_normalized), '-', ' ', 'g'), '\s+', ' ', 'g'))
    = trim(regexp_replace(regexp_replace(lower('${P_ESC}'), '-', ' ', 'g'), '\s+', ' ', 'g'))
  AND CAST(date AS DATE) = DATE '$(sql_escape "${SLATE_DATE}")'
ORDER BY bookmaker NULLS LAST, line;
"
