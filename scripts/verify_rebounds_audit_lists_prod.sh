#!/usr/bin/env bash
# Run audit-list verification against prod rebounds feature parquet on S3 (no temp download).
# Python reads s3:// via boto3 (same credentials / region as aws CLI).
#
# Requires: python3, boto3, pandas, pyarrow, AWS credentials (e.g. aws configure).
#
# By default prints 3 rows with the **latest calendar `date`** in the file (today’s slate if
# the universe is up to date), not a random 3.
#   REBOUNDS_AUDIT_SHOW_ROWS=0  — no console dump
#   REBOUNDS_AUDIT_SHOW_ROWS=10 — print 10 most-recent-by-date rows
#   REBOUNDS_AUDIT_SHOW_BY=verification_sample — print first N of the random verification
#     sample instead (debug only)
#
# Further args are forwarded to scripts/verify_rebounds_audit_lists_parquet.py, e.g.:
#   bash scripts/verify_rebounds_audit_lists_prod.sh --full-scan
#   bash scripts/verify_rebounds_audit_lists_prod.sh -- --team-frame s3://nba-betting-mt/path/to.parquet
#
# Env overrides:
#   AWS_REGION / AWS_DEFAULT_REGION (default us-east-2 for this script export)
#   REBOUNDS_AUDIT_BUCKET (default nba-betting-mt)
#   REBOUNDS_FEAT_KEY (default rebounds/features/rebounds_feature_universe.parquet)
#   REBOUNDS_FEAT_PARQUET_URI — if set, used as full --parquet (overrides bucket + key)
#   REBOUNDS_TEAM_FRAME_URI — optional extra team parquet (legacy; usually unnecessary once
#     the feature object includes team_normalized, home_team_norm, away_team_norm on each row)
#   REBOUNDS_INPUT_UNIVERSE_S3_URI — if auto-build needs it (default
#     s3://$BUCKET/rebounds/input/rebounds_input_universe.parquet for that bucket)
#   REBOUNDS_BUILD_CACHE_DIR, REBOUNDS_BUILD_SEASON — passed through to the builder
#
# When the feature file has no `team_*` context columns, `verify_rebounds_audit_lists_parquet.py`
# can run the same `build_rebounds_feature_universe` step as the Lambda, then re-read S3:
#   REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM=1  (default for this .sh) — build only if those columns
#   are still missing; no-op if the object is already from a current full-universe upload.
#   REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM=0  — only verify (and stderr note) like before.

set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-2}}"
export AWS_REGION="${AWS_REGION:-$REGION}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$REGION}"
export REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM="${REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM:-1}"

BUCKET="${REBOUNDS_AUDIT_BUCKET:-nba-betting-mt}"
FEAT_KEY="${REBOUNDS_FEAT_KEY:-rebounds/features/rebounds_feature_universe.parquet}"
FEAT_URI="${REBOUNDS_FEAT_PARQUET_URI:-s3://${BUCKET}/${FEAT_KEY}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Reading parquet from S3: ${FEAT_URI}" >&2

SHOW_ROWS="${REBOUNDS_AUDIT_SHOW_ROWS:-3}"
SHOW_BY="${REBOUNDS_AUDIT_SHOW_BY:-recent}"
PY_ARGS=(--parquet "${FEAT_URI}")
if [[ "${SHOW_ROWS}" != "0" ]]; then
  PY_ARGS+=(--show-rows "${SHOW_ROWS}" --show-by "${SHOW_BY}")
fi
if [[ -n "${REBOUNDS_TEAM_FRAME_URI:-}" ]]; then
  PY_ARGS+=(--team-frame "${REBOUNDS_TEAM_FRAME_URI}")
fi

python3 scripts/verify_rebounds_audit_lists_parquet.py "${PY_ARGS[@]}" "$@"
