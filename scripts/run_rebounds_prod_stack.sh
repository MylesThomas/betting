#!/usr/bin/env bash
# =============================================================================
# Run rebounds prod stack in order (S3 checks → optional tests → daily pipeline
# → optional Lambda deploy → optional Lambda invoke).
#
# The heavy lifting is scripts/run_rebounds_daily_pipeline.py (build universe,
# fetch live props, slice, train if configured, score, upload run to S3, notify).
#
# Usage (from anywhere):
#   cd /path/to/betting && ./scripts/run_rebounds_prod_stack.sh
#
# With explicit slate (ET calendar date for props/features):
#   ./scripts/run_rebounds_prod_stack.sh --slate-date 2026-04-22
#
# Also deploy container + invoke Lambda (needs Docker + same env as deploy script):
#   export ODDS_API_KEY="..." SNS_TOPIC_ARN="arn:aws:sns:..."
#   ./scripts/run_rebounds_prod_stack.sh --deploy --invoke
#
# Flags:
#   --config PATH     default: config/nba_rebounds_prod.lambda.yaml
#   --slate-date YYYY-MM-DD   passed to run_rebounds_daily_pipeline.py (optional; default ET today)
#   --run-train       pass through to pipeline
#   --notify-which ols|xgb|both   default both
#   --skip-s3-check   skip aws s3 ls
#   --skip-tests      skip pytest audit lists
#   --deploy          run lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh after pipeline
#   --invoke          aws lambda invoke nba-rebounds-daily (requires --deploy or existing image)
#   --invoke-mode M   payload mode: pipeline|settlement|both (default both)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="config/nba_rebounds_prod.lambda.yaml"
SLATE_DATE=""
RUN_TRAIN=0
NOTIFY_WHICH="both"
SKIP_S3=0
SKIP_TESTS=0
DO_DEPLOY=0
DO_INVOKE=0
INVOKE_MODE="both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:?}"
      shift 2
      ;;
    --slate-date)
      SLATE_DATE="${2:?}"
      shift 2
      ;;
    --run-train)
      RUN_TRAIN=1
      shift
      ;;
    --notify-which)
      NOTIFY_WHICH="${2:?}"
      shift 2
      ;;
    --skip-s3-check)
      SKIP_S3=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --deploy)
      DO_DEPLOY=1
      shift
      ;;
    --invoke)
      DO_INVOKE=1
      shift
      ;;
    --invoke-mode)
      INVOKE_MODE="${2:?}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,29p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

AWS_REGION="${AWS_REGION:-us-east-2}"
export AWS_REGION

echo "================================================================================"
echo "rebounds prod stack | repo=${REPO_ROOT}"
echo "config=${CONFIG}"
echo "================================================================================"

if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found" >&2
  exit 1
fi
if ! aws sts get-caller-identity &>/dev/null; then
  echo "ERROR: AWS credentials not working (aws sts get-caller-identity failed)" >&2
  exit 1
fi

if [[ "${SKIP_S3}" -eq 0 ]]; then
  echo ""
  echo "Step 1: S3 layout (input + features + recent runs)"
  aws s3 ls "s3://nba-betting-mt/rebounds/input/" || true
  aws s3 ls "s3://nba-betting-mt/rebounds/features/" || true
  aws s3 ls "s3://nba-betting-mt/rebounds/daily_runs/" 2>/dev/null | tail -n 15 || true
fi

if [[ "${SKIP_TESTS}" -eq 0 ]]; then
  echo ""
  echo "Step 2: Unit tests (rebounds audit lists)"
  python -m pytest tests/unit/test_rebounds_audit_lists.py -v
fi

if [[ -z "${ODDS_API_KEY:-}" ]]; then
  echo ""
  echo "WARNING: ODDS_API_KEY is not set — live props fetch will fail if pipeline needs it." >&2
fi

echo ""
echo "Step 3: Daily pipeline (build → slice → train? → score → S3 run upload → notify?)"
PIPELINE=(
  python scripts/run_rebounds_daily_pipeline.py
  --config "${CONFIG}"
  --notify-which "${NOTIFY_WHICH}"
)
if [[ -n "${SLATE_DATE}" ]]; then
  PIPELINE+=(--slate-date "${SLATE_DATE}")
fi
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  PIPELINE+=(--run-train)
fi
echo "run | ${PIPELINE[*]}"
"${PIPELINE[@]}"

if [[ "${DO_DEPLOY}" -eq 1 ]]; then
  echo ""
  echo "Step 4: Deploy Lambda container (Docker + ECR + update-function)"
  bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh
fi

if [[ "${DO_INVOKE}" -eq 1 ]]; then
  case "${INVOKE_MODE}" in
    pipeline|settlement|both) ;;
    *)
      echo "ERROR: --invoke-mode must be pipeline|settlement|both (got: ${INVOKE_MODE})" >&2
      exit 1
      ;;
  esac
  echo ""
  echo "Step 5: Invoke Lambda (mode=${INVOKE_MODE})"
  OUT="$(mktemp "${TMPDIR:-/tmp}/nba_rebounds_lambda_resp.XXXXXX.json")"
  aws lambda invoke \
    --function-name nba-rebounds-daily \
    --region "${AWS_REGION}" \
    --cli-binary-format raw-in-base64-out \
    --payload "$(printf '%s' "{\"mode\":\"${INVOKE_MODE}\"}")" \
    "${OUT}"
  echo "response written: ${OUT}"
  cat "${OUT}"
  echo ""
fi

echo ""
echo "================================================================================"
echo "Done."
echo "================================================================================"
