#!/bin/bash
###############################################################################
# Deploy NFL Rec Yards Daily Lambda via container image (ECR)
#
# EventBridge schedule (all rules start DISABLED — enable before week 1):
#   Daily 8:30am ET → settle_and_rebuild  (settle yesterday + rebuild spine + ops email)
#   Daily 9:00am ET → pipeline            (score today + send plays/results/all-time email)
#   Weekly Sunday   → spine_update        (full spine rebuild; pre-season use)
#
# Daily flow on a game day (e.g. Monday):
#   8:30am: settle Sunday's games → rebuild spine with Sunday box scores → ops email
#   9:00am: score Monday's props  → send email (today's plays + Sunday results + all-time)
#
# No-op on non-game days: pipeline finds no props and exits cleanly (no email sent).
#
# Pre-season checklist (run before enabling rules on 2026-09-09):
#   1. Verify SES identity: aws ses get-identity-verification-attributes --identities <source>
#   2. Model artifacts already uploaded to S3 (done during research phase)
#   3. Upload initial spine: python src/nfl_rec_yards_modeling/scripts/update_spine.py --season 2026
#   4. Enable EventBridge rules (see bottom of this script)
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:<acct>:betting-arb-alerts"
#   cd ~/dev/betting && bash src/nfl_rec_yards_modeling/lambda/deploy_nfl_rec_yards_lambda.sh
#
# Optional env overrides:
#   export NFL_SEASON=2026
#   export SETTLEMENT_SES_SOURCE="alerts@yourdomain.com"
#   export SETTLEMENT_SES_TO="mylescgthomas@gmail.com"
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="nfl-rec-yards-daily"
ECR_REPO_NAME="nfl-rec-yards-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

RULE_SETTLE_REBUILD="nfl-rec-yards-settle-rebuild-daily-830am-et"
RULE_PIPELINE="nfl-rec-yards-pipeline-daily-9am-et"
RULE_SPINE="nfl-rec-yards-spine-weekly-sunday"

CRON_DAILY_830AM="cron(30 13 * * ? *)"   # 8:30am ET (13:30 UTC)
CRON_DAILY_9AM="cron(0 14 * * ? *)"      # 9:00am ET (14:00 UTC)
CRON_WEEKLY_SUN="cron(0 14 ? * SUN *)"   # Sunday 9am ET

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY NFL REC YARDS DAILY LAMBDA (CONTAINER)"
echo "================================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# ── Step 1: Prerequisites ─────────────────────────────────────────────────────
echo "Step 1: Verifying prerequisites..."

if ! command -v aws &>/dev/null; then
  echo -e "${RED}❌ AWS CLI not found${NC}"; exit 1
fi
if ! command -v docker &>/dev/null; then
  echo -e "${RED}❌ Docker not found${NC}"; exit 1
fi
if ! aws sts get-caller-identity &>/dev/null; then
  echo -e "${RED}❌ AWS credentials not configured${NC}"; exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS configured (Account: $AWS_ACCOUNT_ID)"

[ -z "$ODDS_API_KEY" ]  && echo -e "${YELLOW}⚠️  ODDS_API_KEY not set${NC}"
[ -z "$SNS_TOPIC_ARN" ] && echo -e "${YELLOW}⚠️  SNS_TOPIC_ARN not set${NC}"
[ -z "${SETTLEMENT_SES_SOURCE:-}" ] && echo -e "${YELLOW}⚠️  SETTLEMENT_SES_SOURCE not set${NC}"
[ -z "${SETTLEMENT_SES_TO:-}" ]     && echo -e "${YELLOW}⚠️  SETTLEMENT_SES_TO not set${NC}"

if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &>/dev/null; then
  echo -e "${RED}❌ IAM role '$IAM_ROLE_NAME' not found${NC}"; exit 1
fi
IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role exists"

echo "Running local smoke checks..."
python -m py_compile \
  "src/nfl_rec_yards_modeling/lambda/lambda_function.py" \
  "src/nfl_rec_yards_modeling/scripts/update_spine.py" \
  "src/nfl_rec_yards_modeling/scripts/run_pipeline.py" \
  "src/nfl_rec_yards_modeling/scripts/settle_rec_yards.py"
bash -n "src/nfl_rec_yards_modeling/lambda/deploy_nfl_rec_yards_lambda.sh"
echo "✅ Local smoke checks passed"
echo ""

# ── Step 2: Build + push ECR image ────────────────────────────────────────────
echo "================================================================================"
echo "Step 2: Build and push image to ECR"
echo "================================================================================"
echo ""

if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" &>/dev/null; then
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" --output table
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"
docker buildx build \
  --platform "$DOCKER_PLATFORM" \
  --provenance=false \
  --load \
  -f "src/nfl_rec_yards_modeling/lambda/Dockerfile" \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  .
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo "✅ Image pushed: $IMAGE_URI"
echo ""

# ── Step 3: Deploy Lambda ─────────────────────────────────────────────────────
echo "================================================================================"
echo "Step 3: Deploy Lambda"
echo "================================================================================"
echo ""

if [ -z "${NFL_SEASON:-}" ]; then
  CURRENT_MONTH=$(date +%-m)
  CURRENT_YEAR=$(date +%Y)
  if [ "$CURRENT_MONTH" -ge 8 ]; then
    NFL_SEASON="$CURRENT_YEAR"
  else
    NFL_SEASON=$((CURRENT_YEAR - 1))
  fi
fi
echo "NFL_SEASON: $NFL_SEASON"

ENV_VARS="ODDS_API_KEY=$ODDS_API_KEY,SNS_TOPIC_ARN=$SNS_TOPIC_ARN,NFL_SEASON=$NFL_SEASON,ENABLE_SNS=true"
if [ -n "${SETTLEMENT_SES_SOURCE:-}" ] && [ -n "${SETTLEMENT_SES_TO:-}" ]; then
  ENV_VARS="${ENV_VARS},SETTLEMENT_SES_SOURCE=${SETTLEMENT_SES_SOURCE},SETTLEMENT_SES_TO=${SETTLEMENT_SES_TO}"
  echo "SES HTML emails enabled (source=${SETTLEMENT_SES_SOURCE})"
fi

if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" &>/dev/null; then
  echo "Updating existing Lambda..."
  aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$REGION" \
    --output table
  aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$LAMBDA_NAME" \
    --timeout 900 \
    --memory-size 2048 \
    --environment "Variables={$ENV_VARS}" \
    --region "$REGION" \
    --output table
else
  echo "Creating Lambda..."
  aws lambda create-function \
    --function-name "$LAMBDA_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$IAM_ROLE_ARN" \
    --timeout 900 \
    --memory-size 2048 \
    --environment "Variables={$ENV_VARS}" \
    --region "$REGION" \
    --output table
fi
echo -e "${GREEN}✅ Lambda deployed${NC}"
echo ""

# ── Step 4: EventBridge rules ─────────────────────────────────────────────────
echo "================================================================================"
echo "Step 4: EventBridge rules (all start DISABLED)"
echo "================================================================================"
echo ""

LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --query 'Configuration.FunctionArn' \
  --output text)

declare -A RULE_CRONS
RULE_CRONS["$RULE_SETTLE_REBUILD"]="$CRON_DAILY_830AM"
RULE_CRONS["$RULE_PIPELINE"]="$CRON_DAILY_9AM"
RULE_CRONS["$RULE_SPINE"]="$CRON_WEEKLY_SUN"

declare -A RULE_MODES
RULE_MODES["$RULE_SETTLE_REBUILD"]="settle_and_rebuild"
RULE_MODES["$RULE_PIPELINE"]="pipeline"
RULE_MODES["$RULE_SPINE"]="spine_update"

for RULE_NAME in \
  "$RULE_SETTLE_REBUILD" \
  "$RULE_PIPELINE" \
  "$RULE_SPINE"; do

  CRON="${RULE_CRONS[$RULE_NAME]}"
  MODE="${RULE_MODES[$RULE_NAME]}"

  aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "$CRON" \
    --state DISABLED \
    --description "NFL rec yards: $RULE_NAME" \
    --region "$REGION" \
    --output table

  STMT_ID="EventBridgeInvoke${RULE_NAME//[-]/_}"
  aws lambda remove-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id "$STMT_ID" \
    --region "$REGION" \
    --output text 2>/dev/null || true

  aws lambda add-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id "$STMT_ID" \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$RULE_NAME" \
    --region "$REGION" \
    --output text

  aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "[{\"Id\":\"1\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{\\\"mode\\\":\\\"${MODE}\\\"}\"}]" \
    --region "$REGION" \
    --output table

  echo -e "${GREEN}✅ $RULE_NAME — $CRON (DISABLED)${NC}"
done
echo ""

# ── Step 5: Dry-run invoke ────────────────────────────────────────────────────
echo "================================================================================"
echo "Step 5: Lambda dry-run invoke"
echo "================================================================================"
echo ""

aws lambda wait function-active-v2 --function-name "$LAMBDA_NAME" --region "$REGION"
aws lambda invoke \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --invocation-type DryRun \
  --payload '{"mode":"pipeline"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/nfl_rec_yards_dryrun.json \
  --output table
rm -f /tmp/nfl_rec_yards_dryrun.json
echo "✅ Lambda DryRun accepted (no side effects)"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "================================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "================================================================================"
echo "Lambda  : $LAMBDA_NAME"
echo "Image   : $IMAGE_URI"
echo ""
echo "EventBridge rules (all DISABLED — enable before week 1 on 2026-09-09):"
echo "  $RULE_SETTLE_REBUILD  →  $CRON_DAILY_830AM  (settle + spine rebuild daily)"
echo "  $RULE_PIPELINE        →  $CRON_DAILY_9AM    (scoring email daily)"
echo "  $RULE_SPINE           →  $CRON_WEEKLY_SUN   (full spine rebuild weekly)"
echo ""
echo "Pre-season checklist (before 2026-09-09):"
echo "  1. Verify SES identity in AWS console"
echo "  2. Upload initial 2026 spine:"
echo "       python src/nfl_rec_yards_modeling/scripts/update_spine.py --season 2026"
echo "  3. Enable EventBridge rules:"
echo "       aws events enable-rule --name $RULE_SETTLE_REBUILD --region $REGION"
echo "       aws events enable-rule --name $RULE_PIPELINE        --region $REGION"
echo "       aws events enable-rule --name $RULE_SPINE           --region $REGION"
echo "  4. Test pipeline:          aws lambda invoke --function-name $LAMBDA_NAME --region $REGION --payload '{\"mode\":\"pipeline\"}' /tmp/out.json && cat /tmp/out.json"
echo "  5. Test settle_and_rebuild: aws lambda invoke --function-name $LAMBDA_NAME --region $REGION --payload '{\"mode\":\"settle_and_rebuild\"}' /tmp/out.json && cat /tmp/out.json"
echo ""
