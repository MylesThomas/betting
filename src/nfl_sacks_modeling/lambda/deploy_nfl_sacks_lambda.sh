#!/bin/bash
###############################################################################
# Deploy NFL Sacks Daily Lambda via container image (ECR)
#
# EventBridge schedule (all rules start DISABLED — enable before week 1):
#   Daily 8:30am ET → settle_and_rebuild  (settle yesterday + rebuild spine + Email 1)
#   Daily 9:00am ET → pipeline            (score today + Email 2: plays + results + all-time)
#
# Cron UTC notes (NFL season Sep–Jan):
#   EDT (Sep–Oct, UTC-4): 12:30 UTC = 8:30am ET | 13:00 UTC = 9:00am ET
#   EST (Nov–Jan, UTC-5): 12:30 UTC = 7:30am ET | 13:00 UTC = 8:00am ET
#   → 30-min gap preserved in both zones; spine+settle finish before pipeline fires.
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:<acct>:betting-arb-alerts"
#   export SES_SOURCE="you@example.com"
#   export SES_TO="you@example.com"
#   cd ~/dev/betting && bash src/nfl_sacks_modeling/lambda/deploy_nfl_sacks_lambda.sh
#
# Optional — override current NFL season (default: computed from today):
#   export NFL_SEASON=2026
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="nfl-sacks-daily"
ECR_REPO_NAME="nfl-sacks-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

# Rule names
RULE_SETTLE="nfl-sacks-settle-rebuild-daily-830am-et"
RULE_PIPELINE="nfl-sacks-pipeline-daily-9am-et"

# Cron expressions
CRON_830AM="cron(30 12 * * ? *)"   # 8:30am ET (12:30 UTC; EDT=8:30am / EST=7:30am)
CRON_9AM="cron(0 13 * * ? *)"      # 9:00am ET (13:00 UTC; EDT=9:00am / EST=8:00am)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY NFL SACKS DAILY LAMBDA (CONTAINER)"
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

[ -z "$ODDS_API_KEY" ]       && echo -e "${YELLOW}⚠️  ODDS_API_KEY not set${NC}"
[ -z "$SNS_TOPIC_ARN" ]      && echo -e "${YELLOW}⚠️  SNS_TOPIC_ARN not set${NC}"
[ -z "${SES_SOURCE:-}" ]     && echo -e "${YELLOW}⚠️  SES_SOURCE not set (HTML emails disabled)${NC}"
[ -z "${SES_TO:-}" ]         && echo -e "${YELLOW}⚠️  SES_TO not set (HTML emails disabled)${NC}"

if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &>/dev/null; then
  echo -e "${RED}❌ IAM role '$IAM_ROLE_NAME' not found${NC}"; exit 1
fi
IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role exists"

echo "Running local smoke checks..."
python -m py_compile \
  "src/nfl_sacks_modeling/lambda/lambda_function.py" \
  "src/nfl_sacks_modeling/scripts/fit_model.py" \
  "src/nfl_sacks_modeling/scripts/update_spine.py" \
  "src/nfl_sacks_modeling/scripts/run_pipeline.py" \
  "src/nfl_sacks_modeling/scripts/settle_sacks.py"
bash -n "src/nfl_sacks_modeling/lambda/deploy_nfl_sacks_lambda.sh"
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
  -f "src/nfl_sacks_modeling/lambda/Dockerfile" \
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

ENV_VARS="ODDS_API_KEY=$ODDS_API_KEY,SNS_TOPIC_ARN=$SNS_TOPIC_ARN,NFL_SEASON=$NFL_SEASON"
if [ -n "${SES_SOURCE:-}" ] && [ -n "${SES_TO:-}" ]; then
  ENV_VARS="${ENV_VARS},SES_SOURCE=${SES_SOURCE},SES_TO=${SES_TO}"
  echo "SES HTML emails enabled (source=${SES_SOURCE})"
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

declare -A RULE_MODES
RULE_MODES["$RULE_SETTLE"]="settle_and_rebuild"
RULE_MODES["$RULE_PIPELINE"]="pipeline"

declare -A RULE_CRONS
RULE_CRONS["$RULE_SETTLE"]="$CRON_830AM"
RULE_CRONS["$RULE_PIPELINE"]="$CRON_9AM"

for RULE_NAME in "$RULE_SETTLE" "$RULE_PIPELINE"; do
  MODE="${RULE_MODES[$RULE_NAME]}"
  CRON="${RULE_CRONS[$RULE_NAME]}"

  aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "$CRON" \
    --state DISABLED \
    --description "NFL sacks: $RULE_NAME" \
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
  /tmp/nfl_sacks_dryrun.json \
  --output table
rm -f /tmp/nfl_sacks_dryrun.json
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
echo "  $RULE_SETTLE   →  $CRON_830AM  (daily 8:30am ET)"
echo "  $RULE_PIPELINE →  $CRON_9AM    (daily 9:00am ET)"
echo ""
echo "Pre-season checklist:"
echo "  1. Verify SES identity in AWS console (SES_SOURCE must be a verified address)"
echo "  2. Run fit_model.py to upload trained LR to S3"
echo "  3. Run update_spine.py --season 2026 to upload initial spine to S3"
echo "  4. Enable EventBridge rules:"
echo "       aws events enable-rule --name $RULE_SETTLE   --region $REGION"
echo "       aws events enable-rule --name $RULE_PIPELINE --region $REGION"
echo "  5. Test settle_and_rebuild: aws lambda invoke --payload '{\"mode\":\"settle_and_rebuild\"}' ..."
echo "  6. Test pipeline:           aws lambda invoke --payload '{\"mode\":\"pipeline\"}' ..."
echo ""
