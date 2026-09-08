#!/bin/bash
###############################################################################
# Deploy MLB Game Totals Lambda via container image (ECR)
#
# EventBridge schedule (all rules start DISABLED — enable to go live):
#   Daily 10:00am ET → pipeline  (fetch live totals, train, score, email bets)
#   Daily 11:00am ET → settle    (settle yesterday's 9.5 under bets)
#
# S3 spine must be up to date before enabling:
#   s3://the-odds-api-mt/mlb/game_totals_model/spine/mlb_game_totals_spine.parquet
#
# Strategy: UNDER 9.5 · edge > 0 · re-trained daily (no model artifact needed)
# OOS ROI: +8.45% benchmark (2025+2026, n=2,324)
#          +13.65% with model edge>0 (n=383)
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SES_SOURCE="<verified-ses-email>"
#   export SES_TO="mylescgthomas@gmail.com"
#   cd ~/dev/betting && bash src/mlb_game_totals_modeling/lambda/deploy_mlb_game_totals_lambda.sh
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="mlb-game-totals-daily"
ECR_REPO_NAME="mlb-game-totals-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

RULE_PIPELINE="mlb-gt-pipeline-daily-10am-et"
RULE_SETTLE="mlb-gt-settle-daily-11am-et"

CRON_PIPELINE="cron(0 14 * * ? *)"   # 10:00am ET daily (14:00 UTC, EDT=UTC-4)
CRON_SETTLE="cron(0 15 * * ? *)"     # 11:00am ET daily (15:00 UTC)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY MLB GAME TOTALS DAILY LAMBDA (CONTAINER)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# ── Prerequisites ─────────────────────────────────────────────────────────────
echo "Step 1: Verifying prerequisites..."
command -v aws    &>/dev/null || { echo -e "${RED}❌ AWS CLI not found${NC}"; exit 1; }
command -v docker &>/dev/null || { echo -e "${RED}❌ Docker not found${NC}"; exit 1; }
aws sts get-caller-identity &>/dev/null || { echo -e "${RED}❌ AWS credentials not configured${NC}"; exit 1; }
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS configured (Account: $AWS_ACCOUNT_ID)"

[ -z "$ODDS_API_KEY" ]  && echo -e "${YELLOW}⚠️  ODDS_API_KEY not set${NC}"
[ -z "$SNS_TOPIC_ARN" ] && echo -e "${YELLOW}⚠️  SNS_TOPIC_ARN not set${NC}"
[ -z "${SES_SOURCE:-}" ] && echo -e "${YELLOW}⚠️  SES_SOURCE not set${NC}"

IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role: $IAM_ROLE_ARN"

echo "Running syntax checks..."
python -m py_compile \
  "src/mlb_game_totals_modeling/lambda/lambda_function.py" \
  "src/mlb_game_totals_modeling/scripts/run_pipeline.py" \
  "src/mlb_game_totals_modeling/scripts/settle_game_totals.py"
echo "✅ Syntax checks passed"

# ── Build + push ECR image ─────────────────────────────────────────────────────
echo ""
echo "Step 2: Build and push image to ECR..."
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" &>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" --output table

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"
docker buildx build \
  --platform "$DOCKER_PLATFORM" \
  --provenance=false \
  --load \
  -f "src/mlb_game_totals_modeling/lambda/Dockerfile" \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  .
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo -e "${GREEN}✅ Image pushed: $IMAGE_URI${NC}"

# ── Deploy Lambda ─────────────────────────────────────────────────────────────
echo ""
echo "Step 3: Deploy Lambda..."
_SES_SOURCE="${SES_SOURCE:-myles@thomasquantitativestrategies.com}"
_SES_TO="${SES_TO:-mylescgthomas@gmail.com}"
ENV_VARS="ODDS_API_KEY=${ODDS_API_KEY},SNS_TOPIC_ARN=${SNS_TOPIC_ARN},SES_SOURCE=${_SES_SOURCE},SES_TO=${_SES_TO}"

if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" &>/dev/null; then
  aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" --image-uri "$IMAGE_URI" --region "$REGION" --output table
  aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$LAMBDA_NAME" --timeout 900 --memory-size 2048 \
    --environment "Variables={$ENV_VARS}" --region "$REGION" --output table
else
  aws lambda create-function \
    --function-name "$LAMBDA_NAME" --package-type Image \
    --code ImageUri="$IMAGE_URI" --role "$IAM_ROLE_ARN" \
    --timeout 900 --memory-size 2048 \
    --environment "Variables={$ENV_VARS}" --region "$REGION" --output table
fi
aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"
echo -e "${GREEN}✅ Lambda deployed: $LAMBDA_NAME${NC}"

# ── EventBridge rules ─────────────────────────────────────────────────────────
echo ""
echo "Step 4: EventBridge rules (DISABLED)..."
LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$LAMBDA_NAME" --region "$REGION" \
  --query 'Configuration.FunctionArn' --output text)

declare -A RULE_MODES
RULE_MODES["$RULE_PIPELINE"]="pipeline"
RULE_MODES["$RULE_SETTLE"]="settle"

declare -A RULE_CRONS
RULE_CRONS["$RULE_PIPELINE"]="$CRON_PIPELINE"
RULE_CRONS["$RULE_SETTLE"]="$CRON_SETTLE"

for RULE_NAME in "$RULE_PIPELINE" "$RULE_SETTLE"; do
  MODE="${RULE_MODES[$RULE_NAME]}"
  CRON="${RULE_CRONS[$RULE_NAME]}"

  aws events put-rule --name "$RULE_NAME" \
    --schedule-expression "$CRON" --state DISABLED \
    --description "MLB game totals: $MODE" --region "$REGION" --output table

  STMT_ID="EventBridgeInvoke${RULE_NAME//[-]/_}"
  aws lambda remove-permission --function-name "$LAMBDA_NAME" \
    --statement-id "$STMT_ID" --region "$REGION" --output text 2>/dev/null || true
  aws lambda add-permission --function-name "$LAMBDA_NAME" \
    --statement-id "$STMT_ID" --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$RULE_NAME" \
    --region "$REGION" --output text
  aws events put-targets --rule "$RULE_NAME" --region "$REGION" \
    --targets "[{\"Id\":\"1\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{\\\"mode\\\":\\\"${MODE}\\\"}\"}]" \
    --output table

  echo -e "${GREEN}✅ $RULE_NAME — $CRON (DISABLED)${NC}"
done

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "================================================================================"
echo "Lambda  : $LAMBDA_NAME"
echo "Image   : $IMAGE_URI"
echo ""
echo "PRE-DEPLOY CHECKLIST (before enabling EventBridge rules):"
echo "  1. Confirm spine is current (run spine rebuild if > 7 days old)"
echo "  2. Test pipeline mode with a recent historical date:"
echo "     aws lambda invoke --function-name $LAMBDA_NAME --region $REGION \\"
echo "       --payload '{\"mode\":\"pipeline\",\"gameday\":\"$(date +%Y-%m-%d)\"}' \\"
echo "       --cli-binary-format raw-in-base64-out /tmp/gt_out.json && cat /tmp/gt_out.json"
echo "  3. Monitor for 1 week before increasing bet size"
echo ""
echo "When ready to go live — enable EventBridge rules:"
echo "  aws events enable-rule --name $RULE_PIPELINE --region $REGION"
echo "  aws events enable-rule --name $RULE_SETTLE    --region $REGION"
echo ""
echo "MONITORING: If 9.5 under hit rate drops below 53% sustained over 30 days, pause."
echo ""
