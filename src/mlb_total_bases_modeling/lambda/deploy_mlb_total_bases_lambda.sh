#!/bin/bash
###############################################################################
# Deploy MLB Batter Total Bases Lambda via container image (ECR)
#
# EventBridge schedule (all rules start DISABLED — enable after confirming):
#   Daily 8:30am ET → spine_update  (incremental Statcast fetch + spine rebuild)
#   Daily 9:00am ET → pipeline      (fetch live props, score, email bets)
#   Daily 9:00am ET → settle        (settle prior-day games, email P&L)
#
# All S3 assets are already in place (model + spine uploaded during research):
#   s3://the-odds-api-mt/mlb/total_bases_model/model/mlb_tb_regression_v2.joblib
#   s3://the-odds-api-mt/mlb/total_bases_model/spine/mlb_total_bases_spine.parquet
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:<acct>:betting-arb-alerts"
#   export SES_SOURCE="<verified-ses-email>"
#   export SES_TO="mylescgthomas@gmail.com"
#   cd ~/dev/betting && bash src/mlb_total_bases_modeling/lambda/deploy_mlb_total_bases_lambda.sh
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="mlb-total-bases-daily"
ECR_REPO_NAME="mlb-total-bases-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

RULE_SPINE="mlb-tb-spine-daily-830am-et"
RULE_PIPELINE="mlb-tb-pipeline-daily-9am-et"
RULE_SETTLE="mlb-tb-settle-daily-9am-et"

CRON_SPINE="cron(30 12 * * ? *)"   # 8:30am ET daily (12:30 UTC, EDT=UTC-4)
CRON_PIPELINE="cron(0 13 * * ? *)" # 9:00am ET daily (13:00 UTC)
CRON_SETTLE="cron(0 13 * * ? *)"   # 9:00am ET daily (13:00 UTC)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY MLB TOTAL BASES DAILY LAMBDA (CONTAINER)"
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
  "src/mlb_total_bases_modeling/lambda/lambda_function.py" \
  "src/mlb_total_bases_modeling/scripts/update_spine.py" \
  "src/mlb_total_bases_modeling/scripts/run_pipeline.py" \
  "src/mlb_total_bases_modeling/scripts/settle_total_bases.py"
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
  -f "src/mlb_total_bases_modeling/lambda/Dockerfile" \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  .
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo -e "${GREEN}✅ Image pushed: $IMAGE_URI${NC}"

# ── Deploy Lambda ─────────────────────────────────────────────────────────────
echo ""
echo "Step 3: Deploy Lambda..."
# SES addresses are non-secret; hardcoded here so they're never accidentally dropped.
# Override by exporting SES_SOURCE/SES_TO in the shell before running this script.
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
RULE_MODES["$RULE_SPINE"]="spine_update"
RULE_MODES["$RULE_PIPELINE"]="pipeline"
RULE_MODES["$RULE_SETTLE"]="settle"

declare -A RULE_CRONS
RULE_CRONS["$RULE_SPINE"]="$CRON_SPINE"
RULE_CRONS["$RULE_PIPELINE"]="$CRON_PIPELINE"
RULE_CRONS["$RULE_SETTLE"]="$CRON_SETTLE"

for RULE_NAME in "$RULE_PIPELINE" "$RULE_SETTLE" "$RULE_SPINE"; do
  MODE="${RULE_MODES[$RULE_NAME]}"
  CRON="${RULE_CRONS[$RULE_NAME]}"

  aws events put-rule --name "$RULE_NAME" \
    --schedule-expression "$CRON" --state DISABLED \
    --description "MLB total bases: $MODE" --region "$REGION" --output table

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
echo "Next — test pipeline mode (games that haven't started yet today):"
echo "  aws lambda invoke --function-name $LAMBDA_NAME --region $REGION \\"
echo "    --payload '{\"mode\":\"pipeline\",\"gameday\":\"$(date +%Y-%m-%d)\"}' \\"
echo "    --cli-binary-format raw-in-base64-out /tmp/tb_out.json && cat /tmp/tb_out.json"
echo ""
echo "When ready to go live — enable EventBridge rules:"
echo "  aws events enable-rule --name $RULE_PIPELINE --region $REGION"
echo "  aws events enable-rule --name $RULE_SETTLE   --region $REGION"
echo "  aws events enable-rule --name $RULE_SPINE    --region $REGION"
echo ""
