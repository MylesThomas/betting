#!/bin/bash
###############################################################################
# Deploy MLB Pitcher Strikeouts Lambda via container image (ECR)
#
# EventBridge schedule (all rules start DISABLED — enable at season start):
#   Daily 8:30am ET → spine_update (refresh gamelogs + rebuild spine)
#   Daily 9:00am ET → pipeline     (fetch live props, score, email)
#   Daily 9:00am ET → settle       (settle prior-day games; no-op if none)
#
# Pre-season checklist (run before enabling rules for 2027 season):
#   1. Verify SES identity in AWS console
#   2. Upload model artifacts to S3:
#        aws s3 cp models/mlb_strikeouts_model.joblib    s3://the-odds-api-mt/mlb/strikeouts_model/model/
#        aws s3 cp models/mlb_strikeouts_residuals.npy   s3://the-odds-api-mt/mlb/strikeouts_model/model/
#        aws s3 cp models/mlb_strikeouts_meta.json       s3://the-odds-api-mt/mlb/strikeouts_model/model/
#   3. Upload initial spine:
#        python src/mlb_strikeouts_modeling/scripts/update_spine.py
#   4. Enable EventBridge rules (see bottom of this script)
#   5. Test pipeline: aws lambda invoke --payload '{"mode":"pipeline"}' ...
#   6. Test settle:   aws lambda invoke --payload '{"mode":"settle"}' ...
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:<acct>:betting-arb-alerts"
#   cd ~/dev/betting && bash src/mlb_strikeouts_modeling/lambda/deploy_mlb_strikeouts_lambda.sh
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="mlb-strikeouts-daily"
ECR_REPO_NAME="mlb-strikeouts-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

RULE_PIPELINE="mlb-strikeouts-pipeline-daily-1pm-et"
RULE_SETTLE="mlb-strikeouts-settle-daily-10am-et"
RULE_SPINE="mlb-strikeouts-spine-weekly-sunday"

CRON_DAILY_9AM="cron(0 14 * * ? *)"         # 9am ET (14:00 UTC)
CRON_DAILY_830AM="cron(30 13 * * ? *)"     # 8:30am ET (13:30 UTC)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY MLB STRIKEOUTS DAILY LAMBDA (CONTAINER)"
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
[ -z "${SETTLEMENT_SES_SOURCE:-}" ] && echo -e "${YELLOW}⚠️  SETTLEMENT_SES_SOURCE not set${NC}"

IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role: $IAM_ROLE_ARN"

echo "Running syntax checks..."
python -m py_compile \
  "src/mlb_strikeouts_modeling/lambda/lambda_function.py" \
  "src/mlb_strikeouts_modeling/scripts/update_spine.py" \
  "src/mlb_strikeouts_modeling/scripts/run_pipeline.py" \
  "src/mlb_strikeouts_modeling/scripts/settle_strikeouts.py"
echo "✅ Syntax checks passed"

# ── Build + push ECR image ────────────────────────────────────────────────────
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
  -f "src/mlb_strikeouts_modeling/lambda/Dockerfile" \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  .
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo -e "${GREEN}✅ Image pushed: $IMAGE_URI${NC}"

# ── Deploy Lambda ─────────────────────────────────────────────────────────────
echo ""
echo "Step 3: Deploy Lambda..."
ENV_VARS="ODDS_API_KEY=$ODDS_API_KEY,SNS_TOPIC_ARN=$SNS_TOPIC_ARN,ENABLE_SNS=true"
[ -n "${SETTLEMENT_SES_SOURCE:-}" ] && ENV_VARS="${ENV_VARS},SETTLEMENT_SES_SOURCE=${SETTLEMENT_SES_SOURCE}"
[ -n "${SETTLEMENT_SES_TO:-}" ]     && ENV_VARS="${ENV_VARS},SETTLEMENT_SES_TO=${SETTLEMENT_SES_TO}"

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
RULE_MODES["$RULE_SPINE"]="spine_update"

declare -A RULE_CRONS
RULE_CRONS["$RULE_PIPELINE"]="$CRON_DAILY_9AM"
RULE_CRONS["$RULE_SETTLE"]="$CRON_DAILY_9AM"
RULE_CRONS["$RULE_SPINE"]="$CRON_DAILY_830AM"

for RULE_NAME in "$RULE_PIPELINE" "$RULE_SETTLE" "$RULE_SPINE"; do
  MODE="${RULE_MODES[$RULE_NAME]}"
  CRON="${RULE_CRONS[$RULE_NAME]}"

  aws events put-rule --name "$RULE_NAME" \
    --schedule-expression "$CRON" --state DISABLED \
    --description "MLB strikeouts: $MODE" --region "$REGION" --output table

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
echo "Pre-season checklist (before 2027 MLB season ~2027-03-20):"
echo "  1. Upload model artifacts to S3:"
echo "       aws s3 cp models/mlb_strikeouts_model.joblib  s3://the-odds-api-mt/mlb/strikeouts_model/model/"
echo "       aws s3 cp models/mlb_strikeouts_residuals.npy s3://the-odds-api-mt/mlb/strikeouts_model/model/"
echo "       aws s3 cp models/mlb_strikeouts_meta.json     s3://the-odds-api-mt/mlb/strikeouts_model/model/"
echo "  2. Upload initial spine:"
echo "       python src/mlb_strikeouts_modeling/scripts/update_spine.py"
echo "  3. Enable EventBridge rules:"
echo "       aws events enable-rule --name $RULE_PIPELINE --region $REGION"
echo "       aws events enable-rule --name $RULE_SETTLE   --region $REGION"
echo "       aws events enable-rule --name $RULE_SPINE    --region $REGION"
echo "  4. Test pipeline:     aws lambda invoke --function-name $LAMBDA_NAME --payload '{\"mode\":\"pipeline\"}' /tmp/out.json && cat /tmp/out.json"
echo "  5. Test settle:       aws lambda invoke --function-name $LAMBDA_NAME --payload '{\"mode\":\"settle\"}' /tmp/out.json && cat /tmp/out.json"
echo "  6. Test spine_update: aws lambda invoke --function-name $LAMBDA_NAME --payload '{\"mode\":\"spine_update\"}' /tmp/out.json && cat /tmp/out.json"
echo ""
