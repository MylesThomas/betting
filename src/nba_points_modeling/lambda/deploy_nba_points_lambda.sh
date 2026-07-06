#!/bin/bash
###############################################################################
# Deploy NBA Player Points Daily Lambda via container image (ECR)
#
# Strategy: S3 — UNDER only · shrinkage=0.25 · edge≥5pp · fav_only · all lines
# OOS performance: 1,396 bets · 56.2% win rate · +149.6u · 10.71% ROI · 15.2u MDD
#
# EventBridge schedule (all rules start DISABLED — enable before 2026-10-28):
#   Daily 11:30am ET → pipeline   (fetch props, score, email bets)
#   Daily 10:30am ET → settle     (settle prior-day games)
#   Pre-season       → spine_update (manual invoke to rebuild spine)
#
# Pre-season checklist (run before enabling rules on 2026-10-28):
#   1. Verify SES identity in AWS console
#   2. Upload spine + model artifacts:
#        python src/nba_points_modeling/scripts/update_spine.py
#        (model artifacts already uploaded to S3 during research)
#   3. Enable EventBridge rules (see bottom of this script)
#   4. Test pipeline:  aws lambda invoke --payload '{"mode":"pipeline"}' ...
#   5. Test settle:    aws lambda invoke --payload '{"mode":"settle"}' ...
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:<acct>:betting-arb-alerts"
#   cd ~/dev/betting && bash src/nba_points_modeling/lambda/deploy_nba_points_lambda.sh
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="nba-points-daily"
ECR_REPO_NAME="nba-points-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

RULE_PIPELINE="nba-points-pipeline-daily-1130am-et"
RULE_SETTLE="nba-points-settle-daily-1030am-et"

CRON_DAILY_1130="cron(30 16 * * ? *)"   # 11:30am ET (16:30 UTC)
CRON_DAILY_1030="cron(30 15 * * ? *)"   # 10:30am ET (15:30 UTC)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY NBA PLAYER POINTS DAILY LAMBDA (CONTAINER)"
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
  "src/nba_points_modeling/lambda/lambda_function.py" \
  "src/nba_points_modeling/scripts/update_spine.py" \
  "src/nba_points_modeling/scripts/run_pipeline.py" \
  "src/nba_points_modeling/scripts/settle_points.py"
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
  -f "src/nba_points_modeling/lambda/Dockerfile" \
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

for RULE_NAME in "$RULE_PIPELINE" "$RULE_SETTLE"; do
  if [ "$RULE_NAME" = "$RULE_PIPELINE" ]; then
    CRON="$CRON_DAILY_1130"; MODE="pipeline"
  else
    CRON="$CRON_DAILY_1030"; MODE="settle"
  fi

  aws events put-rule --name "$RULE_NAME" \
    --schedule-expression "$CRON" --state DISABLED \
    --description "NBA points: $MODE" --region "$REGION" --output table

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
echo "Strategy: S3 — UNDER · shrink=0.25 · edge≥5pp · fav_only · all lines"
echo ""
echo "Pre-season checklist (before 2026-10-28):"
echo "  1. Upload new-season spine:"
echo "       python src/nba_points_modeling/scripts/update_spine.py"
echo "  2. Enable EventBridge rules:"
echo "       aws events enable-rule --name $RULE_PIPELINE --region $REGION"
echo "       aws events enable-rule --name $RULE_SETTLE   --region $REGION"
echo "  3. Test pipeline:  aws lambda invoke --function-name $LAMBDA_NAME --payload '{\"mode\":\"pipeline\"}' /tmp/out.json && cat /tmp/out.json"
echo "  4. Test settle:    aws lambda invoke --function-name $LAMBDA_NAME --payload '{\"mode\":\"settle\"}' /tmp/out.json && cat /tmp/out.json"
echo ""
