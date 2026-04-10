#!/bin/bash
###############################################################################
# Deploy NBA Rebounds Daily Lambda via container image (ECR)
#
# Workflow in Lambda:
# 1) python src/nba_rebounds_modeling/00_research/scripts/run_rebounds_daily_pipeline.py --config ...
# 2) python src/nba_rebounds_modeling/00_research/scripts/settle_rebounds_runs.py --latest-only
#
# Usage:
#   export ODDS_API_KEY="your-key"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:232692785472:betting-arb-alerts"
#   cd ~/dev/betting && bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="nba-rebounds-daily"
SCORE_RULE_NAME="nba-rebounds-daily-score-9am-et"
SETTLE_RULE_NAME="nba-rebounds-daily-settle-905am-et"
ECR_REPO_NAME="nba-rebounds-daily"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
# 9:00am ET = 14:00 UTC during EST (winter). Update if you prefer EDT-specific handling.
CRON_SCORE_9AM_ET="cron(0 14 * * ? *)"
CRON_SETTLE_905AM_ET="cron(5 14 * * ? *)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY NBA REBOUNDS DAILY LAMBDA (CONTAINER)"
echo "================================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "Step 1: Verifying prerequisites..."
if ! command -v aws &> /dev/null; then
  echo -e "${RED}❌ AWS CLI not found${NC}"
  exit 1
fi
if ! command -v docker &> /dev/null; then
  echo -e "${RED}❌ Docker not found${NC}"
  exit 1
fi
if ! aws sts get-caller-identity &> /dev/null; then
  echo -e "${RED}❌ AWS credentials not configured${NC}"
  exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS configured (Account: $AWS_ACCOUNT_ID)"

if [ -z "$ODDS_API_KEY" ]; then
  echo -e "${YELLOW}⚠️  ODDS_API_KEY not set${NC}"
fi
if [ -z "$SNS_TOPIC_ARN" ]; then
  echo -e "${YELLOW}⚠️  SNS_TOPIC_ARN not set${NC}"
fi
if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &> /dev/null; then
  echo -e "${RED}❌ IAM role '$IAM_ROLE_NAME' not found${NC}"
  exit 1
fi
IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role exists"

echo "Running local smoke checks..."
python -m py_compile \
  "lambda/nba_rebounds_daily/lambda_function.py" \
  "scripts/rebounds_settle_runs.py"
bash -n "lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh"
echo "✅ Local smoke checks passed"
echo ""

echo "================================================================================"
echo "Step 2: Build and push image to ECR"
echo "================================================================================"
echo ""

if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" &> /dev/null; then
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" --output table
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"
docker buildx build \
  --platform "$DOCKER_PLATFORM" \
  --provenance=false \
  --load \
  -f "lambda/nba_rebounds_daily/Dockerfile" \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  .
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo "✅ Image pushed: $IMAGE_URI"
echo ""

echo "================================================================================"
echo "Step 3: Deploy Lambda"
echo "================================================================================"
echo ""

ENV_VARS="ODDS_API_KEY=$ODDS_API_KEY,SNS_TOPIC_ARN=$SNS_TOPIC_ARN,CONFIG_PATH=config/nba_rebounds_prod.lambda.yaml,SETTLE_BUCKET=nba-betting-mt,SETTLE_PREFIX=rebounds/daily_runs,SETTLE_DAYS_LAG=1,SETTLE_WINDOW_DAYS=1,SETTLE_ALL_TIME_DAYS=999999,SETTLE_MAX_UNMATCHED_BET_ROWS=0"

if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" &> /dev/null; then
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

echo "================================================================================"
echo "Step 4: EventBridge rules"
echo "================================================================================"
echo ""

aws events put-rule \
  --name "$SCORE_RULE_NAME" \
  --schedule-expression "$CRON_SCORE_9AM_ET" \
  --state DISABLED \
  --description "Trigger NBA rebounds daily pipeline at 9:00am ET target" \
  --region "$REGION" \
  --output table

aws events put-rule \
  --name "$SETTLE_RULE_NAME" \
  --schedule-expression "$CRON_SETTLE_905AM_ET" \
  --state ENABLED \
  --description "Trigger NBA rebounds settlement at 9:05am ET target" \
  --region "$REGION" \
  --output table

LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" --query 'Configuration.FunctionArn' --output text)

# Keep EventBridge invoke permission in sync with current rule ARNs.
aws lambda remove-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id EventBridgeInvokeNBAReboundsDailyScore \
  --region "$REGION" \
  --output text 2>/dev/null || true

aws lambda remove-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id EventBridgeInvokeNBAReboundsDailySettle \
  --region "$REGION" \
  --output text 2>/dev/null || true

aws lambda add-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id EventBridgeInvokeNBAReboundsDailyScore \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$SCORE_RULE_NAME" \
  --region "$REGION" \
  --output text

aws lambda add-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id EventBridgeInvokeNBAReboundsDailySettle \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$SETTLE_RULE_NAME" \
  --region "$REGION" \
  --output text

aws events put-targets \
  --rule "$SCORE_RULE_NAME" \
  --targets "[{\"Id\":\"1\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{\\\"mode\\\":\\\"pipeline\\\"}\"}]" \
  --region "$REGION" \
  --output table

aws events put-targets \
  --rule "$SETTLE_RULE_NAME" \
  --targets "[{\"Id\":\"1\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{\\\"mode\\\":\\\"settlement\\\"}\"}]" \
  --region "$REGION" \
  --output table

SCORE_RULE_ARN=$(aws events describe-rule --name "$SCORE_RULE_NAME" --region "$REGION" --query 'Arn' --output text)
SETTLE_RULE_ARN=$(aws events describe-rule --name "$SETTLE_RULE_NAME" --region "$REGION" --query 'Arn' --output text)

echo -e "${GREEN}✅ EventBridge score rule: $CRON_SCORE_9AM_ET (9am ET target)${NC}"
echo -e "${GREEN}✅ EventBridge settle rule: $CRON_SETTLE_905AM_ET (9:05am ET target)${NC}"
echo "Score rule ARN: $SCORE_RULE_ARN"
echo "Settle rule ARN: $SETTLE_RULE_ARN"
echo ""

echo "================================================================================"
echo "Step 5: Lambda dry-run invoke"
echo "================================================================================"
echo ""

aws lambda wait function-active-v2 \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION"

aws lambda invoke \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --invocation-type DryRun \
  --payload '{"mode":"settlement"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/nba_rebounds_daily_dryrun.json \
  --output table
rm -f /tmp/nba_rebounds_daily_dryrun.json
echo "✅ Lambda DryRun accepted (no side effects)"
echo ""

echo "================================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "================================================================================"
echo "Lambda: $LAMBDA_NAME"
echo "Score rule: $SCORE_RULE_NAME ($CRON_SCORE_9AM_ET)"
echo "Settle rule: $SETTLE_RULE_NAME ($CRON_SETTLE_905AM_ET)"
echo "Image: $IMAGE_URI"
echo ""
