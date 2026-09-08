#!/bin/bash
###############################################################################
# Deploy MLB TB Prop Snapshot Lambda via container image (ECR)
#
# EventBridge rule starts DISABLED — enable manually after smoke test passes:
#   aws events enable-rule --name mlb-tb-prop-snapshot-hourly --region us-east-2
#
# Env vars needed at deploy time:
#   ODDS_API_KEY  (required — no default)
#   SES_SOURCE    (default: tqstrats@gmail.com)
#   SES_TO        (default: mylescgthomas@gmail.com)
#   SNS_TOPIC_ARN (default: arn:aws:sns:us-east-2:232692785472:betting-arb-alerts)
#   S3_BUCKET     (default: the-odds-api-mt)
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   cd ~/dev/betting && bash src/mlb_total_bases_modeling/lambda/deploy_mlb_tb_snapshot.sh
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="mlb-tb-prop-snapshot"
ECR_REPO_NAME="mlb-tb-prop-snapshot"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

RULE_HOURLY="mlb-tb-prop-snapshot-hourly"
CRON_HOURLY="cron(0 * * * ? *)"    # top of every hour, all day

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY MLB TB PROP SNAPSHOT LAMBDA (CONTAINER)"
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
[ -z "$SNS_TOPIC_ARN" ] && echo -e "${YELLOW}⚠️  SNS_TOPIC_ARN not set (will use default)${NC}"
[ -z "${SES_SOURCE:-}" ] && echo -e "${YELLOW}⚠️  SES_SOURCE not set (will use default)${NC}"

IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role: $IAM_ROLE_ARN"

echo "Running syntax checks..."
python -m py_compile \
  "src/mlb_total_bases_modeling/lambda/lambda_snapshot.py" \
  "src/mlb_total_bases_modeling/scripts/snapshot_props.py" \
  "src/mlb_total_bases_modeling/scripts/verify_snapshot.py" \
  "src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py" \
  "src/mlb_total_bases_modeling/scripts/compute_clv.py" \
  "src/mlb_total_bases_modeling/scripts/compute_tightening.py" \
  "src/mlb_total_bases_modeling/scripts/verify_clv.py"
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
  -f "src/mlb_total_bases_modeling/lambda/Dockerfile.snapshot" \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  .
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$IMAGE_URI"
echo "Pushing to ECR (retries on network drop)..."
for attempt in 1 2 3 4 5; do
  docker push "$IMAGE_URI" && break
  echo -e "${YELLOW}⚠️  Push attempt $attempt failed, retrying in 5s...${NC}"
  sleep 5
  [ "$attempt" -eq 5 ] && { echo -e "${RED}❌ Push failed after 5 attempts${NC}"; exit 1; }
done
echo -e "${GREEN}✅ Image pushed: $IMAGE_URI${NC}"

# ── Deploy Lambda ─────────────────────────────────────────────────────────────
echo ""
echo "Step 3: Deploy Lambda..."
# SES addresses and SNS ARN are non-secret; hardcoded defaults so they're never
# accidentally dropped. Override by exporting vars in the shell before running.
_SES_SOURCE="${SES_SOURCE:-tqstrats@gmail.com}"
_SES_TO="${SES_TO:-mylescgthomas@gmail.com}"
_SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-2:232692785472:betting-arb-alerts}"
_S3_BUCKET="${S3_BUCKET:-the-odds-api-mt}"
ENV_VARS="ODDS_API_KEY=${ODDS_API_KEY},SES_SOURCE=${_SES_SOURCE},SES_TO=${_SES_TO},SNS_TOPIC_ARN=${_SNS_TOPIC_ARN},S3_BUCKET=${_S3_BUCKET}"

if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" &>/dev/null; then
  aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" --image-uri "$IMAGE_URI" --region "$REGION" --output table
  aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$LAMBDA_NAME" --timeout 300 --memory-size 512 \
    --environment "Variables={$ENV_VARS}" --region "$REGION" --output table
else
  aws lambda create-function \
    --function-name "$LAMBDA_NAME" --package-type Image \
    --code ImageUri="$IMAGE_URI" --role "$IAM_ROLE_ARN" \
    --timeout 300 --memory-size 512 \
    --environment "Variables={$ENV_VARS}" --region "$REGION" --output table
fi
aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"
echo -e "${GREEN}✅ Lambda deployed: $LAMBDA_NAME${NC}"

# ── EventBridge rule ─────────────────────────────────────────────────────────
echo ""
echo "Step 4: EventBridge rule..."
RULE_WAS_ENABLED=$(aws events describe-rule --name "$RULE_HOURLY" --region "$REGION" \
  --query 'State' --output text 2>/dev/null || echo "DISABLED")
echo "  Rule state before deploy: $RULE_WAS_ENABLED"

# Preserve the prior state atomically — put-rule sets state in one call so a
# second rapid deploy never sees a transient DISABLED and skips re-enable.
RULE_TARGET_STATE="DISABLED"
[ "$RULE_WAS_ENABLED" = "ENABLED" ] && RULE_TARGET_STATE="ENABLED"

LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$LAMBDA_NAME" --region "$REGION" \
  --query 'Configuration.FunctionArn' --output text)

aws events put-rule --name "$RULE_HOURLY" \
  --schedule-expression "$CRON_HOURLY" --state "$RULE_TARGET_STATE" \
  --description "MLB TB prop snapshot: hourly" --region "$REGION" --output table

STMT_ID="EventBridgeInvoke${RULE_HOURLY//[-]/_}"
aws lambda remove-permission --function-name "$LAMBDA_NAME" \
  --statement-id "$STMT_ID" --region "$REGION" --output text 2>/dev/null || true
aws lambda add-permission --function-name "$LAMBDA_NAME" \
  --statement-id "$STMT_ID" --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$RULE_HOURLY" \
  --region "$REGION" --output text
aws events put-targets --rule "$RULE_HOURLY" --region "$REGION" \
  --targets "[{\"Id\":\"1\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{}\"}]" \
  --output table

echo -e "${GREEN}✅ $RULE_HOURLY — $CRON_HOURLY ($RULE_TARGET_STATE)${NC}"

# ── DryRun check ──────────────────────────────────────────────────────────────
echo ""
echo "Step 5: DryRun invoke..."
aws lambda invoke \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --invocation-type DryRun \
  --payload '{}' \
  --cli-binary-format raw-in-base64-out \
  /dev/null \
  --output table
echo -e "${GREEN}✅ DryRun passed${NC}"

# ── Real invoke + smoke test ─────────────────────────────────────────────────
echo ""
echo "Step 6: Real invoke + smoke test..."
# Use async (Event) so the CLI never times out and retries. Lambda runs up to 5 min.
aws lambda invoke \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --invocation-type Event \
  --payload '{}' \
  --cli-binary-format raw-in-base64-out \
  /dev/null \
  --output table
echo "Lambda invoked async — waiting 45s for it to complete before smoke test..."
sleep 45

echo ""
echo "Running smoke test..."
uv run --system-certs python src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py
echo -e "${GREEN}✅ Smoke test passed${NC}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "================================================================================"
echo "Lambda  : $LAMBDA_NAME"
echo "Image   : $IMAGE_URI"
echo ""
if [ "$RULE_TARGET_STATE" != "ENABLED" ]; then
  echo "Rule is DISABLED — enable when ready:"
  echo "  aws events enable-rule --name $RULE_HOURLY --region $REGION"
fi
echo ""
