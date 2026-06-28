#!/bin/bash
###############################################################################
# Deploy NFL Sacks Daily Lambda via container image (ECR)
#
# EventBridge schedule (all rules start DISABLED — enable before week 1):
#   Tue 9am ET  → spine_update   (rebuild + upload spine from nfl_data_py)
#   Wed 9am ET  → spine_verify   (re-pull + compare vs S3 spine)
#   Thu 9am ET  → pipeline       (Thursday Night Football)
#   Sun 9am ET  → pipeline       (Sunday games)
#   Mon 9am ET  → pipeline       (Monday Night Football)
#
# Cron expressions use 14:00 UTC (= 9am EST, = 10am EDT during summer).
# NFL regular season runs Sep–Jan, predominantly EST. The +1hr during EDT
# (Sep–Oct) means props are fetched by 10am local — still early enough.
#
# Usage:
#   export ODDS_API_KEY="<key>"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:<acct>:betting-arb-alerts"
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
RULE_SPINE_UPDATE="nfl-sacks-spine-update-tue-9am-et"
RULE_SPINE_VERIFY="nfl-sacks-spine-verify-wed-9am-et"
RULE_PIPELINE_THU="nfl-sacks-pipeline-thu-9am-et"
RULE_PIPELINE_SUN="nfl-sacks-pipeline-sun-9am-et"
RULE_PIPELINE_MON="nfl-sacks-pipeline-mon-9am-et"
RULE_SETTLE_DAILY="nfl-sacks-settle-daily-10am-et"

# Pipeline at 8am ET = 12:00 UTC (covers London games which kick off ~9:30am ET)
# During EDT (Sep-Nov): 12:00 UTC = 8am ET  ← London window
# During EST (Dec-Jan): 12:00 UTC = 7am ET  ← fine, domestic kickoffs not until 1pm ET
# Spine/verify at 9am ET = 14:00 UTC (no game-time constraint, keeps Tue settle gap clean)
# Settle at 10am ET = 15:00 UTC (always after spine_update finishes on Tue)
CRON_TUE="cron(0 14 ? * TUE *)"         # 9am ET  — spine_update
CRON_WED="cron(0 14 ? * WED *)"         # 9am ET  — spine_verify
CRON_THU="cron(0 12 ? * THU *)"         # 8am ET  — pipeline (TNF)
CRON_SUN="cron(0 12 ? * SUN *)"         # 8am ET  — pipeline (covers London + domestic)
CRON_MON="cron(0 12 ? * MON *)"         # 8am ET  — pipeline (MNF)
CRON_DAILY_10AM="cron(0 15 * * ? *)"    # 10am ET — settle (every day)

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

[ -z "$ODDS_API_KEY" ]  && echo -e "${YELLOW}⚠️  ODDS_API_KEY not set${NC}"
[ -z "$SNS_TOPIC_ARN" ] && echo -e "${YELLOW}⚠️  SNS_TOPIC_ARN not set${NC}"
[ -z "${SES_SOURCE:-}" ] && echo -e "${YELLOW}⚠️  SES_SOURCE not set (HTML emails disabled)${NC}"
[ -z "${SES_TO:-}" ]     && echo -e "${YELLOW}⚠️  SES_TO not set (HTML emails disabled)${NC}"

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

# Compute current NFL season if not overridden
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

declare -A RULES
RULES["$RULE_SPINE_UPDATE"]="$CRON_TUE|spine_update|Tue 9am ET spine update"
RULES["$RULE_SPINE_VERIFY"]="$CRON_WED|spine_verify|Wed 9am ET spine verify"
RULES["$RULE_PIPELINE_THU"]="$CRON_THU|pipeline|Thu 8am ET (TNF) pipeline"
RULES["$RULE_PIPELINE_SUN"]="$CRON_SUN|pipeline|Sun 8am ET pipeline (incl. London)"
RULES["$RULE_PIPELINE_MON"]="$CRON_MON|pipeline|Mon 8am ET (MNF) pipeline"
RULES["$RULE_SETTLE_DAILY"]="$CRON_DAILY_10AM|settle|Daily 10am ET settlement email"

for RULE_NAME in "${!RULES[@]}"; do
  IFS='|' read -r CRON MODE DESC <<< "${RULES[$RULE_NAME]}"

  aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "$CRON" \
    --state DISABLED \
    --description "NFL sacks: $DESC" \
    --region "$REGION" \
    --output table

  # Remove + re-add permission (idempotent)
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
echo "  $RULE_SPINE_UPDATE  →  $CRON_TUE"
echo "  $RULE_SPINE_VERIFY  →  $CRON_WED"
echo "  $RULE_PIPELINE_THU  →  $CRON_THU"
echo "  $RULE_PIPELINE_SUN  →  $CRON_SUN"
echo "  $RULE_PIPELINE_MON  →  $CRON_MON"
echo "  $RULE_SETTLE_DAILY  →  $CRON_DAILY_10AM"
echo ""
echo "Pre-season checklist:"
echo "  1. Verify SES identity in AWS console (SES_SOURCE must be a verified address)"
echo "  2. Run fit_model.py to upload trained LR to S3"
echo "  3. Run update_spine.py --season 2026 to upload initial spine to S3"
echo "  4. Enable EventBridge rules in AWS console (or via aws events enable-rule)"
echo "  5. Test pipeline:  aws lambda invoke --payload '{\"mode\":\"pipeline\"}' ..."
echo "  6. Test settle:    aws lambda invoke --payload '{\"mode\":\"settle\"}' ..."
echo ""
