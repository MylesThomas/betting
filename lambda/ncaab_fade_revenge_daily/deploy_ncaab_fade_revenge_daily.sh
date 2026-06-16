#!/bin/bash
###############################################################################
# Deploy NCAAB Fade Revenge Spot Lambda (daily 9am ET)
#
# Lambda: runs fetch scripts for yesterday (lines + game results), then
# finds rematch spots and writes plays to S3; sends SNS email.
#
# Prerequisites:
# - AWS CLI configured
# - IAM role: betting-dashboard-daily-update-role-ille2llh
# - ODDS_API_KEY (required for fetches + today's events)
# - SNS_TOPIC_ARN (optional): full ARN e.g. arn:aws:sns:us-east-2:232692785472:nba-props-alerts
#
# Usage:
#   export ODDS_API_KEY="your-key"
#   export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:YOUR_ACCOUNT_ID:your-topic-name"  # optional, use real account id
#   export NCAAB_PAUSE_UNTIL="2026-11-03"  # optional, skip runs before this date (YYYY-MM-DD)
#   cd ~/dev/betting && bash lambda/ncaab_fade_revenge_daily/deploy_ncaab_fade_revenge_daily.sh
#
# Verify:
#   python tmp/get_lambda_logs.py --lambda-function-name ncaab-fade-revenge-daily
# Per-book spread logs (Away @ Home, then each book's home spread) appear in CloudWatch
# after "Odds API lines: N". Deploy does a test invoke; tail shows last 40 log lines.
###############################################################################

set -e
export AWS_PAGER=""

REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NAME="ncaab-fade-revenge-daily"
EVENTBRIDGE_RULE="ncaab-fade-revenge-daily-9am-et"
RUNTIME="python3.11"

# 9am ET = 14:00 UTC (EST) / 13:00 UTC (EDT). Use 14:00 UTC for 9am ET in winter.
CRON_9AM_ET="cron(0 14 * * ? *)"

PANDAS_LAYER_ARN="arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:25"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "DEPLOY NCAAB FADE REVENGE DAILY LAMBDA"
echo "================================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

###############################################################################
# Step 1: Prerequisites
###############################################################################
echo "Step 1: Verifying prerequisites..."

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found${NC}"
    exit 1
fi
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS configured (Account: $AWS_ACCOUNT_ID)"

if [ -z "$ODDS_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  ODDS_API_KEY not set (fetches and today's events will fail)${NC}"
fi
echo "✅ ODDS_API_KEY set" || true

if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &> /dev/null; then
    echo -e "${RED}❌ IAM role '$IAM_ROLE_NAME' not found${NC}"
    exit 1
fi
IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "✅ IAM role exists"
echo ""

###############################################################################
# Step 2: Package Lambda (fetch scripts + src + config)
###############################################################################
echo "================================================================================"
echo "Step 2: Packaging Lambda code"
echo "================================================================================"
echo ""

rm -f lambda_ncaab_fade_revenge.zip
rm -rf package
mkdir -p package
mkdir -p package/src
mkdir -p package/scripts
mkdir -p package/config

# Dependencies: requests, pyyaml, python-dotenv. Pandas/numpy come from AWSSDKPandas layer;
# we set PYTHONPATH in handler so subprocess includes /opt/python and finds the layer.
pip install -q --target ./package requests pyyaml python-dotenv

# Handler and shared code
cp lambda_function.py package/
cp "$REPO_ROOT/src/config_loader.py" package/src/
cp "$REPO_ROOT/src/ncaab_team_name_mapping.py" package/src/
cp "$REPO_ROOT/src/ncaab_conference_data.py" package/src/
cp "$REPO_ROOT/src/ncaab_conference_inferred.py" package/src/

# Fetch scripts (run for yesterday inside Lambda)
cp "$REPO_ROOT/scripts/fetch_historical_ncaab_season_lines.py" package/scripts/
cp "$REPO_ROOT/scripts/fetch_historical_game_results_espn_api.py" package/scripts/

# Config so get_config() and find_project_root work in fetch scripts
cp "$REPO_ROOT/config/config.yaml" package/config/

# So find_project_root() in scripts finds package root (looks for .gitignore)
touch package/.gitignore

cd package
zip -q -r ../lambda_ncaab_fade_revenge.zip .
cd ..
rm -rf package

if [ ! -f lambda_ncaab_fade_revenge.zip ]; then
    echo -e "${RED}❌ Failed to create zip${NC}"
    exit 1
fi
echo "✅ Package created: lambda_ncaab_fade_revenge.zip ($(du -h lambda_ncaab_fade_revenge.zip | cut -f1))"
echo ""

###############################################################################
# Step 3: Deploy Lambda
###############################################################################
echo "================================================================================"
echo "Step 3: Deploy Lambda"
echo "================================================================================"
echo ""

ENV_VARS="ODDS_API_KEY=$ODDS_API_KEY"
if [ -n "$SNS_TOPIC_ARN" ]; then
    ENV_VARS="$ENV_VARS,SNS_TOPIC_ARN=$SNS_TOPIC_ARN"
fi
if [ -n "$NCAAB_PAUSE_UNTIL" ]; then
    ENV_VARS="$ENV_VARS,NCAAB_PAUSE_UNTIL=$NCAAB_PAUSE_UNTIL"
fi

if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" &> /dev/null; then
    echo "Updating existing Lambda..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_NAME" \
        --zip-file fileb://lambda_ncaab_fade_revenge.zip \
        --region "$REGION" \
        --output table
    echo "Waiting for update..."
    aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"
    echo "Updating configuration..."
    aws lambda update-function-configuration \
        --function-name "$LAMBDA_NAME" \
        --runtime "$RUNTIME" \
        --handler lambda_function.lambda_handler \
        --timeout 300 \
        --memory-size 1024 \
        --layers "$PANDAS_LAYER_ARN" \
        --environment "Variables={$ENV_VARS}" \
        --region "$REGION" \
        --output table
else
    echo "Creating Lambda..."
    aws lambda create-function \
        --function-name "$LAMBDA_NAME" \
        --runtime "$RUNTIME" \
        --role "$IAM_ROLE_ARN" \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://lambda_ncaab_fade_revenge.zip \
        --timeout 300 \
        --memory-size 1024 \
        --layers "$PANDAS_LAYER_ARN" \
        --environment "Variables={$ENV_VARS}" \
        --region "$REGION" \
        --output table
fi
echo -e "${GREEN}✅ Lambda deployed${NC}"
echo ""

###############################################################################
# Step 4: EventBridge rule (9am ET daily)
###############################################################################
echo "================================================================================"
echo "Step 4: EventBridge rule (9am ET daily)"
echo "================================================================================"
echo ""

aws events put-rule \
    --name "$EVENTBRIDGE_RULE" \
    --schedule-expression "$CRON_9AM_ET" \
    --state DISABLED \
    --description "Trigger NCAAB fade revenge Lambda daily at 9am ET" \
    --region "$REGION" \
    --output table

LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" --query 'Configuration.FunctionArn' --output text)

aws lambda add-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id EventBridgeInvokeFadeRevenge \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$EVENTBRIDGE_RULE" \
    --region "$REGION" \
    --output text 2>/dev/null || echo "  (Permission already exists)"

aws events put-targets \
    --rule "$EVENTBRIDGE_RULE" \
    --targets "Id"="1","Arn"="$LAMBDA_ARN" \
    --region "$REGION" \
    --output table

echo -e "${GREEN}✅ EventBridge rule set: $CRON_9AM_ET (9am ET)${NC}"
echo ""

###############################################################################
# Step 5: Test invocation
###############################################################################
echo "================================================================================"
echo "Step 5: Test invocation"
echo "================================================================================"
echo ""

aws lambda invoke \
    --function-name "$LAMBDA_NAME" \
    --region "$REGION" \
    --log-type Tail \
    --query 'LogResult' \
    --output text \
    response.json | base64 --decode | tail -40
echo ""
echo "Response: $(cat response.json)"
rm -f response.json
echo ""

###############################################################################
# Summary
###############################################################################
echo "================================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "Lambda: $LAMBDA_NAME"
echo "Schedule: $CRON_9AM_ET (9am ET daily)"
echo "Plays: s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/"
echo ""
echo "Check logs:"
echo "  python tmp/get_lambda_logs.py --lambda-function-name $LAMBDA_NAME"
echo ""
