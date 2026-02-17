#!/bin/bash
###############################################################################
# Deploy NCAAB + NBA Live Odds Tracking Lambdas
#
# Context:
# - Deploys 2 Lambda functions using the SAME code
# - Lambda 1: NBA (DEFAULT_SPORT=nba)
# - Lambda 2: NCAAB (DEFAULT_SPORT=ncaab)
# - Both run every 1 minute via EventBridge
#
# Prerequisites:
# - AWS CLI configured with credentials
# - IAM role: betting-dashboard-daily-update-role-ille2llh (already exists)
# - ODDS_API_KEY environment variable set
#
# Usage:
#   export ODDS_API_KEY="your-key-here"
#   bash scripts/deploy_live_odds_lambdas.sh
#
# Created: 2026-02-16
###############################################################################

set -e  # Exit on error
export AWS_PAGER=""  # Disable AWS CLI pager for non-interactive execution

# Configuration
REGION="us-east-2"
IAM_ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
LAMBDA_NBA="track-live-odds-nba-per-minute"
LAMBDA_NCAAB="track-live-odds-ncaab-per-minute"
EVENTBRIDGE_NBA="track-live-odds-nba-every-minute"
EVENTBRIDGE_NCAAB="track-live-odds-ncaab-every-minute"
RUNTIME="python3.11"

# AWS SDK for pandas Layer (includes pandas, numpy, pyarrow, boto3, s3fs)
# See: https://aws-sdk-pandas.readthedocs.io/en/stable/layers.html
PANDAS_LAYER_ARN="arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:25"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "DEPLOY LIVE ODDS TRACKING LAMBDAS"
echo "================================================================================"
echo ""

###############################################################################
# STEP 1: Verify Prerequisites
###############################################################################

echo "📋 Step 1: Verifying prerequisites..."
echo ""

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Install it first.${NC}"
    exit 1
fi
echo "✅ AWS CLI found"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS credentials configured (Account: $AWS_ACCOUNT_ID)"

# Check ODDS_API_KEY
if [ -z "$ODDS_API_KEY" ]; then
    echo -e "${RED}❌ ODDS_API_KEY environment variable not set${NC}"
    echo "   Run: export ODDS_API_KEY='your-key-here'"
    exit 1
fi
echo "✅ ODDS_API_KEY environment variable set"

# Verify IAM role exists
echo ""
echo "🔍 Verifying IAM role: $IAM_ROLE_NAME..."
if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &> /dev/null; then
    echo -e "${RED}❌ IAM role '$IAM_ROLE_NAME' not found${NC}"
    exit 1
fi
echo "✅ IAM role exists"

# Get role ARN
IAM_ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_NAME" --query 'Role.Arn' --output text)
echo "   ARN: $IAM_ROLE_ARN"

# Verify required policies are attached
echo ""
echo "🔍 Verifying IAM policies..."
ATTACHED_POLICIES=$(aws iam list-attached-role-policies --role-name "$IAM_ROLE_NAME" --query 'AttachedPolicies[].PolicyName' --output text)

if echo "$ATTACHED_POLICIES" | grep -q "AmazonS3FullAccess"; then
    echo "✅ AmazonS3FullAccess attached"
else
    echo -e "${YELLOW}⚠️  AmazonS3FullAccess not attached (needed for S3 writes)${NC}"
fi

if echo "$ATTACHED_POLICIES" | grep -q "AWSLambdaBasicExecutionRole"; then
    echo "✅ AWSLambdaBasicExecutionRole attached"
else
    echo -e "${YELLOW}⚠️  AWSLambdaBasicExecutionRole not attached (needed for CloudWatch logs)${NC}"
fi

echo ""
echo -e "${GREEN}✅ All prerequisites verified${NC}"
echo ""

###############################################################################
# STEP 2: Package Lambda Code (with minimal dependencies)
###############################################################################

echo "================================================================================"
echo "📦 Step 2: Packaging Lambda code (using layers for pandas/pyarrow)..."
echo "================================================================================"
echo ""

# Clean up old package
rm -f lambda_live_odds.zip
rm -rf package/

# Create package directory
mkdir -p package

# Install ONLY requests (pandas/pyarrow come from AWS layer)
echo "Installing requests (pandas/pyarrow will come from AWS SDK layer)..."
pip install -q --target ./package requests

# Add our code to package
echo "Adding application code..."
cp scripts/lambda_function_track_live_odds.py package/
mkdir -p package/src

# CRITICAL: Include BOTH team name mapping files
# NBA mapping normalizes "Los Angeles Clippers" → "LA Clippers" for ESPN matching
# NCAAB mapping normalizes "St" → "State", "Univ." → "University", etc.
# Without these, Lambda falls back to string matching which misses games
cp src/nba_team_name_mapping.py package/src/
cp src/ncaab_team_name_mapping.py package/src/

# Create deployment package
echo "Creating lambda_live_odds.zip..."
cd package
zip -q -r ../lambda_live_odds.zip .
cd ..

# Clean up
rm -rf package/

# Verify package created
if [ ! -f lambda_live_odds.zip ]; then
    echo -e "${RED}❌ Failed to create lambda_live_odds.zip${NC}"
    exit 1
fi

PACKAGE_SIZE=$(du -h lambda_live_odds.zip | cut -f1)
echo "✅ Package created: lambda_live_odds.zip ($PACKAGE_SIZE)"
echo ""

###############################################################################
# STEP 3: Deploy NBA Lambda
###############################################################################

echo "================================================================================"
echo "🏀 Step 3: Deploying NBA Lambda..."
echo "================================================================================"
echo ""

# Check if Lambda exists
if aws lambda get-function --function-name "$LAMBDA_NBA" --region "$REGION" &> /dev/null; then
    echo "Lambda '$LAMBDA_NBA' already exists - updating code..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_NBA" \
        --zip-file fileb://lambda_live_odds.zip \
        --region "$REGION" \
        --output table
    
    echo ""
    echo "Waiting for code update to complete..."
    # CRITICAL: Verify upload succeeded before proceeding
    # If update fails, we'll deploy broken code and waste time debugging
    if ! aws lambda wait function-updated \
        --function-name "$LAMBDA_NBA" \
        --region "$REGION"; then
        echo -e "${RED}❌ Lambda update failed for $LAMBDA_NBA${NC}"
        echo "Deployment aborted - check AWS console for errors"
        exit 1
    fi
    echo "✅ Code update complete and verified"
    
    echo ""
    echo "Updating configuration..."
    aws lambda update-function-configuration \
        --function-name "$LAMBDA_NBA" \
        --runtime "$RUNTIME" \
        --handler lambda_function_track_live_odds.lambda_handler \
        --timeout 30 \
        --memory-size 512 \
        --layers "$PANDAS_LAYER_ARN" \
        --environment "Variables={DEFAULT_SPORT=nba,ODDS_API_KEY=$ODDS_API_KEY,TRACK_UPCOMING_GAMES=false}" \
        --region "$REGION" \
        --output table
else
    echo "Creating Lambda '$LAMBDA_NBA'..."
    aws lambda create-function \
        --function-name "$LAMBDA_NBA" \
        --runtime "$RUNTIME" \
        --role "$IAM_ROLE_ARN" \
        --handler lambda_function_track_live_odds.lambda_handler \
        --zip-file fileb://lambda_live_odds.zip \
        --timeout 30 \
        --memory-size 512 \
        --layers "$PANDAS_LAYER_ARN" \
        --environment "Variables={DEFAULT_SPORT=nba,ODDS_API_KEY=$ODDS_API_KEY,TRACK_UPCOMING_GAMES=false}" \
        --region "$REGION" \
        --output table
fi

echo ""
echo -e "${GREEN}✅ NBA Lambda deployed${NC}"
echo ""

###############################################################################
# STEP 4: Deploy NCAAB Lambda
###############################################################################

echo "================================================================================"
echo "🏀 Step 4: Deploying NCAAB Lambda..."
echo "================================================================================"
echo ""

# Check if Lambda exists
if aws lambda get-function --function-name "$LAMBDA_NCAAB" --region "$REGION" &> /dev/null; then
    echo "Lambda '$LAMBDA_NCAAB' already exists - updating code..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_NCAAB" \
        --zip-file fileb://lambda_live_odds.zip \
        --region "$REGION" \
        --output table
    
    echo ""
    echo "Waiting for code update to complete..."
    # CRITICAL: Verify upload succeeded before proceeding
    # NCAAB Lambda has larger memory/timeout due to more games to process
    if ! aws lambda wait function-updated \
        --function-name "$LAMBDA_NCAAB" \
        --region "$REGION"; then
        echo -e "${RED}❌ Lambda update failed for $LAMBDA_NCAAB${NC}"
        echo "Deployment aborted - check AWS console for errors"
        exit 1
    fi
    echo "✅ Code update complete and verified"
    
    echo ""
    echo "Updating configuration..."
    aws lambda update-function-configuration \
        --function-name "$LAMBDA_NCAAB" \
        --runtime "$RUNTIME" \
        --handler lambda_function_track_live_odds.lambda_handler \
        --timeout 45 \
        --memory-size 1024 \
        --layers "$PANDAS_LAYER_ARN" \
        --environment "Variables={DEFAULT_SPORT=ncaab,ODDS_API_KEY=$ODDS_API_KEY,TRACK_UPCOMING_GAMES=false}" \
        --region "$REGION" \
        --output table
else
    echo "Creating Lambda '$LAMBDA_NCAAB'..."
    aws lambda create-function \
        --function-name "$LAMBDA_NCAAB" \
        --runtime "$RUNTIME" \
        --role "$IAM_ROLE_ARN" \
        --handler lambda_function_track_live_odds.lambda_handler \
        --zip-file fileb://lambda_live_odds.zip \
        --timeout 45 \
        --memory-size 1024 \
        --layers "$PANDAS_LAYER_ARN" \
        --environment "Variables={DEFAULT_SPORT=ncaab,ODDS_API_KEY=$ODDS_API_KEY,TRACK_UPCOMING_GAMES=false}" \
        --region "$REGION" \
        --output table
fi

echo ""
echo -e "${GREEN}✅ NCAAB Lambda deployed${NC}"
echo ""

###############################################################################
# STEP 5: Create EventBridge Rules
###############################################################################

echo "================================================================================"
echo "⏰ Step 5: Setting up EventBridge rules..."
echo "================================================================================"
echo ""

# NBA EventBridge rule
echo "Creating/updating EventBridge rule: $EVENTBRIDGE_NBA..."
aws events put-rule \
    --name "$EVENTBRIDGE_NBA" \
    --schedule-expression "rate(1 minute)" \
    --state ENABLED \
    --description "Trigger NBA live odds tracking every minute" \
    --region "$REGION" \
    --output table

echo ""
echo "✅ NBA EventBridge rule created"
echo ""

# NCAAB EventBridge rule
echo "Creating/updating EventBridge rule: $EVENTBRIDGE_NCAAB..."
aws events put-rule \
    --name "$EVENTBRIDGE_NCAAB" \
    --schedule-expression "rate(1 minute)" \
    --state ENABLED \
    --description "Trigger NCAAB live odds tracking every minute" \
    --region "$REGION" \
    --output table

echo ""
echo "✅ NCAAB EventBridge rule created"
echo ""

###############################################################################
# STEP 6: Connect EventBridge to Lambdas
###############################################################################

echo "================================================================================"
echo "🔗 Step 6: Connecting EventBridge rules to Lambdas..."
echo "================================================================================"
echo ""

# Get Lambda ARNs
NBA_LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NBA" --region "$REGION" --query 'Configuration.FunctionArn' --output text)
NCAAB_LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NCAAB" --region "$REGION" --query 'Configuration.FunctionArn' --output text)

echo "NBA Lambda ARN: $NBA_LAMBDA_ARN"
echo "NCAAB Lambda ARN: $NCAAB_LAMBDA_ARN"
echo ""

# Add EventBridge permissions to NBA Lambda
echo "Adding EventBridge invoke permission to NBA Lambda..."
aws lambda add-permission \
    --function-name "$LAMBDA_NBA" \
    --statement-id EventBridgeInvokeNBA \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$EVENTBRIDGE_NBA" \
    --region "$REGION" \
    --output text 2>/dev/null || echo "  (Permission already exists)"

# Add EventBridge permissions to NCAAB Lambda
echo "Adding EventBridge invoke permission to NCAAB Lambda..."
aws lambda add-permission \
    --function-name "$LAMBDA_NCAAB" \
    --statement-id EventBridgeInvokeNCAB \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$EVENTBRIDGE_NCAAB" \
    --region "$REGION" \
    --output text 2>/dev/null || echo "  (Permission already exists)"

echo ""

# Link NBA rule to Lambda
echo "Linking NBA EventBridge rule to Lambda..."
aws events put-targets \
    --rule "$EVENTBRIDGE_NBA" \
    --targets "Id"="1","Arn"="$NBA_LAMBDA_ARN" \
    --region "$REGION" \
    --output table

echo ""

# Link NCAAB rule to Lambda
echo "Linking NCAAB EventBridge rule to Lambda..."
aws events put-targets \
    --rule "$EVENTBRIDGE_NCAAB" \
    --targets "Id"="1","Arn"="$NCAAB_LAMBDA_ARN" \
    --region "$REGION" \
    --output table

echo ""
echo -e "${GREEN}✅ EventBridge rules connected to Lambdas${NC}"
echo ""

###############################################################################
# STEP 7: Test Invocation
###############################################################################

echo "================================================================================"
echo "🧪 Step 7: Testing Lambda invocations..."
echo "================================================================================"
echo ""

echo "Testing NBA Lambda..."
# Invoke Lambda and capture response + logs
# Using --log-type Tail to verify deployment worked and see any warnings
INVOKE_RESULT=$(aws lambda invoke \
    --function-name "$LAMBDA_NBA" \
    --region "$REGION" \
    --log-type Tail \
    --query 'LogResult' \
    --output text \
    response_nba.json | base64 --decode)

# Check for the missing mapping warning (should be gone after fix)
if echo "$INVOKE_RESULT" | grep -q "WARNING.*nba_team_name_mapping.py not found"; then
    echo -e "${YELLOW}⚠️  WARNING: NBA team mapping file missing in package${NC}"
    echo "This will cause team matching failures - check package contents"
fi

echo "$INVOKE_RESULT" | tail -20

echo ""
echo "NBA Lambda response saved to: response_nba.json"
echo ""

echo "Testing NCAAB Lambda..."
# Same verification for NCAAB
INVOKE_RESULT=$(aws lambda invoke \
    --function-name "$LAMBDA_NCAAB" \
    --region "$REGION" \
    --log-type Tail \
    --query 'LogResult' \
    --output text \
    response_ncaab.json | base64 --decode)

# Check for the missing mapping warning
if echo "$INVOKE_RESULT" | grep -q "WARNING.*ncaab_team_name_mapping.py not found"; then
    echo -e "${YELLOW}⚠️  WARNING: NCAAB team mapping file missing in package${NC}"
    echo "This will cause team matching failures - check package contents"
fi

echo "$INVOKE_RESULT" | tail -20

echo ""
echo "NCAAB Lambda response saved to: response_ncaab.json"
echo ""

###############################################################################
# SUMMARY
###############################################################################

echo "================================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "📊 Summary:"
echo "  • NBA Lambda: $LAMBDA_NBA"
echo "  • NCAAB Lambda: $LAMBDA_NCAAB"
echo "  • Both running every 1 minute via EventBridge"
echo "  • Region: $REGION"
echo "  • IAM Role: $IAM_ROLE_NAME"
echo ""
echo "📁 S3 Output:"
echo "  • NBA:   s3://nba-betting-mt/data/01_input/live_odds/{the-odds-api,espn}/"
echo "  • NCAAB: s3://ncaab-betting-mt/data/01_input/live_odds/{the-odds-api,espn}/"
echo ""
echo "📝 Next Steps:"
echo "  1. Monitor CloudWatch logs:"
echo "     aws logs tail /aws/lambda/$LAMBDA_NBA --follow"
echo "     aws logs tail /aws/lambda/$LAMBDA_NCAAB --follow"
echo ""
echo "  2. Check S3 for data files (during live games):"
echo "     aws s3 ls s3://nba-betting-mt/data/01_input/live_odds/the-odds-api/"
echo "     aws s3 ls s3://ncaab-betting-mt/data/01_input/live_odds/the-odds-api/"
echo ""
echo "  3. Disable during off-season:"
echo "     aws events disable-rule --name $EVENTBRIDGE_NCAAB  # Disable NCAAB (April-Nov)"
echo "     aws events enable-rule --name $EVENTBRIDGE_NCAAB   # Re-enable NCAAB (Nov)"
echo ""
echo "================================================================================"
