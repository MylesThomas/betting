#!/bin/bash
set -euo pipefail

FUNCTION_NAME="levels-fyi-daily-scraper"
REGION="us-east-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$FUNCTION_NAME"
IAM_ROLE="arn:aws:iam::$ACCOUNT_ID:role/service-role/betting-dashboard-daily-update-role-ille2llh"
S3_BUCKET="levels-fyi-mt"

echo "=== Levels.fyi daily scraper deploy ==="
echo "Account: $ACCOUNT_ID  Region: $REGION"

# ── ECR auth ──────────────────────────────────────────────────────────────────
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# ── create ECR repo if needed ─────────────────────────────────────────────────
aws ecr describe-repositories --repository-names "$FUNCTION_NAME" --region "$REGION" \
  2>/dev/null || \
  aws ecr create-repository --repository-name "$FUNCTION_NAME" --region "$REGION"

# ── build + push ──────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
docker buildx build --platform linux/amd64 --provenance=false -t "$ECR_REPO:latest" . --push
echo "Image pushed: $ECR_REPO:latest"

# ── create or update Lambda ───────────────────────────────────────────────────
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" &>/dev/null; then
  echo "Updating existing Lambda..."
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$ECR_REPO:latest" \
    --region "$REGION"
  aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --timeout 300 \
    --memory-size 512 \
    --environment "Variables={S3_BUCKET=$S3_BUCKET,SSL_VERIFY=true}" \
    --region "$REGION"
else
  echo "Creating new Lambda..."
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$ECR_REPO:latest" \
    --role "$IAM_ROLE" \
    --timeout 300 \
    --memory-size 512 \
    --environment "Variables={S3_BUCKET=$S3_BUCKET,SSL_VERIFY=true}" \
    --region "$REGION"
  aws lambda wait function-active --function-name "$FUNCTION_NAME" --region "$REGION"
fi

# ── EventBridge rule: 7am EST daily (12:00 UTC) ───────────────────────────────
RULE_NAME="levels-fyi-daily-0700-est"
RULE_ARN=$(aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "cron(0 12 * * ? *)" \
  --state ENABLED \
  --region "$REGION" \
  --query RuleArn --output text)
echo "EventBridge rule: $RULE_ARN"

LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION" \
  --query Configuration.FunctionArn --output text)

aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id "levels-fyi-eventbridge" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "$RULE_ARN" \
  --region "$REGION" 2>/dev/null || true

aws events put-targets \
  --rule "$RULE_NAME" \
  --targets "Id=1,Arn=$LAMBDA_ARN" \
  --region "$REGION"

echo ""
echo "=== Deploy complete ==="
echo "Lambda:       $FUNCTION_NAME"
echo "Trigger:      cron(0 12 * * ? *)  — 7am EST / 8am EDT"
echo "S3 bucket:    s3://$S3_BUCKET/"
echo ""
echo "Next: create the S3 bucket if it doesn't exist:"
echo "  aws s3 mb s3://$S3_BUCKET --region $REGION"
echo ""
echo "After 1 week stable, disable local cron in run_scraper.sh"
