# Lambda Deployment Fixes Needed

## Issue 1: Missing NBA team mapping file

**Problem:**
```
⚠️  WARNING: nba_team_name_mapping.py not found - using fallback normalization
```

**Root cause:**
Line 138 in `scripts/deploy_live_odds_lambdas.sh` only copies NCAAB mapping:
```bash
cp src/ncaab_team_name_mapping.py package/src/
```

**Fix:**
Add NBA mapping too:
```bash
cp src/ncaab_team_name_mapping.py package/src/
cp src/nba_team_name_mapping.py package/src/
```

**Location:** Line 138 in `scripts/deploy_live_odds_lambdas.sh`

---

## Issue 2: No error handling for failed uploads

**Problem:**
Lambda update could fail silently - no verification that code actually uploaded.

**Fix options:**

### Option A: Check function update status (simple)
```bash
# After update-function-code
if ! aws lambda wait function-updated --function-name "$LAMBDA_NBA" --region "$REGION"; then
    echo "❌ Lambda update failed for $LAMBDA_NBA"
    exit 1
fi
echo "✅ Code update complete and verified"
```

### Option B: Verify code SHA256 (robust)
```bash
# Before upload
LOCAL_SHA=$(sha256sum lambda_live_odds.zip | cut -d' ' -f1)

# After upload
DEPLOYED_SHA=$(aws lambda get-function --function-name "$LAMBDA_NBA" --query 'Configuration.CodeSha256' --output text)

if [ "$LOCAL_SHA" != "$DEPLOYED_SHA" ]; then
    echo "❌ Upload verification failed - checksums don't match"
    exit 1
fi
echo "✅ Upload verified (SHA256: $DEPLOYED_SHA)"
```

### Option C: Use `--publish` flag and verify version (best)
```bash
# Update function and publish new version
VERSION=$(aws lambda update-function-code \
    --function-name "$LAMBDA_NBA" \
    --zip-file fileb://lambda_live_odds.zip \
    --publish \
    --region "$REGION" \
    --query 'Version' \
    --output text)

if [ -z "$VERSION" ]; then
    echo "❌ Failed to deploy - no version returned"
    exit 1
fi

echo "✅ Deployed version: $VERSION"
```

**Recommended:** Option A (already in script at line 179-181, just need to add error check)

---

## Issue 3: Silent failures on EventBridge permission errors

**Problem:**
EventBridge rules might not have permission to invoke Lambda.

**Fix:**
Add permission verification:
```bash
# After adding Lambda permission
echo "Verifying EventBridge can invoke Lambda..."
if aws lambda get-policy --function-name "$LAMBDA_NBA" 2>/dev/null | grep -q "$EVENTBRIDGE_NBA"; then
    echo "✅ EventBridge permission verified"
else
    echo "❌ EventBridge permission not set correctly"
    exit 1
fi
```

---

## Implementation Plan

1. **Fix missing NBA mapping** (5 min)
   - Edit line 138-139 in deploy script
   - Add `cp src/nba_team_name_mapping.py package/src/`

2. **Add upload verification** (10 min)
   - Wrap `aws lambda wait function-updated` in error check
   - Exit if upload fails

3. **Test locally** (5 min)
   ```bash
   # Package only (don't deploy)
   # Manually inspect lambda_live_odds.zip contents
   unzip -l lambda_live_odds.zip | grep mapping
   ```

4. **Deploy and verify** (5 min)
   ```bash
   bash scripts/deploy_live_odds_lambdas.sh
   ```

---

## Quick Fixes for Tomorrow

```bash
# Fix 1: Add NBA mapping to package
# In scripts/deploy_live_odds_lambdas.sh, line 138-139
cp src/ncaab_team_name_mapping.py package/src/
cp src/nba_team_name_mapping.py package/src/      # ADD THIS LINE

# Fix 2: Add error check (line ~181)
echo "Waiting for code update to complete..."
if ! aws lambda wait function-updated --function-name "$LAMBDA_NBA" --region "$REGION"; then
    echo "❌ Lambda update failed"
    exit 1
fi
echo "✅ Code update complete and verified"

# Same for NCAAB Lambda (line ~238)
```

---

## Verification After Fix

```bash
# Check package contents
unzip -l lambda_live_odds.zip | grep -E "(nba|ncaab)_team_name_mapping"

# Expected output:
#   src/nba_team_name_mapping.py
#   src/ncaab_team_name_mapping.py

# Test Lambda invocation
aws lambda invoke --function-name track-live-odds-nba-per-minute response.json
cat response.json | jq

# Should NOT see warning about missing mapping file
```
