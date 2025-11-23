# AWS Lambda Quick Reference

## Cron Expression for EventBridge

### Understanding the Cron Format
```
cron(Minutes Hours Day-of-month Month Day-of-week Year)
```

### For 7 AM ET Daily:
- **EST (Standard Time):** `cron(0 12 * * ? *)`
  - 12:00 PM UTC = 7:00 AM EST
  - Active: November - March

- **EDT (Daylight Time):** `cron(0 11 * * ? *)`
  - 11:00 AM UTC = 7:00 AM EDT
  - Active: March - November

### Recommended: Use Two Rules
1. **Rule 1:** Active November 1 - March 10
   - Cron: `cron(0 12 * * ? *)`
   
2. **Rule 2:** Active March 11 - October 31
   - Cron: `cron(0 11 * * ? *)`

**OR** just use one rule at 12:00 PM UTC and accept it runs at 7 AM EST / 8 AM EDT.

---

## Useful AWS CLI Commands

### Test Lambda Function
```bash
aws lambda invoke \
  --function-name betting-dashboard-daily-update \
  --payload '{}' \
  response.json

cat response.json
```

### View Latest Logs
```bash
aws logs tail /aws/lambda/betting-dashboard-daily-update --follow
```

### Update Lambda Code
```bash
# Zip your function code
zip -r function.zip lambda_function.py

# Update Lambda
aws lambda update-function-code \
  --function-name betting-dashboard-daily-update \
  --zip-file fileb://function.zip
```

### Check Lambda Execution History
```bash
aws lambda list-functions --query 'Functions[?FunctionName==`betting-dashboard-daily-update`]'
```

---

## Lambda Environment Variables

Set these in Lambda Configuration → Environment variables:

| Variable | Value |
|----------|-------|
| `GITHUB_REPO_URL` | `https://github.com/MylesThomas/betting.git` |
| `GITHUB_USERNAME` | `MylesThomas` |
| `GITHUB_EMAIL` | `mylescgthomas@gmail.com` |
| `SECRET_NAME` | `betting-dashboard-secrets` |
| `AWS_REGION_NAME` | `us-east-1` |

---

## Secrets Manager Format

In AWS Secrets Manager, create secret `betting-dashboard-secrets` with:

```json
{
  "ODDS_API_KEY": "your_odds_api_key_here",
  "GITHUB_TOKEN": "ghp_your_github_token_here"
}
```

---

## Common Troubleshooting

### Lambda Times Out
```bash
# Increase timeout to 15 minutes
aws lambda update-function-configuration \
  --function-name betting-dashboard-daily-update \
  --timeout 900
```

### Out of Memory
```bash
# Increase memory to 1024 MB
aws lambda update-function-configuration \
  --function-name betting-dashboard-daily-update \
  --memory-size 1024
```

### Can't Access Secrets
1. Go to Lambda → Configuration → Permissions
2. Click on Execution Role
3. Attach policy: `SecretsManagerReadWrite`

### Git Push Fails
- Check GitHub token hasn't expired
- Regenerate token if needed: https://github.com/settings/tokens
- Update in Secrets Manager

---

## Monitoring

### View CloudWatch Logs
```bash
# List log streams
aws logs describe-log-streams \
  --log-group-name /aws/lambda/betting-dashboard-daily-update \
  --order-by LastEventTime \
  --descending \
  --max-items 5

# Get latest log stream
aws logs get-log-events \
  --log-group-name /aws/lambda/betting-dashboard-daily-update \
  --log-stream-name '<stream-name-from-above>'
```

### Check EventBridge Rule
```bash
aws events list-rules --name-prefix betting-dashboard
```

### List Lambda Invocations (Last 7 Days)
Go to Lambda Console → Monitor → Metrics

Or use CloudWatch Insights:
```sql
fields @timestamp, @message
| filter @message like /Dashboard update/
| sort @timestamp desc
| limit 20
```

---

## Cost Tracking

### View Current Month Costs
```bash
aws ce get-cost-and-usage \
  --time-period Start=2025-11-01,End=2025-11-30 \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=SERVICE
```

### Set Budget Alert
1. Go to AWS Billing → Budgets
2. Create budget: $5/month
3. Set alert threshold: 80% ($4)
4. Add email notification

---

## Updating the System

### Update Lambda Code
1. Edit `docs/lambda_function.py` locally
2. Copy/paste into Lambda console
3. Click "Deploy"

### Update Python Dependencies
1. Update `docs/lambda_requirements.txt`
2. Rebuild layer:
   ```bash
   cd /Users/thomasmyles/dev/betting
   rm -rf lambda_layer
   mkdir -p lambda_layer/python
   pip install -r docs/lambda_requirements.txt -t lambda_layer/python/
   cd lambda_layer
   zip -r layer.zip python
   ```
3. Upload to Lambda → Layers → Create new version

### Update Repository Code
1. Push changes to GitHub as usual
2. Lambda will pull latest code on next run

---

## Emergency Procedures

### Disable Auto-Updates
Go to EventBridge → Rules → betting-dashboard-daily-7am-et → Disable

### Manually Trigger Update
1. Go to Lambda → Test
2. Click "Test" button
3. Check logs for success

### Rollback Dashboard
1. Go to GitHub → Commits
2. Find last good commit
3. Revert:
   ```bash
   git revert <commit-hash>
   git push
   ```
4. Streamlit auto-deploys the rollback

---

## Resources

- **AWS Lambda Docs:** https://docs.aws.amazon.com/lambda/
- **EventBridge Docs:** https://docs.aws.amazon.com/eventbridge/
- **Secrets Manager:** https://docs.aws.amazon.com/secretsmanager/
- **CloudWatch Logs:** https://docs.aws.amazon.com/cloudwatch/

---

## Support

If something breaks:
1. Check CloudWatch logs first
2. Check GitHub for failed commits
3. Check Streamlit Cloud status
4. Test Lambda function manually
5. Review secrets are valid

