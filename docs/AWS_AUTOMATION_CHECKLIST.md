# AWS Automation Checklist - Daily Dashboard Updates

## Goal
Automatically run `scripts/daily_update.sh` every day at 7 AM ET (6 AM CT) so the dashboard updates without manual intervention.

---

## Architecture Overview

```
AWS EventBridge (Scheduler)
    ↓ triggers at 7 AM ET daily
AWS Lambda (Python function)
    ↓ clones repo, runs update script
    ↓ pushes to GitHub
Streamlit Cloud
    ↓ auto-detects changes
    ↓ redeploys dashboard
```

---

## ✅ Checklist

### **Step 1: Prepare Your Repository**

- [ ] **1.1** Ensure all scripts are in GitHub
  ```bash
  cd /Users/thomasmyles/dev/betting
  git status
  # Make sure everything is committed
  ```

- [ ] **1.2** Create a GitHub Personal Access Token (for Lambda to push)
  - Go to: https://github.com/settings/tokens
  - Click "Generate new token" → "Tokens (classic)"
  - Name: `AWS Lambda - Betting Dashboard`
  - Expiration: `No expiration` (or 1 year)
  - Scopes: Check `repo` (full control of private repositories)
  - Click "Generate token"
  - **SAVE THIS TOKEN** - you'll need it for AWS Secrets Manager

---

### **Step 2: Get Your Odds API Key**

- [ ] **2.1** Locate your Odds API key
  - Check your `.env` file: `cat /Users/thomasmyles/dev/betting/.env`
  - Or get it from: https://the-odds-api.com/account/
  - **SAVE THIS KEY** - you'll need it for AWS Secrets Manager

---

### **Step 3: Set Up AWS Account**

- [ ] **3.1** Create AWS account (if you don't have one)
  - Go to: https://aws.amazon.com/
  - Click "Create an AWS Account"
  - Free tier includes: 1M Lambda requests/month (more than enough!)

- [ ] **3.2** Set up AWS CLI (optional but recommended)
  ```bash
  # Install AWS CLI
  brew install awscli
  
  # Configure with your credentials
  aws configure
  # Enter: Access Key ID, Secret Access Key, Region (us-east-2), Format (json)
  ```

---

### **Step 4: Store Secrets in AWS Secrets Manager**

- [ ] **4.1** Go to AWS Secrets Manager: https://console.aws.amazon.com/secretsmanager/
  
- [ ] **4.2** Click "Store a new secret"
  
- [ ] **4.3** Configure secret:
  - **Secret type:** Other type of secret
  - **Key/value pairs:**
    - Key: `ODDS_API_KEY` | Value: `[your_odds_api_key]`
    - Key: `GITHUB_TOKEN` | Value: `[your_github_personal_access_token]`
  - Click "Next"
  
- [ ] **4.4** Name the secret:
  - **Secret name:** `betting-dashboard-secrets`
  - Click "Next" → "Next" → "Store"

- [ ] **4.5** Note the ARN (Amazon Resource Name)
  - It looks like: `arn:aws:secretsmanager:us-east-1:123456789:secret:betting-dashboard-secrets-xxxxx`
  - **SAVE THIS ARN** - you'll need it for Lambda permissions

---

### **Step 5: Create Lambda Function**

- [ ] **5.1** Go to AWS Lambda: https://console.aws.amazon.com/lambda/

- [ ] **5.2** Click "Create function"

- [ ] **5.3** Configure function:
  - **Function name:** `betting-dashboard-daily-update`
  - **Runtime:** Python 3.12 (or latest available)
  - **Architecture:** x86_64
  - **Permissions:** Create a new role with basic Lambda permissions
  - Click "Create function"

- [ ] **5.4** Upload the Lambda code (see `docs/lambda_function.py` - created below)
  - In the Lambda console, go to "Code" tab
  - Copy/paste the code from `docs/lambda_function.py`
  - Click "Deploy"

- [ ] **5.5** Configure Lambda settings:
  - Go to "Configuration" → "General configuration" → "Edit"
  - **Memory:** 512 MB
  - **Timeout:** 15 minutes (enter `15` min, `0` sec)
  - Click "Save"
  
  - Then go to "Configuration" → "General configuration" → "Edit" again (scroll down)
  - **Ephemeral storage:** 2048 MB (need space to clone repo)
  - Click "Save"

- [ ] **5.6** Add environment variables:
  - Go to "Configuration" → "Environment variables" → "Edit"
  - Add:
    - Key: `GITHUB_REPO_URL` | Value: `https://github.com/MylesThomas/betting.git`
    - Key: `GITHUB_USERNAME` | Value: `MylesThomas`
    - Key: `GITHUB_EMAIL` | Value: `mylescgthomas@gmail.com`
    - Key: `SECRET_NAME` | Value: `betting-dashboard-secrets`
    - Key: `AWS_REGION_NAME` | Value: `us-east-2` (or your region)
  - Click "Save"

---

### **Step 6: Grant Lambda Access to Secrets Manager**

- [ ] **6.1** Go to Lambda "Configuration" → "Permissions"

- [ ] **6.2** Click on the "Execution role" name (opens IAM console in new tab)

- [ ] **6.3** In IAM, click "Add permissions" → "Attach policies"

- [ ] **6.4** Search for and attach: `SecretsManagerReadWrite`

- [ ] **6.5** Verify the policy is attached

---

### **Step 7: Add Lambda Layers**

Lambda needs additional packages (Python dependencies + git) that aren't included by default.

**Part A: Add Git Layer (Required)**

- [ ] **7.1** Add public git layer:
  - Go to your Lambda function
  - Scroll down to "Layers" → "Add a layer"
  - Select "Specify an ARN"
  - **ARN:** `arn:aws:lambda:us-east-2:553035198032:layer:git-lambda2:8`
  - Click "Add"

**Part B: Add Python Dependencies Layer**

- [ ] **7.2** Create layer package (for Linux x86_64):
  ```bash
  cd /Users/thomasmyles/dev/betting
  rm -rf lambda_layer
  mkdir -p lambda_layer/python
  
  # Install packages for Linux x86_64 (Lambda's architecture)
  pip install --platform manylinux2014_x86_64 \
    --target=lambda_layer/python \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade \
    -r docs/lambda_requirements.txt
  
  # Zip it up
  cd lambda_layer
  zip -r layer.zip python
  
  # Check size (should be ~45 MB, under 50 MB limit)
  ls -lh layer.zip
  ```

- [ ] **7.3** Upload layer to Lambda:
  - Go to Lambda console → "Layers" → "Create layer"
  - **Name:** `betting-dashboard-dependencies`
  - **Upload:** `lambda_layer/layer.zip`
  - **Compatible runtimes:** Python 3.12 (match your Lambda function runtime)
  - Click "Create"

- [ ] **7.4** Attach layer to function:
  - Go to your Lambda function
  - Scroll down to "Layers" → "Add a layer"
  - Select "Custom layers" → Choose your layer
  - Click "Add"

**Option B: Use Docker (Alternative)**

- [ ] See `docs/lambda_docker_build.md` for Docker-based approach

---

### **Step 8: Test Lambda Function**

- [ ] **8.1** Create test event:
  - In Lambda console, click "Test" tab
  - **Event name:** `DailyUpdate`
  - **Event JSON:** `{}` (empty object is fine)
  - Click "Save"

- [ ] **8.2** Run test:
  - Click "Test" button
  - Watch the logs in the "Execution results" tab
  - Should see: "✅ Dashboard update complete!"
  - Check GitHub for new commit
  - Check Streamlit dashboard updated

- [ ] **8.3** Troubleshoot if needed:
  - Check CloudWatch Logs: Lambda → Monitor → "View logs in CloudWatch"
  - Common issues:
    - Missing secrets: Check Secrets Manager setup
    - Permission denied: Check IAM role has Secrets Manager access
    - Timeout: Increase Lambda timeout to 15 minutes
    - Out of memory: Increase Lambda memory to 1024 MB

---

### **Step 9: Schedule Daily Runs with EventBridge**

- [ ] **9.1** Go to Amazon EventBridge: https://console.aws.amazon.com/events/

- [ ] **9.2** Click "Rules" → "Create rule"

- [ ] **9.3** Configure rule:
  - **Name:** `betting-dashboard-daily-7am-et`
  - **Description:** `Triggers daily dashboard update at 7 AM ET`
  - **Rule type:** Schedule
  - Click "Next"

- [ ] **9.4** Define schedule:
  - **Schedule pattern:** Cron-based schedule
  - **Cron expression:** `0 12 * * ? *`
    - This is 12:00 PM UTC = 7:00 AM EST / 6:00 AM CST
    - Note: During daylight saving (EST → EDT), this becomes 8 AM EDT
    - For year-round 7 AM ET, you may need two rules (one DST, one standard)
  - **Timezone:** UTC (adjust cron as needed)
  - Click "Next"

- [ ] **9.5** Select target:
  - **Target type:** AWS service
  - **Select a target:** Lambda function
  - **Function:** `betting-dashboard-daily-update`
  - Click "Next"

- [ ] **9.6** Review and create:
  - Click "Next" → "Create rule"

---

### **Step 10: Verify End-to-End Automation**

- [ ] **10.1** Wait for next scheduled run (or trigger manually)
  - Go to EventBridge → Rules → Select your rule → "Test schedule"
  
- [ ] **10.2** Check Lambda executed:
  - Lambda → Monitor → "View logs in CloudWatch"
  - Look for successful execution

- [ ] **10.3** Check GitHub updated:
  - Go to: https://github.com/MylesThomas/betting/commits/main
  - Should see new commit with today's date

- [ ] **10.4** Check Streamlit dashboard updated:
  - Go to: https://tqs-nba-props-dashboard.streamlit.app
  - Should show latest data
  - Check "Last Updated" timestamp

---

### **Step 11: Set Up Monitoring & Alerts (Optional)**

- [ ] **11.1** Create SNS topic for alerts:
  - Go to Amazon SNS: https://console.aws.amazon.com/sns/
  - Create topic: `betting-dashboard-alerts`
  - Create subscription: Email → your email

- [ ] **11.2** Add CloudWatch Alarm for Lambda failures:
  - Go to CloudWatch → Alarms → "Create alarm"
  - **Metric:** Lambda → Errors
  - **Function:** `betting-dashboard-daily-update`
  - **Threshold:** > 0 errors in 1 minute
  - **Notification:** SNS topic `betting-dashboard-alerts`
  - name: dashboard-alarm-1

- [ ] **11.3** Test alert:
  - Manually break something in Lambda (e.g., invalid API key)
  - Run test
  - Should receive email alert

---

## 💰 AWS Cost Estimate

**Monthly Costs (estimated):**

| Service | Usage | Cost |
|---------|-------|------|
| Lambda | 30 executions/month @ ~5 min each | $0.00 (free tier: 1M requests, 400K GB-seconds) |
| Secrets Manager | 1 secret | $0.40/month ($0.05 per 10K API calls) |
| EventBridge | 30 scheduled events | $0.00 (free tier: 14M events/month) |
| CloudWatch Logs | ~500 MB/month | $0.00 (free tier: 5 GB ingestion) |
| **TOTAL** | | **~$0.40/month** |

**Free Tier Eligibility:**
- Lambda: Always free (1M requests/month)
- EventBridge: Always free
- CloudWatch Logs: Free tier for 12 months

---

## 🔧 Maintenance

### When to Update Lambda Code
- [ ] When you add new markets
- [ ] When you change the update script
- [ ] When you upgrade Python dependencies

### How to Update Lambda Code
```bash
# 1. Update local code
cd /Users/thomasmyles/dev/betting
git pull

# 2. Update Lambda function code
# Copy new code to Lambda console (or use AWS CLI/SAM for deployment)
```

### Monitoring Dashboard Health
- [ ] Check Streamlit dashboard daily (first few days)
- [ ] Monitor Lambda execution logs weekly
- [ ] Review AWS costs monthly

---

## 🚨 Troubleshooting

### Lambda times out
- Increase timeout to 15 minutes
- Increase memory to 1024 MB
- Check if API is slow/down

### GitHub push fails
- Check GitHub token is valid (doesn't expire)
- Check Lambda has correct permissions
- Check repo URL is correct

### Streamlit doesn't update
- Check GitHub commit was successful
- Check Streamlit Cloud settings (auto-deploy enabled)
- Manually reboot app in Streamlit Cloud

### API rate limits
- Check API usage at: https://the-odds-api.com/account/
- Reduce markets if needed
- Upgrade API plan if needed

---

## 📚 Additional Resources

- AWS Lambda Docs: https://docs.aws.amazon.com/lambda/
- EventBridge Cron Expressions: https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-rule-schedule.html
- Secrets Manager: https://docs.aws.amazon.com/secretsmanager/

---

## ✅ Success Criteria

You're done when:
- [ ] Wake up at 6 AM CT (7 AM ET)
- [ ] Open https://tqs-nba-props-dashboard.streamlit.app
- [ ] See fresh data from that morning
- [ ] You didn't have to run any commands! 🎉

---

**Estimated Setup Time:** 2-3 hours (first time)

**Ongoing Effort:** 0 minutes/day (fully automated!)

