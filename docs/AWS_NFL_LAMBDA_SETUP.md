# AWS Lambda Setup - NFL Weekly Plays Finder

## Overview

Automate NFL regression betting plays analysis with weekly email notifications.

**Schedule:** Every Monday at 12:00 PM ET (17:00 UTC)

**Email Format:**
1. 📧 **TL;DR**: Play count and recommendations
2. ⚠️ **Warning**: May need re-run after Monday Night Football
3. 📋 **Full Logs**: Complete terminal output
4. 📊 **Summary**: Results repeated for quick reference

---

## Prerequisites

Complete the basic AWS setup first (if not already done):

✅ Follow [`docs/AWS_AUTOMATION_CHECKLIST.md`](./AWS_AUTOMATION_CHECKLIST.md) Steps 1-6:
- AWS account created
- AWS Secrets Manager configured with `ODDS_API_KEY` and `GITHUB_TOKEN`
- Lambda layer built and uploaded (`betting-dashboard-dependencies`)
- IAM permissions configured

---

## Setup Steps

### Step 1: Create SNS Topic for Email Alerts

**1.1** Go to AWS SNS: https://console.aws.amazon.com/sns/

**1.2** Click "Topics" → "Create topic"

**1.3** Configure topic:
- **Type:** Standard
- **Name:** `nfl-plays-alerts`
- **Display name:** NFL Weekly Plays
- Click "Create topic"

**1.4** Create email subscription:
- Click "Create subscription"
- **Protocol:** Email
- **Endpoint:** your.email@example.com
- Click "Create subscription"
- **Check your email** and confirm the subscription

**1.5** Copy the Topic ARN (you'll need it):
```
arn:aws:sns:us-east-2:123456789012:nfl-plays-alerts
```

---

### Step 2: Create Lambda Function

**2.1** Go to AWS Lambda: https://console.aws.amazon.com/lambda/

**2.2** Click "Create function"

**2.3** Configure function:
- **Function name:** `nfl-regression-plays-weekly`
- **Runtime:** Python 3.12
- **Architecture:** x86_64
- **Permissions:** Create a new role with basic Lambda permissions
- Click "Create function"

**2.4** Upload the Lambda code:
- Go to "Code" tab
- Copy/paste code from `docs/lambda_function_nfl.py`
- Click "Deploy"

**2.5** Configure settings:
- Go to "Configuration" → "General configuration" → "Edit"
- **Memory:** 1024 MB
- **Timeout:** 15 minutes (900 seconds)
- **Ephemeral storage:** 2048 MB
- Click "Save"

**2.6** Add environment variables:
- Go to "Configuration" → "Environment variables" → "Edit"
- Add:
  - `GITHUB_REPO_URL` = `https://github.com/MylesThomas/betting.git`
  - `GITHUB_USERNAME` = `MylesThomas`
  - `GITHUB_EMAIL` = `mylescgthomas@gmail.com`
  - `SECRET_NAME` = `betting-dashboard-secrets`
  - `AWS_REGION_NAME` = `us-east-2`
  - `SNS_TOPIC_ARN` = `arn:aws:sns:us-east-2:123456789012:nfl-plays-alerts`
- Click "Save"

---

### Step 3: Grant Lambda Permissions

**3.1** Go to "Configuration" → "Permissions"

**3.2** Click on the "Execution role" name (opens IAM)

**3.3** Click "Add permissions" → "Attach policies"

**3.4** Attach these policies:
- ✅ `SecretsManagerReadWrite` (for API keys)
- ✅ `AmazonSNSFullAccess` (for email notifications)

---

### Step 4: Add Lambda Layers

**4.1** Scroll down to "Layers" → "Add a layer"

**4.2** Add git layer:
- Select "Specify an ARN"
- **ARN:** `arn:aws:lambda:us-east-2:553035198032:layer:git-lambda2:8`
- Click "Add"

**4.3** Add Python dependencies layer:
- Click "Add a layer" again
- Select "Custom layers"
- Choose `betting-dashboard-dependencies` (the layer you created earlier)
- Select latest version
- Click "Add"

---

### Step 5: Test the Function

**5.1** Click "Test" tab

**5.2** Create test event:
- **Event name:** `WeeklyRun`
- **Event JSON:** `{}`
- Click "Save"

**5.3** Click "Test" button

**5.4** Verify:
- ✅ Execution succeeds (green banner)
- ✅ Check your email for play recommendations
- ✅ Check CloudWatch logs for details
- ✅ Verify GitHub commit (if plays found)

---

### Step 6: Schedule Weekly Runs

**6.1** Go to Amazon EventBridge: https://console.aws.amazon.com/events/

**6.2** Click "Rules" → "Create rule"

**6.3** Configure rule:
- **Name:** `nfl-plays-weekly-monday-noon-et`
- **Description:** `Triggers NFL regression plays finder every Monday at 12 PM ET`
- **Rule type:** Schedule
- Click "Next"

**6.4** Define schedule:
- **Schedule pattern:** Cron-based schedule
- **Cron expression:** `0 17 * * MON *`
  - This is 17:00 UTC = 12:00 PM EST (1:00 PM EDT)
  - Runs every Monday
- Click "Next"

**6.5** Select target:
- **Target type:** AWS service
- **Select a target:** Lambda function
- **Function:** `nfl-regression-plays-weekly`
- Click "Next"

**6.6** Review and create:
- Click "Next" → "Create rule"

---

### Step 7: Automated GitHub Commits

The Lambda function automatically commits results to GitHub:

**What gets committed:**
- All files in `data/01_input/` (odds data fetched)
- All files in `data/03_intermediate/` (processed data)
- All files in `data/04_output/` (final play recommendations)

**Commit message format:**
- Dynamically generated based on week analysis
- Format: `nfl: add week X plays (post week Y analysis)`
- Example: `nfl: add week 13 plays (post week 12 analysis)`
- Week X = upcoming week to bet on
- Week Y = week that just finished (used for luck calculation)

**Example commit:**
```
nfl: add week 13 plays (post week 12 analysis)

- data/01_input/the-odds-api/nfl/game_lines/historical/nfl_game_lines_2025-11-25.csv
- data/04_output/nfl/todays_plays/nfl_luck_regression_plays_week_13_2025-11-25_120045.csv
- data/04_output/nfl/todays_plays/nfl_luck_regression_all_teams_week_13_2025-11-25_120045.csv
```

---

### Step 8: Optional - Manual Re-Run After MNF

If Monday Night Football results affect teams with upcoming games, you may want to re-run the analysis:

**Option A: Trigger via AWS Console**
1. Go to Lambda → Functions → `nfl-regression-plays-weekly`
2. Click "Test" tab → "Test"
3. Check your email for updated plays
4. New commit will be pushed: `nfl: add week X plays (post week Y analysis, post-MNF update)`

**Option B: Run Locally**
```bash
cd /Users/thomasmyles/dev/betting
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode --no-safe-mode
```

**When to re-run:**
- MNF team is playing again this week AND had significant luck change
- Example: If MNF loser underperformed by 10+ points and plays Thursday/Sunday

---

## Email Format Example

```
================================================================================
🏈 NFL WEEK 13 BETTING PLAYS - TL;DR
================================================================================

✅ FOUND 1 PLAY THIS WEEK
   • Primary Strategy (Unlucky Favorites): 1
   • Secondary Strategy (Lucky Big Underdogs): 0

🔥 PRIMARY PLAYS (67-73% ATS, +27-39% ROI):
   ✅ BET: TB -3.0
      Game: TB vs ARI
      Last week (W12) luck: -8.1
      This week spread: -3.0 (0-3 favorite)

================================================================================
⚠️  IMPORTANT NOTES
================================================================================

📅 This analysis ran on Monday at 12:00 PM ET
🏈 Monday Night Football may not be included in the analysis

If MNF results affect Week 13 teams with upcoming games:
   → Consider re-running after MNF completes
   → Manually trigger Lambda function, or
   → Run locally: python implementation/find_nfl_regression_plays.py --current-week --verbose-mode --no-safe-mode

Strategy Background:
   • Primary: Back teams that underperformed by 7+ points last week (unlucky)
   • Secondary: Back teams that overperformed by 7+ points last week AND are 7+ point underdogs
   • Based on regression-to-mean analysis through Week 12 of 2025 season

================================================================================
📋 FULL EXECUTION LOGS
================================================================================

[... complete terminal output ...]

================================================================================
📊 SUMMARY (Week 13)
================================================================================

[... results repeated ...]
```

---

## Monitoring & Maintenance

### Weekly Checklist
- [ ] Check Monday 12:30 PM ET email
- [ ] Review play recommendations
- [ ] If MNF affects plays, consider re-run
- [ ] Track play results for strategy validation

### Monthly Maintenance
- [ ] Review CloudWatch logs for errors
- [ ] Check AWS costs (should be ~$0.40/month)
- [ ] Verify Unexpected Points data is up-to-date
- [ ] Update Lambda code if strategy changes

### Updating Lambda Code

When you modify the strategy or finder script:

```bash
# 1. Update local code
cd /Users/thomasmyles/dev/betting
git pull

# 2. Go to Lambda console
# 3. Paste updated code from docs/lambda_function_nfl.py
# 4. Click "Deploy"
# 5. Test with test event
```

---

## Troubleshooting

### No Email Received
- Check SNS subscription is confirmed
- Check Lambda execution succeeded (CloudWatch logs)
- Verify `SNS_TOPIC_ARN` environment variable is correct
- Check spam folder

### Function Times Out
- Increase timeout to 15 minutes
- Increase memory to 1024 MB or higher
- Check if Odds API is slow/down

### "Module not found" Error
- Verify Lambda layer is attached
- Check layer version is latest
- Rebuild layer if needed (see `AWS_AUTOMATION_CHECKLIST.md`)

### No Plays Found
- Normal! Not every week has regression opportunities
- Verify Unexpected Points data is updated through latest week
- Check backtesting results still valid

### MNF Update Required
- Manually run Lambda function after MNF completes
- Or run locally with `--current-week --verbose-mode --no-safe-mode`

---

## Cost Breakdown

**Monthly Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| Lambda | 4-5 executions/month @ ~5 min each | $0.00 (free tier) |
| SNS | 4-5 emails/month | $0.00 (free tier) |
| Secrets Manager | 1 secret | $0.40/month |
| CloudWatch Logs | ~200 MB/month | $0.00 (free tier) |
| **TOTAL** | | **~$0.40/month** |

**Same cost as NBA dashboard!** (shares Secrets Manager)

---

## Future Enhancements

Potential improvements:
- [ ] Automatically fetch Unexpected Points data (currently manual)
- [ ] Add HTML email formatting with tables
- [ ] Include injury report analysis
- [ ] Send follow-up email with MNF-adjusted plays
- [ ] Track historical play performance in DynamoDB
- [ ] Add SMS alerts for high-confidence plays

---

## Resources

- Original NBA Lambda Setup: `docs/AWS_AUTOMATION_CHECKLIST.md`
- Lambda Function Code: `docs/lambda_function_nfl.py`
- Finder Script: `implementation/find_nfl_regression_plays.py`
- Strategy Analysis: `backtesting/20251126_nfl_spread_covering_vs_score_differential.py`
- CloudWatch Logs: https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2
- AWS Lambda Docs: https://docs.aws.amazon.com/lambda/
- AWS SNS Docs: https://docs.aws.amazon.com/sns/

---

## Success Criteria

You're done when:
- [ ] Wake up Monday morning
- [ ] Check email around 12:30 PM ET
- [ ] See "🏈 NFL Week X: Y Plays Found"
- [ ] Read recommendations and reasoning
- [ ] Place bets if plays found
- [ ] You didn't have to run any commands! 🎉

**Estimated Setup Time:** 1 hour (if AWS account already configured)

**Ongoing Effort:** 0 minutes/week (fully automated!)

