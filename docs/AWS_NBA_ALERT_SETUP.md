# 🚨 NBA Arb Alert Lambda Setup

## What It Does

```
Every 15 minutes:
    ↓
Clone GitHub repo
    ↓
Fetch live NBA props (today's games, ET timezone)
    ↓
Find ALL arbs, save to data/04_output/nba/arbs/
    ↓
Commit & push to GitHub
    ↓
If 5%+ edge found → Send email alert 📧
    ↓
Streamlit Cloud auto-deploys with new data
```

## Quick Setup (45 min)

### 1. Create New Lambda Function

```
Name: nba-arb-alerts-15min
Runtime: Python 3.12
Memory: 512 MB
Timeout: 120 seconds
Ephemeral storage: 1024 MB
```

### 2. Attach Lambda Layers

Use the same layers as daily Lambda:
- `git-lambda2` (provides git binaries)
- `betting-dashboard-dependencies` (provides pandas, requests)

### 3. Environment Variables

| Variable | Value |
|----------|-------|
| `GITHUB_REPO_URL` | `https://github.com/MylesThomas/betting.git` |
| `GITHUB_USERNAME` | `MylesThomas` |
| `GITHUB_EMAIL` | `mylescgthomas@gmail.com` |
| `SECRET_NAME` | `betting-dashboard-secrets` |
| `AWS_REGION_NAME` | `us-east-2` |
| `SNS_TOPIC_ARN` | `arn:aws:sns:us-east-2:YOUR_ACCOUNT:betting-arb-alerts` |
| `MIN_PROFIT_PCT` | `10.0` |

### 4. IAM Permissions

Add to Lambda role:
- `SecretsManagerReadWrite` (to get ODDS_API_KEY + GITHUB_TOKEN)
- `AmazonSNSFullAccess` (to send emails)

### 5. Upload Code

Copy contents of `docs/lambda_function_nba_alerts.py` into Lambda console.

### 6. Create SNS Topic (if needed)

```bash
# Create topic
aws sns create-topic --name betting-arb-alerts

# Subscribe your email
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-2:YOUR_ACCOUNT:betting-arb-alerts \
    --protocol email \
    --notification-endpoint your@email.com
```

Confirm the subscription email!

### 7. Create EventBridge Schedule

**⚠️ API CREDIT USAGE:**
- Each run fetches ~10 games × 14 prop markets = ~140 API calls
- Running every 5 min, 24/7 = 288 runs/day × 140 = 40,320 credits/day ❌ (TOO MUCH!)
- Running every 5 min during game hours = 84 runs/day × 140 = 11,760 credits/day ✅

**RECOMMENDED: Every 5 min, only 5pm-midnight ET (game hours)**
```
cron(0/5 22-5 * * ? *)   # 22:00-05:00 UTC = 5pm-midnight ET
```

**Alternative: Every 15 min during game hours (if credits very limited)**
```
cron(0/15 22-5 * * ? *)   # 28 runs/day × 140 = 3,920 credits/day
```

**NOT RECOMMENDED: Running 24/7 (burns through entire monthly quota in days)**

## Test It

1. Click "Test" in Lambda console
2. Use empty event: `{}`
3. Check CloudWatch logs
4. Verify new CSV in GitHub repo: `data/04_output/nba/arbs/arb_output_YYYYMMDD_HHMMSS.csv`
5. If 5%+ arb found, you'll get an email!

## Output Files

Each run creates a file like:
```
data/04_output/nba/arbs/arb_output_20251206_171845.csv
```

- Timestamp is in ET timezone
- Contains ALL arbs found (any profit > 0)
- Dashboard dedupes across files, keeps best profit per player/market/line

## Email Example

```
🚨 HIGH-VALUE NBA ARBS FOUND! 🚨

Time: 2025-12-06 07:15 PM ET
Arbs found: 1

==================================================

#1 - 5.23% PROFIT
   Player: LeBron James
   Market: Points 25.5
   Game: Lakers @ Clippers

   📈 OVER 25.5: +115 @ fanduel
   📉 UNDER 25.5: +110 @ draftkings

   💰 Stake $100 total:
      → $48.54 on OVER @ fanduel
      → $51.46 on UNDER @ draftkings
      → Guaranteed profit: $5.23

--------------------------------------------------

⚡ ACT FAST - Lines move quickly!

Dashboard: https://tqs-props-dashboard.streamlit.app
```

## Cost Estimate

| Component | Cost |
|-----------|------|
| Lambda | ~$0.20/month (96 runs/day × 30 sec × 30 days) |
| SNS | ~$0.01/month (few alerts) |
| Secrets Manager | $0.40/month (already have) |
| **Total** | **~$0.65/month** |

## Troubleshooting

**No alerts but games are on:**
- Check MIN_PROFIT_PCT (maybe no 5%+ arbs exist)
- Try lowering to 3.0 to test

**Lambda timeout:**
- Increase timeout to 180 seconds
- Increase memory to 1024 MB

**Git push fails:**
- Check GITHUB_TOKEN is valid (not expired)
- Verify repo URL is correct

**No games found:**
- Check it's looking at correct date (ET timezone)
- Late at night, games may be over

**Email not received:**
- Check SNS subscription is confirmed
- Check spam folder
- Verify SNS_TOPIC_ARN is correct

## Files

- `docs/lambda_function_nba_alerts.py` - Lambda code
- `docs/lambda_requirements.txt` - Dependencies (already in layer)

