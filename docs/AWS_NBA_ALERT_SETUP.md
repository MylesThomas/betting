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

| Variable | Value | Notes |
|----------|-------|-------|
| `GITHUB_REPO_URL` | `https://github.com/MylesThomas/betting.git` | Required |
| `GITHUB_USERNAME` | `MylesThomas` | Required |
| `GITHUB_EMAIL` | `mylescgthomas@gmail.com` | Required |
| `SECRET_NAME` | `betting-dashboard-secrets` | Required |
| `AWS_REGION_NAME` | `us-east-2` | Required |
| `SNS_TOPIC_ARN` | `arn:aws:sns:us-east-2:YOUR_ACCOUNT:betting-arb-alerts` | Optional |
| `MIN_PROFIT_PCT` | `10.0` | Optional (default: 10.0) |
| `MAX_STALENESS_MINUTES` | `2.0` | Optional (default: 2.0) |
| `EXCLUDED_BOOKMAKERS` | `bovada` | Optional (see "Phantom Arbs" below) |

**Where to add these env vars:**
1. Open AWS Lambda Console → Functions → `nba-arb-alerts-15min`
2. Click **Configuration** tab
3. Click **Environment variables** (left sidebar)
4. Click **Edit** button (top right)
5. Click **Add environment variable** for each one
6. Enter Key and Value
7. Click **Save**

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
🚨 high-value nba arbs found! 🚨

time: 2025-12-06 07:15 PM ET
arbs found: 3 (1 high-value, 2 fresh, 1 stale)

==================================================

✅ FRESH ARBS (NOT STALE):

#1 - 5.23% PROFIT ✅
   Player: LeBron James
   Market: Points 25.5
   Game: Los Angeles Lakers @ Los Angeles Clippers

   📈 OVER 25.5: +115 @ fanduel
      Line updated: 07:14:32 PM ET
   📉 UNDER 25.5: +110 @ draftkings
      Line updated: 07:14:45 PM ET
   🕐 Data pulled: 2025-12-06 07:15:12 ET
   ⏱️  Staleness: 0.6 min < 2.0 min threshold ✅ NOT STALE

   💰 Stake $100 total:
      → $48.54 on OVER @ fanduel
      → $51.46 on UNDER @ draftkings
      → Guaranteed profit: $5.23

--------------------------------------------------

==================================================
📋 other fresh arbs (below threshold):
==================================================

#2 - 2.15% | Kawhi Leonard | Points 28.5 ✅
     Game: Los Angeles Lakers @ Los Angeles Clippers
     Over +108 @ betmgm (updated 07:13:42 PM ET)
     Under -110 @ draftkings (updated 07:14:01 PM ET)
     Data pulled: 2025-12-06 07:15:12 ET
     ⏱️  1.4 min < 2.0 min threshold ✅ NOT STALE

==================================================
⚠️  STALE ARBS (lines may have changed):
==================================================

#3 - 8.42% | Paul George | Threes 2.5 ⚠️ STALE
     Game: Los Angeles Lakers @ Los Angeles Clippers
     Over +220 @ bovada (updated 07:10:15 PM ET)
     Under -125 @ betonlineag (updated 07:14:32 PM ET)
     Data pulled: 2025-12-06 07:15:12 ET
     ⏱️  4.9 min > 2.0 min threshold ⚠️ STALE
     ⚠️  bovada data is 4.9 min old - verify before betting!

⚡ act fast - lines move quickly!
✅ = fresh lines (safe to bet)
⚠️  = stale lines (double-check before betting)

Dashboard: https://tqs-props-dashboard.streamlit.app

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

**Git push failures (alerts sent but no CSV in GitHub):**
- Check CloudWatch logs for git errors
- Verify GITHUB_TOKEN in Secrets Manager
- Verify Lambda has write permissions to repo

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

