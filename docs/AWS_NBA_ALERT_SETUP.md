# 🚨 NBA Arb Alert Lambda Setup

## What It Does

```
Every 15 minutes:
    ↓
Fetch live NBA props (~15 sec)
    ↓
Find arbs with 5%+ edge
    ↓
If found → Send email alert 📧
    ↓
If not → Do nothing (no spam!)
```

## Quick Setup (30 min)

### 1. Create New Lambda Function

```
Name: nba-arb-alerts
Runtime: Python 3.12
Memory: 256 MB
Timeout: 60 seconds
```

### 2. Attach Existing Layer

Use the same layer you already have:
- `betting-dashboard-dependencies`

### 3. Environment Variables

| Variable | Value |
|----------|-------|
| `SECRET_NAME` | `betting-dashboard-secrets` |
| `AWS_REGION_NAME` | `us-east-2` |
| `SNS_TOPIC_ARN` | `arn:aws:sns:us-east-2:YOUR_ACCOUNT:betting-arb-alerts` |
| `MIN_PROFIT_PCT` | `5.0` |

### 4. IAM Permissions

Add to Lambda role:
- `SecretsManagerReadWrite` (to get ODDS_API_KEY)
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

**Option A: Every 15 minutes (24/7)**
```
rate(15 minutes)
```

**Option B: Every 15 min, only during game hours (6am-11pm ET)**
```
cron(0/15 11-3 * * ? *)   # 11am-3am UTC = 6am-10pm ET
```

**Option C: Every 15 min, only on game days (10am-11pm ET)**
```
cron(0/15 15-3 ? * * *)   # 10am-11pm ET
```

## Test It

1. Click "Test" in Lambda console
2. Use empty event: `{}`
3. Check CloudWatch logs
4. If arb found, you'll get an email!

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
| Lambda | ~$0.05/month (96 runs/day × 15 sec × 30 days) |
| SNS | ~$0.01/month (few alerts) |
| Secrets Manager | $0.40/month (already have) |
| **Total** | **~$0.50/month** |

## Troubleshooting

**No alerts but games are on:**
- Check MIN_PROFIT_PCT (maybe no 5%+ arbs exist)
- Try lowering to 3.0 to test

**Lambda timeout:**
- Increase timeout to 120 seconds

**Email not received:**
- Check SNS subscription is confirmed
- Check spam folder
- Verify SNS_TOPIC_ARN is correct

**API errors:**
- Check ODDS_API_KEY is valid
- Check API quota remaining

## Files

- `docs/lambda_function_nba_alerts.py` - Lambda code
- `docs/lambda_requirements.txt` - Dependencies (already in layer)

