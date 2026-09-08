# NBA Player Assists

OLS + Normal CDF, OVER only, edge ≥15pp. OOS: 1,188 bets · +28.97u · +2.44% ROI.

## Lambda
`nba-assists-daily`

## EventBridge rules
- `nba-assists-pipeline-daily-11am-et` — score today's props, send email
- `nba-assists-settle-daily-10am-et` — settle yesterday

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nba_assists_modeling/lambda/deploy_nba_assists_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name nba-assists-pipeline-daily-11am-et --region us-east-2
aws events enable-rule --name nba-assists-settle-daily-10am-et    --region us-east-2
```

## Pre-season checklist (before 2026-10-28)
1. Rebuild spine: `python src/nba_assists_modeling/scripts/update_spine.py`
2. Enable rules above
