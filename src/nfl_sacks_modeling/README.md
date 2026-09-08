# NFL Sacks

Logistic regression, UNDER 0.5 sacks, edge ≥3pp. OOS: 213 bets · 74.2% WR · +29.73u · +14.0% ROI.

## Lambda
`nfl-sacks-daily`

## EventBridge rules
- `nfl-sacks-pipeline-daily-9am-et` — score today's props, send email
- `nfl-sacks-settle-rebuild-daily-830am-et` — settle yesterday + rebuild spine

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nfl_sacks_modeling/lambda/deploy_nfl_sacks_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name nfl-sacks-pipeline-daily-9am-et         --region us-east-2
aws events enable-rule --name nfl-sacks-settle-rebuild-daily-830am-et --region us-east-2
```

## Pre-season checklist (before 2026-09-09)
1. Enable rules above
