# NBA Player Rebounds

OLS + XGBoost ensemble, UNDER only, edge ≥5pp. OOS: +986u (OLS) / +940u (XGB).

## Lambda
`nba-rebounds-daily`

## EventBridge rules
- `nba-rebounds-daily-score-9am-et` — score today's props
- `nba-rebounds-daily-settle-905am-et` — settle yesterday
- `nba-rebounds-daily-9am-et` — daily trigger
- `nba-rebounds-daily-130pm-et` — afternoon trigger

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nba_rebounds_modeling/lambda/deploy_nba_rebounds_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name nba-rebounds-daily-score-9am-et    --region us-east-2
aws events enable-rule --name nba-rebounds-daily-settle-905am-et --region us-east-2
aws events enable-rule --name nba-rebounds-daily-9am-et          --region us-east-2
aws events enable-rule --name nba-rebounds-daily-130pm-et        --region us-east-2
```

## Pre-season checklist (before 2026-10-28)
1. Rebuild spine and re-enable rules above
