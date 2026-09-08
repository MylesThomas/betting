# NBA Player Points

OLS + bootstrap, UNDER only, edge ≥5pp, shrink=0.25, fav_only. OOS: 1,396 bets · +149.6u · +10.71% ROI.

## Lambda
`nba-points-daily`

## EventBridge rules
- `nba-points-pipeline-daily-1130am-et` — score today's props, send email
- `nba-points-settle-daily-1030am-et` — settle yesterday

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nba_points_modeling/lambda/deploy_nba_points_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name nba-points-pipeline-daily-1130am-et --region us-east-2
aws events enable-rule --name nba-points-settle-daily-1030am-et    --region us-east-2
```

## Pre-season checklist (before 2026-10-28)
1. Rebuild spine: `python src/nba_points_modeling/scripts/update_spine.py`
2. Upload model artifacts to S3
3. Enable rules above
