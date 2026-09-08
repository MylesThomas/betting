# MLB Pitcher Strikeouts

OLS model, both directions, edge ≥3pp. OOS: 911 bets · 54.01% WR · +95.08u · +10.44% ROI (v6, 2025–2026).

## Lambda
`mlb-strikeouts-daily`

## EventBridge rules
- `mlb-strikeouts-pipeline-daily-1pm-et` — score today's props, send email
- `mlb-strikeouts-settle-daily-10am-et` — settle yesterday + rebuild spine
- `mlb-strikeouts-spine-weekly-sunday` — full spine rebuild

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/mlb_strikeouts_modeling/lambda/deploy_mlb_strikeouts_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name mlb-strikeouts-pipeline-daily-1pm-et --region us-east-2
aws events enable-rule --name mlb-strikeouts-settle-daily-10am-et  --region us-east-2
aws events enable-rule --name mlb-strikeouts-spine-weekly-sunday    --region us-east-2
```

## Pre-season checklist (before 2027-03-20)
1. Upload model artifacts to S3 (model + residuals + meta)
2. Rebuild spine: `python src/mlb_strikeouts_modeling/scripts/update_spine.py`
3. Enable rules above
