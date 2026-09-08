# MLB Batter Total Bases

XGBoost + Method C calibration, UNDER 1.5 only, edge ≥5pp. OOS: +11.67% ROI (n=71).

## Lambda
`mlb-total-bases-daily`

## EventBridge rules
- `mlb-tb-combined-daily-9am-et` — settle yesterday + score today, single email
- `mlb-tb-spine-daily-830am-et` — incremental spine update

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/mlb_total_bases_modeling/lambda/deploy_mlb_total_bases_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name mlb-tb-combined-daily-9am-et --region us-east-2
aws events enable-rule --name mlb-tb-spine-daily-830am-et  --region us-east-2
```
