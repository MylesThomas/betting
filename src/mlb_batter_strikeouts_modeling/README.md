# MLB Batter Strikeouts

Logistic calibration on consensus line, UNDER 1.5 only, edge ≥5pp. OOF: 2,580 bets · +140.41u · +5.44% ROI.

## Lambda
`mlb-batter-strikeouts-daily`

## EventBridge rules
- `mlb-bs-combined-daily-9am-et` — settle yesterday + score today, single email

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/mlb_batter_strikeouts_modeling/lambda/deploy_mlb_batter_strikeouts_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name mlb-bs-combined-daily-9am-et --region us-east-2
```
