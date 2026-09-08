# MLB Pitcher Outs

Consensus-line model, UNDER minus-odds only, edge ≥10pp, shrink=0.25. OOS: +16.63% ROI (n=446).

## Lambda
`mlb-pitcher-outs-daily`

## EventBridge rules
- `mlb-po-pipeline-daily-830am-et` — score today's props, send email
- `mlb-po-settle-daily-830am-et` — settle yesterday
- `mlb-po-spine-daily-8am-et` — spine update

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/mlb_pitcher_outs_modeling/lambda/deploy_mlb_pitcher_outs_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name mlb-po-pipeline-daily-830am-et --region us-east-2
aws events enable-rule --name mlb-po-settle-daily-830am-et    --region us-east-2
aws events enable-rule --name mlb-po-spine-daily-8am-et       --region us-east-2
```
