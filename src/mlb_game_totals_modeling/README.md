# MLB Game Totals

Ridge regression, 9.5 unders, edge-agnostic benchmark + Strategy B (edge ≥2pp, shrink=0.50). OOS: +196.4u · +8.45% ROI (n=2,324).

## Lambda
`mlb-game-totals-daily`

## EventBridge rules
- `mlb-gt-pipeline-daily-10am-et` — score today's games, send email
- `mlb-gt-settle-daily-11am-et` — settle yesterday

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/mlb_game_totals_modeling/lambda/deploy_mlb_game_totals_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name mlb-gt-pipeline-daily-10am-et --region us-east-2
aws events enable-rule --name mlb-gt-settle-daily-11am-et    --region us-east-2
```

## Pre-season checklist (before 2027-03-20)
1. Rebuild spine with new season data
2. Enable rules above
