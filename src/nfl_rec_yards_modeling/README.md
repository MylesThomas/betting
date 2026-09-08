# NFL Receiving Yards

OLS + NegBin hybrid, WR/TE OVER only, edge ≥3pp, min 3 books. OOS: 57.7% hit rate · +115.6u.

## Lambda
`nfl-rec-yards-daily`

## EventBridge rules
- `nfl-rec-yards-pipeline-daily-9am-et` — score today's props, send email
- `nfl-rec-yards-settle-rebuild-daily-830am-et` — settle yesterday + rebuild spine
- `nfl-rec-yards-spine-weekly-sunday` — full spine rebuild

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nfl_rec_yards_modeling/lambda/deploy_nfl_rec_yards_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
aws events enable-rule --name nfl-rec-yards-pipeline-daily-9am-et          --region us-east-2
aws events enable-rule --name nfl-rec-yards-settle-rebuild-daily-830am-et  --region us-east-2
aws events enable-rule --name nfl-rec-yards-spine-weekly-sunday             --region us-east-2
```

## Pre-season checklist (before 2026-09-09)
1. Rebuild spine: `python src/nfl_rec_yards_modeling/scripts/update_spine.py --season 2026`
2. Enable rules above
