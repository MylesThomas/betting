# NFL Tackles

OLS + NegBin hybrid, both directions. Rules currently DISABLED — enable before 2026-09-09.

## Lambda
`nfl-tackles-daily`

## EventBridge rules
- `nfl-tackles-pipeline-thu-11am-et` — Thursday Night Football
- `nfl-tackles-pipeline-sun-11am-et` — Sunday games
- `nfl-tackles-pipeline-mon-11am-et` — Monday Night Football
- `nfl-tackles-settle-daily-10am-et` — settle prior-day games
- `nfl-tackles-spine-update-tue-9am-et` — rebuild spine
- `nfl-tackles-spine-verify-wed-9am-et` — verify spine

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nfl_tackles_modeling/lambda/deploy_nfl_tackles_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
for rule in nfl-tackles-pipeline-thu-11am-et nfl-tackles-pipeline-sun-11am-et \
            nfl-tackles-pipeline-mon-11am-et nfl-tackles-settle-daily-10am-et \
            nfl-tackles-spine-update-tue-9am-et nfl-tackles-spine-verify-wed-9am-et; do
  aws events enable-rule --name $rule --region us-east-2
done
```

## Pre-season checklist (before 2026-09-09)
1. Rebuild spine: `python src/nfl_tackles_modeling/scripts/update_spine.py --season 2026`
2. Enable rules above
