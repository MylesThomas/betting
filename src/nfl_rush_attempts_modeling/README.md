# NFL Rush Attempts

Ridge model, QB UNDER only, line ≥6.5, edge ≥3pp. OOS: +13.96% ROI.

## Lambda
`nfl-rush-attempts-daily`

## EventBridge rules
- `nfl-rush-attempts-pipeline-thu-11am-et` — Thursday Night Football
- `nfl-rush-attempts-pipeline-sun-11am-et` — Sunday games
- `nfl-rush-attempts-pipeline-mon-11am-et` — Monday Night Football
- `nfl-rush-attempts-settle-daily-10am-et` — settle prior-day games
- `nfl-rush-attempts-spine-update-tue-9am-et` — rebuild spine
- `nfl-rush-attempts-spine-verify-wed-9am-et` — verify spine

## Deploy
```bash
cd ~/dev/betting && source .env && bash src/nfl_rush_attempts_modeling/lambda/deploy_nfl_rush_attempts_lambda.sh
```

After deploying, re-enable rules (deploy script sets them to DISABLED):
```bash
for rule in nfl-rush-attempts-pipeline-thu-11am-et nfl-rush-attempts-pipeline-sun-11am-et \
            nfl-rush-attempts-pipeline-mon-11am-et nfl-rush-attempts-settle-daily-10am-et \
            nfl-rush-attempts-spine-update-tue-9am-et nfl-rush-attempts-spine-verify-wed-9am-et; do
  aws events enable-rule --name $rule --region us-east-2
done
```

## Pre-season checklist (before 2026-09-09)
1. Upload model artifacts: `python src/nfl_rush_attempts_modeling/scripts/upload_artifacts.py`
2. Rebuild spine: `python src/nfl_rush_attempts_modeling/scripts/update_spine.py --season 2026`
3. Enable rules above
