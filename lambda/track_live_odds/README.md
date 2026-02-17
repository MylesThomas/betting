# Live Odds Tracker Lambda

Tracks live betting lines during NBA and NCAAB games. Runs every minute via EventBridge.

## What It Does

- Checks ESPN API for live games (free)
- Fetches odds from The Odds API only when games are live (saves API credits)
- Captures spreads + moneylines from 6 bookmakers
- Saves data to S3 in Parquet format
- Continues tracking post-game for ~1-2 hours (captures final scores)

## Files

- `lambda_function.py` - Main Lambda handler
- `deploy_nba_and_ncaab_lambdas.sh` - Deploys both NBA and NCAAB Lambdas
- `tmp/` - Ad-hoc testing scripts

## Dependencies

From repo root:
- `src/nba_team_name_mapping.py` - Normalizes NBA team names for ESPN matching
- `src/ncaab_team_name_mapping.py` - Normalizes NCAAB team names for ESPN matching

External (via AWS SDK for pandas layer):
- pandas, pyarrow, boto3

Packaged:
- requests

## Deployment

**From this directory:**

```bash
cd lambda/track_live_odds
bash deploy_nba_and_ncaab_lambdas.sh
```

**From repo root:**

```bash
bash lambda/track_live_odds/deploy_nba_and_ncaab_lambdas.sh
```

This will:
1. Package Lambda code + dependencies
2. Deploy to both Lambda functions
3. Create/update EventBridge rules (every 1 minute)
4. Test invocations
5. Clean up temporary files

## Lambda Functions Created

- `track-live-odds-nba-per-minute` (512 MB, 30s timeout)
- `track-live-odds-ncaab-per-minute` (1024 MB, 45s timeout)

## Local Testing

```bash
# From repo root
python lambda/track_live_odds/lambda_function.py --sport nba --prod-run
python lambda/track_live_odds/lambda_function.py --sport ncaab --prod-run
```

## S3 Output

Data saved to:
- `s3://nba-betting-mt/data/01_input/live_odds/{the-odds-api,espn}/`
- `s3://ncaab-betting-mt/data/01_input/live_odds/{the-odds-api,espn}/`

File naming: `YYYYMMDD_HHMMSS.parquet` (e.g., `20260216_193500.parquet`)

## Monitoring

```bash
# CloudWatch logs
aws logs tail /aws/lambda/track-live-odds-nba-per-minute --follow
aws logs tail /aws/lambda/track-live-odds-ncaab-per-minute --follow

# Check recent data
aws s3 ls s3://nba-betting-mt/data/01_input/live_odds/the-odds-api/ | tail -10
aws s3 ls s3://ncaab-betting-mt/data/01_input/live_odds/espn/ | tail -10
```

## Query Snapshots

```bash
# From repo root
./tmp/query_live_odds_snapshot.sh 20260216_193500 nba
./tmp/query_live_odds_snapshot.sh 20260216_193500 ncaab
```

## Cost Optimization

- **ESPN-first check**: Only calls Odds API when games are live (~73% savings)
- **Post-game tracking**: Saves ESPN data only (no Odds API calls)
- **Typical usage**: ~180 API calls/day during season (~$162/month)
- **Off-season**: Disable via `aws events disable-rule --name track-live-odds-ncaab-every-minute`

## Notes

- First deployment was successful on 2026-02-16
- Captures final scores for backtesting
- Handles overtime, postponed games, etc.
- Post-game files have ESPN data only (no odds)
