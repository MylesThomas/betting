# NCAAB Fade Revenge Spot – daily Lambda

Runs **~9am ET** daily. Fades the "revenge" team (0–N vs opponent); historical rematch teams cover &lt;40%.

## Flow

1. **Fetch yesterday** (inside Lambda):
   - `scripts/fetch_historical_ncaab_season_lines.py` — `--s3 --season 2025-26 --start-date $YESTERDAY --end-date $YESTERDAY`
   - `scripts/fetch_historical_game_results_espn_api.py` — `--s3 --season 2025-26 --start-date $YESTERDAY --end-date $YESTERDAY --sport ncaab`
2. Load outcomes + lines from S3 (season start → yesterday), join, normalize (team names from `src/ncaab_team_name_mapping.py`).
3. Find rematch spots: pairs where one team has 0 wins vs the other; focal = that team (we fade them).
4. Get today’s games from The Odds API; for each matchup that is a rematch spot, add a play (bet the opponent of the focal).
5. Write plays CSV to `s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/{date}.csv` (includes `home_conference`, `away_conference`).
6. Send SNS email: yesterday’s results (if any) + today’s plays (with conferences in body).

## Deploy

```bash
export ODDS_API_KEY="your-key"
export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:ACCOUNT:topic-name"  # optional
cd ~/dev/betting && bash lambda/ncaab_fade_revenge_daily/deploy_ncaab_fade_revenge_daily.sh
```

## Check logs

```bash
python tmp/get_lambda_logs.py --lambda-function-name ncaab-fade-revenge-daily
```

## Package contents (no tmp/)

- `lambda_function.py`
- `scripts/`: `fetch_historical_ncaab_season_lines.py`, `fetch_historical_game_results_espn_api.py`
- `src/`: `config_loader.py`, `ncaab_team_name_mapping.py`, `ncaab_conference_data.py`
- `config/config.yaml`
- `.gitignore` (so fetch scripts’ `find_project_root()` resolves to package root)

Analysis script lives in `tmp/analyze_ncaab_conference_rematch_su_ats.py` (run from repo root for historical analysis; not used by this Lambda).
