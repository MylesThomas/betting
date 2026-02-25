# NCAAB Fade Revenge Spot – daily Lambda

Runs **~9am ET** daily. **Bets the away team when the away team is the revenge team** (focal = 0–N vs opponent); focal at home = no play.

## Flow

1. **Fetch data** (inside Lambda):
   - **Yesterday:** `fetch_historical_ncaab_season_lines.py` (lines) + `fetch_historical_game_results_espn_api.py` (results) for `$YESTERDAY`. Used for backtest/joined data.
   - **Today (lines only):** `fetch_historical_ncaab_season_lines.py` for `$TODAY`. Writes `s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/{today}.csv` so step 4 can read it. If that file is missing or empty after the fetch, the handler sends an email saying something is broken (no fallback).
2. Load outcomes + lines from S3 (season start → yesterday), join, normalize (team names from `src/ncaab_team_name_mapping.py`).
3. Find rematch spots: pairs where one team has 0 wins vs the other; focal = that revenge team. We only bet when **focal is the away team**.
4. Load today’s game lines from S3; get today’s games from ESPN; for each rematch spot where the away team is the focal (revenge team), add a play **betting the away team**.
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
- `src/`: `config_loader.py`, `ncaab_team_name_mapping.py`, `ncaab_conference_data.py`, `ncaab_conference_inferred.py`
- `config/config.yaml`
- `.gitignore` (so fetch scripts’ `find_project_root()` resolves to package root)

Analysis script lives in `tmp/analyze_ncaab_conference_rematch_su_ats.py` (run from repo root for historical analysis; not used by this Lambda).
