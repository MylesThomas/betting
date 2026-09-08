"""
Fetch team game-by-game run scores from MLB Stats API for 2024–2026.

For each season, pulls the full regular-season schedule with linescore hydration
to get runs scored/allowed per team per game.

Output schema (one row per game):
  game_pk | game_date | season | home_team | away_team |
  home_runs | away_runs | total_runs |
  home_hits | away_hits | home_errors | away_errors |
  innings | game_type | status

Output paths:
  S3:    s3://the-odds-api-mt/mlb/game_totals_model/scores/mlb_team_game_scores.parquet
  Local: ~/Downloads/tmp/mlb_game_totals/game_scores.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_game_scores.py
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_game_scores.py --seasons 2026
"""
from __future__ import annotations

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
SLEEP_S      = 0.1

S3_BUCKET   = "the-odds-api-mt"
SCORES_KEY  = "mlb/game_totals_model/scores/mlb_team_game_scores.parquet"
LOCAL_OUT   = Path.home() / "Downloads/tmp/mlb_game_totals/game_scores.parquet"

SEASONS = [2024, 2025, 2026]


def fetch_schedule(season: int) -> list[dict]:
    """Fetch full regular-season schedule with linescore for one season."""
    print(f"  Fetching schedule for {season}...")
    r = requests.get(
        f"{MLB_API_BASE}/schedule",
        params={
            "sportId":   1,
            "season":    season,
            "gameType":  "R",
            "hydrate":   "linescore,team",
            "startDate": f"{season}-03-01",
            "endDate":   f"{season}-11-30",
        },
        timeout=60,
    )
    r.raise_for_status()
    dates = r.json().get("dates", [])
    games = []
    for d in dates:
        games.extend(d.get("games", []))
    print(f"    {len(games)} games found for {season}")
    return games


def parse_game(game: dict, season: int) -> dict | None:
    """Extract run scoring from a game dict. Returns None if game not final."""
    status = game.get("status", {}).get("abstractGameState", "")
    if status != "Final":
        return None

    game_pk   = game.get("gamePk")
    game_date = game.get("gameDate", "")[:10]  # YYYY-MM-DD
    game_type = game.get("gameType", "R")

    teams     = game.get("teams", {})
    home_info = teams.get("home", {})
    away_info = teams.get("away", {})

    home_team = home_info.get("team", {}).get("name", "")
    away_team = away_info.get("team", {}).get("name", "")

    linescore = game.get("linescore", {})
    innings   = linescore.get("currentInning", 9)

    teams_ls  = linescore.get("teams", {})
    home_ls   = teams_ls.get("home", {})
    away_ls   = teams_ls.get("away", {})

    home_runs   = home_ls.get("runs")
    away_runs   = away_ls.get("runs")
    home_hits   = home_ls.get("hits")
    away_hits   = away_ls.get("hits")
    home_errors = home_ls.get("errors")
    away_errors = away_ls.get("errors")

    if home_runs is None or away_runs is None:
        return None

    return {
        "game_pk":     game_pk,
        "game_date":   game_date,
        "season":      season,
        "home_team":   home_team,
        "away_team":   away_team,
        "home_runs":   int(home_runs),
        "away_runs":   int(away_runs),
        "total_runs":  int(home_runs) + int(away_runs),
        "home_hits":   int(home_hits) if home_hits is not None else None,
        "away_hits":   int(away_hits) if away_hits is not None else None,
        "home_errors": int(home_errors) if home_errors is not None else None,
        "away_errors": int(away_errors) if away_errors is not None else None,
        "innings":     int(innings),
        "game_type":   game_type,
        "status":      "Final",
    }


def main(seasons: list[int] | None = None) -> None:
    target_seasons = seasons or SEASONS
    all_rows: list[dict] = []

    for season in target_seasons:
        games = fetch_schedule(season)
        season_rows = []
        for g in games:
            row = parse_game(g, season)
            if row:
                season_rows.append(row)
        print(f"    Parsed {len(season_rows)} final games for {season}")
        all_rows.extend(season_rows)
        time.sleep(SLEEP_S)

    df = pd.DataFrame(all_rows).drop_duplicates("game_pk").reset_index(drop=True)
    print(f"\nTotal games: {df.shape[0]}")
    print(f"Date range:  {df['game_date'].min()} – {df['game_date'].max()}")
    print(f"Avg total runs: {df['total_runs'].mean():.2f}")
    print(f"Total runs distribution:\n{df['total_runs'].value_counts().sort_index().head(20)}")

    s3  = boto3.client("s3")
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=SCORES_KEY, Body=buf.getvalue())
    print(f"\nSaved → s3://{S3_BUCKET}/{SCORES_KEY}")

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_OUT, index=False)
    print(f"Saved → {LOCAL_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(seasons=args.seasons)
