"""
Fetch starting pitcher game-by-game stats from MLB Stats API for 2024-2026.

For each season:
  1. Pull all pitchers who led in strikeouts (top 300) — this captures all starters.
  2. For each pitcher, pull per-game splits (strikeOuts, IP, gamesStarted, opponent, isHome, date).
  3. Filter to starts only (gamesStarted == 1).
  4. Save merged parquet to S3 and local.

Output schema:
  player_id, player_name, season, game_date, game_pk,
  strikeouts, innings_pitched, hits, earned_runs, walks,
  opponent_id, opponent_name, is_home, is_win,
  games_started

Output paths:
  S3:    s3://the-odds-api-mt/mlb/strikeouts_model/pitcher_gamelogs/{season}.parquet
  Local: ~/Downloads/tmp/mlb_pitcher_gamelogs.parquet  (merged all seasons)

Usage:
  python src/mlb_strikeouts_modeling/scripts/fetch_pitcher_gamelogs.py
  python src/mlb_strikeouts_modeling/scripts/fetch_pitcher_gamelogs.py --seasons 2026
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
SLEEP_S      = 0.05
LEADER_LIMIT = 300  # top-N by strikeouts — captures all relevant starters

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/strikeouts_model/pitcher_gamelogs"
LOCAL_OUT = Path.home() / "Downloads/tmp/mlb_pitcher_gamelogs.parquet"

SEASONS = [2024, 2025, 2026]


def get_starter_player_ids(season: int) -> list[tuple[int, str]]:
    """Return (mlbam_id, full_name) for all pitchers with GS >= 1 in the season."""
    r = requests.get(
        f"{MLB_API_BASE}/stats",
        params={
            "stats":      "season",
            "group":      "pitching",
            "season":     season,
            "gameType":   "R",
            "playerPool": "All",
            "limit":      2000,
            "offset":     0,
            "sportId":    1,
        },
        timeout=30,
    )
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    return [
        (s["player"]["id"], s["player"]["fullName"])
        for s in splits
        if s.get("stat", {}).get("gamesStarted", 0) >= 1
    ]


def get_pitcher_gamelogs(player_id: int, season: int) -> list[dict]:
    r = requests.get(
        f"{MLB_API_BASE}/people/{player_id}/stats",
        params={"stats": "gameLog", "season": season, "group": "pitching"},
        timeout=30,
    )
    if r.status_code != 200:
        return []
    stats_list = r.json().get("stats", [])
    if not stats_list:
        return []
    splits = stats_list[0].get("splits", [])

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        if stat.get("gamesStarted", 0) < 1:
            continue  # skip relief appearances
        rows.append({
            "player_id":       player_id,
            "season":          season,
            "game_date":       s.get("date", ""),
            "game_pk":         s.get("game", {}).get("gamePk"),
            "strikeouts":      stat.get("strikeOuts", 0),
            "innings_pitched": stat.get("inningsPitched", "0.0"),
            "hits":            stat.get("hits", 0),
            "earned_runs":     stat.get("earnedRuns", 0),
            "walks":           stat.get("baseOnBalls", 0),
            "batters_faced":   stat.get("battersFaced", 0),
            "pitches":         stat.get("numberOfPitches", 0),
            "opponent_id":     s.get("opponent", {}).get("id"),
            "opponent_name":   s.get("opponent", {}).get("name", ""),
            "is_home":         int(s.get("isHome", False)),
            "is_win":          int(s.get("isWin", False)),
            "games_started":   stat.get("gamesStarted", 0),
        })
    return rows


def process_season(season: int) -> pd.DataFrame:
    print(f"\n=== Season {season} ===")
    players = get_starter_player_ids(season)
    print(f"  {len(players)} pitchers found")

    all_rows = []
    for i, (pid, name) in enumerate(players, 1):
        rows = get_pitcher_gamelogs(pid, season)
        if rows:
            for row in rows:
                row["player_name"] = name
            all_rows.extend(rows)
        time.sleep(SLEEP_S)
        if i % 50 == 0:
            print(f"  [{i}/{len(players)}] {len(all_rows)} rows so far")

    if not all_rows:
        print(f"  No data for {season}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["innings_pitched"] = pd.to_numeric(df["innings_pitched"], errors="coerce")
    df = df.drop_duplicates(subset=["player_id", "game_pk"]).reset_index(drop=True)
    print(f"  {len(df):,} starts  |  {df['player_id'].nunique()} pitchers  |  "
          f"date range: {df['game_date'].min()} → {df['game_date'].max()}")
    return df


def upload_season(s3c, df: pd.DataFrame, season: int):
    key = f"{S3_PREFIX}/{season}.parquet"
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3c.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Uploaded → s3://{S3_BUCKET}/{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    args = parser.parse_args()

    s3c = boto3.client("s3")
    all_dfs = []

    for season in sorted(args.seasons):
        df = process_season(season)
        if df.empty:
            continue
        upload_season(s3c, df, season)
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(LOCAL_OUT, index=False)
        print(f"\nAll seasons merged → {LOCAL_OUT}")
        print(f"Total rows: {len(combined):,}  |  Pitchers: {combined['player_id'].nunique()}")


if __name__ == "__main__":
    main()
