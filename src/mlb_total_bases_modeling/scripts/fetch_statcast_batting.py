"""
Pull Statcast pitch-by-pitch data for 2024-2026 and aggregate to
total bases per batter per game. Saves to S3 and ~/Downloads/tmp/.

Total bases = 1*(single) + 2*(double) + 3*(triple) + 4*(home_run)

Output schema:
  game_date, batter_id, player_name, team, opponent, home_team,
  singles, doubles, triples, home_runs, total_bases, ab, hits,
  season, game_pk

Usage:
  python src/mlb_total_bases_modeling/scripts/fetch_statcast_batting.py
  python src/mlb_total_bases_modeling/scripts/fetch_statcast_batting.py --seasons 2026
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import pybaseball as pb

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

pb.cache.enable()

S3_BUCKET  = "the-odds-api-mt"
S3_KEY     = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
LOCAL_OUT  = Path.home() / "Downloads/tmp/mlb_batting_statcast.parquet"

# Season date ranges (inclusive)
SEASON_DATES = {
    2024: ("2024-03-20", "2024-10-01"),  # include Seoul series
    2025: ("2025-03-18", "2025-10-01"),  # include Tokyo series
    2026: ("2026-03-25", "2026-07-03"),  # up to today
}

TB_EVENTS = {"single", "double", "triple", "home_run"}
AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "field_error",
    "fielders_choice", "fielders_choice_out",
}


def fetch_season(season: int) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    print(f"  Pulling Statcast {season}: {start} → {end} ...")

    # Pull in monthly chunks to avoid timeouts
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(days=29), end_ts)
        s, e = cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        print(f"    chunk {s} → {e}", end=" ... ", flush=True)
        try:
            df = pb.statcast(start_dt=s, end_dt=e)
            print(f"{len(df):,} pitches")
            chunks.append(df)
        except Exception as ex:
            print(f"ERROR: {ex}")
        cur = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.5)

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    raw = raw[raw["game_type"] == "R"].copy()  # regular season only
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date.astype(str)

    # Filter to events that ended an at-bat
    events = raw[raw["events"].notna()].copy()

    # Compute per-at-bat TB
    events["tb"] = (
        (events["events"] == "single").astype(int) * 1 +
        (events["events"] == "double").astype(int) * 2 +
        (events["events"] == "triple").astype(int) * 3 +
        (events["events"] == "home_run").astype(int) * 4
    )
    events["is_ab"]  = events["events"].isin(AB_EVENTS).astype(int)
    events["is_hit"] = events["events"].isin(TB_EVENTS).astype(int)

    # NOTE: 'player_name' in Statcast is the PITCHER's name, not the batter's.
    # Aggregate to batter-game level using batter MLBAM ID only.
    agg = events.groupby(["game_date", "batter", "home_team", "away_team", "game_pk"]).agg(
        singles     = ("events", lambda x: (x == "single").sum()),
        doubles     = ("events", lambda x: (x == "double").sum()),
        triples     = ("events", lambda x: (x == "triple").sum()),
        home_runs   = ("events", lambda x: (x == "home_run").sum()),
        total_bases = ("tb", "sum"),
        ab          = ("is_ab", "sum"),
        hits        = ("is_hit", "sum"),
    ).reset_index()

    # Infer team from inning_topbot: Bot = home batting, Top = away batting
    events_team = (
        events[["game_date","batter","game_pk","inning_topbot","home_team","away_team"]]
        .drop_duplicates(subset=["game_date","batter","game_pk"])
        .copy()
    )
    events_team["team"] = events_team.apply(
        lambda r: r["home_team"] if r["inning_topbot"] == "Bot" else r["away_team"], axis=1
    )
    events_team["opponent"] = events_team.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Bot" else r["home_team"], axis=1
    )

    agg = agg.merge(
        events_team[["game_date","batter","game_pk","team","opponent"]],
        on=["game_date","batter","game_pk"], how="left",
    ).drop_duplicates(subset=["game_date","batter","game_pk"])

    # Add batter name via reverse lookup
    batter_ids = agg["batter"].dropna().unique().tolist()
    name_lookup = pb.playerid_reverse_lookup(batter_ids, key_type="mlbam")
    name_lookup["player_name"] = (
        name_lookup["name_first"].str.title() + " " + name_lookup["name_last"].str.title()
    )
    agg = agg.merge(
        name_lookup[["key_mlbam","player_name"]].rename(columns={"key_mlbam":"batter"}),
        on="batter", how="left",
    )

    agg["season"] = season
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()))
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

    all_seasons = []
    for season in sorted(args.seasons):
        df = fetch_season(season)
        if not df.empty:
            print(f"  Season {season}: {len(df):,} batter-games")
            all_seasons.append(df)

    if not all_seasons:
        print("No data fetched.")
        return

    combined = pd.concat(all_seasons, ignore_index=True)
    combined = combined.sort_values(["game_date", "player_name"]).reset_index(drop=True)

    print(f"\nTotal batter-games: {len(combined):,}")
    print(f"Date range: {combined['game_date'].min()} → {combined['game_date'].max()}")
    print(f"Unique players: {combined['player_name'].nunique():,}")

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOCAL_OUT, index=False)
    print(f"Saved locally → {LOCAL_OUT}")

    if not args.no_s3:
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf.getvalue())
        print(f"Saved to S3 → s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == "__main__":
    main()
