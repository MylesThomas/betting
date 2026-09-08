"""
Step 1 — Build game-level batter K spine from Statcast.

For each batter-game, produces:
  batter_id, batter_name, game_date, game_pk, home_team, away_team,
  strikeouts, plate_appearances, pitcher_id (starter), stand (batter hand)

Saves to S3 and ~/Downloads/tmp/mlb_batter_strikeouts_spine.parquet

Usage:
    python src/mlb_batter_strikeouts_modeling/scripts/20260711_build_batting_spine.py
    python src/mlb_batter_strikeouts_modeling/scripts/20260711_build_batting_spine.py --seasons 2024
"""
from __future__ import annotations

import argparse
import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import pybaseball
from pybaseball import playerid_reverse_lookup

warnings.filterwarnings("ignore")
pybaseball.cache.enable()

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET = "the-odds-api-mt"
S3_KEY    = "mlb/batter_strikeouts_model/spine/mlb_batter_strikeouts_spine.parquet"
LOCAL_OUT = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_spine.parquet"

SEASON_DATES = {
    2024: ("2024-03-20", "2024-10-01"),
    2025: ("2025-03-18", "2025-10-01"),
    2026: ("2026-03-25", "2026-07-11"),
}

K_EVENTS = {"strikeout", "strikeout_double_play"}

KEEP_COLS = [
    "batter", "pitcher", "events", "game_date", "game_pk",
    "home_team", "away_team", "stand", "inning",
]


def fetch_season(season: int) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    print(f"  Fetching {season} ({start} → {end})...")
    df = pybaseball.statcast(start, end)
    print(f"  Raw pitches: {len(df):,}")

    # Keep only PA-ending events
    pa = df[df["events"].notna()][KEEP_COLS].copy()
    print(f"  PA-ending rows: {len(pa):,}")
    pa["season"] = season
    return pa


def identify_starters(pa: pd.DataFrame) -> pd.DataFrame:
    """Add starting_pitcher_id per (game_pk, batting_team)."""
    # Starter = pitcher who threw the first PA of inning 1 to that batting side
    inning1 = pa[pa["inning"] == 1].copy()
    # For each game_pk, the home batting side faces the away starter (top of 1st = away bats),
    # the away batting side faces the home starter (bot of 1st = home bats).
    # "stand" is batter handedness, not side — use home_team/away_team logic instead.
    # Approach: for each game_pk, group by game_pk; the pitcher in inning 1 top is away starter,
    # inning 1 bottom is home starter.
    starters = (
        inning1
        .sort_values(["game_pk", "inning"])
        .groupby("game_pk")["pitcher"]
        .first()
        .reset_index()
        .rename(columns={"pitcher": "starting_pitcher_id"})
    )
    return pa.merge(starters, on="game_pk", how="left")


def build_game_level(pa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level PA data to (batter, game_date, game_pk) level."""
    pa["is_k"] = pa["events"].isin(K_EVENTS).astype(int)

    agg = (
        pa.groupby(["batter", "game_date", "game_pk", "home_team", "away_team",
                    "stand", "starting_pitcher_id", "season"])
        .agg(
            strikeouts=("is_k", "sum"),
            plate_appearances=("events", "count"),
        )
        .reset_index()
    )
    return agg


def add_batter_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map MLBAM batter IDs to player names."""
    unique_ids = df["batter"].dropna().unique().tolist()
    print(f"  Looking up {len(unique_ids):,} unique batter IDs...")
    lookup = playerid_reverse_lookup(unique_ids, key_type="mlbam")
    lookup["batter_name"] = (
        lookup["name_first"].str.strip() + " " + lookup["name_last"].str.strip()
    )
    lookup = lookup[["key_mlbam", "batter_name"]].rename(columns={"key_mlbam": "batter"})
    df = df.merge(lookup, on="batter", how="left")
    unmatched = df["batter_name"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} batter rows with no name lookup ({unmatched/len(df)*100:.1f}%)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_DATES.keys()))
    args = parser.parse_args()

    all_frames = []
    for season in sorted(args.seasons):
        pa = fetch_season(season)
        pa = identify_starters(pa)
        game_df = build_game_level(pa)
        game_df = add_batter_names(game_df)
        all_frames.append(game_df)
        print(f"  Season {season}: {len(game_df):,} batter-games  "
              f"({game_df['batter_name'].notna().sum():,} named)")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal batter-games: {len(combined):,}")
    print(f"Unique batters: {combined['batter_name'].nunique():,}")
    print(f"Date range: {combined['game_date'].min()} → {combined['game_date'].max()}")

    # Spot-check: Aaron Judge
    judge = combined[combined["batter_name"].str.lower().str.contains("judge", na=False)]
    if not judge.empty:
        print("\nAaron Judge sample (first 5 games):")
        print(judge[["batter_name","game_date","home_team","away_team","strikeouts","plate_appearances"]].head(5).to_string())
        print(f"  Career K rate: {judge['strikeouts'].sum() / judge['plate_appearances'].sum():.3f}")
        print(f"  Games: {len(judge)}")
    else:
        print("WARNING: Aaron Judge not found in spine")

    # Save local
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved local: {LOCAL_OUT}")

    # Save to S3
    s3c = boto3.client("s3")
    buf = BytesIO()
    combined.to_parquet(buf, index=False)
    s3c.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf.getvalue())
    print(f"Saved S3:    s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == "__main__":
    main()
