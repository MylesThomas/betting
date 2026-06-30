"""
Build the game-level rest edge + spread cover base table.

Joins:
  - team_games (from S3/local, per-team-per-game rest metrics)
  - nfl_data_py schedules (spread_line, scores)

Output: one row per team per game with rest metrics + cover result.
Saves to S3 and ~/Downloads/tmp/game_level_rest.csv.

Usage:
    python scripts/build_game_level_rest.py
    python scripts/build_game_level_rest.py --no-cache
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

import boto3
import nfl_data_py as nfl
import pandas as pd
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

S3_BUCKET = "the-odds-api-mt"
S3_KEY = "nfl/rest_edge/game_level_rest.csv"
TMP_DIR = Path.home() / "Downloads" / "tmp"

SEASONS = list(range(2010, 2026))

# Same mapping used in fetch_historical_schedule
NFL_DATA_ABBR_MAP = {
    "LA":  "LAR",
    "STL": "LAR",
    "OAK": "LV",
    "SD":  "LAC",
}


def remap(abbr: str) -> str:
    return NFL_DATA_ABBR_MAP.get(abbr, abbr)


def load_team_games(seasons: list[int]) -> pd.DataFrame:
    """Load per-team-per-game rest data from local tmp files."""
    dfs = []
    for season in seasons:
        path = TMP_DIR / f"team_games_{season}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run build_historical_rest_edge.py first"
            )
        df = pd.read_csv(path, parse_dates=["game_date"])
        df["game_date"] = df["game_date"].dt.date
        df["season"] = season
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_spreads(seasons: list[int]) -> pd.DataFrame:
    """Load spread lines and scores from nfl_data_py, remapping abbreviations."""
    raw = nfl.import_schedules(seasons)
    reg = raw[raw["game_type"] == "REG"].copy()

    reg["home_team"] = reg["home_team"].map(remap)
    reg["away_team"] = reg["away_team"].map(remap)

    keep = ["season", "week", "home_team", "away_team",
            "spread_line", "home_score", "away_score", "result"]
    return reg[keep].reset_index(drop=True)


def compute_cover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spread_assigned, team_margin, covered, push columns.

    spread_line convention (nfl_data_py): positive = home is FAVORED (expected home margin).
      e.g. spread_line = +3  → home gives 3 points (home -3 in standard notation)
           spread_line = -3  → home gets  3 points (home +3 in standard notation)

    spread_assigned = points the TEAM gives (positive = team is favorite):
      home row:  spread_assigned =  spread_line
      away row:  spread_assigned = -spread_line

    Team covers if team_margin > spread_assigned (beat the line).
    Push if team_margin == spread_assigned.
    """
    df = df.copy()

    df["team_score"] = df.apply(
        lambda r: r["home_score"] if r["is_home"] else r["away_score"], axis=1
    )
    df["opp_score"] = df.apply(
        lambda r: r["away_score"] if r["is_home"] else r["home_score"], axis=1
    )
    df["team_margin"] = df["team_score"] - df["opp_score"]

    df["spread_assigned"] = df.apply(
        lambda r: r["spread_line"] if r["is_home"] else -r["spread_line"], axis=1
    )

    df["covered"] = df["team_margin"] > df["spread_assigned"]
    df["push"]    = df["team_margin"] == df["spread_assigned"]

    return df


def build(seasons: list[int]) -> pd.DataFrame:
    print("Loading team_games rest data...")
    team_games = load_team_games(seasons)
    print(f"  {len(team_games)} team-game rows, {team_games['season'].nunique()} seasons")

    print("Loading spreads from nfl_data_py...")
    spreads = load_spreads(seasons)
    print(f"  {len(spreads)} games")

    # Join: team_games has team + is_home; spreads has home_team + away_team
    # Build a join key: (season, week, home_team) where home_team is the home side
    team_games["home_team"] = team_games.apply(
        lambda r: r["team"] if r["is_home"] else r["opponent"], axis=1
    )

    merged = team_games.merge(
        spreads,
        on=["season", "week", "home_team"],
        how="left",
        suffixes=("", "_sched"),
    )

    # Verify away_team from spreads matches our opponent (sanity check)
    mismatch = merged[
        ~merged["away_team"].isna() &
        merged.apply(lambda r: (r["team"] if not r["is_home"] else r["opponent"]) != r["away_team"], axis=1)
    ]
    if len(mismatch) > 0:
        print(f"  WARNING: {len(mismatch)} rows with away_team mismatch — check abbreviation mapping")
        print(mismatch[["season", "week", "team", "opponent", "is_home", "away_team"]].head(5).to_string())

    # Drop rows with no spread (shouldn't happen but guard anyway)
    no_spread = merged["spread_line"].isna().sum()
    if no_spread > 0:
        print(f"  WARNING: {no_spread} rows missing spread_line — dropping")
    merged = merged[merged["spread_line"].notna()].copy()

    print("Computing cover results...")
    merged = compute_cover(merged)

    cols = [
        "season", "week", "game_date", "team", "opponent", "is_home",
        "game_type", "days_rest", "opp_days_rest", "rest_edge",
        "had_bye", "short_week_road", "post_road_prime", "opp_extra_prep",
        "in_3_in_10", "in_4_in_17",
        "spread_line", "spread_assigned", "team_score", "opp_score",
        "team_margin", "covered", "push",
    ]
    # Only keep columns that exist
    cols = [c for c in cols if c in merged.columns]
    result = merged[cols].sort_values(["season", "week", "team"]).reset_index(drop=True)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not args.no_cache:
        local = TMP_DIR / "game_level_rest.csv"
        if local.exists():
            print(f"Found cached file at {local}")
            df = pd.read_csv(local)
            print(f"  {len(df)} rows, {df['season'].nunique()} seasons")
            return

    df = build(SEASONS)

    print(f"\nSaving {len(df)} rows...")
    csv_bytes = df.to_csv(index=False).encode()

    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=csv_bytes)
    print(f"  s3://{S3_BUCKET}/{S3_KEY}")

    local = TMP_DIR / "game_level_rest.csv"
    local.write_bytes(csv_bytes)
    print(f"  {local}")

    # Summary stats
    print(f"\nRow counts:")
    print(f"  Total team-game rows:  {len(df)}")
    print(f"  Unique games:          {len(df)//2}")
    print(f"  Seasons:               {df['season'].min()}–{df['season'].max()}")
    print(f"  Rows with spread:      {df['spread_line'].notna().sum()}")
    print(f"  Covered:               {df['covered'].sum():.0f}")
    print(f"  Push:                  {df['push'].sum():.0f}")
    print(f"  No result (push excl): {df['covered'].isna().sum():.0f}")
    print(f"\nRows per season:")
    print(df.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
