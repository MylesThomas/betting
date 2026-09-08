"""
Pull NFL weekly player data (passing + rushing) for all QBs, 1999–present.

nflverse changed URL scheme after 2024:
  <=2024: player_stats/player_stats_{year}.parquet
  >=2025: stats_player/stats_player_week_{year}.parquet  (also renamed some columns)

  If a future season returns 404 on both URLs, find the new asset name via:
    curl -s https://api.github.com/repos/nflverse/nflverse-data/releases | python3 -c "
    import sys,json; releases=json.load(sys.stdin)
    for r in releases: print(r['tag_name'], r['published_at'][:10])"
  That lists all release tags. Then inspect the relevant one:
    curl -s https://api.github.com/repos/nflverse/nflverse-data/releases/tags/<tag_name> | python3 -c "
    import sys,json; r=json.load(sys.stdin)
    [print(a['name'], a['browser_download_url']) for a in r['assets'] if 'player' in a['name'].lower()]"
  Update NEW_URL below with the new pattern and add any column renames to NEW_COL_RENAMES.

Saves to ~/Downloads/tmp/pass_yds/weekly_player_data.parquet

Usage:
  uv run python 20260806_pull_player_data.py              # pull all available seasons
  uv run python 20260806_pull_player_data.py --season 2025  # probe/fetch one season
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

OUT_DIR = Path.home() / "Downloads" / "tmp" / "pass_yds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Old URL (1999–2024, used by nfl_data_py internally)
OLD_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.parquet"
# New URL (2025+, nflverse renamed the release)
NEW_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"

# Column renames from new schema → old schema for consistency
NEW_COL_RENAMES = {
    "passing_interceptions": "interceptions",
    "sacks_suffered":        "sacks",
    "sack_yards_lost":       "sack_yards",
    "team":                  "recent_team",
}

KEEP_COLS = [
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "recent_team",
    "season",
    "week",
    "season_type",
    "opponent_team",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_yards",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "fantasy_points",
]


def url_available(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.HTTPError:
        return False


def fetch_season(year: int) -> pd.DataFrame:
    """Fetch one season — tries new URL first, falls back to nfl_data_py for old URL."""
    new_url = NEW_URL.format(year=year)
    if url_available(new_url):
        df = pd.read_parquet(new_url)
        df = df.rename(columns=NEW_COL_RENAMES)
        # new schema has no season_type column — all rows are regular season
        if "season_type" not in df.columns:
            df["season_type"] = "REG"
        return df

    # fall back to nfl_data_py (uses old URL internally)
    return nfl.import_weekly_data([year])


def latest_available_season() -> int:
    for year in range(2026, 2020, -1):
        old_ok = url_available(OLD_URL.format(year=year))
        new_ok = url_available(NEW_URL.format(year=year))
        if old_ok or new_ok:
            return year
    raise RuntimeError("No season found between 2021–2026. Check nflverse URL formats.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Probe/fetch a single season year.")
    args = parser.parse_args()

    if args.season:
        year = args.season
        old_url = OLD_URL.format(year=year)
        new_url = NEW_URL.format(year=year)
        print(f"Probing season {year}...")
        print(f"  old URL available: {url_available(old_url)}  ({old_url})")
        print(f"  new URL available: {url_available(new_url)}  ({new_url})")
        df = fetch_season(year)
        df = df[df["season_type"] == "REG"]
        qbs = df[(df["position"] == "QB") | (df["attempts"] > 0)]
        qbs = qbs[(qbs["attempts"] > 0) | (qbs["carries"] > 0)]
        print(f"  {len(qbs):,} QB rows (reg season, attempts>0 or carries>0)")
        ward = qbs[qbs["player_display_name"].str.contains("Cam Ward", na=False)]
        print(f"\nCam Ward rows: {len(ward)}")
        if len(ward):
            cols = [c for c in ["player_display_name", "season", "week", "attempts", "passing_yards", "rushing_yards"] if c in ward.columns]
            print(ward[cols].to_string(index=False))
        return

    print("Detecting latest available season...")
    max_season = latest_available_season()
    seasons = list(range(1999, max_season + 1))
    print(f"  Latest: {max_season} — pulling {seasons[0]}–{seasons[-1]}\n")

    frames = []
    for year in seasons:
        print(f"  Fetching {year}...", end=" ", flush=True)
        df = fetch_season(year)
        cols = [c for c in KEEP_COLS if c in df.columns]
        frames.append(df[cols].copy())
        print(f"{len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["season_type"] == "REG"].copy()

    qbs = combined[(combined["position"] == "QB") | (combined["attempts"] > 0)].copy()
    qbs = qbs[(qbs["attempts"] > 0) | (qbs["carries"] > 0)].copy()
    qbs = qbs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    out_path = OUT_DIR / "weekly_player_data.parquet"
    qbs.to_parquet(out_path, index=False)

    print(f"\nDone.")
    print(f"  Total rows : {len(qbs):,}")
    print(f"  Seasons    : {qbs['season'].min()}–{qbs['season'].max()}")
    print(f"  Unique QBs : {qbs['player_id'].nunique():,}")
    print(f"  Saved to   : {out_path}")

    allen = qbs[qbs["player_display_name"].str.contains("Josh Allen", na=False)]
    print(f"\nJosh Allen rows: {len(allen)}")
    print(allen[["season", "week", "attempts", "passing_yards", "rushing_yards"]].tail(10).to_string(index=False))

    ward = qbs[qbs["player_display_name"].str.contains("Cam Ward", na=False)]
    print(f"\nCam Ward rows: {len(ward)}")
    if len(ward):
        print(ward[["season", "week", "attempts", "passing_yards", "rushing_yards"]].to_string(index=False))
    else:
        print("  NOT FOUND")


if __name__ == "__main__":
    main()
