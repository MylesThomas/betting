"""
Rebuild the NFL sacks spine for the current season and upload to S3.

Modes:
  --update  (default) Re-fetch current season snap counts + PBP from nfl_data_py,
            rebuild rolling features on full spine, upload to S3.
  --verify  Download spine from S3, re-pull current season, rebuild locally,
            compare completed-game rows. Hard-fail if row count drops;
            warn if values differ. Does NOT upload.

S3 paths:
  s3://the-odds-api-mt/nfl/sacks_model/spine/nfl_sacks_historical_spine.parquet

Run:
  python src/nfl_sacks_modeling/scripts/update_spine.py --season 2026
  python src/nfl_sacks_modeling/scripts/update_spine.py --season 2026 --verify
"""

import argparse
import os
import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from build_historical_spine import (
    build_pbp_stats, build_snap_counts, join_season,
    add_rolling_features, add_position_group, load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

S3_BUCKET = "the-odds-api-mt"
S3_KEY    = "nfl/sacks_model/spine/nfl_sacks_historical_spine.parquet"

SNAP_FIRST_SEASON = 2013  # nfl_data_py snap counts start here


def current_nfl_season() -> int:
    """Season year = calendar year of kickoff (August onwards)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.year if now.month >= 8 else now.year - 1


def download_spine_s3() -> pd.DataFrame:
    print(f"  Downloading spine from s3://{S3_BUCKET}/{S3_KEY}...")
    body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=S3_KEY)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def upload_spine_s3(spine: pd.DataFrame) -> None:
    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf.getvalue())
    print(f"  Uploaded spine → s3://{S3_BUCKET}/{S3_KEY}")


def fetch_seasons(seasons: list[int], cfg: dict) -> pd.DataFrame:
    """Pull snap counts + PBP for one or more seasons and join them."""
    warnings.filterwarnings("ignore")
    pbp   = build_pbp_stats(seasons)
    snaps = build_snap_counts(seasons)
    frames = []
    for s in seasons:
        joined = join_season(snaps[snaps["season"] == s].copy(), pbp[pbp["season"] == s].copy())
        joined = add_position_group(joined, cfg["position_groups"], cfg["position_side"])
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rebuild_spine(historical: pd.DataFrame, current_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Concatenate historical (pre-current season) + fresh current season data,
    recompute rolling features across full career history."""
    combined = pd.concat([historical, current_raw], ignore_index=True)
    combined = combined.sort_values(["pfr_player_id", "season", "week", "game_id"]).reset_index(drop=True)

    windows = cfg["rolling_windows"]
    combined = add_rolling_features(combined, windows)

    col_order = [
        "game_id", "season", "week", "player", "pfr_player_id",
        "position", "pos_group", "pos_side", "team",
        "defense_snaps", "defense_pct", "sacks", "qb_hits",
        "games_played_ytd",
        *[f"{f}_L{('career' if w >= 999 else w)}"
          for f in ["sack_rate", "qbhit_rate", "snap_pct"]
          for w in windows],
    ]
    return combined[[c for c in col_order if c in combined.columns]]


def compare_spines(reference: pd.DataFrame, fresh: pd.DataFrame, season: int) -> bool:
    """Compare completed-game rows between two spines. Returns True if clean."""
    key = "game_id"
    ref_ids  = set(reference[key])
    fresh_ids = set(fresh[key])

    removed = ref_ids - fresh_ids
    added   = fresh_ids - ref_ids

    print(f"\n  Reference spine : {len(reference):,} rows  ({reference['game_id'].nunique():,} games)")
    print(f"  Fresh spine     : {len(fresh):,} rows  ({fresh['game_id'].nunique():,} games)")

    if removed:
        print(f"\n  HARD FAIL: {len(removed)} game_ids disappeared from spine!")
        for gid in sorted(removed)[:10]:
            print(f"    - {gid}")
        return False

    if added:
        print(f"  INFO: {len(added)} new game_ids in fresh spine (new completed games — expected)")

    # Compare values for rows present in both
    common_ids = ref_ids & fresh_ids
    ref_common   = reference[reference[key].isin(common_ids)].sort_values([key, "pfr_player_id"]).reset_index(drop=True)
    fresh_common = fresh[fresh[key].isin(common_ids)].sort_values([key, "pfr_player_id"]).reset_index(drop=True)

    numeric_cols = ["sacks", "qb_hits", "defense_snaps", "defense_pct"]
    numeric_cols = [c for c in numeric_cols if c in ref_common.columns and c in fresh_common.columns]

    n_diffs = 0
    for col in numeric_cols:
        ref_vals   = ref_common[col].fillna(-9999)
        fresh_vals = fresh_common[col].fillna(-9999)
        changed = ~np.isclose(ref_vals, fresh_vals, equal_nan=True)
        n_changed = changed.sum()
        if n_changed:
            print(f"  WARN: {n_changed} rows changed in column '{col}'")
            n_diffs += n_changed

    if n_diffs == 0:
        print("  OK: all completed-game rows are stable (no value changes)")
    else:
        print(f"\n  WARN: {n_diffs} value changes detected across completed games")
        print("  This may indicate nfl_data_py retroactively corrected stats.")
        print("  Consider re-running --update to sync S3 spine with corrected data.")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None,
                        help="Current NFL season year (default: computed from today's date)")
    parser.add_argument("--verify", action="store_true",
                        help="Compare fresh spine vs S3 spine instead of uploading")
    args = parser.parse_args()

    season = args.season or current_nfl_season()
    mode   = "verify" if args.verify else "update"

    print(f"\nNFL Sacks Spine — mode={mode}  season={season}")
    print(f"{'='*55}")

    cfg = load_config()

    # Download current S3 spine and strip current season rows
    s3_spine = pd.DataFrame()
    historical = pd.DataFrame()
    full_rebuild = False
    try:
        s3_spine = download_spine_s3()
        historical = s3_spine[s3_spine["season"] < season].copy()
        n_hist_seasons = historical["season"].nunique() if not historical.empty else 0
        print(f"  S3 spine loaded   : {len(s3_spine):,} rows  →  {len(historical):,} historical rows "
              f"({n_hist_seasons} seasons, pre-{season})")
        # If S3 spine is missing historical seasons, do a full rebuild
        if n_hist_seasons < (season - SNAP_FIRST_SEASON):
            print(f"  WARNING: S3 spine only has {n_hist_seasons} historical seasons "
                  f"(expected {season - SNAP_FIRST_SEASON}). Triggering full rebuild.")
            full_rebuild = True
            historical = pd.DataFrame()
    except Exception as e:
        print(f"  WARNING: Could not download S3 spine ({e})")
        full_rebuild = True

    if full_rebuild:
        all_seasons = list(range(SNAP_FIRST_SEASON, season + 1))
        print(f"\nFull rebuild — fetching all {len(all_seasons)} seasons from nfl_data_py "
              f"({SNAP_FIRST_SEASON}–{season})...")
        all_raw = fetch_seasons(all_seasons, cfg)
        print(f"  Fetched: {len(all_raw):,} player-game rows across {all_raw['season'].nunique()} seasons")
    else:
        print(f"\nFetching {season} season from nfl_data_py (incremental update)...")
        all_raw = fetch_seasons([season], cfg)
        print(f"  Fetched: {len(all_raw):,} player-game rows  ({all_raw['pfr_player_id'].nunique():,} players)")

    # Rebuild spine
    print("\nRebuilding rolling features on full spine...")
    fresh_spine = rebuild_spine(historical, all_raw, cfg)
    n_current = len(fresh_spine[fresh_spine["season"] == season])
    print(f"  Full spine: {len(fresh_spine):,} rows  ({n_current:,} in {season} season)")

    if mode == "verify":
        if s3_spine is None or s3_spine.empty:
            sys.exit("Cannot verify — no S3 spine to compare against.")
        ok = compare_spines(s3_spine, fresh_spine, season)
        if not ok:
            sys.exit(1)
        print(f"\n{'='*55}")
        print(f"  Verify complete — spine is {'STABLE' if ok else 'CHANGED'}")
        print(f"{'='*55}\n")
    else:
        upload_spine_s3(fresh_spine)
        print(f"\n{'='*55}")
        print(f"  Season  : {season}")
        print(f"  Rows    : {len(fresh_spine):,}")
        print(f"  Players : {fresh_spine['pfr_player_id'].nunique():,}")
        print(f"  Seasons : {fresh_spine['season'].min()}–{fresh_spine['season'].max()}")
        print(f"  Upload  : s3://{S3_BUCKET}/{S3_KEY}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
