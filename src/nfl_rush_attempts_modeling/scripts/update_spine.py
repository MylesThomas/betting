"""
Rebuild the NFL rush attempts spine for the current season and upload to S3.

Modes:
  --update  (default) Re-fetch current season from nfl_data_py, rebuild rolling
            features on full spine, upload to S3.
  --verify  Download spine from S3, re-pull current season, rebuild locally,
            compare completed-game rows. Hard-fail if row count drops.
            Does NOT upload.

S3 path:
  s3://the-odds-api-mt/nfl/rush_attempts_model/spine/nfl_rush_attempts_spine.parquet

Run:
  python src/nfl_rush_attempts_modeling/scripts/update_spine.py --season 2026
  python src/nfl_rush_attempts_modeling/scripts/update_spine.py --season 2026 --verify
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from step2_build_spine import (
    _norm,
    load_actuals,
    load_schedule,
    build_opp_rush_defense,
    build_spine,
)

S3_BUCKET          = "the-odds-api-mt"
S3_KEY             = "nfl/rush_attempts_model/spine/nfl_rush_attempts_spine.parquet"
HISTORICAL_SEASONS = [2023, 2024, 2025]

ET = ZoneInfo("America/New_York")
warnings.filterwarnings("ignore")


def current_nfl_season() -> int:
    now = datetime.now(ET)
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
    print(f"  Uploaded → s3://{S3_BUCKET}/{S3_KEY}")


def build_spine_for_seasons(seasons: list[int]) -> pd.DataFrame:
    team_sched = load_schedule()
    team_sched_current = team_sched[team_sched["season"].isin(seasons)].copy()
    actuals = load_actuals(team_sched_current)
    opp_def = build_opp_rush_defense(actuals, team_sched)
    # build_spine computes rolling features — pass empty consensus (not needed for spine)
    spine = build_spine(actuals, team_sched, opp_def, consensus=pd.DataFrame())
    return spine


def compare_spines(ref: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    removed = set(ref["nfl_game_id"]) - set(fresh["nfl_game_id"])
    added   = set(fresh["nfl_game_id"]) - set(ref["nfl_game_id"])
    print(f"  Reference : {len(ref):,} rows  ({ref['nfl_game_id'].nunique():,} games)")
    print(f"  Fresh     : {len(fresh):,} rows  ({fresh['nfl_game_id'].nunique():,} games)")
    if removed:
        print(f"  HARD FAIL: {len(removed)} game_ids disappeared!")
        for gid in sorted(removed)[:10]:
            print(f"    - {gid}")
        return False
    if added:
        print(f"  INFO: {len(added)} new game_ids (expected — newly completed games)")
    common_ids = set(ref["nfl_game_id"]) & set(fresh["nfl_game_id"])
    ref_c   = ref[ref["nfl_game_id"].isin(common_ids)].sort_values(
                    ["nfl_game_id", "player_name_norm"]).reset_index(drop=True)
    fresh_c = fresh[fresh["nfl_game_id"].isin(common_ids)].sort_values(
                    ["nfl_game_id", "player_name_norm"]).reset_index(drop=True)
    n_diffs = 0
    for col in ["carries", "carry_rate_L8"]:
        if col in ref_c.columns and col in fresh_c.columns:
            changed = ~np.isclose(
                ref_c[col].fillna(-9999), fresh_c[col].fillna(-9999), equal_nan=True
            )
            if changed.sum():
                print(f"  WARN: {changed.sum()} rows changed in '{col}'")
                n_diffs += changed.sum()
    if n_diffs == 0:
        print("  OK: all completed-game rows stable")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    args   = parser.parse_args()
    season = args.season or current_nfl_season()
    mode   = "verify" if args.verify else "update"

    print(f"\nNFL Rush Attempts Spine — mode={mode}  season={season}")
    print("=" * 60)

    s3_spine = None
    try:
        s3_spine = download_spine_s3()
        print(f"  S3 spine: {len(s3_spine):,} rows  "
              f"| seasons: {sorted(s3_spine['season'].unique())}")
    except Exception as e:
        print(f"  Could not load S3 spine ({e}) — full rebuild")

    if s3_spine is None:
        # Full rebuild
        all_seasons = sorted(set(HISTORICAL_SEASONS) | {season})
        print(f"\n  Full rebuild: seasons {all_seasons}")
        fresh_spine = build_spine_for_seasons(all_seasons)
    else:
        # Incremental: rebuild current season only, stitch with historical
        historical = s3_spine[s3_spine["season"] < season].copy()
        print(f"\n  Incremental: fetching season {season}...")
        current_raw = build_spine_for_seasons([season])
        fresh_spine = pd.concat([historical, current_raw], ignore_index=True)
        fresh_spine = fresh_spine.sort_values(
            ["player_name_norm", "season", "week"]
        ).reset_index(drop=True)

    n_current = len(fresh_spine[fresh_spine["season"] == season])
    print(f"  Full spine: {len(fresh_spine):,} rows  ({n_current:,} in {season})")

    if mode == "verify":
        if s3_spine is None:
            sys.exit("Cannot verify — no S3 spine to compare against.")
        ok = compare_spines(s3_spine, fresh_spine)
        print(f"\n{'='*60}")
        print(f"  Verify: {'STABLE' if ok else 'CHANGED'}")
        print(f"{'='*60}\n")
        if not ok:
            sys.exit(1)
    else:
        upload_spine_s3(fresh_spine)
        print(f"\n{'='*60}")
        print(f"  Season  : {season}")
        print(f"  Rows    : {len(fresh_spine):,}")
        print(f"  Players : {fresh_spine['player_name_norm'].nunique():,}")
        print(f"  Seasons : {fresh_spine['season'].min()}–{fresh_spine['season'].max()}")
        print(f"  Upload  : s3://{S3_BUCKET}/{S3_KEY}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
