"""
Rebuild the NFL rec yards spine for the current season and upload to S3.

Modes:
  --update  (default) Re-fetch current season data from nfl_data_py,
            rebuild rolling features on full spine, upload to S3.
  --verify  Download spine from S3, re-pull current season, rebuild locally,
            compare completed-game rows. Hard-fail if row count drops.
            Does NOT upload.

S3 path:
  s3://the-odds-api-mt/nfl/rec_yards_model/spine/nfl_rec_yards_historical_spine.parquet

Run:
  python src/nfl_rec_yards_modeling/scripts/update_spine.py --season 2026
  python src/nfl_rec_yards_modeling/scripts/update_spine.py --season 2026 --verify
"""

import argparse
import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from build_historical_spine import (
    _normalize_name, add_rolling_features,
    build_game_context, build_game_id_map, build_id_bridge,
    build_opp_pass_defense, build_opponent_map, build_snap_counts,
    build_weekly_receiving,
)

S3_BUCKET          = "the-odds-api-mt"
S3_KEY             = "nfl/rec_yards_model/spine/nfl_rec_yards_historical_spine.parquet"
HISTORICAL_SEASONS = [2023, 2024, 2025]


def current_nfl_season() -> int:
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
    print(f"  Uploaded → s3://{S3_BUCKET}/{S3_KEY}")


def build_spine_for_seasons(seasons: list[int]) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    id_bridge = build_id_bridge()
    weekly    = build_weekly_receiving(seasons)
    snaps     = build_snap_counts(seasons, id_bridge)
    opp_pass  = build_opp_pass_defense(seasons)
    game_ctx  = build_game_context(seasons)
    game_ids  = build_game_id_map(seasons)

    spine = weekly.merge(game_ids, on=["season", "week", "team"], how="left")
    spine = spine[spine["game_id"].notna()].copy()

    snap_cols = snaps[["game_id", "season", "team", "player_id_snap",
                        "offense_snaps", "offense_pct"]].copy()
    spine = spine.merge(
        snap_cols.rename(columns={"player_id_snap": "player_id"}),
        on=["game_id", "season", "team", "player_id"],
        how="left",
    )
    spine["offense_pct"] = spine["offense_pct"].fillna(0.0)
    spine = spine.merge(opp_pass, on=["game_id", "opponent"], how="left")
    spine = spine.merge(game_ctx, on=["game_id", "team"],     how="left")
    spine = add_rolling_features(spine)
    spine["player_name_norm"] = spine["player_name"].map(_normalize_name)
    return spine


def compare_spines(ref: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    removed = set(ref["game_id"]) - set(fresh["game_id"])
    added   = set(fresh["game_id"]) - set(ref["game_id"])
    print(f"  Reference: {len(ref):,} rows  ({ref['game_id'].nunique():,} games)")
    print(f"  Fresh    : {len(fresh):,} rows  ({fresh['game_id'].nunique():,} games)")
    if removed:
        print(f"  HARD FAIL: {len(removed)} game_ids disappeared!")
        for gid in sorted(removed)[:10]:
            print(f"    - {gid}")
        return False
    if added:
        print(f"  INFO: {len(added)} new game_ids (new completed games)")
    numeric_cols = [c for c in ["receiving_yards", "offense_snaps", "offense_pct"]
                    if c in ref.columns and c in fresh.columns]
    common_ids = set(ref["game_id"]) & set(fresh["game_id"])
    ref_c   = ref[ref["game_id"].isin(common_ids)].sort_values(["game_id", "player_id"]).reset_index(drop=True)
    fresh_c = fresh[fresh["game_id"].isin(common_ids)].sort_values(["game_id", "player_id"]).reset_index(drop=True)
    n_diffs = 0
    for col in numeric_cols:
        changed = ~np.isclose(ref_c[col].fillna(-9999), fresh_c[col].fillna(-9999), equal_nan=True)
        if changed.sum():
            print(f"  WARN: {changed.sum()} rows changed in '{col}'")
            n_diffs += changed.sum()
    if n_diffs == 0:
        print("  OK: all completed-game rows stable")
    else:
        print(f"  WARN: {n_diffs} value changes — consider re-running --update")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    args   = parser.parse_args()
    season = args.season or current_nfl_season()
    mode   = "verify" if args.verify else "update"

    print(f"\nNFL Rec Yards Spine — mode={mode}  season={season}")
    print("=" * 55)

    s3_spine   = None
    historical = pd.DataFrame()
    try:
        s3_spine   = download_spine_s3()
        historical = s3_spine[s3_spine["season"] < season].copy()
        print(f"  S3 spine: {len(s3_spine):,} rows  →  {len(historical):,} historical rows")
    except Exception as e:
        print(f"  Could not load S3 spine ({e}) — full rebuild")

    if historical.empty:
        all_seasons = sorted(set(HISTORICAL_SEASONS) | {season})
        print(f"\n  Full rebuild: seasons {all_seasons}")
        fresh_spine = build_spine_for_seasons(all_seasons)
    else:
        print(f"\n  Incremental: fetching season {season}...")
        current_raw = build_spine_for_seasons([season])
        combined    = pd.concat([historical, current_raw], ignore_index=True)
        combined    = combined.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        fresh_spine = add_rolling_features(combined)
        fresh_spine["player_name_norm"] = fresh_spine["player_name"].map(_normalize_name)

    n_current = len(fresh_spine[fresh_spine["season"] == season])
    print(f"  Full spine: {len(fresh_spine):,} rows  ({n_current:,} in {season})")

    if mode == "verify":
        if s3_spine is None:
            sys.exit("Cannot verify — no S3 spine to compare against.")
        ok = compare_spines(s3_spine, fresh_spine)
        print(f"\n{'='*55}")
        print(f"  Verify: {'STABLE' if ok else 'CHANGED'}")
        print(f"{'='*55}\n")
        if not ok:
            sys.exit(1)
    else:
        upload_spine_s3(fresh_spine)
        print(f"\n{'='*55}")
        print(f"  Season  : {season}")
        print(f"  Rows    : {len(fresh_spine):,}")
        print(f"  Players : {fresh_spine['player_id'].nunique():,}")
        print(f"  Seasons : {fresh_spine['season'].min()}–{fresh_spine['season'].max()}")
        print(f"  Upload  : s3://{S3_BUCKET}/{S3_KEY}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
