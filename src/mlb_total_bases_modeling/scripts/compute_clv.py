"""
Compute CLV (Closing Line Value) for MLB total bases props from hourly snapshots.

Usage:
  uv run python src/mlb_total_bases_modeling/scripts/compute_clv.py
  uv run python src/mlb_total_bases_modeling/scripts/compute_clv.py --gameday 2026-09-05

Output schema (per player × event × bookmaker × market_key):
  player_name, event_id, bookmaker, market_key, game_date, season,
  first_seen_under_odds, nine_am_et_under_odds, closing_under_odds, closing_line,
  clv_first_seen_cents, clv_9am_cents, clv_line_shift, clv_tier,
  consensus_clv_first_seen, consensus_clv_9am
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/total_bases_model/prop_snapshots"


def _clv_tier(clv: Optional[float]) -> str:
    """
    Directional tier based on the CLV value with largest absolute magnitude.
    Positive = beat the close (favorable), negative = lost to close (adverse).
    'ok' has no direction suffix (movement < 5¢ either way).
    """
    if clv is None:
        return "unknown"
    magnitude = abs(clv)
    direction = "+" if clv >= 0 else "-"
    if magnitude < 5:
        return "ok"
    if magnitude < 10:
        return f"mild{direction}"
    if magnitude < 20:
        return f"moderate{direction}"
    if magnitude < 30:
        return f"strong{direction}"
    return f"severe{direction}"


def _ts_to_epoch(ts: str) -> float:
    """Convert ISO or YYYYMMDD_HHMMSS timestamp string to Unix epoch."""
    ts_clean = ts.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(ts_clean, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_all_snapshots(s3_client, season: int, game_date: str) -> pd.DataFrame:
    """Load all snapshot parquets for a given season/game_date from S3."""
    prefix = f"{S3_PREFIX}/{season}/{game_date}/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
    except Exception:
        return pd.DataFrame()

    if not keys:
        return pd.DataFrame()

    frames = []
    for key in sorted(keys):
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            frames.append(pd.read_parquet(BytesIO(obj["Body"].read())))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def compute_clv(
    game_date: str,
    season: int,
    s3_client=None,
    strategy_only: bool = True,
    markets: Optional[list[str]] = None,
    lines: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Compute CLV for all player×event×bookmaker×market_key combos for game_date.

    When strategy_only=True (default), filters to markets + lines and drops rows
    with null under_american_odds before computing. Caller is responsible for
    passing markets/lines from config.

    Returns empty DataFrame if no snapshot data exists for that date.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    df = _load_all_snapshots(s3_client, season, game_date)
    if df.empty:
        return pd.DataFrame()

    if strategy_only and markets is not None and lines is not None:
        df = df[
            df["market_key"].isin(markets)
            & df["under_line"].isin(lines)
            & df["under_american_odds"].notna()
        ].copy()
        if df.empty:
            return pd.DataFrame()

    # 09:00 ET = 14:00 UTC on the game date
    nine_am_epoch = _ts_to_epoch(f"{game_date}T14:00:00")

    results = []
    group_cols = ["player_name", "event_id", "bookmaker", "market_key"]

    for key_tuple, grp in df.groupby(group_cols):
        player_name, event_id, bookmaker, market_key = key_tuple
        grp = grp.sort_values("snapshot_ts_utc").reset_index(drop=True)

        commence_time = str(grp["commence_time"].iloc[0])
        commence_epoch = _ts_to_epoch(commence_time)

        # First-seen row
        first_mask = grp["binary_player_game_first_seen"].astype(bool)
        first_rows = grp[first_mask]
        first_seen_under_odds = _safe_int(
            first_rows["under_american_odds"].iloc[0] if not first_rows.empty else None
        )
        first_seen_line = _safe_float(
            first_rows["under_line"].iloc[0] if not first_rows.empty else None
        )

        # 9am ET snapshot — closest snapshot to 14:00 UTC on game_date
        ts_epochs = grp["snapshot_ts_utc"].astype(str).apply(_ts_to_epoch)
        nine_am_idx = (ts_epochs - nine_am_epoch).abs().idxmin()
        nine_am_et_under_odds = _safe_int(grp.loc[nine_am_idx, "under_american_odds"])

        # Closing snapshot — latest snapshot before game start
        pre_game = grp[ts_epochs < commence_epoch]
        if not pre_game.empty:
            closing_row = pre_game.iloc[-1]
            closing_under_odds = _safe_int(closing_row["under_american_odds"])
            closing_line = _safe_float(closing_row["under_line"])
        else:
            closing_under_odds = None
            closing_line = None

        # CLV: positive = beat the close (got a better price than at close)
        clv_first_seen = (
            first_seen_under_odds - closing_under_odds
            if first_seen_under_odds is not None and closing_under_odds is not None
            else None
        )
        clv_9am = (
            nine_am_et_under_odds - closing_under_odds
            if nine_am_et_under_odds is not None and closing_under_odds is not None
            else None
        )
        clv_line_shift = (
            closing_line - first_seen_line
            if closing_line is not None and first_seen_line is not None
            else None
        )

        # Tier: largest absolute movement of the two CLV references (directional)
        clvs = [c for c in [clv_first_seen, clv_9am] if c is not None]
        biggest_clv = max(clvs, key=abs) if clvs else None
        tier = _clv_tier(biggest_clv)

        results.append({
            "player_name":           player_name,
            "event_id":              event_id,
            "bookmaker":             bookmaker,
            "market_key":            market_key,
            "game_date":             game_date,
            "season":                season,
            "first_seen_under_odds": first_seen_under_odds,
            "nine_am_et_under_odds": nine_am_et_under_odds,
            "closing_under_odds":    closing_under_odds,
            "closing_line":          closing_line,
            "clv_first_seen_cents":  clv_first_seen,
            "clv_9am_cents":         clv_9am,
            "clv_line_shift":        clv_line_shift,
            "clv_tier":              tier,
        })

    if not results:
        return pd.DataFrame()

    out_df = pd.DataFrame(results)

    # Consensus CLV: median across books per (player_name, event_id, market_key)
    consensus_cols = ["player_name", "event_id", "market_key"]
    consensus = (
        out_df.groupby(consensus_cols)[["clv_first_seen_cents", "clv_9am_cents"]]
        .median()
        .rename(columns={
            "clv_first_seen_cents": "consensus_clv_first_seen",
            "clv_9am_cents":        "consensus_clv_9am",
        })
        .reset_index()
    )
    return out_df.merge(consensus, on=consensus_cols, how="left")


def main():
    parser = argparse.ArgumentParser(description="Compute CLV for MLB total bases props.")
    parser.add_argument(
        "--gameday",
        default=date.today().isoformat(),
        help="Game date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()
    game_date = args.gameday
    season = int(game_date[:4])

    print(f"Computing CLV for {game_date} (season {season})...")
    df = compute_clv(game_date, season)

    if df.empty:
        print("No CLV data — no snapshots found for this date.")
        return

    print(f"\n{len(df):,} player×bookmaker rows computed.\n")

    tier_counts = df["clv_tier"].value_counts()
    print("CLV tiers:")
    for tier in ["ok", "mild", "moderate", "strong", "severe", "unknown"]:
        print(f"  {tier:10s}: {tier_counts.get(tier, 0):,}")

    print("\nTop 10 worst CLV (first_seen):")
    sample = df.nsmallest(10, "clv_first_seen_cents")[
        ["player_name", "bookmaker", "market_key",
         "first_seen_under_odds", "closing_under_odds",
         "clv_first_seen_cents", "clv_9am_cents", "clv_line_shift", "clv_tier"]
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
