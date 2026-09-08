"""
Verify that the snapshot Lambda ran correctly for today (or a specified date).

Usage:
  uv run python src/mlb_total_bases_modeling/scripts/verify_snapshot.py
  uv run python src/mlb_total_bases_modeling/scripts/verify_snapshot.py --date 2026-09-01
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/total_bases_model/prop_snapshots"

REQUIRED_COLUMNS = [
    "snapshot_ts_utc",
    "season",
    "game_date",
    "event_id",
    "home_team",
    "away_team",
    "commence_time",
    "bookmaker",
    "market_key",
    "player_name",
    "over_line",
    "over_american_odds",
    "under_line",
    "under_american_odds",
    "binary_player_game_first_seen",
    "last_odds_player_game",
    "credits_before",
    "credits_after",
]

BOOL_COLS   = {"binary_player_game_first_seen"}
INT_COLS    = {"over_american_odds", "under_american_odds", "credits_before", "credits_after"}
FLOAT_COLS  = {"over_line", "under_line"}
# everything else is str (season is int but acceptable to check separately)


def _list_snapshot_keys(s3_client, check_date: str) -> list[str]:
    year = check_date[:4]
    prefix = f"{S3_PREFIX}/{year}/{check_date}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    return sorted(keys)


def _load_parquet(s3_client, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def _load_all_for_date(s3_client, check_date: str) -> pd.DataFrame:
    keys = _list_snapshot_keys(s3_client, check_date)
    if not keys:
        return pd.DataFrame()
    frames = [_load_parquet(s3_client, k) for k in keys]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Verify mlb total bases snapshot for a date.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Game date to check (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()
    check_date = args.date

    s3 = boto3.client("s3")
    failures: list[str] = []

    def ok(label: str):
        print(f"  [PASS] {label}")

    def fail(label: str, detail: str = ""):
        msg = f"  [FAIL] {label}" + (f" — {detail}" if detail else "")
        print(msg)
        failures.append(label)

    print(f"\nVerifying snapshot for date: {check_date}\n")

    # ── Check 1: latest snapshot exists ──────────────────────────────────────
    keys = _list_snapshot_keys(s3, check_date)
    if keys:
        ok(f"Check 1: Latest snapshot exists ({len(keys)} file(s) found)")
    else:
        fail("Check 1: Latest snapshot exists", f"No parquet files under {S3_PREFIX}/<year>/{check_date}/")
        print("\n0/7 checks passed (cannot continue without data)")
        sys.exit(1)

    # Load the latest snapshot for subsequent checks
    latest_key = keys[-1]
    df = _load_parquet(s3, latest_key)
    print(f"  Loaded: s3://{S3_BUCKET}/{latest_key}  ({len(df):,} rows)\n")

    # ── Check 2: schema complete ──────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        fail("Check 2: Schema complete", f"Missing columns: {missing_cols}")
    else:
        dtype_issues = []
        for col in BOOL_COLS:
            if col in df.columns and df[col].dtype != bool:
                dtype_issues.append(f"{col} expected bool, got {df[col].dtype}")
        for col in INT_COLS:
            if col in df.columns:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    # Int cols may be float64 due to NaN; check the non-null values are whole numbers
                    if not all(float(v).is_integer() for v in non_null):
                        dtype_issues.append(f"{col} expected integer values")
        for col in FLOAT_COLS:
            if col in df.columns and not pd.api.types.is_float_dtype(df[col]):
                dtype_issues.append(f"{col} expected float, got {df[col].dtype}")
        if dtype_issues:
            fail("Check 2: Schema complete", "; ".join(dtype_issues))
        else:
            ok("Check 2: Schema complete (all columns present, dtypes correct)")

    # ── Check 3: first-seen rows exist ───────────────────────────────────────
    if df["binary_player_game_first_seen"].any():
        n_first = df["binary_player_game_first_seen"].sum()
        ok(f"Check 3: First-seen rows exist ({n_first:,} first-seen rows)")
    else:
        fail("Check 3: First-seen rows exist", "No binary_player_game_first_seen=True rows found")

    # ── Check 4: last_odds null on first-seen ────────────────────────────────
    first_seen_mask = df["binary_player_game_first_seen"]
    if first_seen_mask.any():
        all_null = df.loc[first_seen_mask, "last_odds_player_game"].isna().all()
        if all_null:
            ok("Check 4: last_odds_player_game is null on all first-seen rows")
        else:
            n_bad = df.loc[first_seen_mask, "last_odds_player_game"].notna().sum()
            fail("Check 4: last_odds null on first-seen", f"{n_bad} first-seen rows have non-null last_odds_player_game")
    else:
        ok("Check 4: last_odds null on first-seen (skipped — no first-seen rows)")

    # ── Check 5: last_odds populated on non-first-seen (if any) ─────────────
    non_first_mask = ~df["binary_player_game_first_seen"]
    if non_first_mask.any():
        n_non_first    = non_first_mask.sum()
        n_populated    = df.loc[non_first_mask, "last_odds_player_game"].notna().sum()
        pct_populated  = n_populated / n_non_first
        if pct_populated >= 0.50:
            ok(f"Check 5: last_odds populated on non-first-seen ({pct_populated:.0%} of {n_non_first:,} rows)")
        else:
            fail(
                "Check 5: last_odds populated on non-first-seen",
                f"Only {pct_populated:.0%} of {n_non_first:,} non-first-seen rows have last_odds (need ≥50%)",
            )
    else:
        ok("Check 5: last_odds populated on non-first-seen (skipped — no non-first-seen rows)")

    # ── Check 6: credits consumed ─────────────────────────────────────────────
    credits_before = df["credits_before"].iloc[0]
    credits_after  = df["credits_after"].iloc[0]
    if credits_before > credits_after:
        ok(f"Check 6: Credits consumed ({credits_before:,} → {credits_after:,}, used {credits_before - credits_after})")
    else:
        fail("Check 6: Credits consumed", f"credits_before={credits_before} not > credits_after={credits_after}")

    # ── Check 7: spot-print (human eyeball) ──────────────────────────────────
    print("\n--- HUMAN EYEBALL CHECK ---")
    all_df = _load_all_for_date(s3, check_date)
    if all_df.empty:
        fail("Check 7: Spot-print (human eyeball)", "No data loaded for date")
    else:
        # Pick 3 random (player_name, event_id, bookmaker) combos
        combos = (
            all_df[["player_name", "event_id", "bookmaker"]]
            .drop_duplicates()
            .sample(n=min(3, len(all_df[["player_name", "event_id", "bookmaker"]].drop_duplicates())), random_state=42)
        )
        for _, combo_row in combos.iterrows():
            player   = combo_row["player_name"]
            event_id = combo_row["event_id"]
            bookmaker= combo_row["bookmaker"]
            subset = (
                all_df[
                    (all_df["player_name"] == player)
                    & (all_df["event_id"]   == event_id)
                    & (all_df["bookmaker"]  == bookmaker)
                ]
                .sort_values("snapshot_ts_utc")
                .tail(3)
            )
            print(f"\n  {player} | event={event_id[:8]}... | book={bookmaker}")
            print(f"  {'snapshot_ts_utc':25s}  {'over_line':>9}  {'over_odds':>9}  {'under_line':>10}  {'under_odds':>10}")
            for _, r in subset.iterrows():
                print(
                    f"  {str(r['snapshot_ts_utc']):25s}  "
                    f"{r['over_line']:>9.1f}  "
                    f"{str(r['over_american_odds']):>9}  "
                    f"{r['under_line']:>10.1f}  "
                    f"{str(r['under_american_odds']):>10}"
                )
        ok("Check 7: Spot-print (human eyeball) — review output above")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_passed = 7 - len(failures)
    print(f"\n{'='*50}")
    print(f"{n_passed}/7 checks passed")
    if failures:
        print("Failed checks:")
        for f in failures:
            print(f"  - {f}")
    print('='*50)

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
