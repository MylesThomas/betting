"""
Verify compute_clv() correctness using synthetic snapshot data with known CLV values.

Usage:
  uv run python src/mlb_total_bases_modeling/scripts/verify_clv.py

Checks:
  1. CLV columns have no unexpected nulls on rows that have a closing snapshot
  2. CLV values within plausible range (-300 to +300)
  3. clv_first_seen_cents equals first_seen_odds - closing_odds for each test case
  4. clv_tier classifications match expected tiers
  5. consensus CLV is median across books for each player+event+market_key
"""
from __future__ import annotations

import sys
from datetime import timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.mlb_total_bases_modeling.scripts.compute_clv import compute_clv

GAME_DATE = "2026-07-10"
SEASON    = 2026
EVENT_ID  = "synthetic_event_001"
COMMENCE  = f"{GAME_DATE}T23:05:00Z"  # 7pm ET

# Synthetic test cases: (player, book, market_key, snap1_odds, snap2_odds, closing_odds, line, expected_clv_first_seen, expected_tier)
_TEST_CASES = [
    # Beat the close by 25 cents — strong+ (favorable)
    ("Aaron Judge",     "fanduel",    "batter_total_bases",           -115, -120, -140, 1.5,  25, "strong+"),
    # Lost to close by 12 cents — moderate- (adverse)
    ("Aaron Judge",     "draftkings", "batter_total_bases",           -125, -120, -113, 1.5, -12, "moderate-"),
    # Tiny movement — ok (no suffix)
    ("Shohei Ohtani",   "fanduel",    "batter_total_bases",           -110, -112, -114, 1.5,   4, "ok"),
    # Large adverse — severe- (worst case)
    ("Shohei Ohtani",   "draftkings", "batter_total_bases",           -105, -110, -140, 1.5,  35, "severe+"),
    # Mild favorable — mild+
    ("Freddie Freeman", "fanduel",    "batter_total_bases_alternate",  110,  108,  103, 0.5,   7, "mild+"),
]


def _build_synthetic_df() -> pd.DataFrame:
    """
    Build a synthetic snapshot DataFrame with 3 snapshots per test case:
      - snap 1: 08:00 UTC (first_seen=True)
      - snap 2: 14:00 UTC (closest to 9am ET = 14:00 UTC)
      - snap 3: 18:00 UTC (closing — before commence_time at 23:05 UTC)
    """
    rows = []
    base_ts = [
        (f"{GAME_DATE}T08:00:00Z", True),
        (f"{GAME_DATE}T14:00:00Z", False),
        (f"{GAME_DATE}T18:00:00Z", False),
    ]
    snap_odds_by_snap = [0, 1, 2]  # index into (snap1_odds, snap2_odds, closing_odds)

    for player, book, market_key, s1, s2, s3, line, _expected_clv, _expected_tier in _TEST_CASES:
        snap_odds = [s1, s2, s3]
        for i, (ts, is_first) in enumerate(base_ts):
            odds = snap_odds[i]
            rows.append({
                "snapshot_ts_utc":              ts,
                "season":                       SEASON,
                "game_date":                    GAME_DATE,
                "event_id":                     EVENT_ID,
                "home_team":                    "New York Yankees",
                "away_team":                    "Los Angeles Dodgers",
                "commence_time":                COMMENCE,
                "bookmaker":                    book,
                "market_key":                   market_key,
                "player_name":                  player,
                "over_line":                    line,
                "over_american_odds":           -120,
                "under_line":                   line,
                "under_american_odds":          odds,
                "binary_player_game_first_seen": is_first,
                "last_odds_player_game":         None,
                "credits_before":               50000,
                "credits_after":                49990,
            })

    return pd.DataFrame(rows)


class _MockBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _MockPaginator:
    def __init__(self, prefix: str):
        self._prefix = prefix

    def paginate(self, **kwargs):
        return [{"Contents": [{"Key": f"{self._prefix}snapshot_synthetic.parquet"}]}]


class _MockS3Client:
    """Mimics boto3 S3 client — returns synthetic DataFrame on get_object."""

    def __init__(self, df: pd.DataFrame, prefix: str):
        self._df = df
        self._prefix = prefix

    def get_paginator(self, operation: str):
        return _MockPaginator(self._prefix)

    def get_object(self, Bucket: str, Key: str):
        buf = BytesIO()
        self._df.to_parquet(buf, index=False)
        buf.seek(0)
        return {"Body": _MockBody(buf.read())}


def main():
    synthetic_df = _build_synthetic_df()
    prefix = f"mlb/total_bases_model/prop_snapshots/{SEASON}/{GAME_DATE}/"
    mock_s3 = _MockS3Client(synthetic_df, prefix)

    print(f"\nRunning CLV verification against {len(_TEST_CASES)} synthetic test cases\n")

    failures: list[str] = []

    def ok(label: str, detail: str = ""):
        msg = f"  [PASS] {label}" + (f" — {detail}" if detail else "")
        print(msg)

    def fail(label: str, detail: str = ""):
        msg = f"  [FAIL] {label}" + (f" — {detail}" if detail else "")
        print(msg)
        failures.append(label)

    df = compute_clv(GAME_DATE, SEASON, s3_client=mock_s3)

    if df.empty:
        print("  [FAIL] compute_clv returned empty DataFrame")
        sys.exit(1)

    # Check 1: no nulls on rows that have a closing snapshot (all synthetic rows have one)
    cols_requiring_close = ["clv_first_seen_cents", "clv_9am_cents", "clv_line_shift"]
    null_counts = {c: df[c].isna().sum() for c in cols_requiring_close if c in df.columns}
    if any(v > 0 for v in null_counts.values()):
        fail("Check 1: No unexpected nulls", f"null counts: {null_counts}")
    else:
        ok("Check 1: No unexpected nulls", "clv_first_seen_cents, clv_9am_cents, clv_line_shift all populated")

    # Check 2: CLV values in plausible range
    for col in ["clv_first_seen_cents", "clv_9am_cents"]:
        if col not in df.columns:
            continue
        out_of_range = df[(df[col].notna()) & ((df[col] < -300) | (df[col] > 300))]
        if not out_of_range.empty:
            fail(f"Check 2: Range ({col})", f"{len(out_of_range)} rows outside [-300, +300]")
        else:
            ok(f"Check 2: Range ({col})", "all values in [-300, +300]")

    # Check 3: exact CLV values match expected
    print("\n  [Check 3] Exact CLV values:")
    header = f"  {'Player':20s}  {'Book':10s}  {'Mkt':10s}  {'Expected':>8}  {'Got':>8}  {'OK?':>5}"
    print(header)
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5}")
    check3_issues = []
    for player, book, market_key, s1, s2, s3, line, expected_clv, _ in _TEST_CASES:
        row = df[
            (df["player_name"] == player)
            & (df["bookmaker"] == book)
            & (df["market_key"] == market_key)
        ]
        if row.empty:
            check3_issues.append(f"Missing row: {player}/{book}/{market_key}")
            print(f"  {player[:20]:20s}  {book[:10]:10s}  {market_key[:10]:10s}  {expected_clv:>8}  {'MISSING':>8}  {'FAIL':>5}")
            continue
        got_clv = row["clv_first_seen_cents"].iloc[0]
        passed = got_clv == expected_clv
        status = "PASS" if passed else "FAIL"
        print(f"  {player[:20]:20s}  {book[:10]:10s}  {market_key[:10]:10s}  {expected_clv:>8}  {str(got_clv):>8}  {status:>5}")
        if not passed:
            check3_issues.append(f"{player}/{book}: expected {expected_clv}, got {got_clv}")
    if check3_issues:
        fail("Check 3: Exact CLV values", "; ".join(check3_issues))
    else:
        ok("Check 3: Exact CLV values", f"all {len(_TEST_CASES)} test cases match")

    # Check 4: tier classifications
    print("\n  [Check 4] Tier classifications:")
    tier_issues = []
    for player, book, market_key, *_, expected_tier in _TEST_CASES:
        row = df[
            (df["player_name"] == player)
            & (df["bookmaker"] == book)
            & (df["market_key"] == market_key)
        ]
        if row.empty:
            continue
        got_tier = row["clv_tier"].iloc[0]
        passed = got_tier == expected_tier
        status = "PASS" if passed else "FAIL"
        print(f"  {player[:20]:20s}  {book[:10]:10s}  expected={expected_tier:10s}  got={got_tier:10s}  {status}")
        if not passed:
            tier_issues.append(f"{player}/{book}: expected {expected_tier}, got {got_tier}")
    if tier_issues:
        fail("Check 4: Tier classifications", "; ".join(tier_issues))
    else:
        ok("Check 4: Tier classifications", "all tiers correct")

    # Check 5: consensus CLV = median across books
    # Aaron Judge has 2 books: clv_first_seen = +25 and -12 → median = (+25 + -12) / 2 = 6.5
    judge_rows = df[df["player_name"] == "Aaron Judge"]
    if not judge_rows.empty and "consensus_clv_first_seen" in df.columns:
        expected_consensus = (25 + (-12)) / 2  # = 6.5
        got_consensus = judge_rows["consensus_clv_first_seen"].iloc[0]
        if abs(float(got_consensus) - expected_consensus) < 0.01:
            ok("Check 5: Consensus CLV", f"Aaron Judge median = {got_consensus} (expected {expected_consensus})")
        else:
            fail("Check 5: Consensus CLV", f"expected {expected_consensus}, got {got_consensus}")
    else:
        fail("Check 5: Consensus CLV", "consensus_clv_first_seen column missing or no Aaron Judge rows")

    # Summary
    n_passed = 6 - len(failures)  # 6 sub-checks above (2 for range check)
    print(f"\n{'='*55}")
    print(f"{5 - len(failures)}/5 checks passed")
    if failures:
        for f in failures:
            print(f"  - {f}")
    print("="*55)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
