"""
Step 1 EDA — MLB Batter Hits
Loads fetched market lines from S3, loads Statcast actuals, and produces:
  1. Coverage check (seasons, books, games, players)
  2. Line distribution
  3. Over/under/push hit rates
  4. DNP rate
  5. Spot-check: Freddie Freeman trace
  6. DuckDB SQL validation tests

Run after 20260727_fetch_market_lines.py has completed.

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_eda.py
  python src/mlb_batter_hits_modeling/scripts/20260727_eda.py --rebuild  # re-pull from S3
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import boto3
import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET      = "the-odds-api-mt"
MARKET_PREFIX  = "mlb/batter_hits_model/market_raw"
ACTUALS_KEY    = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
LOCAL_MARKET   = Path.home() / "Downloads/tmp/mlb_batter_hits_market_raw.parquet"
LOCAL_ACTUALS  = Path.home() / "Downloads/tmp/mlb_batting_statcast.parquet"

SPOT_CHECK_PLAYER = "Freddie Freeman"


def load_market_from_s3() -> pd.DataFrame:
    """Rebuild merged market parquet from all per-event S3 files."""
    s3c    = boto3.client("s3")
    frames = []
    paginator = s3c.get_paginator("list_objects_v2")
    for season in [2024, 2025, 2026]:
        prefix = f"{MARKET_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Size"] == 0:
                    continue
                resp = s3c.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                df = pd.read_parquet(BytesIO(resp["Body"].read()))
                if len(df) > 0:
                    frames.append(df)
        print(f"  Season {season}: loaded {len(frames):,} files so far")

    if not frames:
        raise RuntimeError("No market data found in S3")

    combined = pd.concat(frames, ignore_index=True)
    LOCAL_MARKET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOCAL_MARKET, index=False)
    print(f"Market data saved → {LOCAL_MARKET}  ({len(combined):,} rows)")
    return combined


def load_actuals_from_s3() -> pd.DataFrame:
    """Load Statcast batting actuals from S3."""
    s3c = boto3.client("s3")
    obj = s3c.get_object(Bucket=S3_BUCKET, Key=ACTUALS_KEY)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    LOCAL_ACTUALS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_ACTUALS, index=False)
    return df


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, punctuation, suffixes for fuzzy matching."""
    import unicodedata, re
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"['''\-\.]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def run_eda(market: pd.DataFrame, actuals: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("STEP 1 EDA — MLB Batter Hits")
    print("="*70)

    # ---------------------------------------------------------------
    # 1. Market data overview
    # ---------------------------------------------------------------
    print("\n--- Market Data Overview ---")
    print(f"Total rows:        {len(market):,}")
    print(f"Unique players:    {market['player_name'].nunique():,}")
    print(f"Unique games:      {market['event_id'].nunique():,}")
    print(f"Unique books:      {market['bookmaker'].nunique():,}")
    print(f"Date range:        {market['game_date'].min()} → {market['game_date'].max()}")
    print(f"Seasons:           {sorted(market['season'].unique())}")
    print(f"Market keys:       {sorted(market['market_key'].unique())}")

    print("\nRows by season:")
    print(market.groupby("season").agg(
        rows=("player_name", "count"),
        events=("event_id", "nunique"),
        players=("player_name", "nunique"),
        books=("bookmaker", "nunique"),
    ).to_string())

    print("\nRows by bookmaker (sorted by rows desc):")
    bk = market.groupby("bookmaker").agg(
        rows=("player_name", "count"),
        pct=("player_name", lambda x: 100 * len(x) / len(market)),
    ).sort_values("rows", ascending=False)
    print(bk.to_string())

    # ---------------------------------------------------------------
    # 2. Line distribution
    # ---------------------------------------------------------------
    print("\n--- Line Distribution ---")
    line_dist = (
        market.groupby("line")
        .agg(rows=("player_name", "count"))
        .assign(pct=lambda df: 100 * df["rows"] / df["rows"].sum())
        .sort_index()
    )
    print(line_dist.to_string())

    primary_line_pct = line_dist.loc[0.5, "pct"] if 0.5 in line_dist.index else 0
    print(f"\nPrimary line 0.5: {primary_line_pct:.1f}% of all rows")

    # ---------------------------------------------------------------
    # 3. Null check
    # ---------------------------------------------------------------
    print("\n--- Null Check ---")
    for col in ["player_name", "game_date", "line", "over_price", "under_price", "bookmaker"]:
        null_rate = market[col].isna().mean() * 100
        flag = " ⚑" if null_rate > 1 else ""
        print(f"  {col:20s}: {null_rate:.2f}% null{flag}")

    # ---------------------------------------------------------------
    # 4. Join market to actuals for hit rate analysis
    # ---------------------------------------------------------------
    print("\n--- Join Market to Actuals ---")
    # Normalize names for matching
    market["name_norm"]   = market["player_name"].apply(normalize_name)
    actuals["name_norm"]  = actuals["player_name"].apply(normalize_name)

    # Actuals: deduplicate to player-game level
    # Need to handle doubleheaders: keep row with more AB when game_pk differs
    actuals_dedup = (
        actuals.sort_values("ab", ascending=False)
        .drop_duplicates(subset=["game_date", "name_norm"])  # keep highest-AB game per date
    )

    joined = market.merge(
        actuals_dedup[["game_date", "name_norm", "hits", "ab"]],
        on=["game_date", "name_norm"],
        how="left",
    )
    match_rate = joined["hits"].notna().mean() * 100
    print(f"Match rate (market rows with actuals):  {match_rate:.1f}%")
    print(f"Unmatched rows: {joined['hits'].isna().sum():,} of {len(joined):,}")

    # DNP check: line posted but hits is null
    dnp_rate = joined["hits"].isna().mean() * 100
    print(f"DNP rate (line posted, no actuals): {dnp_rate:.1f}%")

    # ---------------------------------------------------------------
    # 5. Over/under/push hit rates by line (settled rows only)
    # ---------------------------------------------------------------
    print("\n--- Hit Rates by Line (settled rows only) ---")
    settled = joined.dropna(subset=["hits"]).copy()
    settled["outcome_over"]  = (settled["hits"] > settled["line"]).astype(int)
    settled["outcome_under"] = (settled["hits"] < settled["line"]).astype(int)
    settled["outcome_push"]  = (settled["hits"] == settled["line"]).astype(int)

    hit_rates = settled.groupby("line").agg(
        n_bets=("hits", "count"),
        over_rate=("outcome_over", "mean"),
        under_rate=("outcome_under", "mean"),
        push_rate=("outcome_push", "mean"),
    ).reset_index()
    hit_rates["sum_check"] = hit_rates["over_rate"] + hit_rates["under_rate"] + hit_rates["push_rate"]
    print(hit_rates.round(3).to_string(index=False))

    # ---------------------------------------------------------------
    # 6. Implied prob vs actual rate
    # ---------------------------------------------------------------
    print("\n--- Market Implied Prob vs Actual Over Rate (line = 0.5) ---")
    line05 = settled[settled["line"] == 0.5].copy()
    if len(line05) > 0:
        # Odds API stores prices in decimal format already — no conversion needed
        line05["raw_prob_over"]  = 1 / line05["over_price"]
        line05["raw_prob_under"] = 1 / line05["under_price"]

        valid = line05.dropna(subset=["raw_prob_over", "raw_prob_under"])
        print(f"Rows at line 0.5 with valid odds: {len(valid):,}")
        print(f"Avg raw prob over:  {valid['raw_prob_over'].mean():.3f}")
        print(f"Avg raw prob under: {valid['raw_prob_under'].mean():.3f}")
        print(f"Avg vig:            {(valid['raw_prob_over'] + valid['raw_prob_under'] - 1).mean():.3f}")
        print(f"Actual over rate:   {line05['outcome_over'].mean():.3f}")
        print(f"Actual under rate:  {line05['outcome_under'].mean():.3f}")
        print(f"Calibration gap over:  {line05['outcome_over'].mean() - valid['raw_prob_over'].mean() / (valid['raw_prob_over'].mean() + valid['raw_prob_under'].mean()):.3f}")
        print(f"Calibration gap under: {line05['outcome_under'].mean() - valid['raw_prob_under'].mean() / (valid['raw_prob_over'].mean() + valid['raw_prob_under'].mean()):.3f}")

    # ---------------------------------------------------------------
    # 7. Spot-check: Freddie Freeman
    # ---------------------------------------------------------------
    print(f"\n--- Spot-Check: {SPOT_CHECK_PLAYER} ---")
    ff_norm = normalize_name(SPOT_CHECK_PLAYER)
    ff_market = joined[joined["name_norm"] == ff_norm].sort_values(["game_date", "line"])
    print(f"Market rows for {SPOT_CHECK_PLAYER}: {len(ff_market):,}")
    print(f"Games: {ff_market['game_date'].nunique()}")
    print(f"Books: {sorted(ff_market['bookmaker'].unique())}")
    print(f"Lines: {sorted(ff_market['line'].unique())}")
    print("\nSample (last 10 rows, line=0.5):")
    ff_05 = ff_market[ff_market["line"] == 0.5].tail(10)
    ff_05_display = ff_05[["game_date", "bookmaker", "line", "over_price", "under_price", "hits"]].copy()
    ff_05_display["outcome_over"] = (ff_05_display["hits"] > ff_05_display["line"]).where(ff_05_display["hits"].notna())
    print(ff_05_display.to_string(index=False))

    # ---------------------------------------------------------------
    # 8. DuckDB SQL tests
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1 — DuckDB SQL TESTS")
    print("="*70)
    con = duckdb.connect()
    con.register("market", market)
    con.register("actuals", actuals)
    con.register("joined", joined)
    con.register("settled", settled)

    tests = [
        ("T1: Market row count ≥ 10,000",
         "SELECT COUNT(*) >= 10000 AS pass FROM market"),
        ("T2: No nulls in player_name",
         "SELECT COUNT(*) = 0 AS pass FROM market WHERE player_name IS NULL"),
        ("T3: No nulls in game_date",
         "SELECT COUNT(*) = 0 AS pass FROM market WHERE game_date IS NULL"),
        ("T4: No nulls in line",
         "SELECT COUNT(*) = 0 AS pass FROM market WHERE line IS NULL"),
        ("T5: Lines 0.5+1.5+2.5 cover >95% of rows (multi-line market confirmed)",
         "SELECT (SUM(CASE WHEN line IN (0.5, 1.5, 2.5) THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) > 0.95 AS pass FROM market"),
        ("T6: Over+under+push sum to ~1.0 (within 1%) for line 0.5",
         "SELECT ABS(AVG(outcome_over) + AVG(outcome_under) + AVG(outcome_push) - 1.0) < 0.01 AS pass FROM settled WHERE line = 0.5"),
        ("T7: Actuals row count ≥ 50,000",
         "SELECT COUNT(*) >= 50000 AS pass FROM actuals"),
        ("T8: Actuals hits column null rate < 5%",
         "SELECT AVG(CASE WHEN hits IS NULL THEN 1.0 ELSE 0.0 END) < 0.05 AS pass FROM actuals"),
        ("T9: Date overlap — market and actuals share dates",
         "SELECT COUNT(*) > 0 AS pass FROM market m JOIN actuals a ON m.game_date = a.game_date"),
        ("T10: Match rate ≥ 80%",
         f"SELECT AVG(CASE WHEN hits IS NOT NULL THEN 1.0 ELSE 0.0 END) >= 0.80 AS pass FROM joined"),
    ]

    all_pass = True
    for name, sql in tests:
        try:
            result = con.execute(sql).fetchone()[0]
            status = "PASS" if result else "FAIL"
            if not result:
                all_pass = False
            print(f"  [{status}] {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            all_pass = False

    print()
    if all_pass:
        print("All Step 1 tests PASSED.")
    else:
        print("Some Step 1 tests FAILED — review above.")

    return joined, settled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Force reload from S3 even if local file exists")
    args = parser.parse_args()

    s3c = boto3.client("s3")

    if args.rebuild or not LOCAL_MARKET.exists():
        print("Loading market data from S3...")
        market = load_market_from_s3()
    else:
        print(f"Loading market data from local: {LOCAL_MARKET}")
        market = pd.read_parquet(LOCAL_MARKET)

    if args.rebuild or not LOCAL_ACTUALS.exists():
        print("Loading actuals from S3...")
        actuals = load_actuals_from_s3()
    else:
        print(f"Loading actuals from local: {LOCAL_ACTUALS}")
        actuals = pd.read_parquet(LOCAL_ACTUALS)

    run_eda(market, actuals)


if __name__ == "__main__":
    main()
