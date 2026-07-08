"""
Step 1 EDA: Pitcher outs recorded market data.

Loads the merged market parquet (~/Downloads/tmp/mlb_pitcher_outs_market_raw.parquet)
and the pitcher gamelogs from S3, then produces:
  1. Coverage by season / book
  2. Line distribution
  3. Over/under/push hit rates
  4. DNP rate
  5. Spot-check: Freddy Peralta traces
  6. DuckDB SQL validation tests

Usage:
  python src/mlb_pitcher_outs_modeling/scripts/20260706_eda.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_pitcher_outs_market_raw.parquet"
LOCAL_LOGS   = Path.home() / "Downloads/tmp/mlb_pitcher_gamelogs.parquet"
LOCAL_JOINED = Path.home() / "Downloads/tmp/mlb_pitcher_outs_eda.parquet"

S3_BUCKET    = "the-odds-api-mt"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_gamelogs() -> pd.DataFrame:
    """Load pitcher gamelogs from S3 (reused from strikeouts pipeline)."""
    if LOCAL_LOGS.exists():
        return pd.read_parquet(LOCAL_LOGS)
    s3 = boto3.client("s3")
    frames = []
    for yr in [2024, 2025, 2026]:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=f"mlb/strikeouts_model/pitcher_gamelogs/{yr}.parquet")
        frames.append(pd.read_parquet(BytesIO(resp["Body"].read())))
    df = pd.concat(frames, ignore_index=True)
    LOCAL_LOGS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_LOGS, index=False)
    return df


def load_market() -> pd.DataFrame:
    if not LOCAL_MARKET.exists():
        raise FileNotFoundError(
            f"{LOCAL_MARKET} not found. Run fetch_market_lines.py first."
        )
    return pd.read_parquet(LOCAL_MARKET)


# ---------------------------------------------------------------------------
# Derive outs_recorded from innings_pitched
# ---------------------------------------------------------------------------

def ip_to_outs(ip) -> int:
    """Convert innings_pitched float (e.g., 5.1 → 16, 6.0 → 18, 7.2 → 23)."""
    s = str(float(ip))
    full, frac = s.split(".")
    return int(full) * 3 + int(frac[0])  # fractional digit = outs (0, 1, or 2)


# ---------------------------------------------------------------------------
# Normalize names for joining
# ---------------------------------------------------------------------------

import unicodedata, re

def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[^a-z ]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return name.strip()


# ---------------------------------------------------------------------------
# Main EDA
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    market = load_market()
    logs   = load_gamelogs()

    # Filter to pitcher_outs (primary line, not alternate for coverage check)
    props = market[market["market_key"] == "pitcher_outs"].copy()
    props_alt = market[market["market_key"] == "pitcher_outs_alternate"].copy()

    print(f"\n=== Raw counts ===")
    print(f"  market rows (pitcher_outs):           {len(props):,}")
    print(f"  market rows (pitcher_outs_alternate): {len(props_alt):,}")
    print(f"  gamelog rows:                         {len(logs):,}")
    print(f"  Gamelog seasons: {sorted(logs['season'].unique())}")
    print(f"  Market seasons:  {sorted(props['season'].unique())}")

    # -------------------------------------------------------------------------
    # 1. Coverage by season / book
    # -------------------------------------------------------------------------
    print("\n=== Coverage by season ===")
    cov = (props.groupby("season")
           .agg(
               n_rows=("player_name","count"),
               n_books=("bookmaker","nunique"),
               n_players=("player_name","nunique"),
               n_games=("event_id","nunique"),
           )
           .reset_index())
    print(cov.to_string(index=False))

    print("\n=== Coverage by book (all seasons) ===")
    book_cov = (props.groupby("bookmaker")
                .agg(
                    n_rows=("player_name","count"),
                    n_seasons=("season","nunique"),
                    n_games=("event_id","nunique"),
                )
                .sort_values("n_rows", ascending=False)
                .reset_index())
    print(book_cov.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Line distribution
    # -------------------------------------------------------------------------
    print("\n=== Line distribution (pitcher_outs) ===")
    line_dist = props["line"].value_counts().sort_index()
    print(line_dist.to_string())

    # -------------------------------------------------------------------------
    # Derive outs_recorded in gamelogs
    # -------------------------------------------------------------------------
    logs["outs_recorded"] = logs["innings_pitched"].apply(ip_to_outs)

    # -------------------------------------------------------------------------
    # 3. Join market to actuals (for hit rate + DNP)
    # -------------------------------------------------------------------------
    logs["name_norm"] = logs["player_name"].apply(normalize_name)
    props["name_norm"] = props["player_name"].apply(normalize_name)

    # Join on name_norm + game_date
    joined = props.merge(
        logs[["name_norm","game_date","outs_recorded","innings_pitched","season"]].drop_duplicates(),
        on=["name_norm","game_date"],
        how="left",
        suffixes=("","_log"),
    )

    # Mark unmatched
    unmatched_rate = joined["outs_recorded"].isna().mean()
    print(f"\n=== Join quality ===")
    print(f"  Total prop rows: {len(joined):,}")
    print(f"  Matched to gamelog: {(~joined['outs_recorded'].isna()).sum():,} ({1-unmatched_rate:.1%})")
    print(f"  Unmatched:          {joined['outs_recorded'].isna().sum():,} ({unmatched_rate:.1%})")

    # Among matched rows, mark outcome
    jm = joined[joined["outs_recorded"].notna()].copy()
    jm["outcome"] = np.where(
        jm["outs_recorded"] > jm["line"], "over",
        np.where(jm["outs_recorded"] < jm["line"], "under", "push")
    )

    # -------------------------------------------------------------------------
    # 4. Over/under/push hit rates
    # -------------------------------------------------------------------------
    print("\n=== Hit rates (matched rows) ===")
    rates = jm["outcome"].value_counts(normalize=True).sort_index()
    print(rates.apply(lambda x: f"{x:.1%}").to_string())

    # By season
    print("\n=== Hit rates by season ===")
    season_rates = jm.groupby("season")["outcome"].value_counts(normalize=True).unstack(fill_value=0)
    print(season_rates.applymap(lambda x: f"{x:.1%}").to_string())

    # -------------------------------------------------------------------------
    # 5. DNP rate (prop posted but player didn't start)
    # -------------------------------------------------------------------------
    # Unmatched rows where we expected a start but got none
    # Best proxy: join to gamelogs broadly and see how many names appear at all
    all_gamelog_names = set(logs["name_norm"].unique())
    props_names = set(props["name_norm"].unique())
    in_logs = props_names & all_gamelog_names
    not_in_logs = props_names - all_gamelog_names
    print(f"\n=== DNP / name matching ===")
    print(f"  Unique prop player names:            {len(props_names):,}")
    print(f"  Names found in gamelogs:             {len(in_logs):,} ({len(in_logs)/len(props_names):.1%})")
    print(f"  Names NOT in gamelogs (relief only): {len(not_in_logs):,}")
    if not_in_logs:
        print(f"  Sample not-in-logs: {sorted(not_in_logs)[:10]}")

    # -------------------------------------------------------------------------
    # 6. Spot-check: Freddy Peralta
    # -------------------------------------------------------------------------
    print("\n=== Spot-check: Freddy Peralta ===")
    fp_logs = logs[logs["player_name"] == "Freddy Peralta"].sort_values("game_date")
    print(f"  Gamelog starts: {len(fp_logs)}")
    print(fp_logs[["game_date","season","outs_recorded","innings_pitched","strikeouts","opponent_name","is_home"]].tail(10).to_string(index=False))

    fp_props = props[props["name_norm"] == "freddy peralta"].sort_values("game_date")
    print(f"\n  Prop rows: {len(fp_props)}")
    print(fp_props[["game_date","season","bookmaker","line","over_price","under_price",
                    "consensus_home_moneyline","consensus_away_moneyline",
                    "home_run_line_point","away_run_line_point"]].tail(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # Save joined file for DuckDB tests
    # -------------------------------------------------------------------------
    LOCAL_JOINED.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(LOCAL_JOINED, index=False)
    print(f"\nJoined saved → {LOCAL_JOINED}")

    # -------------------------------------------------------------------------
    # DuckDB SQL validation tests
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("DUCKDB VALIDATION TESTS")
    print("="*60)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW market AS SELECT * FROM read_parquet('{LOCAL_MARKET}')")
    con.execute(f"CREATE VIEW gamelogs AS SELECT *, "
                f"  CAST(SPLIT_PART(CAST(innings_pitched AS VARCHAR), '.', 1) AS INT)*3 + "
                f"  CAST(LEFT(SPLIT_PART(CAST(innings_pitched AS VARCHAR), '.', 2), 1) AS INT) AS outs_recorded "
                f"FROM read_parquet('{LOCAL_LOGS}')")
    con.execute(f"CREATE VIEW joined AS SELECT * FROM read_parquet('{LOCAL_JOINED}')")

    tests = []

    # Test 1: Row count in expected range
    n_props = con.execute("SELECT COUNT(*) FROM market WHERE market_key='pitcher_outs'").fetchone()[0]
    tests.append(("Row count pitcher_outs > 10,000", n_props > 10_000, f"n={n_props:,}"))

    # Test 2: No unexpected nulls in key columns
    null_check = con.execute("""
        SELECT COUNT(*) FROM market
        WHERE market_key='pitcher_outs'
          AND (player_name IS NULL OR game_date IS NULL OR line IS NULL OR bookmaker IS NULL)
    """).fetchone()[0]
    tests.append(("No nulls in player_name/game_date/line/bookmaker", null_check == 0, f"null_rows={null_check}"))

    # Test 3: Line distribution — variable market (not binary)
    distinct_lines = con.execute("SELECT COUNT(DISTINCT line) FROM market WHERE market_key='pitcher_outs'").fetchone()[0]
    tests.append(("Variable line market (>3 distinct lines)", distinct_lines > 3, f"distinct_lines={distinct_lines}"))

    # Test 4: Over+under+push rates sum to ~100%
    rate_check = con.execute("""
        WITH outcomes AS (
            SELECT
                CASE WHEN outs_recorded > line THEN 'over'
                     WHEN outs_recorded < line THEN 'under'
                     ELSE 'push' END AS outcome
            FROM joined
            WHERE outs_recorded IS NOT NULL AND market_key='pitcher_outs'
        )
        SELECT
            SUM(CASE WHEN outcome='over' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS over_rate,
            SUM(CASE WHEN outcome='under' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS under_rate,
            SUM(CASE WHEN outcome='push' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS push_rate
        FROM outcomes
    """).fetchone()
    total_rate = sum(r for r in rate_check if r is not None)
    tests.append(("over+under+push rates sum to ~100%", abs(total_rate - 1.0) < 0.01,
                  f"sum={total_rate:.4f} (over={rate_check[0]:.3f} under={rate_check[1]:.3f} push={rate_check[2]:.3f})"))

    # Test 5: Gamelog row count reasonable
    n_logs = con.execute("SELECT COUNT(*) FROM gamelogs").fetchone()[0]
    tests.append(("Gamelog row count > 5,000 starts", n_logs > 5_000, f"n={n_logs:,}"))

    # Test 6: Key stat columns null rate < 10%
    null_ip = con.execute("SELECT AVG(CASE WHEN innings_pitched IS NULL THEN 1 ELSE 0 END) FROM gamelogs").fetchone()[0]
    tests.append(("innings_pitched null rate <10%", null_ip < 0.10, f"null_rate={null_ip:.3f}"))

    # Test 7: Date range overlap between market and gamelog
    market_min, market_max = con.execute(
        "SELECT MIN(game_date), MAX(game_date) FROM market WHERE market_key='pitcher_outs'"
    ).fetchone()
    log_min, log_max = con.execute("SELECT MIN(game_date), MAX(game_date) FROM gamelogs").fetchone()
    overlap = market_min <= log_max and log_min <= market_max
    tests.append(("Market and gamelog dates overlap", overlap,
                  f"market={market_min}→{market_max}, logs={log_min}→{log_max}"))

    # Test 8: Game odds columns populated (team betting features exist)
    null_ml = con.execute("""
        SELECT AVG(CASE WHEN consensus_home_moneyline IS NULL THEN 1 ELSE 0 END)
        FROM market WHERE market_key='pitcher_outs'
    """).fetchone()[0]
    tests.append(("consensus_home_moneyline null rate <20%", null_ml < 0.20, f"null_rate={null_ml:.3f}"))

    # Print results
    passed = 0
    for name, ok, detail in tests:
        status = "PASS" if ok else "FAIL"
        mark   = "✓" if ok else "✗"
        print(f"  [{status}] {mark} {name}")
        print(f"         {detail}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed")

    if passed < len(tests):
        print("\n⚠ Some tests FAILED — investigate before proceeding to Step 2.")
    else:
        print("\n✓ All tests passed — ready to proceed to Step 2.")

    con.close()


if __name__ == "__main__":
    main()
