"""
Step 1 EDA: Pitcher walks market data.

Loads the merged market parquet (~/Downloads/tmp/mlb_pitcher_walks_market_raw.parquet)
and the pitcher gamelogs from S3, then produces:
  1. Coverage by season / book
  2. Line distribution
  3. Over/under/push hit rates (by season and overall)
  4. DNP / name matching quality
  5. Spot-check: Freddy Peralta traces
  6. DuckDB SQL validation tests

Usage:
  python src/mlb_pitcher_walks_modeling/scripts/20260706_eda.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from io import BytesIO
from pathlib import Path

import boto3
import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_pitcher_walks_market_raw.parquet"
LOCAL_LOGS   = Path.home() / "Downloads/tmp/mlb_pitcher_gamelogs.parquet"
LOCAL_JOINED = Path.home() / "Downloads/tmp/mlb_pitcher_walks_eda.parquet"

S3_BUCKET = "the-odds-api-mt"


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
# Name normalization (same as pitcher_outs)
# ---------------------------------------------------------------------------

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

    props = market[market["market_key"] == "pitcher_walks"].copy()

    print(f"\n=== Raw counts ===")
    print(f"  market rows (pitcher_walks): {len(props):,}")
    print(f"  gamelog rows:                {len(logs):,}")
    print(f"  Gamelog seasons: {sorted(logs['season'].unique())}")
    print(f"  Market seasons:  {sorted(props['season'].unique())}")

    # -------------------------------------------------------------------------
    # 1. Coverage by season / book
    # -------------------------------------------------------------------------
    print("\n=== Coverage by season ===")
    cov = (props.groupby("season")
           .agg(
               n_rows=("player_name", "count"),
               n_books=("bookmaker", "nunique"),
               n_players=("player_name", "nunique"),
               n_games=("event_id", "nunique"),
           )
           .reset_index())
    print(cov.to_string(index=False))

    print("\n=== Coverage by book (all seasons) ===")
    book_cov = (props.groupby("bookmaker")
                .agg(
                    n_rows=("player_name", "count"),
                    n_seasons=("season", "nunique"),
                    n_games=("event_id", "nunique"),
                )
                .sort_values("n_rows", ascending=False)
                .reset_index())
    print(book_cov.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Line distribution
    # -------------------------------------------------------------------------
    print("\n=== Line distribution (pitcher_walks) ===")
    line_dist = props["line"].value_counts().sort_index()
    print(line_dist.to_string())

    # -------------------------------------------------------------------------
    # 3. Join market to actuals
    # -------------------------------------------------------------------------
    logs["name_norm"]  = logs["player_name"].apply(normalize_name)
    props["name_norm"] = props["player_name"].apply(normalize_name)

    joined = props.merge(
        logs[["name_norm", "game_date", "walks", "innings_pitched", "season"]].drop_duplicates(),
        on=["name_norm", "game_date"],
        how="left",
        suffixes=("", "_log"),
    )

    unmatched_rate = joined["walks"].isna().mean()
    print(f"\n=== Join quality ===")
    print(f"  Total prop rows:    {len(joined):,}")
    print(f"  Matched to gamelog: {(~joined['walks'].isna()).sum():,} ({1 - unmatched_rate:.1%})")
    print(f"  Unmatched:          {joined['walks'].isna().sum():,} ({unmatched_rate:.1%})")
    unmatched_names = joined[joined["walks"].isna()]["player_name"].unique()
    if len(unmatched_names) > 0:
        print(f"  Sample unmatched names: {sorted(unmatched_names)[:10]}")

    # Among matched rows, mark outcome
    jm = joined[joined["walks"].notna()].copy()
    jm["outcome"] = np.where(
        jm["walks"] > jm["line"], "over",
        np.where(jm["walks"] < jm["line"], "under", "push")
    )

    # -------------------------------------------------------------------------
    # 4. Over/under/push hit rates
    # -------------------------------------------------------------------------
    print("\n=== Hit rates (matched rows, all seasons) ===")
    rates = jm["outcome"].value_counts(normalize=True).sort_index()
    print(rates.apply(lambda x: f"{x:.1%}").to_string())

    print("\n=== Hit rates by season ===")
    season_rates = jm.groupby("season")["outcome"].value_counts(normalize=True).unstack(fill_value=0)
    print(season_rates.applymap(lambda x: f"{x:.1%}").to_string())

    print("\n=== Hit rates by line ===")
    line_rates = jm.groupby("line")["outcome"].value_counts(normalize=True).unstack(fill_value=0)
    print(line_rates.applymap(lambda x: f"{x:.1%}").to_string())

    # -------------------------------------------------------------------------
    # 5. Walks distribution in gamelogs
    # -------------------------------------------------------------------------
    print("\n=== Walks distribution in gamelogs (all seasons, starters only) ===")
    starters = logs[logs.get("games_started", pd.Series(1, index=logs.index)) == 1] if "games_started" in logs.columns else logs
    print(starters["walks"].describe().round(3).to_string())
    print("\nValue counts (walks per start):")
    print(starters["walks"].value_counts().sort_index().to_string())

    # -------------------------------------------------------------------------
    # 6. Spot-check: Freddy Peralta
    # -------------------------------------------------------------------------
    print("\n=== Spot-check: Freddy Peralta ===")
    fp_logs  = logs[logs["player_name"] == "Freddy Peralta"].sort_values("game_date")
    print(f"  Gamelog starts: {len(fp_logs)}")
    print(fp_logs[["game_date", "season", "walks", "strikeouts", "innings_pitched", "opponent_name", "is_home"]].tail(10).to_string(index=False))

    fp_props = props[props["name_norm"] == "freddy peralta"].sort_values("game_date")
    print(f"\n  Prop rows: {len(fp_props)}")
    if not fp_props.empty:
        print(fp_props[["game_date", "season", "bookmaker", "line", "over_price", "under_price",
                         "consensus_home_moneyline", "consensus_away_moneyline"]].tail(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # Save joined file
    # -------------------------------------------------------------------------
    LOCAL_JOINED.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(LOCAL_JOINED, index=False)
    print(f"\nJoined saved → {LOCAL_JOINED}")

    # -------------------------------------------------------------------------
    # DuckDB SQL validation tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DUCKDB VALIDATION TESTS")
    print("=" * 60)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW market   AS SELECT * FROM read_parquet('{LOCAL_MARKET}')")
    con.execute(f"CREATE VIEW gamelogs AS SELECT * FROM read_parquet('{LOCAL_LOGS}')")
    con.execute(f"CREATE VIEW joined   AS SELECT * FROM read_parquet('{LOCAL_JOINED}')")

    tests = []

    n_props = con.execute("SELECT COUNT(*) FROM market WHERE market_key='pitcher_walks'").fetchone()[0]
    tests.append(("Row count pitcher_walks > 5,000", n_props > 5_000, f"n={n_props:,}"))

    null_check = con.execute("""
        SELECT COUNT(*) FROM market
        WHERE market_key='pitcher_walks'
          AND (player_name IS NULL OR game_date IS NULL OR line IS NULL OR bookmaker IS NULL)
    """).fetchone()[0]
    tests.append(("No nulls in player_name/game_date/line/bookmaker", null_check == 0, f"null_rows={null_check}"))

    distinct_lines = con.execute("SELECT COUNT(DISTINCT line) FROM market WHERE market_key='pitcher_walks'").fetchone()[0]
    tests.append(("Has multiple distinct lines (>1)", distinct_lines > 1, f"distinct_lines={distinct_lines}"))

    rate_check = con.execute("""
        WITH outcomes AS (
            SELECT
                CASE WHEN walks > line THEN 'over'
                     WHEN walks < line THEN 'under'
                     ELSE 'push' END AS outcome
            FROM joined
            WHERE walks IS NOT NULL AND market_key='pitcher_walks'
        )
        SELECT
            SUM(CASE WHEN outcome='over'  THEN 1 ELSE 0 END)*1.0/COUNT(*) AS over_rate,
            SUM(CASE WHEN outcome='under' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS under_rate,
            SUM(CASE WHEN outcome='push'  THEN 1 ELSE 0 END)*1.0/COUNT(*) AS push_rate
        FROM outcomes
    """).fetchone()
    total_rate = sum(r for r in rate_check if r is not None)
    tests.append(("over+under+push rates sum to ~100%", abs(total_rate - 1.0) < 0.01,
                  f"sum={total_rate:.4f} (over={rate_check[0]:.3f} under={rate_check[1]:.3f} push={rate_check[2]:.3f})"))

    n_logs = con.execute("SELECT COUNT(*) FROM gamelogs").fetchone()[0]
    tests.append(("Gamelog row count > 5,000 starts", n_logs > 5_000, f"n={n_logs:,}"))

    null_walks = con.execute("SELECT AVG(CASE WHEN walks IS NULL THEN 1 ELSE 0 END) FROM gamelogs").fetchone()[0]
    tests.append(("walks null rate in gamelogs <5%", null_walks < 0.05, f"null_rate={null_walks:.3f}"))

    market_min, market_max = con.execute(
        "SELECT MIN(game_date), MAX(game_date) FROM market WHERE market_key='pitcher_walks'"
    ).fetchone()
    log_min, log_max = con.execute("SELECT MIN(game_date), MAX(game_date) FROM gamelogs").fetchone()
    overlap = market_min <= log_max and log_min <= market_max
    tests.append(("Market and gamelog dates overlap", overlap,
                  f"market={market_min}→{market_max}, logs={log_min}→{log_max}"))

    null_ml = con.execute("""
        SELECT AVG(CASE WHEN consensus_home_moneyline IS NULL THEN 1 ELSE 0 END)
        FROM market WHERE market_key='pitcher_walks'
    """).fetchone()[0]
    tests.append(("consensus_home_moneyline null rate <20%", null_ml < 0.20, f"null_rate={null_ml:.3f}"))

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
