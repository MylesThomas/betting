"""
Step 1.5 — Market Calibration by Line — MLB Batter Hits
Join market lines to actuals, compute calibration table per line value.

Output columns per line:
  n_bets, over_rate, under_rate, push_rate,
  avg_raw_prob_over, avg_raw_prob_under, avg_combined_vig,
  avg_novig_prob_over, avg_novig_prob_under,
  calibration_gap_over, calibration_gap_under

Also: calibration by line × bookmaker (to spot outlier books).

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_MARKET  = Path.home() / "Downloads/tmp/mlb_batter_hits_market_raw.parquet"
LOCAL_ACTUALS = Path.home() / "Downloads/tmp/mlb_batting_statcast.parquet"


def normalize_name(name: str) -> str:
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


def main() -> None:
    print("Loading market data...")
    market = pd.read_parquet(LOCAL_MARKET)
    print(f"  {len(market):,} rows")

    print("Loading actuals...")
    actuals = pd.read_parquet(LOCAL_ACTUALS)
    print(f"  {len(actuals):,} rows")

    # Normalize names
    market["name_norm"]  = market["player_name"].apply(normalize_name)
    actuals["name_norm"] = actuals["player_name"].apply(normalize_name)

    # Dedup actuals to player-game level (keep highest AB per date for doubleheaders)
    actuals_dedup = (
        actuals.sort_values("ab", ascending=False)
        .drop_duplicates(subset=["game_date", "name_norm"])
    )

    # Join
    joined = market.merge(
        actuals_dedup[["game_date", "name_norm", "hits", "ab"]],
        on=["game_date", "name_norm"],
        how="left",
    )
    settled = joined.dropna(subset=["hits"]).copy()
    settled["outcome_over"]  = (settled["hits"] > settled["line"]).astype(int)
    settled["outcome_under"] = (settled["hits"] < settled["line"]).astype(int)
    settled["outcome_push"]  = (settled["hits"] == settled["line"]).astype(int)

    print(f"\nSettled rows: {len(settled):,} of {len(joined):,} joined ({len(settled)/len(joined)*100:.1f}%)")

    # ---------------------------------------------------------------
    # Two-sided rows only for calibration (both over_price and under_price present)
    # ---------------------------------------------------------------
    two_sided = settled.dropna(subset=["over_price", "under_price"]).copy()
    two_sided["raw_prob_over"]  = 1.0 / two_sided["over_price"]
    two_sided["raw_prob_under"] = 1.0 / two_sided["under_price"]
    two_sided["vig"]            = two_sided["raw_prob_over"] + two_sided["raw_prob_under"] - 1.0
    two_sided["novig_prob_over"]  = two_sided["raw_prob_over"]  / (two_sided["raw_prob_over"] + two_sided["raw_prob_under"])
    two_sided["novig_prob_under"] = two_sided["raw_prob_under"] / (two_sided["raw_prob_over"] + two_sided["raw_prob_under"])

    print(f"Two-sided rows (both prices): {len(two_sided):,} ({len(two_sided)/len(settled)*100:.1f}% of settled)")

    # ---------------------------------------------------------------
    # Calibration table by line
    # ---------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1.5 — CALIBRATION BY LINE")
    print("="*80)

    con = duckdb.connect()
    con.register("ts", two_sided)
    con.register("settled", settled)

    calib_sql = """
    SELECT
        line,
        COUNT(*)                                          AS n_bets,
        ROUND(AVG(outcome_over),  3)                     AS over_rate,
        ROUND(AVG(outcome_under), 3)                     AS under_rate,
        ROUND(AVG(outcome_push),  3)                     AS push_rate,
        ROUND(AVG(raw_prob_over), 3)                     AS avg_raw_prob_over,
        ROUND(AVG(raw_prob_under),3)                     AS avg_raw_prob_under,
        ROUND(AVG(vig),           3)                     AS avg_combined_vig,
        ROUND(AVG(novig_prob_over), 3)                   AS avg_novig_prob_over,
        ROUND(AVG(novig_prob_under),3)                   AS avg_novig_prob_under,
        ROUND(AVG(outcome_over)  - AVG(novig_prob_over), 3)  AS calibration_gap_over,
        ROUND(AVG(outcome_under) - AVG(novig_prob_under),3)  AS calibration_gap_under
    FROM ts
    GROUP BY line
    ORDER BY line
    """
    calib = con.execute(calib_sql).df()
    print(calib.to_string(index=False))

    print("\n--- Flagged lines where |calibration_gap| > 0.05 ---")
    flagged = calib[
        (calib["calibration_gap_over"].abs() > 0.05) |
        (calib["calibration_gap_under"].abs() > 0.05)
    ]
    if len(flagged) == 0:
        print("  None — all lines within 5pp of fair.")
    else:
        print(flagged.to_string(index=False))

    # ---------------------------------------------------------------
    # Two-sided coverage by line (how many rows have both prices)
    # ---------------------------------------------------------------
    print("\n--- Two-sided coverage by line ---")
    coverage_sql = """
    SELECT
        line,
        COUNT(*) AS total_settled,
        SUM(CASE WHEN under_price IS NOT NULL THEN 1 ELSE 0 END) AS two_sided,
        ROUND(SUM(CASE WHEN under_price IS NOT NULL THEN 1.0 ELSE 0.0 END) / COUNT(*), 3) AS two_sided_pct
    FROM settled
    GROUP BY line
    ORDER BY line
    """
    coverage = con.execute(coverage_sql).df()
    print(coverage.to_string(index=False))

    # ---------------------------------------------------------------
    # Calibration by line × bookmaker (top books only)
    # ---------------------------------------------------------------
    print("\n--- Calibration by Bookmaker at line = 0.5 ---")
    bk_sql = """
    SELECT
        bookmaker,
        COUNT(*)                                                     AS n_bets,
        ROUND(AVG(outcome_over), 3)                                  AS over_rate,
        ROUND(AVG(novig_prob_over), 3)                               AS avg_novig_over,
        ROUND(AVG(outcome_over) - AVG(novig_prob_over), 3)           AS gap_over,
        ROUND(AVG(vig), 3)                                           AS avg_vig
    FROM ts
    WHERE line = 0.5
    GROUP BY bookmaker
    HAVING COUNT(*) >= 100
    ORDER BY gap_over DESC
    """
    bk_calib = con.execute(bk_sql).df()
    print(bk_calib.to_string(index=False))

    # ---------------------------------------------------------------
    # Calibration by line × bookmaker at line 1.5
    # ---------------------------------------------------------------
    print("\n--- Calibration by Bookmaker at line = 1.5 ---")
    bk15_sql = """
    SELECT
        bookmaker,
        COUNT(*)                                                     AS n_bets,
        ROUND(AVG(outcome_over), 3)                                  AS over_rate,
        ROUND(AVG(novig_prob_over), 3)                               AS avg_novig_over,
        ROUND(AVG(outcome_over) - AVG(novig_prob_over), 3)           AS gap_over,
        ROUND(AVG(vig), 3)                                           AS avg_vig
    FROM ts
    WHERE line = 1.5
    GROUP BY bookmaker
    HAVING COUNT(*) >= 30
    ORDER BY gap_over DESC
    """
    bk15_calib = con.execute(bk15_sql).df()
    print(bk15_calib.to_string(index=False))

    # ---------------------------------------------------------------
    # Spot-check: Freddie Freeman calibration
    # ---------------------------------------------------------------
    print("\n--- Spot-Check: Freddie Freeman Calibration by Line ---")
    ff_sql = """
    SELECT
        line,
        COUNT(*) AS n_bets,
        ROUND(AVG(outcome_over), 3) AS over_rate,
        ROUND(AVG(novig_prob_over), 3) AS avg_novig_over,
        ROUND(AVG(outcome_over) - AVG(novig_prob_over), 3) AS gap_over
    FROM ts
    WHERE name_norm = 'freddie freeman'
    GROUP BY line
    ORDER BY line
    """
    ff_calib = con.execute(ff_sql).df()
    print(ff_calib.to_string(index=False))

    # ---------------------------------------------------------------
    # DuckDB SQL Tests
    # ---------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1.5 — DuckDB SQL TESTS")
    print("="*80)

    tests = [
        ("T1: over+under+push sum to ~1.0 at every line (within 1%)",
         "SELECT BOOL_AND(ABS(over_rate + under_rate + push_rate - 1.0) < 0.01) AS pass FROM ("
         "  SELECT line, ROUND(AVG(outcome_over),3) AS over_rate, ROUND(AVG(outcome_under),3) AS under_rate, ROUND(AVG(outcome_push),3) AS push_rate"
         "  FROM ts GROUP BY line"
         ")"),
        ("T2: Lines 0.5 and 1.5 each have >=1000 two-sided observations",
         "SELECT BOOL_AND(n >= 1000) AS pass FROM (SELECT line, COUNT(*) AS n FROM ts WHERE line IN (0.5, 1.5) GROUP BY line)"),
        ("T3: avg_raw_prob_over between 0 and 1 for all lines",
         "SELECT BOOL_AND(avg_raw_prob_over BETWEEN 0 AND 1) AS pass FROM ("
         "  SELECT line, AVG(raw_prob_over) AS avg_raw_prob_over FROM ts GROUP BY line"
         ")"),
        ("T4: avg_raw_prob_under between 0 and 1 for all lines",
         "SELECT BOOL_AND(avg_raw_prob_under BETWEEN 0 AND 1) AS pass FROM ("
         "  SELECT line, AVG(raw_prob_under) AS avg_raw_prob_under FROM ts GROUP BY line"
         ")"),
        ("T5: Total bets in two-sided table matches expected (>500,000)",
         "SELECT COUNT(*) > 500000 AS pass FROM ts"),
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
        print("All Step 1.5 tests PASSED.")
    else:
        print("Some Step 1.5 tests FAILED.")

    return calib, bk_calib, coverage


if __name__ == "__main__":
    main()
