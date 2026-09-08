"""
Step 1.5 — Market Calibration by Line for MLB Game Totals.

Joins totals lines with actual game scores, groups by line value,
and computes calibration metrics (actual hit rate vs market implied probability).

Identifies lines where the market systematically misprices the over/under.

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step15_calibration.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET  = "the-odds-api-mt"
LINES_KEY  = "mlb/game_totals_model/lines/mlb_game_totals_lines.parquet"
SCORES_KEY = "mlb/game_totals_model/scores/mlb_team_game_scores.parquet"
LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
JOINED_OUT = LOCAL_DIR / "step1_joined.parquet"

# Lines > MAX_LINE are excluded as data quality issues (alt lines / live lines)
MAX_LINE = 13.0

TEAM_NORMALIZE = {"Athletics": "Oakland Athletics"}


def normalize_team(name: str) -> str:
    return TEAM_NORMALIZE.get(name, name)


def load_s3_parquet(key: str) -> pd.DataFrame:
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from S3...")
    lines  = load_s3_parquet(LINES_KEY)
    scores = load_s3_parquet(SCORES_KEY)

    print(f"  Lines:  {len(lines)} rows, {lines.event_id.nunique()} events")
    print(f"  Scores: {len(scores)} rows")

    # Normalize team names
    for df in [lines, scores]:
        df["home_team"] = df["home_team"].map(normalize_team)
        df["away_team"] = df["away_team"].map(normalize_team)

    # Filter out implausible lines
    n_before = len(lines)
    lines = lines[lines["line"] <= MAX_LINE].copy()
    print(f"  Filtered lines > {MAX_LINE}: removed {n_before - len(lines)} rows")

    # Compute novig probabilities on lines before filtering
    lines["novig_prob_over"]  = lines["raw_prob_over"]  / (lines["raw_prob_over"] + lines["raw_prob_under"])
    lines["novig_prob_under"] = lines["raw_prob_under"] / (lines["raw_prob_over"] + lines["raw_prob_under"])

    # Handle doubleheaders — keep first game per matchup-date
    dh_keys = set(
        map(tuple,
            scores.groupby(["game_date", "home_team", "away_team"])
            .filter(lambda g: len(g) > 1)[["game_date", "home_team", "away_team"]]
            .drop_duplicates().values
        )
    )
    scores_single = (
        scores[scores.apply(lambda r: (r.game_date, r.home_team, r.away_team) not in dh_keys, axis=1)]
    )
    scores_dh_first = (
        scores[scores.apply(lambda r: (r.game_date, r.home_team, r.away_team) in dh_keys, axis=1)]
        .sort_values("game_pk")
        .groupby(["game_date", "home_team", "away_team"]).first().reset_index()
    )
    scores_clean = pd.concat([scores_single, scores_dh_first], ignore_index=True)

    # Join
    joined = lines.merge(
        scores_clean[["game_date", "home_team", "away_team", "total_runs"]],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )
    settled = joined[joined["total_runs"].notna()].copy()

    print(f"\nSettled rows (have actuals): {len(settled)}")

    # Outcome flags
    settled["hit_over"]  = (settled["total_runs"] > settled["line"]).astype(int)
    settled["hit_under"] = (settled["total_runs"] < settled["line"]).astype(int)
    settled["hit_push"]  = (settled["total_runs"] == settled["line"]).astype(int)

    # ── Calibration by line ──────────────────────────────────────────────────
    cal = (
        settled.groupby("line")
        .agg(
            n_bets           = ("hit_over",         "count"),
            over_rate        = ("hit_over",          "mean"),
            under_rate       = ("hit_under",         "mean"),
            push_rate        = ("hit_push",          "mean"),
            avg_raw_over     = ("raw_prob_over",     "mean"),
            avg_raw_under    = ("raw_prob_under",    "mean"),
        )
        .reset_index()
    )
    cal["avg_combined_vig"]    = cal["avg_raw_over"] + cal["avg_raw_under"] - 1
    cal["avg_novig_over"]      = cal["avg_raw_over"]  / (cal["avg_raw_over"] + cal["avg_raw_under"])
    cal["avg_novig_under"]     = cal["avg_raw_under"] / (cal["avg_raw_over"] + cal["avg_raw_under"])
    cal["calibration_gap_over"]  = cal["over_rate"]  - cal["avg_novig_over"]
    cal["calibration_gap_under"] = cal["under_rate"] - cal["avg_novig_under"]

    for col in ["over_rate", "under_rate", "push_rate",
                "avg_raw_over", "avg_raw_under", "avg_combined_vig",
                "avg_novig_over", "avg_novig_under",
                "calibration_gap_over", "calibration_gap_under"]:
        cal[col] = cal[col].round(3)

    print("\n=== CALIBRATION TABLE BY LINE ===")
    print(cal.to_string(index=False))

    # Flag miscalibrated lines
    print("\nFlagged rows (|calibration_gap| > 0.05):")
    flagged = cal[
        (cal["calibration_gap_over"].abs() > 0.05) |
        (cal["calibration_gap_under"].abs() > 0.05)
    ]
    if len(flagged) > 0:
        print(flagged[["line", "n_bets", "over_rate", "avg_novig_over",
                        "calibration_gap_over", "calibration_gap_under"]].to_string(index=False))
    else:
        print("  None — market appears well-calibrated across all lines")

    # ── NYY @ BOS spot check ────────────────────────────────────────────────
    spot = settled[
        (settled["home_team"] == "Boston Red Sox") &
        (settled["away_team"] == "New York Yankees")
    ]
    print(f"\n=== SPOT CHECK: NYY @ BOS ({len(spot)} rows) ===")
    if not spot.empty:
        print(spot[["game_date", "bookmaker", "line", "over_price", "under_price",
                    "raw_prob_over", "novig_prob_over", "total_runs",
                    "hit_over", "hit_under"]].to_string(index=False))
    else:
        print("  No settled NYY @ BOS rows yet")

    # ── SQL Tests ────────────────────────────────────────────────────────────
    print("\n=== STEP 1.5 SQL TESTS ===")
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE cal AS SELECT * FROM cal")
    con.execute("CREATE TABLE settled AS SELECT * FROM settled")

    tests = [
        ("1. over + under + push sum to ~1 for all lines (within 1%)", """
            SELECT line, ROUND(over_rate + under_rate + push_rate, 3) AS sum_rates,
                   ABS(over_rate + under_rate + push_rate - 1) AS abs_err
            FROM cal
            WHERE ABS(over_rate + under_rate + push_rate - 1) > 0.01
        """, "Should return 0 rows"),
        ("2. Lines with < 30 observations (low-confidence calibration)", """
            SELECT line, n_bets FROM cal WHERE n_bets < 30 ORDER BY n_bets
        """, "Flag low-N lines"),
        ("3. avg_raw_prob_over and avg_raw_prob_under between 0 and 1", """
            SELECT COUNT(*) FILTER (WHERE avg_raw_over < 0 OR avg_raw_over > 1) AS bad_over,
                   COUNT(*) FILTER (WHERE avg_raw_under < 0 OR avg_raw_under > 1) AS bad_under
            FROM cal
        """, "Should both be 0"),
        ("4. Total bets in cal matches settled row count", """
            SELECT SUM(n_bets) AS total_bets_in_cal,
                   (SELECT COUNT(*) FROM settled) AS total_settled_rows
            FROM cal
        """, "Should match"),
    ]

    all_pass = True
    for name, sql, expectation in tests:
        print(f"\n  {name} ({expectation})")
        result = con.execute(sql).fetchdf()
        print(result.to_string(index=False))
        if result.empty:
            print("  PASS (0 rows)")
        elif "bad_over" in result.columns:
            if result["bad_over"].iloc[0] == 0 and result["bad_under"].iloc[0] == 0:
                print("  PASS")
            else:
                print("  FAIL — invalid probability values")
                all_pass = False

    con.close()
    print(f"\n{'ALL TESTS PASS' if all_pass else 'SOME TESTS NEED REVIEW'}")


if __name__ == "__main__":
    main()
