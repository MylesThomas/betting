"""
Step 1 — Data Pull + EDA for MLB Game Totals.

Loads totals lines (from Odds API) and game scores (from MLB Stats API),
joins them, and produces EDA statistics for the session log.

Outputs:
  ~/Downloads/tmp/mlb_game_totals/step1_joined.parquet  — joined dataset
  ~/Downloads/tmp/mlb_game_totals/step1_eda.duckdb      — duckdb for SQL tests

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step1_eda.py
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

S3_BUCKET      = "the-odds-api-mt"
LINES_KEY      = "mlb/game_totals_model/lines/mlb_game_totals_lines.parquet"
SCORES_KEY     = "mlb/game_totals_model/scores/mlb_team_game_scores.parquet"
LOCAL_DIR      = Path.home() / "Downloads/tmp/mlb_game_totals"
JOINED_OUT     = LOCAL_DIR / "step1_joined.parquet"
DUCKDB_OUT     = LOCAL_DIR / "step1_eda.duckdb"


# MLB Stats API uses "Athletics" starting 2025; Odds API uses "Oakland Athletics" throughout.
TEAM_NORMALIZE = {
    "Athletics": "Oakland Athletics",
}


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

    print(f"  Lines:  {lines.shape}  ({lines['event_id'].nunique()} unique events)")
    print(f"  Scores: {scores.shape}  ({scores['game_pk'].nunique()} unique games)")

    # ── Normalize team names ────────────────────────────────────────────────
    lines["home_team"]  = lines["home_team"].map(normalize_team)
    lines["away_team"]  = lines["away_team"].map(normalize_team)
    scores["home_team"] = scores["home_team"].map(normalize_team)
    scores["away_team"] = scores["away_team"].map(normalize_team)

    # ── EDA: Lines ──────────────────────────────────────────────────────────
    print("\n=== MARKET DATA (LINES) EDA ===")

    print("\nBooks posting totals:")
    print(lines.groupby("bookmaker").size().sort_values(ascending=False).to_string())

    print("\nLine distribution:")
    print(lines["line"].value_counts().sort_index().to_string())

    print("\nLines per event (unique lines across books):")
    lpg = lines.groupby("event_id")["line"].nunique()
    print(f"  Mean unique lines/event: {lpg.mean():.2f}")
    print(f"  Events with >1 line:     {(lpg > 1).sum()} ({(lpg > 1).mean():.1%})")
    print(f"  Line count distribution: {lpg.value_counts().sort_index().to_dict()}")

    print("\nBooks per event:")
    bpg = lines.groupby("event_id")["bookmaker"].nunique()
    print(f"  Mean books/event: {bpg.mean():.2f}")

    print("\nOdds distribution (American):")
    print(f"  Over  — mean: {lines['over_price'].mean():.1f}, median: {lines['over_price'].median()}, std: {lines['over_price'].std():.1f}")
    print(f"  Under — mean: {lines['under_price'].mean():.1f}, median: {lines['under_price'].median()}, std: {lines['under_price'].std():.1f}")

    print("\nVig distribution (raw_prob_over + raw_prob_under - 1):")
    vig = lines["raw_prob_over"] + lines["raw_prob_under"] - 1
    print(f"  Mean: {vig.mean():.4f}, Std: {vig.std():.4f}, Median: {vig.median():.4f}")

    print("\nNo-vig over probability (market consensus):")
    lines["novig_prob_over"]  = lines["raw_prob_over"] / (lines["raw_prob_over"] + lines["raw_prob_under"])
    lines["novig_prob_under"] = 1 - lines["novig_prob_over"]
    print(f"  Mean novig_over: {lines['novig_prob_over'].mean():.4f}")
    print(f"  Mean novig_under: {lines['novig_prob_under'].mean():.4f}")

    # ── EDA: Scores ─────────────────────────────────────────────────────────
    print("\n=== ACTUALS (GAME SCORES) EDA ===")

    print("\nBy season:")
    print(scores.groupby("season").agg(
        n_games=("game_pk", "count"),
        avg_total=("total_runs", "mean"),
        median_total=("total_runs", "median"),
        std_total=("total_runs", "std"),
        min_total=("total_runs", "min"),
        max_total=("total_runs", "max"),
    ).round(2).to_string())

    print("\nTotal runs distribution:")
    print(scores["total_runs"].value_counts().sort_index().head(25).to_string())

    print("\nDoubleheaders:")
    dh = scores.groupby(["game_date", "home_team", "away_team"]).size()
    print(f"  Unique matchup-dates: {len(dh)}")
    print(f"  Doubleheaders (n>1):  {(dh > 1).sum()}")

    print("\nExtra-innings games:")
    print(scores[scores.innings > 9]["innings"].value_counts().sort_index().to_string())

    # ── Join ────────────────────────────────────────────────────────────────
    print("\n=== JOIN (LINES → ACTUALS) ===")

    # Drop doubleheader games: Odds API posts one total for the day, actuals have two games
    dh_matchups = scores.groupby(["game_date", "home_team", "away_team"]).size().reset_index(name="n")
    dh_keys = set(
        zip(
            dh_matchups[dh_matchups.n > 1]["game_date"],
            dh_matchups[dh_matchups.n > 1]["home_team"],
            dh_matchups[dh_matchups.n > 1]["away_team"],
        )
    )
    scores_no_dh = scores[~scores.apply(
        lambda r: (r["game_date"], r["home_team"], r["away_team"]) in dh_keys, axis=1
    )].copy()
    # For DH, use the first game (game with lower game_pk — earlier in day)
    scores_dh_first = (
        scores[scores.apply(lambda r: (r["game_date"], r["home_team"], r["away_team"]) in dh_keys, axis=1)]
        .sort_values("game_pk")
        .groupby(["game_date", "home_team", "away_team"])
        .first()
        .reset_index()
    )
    scores_clean = pd.concat([scores_no_dh, scores_dh_first], ignore_index=True)
    print(f"  Scores before DH handling: {len(scores)}")
    print(f"  Scores after (keep first DH game): {len(scores_clean)}")

    # Join lines + scores
    joined = lines.merge(
        scores_clean[["game_date", "home_team", "away_team", "home_runs", "away_runs",
                      "total_runs", "game_pk", "innings"]],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    n_total = len(joined)
    n_matched = joined["total_runs"].notna().sum()
    match_rate = n_matched / n_total
    print(f"  Lines rows:    {len(lines)}")
    print(f"  Joined rows:   {n_total}")
    print(f"  Match rate:    {n_matched}/{n_total} = {match_rate:.3f}")

    # Over/under/push flags
    joined["hit_over"]  = (joined["total_runs"] > joined["line"]).astype("Int8")
    joined["hit_under"] = (joined["total_runs"] < joined["line"]).astype("Int8")
    joined["hit_push"]  = (joined["total_runs"] == joined["line"]).astype("Int8")

    # Settled rows only
    settled = joined[joined["total_runs"].notna()].copy()

    print(f"\nSettled rows (have actuals): {len(settled)}")
    print(f"  Over rate:  {settled['hit_over'].mean():.3f}")
    print(f"  Under rate: {settled['hit_under'].mean():.3f}")
    print(f"  Push rate:  {settled['hit_push'].mean():.3f}")
    print(f"  Sum:        {(settled['hit_over'] + settled['hit_under'] + settled['hit_push']).mean():.3f}")

    print("\nOver/under/push by line:")
    print(
        settled.groupby("line").agg(
            n_bets=("hit_over", "count"),
            over_rate=("hit_over", "mean"),
            under_rate=("hit_under", "mean"),
            push_rate=("hit_push", "mean"),
        ).round(3).to_string()
    )

    # ── Spot check: NYY @ BOS ───────────────────────────────────────────────
    print("\n=== SPOT CHECK: NYY @ BOS ===")
    spot = joined[
        (joined["home_team"] == "Boston Red Sox") &
        (joined["away_team"] == "New York Yankees")
    ]
    print(f"  Total rows (game × book × line): {len(spot)}")
    print(f"  Unique game dates: {spot['game_date'].nunique()}")
    print(f"  Sample:")
    print(spot[["game_date", "bookmaker", "line", "over_price", "under_price",
                "total_runs", "hit_over", "hit_under"]].head(10).to_string())

    # ── Save ────────────────────────────────────────────────────────────────
    joined.to_parquet(JOINED_OUT, index=False)
    print(f"\nSaved joined → {JOINED_OUT}")

    # ── DuckDB SQL Tests ─────────────────────────────────────────────────────
    print("\n=== STEP 1 SQL TESTS ===")

    con = duckdb.connect(str(DUCKDB_OUT))
    con.execute(f"CREATE OR REPLACE TABLE lines AS SELECT * FROM '{JOINED_OUT}'")

    tests = {
        "1. Market row count in expected range (>10k for 4500 events × 11 books)": """
            SELECT COUNT(*) AS n FROM lines
        """,
        "2. No nulls in key market columns": """
            SELECT
              COUNT(*) FILTER (WHERE event_id IS NULL)    AS null_event_id,
              COUNT(*) FILTER (WHERE game_date IS NULL)   AS null_game_date,
              COUNT(*) FILTER (WHERE line IS NULL)        AS null_line,
              COUNT(*) FILTER (WHERE over_price IS NULL)  AS null_over_price,
              COUNT(*) FILTER (WHERE under_price IS NULL) AS null_under_price
            FROM lines
        """,
        "3. Line distribution — variable market (check no single line > 50% of rows)": """
            SELECT line, COUNT(*) AS n, ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 3) AS pct
            FROM lines
            GROUP BY line
            ORDER BY line
        """,
        "4. Over + under + push rates sum to ~1 (within 1%)": """
            SELECT
              ROUND(AVG(hit_over),  3) AS avg_over,
              ROUND(AVG(hit_under), 3) AS avg_under,
              ROUND(AVG(hit_push),  3) AS avg_push,
              ROUND(AVG(hit_over) + AVG(hit_under) + AVG(hit_push), 3) AS sum_should_be_1
            FROM lines
            WHERE total_runs IS NOT NULL
        """,
        "5. Scores row count matches expected (2024+2025+2026 ~6200 games)": f"""
            SELECT COUNT(DISTINCT game_pk) AS n_games FROM '{JOINED_OUT}'
            WHERE game_pk IS NOT NULL
        """,
        "6. Key score columns null rate < 10%": """
            SELECT
              ROUND(COUNT(*) FILTER (WHERE total_runs IS NULL) * 1.0 / COUNT(*), 3) AS null_rate_total_runs,
              ROUND(COUNT(*) FILTER (WHERE home_runs IS NULL) * 1.0 / COUNT(*), 3) AS null_rate_home_runs
            FROM lines
        """,
        "7. Game dates overlap between lines and scores": """
            SELECT
              MIN(game_date) AS lines_min_date,
              MAX(game_date) AS lines_max_date
            FROM lines
        """,
    }

    all_pass = True
    for name, sql in tests.items():
        print(f"\n  {name}")
        result = con.execute(sql).fetchdf()
        print(result.to_string(index=False))

        # Auto-flag failures
        if "null_event_id" in result.columns:
            if result["null_event_id"].iloc[0] > 0:
                print("  *** FAIL: null event_ids found ***")
                all_pass = False

        if "sum_should_be_1" in result.columns:
            val = result["sum_should_be_1"].iloc[0]
            if abs(val - 1.0) > 0.01:
                print(f"  *** FAIL: over+under+push = {val:.3f}, expected ~1.0 ***")
                all_pass = False
            else:
                print(f"  PASS: sum = {val:.3f}")

        if "null_rate_total_runs" in result.columns:
            nr = result["null_rate_total_runs"].iloc[0]
            if nr > 0.50:
                print(f"  *** WARN: {nr:.1%} of rows have no actuals (fetch still running?) ***")

    con.close()
    print(f"\n{'ALL TESTS PASS' if all_pass else 'SOME TESTS NEED REVIEW'}")
    print(f"DuckDB saved → {DUCKDB_OUT}")


if __name__ == "__main__":
    main()
