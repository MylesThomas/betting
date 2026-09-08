"""
Step 1 + 1.5 EDA and calibration for MLB game totals.

Requires:
  - mlb/game_totals_model/lines/mlb_game_totals_lines.parquet (fetch complete)
  - mlb/game_totals_model/scores/mlb_team_game_scores.parquet (fetched)

Outputs:
  - Prints EDA tables to stdout
  - Saves DuckDB test results
  - ~/Downloads/tmp/mlb_game_totals/eda_results.csv

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_eda_and_calibration.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
LINES_KEY   = "mlb/game_totals_model/lines/mlb_game_totals_lines.parquet"
SCORES_KEY  = "mlb/game_totals_model/scores/mlb_team_game_scores.parquet"
LOCAL_OUT   = Path.home() / "Downloads/tmp/mlb_game_totals/eda_results.csv"

PARK_FACTORS = {
    # Static park factors (runs per game relative to league average = 1.000)
    # Source: known ballpark characteristics, approximate
    "Colorado Rockies":    1.18,   # Coors Field — extreme altitude
    "Arizona Diamondbacks": 1.07,  # Chase Field — retractable roof, warm
    "Cincinnati Reds":     1.06,   # Great American Ball Park — hitter-friendly
    "Texas Rangers":       1.05,   # Globe Life Field — warm, humid
    "Boston Red Sox":      1.03,   # Fenway Park — short LF wall
    "Chicago Cubs":        1.03,   # Wrigley Field — wind-dependent
    "Philadelphia Phillies": 1.02, # Citizens Bank Park — fair hitter's park
    "Baltimore Orioles":   1.01,
    "Toronto Blue Jays":   1.00,
    "New York Yankees":    1.00,   # Yankee Stadium — slightly hitter-friendly
    "Houston Astros":      0.99,
    "Cleveland Guardians": 0.99,
    "Kansas City Royals":  0.99,
    "Atlanta Braves":      0.99,
    "Washington Nationals": 0.99,
    "New York Mets":       0.98,
    "Detroit Tigers":      0.98,
    "Minnesota Twins":     0.98,   # Target Field — cold April/May
    "Pittsburgh Pirates":  0.98,
    "Los Angeles Angels":  0.97,
    "St. Louis Cardinals": 0.97,
    "Milwaukee Brewers":   0.97,
    "Tampa Bay Rays":      0.97,   # Tropicana Field — dome, pitcher-friendly
    "Chicago White Sox":   0.97,
    "Miami Marlins":       0.96,
    "Oakland Athletics":   0.96,
    "Seattle Mariners":    0.95,
    "Los Angeles Dodgers": 0.94,   # Dodger Stadium — spacious
    "Athletics":           0.96,   # Sacramento Athletics (same franchise)
    "San Francisco Giants": 0.92,  # Oracle Park — cold, foggy
    "San Diego Padres":    0.91,   # Petco Park — large outfield
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    s3 = boto3.client("s3")

    body  = s3.get_object(Bucket=S3_BUCKET, Key=LINES_KEY)["Body"].read()
    lines = pd.read_parquet(BytesIO(body))

    body   = s3.get_object(Bucket=S3_BUCKET, Key=SCORES_KEY)["Body"].read()
    scores = pd.read_parquet(BytesIO(body))

    return lines, scores


def compute_novig(row: pd.Series) -> tuple[float, float]:
    raw_over  = row["raw_prob_over"]
    raw_under = row["raw_prob_under"]
    total     = raw_over + raw_under
    return raw_over / total, raw_under / total


def main() -> None:
    print("Loading data...")
    lines, scores = load_data()

    print(f"Lines: {len(lines)} rows, {lines.event_id.nunique()} events")
    print(f"Scores: {len(scores)} rows")
    print()

    # ── JOIN ────────────────────────────────────────────────────────────────
    merged = lines.merge(
        scores[["game_date", "home_team", "away_team", "total_runs", "innings"]],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    print("=== JOIN QUALITY ===")
    match_rate = merged["total_runs"].notna().mean()
    print(f"Match rate: {match_rate:.4f}")
    print(f"Unmatched: {merged['total_runs'].isna().sum()} rows")
    if merged["total_runs"].isna().sum() > 0:
        print("Sample unmatched:")
        print(merged[merged["total_runs"].isna()][["game_date","home_team","away_team"]].drop_duplicates().head(5))
    print()

    # Exclude short games (<9 innings — typically "No Action")
    merged_full   = merged.copy()
    short_games   = merged_full[merged_full["innings"] < 9].event_id.unique()
    merged_scored = merged_full[merged_full["total_runs"].notna()].copy()
    merged_valid  = merged_scored[~merged_scored["event_id"].isin(short_games)].copy()
    print(f"Short games excluded: {len(short_games)} events ({len(short_games)/merged_full.event_id.nunique()*100:.1f}%)")
    print(f"Valid (9+ innings): {merged_valid.event_id.nunique()} events, {len(merged_valid)} rows")
    print()

    # ── MARKET DATA EDA ─────────────────────────────────────────────────────
    print("=== MARKET DATA EDA ===")
    print(f"Date range: {lines.game_date.min()} to {lines.game_date.max()}")
    print(f"Total events: {lines.event_id.nunique()}")

    # Book coverage
    book_cov = lines.groupby("bookmaker")["event_id"].nunique().sort_values(ascending=False)
    print("\nBook coverage (events):")
    for book, n in book_cov.items():
        pct = n / lines.event_id.nunique() * 100
        print(f"  {book:20s}: {n:4d} ({pct:.0f}%)")

    # Lines per event
    lpg = lines.groupby("event_id")["line"].nunique()
    print("\nUnique lines per event:")
    print(lpg.value_counts().sort_index().to_string())

    # Line distribution
    print("\nLine distribution (all rows):")
    for line, n in lines.line.value_counts().sort_index().items():
        print(f"  {line:.1f}: {n:6d} ({n/len(lines)*100:.1f}%)")

    # Vig
    lines["vig"] = lines.raw_prob_over + lines.raw_prob_under - 1.0
    print(f"\nAvg vig: {lines.vig.mean():.4f} ({lines.vig.mean()*100:.2f}%)")
    print(f"Vig by book:")
    print(lines.groupby("bookmaker")["vig"].mean().sort_values().round(4).to_string())
    print()

    # ── ACTUALS EDA ─────────────────────────────────────────────────────────
    print("=== ACTUALS EDA ===")
    uniq_scores = scores.drop_duplicates("game_pk")
    print(f"Total unique games: {len(uniq_scores)}")
    print(f"Avg total runs: {uniq_scores.total_runs.mean():.2f}")
    by_season = uniq_scores.groupby("season").agg(
        games=("game_pk","count"),
        avg_total=("total_runs","mean"),
    ).round(2)
    print("By season:")
    print(by_season.to_string())
    print(f"Short games (<9 inn): {(uniq_scores.innings < 9).sum()}")
    print(f"Extra-inning games: {(uniq_scores.innings > 9).sum()} ({(uniq_scores.innings > 9).mean()*100:.1f}%)")
    print()

    # ── STEP 1.5 CALIBRATION ────────────────────────────────────────────────
    print("=== STEP 1.5 — MARKET CALIBRATION BY LINE ===")

    df = merged_valid.copy()
    df["novig_prob_over"]  = df["raw_prob_over"]  / (df["raw_prob_over"]  + df["raw_prob_under"])
    df["novig_prob_under"] = df["raw_prob_under"] / (df["raw_prob_over"]  + df["raw_prob_under"])
    df["is_over"]  = df["total_runs"] > df["line"]
    df["is_under"] = df["total_runs"] < df["line"]
    df["is_push"]  = df["total_runs"] == df["line"]

    calibration = (
        df.groupby("line")
        .agg(
            n_bets                = ("event_id", "count"),
            over_rate             = ("is_over",  "mean"),
            under_rate            = ("is_under", "mean"),
            push_rate             = ("is_push",  "mean"),
            avg_raw_prob_over     = ("raw_prob_over",   "mean"),
            avg_raw_prob_under    = ("raw_prob_under",  "mean"),
            avg_novig_prob_over   = ("novig_prob_over", "mean"),
            avg_novig_prob_under  = ("novig_prob_under","mean"),
        )
        .reset_index()
    )
    calibration["avg_combined_vig"]     = calibration["avg_raw_prob_over"] + calibration["avg_raw_prob_under"] - 1.0
    calibration["calibration_gap_over"] = calibration["over_rate"]  - calibration["avg_novig_prob_over"]
    calibration["calibration_gap_under"]= calibration["under_rate"] - calibration["avg_novig_prob_under"]

    round_cols = ["over_rate","under_rate","push_rate","avg_raw_prob_over","avg_raw_prob_under",
                  "avg_combined_vig","avg_novig_prob_over","avg_novig_prob_under",
                  "calibration_gap_over","calibration_gap_under"]
    calibration[round_cols] = calibration[round_cols].round(3)

    print(calibration[calibration.n_bets >= 30].to_string(index=False))
    print()

    # Flag large calibration gaps
    large_gap = calibration[(calibration["calibration_gap_over"].abs() > 0.05) & (calibration.n_bets >= 30)]
    if not large_gap.empty:
        print("FLAG: Lines with |calibration_gap_over| > 5pp:")
        print(large_gap[["line","n_bets","calibration_gap_over","calibration_gap_under"]].to_string(index=False))
    print()

    # ── SPOT-CHECK GAME: NYY @ BOS ───────────────────────────────────────────
    print("=== SPOT-CHECK GAME: NYY @ BOS ===")
    nyy_bos = df[
        (df["away_team"] == "New York Yankees") &
        (df["home_team"] == "Boston Red Sox")
    ]
    print(f"NYY @ BOS games: {nyy_bos.event_id.nunique()} | rows: {len(nyy_bos)}")
    for event_id, grp in nyy_bos.groupby(["game_date", "event_id"]):
        g = grp.iloc[0]
        print(f"\n  {g['game_date']} | Total: {g['total_runs']} runs | Line range: {grp['line'].min():.1f}–{grp['line'].max():.1f}")
        print(f"  Books: {grp['bookmaker'].nunique()} | is_over: {bool(g['is_over'])} | is_under: {bool(g['is_under'])} | is_push: {bool(g['is_push'])}")

    print()
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(LOCAL_OUT, index=False)
    print(f"Calibration table saved to {LOCAL_OUT}")


if __name__ == "__main__":
    main()
