"""
Parameter sweep for NFL tackles inference strategy.

Runs inference once, then evaluates all combinations of:
  direction      : OVER / UNDER / BOTH
  edge_threshold : minimum |p_hybrid − p_market| to recommend a bet
  line_range     : (LINE_MIN, LINE_MAX) — lines outside this are skipped
  min_books      : minimum number of books with a line (filters thin markets)

Primary metric: EV per unit at -110 odds
  EV = hit_rate × (100/110) − (1 − hit_rate)
  Break-even at -110 = 52.38% hit rate → EV = 0.0

NOTE: all results are IN-SAMPLE (model trained on this data).
Hit rates will be inflated. Use this sweep to:
  1. Find which param combos are directionally best
  2. Understand volume vs accuracy tradeoffs
  3. Set production defaults before OOS validation

Run:
  python src/nfl_tackles_modeling/scripts/param_sweep.py
  python src/nfl_tackles_modeling/scripts/param_sweep.py --sort hit_rate
  python src/nfl_tackles_modeling/scripts/param_sweep.py --min-bets 50
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse inference machinery
from infer import (
    ARTIFACT_DIR,
    DROP_POSITIONS,
    LABELED_PATH,
    N_BOOT,
    TARGET,
    add_derived,
    filter_bets,
    load_artifacts,
    run_inference,
)

warnings.filterwarnings("ignore")

# ── Grid ──────────────────────────────────────────────────────────────────────

DIRECTIONS = ["UNDER", "OVER", "BOTH"]

EDGE_THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20]

LINE_RANGES = [
    (2.5, 9.5),   # full calibrated range
    (4.5, 9.5),   # drop low-line tail (NegBin least accurate there)
    (4.5, 8.5),   # tightest well-calibrated zone
]

MIN_BOOKS_OPTIONS = [1, 3, 5]

# ── Evaluation ────────────────────────────────────────────────────────────────

JUICE = 110   # standard -110 juice; win 100 per 110 wagered
WIN_PAYOUT = 100 / JUICE   # ~0.909 per unit wagered


def ev_at_juice(hit_rate: float) -> float:
    """Expected value per unit wagered at JUICE odds (-110 default)."""
    return hit_rate * WIN_PAYOUT - (1 - hit_rate)


BREAKEVEN_HIT_RATE = JUICE / (JUICE + 100)   # 52.38% at -110


def apply_filter(
    results: pd.DataFrame,
    direction: str,
    edge_threshold: float,
    line_min: float,
    line_max: float,
    min_books: int,
) -> pd.DataFrame:
    mask = (
        (results["offered_line"] >= line_min) &
        (results["offered_line"] <= line_max) &
        (results["edge"].abs() >= edge_threshold) &
        (results["ols_pred"].notna())
    )
    if "n_books" in results.columns:
        mask &= results["n_books"] >= min_books

    if direction == "OVER":
        mask &= results["recommendation"] == "OVER"
    elif direction == "UNDER":
        mask &= results["recommendation"] == "UNDER"
    else:  # BOTH
        mask &= results["recommendation"].isin(["OVER", "UNDER"])

    return results[mask].copy()


def evaluate(bets: pd.DataFrame) -> dict:
    if len(bets) == 0:
        return {"n_bets": 0, "hit_rate": np.nan, "ev_per_unit": np.nan,
                "pct_over": np.nan, "mean_edge_pp": np.nan, "mean_line": np.nan}
    hit_rate     = bets["bet_correct"].mean()
    ev           = ev_at_juice(hit_rate)
    pct_over     = (bets["recommendation"] == "OVER").mean() * 100
    mean_edge_pp = bets["edge"].abs().mean() * 100
    mean_line    = bets["offered_line"].mean()
    return {
        "n_bets":       len(bets),
        "hit_rate":     round(hit_rate * 100, 2),
        "ev_per_unit":  round(ev, 4),
        "pct_over":     round(pct_over, 1),
        "mean_edge_pp": round(mean_edge_pp, 2),
        "mean_line":    round(mean_line, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--sort",         default="ev_per_unit",
                        choices=["ev_per_unit", "hit_rate", "n_bets"])
    parser.add_argument("--min-bets",     type=int, default=20,
                        help="Exclude combos with fewer than this many bets")
    parser.add_argument("--top",          type=int, default=30,
                        help="Print top N rows per direction group")
    args = parser.parse_args()

    # ── Load + infer once ─────────────────────────────────────────────────────
    print(f"\n  Loading artifacts from {args.artifact_dir}...")
    artifacts = load_artifacts(args.artifact_dir)
    meta      = artifacts["meta"]
    print(f"    Trained seasons : {meta['train_seasons']}  |  "
          f"In-sample MAE: {meta['in_sample_mae']}  |  "
          f"NegBin α: {meta['nb_alpha']}")

    print(f"\n  Loading labeled dataset...")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].notna() & ~df["position"].isin(DROP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"    Rows: {len(df):,}  |  Seasons: {sorted(df['season'].unique())}")

    print(f"\n  Running inference [bootstrap: {N_BOOT:,} draws per row]...")
    results = run_inference(df, artifacts)

    # Attach actual outcome
    results["actual_over"] = (results[TARGET] > results["offered_line"]).astype(float)
    results["bet_correct"] = np.where(
        results["recommendation"] == "OVER",  results["actual_over"],
        np.where(
            results["recommendation"] == "UNDER", 1 - results["actual_over"],
            np.nan,
        ),
    )

    scored = results["ols_pred"].notna().sum()
    print(f"    Scored rows: {scored:,}")

    # ── Grid sweep ────────────────────────────────────────────────────────────
    print(f"\n  Sweeping {len(DIRECTIONS)} directions × "
          f"{len(EDGE_THRESHOLDS)} edge thresholds × "
          f"{len(LINE_RANGES)} line ranges × "
          f"{len(MIN_BOOKS_OPTIONS)} min_books "
          f"= {len(DIRECTIONS)*len(EDGE_THRESHOLDS)*len(LINE_RANGES)*len(MIN_BOOKS_OPTIONS)} combos...")

    rows = []
    for direction, edge, (lmin, lmax), min_books in itertools.product(
        DIRECTIONS, EDGE_THRESHOLDS, LINE_RANGES, MIN_BOOKS_OPTIONS
    ):
        bets    = apply_filter(results, direction, edge, lmin, lmax, min_books)
        metrics = evaluate(bets)
        rows.append({
            "direction":    direction,
            "edge":         edge,
            "line_min":     lmin,
            "line_max":     lmax,
            "min_books":    min_books,
            **metrics,
        })

    sweep = pd.DataFrame(rows)

    # ── Display ───────────────────────────────────────────────────────────────
    W = 120
    sort_col = args.sort

    print(f"\n{'='*W}")
    print(f"  PARAMETER SWEEP  (⚠  in-sample — inflated hit rates expected)")
    print(f"  Break-even hit rate at -110 juice: {BREAKEVEN_HIT_RATE*100:.2f}%")
    print(f"  Sorted by {sort_col}  |  Minimum bets: {args.min_bets}")
    print(f"{'='*W}\n")

    display_cols = [
        "direction", "edge", "line_min", "line_max", "min_books",
        "n_bets", "hit_rate", "ev_per_unit", "pct_over", "mean_edge_pp", "mean_line",
    ]

    # Print full table sorted globally
    valid = sweep[sweep["n_bets"] >= args.min_bets].sort_values(
        sort_col, ascending=False
    )
    print(valid[display_cols].head(args.top).to_string(index=False))

    # ── Summary: best combo per direction ─────────────────────────────────────
    print(f"\n{'='*W}")
    print("  BEST COMBO PER DIRECTION  (by ev_per_unit, n_bets ≥ 50)")
    print(f"{'='*W}\n")

    for direction in DIRECTIONS:
        sub = sweep[
            (sweep["direction"] == direction) & (sweep["n_bets"] >= 50)
        ].sort_values("ev_per_unit", ascending=False)
        if sub.empty:
            print(f"  {direction}: no combos with ≥50 bets\n")
            continue
        best = sub.iloc[0]
        print(f"  {direction}:")
        print(f"    edge={best['edge']}  lines={best['line_min']}–{best['line_max']}  "
              f"min_books={best['min_books']}")
        print(f"    n_bets={best['n_bets']:.0f}  hit_rate={best['hit_rate']:.1f}%  "
              f"ev_per_unit={best['ev_per_unit']:+.4f}\n")

    # ── Edge threshold sensitivity (BOTH direction, best line range) ──────────
    print(f"{'='*W}")
    print("  EDGE THRESHOLD SENSITIVITY  (direction=BOTH, line 4.5–8.5, min_books=3)")
    print(f"{'='*W}\n")
    sens = sweep[
        (sweep["direction"] == "BOTH") &
        (sweep["line_min"] == 4.5) &
        (sweep["line_max"] == 8.5) &
        (sweep["min_books"] == 3)
    ].sort_values("edge")
    if not sens.empty:
        print(sens[display_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
