"""
NBA Player Points — Step 5: Out-of-Sample Grid Search (Regression)
====================================================================
Grid search over min_edge, shrinkage, direction, odds bucket, line bucket.
Uses OOF bootstrap P(under/over) from Step 4 — truly out-of-sample.

Shrinkage: mean_adj = offered_line + (1-shrink) * (yhat - offered_line)
  shrink=0.0 → full model signal (raw yhat)
  shrink=0.5 → blend model and line
  shrink=1.0 → no signal (would never bet)
Shrinkage controls overconfidence: at high shrinkage the model must show
a large point difference to generate any edge.

Units: flat 1-unit bets at American odds derived from novig p_market.
Max drawdown: largest peak-to-trough loss in chronological bet sequence.

Outputs:
  ~/Downloads/tmp/points_eda/step5_grid_oos.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path.home() / "Downloads/tmp/points_eda"

MIN_EDGES    = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
SHRINKAGES   = [0.0, 0.25, 0.50, 0.75]
DIRECTIONS   = ["under_only", "over_only", "both"]
ODDS_BUCKETS = ["all", "dog_only", "fav_only"]
LINE_BUCKETS = ["all", "low_14", "mid_15_24", "high_25plus"]

N_BOOT = 10_000
RNG    = np.random.default_rng(42)


def bootstrap_p_under_batch(yhat: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samples = RNG.choice(residuals, size=(len(yhat), N_BOOT), replace=True)
    sims = yhat[:, None] + samples
    return (sims <= line[:, None]).mean(axis=1)


def max_drawdown_units(pnl_series: np.ndarray) -> float:
    cum = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


def compute_unit_pnl(is_under: float, side: str, american_odds: float) -> float:
    bet_hits = (is_under == 1.0) if side == "under" else (is_under == 0.0)
    if bet_hits:
        return american_odds / 100.0 if american_odds >= 0 else 100.0 / abs(american_odds)
    return -1.0


def p_market_to_american(p: float) -> float:
    """Convert novig probability to American odds (for the side we're betting)."""
    if p >= 0.5:
        return -(p / (1 - p) * 100)
    return (1 - p) / p * 100


def run_grid_search(df: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)
    yhat_arr  = df["yhat"].values
    line_arr  = df["offered_line"].values
    rows = []

    for shrink in SHRINKAGES:
        # Apply shrinkage: blend yhat toward offered_line
        mean_adj = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)

        # Bootstrap P(under/over) with shrunk mean
        p_model_under = bootstrap_p_under_batch(mean_adj, line_arr, residuals)
        p_model_over  = 1.0 - p_model_under

        p_market_under = df["p_market_under"].values
        p_market_over  = df["p_market_over"].values

        edge_under = p_model_under - p_market_under
        edge_over  = p_model_over  - p_market_over

        for min_edge, direction, odds_bucket, line_bucket in product(
            MIN_EDGES, DIRECTIONS, ODDS_BUCKETS, LINE_BUCKETS
        ):
            if direction == "under_only":
                bet_mask = edge_under >= min_edge
                sides = np.where(bet_mask, "under", None)
            elif direction == "over_only":
                bet_mask = edge_over >= min_edge
                sides = np.where(bet_mask, "over", None)
            else:  # both: pick the stronger edge side
                under_q = edge_under >= min_edge
                over_q  = edge_over  >= min_edge
                bet_mask = under_q | over_q
                sides = np.where(
                    under_q & (~over_q | (edge_under >= edge_over)), "under",
                    np.where(over_q, "over", None)
                )

            # Odds bucket filter (based on p_market_under — dog = market favors over = p_under < 0.50)
            if odds_bucket == "dog_only":
                # We're betting under; dog = market favors OVER → p_market_under < 0.50
                bet_mask = bet_mask & (p_market_under < 0.50)
            elif odds_bucket == "fav_only":
                bet_mask = bet_mask & (p_market_under >= 0.50)

            # Line bucket filter
            if line_bucket == "low_14":
                bet_mask = bet_mask & (line_arr <= 14.5)
            elif line_bucket == "mid_15_24":
                bet_mask = bet_mask & (line_arr > 14.5) & (line_arr <= 24.5)
            elif line_bucket == "high_25plus":
                bet_mask = bet_mask & (line_arr > 24.5)

            idx = np.where(bet_mask)[0]
            n_bets = len(idx)
            if n_bets < 30:
                continue

            pnls      = []
            dec_odds  = []   # decimal odds per bet (for avg_odds)
            is_push   = []   # push flags
            for i in idx:
                side = sides[i]
                if side is None:
                    continue
                if side == "under":
                    p_mkt = float(p_market_under[i])
                    actual_under = float(df["is_under"].iloc[i])
                    is_push.append(actual_under == 0.5)  # rare; flag if line is integer
                else:
                    p_mkt = float(p_market_over[i])
                    actual_under = float(df["is_under"].iloc[i])
                    is_push.append(actual_under == 0.5)
                dec_odds.append(1.0 / p_mkt)
                am_odds = p_market_to_american(p_mkt)
                pnl = compute_unit_pnl(actual_under, side, am_odds)
                pnls.append(pnl)

            pnls     = np.array(pnls)
            dec_odds = np.array(dec_odds)
            wins     = (pnls > 0).sum()
            pushes   = sum(is_push)
            units    = float(pnls.sum())
            mdd      = max_drawdown_units(pnls)
            n        = len(pnls)

            rows.append({
                "shrinkage":     shrink,
                "min_edge":      min_edge,
                "clf_threshold": "n/a",   # regression — no classification threshold
                "direction":     direction,
                "odds_bucket":   odds_bucket,
                "line_bucket":   line_bucket,
                "n_bets":        n,
                "win_rate":      round(wins / n, 4),
                "push_rate":     round(pushes / n, 4),
                "units_won":     round(units, 2),
                "roi":           round(units / n, 4),
                "avg_odds":      round(float(dec_odds.mean()), 4),
                "max_drawdown":  round(mdd, 2),
                "drawdown_flag": mdd > units,
            })

    return pd.DataFrame(rows).sort_values("units_won", ascending=False)


def main():
    print("Loading edge dataset...", flush=True)
    df = pd.read_parquet(OUT_DIR / "step4_edge.parquet")
    residuals = np.load(OUT_DIR / "step3_oof_residuals.npy")
    print(f"  Rows: {len(df):,}  Residuals: {len(residuals):,}")

    print(f"\nRunning OOS grid search...")
    print(f"  Dimensions: {len(SHRINKAGES)} shrinkages × {len(MIN_EDGES)} edges × "
          f"{len(DIRECTIONS)} dirs × {len(ODDS_BUCKETS)} odds × {len(LINE_BUCKETS)} lines")

    results = run_grid_search(df, residuals)

    results.to_csv(OUT_DIR / "step5_grid_oos.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/step5_grid_oos.csv  ({len(results):,} valid strategies)")

    print(f"\n── Top 25 by units_won ──")
    print(results.head(25)[
        ["shrinkage", "min_edge", "direction", "odds_bucket", "line_bucket",
         "n_bets", "win_rate", "units_won", "roi", "max_drawdown", "drawdown_flag"]
    ].to_string(index=False))

    print(f"\n── Suspicious: ROI > 25% with > 100 bets ──")
    suspicious = results[(results["roi"] > 0.25) & (results["n_bets"] > 100)]
    print(f"  {len(suspicious):,} suspicious strategies (flag for leakage review)")

    print(f"\n── Drawdown-flagged strategies (max_drawdown > units_won) ──")
    flagged = results[results["drawdown_flag"]]
    print(f"  {len(flagged):,} of {len(results):,} strategies flagged")

    print("\nDone.")


if __name__ == "__main__":
    main()
