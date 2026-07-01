"""
Step 5 — Out-of-Sample Grid Search (OOF predictions only)

For regression markets: only edge threshold applies (no classification threshold).
Edge = p_model - p_market.
  Over bets:  place when  edge >= threshold  (model favors over vs market)
  Under bets: place when -edge >= threshold  (model favors under vs market)

Flat betting: 1 unit per bet.
  Win:  profit = decimal_odds - 1
  Loss: profit = -1
  Push: 0 (impossible — all lines are .5)

Grid dimensions:
  edge_threshold : [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
  direction      : [over, under, both]
  line_filter    : [all, low (<6.5 — QBs/backups), high (>=6.5 — featured RBs)]
  position_filter: [all, RB, QB]

Output:
  ~/Downloads/tmp/rush_attempts/step5_grid.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

BETS_PATH = Path.home() / "Downloads" / "tmp" / "rush_attempts" / "step4_bets.parquet"
OUT_DIR   = Path.home() / "Downloads" / "tmp" / "rush_attempts"

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
DIRECTIONS      = ["over", "under", "both"]
LINE_FILTERS    = ["all", "low", "high"]
POS_FILTERS     = ["all", "RB", "QB"]


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Largest peak-to-trough loss in units (chronological order)."""
    cum = np.cumsum(pnl_series)
    peak = cum[0]
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def evaluate_strategy(df: pd.DataFrame, direction: str,
                      edge_threshold: float) -> dict:
    """
    Given a filtered DataFrame and strategy params, return performance metrics.
    df must already be filtered to the desired line/position segment.
    """
    edge = df["edge"].values
    is_over = df["is_over"].values
    over_price = df["book_over_price"].values
    under_price = df["book_under_price"].values

    # Select bets based on direction and threshold
    if direction == "over":
        mask = edge >= edge_threshold
        correct = is_over[mask] == 1
        prices  = over_price[mask]
    elif direction == "under":
        mask = (-edge) >= edge_threshold
        correct = is_over[mask] == 0
        prices  = under_price[mask]
    else:  # both
        over_mask  = edge >= edge_threshold
        under_mask = (-edge) >= edge_threshold
        mask = over_mask | under_mask
        # For "both" direction: each row has its own direction
        bet_over = np.where(over_mask, True, False)
        correct = np.where(
            bet_over,
            is_over == 1,
            is_over == 0,
        )[mask]
        prices = np.where(bet_over, over_price, under_price)[mask]

    n_bets = int(mask.sum())
    if n_bets == 0:
        return None

    # Payout per bet
    decimals = np.vectorize(american_to_decimal)(prices)
    profit   = np.where(correct, decimals - 1, -1.0)

    units_won = round(float(profit.sum()), 4)
    win_rate  = round(float(correct.mean()), 4)
    roi       = round(units_won / n_bets, 4)
    max_dd    = max_drawdown(profit)

    # Convert avg decimal odds to avg American odds for display
    avg_dec = float(decimals.mean())
    if avg_dec >= 2.0:
        avg_odds_american = round((avg_dec - 1) * 100, 1)
    else:
        avg_odds_american = round(-100 / (avg_dec - 1), 1)

    return {
        "n_bets":      n_bets,
        "win_rate":    win_rate,
        "push_rate":   0.0,
        "units_won":   units_won,
        "roi":         roi,
        "avg_odds":    avg_odds_american,
        "max_drawdown": max_dd,
    }


def run():
    df = pd.read_parquet(BETS_PATH)
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows | seasons: {sorted(df['season'].unique())}")

    results = []

    for edge_thresh, direction, line_filter, pos_filter in itertools.product(
        EDGE_THRESHOLDS, DIRECTIONS, LINE_FILTERS, POS_FILTERS
    ):
        sub = df.copy()

        # Position filter
        if pos_filter == "RB":
            sub = sub[sub["position"] == "RB"]
        elif pos_filter == "QB":
            sub = sub[sub["position"] == "QB"]

        # Line filter
        if line_filter == "low":
            sub = sub[sub["book_line"] < 6.5]
        elif line_filter == "high":
            sub = sub[sub["book_line"] >= 6.5]

        if len(sub) == 0:
            continue

        metrics = evaluate_strategy(sub, direction, edge_thresh)
        if metrics is None:
            continue

        row = {
            "edge_threshold":  edge_thresh,
            "direction":       direction,
            "line_filter":     line_filter,
            "position_filter": pos_filter,
            **metrics,
        }
        results.append(row)

    grid = pd.DataFrame(results)
    grid = grid.sort_values(["units_won", "n_bets"], ascending=[False, False])

    out_path = OUT_DIR / "step5_grid.csv"
    grid.to_csv(out_path, index=False)
    print(f"Grid search: {len(grid):,} combos evaluated")

    # ── Summary prints ────────────────────────────────────────────────────────
    print("\n=== TOP 20 by units won (≥50 bets) ===")
    top = grid[grid["n_bets"] >= 50].head(20)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(top[["edge_threshold","direction","line_filter","position_filter",
               "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]].to_string(index=False))

    print("\n=== UNDER strategies only (≥50 bets), top 15 by ROI ===")
    under = grid[(grid["direction"].isin(["under","both"])) & (grid["n_bets"] >= 50)]
    under_top = under.sort_values("roi", ascending=False).head(15)
    print(under_top[["edge_threshold","direction","line_filter","position_filter",
                      "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]].to_string(index=False))

    print("\n=== QB-specific strategies (≥30 bets) ===")
    qb = grid[(grid["position_filter"] == "QB") & (grid["n_bets"] >= 30)]
    print(qb.sort_values("roi", ascending=False).head(10)[
        ["edge_threshold","direction","line_filter","position_filter",
         "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]
    ].to_string(index=False))

    print("\n=== Strategies with ROI > 5% (flag if >25% with >100 bets — possible leakage) ===")
    suspect = grid[(grid["roi"] > 0.05) & (grid["n_bets"] >= 50)]
    print(suspect[["edge_threshold","direction","line_filter","position_filter",
                   "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]].head(20).to_string(index=False))

    print(f"\nSaved grid to {out_path}")
    print("=== Step 5 complete ===")


if __name__ == "__main__":
    run()
