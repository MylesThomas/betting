"""
MLB Pitcher Strikeouts — Step 5: Out-of-Sample Grid Search
===========================================================
Grid search over shrinkage, min_edge, direction, odds bucket, line bucket.
Uses OOF bootstrap P(over) from Step 4 — truly out-of-sample.

Shrinkage: mean_adj = line + (1-shrink) * (yhat - line)
  shrink=0.0 → raw yhat (full model signal)
  shrink=0.5 → blend model and line
  shrink=1.0 → no signal (would never bet)

Units: flat 1-unit bets at American odds derived from novig p_market.
Max drawdown: largest peak-to-trough loss in chronological bet sequence.

Outputs:
  ~/Downloads/tmp/mlb_strikeouts/step5_grid_oos.csv

Usage:
  python src/mlb_strikeouts_modeling/scripts/v5_grid_search_oos.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path.home() / "Downloads/tmp/mlb_strikeouts"

MIN_EDGES    = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
SHRINKAGES   = [0.0, 0.25, 0.50, 0.75]
DIRECTIONS   = ["under_only", "over_only", "both"]
ODDS_BUCKETS = ["all", "dog_only", "fav_only"]
LINE_BUCKETS = ["all", "low_le4.5", "mid_5.5_6.5", "high_ge7.5"]

N_BOOT = 10_000
RNG    = np.random.default_rng(42)


def bootstrap_p_over_batch(yhat: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samples = RNG.choice(residuals, size=(len(yhat), N_BOOT), replace=True)
    sims    = yhat[:, None] + samples
    return (sims > line[:, None]).mean(axis=1)


def max_drawdown_units(pnl_series: np.ndarray) -> float:
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum)
    return float((running_max - cum).max())


def p_market_to_american(p: float) -> float:
    if p >= 0.5:
        return -(p / (1 - p) * 100)
    return (1 - p) / p * 100


def compute_unit_pnl(is_over: int, side: str, am_odds: float) -> float:
    bet_hits = (is_over == 1) if side == "over" else (is_over == 0)
    if bet_hits:
        return am_odds / 100.0 if am_odds >= 0 else 100.0 / abs(am_odds)
    return -1.0


def run_grid_search(df: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)
    yhat_arr = df["yhat"].values
    line_arr = df["line"].values
    rows = []

    for shrink in SHRINKAGES:
        mean_adj = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)

        p_model_over  = bootstrap_p_over_batch(mean_adj, line_arr, residuals)
        p_model_under = 1.0 - p_model_over

        p_market_over  = df["p_market_over"].values
        p_market_under = df["p_market_under"].values
        is_over_arr    = df["is_over"].values

        edge_over  = p_model_over  - p_market_over
        edge_under = p_model_under - p_market_under

        for min_edge, direction, odds_bucket, line_bucket in product(
            MIN_EDGES, DIRECTIONS, ODDS_BUCKETS, LINE_BUCKETS
        ):
            if direction == "under_only":
                bet_mask = edge_under >= min_edge
                sides    = np.where(bet_mask, "under", None)
            elif direction == "over_only":
                bet_mask = edge_over >= min_edge
                sides    = np.where(bet_mask, "over", None)
            else:  # both
                under_q  = edge_under >= min_edge
                over_q   = edge_over  >= min_edge
                bet_mask = under_q | over_q
                sides    = np.where(
                    under_q & (~over_q | (edge_under >= edge_over)), "under",
                    np.where(over_q, "over", None),
                )

            if odds_bucket == "dog_only":
                # Betting side is underdog (market prices you below 50%)
                under_dog = (direction in ("under_only", "both")) & (p_market_under < 0.50)
                over_dog  = (direction in ("over_only",  "both")) & (p_market_over  < 0.50)
                bet_mask  = bet_mask & (under_dog | over_dog)
            elif odds_bucket == "fav_only":
                under_fav = (direction in ("under_only", "both")) & (p_market_under >= 0.50)
                over_fav  = (direction in ("over_only",  "both")) & (p_market_over  >= 0.50)
                bet_mask  = bet_mask & (under_fav | over_fav)

            if line_bucket == "low_le4.5":
                bet_mask = bet_mask & (line_arr <= 4.5)
            elif line_bucket == "mid_5.5_6.5":
                bet_mask = bet_mask & (line_arr >= 5.5) & (line_arr <= 6.5)
            elif line_bucket == "high_ge7.5":
                bet_mask = bet_mask & (line_arr >= 7.5)

            idx    = np.where(bet_mask)[0]
            n_bets = len(idx)
            if n_bets < 30:
                continue

            pnls      = []
            dec_odds  = []
            for i in idx:
                side = sides[i]
                if side is None:
                    continue
                p_mkt = float(p_market_over[i]) if side == "over" else float(p_market_under[i])
                dec_odds.append(1.0 / p_mkt)
                am_odds = p_market_to_american(p_mkt)
                pnls.append(compute_unit_pnl(int(is_over_arr[i]), side, am_odds))

            pnls      = np.array(pnls)
            dec_odds  = np.array(dec_odds)
            units     = float(pnls.sum())
            wins      = int((pnls > 0).sum())
            mdd       = max_drawdown_units(pnls)
            n         = len(pnls)

            rows.append({
                "shrinkage":     shrink,
                "min_edge":      min_edge,
                "direction":     direction,
                "odds_bucket":   odds_bucket,
                "line_bucket":   line_bucket,
                "n_bets":        n,
                "win_rate":      round(wins / n, 4),
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
    sigma = residuals.std()
    residuals = np.clip(residuals, -5 * sigma, 5 * sigma)
    # season column may be season_x or season_y from merge in v4
    if "season" not in df.columns:
        df["season"] = df.get("season_y", df.get("season_x"))
    print(f"  Rows: {len(df):,}  Residuals: {len(residuals):,}  σ={sigma:.4f}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # Deduplicate to one row per (player_key, game_date, line).
    # Multiple books post the same line — the model edge is identical across them.
    # Keeping all would inflate n_bets by ~n_books with no new information.
    n_before = len(df)
    df = df.drop_duplicates(subset=["player_key", "game_date", "line"], keep="first")
    print(f"  Deduped to 1 row per (player, game, line): {n_before:,} → {len(df):,} rows")

    print(f"\nRunning OOS grid search...")
    print(f"  Dimensions: {len(SHRINKAGES)} shrink × {len(MIN_EDGES)} edge × "
          f"{len(DIRECTIONS)} dir × {len(ODDS_BUCKETS)} odds × {len(LINE_BUCKETS)} line "
          f"= {len(SHRINKAGES)*len(MIN_EDGES)*len(DIRECTIONS)*len(ODDS_BUCKETS)*len(LINE_BUCKETS):,} combos")

    results = run_grid_search(df, residuals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "step5_grid_oos.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/step5_grid_oos.csv  ({len(results):,} valid strategies ≥30 bets)")

    print(f"\n── Top 25 by units_won ──")
    print(results.head(25)[
        ["shrinkage", "min_edge", "direction", "odds_bucket", "line_bucket",
         "n_bets", "win_rate", "units_won", "roi", "max_drawdown", "drawdown_flag"]
    ].to_string(index=False))

    print(f"\n── Suspicious: ROI > 30% with > 100 bets ──")
    suspicious = results[(results["roi"] > 0.30) & (results["n_bets"] > 100)]
    print(f"  {len(suspicious):,} suspicious strategies — review for leakage")

    print(f"\n── Drawdown-flagged (max_drawdown > units_won) ──")
    print(f"  {results['drawdown_flag'].sum():,} of {len(results):,} strategies flagged")

    print("\nDone.")


if __name__ == "__main__":
    main()
