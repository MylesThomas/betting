"""
NBA Player Points — Step 6: In-Sample Grid Search (Regression)
===============================================================
Same grid as Step 5 but using full-data OLS model predictions on the same
data used for training. IS ROI is inflated by construction.

Key check: IS/OOS ratio < 5x means OOS signal is genuine, not memorization.

Outputs:
  ~/Downloads/tmp/points_eda/step6_grid_is.csv
  ~/Downloads/tmp/points_eda/step6_is_predictions.parquet
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR    = Path.home() / "Downloads/tmp/points_eda"
MODELS_DIR = REPO_ROOT / "models"

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
    if p >= 0.5:
        return -(p / (1 - p) * 100)
    return (1 - p) / p * 100


def run_grid_search(df: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)
    yhat_arr = df["yhat"].values
    line_arr = df["offered_line"].values
    rows = []

    for shrink in SHRINKAGES:
        mean_adj = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)
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
            else:
                under_q = edge_under >= min_edge
                over_q  = edge_over  >= min_edge
                bet_mask = under_q | over_q
                sides = np.where(
                    under_q & (~over_q | (edge_under >= edge_over)), "under",
                    np.where(over_q, "over", None)
                )

            if odds_bucket == "dog_only":
                bet_mask = bet_mask & (p_market_under < 0.50)
            elif odds_bucket == "fav_only":
                bet_mask = bet_mask & (p_market_under >= 0.50)

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

            pnls     = []
            dec_odds = []
            is_push  = []
            for i in idx:
                side = sides[i]
                if side is None:
                    continue
                p_mkt = float(p_market_under[i] if side == "under" else p_market_over[i])
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
                "clf_threshold": "n/a",
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
    print("Loading spine + full OLS model...", flush=True)
    spine = pd.read_parquet(OUT_DIR / "points_spine.parquet")
    settled = spine[spine["pts_actual"].notna()].copy()

    meta_path = MODELS_DIR / "nba_points_meta.json"
    meta = json.loads(meta_path.read_text())
    features = meta["features"]
    model_type = meta["model_type"]

    print(f"  Model type: {model_type}  Features: {features}")

    if model_type == "ols":
        model = joblib.load(MODELS_DIR / "nba_points_model_ols.joblib")
    else:
        model = joblib.load(MODELS_DIR / "nba_points_model_xgb.joblib")

    residuals = np.load(MODELS_DIR / "nba_points_residuals.npy")
    print(f"  Residuals: {len(residuals):,}  σ={residuals.std():.4f}")

    # Score all settled rows with full-data model (in-sample)
    valid = settled.dropna(subset=features).copy()
    valid["yhat"]          = model.predict(valid[features])
    valid["p_market_over"] = valid["novig_prob_over"]
    valid["p_market_under"]= 1.0 - valid["novig_prob_over"]
    valid["is_under"]      = (valid["is_over"] == 0).astype(float)

    print(f"  Settled rows scored: {len(valid):,}")
    print(f"  yhat range: [{valid['yhat'].min():.2f}, {valid['yhat'].max():.2f}]")

    valid.to_parquet(OUT_DIR / "step6_is_predictions.parquet", index=False)
    print(f"  IS predictions saved")

    print("\nRunning IS grid search...", flush=True)
    results = run_grid_search(valid, residuals)
    results.to_csv(OUT_DIR / "step6_grid_is.csv", index=False)
    print(f"Saved: {OUT_DIR}/step6_grid_is.csv  ({len(results):,} strategies)")

    print(f"\n── Top 20 IS strategies by units_won ──")
    print(results.head(20)[
        ["shrinkage", "min_edge", "direction", "odds_bucket", "line_bucket",
         "n_bets", "win_rate", "units_won", "roi", "max_drawdown"]
    ].to_string(index=False))

    # IS vs OOS comparison for candidate strategies
    oos = pd.read_csv(OUT_DIR / "step5_grid_oos.csv")

    candidates = [
        {"shrinkage": 0.50, "min_edge": 0.03, "direction": "under_only", "odds_bucket": "fav_only",  "line_bucket": "high_25plus"},
        {"shrinkage": 0.00, "min_edge": 0.05, "direction": "under_only", "odds_bucket": "all",       "line_bucket": "high_25plus"},
        {"shrinkage": 0.25, "min_edge": 0.05, "direction": "under_only", "odds_bucket": "fav_only",  "line_bucket": "all"},
    ]

    print(f"\n── IS vs OOS comparison for candidate strategies ──")
    for c in candidates:
        om = oos.copy()
        im = results.copy()
        for k, v in c.items():
            om = om[om[k] == v]
            im = im[im[k] == v]

        if len(om) > 0 and len(im) > 0:
            o = om.iloc[0]
            i = im.iloc[0]
            ratio = i["roi"] / o["roi"] if o["roi"] != 0 else float("inf")
            tag = f'shrink={c["shrinkage"]:.2f}|edge={c["min_edge"]:.2f}|{c["direction"][:5]}|{c["odds_bucket"][:3]}|{c["line_bucket"][:9]}'
            print(f"  {tag}")
            print(f"    OOS: n={o['n_bets']:,}, wr={o['win_rate']:.4f}, units={o['units_won']:.2f}, roi={o['roi']:.4f}, mdd={o['max_drawdown']:.2f}")
            print(f"    IS:  n={i['n_bets']:,}, wr={i['win_rate']:.4f}, units={i['units_won']:.2f}, roi={i['roi']:.4f}, mdd={i['max_drawdown']:.2f}")
            print(f"    IS/OOS ratio: {ratio:.2f}x {'PASS' if ratio < 5.0 else 'FLAG - OVERFIT'}")
        else:
            print(f"  No match for {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
