"""
Step 6 — In-Sample (IS) grid search. Same 6 dimensions as Step 5 OOF grid.

IS vs OOF difference:
  - OOF: logistic calibration fit on training folds, predicted on val folds (Step 5)
  - IS:  logistic calibration fit on ALL data, predicted on ALL data (this script)

Ridge yhat == yhat_oof at this dataset size (they converge). IS vs OOF gap comes
entirely from the logistic calibration overfitting when fit on all data.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT   = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "src/mlb_batter_strikeouts_modeling/config.yaml"
OOF_PATH    = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_oof_regression.parquet"
OUT_PATH    = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_step6_is_grid.csv"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def is_logistic_p_over(yhat: np.ndarray, lines: np.ndarray,
                        y: np.ndarray) -> np.ndarray:
    """Fit logistic on ALL data, predict on ALL data (IS)."""
    X = np.column_stack([yhat, lines])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_s, y)
    return lr.predict_proba(X_s)[:, 1]


def is_logistic_with_shrinkage(yhat: np.ndarray, lines: np.ndarray,
                                 y: np.ndarray, shrinkage: float) -> np.ndarray:
    """Apply global-mean shrinkage then fit IS logistic on all data."""
    global_mean = float(yhat.mean())
    yhat_s = (1 - shrinkage) * yhat + shrinkage * global_mean
    return is_logistic_p_over(yhat_s, lines, y)


def compute_roi(sub: pd.DataFrame, direction: str) -> dict:
    if len(sub) == 0:
        return dict(n_bets=0, win_rate=0.0, units_won=0.0, roi=0.0, max_drawdown=0.0, avg_odds=0.0)

    if direction == "under":
        hit  = (sub["over_flag"] == 0).values.astype(int)
        odds = sub["under_price"].values
    elif direction == "over":
        hit  = (sub["over_flag"] == 1).values.astype(int)
        odds = sub["over_price"].values
    else:
        under_better = sub["_edge_under"].values >= sub["_edge_over"].values
        hit  = np.where(under_better,
                        (sub["over_flag"] == 0).values.astype(int),
                        (sub["over_flag"] == 1).values.astype(int))
        odds = np.where(under_better, sub["under_price"].values, sub["over_price"].values)

    profit = np.where(hit == 1, odds - 1.0, -1.0)
    cumul  = np.cumsum(profit)
    peak   = np.maximum.accumulate(cumul)
    max_dd = float((peak - cumul).max())

    return dict(
        n_bets       = len(sub),
        win_rate     = round(float(hit.mean()), 4),
        units_won    = round(float(profit.sum()), 3),
        roi          = round(float(profit.sum()) / len(sub) * 100, 3),
        max_drawdown = round(max_dd, 2),
        avg_odds     = round(float(odds.mean()), 3),
    )


def main():
    cfg = load_config()
    gs  = cfg["grid_search"]

    print("Loading parquet...")
    df = pd.read_parquet(OOF_PATH)
    dv = df[df["yhat_oof"].notna() & df["over_flag"].notna()].copy()
    dv = dv.sort_values("game_date").reset_index(drop=True)
    total_rows = len(dv)
    print(f"  {total_rows:,} valid rows")

    yhat_arr = dv["yhat_oof"].values
    lines    = dv["offered_line"].values
    y        = dv["over_flag"].values

    consensus_yhat = (dv.groupby(["player_key", "game_date"])["offered_line"]
                        .transform("mean").values)

    dv["under_odds_sign"] = np.where(dv["under_price"] > 2.0, "plus",
                            np.where(dv["under_price"] < 2.0, "minus", "even"))
    dv["over_odds_sign"]  = np.where(dv["over_price"]  > 2.0, "plus",
                            np.where(dv["over_price"]  < 2.0, "minus", "even"))

    # Pre-compute IS p_over for each (method, shrinkage)
    p_cache: dict[tuple, np.ndarray] = {}
    shrinkage_vals = gs["shrinkage"]
    pred_methods   = gs["prediction_method"]

    print("Pre-computing IS logistic calibration...")
    for pm in pred_methods:
        for s in shrinkage_vals:
            if pm == "consensus_line" and s != 0:
                continue
            key = (pm, s)
            yhat_in = yhat_arr if pm == "model" else consensus_yhat
            print(f"  ({pm}, shrink={s})...", end="", flush=True)
            if s == 0:
                p_over = is_logistic_p_over(yhat_in, lines, y)
            else:
                p_over = is_logistic_with_shrinkage(yhat_in, lines, y, s)
            p_cache[key] = p_over
            print(" done")

    # Grid loop
    results = []
    combos = 0
    for pm in pred_methods:
        for s in shrinkage_vals:
            if pm == "consensus_line" and s != 0:
                continue
            key = (pm, s)
            p_over  = p_cache[key]
            p_under = 1.0 - p_over

            edge_under = p_under - dv["raw_implied_prob_under"].values
            edge_over  = p_over  - dv["raw_implied_prob_over"].values

            dv["_p_over"]     = p_over
            dv["_p_under"]    = p_under
            dv["_edge_under"] = edge_under
            dv["_edge_over"]  = edge_over

            for direction in gs["direction"]:
                if direction == "under":
                    edge_col      = "_edge_under"
                    odds_sign_col = "under_odds_sign"
                elif direction == "over":
                    edge_col      = "_edge_over"
                    odds_sign_col = "over_odds_sign"
                else:
                    dv["_edge_best"] = np.maximum(
                        dv["_edge_under"].fillna(-99),
                        dv["_edge_over"].fillna(-99)
                    )
                    edge_col      = "_edge_best"
                    odds_sign_col = None

                for odds_bucket in gs["odds_bucket"]:
                    for lf in gs["line_filter"]:
                        for et in gs["edge_threshold"]:
                            mask = dv[edge_col].values >= et

                            if lf == "0.5_only":
                                mask &= dv["offered_line"].values == 0.5
                            elif lf == "1.5_only":
                                mask &= dv["offered_line"].values == 1.5

                            if odds_bucket != "all":
                                target = "plus" if odds_bucket == "plus_odds" else "minus"
                                if odds_sign_col:
                                    mask &= (dv[odds_sign_col].values == target)
                                else:
                                    under_better = (dv["_edge_under"].values >= dv["_edge_over"].values)
                                    bet_sign = np.where(under_better, dv["under_odds_sign"].values,
                                                        dv["over_odds_sign"].values)
                                    mask &= (bet_sign == target)

                            sub = dv[mask]
                            m = compute_roi(sub, direction)

                            results.append({
                                "prediction_method": pm,
                                "shrinkage":         s,
                                "direction":         direction,
                                "odds_bucket":       odds_bucket,
                                "line_filter":       lf,
                                "edge_threshold":    et,
                                "n_bets":            m["n_bets"],
                                "pct_of_universe":   round(m["n_bets"] / total_rows * 100, 3),
                                "win_rate":          m["win_rate"],
                                "units_won":         m["units_won"],
                                "roi":               m["roi"],
                                "max_drawdown":      m["max_drawdown"],
                                "avg_odds":          m["avg_odds"],
                            })
                            combos += 1

    grid = (pd.DataFrame(results)
              .sort_values("units_won", ascending=False)
              .reset_index(drop=True))
    grid.to_csv(OUT_PATH, index=False)
    print(f"\nIS Grid: {combos:,} combos → {len(grid):,} rows saved to {OUT_PATH}")

    print(f"\n=== Top 20 IS by units_won (n >= 30) ===")
    top = grid[grid["n_bets"] >= 30].head(20)
    print(top[["prediction_method","shrinkage","direction","odds_bucket","line_filter",
               "edge_threshold","n_bets","units_won","roi","win_rate","max_drawdown"]
             ].to_string(index=False))

    print(f"\n=== IS vs OOF: production config (model, shrink=0, under, all, all, edge=0.10) ===")
    prod = grid[(grid["prediction_method"]=="model") & (grid["shrinkage"]==0) &
                (grid["direction"]=="under") & (grid["odds_bucket"]=="all") &
                (grid["line_filter"]=="all") & (grid["edge_threshold"]==0.10)]
    print(prod[["n_bets","units_won","roi","win_rate","max_drawdown"]].to_string(index=False))
    print(f"  OOF reference: n=1,713, +95.82u, +5.594% ROI, max_dd=59.65")

    print(f"\n=== IS shrinkage sweep — model, under, all, all, edge=0.10 ===")
    filt = ((grid["prediction_method"]=="model") & (grid["direction"]=="under") &
            (grid["odds_bucket"]=="all") & (grid["line_filter"]=="all") &
            (grid["edge_threshold"]==0.10))
    print(grid[filt].sort_values("shrinkage")
          [["shrinkage","n_bets","units_won","roi","win_rate","max_drawdown"]]
          .to_string(index=False))

    print(f"\n=== IS odds bucket sweep — model, shrink=0, under, all lines, edge=0.10 ===")
    filt2 = ((grid["prediction_method"]=="model") & (grid["shrinkage"]==0) &
             (grid["direction"]=="under") & (grid["line_filter"]=="all") &
             (grid["edge_threshold"]==0.10))
    print(grid[filt2][["odds_bucket","n_bets","units_won","roi","win_rate","max_drawdown","avg_odds"]]
          .to_string(index=False))

    # ── Minus_odds deep-dive (per user request) ───────────────────────────────
    print(f"\n=== MINUS_ODDS deep-dive — model, shrink=0, under, all edge thresholds ===")
    filt3 = ((grid["prediction_method"]=="model") & (grid["shrinkage"]==0) &
             (grid["direction"]=="under") & (grid["odds_bucket"]=="minus_odds"))
    print(grid[filt3].sort_values("edge_threshold")
          [["line_filter","edge_threshold","n_bets","units_won","roi","win_rate","max_drawdown","avg_odds"]]
          .to_string(index=False))

    susp = grid[(grid["roi"] > 50) & (grid["n_bets"] >= 100)]
    if len(susp):
        print(f"\n⚠ {len(susp)} IS rows with ROI > 50% and n >= 100 (overfit flag):")
        print(susp[["prediction_method","shrinkage","direction","odds_bucket",
                     "line_filter","edge_threshold","n_bets","roi"]].head(10).to_string(index=False))
    else:
        print("\n✅ No IS rows with ROI > 50% and n >= 100")


if __name__ == "__main__":
    main()
