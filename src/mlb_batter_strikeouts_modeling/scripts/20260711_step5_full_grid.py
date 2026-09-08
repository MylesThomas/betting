"""
Step 5 — Full OOS grid search. All dims loaded from config.yaml.

Grid dimensions:
  edge_threshold, direction, odds_bucket, shrinkage,
  prediction_method, line_filter

Shrinkage: yhat_shrunk = (1-s) * yhat_oof + s * fold_train_mean
  - Applied BEFORE logistic calibration, fold-aware (no leakage).
  - For consensus_line: shrinkage is meaningless — skipped (fixed at 0).

Prediction methods:
  - "model"          : Ridge OOF yhat (shrunken) → OOF logistic → P(over)
  - "consensus_line" : avg offered_line as yhat → OOF logistic → P(over)

Odds bucket (based on the bet side's decimal price):
  - "all"        : no filter
  - "plus_odds"  : decimal price > 2.0
  - "minus_odds" : decimal price < 2.0

Edge = p_model_side - raw_implied_prob_side (vig-inclusive, per-book).
Sorted by units_won descending.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT   = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "src/mlb_batter_strikeouts_modeling/config.yaml"
OOF_PATH    = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_oof_regression.parquet"
OUT_PATH    = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_step5_full_grid.csv"

N_FOLDS = 5


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def oof_logistic_p_over(yhat: np.ndarray, lines: np.ndarray,
                         y: np.ndarray, sort_idx: np.ndarray) -> np.ndarray:
    """
    Compute OOF P(over) via logistic regression on (yhat, line).
    sort_idx: row indices in temporal order (already sorted by game_date).
    Returns p_over in the SAME ORDER as the input arrays.
    """
    n = len(yhat)
    p_over = np.full(n, np.nan)
    fold_size = n // N_FOLDS

    for fold in range(N_FOLDS):
        val_start = fold_size * fold
        val_end   = fold_size * (fold + 1) if fold < N_FOLDS - 1 else n
        tr_idx    = np.concatenate([sort_idx[:val_start], sort_idx[val_end:]])
        vl_idx    = sort_idx[val_start:val_end]

        if len(tr_idx) < 50 or len(vl_idx) < 10:
            continue

        X_tr = np.column_stack([yhat[tr_idx], lines[tr_idx]])
        y_tr = y[tr_idx]
        X_vl = np.column_stack([yhat[vl_idx], lines[vl_idx]])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_vl = scaler.transform(X_vl)

        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(X_tr, y_tr)
        p_over[vl_idx] = lr.predict_proba(X_vl)[:, 1]

    return p_over


def oof_logistic_with_shrinkage(yhat_oof: np.ndarray, lines: np.ndarray,
                                  y: np.ndarray, sort_idx: np.ndarray,
                                  shrinkage: float) -> np.ndarray:
    """
    Apply fold-aware shrinkage then compute OOF logistic P(over).
    Fold train mean is used for shrinkage target — no leakage.
    """
    n = len(yhat_oof)
    yhat_s = yhat_oof.copy()
    fold_size = n // N_FOLDS

    # Step 1: build shrunken yhat array using training-fold means
    for fold in range(N_FOLDS):
        val_start = fold_size * fold
        val_end   = fold_size * (fold + 1) if fold < N_FOLDS - 1 else n
        tr_idx    = np.concatenate([sort_idx[:val_start], sort_idx[val_end:]])
        vl_idx    = sort_idx[val_start:val_end]

        fold_mean = float(yhat_oof[tr_idx].mean())
        yhat_s[vl_idx] = (1 - shrinkage) * yhat_oof[vl_idx] + shrinkage * fold_mean

    # Step 2: OOF logistic on shrunken yhat
    return oof_logistic_p_over(yhat_s, lines, y, sort_idx)


def compute_roi(sub: pd.DataFrame, direction: str) -> dict:
    if len(sub) == 0:
        return dict(n_bets=0, win_rate=0.0, push_rate=0.0, units_won=0.0,
                    roi=0.0, max_drawdown=0.0, avg_odds=0.0)

    if direction == "under":
        hit  = (sub["over_flag"] == 0).values.astype(int)
        odds = sub["under_price"].values
    elif direction == "over":
        hit  = (sub["over_flag"] == 1).values.astype(int)
        odds = sub["over_price"].values
    else:  # both — best edge side per row
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
        push_rate    = 0.0,
        units_won    = round(float(profit.sum()), 3),
        roi          = round(float(profit.sum()) / len(sub) * 100, 3),
        max_drawdown = round(max_dd, 2),
        avg_odds     = round(float(odds.mean()), 3),
    )


def main():
    cfg = load_config()
    gs  = cfg["grid_search"]

    print("Loading OOF parquet...")
    df = pd.read_parquet(OOF_PATH)
    dv = df[df["yhat_oof"].notna() & df["over_flag"].notna()].copy()

    # Sort temporally — sort_idx maps temporal position → row position in dv
    dv = dv.sort_values("game_date").reset_index(drop=True)
    sort_idx = np.arange(len(dv))  # already sorted
    total_rows = len(dv)
    print(f"  {total_rows:,} valid OOF rows (sorted by game_date)")

    yhat_oof = dv["yhat_oof"].values
    lines    = dv["offered_line"].values
    y        = dv["over_flag"].values

    # ── Consensus line yhat (book-invariant: avg line per player-game) ────────
    consensus_yhat = (dv.groupby(["player_key", "game_date"])["offered_line"]
                        .transform("mean").values)

    # ── Odds sign ─────────────────────────────────────────────────────────────
    dv["under_odds_sign"] = np.where(dv["under_price"] > 2.0, "plus",
                            np.where(dv["under_price"] < 2.0, "minus", "even"))
    dv["over_odds_sign"]  = np.where(dv["over_price"]  > 2.0, "plus",
                            np.where(dv["over_price"]  < 2.0, "minus", "even"))

    # ── Pre-compute OOF p_over for every (pred_method, shrinkage) combo ───────
    # This is the expensive step — cache results so grid loop is fast.
    p_cache: dict[tuple, np.ndarray] = {}

    shrinkage_vals  = gs["shrinkage"]
    pred_methods    = gs["prediction_method"]

    print("Pre-computing OOF logistic calibration for each (method, shrinkage)...")
    for pm in pred_methods:
        for s in shrinkage_vals:
            if pm == "consensus_line" and s != 0:
                continue  # shrinkage meaningless for consensus_line
            key = (pm, s)
            if pm == "model":
                yhat_in = yhat_oof
            else:
                yhat_in = consensus_yhat

            print(f"  ({pm}, shrink={s})...", end="", flush=True)
            if s == 0:
                p_over = oof_logistic_p_over(yhat_in, lines, y, sort_idx)
            else:
                p_over = oof_logistic_with_shrinkage(yhat_in, lines, y, sort_idx, s)
            p_cache[key] = p_over
            valid = ~np.isnan(p_over)
            print(f" {valid.sum():,} valid predictions")

    # ── Grid loop ─────────────────────────────────────────────────────────────
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

            # Mask out rows with no prediction (first OOF fold)
            has_pred = ~np.isnan(p_over)

            for direction in gs["direction"]:
                if direction == "under":
                    edge_col      = "_edge_under"
                    odds_sign_col = "under_odds_sign"
                elif direction == "over":
                    edge_col      = "_edge_over"
                    odds_sign_col = "over_odds_sign"
                else:
                    # both: max edge side
                    dv["_edge_best"] = np.maximum(
                        dv["_edge_under"].fillna(-99),
                        dv["_edge_over"].fillna(-99)
                    )
                    edge_col      = "_edge_best"
                    odds_sign_col = None

                for odds_bucket in gs["odds_bucket"]:
                    for lf in gs["line_filter"]:
                        for et in gs["edge_threshold"]:
                            mask = has_pred & (dv[edge_col].values >= et)

                            if lf == "0.5_only":
                                mask &= dv["offered_line"].values == 0.5
                            elif lf == "1.5_only":
                                mask &= dv["offered_line"].values == 1.5

                            if odds_bucket != "all":
                                target = "plus" if odds_bucket == "plus_odds" else "minus"
                                if odds_sign_col:
                                    mask &= (dv[odds_sign_col].values == target)
                                else:
                                    # "both" direction: check the better-edge side's sign
                                    under_better = (dv["_edge_under"].values >=
                                                    dv["_edge_over"].values)
                                    bet_sign = np.where(
                                        under_better,
                                        dv["under_odds_sign"].values,
                                        dv["over_odds_sign"].values
                                    )
                                    mask &= (bet_sign == target)

                            sub = dv[mask]
                            m = compute_roi(sub, direction)
                            pct_u = round(m["n_bets"] / total_rows * 100, 3)

                            results.append({
                                "prediction_method": pm,
                                "shrinkage":         s,
                                "direction":         direction,
                                "odds_bucket":       odds_bucket,
                                "line_filter":       lf,
                                "edge_threshold":    et,
                                "n_bets":            m["n_bets"],
                                "pct_of_universe":   pct_u,
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
    print(f"\nGrid: {combos:,} combos → {len(grid):,} rows saved to {OUT_PATH}")

    # ── Print top 30 ──────────────────────────────────────────────────────────
    print(f"\n=== Top 30 by units_won (n >= 30) ===")
    top = grid[grid["n_bets"] >= 30].head(30)
    print(top[["prediction_method","shrinkage","direction","odds_bucket","line_filter",
               "edge_threshold","n_bets","units_won","roi","win_rate","max_drawdown"]
             ].to_string(index=False))

    # ── Shrinkage sweep at best prior config ──────────────────────────────────
    print(f"\n=== Shrinkage sweep — model, under, all, all, edge=0.10 ===")
    filt = ((grid["prediction_method"]=="model") & (grid["direction"]=="under") &
            (grid["odds_bucket"]=="all") & (grid["line_filter"]=="all") &
            (grid["edge_threshold"]==0.10))
    print(grid[filt].sort_values("shrinkage")
          [["shrinkage","n_bets","units_won","roi","win_rate","max_drawdown"]]
          .to_string(index=False))

    # ── Direction sweep ───────────────────────────────────────────────────────
    print(f"\n=== Direction sweep — model, shrink=0, all, all, edge=0.10 ===")
    filt2 = ((grid["prediction_method"]=="model") & (grid["shrinkage"]==0) &
             (grid["odds_bucket"]=="all") & (grid["line_filter"]=="all") &
             (grid["edge_threshold"]==0.10))
    print(grid[filt2].sort_values("direction")
          [["direction","n_bets","units_won","roi","win_rate","max_drawdown"]]
          .to_string(index=False))

    # ── Model vs consensus_line ───────────────────────────────────────────────
    print(f"\n=== model vs consensus_line — under, shrink=0, all, all, edge=0.10 ===")
    filt3 = ((grid["direction"]=="under") & (grid["shrinkage"]==0) &
             (grid["odds_bucket"]=="all") & (grid["line_filter"]=="all") &
             (grid["edge_threshold"]==0.10))
    print(grid[filt3][["prediction_method","n_bets","units_won","roi","win_rate","max_drawdown"]]
          .to_string(index=False))

    # ── Flags ─────────────────────────────────────────────────────────────────
    susp = grid[(grid["roi"] > 25) & (grid["n_bets"] >= 100)]
    if len(susp):
        print(f"\n⚠ {len(susp)} rows with ROI > 25% and n >= 100:")
        print(susp[["prediction_method","shrinkage","direction","odds_bucket",
                     "line_filter","edge_threshold","n_bets","roi"]].head(10).to_string(index=False))
    else:
        print("\n✅ No rows with ROI > 25% and n >= 100")


if __name__ == "__main__":
    main()
