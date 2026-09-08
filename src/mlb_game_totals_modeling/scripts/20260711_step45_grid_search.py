"""
Steps 4+5 — Probability Conversion + Grid Search Backtest for MLB Game Totals.

Two prediction methods:
  1. model:          Logistic regression (top-5 features) predicting P(hit_over) and P(hit_under)
  2. consensus_line: Empirical hit rate per line bucket (calibration table lookup)

For each method × direction (over/under/both) × edge threshold × line filter:
  - edge = p_model - raw_prob  (using raw market implied probability, per skill spec)
  - ROI = (net_units / n_bets) * 100
  - Hit rate = fraction of bets that won

Output:
  ~/Downloads/tmp/mlb_game_totals/step5_grid_search.csv  — full results
  Prints best configs by ROI and Sharpe-like ratio

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step45_grid_search.py
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
LOCAL_PATH = LOCAL_DIR / "game_totals_spine.parquet"
OUT_CSV    = LOCAL_DIR / "step5_grid_search.csv"

N_FOLDS = 5
SEED    = 42

# Grid search params
EDGE_THRESHOLDS    = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
DIRECTIONS         = ["over", "under", "both"]
LINE_FILTERS       = [None, "half_only", [9.5], [8.5, 9.5], [9.5, 10.5]]
MIN_BOOKS          = [1, 3]


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(LOCAL_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["hit_over"]  = df["hit_over"].astype(int)
    df["hit_under"] = df["hit_under"].astype(int)
    return df


def compute_oof_model_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    OOF logistic regression with top-5 features.
    Returns df with p_model_over and p_model_under columns.
    """
    feature_cols = ["line_is_integer", "home_ra_career", "home_ra_L20", "home_rs_L10", "line"]
    df = df.copy()
    df["p_model_over"]  = np.nan
    df["p_model_under"] = np.nan

    valid_mask = df[feature_cols + ["hit_over", "hit_under"]].notna().all(axis=1)
    valid_idx  = df[valid_mask].index
    sub = df.loc[valid_idx].sort_values("game_date").reset_index(drop=True)

    X      = sub[feature_cols].values.astype(float)
    y_over  = sub["hit_over"].values
    y_under = sub["hit_under"].values
    groups  = sub["game_pk"].values
    dates   = sub["game_date"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_over  = np.zeros(len(sub))
    oof_under = np.zeros(len(sub))

    for train_idx, test_idx in gkf.split(X, y_over, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_idx])
        Xte = sc.transform(X[test_idx])

        clf_over  = LogisticRegression(max_iter=500)
        clf_under = LogisticRegression(max_iter=500)
        clf_over.fit(Xtr,  y_over[train_idx])
        clf_under.fit(Xtr, y_under[train_idx])

        oof_over[test_idx]  = clf_over.predict_proba(Xte)[:, 1]
        oof_under[test_idx] = clf_under.predict_proba(Xte)[:, 1]

    sub["p_model_over"]  = oof_over
    sub["p_model_under"] = oof_under

    # Map back to original df by index position
    original_indices = df[valid_mask].index
    df.loc[original_indices, "p_model_over"]  = sub["p_model_over"].values
    df.loc[original_indices, "p_model_under"] = sub["p_model_under"].values

    return df


def compute_consensus_line_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    OOF calibration table lookup.
    p_model_over  = historical over_rate  at this line (from training fold)
    p_model_under = historical under_rate at this line (from training fold)
    """
    df = df.copy()
    df["p_cal_over"]  = np.nan
    df["p_cal_under"] = np.nan

    sub = df[["line", "hit_over", "hit_under", "game_pk", "game_date"]].dropna()
    sub = sub.sort_values("game_date").reset_index(drop=True)
    groups = sub["game_pk"].values
    y = sub["hit_over"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_over  = np.zeros(len(sub))
    oof_under = np.zeros(len(sub))

    for train_idx, test_idx in gkf.split(sub.values, y, groups):
        tr = sub.iloc[train_idx]
        te = sub.iloc[test_idx]

        global_over  = tr["hit_over"].mean()
        global_under = tr["hit_under"].mean()
        cal_over  = tr.groupby("line")["hit_over"].mean()
        cal_under = tr.groupby("line")["hit_under"].mean()

        oof_over[test_idx]  = te["line"].map(cal_over).fillna(global_over).values
        oof_under[test_idx] = te["line"].map(cal_under).fillna(global_under).values

    # Map back to original df
    valid_idx = df[["line", "hit_over", "hit_under", "game_pk", "game_date"]].dropna().index
    df.loc[valid_idx, "p_cal_over"]  = oof_over
    df.loc[valid_idx, "p_cal_under"] = oof_under

    return df


def compute_pnl(row: pd.Series) -> float:
    """Compute P&L (in units) for a single bet. Bet 1 unit."""
    if row["result"] == "win":
        # Positive American: win (odds/100) units; Negative American: win (100/|odds|) units
        odds = row["odds"]
        if odds > 0:
            return odds / 100
        else:
            return 100 / abs(odds)
    elif row["result"] == "push":
        return 0.0
    else:  # loss
        return -1.0


def run_backtest(df: pd.DataFrame, p_model_col: str, direction: str,
                 edge_threshold: float, line_filter, min_books: int) -> dict:
    """Run backtest for a specific parameter combination."""
    sub = df[df["n_books_total"] >= min_books].copy()

    # Apply line filter
    if line_filter == "half_only":
        sub = sub[sub["line_is_integer"] == 0]
    elif isinstance(line_filter, list):
        sub = sub[sub["line"].isin(line_filter)]

    # Compute edge per direction
    if direction == "over":
        sub["p_model"] = sub[p_model_col + "_over"] if p_model_col + "_over" in sub.columns else sub[p_model_col]
        sub["raw_prob_dir"] = sub["raw_prob_over"]
        sub["novig_prob_dir"] = sub["novig_prob_over"]
        sub["hit"] = sub["hit_over"]
        sub["odds"] = sub["over_price"]
    elif direction == "under":
        sub["p_model"] = sub[p_model_col + "_under"] if p_model_col + "_under" in sub.columns else sub[p_model_col]
        sub["raw_prob_dir"] = sub["raw_prob_under"]
        sub["novig_prob_dir"] = sub["novig_prob_under"]
        sub["hit"] = sub["hit_under"]
        sub["odds"] = sub["under_price"]
    else:  # both
        over_rows = sub.copy()
        over_rows["p_model"] = sub[p_model_col + "_over"]
        over_rows["raw_prob_dir"] = sub["raw_prob_over"]
        over_rows["novig_prob_dir"] = sub["novig_prob_over"]
        over_rows["hit"] = sub["hit_over"]
        over_rows["odds"] = sub["over_price"]
        under_rows = sub.copy()
        under_rows["p_model"] = sub[p_model_col + "_under"]
        under_rows["raw_prob_dir"] = sub["raw_prob_under"]
        under_rows["novig_prob_dir"] = sub["novig_prob_under"]
        under_rows["hit"] = sub["hit_under"]
        under_rows["odds"] = sub["under_price"]
        sub = pd.concat([over_rows, under_rows], ignore_index=True)

    sub = sub.dropna(subset=["p_model", "raw_prob_dir", "hit", "odds"])

    # Edge = p_model - raw_prob (per skill spec)
    sub["edge"] = sub["p_model"] - sub["raw_prob_dir"]

    # Filter by edge threshold
    bets = sub[sub["edge"] >= edge_threshold].copy()

    if len(bets) == 0:
        return None

    # Compute P&L
    bets["result"] = "loss"
    bets.loc[bets["hit"] == 1, "result"] = "win"
    bets.loc[bets["hit_push"] == 1, "result"] = "push"

    bets["pnl"] = bets.apply(compute_pnl, axis=1)

    n_bets  = len(bets)
    n_wins  = (bets["result"] == "win").sum()
    n_push  = (bets["result"] == "push").sum()
    n_loss  = (bets["result"] == "loss").sum()
    net_pnl = bets["pnl"].sum()
    roi     = net_pnl / (n_bets - n_push) * 100 if (n_bets - n_push) > 0 else 0
    hit_rate = n_wins / (n_bets - n_push) if (n_bets - n_push) > 0 else 0

    return {
        "p_model":        p_model_col,
        "direction":      direction,
        "edge_threshold": edge_threshold,
        "line_filter":    str(line_filter),
        "min_books":      min_books,
        "n_bets":         n_bets,
        "n_wins":         n_wins,
        "n_push":         n_push,
        "hit_rate":       round(hit_rate, 4),
        "net_pnl":        round(net_pnl, 2),
        "roi":            round(roi, 4),
    }


def main() -> None:
    print("Loading spine and computing OOF probabilities...")
    df = load_spine()

    print("  Computing logistic model OOF probabilities...")
    df = compute_oof_model_probs(df)

    print("  Computing consensus-line OOF probabilities...")
    df = compute_consensus_line_probs(df)

    print(f"  {df['p_model_over'].notna().sum()} rows with model probs")
    print(f"  {df['p_cal_over'].notna().sum()} rows with calibration probs\n")

    # Spot check: 9.5 under signal
    check = df[df["line"] == 9.5][["line", "hit_under", "novig_prob_under",
                                   "p_model_under", "p_cal_under"]].dropna()
    print(f"=== SPOT CHECK: Line 9.5 under (n={len(check)}) ===")
    print(f"  Actual hit rate:    {check['hit_under'].mean():.3f}")
    print(f"  Novig implied:      {check['novig_prob_under'].mean():.3f}")
    print(f"  Model p_model:      {check['p_model_under'].mean():.3f}")
    print(f"  Calibration p_cal:  {check['p_cal_under'].mean():.3f}\n")

    print("Running grid search...")
    all_results = []

    p_model_map = {
        "p_model": ("p_model_over", "p_model_under"),
        "p_cal":   ("p_cal_over",   "p_cal_under"),
    }

    for p_name, edge_th, direction, line_f, min_b in product(
        ["p_model", "p_cal"], EDGE_THRESHOLDS, DIRECTIONS, LINE_FILTERS, MIN_BOOKS
    ):
        # Patch column names for both directions
        sub_df = df.copy()
        p_over_col, p_under_col = p_model_map[p_name]
        sub_df["_p_over"]  = sub_df[p_over_col]
        sub_df["_p_under"] = sub_df[p_under_col]

        if direction == "over":
            sub_df["p_model_over"]  = sub_df["_p_over"]
        elif direction == "under":
            sub_df["p_model_under"] = sub_df["_p_under"]
        else:
            sub_df["p_model_over"]  = sub_df["_p_over"]
            sub_df["p_model_under"] = sub_df["_p_under"]

        res = run_backtest(sub_df, "p_model", direction, edge_th, line_f, min_b)
        if res is not None:
            res["p_model"] = p_name
            all_results.append(res)

    results_df = pd.DataFrame(all_results)

    print(f"  {len(results_df)} parameter combos with bets\n")

    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}\n")

    # Top configs by ROI (min 50 bets)
    print("=== TOP 20 CONFIGS BY ROI (min 50 bets) ===")
    top_roi = (results_df[results_df["n_bets"] >= 50]
               .sort_values("roi", ascending=False)
               .head(20))
    print(top_roi[["p_model", "direction", "edge_threshold", "line_filter",
                   "min_books", "n_bets", "hit_rate", "net_pnl", "roi"]].to_string(index=False))

    # Top configs by net_pnl (min 50 bets)
    print("\n=== TOP 10 BY NET P&L (min 50 bets) ===")
    top_pnl = (results_df[results_df["n_bets"] >= 50]
               .sort_values("net_pnl", ascending=False)
               .head(10))
    print(top_pnl[["p_model", "direction", "edge_threshold", "line_filter",
                   "min_books", "n_bets", "hit_rate", "net_pnl", "roi"]].to_string(index=False))

    # Best under configs specifically
    print("\n=== TOP UNDER CONFIGS (min 100 bets) ===")
    top_under = (results_df[
        (results_df["n_bets"] >= 100) &
        (results_df["direction"].isin(["under", "both"]))
    ].sort_values("roi", ascending=False).head(15))
    print(top_under[["p_model", "direction", "edge_threshold", "line_filter",
                     "min_books", "n_bets", "hit_rate", "net_pnl", "roi"]].to_string(index=False))

    # Focus on 9.5 line specifically
    print("\n=== ALL CONFIGS TARGETING LINE 9.5 ===")
    nine5 = results_df[results_df["line_filter"].str.contains("9.5", na=False)].sort_values("roi", ascending=False)
    print(nine5[["p_model", "direction", "edge_threshold", "line_filter",
                 "min_books", "n_bets", "hit_rate", "net_pnl", "roi"]].to_string(index=False))


if __name__ == "__main__":
    main()
