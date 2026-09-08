"""
Step 4 — Backtest + Grid Search for MLB Game Totals.

Takes logistic OOF p_over predictions from Step 3b, expands to full spine grain
(one row per game × book × line), computes edge per book, then runs a grid search
over edge thresholds, directions, and odds filters.

Edge = p_model_over - novig_prob_over  (over edge)
     = p_model_under - novig_prob_under = novig_prob_over - p_model_over  (under edge)

Backtest metric: ROI (units won / units bet) at 1 unit per qualifying bet.

Output:
  ~/Downloads/tmp/mlb_game_totals/step4_grid_search.csv
  ~/Downloads/tmp/mlb_game_totals/step4_full_spine_preds.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step4_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_spine.parquet"
OUT_GRID     = Path.home() / "Downloads/tmp/mlb_game_totals/step4_grid_search.csv"
OUT_PREDS    = Path.home() / "Downloads/tmp/mlb_game_totals/step4_full_spine_preds.parquet"

N_SPLITS = 5

GAME_FEATURES = [
    "home_rs_L5", "home_rs_L10", "home_rs_season",
    "home_ra_L5", "home_ra_L10", "home_ra_L20", "home_ra_season",
    "away_rs_L3", "away_rs_L5", "away_rs_season",
    "away_ra_L3", "away_ra_L5", "away_ra_L10", "away_ra_season",
    "park_factor",
    "month",
    "day_of_week",
    "line_is_integer",
    "line",
]

GRID = {
    "direction":       ["over", "under", "both"],
    "edge_threshold":  [0.00, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
    "odds_filter":     ["all", "plus_odds", "minus_odds"],
    "min_books":       [1, 3],
    "line_filter":     ["all", "half_only", "integer_only"],
}


def dedup_to_game_line(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"novig_prob_over": "mean", "novig_prob_under": "mean"}
    first_cols = {c: "first" for c in df.columns
                  if c not in agg and c not in ("game_pk", "line", "novig_prob_over", "novig_prob_under")}
    return df.groupby(["game_pk", "line"]).agg({**first_cols, **agg}).reset_index()


def payout_roi(row: pd.Series, direction: str) -> float:
    """Return profit/loss for a 1-unit bet given outcome and American odds."""
    if direction == "over":
        hit = int(row["hit_over"])
        price = row["over_price"]
    else:
        hit = int(row["hit_under"])
        price = row["under_price"]

    push = int(row["hit_push"])
    if push:
        return 0.0  # stake returned
    if hit:
        if price >= 0:
            return price / 100.0
        else:
            return 100.0 / abs(price)
    return -1.0


def run_backtest(df: pd.DataFrame, direction: str, edge_threshold: float,
                 odds_filter: str, min_books: int, line_filter: str) -> dict:
    d = df.copy()

    # Direction filter
    if direction == "over":
        d = d[d["book_over_edge"] >= edge_threshold]
    elif direction == "under":
        d = d[d["book_under_edge"] >= edge_threshold]
    else:  # both
        d = d[(d["book_over_edge"] >= edge_threshold) | (d["book_under_edge"] >= edge_threshold)]

    # Odds filter
    if odds_filter == "plus_odds":
        if direction in ("over", "both"):
            d = d[d["over_price"] > 0]
        if direction == "under":
            d = d[d["under_price"] > 0]
    elif odds_filter == "minus_odds":
        if direction in ("over", "both"):
            d = d[d["over_price"] < 0]
        if direction == "under":
            d = d[d["under_price"] < 0]

    # Min books filter
    if min_books > 1:
        d = d[d["n_books_total"] >= min_books]

    # Line filter
    if line_filter == "half_only":
        d = d[d["line_is_integer"] == 0]
    elif line_filter == "integer_only":
        d = d[d["line_is_integer"] == 1]

    if len(d) < 10:
        return {"direction": direction, "edge_threshold": edge_threshold,
                "odds_filter": odds_filter, "min_books": min_books,
                "line_filter": line_filter, "n_bets": 0, "roi": np.nan,
                "units_won": np.nan, "hit_rate": np.nan}

    # Compute payout
    if direction == "over":
        bet_dir_col = "over"
    elif direction == "under":
        bet_dir_col = "under"
    else:
        # For "both": bet whichever side has edge
        d = d.copy()
        d["_bet_dir"] = np.where(d["book_over_edge"] >= d["book_under_edge"], "over", "under")
        results = []
        for _, row in d.iterrows():
            results.append(payout_roi(row, row["_bet_dir"]))
        n_bets = len(d)
        units = sum(results)
        hits = sum(1 for r in results if r > 0)
        return {"direction": direction, "edge_threshold": edge_threshold,
                "odds_filter": odds_filter, "min_books": min_books,
                "line_filter": line_filter,
                "n_bets": n_bets, "roi": round(units / n_bets, 4),
                "units_won": round(units, 2), "hit_rate": round(hits / n_bets, 4)}

    results = [payout_roi(row, bet_dir_col) for _, row in d.iterrows()]
    n_bets = len(results)
    units = sum(results)
    hits = sum(1 for r in results if r > 0)

    return {"direction": direction, "edge_threshold": edge_threshold,
            "odds_filter": odds_filter, "min_books": min_books,
            "line_filter": line_filter,
            "n_bets": n_bets, "roi": round(units / n_bets, 4),
            "units_won": round(units, 2), "hit_rate": round(hits / n_bets, 4)}


def main() -> None:
    print("Loading spine...")
    df_all = pd.read_parquet(LOCAL_SPINE)
    print(f"  {len(df_all)} rows, {df_all.game_pk.nunique()} games")

    # ── Step 1: Re-train logistic OOF at (game_pk, line) grain ──────────────
    print("Deduplicating to (game_pk, line) for training...")
    df_gl = dedup_to_game_line(df_all)
    df_gl["hit_over"] = df_gl["hit_over"].astype(float)

    valid_gl = df_gl.dropna(subset=GAME_FEATURES + ["hit_over"])
    X = valid_gl[GAME_FEATURES].values
    y = valid_gl["hit_over"].values.astype(int)
    groups = valid_gl["game_pk"].values

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_probs = np.full(len(y), np.nan)

    print(f"Training logistic OOF ({N_SPLITS} folds)...")
    for tr, val in gkf.split(X, y, groups):
        X_tr_s = scaler.fit_transform(X[tr])
        X_val_s = scaler.transform(X[val])
        clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        clf.fit(X_tr_s, y[tr])
        oof_probs[val] = clf.predict_proba(X_val_s)[:, 1]

    valid_gl = valid_gl.copy()
    valid_gl["p_model_over"]  = oof_probs
    valid_gl["p_model_under"] = 1.0 - oof_probs

    # ── Step 2: Expand p_model back to full spine ────────────────────────────
    print("Expanding model predictions to full spine grain...")
    pred_map = valid_gl[["game_pk", "line", "p_model_over", "p_model_under"]].copy()
    df_full = df_all.merge(pred_map, on=["game_pk", "line"], how="left")

    # Compute per-book edge
    df_full["book_over_edge"]  = df_full["p_model_over"]  - df_full["novig_prob_over"]
    df_full["book_under_edge"] = df_full["p_model_under"] - df_full["novig_prob_under"]

    print(f"  Full spine with predictions: {len(df_full)} rows")
    print(f"  Rows with p_model: {df_full['p_model_over'].notna().sum()} ({df_full['p_model_over'].notna().mean():.1%})")

    # Drop rows where we have no model prediction
    df_pred = df_full.dropna(subset=["p_model_over", "hit_over"]).copy()
    print(f"  Rows with both prediction and outcome: {len(df_pred)}")

    # ── Step 3: Edge distribution ─────────────────────────────────────────────
    print(f"\nAvg over edge by line:")
    by_line = df_pred.groupby("line").agg(
        n=("book_over_edge","count"),
        avg_over_edge=("book_over_edge","mean"),
        avg_under_edge=("book_under_edge","mean"),
        actual_over=("hit_over","mean"),
    ).round(3)
    print(by_line[by_line["n"] >= 50].to_string())

    # ── Step 4: Grid search ─────────────────────────────────────────────────
    print("\nRunning grid search...")
    rows = []
    for direction in GRID["direction"]:
        for edge in GRID["edge_threshold"]:
            for odds in GRID["odds_filter"]:
                for min_b in GRID["min_books"]:
                    for lf in GRID["line_filter"]:
                        r = run_backtest(df_pred, direction, edge, odds, min_b, lf)
                        rows.append(r)

    results = pd.DataFrame(rows)
    results = results.sort_values("roi", ascending=False)

    print("\nTop 20 strategies by ROI (min 100 bets):")
    top = results[results["n_bets"] >= 100].head(20)
    print(top.to_string(index=False))

    print("\nTop 5 strategies for UNDER (min 100 bets):")
    top_under = results[(results["direction"].isin(["under"])) & (results["n_bets"] >= 100)].head(5)
    print(top_under.to_string(index=False))

    results.to_csv(OUT_GRID, index=False)
    df_full.to_parquet(OUT_PREDS, index=False)
    print(f"\nSaved grid → {OUT_GRID}")
    print(f"Saved preds → {OUT_PREDS}")


if __name__ == "__main__":
    main()
