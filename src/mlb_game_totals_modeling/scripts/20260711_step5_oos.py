"""
Step 5 — Out-of-Sample Validation for MLB Game Totals.

Time-based split: train on 2024+2025, test on 2026.
Tests the top strategies from the IS grid search on unseen 2026 data.

Output:
  ~/Downloads/tmp/mlb_game_totals/step5_oos_results.csv

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step5_oos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_spine.parquet"
OUT_CSV     = Path.home() / "Downloads/tmp/mlb_game_totals/step5_oos_results.csv"

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


def dedup_to_game_line(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"novig_prob_over": "mean", "novig_prob_under": "mean"}
    first_cols = {c: "first" for c in df.columns
                  if c not in agg and c not in ("game_pk", "line", "novig_prob_over", "novig_prob_under")}
    return df.groupby(["game_pk", "line"]).agg({**first_cols, **agg}).reset_index()


def eval_strategy(df: pd.DataFrame, direction: str, edge_threshold: float,
                  odds_filter: str, line_filter: str, label: str = "") -> dict:
    d = df.copy()

    if direction == "over":
        d = d[d["over_edge"] >= edge_threshold]
    else:
        d = d[d["under_edge"] >= edge_threshold]

    if odds_filter == "plus_odds":
        price_col = "over_price" if direction == "over" else "under_price"
        d = d[d[price_col] > 0]
    elif odds_filter == "minus_odds":
        price_col = "over_price" if direction == "over" else "under_price"
        d = d[d[price_col] < 0]

    if line_filter == "half_only":
        d = d[d["line_is_integer"] == 0]
    elif line_filter == "integer_only":
        d = d[d["line_is_integer"] == 1]

    n = len(d)
    if n < 5:
        return {"label": label, "n_bets": n, "roi": np.nan, "units_won": np.nan,
                "hit_rate": np.nan, "direction": direction, "edge": edge_threshold,
                "odds_filter": odds_filter, "line_filter": line_filter}

    price_col = "over_price" if direction == "over" else "under_price"
    hit_col   = "hit_over"   if direction == "over" else "hit_under"

    def pnl(row):
        push = int(row["hit_push"])
        if push:
            return 0.0
        hit = int(row[hit_col])
        price = row[price_col]
        if hit:
            return price / 100.0 if price >= 0 else 100.0 / abs(price)
        return -1.0

    results = [pnl(row) for _, row in d.iterrows()]
    units = sum(results)
    hits = sum(1 for r in results if r > 0)

    return {"label": label, "direction": direction, "edge": edge_threshold,
            "odds_filter": odds_filter, "line_filter": line_filter,
            "n_bets": n, "roi": round(units / n, 4), "units_won": round(units, 2),
            "hit_rate": round(hits / n, 4)}


def main() -> None:
    print("Loading spine...")
    df_all = pd.read_parquet(LOCAL_SPINE)
    df_all["season"] = pd.to_datetime(df_all["game_date"]).dt.year
    print(f"  {len(df_all)} rows, {df_all.game_pk.nunique()} games")
    print(f"  Season breakdown:\n{df_all.groupby('season')['game_pk'].nunique().to_string()}")

    # Deduplicate to (game_pk, line) for model training
    df_gl = dedup_to_game_line(df_all)
    df_gl["hit_over"] = df_gl["hit_over"].astype(float)

    # Time-based split
    is_mask  = df_gl["season"].isin([2024, 2025])
    oos_mask = df_gl["season"] == 2026

    df_train = df_gl[is_mask].dropna(subset=GAME_FEATURES + ["hit_over"])
    df_test  = df_gl[oos_mask].dropna(subset=GAME_FEATURES + ["hit_over"])

    print(f"\nTrain (2024+2025): {len(df_train)} game-line rows")
    print(f"Test  (2026):       {len(df_test)} game-line rows")

    # Train logistic on IS
    X_tr = df_train[GAME_FEATURES].values
    y_tr = df_train["hit_over"].values.astype(int)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    clf.fit(X_tr_s, y_tr)

    # Predict on OOS
    X_te = df_test[GAME_FEATURES].values
    X_te_s = scaler.transform(X_te)
    p_model = clf.predict_proba(X_te_s)[:, 1]
    df_test = df_test.copy()
    df_test["p_model_over"]  = p_model
    df_test["p_model_under"] = 1.0 - p_model

    auc   = roc_auc_score(df_test["hit_over"].astype(int), p_model)
    brier = brier_score_loss(df_test["hit_over"].astype(int), p_model)
    print(f"\nOOS Model metrics — AUC={auc:.4f}, Brier={brier:.4f}")

    # Expand to full spine (all books) for OOS
    pred_map = df_test[["game_pk", "line", "p_model_over", "p_model_under"]]
    df_oos_full = df_all[df_all["season"] == 2026].merge(
        pred_map, on=["game_pk", "line"], how="inner"
    )
    df_oos_full["over_edge"]  = df_oos_full["p_model_over"]  - df_oos_full["novig_prob_over"]
    df_oos_full["under_edge"] = df_oos_full["p_model_under"] - df_oos_full["novig_prob_under"]
    df_oos_full = df_oos_full.dropna(subset=["hit_over"])
    print(f"\nOOS spine rows: {len(df_oos_full)}")

    # OOS calibration by line
    print("\nOOS calibration by line:")
    by_line = df_oos_full.groupby("line").agg(
        n=("over_edge","count"),
        actual_over=("hit_over","mean"),
        model_over=("p_model_over","mean"),
        market_over=("novig_prob_over","mean"),
        avg_over_edge=("over_edge","mean"),
        avg_under_edge=("under_edge","mean"),
    ).round(3)
    print(by_line[by_line["n"] >= 20].to_string())

    # Test top strategies on OOS
    strategies = [
        # Top IS strategy (suspicious)
        ("OVER 10pp minus half", "over", 0.10, "minus_odds", "half_only"),
        # Under strategies
        ("UNDER 5pp plus half", "under", 0.05, "plus_odds", "half_only"),
        ("UNDER 3pp plus half", "under", 0.03, "plus_odds", "half_only"),
        ("UNDER 5pp all half",  "under", 0.05, "all",        "half_only"),
        ("UNDER 3pp all half",  "under", 0.03, "all",        "half_only"),
        ("UNDER 5pp all all",   "under", 0.05, "all",        "all"),
        ("UNDER 3pp minus half","under", 0.03, "minus_odds",  "half_only"),
        ("UNDER 5pp minus half","under", 0.05, "minus_odds",  "half_only"),
        # Integer line strategies
        ("UNDER 3pp all int",   "under", 0.03, "all",         "integer_only"),
        ("UNDER 5pp all int",   "under", 0.05, "all",         "integer_only"),
        # Baseline: no edge filter
        ("UNDER 0pp all all",   "under", 0.00, "all",         "all"),
        ("OVER 0pp all all",    "over",  0.00, "all",         "all"),
    ]

    oos_rows = []
    for label, direction, edge, odds, lf in strategies:
        r = eval_strategy(df_oos_full, direction, edge, odds, lf, label)
        oos_rows.append(r)
        print(f"  {label}: n={r['n_bets']}, roi={r['roi']:.4f}, units={r['units_won']}, hit={r['hit_rate']}")

    oos_results = pd.DataFrame(oos_rows)
    oos_results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    # Spot check NYY @ BOS (OOS 2026)
    print("\nSpot check NYY @ BOS (OOS 2026):")
    spot = df_oos_full[
        (df_oos_full["home_team"] == "Boston Red Sox") &
        (df_oos_full["away_team"] == "New York Yankees")
    ].sort_values("game_date").drop_duplicates(["game_date","line","bookmaker"])
    print(spot[["game_date","bookmaker","line","total_runs","p_model_over","over_edge","novig_prob_over","hit_over"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
