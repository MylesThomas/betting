"""
Steps 5+6 — Temporal OOS Backtest for MLB Game Totals.

Uses proper temporal train/test splits (NOT GroupKFold which allows future leakage):
  - OOS Split 1: Train 2024 → Test 2025
  - OOS Split 2: Train 2024+2025 → Test 2026

For each split, re-trains the full pipeline:
  1. Ridge regression (same 12 features) → y_hat on test games
  2. Method C per-line calibration on training games → p_model_over/under on test games
  3. edge_under = p_model_under - raw_prob_under  (per skill spec: raw, not novig)

Grid search across edge thresholds and line filters.
Key benchmark: "bet all 9.5 unders" regardless of model (pure calibration signal).

Output:
  ~/Downloads/tmp/mlb_game_totals/step56_oos_results.csv  — OOS ROI table
  Prints OOS summary per split

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step56_backtest.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
SPINE_PATH = LOCAL_DIR / "game_totals_spine.parquet"
OUT_CSV    = LOCAL_DIR / "step56_oos_results.csv"

FEATURE_COLS = [
    "consensus_line",
    "park_factor",
    "combined_ra_L10",
    "home_ra_L20",
    "combined_ra_L5",
    "combined_rs_L10",
    "home_rs_L10",
    "away_rs_L3",
    "combined_ra_career",
    "away_ra_L5",
    "home_ra_L10",
    "away_ra_L10",
]

MIN_GAMES_CAL = 30   # min games per line to fit per-line calibration
MAX_LINE      = 13.0
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
LINE_FILTERS    = [None, [9.5], [8.5, 9.5], [9.0, 9.5, 10.0]]


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(SPINE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]
    df = df[df["line"] <= MAX_LINE]
    return df


def get_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one row per game for regression training."""
    cols = ["game_pk", "game_date", "season", "total_runs"] + FEATURE_COLS
    cols = list(dict.fromkeys(cols))
    return df[cols].drop_duplicates("game_pk").dropna(subset=["total_runs"] + FEATURE_COLS)


def fit_regression(games_train: pd.DataFrame, games_test: pd.DataFrame) -> np.ndarray:
    """Fit Ridge regression on training games, predict on test games."""
    X_tr = games_train[FEATURE_COLS].values.astype(float)
    y_tr = games_train["total_runs"].values.astype(float)
    X_te = games_test[FEATURE_COLS].values.astype(float)

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    mdl  = Ridge(alpha=50)
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te)


def fit_calibration(games_train: pd.DataFrame, spine_test: pd.DataFrame) -> pd.DataFrame:
    """
    Method C per-line calibration:
      For each line bucket in test, fit logistic(y_hat) → P(hit_over) on training data.
      Falls back to global calibration for lines with < MIN_GAMES_CAL training examples.
    """
    spine_test = spine_test.copy()
    spine_test["p_model_over"]  = np.nan
    spine_test["p_model_under"] = np.nan

    # Global fallback
    g_valid = games_train.dropna(subset=["y_hat_train", "hit_over"])
    if len(g_valid) < 10:
        spine_test["p_model_over"]  = 0.5
        spine_test["p_model_under"] = 0.5
        return spine_test

    sc_g = StandardScaler()
    X_g  = sc_g.fit_transform(g_valid[["y_hat_train"]].values.astype(float))
    clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(X_g, g_valid["hit_over"].values.astype(int))

    for line_val in spine_test["line"].unique():
        te_mask = spine_test["line"] == line_val
        tr_mask = games_train["line"] == line_val

        n_tr = tr_mask.sum()
        te_rows = spine_test[te_mask]

        if n_tr < MIN_GAMES_CAL:
            X_te = sc_g.transform(te_rows[["y_hat_test"]].values.astype(float))
            preds = clf_g.predict_proba(X_te)[:, 1]
        else:
            tr_sub = games_train[tr_mask].dropna(subset=["y_hat_train", "hit_over"])
            if len(tr_sub) < MIN_GAMES_CAL:
                X_te = sc_g.transform(te_rows[["y_hat_test"]].values.astype(float))
                preds = clf_g.predict_proba(X_te)[:, 1]
            else:
                sc  = StandardScaler()
                X_tr = sc.fit_transform(tr_sub[["y_hat_train"]].values.astype(float))
                X_te = sc.transform(te_rows[["y_hat_test"]].values.astype(float))
                clf  = LogisticRegression(max_iter=500, C=0.5)
                clf.fit(X_tr, tr_sub["hit_over"].values.astype(int))
                preds = clf.predict_proba(X_te)[:, 1]

        spine_test.loc[te_mask, "p_model_over"]  = preds
        spine_test.loc[te_mask, "p_model_under"] = 1.0 - preds

    spine_test["edge_under"] = spine_test["p_model_under"] - spine_test["raw_prob_under"]
    spine_test["edge_over"]  = spine_test["p_model_over"]  - spine_test["raw_prob_over"]
    return spine_test


def compute_pnl(row: pd.Series) -> float:
    """Compute P&L for 1-unit bet."""
    odds = row["odds"]
    if row["hit"] == 1:
        return float(odds / 100) if odds > 0 else float(100 / abs(odds))
    elif row.get("is_push", False):
        return 0.0
    else:
        return -1.0


def run_backtest_config(spine: pd.DataFrame, direction: str,
                        edge_threshold: float, line_filter) -> dict | None:
    sub = spine.copy()

    if line_filter is not None:
        sub = sub[sub["line"].isin(line_filter)]

    if direction == "under":
        sub["edge"] = sub["edge_under"]
        sub["hit"]  = sub["hit_under"]
        sub["odds"] = sub["under_price"]
    else:
        sub["edge"] = sub["edge_over"]
        sub["hit"]  = sub["hit_over"]
        sub["odds"] = sub["over_price"]

    sub = sub.dropna(subset=["edge", "hit", "odds"])
    bets = sub[sub["edge"] >= edge_threshold].copy()
    if len(bets) == 0:
        return None

    bets["is_push"] = bets["hit_push"]
    bets["pnl"] = bets.apply(compute_pnl, axis=1)

    n       = len(bets)
    n_push  = int(bets["hit_push"].sum())
    n_win   = int(bets["hit"].sum())
    net     = float(bets["pnl"].sum())
    denom   = n - n_push
    roi     = net / denom * 100 if denom > 0 else 0
    hr      = n_win / denom if denom > 0 else 0

    return {
        "direction":      direction,
        "edge_threshold": edge_threshold,
        "line_filter":    str(line_filter),
        "n_bets":         n,
        "n_wins":         n_win,
        "n_push":         n_push,
        "hit_rate":       round(hr, 4),
        "net_pnl":        round(net, 2),
        "roi":            round(roi, 4),
    }


def run_split(spine: pd.DataFrame, train_seasons: list[int],
              test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train pipeline on train_seasons, evaluate on test_season.
    Returns (all_results_df, test_spine_with_pmodel).
    """
    print(f"\n{'='*60}")
    print(f"  TRAIN {train_seasons} → TEST {test_season}")
    print(f"{'='*60}")

    train_df = spine[spine["season"].isin(train_seasons)]
    test_df  = spine[spine["season"] == test_season]

    games_train = get_game_level(train_df)
    games_test  = get_game_level(test_df)
    print(f"  Train games: {len(games_train)}  |  Test games: {len(games_test)}")

    # 1. Fit regression
    y_hat_test = fit_regression(games_train, games_test)
    games_test = games_test.copy()
    games_test["y_hat_test"]  = y_hat_test

    # For calibration, we need y_hat on training games too (for per-line calibration)
    # Fit on training, predict training (in-sample for calibration fitting)
    y_hat_train = fit_regression(games_train, games_train)
    games_train = games_train.copy()
    games_train["y_hat_train"] = y_hat_train
    # Add hit_over to games_train for calibration fitting
    hit_over_map = train_df.drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    games_train["hit_over"] = games_train["game_pk"].map(hit_over_map)
    # Add line to games_train (use consensus_line)
    games_train["line"] = games_train["consensus_line"].round(1)

    # 2. Broadcast y_hat to test spine rows
    test_spine = test_df.copy()
    yhat_map = games_test.set_index("game_pk")["y_hat_test"]
    test_spine["y_hat_test"] = test_spine["game_pk"].map(yhat_map)
    test_spine = test_spine.dropna(subset=["y_hat_test"])

    # 3. Method C calibration
    test_spine = fit_calibration(games_train, test_spine)

    n_calibrated = test_spine["p_model_under"].notna().sum()
    print(f"  Test rows with p_model: {n_calibrated}")

    # y_hat stats
    print(f"  y_hat test: mean={y_hat_test.mean():.2f}  std={y_hat_test.std():.2f}  "
          f"range=[{y_hat_test.min():.2f}, {y_hat_test.max():.2f}]")

    # 4. Grid search backtest
    print(f"\n  Running grid search...")
    all_results = []
    for direction in ["under", "over"]:
        for edge_th in EDGE_THRESHOLDS:
            for lf in LINE_FILTERS:
                res = run_backtest_config(test_spine, direction, edge_th, lf)
                if res is not None:
                    res["test_season"] = test_season
                    all_results.append(res)

    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    print(f"  Grid combos with bets: {len(results_df)}")

    # Benchmark: bet all 9.5 unders regardless of model edge
    bm = test_spine[(test_spine["line"] == 9.5)].copy()
    bm["hit"] = bm["hit_under"]
    bm["odds"] = bm["under_price"]
    bm["is_push"] = bm["hit_push"]
    bm = bm.dropna(subset=["hit", "odds"])
    if len(bm) > 0:
        bm["pnl"] = bm.apply(compute_pnl, axis=1)
        n    = len(bm)
        npush = int(bm["hit_push"].sum())
        nwin  = int(bm["hit_under"].sum())
        net   = float(bm["pnl"].sum())
        denom = n - npush
        roi   = net / denom * 100 if denom > 0 else 0
        hr    = nwin / denom if denom > 0 else 0
        print(f"\n  BENCHMARK (bet all 9.5 unders): n={n}  hit_rate={hr:.3f}  net={net:.1f}u  ROI={roi:.2f}%")

    # Also: simple calibration only (p_cal = historical hit rate, no regression)
    # Use novig_prob_under as comparison baseline
    novig_bm = test_spine[(test_spine["line"] == 9.5)].copy()
    novig_bm["edge_novig"] = (1 - novig_bm["novig_prob_over"]) - novig_bm["raw_prob_under"]
    novig_bets = novig_bm[novig_bm["edge_novig"] > 0].copy()
    novig_bets["hit"]    = novig_bets["hit_under"]
    novig_bets["odds"]   = novig_bets["under_price"]
    novig_bets["is_push"] = novig_bets["hit_push"]
    novig_bets = novig_bets.dropna(subset=["hit", "odds"])
    if len(novig_bets) > 0:
        novig_bets["pnl"] = novig_bets.apply(compute_pnl, axis=1)
        n2 = len(novig_bets)
        net2 = float(novig_bets["pnl"].sum())
        hr2  = novig_bets["hit_under"].mean()
        roi2 = net2 / n2 * 100 if n2 > 0 else 0
        print(f"  NOVIG SIGNAL (9.5 under, novig edge>0): n={n2}  hit_rate={hr2:.3f}  net={net2:.1f}u  ROI={roi2:.2f}%")

    # Top model configs
    if len(results_df) > 0:
        under_res = results_df[results_df["direction"] == "under"]
        if len(under_res) > 0:
            top5 = (under_res[under_res["n_bets"] >= 50]
                    .sort_values("roi", ascending=False)
                    .head(5))
            if len(top5) > 0:
                print(f"\n  TOP 5 UNDER CONFIGS (n>=50) by ROI:")
                print(top5[["edge_threshold","line_filter","n_bets","hit_rate","net_pnl","roi"]].to_string(index=False))

    return results_df, test_spine


def main() -> None:
    print("Loading spine...")
    spine = load_spine()
    print(f"  {len(spine)} rows, {spine['game_pk'].nunique()} games")
    print(f"  Seasons: {sorted(spine['season'].unique())}")

    all_results = []

    # OOS Split 1: Train 2024 → Test 2025
    res1, spine_2025 = run_split(spine, [2024], 2025)
    if len(res1) > 0:
        all_results.append(res1)

    # OOS Split 2: Train 2024+2025 → Test 2026
    res2, spine_2026 = run_split(spine, [2024, 2025], 2026)
    if len(res2) > 0:
        all_results.append(res2)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUT_CSV, index=False)
        print(f"\n\nFull results saved → {OUT_CSV}")

        # Combined OOS view: sum 2025 + 2026 results at each config
        print("\n\n=== COMBINED OOS SUMMARY (2025+2026) ===")
        print("Top configs for UNDER by combined ROI (min 100 bets total):")
        under_comb = combined[combined["direction"] == "under"]
        grouped = (under_comb.groupby(["edge_threshold", "line_filter"])
                   .agg(n_bets_total=("n_bets", "sum"),
                        net_pnl_total=("net_pnl", "sum"),
                        n_wins_total=("n_wins", "sum"))
                   .reset_index())
        grouped["roi_combined"] = grouped["net_pnl_total"] / grouped["n_bets_total"] * 100
        grouped["hit_rate_combined"] = grouped["n_wins_total"] / grouped["n_bets_total"]
        top = grouped[grouped["n_bets_total"] >= 100].sort_values("roi_combined", ascending=False).head(15)
        print(top[["edge_threshold","line_filter","n_bets_total","hit_rate_combined","net_pnl_total","roi_combined"]].to_string(index=False))

        # Benchmark: all 9.5 unders across both OOS years
        print("\n\n=== BENCHMARK: ALL 9.5 UNDERS, OOS 2025+2026 ===")
        test_spines = pd.concat([spine_2025, spine_2026])
        bm = test_spines[(test_spines["line"] == 9.5)].copy()
        bm["hit"]    = bm["hit_under"]
        bm["odds"]   = bm["under_price"]
        bm["is_push"] = bm["hit_push"]
        bm = bm.dropna(subset=["hit", "odds"])
        if len(bm) > 0:
            bm["pnl"] = bm.apply(compute_pnl, axis=1)
            n    = len(bm)
            npush = int(bm["hit_push"].sum())
            nwin  = int(bm["hit_under"].sum())
            net   = float(bm["pnl"].sum())
            denom = n - npush
            roi   = net / denom * 100 if denom > 0 else 0
            hr    = nwin / denom if denom > 0 else 0
            print(f"  All 9.5 unders: n={n}  hit_rate={hr:.3f}  net={net:.1f}u  ROI={roi:.2f}%")

            # By year
            for yr in [2025, 2026]:
                bm_yr = bm[bm["season"] == yr]
                if len(bm_yr) > 0:
                    npush_yr = int(bm_yr["hit_push"].sum())
                    nwin_yr  = int(bm_yr["hit_under"].sum())
                    net_yr   = float(bm_yr["pnl"].sum())
                    denom_yr = len(bm_yr) - npush_yr
                    roi_yr   = net_yr / denom_yr * 100 if denom_yr > 0 else 0
                    hr_yr    = nwin_yr / denom_yr if denom_yr > 0 else 0
                    print(f"  {yr}: n={len(bm_yr)}  hit_rate={hr_yr:.3f}  net={net_yr:.1f}u  ROI={roi_yr:.2f}%")

        print("\n=== CONCLUSION ===")
        print("""
  Key question: does the regression+calibration model add edge beyond
  the raw '9.5 under calibration gap' signal?

  Compare:
    Benchmark: bet all 9.5 unders (edge-agnostic) → ROI shown above
    Model:     bet 9.5 unders where model edge > threshold → top configs above

  If model ROI ≈ benchmark ROI at low thresholds, the regression is not adding
  meaningful within-line discrimination. The signal is in the calibration gap
  (market underprices unders at 9.5), not in identifying which specific games.
        """)


if __name__ == "__main__":
    main()
