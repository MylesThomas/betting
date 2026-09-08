"""
Step 6c — Extended grid search for MLB Game Totals OOS backtest.

New dimensions not swept in step56:
  - odds_bucket:       ["all", "plus_odds", "minus_odds"]
  - shrinkage:         [0, 0.25, 0.50, 0.75]
      Shrinks p_model toward novig_prob:
      p_eff = (1 - shrinkage) * p_model + shrinkage * novig_prob
      At 0 = pure model; at 0.75 = mostly market
  - prediction_method: ["model", "consensus_line"]
      "model"          = full 12-feature Ridge + per-line calibration
      "consensus_line" = single-feature Ridge (consensus_line only) + per-line calibration
      Tests whether rolling RA/RS features add anything beyond the market's own priced total.

Combined with existing dims:
  - direction:      ["under", "over"]
  - edge_threshold: [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
  - line_filter:    [None, [9.5], [8.5, 9.5], [9.0, 9.5, 10.0]]

Each config also reports net/MDD (Calmar-equivalent) alongside ROI.
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
OUT_CSV    = LOCAL_DIR / "step6c_extended_grid.csv"

FEATURE_COLS_FULL = [
    "consensus_line", "park_factor", "combined_ra_L10", "home_ra_L20",
    "combined_ra_L5", "combined_rs_L10", "home_rs_L10", "away_rs_L3",
    "combined_ra_career", "away_ra_L5", "home_ra_L10", "away_ra_L10",
]
FEATURE_COLS_CONSENSUS = ["consensus_line"]

MIN_GAMES_CAL = 30
MAX_LINE      = 13.0

EDGE_THRESHOLDS    = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
LINE_FILTERS       = [None, [9.5], [8.5, 9.5], [9.0, 9.5, 10.0]]
ODDS_BUCKETS       = ["all", "plus_odds", "minus_odds"]
SHRINKAGE_VALS     = [0.00, 0.25, 0.50, 0.75]
PRED_METHODS       = ["model", "consensus_line"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(SPINE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]
    return df[df["line"] <= MAX_LINE]


def get_game_level(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cols = ["game_pk", "game_date", "season", "total_runs"] + feature_cols
    cols = list(dict.fromkeys(cols))
    return df[cols].drop_duplicates("game_pk").dropna(subset=["total_runs"] + feature_cols)


# ── Model fitting ──────────────────────────────────────────────────────────────

def fit_ridge(games_train: pd.DataFrame, games_pred: pd.DataFrame,
              feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_hat_train_insample, y_hat_pred)."""
    X_tr = games_train[feature_cols].values.astype(float)
    y_tr = games_train["total_runs"].values.astype(float)
    X_pr = games_pred[feature_cols].values.astype(float)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_pr_s = sc.transform(X_pr)

    mdl = Ridge(alpha=50)
    mdl.fit(X_tr_s, y_tr)
    return mdl.predict(X_tr_s), mdl.predict(X_pr_s)


def fit_calibration(games_train: pd.DataFrame, test_spine: pd.DataFrame) -> pd.DataFrame:
    """Method C per-line logistic calibration. Returns scored test_spine."""
    test_spine = test_spine.copy()

    g_valid = games_train.dropna(subset=["y_hat", "hit_over"])
    sc_g = StandardScaler()
    X_g  = sc_g.fit_transform(g_valid[["y_hat"]].values.astype(float))
    clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(X_g, g_valid["hit_over"].values.astype(int))
    calib = {None: (sc_g, clf_g)}

    for line_val in sorted(g_valid["line_bucket"].unique()):
        sub = g_valid[g_valid["line_bucket"] == line_val]
        if len(sub) < MIN_GAMES_CAL:
            continue
        sc2 = StandardScaler()
        X_s = sc2.fit_transform(sub[["y_hat"]].values.astype(float))
        clf2 = LogisticRegression(max_iter=500, C=0.5)
        clf2.fit(X_s, sub["hit_over"].values.astype(int))
        calib[line_val] = (sc2, clf2)

    p_over = []
    for _, row in test_spine.iterrows():
        bucket = round(float(row["line"]), 1)
        sc2, clf2 = calib.get(bucket, calib[None])
        X = sc2.transform(np.array([[row["y_hat_test"]]]))
        p_over.append(float(clf2.predict_proba(X)[0, 1]))

    test_spine["p_model_over"]  = p_over
    test_spine["p_model_under"] = 1.0 - test_spine["p_model_over"]
    return test_spine


def run_pipeline_for_method(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             spine_train: pd.DataFrame,
                             feature_cols: list[str]) -> pd.DataFrame:
    """Full Ridge + calibration pipeline for one prediction method."""
    games_train = get_game_level(train_df, feature_cols).dropna(subset=["total_runs"])
    games_test  = get_game_level(test_df, feature_cols)

    y_hat_train, y_hat_test = fit_ridge(games_train, games_test, feature_cols)
    games_train = games_train.copy()
    games_train["y_hat"] = y_hat_train

    hit_over_map = spine_train.drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    games_train["hit_over"]    = games_train["game_pk"].map(hit_over_map)
    games_train["line_bucket"] = games_train["consensus_line"].round(1)

    yhat_map   = dict(zip(games_test["game_pk"], y_hat_test))
    test_spine = test_df.copy()
    test_spine["y_hat_test"] = test_spine["game_pk"].map(yhat_map)
    test_spine = test_spine.dropna(subset=["y_hat_test"])

    return fit_calibration(games_train, test_spine)


# ── Drawdown ───────────────────────────────────────────────────────────────────

def mdd_from_pnl(pnl_series: pd.Series) -> float:
    if len(pnl_series) == 0:
        return 0.0
    cumsum      = pnl_series.cumsum()
    rolling_max = cumsum.cummax()
    return float((cumsum - rolling_max).min())


# ── P&L ───────────────────────────────────────────────────────────────────────

def pnl_bet(price: float, hit: int) -> float:
    p = float(price)
    if hit == 1:
        return float(p / 100) if p > 0 else float(100 / abs(p))
    return -1.0


# ── Grid evaluation ────────────────────────────────────────────────────────────

def evaluate_config(
    test_spine: pd.DataFrame,
    direction: str,
    edge_threshold: float,
    line_filter,
    odds_bucket: str,
    shrinkage: float,
) -> dict | None:
    sub = test_spine.copy()

    # Line filter
    if line_filter is not None:
        sub = sub[sub["line"].isin(line_filter)]

    if direction == "both":
        # Bet whichever side the model gives positive edge to — at most one bet per row.
        # edge_under + edge_over = -vig so only one side can be >= threshold at a time.
        p_u = (1 - shrinkage) * sub["p_model_under"] + shrinkage * sub["novig_prob_under"]
        p_o = (1 - shrinkage) * sub["p_model_over"]  + shrinkage * sub["novig_prob_over"]
        sub["edge_u"] = p_u - sub["raw_prob_under"]
        sub["edge_o"] = p_o - sub["raw_prob_over"]
        sub["edge_eff"] = sub[["edge_u", "edge_o"]].max(axis=1)
        sub["bet_under"] = sub["edge_u"] >= sub["edge_o"]
        sub["hit"]   = sub.apply(lambda r: r["hit_under"]  if r["bet_under"] else r["hit_over"],  axis=1)
        sub["price"] = sub.apply(lambda r: r["under_price"] if r["bet_under"] else r["over_price"], axis=1)
        sub = sub[sub["edge_eff"] >= edge_threshold]
    elif direction == "under":
        p_eff = (1 - shrinkage) * sub["p_model_under"] + shrinkage * sub["novig_prob_under"]
        sub["edge_eff"] = p_eff - sub["raw_prob_under"]
        sub["hit"]      = sub["hit_under"]
        sub["price"]    = sub["under_price"]
        sub = sub[sub["edge_eff"] >= edge_threshold]
    else:
        p_eff = (1 - shrinkage) * sub["p_model_over"] + shrinkage * sub["novig_prob_over"]
        sub["edge_eff"] = p_eff - sub["raw_prob_over"]
        sub["hit"]      = sub["hit_over"]
        sub["price"]    = sub["over_price"]
        sub = sub[sub["edge_eff"] >= edge_threshold]

    # Odds bucket filter
    if odds_bucket == "plus_odds":
        sub = sub[sub["price"] > 0]
    elif odds_bucket == "minus_odds":
        sub = sub[sub["price"] < 0]

    sub = sub.dropna(subset=["hit", "price"])
    sub = sub.sort_values("game_date").reset_index(drop=True)

    if len(sub) < 10:
        return None

    sub["pnl"]     = sub.apply(lambda r: pnl_bet(r["price"], r["hit"]), axis=1)
    n              = len(sub)
    n_push         = int(sub["hit_push"].sum()) if "hit_push" in sub.columns else 0
    n_wins         = int(sub["hit"].sum())
    net            = float(sub["pnl"].sum())
    denom          = n - n_push
    roi            = net / denom * 100 if denom > 0 else 0.0
    hr             = n_wins / denom if denom > 0 else 0.0
    mdd            = mdd_from_pnl(sub["pnl"])
    net_per_mdd    = round(net / abs(mdd), 2) if mdd != 0 else float("inf")

    return {
        "direction":      direction,
        "edge_threshold": edge_threshold,
        "line_filter":    str(line_filter),
        "odds_bucket":    odds_bucket,
        "shrinkage":      shrinkage,
        "n_bets":         n,
        "n_wins":         n_wins,
        "hit_rate":       round(hr, 4),
        "net_pnl":        round(net, 2),
        "roi_pct":        round(roi, 2),
        "mdd":            round(mdd, 2),
        "net_per_mdd":    net_per_mdd,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading spine...")
    spine = load_spine()
    print(f"  {len(spine):,} rows  |  {spine['game_pk'].nunique():,} games")

    # Check plus/minus odds distribution at 9.5
    sub95 = spine[spine["line"] == 9.5]
    plus  = (sub95["under_price"] > 0).sum()
    minus = (sub95["under_price"] < 0).sum()
    print(f"  9.5 line under prices: {minus:,} minus-odds, {plus:,} plus-odds ({plus/(plus+minus):.1%} plus)")

    all_test_spines: dict[str, list[pd.DataFrame]] = {"model": [], "consensus_line": []}

    for train_seasons, test_season in [([2024], 2025), ([2024, 2025], 2026)]:
        print(f"\n--- Train {train_seasons} → Test {test_season} ---")
        train_df = spine[spine["season"].isin(train_seasons)]
        test_df  = spine[spine["season"] == test_season]

        for method in PRED_METHODS:
            feat_cols = FEATURE_COLS_FULL if method == "model" else FEATURE_COLS_CONSENSUS
            scored = run_pipeline_for_method(train_df, test_df, train_df, feat_cols)
            scored["test_season"]   = test_season
            scored["pred_method"]   = method
            all_test_spines[method].append(scored)
            print(f"  [{method}] scored {len(scored):,} test rows")

    all_results = []

    for method in PRED_METHODS:
        combined = pd.concat(all_test_spines[method], ignore_index=True)
        print(f"\n=== Sweeping {method} ({len(combined):,} total test rows) ===")

        for direction in ["under", "over", "both"]:
            for edge_th in EDGE_THRESHOLDS:
                for lf in LINE_FILTERS:
                    for ob in ODDS_BUCKETS:
                        for sh in SHRINKAGE_VALS:
                            r = evaluate_config(combined, direction, edge_th, lf, ob, sh)
                            if r:
                                r["pred_method"] = method
                                all_results.append(r)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(results_df):,} configs → {OUT_CSV}")

    # ── Summary tables ─────────────────────────────────────────────────────────

    under_df = results_df[
        (results_df["direction"] == "under") & (results_df["n_bets"] >= 50)
    ].copy()

    print("\n\n=== TOP 20 UNDER CONFIGS by ROI (n≥50) ===")
    top_roi = under_df.sort_values("roi_pct", ascending=False).head(20)
    print(top_roi[[
        "pred_method","edge_threshold","line_filter","odds_bucket","shrinkage",
        "n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    print("\n\n=== TOP 20 UNDER CONFIGS by net/MDD (n≥50) ===")
    top_mdd = under_df[under_df["net_per_mdd"] != float("inf")].sort_values("net_per_mdd", ascending=False).head(20)
    print(top_mdd[[
        "pred_method","edge_threshold","line_filter","odds_bucket","shrinkage",
        "n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    print("\n\n=== TOP 20 UNDER CONFIGS by net_pnl (n≥50) ===")
    top_net = under_df.sort_values("net_pnl", ascending=False).head(20)
    print(top_net[[
        "pred_method","edge_threshold","line_filter","odds_bucket","shrinkage",
        "n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    # ── Shrinkage dimension ────────────────────────────────────────────────────
    print("\n\n=== SHRINKAGE EFFECT on benchmark (9.5 under, edge>=0, all odds) ===")
    bench_rows = under_df[
        (under_df["line_filter"] == "[9.5]") &
        (under_df["edge_threshold"] == 0.0) &
        (under_df["odds_bucket"] == "all")
    ].sort_values(["pred_method", "shrinkage"])
    print(bench_rows[[
        "pred_method","shrinkage","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    # ── Odds bucket dimension ──────────────────────────────────────────────────
    print("\n\n=== ODDS BUCKET EFFECT on 9.5 unders (edge>=0, shrinkage=0) ===")
    ob_rows = under_df[
        (under_df["line_filter"] == "[9.5]") &
        (under_df["edge_threshold"] == 0.0) &
        (under_df["shrinkage"] == 0.0)
    ].sort_values(["pred_method", "odds_bucket"])
    print(ob_rows[[
        "pred_method","odds_bucket","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    # ── Prediction method comparison ──────────────────────────────────────────
    print("\n\n=== MODEL vs CONSENSUS_LINE (9.5 under, all odds, shrinkage=0) ===")
    pm_rows = under_df[
        (under_df["line_filter"] == "[9.5]") &
        (under_df["odds_bucket"] == "all") &
        (under_df["shrinkage"] == 0.0)
    ].sort_values(["pred_method", "edge_threshold"])
    print(pm_rows[[
        "pred_method","edge_threshold","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    # ── "Both" direction: does flipping to over when model prefers it help? ──────
    print("\n\n=== DIRECTION='BOTH' vs 'UNDER' — 9.5 line, all odds, shrinkage=0 ===")
    both_rows = results_df[
        (results_df["line_filter"] == "[9.5]") &
        (results_df["odds_bucket"] == "all") &
        (results_df["shrinkage"] == 0.0) &
        (results_df["pred_method"] == "model") &
        (results_df["direction"].isin(["under", "both"]))
    ].sort_values(["direction", "edge_threshold"])
    print(both_rows[[
        "direction","edge_threshold","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    print("\n\n=== TOP 10 'BOTH' CONFIGS by net_pnl (n≥50) ===")
    both_top = results_df[
        (results_df["direction"] == "both") & (results_df["n_bets"] >= 50)
    ].sort_values("net_pnl", ascending=False).head(10)
    print(both_top[[
        "pred_method","edge_threshold","line_filter","odds_bucket","shrinkage",
        "n_bets","hit_rate","net_pnl","roi_pct","mdd","net_per_mdd"
    ]].to_string(index=False))

    # ── Benchmark reference ────────────────────────────────────────────────────
    print("\n\n=== BENCHMARK REFERENCE (all 9.5 unders, edge=-999, no model) ===")
    bm_rows = results_df[
        (results_df["direction"] == "under") &
        (results_df["line_filter"] == "[9.5]") &
        (results_df["edge_threshold"] == 0.0) &
        (results_df["odds_bucket"] == "all") &
        (results_df["shrinkage"] == 0.0) &
        (results_df["pred_method"] == "model")
    ]
    if len(bm_rows) > 0:
        r = bm_rows.iloc[0]
        print(f"  n={r['n_bets']}  hit={r['hit_rate']:.3f}  net={r['net_pnl']:+.1f}u  ROI={r['roi_pct']:+.2f}%  MDD={r['mdd']:+.1f}u  net/MDD={r['net_per_mdd']:.2f}x")
        print("  (Note: benchmark with edge>=0 filter includes only model-positive rows,")
        print("   not ALL 9.5 unders — the true benchmark is n=2,324 from step6b)")


if __name__ == "__main__":
    main()
