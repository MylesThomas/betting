"""
Step 6b — Drawdown analysis for MLB Game Totals OOS backtest.

Adds max drawdown (MDD) to the key strategy configs from step56.
Re-runs the OOS pipeline and computes equity curves + MDD per config.

Key configs to evaluate:
  - BENCHMARK: all 9.5 unders (edge-agnostic)
  - edge>0.00, line=[9.5]
  - edge>0.01, line=[9.5]
  - edge>0.02, line=[9.5]

Sorted by net_pnl descending (not ROI%) since we flat-bet 1u per qualifying bet.
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
OUT_CSV    = LOCAL_DIR / "step6b_drawdown_results.csv"

FEATURE_COLS = [
    "consensus_line", "park_factor", "combined_ra_L10", "home_ra_L20",
    "combined_ra_L5", "combined_rs_L10", "home_rs_L10", "away_rs_L3",
    "combined_ra_career", "away_ra_L5", "home_ra_L10", "away_ra_L10",
]
MIN_GAMES_CAL = 30
MAX_LINE      = 13.0


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(SPINE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]
    return df[df["line"] <= MAX_LINE]


def get_game_level(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_pk", "game_date", "season", "total_runs"] + FEATURE_COLS
    cols = list(dict.fromkeys(cols))
    return df[cols].drop_duplicates("game_pk").dropna(subset=["total_runs"] + FEATURE_COLS)


def fit_pipeline(games_train: pd.DataFrame, games_test: pd.DataFrame,
                 spine_test: pd.DataFrame) -> pd.DataFrame:
    """Full OOS pipeline: Ridge regression + Method C calibration → scored test spine."""
    X_tr = games_train[FEATURE_COLS].values.astype(float)
    y_tr = games_train["total_runs"].values.astype(float)
    X_te = games_test[FEATURE_COLS].values.astype(float)

    sc   = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    mdl = Ridge(alpha=50)
    mdl.fit(X_tr_s, y_tr)

    # In-sample y_hat for calibration fitting
    y_hat_train = mdl.predict(X_tr_s)
    games_train = games_train.copy()
    games_train["y_hat"] = y_hat_train
    hit_over_map = spine_test.drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    # Use train split's hit_over
    train_hit = (games_train["game_pk"].map(
        pd.read_parquet(SPINE_PATH).drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    ))
    games_train["hit_over"]   = train_hit
    games_train["line_bucket"] = games_train["consensus_line"].round(1)

    # Global fallback calibration
    tr_valid = games_train.dropna(subset=["y_hat", "hit_over"])
    sc_g = StandardScaler()
    X_g  = sc_g.fit_transform(tr_valid[["y_hat"]].values.astype(float))
    clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(X_g, tr_valid["hit_over"].values.astype(int))

    # Per-line calibration models
    calib = {None: (sc_g, clf_g)}
    for line_val in sorted(tr_valid["line_bucket"].unique()):
        sub = tr_valid[tr_valid["line_bucket"] == line_val]
        if len(sub) < MIN_GAMES_CAL:
            continue
        sc2 = StandardScaler()
        X_s = sc2.fit_transform(sub[["y_hat"]].values.astype(float))
        clf2 = LogisticRegression(max_iter=500, C=0.5)
        clf2.fit(X_s, sub["hit_over"].values.astype(int))
        calib[line_val] = (sc2, clf2)

    # Score test spine
    y_hat_test = mdl.predict(X_te_s)
    yhat_map = dict(zip(games_test["game_pk"], y_hat_test))

    spine_test = spine_test.copy()
    spine_test["y_hat"] = spine_test["game_pk"].map(yhat_map)
    spine_test = spine_test.dropna(subset=["y_hat"])

    p_over_arr = []
    for _, row in spine_test.iterrows():
        bucket = round(float(row["line"]), 1)
        if bucket in calib:
            sc2, clf2 = calib[bucket]
        else:
            sc2, clf2 = calib[None]
        X = sc2.transform(np.array([[row["y_hat"]]]))
        p_over_arr.append(float(clf2.predict_proba(X)[0, 1]))

    spine_test["p_model_over"]  = p_over_arr
    spine_test["p_model_under"] = 1.0 - spine_test["p_model_over"]
    spine_test["edge_under"]    = spine_test["p_model_under"] - spine_test["raw_prob_under"]
    spine_test["edge_over"]     = spine_test["p_model_over"]  - spine_test["raw_prob_over"]
    return spine_test


def pnl_bet(under_price: float, hit_under: int) -> float:
    p = float(under_price)
    if hit_under == 1:
        return float(p / 100) if p > 0 else float(100 / abs(p))
    return -1.0


def drawdown_stats(pnl_series: pd.Series, dates: pd.Series | None = None) -> dict:
    """
    Compute peak-to-trough max drawdown and (optionally) its dates.

    MDD is peak-to-trough on the cumulative equity curve, NOT from zero.
    Being -47u from a prior +88u peak (MDD = -47u) while up +41u overall
    is very different from being -47u underwater — the Calmar framing is
    what matters for live risk management.

    Returns:
      mdd               — max peak-to-trough drawdown in units
      mdd_peak_date     — date equity hit the peak before the deepest trough
      mdd_trough_date   — date equity hit the trough
      mdd_recovery_date — date equity recovered to prior peak (None if unrecovered)
      still_in_drawdown — True if max drawdown has NOT recovered by end of sample
    """
    cumsum      = pnl_series.cumsum()
    rolling_max = cumsum.cummax()
    drawdown    = cumsum - rolling_max
    mdd         = float(drawdown.min())

    result = {
        "mdd":               round(mdd, 2),
        "mdd_peak_date":     None,
        "mdd_trough_date":   None,
        "mdd_recovery_date": None,
        "still_in_drawdown": False,
    }

    if dates is None or len(cumsum) == 0 or mdd == 0:
        return result

    trough_idx   = int(drawdown.idxmin())
    peak_val     = float(rolling_max.iloc[trough_idx])

    # Peak: last point before trough where equity == rolling_max value
    peak_candidates = cumsum[:trough_idx + 1]
    peak_idx = int((peak_candidates[peak_candidates == peak_val]).index[-1])

    result["mdd_peak_date"]   = str(dates.iloc[peak_idx])[:10]
    result["mdd_trough_date"] = str(dates.iloc[trough_idx])[:10]

    # Recovery: first point after trough where cumsum >= peak_val
    after_trough = cumsum.iloc[trough_idx + 1:]
    recovered    = after_trough[after_trough >= peak_val]
    if len(recovered) > 0:
        result["mdd_recovery_date"] = str(dates.iloc[int(recovered.index[0])])[:10]
        result["still_in_drawdown"] = False
    else:
        result["still_in_drawdown"] = True

    return result


def analyze_config(test_spine: pd.DataFrame, label: str,
                   line_filter=None, edge_min: float = -999.0) -> dict | None:
    sub = test_spine.copy()
    if line_filter is not None:
        sub = sub[sub["line"].isin(line_filter)]
    sub = sub[sub["edge_under"] >= edge_min].copy()
    sub = sub.dropna(subset=["hit_under", "under_price"])
    sub = sub.sort_values("game_date").reset_index(drop=True)

    if len(sub) == 0:
        return None

    sub["pnl"] = sub.apply(lambda r: pnl_bet(r["under_price"], r["hit_under"]), axis=1)
    n       = len(sub)
    n_push  = int(sub["hit_push"].sum()) if "hit_push" in sub else 0
    n_wins  = int(sub["hit_under"].sum())
    net     = float(sub["pnl"].sum())
    denom   = n - n_push
    roi     = net / denom * 100 if denom > 0 else 0
    hr      = n_wins / denom if denom > 0 else 0

    dd      = drawdown_stats(sub["pnl"], sub["game_date"])

    return {
        "config":              label,
        "n_bets":              n,
        "n_wins":              n_wins,
        "hit_rate":            round(hr, 3),
        "net_pnl":             round(net, 2),
        "roi_pct":             round(roi, 2),
        "mdd":                 dd["mdd"],
        "net_per_mdd":         round(net / abs(dd["mdd"]), 2) if dd["mdd"] != 0 else float("inf"),
        "mdd_peak_date":       dd["mdd_peak_date"],
        "mdd_trough_date":     dd["mdd_trough_date"],
        "mdd_recovery_date":   dd["mdd_recovery_date"],
        "still_in_drawdown":   dd["still_in_drawdown"],
    }


def main() -> None:
    print("Loading spine...")
    spine = load_spine()
    print(f"  {len(spine)} rows, {spine['game_pk'].nunique()} games")

    all_results = []
    all_test_spines = []

    for train_seasons, test_season in [([2024], 2025), ([2024, 2025], 2026)]:
        print(f"\n--- Train {train_seasons} → Test {test_season} ---")
        train_df = spine[spine["season"].isin(train_seasons)]
        test_df  = spine[spine["season"] == test_season]

        games_train = get_game_level(train_df).dropna(subset=["total_runs"])
        games_test  = get_game_level(test_df)

        print(f"  Train: {len(games_train)} games  |  Test: {len(games_test)} games")
        test_spine = fit_pipeline(games_train, games_test, test_df)
        test_spine["test_season"] = test_season
        all_test_spines.append(test_spine)

    combined_test = pd.concat(all_test_spines, ignore_index=True)
    print(f"\nCombined test spine: {len(combined_test)} rows")

    # Define configs to evaluate
    configs = [
        # label, line_filter, edge_min
        ("BENCHMARK: all 9.5 unders (edge-agnostic)",   [9.5], -999.0),
        ("edge>0.00, line=[9.5]",                        [9.5],  0.00),
        ("edge>0.01, line=[9.5]",                        [9.5],  0.01),
        ("edge>0.02, line=[9.5]",                        [9.5],  0.02),
        ("edge>0.03, line=[9.5]",                        [9.5],  0.03),
        ("edge>0.05, line=[9.5]",                        [9.5],  0.05),
        ("edge>0.00, line=[8.5,9.5]",                    [8.5, 9.5], 0.00),
        ("edge>0.01, line=[8.5,9.5]",                    [8.5, 9.5], 0.01),
    ]

    print("\n=== COMBINED OOS 2025+2026 — with DRAWDOWN (peak-to-trough) ===")
    print(f"  {'Config':<44}  {'n':>5}  {'hit%':>6}  {'net':>8}  {'ROI%':>7}  {'MDD':>8}  {'net/MDD':>8}  {'peak→trough':<24}  {'recovered?'}")
    print(f"  {'-'*44}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*24}  {'-'*12}")
    rows = []
    for label, lf, em in configs:
        r = analyze_config(combined_test, label, line_filter=lf, edge_min=em)
        if r:
            rows.append(r)
            peak_trough = f"{r['mdd_peak_date']} → {r['mdd_trough_date']}" if r['mdd_peak_date'] else "—"
            recovered   = ("yes " + r['mdd_recovery_date']) if r['mdd_recovery_date'] else ("still in DD" if r['still_in_drawdown'] else "—")
            print(f"  {r['config']:<44}  {r['n_bets']:>5}  {r['hit_rate']:>6.1%}  "
                  f"{r['net_pnl']:>+8.1f}u  {r['roi_pct']:>+7.2f}%  "
                  f"{r['mdd']:>+8.1f}u  {r['net_per_mdd']:>8.2f}x  {peak_trough:<24}  {recovered}")

    # Per-season view for the key configs
    print("\n=== PER-SEASON VIEW ===")
    for label, lf, em in configs[:4]:
        for yr in [2025, 2026]:
            yr_spine = combined_test[combined_test["test_season"] == yr]
            r = analyze_config(yr_spine, label, line_filter=lf, edge_min=em)
            if r:
                print(f"  {yr}  {label:<44}  n={r['n_bets']:>4}  "
                      f"hit={r['hit_rate']:.1%}  net={r['net_pnl']:>+7.1f}u  "
                      f"ROI={r['roi_pct']:>+6.2f}%  MDD={r['mdd']:>+7.1f}u")

    # Save
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    print("\n=== CONCLUSION ===")
    bench = next((r for r in rows if "BENCHMARK" in r["config"]), None)
    model = next((r for r in rows if r["config"] == "edge>0.00, line=[9.5]"), None)
    if bench and model:
        print(f"  Benchmark (all 9.5 unders):  {bench['net_pnl']:+.1f}u  ROI={bench['roi_pct']:+.2f}%  MDD={bench['mdd']:+.1f}u  net/MDD={bench['net_per_mdd']:.2f}x")
        print(f"  Model edge>0:                {model['net_pnl']:+.1f}u  ROI={model['roi_pct']:+.2f}%  MDD={model['mdd']:+.1f}u  net/MDD={model['net_per_mdd']:.2f}x")
        if bench["net_per_mdd"] > model["net_per_mdd"]:
            print(f"\n  → BENCHMARK wins on net/MDD. Use all 9.5 unders as primary strategy.")
        else:
            print(f"\n  → MODEL edge>0 wins on net/MDD. Use edge filter as primary strategy.")


if __name__ == "__main__":
    main()
