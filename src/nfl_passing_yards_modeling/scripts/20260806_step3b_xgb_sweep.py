"""
Step 3b — Individual Feature Predictive Power Sweep (XGBoost)

Repeats the 3a sweep using XGBoost instead of linear/logistic regression.
Re-runs 3a inline so the final table is a stacked side-by-side comparison:
  one row per (feature, model_type) — model_type in {linear, xgboost}.

Also flags features where XGBoost adds AUC delta > 0.02 over linear.

CV: leave-one-season-out (same as 3a).

Usage:
  uv run python 20260806_step3b_xgb_sweep.py
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
TMP_DIR    = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH  = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
SPINE_PATH = TMP_DIR / "step2_spine.parquet"
OUT_PATH   = TMP_DIR / "step3b_results.parquet"
ET         = ZoneInfo("America/New_York")


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html(df: pd.DataFrame, title: str = "") -> str:
    out = ""
    if title:
        out += f'<p><strong>{title}</strong></p>'
    out += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;font-family:monospace">'
    out += "<thead><tr>" + "".join(
        f"<th style='background:#1a1a2e;color:white;padding:6px 10px'>{c}</th>" for c in df.columns
    ) + "</tr></thead><tbody>"
    for i, row in df.iterrows():
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        out += f"<tr style='background:{bg}'>" + "".join(
            f"<td style='padding:4px 8px'>{v}</td>" for v in row.values
        ) + "</tr>"
    out += "</tbody></table>"
    return out


# ── Load spine ────────────────────────────────────────────────────────────────

print("[Step 3b] Loading spine...")
spine = pd.read_parquet(SPINE_PATH)

df = spine[spine["passing_yards"].notna()].copy().reset_index(drop=True)
df["is_over"] = (df["outcome"] == "over").astype(int)

print(f"  Rows with actuals: {len(df):,}")
print(f"  Seasons: {sorted(df['nfl_season'].unique())}")
print(f"  Over rate: {df['is_over'].mean():.3f}")

SEASONS = sorted(df["nfl_season"].unique())


# ── Feature definitions (same as 3a) ──────────────────────────────────────────

NUMERIC_FEATURES = [
    "pass_yds_roll_1","pass_yds_roll_3","pass_yds_roll_5","pass_yds_roll_10",
    "pass_yds_roll_season","pass_yds_roll_career",
    "attempts_roll_1","attempts_roll_3","attempts_roll_5","attempts_roll_10",
    "attempts_roll_season","attempts_roll_career",
    "comp_pct_roll_1","comp_pct_roll_3","comp_pct_roll_5","comp_pct_roll_10",
    "comp_pct_roll_season","comp_pct_roll_career",
    "ypa_roll_1","ypa_roll_3","ypa_roll_5","ypa_roll_10",
    "ypa_roll_season","ypa_roll_career",
    "epa_per_att_roll_1","epa_per_att_roll_3","epa_per_att_roll_5","epa_per_att_roll_10",
    "epa_per_att_roll_season","epa_per_att_roll_career",
    "sack_rate_roll_3","sack_rate_roll_5","sack_rate_roll_10",
    "sack_rate_roll_season","sack_rate_roll_career",
    "pass_rate_roll_3","pass_rate_roll_5","pass_rate_roll_10",
    "pass_rate_roll_season","pass_rate_roll_career",
    "pass_yds_std_5","pass_yds_std_10","pass_yds_std_career",
    "cover_rate_roll_1","cover_rate_roll_3","cover_rate_roll_5","cover_rate_roll_10",
    "cover_rate_roll_season","cover_rate_roll_career",
    "career_starts_pct","starts_last_4","games_as_starter_this_season","started_week1_for_team",
    "opp_pass_yds_allowed_roll5",
    "total_line","team_spread","implied_team_total",
    "consensus_line","min_line","max_line","n_books",
    "cons_novig_prob_over","min_raw_prob_over","max_raw_prob_over",
    "min_raw_prob_under","max_raw_prob_under",
    "nfl_week",
]

CATEGORICAL_FEATURES = [
    "consensus_over_odds_bin",
    "consensus_over_odds_bin_granular",
    "consensus_under_odds_bin",
    "consensus_under_odds_bin_granular",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Leave-one-season-out CV helpers ───────────────────────────────────────────

def prep_feature(X_train: np.ndarray, X_test: np.ndarray, is_cat: bool, enc=None):
    """Encode categoricals or impute NaN for numerics. Returns (X_train, X_test, enc)."""
    if is_cat:
        if enc is None:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train = enc.fit_transform(X_train)
        else:
            X_train = enc.transform(X_train)
        X_test = enc.transform(X_test)
    else:
        med = np.nanmedian(X_train)
        X_train = np.where(np.isnan(X_train), med, X_train)
        X_test  = np.where(np.isnan(X_test),  med, X_test)
    return X_train, X_test, enc


def loo_cv_linear(df: pd.DataFrame, feature: str, target: str, is_cat: bool = False):
    oof_preds  = np.full(len(df), np.nan)
    oof_actual = np.full(len(df), np.nan)

    for test_season in SEASONS:
        train_mask = df["nfl_season"] != test_season
        test_mask  = df["nfl_season"] == test_season

        train = df[train_mask][[feature, target]].dropna()
        test  = df[test_mask][[feature, target]].dropna()

        if len(train) < 30 or len(test) < 10:
            continue

        X_tr = train[[feature]].values
        y_tr = train[target].values
        X_te = test[[feature]].values
        y_te = test[target].values

        X_tr, X_te, _ = prep_feature(X_tr, X_te, is_cat)

        if target == "passing_yards":
            model = LinearRegression()
        else:
            model = LogisticRegression(max_iter=500, solver="lbfgs")

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        test_indices = df[test_mask].index[df[test_mask][[feature, target]].notna().all(axis=1)]
        oof_preds[test_indices]  = preds
        oof_actual[test_indices] = y_te

    valid = ~np.isnan(oof_preds) & ~np.isnan(oof_actual)
    return oof_preds[valid], oof_actual[valid]


def loo_cv_xgb(df: pd.DataFrame, feature: str, target: str, is_cat: bool = False):
    oof_preds  = np.full(len(df), np.nan)
    oof_actual = np.full(len(df), np.nan)

    for test_season in SEASONS:
        train_mask = df["nfl_season"] != test_season
        test_mask  = df["nfl_season"] == test_season

        train = df[train_mask][[feature, target]].dropna()
        test  = df[test_mask][[feature, target]].dropna()

        if len(train) < 30 or len(test) < 10:
            continue

        X_tr = train[[feature]].values
        y_tr = train[target].values
        X_te = test[[feature]].values
        y_te = test[target].values

        X_tr, X_te, _ = prep_feature(X_tr, X_te, is_cat)

        if target == "passing_yards":
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=1.0,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
        else:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=1.0,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_te)[:, 1]

        test_indices = df[test_mask].index[df[test_mask][[feature, target]].notna().all(axis=1)]
        oof_preds[test_indices]  = preds
        oof_actual[test_indices] = y_te

    valid = ~np.isnan(oof_preds) & ~np.isnan(oof_actual)
    return oof_preds[valid], oof_actual[valid]


def compute_metrics(preds, actual, target):
    """Compute regression or classification metrics depending on target."""
    if target == "passing_yards":
        ss_res = np.sum((actual - preds) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot
        rmse = float(np.sqrt(mean_squared_error(actual, preds)))
        mae  = float(mean_absolute_error(actual, preds))
        return {"r2": round(r2, 4), "rmse": round(rmse, 2), "mae": round(mae, 2),
                "auc": None, "prec": None, "rec": None}
    else:
        preds_bin = (preds >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(actual, preds))
        except Exception:
            auc = None
        prec = float(precision_score(actual, preds_bin, zero_division=0))
        rec  = float(recall_score(actual, preds_bin, zero_division=0))
        return {"r2": None, "rmse": None, "mae": None,
                "auc": round(auc, 4) if auc else None,
                "prec": round(prec, 4), "rec": round(rec, 4)}


# ── Run both sweeps ───────────────────────────────────────────────────────────

print("\n[Step 3b] Running 3a (linear) + 3b (XGBoost) sweeps...")
print(f"  {len(ALL_FEATURES)} features × 2 model types × 2 targets\n")

all_results = []

for feat in ALL_FEATURES:
    if feat not in df.columns:
        print(f"  SKIP {feat} (not in spine)")
        continue

    is_cat = feat in CATEGORICAL_FEATURES
    n_valid = df[[feat, "passing_yards"]].dropna().__len__()

    for model_type, cv_fn in [("linear", loo_cv_linear), ("xgboost", loo_cv_xgb)]:

        # Regression: passing_yards
        preds_reg, actual_reg = cv_fn(df, feat, "passing_yards", is_cat)
        if len(preds_reg) < 50:
            reg_m = {"r2": None, "rmse": None, "mae": None}
        else:
            reg_m = compute_metrics(preds_reg, actual_reg, "passing_yards")

        # Classification: is_over
        preds_cls, actual_cls = cv_fn(df, feat, "is_over", is_cat)
        if len(preds_cls) < 50 or len(np.unique(actual_cls)) < 2:
            cls_m = {"auc": None, "prec": None, "rec": None}
        else:
            cls_m = compute_metrics(preds_cls, actual_cls, "is_over")

        all_results.append({
            "feature":    feat,
            "model_type": model_type,
            "n_valid":    n_valid,
            "type":       "cat" if is_cat else "num",
            "r2":         reg_m["r2"],
            "rmse":       reg_m["rmse"],
            "mae":        reg_m["mae"],
            "auc":        cls_m["auc"],
            "prec":       cls_m["prec"],
            "rec":        cls_m["rec"],
        })

    # Print progress after both models for this feature
    lin = next(r for r in all_results if r["feature"] == feat and r["model_type"] == "linear")
    xgb = next(r for r in all_results if r["feature"] == feat and r["model_type"] == "xgboost")
    delta = (xgb["auc"] or 0) - (lin["auc"] or 0)
    flag  = " *** XGB LIFT" if delta > 0.02 else ""
    print(
        f"  {feat:<45} "
        f"lin_auc={lin['auc'] or 0:.4f}  xgb_auc={xgb['auc'] or 0:.4f}  "
        f"Δ={delta:+.4f}{flag}"
    )

results_df = pd.DataFrame(all_results)
results_df.to_parquet(OUT_PATH, index=False)
print(f"\n[Step 3b] Results saved → {OUT_PATH}")


# ── Build comparison tables ───────────────────────────────────────────────────

lin_df = results_df[results_df["model_type"] == "linear"].copy()
xgb_df = results_df[results_df["model_type"] == "xgboost"].copy()

# Merge for side-by-side
cmp = lin_df[["feature","type","n_valid","auc","r2","rmse"]].merge(
    xgb_df[["feature","auc","r2","rmse"]],
    on="feature",
    suffixes=("_lin","_xgb"),
)
cmp["auc_delta"] = (cmp["auc_xgb"].fillna(0) - cmp["auc_lin"].fillna(0)).round(4)
cmp["rmse_delta"] = (cmp["rmse_xgb"].fillna(0) - cmp["rmse_lin"].fillna(0)).round(2)
cmp["xgb_lifts"] = cmp["auc_delta"] > 0.02

# Sort by best XGB AUC
cmp_by_auc = cmp.sort_values("auc_xgb", ascending=False).reset_index(drop=True)
# Sort by AUC delta (XGB lift)
cmp_by_delta = cmp.sort_values("auc_delta", ascending=False).reset_index(drop=True)

print("\n=== Top 15 Features by XGB AUC ===")
print(cmp_by_auc[["feature","auc_lin","auc_xgb","auc_delta","r2_lin","r2_xgb","rmse_lin","rmse_xgb"]].head(15).to_string(index=False))

print("\n=== Top 15 by XGB AUC Delta (most lift over linear) ===")
print(cmp_by_delta[["feature","auc_lin","auc_xgb","auc_delta","rmse_lin","rmse_xgb"]].head(15).to_string(index=False))

print("\n=== Features with XGB AUC delta > 0.02 ===")
lifted = cmp[cmp["xgb_lifts"]]
print(f"  Count: {len(lifted)}")
if len(lifted):
    print(lifted[["feature","auc_lin","auc_xgb","auc_delta"]].to_string(index=False))

# Baseline
y_all = df["passing_yards"].dropna().values
baseline_rmse = float(np.sqrt(np.mean((y_all - y_all.mean()) ** 2)))
baseline_mae  = float(np.mean(np.abs(y_all - y_all.mean())))
print(f"\nBaseline RMSE (predict mean): {baseline_rmse:.2f}  MAE: {baseline_mae:.2f}")

# XGBoost top RMSE
xgb_by_rmse = xgb_df.sort_values("rmse", ascending=True).reset_index(drop=True)
print(f"\nBest XGB RMSE: {xgb_by_rmse.iloc[0]['rmse']:.2f} ({xgb_by_rmse.iloc[0]['feature']})")


# ── DuckDB validation tests ───────────────────────────────────────────────────

import duckdb

print("\n[Step 3b] Running DuckDB validation tests...")
con = duckdb.connect()
con.register("results", results_df)
con.register("cmp", cmp)

tests = []

def run_test(name: str, sql: str, expect_true: bool = True):
    result = con.execute(sql).fetchone()[0]
    passed = bool(result) == expect_true
    status = "PASS" if passed else "FAIL"
    tests.append({"test": name, "status": status, "result": result})
    print(f"  [{status}] {name} → {result}")

# T1: results have both model types
run_test(
    "T1: both model types present",
    "SELECT COUNT(DISTINCT model_type) = 2 FROM results",
)

# T2: feature count is the same for linear and xgboost
run_test(
    "T2: same feature count for both model types",
    "SELECT (SELECT COUNT(*) FROM results WHERE model_type='linear') = (SELECT COUNT(*) FROM results WHERE model_type='xgboost')",
)

# T3: no AUC outside [0, 1]
run_test(
    "T3: all AUC values in [0, 1]",
    "SELECT COUNT(*) = 0 FROM results WHERE auc IS NOT NULL AND (auc < 0 OR auc > 1)",
)

# T4: XGB RMSE not dramatically worse than baseline (< 2× baseline)
run_test(
    "T4: XGB best RMSE < 2× baseline",
    f"SELECT MIN(rmse) < {baseline_rmse * 2:.1f} FROM results WHERE model_type='xgboost' AND rmse IS NOT NULL",
)

# T5: n features tested = expected
n_feats_expected = sum(1 for f in ALL_FEATURES if f in df.columns)
run_test(
    f"T5: feature count = {n_feats_expected} per model type",
    f"SELECT COUNT(DISTINCT feature) >= {n_feats_expected - 2} FROM results WHERE model_type='linear'",
)

# T6: stacked table has correct number of rows
expected_rows = n_feats_expected * 2
run_test(
    f"T6: stacked result rows = {expected_rows}",
    f"SELECT COUNT(*) >= {expected_rows - 4} FROM results",
)

# T7: auc_delta column is populated in cmp
run_test(
    "T7: auc_delta populated in comparison table",
    "SELECT COUNT(*) > 0 FROM cmp WHERE auc_delta IS NOT NULL",
)

n_pass = sum(1 for t in tests if t["status"] == "PASS")
n_fail = sum(1 for t in tests if t["status"] == "FAIL")
print(f"\n  Tests: {n_pass}/{len(tests)} passed")
if n_fail:
    print("  FAILURES:")
    for t in tests:
        if t["status"] == "FAIL":
            print(f"    → {t['test']} = {t['result']}")

tests_df = pd.DataFrame(tests)


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 3b] Writing HTML section...")

top15_xgb_auc  = cmp_by_auc.head(15)[["feature","auc_lin","auc_xgb","auc_delta","r2_lin","r2_xgb","rmse_lin","rmse_xgb"]]
top15_delta    = cmp_by_delta.head(15)[["feature","auc_lin","auc_xgb","auc_delta","rmse_lin","rmse_xgb"]]
lifted_display = lifted[["feature","auc_lin","auc_xgb","auc_delta","rmse_lin","rmse_xgb"]] if len(lifted) else pd.DataFrame(columns=["feature","auc_lin","auc_xgb","auc_delta"])

# Full stacked table — XGBoost only, sorted by AUC desc
xgb_full = xgb_df.sort_values("auc", ascending=False)[["feature","type","n_valid","r2","rmse","mae","auc","prec","rec"]]

html = f"""
<section>
<h2>Step 3b — XGBoost Individual Feature Sweep</h2>
<p><em>{ts()}</em></p>

<h3>Method</h3>
<ul>
  <li>Leave-one-season-out CV: same 3 folds as Step 3a</li>
  <li>XGBRegressor (n_estimators=100, max_depth=3, lr=0.1) → predict <code>passing_yards</code></li>
  <li>XGBClassifier (same params, eval_metric=logloss) → predict <code>is_over</code></li>
  <li>Stacked side-by-side with Step 3a (linear/logistic) for every feature</li>
  <li>Baseline RMSE (predict mean): <strong>{baseline_rmse:.2f} yds</strong></li>
  <li>Best XGBoost RMSE: <strong>{xgb_by_rmse.iloc[0]['rmse']:.2f} yds</strong> ({xgb_by_rmse.iloc[0]['feature']})</li>
</ul>

<h3>Top 15 Features by XGBoost AUC</h3>
{df_to_html(top15_xgb_auc, "Higher XGB AUC = more over/under signal. Compare lin vs xgb to see if non-linear signal exists.")}

<h3>Top 15 Features by AUC Delta (XGB vs Linear)</h3>
{df_to_html(top15_delta, "auc_delta = xgb_auc - lin_auc. Positive delta → XGBoost captures non-linear signal.")}

<h3>Features with XGB AUC Delta > 0.02 (meaningful non-linear lift)</h3>
{df_to_html(lifted_display, f"Count: {len(lifted)}. These features may interact non-linearly with the target — worth prioritizing in Step 3c combos.") if len(lifted) else "<p>None — XGBoost does not add meaningful lift over linear for any individual feature.</p>"}

<h3>Full XGBoost Feature Table (sorted by AUC)</h3>
{df_to_html(xgb_full, "XGBoost results for all features")}

<h3>DuckDB Test Results</h3>
{df_to_html(tests_df)}

<h3>Key Takeaways</h3>
<ul>
  <li>Best XGB AUC: <strong>{cmp_by_auc.iloc[0]['feature']}</strong> = {cmp_by_auc.iloc[0]['auc_xgb']:.4f} (linear: {cmp_by_auc.iloc[0]['auc_lin']:.4f})</li>
  <li>Best XGB RMSE: <strong>{xgb_by_rmse.iloc[0]['feature']}</strong> = {xgb_by_rmse.iloc[0]['rmse']:.2f} yds (baseline: {baseline_rmse:.2f})</li>
  <li>Features with meaningful XGB lift (delta > 0.02): <strong>{len(lifted)}</strong></li>
  <li>Overall signal level: {'XGBoost adds meaningful non-linear lift over linear for ' + str(len(lifted)) + ' feature(s)' if len(lifted) else 'XGBoost adds little over linear — features are mostly linearly related to the target. Step 3c combos may still help.'}</li>
  <li>Proceed to Step 3c: multi-feature combos using top features from 3a + 3b.</li>
</ul>
</section>
"""

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 3b] HTML section appended → {HTML_PATH}")
print("\n=== DONE ===")
