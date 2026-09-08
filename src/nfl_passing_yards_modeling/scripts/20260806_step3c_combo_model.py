"""
Step 3c — Multi-Feature Combo Models

Builds multi-feature regression and classification models using the top predictors
from Steps 3a + 3b. Runs leave-one-season-out CV. Saves the best model artifact.

Targets:
  - passing_yards (regression): predict raw yardage → later converted to P(over)
  - is_over (classification): direct over/under prediction (diagnostic only)

Models:
  1. OLS (LinearRegression) — interpretable baseline
  2. Ridge regression — handles correlated rolling features
  3. XGBoost regressor — captures non-linear interactions
  4. XGBoost classifier — for direct is_over comparison

Feature sets:
  A. Core rolling (best from 3a/3b)
  B. Core + market features (consensus_line, implied_team_total, n_books)
  C. Core + market + starter flags
  D. Full set (all top features)

Saves best model + config block to:
  ~/Downloads/tmp/pass_yds/step3c_model.pkl
  ~/Downloads/tmp/pass_yds/step3c_results.parquet

Usage:
  uv run python 20260806_step3c_combo_model.py
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
TMP_DIR    = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH  = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
SPINE_PATH = TMP_DIR / "step2_spine.parquet"
MODEL_PATH = TMP_DIR / "step3c_model.pkl"
OUT_PATH   = TMP_DIR / "step3c_results.parquet"
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

print("[Step 3c] Loading spine...")
spine = pd.read_parquet(SPINE_PATH)

df = spine[spine["passing_yards"].notna()].copy().reset_index(drop=True)
df["is_over"] = (df["outcome"] == "over").astype(int)

print(f"  Rows with actuals: {len(df):,}")
SEASONS = sorted(df["nfl_season"].unique())

y_all = df["passing_yards"].dropna().values
BASELINE_RMSE = float(np.sqrt(np.mean((y_all - y_all.mean()) ** 2)))
BASELINE_MAE  = float(np.mean(np.abs(y_all - y_all.mean())))
print(f"  Baseline RMSE: {BASELINE_RMSE:.2f}  MAE: {BASELINE_MAE:.2f}")

# ── Feature sets ──────────────────────────────────────────────────────────────

# Best from 3a (linear): cover_rate_roll_1, pass_yds_roll_3/5, pass_yds_roll_season
# Best from 3b (XGB):    attempts_roll_career/3, comp_pct_roll_3, pass_rate_roll_3/10
# Starter flags (XGB lift): starts_last_4, games_as_starter_this_season
# Market: consensus_line (best RMSE), implied_team_total, team_spread, n_books
# Opponent: opp_pass_yds_allowed_roll5
# Context: nfl_week

CORE_ROLLING = [
    "pass_yds_roll_3",
    "pass_yds_roll_5",
    "pass_yds_roll_season",
    "cover_rate_roll_1",
    "attempts_roll_career",
    "attempts_roll_3",
    "comp_pct_roll_3",
    "ypa_roll_career",
    "pass_rate_roll_3",
    "epa_per_att_roll_10",
    "opp_pass_yds_allowed_roll5",
]

MARKET = [
    "consensus_line",
    "implied_team_total",
    "team_spread",
    "n_books",
]

STARTER = [
    "starts_last_4",
    "games_as_starter_this_season",
    "career_starts_pct",
]

CONTEXT = [
    "nfl_week",
]

FEATURE_SETS = {
    "A_core":          CORE_ROLLING,
    "B_core_market":   CORE_ROLLING + MARKET,
    "C_core_mkt_start": CORE_ROLLING + MARKET + STARTER,
    "D_full":          CORE_ROLLING + MARKET + STARTER + CONTEXT,
}

# ── LOO-season CV helper ───────────────────────────────────────────────────────

def loo_cv(df, features, target, model_fn):
    """
    Returns oof_preds array (len = df with actuals) and oof_actual.
    model_fn takes (X_train, y_train) and returns a fitted model with .predict().
    For XGBClassifier, caller overrides to use predict_proba.
    """
    oof_preds  = np.full(len(df), np.nan)
    oof_actual = np.full(len(df), np.nan)
    oof_season = np.full(len(df), np.nan)

    for test_season in SEASONS:
        train_mask = df["nfl_season"] != test_season
        test_mask  = df["nfl_season"] == test_season

        # Drop rows missing any feature or target
        cols = features + [target]
        train = df[train_mask][cols].dropna()
        test  = df[test_mask][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            continue

        X_tr = train[features].values.astype(float)
        y_tr = train[target].values
        X_te = test[features].values.astype(float)
        y_te = test[target].values

        # NaN imputation: median from train set
        medians = np.nanmedian(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = medians[j]
            X_te[np.isnan(X_te[:, j]), j] = medians[j]

        model = model_fn(X_tr, y_tr)
        preds = model.predict(X_te)
        # For classifiers that need predict_proba, model_fn wraps it
        if hasattr(model, "_use_proba") and model._use_proba:
            preds = model.predict_proba(X_te)[:, 1]

        test_indices = df[test_mask].index[df[test_mask][cols].notna().all(axis=1)]
        oof_preds[test_indices]  = preds
        oof_actual[test_indices] = y_te
        oof_season[test_indices] = test_season

    valid = ~np.isnan(oof_preds) & ~np.isnan(oof_actual)
    return oof_preds[valid], oof_actual[valid], oof_season[valid]


# ── Model factories ───────────────────────────────────────────────────────────

def make_ols(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m

class ScaledRidge:
    def __init__(self, scaler, ridge):
        self.scaler = scaler
        self.ridge  = ridge
        self._use_proba = False
    def predict(self, X):
        return self.ridge.predict(self.scaler.transform(X))


def make_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    m = Ridge(alpha=10.0)
    m.fit(Xs, y)
    return ScaledRidge(scaler, m)

def make_xgb_reg(X, y):
    m = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
    )
    m.fit(X, y)
    return m

def make_xgb_cls(X, y):
    m = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    m.fit(X, y)
    m._use_proba = True
    return m


# ── Metric helpers ────────────────────────────────────────────────────────────

def reg_metrics(preds, actual):
    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot)
    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    mae  = float(mean_absolute_error(actual, preds))
    return round(r2, 4), round(rmse, 2), round(mae, 2)

def cls_metrics(preds, actual):
    preds_bin = (preds >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(actual, preds))
    except Exception:
        auc = np.nan
    prec = float(precision_score(actual, preds_bin, zero_division=0))
    rec  = float(recall_score(actual, preds_bin, zero_division=0))
    return round(auc, 4), round(prec, 4), round(rec, 4)


# ── Run all combos ────────────────────────────────────────────────────────────

print("\n[Step 3c] Running combo models...")

MODELS_REG = {"ols": make_ols, "ridge": make_ridge, "xgb_reg": make_xgb_reg}
MODELS_CLS = {"xgb_cls": make_xgb_cls}

results = []
oof_store = {}  # Store OOF preds for best model

for fset_name, features in FEATURE_SETS.items():
    # Filter to features that exist
    feats = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  Warning: {fset_name} missing features: {missing}")

    n_rows = df[feats + ["passing_yards"]].dropna().__len__()
    print(f"\n  Feature set {fset_name}: {len(feats)} features, {n_rows} complete rows")

    # Regression models
    for model_name, model_fn in MODELS_REG.items():
        print(f"    {model_name} (passing_yards)...", end="", flush=True)
        preds, actual, seasons = loo_cv(df, feats, "passing_yards", model_fn)
        if len(preds) < 50:
            print(" [skip — too few rows]")
            continue
        r2, rmse, mae = reg_metrics(preds, actual)
        print(f" r2={r2:+.4f}  rmse={rmse:.2f}  mae={mae:.2f}  n={len(preds)}")

        results.append({
            "fset": fset_name, "model": model_name, "target": "passing_yards",
            "n": len(preds), "r2": r2, "rmse": rmse, "mae": mae,
            "auc": None, "prec": None, "rec": None,
        })
        oof_store[(fset_name, model_name)] = {"preds": preds, "actual": actual, "seasons": seasons, "features": feats}

    # Classifier
    for model_name, model_fn in MODELS_CLS.items():
        print(f"    {model_name} (is_over)...", end="", flush=True)
        preds, actual, seasons = loo_cv(df, feats, "is_over", model_fn)
        if len(preds) < 50:
            print(" [skip — too few rows]")
            continue
        auc, prec, rec = cls_metrics(preds, actual)
        print(f" auc={auc:.4f}  prec={prec:.4f}  rec={rec:.4f}  n={len(preds)}")

        results.append({
            "fset": fset_name, "model": model_name, "target": "is_over",
            "n": len(preds), "r2": None, "rmse": None, "mae": None,
            "auc": auc, "prec": prec, "rec": rec,
        })

results_df = pd.DataFrame(results)
results_df.to_parquet(OUT_PATH, index=False)
print(f"\n[Step 3c] Results saved → {OUT_PATH}")


# ── Summary tables ────────────────────────────────────────────────────────────

reg_results = results_df[results_df["target"] == "passing_yards"].sort_values("rmse")
cls_results = results_df[results_df["target"] == "is_over"].sort_values("auc", ascending=False)

print("\n=== Regression Results (sorted by RMSE) ===")
print(reg_results[["fset","model","n","r2","rmse","mae"]].to_string(index=False))

print("\n=== Classification Results (sorted by AUC) ===")
print(cls_results[["fset","model","n","auc","prec","rec"]].to_string(index=False))

# Best regression model
best_reg = reg_results.iloc[0]
print(f"\nBest regression: {best_reg['fset']} / {best_reg['model']} — RMSE={best_reg['rmse']:.2f}  R²={best_reg['r2']:.4f}")
print(f"Baseline RMSE: {BASELINE_RMSE:.2f}  →  Improvement: {BASELINE_RMSE - best_reg['rmse']:.2f} yds")

best_cls = cls_results.iloc[0]
print(f"Best classifier: {best_cls['fset']} / {best_cls['model']} — AUC={best_cls['auc']:.4f}")

# ── Train + save best model on ALL data ──────────────────────────────────────

# Best model for production = best regression (XGB or Ridge on D_full or C set)
# We want passing_yards prediction so we can convert to prob in Step 4
best_key_candidates = [
    ("D_full", "xgb_reg"),
    ("C_core_mkt_start", "xgb_reg"),
    ("D_full", "ridge"),
    ("C_core_mkt_start", "ridge"),
]

best_prod_key = None
best_prod_rmse = float("inf")
for key in best_key_candidates:
    row = results_df[(results_df["fset"] == key[0]) & (results_df["model"] == key[1]) & (results_df["target"] == "passing_yards")]
    if len(row) and not pd.isna(row.iloc[0]["rmse"]):
        r = float(row.iloc[0]["rmse"])
        if r < best_prod_rmse:
            best_prod_rmse = r
            best_prod_key = key

if best_prod_key is None:
    # Fallback to overall best regression
    best_prod_key = (best_reg["fset"], best_reg["model"])

fset_name, model_name = best_prod_key
prod_features = [f for f in FEATURE_SETS[fset_name] if f in df.columns]
print(f"\n[Step 3c] Saving prod model: {fset_name} / {model_name} ({len(prod_features)} features)")

# Train on ALL available data
cols = prod_features + ["passing_yards"]
train_all = df[cols].dropna()
X_all = train_all[prod_features].values.astype(float)
y_all_reg = train_all["passing_yards"].values

medians_all = np.nanmedian(X_all, axis=0)
for j in range(X_all.shape[1]):
    X_all[np.isnan(X_all[:, j]), j] = medians_all[j]

fn = {"ols": make_ols, "ridge": make_ridge, "xgb_reg": make_xgb_reg}[model_name]
prod_model = fn(X_all, y_all_reg)

model_artifact = {
    "model":        prod_model,
    "features":     prod_features,
    "medians":      medians_all,
    "fset":         fset_name,
    "model_name":   model_name,
    "oof_rmse":     best_prod_rmse,
    "baseline_rmse": BASELINE_RMSE,
}
with open(MODEL_PATH, "wb") as fh:
    pickle.dump(model_artifact, fh)
print(f"  Saved → {MODEL_PATH}")

# ── Per-season breakdown for best reg model ────────────────────────────────────

oof_key = (fset_name, model_name)
if oof_key in oof_store:
    oof = oof_store[oof_key]
    seasonal_rows = []
    for s in SEASONS:
        mask = oof["seasons"] == s
        if mask.sum() < 10:
            continue
        r2s, rmsep, maes = reg_metrics(oof["preds"][mask], oof["actual"][mask])
        seasonal_rows.append({"season": int(s), "n": mask.sum(), "r2": r2s, "rmse": rmsep, "mae": maes})
    seasonal_df = pd.DataFrame(seasonal_rows)
    print("\n=== Per-season breakdown (best reg model) ===")
    print(seasonal_df.to_string(index=False))
else:
    seasonal_df = pd.DataFrame()

# ── Spot-check: Josh Allen prediction trace ────────────────────────────────────

allen_df = pd.DataFrame()
allen = df[df["player_norm"].str.contains("josh allen", case=False, na=False)].copy()
if len(allen):
    # Re-run OOF with OLS for display (fast, consistent)
    ols_preds, ols_actual, ols_seasons = loo_cv(df, prod_features, "passing_yards", make_ols)
    allen_rows = []
    seen = set()
    for _, row in allen.iterrows():
        orig_idx = row.name
        key = (int(row["nfl_season"]), int(row["nfl_week"]) if not pd.isna(row["nfl_week"]) else -1)
        if key in seen:
            continue
        seen.add(key)
        pred = round(float(ols_preds[orig_idx]), 1) if orig_idx < len(ols_preds) and not np.isnan(ols_preds[orig_idx]) else None
        allen_rows.append({
            "season": int(row["nfl_season"]),
            "week":   key[1],
            "line":   row["line"],
            "actual": row["passing_yards"],
            "ols_pred": pred,
            "outcome": row["outcome"],
        })
    allen_df = pd.DataFrame(allen_rows).head(20)
    print("\n=== Josh Allen OOF Trace (OLS, D_full features) ===")
    print(allen_df.to_string(index=False))


# ── Config YAML block ──────────────────────────────────────────────────────────

config_block = f"""# nfl_passing_yards model config (Step 3c)
model:
  fset:           {fset_name}
  model_type:     {model_name}
  features:
{chr(10).join('    - ' + f for f in prod_features)}
  oof_rmse:       {best_prod_rmse:.2f}
  baseline_rmse:  {BASELINE_RMSE:.2f}
  rmse_improvement: {BASELINE_RMSE - best_prod_rmse:.2f}
"""
print("\n=== Config YAML block ===")
print(config_block)


# ── DuckDB tests ──────────────────────────────────────────────────────────────

import duckdb

print("\n[Step 3c] Running DuckDB validation tests...")
con = duckdb.connect()
con.register("results", results_df)

tests = []

def run_test(name, sql, expect_true=True):
    result = con.execute(sql).fetchone()[0]
    passed = bool(result) == expect_true
    status = "PASS" if passed else "FAIL"
    tests.append({"test": name, "status": status, "result": result})
    print(f"  [{status}] {name} → {result}")

# T1: all 4 feature sets present
run_test(
    "T1: 4 feature sets present",
    f"SELECT COUNT(DISTINCT fset) = 4 FROM results",
)

# T2: best RMSE < baseline
run_test(
    "T2: best combo RMSE < baseline",
    f"SELECT MIN(rmse) < {BASELINE_RMSE:.2f} FROM results WHERE target='passing_yards' AND rmse IS NOT NULL",
)

# T3: best AUC for combo > 0.50
run_test(
    "T3: best combo AUC > 0.50",
    "SELECT MAX(auc) > 0.50 FROM results WHERE target='is_over' AND auc IS NOT NULL",
)

# T4: no null n_rows for regression results
run_test(
    "T4: all regression rows have n > 0",
    "SELECT COUNT(*) = 0 FROM results WHERE target='passing_yards' AND (n IS NULL OR n = 0)",
)

# T5: xgb_reg present in results
run_test(
    "T5: xgb_reg model present",
    "SELECT COUNT(*) > 0 FROM results WHERE model='xgb_reg'",
)

# T6: prod model was saved
run_test(
    "T6: model artifact file exists",
    f"SELECT {MODEL_PATH.exists()}",
)

n_pass = sum(1 for t in tests if t["status"] == "PASS")
n_fail = sum(1 for t in tests if t["status"] == "FAIL")
print(f"\n  Tests: {n_pass}/{len(tests)} passed")
tests_df = pd.DataFrame(tests)


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 3c] Writing HTML section...")

html = f"""
<section>
<h2>Step 3c — Multi-Feature Combo Models</h2>
<p><em>{ts()}</em></p>

<h3>Feature Sets</h3>
<ul>
  <li><strong>A_core</strong>: {len(FEATURE_SETS['A_core'])} rolling features (pass_yds, attempts, comp_pct, ypa, cover_rate, pass_rate, EPA, opp_defense)</li>
  <li><strong>B_core_market</strong>: A + consensus_line, implied_team_total, team_spread, n_books</li>
  <li><strong>C_core_mkt_start</strong>: B + starts_last_4, games_as_starter_this_season, career_starts_pct</li>
  <li><strong>D_full</strong>: C + nfl_week</li>
</ul>

<h3>Regression Results — Predict Passing Yards (sorted by RMSE)</h3>
{df_to_html(reg_results[['fset','model','n','r2','rmse','mae']].reset_index(drop=True), f"Baseline RMSE = {BASELINE_RMSE:.2f} yds")}

<h3>Classification Results — Predict is_over (sorted by AUC)</h3>
{df_to_html(cls_results[['fset','model','n','auc','prec','rec']].reset_index(drop=True), "")}

<h3>Per-Season Breakdown (Best Regression Model: {fset_name} / {model_name})</h3>
{df_to_html(seasonal_df.reset_index(drop=True)) if len(seasonal_df) else "<p>N/A</p>"}

{f'<h3>Josh Allen OOF Trace (D_full / OLS)</h3>{df_to_html(allen_df.reset_index(drop=True))}' if len(allen_df) else ''}

<h3>Model Config Block</h3>
<pre style="background:#1a1a2e;color:#e0e0e0;padding:12px;border-radius:4px;font-size:12px">{config_block}</pre>

<h3>DuckDB Test Results</h3>
{df_to_html(tests_df)}

<h3>Key Takeaways</h3>
<ul>
  <li>Best regression: <strong>{best_reg['fset']} / {best_reg['model']}</strong> — RMSE={best_reg['rmse']:.2f} yds (baseline: {BASELINE_RMSE:.2f}, improvement: {BASELINE_RMSE - best_reg['rmse']:.2f} yds)</li>
  <li>Best classifier: <strong>{best_cls['fset']} / {best_cls['model']}</strong> — AUC={best_cls['auc']:.4f}</li>
  <li>Production model saved: <code>{MODEL_PATH}</code></li>
  <li>Next: Step 4 — convert ypred (passing_yards) to P(over/under) using Normal or Student-t distribution. Evaluate Brier score + calibration curves.</li>
</ul>
</section>
"""

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 3c] HTML section appended → {HTML_PATH}")
print("\n=== DONE ===")
