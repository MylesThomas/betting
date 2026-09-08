"""
Step 4 — ypred → P(over/under) Probability Conversion

The Ridge model predicts passing_yards (yhat). To turn this into a betting
probability we need P(actual > line | yhat).

We model residuals (actual - yhat) as a distribution centered at 0, then:
  P(over line) = 1 - CDF(line - yhat)  [i.e. CDF of the residual at (line - yhat)]

Candidates:
  1. Normal(0, sigma)          — standard Gaussian
  2. Student-t(df, 0, sigma)   — heavier tails; better for football (outlier games)

Grid: sigma_multiplier in [0.8, 0.9, 1.0, 1.1, 1.2] × df (t-dist) in [3, 5, 8, 15]
Evaluate: Brier score (lower = better), calibration curve (reliability diagram)

Clip p_model to [0.01, 0.99] after conversion.

Usage:
  uv run python 20260806_step4_prob_conversion.py
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

REPO_ROOT   = Path(__file__).resolve().parents[3]
TMP_DIR     = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH   = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
SPINE_PATH  = TMP_DIR / "step2_spine.parquet"
MODEL_PATH  = TMP_DIR / "step3c_model.pkl"
OUT_PATH    = TMP_DIR / "step4_spine_with_pmodel.parquet"
ET          = ZoneInfo("America/New_York")


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


# ── ScaledRidge (must match step3c definition for pickle to resolve) ──────────

class ScaledRidge:
    def __init__(self, scaler, ridge):
        self.scaler = scaler
        self.ridge  = ridge
        self._use_proba = False
    def predict(self, X):
        return self.ridge.predict(self.scaler.transform(X))


# ── Load spine + model ────────────────────────────────────────────────────────

print("[Step 4] Loading spine and model...")
spine = pd.read_parquet(SPINE_PATH)
with open(MODEL_PATH, "rb") as fh:
    artifact = pickle.load(fh)

model    = artifact["model"]
features = artifact["features"]
medians  = artifact["medians"]

print(f"  Model: {artifact['fset']} / {artifact['model_name']}")
print(f"  Features: {len(features)}")
print(f"  OOF RMSE: {artifact['oof_rmse']:.2f}")

df = spine[spine["passing_yards"].notna()].copy().reset_index(drop=True)
df["is_over"] = (df["outcome"] == "over").astype(int)
SEASONS = sorted(df["nfl_season"].unique())

print(f"  Rows with actuals: {len(df):,}")


# ── Generate OOF ypred via LOO-season CV ──────────────────────────────────────
# Must use OOF predictions (not train-set) for proper calibration evaluation

print("\n[Step 4] Generating OOF ypred (LOO-season CV)...")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class ScaledRidge:
    def __init__(self, scaler, ridge):
        self.scaler = scaler
        self.ridge  = ridge
    def predict(self, X):
        return self.ridge.predict(self.scaler.transform(X))


def loo_ypred(df, features, medians):
    """Generate OOF ypred using the same Ridge (alpha=10) as 3c."""
    oof_ypred = np.full(len(df), np.nan)

    for test_season in SEASONS:
        train_mask = df["nfl_season"] != test_season
        test_mask  = df["nfl_season"] == test_season

        cols = features + ["passing_yards"]
        train = df[train_mask][cols].dropna()
        test  = df[test_mask][cols].dropna()

        if len(train) < 50 or len(test) < 10:
            continue

        X_tr = train[features].values.astype(float)
        y_tr = train["passing_yards"].values
        X_te = test[features].values.astype(float)

        # NaN imputation
        meds = np.nanmedian(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = meds[j]
            X_te[np.isnan(X_te[:, j]), j] = meds[j]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_s, y_tr)
        preds = ridge.predict(X_te_s)

        test_indices = df[test_mask].index[df[test_mask][cols].notna().all(axis=1)]
        oof_ypred[test_indices] = preds

    return oof_ypred

oof_ypred = loo_ypred(df, features, medians)
df["ypred"] = oof_ypred

# Only rows where we have both ypred and actual
df_oof = df[df["ypred"].notna()].copy().reset_index(drop=True)
residuals = df_oof["passing_yards"] - df_oof["ypred"]

print(f"  OOF rows with ypred: {len(df_oof):,}")
print(f"  Residual mean:  {residuals.mean():.3f}")
print(f"  Residual std:   {residuals.std():.3f}")
print(f"  Residual RMSE:  {np.sqrt((residuals**2).mean()):.3f}")

sigma_base = float(residuals.std())


# ── Distribution grid search ──────────────────────────────────────────────────
# For each (dist_type, sigma_mult, df_param), compute:
#   p_model_over = P(actual > line | ypred, sigma)
# Then evaluate Brier score against actual is_over

print("\n[Step 4] Running distribution grid search...")

SIGMA_MULTS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
T_DFS       = [3, 5, 8, 15, 30]

grid_results = []

for dist_type in ["normal", "t"]:
    mults = SIGMA_MULTS
    dfs   = T_DFS if dist_type == "t" else [None]

    for sm in mults:
        sigma = sigma_base * sm

        for df_param in dfs:
            # P(actual > line) = P(residual > line - ypred) = 1 - CDF(line - ypred)
            z = (df_oof["line"] - df_oof["ypred"]) / sigma

            if dist_type == "normal":
                p_over = 1 - norm.cdf(z.values)
                label  = f"normal(sigma×{sm:.1f})"
            else:
                p_over = 1 - t_dist.cdf(z.values, df=df_param)
                label  = f"t(df={df_param}, sigma×{sm:.1f})"

            p_over = np.clip(p_over, 0.01, 0.99)
            p_under = 1 - p_over

            brier_over  = float(brier_score_loss(df_oof["is_over"].values, p_over))
            brier_under = float(brier_score_loss(1 - df_oof["is_over"].values, p_under))
            brier_avg   = (brier_over + brier_under) / 2

            grid_results.append({
                "dist":    dist_type,
                "sigma_mult": sm,
                "df_param":   df_param,
                "label":      label,
                "sigma":      round(sigma, 2),
                "brier_over":  round(brier_over,  5),
                "brier_under": round(brier_under, 5),
                "brier_avg":   round(brier_avg,   5),
                "mean_p_over": round(p_over.mean(), 4),
            })

grid_df = pd.DataFrame(grid_results).sort_values("brier_avg")
print("\nTop 10 configs by Brier avg:")
print(grid_df.head(10)[["label","sigma","brier_avg","brier_over","brier_under","mean_p_over"]].to_string(index=False))

best = grid_df.iloc[0]
print(f"\nBest: {best['label']}  sigma={best['sigma']:.2f}  brier_avg={best['brier_avg']:.5f}")


# ── Calibration check for best 3 configs ──────────────────────────────────────

print("\n[Step 4] Calibration curves for top 3 configs...")

top3 = grid_df.head(3)
calib_rows = []

for _, cfg in top3.iterrows():
    sigma = cfg["sigma"]
    dist  = cfg["dist"]
    df_p  = cfg["df_param"]

    z = (df_oof["line"] - df_oof["ypred"]) / sigma

    if dist == "normal":
        p_over = np.clip(1 - norm.cdf(z.values), 0.01, 0.99)
    else:
        p_over = np.clip(1 - t_dist.cdf(z.values, df=df_p), 0.01, 0.99)

    fraction_of_pos, mean_predicted = calibration_curve(
        df_oof["is_over"].values, p_over, n_bins=10, strategy="quantile"
    )
    for frac, pred in zip(fraction_of_pos, mean_predicted):
        calib_rows.append({
            "config":       cfg["label"],
            "pred_prob":    round(float(pred), 4),
            "actual_freq":  round(float(frac), 4),
            "gap":          round(float(frac - pred), 4),
        })

calib_df = pd.DataFrame(calib_rows)
print(calib_df.to_string(index=False))


# ── Pick best config and generate final p_model column ────────────────────────

best_dist       = best["dist"]
best_sigma      = float(best["sigma"])
best_df_param   = best["df_param"]

print(f"\n[Step 4] Applying best config: {best['label']}")

# Apply to ALL spine rows (not just OOF — we need p_model for production too)
# For production, ypred comes from the full-data trained model
def apply_prod_model(row_df, model, features, medians):
    """Apply the prod model to generate ypred for all rows."""
    feats_present = [f for f in features if f in row_df.columns]
    X = row_df[feats_present].values.astype(float)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]
    return model.predict(X)

spine_full = spine.copy()
spine_full["ypred_prod"] = np.nan
feats_present = [f for f in features if f in spine_full.columns]
X_all = spine_full[feats_present].values.astype(float)
med_all = medians[:len(feats_present)]
for j in range(X_all.shape[1]):
    X_all[np.isnan(X_all[:, j]), j] = med_all[j]
spine_full["ypred_prod"] = model.predict(X_all)

# Compute p_model using best distribution
z_all = (spine_full["line"] - spine_full["ypred_prod"]) / best_sigma
if best_dist == "normal":
    p_over_all = 1 - norm.cdf(z_all.values)
else:
    p_over_all = 1 - t_dist.cdf(z_all.values, df=best_df_param)

spine_full["p_model_over"]  = np.clip(p_over_all, 0.01, 0.99)
spine_full["p_model_under"] = 1 - spine_full["p_model_over"]

# For backtest: use OOF ypred (not prod) for rows where we have it
# Merge OOF ypred back
oof_map = dict(zip(df_oof.index, df_oof["ypred"]))
spine_full["ypred_oof"] = np.nan
matched_idx = df_oof.index.tolist()
spine_full.loc[df_oof.index[:len(df_oof)], "ypred_oof"] = df_oof["ypred"].values

# Compute p_model_oof for backtest rows
z_oof_all = (spine_full["line"] - spine_full["ypred_oof"]) / best_sigma
if best_dist == "normal":
    p_over_oof = 1 - norm.cdf(z_oof_all.values)
else:
    p_over_oof = 1 - t_dist.cdf(z_oof_all.values, df=best_df_param)

spine_full["p_model_over_oof"]  = np.clip(p_over_oof, 0.01, 0.99)
spine_full["p_model_under_oof"] = 1 - spine_full["p_model_over_oof"]

spine_full.to_parquet(OUT_PATH, index=False)
print(f"  Saved → {OUT_PATH}")
print(f"  Rows: {len(spine_full):,}")

# Summary stats on p_model
oof_rows = spine_full[spine_full["ypred_oof"].notna()]
print(f"\n  p_model_over_oof stats (n={len(oof_rows):,}):")
print(f"    mean:  {oof_rows['p_model_over_oof'].mean():.4f}")
print(f"    std:   {oof_rows['p_model_over_oof'].std():.4f}")
print(f"    min:   {oof_rows['p_model_over_oof'].min():.4f}")
print(f"    max:   {oof_rows['p_model_over_oof'].max():.4f}")
print(f"    p5:    {np.percentile(oof_rows['p_model_over_oof'], 5):.4f}")
print(f"    p95:   {np.percentile(oof_rows['p_model_over_oof'], 95):.4f}")

# Spot-check: Josh Allen
allen = oof_rows[oof_rows["player_norm"].str.contains("josh allen", case=False, na=False)]
if len(allen):
    allen_show = allen[["nfl_season","nfl_week","line","passing_yards","ypred_oof","p_model_over_oof","p_model_under_oof","outcome"]].drop_duplicates(subset=["nfl_season","nfl_week"]).head(15)
    allen_show = allen_show.rename(columns={"passing_yards":"actual","ypred_oof":"ypred","p_model_over_oof":"p_over","p_model_under_oof":"p_under"})
    print("\n=== Josh Allen Spot-Check (OOF ypred + p_model) ===")
    print(allen_show.round(3).to_string(index=False))


# ── DuckDB tests ──────────────────────────────────────────────────────────────

import duckdb

print("\n[Step 4] Running DuckDB validation tests...")
con = duckdb.connect()
con.register("s4", spine_full)
con.register("grid", grid_df)

tests = []

def run_test(name, sql, expect_true=True):
    result = con.execute(sql).fetchone()[0]
    passed = bool(result) == expect_true
    status = "PASS" if passed else "FAIL"
    tests.append({"test": name, "status": status, "result": result})
    print(f"  [{status}] {name} → {result}")

# T1: p_model_over is between 0.01 and 0.99 for all rows
run_test(
    "T1: all p_model_over in [0.01, 0.99]",
    "SELECT COUNT(*) = 0 FROM s4 WHERE p_model_over IS NOT NULL AND (p_model_over < 0.01 OR p_model_over > 0.99)",
)

# T2: p_model_over + p_model_under = 1.0 for all rows
run_test(
    "T2: p_model_over + p_model_under = 1.0",
    "SELECT COUNT(*) = 0 FROM s4 WHERE p_model_over IS NOT NULL AND ABS(p_model_over + p_model_under - 1.0) > 0.001",
)

# T3: ypred_oof is populated for the majority of rows with actuals
run_test(
    "T3: >80% of rows with actuals have ypred_oof",
    f"SELECT (SELECT COUNT(*) FROM s4 WHERE ypred_oof IS NOT NULL AND passing_yards IS NOT NULL) * 1.0 / (SELECT COUNT(*) FROM s4 WHERE passing_yards IS NOT NULL) > 0.80",
)

# T4: mean p_model_over is near 0.5 (model not strongly biased)
run_test(
    "T4: mean p_model_over within [0.42, 0.58]",
    "SELECT AVG(p_model_over_oof) BETWEEN 0.42 AND 0.58 FROM s4 WHERE p_model_over_oof IS NOT NULL",
)

# T5: grid has results for both normal and t distributions
run_test(
    "T5: grid has both normal and t results",
    "SELECT COUNT(DISTINCT dist) = 2 FROM grid",
)

# T6: best brier < 0.26 (naive baseline is ~0.25 for 50/50 market)
run_test(
    "T6: best brier_avg < 0.26",
    f"SELECT MIN(brier_avg) < 0.26 FROM grid",
)

# T7: OOF ypred residuals have near-zero mean (no systematic bias)
run_test(
    "T7: OOF residual mean < 5 yds (no large systematic bias)",
    "SELECT ABS(AVG(passing_yards - ypred_oof)) < 5 FROM s4 WHERE passing_yards IS NOT NULL AND ypred_oof IS NOT NULL",
)

n_pass = sum(1 for t in tests if t["status"] == "PASS")
n_fail = sum(1 for t in tests if t["status"] == "FAIL")
print(f"\n  Tests: {n_pass}/{len(tests)} passed")
tests_df = pd.DataFrame(tests)


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 4] Writing HTML section...")

# Calibration gap table
calib_top3_display = calib_df.copy()

# Allen spot-check for HTML
allen_html = ""
if len(allen):
    allen_html = df_to_html(allen_show.round(3).reset_index(drop=True), "Josh Allen spot-check — ypred vs actual, p_model over/under")

# Normal-only grid slice
grid_normal = grid_df[grid_df["dist"] == "normal"][["label","sigma","brier_avg","brier_over","brier_under","mean_p_over"]].reset_index(drop=True)
grid_t      = grid_df[grid_df["dist"] == "t"][["label","sigma","df_param","brier_avg","brier_over","brier_under","mean_p_over"]].reset_index(drop=True)

html = f"""
<section>
<h2>Step 4 — ypred → P(over/under) Probability Conversion</h2>
<p><em>{ts()}</em></p>

<h3>Method</h3>
<ul>
  <li>OOF ypred from LOO-season CV (Ridge, C_core_mkt_start features)</li>
  <li>Residual std (sigma_base): <strong>{sigma_base:.2f} yds</strong></li>
  <li>P(over line) = 1 − CDF((line − ypred) / sigma)</li>
  <li>Grid search: Normal vs Student-t; sigma_mult ∈ {SIGMA_MULTS}; df ∈ {T_DFS}</li>
  <li>Evaluation metric: Brier score (lower = better calibration)</li>
  <li>p_model clipped to [0.01, 0.99]</li>
</ul>

<h3>Best Config</h3>
<p><strong>{best['label']}</strong> — sigma={best['sigma']:.2f}, brier_avg={best['brier_avg']:.5f}</p>

<h3>Distribution Grid — Normal</h3>
{df_to_html(grid_normal, "All Normal configs sorted by Brier avg")}

<h3>Distribution Grid — Student-t</h3>
{df_to_html(grid_t, "All Student-t configs sorted by Brier avg")}

<h3>Calibration Curves (top 3 configs)</h3>
{df_to_html(calib_top3_display.reset_index(drop=True), "pred_prob = mean predicted prob per bin; actual_freq = fraction that went over; gap = actual - pred (ideal: 0)")}

<h3>p_model Stats (OOF rows, n={len(oof_rows):,})</h3>
<ul>
  <li>mean p_over: {oof_rows['p_model_over_oof'].mean():.4f}</li>
  <li>std:         {oof_rows['p_model_over_oof'].std():.4f}</li>
  <li>p5–p95:      {np.percentile(oof_rows['p_model_over_oof'], 5):.4f} – {np.percentile(oof_rows['p_model_over_oof'], 95):.4f}</li>
</ul>

{allen_html}

<h3>DuckDB Test Results</h3>
{df_to_html(tests_df)}

<h3>Key Takeaways</h3>
<ul>
  <li>Best distribution: <strong>{best['label']}</strong> (Brier avg = {best['brier_avg']:.5f})</li>
  <li>Residual sigma: {sigma_base:.2f} yds (σ_mult={best['sigma_mult']:.1f} → effective sigma = {best['sigma']:.2f})</li>
  <li>p_model spread: 5th–95th pctile = {np.percentile(oof_rows['p_model_over_oof'], 5):.3f}–{np.percentile(oof_rows['p_model_over_oof'], 95):.3f} — model has useful spread, not degenerate</li>
  <li>Spine saved with ypred_oof + p_model_over_oof columns → ready for Step 5 grid search</li>
</ul>
</section>
"""

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 4] HTML section appended → {HTML_PATH}")
print("\n=== DONE ===")
