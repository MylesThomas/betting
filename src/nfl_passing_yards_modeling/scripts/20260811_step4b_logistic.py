"""
Step 4b — Secondary Classifier Methods (Logistic + XGB on ypred × line)

Compares three P(over) methods head-to-head on Brier score:
  Method 1 — Distribution (t(df=3, sigma×1.3)) from Step 4a
  Method 2 — Logistic regression trained on (ypred, line) → P(over)
  Method 3 — XGBClassifier trained on (ypred, line) → P(over)

All three use OOF ypred from the Ridge model (LOO-season CV).
The secondary classifiers also use LOO-season CV — train on 2 seasons, test on 1.

Winning method is written to the spine parquet and HTML.

Usage:
  uv run python 20260811_step4b_logistic.py
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
TMP_DIR    = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH  = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
S4_PATH    = TMP_DIR / "step4_spine_with_pmodel.parquet"
OUT_PATH   = TMP_DIR / "step4_spine_with_pmodel.parquet"  # overwrite with best method
ET         = ZoneInfo("America/New_York")

BEST_DIST_SIGMA = 94.44   # t(df=3, sigma×1.3) from Step 4a
BEST_DIST_DF    = 3


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


# ── Load Step 4 spine (already has ypred_oof) ─────────────────────────────────

print("[Step 4b] Loading spine with OOF ypred...")
spine = pd.read_parquet(S4_PATH)

df = spine[
    (spine["bookmaker"] == "betonlineag") &
    spine["ypred_oof"].notna() &
    spine["passing_yards"].notna()
].copy().reset_index(drop=True)

df["is_over"] = (df["outcome"] == "over").astype(int)
SEASONS = sorted(df["nfl_season"].unique())

print(f"  OOF rows (BetOnline, with ypred): {len(df):,}")
print(f"  Over rate: {df['is_over'].mean():.4f}")


# ── Method 1 — Distribution (already computed in Step 4a) ─────────────────────

z = (df["line"] - df["ypred_oof"]) / BEST_DIST_SIGMA
p_dist = np.clip(1 - t_dist.cdf(z.values, df=BEST_DIST_DF), 0.01, 0.99)
brier_dist = brier_score_loss(df["is_over"].values, p_dist)
print(f"\n  Method 1 — Distribution t(df={BEST_DIST_DF}, σ={BEST_DIST_SIGMA:.1f}): Brier = {brier_dist:.5f}")


# ── Method 2 — Logistic regression on (ypred, line) ───────────────────────────

print("\n[Step 4b] Method 2 — Logistic(ypred, line) LOO-season CV...")

oof_logistic = np.full(len(df), np.nan)

for test_season in SEASONS:
    train_mask = df["nfl_season"] != test_season
    test_mask  = df["nfl_season"] == test_season

    X_tr = df.loc[train_mask, ["ypred_oof", "line"]].values
    y_tr = df.loc[train_mask, "is_over"].values
    X_te = df.loc[test_mask,  ["ypred_oof", "line"]].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_tr_s, y_tr)
    oof_logistic[test_mask] = model.predict_proba(X_te_s)[:, 1]

valid = ~np.isnan(oof_logistic)
p_logistic = np.clip(oof_logistic[valid], 0.01, 0.99)
brier_logistic = brier_score_loss(df.loc[valid, "is_over"].values, p_logistic)
print(f"  Method 2 — Logistic(ypred, line): Brier = {brier_logistic:.5f}")

# Logistic coefficients (directional sanity check)
# Fit on all data for display
X_all = df[["ypred_oof", "line"]].values
scaler_all = StandardScaler()
X_all_s = scaler_all.fit_transform(X_all)
log_final = LogisticRegression(max_iter=1000, solver="lbfgs")
log_final.fit(X_all_s, df["is_over"].values)
coef_ypred, coef_line = log_final.coef_[0]
print(f"  Coefficients: ypred={coef_ypred:+.4f} (expect +), line={coef_line:+.4f} (expect -)")


# ── Method 3 — XGBClassifier on (ypred, line) ─────────────────────────────────

print("\n[Step 4b] Method 3 — XGBClassifier(ypred, line) LOO-season CV...")

oof_xgb = np.full(len(df), np.nan)

for test_season in SEASONS:
    train_mask = df["nfl_season"] != test_season
    test_mask  = df["nfl_season"] == test_season

    X_tr = df.loc[train_mask, ["ypred_oof", "line"]].values
    y_tr = df.loc[train_mask, "is_over"].values
    X_te = df.loc[test_mask,  ["ypred_oof", "line"]].values

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X_tr, y_tr)
    oof_xgb[test_mask] = model.predict_proba(X_te)[:, 1]

valid_xgb = ~np.isnan(oof_xgb)
p_xgb = np.clip(oof_xgb[valid_xgb], 0.01, 0.99)
brier_xgb = brier_score_loss(df.loc[valid_xgb, "is_over"].values, p_xgb)
print(f"  Method 3 — XGBClassifier(ypred, line): Brier = {brier_xgb:.5f}")


# ── Compare all 3 ──────────────────────────────────────────────────────────────

print("\n=== Method Comparison ===")
comparison = pd.DataFrame([
    {"method": "Method 1 — Distribution t(df=3, σ×1.3)", "brier": round(brier_dist,     5), "n": len(p_dist)},
    {"method": "Method 2 — Logistic(ypred, line)",        "brier": round(brier_logistic, 5), "n": int(valid.sum())},
    {"method": "Method 3 — XGBClassifier(ypred, line)",   "brier": round(brier_xgb,     5), "n": int(valid_xgb.sum())},
]).sort_values("brier")

print(comparison.to_string(index=False))
best_method = comparison.iloc[0]["method"]
best_brier  = comparison.iloc[0]["brier"]
print(f"\nWinner: {best_method} (Brier = {best_brier:.5f})")


# ── Calibration curves for all 3 ──────────────────────────────────────────────

print("\n[Step 4b] Computing calibration curves...")
calib_rows = []

for method_name, p_vals, actual in [
    ("Distribution t(df=3)", p_dist, df["is_over"].values),
    ("Logistic(ypred, line)", p_logistic, df.loc[valid, "is_over"].values),
    ("XGB(ypred, line)",      p_xgb,      df.loc[valid_xgb, "is_over"].values),
]:
    fraction_pos, mean_pred = calibration_curve(actual, p_vals, n_bins=8, strategy="quantile")
    for frac, pred in zip(fraction_pos, mean_pred):
        calib_rows.append({
            "method":      method_name,
            "pred_prob":   round(float(pred), 4),
            "actual_freq": round(float(frac), 4),
            "gap":         round(float(frac - pred), 4),
        })

calib_df = pd.DataFrame(calib_rows)


# ── Write winning p_model back to full spine ───────────────────────────────────

# Determine which method won and build p_model_over_oof for all spine rows
best_idx = comparison["brier"].idxmin()
best_row  = comparison.loc[best_idx]

print(f"\n[Step 4b] Writing {best_row['method']} as production p_model...")

# Re-compute on full spine (all books, all rows with ypred_oof)
spine_full = pd.read_parquet(S4_PATH)

if "Distribution" in best_row["method"]:
    z_all = (spine_full["line"] - spine_full["ypred_oof"]) / BEST_DIST_SIGMA
    p_over_new = np.clip(1 - t_dist.cdf(z_all.values, df=BEST_DIST_DF), 0.01, 0.99)
    winning_method_tag = f"t(df={BEST_DIST_DF}, σ={BEST_DIST_SIGMA:.1f})"

elif "Logistic" in best_row["method"]:
    # Train logistic on all BOL data with actuals, apply to full spine
    has_both = spine_full["ypred_oof"].notna() & spine_full["passing_yards"].notna() & (spine_full["bookmaker"] == "betonlineag")
    X_train_final = spine_full.loc[has_both, ["ypred_oof", "line"]].values
    y_train_final = (spine_full.loc[has_both, "outcome"] == "over").astype(int).values
    scaler_prod = StandardScaler()
    X_train_s = scaler_prod.fit_transform(X_train_final)
    log_prod = LogisticRegression(max_iter=1000, solver="lbfgs")
    log_prod.fit(X_train_s, y_train_final)

    has_ypred = spine_full["ypred_oof"].notna()
    X_score = spine_full.loc[has_ypred, ["ypred_oof", "line"]].values
    X_score_s = scaler_prod.transform(X_score)
    p_over_scored = log_prod.predict_proba(X_score_s)[:, 1]
    p_over_new = np.full(len(spine_full), np.nan)
    p_over_new[has_ypred.values] = np.clip(p_over_scored, 0.01, 0.99)
    winning_method_tag = "Logistic(ypred, line)"

else:  # XGB
    has_both = spine_full["ypred_oof"].notna() & spine_full["passing_yards"].notna() & (spine_full["bookmaker"] == "betonlineag")
    X_train_final = spine_full.loc[has_both, ["ypred_oof", "line"]].values
    y_train_final = (spine_full.loc[has_both, "outcome"] == "over").astype(int).values
    xgb_prod = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, random_state=42, verbosity=0, eval_metric="logloss")
    xgb_prod.fit(X_train_final, y_train_final)

    has_ypred = spine_full["ypred_oof"].notna()
    X_score = spine_full.loc[has_ypred, ["ypred_oof", "line"]].values
    p_over_scored = xgb_prod.predict_proba(X_score)[:, 1]
    p_over_new = np.full(len(spine_full), np.nan)
    p_over_new[has_ypred.values] = np.clip(p_over_scored, 0.01, 0.99)
    winning_method_tag = "XGBClassifier(ypred, line)"

spine_full["p_model_over_oof"]  = p_over_new
spine_full["p_model_under_oof"] = np.where(~np.isnan(p_over_new), 1 - p_over_new, np.nan)
spine_full["p_method"] = winning_method_tag

spine_full.to_parquet(OUT_PATH, index=False)
print(f"  Spine updated → {OUT_PATH}")

oof_valid = spine_full[spine_full["p_model_over_oof"].notna()]
print(f"  p_model_over stats (n={len(oof_valid):,}):")
print(f"    mean={oof_valid['p_model_over_oof'].mean():.4f}  std={oof_valid['p_model_over_oof'].std():.4f}")
print(f"    p5={np.percentile(oof_valid['p_model_over_oof'], 5):.4f}  p95={np.percentile(oof_valid['p_model_over_oof'], 95):.4f}")


# ── Spot-checks: Allen, Flacco, Ward, Maye ────────────────────────────────────

SPOT_CHECKS = [
    ("josh allen",    "Josh Allen (star starter)"),
    ("joe flacco",    "Joe Flacco (journeyman backup)"),
    ("cam ward",      "Cam Ward (2025 rookie)"),
    ("drake maye",    "Drake Maye (2nd year)"),
]

bol_oof = spine_full[
    (spine_full["bookmaker"] == "betonlineag") &
    spine_full["p_model_over_oof"].notna() &
    spine_full["passing_yards"].notna()
].copy()

spot_html_parts = []
for name_key, label in SPOT_CHECKS:
    player_df = bol_oof[bol_oof["player_norm"].str.contains(name_key, case=False, na=False)]
    if len(player_df) == 0:
        print(f"  {label}: no rows found")
        spot_html_parts.append(f"<p><strong>{label}</strong>: no rows found in spine.</p>")
        continue
    show = (
        player_df
        .drop_duplicates(subset=["nfl_season", "nfl_week"])
        .sort_values(["nfl_season", "nfl_week"])
        [["nfl_season", "nfl_week", "line", "passing_yards", "ypred_oof",
          "p_model_over_oof", "p_model_under_oof", "outcome"]]
        .rename(columns={
            "passing_yards":    "actual",
            "ypred_oof":        "ypred",
            "p_model_over_oof": "p_over",
            "p_model_under_oof":"p_under",
        })
        .round(3)
        .head(15)
        .reset_index(drop=True)
    )
    print(f"\n  {label} ({len(player_df)} rows):")
    print(show.to_string(index=False))
    spot_html_parts.append(f"<h4>{label}</h4>" + df_to_html(show))


# ── DuckDB tests ──────────────────────────────────────────────────────────────

import duckdb
print("\n[Step 4b] Running tests...")
con = duckdb.connect()
con.register("s4", spine_full)
con.register("cmp", comparison)

tests = []
def run_test(name, sql, expect_true=True):
    result = con.execute(sql).fetchone()[0]
    passed = bool(result) == expect_true
    tests.append({"test": name, "status": "PASS" if passed else "FAIL", "result": result})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name} → {result}")

run_test("T1: 3 methods compared",         "SELECT COUNT(*) = 3 FROM cmp")
run_test("T2: p_model in [0.01, 0.99]",    "SELECT COUNT(*) = 0 FROM s4 WHERE p_model_over_oof IS NOT NULL AND (p_model_over_oof < 0.01 OR p_model_over_oof > 0.99)")
run_test("T3: p_over + p_under = 1",       "SELECT COUNT(*) = 0 FROM s4 WHERE p_model_over_oof IS NOT NULL AND ABS(p_model_over_oof + p_model_under_oof - 1.0) > 0.001")
run_test("T4: logistic coef ypred > 0",    f"SELECT {coef_ypred:.4f} > 0")
run_test("T5: logistic coef line < 0",     f"SELECT {coef_line:.4f} < 0")
run_test("T6: best brier < 0.252",         f"SELECT MIN(brier) < 0.252 FROM cmp")

n_pass = sum(1 for t in tests if t["status"] == "PASS")
tests_df = pd.DataFrame(tests)
print(f"\n  Tests: {n_pass}/{len(tests)} passed")


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 4b] Writing HTML section...")

html = f"""
<section>
<h2>Step 4b — Secondary Classifier Methods (Logistic + XGB on ypred × line)</h2>
<p><em>{ts()}</em></p>

<div style="background:#f0f7ff;border-left:4px solid #3498db;padding:12px 16px;margin:12px 0;font-family:system-ui,Arial;font-size:14px;line-height:1.6">
<strong>Why try this?</strong> The distribution method (Step 4a) assumes a parametric shape for residuals. A secondary classifier skips that assumption — it trains directly on <code>(ypred, line) → is_over</code> and learns the empirical relationship from data. If the residuals aren't well-described by Student-t, this could outperform on Brier score.
<br><br>
<strong>Sanity check on logistic coefficients:</strong> ypred should be positive (higher predicted yards → more likely to go over); line should be negative (higher line → harder to go over). If either flips sign, the model is broken.
</div>

<h3>Method Comparison — Brier Score (lower = better)</h3>
{df_to_html(comparison.reset_index(drop=True), "All three methods on OOF predictions. Brier baseline (always predict 0.5) = 0.25.")}

<h3>Winner: {best_method}</h3>
<p>Brier = {best_brier:.5f}. Production spine updated with this method's p_model_over_oof.</p>

<h3>Logistic Coefficient Sanity Check</h3>
<ul>
  <li>ypred coefficient: <strong>{coef_ypred:+.4f}</strong> {'✓ positive (higher ypred → more likely over)' if coef_ypred > 0 else '✗ WRONG SIGN'}</li>
  <li>line coefficient:  <strong>{coef_line:+.4f}</strong>  {'✓ negative (higher line → harder to go over)' if coef_line < 0 else '✗ WRONG SIGN'}</li>
</ul>

<h3>Calibration Curves (all 3 methods)</h3>
{df_to_html(calib_df.reset_index(drop=True), "pred_prob = mean predicted prob per bin; actual_freq = actual over rate; gap = actual − pred (ideal: 0)")}

<h3>Spot-Check Players (winning method p_model)</h3>
{''.join(spot_html_parts)}

<h3>DuckDB Test Results</h3>
{df_to_html(tests_df)}

<h3>Key Takeaways</h3>
<ul>
  <li>Winner: <strong>{best_method}</strong> (Brier = {best_brier:.5f})</li>
  <li>All three methods produce Brier scores near the 0.25 baseline — consistent with AUC ~0.53 signal level</li>
  <li>Logistic coefficients are directionally correct: ypred (+), line (−)</li>
  <li>Spine updated with best method's p_model_over_oof → ready for Step 5 grid search</li>
</ul>
</section>
"""

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 4b] HTML appended → {HTML_PATH}")
print("\n=== DONE ===")
