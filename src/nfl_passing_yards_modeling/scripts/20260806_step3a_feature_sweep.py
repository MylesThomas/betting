"""
Step 3a — Individual Feature Predictive Power Sweep

For each candidate feature independently:
  - Linear regression on actual passing_yards (continuous target) → R², RMSE, MAE
  - Logistic regression on is_over (binary target) → AUC, precision, recall

CV: leave-one-season-out (train on 2 seasons, test on the held-out season, 3 folds).
No future leakage — this mirrors actual deployment where we train on past seasons.

Appends a section to knowledge-base/raw/20260806-nfl-qb-pass-yds.html.

Usage:
  uv run python 20260806_step3a_feature_sweep.py
"""

from __future__ import annotations

import sys
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

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
TMP_DIR    = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH  = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
SPINE_PATH = TMP_DIR / "step2_spine.parquet"
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

print("[Step 3a] Loading spine...")
spine = pd.read_parquet(SPINE_PATH)

# Keep only rows with actuals (matched games)
df = spine[spine["passing_yards"].notna()].copy().reset_index(drop=True)
df["is_over"] = (df["outcome"] == "over").astype(int)

print(f"  Rows with actuals: {len(df):,}")
print(f"  Seasons: {sorted(df['nfl_season'].unique())}")
print(f"  Over rate: {df['is_over'].mean():.3f}")

SEASONS = sorted(df["nfl_season"].unique())


# ── Feature definitions ───────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # Rolling passing yards
    "pass_yds_roll_1","pass_yds_roll_3","pass_yds_roll_5","pass_yds_roll_10",
    "pass_yds_roll_season","pass_yds_roll_career",
    # Rolling attempts
    "attempts_roll_1","attempts_roll_3","attempts_roll_5","attempts_roll_10",
    "attempts_roll_season","attempts_roll_career",
    # Rolling comp pct
    "comp_pct_roll_1","comp_pct_roll_3","comp_pct_roll_5","comp_pct_roll_10",
    "comp_pct_roll_season","comp_pct_roll_career",
    # Rolling YPA
    "ypa_roll_1","ypa_roll_3","ypa_roll_5","ypa_roll_10",
    "ypa_roll_season","ypa_roll_career",
    # Rolling EPA/att
    "epa_per_att_roll_1","epa_per_att_roll_3","epa_per_att_roll_5","epa_per_att_roll_10",
    "epa_per_att_roll_season","epa_per_att_roll_career",
    # Rolling sack rate
    "sack_rate_roll_3","sack_rate_roll_5","sack_rate_roll_10",
    "sack_rate_roll_season","sack_rate_roll_career",
    # Rolling pass rate
    "pass_rate_roll_3","pass_rate_roll_5","pass_rate_roll_10",
    "pass_rate_roll_season","pass_rate_roll_career",
    # Rolling std dev (consistency)
    "pass_yds_std_5","pass_yds_std_10","pass_yds_std_career",
    # Cover rate (vs BetOnline line)
    "cover_rate_roll_1","cover_rate_roll_3","cover_rate_roll_5","cover_rate_roll_10",
    "cover_rate_roll_season","cover_rate_roll_career",
    # Starter flags
    "career_starts_pct","starts_last_4","games_as_starter_this_season","started_week1_for_team",
    # Opponent defense
    "opp_pass_yds_allowed_roll5",
    # Game lines
    "total_line","team_spread","implied_team_total",
    # Market
    "consensus_line","min_line","max_line","n_books",
    "cons_novig_prob_over","min_raw_prob_over","max_raw_prob_over",
    "min_raw_prob_under","max_raw_prob_under",
    # Context
    "nfl_week",
]

CATEGORICAL_FEATURES = [
    "consensus_over_odds_bin",
    "consensus_over_odds_bin_granular",
    "consensus_under_odds_bin",
    "consensus_under_odds_bin_granular",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Leave-one-season-out CV ───────────────────────────────────────────────────

def loo_season_cv(df: pd.DataFrame, feature: str, target: str, is_cat: bool = False):
    """
    For each test season, train on all other seasons, predict on test season.
    Returns concatenated OOF predictions.
    """
    oof_preds  = np.full(len(df), np.nan)
    oof_actual = np.full(len(df), np.nan)

    for test_season in SEASONS:
        train_mask = df["nfl_season"] != test_season
        test_mask  = df["nfl_season"] == test_season

        train = df[train_mask][[feature, target]].dropna()
        test  = df[test_mask][[feature, target]].dropna()

        if len(train) < 30 or len(test) < 10:
            continue

        X_train = train[[feature]].values
        y_train = train[target].values
        X_test  = test[[feature]].values
        y_test  = test[target].values

        if is_cat:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train = enc.fit_transform(X_train)
            X_test  = enc.transform(X_test)

        # Fill NaN with median (for numeric)
        if not is_cat:
            med = np.nanmedian(X_train)
            X_train = np.where(np.isnan(X_train), med, X_train)
            med_t   = np.nanmedian(X_test) if np.isnan(X_test).any() else med
            X_test  = np.where(np.isnan(X_test), med_t, X_test)

        if target == "passing_yards":
            model = LinearRegression()
        else:
            model = LogisticRegression(max_iter=500, solver="lbfgs")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Store predictions for matched test rows
        test_indices = df[test_mask].index[df[test_mask][[feature, target]].notna().all(axis=1)]
        oof_preds[test_indices]  = preds
        oof_actual[test_indices] = y_test

    valid = ~np.isnan(oof_preds) & ~np.isnan(oof_actual)
    return oof_preds[valid], oof_actual[valid]


# ── Run sweep ─────────────────────────────────────────────────────────────────

print("\n[Step 3a] Running feature sweep...")
print(f"  {len(ALL_FEATURES)} features × 2 targets (passing_yards, is_over)\n")

results = []

for feat in ALL_FEATURES:
    if feat not in df.columns:
        print(f"  SKIP {feat} (not in spine)")
        continue

    is_cat = feat in CATEGORICAL_FEATURES
    n_valid = df[[feat, "passing_yards"]].dropna().__len__()

    # ── Regression: predict actual passing yards ──────────────────────────────
    preds_reg, actual_reg = loo_season_cv(df, feat, "passing_yards", is_cat)
    if len(preds_reg) < 50:
        r2, rmse, mae = np.nan, np.nan, np.nan
    else:
        ss_res = np.sum((actual_reg - preds_reg) ** 2)
        ss_tot = np.sum((actual_reg - actual_reg.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot
        rmse = np.sqrt(mean_squared_error(actual_reg, preds_reg))
        mae  = mean_absolute_error(actual_reg, preds_reg)

    # ── Classification: predict is_over ──────────────────────────────────────
    preds_cls, actual_cls = loo_season_cv(df, feat, "is_over", is_cat)
    if len(preds_cls) < 50 or len(np.unique(actual_cls)) < 2:
        auc, prec, rec = np.nan, np.nan, np.nan
    else:
        preds_bin = (preds_cls >= 0.5).astype(int)
        try:
            auc  = roc_auc_score(actual_cls, preds_cls)
        except Exception:
            auc  = np.nan
        prec = precision_score(actual_cls, preds_bin, zero_division=0)
        rec  = recall_score(actual_cls, preds_bin, zero_division=0)

    results.append({
        "feature":  feat,
        "n_valid":  n_valid,
        "type":     "cat" if is_cat else "num",
        "r2":       round(r2,   4) if not np.isnan(r2)   else None,
        "rmse":     round(rmse, 2) if not np.isnan(rmse) else None,
        "mae":      round(mae,  2) if not np.isnan(mae)  else None,
        "auc":      round(auc,  4) if not np.isnan(auc)  else None,
        "prec":     round(prec, 4) if not np.isnan(prec) else None,
        "rec":      round(rec,  4) if not np.isnan(rec)  else None,
    })

    tag = "cat" if is_cat else "num"
    print(f"  {feat:<45} r2={r2:+.4f}  rmse={rmse:.1f}  auc={auc:.4f}")

results_df = pd.DataFrame(results)


# ── Rank and summarize ────────────────────────────────────────────────────────

# Sort by AUC descending (primary signal for over/under prediction)
by_auc  = results_df.sort_values("auc", ascending=False).reset_index(drop=True)
by_r2   = results_df.sort_values("r2",  ascending=False).reset_index(drop=True)
by_rmse = results_df.sort_values("rmse", ascending=True).reset_index(drop=True)

print("\n=== Top 15 by AUC (is_over prediction) ===")
print(by_auc[["feature","auc","prec","rec","r2","rmse"]].head(15).to_string(index=False))

print("\n=== Top 15 by R² (passing_yards regression) ===")
print(by_r2[["feature","r2","rmse","mae","auc"]].head(15).to_string(index=False))

print("\n=== Consensus odds bins ===")
cats = results_df[results_df["type"] == "cat"]
print(cats[["feature","auc","r2","rmse"]].to_string(index=False))

print("\n=== Starter flag features ===")
starters = results_df[results_df["feature"].str.contains("start|career_starts", na=False)]
print(starters[["feature","auc","r2","rmse"]].to_string(index=False))

print("\n=== Market features (line, prob, books) ===")
mkt_feats = results_df[results_df["feature"].str.contains("line|prob|n_books|novig", na=False)]
print(mkt_feats[["feature","auc","r2","rmse"]].to_string(index=False))

# Baseline (no features — just predicting mean)
y_all = df["passing_yards"].dropna().values
baseline_rmse = np.sqrt(np.mean((y_all - y_all.mean()) ** 2))
baseline_mae  = np.mean(np.abs(y_all - y_all.mean()))
print(f"\nBaseline (predict mean): RMSE={baseline_rmse:.2f}  MAE={baseline_mae:.2f}")


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 3a] Writing HTML section...")

top10_auc  = by_auc.head(10)[["feature","n_valid","auc","prec","rec","r2","rmse","mae"]]
top10_r2   = by_r2.head(10)[["feature","n_valid","r2","rmse","mae","auc"]]
full_table = by_auc[["feature","type","n_valid","r2","rmse","mae","auc","prec","rec"]]

html = f"""
<section>
<h2>Step 3a — Individual Feature Predictive Power Sweep</h2>
<p><em>{ts()}</em></p>

<h3>Method</h3>
<ul>
  <li>Leave-one-season-out CV: train on 2 of 3 seasons, test on held-out season (×3 folds)</li>
  <li>Linear regression → predict actual <code>passing_yards</code> (R², RMSE, MAE)</li>
  <li>Logistic regression → predict <code>is_over</code> binary (AUC, precision, recall)</li>
  <li>Each feature tested independently; NaN → median imputation for numeric</li>
  <li>Baseline RMSE (predict mean): <strong>{baseline_rmse:.2f}</strong> | MAE: <strong>{baseline_mae:.2f}</strong></li>
</ul>

<h3>Top 10 Features by AUC (Over/Under prediction)</h3>
{df_to_html(top10_auc, "Higher AUC = more signal for predicting whether outcome goes over the line")}

<h3>Top 10 Features by R² (Passing Yards regression)</h3>
{df_to_html(top10_r2, "Higher R² = better at predicting raw passing yards")}

<h3>Consensus Odds Bin Features</h3>
{df_to_html(cats[["feature","auc","prec","rec","r2","rmse"]], "4 categorical market features — do they have standalone predictive power?")}

<h3>Starter Flag Features</h3>
{df_to_html(starters[["feature","n_valid","auc","r2","rmse"]], "Starter status signals")}

<h3>Market Features (line, prob, n_books)</h3>
{df_to_html(mkt_feats[["feature","n_valid","auc","r2","rmse"]], "Line and probability features")}

<h3>Full Feature Sweep Table</h3>
{df_to_html(full_table, "All features sorted by AUC descending")}

<h3>Key Takeaways</h3>
<ul>
"""

# Auto-generate takeaways
top3_auc  = by_auc.head(3)["feature"].tolist()
top3_r2   = by_r2.head(3)["feature"].tolist()
best_auc  = by_auc.iloc[0]
worst_auc = by_auc.iloc[-1]

html += f"<li>Best AUC: <strong>{best_auc['feature']}</strong> = {best_auc['auc']:.4f}</li>"
html += f"<li>Top 3 by AUC: {', '.join(top3_auc)}</li>"
html += f"<li>Top 3 by R²: {', '.join(top3_r2)}</li>"
html += f"<li>Baseline RMSE = {baseline_rmse:.2f} yds — best individual feature RMSE = {by_rmse.iloc[0]['rmse']:.2f} yds ({by_rmse.iloc[0]['feature']})</li>"

odds_bin_aucs = cats["auc"].dropna()
if len(odds_bin_aucs):
    html += f"<li>Consensus odds bin AUCs: {odds_bin_aucs.min():.4f}–{odds_bin_aucs.max():.4f} — {'strong signal, include in model' if odds_bin_aucs.max() > 0.55 else 'weak standalone signal'}</li>"

starter_aucs = starters["auc"].dropna()
if len(starter_aucs):
    html += f"<li>Starter flag AUCs: {starter_aucs.min():.4f}–{starter_aucs.max():.4f}</li>"

html += "</ul></section>"

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 3a] HTML section appended: {HTML_PATH}")
print("\n=== DONE ===")
