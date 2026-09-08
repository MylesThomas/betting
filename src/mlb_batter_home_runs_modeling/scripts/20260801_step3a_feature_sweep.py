"""
Step 3a — Individual feature sweep (n=1 logistic + linear regression, OOF CV).

For each candidate feature independently:
  - Logistic regression (binary target: hr_over_0_5)
  - AUC, precision, recall, F1, coefficients

Also includes 4 consensus odds bin features + 6 min/max features.

Output: printed table + HTML appended to session log.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_batter_hr_spine.parquet"
HTML_LOG    = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
ET          = ZoneInfo("America/New_York")


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html_table(df: pd.DataFrame, caption: str = "") -> str:
    rows_html = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows_html.append(f"<tr>{cells}</tr>")
    cap = f"<caption style='font-weight:bold;text-align:left;padding:6px 0'>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


# Feature candidates — all player-game-level (book-invariant)
NUMERIC_CANDIDATES = [
    "hr_roll_L1",
    "hr_roll_L3",
    "hr_roll_L5",
    "hr_roll_L10",
    "hr_roll_L20",
    "hr_roll_season",
    "hr_roll_career",
    "ab_roll_L5",
    "ab_roll_career",
    "games_played_career",
    "opp_hr_rate_career",
    "opp_hr_rate_L20",
    "is_home",
    "min_line",
    "max_line",
    "min_raw_implied_prob_over",
    "max_raw_implied_prob_over",
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_under",
    "consensus_line",
]

CATEGORICAL_CANDIDATES = [
    "consensus_over_odds_bin",
    "consensus_over_odds_bin_granular",
    "consensus_under_odds_bin",
    "consensus_under_odds_bin_granular",
]

N_SPLITS = 5


def oof_logistic_numeric(df: pd.DataFrame, feature: str, target: str) -> dict:
    """OOF logistic regression for a single numeric feature."""
    valid = df[[feature, target]].dropna()
    X = valid[[feature]].values
    y = valid[target].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y_pred_proba = np.zeros(len(valid))

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_train, y_train)
        y_pred_proba[val_idx] = clf.predict_proba(X_val)[:, 1]

    # Use threshold 0.5 for classification metrics
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    # Coefficient from full-data fit for direction check
    clf_full = LogisticRegression(max_iter=500, C=1.0)
    clf_full.fit(X, y)
    coef = float(clf_full.coef_[0][0])

    try:
        auc = roc_auc_score(y, y_pred_proba)
    except Exception:
        auc = np.nan

    return {
        "feature":     feature,
        "model_type":  "logistic",
        "n_samples":   len(valid),
        "auc":         round(auc, 4),
        "precision":   round(precision_score(y, y_pred_binary, zero_division=0), 4),
        "recall":      round(recall_score(y, y_pred_binary, zero_division=0), 4),
        "f1":          round(f1_score(y, y_pred_binary, zero_division=0), 4),
        "coefficient": round(coef, 4),
    }


def oof_logistic_categorical(df: pd.DataFrame, feature: str, target: str) -> dict:
    """OOF logistic regression for a single categorical feature (label-encoded)."""
    valid = df[[feature, target]].dropna()
    le = LabelEncoder()
    X = le.fit_transform(valid[feature]).reshape(-1, 1)
    y = valid[target].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y_pred_proba = np.zeros(len(valid))

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_train, y_train)
        y_pred_proba[val_idx] = clf.predict_proba(X_val)[:, 1]

    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y, y_pred_proba)
    except Exception:
        auc = np.nan

    return {
        "feature":     feature,
        "model_type":  "logistic_cat",
        "n_samples":   len(valid),
        "auc":         round(auc, 4),
        "precision":   round(precision_score(y, y_pred_binary, zero_division=0), 4),
        "recall":      round(recall_score(y, y_pred_binary, zero_division=0), 4),
        "f1":          round(f1_score(y, y_pred_binary, zero_division=0), 4),
        "coefficient": None,
    }


def main() -> None:
    if not LOCAL_SPINE.exists():
        print(f"ERROR: {LOCAL_SPINE} not found — run 20260801_build_spine.py first")
        sys.exit(1)

    spine = pd.read_parquet(LOCAL_SPINE)
    print(f"Loaded spine: {len(spine):,} rows")

    # Deduplicate to player-game level for model training
    # (one row per player-game — take first book row for each)
    df = (
        spine
        .dropna(subset=["hr_over_0_5"])
        .sort_values(["player_key", "game_date", "game_date"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    print(f"Player-game level rows (deduped for training): {len(df):,}")
    print(f"Target distribution: {df['hr_over_0_5'].value_counts().to_dict()}")

    TARGET = "hr_over_0_5"

    results = []

    print("\nNumeric feature sweep...")
    for feat in NUMERIC_CANDIDATES:
        if feat not in df.columns:
            print(f"  SKIP {feat} — not in spine")
            continue
        null_rate = df[feat].isna().mean()
        if null_rate > 0.5:
            print(f"  SKIP {feat} — null rate {null_rate:.1%}")
            continue
        res = oof_logistic_numeric(df, feat, TARGET)
        results.append(res)
        print(f"  {feat:35s}  auc={res['auc']:.4f}  coef={res['coefficient']:+.4f}")

    print("\nCategorical feature sweep...")
    for feat in CATEGORICAL_CANDIDATES:
        if feat not in df.columns:
            print(f"  SKIP {feat} — not in spine")
            continue
        res = oof_logistic_categorical(df, feat, TARGET)
        results.append(res)
        print(f"  {feat:35s}  auc={res['auc']:.4f}")

    results_df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    print("\n=== Feature Ranking by AUC (descending) ===")
    print(results_df.to_string(index=False))

    # Spot-check: Aaron Judge
    judge = df[df["player_key"] == "aaron judge"].sort_values("game_date")
    print(f"\n=== Spot-check Aaron Judge: {len(judge)} player-game rows ===")
    show_cols = [c for c in ["game_date", "hr_over_0_5", "hr_roll_L5", "hr_roll_L10",
                              "hr_roll_career", "opp_hr_rate_career",
                              "min_raw_implied_prob_under", "max_raw_implied_prob_under"] if c in judge.columns]
    print(judge[show_cols].tail(10).to_string(index=False))

    # HTML
    section_html = f"""
<section>
<h2>Step 3a — Individual Feature Sweep (n=1 Logistic Regression, OOF)</h2>
<p class="timestamp">{ts()}</p>

<h3>Feature Rankings by AUC (descending)</h3>
{df_to_html_table(results_df, f"All feature candidates — logistic regression OOF (n_splits={N_SPLITS})")}

<h3>Spot-check: Aaron Judge</h3>
{df_to_html_table(judge[show_cols].tail(10).reset_index(drop=True))}

<h3>Observations</h3>
<ul>
  <li>Target: <code>hr_over_0_5</code> (1 if home_runs ≥ 1 else 0). Base rate ~10%.</li>
  <li>n=1 features ranked by OOF AUC. AUC > 0.5 = better than random. AUC > 0.55 = meaningful signal.</li>
  <li>Market-derived features (min/max raw implied prob) likely rank near top — market prices HR probability based on batter quality. Understand whether they add signal beyond what rolling career stats already capture.</li>
</ul>
</section>
"""
    with open(HTML_LOG, "a") as f:
        f.write(section_html)
    print(f"\nHTML appended → {HTML_LOG}")


if __name__ == "__main__":
    main()
