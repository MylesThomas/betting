"""
Rank all single predictors for the sacks model.

Train : 2024  |  Holdout : 2025  |  Temporal split — no look-ahead
Target: 1 = sacks >= 1.0, 0 = sacks == 0.0  (pushes dropped)

For each predictor, trains:
  - LR  (LogisticRegression, C=1.0)
  - XGB (XGBClassifier, shallow tree, early stopping off)

Baseline: LR with prop_median_impl_over (market implied probability alone).
All models ranked by holdout AUC delta vs baseline.

Outputs:
  ~/Downloads/tmp/sacks_predictor_ranking.csv
  Printed ranked table

Run:
  python src/nfl_sacks_modeling/scripts/rank_predictors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TMP    = Path.home() / "Downloads" / "tmp"
F24    = TMP / "nfl_sacks_features_2024.parquet"
F25    = TMP / "nfl_sacks_features_2025.parquet"
OUT    = TMP / "sacks_predictor_ranking.csv"

THRESHOLD = 0.30  # predict Under if P(over) < THRESHOLD

NUMERIC_FEATS = [
    "sack_rate_L1", "sack_rate_L3", "sack_rate_L5", "sack_rate_L8",
    "sack_rate_L16", "sack_rate_Lcareer",
    "qbhit_rate_L1", "qbhit_rate_L3", "qbhit_rate_L5", "qbhit_rate_L8",
    "qbhit_rate_L16", "qbhit_rate_Lcareer",
    "snap_pct_L1", "snap_pct_L3", "snap_pct_L5", "snap_pct_L8",
    "snap_pct_L16", "snap_pct_Lcareer",
    "game_total", "team_spread", "games_played_ytd",
    "prop_median_impl_over", "prop_median_impl_under",
    "prop_mean_impl_over", "prop_mean_impl_under",
    "prop_min_impl_over", "prop_max_impl_over",
    "prop_min_impl_under", "prop_max_impl_under",
    "prop_book_spread_over", "prop_book_spread_under",
    "prop_n_books",
    "fanduel_over_0p5_implied",
    "betonline_over_0p5_implied", "betonline_under_0p5_implied",
    "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
]

CATEGORICAL_FEATS = [
    "pos_group", "pos_side",
    "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
    "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
]

BASELINE_FEAT = "prop_median_impl_over"

SEP = "=" * 80


def load_data():
    train_raw = pd.read_parquet(F24)
    hold_raw  = pd.read_parquet(F25)
    train = train_raw[train_raw["target"].notna()].copy()
    hold  = hold_raw[hold_raw["target"].notna()].copy()
    print(f"  Train (2024): {len(train):,} rows  pos={int(train['target'].sum())}  neg={int((train['target']==0).sum())}")
    print(f"  Hold  (2025): {len(hold):,} rows   pos={int(hold['target'].sum())}  neg={int((hold['target']==0).sum())}")
    return train, hold


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_numeric(feat, train, hold):
    X_tr = train[[feat]].fillna(train[feat].mean())
    X_ho = hold[[feat]].fillna(train[feat].mean())
    y_tr = train["target"].astype(int)
    y_ho = hold["target"].astype(int)

    results = {}

    # LR
    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"))])
    lr.fit(X_tr, y_tr)
    results["LR"] = metrics(y_ho, lr.predict_proba(X_ho)[:, 1])

    # XGB
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         subsample=0.8, eval_metric="logloss",
                         verbosity=0, random_state=42)
    xgb.fit(X_tr, y_tr)
    results["XGB"] = metrics(y_ho, xgb.predict_proba(X_ho)[:, 1])

    return results


def run_categorical(feat, train, hold):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = ohe.fit_transform(train[[feat]].fillna("missing"))
    X_ho = ohe.transform(hold[[feat]].fillna("missing"))
    y_tr = train["target"].astype(int)
    y_ho = hold["target"].astype(int)

    results = {}

    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    results["LR"] = metrics(y_ho, lr.predict_proba(X_ho)[:, 1])

    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         subsample=0.8, eval_metric="logloss",
                         verbosity=0, random_state=42)
    xgb.fit(X_tr, y_tr)
    results["XGB"] = metrics(y_ho, xgb.predict_proba(X_ho)[:, 1])

    return results


def main():
    print(f"\n{SEP}")
    print("  Single-predictor ranking — Train: 2024 | Holdout: 2025")
    print(SEP)

    train, hold = load_data()
    y_ho = hold["target"].astype(int)

    # ── Baseline ──────────────────────────────────────────────────────────────
    base_results = run_numeric(BASELINE_FEAT, train, hold)
    baseline_auc = base_results["LR"]["auc"]
    print(f"\n  Baseline (LR, {BASELINE_FEAT}): AUC = {baseline_auc:.4f}")

    # ── Run all predictors ────────────────────────────────────────────────────
    rows = []
    all_feats = [(f, "numeric") for f in NUMERIC_FEATS] + \
                [(f, "categorical") for f in CATEGORICAL_FEATS]

    for feat, ftype in all_feats:
        if feat not in train.columns:
            continue
        try:
            res = run_numeric(feat, train, hold) if ftype == "numeric" \
                  else run_categorical(feat, train, hold)
            for model, m in res.items():
                rows.append({
                    "feature":    feat,
                    "type":       ftype,
                    "model":      model,
                    "auc":        m["auc"],
                    "auc_delta":  round(m["auc"] - baseline_auc, 4),
                    "accuracy":   m["accuracy"],
                    "precision":  m["precision"],
                    "recall":     m["recall"],
                    "f1":         m["f1"],
                })
        except Exception as e:
            print(f"  SKIP {feat}: {e}")

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Rank':<5} {'Feature':<35} {'Model':<5} {'AUC':>6} {'Delta':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 85)
    for _, r in df.iterrows():
        delta_str = f"{r['auc_delta']:+.4f}"
        print(f"{int(r['rank']):<5} {r['feature']:<35} {r['model']:<5} "
              f"{r['auc']:>6.4f} {delta_str:>7} {r['accuracy']:>6.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f}")

    df.to_csv(OUT, index=False)
    print(f"\n  Saved → {OUT}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
