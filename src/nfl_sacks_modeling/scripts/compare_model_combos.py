"""
Compare feature combinations × 2 models (LR, XGB).
Train: 2024  |  Holdout: 2025  |  Temporal split

Original 8 combos (M1-M8) — numeric only:
  M1  Baseline         prop_median_impl_over only
  M2  Market both      + prop_median_impl_under
  M3  Full mkt over    median + mean + min + max impl_over
  M4  + best non-mkt   M1 + qbhit_rate_L16
  M5  + pressure pair  M1 + qbhit_rate_L16 + qbhit_rate_Lcareer
  M6  + sack history   M1 + sack_rate_Lcareer + sack_rate_L16
  M7  Parsimonious     M1 + qbhit_rate_L16 + sack_rate_Lcareer
  M8  Kitchen sink     all 37 numeric features

Categorical combos (C1-C8) — M7 base + cat vars:
  C1  M7 + pos_side
  C2  M7 + pos_group
  C3  M7 + pos_side + pos_group
  C4  M7 + prop_median_impl_over_bin
  C5  M7 + pos_side + prop_median_impl_over_bin
  C6  M5 + pos_side
  C7  Baseline + pos_side
  C8  M7 + pos_side + pos_group + prop_median_impl_over_bin

Outputs:
  ~/Downloads/tmp/sacks_model_combos.csv
  Printed ranked table

Run:
  python src/nfl_sacks_modeling/scripts/compare_model_combos.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TMP = Path.home() / "Downloads" / "tmp"
F24 = TMP / "nfl_sacks_features_2024.parquet"
F25 = TMP / "nfl_sacks_features_2025.parquet"
OUT = TMP / "sacks_model_combos.csv"

SEP = "=" * 90

ALL_NUMERIC = [
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

# (numeric_features, categorical_features)
M7_NUM = ["prop_median_impl_over", "qbhit_rate_L16", "sack_rate_Lcareer"]
M5_NUM = ["prop_median_impl_over", "qbhit_rate_L16", "qbhit_rate_Lcareer"]

COMBOS = {
    # ── original numeric-only combos ─────────────────────────────────────────
    "M1-baseline":      (["prop_median_impl_over"], []),
    "M2-mkt-both":      (["prop_median_impl_over", "prop_median_impl_under"], []),
    "M3-mkt-over-full": (["prop_median_impl_over", "prop_mean_impl_over",
                          "prop_min_impl_over", "prop_max_impl_over"], []),
    "M4-+qbhit-L16":    (["prop_median_impl_over", "qbhit_rate_L16"], []),
    "M5-+qbhit-pair":   (M5_NUM, []),
    "M6-+sack-hist":    (["prop_median_impl_over", "sack_rate_Lcareer", "sack_rate_L16"], []),
    "M7-parsimonious":  (M7_NUM, []),
    "M8-kitchen-sink":  (ALL_NUMERIC, []),
    # ── categorical combos ────────────────────────────────────────────────────
    "C1-M7+pos_side":              (M7_NUM, ["pos_side"]),
    "C2-M7+pos_group":             (M7_NUM, ["pos_group"]),
    "C3-M7+pos_both":              (M7_NUM, ["pos_side", "pos_group"]),
    "C4-M7+impl_bin":              (M7_NUM, ["prop_median_impl_over_bin"]),
    "C5-M7+pos_side+impl_bin":     (M7_NUM, ["pos_side", "prop_median_impl_over_bin"]),
    "C6-M5+pos_side":              (M5_NUM, ["pos_side"]),
    "C7-baseline+pos_side":        (["prop_median_impl_over"], ["pos_side"]),
    "C8-M7+all_cat":               (M7_NUM, ["pos_side", "pos_group", "prop_median_impl_over_bin"]),
}


def load_data():
    train = pd.read_parquet(F24)
    hold  = pd.read_parquet(F25)
    train = train[train["target"].notna()].copy()
    hold  = hold[hold["target"].notna()].copy()
    return train, hold


def build_pipeline_mixed(n_cols, c_cols):
    transformers = []
    if n_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler()),
        ]), n_cols))
    if c_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), c_cols))
    pre = ColumnTransformer(transformers)
    return Pipeline([
        ("pre", pre),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])


def run_combo(num_features, cat_features, train, hold):
    n_cols = [f for f in num_features if f in train.columns]
    c_cols = [f for f in cat_features if f in train.columns]
    all_cols = n_cols + c_cols

    X_tr = train[all_cols]
    X_ho = hold[all_cols]
    y_tr = train["target"].astype(int)
    y_ho = hold["target"].astype(int)

    results = {}

    # LR with ColumnTransformer
    lr_pipe = build_pipeline_mixed(n_cols, c_cols)
    lr_pipe.fit(X_tr, y_tr)
    lr_prob = lr_pipe.predict_proba(X_ho)[:, 1]
    results["LR"] = score(y_ho, lr_prob)

    # XGB — fill numeric with train mean, leave categoricals as-is (XGB handles strings via OHE)
    num_fill = {f: train[f].mean() for f in n_cols}
    X_tr_xgb = X_tr.copy()
    X_ho_xgb = X_ho.copy()
    for f, v in num_fill.items():
        X_tr_xgb[f] = X_tr_xgb[f].fillna(v)
        X_ho_xgb[f] = X_ho_xgb[f].fillna(v)

    if c_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_tr = ohe.fit_transform(X_tr_xgb[c_cols].fillna("missing"))
        cat_ho = ohe.transform(X_ho_xgb[c_cols].fillna("missing"))
        num_tr = X_tr_xgb[n_cols].values if n_cols else np.empty((len(X_tr_xgb), 0))
        num_ho = X_ho_xgb[n_cols].values if n_cols else np.empty((len(X_ho_xgb), 0))
        Xtr_xgb_arr = np.hstack([num_tr, cat_tr])
        Xho_xgb_arr = np.hstack([num_ho, cat_ho])
    else:
        Xtr_xgb_arr = X_tr_xgb[n_cols].values
        Xho_xgb_arr = X_ho_xgb[n_cols].values

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    xgb.fit(Xtr_xgb_arr, y_tr)
    xgb_prob = xgb.predict_proba(Xho_xgb_arr)[:, 1]
    results["XGB"] = score(y_ho, xgb_prob)

    return results, len(n_cols) + len(c_cols)


def score(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def main():
    print(f"\n{SEP}")
    print("  Model combo comparison — Train: 2024 | Holdout: 2025")
    print(SEP)

    train, hold = load_data()
    print(f"  Train: {len(train):,} rows  |  Hold: {len(hold):,} rows\n")

    rows = []
    for combo_name, (num_feats, cat_feats) in COMBOS.items():
        results, n_feats = run_combo(num_feats, cat_feats, train, hold)
        for model, m in results.items():
            rows.append({
                "combo":     combo_name,
                "model":     model,
                "n_feats":   n_feats,
                "auc":       m["auc"],
                "accuracy":  m["accuracy"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
            })
        cat_str = f" cat={cat_feats}" if cat_feats else ""
        print(f"  {combo_name} ({n_feats} feats{cat_str}) — "
              f"LR AUC={results['LR']['auc']:.4f}  XGB AUC={results['XGB']['auc']:.4f}")

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    baseline_auc = df.loc[(df["combo"] == "M1-baseline") & (df["model"] == "LR"), "auc"].iloc[0]
    df["auc_delta"] = (df["auc"] - baseline_auc).round(4)

    print(f"\n{SEP}")
    print(f"  Ranked by holdout AUC  (baseline LR M1 = {baseline_auc:.4f})")
    print(SEP)
    print(f"{'Rank':<5} {'Combo':<30} {'Mdl':<5} {'Feats':>5} {'AUC':>6} {'Delta':>7} "
          f"{'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 83)
    for _, r in df.iterrows():
        print(f"{int(r['rank']):<5} {r['combo']:<30} {r['model']:<5} {int(r['n_feats']):>5} "
              f"{r['auc']:>6.4f} {r['auc_delta']:>+7.4f} {r['accuracy']:>6.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f}")

    df.to_csv(OUT, index=False)
    print(f"\n  Saved → {OUT}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
