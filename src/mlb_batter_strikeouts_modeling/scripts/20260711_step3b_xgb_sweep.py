"""
Step 3b — Individual feature sweep with XGBoost (OOF CV).

Same features and OOF structure as 3a. Stacks results with 3a CSV for
side-by-side comparison. Flags features where XGBoost AUC delta > 0.02
over logistic regression.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

FEATURES_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_features.parquet"
SWEEP_3A_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_3a_sweep.csv"
OUT_PATH      = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_3ab_sweep.csv"

N_FOLDS = 5


def temporal_oof_xgb(df: pd.DataFrame, feature_cols: list[str], target_col: str = "over_flag"):
    df_sorted = df.dropna(subset=feature_cols + [target_col]).sort_values("game_date").reset_index(drop=True)
    n = len(df_sorted)
    fold_size = n // N_FOLDS
    oof_preds = np.full(len(df_sorted), np.nan)

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 1)
        val_idx   = df_sorted.index[train_end:] if fold == N_FOLDS - 1 else df_sorted.index[train_end: train_end + fold_size]
        train_idx = df_sorted.index[:train_end]

        if len(train_idx) < 50 or len(val_idx) < 10:
            continue

        X_train = df_sorted.loc[train_idx, feature_cols].values
        y_train = df_sorted.loc[train_idx, target_col].values
        X_val   = df_sorted.loc[val_idx, feature_cols].values

        clf = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf.fit(X_train, y_train)
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]

    valid_mask = ~np.isnan(oof_preds)
    y_true = df_sorted[target_col].values[valid_mask]
    y_pred = oof_preds[valid_mask]
    y_bin  = (y_pred >= 0.5).astype(int)

    return {
        "auc":       round(roc_auc_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_bin, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_bin, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_bin, zero_division=0), 4),
        "n_samples": int(valid_mask.sum()),
        "coefficient": np.nan,
    }


def encode_categorical(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)

    for col in ["stand", "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
                "consensus_under_odds_bin", "consensus_under_odds_bin_granular"]:
        df, _ = encode_categorical(df, col)

    numeric_features = [
        "offered_line",
        "k_roll_L1", "k_roll_L5", "k_roll_L10", "k_roll_L20",
        "k_roll_season", "k_roll_career",
        "k_rate_L5", "k_rate_career",
        "pa_roll_L5", "pa_roll_career",
        "opp_k_rate_career", "opp_k_rate_L5",
        "is_home",
        "novig_prob_over",
        "min_line", "max_line",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
    ]
    categorical_groups = {
        "consensus_over_odds_bin":           [c for c in df.columns if c.startswith("consensus_over_odds_bin_") and "_granular" not in c],
        "consensus_over_odds_bin_granular":  [c for c in df.columns if c.startswith("consensus_over_odds_bin_granular_")],
        "consensus_under_odds_bin":          [c for c in df.columns if c.startswith("consensus_under_odds_bin_") and "_granular" not in c],
        "consensus_under_odds_bin_granular": [c for c in df.columns if c.startswith("consensus_under_odds_bin_granular_")],
        "stand":                             [c for c in df.columns if c.startswith("stand_")],
    }

    results = []
    for feat in numeric_features:
        if feat not in df.columns:
            continue
        print(f"  {feat}...", end=" ", flush=True)
        try:
            metrics = temporal_oof_xgb(df, [feat])
            results.append({"feature": feat, "model_type": "xgboost", "n_features": 1, **metrics})
            print(f"AUC={metrics['auc']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    for feat_name, feat_cols in categorical_groups.items():
        if not feat_cols:
            continue
        print(f"  {feat_name}...", end=" ", flush=True)
        try:
            metrics = temporal_oof_xgb(df, feat_cols)
            results.append({"feature": feat_name, "model_type": "xgboost", "n_features": len(feat_cols), **metrics})
            print(f"AUC={metrics['auc']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    xgb_df = pd.DataFrame(results)

    # Combine with 3a
    lr_df = pd.read_csv(SWEEP_3A_PATH)
    combined = pd.concat([lr_df, xgb_df], ignore_index=True).sort_values(["feature", "model_type"])

    # Flag XGBoost lift > 0.02 AUC
    pivot = combined.pivot(index="feature", columns="model_type", values="auc")
    if "xgboost" in pivot.columns and "logistic_regression" in pivot.columns:
        pivot["xgb_lift"] = pivot["xgboost"] - pivot["logistic_regression"]
        lifted = pivot[pivot["xgb_lift"] > 0.02]
        if not lifted.empty:
            print(f"\nFeatures with XGB lift > 0.02 AUC:")
            print(lifted[["logistic_regression","xgboost","xgb_lift"]].to_string())
        else:
            print("\nNo features with XGB lift > 0.02 AUC")

    # Print combined sorted by AUC
    print(f"\n{'='*70}")
    print(f"{'Feature':<40} {'Model':<22} {'AUC':>7} {'F1':>7}")
    print(f"{'='*70}")
    for _, row in combined.sort_values("auc", ascending=False).head(30).iterrows():
        print(f"{row['feature']:<40} {row['model_type']:<22} {row['auc']:>7.4f} {row['f1']:>7.4f}")

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
