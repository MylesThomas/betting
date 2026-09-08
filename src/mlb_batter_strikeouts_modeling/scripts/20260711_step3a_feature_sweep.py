"""
Step 3a — Individual feature sweep (logistic regression, OOF cross-validation).

For each candidate feature independently, trains a logistic regression using
temporal OOF CV and reports AUC, precision, recall, F1.

Target: over_flag (strikeouts > line)
Grain: (player_key, game_date, bookmaker, offered_line)
OOF: 5-fold temporal split (sorted by game_date)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

FEATURES_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_features.parquet"
OUT_PATH      = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_3a_sweep.csv"

N_FOLDS = 5


def temporal_oof_cv(df: pd.DataFrame, feature_cols: list[str], target_col: str = "over_flag"):
    """5-fold temporal OOF — split on sorted game_date."""
    df_sorted = df.dropna(subset=feature_cols + [target_col]).sort_values("game_date").reset_index(drop=True)
    n = len(df_sorted)
    fold_size = n // N_FOLDS
    oof_preds = np.full(len(df_sorted), np.nan)

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 1)
        if fold == N_FOLDS - 1:
            val_idx = df_sorted.index[train_end:]
        else:
            val_idx = df_sorted.index[train_end: train_end + fold_size]
        train_idx = df_sorted.index[:train_end]

        if len(train_idx) < 50 or len(val_idx) < 10:
            continue

        X_train = df_sorted.loc[train_idx, feature_cols].values
        y_train = df_sorted.loc[train_idx, target_col].values
        X_val   = df_sorted.loc[val_idx, feature_cols].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]

    valid_mask = ~np.isnan(oof_preds)
    y_true = df_sorted[target_col].values[valid_mask]
    y_pred = oof_preds[valid_mask]
    y_bin  = (y_pred >= 0.5).astype(int)

    auc       = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall    = recall_score(y_true, y_bin, zero_division=0)
    f1        = f1_score(y_true, y_bin, zero_division=0)
    n_samples = int(valid_mask.sum())

    # Coefficient (last fold, single feature)
    if len(feature_cols) == 1:
        X_last = df_sorted[feature_cols].values
        scaler = StandardScaler()
        X_last = scaler.fit_transform(X_last)
        clf_full = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf_full.fit(X_last, df_sorted[target_col].values)
        coef = float(clf_full.coef_[0][0])
    else:
        coef = np.nan

    return {
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_samples": n_samples,
        "coefficient": round(coef, 4) if not np.isnan(coef) else np.nan,
    }


def encode_categorical(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode a categorical column, return df + new column names."""
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(df):,} rows, {df['over_flag'].mean():.3f} over rate")

    # Encode categoricals
    for col in ["stand", "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
                "consensus_under_odds_bin", "consensus_under_odds_bin_granular"]:
        df, _ = encode_categorical(df, col)

    # Candidate features
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

    # Numeric features — one at a time
    for feat in numeric_features:
        if feat not in df.columns:
            print(f"  SKIP {feat} (not in df)")
            continue
        print(f"  {feat}...", end=" ", flush=True)
        try:
            metrics = temporal_oof_cv(df, [feat])
            results.append({"feature": feat, "model_type": "logistic_regression",
                             "n_features": 1, **metrics})
            print(f"AUC={metrics['auc']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Categorical features — encode as group
    for feat_name, feat_cols in categorical_groups.items():
        if not feat_cols:
            continue
        print(f"  {feat_name} ({len(feat_cols)} dummies)...", end=" ", flush=True)
        try:
            metrics = temporal_oof_cv(df, feat_cols)
            metrics["coefficient"] = np.nan  # multi-column, no single coef
            results.append({"feature": feat_name, "model_type": "logistic_regression",
                             "n_features": len(feat_cols), **metrics})
            print(f"AUC={metrics['auc']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    out = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(f"\n{'='*65}")
    print(f"{'Feature':<40} {'AUC':>7} {'F1':>7} {'N':>8} {'Coef':>8}")
    print(f"{'='*65}")
    for _, row in out.iterrows():
        coef_str = f"{row['coefficient']:+.3f}" if pd.notna(row["coefficient"]) else "  —"
        print(f"{row['feature']:<40} {row['auc']:>7.4f} {row['f1']:>7.4f} {row['n_samples']:>8,} {coef_str:>8}")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
