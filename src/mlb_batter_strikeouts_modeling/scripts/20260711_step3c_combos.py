"""
Step 3c — Combo model training (classification + regression).

Tests combinations of features using OOF temporal CV.
Saves best model artifacts to ~/Downloads/tmp/ and S3.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

FEATURES_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_features.parquet"
OUT_PATH      = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_3c_combos.csv"
MODEL_DIR     = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_models"

S3_BUCKET    = "the-odds-api-mt"
S3_MODEL_PFX = "mlb/batter_strikeouts_model/artifacts"

N_FOLDS = 5
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMBO_SPECS = [
    # (name, numeric_features, categorical_features, model_type, target)
    ("market_only_lr",
     ["novig_prob_over", "offered_line"],
     [],
     "logistic_regression", "classification"),

    ("market_granular_lr",
     ["novig_prob_over", "offered_line"],
     ["consensus_under_odds_bin_granular", "consensus_over_odds_bin_granular"],
     "logistic_regression", "classification"),

    ("market_batter_lr",
     ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_L20"],
     [],
     "logistic_regression", "classification"),

    ("market_batter_pitcher_lr",
     ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_L20", "opp_k_rate_career"],
     [],
     "logistic_regression", "classification"),

    ("market_batter_full_lr",
     ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_L20", "opp_k_rate_career", "is_home"],
     ["consensus_under_odds_bin_granular"],
     "logistic_regression", "classification"),

    ("market_batter_xgb",
     ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_career",
      "k_roll_L5", "k_roll_L20", "opp_k_rate_career", "opp_k_rate_L5", "is_home"],
     ["consensus_under_odds_bin_granular", "consensus_over_odds_bin_granular"],
     "xgboost", "classification"),

    # Regression: predict raw strikeout count
    ("regression_market_ridge",
     ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_L20", "opp_k_rate_career"],
     [],
     "ridge", "regression"),

    ("regression_market_xgb",
     ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_career",
      "k_roll_L5", "k_roll_L20", "opp_k_rate_career", "opp_k_rate_L5", "is_home"],
     ["consensus_under_odds_bin_granular"],
     "xgboost", "regression"),
]


def encode_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
    return pd.concat([df, dummies], axis=1)


def get_dummy_cols(df: pd.DataFrame, cat_group: str) -> list[str]:
    return [c for c in df.columns if c.startswith(cat_group + "_")
            and c != cat_group]


def run_oof(df: pd.DataFrame, feature_cols: list[str], target_col: str, model_type: str):
    df_s = df.dropna(subset=feature_cols + [target_col]).sort_values("game_date").reset_index(drop=True)
    n = len(df_s)
    fold_size = n // N_FOLDS
    oof_preds = np.full(n, np.nan)
    trained_models = []

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 1)
        val_idx   = df_s.index[train_end:] if fold == N_FOLDS - 1 else df_s.index[train_end: train_end + fold_size]
        train_idx = df_s.index[:train_end]
        if len(train_idx) < 50 or len(val_idx) < 10:
            continue

        X_tr = df_s.loc[train_idx, feature_cols].values
        y_tr = df_s.loc[train_idx, target_col].values
        X_vl = df_s.loc[val_idx,   feature_cols].values

        if model_type == "logistic_regression":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_vl)[:, 1]
            trained_models.append((scaler, clf))

        elif model_type == "xgboost" and target_col == "over_flag":
            clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=42, verbosity=0)
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_vl)[:, 1]
            trained_models.append((None, clf))

        elif model_type == "ridge":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)
            reg = Ridge(alpha=1.0)
            reg.fit(X_tr, y_tr)
            oof_preds[val_idx] = reg.predict(X_vl)
            trained_models.append((scaler, reg))

        elif model_type == "xgboost" and target_col == "strikeouts":
            reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=0)
            reg.fit(X_tr, y_tr)
            oof_preds[val_idx] = reg.predict(X_vl)
            trained_models.append((None, reg))

    valid = ~np.isnan(oof_preds)
    return df_s, oof_preds, valid, trained_models


def eval_classification(y_true, y_pred):
    y_bin = (y_pred >= 0.5).astype(int)
    return {
        "auc":       round(roc_auc_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_bin, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_bin, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_bin, zero_division=0), 4),
        "rmse": np.nan, "mae": np.nan, "r2": np.nan,
    }


def eval_regression(y_true, y_pred):
    return {
        "auc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }


def train_full_model(df: pd.DataFrame, feature_cols: list[str], target_col: str, model_type: str):
    """Train on all data for the production artifact."""
    df_s = df.dropna(subset=feature_cols + [target_col])
    X = df_s[feature_cols].values
    y = df_s[target_col].values

    if model_type == "logistic_regression":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X, y)
        return {"scaler": scaler, "model": clf, "feature_cols": feature_cols, "model_type": model_type, "target": target_col}
    elif model_type == "xgboost" and target_col == "over_flag":
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=42, verbosity=0)
        clf.fit(X, y)
        return {"scaler": None, "model": clf, "feature_cols": feature_cols, "model_type": model_type, "target": target_col}
    elif model_type == "ridge":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        reg = Ridge(alpha=1.0)
        reg.fit(X, y)
        return {"scaler": scaler, "model": reg, "feature_cols": feature_cols, "model_type": model_type, "target": target_col}
    elif model_type == "xgboost" and target_col == "strikeouts":
        reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0)
        reg.fit(X, y)
        return {"scaler": None, "model": reg, "feature_cols": feature_cols, "model_type": model_type, "target": target_col}


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)

    for col in ["consensus_over_odds_bin", "consensus_over_odds_bin_granular",
                "consensus_under_odds_bin", "consensus_under_odds_bin_granular", "stand"]:
        df = encode_categorical(df, col)

    results = []
    best_auc = 0.0
    best_combo = None

    for name, num_feats, cat_groups, model_type, task in COMBO_SPECS:
        target = "over_flag" if task == "classification" else "strikeouts"

        # Expand categorical groups to dummy column names
        cat_feat_cols = []
        for cg in cat_groups:
            cat_feat_cols.extend(get_dummy_cols(df, cg))

        feature_cols = [f for f in num_feats if f in df.columns] + cat_feat_cols
        missing = [f for f in num_feats if f not in df.columns]
        if missing:
            print(f"  SKIP {name} — missing: {missing}")
            continue

        print(f"  {name} ({len(feature_cols)} feats)...", end=" ", flush=True)
        try:
            df_s, oof_preds, valid, _ = run_oof(df, feature_cols, target, model_type)
            y_true = df_s[target].values[valid]
            y_pred = oof_preds[valid]

            if task == "classification":
                metrics = eval_classification(y_true, y_pred)
                print(f"AUC={metrics['auc']:.4f}")
            else:
                metrics = eval_regression(y_true, y_pred)
                print(f"RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

            row = {
                "combo_name": name, "model_type": model_type, "task": task,
                "n_features": len(feature_cols), "n_samples": int(valid.sum()),
                "features_included": str(num_feats + cat_groups),
                **metrics
            }
            results.append(row)

            if task == "classification" and metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_combo = (name, feature_cols, model_type, task, target)

        except Exception as e:
            print(f"ERROR: {e}")

    out = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"{'Combo':<35} {'Model':<22} {'Task':<15} {'AUC':>7} {'RMSE':>7} {'R²':>7}")
    print(f"{'='*80}")
    for _, row in out.iterrows():
        auc_str  = f"{row['auc']:.4f}"  if pd.notna(row['auc'])  else "    —"
        rmse_str = f"{row['rmse']:.4f}" if pd.notna(row['rmse']) else "    —"
        r2_str   = f"{row['r2']:.4f}"   if pd.notna(row['r2'])   else "    —"
        print(f"{row['combo_name']:<35} {row['model_type']:<22} {row['task']:<15} {auc_str:>7} {rmse_str:>7} {r2_str:>7}")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved results: {OUT_PATH}")

    # Save best classification model
    if best_combo:
        name, feature_cols, model_type, task, target = best_combo
        print(f"\nBest classification model: {name} (AUC={best_auc:.4f})")
        artifact = train_full_model(df, feature_cols, target, model_type)
        artifact["combo_name"] = name

        local_path = MODEL_DIR / f"model_{name}.pkl"
        with open(local_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved model: {local_path}")

        s3c = boto3.client("s3")
        buf = BytesIO()
        pickle.dump(artifact, buf)
        s3_key = f"{S3_MODEL_PFX}/model_{name}.pkl"
        s3c.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf.getvalue())
        print(f"Saved S3:    s3://{S3_BUCKET}/{s3_key}")

    # sklearn version for Dockerfile pinning
    import sklearn
    print(f"\nsklearn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()
