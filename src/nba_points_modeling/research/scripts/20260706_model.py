"""
NBA Player Points — Step 3: Regression Model Training
=======================================================
Predict pts_actual (continuous) via OLS + XGBoost regression, OOF walk-forward.

3a: Individual predictors — OLS (RMSE, MAE, R²)
3b: Individual predictors — XGBoost regressor
3c: Combo models; best model saved with training residuals for bootstrap inference

OOF design: temporal walk-forward
  Fold 1: train 2023-24, test 2024-25
  Fold 2: train 2023-24 + 2024-25, test 2025-26

Artifacts saved:
  models/nba_points_model_ols.joblib   — sklearn Pipeline (scaler + LinearRegression)
  models/nba_points_residuals.npy      — OLS training residuals (for bootstrap)
  models/nba_points_meta.json          — training metadata

Outputs:
  ~/Downloads/tmp/points_eda/step3a_individual_ols.csv
  ~/Downloads/tmp/points_eda/step3b_individual_xgb.csv
  ~/Downloads/tmp/points_eda/step3c_combos.csv
  ~/Downloads/tmp/points_eda/step3_oof_predictions.parquet  — yhat per row
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR    = Path.home() / "Downloads/tmp/points_eda"
MODELS_DIR = REPO_ROOT / "models"
SPINE_PATH = OUT_DIR / "points_spine.parquet"
SPOT_CHECK = "stephen curry"

SEASONS = ["2023-24", "2024-25", "2025-26"]

OOF_FOLDS = [
    ("2024-25", ["2023-24"]),
    ("2025-26", ["2023-24", "2024-25"]),
]

ALL_FEATURES = [
    "pts_L1", "pts_L3", "pts_L5", "pts_L10", "pts_L20", "pts_career",
    "min_L5", "min_L20", "fga_L5",
    "is_home", "days_rest", "games_into_season",
    "opp_pts_allowed_L10",
    "offered_line", "novig_prob_over",
]

TARGET = "pts_actual"


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    n    = len(y_true)
    return {"n_samples": n, "rmse": rmse, "mae": mae, "r2": r2}


def run_ols_oof(df: pd.DataFrame, features: str | list[str]):
    feats = [features] if isinstance(features, str) else list(features)
    all_preds = []
    fold_metrics = []
    all_residuals = []

    for test_season, train_seasons in OOF_FOLDS:
        train = df[df["season"].isin(train_seasons)].dropna(subset=feats + [TARGET])
        test  = df[df["season"] == test_season].dropna(subset=feats + [TARGET])
        if len(train) < 100 or len(test) < 50:
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        pipe.fit(train[feats], train[TARGET])

        train_yhat = pipe.predict(train[feats])
        residuals  = (train[TARGET].values - train_yhat).tolist()
        all_residuals.extend(residuals)

        yhat = pipe.predict(test[feats])
        fold_metrics.append(regression_metrics(test[TARGET].values, yhat))
        all_preds.append(pd.DataFrame({
            "player_key": test["player_key"].values,
            "game_date":  test["game_date"].values,
            "season":     test["season"].values,
            TARGET:       test[TARGET].values,
            "yhat":       yhat,
            "fold":       test_season,
            "features":   str(feats),
        }))

    if not fold_metrics:
        return {"n_samples": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}, \
               pd.DataFrame(), np.array([])

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return avg, preds, np.array(all_residuals)


def run_xgb_oof(df: pd.DataFrame, features: str | list[str]):
    feats = [features] if isinstance(features, str) else list(features)
    all_preds = []
    fold_metrics = []
    all_residuals = []

    for test_season, train_seasons in OOF_FOLDS:
        train = df[df["season"].isin(train_seasons)].dropna(subset=feats + [TARGET])
        test  = df[df["season"] == test_season].dropna(subset=feats + [TARGET])
        if len(train) < 100 or len(test) < 50:
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=0.5,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(train[feats], train[TARGET])

        train_yhat = model.predict(train[feats])
        all_residuals.extend((train[TARGET].values - train_yhat).tolist())

        yhat = model.predict(test[feats])
        fold_metrics.append(regression_metrics(test[TARGET].values, yhat))
        all_preds.append(pd.DataFrame({
            "player_key": test["player_key"].values,
            "game_date":  test["game_date"].values,
            "season":     test["season"].values,
            TARGET:       test[TARGET].values,
            "yhat":       yhat,
            "fold":       test_season,
            "features":   str(feats),
        }))

    if not fold_metrics:
        return {"n_samples": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}, \
               pd.DataFrame(), np.array([])

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return avg, preds, np.array(all_residuals)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading spine...", flush=True)
    df = pd.read_parquet(SPINE_PATH)
    settled = df[df["pts_actual"].notna()].copy()
    print(f"  Settled rows: {len(settled):,}")
    for s in SEASONS:
        print(f"  {s}: {len(settled[settled['season']==s]):,}")

    # ── Step 3a: Individual OLS ───────────────────────────────────────────────
    print("\n── STEP 3a: Individual OLS (regression) ──", flush=True)
    rows_3a = []
    for feat in ALL_FEATURES:
        result, _, _ = run_ols_oof(settled, feat)
        rows_3a.append({"feature": feat, "model_type": "ols", **result,
                         "auc": float("nan"), "precision": float("nan"),
                         "recall": float("nan"), "f1": float("nan"), "coefficient": float("nan")})
        print(f"  {feat:<25}: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R²={result['r2']:.4f}")

    df_3a = pd.DataFrame(rows_3a).sort_values("rmse")
    df_3a.to_csv(OUT_DIR / "step3a_individual_ols.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/step3a_individual_ols.csv")

    # ── Step 3b: Individual XGBoost ───────────────────────────────────────────
    print("\n── STEP 3b: Individual XGBoost (regression) ──", flush=True)
    rows_3b = []
    for feat in ALL_FEATURES:
        result, _, _ = run_xgb_oof(settled, feat)
        rows_3b.append({"feature": feat, "model_type": "xgboost", **result,
                         "auc": float("nan"), "precision": float("nan"),
                         "recall": float("nan"), "f1": float("nan"), "coefficient": float("nan")})
        print(f"  {feat:<25}: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R²={result['r2']:.4f}")

    df_3b = pd.DataFrame(rows_3b).sort_values("rmse")
    df_3b.to_csv(OUT_DIR / "step3b_individual_xgb.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/step3b_individual_xgb.csv")

    print("\n── XGBoost vs OLS RMSE delta ──")
    merged = df_3a[["feature", "rmse"]].merge(df_3b[["feature", "rmse"]], on="feature", suffixes=("_ols", "_xgb"))
    merged["delta_rmse"] = merged["rmse_xgb"] - merged["rmse_ols"]  # negative = XGB better
    print(merged.sort_values("delta_rmse").to_string(index=False))

    # ── Step 3c: Combo models ─────────────────────────────────────────────────
    print("\n── STEP 3c: Combo models ──", flush=True)

    top_feats = df_3a.head(5)["feature"].tolist()
    print(f"  Top 5 features by RMSE: {top_feats}")

    combos = [
        # Non-market features only
        [f for f in top_feats if f not in ("novig_prob_over", "offered_line")],
        # Top features including market
        top_feats,
        # All features
        [f for f in ALL_FEATURES if f not in ("novig_prob_over", "offered_line")],
        ALL_FEATURES,
        # Rolling PTS window sweep
        ["pts_L5", "pts_career", "min_L5", "opp_pts_allowed_L10", "offered_line", "novig_prob_over"],
        ["pts_L10", "pts_career", "min_L5", "opp_pts_allowed_L10", "offered_line", "novig_prob_over"],
        ["pts_L20", "pts_career", "min_L5", "opp_pts_allowed_L10", "offered_line", "novig_prob_over"],
        ["pts_L3", "pts_career", "min_L5", "fga_L5", "opp_pts_allowed_L10", "offered_line", "novig_prob_over"],
    ]

    rows_3c = []
    best_rmse = float("inf")
    best_model_fn = None
    best_feats = None
    best_preds = None
    best_residuals = None

    for feats in combos:
        if len(feats) == 0:
            continue
        for model_type, fn in [("ols", run_ols_oof), ("xgboost", run_xgb_oof)]:
            result, preds, residuals = fn(settled, feats)
            rows_3c.append({
                "features_included": str(feats),
                "n_features": len(feats),
                "model_type": model_type,
                **result,
                "auc": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"),
                "rationale": f"{model_type} on {len(feats)} features",
            })
            print(f"  [{model_type:<7}] {len(feats)} feats: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R²={result['r2']:.4f}")

            if result["rmse"] < best_rmse and not np.isnan(result["rmse"]):
                best_rmse = result["rmse"]
                best_model_fn = fn
                best_feats = feats
                best_preds = preds
                best_residuals = residuals

    df_3c = pd.DataFrame(rows_3c).sort_values("rmse")
    df_3c.to_csv(OUT_DIR / "step3c_combos.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/step3c_combos.csv")
    print(f"\nBest combo: RMSE={best_rmse:.4f}, features={best_feats}")

    # ── Train best model on all settled data, save artifacts ─────────────────
    print("\nTraining best model on all settled data...", flush=True)
    if best_feats:
        full_data = settled.dropna(subset=best_feats + [TARGET])

        if best_model_fn == run_xgb_oof:
            final_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=0.5,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            final_model.fit(full_data[best_feats], full_data[TARGET])
            yhat_full = final_model.predict(full_data[best_feats])
            final_residuals = full_data[TARGET].values - yhat_full
            joblib.dump(final_model, MODELS_DIR / "nba_points_model_xgb.joblib")
        else:
            final_model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
            final_model.fit(full_data[best_feats], full_data[TARGET])
            yhat_full = final_model.predict(full_data[best_feats])
            final_residuals = full_data[TARGET].values - yhat_full
            joblib.dump(final_model, MODELS_DIR / "nba_points_model_ols.joblib")

        np.save(MODELS_DIR / "nba_points_residuals.npy", final_residuals)

        meta = {
            "features": best_feats,
            "target": TARGET,
            "model_type": "xgboost" if best_model_fn == run_xgb_oof else "ols",
            "n_rows_train": int(len(full_data)),
            "oof_rmse": round(best_rmse, 4),
            "residual_std": round(float(final_residuals.std()), 4),
            "residual_mean": round(float(final_residuals.mean()), 4),
        }
        (MODELS_DIR / "nba_points_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Model artifacts saved to {MODELS_DIR}")
        print(f"  Residual σ: {final_residuals.std():.4f}  mean: {final_residuals.mean():.4f}")
        print(f"  n_residuals: {len(final_residuals):,}")

    # ── Save OOF predictions ──────────────────────────────────────────────────
    if best_preds is not None and len(best_preds) > 0:
        best_preds.to_parquet(OUT_DIR / "step3_oof_predictions.parquet", index=False)
        print(f"  Saved OOF predictions → {OUT_DIR}/step3_oof_predictions.parquet")
        print(f"  OOF rows: {len(best_preds):,}")

        # Residuals from OOF training folds
        np.save(OUT_DIR / "step3_oof_residuals.npy", best_residuals)
        print(f"  Saved OOF residuals → {OUT_DIR}/step3_oof_residuals.npy  ({len(best_residuals):,} values)")

        # Spot-check Curry
        curry = best_preds[best_preds["player_key"] == SPOT_CHECK]
        if len(curry) > 0:
            print(f"\n── Spot-check {SPOT_CHECK} OOF predictions (last 8) ──")
            print(curry[["game_date", "season", TARGET, "yhat"]].tail(8).to_string(index=False))
            print(f"  mean yhat={curry['yhat'].mean():.2f}  mean pts_actual={curry[TARGET].mean():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
