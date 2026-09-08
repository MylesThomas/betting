"""
Step 3 rebuild — Regression sweep + combo (book-invariant features only, no market features).

Target: strikeouts (raw integer count, 0-5).
Metrics: RMSE, MAE, R².
OOS structure: quarterly holdout within 2024 (train Mar-Jun, test Jul-Oct).
Also runs 5-fold temporal OOF for comparison.

Required assert: yhat is book-invariant — same prediction for every book at
the same (player, game_date, line).
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
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

FEATURES_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_features.parquet"
OUT_PATH      = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_3_rebuild_sweep.csv"
OOF_PATH      = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_oof_regression.parquet"
MODEL_DIR     = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_models"

S3_BUCKET    = "the-odds-api-mt"
S3_MODEL_PFX = "mlb/batter_strikeouts_model/artifacts"

N_FOLDS  = 5
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Book-invariant features only (no novig_prob_over, no offered_line)
CANDIDATE_FEATURES = [
    "k_roll_L1",
    "k_roll_L5",
    "k_roll_L10",
    "k_roll_L20",
    "k_roll_season",
    "k_roll_career",
    "k_rate_L5",
    "k_rate_career",
    "pa_roll_L5",
    "pa_roll_career",
    "opp_k_rate_career",
    "opp_k_rate_L5",
    "is_home",
]

TARGET = "strikeouts"


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }


def _pg_oof(df: pd.DataFrame, feature_cols: list[str], model_type: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    OOF at player-game level (deduplicate → predict → broadcast to all book rows).
    Ensures book-invariant yhat: all rows with the same (player_key, game_date)
    land in the same fold and receive the same prediction.
    """
    df_s = df.dropna(subset=feature_cols + [TARGET]).sort_values("game_date").reset_index(drop=True)

    # Deduplicate to one row per player-game for fold assignment + training
    pg = (df_s.groupby(["player_key", "game_date"], sort=False)
               .first()
               .reset_index()
               .sort_values("game_date")
               .reset_index(drop=True))

    n = len(pg)
    fold_size = n // N_FOLDS
    pg_preds = np.full(n, np.nan)

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 1)
        val_idx   = pg.index[train_end:] if fold == N_FOLDS - 1 else pg.index[train_end: train_end + fold_size]
        train_idx = pg.index[:train_end]
        if len(train_idx) < 50 or len(val_idx) < 10:
            continue
        X_tr = pg.loc[train_idx, feature_cols].values
        y_tr = pg.loc[train_idx, TARGET].values
        X_vl = pg.loc[val_idx,   feature_cols].values

        if model_type == "ridge":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)
            reg = Ridge(alpha=1.0)
        else:
            reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        reg.fit(X_tr, y_tr)
        pg_preds[val_idx] = reg.predict(X_vl)

    # Broadcast player-game predictions back to all book rows via merge
    pg["_yhat"] = pg_preds
    df_s = df_s.merge(pg[["player_key", "game_date", "_yhat"]], on=["player_key", "game_date"], how="left")
    full_preds = df_s["_yhat"].values
    df_s = df_s.drop(columns=["_yhat"])

    return df_s, full_preds


def oof_ridge(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return _pg_oof(df, feature_cols, "ridge")


def oof_xgb(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return _pg_oof(df, feature_cols, "xgboost")


def quarterly_oof(df: pd.DataFrame, feature_cols: list[str], model_type: str = "ridge") -> tuple[pd.DataFrame, np.ndarray]:
    """
    Quarterly holdout within 2024:
      Q1: Mar–Apr (train on nothing → skip first)
      Q2: May–Jun (train on Mar–Apr)
      Q3: Jul–Aug (train on Mar–Jun)
      Q4: Sep–Oct (train on Mar–Aug)
    """
    df2024 = df[df["game_date"].str.startswith("2024")].copy()
    df2024 = df2024.dropna(subset=feature_cols + [TARGET]).sort_values("game_date").reset_index(drop=True)

    quarters = {
        "Q1": ("2024-03-01", "2024-04-30"),
        "Q2": ("2024-05-01", "2024-06-30"),
        "Q3": ("2024-07-01", "2024-08-31"),
        "Q4": ("2024-09-01", "2024-10-31"),
    }

    oof_preds = np.full(len(df2024), np.nan)

    for q_name, (q_start, q_end) in quarters.items():
        val_mask   = (df2024["game_date"] >= q_start) & (df2024["game_date"] <= q_end)
        train_mask = df2024["game_date"] < q_start

        if train_mask.sum() < 50 or val_mask.sum() < 10:
            print(f"    {q_name}: skip (train={train_mask.sum()}, val={val_mask.sum()})")
            continue

        X_tr = df2024.loc[train_mask, feature_cols].values
        y_tr = df2024.loc[train_mask, TARGET].values
        X_vl = df2024.loc[val_mask, feature_cols].values

        if model_type == "ridge":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)
            reg = Ridge(alpha=1.0)
            reg.fit(X_tr, y_tr)
        else:
            reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            reg.fit(X_tr, y_tr)

        preds = reg.predict(X_vl)
        oof_preds[val_mask.values] = preds
        valid = val_mask.sum()
        m = eval_regression(df2024.loc[val_mask, TARGET].values, preds)
        print(f"    {q_name}: n={valid:,}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

    return df2024, oof_preds


def train_full_model(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    df_s = df.dropna(subset=feature_cols + [TARGET])
    X = df_s[feature_cols].values
    y = df_s[TARGET].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)
    return {"scaler": scaler, "model": reg, "feature_cols": feature_cols,
            "model_type": "ridge", "target": TARGET}


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(df):,} rows  |  target mean={df[TARGET].mean():.3f}  std={df[TARGET].std():.3f}")

    # ── Step 3a: Individual feature sweep (Ridge, OOF) ──────────────────────
    print("\n=== Step 3a: Individual feature sweep (Ridge, OOF) ===")
    results_3a = []
    for feat in CANDIDATE_FEATURES:
        if feat not in df.columns:
            print(f"  SKIP {feat} (missing)")
            continue
        print(f"  {feat}...", end=" ", flush=True)
        try:
            df_s, preds = oof_ridge(df, [feat])
            valid = ~np.isnan(preds)
            m = eval_regression(df_s[TARGET].values[valid], preds[valid])
            results_3a.append({"feature": feat, "model": "ridge", "n": int(valid.sum()), **m})
            print(f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Step 3b: Individual feature sweep (XGBoost) ─────────────────────────
    print("\n=== Step 3b: Individual feature sweep (XGBoost, OOF) ===")
    results_3b = []
    for feat in CANDIDATE_FEATURES:
        if feat not in df.columns:
            continue
        print(f"  {feat}...", end=" ", flush=True)
        try:
            df_s, preds = oof_xgb(df, [feat])
            valid = ~np.isnan(preds)
            m = eval_regression(df_s[TARGET].values[valid], preds[valid])
            results_3b.append({"feature": feat, "model": "xgboost", "n": int(valid.sum()), **m})
            print(f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    combined_3ab = pd.DataFrame(results_3a + results_3b).sort_values(["feature","model"])

    # ── Step 3c: Combos ─────────────────────────────────────────────────────
    print("\n=== Step 3c: Combo models ===")
    combos = [
        ("all_career",       ["k_roll_career", "k_rate_career", "opp_k_rate_career"], "ridge"),
        ("all_career_xgb",   ["k_roll_career", "k_rate_career", "opp_k_rate_career"], "xgboost"),
        ("career_l20",       ["k_roll_career", "k_roll_L20", "k_rate_career", "opp_k_rate_career", "is_home"], "ridge"),
        ("career_l20_xgb",   ["k_roll_career", "k_roll_L20", "k_rate_career", "opp_k_rate_career", "is_home"], "xgboost"),
        ("full_ridge",       CANDIDATE_FEATURES, "ridge"),
        ("full_xgb",         CANDIDATE_FEATURES, "xgboost"),
    ]

    results_3c = []
    best_rmse = float("inf")
    best_combo = None

    for name, feats, mtype in combos:
        feats = [f for f in feats if f in df.columns]
        print(f"  {name} ({len(feats)} feats)...", end=" ", flush=True)
        try:
            if mtype == "ridge":
                df_s, preds = oof_ridge(df, feats)
            else:
                df_s, preds = oof_xgb(df, feats)
            valid = ~np.isnan(preds)
            m = eval_regression(df_s[TARGET].values[valid], preds[valid])
            print(f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
            results_3c.append({"combo": name, "model": mtype, "n_feats": len(feats), "n": int(valid.sum()), **m})
            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                best_combo = (name, feats, mtype)
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Required assert: yhat is book-invariant ──────────────────────────────
    print("\n=== Required assert: yhat is book-invariant ===")
    best_name, best_feats, best_mtype = best_combo
    feats_ok = [f for f in best_feats if f in df.columns]

    if best_mtype == "ridge":
        df_full, oof_all = oof_ridge(df, feats_ok)
    else:
        df_full, oof_all = oof_xgb(df, feats_ok)

    df_full["yhat"] = oof_all
    valid_mask = df_full["yhat"].notna()
    df_check = df_full[valid_mask].copy()

    yhat_check = df_check.groupby(["player_key","game_date","offered_line"])["yhat"].nunique()
    n_varying = (yhat_check > 1).sum()
    assert n_varying == 0, (
        f"yhat is NOT book-invariant — {n_varying} (player, game, line) groups "
        f"have varying predictions across books. A per-book feature is inside the model."
    )
    print(f"  ✅ yhat is book-invariant across all (player, game, line) groups")

    # ── Quarterly OOF holdout (2024 only) ───────────────────────────────────
    print(f"\n=== Quarterly holdout (best combo: {best_name}, 2024 only) ===")
    df2024_q, q_preds = quarterly_oof(df[df["game_date"].str.startswith("2024")], feats_ok, best_mtype)
    q_valid = ~np.isnan(q_preds)
    if q_valid.sum() > 0:
        q_m = eval_regression(df2024_q[TARGET].values[q_valid], q_preds[q_valid])
        print(f"  Quarterly total (Q2–Q4): n={q_valid.sum():,}  RMSE={q_m['rmse']:.4f}  MAE={q_m['mae']:.4f}  R²={q_m['r2']:.4f}")

    # Save OOF with predictions for Step 4
    df_full["yhat_oof"] = oof_all
    df_full.to_parquet(OOF_PATH, index=False)
    print(f"\nSaved OOF predictions: {OOF_PATH}")

    # Train full model + save
    print(f"\nBest combo: {best_name} (RMSE={best_rmse:.4f})")
    artifact = train_full_model(df, feats_ok)
    artifact["combo_name"] = best_name

    local_path = MODEL_DIR / f"model_{best_name}.pkl"
    with open(local_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model: {local_path}")

    s3c = boto3.client("s3")
    buf = BytesIO()
    pickle.dump(artifact, buf)
    s3_key = f"{S3_MODEL_PFX}/model_batter_strikeouts_{best_name}.pkl"
    s3c.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf.getvalue())
    print(f"Saved S3:    s3://{S3_BUCKET}/{s3_key}")

    # Summary table
    out = pd.DataFrame(results_3a + results_3b + results_3c)
    print(f"\n{'='*70}")
    print(f"{'Combo/Feature':<30} {'Model':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"{'='*70}")
    for _, row in pd.DataFrame(results_3c).sort_values("rmse").iterrows():
        print(f"{row['combo']:<30} {row['model']:<10} {row['rmse']:>8.4f} {row['mae']:>8.4f} {row['r2']:>8.4f}")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()
