"""
Step 3c — OLS-bootstrap combo model for MLB pitcher_walks.

Follows the strikeouts v5 pattern:
  - Deduplicate spine to one row per (player_key, game_date) for OLS regression
  - Test various feature combos with OOF (train 2025, test 2026)
  - Select best combo by OOS AUC (bootstrap P(over))
  - Save final model + residuals to S3

Post-spine filter: abs(home_run_line_point) <= 2.0

Artifacts:
  S3:    mlb/pitcher_walks_model/model/mlb_pitcher_walks_model.joblib
         mlb/pitcher_walks_model/model/mlb_pitcher_walks_residuals.npy
  Local: ~/Downloads/tmp/mlb_pitcher_walks_step3c_results.csv
         ~/Downloads/tmp/mlb_pitcher_walks_oof_preds.parquet

Usage:
  python src/mlb_pitcher_walks_modeling/scripts/20260706_step3c_model.py
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_pitcher_walks_spine.parquet"
OUT_DIR     = Path.home() / "Downloads/tmp"
MODELS_DIR  = REPO_ROOT / "models"
S3_BUCKET   = "the-odds-api-mt"
MODEL_KEY   = "mlb/pitcher_walks_model/model/mlb_pitcher_walks_model.joblib"
RESID_KEY   = "mlb/pitcher_walks_model/model/mlb_pitcher_walks_residuals.npy"
META_KEY    = "mlb/pitcher_walks_model/model/mlb_pitcher_walks_meta.json"

TARGET = "walks"
SPOT_CHECK = "642547.0_freddy_peralta"

CAT_FEATURES = {"over_price_bucket_fine", "under_price_bucket_fine",
                "consensus_over_odds_bin", "consensus_under_odds_bin",
                "consensus_over_odds_bin_granular", "consensus_under_odds_bin_granular"}

OOF_FOLDS = [
    (2026, [2025]),         # primary OOS fold
    (2025, [2024, 2025]),   # secondary (2024 has only 21 games — minimal impact)
]

# Feature combos to evaluate
COMBOS = [
    # Baseline: just rolling walks
    ["walks_roll_career"],
    ["walks_roll_career", "walks_roll_season"],
    ["walks_roll_career", "walks_roll_c5"],
    # + line
    ["walks_roll_career", "consensus_line"],
    ["walks_roll_career", "walks_roll_season", "consensus_line"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line"],
    # + game context
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "is_home"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "is_home", "games_into_season"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "is_home", "days_rest"],
    ["walks_roll_career", "walks_roll_season", "consensus_line", "is_home", "games_into_season"],
    # + market features (book-invariant odds bins)
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "over_price_bucket_fine"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "under_price_bucket_fine"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "over_price_bucket_fine", "under_price_bucket_fine"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "is_home",
     "over_price_bucket_fine", "under_price_bucket_fine"],
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "is_home", "games_into_season",
     "over_price_bucket_fine", "under_price_bucket_fine"],
    # + min/max line
    ["walks_roll_career", "walks_roll_c5", "consensus_line", "min_line", "max_line"],
    ["walks_roll_career", "consensus_line", "min_line", "max_line",
     "over_price_bucket_fine", "under_price_bucket_fine"],
    # Full kitchen sink
    ["walks_roll_career", "walks_roll_season", "walks_roll_c5",
     "consensus_line", "min_line", "max_line", "is_home", "games_into_season",
     "over_price_bucket_fine", "under_price_bucket_fine"],
    # Without line features (purer player-history model)
    ["walks_roll_career", "walks_roll_c5", "is_home", "games_into_season"],
    ["walks_roll_career", "walks_roll_c5", "is_home", "over_price_bucket_fine", "under_price_bucket_fine"],
]


def encode_cat_columns(df: pd.DataFrame, cat_cols: set[str]) -> pd.DataFrame:
    """Label-encode categorical bucket columns in-place (new _enc suffix columns)."""
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            continue
        enc_col = c + "_enc"
        le = LabelEncoder()
        valid = df[c].dropna()
        le.fit(valid)
        df[enc_col] = df[c].map(lambda x, le=le: le.transform([x])[0] if pd.notna(x) else -1).astype(float)
    return df


def remap_cat_features(features: list[str]) -> list[str]:
    """Replace raw cat feature names with their _enc equivalents."""
    return [f + "_enc" if f in CAT_FEATURES else f for f in features]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def bootstrap_p_over(yhat: float, line: float, residuals: np.ndarray, n_boot: int = 10000, rng_seed: int = 42) -> float:
    rng = np.random.default_rng(rng_seed)
    draws = yhat + rng.choice(residuals, size=n_boot, replace=True)
    return float((draws > line).mean())


def run_ols_combo(
    df: pd.DataFrame,
    features: list[str],
    n_boot: int = 10000,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """OOF OLS regression with bootstrap P(over) computation."""
    fold_mets = []
    all_preds: list[pd.DataFrame] = []
    all_residuals: list[float] = []

    for test_season, train_seasons in OOF_FOLDS:
        train = df[df["season"].isin(train_seasons)].dropna(subset=features + [TARGET, "line"])
        test  = df[df["season"] == test_season].dropna(subset=features + [TARGET, "line"])
        if len(train) < 50 or len(test) < 20:
            continue

        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(train[features], train[TARGET])

        resid = (train[TARGET].values - pipe.predict(train[features])).tolist()
        all_residuals.extend(resid)

        yhat_te = pipe.predict(test[features])
        reg_met = regression_metrics(test[TARGET].values, yhat_te)

        # Bootstrap P(over) per row using training residuals accumulated so far
        resid_arr = np.array(all_residuals)
        rng = np.random.default_rng(42)
        probs_over = []
        for yh, ln in zip(yhat_te, test["line"].values):
            draws = yh + rng.choice(resid_arr, size=n_boot, replace=True)
            probs_over.append(float((draws > ln).mean()))

        probs_arr = np.array(probs_over)
        y_bin = test["target_over"].values.astype(int)
        try:
            auc = float(roc_auc_score(y_bin, probs_arr))
        except Exception:
            auc = float("nan")

        fold_mets.append({**reg_met, "oos_auc": auc, "season": test_season})
        all_preds.append(pd.DataFrame({
            "player_key": test["player_key"].values,
            "game_date":  test["game_date"].values,
            "season":     test_season,
            TARGET:       test[TARGET].values,
            "line":       test["line"].values,
            "target_over": y_bin,
            "yhat":       yhat_te,
            "p_model":    probs_arr,
        }))

    if not fold_mets:
        empty = {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "oos_auc": float("nan")}
        return empty, pd.DataFrame(), np.array([])

    avg   = {k: float(np.nanmean([m[k] for m in fold_mets if k in m])) for k in ["n", "rmse", "mae", "r2", "oos_auc"]}
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return avg, preds, np.array(all_residuals)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading spine...")
    spine = pd.read_parquet(LOCAL_SPINE)
    spine = spine[spine["home_run_line_point"].abs() <= 2.0].copy()
    print(f"Filtered spine: {len(spine):,} rows, {spine['season'].unique().tolist()}")

    # Deduplicate to one row per (player_key, game_date) for regression
    # Rolling features and line features are book-invariant at this grain
    df = spine.drop_duplicates(subset=["player_key", "game_date"], keep="first").copy()
    df = df[df[TARGET].notna()].copy()
    print(f"Deduped to player-game: {len(df):,} rows")
    for s in [2024, 2025, 2026]:
        sub = df[df["season"] == s]
        print(f"  {s}: {len(sub):,} starts, {sub['player_key'].nunique()} pitchers, avg walks={sub[TARGET].mean():.3f}")

    # Encode categorical columns
    df = encode_cat_columns(df, CAT_FEATURES)

    # Spot-check player
    fp = df[df["player_key"] == SPOT_CHECK].sort_values("game_date")
    print(f"\n── Spot-check: {SPOT_CHECK} ({len(fp)} starts) ──")
    print(fp[["game_date", "season", "walks", "walks_roll_L5", "walks_roll_career",
               "walks_roll_season", "consensus_line"]].tail(10).to_string(index=False))

    # --- Combo sweep ---
    print("\n── Step 3c: OLS combo sweep ──")
    results = []
    best_auc = -1.0
    best_preds = pd.DataFrame()
    best_residuals = np.array([])
    best_features: list[str] = []

    for combo in COMBOS:
        # Remap cat feature names to their _enc equivalents
        avail_raw = combo
        avail = remap_cat_features(avail_raw)
        avail = [f for f in avail if f in df.columns]
        if not avail or len(avail) < len(avail_raw):
            missing = [f for f in avail_raw if remap_cat_features([f])[0] not in df.columns]
            print(f"  SKIP {avail_raw[:3]}... — missing: {missing}")
            continue

        metrics, preds, residuals = run_ols_combo(df, avail)
        results.append({
            "features": str(avail),
            "n_feats": len(avail),
            **metrics,
        })
        auc = metrics.get("oos_auc", float("nan"))
        print(f"  {str(avail):<80}  AUC={auc:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_preds = preds
            best_residuals = residuals
            best_features = avail

    results_df = pd.DataFrame(results).sort_values("oos_auc", ascending=False).reset_index(drop=True)
    print("\n── Top 10 combos by OOS AUC ──")
    print(results_df[["features", "oos_auc", "rmse", "r2"]].head(10).to_string(index=False))

    print(f"\n✓ Best combo: {best_features}")
    print(f"  OOS AUC={best_auc:.4f}")

    # --- Train final model on all data (2025+2026) for production ---
    print("\nTraining final model on 2025+2026...")
    train_all = df[df["season"].isin([2025, 2026])].dropna(subset=best_features + [TARGET])
    final_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    final_pipe.fit(train_all[best_features], train_all[TARGET])
    final_residuals = (train_all[TARGET].values - final_pipe.predict(train_all[best_features]))
    print(f"  Training rows: {len(train_all):,}")
    print(f"  Residual std: {final_residuals.std():.4f}")
    print(f"  Residual mean: {final_residuals.mean():.4f}")
    coef = dict(zip(best_features, final_pipe.named_steps["lr"].coef_))
    print(f"  Coefficients: {coef}")

    # Save locally
    model_local = MODELS_DIR / "mlb_pitcher_walks_model.joblib"
    resid_local = MODELS_DIR / "mlb_pitcher_walks_residuals.npy"
    joblib.dump(final_pipe, model_local)
    np.save(resid_local, final_residuals)
    print(f"\nModel saved → {model_local}")
    print(f"Residuals saved → {resid_local}")

    # Save to S3
    s3c = boto3.client("s3")
    buf = BytesIO(); joblib.dump(final_pipe, buf); buf.seek(0)
    s3c.put_object(Bucket=S3_BUCKET, Key=MODEL_KEY, Body=buf.getvalue())
    print(f"Model uploaded → s3://{S3_BUCKET}/{MODEL_KEY}")

    resid_buf = BytesIO(); np.save(resid_buf, final_residuals); resid_buf.seek(0)
    s3c.put_object(Bucket=S3_BUCKET, Key=RESID_KEY, Body=resid_buf.getvalue())
    print(f"Residuals uploaded → s3://{S3_BUCKET}/{RESID_KEY}")

    meta = {
        "features": best_features,
        "target": TARGET,
        "oos_auc": best_auc,
        "n_train": len(train_all),
        "residual_std": float(final_residuals.std()),
    }
    s3c.put_object(Bucket=S3_BUCKET, Key=META_KEY, Body=json.dumps(meta, indent=2).encode())
    print(f"Meta uploaded → s3://{S3_BUCKET}/{META_KEY}")

    # Save OOF preds and results
    results_df.to_csv(OUT_DIR / "mlb_pitcher_walks_step3c_results.csv", index=False)
    if not best_preds.empty:
        best_preds.to_parquet(OUT_DIR / "mlb_pitcher_walks_oof_preds.parquet", index=False)
        print(f"\nOOF preds: {len(best_preds):,} rows")

    # Spot-check prediction on Freddy Peralta
    if not best_preds.empty:
        fp_preds = best_preds[best_preds["player_key"] == SPOT_CHECK].sort_values("game_date") \
            if "player_key" in best_preds.columns else pd.DataFrame()
        if not fp_preds.empty:
            print(f"\n── OOF predictions: {SPOT_CHECK} ──")
            print(fp_preds[["game_date", "season", "walks", "line", "yhat", "p_model", "target_over"]].tail(10).to_string(index=False))

    print(f"\n✓ Step 3c complete. Best OOS AUC: {best_auc:.4f}")
    print(f"  Features: {best_features}")


if __name__ == "__main__":
    main()
