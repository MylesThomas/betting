"""
Fit and serialize NFL tackles model artifacts for inference.

Trains OLS (market_L16_game_ctx_pos_overprob) + NegBin NB2 on the full
labeled dataset, then saves serialized artifacts locally and optionally
to S3 for Lambda consumption.

Artifacts saved to ARTIFACT_DIR (and optionally s3://S3_BUCKET/S3_KEY_PREFIX/):
  ols_pipeline.joblib  — sklearn Pipeline (StandardScaler + LinearRegression)
  residuals.npy        — OLS training residuals, used for Bootstrap P(over)
  nb_coefs.npy         — NB2 log-link coefficient vector (const + 9 features)
  nb_alpha.npy         — NB2 dispersion parameter α (scalar array)
  meta.json            — training metadata (seasons, n, MAE, features, etc.)

Run:
  python src/nfl_tackles_modeling/scripts/train.py
  python src/nfl_tackles_modeling/scripts/train.py --upload-s3
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial

warnings.filterwarnings("ignore")

LABELED_PATH    = Path.home() / "Downloads" / "tmp" / "nfl_tackles_per_book.parquet"
ARTIFACT_DIR    = Path.home() / "Downloads" / "tmp" / "nfl_tackles_artifacts"
S3_BUCKET       = "the-odds-api-mt"
S3_KEY_PREFIX   = "nfl/tackles_model/artifacts"

TARGET = "tackles_combined"

POS_GROUP_MAP = {
    "LB": "LB", "CB": "CB", "DB": "CB",
    "S":  "S",  "FS": "S",  "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}

BEST_FEATS = [
    "offered_line", "game_total", "proj_opp_score", "tackle_rate_L16",
    "pos_LB", "pos_CB", "pos_S", "pos_DL", "market_under_prob",
]

DROP_POSITIONS = ["WR", "FB"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add position dummies. market_under_prob already present in per-book dataset."""
    df = df.copy()
    df["position_group"] = df["position"].map(POS_GROUP_MAP)
    for g in ["LB", "CB", "S", "DL"]:
        df[f"pos_{g}"] = (df["position_group"] == g).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-s3", action="store_true",
                        help="Upload artifacts to S3 after saving locally")
    args = parser.parse_args()

    # ── Load + filter ─────────────────────────────────────────────────────────
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].notna() & ~df["position"].isin(DROP_POSITIONS)].copy()
    df = add_derived(df)

    seasons = sorted(df["season"].unique())
    sub     = df[BEST_FEATS + [TARGET]].dropna()
    X       = sub[BEST_FEATS].to_numpy(dtype=float)
    y       = sub[TARGET].to_numpy(dtype=float)

    print(f"\nTraining on seasons: {seasons}  ({len(df):,} rows → {len(sub):,} after dropna)")

    # ── OLS ───────────────────────────────────────────────────────────────────
    print("\n  Fitting OLS pipeline...")
    ols       = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    ols.fit(X, y)
    residuals = y - ols.predict(X)
    in_mae    = mean_absolute_error(y, ols.predict(X))
    print(f"    In-sample MAE  : {in_mae:.4f}")
    print(f"    Residual σ     : {residuals.std():.4f}")
    print(f"    Residual skew  : {pd.Series(residuals).skew():.3f}")

    # ── NegBin NB2 ───────────────────────────────────────────────────────────
    print("\n  Fitting NegBin NB2 GLM (same 9 features)...")
    X_const   = sm.add_constant(X)     # (n, 10): const + 9 features
    nb_result = NegativeBinomial(y, X_const).fit(disp=False, maxiter=300)
    nb_coefs  = nb_result.params[:-1]   # shape (10,): const + 9 features; last elem is lnalpha
    nb_alpha  = float(np.exp(nb_result.lnalpha))
    converged = nb_result.mle_retvals["converged"]
    print(f"    NegBin α       : {nb_alpha:.6f}  (var = μ + α·μ²)")
    print(f"    Converged      : {converged}")
    if not converged:
        print("    WARNING: NegBin did not converge — consider increasing maxiter")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(ols,                     ARTIFACT_DIR / "ols_pipeline.joblib")
    np.save(ARTIFACT_DIR / "residuals.npy", residuals)
    np.save(ARTIFACT_DIR / "nb_coefs.npy",  nb_coefs)
    np.save(ARTIFACT_DIR / "nb_alpha.npy",  np.array([nb_alpha]))

    meta = {
        "train_seasons":   [int(s) for s in seasons],
        "n_rows_total":    int(len(df)),
        "n_rows_train":    int(len(sub)),
        "features":        BEST_FEATS,
        "target":          TARGET,
        "drop_positions":  DROP_POSITIONS,
        "in_sample_mae":   round(in_mae, 4),
        "residual_std":    round(float(residuals.std()), 4),
        "nb_alpha":        round(nb_alpha, 6),
        "nb_converged":    bool(converged),
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Artifacts saved → {ARTIFACT_DIR}")
    for f in sorted(ARTIFACT_DIR.iterdir()):
        print(f"    {f.name:<30}  {f.stat().st_size:>10,} bytes")

    # ── Optional S3 upload ────────────────────────────────────────────────────
    if args.upload_s3:
        print(f"\n  Uploading to s3://{S3_BUCKET}/{S3_KEY_PREFIX}/...")
        s3 = boto3.client("s3")
        for f in sorted(ARTIFACT_DIR.iterdir()):
            key = f"{S3_KEY_PREFIX}/{f.name}"
            s3.upload_file(str(f), S3_BUCKET, key)
            print(f"    {f.name} → s3://{S3_BUCKET}/{key}")
        print("  Done.")

    print()


if __name__ == "__main__":
    main()
