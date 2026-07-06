"""
Fit and serialize NFL WR/TE receiving yards model artifacts for inference.

Trains OLS (BEST_FEATS, 8 features) + NegBin NB2 on the full labeled dataset,
then saves serialized artifacts locally and optionally to S3 for Lambda consumption.

Artifacts saved to ARTIFACT_DIR:
  ols_pipeline.joblib  — sklearn Pipeline (StandardScaler + LinearRegression)
  residuals.npy        — OLS training residuals, used for Bootstrap P(under)
  nb_coefs.npy         — NB2 log-link coefficient vector (const + 8 features)
  nb_alpha.npy         — NB2 dispersion parameter α (scalar array)
  meta.json            — training metadata (seasons, n, MAE, features, etc.)

Run:
  python src/nfl_rec_yards_modeling/scripts/train.py
  python src/nfl_rec_yards_modeling/scripts/train.py --upload-s3
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

LABELED_PATH  = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
ARTIFACT_DIR  = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_artifacts"
S3_BUCKET     = "the-odds-api-mt"
S3_KEY_PREFIX = "nfl/rec_yards_model/artifacts"

TARGET = "receiving_yards"

BEST_FEATS = [
    "offered_line",
    "game_total",
    "proj_own_score",
    "rec_yards_L8",
    "target_share_L8",
    "snap_pct_L8",
    "pos_TE",
    "market_under_prob",
]

KEEP_POSITIONS = ["WR", "TE"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-s3", action="store_true",
                        help="Upload artifacts to S3 after saving locally")
    args = parser.parse_args()

    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
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
    print("\n  Fitting NegBin NB2 GLM (same features)...")
    X_const   = sm.add_constant(X)
    nb_result = NegativeBinomial(y, X_const).fit(disp=False, maxiter=300)
    nb_coefs  = nb_result.params[:-1]
    nb_alpha  = float(np.exp(nb_result.lnalpha))
    converged = nb_result.mle_retvals["converged"]
    print(f"    NegBin α       : {nb_alpha:.6f}  (var = μ + α·μ²)")
    print(f"    Converged      : {converged}")
    if not converged:
        print("    WARNING: NegBin did not converge — consider increasing maxiter")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(ols, ARTIFACT_DIR / "ols_pipeline.joblib")
    np.save(ARTIFACT_DIR / "residuals.npy", residuals)
    np.save(ARTIFACT_DIR / "nb_coefs.npy",  nb_coefs)
    np.save(ARTIFACT_DIR / "nb_alpha.npy",  np.array([nb_alpha]))

    meta = {
        "train_seasons":  [int(s) for s in seasons],
        "n_rows_total":   int(len(df)),
        "n_rows_train":   int(len(sub)),
        "features":       BEST_FEATS,
        "target":         TARGET,
        "keep_positions": KEEP_POSITIONS,
        "in_sample_mae":  round(in_mae, 4),
        "residual_std":   round(float(residuals.std()), 4),
        "nb_alpha":       round(nb_alpha, 6),
        "nb_converged":   bool(converged),
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Artifacts saved → {ARTIFACT_DIR}")
    for f in sorted(ARTIFACT_DIR.iterdir()):
        print(f"    {f.name:<30}  {f.stat().st_size:>10,} bytes")

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
