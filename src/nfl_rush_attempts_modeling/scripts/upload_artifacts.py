"""
Upload model artifacts to S3 (run once before season start).

Saves:
  1. Residual CDFs (per predicted-carry bucket) — rebuilt from OOF predictions
  2. best_model.pkl — the trained Ridge model artifact

S3 outputs:
  s3://the-odds-api-mt/nfl/rush_attempts_model/artifacts/best_model.pkl
  s3://the-odds-api-mt/nfl/rush_attempts_model/artifacts/residual_cdfs.pkl

Run:
  python src/nfl_rush_attempts_modeling/scripts/upload_artifacts.py
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
from scipy import stats
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
OOF_PATH   = Path.home() / "Downloads" / "tmp" / "rush_attempts" / "oof_predictions.parquet"
MODEL_PATH = REPO_ROOT / "models" / "nfl_rush_attempts" / "best_model.pkl"

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/rush_attempts_model/artifacts"

PRED_BINS   = [0, 5, 10, 15, 20, np.inf]
PRED_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]


def build_residual_cdfs(oof: pd.DataFrame) -> dict:
    """
    Rebuild stratified KDE residual CDFs from OOF predictions.
    Returns dict: bucket_label → callable cdf(x) = P(residual <= x).
    Mirrors step4_calibration.py logic exactly.
    """
    df = oof[oof["oof_carries"].notna()].copy()
    df["residual"]    = df["carries"] - df["oof_carries"]
    df["pred_bucket"] = pd.cut(df["oof_carries"], bins=PRED_BINS,
                                labels=PRED_LABELS, right=False)

    cdfs = {}
    global_r = df["residual"].dropna().values
    print("\nResidual CDFs (OOF):")
    print(f"  {'bucket':<10} {'n':>6}  {'mean':>8}  {'std':>8}")

    for label in PRED_LABELS:
        r = df[df["pred_bucket"] == label]["residual"].dropna().values
        if len(r) < 20:
            r = global_r
        print(f"  {label:<10} {len(r):>6}  {r.mean():>+8.3f}  {r.std():>8.3f}")

        kde     = stats.gaussian_kde(r, bw_method="scott")
        std_r   = r.std()
        x_min   = r.min() - 3 * std_r
        x_max   = r.max() + 3 * std_r
        xs      = np.linspace(x_min, x_max, 500)
        pdf     = kde(xs)
        cdf_v   = np.cumsum(pdf) * (xs[1] - xs[0])
        cdf_v   = np.clip(cdf_v / cdf_v[-1], 0, 1)
        cdf_fn  = interp1d(xs, cdf_v, kind="linear",
                           bounds_error=False, fill_value=(0.0, 1.0))
        cdfs[label] = cdf_fn

    return cdfs


def upload_pkl(obj: object, key: str) -> None:
    buf = BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Uploaded → s3://{S3_BUCKET}/{key}")


def run():
    print("Loading OOF predictions...")
    oof = pd.read_parquet(OOF_PATH)
    print(f"  {len(oof):,} OOF rows  |  {oof['oof_carries'].notna().sum():,} with predictions")

    cdfs = build_residual_cdfs(oof)

    print("\nLoading model artifact...")
    with open(MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)
    print(f"  Model type: {model_artifact['model_type']}")
    print(f"  Features:   {model_artifact['features']}")

    print("\nUploading to S3...")
    upload_pkl(cdfs, f"{S3_PREFIX}/residual_cdfs.pkl")
    upload_pkl(model_artifact, f"{S3_PREFIX}/best_model.pkl")

    print("\nVerifying uploads...")
    s3 = boto3.client("s3")
    for key in [f"{S3_PREFIX}/residual_cdfs.pkl", f"{S3_PREFIX}/best_model.pkl"]:
        obj = s3.head_object(Bucket=S3_BUCKET, Key=key)
        print(f"  ✓ {key}  ({obj['ContentLength']:,} bytes)")

    print("\n=== Artifacts uploaded ===")


if __name__ == "__main__":
    run()
