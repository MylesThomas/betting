"""
Train and save production OLS + Logistic models for MLB Batter Hits.

Trains on all settled data (full data, no OOF).
Saves two joblib artifacts to S3:
  mlb/batter_hits_model/model/mlb_batter_hits_ols.joblib
  mlb/batter_hits_model/model/mlb_batter_hits_logit.joblib

Also saves the spine to S3:
  mlb/batter_hits_model/spine/mlb_batter_hits_spine.parquet

Run this once before deploying the Lambda (and re-run when the spine is rebuilt).

Usage:
  python src/mlb_batter_hits_modeling/scripts/train_model.py
  python src/mlb_batter_hits_modeling/scripts/train_model.py --dry-run  # skip S3 upload
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH  = Path(__file__).resolve().parents[1] / "config.yaml"
LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip S3 upload")
    args = parser.parse_args()

    cfg = load_config()
    s3_bucket   = cfg["data"]["s3_bucket"]
    spine_key   = cfg["data"]["spine_key"]
    ols_key     = cfg["data"]["ols_model_key"]
    logit_key   = cfg["data"]["logit_model_key"]
    ols_feats   = cfg["model"]["ols_features"]

    print("Loading spine...")
    spine = pd.read_parquet(LOCAL_SPINE)
    settled = spine[spine["hits_actual"].notna()].copy()
    settled["over_flag"] = (settled["hits_actual"] > settled["offered_line"]).astype(int)
    print(f"  {len(settled):,} settled rows")

    # ---------------------------------------------------------------
    # Stage 1: OLS on player-game level
    # ---------------------------------------------------------------
    has_dk = settled[settled["bookmaker"] == "draftkings"]
    no_dk  = settled[~settled.index.isin(has_dk.index)]
    pg = (
        pd.concat([has_dk, no_dk])
        .sort_values(["player_key", "game_date", "bookmaker"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    avail = [f for f in ols_feats if f in pg.columns]
    sub = pg[avail + ["hits_actual"]].dropna()
    X1  = sub[avail].values
    y1  = sub["hits_actual"].values

    ols = LinearRegression()
    ols.fit(X1, y1)
    yhat_is = ols.predict(X1)
    r2_is = 1 - np.sum((y1 - yhat_is)**2) / np.sum((y1 - y1.mean())**2)
    print(f"\nOLS IS r²={r2_is:.4f}  RMSE={np.sqrt(np.mean((y1-yhat_is)**2)):.4f}")
    print(f"  Coefficients: {dict(zip(avail, ols.coef_.round(4)))}")

    # ---------------------------------------------------------------
    # Stage 2: Logistic on full spine (all books, all lines)
    # ---------------------------------------------------------------
    pg["yhat_ols"] = np.nan
    pg.loc[sub.index, "yhat_ols"] = yhat_is

    settled = settled.merge(
        pg[["player_key", "game_date", "yhat_ols"]],
        on=["player_key", "game_date"], how="left",
    )

    # Assert yhat is book-invariant: same (player, game, line) must have identical yhat
    yhat_spread = (
        settled[settled["yhat_ols"].notna()]
        .groupby(["player_key", "game_date", "offered_line"])["yhat_ols"]
        .agg(lambda x: x.max() - x.min())
    )
    assert yhat_spread.max() < 1e-8, (
        f"yhat_ols is not book-invariant — max spread across books: {yhat_spread.max():.2e}. "
        f"A per-book feature is inside the OLS model. Check ols_features in config.yaml."
    )
    print(f"  yhat book-invariance check: PASS (max spread={yhat_spread.max():.2e})")

    logit_sub = settled[settled["yhat_ols"].notna() & settled["over_price"].notna()].copy()
    X2 = logit_sub[["yhat_ols", "offered_line"]].values
    y2 = logit_sub["over_flag"].values

    logit = LogisticRegression(max_iter=500, solver="lbfgs")
    logit.fit(X2, y2)
    p_is = logit.predict_proba(X2)[:, 1]
    auc_is = roc_auc_score(y2, p_is)
    print(f"\nLogistic IS AUC={auc_is:.4f}")
    print(f"  Coefficients: intercept={logit.intercept_[0]:.4f}  yhat={logit.coef_[0][0]:.4f}  line={logit.coef_[0][1]:.4f}")

    # Quick IS edge check
    logit_sub = logit_sub.copy()
    logit_sub["p_model"] = p_is
    logit_sub["under_edge"] = (1 - p_is) - (1.0 / logit_sub["under_price"].where(logit_sub["under_price"].notna()))
    sub_strat = logit_sub[
        (logit_sub["offered_line"] == 0.5) &
        logit_sub["under_edge"].notna() &
        (logit_sub["under_edge"] >= 0.02) &
        logit_sub["under_price"].notna()
    ]
    if len(sub_strat) > 0:
        under_flag = (sub_strat["hits_actual"] < sub_strat["offered_line"]).astype(int)
        pnl = np.where(under_flag == 1, sub_strat["under_price"] - 1, -1.0)
        print(f"\nIS check (0.5 UNDER edge≥2pp): n={len(sub_strat):,}  hit={under_flag.mean():.3f}  net={pnl.sum():.1f}u  ROI={pnl.sum()/len(sub_strat)*100:.2f}%")

    if args.dry_run:
        print("\nDry run — skipping S3 uploads")
        return

    # ---------------------------------------------------------------
    # Save models to S3
    # ---------------------------------------------------------------
    s3 = boto3.client("s3")

    def upload_joblib(obj, key: str) -> None:
        buf = BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        s3.put_object(Bucket=s3_bucket, Key=key, Body=buf.read())
        print(f"  Uploaded → s3://{s3_bucket}/{key}")

    print("\nUploading models to S3...")
    upload_joblib(ols, ols_key)
    upload_joblib(logit, logit_key)

    # Save spine to S3
    print("Uploading spine to S3...")
    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=s3_bucket, Key=spine_key, Body=buf.read())
    print(f"  Uploaded → s3://{s3_bucket}/{spine_key}  ({len(spine):,} rows)")

    print("\nAll artifacts saved. Lambda is ready to deploy.")


if __name__ == "__main__":
    main()
