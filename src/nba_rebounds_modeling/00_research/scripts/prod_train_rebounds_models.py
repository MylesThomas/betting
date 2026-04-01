"""
Train OLS + XGB on rebounds feature parquet; write artifacts for prod_score_rebounds_slate.py (same repo).

Context:
- Features: B_min_max 6 columns (rebounds_feature_spec.py), same as v3/v5.
- No API calls; reads local/S3-mounted parquet path from CLI or config override.
- Optional S3 upload of artifact directory when config s3_bucket is non-null.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/prod_train_rebounds_models.py \\
        --config config/nba_rebounds_prod.yaml \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --output-dir ~/Downloads/tmp/rebounds_prod_models/run_001
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import yaml


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

from src.config_loader import get_project_root  # noqa: E402
from src.nba_rebounds_modeling.rebounds_feature_spec import (  # noqa: E402
    B_MIN_MAX_FEATS,
    GROUP_KEYS,
    TARGET,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train rebounds OLS + XGB for prod.")
    p.add_argument("--config", type=str, default="", help="YAML config (see config/nba_rebounds_prod.example.yaml).")
    p.add_argument("--feat", type=str, required=True, help="rebounds_model_features_v2.parquet path.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory for manifest + model files.")
    return p.parse_args()


def load_yaml_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty YAML: {path}")
    return raw


def try_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(get_project_root()),
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def fit_xgb_same_as_v5(X: pd.DataFrame, y: pd.Series):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=69,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def maybe_upload_dir(local_dir: Path, bucket: str, prefix: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    for fp in local_dir.iterdir():
        if fp.is_file():
            key = f"{prefix.rstrip('/')}/{fp.name}"
            body = fp.read_bytes()
            s3.put_object(Bucket=bucket, Key=key, Body=body)
            print(f"uploaded s3://{bucket}/{key}")


def main() -> None:
    args = parse_args()
    feat_path = Path(args.feat).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict = {}
    if args.config.strip():
        cfg = load_yaml_config(Path(args.config).expanduser())

    feature_columns = cfg["feature_columns"] if cfg else B_MIN_MAX_FEATS
    if list(feature_columns) != list(B_MIN_MAX_FEATS):
        raise ValueError(
            "feature_columns in config must match rebounds_feature_spec.B_MIN_MAX_FEATS exactly "
            f"(got {feature_columns})"
        )

    train_end = cfg["train_end_date"] if cfg else None
    sigma_floor = float(cfg["sigma_floor"]) if cfg else 0.25
    prod_shrink = float(cfg["prod_shrink"]) if cfg else 0.0
    prod_min_edge = float(cfg["prod_min_edge"]) if cfg else 0.05
    sigma_col = cfg["sigma_column"] if cfg else "roll_reb_std_5"

    df = pd.read_parquet(feat_path)
    cols_needed = list(B_MIN_MAX_FEATS) + [TARGET] + GROUP_KEYS
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"feat parquet missing column: {c}")

    work = df.dropna(subset=cols_needed).copy()
    if train_end is not None:
        end_ts = pd.Timestamp(str(train_end)).normalize()
        work["__d"] = pd.to_datetime(work["date"]).dt.normalize()
        work = work.loc[work["__d"] <= end_ts].drop(columns=["__d"])

    X = work[B_MIN_MAX_FEATS].astype(float)
    y = work[TARGET].astype(float)
    X_const = sm.add_constant(X, has_constant="add")

    ols = sm.OLS(y, X_const).fit()
    with open(out_dir / "ols_model.pkl", "wb") as f:
        pickle.dump(ols, f)

    xgb_model = fit_xgb_same_as_v5(X, y)
    xgb_model.save_model(str(out_dir / "xgb_model.json"))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": try_git_sha(),
        "feat_path": str(feat_path),
        "n_train_rows": int(len(work)),
        "date_min": str(pd.to_datetime(work["date"]).min().date()),
        "date_max": str(pd.to_datetime(work["date"]).max().date()),
        "feature_columns": B_MIN_MAX_FEATS,
        "target": TARGET,
        "group_keys": GROUP_KEYS,
        "sigma_column": sigma_col,
        "sigma_floor": sigma_floor,
        "prod_shrink": prod_shrink,
        "prod_min_edge": prod_min_edge,
        "side_policy": cfg["side_policy"] if cfg else "under_only",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        "prod_train_rebounds_models",
        f"n_train={len(work):,}",
        f"output_dir={out_dir}",
        f"run_id={run_id}",
        sep=" | ",
    )

    bucket = cfg["s3_bucket"] if cfg else None
    prefix = cfg["s3_models_prefix"] if cfg else None
    if bucket and prefix:
        maybe_upload_dir(out_dir, bucket, f"{prefix}/{run_id}")
    elif bucket or prefix:
        raise ValueError("Set both s3_bucket and s3_models_prefix in config to upload, or neither.")


if __name__ == "__main__":
    main()
