"""
Score a rebounds slate: join feature slice + v3-style props, OLS + XGB yhat, Option A under_only plays.

Context:
- Loads models from prod_train_rebounds_slate output directory (ols_model.pkl, xgb_model.json, manifest.json).
- Props file must match v3_rebounds_props_raw schema (per-book lines + no-vig probs).
- Ingestion for *live* props is a separate fetch job; this script only reads parquet/CSV paths.

Locked policy defaults match manifest.json; CLI can override min_edge for experiments.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/prod_score_rebounds_slate.py \\
        --models-dir ~/Downloads/tmp/rebounds_prod_models/run_001 \\
        --feat-slice ~/Downloads/tmp/rebounds_features_slice_2025-03-15.parquet \\
        --props ~/Downloads/tmp/v3_rebounds_props_raw.parquet \\
        --slate-date 2025-03-15 \\
        --output ~/Downloads/tmp/rebounds_scored_2025-03-15.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


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

from src.nba_rebounds_modeling.option_a_scoring import (  # noqa: E402
    option_a_vector_batch,
    play_under_only_mask,
)
from src.nba_rebounds_modeling.rebounds_feature_spec import (  # noqa: E402
    B_MIN_MAX_FEATS,
    GROUP_KEYS,
    V3_PROPS_SCORE_COLS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score rebounds slate for prod.")
    p.add_argument("--models-dir", type=str, required=True, help="Directory with ols_model.pkl, xgb_model.json, manifest.json.")
    p.add_argument("--feat-slice", type=str, required=True, help="Feature rows for slate (e.g. prod_slice output).")
    p.add_argument("--props", type=str, required=True, help="v3_rebounds_props_raw-style parquet.")
    p.add_argument("--slate-date", type=str, required=True, help="YYYY-MM-DD; filters props + must match feat slice.")
    p.add_argument("--output", type=str, required=True, help="Scored parquet path.")
    p.add_argument("--min-edge", type=float, default=-1.0, help="Override manifest prod_min_edge if >= 0.")
    p.add_argument("--s3-uri", type=str, default="", help="Optional s3://bucket/key.parquet upload after write.")
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_xgb_model(path: Path):
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def write_parquet_s3(df: pd.DataFrame, s3_uri: str) -> None:
    import boto3

    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must be s3://bucket/key")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3_uri: {s3_uri}")
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"uploaded {s3_uri}")


def american_to_implied_prob_vigged(american: np.ndarray) -> np.ndarray:
    odds = american.astype(np.float64, copy=False)
    out = np.empty_like(odds, dtype=np.float64)
    neg = odds < 0
    out[neg] = (-odds[neg]) / ((-odds[neg]) + 100.0)
    out[~neg] = 100.0 / (odds[~neg] + 100.0)
    return out


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir).expanduser()
    feat_path = Path(args.feat_slice).expanduser()
    props_path = Path(args.props).expanduser()
    out_path = Path(args.output).expanduser()
    slate = pd.Timestamp(args.slate_date).normalize()

    manifest_path = models_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    prod_shrink = float(manifest["prod_shrink"])
    prod_min_edge = float(manifest["prod_min_edge"])
    if args.min_edge >= 0.0:
        prod_min_edge = float(args.min_edge)
    sigma_floor = float(manifest["sigma_floor"])
    sigma_col = manifest["sigma_column"]

    with open(models_dir / "ols_model.pkl", "rb") as f:
        ols = pickle.load(f)

    xgb_model = load_xgb_model(models_dir / "xgb_model.json")

    feat = read_table(feat_path)
    props = read_table(props_path)

    for c in GROUP_KEYS + B_MIN_MAX_FEATS:
        if c not in feat.columns:
            raise ValueError(f"feat slice missing column: {c}")
    for c in V3_PROPS_SCORE_COLS:
        if c not in props.columns:
            raise ValueError(f"props missing column: {c}")
    if sigma_col not in feat.columns:
        raise ValueError(f"feat slice missing sigma column {sigma_col}")

    fd = pd.to_datetime(feat["date"]).dt.normalize()
    feat_s = feat.loc[fd == slate].copy()
    pd_dt = pd.to_datetime(props["date"]).dt.normalize()
    props_s = props.loc[pd_dt == slate].copy()

    dup = [c for c in feat_s.columns if c in props_s.columns and c not in GROUP_KEYS]
    feat_m = feat_s.drop(columns=dup)
    base = props_s.merge(feat_m, on=GROUP_KEYS, how="inner")
    if len(base) == 0:
        raise ValueError("merge produced 0 rows; check slate-date, keys, and slice paths")

    X = base[B_MIN_MAX_FEATS].astype(float)
    X_const = sm.add_constant(X, has_constant="add")
    yhat_ols = ols.predict(X_const).to_numpy()
    yhat_xgb = xgb_model.predict(X)

    consensus = base["consensus_reb_line"].astype(float).to_numpy()
    line = base["line"].astype(float).to_numpy()
    sigma_raw = base[sigma_col].astype(float).to_numpy()
    over_odds = base["over_odds"].astype(float).to_numpy()
    under_odds = base["under_odds"].astype(float).to_numpy()
    p_book_o = american_to_implied_prob_vigged(over_odds)
    p_book_u = american_to_implied_prob_vigged(under_odds)

    ma_o, _, _, pu_o, _, eu_o = option_a_vector_batch(
        consensus, yhat_ols, line, sigma_raw, prod_shrink, p_book_o, p_book_u, sigma_floor=sigma_floor
    )
    ma_x, _, _, pu_x, _, eu_x = option_a_vector_batch(
        consensus, yhat_xgb, line, sigma_raw, prod_shrink, p_book_o, p_book_u, sigma_floor=sigma_floor
    )

    sig_used = np.maximum(sigma_raw.astype(np.float64), sigma_floor)
    play_o = play_under_only_mask(eu_o, prod_min_edge)
    play_x = play_under_only_mask(eu_x, prod_min_edge)

    out = base.copy()
    out["yhat_ols"] = yhat_ols
    out["yhat_xgb"] = yhat_xgb
    out["mean_adj_ols"] = ma_o
    out["mean_adj_xgb"] = ma_x
    out["sigma_used"] = sig_used
    out["p_under_ols"] = pu_o
    out["p_under_xgb"] = pu_x
    out["p_under_book_raw"] = p_book_u
    out["edge_under_ols"] = eu_o
    out["edge_under_xgb"] = eu_x
    out["play_under_ols"] = play_o
    out["play_under_xgb"] = play_x
    out["play_both"] = play_o & play_x
    out["play_ols_only"] = play_o & ~play_x
    out["play_xgb_only"] = play_x & ~play_o
    out["play_neither"] = ~play_o & ~play_x

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out["score_run_id"] = run_id
    out["score_manifest_run_id"] = manifest["run_id"]
    out["score_min_edge_used"] = prod_min_edge
    out["score_edge_basis"] = "raw_implied"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        "prod_score_rebounds_slate",
        f"slate={args.slate_date}",
        f"rows={len(out):,}",
        f"n_play_ols={int(play_o.sum())}",
        f"n_play_xgb={int(play_x.sum())}",
        f"output={out_path}",
        sep=" | ",
    )

    if args.s3_uri.strip():
        write_parquet_s3(out, args.s3_uri.strip())


if __name__ == "__main__":
    main()
