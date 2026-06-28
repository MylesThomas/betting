"""
Fit final LR model on 2024+2025 combined data, serialize, upload to S3.

Reads from:
  ~/Downloads/tmp/nfl_sacks_features_2024.parquet
  ~/Downloads/tmp/nfl_sacks_features_2025.parquet

Outputs:
  ~/Downloads/tmp/lr_model.pkl   (local copy)
  s3://the-odds-api-mt/nfl/sacks_model/model/lr_model.pkl

Run:
  python src/nfl_sacks_modeling/scripts/fit_model.py
"""

import sys
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
TMP = Path.home() / "Downloads" / "tmp"

S3_BUCKET = "the-odds-api-mt"
S3_KEY = "nfl/sacks_model/model/lr_model.pkl"
THRESHOLD = 0.30
TRAIN_SEASONS = [2024, 2025]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    windows = cfg["rolling_windows"]
    rolling = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    market_num = [
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over",   "prop_mean_impl_under",
        "prop_min_impl_over",    "prop_max_impl_over",
        "prop_min_impl_under",   "prop_max_impl_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books",
        "fanduel_over_0p5_implied",
        "betonline_over_0p5_implied", "betonline_under_0p5_implied",
        "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
    ]
    market_cat = [
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
    ]
    numeric = rolling + ["game_total", "team_spread", "games_played_ytd"] + market_num
    categorical = ["pos_group", "pos_side"] + market_cat
    return numeric, categorical


def build_pipeline(n_cols: list[str], c_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler()),
        ]), n_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), c_cols),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])


def main():
    cfg = load_config()
    n_cols, c_cols = feature_lists(cfg)

    print(f"\nFitting LR on seasons {TRAIN_SEASONS}")
    print(f"{'='*55}")

    frames = []
    for season in TRAIN_SEASONS:
        path = TMP / f"nfl_sacks_features_{season}.parquet"
        if not path.exists():
            sys.exit(f"Missing: {path}\nRun build_sacks_features.py --season {season} first.")
        df = pd.read_parquet(path)
        df["season"] = season
        frames.append(df)
        print(f"  {season}: {len(df):,} rows  ({df['target'].notna().sum():,} with target)")

    combined = pd.concat(frames, ignore_index=True)
    train = combined[combined["target"].notna()].copy()

    X = train[[c for c in n_cols + c_cols if c in train.columns]]
    y = train["target"].astype(int)

    missing_cols = [c for c in n_cols + c_cols if c not in train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing from data: {missing_cols[:5]}...")
    n_cols_present = [c for c in n_cols if c in train.columns]
    c_cols_present = [c for c in c_cols if c in train.columns]

    pipe = build_pipeline(n_cols_present, c_cols_present)
    pipe.fit(X[n_cols_present + c_cols_present], y)

    probas = pipe.predict_proba(X[n_cols_present + c_cols_present])[:, 1]
    n_under = int((probas < THRESHOLD).sum())
    pct_under = n_under / len(probas) * 100

    n_train = len(train)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    print(f"\n  Train rows : {n_train:,}  (pos={n_pos:,}, neg={n_neg:,})")
    print(f"  Numeric    : {len(n_cols_present)} features")
    print(f"  Categorical: {len(c_cols_present)} features")
    print(f"  Threshold  : {THRESHOLD}  → {n_under:,} rows ({pct_under:.1f}%) flagged as Under bets")

    artifact = {
        "pipeline":        pipe,
        "n_cols":          n_cols_present,
        "c_cols":          c_cols_present,
        "threshold":       THRESHOLD,
        "trained_seasons": TRAIN_SEASONS,
        "n_train":         n_train,
        "n_pos":           n_pos,
        "n_neg":           n_neg,
        "pos_rate":        n_pos / n_train,
    }

    local_path = TMP / "lr_model.pkl"
    joblib.dump(artifact, local_path)
    print(f"\n  Saved local : {local_path}")

    buf = BytesIO()
    joblib.dump(artifact, buf)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf.getvalue())
    print(f"  Uploaded    : s3://{S3_BUCKET}/{S3_KEY}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
