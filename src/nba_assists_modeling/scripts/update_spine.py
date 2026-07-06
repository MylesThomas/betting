"""
Rebuild the NBA assists rolling-feature spine from S3 game logs and upload.

Fetches all player game logs from s3://nba-api-mt/player_game_logs/{season}/*.csv,
computes rolling features with strict no-lookahead (shift(1)), and uploads the
resulting spine parquet to S3.

Run:
    python src/nba_assists_modeling/scripts/update_spine.py
    python src/nba_assists_modeling/scripts/update_spine.py --verify   # compare, don't upload
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

GAMELOG_BUCKET  = "nba-api-mt"
GAMELOG_PREFIX  = "player_game_logs"
SPINE_BUCKET    = "the-odds-api-mt"
SPINE_KEY       = "nba/assists_model/spine/nba_assists_spine.parquet"
SEASONS         = ["2023-24", "2024-25", "2025-26"]

ROLL_WINDOWS    = [1, 3, 5, 10, 20]   # season-scoped
PROD_FEATURES   = ["min_line", "max_line", "ast_roll_20"]


def _s3():
    return boto3.client("s3")


def _normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def load_gamelogs() -> pd.DataFrame:
    s3 = _s3()
    frames = []
    for season in SEASONS:
        prefix = f"{GAMELOG_PREFIX}/{season}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=GAMELOG_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=GAMELOG_BUCKET, Key=obj["Key"])["Body"].read()
                df = pd.read_csv(BytesIO(body))
                df["season"] = season
                frames.append(df)
    if not frames:
        raise RuntimeError("No game log files found in S3")
    return pd.concat(frames, ignore_index=True)


def build_spine(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed").dt.date
    df["game_date"] = df["GAME_DATE"].astype(str)
    df["player_key"] = df["PLAYER_NAME"].apply(_normalize_name)
    df["is_home"]    = df["MATCHUP"].apply(lambda m: 0 if "@" in str(m) else 1)
    df["ast_min_ratio"] = np.where(df["MIN"] > 0, df["AST"] / df["MIN"], np.nan)

    df = df.sort_values(["player_key", "season", "game_date"]).reset_index(drop=True)

    grp_season = df.groupby(["player_key", "season"], sort=False)
    grp_career = df.groupby("player_key", sort=False)

    def roll_shift(series, w, min_p=None):
        if min_p is None:
            min_p = min(3, w)
        return series.shift(1).rolling(w, min_periods=min_p).mean()

    for w in ROLL_WINDOWS:
        df[f"ast_roll_{w}"] = grp_season["AST"].transform(lambda s: roll_shift(s, w))

    df["ast_roll_career"] = grp_career["AST"].transform(lambda s: roll_shift(s, 9999, min_p=3))

    keep = [
        "player_key", "PLAYER_NAME", "game_date", "season",
        "AST", "MIN", "TOV", "FG3A", "is_home", "MATCHUP",
        "ast_min_ratio",
    ] + [f"ast_roll_{w}" for w in ROLL_WINDOWS] + ["ast_roll_career"]

    return df[[c for c in keep if c in df.columns]].copy()


def upload_spine(spine: pd.DataFrame) -> None:
    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=SPINE_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    print(f"Uploaded {len(spine):,} rows → s3://{SPINE_BUCKET}/{SPINE_KEY}")


def load_s3_spine() -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=SPINE_BUCKET, Key=SPINE_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Rebuild locally and compare vs S3; do not upload")
    args = parser.parse_args()

    print("Loading game logs from S3...", flush=True)
    logs = load_gamelogs()
    print(f"  Raw rows: {len(logs):,}")

    print("Building spine...", flush=True)
    spine = build_spine(logs)
    print(f"  Spine rows: {len(spine):,}")
    print(f"  Seasons: {spine['season'].value_counts().to_dict()}")
    print(f"  Players: {spine['player_key'].nunique():,}")
    print(f"  ast_roll_20 null rate: {spine['ast_roll_20'].isna().mean()*100:.1f}%")

    if args.verify:
        existing = load_s3_spine()
        if existing is None:
            print("No existing S3 spine to compare against.")
        else:
            print(f"\nS3 spine rows: {len(existing):,}  | Local: {len(spine):,}")
            diff = len(spine) - len(existing)
            print(f"  Diff: {diff:+,} rows")
        return

    upload_spine(spine)
    print("Done.")


if __name__ == "__main__":
    main()
