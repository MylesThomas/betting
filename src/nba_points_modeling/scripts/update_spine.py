"""
Rebuild the NBA player points rolling-feature spine from S3 game logs and upload.

Fetches all player game logs from s3://nba-api-mt/player_game_logs/{season}/*.csv,
computes rolling features (pts_L1..L20, career, min_L5/L20, fga_L5, is_home,
days_rest, games_into_season, opp_pts_allowed_L10), and uploads the resulting
spine parquet to S3. No market data — market features (offered_line, novig_prob_over)
are joined at inference time from live API data.

Run:
    python src/nba_points_modeling/scripts/update_spine.py
    python src/nba_points_modeling/scripts/update_spine.py --verify
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

GAMELOG_BUCKET = "nba-api-mt"
GAMELOG_PREFIX = "player_game_logs"
SPINE_BUCKET   = "the-odds-api-mt"
SPINE_KEY      = "nba/points_model/spine/nba_points_spine.parquet"
SEASONS        = ["2023-24", "2024-25", "2025-26"]

ROLL_WINDOWS = [1, 3, 5, 10, 20]


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


def _parse_matchup(matchup: str):
    m = str(matchup)
    if " vs. " in m:
        parts = m.split(" vs. ")
        return parts[0].strip(), parts[1].strip(), 1
    elif " @ " in m:
        parts = m.split(" @ ")
        return parts[0].strip(), parts[1].strip(), 0
    return None, None, None


def build_spine(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    df["game_date"] = pd.to_datetime(df["GAME_DATE"], format="mixed").dt.date.astype(str)
    df["player_key"] = df["PLAYER_NAME"].apply(_normalize_name)

    parsed = df["MATCHUP"].apply(
        lambda m: pd.Series(_parse_matchup(m), index=["team_abbr", "opp_abbr", "is_home"])
    )
    df = pd.concat([df, parsed], axis=1)

    # Opponent pts allowed (rolling 10-game) — team-level, strictly before current game
    team_pts = (
        df.groupby(["game_date", "team_abbr"])["PTS"].sum()
        .reset_index()
        .rename(columns={"team_abbr": "opp_abbr", "PTS": "opp_pts_on_date"})
    )
    team_pts = team_pts.sort_values(["opp_abbr", "game_date"])
    team_pts["opp_pts_allowed_L10"] = (
        team_pts.groupby("opp_abbr")["opp_pts_on_date"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )
    df = df.merge(
        team_pts[["opp_abbr", "game_date", "opp_pts_allowed_L10"]],
        on=["opp_abbr", "game_date"],
        how="left",
    )

    df = df.sort_values(["player_key", "season", "game_date"]).reset_index(drop=True)

    # Days rest (capped at 14)
    df["prev_date"] = df.groupby("player_key")["game_date"].shift(1)
    df["days_rest"] = (
        pd.to_datetime(df["game_date"]) - pd.to_datetime(df["prev_date"])
    ).dt.days.clip(upper=14).fillna(3)

    # Games into season (0-indexed games before this one)
    df["games_into_season"] = df.groupby(["player_key", "season"]).cumcount()

    def roll_shift(series, w, min_p=None):
        if min_p is None:
            min_p = min(3, w)
        return series.shift(1).rolling(w, min_periods=min_p).mean()

    grp_season = df.groupby(["player_key", "season"], sort=False)
    grp_career = df.groupby("player_key", sort=False)

    for w in ROLL_WINDOWS:
        df[f"pts_L{w}"] = grp_season["PTS"].transform(lambda s: roll_shift(s, w))

    df["pts_career"] = grp_career["PTS"].transform(lambda s: roll_shift(s, 9999, min_p=3))
    df["min_L5"]     = grp_season["MIN"].transform(lambda s: roll_shift(s, 5))
    df["min_L20"]    = grp_season["MIN"].transform(lambda s: roll_shift(s, 20))
    df["fga_L5"]     = grp_season["FGA"].transform(lambda s: roll_shift(s, 5))

    keep = [
        "player_key", "PLAYER_NAME", "game_date", "season",
        "PTS", "MIN", "FGA", "is_home", "days_rest", "games_into_season",
        "team_abbr", "opp_abbr", "opp_pts_allowed_L10",
        "pts_L1", "pts_L3", "pts_L5", "pts_L10", "pts_L20", "pts_career",
        "min_L5", "min_L20", "fga_L5",
    ]
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
    print(f"  Players: {spine['player_key'].nunique():,}")
    for s in SEASONS:
        n = (spine["season"] == s).sum()
        null_pct = spine[spine["season"] == s]["pts_L5"].isna().mean() * 100
        print(f"  {s}: {n:,} rows  pts_L5 null={null_pct:.1f}%")

    if args.verify:
        existing = load_s3_spine()
        if existing is None:
            print("No existing S3 spine to compare against.")
        else:
            print(f"\nS3 spine rows: {len(existing):,}  | Local: {len(spine):,}")
            print(f"  Diff: {len(spine) - len(existing):+,} rows")
        return

    upload_spine(spine)
    print("Done.")


if __name__ == "__main__":
    main()
