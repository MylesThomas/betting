"""
Build historical NFL rest edge for seasons 2015-2025.

Uses nfl_data_py for schedule data, then runs the same compute_rest_metrics()
pipeline as the 2026 analysis. Saves per-season summary CSVs to S3.

Usage:
    python scripts/build_historical_rest_edge.py
    python scripts/build_historical_rest_edge.py --seasons 2020 2021
    python scripts/build_historical_rest_edge.py --no-cache
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_rest_edge.fetch_schedule import fetch_historical_schedule
from nfl_rest_edge.compute_rest import compute_rest_metrics, compute_team_summary

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/rest_edge"
TMP_DIR = Path.home() / "Downloads" / "tmp"


def _s3_summary_key(season: int) -> str:
    return f"{S3_PREFIX}/{season}/summary_{season}.csv"


def _read_summary_from_s3(season: int) -> pd.DataFrame | None:
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_s3_summary_key(season))
        return pd.read_csv(StringIO(obj["Body"].read().decode()))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def build_season(season: int, use_cache: bool = True) -> pd.DataFrame:
    """Compute and return the per-team rest summary for one season."""
    # Check if summary already in S3
    if use_cache:
        cached = _read_summary_from_s3(season)
        if cached is not None:
            print(f"  {season}: summary loaded from S3")
            return cached

    print(f"  {season}: fetching schedule...", end=" ", flush=True)
    schedule = fetch_historical_schedule(season, use_cache=use_cache)
    print(f"{len(schedule)} games")

    team_games = compute_rest_metrics(schedule, season=season)
    summary = compute_team_summary(team_games)
    summary["season"] = season

    # Save to S3
    s3 = boto3.client("s3")
    for key, df, idx in [
        (f"{S3_PREFIX}/{season}/team_games_{season}.csv", team_games, False),
        (_s3_summary_key(season), summary, False),
    ]:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=idx).encode())

    # Local mirror
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for fname, df, idx in [
        (f"team_games_{season}.csv", team_games, False),
        (f"rest_summary_{season}.csv", summary, False),
    ]:
        (TMP_DIR / fname).write_bytes(df.to_csv(index=idx).encode())

    best = summary.iloc[0]
    worst = summary.iloc[-1]
    swing = int(best["net_rest"]) - int(worst["net_rest"])
    print(f"    best: {best['team']} +{int(best['net_rest'])}, worst: {worst['team']} {int(worst['net_rest'])}, swing: {swing}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2015, 2026)))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print(f"Building historical rest edge: seasons {args.seasons[0]}–{args.seasons[-1]}")
    all_summaries = []
    for season in args.seasons:
        summary = build_season(season, use_cache=not args.no_cache)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)

    # Save combined net_rest table
    net_rest_cols = ["season", "team", "net_rest", "rest_adv_games", "rest_disadv_games",
                     "short_week_road", "negated_bye", "in_4_in_17"]
    combined_slim = combined[net_rest_cols]

    key = f"{S3_PREFIX}/net_rest_all.csv"
    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=combined_slim.to_csv(index=False).encode())
    local = TMP_DIR / "net_rest_all.csv"
    local.write_bytes(combined_slim.to_csv(index=False).encode())

    print(f"\nCombined net rest table: {len(combined_slim)} rows")
    print(f"  s3://{S3_BUCKET}/{key}")
    print(f"  {local}")

    print("\nTop net rest by season (best + worst):")
    for season, grp in combined_slim.groupby("season"):
        grp = grp.sort_values("net_rest", ascending=False)
        best = grp.iloc[0]
        worst = grp.iloc[-1]
        swing = int(best["net_rest"]) - int(worst["net_rest"])
        print(f"  {season}: {best['team']} +{int(best['net_rest'])}  /  {worst['team']} {int(worst['net_rest'])}  (swing {swing})")


if __name__ == "__main__":
    main()
