"""
Normalize NCAAB game lines CSVs: Odds API team names → ESPN (canonical).

Processes all CSV files in the Odds API NCAAB game_lines folder (S3 or local).
Rewrites home_team and away_team using src.ncaab_team_name_mapping so downstream
joins (e.g. Lambda outcomes + lines) match without per-read mapping.

Context:
- Game lines are saved by fetch_historical_ncaab_season_lines.py with Odds API names.
- Outcomes use ESPN display names. Join fails when names differ (e.g. American Eagles vs American University Eagles).
- Running this script once (or after new mapping updates) normalizes stored data to ESPN names.

Usage:
    # S3 (default): list and rewrite all CSVs under game_lines/
    python scripts/normalize_ncaab_game_lines_team_names.py --s3

    # Dry run: only report what would change, do not write
    python scripts/normalize_ncaab_game_lines_team_names.py --s3 --dry-run

    # Local folder
    python scripts/normalize_ncaab_game_lines_team_names.py --local data/01_input/the-odds-api/ncaab/game_lines
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root for imports (find .gitignore per workspace rules)
def _find_project_root():
    current = Path.cwd()
    while current != current.parent:
        if (current / ".gitignore").exists():
            return current
        current = current.parent
    return Path.cwd()


_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ncaab_team_name_mapping import normalize_ncaab_team_name

# Match fetch script
S3_BUCKET = "ncaab-betting-mt"
S3_PREFIX = "data/01_input/the-odds-api/ncaab/game_lines/"


def _list_s3_csv_keys(s3_client):
    """List object keys under S3 prefix that end with .csv."""
    out = []
    pag = s3_client.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".csv"):
                out.append(k)
    return sorted(out)


def _read_s3_csv(s3_client, key):
    """Read one CSV from S3 into DataFrame."""
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(resp["Body"])


def _write_s3_csv(s3_client, key, df):
    """Write DataFrame to S3 as CSV."""
    from io import StringIO
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buf.getvalue(),
        ContentType="text/csv",
    )


def _normalize_df(df):
    """Normalize home_team and away_team in place; return (df, n_home_changed, n_away_changed)."""
    if df.empty or "home_team" not in df.columns or "away_team" not in df.columns:
        return df, 0, 0
    home_before = df["home_team"].astype(str).str.strip()
    away_before = df["away_team"].astype(str).str.strip()
    df = df.copy()
    df["home_team"] = home_before.apply(normalize_ncaab_team_name)
    df["away_team"] = away_before.apply(normalize_ncaab_team_name)
    n_home = (home_before != df["home_team"]).sum()
    n_away = (away_before != df["away_team"]).sum()
    return df, int(n_home), int(n_away)


def run_s3(dry_run: bool):
    import boto3
    s3 = boto3.client("s3")
    keys = _list_s3_csv_keys(s3)
    if not keys:
        print(f"No CSV files under s3://{S3_BUCKET}/{S3_PREFIX}")
        return
    print(f"Found {len(keys)} CSV(s) under s3://{S3_BUCKET}/{S3_PREFIX}")
    if dry_run:
        print("(dry-run: no writes)")
    total_files = 0
    total_home = 0
    total_away = 0
    for key in keys:
        df = _read_s3_csv(s3, key)
        df_norm, n_home, n_away = _normalize_df(df)
        total_files += 1
        total_home += n_home
        total_away += n_away
        if n_home or n_away:
            print(f"  {key}: home_team changes={n_home}, away_team changes={n_away}")
        if not dry_run and (n_home or n_away):
            _write_s3_csv(s3, key, df_norm)
    print(f"Processed {total_files} file(s); total home_team changes={total_home}, away_team changes={total_away}")


def run_local(local_dir: Path, dry_run: bool):
    if not local_dir.is_dir():
        print(f"Not a directory: {local_dir}")
        return
    csvs = sorted(local_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files in {local_dir}")
        return
    print(f"Found {len(csvs)} CSV(s) in {local_dir}")
    if dry_run:
        print("(dry-run: no writes)")
    total_files = 0
    total_home = 0
    total_away = 0
    for path in csvs:
        df = pd.read_csv(path)
        df_norm, n_home, n_away = _normalize_df(df)
        total_files += 1
        total_home += n_home
        total_away += n_away
        if n_home or n_away:
            print(f"  {path.name}: home_team changes={n_home}, away_team changes={n_away}")
        if not dry_run:
            df_norm.to_csv(path, index=False)
    print(f"Processed {total_files} file(s); total home_team changes={total_home}, away_team changes={total_away}")


def main():
    ap = argparse.ArgumentParser(description="Normalize NCAAB game lines team names (Odds API → ESPN).")
    ap.add_argument("--s3", action="store_true", help="Process CSVs in S3 (ncaab-betting-mt/game_lines/)")
    ap.add_argument("--local", type=Path, metavar="DIR", help="Process CSVs in local directory")
    ap.add_argument("--dry-run", action="store_true", help="Report changes only; do not write")
    args = ap.parse_args()
    if args.s3:
        run_s3(dry_run=args.dry_run)
    elif args.local is not None:
        run_local(args.local.resolve(), dry_run=args.dry_run)
    else:
        print("Use --s3 or --local DIR. See script docstring for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
