"""
Add missing bookmaker_stale column to live betting signal parquets in S3.

Older parquet files were written before we added bookmaker_stale; appending new
signals (which have the column) causes DuckDB "Set operations can only apply to
expressions with the same number of result columns". This script adds
bookmaker_stale = NA (null) to all existing rows in files that lack the column.
Evals treat NA as stale (excluded from ROI/Brier).

S3 prefix: s3://nba-betting-mt/data/04_output/live_betting_signals/player_points/

Usage:
  python tmp/add_bookmaker_stale_to_signal_parquets.py --dry-run   # default: only print what would be done
  python tmp/add_bookmaker_stale_to_signal_parquets.py             # apply changes to S3
"""

import argparse
import sys
from pathlib import Path

import boto3
import pandas as pd
from io import BytesIO

# Repo root for imports if needed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET = "nba-betting-mt"
S3_PREFIX = "data/04_output/live_betting_signals/player_points"
COLUMN_TO_ADD = "bookmaker_stale"


def list_signal_parquets(s3_client) -> list[str]:
    """List all parquet keys under the signals prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)
    return sorted(keys)


def add_column_if_missing(s3_client, key: str, dry_run: bool) -> bool:
    """
    Read parquet from S3; if bookmaker_stale is missing, add it (True) and write back.
    Returns True if file was modified (or would be in dry run), False if already had column.
    """
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    if COLUMN_TO_ADD in df.columns:
        print(f"  Skip: {key} (already has '{COLUMN_TO_ADD}')")
        return False

    n_rows = len(df)
    # Nullable boolean so parquet stores NA (evals treat NA as stale)
    df[COLUMN_TO_ADD] = pd.array([pd.NA] * n_rows, dtype="boolean")

    if dry_run:
        print(f"  [DRY RUN] Would add '{COLUMN_TO_ADD}=NA' to {key} ({n_rows} rows)")
        return True

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buffer.getvalue())
    print(f"  Updated {key} ({n_rows} rows)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add bookmaker_stale to S3 signal parquets that lack it")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Only print what would be done (default: True)")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes to S3")
    args = parser.parse_args()
    dry_run = not args.apply

    if not dry_run:
        print("Applying changes to S3 (--apply).")
    else:
        print("Dry run (use --apply to write to S3).")

    s3 = boto3.client("s3", region_name="us-east-2")
    keys = list_signal_parquets(s3)
    print(f"Found {len(keys)} parquet file(s) under s3://{S3_BUCKET}/{S3_PREFIX}/")

    updated = 0
    for key in keys:
        if add_column_if_missing(s3, key, dry_run):
            updated += 1

    print(f"\n{'Would update' if dry_run else 'Updated'}: {updated} file(s). Skipped (already have column): {len(keys) - updated}.")


if __name__ == "__main__":
    main()
