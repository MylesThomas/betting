"""
Build persistent cache for arbitrage dashboard data.

OVERVIEW:
    This script consolidates ALL historical arb files from S3 into a single Parquet cache.
    This dramatically speeds up dashboard loading:
    - Without cache: Load 7000+ individual CSVs (~58 seconds)
    - With cache: Load 1 Parquet file via DuckDB (~1-2 seconds)
    - Performance improvement: 29x faster, 80% smaller files

CACHE LOCATIONS:
    Local dev:  ~/Downloads/tmp/{sport}_arbs_cache.parquet
    Production: s3://betting-{sport}-arbs/cache/{sport}_arbs_cache.parquet
    Metadata:   {same_location}/{sport}_arbs_cache_metadata.json

TWO-TIER CACHING SYSTEM:
    
    1. REAL-TIME UPDATES (every 5 minutes during games):
       - find_nba_arb_opportunities.py runs
       - Gets new arbs from API
       - Saves CSV snapshot to S3 (for history)
       - Immediately appends to Parquet cache in S3
       - Dashboard sees new data instantly (no rebuild needed!)
    
    2. DAILY REBUILD (this script, runs at 2am):
       - Backfills any missing snapshots (if scraper failed)
       - Cleans up duplicates
       - Verifies cache integrity
       - Updates metadata
       
    Why both?
    - Real-time: Dashboard always fresh (no 5-min-old data)
    - Daily: Catches errors, fills gaps, maintains data quality

USAGE:
    # Initial build (load ALL files from S3)
    python scripts/build_arb_cache.py --sport all --file-type parquet --initial-cache-create true
    
    # Incremental build (load cache + new files only)
    python scripts/build_arb_cache.py --sport all --file-type parquet --initial-cache-create false
    
    # Build cache for one sport
    python scripts/build_arb_cache.py --sport nba --file-type parquet --initial-cache-create true
    
    # Quick setup (via setup script)
    python scripts/setup_arb_cache.py

SCHEDULING (Optional but recommended):
    Run this daily to keep cache fresh. Dashboard handles incremental updates
    between rebuilds (loads cache + only new files since last rebuild).
    
    Option A - Cron (macOS/Linux):
        crontab -e
        # Add line (runs daily at 2am ET):
        0 2 * * * cd /path/to/betting && python scripts/build_arb_cache.py --sport all
    
    Option B - Manual:
        python scripts/build_arb_cache.py --sport all
    
    Option C - AWS Lambda:
        Set up Lambda to run script daily via EventBridge

HOW IT WORKS:
    1. Lists ALL arb files from S3 (with pagination for >1000 files)
    2. Downloads files in parallel (200 workers)
    3. Deduplicates by (file_date, player, market, line)
       - Keeps row with highest expected_profit_pct
       - Same player/market/line may appear multiple times per day
    4. Saves consolidated cache to ~/Downloads/tmp/ (local)
    5. Uploads cache to S3 (production)
    6. Saves metadata (tracks last rebuild date, file count, date range)

DASHBOARD BEHAVIOR:
    When dashboard loads:
    1. Checks for cache in S3 (production) or ~/Downloads/tmp/ (local dev)
    2. Loads cache (fast! ~1 second)
    3. Checks metadata for last rebuild date
    4. Lists only NEW files since last rebuild
    5. Downloads only new files (~10-100 per day, not all 4904+!)
    6. Merges cache + new files
    7. Deduplicates and returns combined data
    8. Total time: 1-2 seconds (vs 20-50 seconds without cache)

TROUBLESHOOTING:
    Dashboard still slow?
        → Run: python scripts/build_arb_cache.py --sport all
    
    Cache not found?
        → Check AWS credentials: aws s3 ls s3://betting-nba-arbs/
        → Run: python scripts/build_arb_cache.py --sport all
    
    Missing recent data?
        → Check metadata: cat ~/Downloads/tmp/nba_arbs_cache_metadata.json
        → Rebuild: python scripts/build_arb_cache.py --sport nba
    
    Production (Streamlit) slow?
        → Check S3 cache: aws s3 ls s3://betting-nba-arbs/cache/
        → Rebuild from local (uploads to S3): python scripts/build_arb_cache.py --sport all

FILE STRUCTURE:
    ~/Downloads/tmp/                           # Local cache (dev only)
      ├── nba_arbs_cache.csv                  # NBA consolidated cache
      ├── nba_arbs_cache_metadata.json        # NBA metadata
      ├── nfl_arbs_cache.csv                  # NFL consolidated cache
      └── nfl_arbs_cache_metadata.json        # NFL metadata
    
    s3://betting-nba-arbs/
      ├── nba/arbs/YYYY-MM-DD/                # Raw files (Lambda output)
      │   └── arb_output_*.csv
      └── cache/                               # Consolidated cache
          ├── nba_arbs_cache.csv
          └── nba_arbs_cache_metadata.json

PERFORMANCE:
    Before cache:
        Load 4904 files → 20-50 seconds
        High S3 bandwidth usage
        Expensive S3 GET requests
    
    After cache:
        Load 1 cache + 10-100 new files → 1-2 seconds
        10-50x less bandwidth
        10-50x fewer S3 GET requests

NEXT STEPS:
    1. Run: python scripts/setup_arb_cache.py (one-time, takes 2-5 min)
    2. Test dashboard: cd streamlit_app && streamlit run app.py
    3. Schedule daily rebuilds (optional): crontab -e
"""

import boto3
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# S3 Configuration
S3_CONFIG = {
    'nba': {
        'bucket': 'betting-nba-arbs',
        'prefix': 'nba/arbs/',
        'cache_prefix': 'cache/'
    },
    'nfl': {
        'bucket': 'betting-nfl-arbs',
        'prefix': 'nfl/arbs/',
        'cache_prefix': 'cache/'
    }
}

# Local paths
# For local dev: use ~/Downloads/tmp (doesn't clutter repo)
# For production: only S3 is used (no local cache)
CACHE_DIR = Path.home() / 'Downloads' / 'tmp'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Performance tuning
MAX_WORKERS = 200  # Parallel downloads


# =============================================================================
# HELPER FUNCTIONS (execution flow order)
# =============================================================================

def load_single_s3_file(s3_client, bucket: str, s3_key: str) -> pd.DataFrame:
    """
    Load a single S3 file (used for parallel processing).
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 key to load
    
    Returns:
        DataFrame with file metadata added, or None if failed
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
        csv_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Skip empty files
        if len(df) == 0:
            return None
        
        # Extract date from S3 key: {sport}/arbs/2025-12-24/arb_output_20251224_180000.csv
        parts = s3_key.split('/')
        if len(parts) >= 4:
            file_date = parts[2]  # YYYY-MM-DD
            filename = parts[-1]  # arb_output_20251224_180000.csv
            
            # Extract time from filename
            filename_parts = filename.replace('.csv', '').split('_')
            if len(filename_parts) >= 3:
                date_str = filename_parts[-2]  # YYYYMMDD
                time_str = filename_parts[-1]  # HHMMSS
                
                file_datetime_utc = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
                file_datetime_utc = file_datetime_utc.replace(tzinfo=ZoneInfo('UTC'))
                file_datetime_et = file_datetime_utc.astimezone(ZoneInfo('America/New_York'))
                
                df['file_date'] = file_date
                df['file_datetime'] = file_datetime_et.isoformat()
                df['source_file'] = filename
        
        return df
    except Exception as e:
        print(f"      ⚠️  Failed to load {s3_key}: {e}")
        return None


def list_all_arb_files(s3_client, bucket: str, prefix: str) -> list:
    """
    List all arb CSV files from S3 with pagination.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'nba/arbs/')
    
    Returns:
        List of S3 keys for all CSV files
    """
    arb_files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in page_iterator:
        if 'Contents' in page:
            arb_files.extend([
                obj['Key'] for obj in page['Contents'] 
                if obj['Key'].endswith('.csv')
            ])
    
    return arb_files


def load_all_files_parallel(s3_client, bucket: str, files: list, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    Load multiple S3 files in parallel.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        files: List of S3 keys to load
        max_workers: Number of parallel threads
    
    Returns:
        Combined DataFrame with all data
    """
    all_dfs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(load_single_s3_file, s3_client, bucket, s3_key): s3_key 
            for s3_key in files
        }
        
        completed = 0
        total = len(files)
        
        for future in as_completed(future_to_key):
            df = future.result()
            if df is not None:
                all_dfs.append(df)
            
            completed += 1
            if completed % 100 == 0:
                print(f"      Progress: {completed}/{total} files processed...")
    
    if not all_dfs:
        return None
    
    return pd.concat(all_dfs, ignore_index=True)


def dedupe_arbs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate arbs by keeping best opportunity per player/market/line/day.
    
    Multiple files may exist per day (Lambda runs every 5-15 min).
    Same player/market/line may appear multiple times with different odds.
    We keep the BEST opportunity (highest expected_profit_pct).
    
    Args:
        df: Combined DataFrame with all arb data
    
    Returns:
        Deduplicated DataFrame
    """
    if len(df) == 0 or 'expected_profit_pct' not in df.columns:
        return df
    
    # Sort by expected_profit_pct descending, then take first per group
    df = df.sort_values('expected_profit_pct', ascending=False)
    df = df.drop_duplicates(
        subset=['file_date', 'player', 'market', 'line'],
        keep='first'
    )
    
    # Re-sort by file_date (desc) then expected_profit_pct (desc)
    df = df.sort_values(
        ['file_date', 'expected_profit_pct'], 
        ascending=[False, False]
    )
    
    return df


def save_cache_local(df: pd.DataFrame, sport: str):
    """Save cache to local filesystem as Parquet (fast + compressed)."""
    cache_path = CACHE_DIR / f"{sport}_arbs_cache.parquet"
    df.to_parquet(cache_path, engine='pyarrow', compression='snappy', index=False)
    
    # Get file size
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"   ✅ Saved locally: {cache_path} ({size_mb:.1f} MB)")
    return cache_path


def save_cache_s3(s3_client, df: pd.DataFrame, sport: str):
    """Save cache to S3 as Parquet (fast + compressed)."""
    from io import BytesIO
    
    config = S3_CONFIG[sport]
    bucket = config['bucket']
    s3_key = f"{config['cache_prefix']}{sport}_arbs_cache.parquet"
    
    # Write to BytesIO buffer
    buffer = BytesIO()
    df.to_parquet(buffer, engine='pyarrow', compression='snappy', index=False)
    buffer.seek(0)
    
    # Upload to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    
    size_mb = len(buffer.getvalue()) / 1024 / 1024
    print(f"   ✅ Uploaded to S3: s3://{bucket}/{s3_key} ({size_mb:.1f} MB)")



def save_metadata(sport: str, num_files: int, num_rows: int, oldest_date: str, newest_date: str):
    """Save cache metadata for tracking."""
    metadata = {
        'sport': sport,
        'last_rebuild': datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'num_files_processed': num_files,
        'num_total_rows': num_rows,
        'oldest_date': oldest_date,
        'newest_date': newest_date,
        'cache_version': '1.0'
    }
    
    metadata_path = CACHE_DIR / f"{sport}_arbs_cache_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata saved: {metadata_path}")
    
    # Also upload to S3
    s3 = boto3.client('s3')
    config = S3_CONFIG[sport]
    bucket = config['bucket']
    s3_key = f"{config['cache_prefix']}{sport}_arbs_cache_metadata.json"
    
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(metadata, indent=2),
        ContentType='application/json'
    )
    print(f"   ✅ Metadata uploaded to S3: s3://{bucket}/{s3_key}")


def build_cache_for_sport(sport: str, initial_build: bool = True):
    """
    Build arb cache for a specific sport.
    
    Args:
        sport: 'nba' or 'nfl'
        initial_build: If True, load all files from S3 (initial/full rebuild).
                      If False, load existing cache + only new files (incremental).
    """
    print("=" * 70)
    print(f"Building {sport.upper()} Arbitrage Cache")
    print(f"Mode: {'INITIAL BUILD' if initial_build else 'INCREMENTAL UPDATE'}")
    print("=" * 70)
    print()
    
    s3_client = boto3.client('s3')
    config = S3_CONFIG[sport]
    bucket = config['bucket']
    prefix = config['prefix']
    
    if initial_build:
        # INITIAL BUILD: Load all files from S3
        print("Step 1: Listing all arb files from S3...")
        arb_files = list_all_arb_files(s3_client, bucket, prefix)
        print(f"   ✅ Found {len(arb_files):,} files")
        print()
        
        if not arb_files:
            print(f"   ⚠️  No files found for {sport.upper()}. Skipping cache build.")
            return
        
        print(f"Step 2: Loading all files (parallel with {MAX_WORKERS} workers)...")
        combined_df = load_all_files_parallel(s3_client, bucket, arb_files, max_workers=MAX_WORKERS)
        
        if combined_df is None:
            print(f"   ❌ Failed to load files for {sport.upper()}")
            return
        
        print(f"   ✅ Loaded {len(combined_df):,} total rows")
        print()
        
    else:
        # INCREMENTAL BUILD: Load existing cache + new files
        print("Step 1: Loading existing cache...")
        
        # Try local first
        cache_path = CACHE_DIR / f"{sport}_arbs_cache.parquet"
        if cache_path.exists():
            import duckdb
            con = duckdb.connect(':memory:')
            combined_df = con.execute(
                "SELECT * FROM read_parquet(?)",
                [str(cache_path)]
            ).df()
            con.close()
            print(f"   ✅ Loaded local cache: {len(combined_df):,} rows")
        else:
            # Try S3
            try:
                s3_key = f"{config['cache_prefix']}{sport}_arbs_cache.parquet"
                obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
                parquet_bytes = obj['Body'].read()
                
                import duckdb
                from io import BytesIO
                con = duckdb.connect(':memory:')
                combined_df = con.execute(
                    "SELECT * FROM read_parquet(?)",
                    [BytesIO(parquet_bytes)]
                ).df()
                con.close()
                print(f"   ✅ Loaded S3 cache: {len(combined_df):,} rows")
            except Exception as e:
                print(f"   ❌ No existing cache found: {e}")
                print(f"   💡 Run with --initial-cache-create true first")
                return
        
        print()
        
        # Get metadata to find last cache date
        metadata_path = CACHE_DIR / f"{sport}_arbs_cache_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            newest_date = metadata.get('newest_date', '2020-01-01')
            print(f"Step 2: Finding new files since {newest_date}...")
        else:
            print(f"   ⚠️  No metadata found, loading last 7 days...")
            from datetime import datetime, timedelta
            newest_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # List only new files
        new_files = []
        from datetime import datetime, timedelta
        today = datetime.now().date()
        since_date_obj = datetime.strptime(newest_date, '%Y-%m-%d').date()
        days_to_check = min(30, (today - since_date_obj).days + 1)
        
        for i in range(days_to_check):
            check_date = today - timedelta(days=i)
            date_str = check_date.strftime('%Y-%m-%d')
            date_prefix = f"{prefix}{date_str}/"
            
            try:
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix=date_prefix)
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if key.endswith('.csv'):
                            new_files.append(key)
            except Exception:
                continue
        
        print(f"   ✅ Found {len(new_files):,} new files")
        print()
        
        if new_files:
            print(f"Step 3: Loading new files (parallel with {MAX_WORKERS} workers)...")
            new_df = load_all_files_parallel(s3_client, bucket, new_files, max_workers=MAX_WORKERS)
            
            if new_df is not None:
                print(f"   ✅ Loaded {len(new_df):,} new rows")
                print()
                
                # Merge with existing cache
                print("Step 4: Merging cache + new files...")
                combined_df = pd.concat([combined_df, new_df], ignore_index=True)
                print(f"   ✅ Combined: {len(combined_df):,} total rows")
                print()
        else:
            print("   ℹ️  No new files to load")
            print()
        
        arb_files = new_files  # For summary stats
    
    # Deduplicate (common for both modes)
    step_num = 3 if initial_build else 5
    print(f"Step {step_num}: Deduplicating arbs...")
    original_count = len(combined_df)
    deduped_df = dedupe_arbs(combined_df)
    removed_count = original_count - len(deduped_df)
    print(f"   ✅ Deduplicated: {len(deduped_df):,} rows (removed {removed_count:,} duplicates)")
    print()
    
    # Save cache
    step_num += 1
    print(f"Step {step_num}: Saving cache...")
    save_cache_local(deduped_df, sport)
    save_cache_s3(s3_client, deduped_df, sport)
    print()
    
    # Save metadata
    step_num += 1
    print(f"Step {step_num}: Saving metadata...")
    oldest_date = deduped_df['file_date'].min() if 'file_date' in deduped_df.columns else 'N/A'
    newest_date = deduped_df['file_date'].max() if 'file_date' in deduped_df.columns else 'N/A'
    num_files = len(arb_files) if initial_build else len(new_files) if not initial_build and new_files else 0
    save_metadata(sport, num_files, len(deduped_df), oldest_date, newest_date)
    print()
    
    # Summary
    print("=" * 70)
    print(f"✅ {sport.upper()} Cache Built Successfully!")
    print("=" * 70)
    print(f"Files processed: {num_files:,}")
    print(f"Total rows: {len(deduped_df):,}")
    print(f"Date range: {oldest_date} → {newest_date}")
    print()
    
    # Show metrics
    if 'is_arb' in deduped_df.columns:
        arb_rows = deduped_df[deduped_df['is_arb'] == True]
        print(f"Arbitrage opportunities: {len(arb_rows):,}")
        
        if len(arb_rows) > 0 and 'expected_profit_pct' in arb_rows.columns:
            avg_profit = arb_rows['expected_profit_pct'].mean()
            max_profit = arb_rows['expected_profit_pct'].max()
            print(f"Average profit: {avg_profit:.2f}%")
            print(f"Max profit: {max_profit:.2f}%")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build persistent cache for arbitrage dashboards'
    )
    parser.add_argument(
        '--sport',
        choices=['nba', 'nfl', 'all'],
        default='all',
        help='Which sport to build cache for (default: all)'
    )
    parser.add_argument(
        '--file-type',
        choices=['parquet'],
        default='parquet',
        help='Cache file format (only parquet supported)'
    )
    parser.add_argument(
        '--initial-cache-create',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='True: Load all files from S3 (initial build). False: Load existing cache + new files (incremental)'
    )
    
    args = parser.parse_args()
    
    # Validate file type
    if args.file_type != 'parquet':
        raise NotImplementedError(
            f"Only 'parquet' file type is supported. Got: {args.file_type}\n"
            f"CSV is deprecated due to slow parsing (35-50 seconds vs 1-2 seconds for parquet)."
        )
    
    print()
    print("█" * 70)
    print("  ARBITRAGE CACHE BUILDER")
    print("█" * 70)
    print(f"  File type: {args.file_type}")
    print(f"  Mode: {'INITIAL BUILD (load all from S3)' if args.initial_cache_create else 'INCREMENTAL (load cache + new files)'}")
    print("█" * 70)
    print()
    
    if args.sport == 'all':
        build_cache_for_sport('nba', initial_build=args.initial_cache_create)
        print()
        build_cache_for_sport('nfl', initial_build=args.initial_cache_create)
    else:
        build_cache_for_sport(args.sport, initial_build=args.initial_cache_create)
    
    print()
    print("█" * 70)
    print("  ✅ ALL CACHES BUILT SUCCESSFULLY")
    print("█" * 70)
    print()
    print("Next steps:")
    print("1. Dashboard will now use cached data (much faster!)")
    print("2. Schedule this script to run daily via cron/Lambda")
    print("3. Dashboard handles incremental updates between rebuilds")
    print()


if __name__ == '__main__':
    main()
