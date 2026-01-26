"""
Shared cache utilities for arbitrage dashboards.

OVERVIEW:
    This module provides fast loading for arb dashboards using a persistent cache.
    Instead of loading 4904+ files every time (20-50 seconds), it:
    1. Loads pre-built Parquet cache from S3/local using DuckDB (1-2 seconds)
    2. Checks for new files since last cache rebuild
    3. Downloads only NEW files (10-100 per day)
    4. Merges and returns combined data (total: 2-3 seconds)

CACHE LOCATIONS:
    Local dev:  ~/Downloads/tmp/{sport}_arbs_cache.parquet
    Production: s3://betting-{sport}-arbs/cache/{sport}_arbs_cache.parquet
    Metadata:   {same_location}/{sport}_arbs_cache_metadata.json

CACHE FORMAT:
    - Parquet (fast, compressed, ~5x smaller than CSV)
    - Read using DuckDB (10-20x faster than pandas CSV reading)
    - Typical: 121K rows load in 1-2 seconds vs 35-50 seconds for CSV

FUNCTIONS:
    load_all_arbs_with_cache(sport, max_workers) - Main function used by dashboards
        - Loads Parquet cache from S3/local using DuckDB
        - Checks for new files since last cache rebuild
        - Downloads only new files (incremental)
        - Merges and deduplicates
        - Returns combined data

    Helper functions:
        - load_cache_from_s3() - Load Parquet cache from S3 with DuckDB
        - load_cache_from_local() - Load Parquet cache from ~/Downloads/tmp/ with DuckDB
        - get_cache_metadata() - Get last rebuild date
        - list_new_files_since_date() - Find files created after date
        - load_files_parallel() - Download multiple files in parallel
        - dedupe_arbs() - Deduplicate by player/market/line/day

USAGE (from dashboards):
    from utils.arb_cache import load_all_arbs_with_cache
    
    @st.cache_data(ttl=60)
    def load_all_arbs():
        return load_all_arbs_with_cache('nba', max_workers=100)

CACHE REBUILD:
    Cache is rebuilt daily by scripts/build_arb_cache.py
    Dashboard uses metadata to know when cache was last rebuilt
    Only downloads files created after last rebuild (incremental loading)

PERFORMANCE:
    Before cache: Load 4904 CSV files → 20-50 seconds
    After cache (CSV):  Load 1 cache → 35-50 seconds (pandas CSV parsing slow!)
    After cache (Parquet): Load 1 cache → 1-2 seconds (DuckDB Parquet reading fast!)
    Improvement:  20-40x faster!

Used by: streamlit_app/pages/1_NBA_Arbs.py
         streamlit_app/pages/2_NFL_Arbs.py
"""

import boto3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import timing utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from timing import timed, timed_section


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


# =============================================================================
# HELPER FUNCTIONS (execution flow order)
# =============================================================================

def load_single_s3_file(s3_key: str, bucket: str, s3_client) -> pd.DataFrame:
    """
    Load a single S3 file (used for parallel processing).
    
    Args:
        s3_key: S3 key to load
        bucket: S3 bucket name
        s3_client: Boto3 S3 client
    
    Returns:
        DataFrame with file metadata added, or None if failed
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
        csv_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        if len(df) == 0:
            return None
        
        # Extract date from S3 key: {sport}/arbs/2025-12-24/arb_output_20251224_180000.csv
        parts = s3_key.split('/')
        if len(parts) >= 4:
            file_date = parts[2]  # YYYY-MM-DD
            filename = parts[-1]  # arb_output_20251224_180000.csv
            
            filename_parts = filename.replace('.csv', '').split('_')
            if len(filename_parts) >= 3:
                date_str = filename_parts[-2]  # YYYYMMDD
                time_str = filename_parts[-1]  # HHMMSS
                
                file_datetime_utc = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
                file_datetime_utc = file_datetime_utc.replace(tzinfo=ZoneInfo('UTC'))
                file_datetime_et = file_datetime_utc.astimezone(ZoneInfo('America/New_York'))
                
                df['file_date'] = file_date
                df['file_datetime'] = file_datetime_et
                df['source_file'] = filename
        
        return df
    except Exception:
        return None


@timed
def load_cache_from_s3(sport: str, s3_client) -> pd.DataFrame:
    """
    Load cache file from S3 using DuckDB for fast Parquet reading.
    
    Downloads to temp file first, then uses DuckDB to stream from disk.
    This is faster and more memory-efficient than loading into memory.
    
    Args:
        sport: 'nba' or 'nfl'
        s3_client: Boto3 S3 client
    
    Returns:
        DataFrame with cached data, or None if cache doesn't exist
    """
    try:
        import duckdb
        import tempfile
        
        config = S3_CONFIG[sport]
        bucket = config['bucket']
        s3_key = f"{config['cache_prefix']}{sport}_arbs_cache.parquet"
        
        with timed_section("S3: download to temp file"):
            # Download to temp file (cleaned up automatically)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
                tmp_path = tmp_file.name
                s3_client.download_fileobj(
                    Bucket=bucket,
                    Key=s3_key,
                    Fileobj=tmp_file
                )
                file_size = Path(tmp_path).stat().st_size
                print(f"   📦 Downloaded {file_size:,} bytes ({file_size/1024/1024:.1f} MB) to {tmp_path}")
        
        with timed_section("DuckDB: read Parquet from temp file"):
            # Use DuckDB to read from disk (faster than pandas!)
            con = duckdb.connect(':memory:')
            df = con.execute(
                "SELECT * FROM read_parquet(?)",
                [tmp_path]
            ).df()
            con.close()
            
            print(f"   📊 Loaded {len(df):,} rows with DuckDB")
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return df
    except Exception as e:
        import traceback
        print(f"   ❌ S3 cache load failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None


@timed
def load_cache_from_local(sport: str) -> pd.DataFrame:
    """
    Load cache file from local filesystem using DuckDB for fast Parquet reading.
    
    Args:
        sport: 'nba' or 'nfl'
    
    Returns:
        DataFrame with cached data, or None if cache doesn't exist
    """
    try:
        import duckdb
        
        cache_path = CACHE_DIR / f"{sport}_arbs_cache.parquet"
        if not cache_path.exists():
            return None
        
        # Use DuckDB to read Parquet (SUPER FAST!)
        con = duckdb.connect(':memory:')
        df = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(cache_path)]
        ).df()
        con.close()
        
        return df
    except Exception:
        return None


def get_cache_metadata(sport: str, s3_client) -> dict:
    """
    Get cache metadata (tracks last rebuild date).
    
    Args:
        sport: 'nba' or 'nfl'
        s3_client: Boto3 S3 client
    
    Returns:
        Metadata dict, or None if not found
    """
    try:
        # Try S3 first
        config = S3_CONFIG[sport]
        bucket = config['bucket']
        s3_key = f"{config['cache_prefix']}{sport}_arbs_cache_metadata.json"
        
        obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
        metadata = json.loads(obj['Body'].read().decode('utf-8'))
        return metadata
    except Exception:
        # Fall back to local
        try:
            metadata_path = CACHE_DIR / f"{sport}_arbs_cache_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
    
    return None


@timed
def list_new_files_since_date(sport: str, since_date: str, s3_client) -> list:
    """
    List files created after a specific date.
    
    OPTIMIZATION: Only checks last 30 days of folders to avoid scanning 7000+ files.
    
    Args:
        sport: 'nba' or 'nfl'
        since_date: Date string in YYYY-MM-DD format
        s3_client: Boto3 S3 client
    
    Returns:
        List of S3 keys for new files
    """
    from datetime import datetime, timedelta
    
    config = S3_CONFIG[sport]
    bucket = config['bucket']
    prefix = config['prefix']
    
    # Parse since_date
    since_date_obj = datetime.strptime(since_date, '%Y-%m-%d').date()
    
    # Only check last 30 days of folders (optimization)
    today = datetime.now().date()
    days_to_check = min(30, (today - since_date_obj).days + 1)
    
    print(f"   🔍 Checking last {days_to_check} days of folders (vs checking ALL 7000+ files)")
    
    new_files = []
    
    with timed_section(f"List files in {days_to_check} date folders"):
        for i in range(days_to_check):
            check_date = today - timedelta(days=i)
            date_str = check_date.strftime('%Y-%m-%d')
            
            # Only check this date's folder
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
    
    print(f"   ✅ Found {len(new_files):,} files in last {days_to_check} days")
    
    return new_files


@timed
def load_files_parallel(files: list, bucket: str, s3_client, max_workers: int = 100) -> pd.DataFrame:
    """
    Load multiple S3 files in parallel.
    
    Args:
        files: List of S3 keys to load
        bucket: S3 bucket name
        s3_client: Boto3 S3 client
        max_workers: Number of parallel threads
    
    Returns:
        Combined DataFrame with all data
    """
    if not files:
        return None
    
    all_dfs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(load_single_s3_file, s3_key, bucket, s3_client): s3_key 
            for s3_key in files
        }
        
        for future in as_completed(future_to_key):
            df = future.result()
            if df is not None:
                all_dfs.append(df)
    
    if not all_dfs:
        return None
    
    return pd.concat(all_dfs, ignore_index=True)


def dedupe_arbs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate arbs by keeping best opportunity per player/market/line/day.
    
    Args:
        df: Combined DataFrame with all arb data
    
    Returns:
        Deduplicated DataFrame
    """
    if len(df) == 0 or 'expected_profit_pct' not in df.columns:
        return df
    
    df = df.sort_values('expected_profit_pct', ascending=False)
    df = df.drop_duplicates(
        subset=['file_date', 'player', 'market', 'line'],
        keep='first'
    )
    
    df = df.sort_values(
        ['file_date', 'expected_profit_pct'], 
        ascending=[False, False]
    )
    
    return df


@timed
def load_all_arbs_with_cache(sport: str, max_workers: int = 100) -> pd.DataFrame:
    """
    Load all arbs using persistent cache + incremental updates.
    
    This is the main function used by dashboards. It:
    1. Loads existing cache (fast)
    2. Checks for new files since last cache rebuild
    3. Loads only new files (incremental)
    4. Merges and deduplicates
    5. Returns combined data
    
    Args:
        sport: 'nba' or 'nfl'
        max_workers: Number of parallel download threads
    
    Returns:
        DataFrame with all arb data (cached + new)
    """
    print(f"\n{'='*70}")
    print(f"Loading arbs for {sport.upper()} with cache system")
    print(f"{'='*70}")
    
    with timed_section("Initialize S3 client"):
        s3_client = boto3.client('s3')
        config = S3_CONFIG[sport]
    
    # Step 1: Load cache from S3 (PRODUCTION - NO FALLBACK)
    with timed_section("Load cache from S3"):
        cache_df = load_cache_from_s3(sport, s3_client)
    
    if cache_df is not None:
        print(f"✅ Cache loaded: {len(cache_df):,} rows")
    else:
        # Show helpful error message
        error_msg = (
            f"❌ Failed to load S3 cache for {sport}.\n\n"
            f"Possible causes:\n"
            f"1. AWS credentials not configured in Streamlit Cloud Secrets\n"
            f"2. Cache file doesn't exist: s3://{S3_CONFIG[sport]['bucket']}/cache/{sport}_arbs_cache.parquet\n"
            f"3. Insufficient S3 permissions\n\n"
            f"To fix:\n"
            f"- Add AWS credentials to Streamlit Cloud → Settings → Secrets\n"
            f"- Or run: python scripts/build_arb_cache.py --sport {sport}\n"
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Step 2: Get cache metadata to find last rebuild date
    with timed_section("Get cache metadata"):
        metadata = get_cache_metadata(sport, s3_client)
    
    if metadata:
        newest_date = metadata.get('newest_date', '2020-01-01')
        print(f"📅 Cache last rebuilt: {metadata.get('last_rebuild', 'Unknown')}")
        print(f"📅 Cache newest date: {newest_date}")
    else:
        newest_date = '2020-01-01'  # Load everything if no metadata
        print(f"⚠️  No metadata found - will check all files")
    
    # Step 3: Check for new files since last cache update
    with timed_section("List new files since cache"):
        new_files = list_new_files_since_date(sport, newest_date, s3_client)
    
    print(f"📂 New files found: {len(new_files):,}")
    
    # Step 4: If cache failed to load, we already raised exception above
    # No need to check again here
    
    # Step 5: Load new files if they exist
    if new_files:
        with timed_section(f"Load {len(new_files)} new files (parallel)"):
            new_df = load_files_parallel(new_files, config['bucket'], s3_client, max_workers)
        
        if new_df is not None:
            print(f"✅ New files loaded: {len(new_df):,} rows")
            
            # Merge with cache (cache is guaranteed to exist at this point)
            with timed_section("Merge cache + new files"):
                combined_df = pd.concat([cache_df, new_df], ignore_index=True)
                print(f"📊 Combined: {len(combined_df):,} rows")
            
            # Deduplicate
            with timed_section("Deduplicate arbs"):
                combined_df = dedupe_arbs(combined_df)
                print(f"✅ After deduplication: {len(combined_df):,} rows")
            
            return combined_df
    else:
        print(f"✅ No new files - using cache as-is")
    
    # No new files, return cache as-is
    return cache_df
