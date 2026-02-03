"""
Clean Up Live Odds Parquet Files - Remove Pre-Game Data

PURPOSE:
Filter existing live odds parquet files to remove rows where game_status != 'in'.
This removes pre-game and post-game data, keeping only true live in-game odds.

CONTEXT:
The lambda was previously saving odds for ALL games (live + upcoming + finished).
This script cleans up historical data to only include live game data.

FUNCTIONALITY:
1. Backup all files to ~/Downloads/tmp/live_odds_backup/ first
2. Read each parquet file from S3
3. Filter to only rows where game_status == 'in'
4. If 0 rows remain: Delete the file from S3
5. If rows remain: Write filtered file back to S3
6. Report statistics on what was cleaned

FILES PROCESSED:
- s3://nba-betting-mt/data/01_input/live_odds/the-odds-api/*.parquet
- s3://nba-betting-mt/data/01_input/live_odds/espn/*.parquet

USAGE:
    # Dry run (preview changes without modifying anything)
    python tmp/cleanup_live_odds_parquet.py --dry-run
    
    # Execute cleanup
    python tmp/cleanup_live_odds_parquet.py
    
    # Process specific date
    python tmp/cleanup_live_odds_parquet.py --date 20260202

AUTHOR: Thomas Myles
CREATED: 2026-02-02
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil
from zoneinfo import ZoneInfo

# Find project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
S3_ODDS_PATH = 'data/01_input/live_odds/the-odds-api/'
S3_ESPN_PATH = 'data/01_input/live_odds/espn/'

BACKUP_DIR = Path.home() / 'Downloads' / 'tmp' / 'live_odds_backup'

EMOJI = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'refresh': '🔄',
    'save': '💾',
    'trash': '🗑️',
    'chart': '📊',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_s3_files(bucket: str, prefix: str) -> list:
    """List all parquet files in S3 prefix."""
    import boto3
    
    s3_client = boto3.client('s3')
    
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('.parquet'):
                files.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                })
    
    return files


def backup_file_from_s3(bucket: str, key: str, backup_dir: Path):
    """Backup a single file from S3 to local directory."""
    import boto3
    
    s3_client = boto3.client('s3')
    
    # Preserve directory structure
    relative_path = Path(key)
    local_path = backup_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    s3_client.download_file(bucket, key, str(local_path))
    
    return local_path


def read_parquet_from_s3(bucket: str, key: str):
    """Read parquet file from S3."""
    import pandas as pd
    import boto3
    import io
    
    s3_client = boto3.client('s3')
    
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    parquet_buffer = io.BytesIO(obj['Body'].read())
    
    df = pd.read_parquet(parquet_buffer)
    
    return df


def write_parquet_to_s3(df, bucket: str, key: str):
    """Write parquet file to S3."""
    import pandas as pd
    import boto3
    import io
    
    s3_client = boto3.client('s3')
    
    # Convert to parquet in memory
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    parquet_buffer.seek(0)
    
    # Write to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=parquet_buffer.getvalue()
    )


def delete_file_from_s3(bucket: str, key: str):
    """Delete file from S3."""
    import boto3
    
    s3_client = boto3.client('s3')
    s3_client.delete_object(Bucket=bucket, Key=key)


def filter_df_to_live_only(df, is_espn_file: bool = False):
    """
    Filter dataframe to only live games (game_status == 'in').
    
    Args:
        df: DataFrame to filter
        is_espn_file: True if this is an ESPN file (has game_status column)
    
    Returns:
        Filtered DataFrame, or original if no game_status column
    """
    if 'game_status' not in df.columns:
        if is_espn_file:
            print(f"      ⚠️ WARNING: ESPN file missing game_status column")
            return df
        else:
            # Odds files don't have game_status - can't filter directly
            print(f"      ℹ️ Odds file (no game_status column)")
            return None  # Signal that we need to match with ESPN data
    
    return df[df['game_status'] == 'in'].copy()


def get_live_games_from_espn_file(bucket: str, espn_key: str, verbose: bool = False) -> set:
    """
    Get set of (away_team, home_team) tuples for live games from ESPN file.
    
    Returns:
        Set of (away_team, home_team) tuples that were live
    """
    try:
        df = read_parquet_from_s3(bucket, espn_key)
        
        if verbose:
            print(f"      📋 ESPN file columns: {list(df.columns)}")
        
        if 'game_status' not in df.columns:
            if verbose:
                print(f"      ⚠️ ESPN file missing game_status column")
            return set()
        
        live_df = df[df['game_status'] == 'in']
        
        live_games = set()
        for _, row in live_df.iterrows():
            away = row.get('away_team_espn')
            home = row.get('home_team_espn')
            if away and home:
                live_games.add((away, home))
        
        if verbose:
            print(f"      ✅ Found {len(live_games)} live games in ESPN file")
        return live_games
        
    except Exception as e:
        if verbose:
            print(f"      ❌ Error reading ESPN file: {e}")
        return set()


def process_file(bucket: str, key: str, backup_dir: Path, live_games_map: dict = None, dry_run: bool = False, verbose: bool = True) -> dict:
    """
    Process a single parquet file.
    
    Args:
        bucket: S3 bucket
        key: S3 key
        backup_dir: Local backup directory
        live_games_map: Dict mapping timestamp -> set of (away_team, home_team) for live games
        dry_run: If True, don't modify anything
    
    Returns:
        Dict with stats: {
            'original_rows': int,
            'filtered_rows': int,
            'action': 'kept' | 'deleted' | 'skipped',
            'backup_path': str,
        }
    """
    try:
        # Determine file type
        is_espn_file = '/espn/' in key
        is_odds_file = '/the-odds-api/' in key
        
        if verbose:
            print(f"      📁 File type: {'ESPN' if is_espn_file else 'Odds API'}")
        
        # Step 1: Backup
        if not dry_run:
            backup_path = backup_file_from_s3(bucket, key, backup_dir)
            backup_path_str = str(backup_path)
        else:
            backup_path_str = str(backup_dir / key)
        
        # Step 2: Read
        df = read_parquet_from_s3(bucket, key)
        original_rows = len(df)
        
        if verbose:
            print(f"      📊 Original rows: {original_rows}")
            print(f"      📋 Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        
        # Step 3: Filter based on file type
        if is_espn_file:
            # ESPN files have game_status column
            if 'game_status' not in df.columns:
                if verbose:
                    print(f"      ⚠️ ESPN file missing game_status - keeping all rows")
                df_filtered = df
            else:
                df_filtered = df[df['game_status'] == 'in'].copy()
                if verbose:
                    print(f"      ✅ Filtered by game_status == 'in'")
        
        elif is_odds_file:
            # Odds files need to be matched with ESPN data
            # Extract timestamp from filename (e.g., "20260201_182002.parquet" -> "20260201_182002")
            filename = Path(key).name
            timestamp = filename.replace('.parquet', '')
            
            if verbose:
                print(f"      🕐 Timestamp: {timestamp}")
            
            if live_games_map and timestamp in live_games_map:
                live_games = live_games_map[timestamp]
                if verbose:
                    print(f"      🎯 Live games at this time: {len(live_games)}")
                
                # Filter odds records to only live games
                if 'away_team' in df.columns and 'home_team' in df.columns:
                    df_filtered = df[
                        df.apply(lambda row: (row['away_team'], row['home_team']) in live_games, axis=1)
                    ].copy()
                    if verbose:
                        print(f"      ✅ Filtered by matching live games")
                else:
                    if verbose:
                        print(f"      ⚠️ Missing away_team/home_team columns - keeping all rows")
                    df_filtered = df
            else:
                if verbose:
                    print(f"      ⚠️ No ESPN data found for timestamp {timestamp} - assuming no live games")
                df_filtered = df.iloc[0:0]  # Empty dataframe with same columns
        
        else:
            if verbose:
                print(f"      ⚠️ Unknown file type - keeping all rows")
            df_filtered = df
        
        filtered_rows = len(df_filtered)
        if verbose:
            print(f"      📊 Filtered rows: {filtered_rows}")
        
        # Step 4: Decide action
        if filtered_rows == 0:
            # Delete file
            action = 'deleted'
            if not dry_run:
                delete_file_from_s3(bucket, key)
        elif filtered_rows == original_rows:
            # No change needed
            action = 'skipped'
        else:
            # Write filtered file back
            action = 'kept'
            if not dry_run:
                write_parquet_to_s3(df_filtered, bucket, key)
        
        return {
            'original_rows': original_rows,
            'filtered_rows': filtered_rows,
            'action': action,
            'backup_path': backup_path_str,
        }
        
    except Exception as e:
        import traceback
        if verbose:
            print(f"      ❌ ERROR: {e}")
            print(f"      {traceback.format_exc()}")
        return {
            'original_rows': 0,
            'filtered_rows': 0,
            'action': 'error',
            'error': str(e),
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(dry_run: bool = False, date_filter: str = None):
    """
    Main cleanup function.
    
    Args:
        dry_run: If True, preview changes without modifying anything
        date_filter: Optional date string (YYYYMMDD) to only process files from that date
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['refresh']} LIVE ODDS CLEANUP - Remove Pre-Game Data")
    print(f"{'='*80}\n")
    
    if dry_run:
        print(f"{EMOJI['warning']} DRY RUN MODE - No changes will be made\n")
    else:
        print(f"{EMOJI['info']} Backup location: {BACKUP_DIR}\n")
    
    # Step 1: List all files
    print(f"{EMOJI['refresh']} Step 1: Listing files from S3...")
    
    odds_files = list_s3_files(S3_BUCKET, S3_ODDS_PATH)
    espn_files = list_s3_files(S3_BUCKET, S3_ESPN_PATH)
    
    # Apply date filter if specified
    if date_filter:
        odds_files = [f for f in odds_files if date_filter in f['key']]
        espn_files = [f for f in espn_files if date_filter in f['key']]
        print(f"{EMOJI['info']} Filtering to date: {date_filter}")
    
    total_files = len(odds_files) + len(espn_files)
    total_size_mb = sum(f['size'] for f in odds_files + espn_files) / (1024 * 1024)
    
    print(f"{EMOJI['success']} Found {total_files} files ({total_size_mb:.2f} MB)")
    print(f"   Odds files: {len(odds_files)}")
    print(f"   ESPN files: {len(espn_files)}\n")
    
    if total_files == 0:
        print(f"{EMOJI['info']} No files to process\n")
        return
    
    # Step 2: Build map of live games from ESPN files (only ones with corresponding odds files)
    print(f"{EMOJI['refresh']} Step 2: Building live games map from ESPN files...")
    
    # Get timestamps that have odds files
    odds_timestamps = {Path(f['key']).name.replace('.parquet', '') for f in odds_files}
    print(f"   Found {len(odds_timestamps)} unique timestamps with odds files")
    
    live_games_map = {}  # timestamp -> set of (away_team, home_team) tuples
    espn_files_to_process = [f for f in espn_files if Path(f['key']).name.replace('.parquet', '') in odds_timestamps]
    
    print(f"   Processing {len(espn_files_to_process)} ESPN files (matching odds timestamps)...")
    
    for i, espn_file in enumerate(espn_files_to_process, 1):
        key = espn_file['key']
        filename = Path(key).name
        timestamp = filename.replace('.parquet', '')
        
        if i % 50 == 0 or i == len(espn_files_to_process):
            print(f"   Processing ESPN file {i}/{len(espn_files_to_process)}...", end='\r')
        
        try:
            live_games = get_live_games_from_espn_file(S3_BUCKET, key)
            if live_games:
                live_games_map[timestamp] = live_games
        except Exception as e:
            pass  # Continue on error
    
    print(f"\n{EMOJI['success']} Built live games map: {len(live_games_map)} timestamps with live games\n")
    print(f"   📊 Sample timestamps with live games: {list(live_games_map.keys())[:5]}\n")
    
    # Step 3: Process files
    print(f"{EMOJI['refresh']} Step 3: Processing files...")
    
    stats = {
        'total_processed': 0,
        'total_kept': 0,
        'total_deleted': 0,
        'total_skipped': 0,
        'total_errors': 0,
        'original_rows': 0,
        'filtered_rows': 0,
    }
    
    all_files = [
        (S3_BUCKET, f['key'], 'espn') for f in espn_files
    ] + [
        (S3_BUCKET, f['key'], 'odds') for f in odds_files
    ]
    
    for i, (bucket, key, file_type) in enumerate(all_files, 1):
        filename = Path(key).name
        
        # Show verbose output for first 10 files, then every 100th file
        show_details = (i <= 10) or (i % 100 == 0) or (i == total_files)
        
        if show_details:
            print(f"\n[{i}/{total_files}] {filename} ({file_type})...")
        else:
            # Just show progress indicator
            print(f"Processing {i}/{total_files}...", end='\r')
        
        result = process_file(bucket, key, BACKUP_DIR, live_games_map=live_games_map, dry_run=dry_run, verbose=show_details)
        
        stats['total_processed'] += 1
        stats['original_rows'] += result['original_rows']
        stats['filtered_rows'] += result['filtered_rows']
        
        if show_details:
            if result['action'] == 'kept':
                stats['total_kept'] += 1
                rows_removed = result['original_rows'] - result['filtered_rows']
                print(f"   {EMOJI['save']} Filtered: {result['original_rows']} → {result['filtered_rows']} rows ({rows_removed} removed)")
            elif result['action'] == 'deleted':
                stats['total_deleted'] += 1
                print(f"   {EMOJI['trash']} Deleted (0 live rows)")
            elif result['action'] == 'skipped':
                stats['total_skipped'] += 1
                print(f"   {EMOJI['success']} Skipped (already clean)")
            elif result['action'] == 'error':
                stats['total_errors'] += 1
                print(f"   {EMOJI['error']} Error: {result.get('error', 'unknown')}")
        else:
            # Just count the action
            if result['action'] == 'kept':
                stats['total_kept'] += 1
            elif result['action'] == 'deleted':
                stats['total_deleted'] += 1
            elif result['action'] == 'skipped':
                stats['total_skipped'] += 1
            elif result['action'] == 'error':
                stats['total_errors'] += 1
    
    print()  # New line after progress indicator
    
    # Step 4: Report summary
    print(f"\n{'='*80}")
    print(f"{EMOJI['chart']} CLEANUP SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {stats['total_processed']}")
    print(f"Files kept (filtered): {stats['total_kept']}")
    print(f"Files deleted (no live data): {stats['total_deleted']}")
    print(f"Files skipped (already clean): {stats['total_skipped']}")
    print(f"Files with errors: {stats['total_errors']}")
    print(f"\nRows:")
    print(f"  Original: {stats['original_rows']:,}")
    print(f"  After filtering: {stats['filtered_rows']:,}")
    print(f"  Removed: {stats['original_rows'] - stats['filtered_rows']:,}")
    
    if not dry_run:
        print(f"\n{EMOJI['save']} Backup saved to: {BACKUP_DIR}")
    else:
        print(f"\n{EMOJI['warning']} DRY RUN - No changes made. Run without --dry-run to execute.")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean up live odds parquet files')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying anything'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Only process files from specific date (YYYYMMDD format)'
    )
    
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, date_filter=args.date)
