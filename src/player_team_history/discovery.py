"""
Player discovery from S3 betting/props data.

Functions to scan S3 buckets and extract unique player names from betting data.
"""

import boto3
import pandas as pd
from io import BytesIO
from typing import Set
from pathlib import Path
import sys

# Find repo root
current_dir = Path(__file__).resolve()
repo_root = current_dir
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
    if repo_root == repo_root.parent:
        raise RuntimeError("Could not find repo root")

sys.path.append(str(repo_root))

from src.config import EMOJI


# S3 Configuration
ODDS_API_BUCKET = 'the-odds-api-mt'
ODDS_API_PREFIXES = [
    'nba/historical_player_props/2023-24/',
    'nba/historical_player_props/2024-25/',
    'nba/historical_player_props/2025-26/',
]


def get_all_props_files_from_s3(bucket, prefix, max_files=None):
    """
    List all props CSV files in S3.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to search under
        max_files: Optional limit on number of files to return
    
    Returns:
        List of S3 keys
    """
    s3_client = boto3.client('s3')
    
    keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            # Only CSV files
            if key.endswith('.csv'):
                keys.append(key)
                
                if max_files and len(keys) >= max_files:
                    return keys
    
    return keys


def extract_players_from_props_file(bucket, key):
    """
    Extract unique player names from a single props CSV in S3.
    
    Normalizes all player names to canonical form and filters out garbage.
    Uses Odds API normalization since S3 data comes from The Odds API.
    
    Args:
        bucket: S3 bucket name
        key: S3 key to CSV file
    
    Returns:
        Set of normalized player names (garbage filtered out)
    """
    from src.player_team_history.name_normalization import normalize_from_odds_api
    
    s3_client = boto3.client('s3')
    
    try:
        # Download file
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read CSV
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        
        # Find player column
        player_col = None
        for col in df.columns:
            if 'player' in col.lower() and col.lower() != 'player_id':
                player_col = col
                break
        
        if not player_col:
            return set()
        
        # Extract and NORMALIZE player names (using Odds API normalization)
        raw_players = df[player_col].dropna().unique()
        normalized_players = set()
        
        for player in raw_players:
            normalized = normalize_from_odds_api(player)
            if normalized:  # Only include valid player names
                normalized_players.add(normalized)
        
        return normalized_players
        
    except Exception as e:
        # Skip files that can't be read
        return set()


def discover_players_from_s3(bucket=ODDS_API_BUCKET, prefixes=None, 
                             sample_size=None, verbose=True):
    """
    Scan S3 betting data to find all unique player names.
    
    Args:
        bucket: S3 bucket name
        prefixes: List of S3 prefixes to search (default: all historical_player_props seasons)
        sample_size: If specified, only scan this many files total (for testing)
        verbose: If True, print progress
    
    Returns:
        Set of unique player names
    """
    if prefixes is None:
        prefixes = ODDS_API_PREFIXES
    
    if verbose:
        print(f"\n{EMOJI['chart']} Discovering players from S3...")
        print(f"   Bucket: s3://{bucket}/")
        print(f"   Prefixes: {len(prefixes)} season(s)")
    
    all_players = set()
    files_processed = 0
    
    for prefix in prefixes:
        if verbose:
            print(f"\n   Scanning: {prefix}")
        
        # Get props files for this prefix
        remaining_sample = None
        if sample_size:
            remaining_sample = sample_size - files_processed
            if remaining_sample <= 0:
                break
        
        props_files = get_all_props_files_from_s3(bucket, prefix, max_files=remaining_sample)
        
        if not props_files:
            if verbose:
                print(f"      {EMOJI['warning']} No files found")
            continue
        
        if verbose:
            print(f"      Found {len(props_files)} files")
        
        for i, key in enumerate(props_files, 1):
            players_in_file = extract_players_from_props_file(bucket, key)
            all_players.update(players_in_file)
            files_processed += 1
            
            if verbose and files_processed % 100 == 0:
                print(f"      Processed {files_processed} files total... ({len(all_players)} unique players)")
            
            if sample_size and files_processed >= sample_size:
                break
    
    if verbose:
        print(f"\n{EMOJI['success']} Found {len(all_players)} unique players from {files_processed} S3 files")
    
    return all_players


def extract_players_from_local_file(csv_path):
    """
    Extract unique player names from a local CSV file.
    
    Args:
        csv_path: Path to local CSV file
    
    Returns:
        Set of normalized player names
    """
    from src.player_team_history.name_normalization import normalize_from_odds_api
    
    try:
        df = pd.read_csv(csv_path)
        
        # Find player column
        player_col = None
        for col in df.columns:
            if 'player' in col.lower() and col.lower() != 'player_id':
                player_col = col
                break
        
        if not player_col:
            return set()
        
        # Extract and normalize player names
        raw_players = df[player_col].dropna().unique()
        normalized_players = set()
        
        for player in raw_players:
            normalized = normalize_from_odds_api(player)
            if normalized:
                normalized_players.add(normalized)
        
        return normalized_players
        
    except Exception:
        return set()


def discover_players_from_local(local_dir, sample_size=None, verbose=True):
    """
    Discover players from local cached CSV files (much faster than S3).
    
    Args:
        local_dir: Path to local directory containing cached CSVs
        sample_size: Optional limit on files to scan (for testing)
        verbose: If True, print progress
    
    Returns:
        Set of unique player names
    """
    local_path = Path(local_dir)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Local cache not found: {local_path}")
    
    if verbose:
        print(f"\n{EMOJI['chart']} Discovering players from local cache...")
        print(f"   Location: {local_path}")
    
    all_players = set()
    files_processed = 0
    
    # Get all CSV files
    csv_files = list(local_path.rglob('*.csv'))
    
    if sample_size:
        csv_files = csv_files[:sample_size]
    
    if verbose:
        print(f"   Found {len(csv_files)} CSV files")
    
    for csv_path in csv_files:
        players_in_file = extract_players_from_local_file(csv_path)
        all_players.update(players_in_file)
        files_processed += 1
        
        if verbose and files_processed % 100 == 0:
            print(f"      Processed {files_processed} files... ({len(all_players)} unique players)")
    
    if verbose:
        print(f"\n{EMOJI['success']} Found {len(all_players)} unique players from {files_processed} local files")
    
    return all_players


def discover_all_players(s3_sample_size=None, verbose=True):
    """
    Discover all players from local cache or S3 betting data.
    
    Checks for local cache first (fast), falls back to S3 if not found.
    
    Input: 
        Local: ~/Downloads/tmp/player_props_raw/{2023-24,2024-25,2025-26}/*.csv
        S3: s3://the-odds-api-mt/nba/historical_player_props/{2023-24,2024-25,2025-26}/*.csv
    
    Args:
        s3_sample_size: Optional limit on S3 files to scan (for testing)
        verbose: If True, print progress
    
    Returns:
        Set of unique player names
    """
    # Check for local cache first
    local_cache = Path.home() / 'Downloads' / 'tmp' / 'player_props_raw'
    
    if local_cache.exists():
        if verbose:
            print(f"\n{EMOJI['success']} Using local cache (fast!)")
        players = discover_players_from_local(
            local_cache,
            sample_size=s3_sample_size,
            verbose=verbose
        )
    else:
        if verbose:
            print(f"\n{EMOJI['warning']} Local cache not found, using S3 (slow)")
            print(f"   To speed up: Download cache first with:")
            print(f"   aws s3 sync s3://the-odds-api-mt/nba/historical_player_props/ ~/Downloads/tmp/player_props_raw/")
        players = discover_players_from_s3(
            bucket=ODDS_API_BUCKET,
            prefixes=ODDS_API_PREFIXES,
            sample_size=s3_sample_size,
            verbose=verbose
        )
    
    if verbose:
        print(f"\n{EMOJI['success']} Total unique players discovered: {len(players)}")
    
    return players


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    """Test player discovery from S3."""
    print("="*80)
    print(f"{EMOJI['test']} Testing Player Discovery from S3")
    print("="*80)
    print()
    print("Input: s3://the-odds-api-mt/nba/historical_player_props/{2023-24,2024-25,2025-26}/*.csv")
    print()
    
    # Test with small sample
    players = discover_all_players(s3_sample_size=20, verbose=True)
    
    print(f"\n{EMOJI['success']} Sample players found:")
    for i, player in enumerate(sorted(list(players))[:20], 1):
        print(f"   {i}. {player}")
    
    if len(players) > 20:
        print(f"   ... and {len(players) - 20} more")
