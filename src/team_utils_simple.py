"""
Simple team utilities - just load cache, no writes.

For Streamlit Cloud where we can't write files.
Reads player-team cache from S3.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import sys
import boto3
from io import BytesIO

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))
from player_name_utils import normalize_player_name

# S3 Configuration
S3_BUCKET = 'nba-betting-mt'
S3_CACHE_KEY = 'data/02_cache/player_team_cache.csv'

# Local fallback path (for dev environments without S3 access)
try:
    from config_loader import get_file_path
    LOCAL_CACHE_PATH = Path(__file__).parent.parent / get_file_path('player_team_cache')
except:
    LOCAL_CACHE_PATH = Path(__file__).parent.parent / "data" / "02_cache" / "player_team_cache.csv"


def load_player_teams() -> Dict[str, str]:
    """
    Load player-to-team mapping from S3 cache (with local fallback).
    
    Returns:
        Dict mapping normalized player names to team abbreviations
        Returns empty dict if cache doesn't exist
    """
    # Try S3 first
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_CACHE_KEY)
        cache_df = pd.read_csv(BytesIO(obj['Body'].read()))
        mapping = dict(zip(cache_df['player_normalized'], cache_df['team']))
        print(f"✅ Loaded {len(mapping)} players from S3 cache")
        return mapping
    
    except Exception as s3_error:
        print(f"⚠️  S3 read failed, trying local fallback: {s3_error}")
        
        # Fallback to local file
        try:
            if not LOCAL_CACHE_PATH.exists():
                print(f"❌ Local cache not found: {LOCAL_CACHE_PATH}")
                return {}
            
            cache_df = pd.read_csv(LOCAL_CACHE_PATH)
            mapping = dict(zip(cache_df['player_normalized'], cache_df['team']))
            print(f"✅ Loaded {len(mapping)} players from local cache")
            return mapping
        
        except Exception as local_error:
            print(f"❌ Error loading local cache: {local_error}")
            return {}


def add_team_column_simple(df: pd.DataFrame, player_col: str = 'player') -> pd.DataFrame:
    """
    Add team column to dataframe using cache ONLY.
    
    If player not in cache → NULL (you need to manually update cache).
    No writes, no API calls, no complexity.
    
    Args:
        df: DataFrame with player column
        player_col: Name of player column (default: 'player')
        
    Returns:
        DataFrame with new 'team' column added (NULL if not in cache)
    """
    df = df.copy()
    
    # Load cache
    mapping = load_player_teams()
    
    # Normalize player names, then apply name mappings for nickname variations
    from player_name_utils import get_name_mappings
    name_mappings = get_name_mappings()
    
    df['player_normalized'] = df[player_col].apply(normalize_player_name)
    # Apply mappings to convert Odds API nicknames to NBA API nicknames
    df['player_normalized'] = df['player_normalized'].map(lambda x: name_mappings.get(x, x))
    df['team'] = df['player_normalized'].map(mapping)
    df = df.drop('player_normalized', axis=1)
    
    # Count how many unique players are missing
    missing_rows = df['team'].isna().sum()
    if missing_rows > 0:
        missing_players = df[df['team'].isna()][player_col].nunique()
        print(f"⚠️ {missing_players} unique players not in cache ({missing_rows} rows showing NULL)")
    
    return df


if __name__ == '__main__':
    # Test
    print("Testing simple team utils...")
    print(f"Cache path: {PLAYER_TEAM_CACHE_PATH}")
    print()
    
    mapping = load_player_teams()
    print(f"\nFirst 5 players in cache:")
    for player, team in list(mapping.items())[:5]:
        print(f"  {player}: {team}")

