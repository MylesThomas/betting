"""
Simple team utilities - just load cache, no writes.

For Streamlit Cloud where we can't write files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))
from player_name_utils import normalize_player_name

try:
    from config_loader import get_file_path
    PLAYER_TEAM_CACHE_PATH = Path(__file__).parent.parent / get_file_path('player_team_cache')
except:
    # Fallback if config_loader not available
    PLAYER_TEAM_CACHE_PATH = Path(__file__).parent.parent / "data" / "02_cache" / "player_team_cache.csv"


def load_player_teams() -> Dict[str, str]:
    """
    Load player-to-team mapping from cache file (read-only).
    
    Returns:
        Dict mapping normalized player names to team abbreviations
        Returns empty dict if cache doesn't exist
    """
    try:
        if not PLAYER_TEAM_CACHE_PATH.exists():
            print(f"⚠️ Cache file not found: {PLAYER_TEAM_CACHE_PATH}")
            return {}
        
        cache_df = pd.read_csv(PLAYER_TEAM_CACHE_PATH)
        mapping = dict(zip(cache_df['player_normalized'], cache_df['team']))
        print(f"✅ Loaded {len(mapping)} players from cache")
        return mapping
    
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
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

