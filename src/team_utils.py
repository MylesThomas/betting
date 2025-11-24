"""
Team utilities for NBA betting analysis.

Provides player-to-team mapping using a multi-tier approach:
1. Intersection method: For players appearing in multiple games
2. Historical cross-reference: Match recent game results with tonight's game
3. NBA API fallback: Query live roster data for edge cases

This approach handles:
- Current rosters from multiple data sources
- Recent trades (player's historical team vs tonight's game)
- Name normalization for consistent matching
- Real-time roster updates via API fallback

Usage:
    from src.team_utils import add_team_column_from_props
    
    # Add team column to dataframe (recommended)
    props_df = add_team_column_from_props(props_df)
    
    # Or build mapping manually
    player_team_map = build_player_team_mapping(props_df)

Strategy:
    Multi-tier approach:
    1. Intersection: If player appears in multiple games, find common team
    2. Historical match: Check if player's recent team matches tonight's game
    3. NBA API: Query current rosters as fallback for uncertain cases

Author: Myles Thomas
Date: 2025-11-22
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Set
from collections import Counter
from functools import lru_cache
from datetime import datetime, timedelta

# Add parent to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

from player_name_utils import normalize_player_name
from config import PLAYER_TEAM_CACHE_MAX_AGE_HOURS
from config_loader import get_file_path


# Constants
GAME_RESULTS_PATH = Path(__file__).parent.parent / get_file_path('nba_game_results_current')
PLAYER_TEAM_CACHE_PATH = Path(__file__).parent.parent / get_file_path('player_team_cache')


# Team abbreviation to full name mapping (for NBA teams only)
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}

# Reverse mapping: full name -> abbreviation
TEAM_NAME_TO_ABBR = {v: k for k, v in NBA_TEAMS.items()}


@lru_cache(maxsize=1)
def load_historical_team_mapping() -> Dict[str, str]:
    """
    Load player-to-team mapping from historical game results.
    
    Returns:
        Dict mapping normalized player names to team abbreviations
        Based on most recent game each player played
    """
    if not GAME_RESULTS_PATH.exists():
        return {}
    
    try:
        # Load game results
        df = pd.read_csv(GAME_RESULTS_PATH)
        
        # Filter to NBA teams only (exclude G-League)
        df = df[df['team'].isin(NBA_TEAMS.keys())]
        
        # Normalize player names
        df['player_normalized'] = df['player'].apply(normalize_player_name)
        
        # Sort by date (most recent last)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Get most recent team for each player
        mapping = df.groupby('player_normalized')['team'].last().to_dict()
        
        return mapping
    
    except Exception as e:
        print(f"Error loading historical team mapping: {e}")
        return {}


def parse_teams_from_game(game_str: str) -> tuple:
    """
    Parse team abbreviations from game string.
    
    Args:
        game_str: Game string like "Los Angeles Clippers @ Charlotte Hornets"
        
    Returns:
        Tuple of (away_team_abbr, home_team_abbr)
        Example: ('LAC', 'CHA')
    """
    if not game_str or ' @ ' not in game_str:
        return (None, None)
    
    try:
        away, home = game_str.split(' @ ')
        away = away.strip()
        home = home.strip()
        
        # Convert full names to abbreviations
        away_abbr = TEAM_NAME_TO_ABBR.get(away)
        home_abbr = TEAM_NAME_TO_ABBR.get(home)
        
        return (away_abbr, home_abbr)
    except:
        return (None, None)


def load_player_team_cache() -> Dict[str, Dict]:
    """
    Load cached player-to-team mapping from CSV file.
    
    CSV Format:
        player_normalized,team,timestamp
        lebron james,LAL,2025-11-22 10:30:00
        john collins,LAC,2025-11-22 10:30:00
    
    Returns:
        Dict with 'mapping' (player->team) and 'timestamp' (when cached)
        Returns empty dict if cache doesn't exist or is invalid
    """
    try:
        if not PLAYER_TEAM_CACHE_PATH.exists():
            return {'mapping': {}, 'timestamp': None}
        
        # Read CSV into DataFrame
        cache_df = pd.read_csv(PLAYER_TEAM_CACHE_PATH)
        
        # Convert to dict mapping
        mapping = dict(zip(cache_df['player_normalized'], cache_df['team']))
        
        # Get most recent timestamp
        timestamp = cache_df['timestamp'].iloc[0] if len(cache_df) > 0 else None
        
        return {'mapping': mapping, 'timestamp': timestamp}
    except Exception as e:
        return {'mapping': {}, 'timestamp': None}


def save_player_team_cache(mapping: Dict[str, str]) -> None:
    """
    Save player-to-team mapping to CSV cache file.
    
    Args:
        mapping: Dict mapping normalized player names to team abbreviations
    """
    try:
        # Ensure data directory exists
        PLAYER_TEAM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        timestamp = datetime.now().isoformat()
        cache_df = pd.DataFrame([
            {'player_normalized': player, 'team': team, 'timestamp': timestamp}
            for player, team in mapping.items()
        ])
        
        # Save to CSV
        cache_df.to_csv(PLAYER_TEAM_CACHE_PATH, index=False)
    except Exception as e:
        # Fail silently - caching is optional
        pass


def is_cache_stale(cache_timestamp: Optional[str], max_age_hours: int = 24) -> bool:
    """
    Check if cache is stale (older than max_age_hours).
    
    Args:
        cache_timestamp: ISO format timestamp string
        max_age_hours: Maximum age in hours before cache is considered stale
        
    Returns:
        True if cache is stale or invalid, False otherwise
    """
    if not cache_timestamp:
        return True
    
    try:
        cache_time = datetime.fromisoformat(cache_timestamp)
        age = datetime.now() - cache_time
        return age > timedelta(hours=max_age_hours)
    except:
        return True


def lookup_player_team_from_api(player_name: str, game_teams: Set[str] = None) -> Optional[str]:
    """
    Lookup player's current team using NBA API.
    
    This is the PRIMARY method for determining player teams, as it provides
    the most up-to-date roster information including recent trades.
    
    Args:
        player_name: Player name (will be normalized)
        game_teams: Optional set of teams in the game (for validation)
        
    Returns:
        Team abbreviation or None if not found
        
    Strategy:
        1. Query nba_api for current player roster data
        2. Find player by name (with normalized fuzzy matching)
        3. Return current team from API
        4. Validate against game teams if provided
    """
    try:
        from nba_api.stats.static import players
        import requests
        import urllib3
        
        # Disable SSL warnings for development
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Find player ID from static data
        all_players = players.get_players()
        normalized_search = normalize_player_name(player_name)
        
        player_dict = None
        for p in all_players:
            if normalize_player_name(p['full_name']) == normalized_search:
                player_dict = p
                break
        
        if not player_dict:
            # Player not found in API
            return None
        
        # Query NBA stats API directly with SSL verification disabled
        player_id = player_dict['id']
        url = f'https://stats.nba.com/stats/commonplayerinfo'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://stats.nba.com/',
        }
        params = {
            'PlayerID': player_id,
            'LeagueID': '00'
        }
        
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10,
            verify=False  # Disable SSL verification (for development)
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Parse the response
        if 'resultSets' not in data or len(data['resultSets']) == 0:
            return None
        
        headers_list = data['resultSets'][0]['headers']
        row_data = data['resultSets'][0]['rowSet']
        
        if len(row_data) == 0:
            return None
        
        # Find team abbreviation column
        try:
            team_idx = headers_list.index('TEAM_ABBREVIATION')
            team_abbr = row_data[0][team_idx]
        except (ValueError, IndexError):
            return None
        
        # Handle special cases (free agents, retired, etc.)
        if not team_abbr or team_abbr == '' or pd.isna(team_abbr):
            return None
        
        # If game_teams provided, validate the result
        if game_teams and team_abbr not in game_teams:
            # API returned a team not in tonight's game - might be stale or error
            # Still return it, but caller can decide what to do
            pass
        
        return team_abbr
        
    except ImportError:
        # requests or nba_api not installed - skip this method
        return None
    except Exception as e:
        # API error - skip this method gracefully
        # Common errors: SSL cert verification, rate limiting, network issues
        return None


def build_player_team_mapping(df: pd.DataFrame, player_col: str = 'player', game_col: str = 'game', use_cache: bool = True) -> Dict[str, str]:
    """
    Build player-to-team mapping using multi-tier approach with caching.
    
    Caching Strategy:
        - Loads cached mapping from JSON file if available and fresh (<24hrs old)
        - Only queries NBA API for players NOT in cache
        - Saves updated mapping to cache for future runs
        - This dramatically speeds up subsequent loads (instant vs 60+ seconds)
    
    Manual Cache Update:
        If you need to manually rebuild the cache (e.g., after trades):
        1. Run: python scripts/build_full_roster_cache.py
        2. Then convert to player_team_cache format (see streamlit_app/app.py docstring)
        - run: python scripts/update_player_cache.py
        
        Or use the "Invalidate Cache" button in the Streamlit dashboard sidebar.
    

    Strategy (in order of priority):
        0. Cache Lookup (if enabled):
           - Check CSV cache file first
           - Skip API calls for cached players
           - Refresh cache if older than PLAYER_TEAM_CACHE_MAX_AGE_HOURS (7 days default)
        
        1. NBA API (PRIMARY):
           - Query live roster data from nba_api
           - Most accurate and up-to-date (handles trades immediately)
           - Validates against tonight's game teams
           - Falls back gracefully if API unavailable
           - NOTE: Currently disabled due to timeout issues. Re-enable by uncommenting
             the lookup_player_team_from_api() call in this function.
        
        2. Intersection Method:
           - If player appears in multiple games, find common team
           - Most reliable when available (but rare in single-day data)
        
        3. Historical Cross-Reference:
           - Load player's recent team from game results
           - If it matches one of tonight's game teams, use it
           - Good for established players who haven't been traded
        
        4. NBA API (Unvalidated):
           - Use API result even if it doesn't match tonight's game
           - API data is trusted over historical data
    
    Args:
        df: DataFrame with player and game columns (from props/arbs)
        player_col: Name of player column (default: 'player')
        game_col: Name of game column (default: 'game')
        use_cache: Whether to use cached mappings (default: True)
        
    Returns:
        Dict mapping normalized player names to team abbreviations
        Example: {'lebron james': 'LAL', 'john collins': 'LAC', ...}
    """
    if player_col not in df.columns or game_col not in df.columns:
        return {}
    
    # Load cache if enabled
    cache_data = load_player_team_cache() if use_cache else {'mapping': {}, 'timestamp': None}
    cached_mapping = cache_data.get('mapping', {})
    cache_timestamp = cache_data.get('timestamp')
    cache_is_stale = is_cache_stale(cache_timestamp, max_age_hours=PLAYER_TEAM_CACHE_MAX_AGE_HOURS)
    
    # Load historical team mapping once
    historical_mapping = load_historical_team_mapping()
    
    final_mapping = {}
    players_needing_api_lookup = []
    
    # Process each unique player
    for player in df[player_col].unique():
        normalized_name = normalize_player_name(player)
        
        # Check cache first (if not stale)
        if use_cache and not cache_is_stale and normalized_name in cached_mapping:
            final_mapping[normalized_name] = cached_mapping[normalized_name]
            continue
        
        # Get this player's game(s)
        player_rows = df[df[player_col] == player]
        games = player_rows[game_col].unique()
        
        if len(games) == 0:
            continue
        
        # Parse all teams from all games
        all_teams_sets = []
        for game in games:
            away, home = parse_teams_from_game(game)
            teams_in_game = {away, home} - {None}
            if teams_in_game:
                all_teams_sets.append(teams_in_game)
        
        if not all_teams_sets:
            continue
        
        all_game_teams = set()
        for teams_set in all_teams_sets:
            all_game_teams.update(teams_set)
        
        # Store for potential API lookup
        players_needing_api_lookup.append((player, normalized_name, all_game_teams, all_teams_sets))
    
    # Batch API lookups for uncached players
    for player, normalized_name, all_game_teams, all_teams_sets in players_needing_api_lookup:
        
        # METHOD 1: NBA API (Primary - Most Accurate for Current Rosters)
        # TEMPORARILY DISABLED - API calls timing out
        # 
        # To update cache manually instead of using API:
        #   1. Run: python scripts/build_full_roster_cache.py
        #   2. Convert to player_team_cache format (see streamlit_app/app.py docstring)
        # 
        # Try to lookup player's current team via API first
        # This handles trades and is always up-to-date
        # api_team = lookup_player_team_from_api(player, all_game_teams)
        api_team = None  # Disabled: lookup_player_team_from_api(player, all_game_teams)
        if api_team and api_team in all_game_teams:
            # API returned a team that matches tonight's game - use it!
            final_mapping[normalized_name] = api_team
            continue
        
        # METHOD 2: INTERSECTION
        # If player appears in multiple games, find common team
        if len(all_teams_sets) > 1:
            common_teams = set.intersection(*all_teams_sets)
            if len(common_teams) == 1:
                # Perfect! Player's team appears in all their games
                final_mapping[normalized_name] = list(common_teams)[0]
                continue
            elif len(common_teams) > 1:
                # Multiple common teams (shouldn't happen) - use API or historical to disambiguate
                if api_team and api_team in common_teams:
                    final_mapping[normalized_name] = api_team
                    continue
                historical_team = historical_mapping.get(normalized_name)
                if historical_team and historical_team in common_teams:
                    final_mapping[normalized_name] = historical_team
                    continue
        
        # METHOD 3: HISTORICAL CROSS-REFERENCE
        # Check if player's historical team matches one of tonight's teams
        historical_team = historical_mapping.get(normalized_name)
        if historical_team and historical_team in all_game_teams:
            final_mapping[normalized_name] = historical_team
            continue
        
        # METHOD 4: NBA API (Even if not validated against game)
        # If API returned a team (even if not in tonight's game), trust it
        if api_team:
            final_mapping[normalized_name] = api_team
            continue
        
        # LAST RESORT: Use most common team or first team
        # This is imperfect but better than nothing
        team_counter = Counter()
        for teams_set in all_teams_sets:
            for team in teams_set:
                team_counter[team] += 1
        
        if team_counter:
            most_common_team = team_counter.most_common(1)[0][0]
            final_mapping[normalized_name] = most_common_team
    
    # Save updated mapping to cache (merge with existing cache)
    if use_cache:
        merged_mapping = {**cached_mapping, **final_mapping}
        save_player_team_cache(merged_mapping)
    
    return final_mapping


def add_team_column_from_props(df: pd.DataFrame, player_col: str = 'player', game_col: str = 'game') -> pd.DataFrame:
    """
    Add team column to props/arbs dataframe using multi-tier mapping.
    
    This is the recommended way to add teams - it uses intersection,
    historical data, and API fallback for maximum accuracy.
    
    Args:
        df: DataFrame with player and game columns
        player_col: Name of player column (default: 'player')  
        game_col: Name of game column (default: 'game')
        
    Returns:
        DataFrame with new 'team' column added
        
    Example:
        >>> props_df = pd.read_csv('arb_threes_20251122.csv')
        >>> props_df = add_team_column_from_props(props_df)
        >>> props_df[['player', 'team', 'game']]
           player         team    game
        0  LeBron James   LAL     Los Angeles Lakers @ Boston Celtics
        1  Stephen Curry  GSW     Golden State Warriors @ Denver Nuggets
    """
    df = df.copy()
    
    # Build mapping using multi-tier approach
    mapping = build_player_team_mapping(df, player_col, game_col)
    
    # Apply mapping
    df['player_normalized'] = df[player_col].apply(normalize_player_name)
    df['team'] = df['player_normalized'].map(mapping)
    df = df.drop('player_normalized', axis=1)
    
    return df


def get_team_full_name(team_abbr: str) -> str:
    """
    Get full team name from abbreviation.
    
    Args:
        team_abbr: Team abbreviation (e.g., 'LAL')
        
    Returns:
        Full team name (e.g., 'Los Angeles Lakers')
    """
    return NBA_TEAMS.get(team_abbr, team_abbr)


def get_all_teams() -> List[str]:
    """
    Get list of all NBA teams.
    
    Returns:
        Sorted list of NBA team abbreviations
        Example: ['ATL', 'BOS', 'BKN', 'CHA', ...]
    """
    return sorted(NBA_TEAMS.keys())


if __name__ == '__main__':
    # Test the module with sample data
    print("Testing team_utils.py with multi-tier approach")
    print("=" * 50)
    
    # Create sample dataframe with various scenarios
    sample_data = {
        'player': [
            'LeBron James',      # Should be LAL (in Lakers @ Celtics)
            'Anthony Black',     # Should be ORL (in Knicks @ Magic)
            'John Collins',      # Recently traded - should be CHA
            'LeBron James',      # Same player in another game (for intersection test)
        ],
        'game': [
            'Los Angeles Lakers @ Boston Celtics',
            'New York Knicks @ Orlando Magic',
            'Los Angeles Clippers @ Charlotte Hornets',
            'Los Angeles Lakers @ Miami Heat',  # LeBron in 2 games - LAL is common
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nSample data:")
    print(df)
    print()
    
    # Build mapping
    mapping = build_player_team_mapping(df)
    print("Player-Team Mapping:")
    for player, team in mapping.items():
        full_name = get_team_full_name(team)
        print(f"  {player}: {team} ({full_name})")
    print()
    
    # Add team column
    df = add_team_column_from_props(df)
    print("DataFrame with team column:")
    print(df[['player', 'team', 'game']])
