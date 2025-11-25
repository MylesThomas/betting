"""
Utility functions for working with NBA game logs.

This module provides:
- Column mappings for NBA API game log data
- Parsing and cleaning functions
- Data structure definitions
- Helper functions for game log analysis

Used by:
- scripts/build_season_game_logs.py
- Any script that needs to work with game log data
"""

import pandas as pd


# ============================================================================
# COLUMN MAPPINGS
# ============================================================================

# Standard NBA API game log columns mapped to our clean column names
NBA_GAMELOG_COLUMNS = {
    'PLAYER_NAME': 'player',
    'PLAYER_ID': 'player_id',
    'GAME_DATE': 'date',
    'MATCHUP': 'matchup',
    'WL': 'result',
    'MIN': 'minutes',
    'FGM': 'fgm',
    'FGA': 'fga',
    'FG_PCT': 'fg_pct',
    'FG3M': 'threes_made',
    'FG3A': 'threes_attempted',
    'FG3_PCT': 'three_pct',
    'FTM': 'ftm',
    'FTA': 'fta',
    'FT_PCT': 'ft_pct',
    'OREB': 'oreb',
    'DREB': 'dreb',
    'REB': 'reb',
    'AST': 'ast',
    'STL': 'stl',
    'BLK': 'blk',
    'TOV': 'tov',
    'PF': 'pf',
    'PTS': 'pts',
    'PLUS_MINUS': 'plus_minus',
    'GAME_ID': 'game_id',
}


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_minutes(min_str):
    """
    Parse minutes string to float.
    
    Handles various formats:
    - "34:25" -> 34.42 (34 minutes 25 seconds)
    - "DNP" -> 0.0 (Did Not Play)
    - None -> 0.0
    - 34 -> 34.0
    
    Args:
        min_str: Minutes string from NBA API
    
    Returns:
        Float representing minutes played
    
    Examples:
        >>> parse_minutes("34:25")
        34.416666666666664
        >>> parse_minutes("DNP")
        0.0
        >>> parse_minutes(None)
        0.0
    """
    if pd.isna(min_str) or min_str == 'DNP' or min_str == '' or min_str == 'N/A':
        return 0.0
    
    try:
        if ':' in str(min_str):
            mins, secs = str(min_str).split(':')
            return float(mins) + float(secs) / 60
        else:
            return float(min_str)
    except:
        return 0.0


def extract_opponent(matchup_str):
    """
    Extract opponent team abbreviation from matchup string.
    
    Args:
        matchup_str: Matchup string like "LAL vs. GSW" or "LAL @ GSW"
    
    Returns:
        Opponent team abbreviation (e.g., "GSW")
    
    Examples:
        >>> extract_opponent("LAL vs. GSW")
        'GSW'
        >>> extract_opponent("LAL @ GSW")
        'GSW'
    """
    if pd.isna(matchup_str):
        return None
    
    try:
        # Split on "vs." or "@"
        if ' vs. ' in matchup_str:
            return matchup_str.split(' vs. ')[1].strip()
        elif ' @ ' in matchup_str:
            return matchup_str.split(' @ ')[1].strip()
        else:
            return None
    except:
        return None


def determine_home_away(matchup_str):
    """
    Determine if game was home or away.
    
    Args:
        matchup_str: Matchup string like "LAL vs. GSW" or "LAL @ GSW"
    
    Returns:
        'HOME' if home game, 'AWAY' if away game, None if unknown
    
    Examples:
        >>> determine_home_away("LAL vs. GSW")
        'HOME'
        >>> determine_home_away("LAL @ GSW")
        'AWAY'
    """
    if pd.isna(matchup_str):
        return None
    
    if ' vs. ' in matchup_str:
        return 'HOME'
    elif ' @ ' in matchup_str:
        return 'AWAY'
    else:
        return None


def get_empty_gamelog_dataframe():
    """
    Create an empty DataFrame with the correct game log structure.
    
    Useful for:
    - Creating placeholder files for players with no games
    - Initializing empty DataFrames with correct schema
    
    Returns:
        Empty DataFrame with all expected game log columns
    """
    # Get all output column names from the mapping
    columns = list(NBA_GAMELOG_COLUMNS.values())
    
    # Add the extra columns that parse_game_logs adds
    columns.extend(['opponent', 'home_away', 'player_normalized'])
    
    return pd.DataFrame(columns=columns)


def parse_game_logs(df, add_normalized_name=True):
    """
    Parse and clean NBA game log DataFrame.
    
    Takes raw NBA API game log data and:
    - Renames columns to clean names
    - Parses dates
    - Converts minutes to float
    - Extracts opponent and home/away
    - Optionally adds normalized player name
    - Sorts by date (most recent first)
    
    Args:
        df: Raw DataFrame from NBA API
        add_normalized_name: Whether to add normalized player name column
    
    Returns:
        Cleaned DataFrame with standardized columns
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Select only columns that exist in the data
    available_cols = [col for col in NBA_GAMELOG_COLUMNS.keys() if col in df.columns]
    df_clean = df[available_cols].copy()
    
    # Rename columns
    df_clean = df_clean.rename(columns=NBA_GAMELOG_COLUMNS)
    
    # Parse date
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Convert minutes to float
    if 'minutes' in df_clean.columns:
        df_clean['minutes'] = df_clean['minutes'].apply(parse_minutes)
    
    # Extract opponent and home/away from matchup
    if 'matchup' in df_clean.columns:
        df_clean['opponent'] = df_clean['matchup'].apply(extract_opponent)
        df_clean['home_away'] = df_clean['matchup'].apply(determine_home_away)
    
    # Add normalized player name for matching
    if add_normalized_name and 'player' in df_clean.columns:
        from src.player_name_utils import normalize_player_name
        df_clean['player_normalized'] = df_clean['player'].apply(normalize_player_name)
    
    # Sort by date (most recent first)
    if 'date' in df_clean.columns:
        df_clean = df_clean.sort_values('date', ascending=False).reset_index(drop=True)
    
    return df_clean


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def filter_by_minutes(df, min_minutes=10):
    """
    Filter game logs to only include games where player played >= min_minutes.
    
    Useful to exclude garbage time appearances and DNP-CD situations.
    
    Args:
        df: Game log DataFrame
        min_minutes: Minimum minutes threshold
    
    Returns:
        Filtered DataFrame
    """
    if df.empty or 'minutes' not in df.columns:
        return df
    
    return df[df['minutes'] >= min_minutes].copy()


def filter_by_date_range(df, start_date=None, end_date=None):
    """
    Filter game logs by date range.
    
    Args:
        df: Game log DataFrame
        start_date: Start date (inclusive), can be string or datetime
        end_date: End date (inclusive), can be string or datetime
    
    Returns:
        Filtered DataFrame
    """
    if df.empty or 'date' not in df.columns:
        return df
    
    df_filtered = df.copy()
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered['date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered['date'] <= end_date]
    
    return df_filtered


def get_last_n_games(df, n=5, player_name=None):
    """
    Get last N games for a player.
    
    Args:
        df: Game log DataFrame (should be sorted by date descending)
        n: Number of games to return
        player_name: Optional player name to filter by
    
    Returns:
        DataFrame with last N games
    """
    if df.empty:
        return df
    
    df_filtered = df.copy()
    
    if player_name and 'player' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['player'] == player_name]
    
    return df_filtered.head(n)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_player_averages(df, stat_columns=None):
    """
    Calculate per-game averages for specified stats.
    
    Args:
        df: Game log DataFrame
        stat_columns: List of stat columns to average (default: key shooting stats)
    
    Returns:
        Series with averages
    """
    if df.empty:
        return pd.Series()
    
    if stat_columns is None:
        stat_columns = ['threes_made', 'threes_attempted', 'three_pct', 'pts', 'ast', 'reb']
    
    # Only include columns that exist
    available_cols = [col for col in stat_columns if col in df.columns]
    
    return df[available_cols].mean()


def calculate_streak(df, stat_column, threshold, comparison='>='):
    """
    Calculate current streak for a stat vs threshold.
    
    Example: How many consecutive games has player hit OVER 2.5 threes?
    
    Args:
        df: Game log DataFrame (must be sorted by date descending - most recent first)
        stat_column: Column to check (e.g., 'threes_made')
        threshold: Threshold value (e.g., 2.5)
        comparison: Comparison operator ('>=', '>', '<', '<=')
    
    Returns:
        Integer representing streak length
    
    Examples:
        >>> # Last 5 games: [3, 4, 2, 3, 1] threes made, threshold 2.5
        >>> calculate_streak(df, 'threes_made', 2.5, '>=')
        2  # Hit in last 2 games (3, 4)
    """
    if df.empty or stat_column not in df.columns:
        return 0
    
    streak = 0
    
    for _, row in df.iterrows():
        value = row[stat_column]
        
        if pd.isna(value):
            break
        
        # Check if condition is met
        if comparison == '>=':
            hit = value >= threshold
        elif comparison == '>':
            hit = value > threshold
        elif comparison == '<':
            hit = value < threshold
        elif comparison == '<=':
            hit = value <= threshold
        else:
            break
        
        if hit:
            streak += 1
        else:
            break
    
    return streak


# ============================================================================
# DATA QUALITY FUNCTIONS
# ============================================================================

def validate_game_log(df):
    """
    Validate that game log DataFrame has required columns and data.
    
    Args:
        df: Game log DataFrame
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    required_cols = ['player', 'date', 'threes_made', 'threes_attempted']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            errors.append("'date' column is not datetime type")
    
    return len(errors) == 0, errors


if __name__ == '__main__':
    # Demo/test
    print("NBA Game Log Utilities")
    print("=" * 50)
    print()
    
    print("Available functions:")
    print("  - parse_game_logs(df)")
    print("  - filter_by_minutes(df, min_minutes=10)")
    print("  - filter_by_date_range(df, start_date, end_date)")
    print("  - get_last_n_games(df, n=5)")
    print("  - calculate_player_averages(df)")
    print("  - calculate_streak(df, 'threes_made', 2.5, '>=')")
    print()
    
    # Test parsing functions
    print("Testing parse_minutes():")
    test_cases = ["34:25", "DNP", None, "45:00", "0:30"]
    for test in test_cases:
        result = parse_minutes(test)
        print(f"  {str(test):>10} -> {result:.2f} minutes")
    
    print()
    print("Testing extract_opponent():")
    test_cases = ["LAL vs. GSW", "LAL @ GSW", "BOS @ MIA"]
    for test in test_cases:
        opponent = extract_opponent(test)
        home_away = determine_home_away(test)
        print(f"  {test:>20} -> Opponent: {opponent}, Location: {home_away}")

