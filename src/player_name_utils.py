"""
Player Name Normalization Utilities

Handles common player name variations between different data sources (props, game results, etc.)
to ensure consistent matching.

Common issues this handles:
1. Accented characters (Dončić -> Doncic, Porziņģis -> Porzingis)
2. Suffix variations (Jr. vs Jr, III vs Iii)
3. Nickname vs full name (Herb Jones vs Herbert Jones)
4. Initials (P.J. vs PJ vs Pj)

Date: 2025-11-20
Author: Myles Thomas
"""

import pandas as pd
import unicodedata


def remove_accents(text):
    """
    Remove accents/diacritics from text.
    
    Examples:
        - Luka Dončić -> Luka Doncic
        - Kristaps Porziņģis -> Kristaps Porzingis
        - Bogdan Bogdanović -> Bogdan Bogdanovic
    """
    if pd.isna(text):
        return text
    
    # Normalize to NFD (decompose), filter out combining characters
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def normalize_player_name(name, keep_case=False):
    """
    Normalize a single player name for consistent matching.
    
    Args:
        name: Player name string
        keep_case: If True, preserves original case. If False, converts to Title Case.
    
    Returns:
        Normalized player name
    """
    if pd.isna(name):
        return name
    
    # Strip whitespace
    name = name.strip()
    
    # Remove ALL periods (handles A.J. vs AJ, P.J. vs PJ, Jr. vs Jr, etc.)
    name = name.replace('.', '')
    
    # Convert to Title Case unless keep_case is True
    if not keep_case:
        name = name.title()
    
    # Remove accents
    name = remove_accents(name)
    
    # Remove ONLY generational numbers at END of name (III, II, IV, V)
    # Keep Jr and Sr as they're more stable across sources
    # Use endswith to avoid removing letters from middle of names (e.g., "Valanciunas")
    if name.endswith(' Iii'):
        name = name[:-4]
    elif name.endswith(' Ii'):
        name = name[:-3]
    elif name.endswith(' Iv'):
        name = name[:-3]
    elif name.endswith(' V'):
        name = name[:-2]
    
    # Clean up multiple spaces
    name = ' '.join(name.split())
    
    return name


def get_name_mappings():
    """
    Get dictionary of known player name variations.
    
    MAPPING DIRECTION:
        Key (left side) = Name from ODDS API (The Odds API, props data)
        Value (right side) = Name from NBA API (nba_api, cache, game results)
    
    Usage: After normalizing both sides, apply this mapping to convert
           Odds API names to match what's in the NBA API cache.
    
    Returns:
        Dictionary {odds_api_name: nba_api_name}
    """
    return {
        # ===================================================================
        # NICKNAMES: Odds API vs NBA API use different versions  
        # (Applied AFTER normalization, so use normalized forms)
        # ===================================================================
        # Odds API (normalized) → NBA API (normalized)
        'Herb Jones': 'Herbert Jones',            # Odds uses nickname, NBA uses full
        'Moe Wagner': 'Moritz Wagner',            # Odds uses nickname, NBA uses full
        'Nicolas Claxton': 'Nic Claxton',         # Odds uses full, NBA uses nickname
        'Ron Holland': 'Ronald Holland',          # Odds uses "Ron", NBA uses "Ronald" 
        'Vincent Williams Jr': 'Vince Williams Jr',  # Odds uses full, NBA uses nickname
        
        # ===================================================================
        # NAME VARIATIONS
        # ===================================================================
        # Odds API (normalized) → NBA API (normalized)
        'Derrick Jones': 'Derrick Jones Jr',
        'Bruce Brown Jr': 'Bruce Brown',
        'Kenyon Martin Jr': 'Kj Martin',
        'Paul Reed Jr': 'Paul Reed',
        'Carlton Carrington': 'Bub Carrington',   # Rookie name change
    }


def normalize_player_names_df(df, player_col='player'):
    """
    Normalize all player names in a DataFrame column.
    
    Args:
        df: DataFrame containing player names
        player_col: Name of the column containing player names
    
    Returns:
        DataFrame with normalized player names
    """
    df = df.copy()
    
    # Apply basic normalization
    df[player_col] = df[player_col].apply(normalize_player_name)
    
    # Apply known name mappings
    name_mappings = get_name_mappings()
    df[player_col] = df[player_col].replace(name_mappings)
    
    return df


def find_similar_player_names(target_name, player_list, threshold=0.6):
    """
    Find similar player names from a list using fuzzy matching.
    
    Useful for debugging when exact matches aren't found.
    
    Args:
        target_name: The name to search for
        player_list: List of player names to search in
        threshold: Similarity threshold (0-1)
    
    Returns:
        List of (name, similarity_score) tuples, sorted by score
    """
    from difflib import SequenceMatcher
    
    similarities = []
    
    for name in player_list:
        ratio = SequenceMatcher(None, target_name.lower(), name.lower()).ratio()
        if ratio >= threshold:
            similarities.append((name, ratio))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)


def check_player_name_match(props_df, results_df, props_col='player', results_col='player'):
    """
    Check which players from props exist in game results.
    
    Useful for debugging name mismatch issues.
    
    Args:
        props_df: DataFrame with prop data
        results_df: DataFrame with game results
        props_col: Column name for player in props_df
        results_col: Column name for player in results_df
    
    Returns:
        Dictionary with:
            - 'matched': List of players found in both
            - 'missing': List of players in props but not in results
            - 'suggestions': Dict mapping missing players to suggested matches
    """
    props_players = set(props_df[props_col].dropna().unique())
    results_players = set(results_df[results_col].dropna().unique())
    
    matched = props_players & results_players
    missing = props_players - results_players
    
    # For missing players, find similar names
    suggestions = {}
    for player in missing:
        similar = find_similar_player_names(player, list(results_players), threshold=0.6)
        if similar:
            suggestions[player] = similar[:3]  # Top 3 matches
    
    return {
        'matched': sorted(matched),
        'missing': sorted(missing),
        'suggestions': suggestions
    }


def print_name_mismatch_report(props_df, results_df, props_col='player', results_col='player'):
    """
    Print a detailed report of player name mismatches.
    
    Args:
        props_df: DataFrame with prop data
        results_df: DataFrame with game results
        props_col: Column name for player in props_df
        results_col: Column name for player in results_df
    """
    print("="*70)
    print("PLAYER NAME MATCH REPORT")
    print("="*70)
    print()
    
    report = check_player_name_match(props_df, results_df, props_col, results_col)
    
    props_count = len(set(props_df[props_col].dropna().unique()))
    matched_count = len(report['matched'])
    missing_count = len(report['missing'])
    
    print(f"Total players in props: {props_count}")
    print(f"Matched in game results: {matched_count} ({matched_count/props_count*100:.1f}%)")
    print(f"Missing from game results: {missing_count} ({missing_count/props_count*100:.1f}%)")
    print()
    
    if missing_count > 0:
        print("Players in props but NOT in game results:")
        print()
        
        for player in report['missing'][:20]:  # Show first 20
            print(f"  {player}")
            if player in report['suggestions']:
                print(f"    Possible matches: {[s[0] for s in report['suggestions'][player]]}")
        
        if missing_count > 20:
            print(f"  ... and {missing_count - 20} more")
        print()
    else:
        print("✅ All players matched!")
    
    print("="*70)


if __name__ == '__main__':
    # Demo/test the functions
    test_names = [
        "Luka Dončić",
        "Kristaps Porziņģis", 
        "Bogdan Bogdanović",
        "P.J. Washington",
        "Herb Jones",
        "Jimmy Butler III",
        "Derrick Jones Jr.",
        "Gary Trent Jr."
    ]
    
    print("Player Name Normalization Demo")
    print("="*70)
    print()
    
    for name in test_names:
        normalized = normalize_player_name(name)
        print(f"{name:30} -> {normalized}")
    print()

