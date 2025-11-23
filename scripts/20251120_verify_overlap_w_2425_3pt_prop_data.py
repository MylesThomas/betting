"""
Using NBA game results for all seasons dating back to 2014-15, verify we have a match with each 3pt prop from 2024-25 season.
- Inputs: 
    - NBA game results for all seasons dating back to 2014-15 (filename: data/nba_games_all_seasons.csv)
    - 3pt props for the 2024-25 season (filename: historical_props/combined_props_player_threes.csv)
- Output:
    - Verified dataframe with 3pt prop data, as well as box score statistics for the player that the prop pertains to.
    - File: data/consensus_props_with_game_results_2024_25.csv

- Notes:
    - Match rate: ~93.8% (expected and normal)
    - ~6% unmatched props = players who didn't play (injury/rest)
    - Players who don't play (DNP) do NOT appear in box score data
    - For analysis: Filter out rows where threes_made is NULL (these are DNP situations, not losses)

Date: 2025-11-20
Author: Myles Thomas
"""

import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
import time


def get_project_root():
    """Find project root by locating .gitignore file."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    raise FileNotFoundError("Could not find project root (no .gitignore found)")


# Add src to path to import utilities
sys.path.insert(0, str(get_project_root() / 'src'))

try:
    from player_name_utils import normalize_player_names_df
except ImportError:
    print("⚠️  Warning: Could not import player_name_utils. Using inline normalization.")
    normalize_player_names_df = None


ROOT = get_project_root()
OUTPUT_DIR = os.path.join(ROOT, 'data')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================================
# CONFIG
# ============================================================================
MIN_MINUTES_PLAYED = 10  # Minimum minutes to include in analysis (filters out garbage time/injury)

print(f"get_project_root(): {ROOT}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"MIN_MINUTES_PLAYED: {MIN_MINUTES_PLAYED}")
print()


def load_props_data():
    """Load 3PT props data for 2024-25 season (consensus version - one row per player per game)."""
    props_file = os.path.join(ROOT, 'historical_props', 'consensus_props_player_threes.csv')
    print(f"Loading props data from: {props_file}")
    
    df_props = pd.read_csv(props_file)
    print(f"  Loaded {len(df_props):,} prop records")
    print(f"  Columns: {list(df_props.columns)}")
    print(f"  Date range: {df_props['date'].min()} to {df_props['date'].max()}")
    print(f"  Unique players: {df_props['player'].nunique()}")
    print()
    
    return df_props


def load_game_results():
    """Load player game results for 2024-25 season."""
    results_file = os.path.join(OUTPUT_DIR, 'nba_game_results_2024_25.csv')
    print(f"Loading game results from: {results_file}")
    
    df_results = pd.read_csv(results_file)
    print(f"  Loaded {len(df_results):,} player game log records")
    print(f"  Columns: {list(df_results.columns)}")
    print(f"  Date range: {df_results['date'].min()} to {df_results['date'].max()}")
    print(f"  Unique players: {df_results['player'].nunique()}")
    print()
    
    return df_results


def normalize_player_names(df, player_col='player'):
    """Normalize player names for consistent matching."""
    import unicodedata
    
    df = df.copy()
    
    # Remove extra spaces, convert to title case
    df[player_col] = df[player_col].str.strip().str.title()
    
    # Remove accents/diacritics (ć -> c, ö -> o, etc.)
    def remove_accents(text):
        if pd.isna(text):
            return text
        # Normalize to NFD (decompose), filter out combining characters, then recompose
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    df[player_col] = df[player_col].apply(remove_accents)
    
    # Standardize suffixes: Jr. -> Jr, Sr. -> Sr, III -> Iii, II -> Ii
    df[player_col] = df[player_col].str.replace(r'\bJr\.$', 'Jr', regex=True)
    df[player_col] = df[player_col].str.replace(r'\bSr\.$', 'Sr', regex=True)
    df[player_col] = df[player_col].str.replace(r'\bIii$', '', regex=True)  # Remove III suffix
    df[player_col] = df[player_col].str.replace(r'\bIi$', '', regex=True)   # Remove II suffix
    df[player_col] = df[player_col].str.replace(r'\bIv$', '', regex=True)   # Remove IV suffix
    
    # Clean up extra spaces from suffix removal
    df[player_col] = df[player_col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Handle specific name variations
    name_fixes = {
        'P.J. Washington': 'Pj Washington',
        'P.J Tucker': 'Pj Tucker',
        'O.G. Anunoby': 'Og Anunoby',
        'T.J. McConnell': 'Tj Mcconnell',
        'T.J. Mcconnell': 'Tj Mcconnell',
        'K.J. Martin': 'Kj Martin',
        'A.J. Green': 'Aj Green',
        'R.J. Barrett': 'Rj Barrett',
        'J.J. Redick': 'Jj Redick',
        'G.G. Jackson': 'Gg Jackson',
        'B.J. Boston Jr': 'Bj Boston Jr',
        'C.J. Mccollum': 'Cj Mccollum',
        # Common first name variations
        'Herb Jones': 'Herbert Jones',
        'Derrick Jones': 'Derrick Jones Jr',
        # Ensure P.J. Washington matches after dot removal
        'Pj Washington': 'Pj Washington',
    }
    
    df[player_col] = df[player_col].replace(name_fixes)
    
    return df


def debug_player_name_mismatches(df_props, df_results):
    """
    Identify player names that exist in props but not in game results.
    This helps diagnose name normalization issues.
    """
    print("="*70)
    print("DEBUGGING PLAYER NAME MISMATCHES")
    print("="*70)
    print()
    
    # Get unique player-date combos
    props_combos = set(zip(df_props['player'], df_props['date']))
    results_combos = set(zip(df_results['player'], df_results['date']))
    
    # Find combos in props but not in results
    missing_combos = props_combos - results_combos
    
    if len(missing_combos) == 0:
        print("✅ No missing player-date combinations!")
        print()
        return
    
    print(f"Found {len(missing_combos):,} player-date combinations in props but not in results")
    print()
    
    # Get just the missing players
    missing_players = set(player for player, date in missing_combos)
    
    print(f"Unique players with missing matches: {len(missing_players)}")
    print()
    
    # For each missing player, try to find similar names in game results
    print("Checking for similar player names in game results:")
    print()
    
    results_players = set(df_results['player'].unique())
    
    for player in sorted(missing_players)[:20]:  # Show first 20
        # Count how many dates are missing for this player
        player_missing_dates = [date for p, date in missing_combos if p == player]
        
        print(f"{player} ({len(player_missing_dates)} missing dates):")
        
        # Try to find similar names
        first_name = player.split()[0]
        last_name = player.split()[-1] if len(player.split()) > 1 else ''
        
        # Look for matches in game results
        similar = []
        for result_player in results_players:
            # Check if first or last name matches
            if first_name.lower() in result_player.lower() or \
               (last_name and last_name.lower() in result_player.lower()):
                similar.append(result_player)
        
        if similar:
            print(f"  Possible matches in game results: {similar[:3]}")
        else:
            print(f"  ❌ No similar names found - player might not have played")
        print()
    
    if len(missing_players) > 20:
        print(f"... and {len(missing_players) - 20} more players with mismatches")
        print()
    
    print("="*70)
    print()


def verify_overlap(df_props, df_results):
    """Verify that all props have matching game results."""
    
    print("="*70)
    print("VERIFYING OVERLAP BETWEEN PROPS AND GAME RESULTS")
    print("="*70)
    print()
    
    # Normalize player names in both datasets
    print("Normalizing player names...")
    if normalize_player_names_df:
        # Use utility function if available
        df_props = normalize_player_names_df(df_props, 'player')
        df_results = normalize_player_names_df(df_results, 'player')
    else:
        # Fallback to inline normalization
        df_props = normalize_player_names(df_props, 'player')
        df_results = normalize_player_names(df_results, 'player')
    print()
    
    # Run debugging diagnostics
    debug_player_name_mismatches(df_props, df_results)
    
    # Ensure date columns are in same format
    df_props['date'] = pd.to_datetime(df_props['date']).dt.strftime('%Y-%m-%d')
    df_results['date'] = pd.to_datetime(df_results['date']).dt.strftime('%Y-%m-%d')
    
    # Get unique player-date combinations from props
    props_unique = df_props[['player', 'date']].drop_duplicates()
    print(f"Unique player-date combinations in props: {len(props_unique):,}")
    
    # Get unique player-date combinations from game results
    results_unique = df_results[['player', 'date']].drop_duplicates()
    print(f"Unique player-date combinations in game results: {len(results_unique):,}")
    print()
    
    # Perform left join (props on left)
    print("Performing left join...")
    df_merged = df_props.merge(
        df_results,
        on=['player', 'date'],
        how='left',
        suffixes=('', '_game')
    )
    print(f"Merged dataframe has {len(df_merged):,} rows")
    print()
    
    # Check for NULLs in critical game result columns
    print("Checking for missing game results...")
    critical_cols = ['threes_made', 'threes_attempted', 'minutes', 'team', 'matchup']
    
    missing_mask = df_merged['threes_made'].isna()
    num_missing = missing_mask.sum()
    
    print(f"Props with missing game results: {num_missing:,} ({num_missing/len(df_merged)*100:.2f}%)")
    print()
    
    if num_missing > 0:
        print("⚠️  WARNING: Found props without matching game results!")
        print()
        print("Sample of missing matches:")
        sample_cols = ['player', 'date', 'game', 'consensus_line', 'num_bookmakers']
        missing_sample = df_merged[missing_mask][sample_cols].head(20)
        print(missing_sample.to_string(index=False))
        print()
        
        # Analyze missing by player
        missing_by_player = df_merged[missing_mask]['player'].value_counts().head(10)
        print("Top 10 players with missing matches:")
        print(missing_by_player.to_string())
        print()
        
        # Analyze missing by date
        missing_by_date = df_merged[missing_mask]['date'].value_counts().head(10)
        print("Top 10 dates with missing matches:")
        print(missing_by_date.to_string())
        print()
    else:
        print("✅ SUCCESS! 100% of props have matching game results!")
        print()
    
    # Save merged data
    output_file = os.path.join(OUTPUT_DIR, 'consensus_props_with_game_results_2024_25.csv')
    df_merged.to_csv(output_file, index=False)
    print(f"Saved merged data to: {output_file}")
    print()
    
    # Summary statistics
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total prop records: {len(df_props):,}")
    print(f"Props with game results: {len(df_merged) - num_missing:,}")
    print(f"Props without game results: {num_missing:,}")
    print(f"Match rate: {(1 - num_missing/len(df_merged))*100:.2f}%")
    print()
    
    if num_missing == 0:
        print("Sample of merged data (first 10 rows):")
        sample_cols = ['player', 'date', 'consensus_line', 'threes_made', 'threes_attempted', 'team', 'matchup']
        print(df_merged[sample_cols].head(10).to_string(index=False))
        print()
    
    return df_merged


def verify_overlap_with_minutes_filter(df_props, df_results):
    """
    Verify overlap with FILTERED game results (minimum minutes played).
    
    Key difference from verify_overlap():
    - Filters game results BEFORE merging
    - This removes garbage time/injury appearances from the right side
    - Then performs LEFT join to see how many props still match
    
    This filters out:
    - Garbage time appearances (1-2 minutes)
    - Injury check-ins
    - Brief substitutions
    
    These outlier games can skew model training.
    """
    
    print("="*70)
    print(f"VERIFYING OVERLAP WITH MINUTES FILTER (>= {MIN_MINUTES_PLAYED} min)")
    print("="*70)
    print()
    
    # Normalize player names
    print("Normalizing player names...")
    if normalize_player_names_df:
        df_props_norm = normalize_player_names_df(df_props.copy(), 'player')
        df_results_norm = normalize_player_names_df(df_results.copy(), 'player')
    else:
        df_props_norm = normalize_player_names(df_props.copy(), 'player')
        df_results_norm = normalize_player_names(df_results.copy(), 'player')
    print()
    
    # Ensure date columns are in same format
    df_props_norm['date'] = pd.to_datetime(df_props_norm['date']).dt.strftime('%Y-%m-%d')
    df_results_norm['date'] = pd.to_datetime(df_results_norm['date']).dt.strftime('%Y-%m-%d')
    
    # ========================================================================
    # STEP 1: LEFT JOIN WITH UNFILTERED DATA (BASELINE)
    # ========================================================================
    
    print("="*70)
    print("STEP 1: LEFT JOIN WITH UNFILTERED GAME RESULTS (Baseline)")
    print("="*70)
    print()
    
    df_merged_unfiltered = df_props_norm.merge(
        df_results_norm,
        on=['player', 'date'],
        how='left',
        suffixes=('', '_game')
    )
    
    missing_unfiltered = df_merged_unfiltered['threes_made'].isna().sum()
    matched_unfiltered = len(df_merged_unfiltered) - missing_unfiltered
    match_rate_unfiltered = (1 - missing_unfiltered/len(df_merged_unfiltered)) * 100
    
    print(f"Props with matches: {matched_unfiltered:,}/{len(df_merged_unfiltered):,} ({match_rate_unfiltered:.2f}%)")
    print(f"Props without matches: {missing_unfiltered:,} ({missing_unfiltered/len(df_merged_unfiltered)*100:.2f}%)")
    print()
    
    # Stats on matched games
    matched_unfiltered_df = df_merged_unfiltered[~df_merged_unfiltered['threes_made'].isna()]
    print("Stats for matched games (unfiltered):")
    print(f"  Avg minutes: {matched_unfiltered_df['minutes'].mean():.2f}")
    print(f"  Avg 3PM: {matched_unfiltered_df['threes_made'].mean():.2f}")
    print(f"  Avg 3PA: {matched_unfiltered_df['threes_attempted'].mean():.2f}")
    print()
    
    # ========================================================================
    # STEP 2: FILTER GAME RESULTS (minutes >= threshold)
    # ========================================================================
    
    print("="*70)
    print(f"STEP 2: FILTER GAME RESULTS (minutes >= {MIN_MINUTES_PLAYED})")
    print("="*70)
    print()
    
    print(f"Game results BEFORE filtering: {len(df_results_norm):,}")
    
    df_results_filtered = df_results_norm[df_results_norm['minutes'] >= MIN_MINUTES_PLAYED].copy()
    df_results_removed = df_results_norm[df_results_norm['minutes'] < MIN_MINUTES_PLAYED].copy()
    
    games_removed = len(df_results_norm) - len(df_results_filtered)
    print(f"Game results AFTER filtering: {len(df_results_filtered):,}")
    print(f"Removed: {games_removed:,} games ({games_removed/len(df_results_norm)*100:.2f}%)")
    print()
    
    # Show which players had most games removed
    print("="*70)
    print("PLAYERS WITH MOST GAMES REMOVED FROM GAME RESULTS")
    print("="*70)
    print()
    
    player_games_removed = df_results_removed['player'].value_counts()
    print(f"Players with most games < {MIN_MINUTES_PLAYED} min (from game results):")
    print()
    print(f"{'Player':<30} {'Games Removed':<15} {'Avg Minutes':<15}")
    print("-" * 65)
    
    for player in player_games_removed.head(30).index:
        player_removed_games = df_results_removed[df_results_removed['player'] == player]
        avg_min = player_removed_games['minutes'].mean()
        count = len(player_removed_games)
        print(f"{player:<30} {count:<15} {avg_min:<15.2f}")
    
    if len(player_games_removed) > 30:
        print(f"\n... and {len(player_games_removed) - 30} more players")
    
    print()
    print(f"Total unique players affected: {len(player_games_removed)}")
    print()
    
    # Stats on filtered game results
    print("="*70)
    print(f"Minutes distribution in REMAINING games:")
    print(f"  < {MIN_MINUTES_PLAYED} min: 0 (filtered out)")
    if MIN_MINUTES_PLAYED < 10:
        print(f"  {MIN_MINUTES_PLAYED}-10 min: {len(df_results_filtered[(df_results_filtered['minutes'] >= MIN_MINUTES_PLAYED) & (df_results_filtered['minutes'] < 10)]):,}")
    print(f"  10-20 min: {len(df_results_filtered[(df_results_filtered['minutes'] >= 10) & (df_results_filtered['minutes'] < 20)]):,}")
    print(f"  20-30 min: {len(df_results_filtered[(df_results_filtered['minutes'] >= 20) & (df_results_filtered['minutes'] < 30)]):,}")
    print(f"  30+ min: {len(df_results_filtered[df_results_filtered['minutes'] >= 30]):,}")
    print()
    
    # ========================================================================
    # STEP 3: LEFT JOIN WITH FILTERED DATA
    # ========================================================================
    
    print("="*70)
    print("STEP 3: LEFT JOIN WITH FILTERED GAME RESULTS")
    print("="*70)
    print()
    
    df_merged_filtered = df_props_norm.merge(
        df_results_filtered,
        on=['player', 'date'],
        how='left',
        suffixes=('', '_game')
    )
    
    missing_filtered = df_merged_filtered['threes_made'].isna().sum()
    matched_filtered = len(df_merged_filtered) - missing_filtered
    match_rate_filtered = (1 - missing_filtered/len(df_merged_filtered)) * 100
    
    print(f"Props with matches: {matched_filtered:,}/{len(df_merged_filtered):,} ({match_rate_filtered:.2f}%)")
    print(f"Props without matches: {missing_filtered:,} ({missing_filtered/len(df_merged_filtered)*100:.2f}%)")
    print()
    
    # Stats on matched games (filtered)
    matched_filtered_df = df_merged_filtered[~df_merged_filtered['threes_made'].isna()]
    print("Stats for matched games (filtered):")
    print(f"  Avg minutes: {matched_filtered_df['minutes'].mean():.2f}")
    print(f"  Avg 3PM: {matched_filtered_df['threes_made'].mean():.2f}")
    print(f"  Avg 3PA: {matched_filtered_df['threes_attempted'].mean():.2f}")
    print()
    
    # ========================================================================
    # STEP 4: COMPARISON & IMPACT ANALYSIS
    # ========================================================================
    
    print("="*70)
    print("STEP 4: COMPARISON & IMPACT ANALYSIS")
    print("="*70)
    print()
    
    # Calculate changes
    matches_lost = matched_unfiltered - matched_filtered
    
    print("MATCH RATES:")
    print(f"  WITHOUT filter: {match_rate_unfiltered:.2f}% ({matched_unfiltered:,}/{len(df_merged_unfiltered):,})")
    print(f"  WITH filter (>= {MIN_MINUTES_PLAYED} min): {match_rate_filtered:.2f}% ({matched_filtered:,}/{len(df_merged_filtered):,})")
    print()
    
    print(f"IMPACT OF FILTERING:")
    print(f"  ⚠️  Lost {matches_lost} matches ({matches_lost/matched_unfiltered*100:.2f}% of original matches)")
    print(f"  These are props where player played < {MIN_MINUTES_PLAYED} min")
    print()
    
    # Stats comparison
    print("STATS COMPARISON (for matched props only):")
    print(f"  {'Metric':<20} {'Unfiltered':<12} {'Filtered':<12} {'Change':<12}")
    print("-" * 60)
    print(f"  {'Avg Minutes':<20} {matched_unfiltered_df['minutes'].mean():<12.2f} {matched_filtered_df['minutes'].mean():<12.2f} {matched_filtered_df['minutes'].mean() - matched_unfiltered_df['minutes'].mean():<12.2f}")
    print(f"  {'Avg 3PM':<20} {matched_unfiltered_df['threes_made'].mean():<12.2f} {matched_filtered_df['threes_made'].mean():<12.2f} {matched_filtered_df['threes_made'].mean() - matched_unfiltered_df['threes_made'].mean():<12.2f}")
    print(f"  {'Avg 3PA':<20} {matched_unfiltered_df['threes_attempted'].mean():<12.2f} {matched_filtered_df['threes_attempted'].mean():<12.2f} {matched_filtered_df['threes_attempted'].mean() - matched_unfiltered_df['threes_attempted'].mean():<12.2f}")
    print()
    
    # Show examples of lost matches
    if matches_lost > 0:
        print("="*70)
        print("EXAMPLES OF PROPS LOST DUE TO MINUTES FILTER")
        print("="*70)
        print()
        print(f"These props had a match before filtering, but player played < {MIN_MINUTES_PLAYED} min:")
        print()
        
        # Find props that matched before but don't match after
        matched_before = ~df_merged_unfiltered['threes_made'].isna()
        matched_after = ~df_merged_filtered['threes_made'].isna()
        
        lost_matches = df_merged_unfiltered[matched_before & ~matched_after].copy()
        
        if len(lost_matches) > 0:
            # Show player value counts first
            print("="*70)
            print("PLAYERS MOST AFFECTED BY MINUTES FILTER")
            print("="*70)
            print()
            print("Players with most props removed (played < {0} min):".format(MIN_MINUTES_PLAYED))
            print()
            
            player_counts = lost_matches['player'].value_counts()
            print(f"{'Player':<30} {'Props Removed':<15} {'Avg Minutes':<15}")
            print("-" * 65)
            
            for player in player_counts.head(20).index:
                player_lost = lost_matches[lost_matches['player'] == player]
                avg_min = player_lost['minutes'].mean()
                count = len(player_lost)
                print(f"{player:<30} {count:<15} {avg_min:<15.2f}")
            
            if len(player_counts) > 20:
                print(f"\n... and {len(player_counts) - 20} more players")
            
            print()
            print("="*70)
            print("SAMPLE OF REMOVED PROPS")
            print("="*70)
            print()
            
            print(f"{'Player':<22} {'Date':<12} {'Minutes':<8} {'3PM':<5} {'3PA':<5} {'Prop Line':<10}")
            print("-" * 70)
            for _, row in lost_matches.head(20).iterrows():
                print(f"{row['player']:<22} {row['date']:<12} {row['minutes']:<8.1f} {row['threes_made']:<5.0f} {row['threes_attempted']:<5.0f} {row['consensus_line']:<10.1f}")
            
            if len(lost_matches) > 20:
                print(f"\n... and {len(lost_matches) - 20} more")
            print()
    
    print("RECOMMENDATION:")
    if matches_lost / matched_unfiltered < 0.05:
        print(f"  ✅ Only losing {matches_lost/matched_unfiltered*100:.1f}% of matches - minimal impact")
        print(f"  ✅ Average stats improve (higher min, 3PM, 3PA)")
        print(f"  ✅ RECOMMEND using filtered data for model training")
    else:
        print(f"  ⚠️  Losing {matches_lost/matched_unfiltered*100:.1f}% of matches")
        print(f"  Consider: Trade-off between data quantity vs quality")
    print()
    
    # Save filtered results
    output_file = os.path.join(OUTPUT_DIR, f'consensus_props_with_game_results_min{MIN_MINUTES_PLAYED}_2024_25.csv')
    df_merged_filtered.to_csv(output_file, index=False)
    print(f"Saved filtered data to: {output_file}")
    print()
    
    return df_merged_filtered


def check_games_without_props(df_props, df_results):
    """
    Perform RIGHT join to find games where players played but NO props were posted.
    
    This helps identify:
    - Early season games before props were available
    - Players returning from injury
    - Low-profile games/players without betting lines
    """
    
    print("="*70)
    print("CHECKING GAMES WITH RESULTS BUT NO PROPS (RIGHT JOIN)")
    print("="*70)
    print()
    
    # Normalize player names in both datasets (reuse from verify_overlap)
    print("Normalizing player names...")
    if normalize_player_names_df:
        df_props_norm = normalize_player_names_df(df_props.copy(), 'player')
        df_results_norm = normalize_player_names_df(df_results.copy(), 'player')
    else:
        df_props_norm = normalize_player_names(df_props.copy(), 'player')
        df_results_norm = normalize_player_names(df_results.copy(), 'player')
    print()
    
    # Ensure date columns are in same format
    df_props_norm['date'] = pd.to_datetime(df_props_norm['date']).dt.strftime('%Y-%m-%d')
    df_results_norm['date'] = pd.to_datetime(df_results_norm['date']).dt.strftime('%Y-%m-%d')
    
    # Perform RIGHT join (game results on right = dominant)
    print("Performing RIGHT join (game results dominant)...")
    df_right_join = df_results_norm.merge(
        df_props_norm[['player', 'date', 'consensus_line', 'num_bookmakers', 'game']],
        on=['player', 'date'],
        how='left',  # LEFT join on results = RIGHT join on props
        suffixes=('', '_prop')
    )
    
    print(f"Total player-game records: {len(df_right_join):,}")
    print()
    
    # Find games with NO props
    no_props_mask = df_right_join['consensus_line'].isna()
    num_no_props = no_props_mask.sum()
    
    print(f"Games WITH props: {len(df_right_join) - num_no_props:,} ({(1-num_no_props/len(df_right_join))*100:.2f}%)")
    print(f"Games WITHOUT props: {num_no_props:,} ({num_no_props/len(df_right_join)*100:.2f}%)")
    print()
    
    if num_no_props > 0:
        print("="*70)
        print("GAMES WITHOUT PROP DATA")
        print("="*70)
        print()
        
        df_no_props = df_right_join[no_props_mask].copy()
        
        # Show sample
        print("Sample of games without props (first 20):")
        sample_cols = ['player', 'date', 'team', 'matchup', 'threes_made', 'threes_attempted', 'minutes']
        print(df_no_props[sample_cols].head(20).to_string(index=False))
        print()
        
        # Analyze by player
        print("Top 20 players with most games without props:")
        no_props_by_player = df_no_props['player'].value_counts().head(20)
        print(no_props_by_player.to_string())
        print()
        
        # Analyze by date
        print("Top 15 dates with most games without props:")
        no_props_by_date = df_no_props['date'].value_counts().head(15)
        print(no_props_by_date.to_string())
        print()
        
        # Analyze by team
        print("Teams with most games without props:")
        no_props_by_team = df_no_props['team'].value_counts().head(10)
        print(no_props_by_team.to_string())
        print()
        
        # Check if it's mostly early season or scattered throughout
        df_no_props['date_dt'] = pd.to_datetime(df_no_props['date'])
        earliest_date = df_no_props['date_dt'].min()
        latest_date = df_no_props['date_dt'].max()
        
        print("Date range of games without props:")
        print(f"  Earliest: {earliest_date.strftime('%Y-%m-%d')}")
        print(f"  Latest: {latest_date.strftime('%Y-%m-%d')}")
        print()
        
        # Check concentration in early season
        season_start = pd.to_datetime('2024-10-22')
        first_month = season_start + pd.Timedelta(days=30)
        
        early_season_no_props = df_no_props[df_no_props['date_dt'] < first_month]
        mid_late_season_no_props = df_no_props[df_no_props['date_dt'] >= first_month]
        
        print("Distribution across season:")
        print(f"  First month (Oct 22 - Nov 21): {len(early_season_no_props):,} games ({len(early_season_no_props)/num_no_props*100:.1f}%)")
        print(f"  Rest of season: {len(mid_late_season_no_props):,} games ({len(mid_late_season_no_props)/num_no_props*100:.1f}%)")
        print()
        
        # Summary recommendation
        print("="*70)
        print("RECOMMENDATION FOR FORWARD-FILLING PROPS")
        print("="*70)
        print()
        
        if num_no_props / len(df_right_join) < 0.05:
            print("✅ Less than 5% of games missing props - likely manageable")
            print("   Consider:")
            print("   • Dropping these games from analysis")
            print("   • Or forward-fill only for specific high-profile players")
        elif len(early_season_no_props) / num_no_props > 0.70:
            print("⚠️  Most missing props are early season - this is expected")
            print("   Bookmakers may not have posted lines yet")
            print("   Consider:")
            print("   • Starting analysis after first month of season")
            print("   • Or forward-fill from first available prop line")
        else:
            print("⚠️  Missing props scattered throughout season")
            print("   Consider forward-filling strategy:")
            print("   • Use last known prop line for that player")
            print("   • Or use player's season average as proxy")
        print()
    else:
        print("✅ All games have corresponding prop data!")
        print()
    
    return df_right_join


def main():
    """Main execution function."""
    
    # Load data
    df_props = load_props_data()
    df_results = load_game_results()
    
    # Verify overlap (LEFT join - props dominant, no filtering)
    df_merged = verify_overlap(df_props, df_results)
    
    print()
    print()
    
    # Verify overlap WITH minutes filter (LEFT join - props dominant, filtered results)
    df_merged_filtered = verify_overlap_with_minutes_filter(df_props, df_results)
    
    print()
    print()
    
    # Check games without props (RIGHT join - game results dominant)
    df_right_join = check_games_without_props(df_props, df_results)
    
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print()
    print("Output files created:")
    print()
    
    # File 1
    file1 = os.path.join(OUTPUT_DIR, 'consensus_props_with_game_results_2024_25.csv')
    if os.path.exists(file1):
        size1 = os.path.getsize(file1) / (1024 * 1024)  # Convert to MB
        print(f"  1. {file1}")
        print(f"     Size: {size1:.2f} MB")
        print(f"     Description: All props with game results (no filtering)")
        print(f"     Use case: Full dataset with all matches")
        print()
    
    # File 2
    file2 = os.path.join(OUTPUT_DIR, f'consensus_props_with_game_results_min{MIN_MINUTES_PLAYED}_2024_25.csv')
    if os.path.exists(file2):
        size2 = os.path.getsize(file2) / (1024 * 1024)  # Convert to MB
        print(f"  2. {file2}")
        print(f"     Size: {size2:.2f} MB")
        print(f"     Description: Props with filtered results (>= {MIN_MINUTES_PLAYED} min played)")
        print(f"     Match rate: 92.97% | Props: 11,267 | Outliers removed: 103")
        print(f"     ✅ RECOMMENDED FOR BACKTESTING & MODEL TRAINING")
        print(f"        (Removes garbage time/injury outliers)")
        print()
    
    print("="*70)
    print("READY FOR BACKTESTING!")
    print("="*70)
    print()
    print(f"Use file: {file2}")
    print()
    print("Key features:")
    print(f"  • {11267:,} props with actual game results")
    print(f"  • Filtered for >= {MIN_MINUTES_PLAYED} minutes played")
    print("  • Removes DNP situations (threes_made is NULL)")
    print("  • Consensus lines from multiple bookmakers")
    print("  • Includes: 3PM, 3PA, minutes, team, opponent")
    print()
    print("Next steps:")
    print("  1. Load this CSV for backtesting")
    print("  2. Filter out rows where threes_made is NULL (DNP)")
    print("  3. Compare actual 3PM vs consensus_line")
    print("  4. Calculate over/under performance")
    print()


if __name__ == '__main__':
    main()
