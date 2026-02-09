"""
Debug script to trace where Kawhi Leonard gets filtered out during backtest.

This simulates the backtest process for 2026-02-08 only and tracks Kawhi at each step.

Run:
    cd ~/dev/betting
    python3 tmp/debug_backtest_kawhi.py
"""

import sys
import os
from pathlib import Path
import boto3
import pandas as pd
from io import StringIO
import unicodedata

# Add src to path
root_dir = Path(__file__).resolve()
while not (root_dir / '.gitignore').exists() and root_dir != root_dir.parent:
    root_dir = root_dir.parent
sys.path.insert(0, str(root_dir / 'src'))

# =============================================================================
# COPY FUNCTIONS FROM LAMBDA
# =============================================================================

def remove_accents(text):
    if pd.isna(text):
        return text
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

def normalize_player_name(name):
    if pd.isna(name):
        return name
    name = name.strip().replace('.', '').title()
    name = remove_accents(name)
    if name.endswith(' Iii'):
        name = name[:-4]
    elif name.endswith(' Ii'):
        name = name[:-3]
    elif name.endswith(' Iv'):
        name = name[:-3]
    elif name.endswith(' V'):
        name = name[:-2]
    name = ' '.join(name.split())
    mappings = {
        'Herb Jones': 'Herbert Jones',
        'Moe Wagner': 'Moritz Wagner',
        'Nicolas Claxton': 'Nic Claxton',
        'Ron Holland': 'Ronald Holland',
        'Vincent Williams Jr': 'Vince Williams Jr',
        'Derrick Jones': 'Derrick Jones Jr',
        'Bruce Brown Jr': 'Bruce Brown',
        'Kenyon Martin Jr': 'Kj Martin',
        'Paul Reed Jr': 'Paul Reed',
        'Carlton Carrington': 'Bub Carrington',
        'Alfred Joel Horford Reynoso': 'Al Horford',
        'Anthony Davis Jr': 'Anthony Davis',
    }
    return mappings.get(name, name)

# =============================================================================
# MAIN DEBUG SCRIPT
# =============================================================================

def main():
    """Debug where Kawhi gets filtered."""
    print("="*80)
    print("DEBUG: Tracing Kawhi Leonard Through Backtest Pipeline")
    print("="*80)
    print()
    
    DATE = '2026-02-08'
    SEASON = '2025-26'
    
    s3 = boto3.client('s3')
    
    # ==========================================================================
    # STEP 1: Load Props
    # ==========================================================================
    print("STEP 1: Loading player props...")
    response = s3.get_object(Bucket='the-odds-api-mt', Key=f'nba/historical_player_props/{SEASON}/{DATE}.csv')
    df_props = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    
    print(f"   Total props loaded: {len(df_props):,}")
    
    # Normalize and add game_date
    df_props['player_normalized'] = df_props['player'].apply(normalize_player_name)
    df_props['game_date'] = DATE
    
    kawhi_props = df_props[df_props['player_normalized'] == 'Kawhi Leonard']
    print(f"   ✅ Kawhi props found: {len(kawhi_props)}")
    if len(kawhi_props) > 0:
        kawhi_points_props = kawhi_props[kawhi_props['market'] == 'player_points']
        print(f"   ✅ Kawhi player_points props: {len(kawhi_points_props)}")
        if len(kawhi_points_props) > 0:
            print(f"      Prop lines: {sorted(kawhi_points_props['prop_line'].unique())}")
    print()
    
    # ==========================================================================
    # STEP 2: Load Game Logs
    # ==========================================================================
    print("STEP 2: Loading game logs...")
    response = s3.get_object(Bucket='nba-api-mt', Key=f'player_game_logs/{SEASON}/{DATE}.csv')
    df_games = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    
    print(f"   Total game logs loaded: {len(df_games):,}")
    
    # Normalize
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
    df_games['game_date'] = df_games['GAME_DATE'].dt.date.astype(str)
    df_games['player_normalized'] = df_games['PLAYER_NAME'].apply(normalize_player_name)
    
    # Filter to players who actually played
    df_games_played = df_games[df_games['MIN'].notna() & (df_games['MIN'] > 0)].copy()
    
    kawhi_game = df_games_played[df_games_played['player_normalized'] == 'Kawhi Leonard']
    print(f"   ✅ Kawhi game log found: {len(kawhi_game)}")
    if len(kawhi_game) > 0:
        print(f"      Points scored: {kawhi_game.iloc[0]['PTS']}")
        print(f"      Minutes: {kawhi_game.iloc[0]['MIN']}")
        print(f"      Team: {kawhi_game.iloc[0]['TEAM_NAME']}")
        print(f"      Matchup: {kawhi_game.iloc[0]['MATCHUP']}")
    print()
    
    # ==========================================================================
    # STEP 3: Load Game Lines (Spreads)
    # ==========================================================================
    print("STEP 3: Loading game lines...")
    response = s3.get_object(Bucket='the-odds-api-mt', Key=f'nba/historical_game_lines/{SEASON}/nba_game_lines_{DATE}.csv')
    df_lines_raw = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    
    print(f"   Total game lines loaded: {len(df_lines_raw):,}")
    
    # Process spreads
    spreads = df_lines_raw[df_lines_raw['market'] == 'spread'].copy()
    spreads['game_date'] = DATE
    
    # Calculate consensus
    consensus = spreads.groupby(['game_id', 'game_date', 'away_team', 'home_team', 'market']).agg({
        'away_line': 'mean',
        'home_line': 'mean'
    }).reset_index()
    
    df_lines = consensus[['game_id', 'game_date', 'away_team', 'home_team', 'away_line', 'home_line']]
    df_lines.columns = ['game_id', 'game_date', 'away_team', 'home_team', 'away_spread', 'home_spread']
    
    print(f"   ✅ Consensus spreads: {len(df_lines)}")
    
    # Check LAC game
    lac_game = df_lines[(df_lines['away_team'] == 'Los Angeles Clippers') | (df_lines['home_team'] == 'Los Angeles Clippers')]
    if len(lac_game) > 0:
        print(f"   ✅ LAC game found:")
        print(f"      {lac_game.iloc[0]['away_team']} @ {lac_game.iloc[0]['home_team']}")
        print(f"      Spread: {lac_game.iloc[0]['away_spread']} / {lac_game.iloc[0]['home_spread']}")
    print()
    
    # ==========================================================================
    # STEP 4: Join Props (aggregate first)
    # ==========================================================================
    print("STEP 4: Joining props to game logs...")
    
    # Filter to player_points only and aggregate
    df_props_points = df_props[df_props['market'] == 'player_points'].copy()
    props_agg = df_props_points.groupby(['player_normalized', 'game_date']).agg({
        'prop_line': 'mean'
    }).reset_index()
    props_agg.columns = ['player_normalized', 'game_date', 'points_line']
    
    print(f"   Props aggregated: {len(props_agg)} player-dates")
    
    kawhi_props_agg = props_agg[props_agg['player_normalized'] == 'Kawhi Leonard']
    print(f"   ✅ Kawhi in aggregated props: {len(kawhi_props_agg)}")
    if len(kawhi_props_agg) > 0:
        print(f"      Average prop line: {kawhi_props_agg.iloc[0]['points_line']:.1f}")
    print()
    
    # Join props to games
    df_merged = df_games_played.merge(props_agg, on=['player_normalized', 'game_date'], how='left')
    
    print(f"   After merge: {len(df_merged)} records")
    
    kawhi_merged = df_merged[df_merged['player_normalized'] == 'Kawhi Leonard']
    print(f"   ✅ Kawhi after props merge: {len(kawhi_merged)}")
    if len(kawhi_merged) > 0:
        print(f"      points_line: {kawhi_merged.iloc[0]['points_line']}")
        print(f"      Is NaN: {pd.isna(kawhi_merged.iloc[0]['points_line'])}")
    print()
    
    # =========================================================================
    # STEP 5: Join Game Lines (Spreads)
    # =========================================================================
    print("STEP 5: Joining game lines (spreads)...")
    
    # Normalize Odds API team names to NBA API format (source of truth)
    ODDS_TO_NBA_TEAM_MAP = {'Los Angeles Clippers': 'LA Clippers'}
    df_lines['away_team'] = df_lines['away_team'].replace(ODDS_TO_NBA_TEAM_MAP)
    df_lines['home_team'] = df_lines['home_team'].replace(ODDS_TO_NBA_TEAM_MAP)
    
    # Determine home/away
    df_merged['is_home'] = ~df_merged['MATCHUP'].str.contains('@')
    
    # Split and join
    df_merged_home = df_merged[df_merged['is_home']].copy()
    df_merged_away = df_merged[~df_merged['is_home']].copy()
    
    df_merged_home = df_merged_home.merge(
        df_lines[['game_date', 'home_team', 'home_spread']],
        left_on=['game_date', 'TEAM_NAME'],
        right_on=['game_date', 'home_team'],
        how='left'
    )
    df_merged_home['team_spread'] = df_merged_home['home_spread']
    
    df_merged_away = df_merged_away.merge(
        df_lines[['game_date', 'away_team', 'away_spread']],
        left_on=['game_date', 'TEAM_NAME'],
        right_on=['game_date', 'away_team'],
        how='left'
    )
    df_merged_away['team_spread'] = df_merged_away['away_spread']
    
    df_merged = pd.concat([df_merged_home, df_merged_away], ignore_index=True)
    
    print(f"   After spread merge: {len(df_merged)} records")
    
    kawhi_merged = df_merged[df_merged['player_normalized'] == 'Kawhi Leonard']
    print(f"   ✅ Kawhi after spread merge: {len(kawhi_merged)}")
    if len(kawhi_merged) > 0:
        print(f"      team_spread: {kawhi_merged.iloc[0]['team_spread']}")
        print(f"      Is NaN: {pd.isna(kawhi_merged.iloc[0]['team_spread'])}")
        print(f"      TEAM_NAME: {kawhi_merged.iloc[0]['TEAM_NAME']}")
    print()
    
    # ==========================================================================
    # STEP 6: Filter to rows with props
    # ==========================================================================
    print("STEP 6: Filtering to rows with props (THIS IS WHERE FILTERING HAPPENS)...")
    
    before_filter = len(df_merged)
    df_merged = df_merged[df_merged['points_line'].notna()].copy()
    after_filter = len(df_merged)
    
    print(f"   Before filter: {before_filter} records")
    print(f"   After filter: {after_filter} records")
    print(f"   Filtered out: {before_filter - after_filter} records")
    print()
    
    kawhi_merged = df_merged[df_merged['player_normalized'] == 'Kawhi Leonard']
    if len(kawhi_merged) > 0:
        print(f"   ✅ KAWHI SURVIVED THE FILTER!")
        print(f"      points_line: {kawhi_merged.iloc[0]['points_line']}")
        print(f"      team_spread: {kawhi_merged.iloc[0]['team_spread']}")
        print(f"      PTS: {kawhi_merged.iloc[0]['PTS']}")
    else:
        print(f"   ❌ KAWHI WAS FILTERED OUT!")
        print()
        print("   This means df_merged[df_merged['points_line'].notna()] removed Kawhi")
        print("   Checking why...")
        
        # Go back and check
        df_merged_unfiltered = pd.concat([df_merged_home, df_merged_away], ignore_index=True)
        kawhi_unfiltered = df_merged_unfiltered[df_merged_unfiltered['player_normalized'] == 'Kawhi Leonard']
        
        if len(kawhi_unfiltered) > 0:
            print()
            print("   Kawhi's row before filter:")
            print(f"      points_line: {kawhi_unfiltered.iloc[0]['points_line']}")
            print(f"      Is NaN: {pd.isna(kawhi_unfiltered.iloc[0]['points_line'])}")
            
            if pd.isna(kawhi_unfiltered.iloc[0]['points_line']):
                print()
                print("   ROOT CAUSE: points_line is NaN!")
                print("   This means the props join failed.")
                print()
                print("   Checking join keys:")
                print(f"      Game log player_normalized: '{kawhi_unfiltered.iloc[0]['player_normalized']}'")
                print(f"      Game log game_date: '{kawhi_unfiltered.iloc[0]['game_date']}'")
                print()
                print(f"      Props agg has Kawhi: {len(kawhi_props_agg) > 0}")
                if len(kawhi_props_agg) > 0:
                    print(f"      Props agg player_normalized: '{kawhi_props_agg.iloc[0]['player_normalized']}'")
                    print(f"      Props agg game_date: '{kawhi_props_agg.iloc[0]['game_date']}'")
    
    print()
    print("="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
