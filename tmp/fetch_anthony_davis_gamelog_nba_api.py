"""
Fetch Anthony Davis Game Log History using nba_api

Context:
--------
After exploring ESPN API endpoints (see fetch_anthony_davis_gamelog.py), 
it's clear that the nba_api package provides more reliable access to game log data.
This script uses nba_api to fetch Anthony Davis's game-by-game statistics.

Goal:
-----
1. Find Anthony Davis's player ID in the NBA API
2. Fetch his game log history for current season
3. Display game-by-game stats (points, rebounds, assists, etc.)
4. Save to CSV for analysis

Data Available:
--------------
Via PlayerGameLog endpoint:
- Date, opponent, home/away
- Minutes played
- FGM/FGA, 3PM/3PA, FTM/FTA
- Points, rebounds (offensive/defensive/total), assists
- Steals, blocks, turnovers, fouls
- Plus/minus, game result

nba_api Documentation:
---------------------
- GitHub: https://github.com/swar/nba_api
- Player finder: from nba_api.stats.static import players
- Game logs: from nba_api.stats.endpoints import playergamelog

Usage:
------
# Fetch current season (2025-26)
python tmp/fetch_anthony_davis_gamelog_nba_api.py

# Fetch specific season
python tmp/fetch_anthony_davis_gamelog_nba_api.py --season 2024-25

# Fetch entire career
python tmp/fetch_anthony_davis_gamelog_nba_api.py --season career

# Fetch different player
python tmp/fetch_anthony_davis_gamelog_nba_api.py --player "LeBron James" --season career

# Enable debug output to see all available columns
python tmp/fetch_anthony_davis_gamelog_nba_api.py --season career --debug

Installation (if needed):
-------------------------
pip install nba_api

Author: Thomas Myles
Date: 2026-02-04
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import argparse
import time

# Add src to path (finding root via .gitignore)
current_dir = Path(__file__).parent
repo_root = current_dir.parent
sys.path.insert(0, str(repo_root / 'src'))

try:
    from nba_api.stats.static import players
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.endpoints import commonplayerinfo
except ImportError:
    print("❌ nba_api not found. Install with: pip install nba_api")
    sys.exit(1)


# =============================================================================
# PLAYER SEARCH
# =============================================================================

def find_player(player_name):
    """
    Find player by name using nba_api.
    
    Args:
        player_name: Player's name (e.g., "Anthony Davis")
    
    Returns:
        Player dict with id, full_name, etc.
    """
    print(f"🔍 Searching for player: {player_name}")
    
    # Get all players
    all_players = players.get_players()
    
    # Search for player
    player = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
    
    if not player:
        print(f"   ❌ Player not found")
        return None
    
    if len(player) > 1:
        print(f"   ⚠️  Multiple players found:")
        for p in player:
            print(f"      - {p['full_name']} (ID: {p['id']}, Active: {p['is_active']})")
        # Return the active one or first one
        active = [p for p in player if p['is_active']]
        if active:
            player = active[0]
            print(f"   Using active player: {player['full_name']}")
        else:
            player = player[0]
            print(f"   Using first match: {player['full_name']}")
    else:
        player = player[0]
    
    print(f"   ✅ Found: {player['full_name']}")
    print(f"      Player ID: {player['id']}")
    print(f"      Active: {player['is_active']}")
    
    return player


# =============================================================================
# GAME LOG FETCHING
# =============================================================================

def get_player_career_years(player_id):
    """
    Get the range of seasons a player has been active.
    
    Args:
        player_id: NBA API player ID
    
    Returns:
        List of season strings (e.g., ['2012-13', '2013-14', ...])
    """
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = player_info.get_data_frames()[0]
        
        if df.empty:
            return []
        
        # Get from and to years
        from_year = df['FROM_YEAR'].iloc[0]
        to_year = df['TO_YEAR'].iloc[0]
        
        # Generate season strings
        seasons = []
        for year in range(int(from_year), int(to_year) + 1):
            season = f"{year}-{str(year + 1)[-2:]}"
            seasons.append(season)
        
        return seasons
        
    except Exception as e:
        print(f"   ⚠️  Could not fetch career info: {e}")
        return []


def extract_team_from_matchup(matchup):
    """
    Extract player's team from MATCHUP string.
    
    MATCHUP format:
    - "TEAM @ OPP" (away game)
    - "TEAM vs. OPP" (home game)
    
    Args:
        matchup: MATCHUP string (e.g., "DAL @ UTA")
    
    Returns:
        Team abbreviation (e.g., "DAL")
    """
    if pd.isna(matchup):
        return None
    
    # Split by @ or vs.
    if '@' in matchup:
        team = matchup.split('@')[0].strip()
    elif 'vs.' in matchup:
        team = matchup.split('vs.')[0].strip()
    else:
        return None
    
    return team


def extract_opponent_from_matchup(matchup):
    """
    Extract opponent team from MATCHUP string.
    
    Args:
        matchup: MATCHUP string (e.g., "DAL @ UTA")
    
    Returns:
        Opponent team abbreviation (e.g., "UTA")
    """
    if pd.isna(matchup):
        return None
    
    # Split by @ or vs.
    if '@' in matchup:
        opponent = matchup.split('@')[1].strip()
    elif 'vs.' in matchup:
        opponent = matchup.split('vs.')[1].strip()
    else:
        return None
    
    return opponent


def get_player_game_log(player_id, season='2025-26', fetch_career=False, debug=False):
    """
    Fetch player's game log for a season or entire career using nba_api.
    
    Args:
        player_id: NBA API player ID
        season: Season string (e.g., '2025-26') - ignored if fetch_career=True
        fetch_career: If True, fetch all seasons
        debug: If True, print debug info about available columns
    
    Returns:
        DataFrame with game log data
    """
    if fetch_career:
        print(f"\n📊 Fetching career game log for player ID {player_id}...")
        
        # Get all seasons
        seasons = get_player_career_years(player_id)
        
        if not seasons:
            print(f"   ❌ Could not determine career seasons")
            return None
        
        print(f"   Found {len(seasons)} seasons: {seasons[0]} to {seasons[-1]}")
        
        all_games = []
        
        for i, season in enumerate(seasons, 1):
            print(f"   [{i}/{len(seasons)}] Fetching {season}...", end=' ')
            
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                )
                df = gamelog.get_data_frames()[0]
                
                if not df.empty:
                    all_games.append(df)
                    print(f"✅ {len(df)} games")
                else:
                    print(f"⚠️  No games")
                
                # Rate limiting - be nice to NBA API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if not all_games:
            print(f"   ❌ No game data fetched")
            return None
        
        # Combine all seasons
        combined_df = pd.concat(all_games, ignore_index=True)
        print(f"\n   ✅ Total career games: {len(combined_df)}")
        
        # Debug: Show available columns
        if debug:
            print(f"\n   🔍 DEBUG - Available columns:")
            for col in combined_df.columns:
                print(f"      - {col}")
        
        # Add team and opponent columns
        if 'MATCHUP' in combined_df.columns:
            combined_df['TEAM'] = combined_df['MATCHUP'].apply(extract_team_from_matchup)
            combined_df['OPPONENT'] = combined_df['MATCHUP'].apply(extract_opponent_from_matchup)
            combined_df['IS_HOME'] = combined_df['MATCHUP'].str.contains('vs.', na=False)
            print(f"   ✅ Added TEAM, OPPONENT, IS_HOME columns")
        
        return combined_df
        
    else:
        print(f"\n📊 Fetching game log for player ID {player_id}, season {season}...")
        
        try:
            # Fetch game log
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            
            # Get DataFrame
            df = gamelog.get_data_frames()[0]
            
            print(f"   ✅ Fetched {len(df)} games")
            
            # Debug: Show available columns
            if debug:
                print(f"\n   🔍 DEBUG - Available columns:")
                for col in df.columns:
                    print(f"      - {col}")
            
            # Add team and opponent columns
            if 'MATCHUP' in df.columns:
                df['TEAM'] = df['MATCHUP'].apply(extract_team_from_matchup)
                df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent_from_matchup)
                df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', na=False)
                print(f"   ✅ Added TEAM, OPPONENT, IS_HOME columns")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching game log: {e}")
            return None


def display_game_log_summary(df):
    """
    Display summary statistics from game log.
    
    Args:
        df: Game log DataFrame
    """
    if df is None or df.empty:
        print("   No data to display")
        return
    
    print(f"\n📋 Game Log Summary:")
    print(f"   Games Played: {len(df)}")
    
    # Check if career data (multiple seasons)
    if 'SEASON_ID' in df.columns:
        unique_seasons = df['SEASON_ID'].nunique()
        if unique_seasons > 1:
            print(f"   Seasons: {unique_seasons}")
            
            # Season range
            seasons = sorted(df['SEASON_ID'].unique())
            first_season = seasons[0]
            last_season = seasons[-1]
            print(f"   Range: {first_season} to {last_season}")
    
    # Key stats
    if 'PTS' in df.columns:
        total_pts = df['PTS'].sum()
        print(f"   Points: {df['PTS'].mean():.1f} avg, {df['PTS'].max():.0f} high, {total_pts:,.0f} total")
    if 'REB' in df.columns:
        total_reb = df['REB'].sum()
        print(f"   Rebounds: {df['REB'].mean():.1f} avg, {df['REB'].max():.0f} high, {total_reb:,.0f} total")
    if 'AST' in df.columns:
        total_ast = df['AST'].sum()
        print(f"   Assists: {df['AST'].mean():.1f} avg, {df['AST'].max():.0f} high, {total_ast:,.0f} total")
    if 'BLK' in df.columns:
        print(f"   Blocks: {df['BLK'].mean():.1f} avg, {df['BLK'].max():.0f} high")
    if 'STL' in df.columns:
        print(f"   Steals: {df['STL'].mean():.1f} avg, {df['STL'].max():.0f} high")
    if 'MIN' in df.columns:
        print(f"   Minutes: {df['MIN'].mean():.1f} avg")
    
    # Win/Loss record
    if 'WL' in df.columns:
        wins = (df['WL'] == 'W').sum()
        losses = (df['WL'] == 'L').sum()
        win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0
        print(f"   Record: {wins}-{losses} ({win_pct:.1%})")
    
    # Recent games
    num_recent = min(5, len(df))
    print(f"\n📅 Last {num_recent} Games:")
    recent = df.head(num_recent)
    
    for _, game in recent.iterrows():
        game_date = game.get('GAME_DATE', 'N/A')
        matchup = game.get('MATCHUP', 'N/A')
        team = game.get('TEAM', '')
        opponent = game.get('OPPONENT', '')
        is_home = game.get('IS_HOME', False)
        pts = game.get('PTS', 0)
        reb = game.get('REB', 0)
        ast = game.get('AST', 0)
        wl = game.get('WL', '')
        
        # Show team if available
        if team:
            location = 'vs' if is_home else '@'
            matchup_display = f"{team} {location} {opponent}"
        else:
            matchup_display = matchup
        
        print(f"   {game_date} | {matchup_display:20} | {pts:2.0f} PTS, {reb:2.0f} REB, {ast:2.0f} AST | {wl}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch Anthony Davis game log from NBA API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch current season (2025-26)
  python tmp/fetch_anthony_davis_gamelog_nba_api.py
  
  # Fetch specific season
  python tmp/fetch_anthony_davis_gamelog_nba_api.py --season 2024-25
  
  # Fetch entire career
  python tmp/fetch_anthony_davis_gamelog_nba_api.py --season career
        """
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2025-26',
        help='Season to fetch (e.g., "2025-26" or "career" for all seasons)'
    )
    parser.add_argument(
        '--player',
        type=str,
        default='Anthony Davis',
        help='Player name to search for (default: Anthony Davis)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output to see available columns'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"🏀 FETCH {args.player.upper()} GAME LOG - NBA API")
    print("="*80)
    print()
    
    # Step 1: Find player
    player = find_player(args.player)
    
    if not player:
        print(f"\n❌ Could not find {args.player}")
        return
    
    player_id = player['id']
    
    # Step 2: Fetch game log (career or single season)
    fetch_career = (args.season.lower() == 'career')
    season_label = 'career' if fetch_career else args.season
    
    game_log_df = get_player_game_log(
        player_id, 
        season=args.season if not fetch_career else '2025-26',  # Default for career fetch
        fetch_career=fetch_career,
        debug=args.debug
    )
    
    if game_log_df is None or game_log_df.empty:
        print("\n❌ No game log data available")
        return
    
    # Step 3: Display summary
    display_game_log_summary(game_log_df)
    
    # Step 4: Display columns available
    print(f"\n📊 Available Columns ({len(game_log_df.columns)} total):")
    
    # Highlight important columns
    important_cols = ['GAME_DATE', 'TEAM', 'OPPONENT', 'IS_HOME', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'MIN']
    print(f"   Key columns:")
    for col in important_cols:
        if col in game_log_df.columns:
            print(f"      • {col}")
    
    if args.debug:
        print(f"\n   All columns:")
        for i, col in enumerate(game_log_df.columns, 1):
            print(f"      {i:2}. {col}")
    else:
        print(f"\n   (Use --debug to see all {len(game_log_df.columns)} columns)")
    
    # Step 5: Save to CSV
    output_dir = Path.home() / 'Downloads' / 'tmp'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize player name for filename
    player_name_safe = player['full_name'].lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d")
    
    output_file = output_dir / f'{player_name_safe}_gamelog_{season_label.replace("-", "")}_{timestamp}.csv'
    game_log_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved game log to: {output_file}")
    
    # Also save a preview (only if career)
    if fetch_career:
        preview_file = output_dir / f'{player_name_safe}_gamelog_{season_label}_preview.csv'
        game_log_df.head(20).to_csv(preview_file, index=False)
        print(f"💾 Saved preview (20 games) to: {preview_file}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
