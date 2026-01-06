"""
Find tonight's NBA player prop plays using the Role-Spread Points Model.

Strategy Overview:
This script implements the Role-Spread Points Model, which identifies betting
edges by categorizing players based on two factors:
    1. Player Role: Scoring tier based on their points prop line
    2. Game Spread: Team's competitive situation (favorite/underdog/pick'em)

The model uses historical data to find combinations where over/under hit rates
significantly deviate from baseline, creating positive expected ROI opportunities.

Context:
Takes tonight's games, bins each player by their points line tier and team spread,
then matches them against pre-defined strategies with proven positive ROI.

Strategies are hardcoded based on analyze_player_props_matrix.py analysis results.

Usage:
    python scripts/find_role_spread_points_model_plays.py
    python scripts/find_role_spread_points_model_plays.py --date 2026-01-06
    python scripts/find_role_spread_points_model_plays.py --granularity coarse
    python scripts/find_role_spread_points_model_plays.py --show-all
"""

import sys
from pathlib import Path

# Add project root to path
import os
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

import pandas as pd
import argparse
from datetime import datetime, date

# Import player name normalization and team mapping
sys.path.insert(0, str(root / 'src'))
from player_name_utils import normalize_player_name
from team_utils import TEAM_NAME_TO_ABBR, load_player_team_cache


# =============================================================================
# STRATEGIES (from analyze_player_props_matrix.py analysis)
# =============================================================================

# These strategies are based on historical analysis with 50+ game sample sizes
# Format: (line_tier, spread_bin, bet_side, edge, roi, games)

STRATEGIES_FINE = {
    # =========================================================================
    # OVER STRATEGIES (Top 10 from fine granularity analysis)
    # =========================================================================
    'high_star_small_dog_over': {
        'line_tier': '25-30 (High Star)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'OVER',
        'edge': 7.3,
        'roi': 7.0,
        'games': 107,
        'hit_rate': 56.1,
    },
    'star_pickem_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': 'Pick\'em (-2 to +2)',
        'bet_side': 'OVER',
        'edge': 7.1,
        'roi': 6.7,
        'games': 93,
        'hit_rate': 55.9,
    },
    'role_player_big_dog_over': {
        'line_tier': '10-15 (Role Player)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'OVER',
        'edge': 6.7,
        'roi': 5.9,
        'games': 173,
        'hit_rate': 55.5,
    },
    'star_big_fav_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '10-15 Fav',
        'bet_side': 'OVER',
        'edge': 6.1,
        'roi': 4.8,
        'games': 52,
        'hit_rate': 54.9,
    },
    'high_star_big_dog_over': {
        'line_tier': '25-30 (High Star)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'OVER',
        'edge': 5.9,
        'roi': 4.4,
        'games': 64,
        'hit_rate': 54.7,
    },
    'star_medium_fav_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '6-10 Fav',
        'bet_side': 'OVER',
        'edge': 5.8,
        'roi': 4.1,
        'games': 121,
        'hit_rate': 54.5,
    },
    'bench_small_dog_over': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'OVER',
        'edge': 4.7,
        'roi': 2.1,
        'games': 379,
        'hit_rate': 53.5,
    },
    'star_medium_dog_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '6-10 Dog',
        'bet_side': 'OVER',
        'edge': 4.6,
        'roi': 1.8,
        'games': 120,
        'hit_rate': 53.3,
    },
    'bench_medium_fav_over': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '6-10 Fav',
        'bet_side': 'OVER',
        'edge': 4.3,
        'roi': 1.3,
        'games': 328,
        'hit_rate': 53.0,
    },
    'bench_huge_dog_over': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '15+ Dog',
        'bet_side': 'OVER',
        'edge': 4.2,
        'roi': 1.1,
        'games': 51,
        'hit_rate': 52.9,
    },
    
    # =========================================================================
    # UNDER STRATEGIES (Top 10 from fine granularity analysis)
    # =========================================================================
    'bench_pickem_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': 'Pick\'em (-2 to +2)',
        'bet_side': 'UNDER',
        'edge': 7.3,
        'roi': 11.8,
        'games': 210,
        'hit_rate': 58.6,
    },
    'star_small_fav_under': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '2-6 Fav',
        'bet_side': 'UNDER',
        'edge': 5.4,
        'roi': 8.2,
        'games': 150,
        'hit_rate': 56.7,
    },
    'bench_huge_fav_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '15+ Fav',
        'bet_side': 'UNDER',
        'edge': 4.8,
        'roi': 6.9,
        'games': 50,
        'hit_rate': 56.0,
    },
    'star_small_dog_under': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'UNDER',
        'edge': 4.5,
        'roi': 6.3,
        'games': 149,
        'hit_rate': 55.7,
    },
    'bench_big_fav_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '10-15 Fav',
        'bet_side': 'UNDER',
        'edge': 4.2,
        'roi': 5.8,
        'games': 202,
        'hit_rate': 55.4,
    },
    'high_role_pickem_under': {
        'line_tier': '15-20 (High Role)',
        'spread_bin': 'Pick\'em (-2 to +2)',
        'bet_side': 'UNDER',
        'edge': 4.0,
        'roi': 5.4,
        'games': 154,
        'hit_rate': 55.2,
    },
    'high_role_big_dog_under': {
        'line_tier': '15-20 (High Role)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'UNDER',
        'edge': 3.4,
        'roi': 4.3,
        'games': 108,
        'hit_rate': 54.6,
    },
    'bench_big_dog_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'UNDER',
        'edge': 2.8,
        'roi': 3.2,
        'games': 164,
        'hit_rate': 54.0,
    },
    'high_role_small_dog_under': {
        'line_tier': '15-20 (High Role)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'UNDER',
        'edge': 2.6,
        'roi': 2.7,
        'games': 264,
        'hit_rate': 53.8,
    },
    'bench_small_fav_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '2-6 Fav',
        'bet_side': 'UNDER',
        'edge': 2.4,
        'roi': 2.4,
        'games': 402,
        'hit_rate': 53.6,
    },
}

STRATEGIES_COARSE = {
    # TODO: Add coarse strategies when we run coarse analysis
    # For now, just use fine as default
}


# =============================================================================
# BINNING FUNCTIONS (must match analyze_player_props_matrix.py)
# =============================================================================

def bin_points_line(line, granularity='fine'):
    """Bin player points line into tiers"""
    if pd.isna(line):
        return 'Unknown'
    
    if granularity == 'coarse':
        if line < 10:
            return '<10 (Bench)'
        elif line < 20:
            return '10-20 (Role)'
        elif line < 30:
            return '20-30 (Star)'
        else:
            return '30+ (Superstar)'
    else:  # fine
        if line < 5:
            return '<5 (Deep Bench)'
        elif line < 10:
            return '5-10 (Bench)'
        elif line < 15:
            return '10-15 (Role Player)'
        elif line < 20:
            return '15-20 (High Role)'
        elif line < 25:
            return '20-25 (Star)'
        elif line < 30:
            return '25-30 (High Star)'
        else:
            return '30+ (Superstar)'


def bin_team_spread(spread, granularity='fine'):
    """
    Bin team spread into categories
    
    Args:
        spread: Team spread (positive = underdog, negative = favorite)
    """
    if pd.isna(spread):
        return 'Unknown'
    
    if granularity == 'coarse':
        if spread < -5:
            return 'Favorite'
        elif spread <= 5:
            return 'Pick\'em'
        else:
            return 'Underdog'
    else:  # fine
        if spread < -15:
            return '15+ Fav'
        elif spread < -10:
            return '10-15 Fav'
        elif spread < -6:
            return '6-10 Fav'
        elif spread < -2:
            return '2-6 Fav'
        elif spread <= 2:
            return 'Pick\'em (-2 to +2)'
        elif spread <= 6:
            return '2-6 Dog'
        elif spread <= 10:
            return '6-10 Dog'
        elif spread <= 15:
            return '10-15 Dog'
        else:
            return '15+ Dog'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_tonights_games(target_date=None):
    """
    Load tonight's games with player props and team spreads
    
    Uses The Odds API to fetch:
    1. Today's NBA games
    2. Player points props for each game  
    3. Game spreads for each team
    
    Also loads player-team mapping from historical game data
    
    Returns:
        DataFrame with columns: PLAYER_NAME, points_line, team_abbr, team_spread, opponent
    """
    import requests
    import ssl
    import urllib3
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    # Disable SSL warnings (common with macOS)
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Load API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ No valid API key found!")
        print("Get your API key at: https://the-odds-api.com/")
        print("Add it to .env file as: ODDS_API_KEY=your_key")
        # print("\n⚠️  Using mock data for now...\n")
        
        # Return mock data
        return -99
    
    # Load player-team mapping from cache using utility function
    print("📋 Loading player-team mapping from cache...")
    try:
        cache_data = load_player_team_cache()
        player_team_map = cache_data['mapping']
        cache_timestamp = cache_data['timestamp']
        
        if player_team_map:
            print(f"   ✅ Loaded {len(player_team_map):,} player-team mappings from cache")
            if cache_timestamp:
                print(f"   📅 Cache timestamp: {cache_timestamp}\n")
            else:
                print()
        else:
            print(f"   ⚠️  No player-team mapping available in cache")
            print(f"   Will skip players without known teams\n")
    except Exception as e:
        print(f"   ⚠️  Error loading player-team mapping: {e}")
        print(f"   Will skip players without known teams\n")
        player_team_map = {}
    
    # API Configuration
    API_BASE_URL = 'https://api.the-odds-api.com/v4'
    SPORT = 'basketball_nba'
    TIMEZONE = 'America/New_York'
    
    print(f"📊 Fetching today's NBA games and player props...")
    print(f"📅 {datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')}\n")
    
    try:
        # Step 1: Get today's NBA events
        url = f"{API_BASE_URL}/sports/{SPORT}/events"
        params = {'apiKey': api_key}
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        events = response.json()
        
        # Filter for today's games
        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)
        today = now.date()
        
        todays_events = []
        for event in events:
            event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            event_time_local = event_time_utc.astimezone(tz)
            
            if event_time_local.date() == today:
                todays_events.append(event)
        
        if not todays_events:
            print("❌ No NBA games found for today")
            print("⚠️  Using mock data...\n")
            return pd.DataFrame({
                'PLAYER_NAME': ['Jalen Brunson', 'Karl-Anthony Towns'],
                'points_line': [28.5, 23.5],
                'team_abbr': ['NYK', 'NYK'],
                'team_spread': [3.0, 3.0],
                'opponent': ['BOS', 'BOS'],
            })
        
        print(f"✅ Found {len(todays_events)} games today\n")
        
        # Step 2: Fetch player props AND spreads for each game
        all_player_data = []
        unmapped_players = []  # Track players without team mapping
        game_info = []  # Store game metadata (tip time, teams, spreads)
        
        for i, event in enumerate(todays_events, 1):
            event_id = event['id']
            away_team = event['away_team']
            home_team = event['home_team']
            event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            event_time_local = event_time_utc.astimezone(tz)
            
            print(f"📥 Game {i}/{len(todays_events)}: {away_team} @ {home_team}")
            
            # Get player points props
            url = f"{API_BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'player_points,spreads',  # Get both player props and spreads
                'oddsFormat': 'american',
            }
            
            try:
                response = requests.get(url, params=params, verify=False)
                response.raise_for_status()
                odds_data = response.json()
                
                # Extract spreads first (to map teams to spreads)
                team_spreads = {}
                for bookmaker in odds_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'spreads':
                            for outcome in market.get('outcomes', []):
                                team_name = outcome['name']
                                spread = outcome.get('point', 0)
                                # Use first spread found for each team
                                if team_name not in team_spreads:
                                    team_spreads[team_name] = spread
                
                # Store game info for later use
                away_abbr_temp = TEAM_NAME_TO_ABBR.get(away_team, away_team)
                home_abbr_temp = TEAM_NAME_TO_ABBR.get(home_team, home_team)
                game_info.append({
                    'away_team': away_abbr_temp,
                    'home_team': home_abbr_temp,
                    'away_spread': team_spreads.get(away_team, 0),
                    'home_spread': team_spreads.get(home_team, 0),
                    'game_time': event_time_local,
                })
                
                # Extract player props
                for bookmaker in odds_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'player_points':
                            # Group by (player, line) to deduplicate
                            player_lines_seen = set()
                            
                            for outcome in market.get('outcomes', []):
                                player = outcome.get('description', 'Unknown')
                                line = outcome.get('point')
                                
                                key = (player, line)
                                if key in player_lines_seen:
                                    continue
                                player_lines_seen.add(key)
                                
                                # Normalize player name to match cache
                                player_normalized = normalize_player_name(player)
                                
                                # Determine player's team using cache mapping
                                player_team_abbr = player_team_map.get(player_normalized)
                                
                                # Convert API's full team names to abbreviations for comparison
                                away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team)
                                home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team)
                                
                                if player_team_abbr and player_team_abbr in [away_abbr, home_abbr]:
                                    # We know the team from cache
                                    opponent_abbr = home_abbr if player_team_abbr == away_abbr else away_abbr
                                    
                                    # Get spread for player's team (API returns spreads with full names)
                                    player_team_full = away_team if player_team_abbr == away_abbr else home_team
                                    spread = team_spreads.get(player_team_full, 0)
                                    
                                    # Get opponent full name
                                    opponent_full = home_team if player_team_abbr == away_abbr else away_team
                                    
                                    all_player_data.append({
                                        'PLAYER_NAME': player,
                                        'points_line': line,
                                        'team_abbr': player_team_abbr,
                                        'team_spread': spread,
                                        'opponent': opponent_abbr,
                                        'game_time': event_time_local,
                                    })
                                else:
                                    # Track unmapped player (cache might be outdated or player recently traded)
                                    unmapped_players.append({
                                        'player': player,
                                        'normalized': player_normalized,
                                        'game': f"{away_team} @ {home_team}",
                                        'in_cache': player_normalized in player_team_map,
                                        'cached_team': player_team_map.get(player_normalized, 'N/A'),
                                    })
                
                print(f"   ✅ Found props and spreads")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if not all_player_data:
            print("\n❌ No player props found with team mapping")
            print("⚠️  Using mock data...\n")
            return pd.DataFrame({
                'PLAYER_NAME': ['Jalen Brunson', 'Karl-Anthony Towns'],
                'points_line': [28.5, 23.5],
                'team_abbr': ['NYK', 'NYK'],
                'team_spread': [3.0, 3.0],
                'opponent': ['BOS', 'BOS'],
            })
        
        df = pd.DataFrame(all_player_data)
        
        # Remove duplicates (same player might appear in multiple bookmaker markets)
        df = df.drop_duplicates(subset=['PLAYER_NAME', 'points_line', 'team_abbr'])
        
        total_props_before_consensus = len(df)
        total_players_before_consensus = df['PLAYER_NAME'].nunique()
        print(f"   Raw props before consensus: {total_props_before_consensus} props for {total_players_before_consensus} players")
        
        # Get consensus line for each player (median line)
        # This removes the issue of multiple lines per player (21.5, 22.5, 23.5, etc.)
        df_consensus = df.groupby('PLAYER_NAME', as_index=False).agg({
            'points_line': 'median',  # Use median line as consensus
            'team_abbr': 'first',
            'team_spread': 'first',
            'opponent': 'first',
            'game_time': 'first',
        })
        
        players_mapped = df_consensus['PLAYER_NAME'].nunique()
        total_props = len(df_consensus)
        total_unmapped = len(set(u['normalized'] for u in unmapped_players))
        
        print(f"\n✅ Loaded {total_props} player props from {len(todays_events)} games (consensus lines)")
        print(f"   Successfully mapped: {players_mapped} players")
        if total_unmapped > 0:
            print(f"   Unmapped (skipped): {total_unmapped} players")
        
        # Log unmapped players if any
        if unmapped_players:
            # Deduplicate unmapped players
            unmapped_df = pd.DataFrame(unmapped_players)
            unmapped_df = unmapped_df.drop_duplicates(subset=['normalized'])
            
            print(f"\n⚠️  {len(unmapped_df)} players without team mapping:")
            for _, row in unmapped_df.iterrows():
                if row['in_cache']:
                    # Player is in cache but team doesn't match this game (likely traded/wrong mapping)
                    print(f"   - {row['player']} (cached as {row['cached_team']}, not in {row['game']})")
                else:
                    # Player not in cache at all
                    print(f"   - {row['player']} (not in cache, game: {row['game']})")
        
        print()
        
        return df_consensus, game_info
        
    except Exception as e:
        print(f"\n❌ Error loading games: {e}")
        print("⚠️  Using mock data...\n")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({
            'PLAYER_NAME': ['Jalen Brunson', 'Karl-Anthony Towns'],
            'points_line': [28.5, 23.5],
            'team_abbr': ['NYK', 'NYK'],
            'team_spread': [3.0, 3.0],
            'opponent': ['BOS', 'BOS'],
        })


# =============================================================================
# PLAY FINDING
# =============================================================================

def find_plays(df_games, granularity='fine'):
    """
    Find betting plays by matching games to strategies
    
    Returns:
        DataFrame with plays and reasoning
    """
    strategies = STRATEGIES_FINE if granularity == 'fine' else STRATEGIES_COARSE
    
    if not strategies:
        print(f"No strategies defined for granularity: {granularity}")
        return pd.DataFrame()
    
    # Bin each player/team
    df_games['line_tier'] = df_games['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df_games['spread_bin'] = df_games['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    plays = []
    
    for idx, row in df_games.iterrows():
        player = row['PLAYER_NAME']
        line = row['points_line']
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        team = row['team_abbr']
        opp = row['opponent']
        spread = row['team_spread']
        
        # Check if this combination matches any strategy
        for strat_name, strat in strategies.items():
            if strat['line_tier'] == line_tier and strat['spread_bin'] == spread_bin:
                # Generate strategy display name from bins + bet side
                strategy_display_name = f"{strat['line_tier']} + {strat['spread_bin']} {strat['bet_side']}"
                
                plays.append({
                    'player': player,
                    'line': line,
                    'bet_side': strat['bet_side'],
                    'team': team,
                    'opponent': opp,
                    'spread': spread,
                    'line_tier': line_tier,
                    'spread_bin': spread_bin,
                    'strategy_key': strat_name,
                    'strategy_name': strategy_display_name,
                    'strategy_roi': strat['roi'],
                    'strategy_edge': strat['edge'],
                    'strategy_hit_rate': strat['hit_rate'],
                    'strategy_games': strat['games'],
                    'reason': f"{strat['bet_side']} - {line_tier} in {spread_bin} games ({strat['edge']:+.1f}% edge, {strat['roi']:+.1f}% ROI, {strat['games']} games)"
                })
    
    return pd.DataFrame(plays)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_plays(df_plays, all_games_info):
    """Pretty print plays grouped by game, showing ALL games"""
    
    print(f"\n{'='*80}")
    print(f"🎯 TONIGHT'S PLAYS ({date.today()})")
    print(f"{'='*80}\n")
    
    # Sort games by tip time
    all_games_info = sorted(all_games_info, key=lambda g: g['game_time'])
    
    if len(df_plays) == 0:
        print("❌ No plays found matching strategies\n")
        # Still show all games even if no plays
        for game_num, game in enumerate(all_games_info, 1):
            tip_time = game['game_time'].strftime('%I:%M %p ET')
            print(f"{'='*80}")
            print(f"🏀 GAME {game_num}/{len(all_games_info)}: {game['away_team']} @ {game['home_team']}")
            print(f"⏰ Tip: {tip_time}")
            print(f"{'='*80}\n")
            print(f"📍 {game['away_team']}: No plays")
            print(f"🏠 {game['home_team']}: No plays\n")
        return
    
    # Add game_teams column for grouping
    def get_game_teams(row):
        teams = sorted([row['team'], row['opponent']])
        return tuple(teams)
    
    df_plays['game_teams'] = df_plays.apply(get_game_teams, axis=1)
    
    # Create a map of plays by game
    plays_by_game = {}
    for game_teams, group in df_plays.groupby('game_teams'):
        plays_by_game[game_teams] = group
    
    # Show all games (even ones without plays), sorted by tip time
    for game_num, game in enumerate(all_games_info, 1):
        away_team = game['away_team']
        home_team = game['home_team']
        tip_time = game['game_time'].strftime('%I:%M %p ET')
        game_teams = tuple(sorted([away_team, home_team]))
        
        print(f"{'='*80}")
        print(f"🏀 GAME {game_num}/{len(all_games_info)}: {away_team} @ {home_team}")
        print(f"⏰ Tip: {tip_time}")
        print(f"{'='*80}\n")
        
        # Check if this game has any plays
        if game_teams not in plays_by_game:
            print(f"📍 {away_team}: No plays")
            print(f"🏠 {home_team}: No plays\n")
            continue
        
        game_plays = plays_by_game[game_teams]
        
        # Group plays by team
        away_plays = game_plays[game_plays['team'] == away_team]
        home_plays = game_plays[game_plays['team'] == home_team]
        
        # Show away team plays
        if len(away_plays) > 0:
            print(f"📍 {away_team} plays ({len(away_plays)}):")
            print(f"{'─'*80}")
            
            # Sort by ROI descending within team
            away_plays = away_plays.sort_values('strategy_roi', ascending=False)
            
            for idx, play in away_plays.iterrows():
                breakeven_rate = 52.38
                edge_vs_breakeven = play['strategy_hit_rate'] - breakeven_rate
                
                print(f"🔥 {play['bet_side']}: {play['player']} {play['line']} pts")
                print(f"   Strategy: {play['strategy_name']}")
                print(f"   ├─ Historical: {play['strategy_hit_rate']:.1f}% hit rate ({play['strategy_games']} games)")
                print(f"   ├─ Edge vs Baseline: {play['strategy_edge']:+.1f}% | Edge vs Breakeven: {edge_vs_breakeven:+.1f}%")
                print(f"   └─ Expected ROI: {play['strategy_roi']:+.1f}%")
                print()
        else:
            print(f"📍 {away_team}: No plays\n")
        
        # Show home team plays
        if len(home_plays) > 0:
            print(f"🏠 {home_team} plays ({len(home_plays)}):")
            print(f"{'─'*80}")
            
            # Sort by ROI descending within team
            home_plays = home_plays.sort_values('strategy_roi', ascending=False)
            
            for idx, play in home_plays.iterrows():
                breakeven_rate = 52.38
                edge_vs_breakeven = play['strategy_hit_rate'] - breakeven_rate
                
                print(f"🔥 {play['bet_side']}: {play['player']} {play['line']} pts")
                print(f"   Strategy: {play['strategy_name']}")
                print(f"   ├─ Historical: {play['strategy_hit_rate']:.1f}% hit rate ({play['strategy_games']} games)")
                print(f"   ├─ Edge vs Baseline: {play['strategy_edge']:+.1f}% | Edge vs Breakeven: {edge_vs_breakeven:+.1f}%")
                print(f"   └─ Expected ROI: {play['strategy_roi']:+.1f}%")
                print()
        else:
            print(f"🏠 {home_team}: No plays\n")
        
        print()
    
    print(f"{'='*80}")
    print(f"Total plays: {len(df_plays)}")
    print(f"Avg ROI: {df_plays['strategy_roi'].mean():.1f}%")
    print(f"{'='*80}\n")


def print_no_plays_reasoning(df_games, granularity='fine'):
    """Show why we don't have plays for certain games"""
    
    strategies = STRATEGIES_FINE if granularity == 'fine' else STRATEGIES_COARSE
    
    df_games['line_tier'] = df_games['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df_games['spread_bin'] = df_games['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    print(f"\n{'='*80}")
    print("🔍 ALL PLAYERS & STRATEGY MATCHES")
    print(f"{'='*80}\n")
    
    for idx, row in df_games.iterrows():
        player = row['PLAYER_NAME']
        line = row['points_line']
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        
        # Check for matches
        matches = []
        for strat_name, strat in strategies.items():
            if strat['line_tier'] == line_tier and strat['spread_bin'] == spread_bin:
                matches.append(strat)
        
        if matches:
            print(f"✅ {player} {line} pts")
            print(f"   Category: {line_tier} + {spread_bin}")
            for strat in matches:
                print(f"   → {strat['bet_side']}: {strat['edge']:+.1f}% edge, {strat['roi']:+.1f}% ROI")
        else:
            print(f"❌ {player} {line} pts")
            print(f"   Category: {line_tier} + {spread_bin}")
            print(f"   → No strategy for this combination")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Find tonight\'s NBA player prop plays')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--granularity', choices=['coarse', 'fine'], default='fine',
                       help='Binning granularity (fine = 7x9 grid, coarse = 4x3 grid)')
    parser.add_argument('--min-roi', type=float, default=5.0,
                       help='Minimum ROI threshold (default: 5.0%%)')
    parser.add_argument('--show-all', action='store_true',
                       help='Show reasoning for all players, not just plays')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\n📊 Loading games for {args.date or 'today'}...")
    result = load_tonights_games(args.date)
    
    # Unpack result
    if isinstance(result, tuple):
        df_games, all_games_info = result
    else:
        # Fallback for mock data (doesn't return game info)
        df_games = result
        all_games_info = []
    
    print(f"   Found {len(df_games)} players with props\n")
    
    # Find plays
    df_plays = find_plays(df_games, args.granularity)
    
    # Filter by minimum ROI
    plays_before_filter = len(df_plays)
    df_plays = df_plays[df_plays['strategy_roi'] >= args.min_roi]
    
    if plays_before_filter > len(df_plays):
        filtered_count = plays_before_filter - len(df_plays)
        print(f"🔍 Filtered to ROI >= {args.min_roi}% ({filtered_count} plays below threshold)\n")
    
    # Output
    if args.show_all:
        print_no_plays_reasoning(df_games, args.granularity)
    
    print_plays(df_plays, all_games_info)


if __name__ == '__main__':
    main()

