"""
NBA API Setup and Testing Script

This script tests the nba_api package and demonstrates how to fetch:
- Player game logs (for trend analysis)
- Current season stats
- Historical data

No API key required - uses official NBA stats endpoints
"""

import pandas as pd
from datetime import datetime
import time
import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

from nba_api.stats.endpoints import playergamelog, commonplayerinfo, leaguedashplayerstats
from nba_api.stats.static import players, teams


def get_all_active_players():
    """Get list of all active NBA players"""
    print("Fetching all active NBA players...")
    all_players = players.get_active_players()
    df = pd.DataFrame(all_players)
    print(f"Found {len(df)} active players")
    return df


def get_player_id(player_name):
    """Find player ID by name"""
    all_players = players.find_players_by_full_name(player_name)
    if not all_players:
        print(f"Player '{player_name}' not found")
        return None
    return all_players[0]['id']


def get_player_game_log(player_name, season="2024-25"):
    """
    Get game-by-game stats for a player
    
    Args:
        player_name: Full player name (e.g., "Draymond Green")
        season: NBA season format (e.g., "2024-25")
    
    Returns:
        DataFrame with game log including all stats
    """
    print(f"\nFetching game log for {player_name} ({season})...")
    
    player_id = get_player_id(player_name)
    if not player_id:
        return None
    
    # Fetch game log
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star='Regular Season'
    )
    
    df = gamelog.get_data_frames()[0]
    
    # Display sample
    print(f"Total games: {len(df)}")
    print("\nRecent games:")
    print(df[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'FG3M', 'MIN']].head(10))
    
    return df


def analyze_trend(df, stat_column, threshold, direction='under'):
    """
    Analyze consecutive games over/under a threshold
    
    Args:
        df: Game log DataFrame
        stat_column: Column to analyze (e.g., 'FG3M' for 3-pointers made)
        threshold: The prop line threshold
        direction: 'under' or 'over'
    
    Returns:
        Trend analysis including current streak
    """
    print(f"\n=== Analyzing {stat_column} {direction.upper()} {threshold} trend ===")
    
    if direction == 'under':
        df['hit'] = df[stat_column] < threshold
    else:
        df['hit'] = df[stat_column] > threshold
    
    # Calculate current streak
    current_streak = 0
    for hit in df['hit'].values:
        if hit:
            current_streak += 1
        else:
            break
    
    # Calculate overall hit rate
    hit_rate = df['hit'].mean() * 100
    
    print(f"Current streak: {current_streak} games")
    print(f"Overall hit rate: {hit_rate:.1f}%")
    print(f"\nLast 10 games {stat_column} values:")
    print(df[['GAME_DATE', 'MATCHUP', stat_column]].head(10))
    
    return {
        'current_streak': current_streak,
        'hit_rate': hit_rate,
        'stat_column': stat_column,
        'threshold': threshold,
        'direction': direction
    }


def find_long_bad_trends(players_to_check, stat_column='FG3M', threshold=0.5, min_streak=5):
    """
    Scan multiple players for long negative trends (potential regression opportunities)
    
    Args:
        players_to_check: List of player names
        stat_column: Stat to analyze
        threshold: The prop line
        min_streak: Minimum streak length to flag
    """
    print(f"\n{'='*60}")
    print(f"SCANNING FOR LONG TRENDS ({stat_column} < {threshold})")
    print(f"{'='*60}")
    
    opportunities = []
    
    for player_name in players_to_check:
        try:
            player_id = get_player_id(player_name)
            if not player_id:
                continue
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season="2024-25",
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            
            # Check current streak
            current_streak = 0
            for value in df[stat_column].values:
                if value < threshold:
                    current_streak += 1
                else:
                    break
            
            if current_streak >= min_streak:
                avg_stat = df[stat_column].head(current_streak).mean()
                opportunities.append({
                    'player': player_name,
                    'streak': current_streak,
                    'avg_in_streak': avg_stat,
                    'threshold': threshold
                })
                print(f"🚨 {player_name}: {current_streak} game streak (avg: {avg_stat:.2f})")
            
            # Rate limit (NBA API is sensitive to rapid requests)
            time.sleep(0.6)
            
        except Exception as e:
            print(f"Error fetching {player_name}: {e}")
    
    return opportunities


def test_nba_api():
    """Run comprehensive test of NBA API functionality"""
    print("="*60)
    print("NBA API SETUP TEST")
    print("="*60)
    
    # Test 1: Get active players
    try:
        active_players = get_all_active_players()
        print("✅ Successfully fetched active players\n")
    except Exception as e:
        print(f"❌ Error fetching players: {e}\n")
        return False
    
    # Test 2: Get specific player game log (Draymond Green example)
    try:
        draymond_log = get_player_game_log("Draymond Green", "2024-25")
        print("✅ Successfully fetched player game log\n")
        
        # Test 3: Analyze 3PT trend
        if draymond_log is not None and not draymond_log.empty:
            trend = analyze_trend(draymond_log, 'FG3M', 0.5, 'under')
            print("✅ Successfully analyzed trend\n")
        
    except Exception as e:
        print(f"❌ Error fetching game log: {e}\n")
        return False
    
    print("="*60)
    print("✅ NBA API SETUP COMPLETE!")
    print("="*60)
    return True


if __name__ == "__main__":
    # Run the test
    success = test_nba_api()
    
    if success:
        print("\n" + "="*60)
        print("BONUS: Scanning for trend opportunities...")
        print("="*60)
        
        # Example: Check multiple players for 3PT unders trend
        players_to_scan = [
            "Draymond Green",
            "Rudy Gobert", 
            "Clint Capela",
            "Steven Adams",
            "Domantas Sabonis"
        ]
        
        opportunities = find_long_bad_trends(
            players_to_scan, 
            stat_column='FG3M', 
            threshold=0.5,
            min_streak=5
        )
        
        if opportunities:
            print(f"\n🎯 Found {len(opportunities)} potential regression opportunities!")
        else:
            print("\n📊 No significant trends found in this sample")

