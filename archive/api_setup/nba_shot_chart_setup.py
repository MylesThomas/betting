"""
NBA Shot Chart Data Setup

This script fetches detailed shot data from the NBA Stats API, including:
- Shot distance (feet)
- Shot location (x, y coordinates)
- Shot result (made/missed)
- Shot type (Dunk, Layup, Jump Shot, etc.)
- Game context (date, opponent, score)

User Request:
"I wanna do analysis on shots within 6 feet in the nba can we get data by shot distance?"

Use Cases:
- Analyze close-range shooting (0-6 feet)
- Study rim efficiency
- Compare player finishing ability
- Track shooting trends over time

API Endpoint:
- shotchartdetail (from nba_api.stats.endpoints)
- Returns every shot a player/team has taken with detailed metadata

No API key required - uses official NBA stats endpoints
"""

import pandas as pd
from datetime import datetime
import time
import ssl
import urllib3
import requests

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players, teams


def get_player_shot_chart(player_name, season="2024-25", season_type="Regular Season"):
    """
    Get all shots for a player in a season
    
    Args:
        player_name: Full player name (e.g., "LeBron James")
        season: NBA season format (e.g., "2024-25")
        season_type: "Regular Season" or "Playoffs"
    
    Returns:
        DataFrame with every shot including:
        - SHOT_DISTANCE (feet)
        - LOC_X, LOC_Y (court coordinates)
        - SHOT_MADE_FLAG (1=made, 0=missed)
        - SHOT_TYPE (2PT Field Goal, 3PT Field Goal)
        - ACTION_TYPE (Dunk, Layup, Jump Shot, etc.)
        - GAME_DATE
        - HTM, VTM (home/visiting team)
        - And many more fields...
    """
    print(f"\n🏀 Fetching shot chart for {player_name} ({season})...")
    
    # Get player ID
    all_players = players.find_players_by_full_name(player_name)
    if not all_players:
        print(f"❌ Player '{player_name}' not found")
        return None
    
    player_id = all_players[0]['id']
    print(f"   Player ID: {player_id}")
    
    # Fetch shot chart data
    try:
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=0,  # 0 = all teams
            player_id=player_id,
            season_nullable=season,
            season_type_all_star=season_type,
            context_measure_simple='FGA'  # Field Goals Attempted
        )
        
        shots_df = shot_chart.get_data_frames()[0]
        
        print(f"✅ Found {len(shots_df)} shots")
        print(f"\n📊 Shot Distance Stats:")
        print(shots_df['SHOT_DISTANCE'].describe())
        
        return shots_df
        
    except Exception as e:
        print(f"❌ Error fetching shot chart: {e}")
        return None


def get_team_shot_chart(team_abbreviation, season="2024-25", season_type="Regular Season"):
    """
    Get all shots for a team in a season
    
    Args:
        team_abbreviation: Team abbreviation (e.g., "LAL", "GSW")
        season: NBA season format (e.g., "2024-25")
        season_type: "Regular Season" or "Playoffs"
    
    Returns:
        DataFrame with every shot for the entire team
    """
    print(f"\n🏀 Fetching shot chart for {team_abbreviation} ({season})...")
    
    # Get team ID
    all_teams = teams.find_teams_by_abbreviation(team_abbreviation)
    if not all_teams:
        print(f"❌ Team '{team_abbreviation}' not found")
        return None
    
    team_id = all_teams[0]['id']
    print(f"   Team ID: {team_id}")
    
    # Fetch shot chart data
    try:
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=0,  # 0 = all players
            season_nullable=season,
            season_type_all_star=season_type,
            context_measure_simple='FGA'
        )
        
        shots_df = shot_chart.get_data_frames()[0]
        
        print(f"✅ Found {len(shots_df)} shots")
        print(f"\n📊 Shot Distance Stats:")
        print(shots_df['SHOT_DISTANCE'].describe())
        
        return shots_df
        
    except Exception as e:
        print(f"❌ Error fetching shot chart: {e}")
        return None


def analyze_close_range_shots(shots_df, max_distance=6):
    """
    Analyze shots within specified distance (default: 6 feet)
    
    Args:
        shots_df: DataFrame from get_player_shot_chart or get_team_shot_chart
        max_distance: Maximum distance in feet (default 6)
    
    Returns:
        Dict with analysis results
    """
    if shots_df is None or shots_df.empty:
        print("❌ No shot data to analyze")
        return None
    
    print(f"\n🎯 Analyzing shots within {max_distance} feet...")
    
    # Filter to close-range shots
    close_shots = shots_df[shots_df['SHOT_DISTANCE'] <= max_distance].copy()
    
    total_shots = len(shots_df)
    close_shot_count = len(close_shots)
    close_shot_pct = (close_shot_count / total_shots * 100) if total_shots > 0 else 0
    
    # Calculate shooting percentage
    close_makes = close_shots['SHOT_MADE_FLAG'].sum()
    close_fg_pct = (close_makes / close_shot_count * 100) if close_shot_count > 0 else 0
    
    # Shot type breakdown
    shot_type_breakdown = close_shots['ACTION_TYPE'].value_counts()
    
    # Results
    results = {
        'total_shots_season': total_shots,
        'close_range_attempts': close_shot_count,
        'close_range_pct_of_total': close_shot_pct,
        'close_range_makes': close_makes,
        'close_range_fg_pct': close_fg_pct,
        'shot_types': shot_type_breakdown.to_dict()
    }
    
    print(f"\n📈 Results:")
    print(f"   Total shots this season: {total_shots:,}")
    print(f"   Shots within {max_distance} feet: {close_shot_count:,} ({close_shot_pct:.1f}%)")
    print(f"   FG% within {max_distance} feet: {close_fg_pct:.1f}%")
    print(f"\n🏀 Shot Type Breakdown (within {max_distance} feet):")
    for shot_type, count in shot_type_breakdown.head(10).items():
        pct = (count / close_shot_count * 100)
        print(f"   {shot_type:<30}: {count:>4} ({pct:>5.1f}%)")
    
    return results


def compare_players_close_range(player_names, season="2024-25", max_distance=6):
    """
    Compare multiple players' close-range shooting
    
    Args:
        player_names: List of player names
        season: NBA season
        max_distance: Max distance for "close range"
    
    Returns:
        DataFrame comparing players
    """
    print(f"\n{'='*70}")
    print(f"COMPARING CLOSE-RANGE SHOOTING ({max_distance} feet)")
    print(f"{'='*70}")
    
    results = []
    
    for player_name in player_names:
        shots_df = get_player_shot_chart(player_name, season)
        
        if shots_df is not None and not shots_df.empty:
            analysis = analyze_close_range_shots(shots_df, max_distance)
            
            if analysis:
                results.append({
                    'player': player_name,
                    'total_shots': analysis['total_shots_season'],
                    'close_attempts': analysis['close_range_attempts'],
                    'close_pct_of_total': analysis['close_range_pct_of_total'],
                    'close_fg_pct': analysis['close_range_fg_pct']
                })
        
        # Rate limiting
        time.sleep(0.6)
    
    if not results:
        print("❌ No data found for any players")
        return None
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('close_fg_pct', ascending=False)
    
    print(f"\n📊 Close-Range Shooting Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def save_shot_data(shots_df, filename, output_dir="../data/01_input/nba_api"):
    """
    Save shot chart data to CSV
    
    Args:
        shots_df: Shot chart DataFrame
        filename: Output filename (without .csv extension)
        output_dir: Output directory
    """
    if shots_df is None or shots_df.empty:
        print("❌ No data to save")
        return
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{filename}.csv")
    shots_df.to_csv(filepath, index=False)
    print(f"💾 Saved {len(shots_df)} shots to: {filepath}")


def demo_shot_analysis():
    """
    Demo script showing how to use shot chart analysis
    """
    print("="*70)
    print("NBA SHOT CHART ANALYSIS - DEMO")
    print("="*70)
    
    # Example 1: Single player analysis
    print("\n" + "="*70)
    print("EXAMPLE 1: Analyzing LeBron James close-range shooting")
    print("="*70)
    
    lebron_shots = get_player_shot_chart("LeBron James", "2024-25")
    
    if lebron_shots is not None:
        # Analyze shots within 6 feet
        lebron_analysis = analyze_close_range_shots(lebron_shots, max_distance=6)
        
        # Show sample of close-range shots
        close_shots = lebron_shots[lebron_shots['SHOT_DISTANCE'] <= 6]
        print("\n📋 Sample Close-Range Shots:")
        print(close_shots[['GAME_DATE', 'SHOT_DISTANCE', 'ACTION_TYPE', 
                           'SHOT_MADE_FLAG', 'HTM', 'VTM']].head(10))
        
        # Save to file
        save_shot_data(lebron_shots, "lebron_james_2024_25_shot_chart")
    
    time.sleep(0.6)
    
    # Example 2: Compare multiple players
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing multiple players")
    print("="*70)
    
    players_to_compare = [
        "Giannis Antetokounmpo",
        "Joel Embiid",
        "Nikola Jokic"
    ]
    
    comparison = compare_players_close_range(players_to_compare, "2024-25", max_distance=6)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    
    return lebron_shots, comparison


if __name__ == "__main__":
    # Run the demo
    shots_df, comparison_df = demo_shot_analysis()
    
    print("\n" + "="*70)
    print("AVAILABLE SHOT DATA COLUMNS:")
    print("="*70)
    if shots_df is not None:
        print("\nKey columns for analysis:")
        key_cols = [
            'SHOT_DISTANCE',
            'LOC_X', 
            'LOC_Y',
            'SHOT_MADE_FLAG',
            'SHOT_TYPE',
            'ACTION_TYPE',
            'GAME_DATE',
            'PERIOD',
            'MINUTES_REMAINING',
            'SECONDS_REMAINING',
            'HTM',
            'VTM',
            'SHOT_ZONE_BASIC',
            'SHOT_ZONE_AREA',
            'SHOT_ZONE_RANGE'
        ]
        
        for col in key_cols:
            if col in shots_df.columns:
                print(f"  ✓ {col}")
        
        print(f"\n📋 All available columns ({len(shots_df.columns)} total):")
        print(list(shots_df.columns))

