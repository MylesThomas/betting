"""
Test NBA API for player defensive performance stats (2025-26 season).

Purpose:
- Explore different NBA API endpoints for defensive metrics
- Test fetching defensive stats: DEF_RATING, STL, BLK, opponent FG%, etc.
- Validate data availability and structure for 2025-26 season

Context:
The NBA API provides multiple endpoints for defensive data:
1. LeagueDashPlayerStats (with measure_type="Defense")
2. PlayerDefenseDashboard (opponent shooting when guarded)
3. PlayerDashPtDefend (tracking data - defended shot quality)

This script tests all three to see what data is available and useful for betting.

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 testing/test_nba_defensive_stats.py
"""

import pandas as pd
import ssl
import urllib3
import requests
import time
from pprint import pprint

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Import after SSL fixes
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    playerdashptshotdefend,
    leaguedashptdefend,
    defensehub
)
from nba_api.stats.static import players

# Test players with known defensive skills
TEST_PLAYERS = [
    "Rudy Gobert",      # Elite rim protector
    "Draymond Green",   # Versatile defender
    "Jrue Holiday",     # Perimeter defender
    "Bam Adebayo",      # Switchable big
    "Anthony Davis",    # Two-way star
]

SEASON = "2025-26"


def get_player_id(player_name):
    """Find player ID by name"""
    all_players = players.find_players_by_full_name(player_name)
    if not all_players:
        print(f"❌ Player '{player_name}' not found")
        return None
    return all_players[0]['id']


def test_league_defensive_stats():
    """
    Test 1: League-wide defensive stats (basic approach)
    
    Endpoint: LeagueDashPlayerStats with measure_type="Defense"
    
    Returns:
        DataFrame with defensive metrics for all players
    """
    print("\n" + "="*80)
    print("TEST 1: League-Wide Defensive Stats")
    print("="*80)
    print(f"Endpoint: LeagueDashPlayerStats(measure_type='Defense')")
    print(f"Season: {SEASON}\n")
    
    try:
        # Fetch defensive stats for all players
        defensive_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame"
        )
        
        df = defensive_stats.get_data_frames()[0]
        
        print(f"✅ SUCCESS! Fetched {len(df)} players\n")
        
        # Show defensive columns available
        print("📊 Defensive Columns Available:")
        defensive_cols = [col for col in df.columns if any(
            keyword in col.upper() for keyword in 
            ['DEF', 'STL', 'BLK', 'DREB', 'OPP']
        )]
        for col in defensive_cols:
            print(f"   • {col}")
        
        # Show top 5 players by steals
        print(f"\n🏀 Top 5 Players by Steals (Per Game):")
        if 'STL' in df.columns:
            top_stl = df.nlargest(5, 'STL')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'STL', 'BLK', 'MIN']]
            print(top_stl.to_string(index=False))
        
        # Show sample data for our test players
        print(f"\n🎯 Test Players Sample Data:")
        for test_player in TEST_PLAYERS[:3]:  # Show first 3
            player_data = df[df['PLAYER_NAME'] == test_player]
            if not player_data.empty:
                row = player_data.iloc[0]
                print(f"\n   {test_player}:")
                print(f"      Team: {row.get('TEAM_ABBREVIATION', 'N/A')}")
                print(f"      STL: {row.get('STL', 'N/A'):.1f}")
                print(f"      BLK: {row.get('BLK', 'N/A'):.1f}")
                print(f"      DREB: {row.get('DREB', 'N/A'):.1f}")
                if 'DEF_RATING' in df.columns:
                    print(f"      DEF_RATING: {row.get('DEF_RATING', 'N/A'):.1f}")
        
        return df
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_player_defense_dashboard(player_name):
    """
    Test 2: Player Defense Dashboard (opponent shooting data)
    
    Endpoint: PlayerDashPtShotDefend
    
    Shows:
    - Opponent FG% when guarded by this player
    - Breakdown by shot distance
    - Breakdown by play type
    
    Args:
        player_name: Player name to test
        
    Returns:
        dict: Multiple DataFrames with defensive breakdowns
    """
    print("\n" + "="*80)
    print(f"TEST 2: Player Defense Dashboard - {player_name}")
    print("="*80)
    print(f"Endpoint: PlayerDashPtShotDefend")
    print(f"Season: {SEASON}\n")
    
    player_id = get_player_id(player_name)
    if not player_id:
        return None
    
    try:
        # PlayerDashPtShotDefend requires team_id (use 0 for all teams)
        defense_dashboard = playerdashptshotdefend.PlayerDashPtShotDefend(
            player_id=player_id,
            team_id=0,  # 0 = all teams
            season=SEASON,
            season_type_all_star="Regular Season"
        )
        
        dfs = defense_dashboard.get_data_frames()
        
        print(f"✅ SUCCESS! Fetched {len(dfs)} DataFrames\n")
        
        # Show what data is available
        print("📊 Available DataFrames:")
        for i, df in enumerate(dfs):
            print(f"   DataFrame {i}: {len(df)} rows, {len(df.columns)} columns")
            if len(df) > 0:
                print(f"      Columns: {', '.join(df.columns[:5])}...")
        
        # Show overall defensive stats (usually first DataFrame)
        if len(dfs) > 0 and len(dfs[0]) > 0:
            print(f"\n🎯 Overall Defensive Stats:")
            overall = dfs[0].iloc[0]
            print(f"   Player: {overall.get('PLAYER_NAME', player_name)}")
            if 'DFGM' in overall:
                print(f"   Defensive FGM: {overall.get('DFGM', 'N/A')}")
            if 'DFGA' in overall:
                print(f"   Defensive FGA: {overall.get('DFGA', 'N/A')}")
            if 'DFG_PCT' in overall:
                print(f"   Opponent FG%: {overall.get('DFG_PCT', 'N/A'):.1%}")
        
        return dfs
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_player_tracking_defense(player_name):
    """
    Test 3: Player Tracking Defense (most advanced)
    
    Endpoint: LeagueDashPtDefend (league-wide tracking defense)
    
    Shows:
    - Opponent shooting stats when defended by player
    - Breakdown by shot distance (0-6 ft, 6-10 ft, 10-15 ft, etc.)
    - Most detailed defensive data available
    
    Args:
        player_name: Player name to test
        
    Returns:
        DataFrame with tracking defensive data
    """
    print("\n" + "="*80)
    print(f"TEST 3: Player Tracking Defense - {player_name}")
    print("="*80)
    print(f"Endpoint: LeagueDashPtDefend")
    print(f"Season: {SEASON}\n")
    
    player_id = get_player_id(player_name)
    if not player_id:
        return None
    
    try:
        tracking_defense = leaguedashptdefend.LeagueDashPtDefend(
            season=SEASON,
            season_type_all_star="Regular Season",
            defense_category="Overall",
            per_mode_simple="PerGame"
        )
        
        df = tracking_defense.get_data_frames()[0]
        
        # Filter for specific player by name (column might be PLAYER_NAME not PLAYER_ID)
        if 'PLAYER_NAME' in df.columns:
            df = df[df['PLAYER_NAME'] == player_name]
        elif 'CLOSE_DEF_PERSON_ID' in df.columns:
            df = df[df['CLOSE_DEF_PERSON_ID'] == player_id]
        
        if len(df) == 0:
            print(f"⚠️  Player not found in tracking data")
            print(f"   Available columns: {', '.join(df.columns[:10])}...")
            return None
        
        print(f"✅ SUCCESS! Fetched {len(df)} records\n")
        
        # Get all columns first
        all_cols = df.columns.tolist()
        
        # Show columns
        print("📊 Tracking Columns Available:")
        for col in all_cols[:10]:  # Show first 10 columns
            print(f"   • {col}")
        if len(all_cols) > 10:
            print(f"   ... and {len(all_cols) - 10} more columns")
        
        # Show sample data
        if len(df) > 0:
            print(f"\n🎯 Defensive Tracking Data for {player_name}:")
            print("-" * 80)
            
            # Show first row's data
            row = df.iloc[0]
            for col in all_cols[:15]:  # Show first 15 columns
                value = row.get(col, 'N/A')
                print(f"   {col}: {value}")
            
            if len(all_cols) > 15:
                print(f"   ... and {len(all_cols) - 15} more fields")
        
        return df
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def compare_defensive_endpoints():
    """
    Test 4: Compare data from all three endpoints
    
    Shows which endpoint provides which metrics
    """
    print("\n" + "="*80)
    print("TEST 4: Endpoint Comparison")
    print("="*80)
    print("Comparing what defensive metrics each endpoint provides:\n")
    
    metrics_by_endpoint = {
        "LeagueDashPlayerStats (Defense)": [
            "STL (Steals)",
            "BLK (Blocks)",
            "DREB (Defensive Rebounds)",
            "DEF_RATING (Defensive Rating)",
            "PLUS_MINUS (Plus/Minus)",
            "PF (Personal Fouls)"
        ],
        "PlayerDashPtShotDefend": [
            "DFGM (Defensive FG Made - opponent)",
            "DFGA (Defensive FG Attempted - opponent)",
            "DFG_PCT (Opponent FG%)",
            "Breakdown by shot type",
            "Breakdown by play type"
        ],
        "LeagueDashPtDefend (Tracking)": [
            "FGA/FGM by shot distance when defended",
            "Most detailed opponent shooting data",
            "0-6 ft, 6-10 ft, 10-15 ft, 15+ ft breakdowns",
            "Exact opponent FG% when contested"
        ]
    }
    
    for endpoint, metrics in metrics_by_endpoint.items():
        print(f"📊 {endpoint}:")
        for metric in metrics:
            print(f"   • {metric}")
        print()
    
    print("💡 RECOMMENDATION:")
    print("   Use LeagueDashPlayerStats for basic defensive stats (STL, BLK)")
    print("   Use LeagueDashPtDefend for advanced analysis (opponent shooting)")
    print("   Use PlayerDashPtShotDefend for individual player shot defense data")


def main():
    """Run all defensive stats tests"""
    print("="*80)
    print("NBA DEFENSIVE STATS API TEST (2025-26 SEASON)")
    print("="*80)
    print(f"\nTesting defensive data endpoints for season: {SEASON}")
    print(f"Test players: {', '.join(TEST_PLAYERS)}\n")
    
    results = {
        'league_stats': False,
        'defense_dashboard': False,
        'tracking_defense': False
    }
    
    # Test 1: League-wide defensive stats
    league_df = test_league_defensive_stats()
    results['league_stats'] = league_df is not None
    
    time.sleep(1)  # Rate limit
    
    # Test 2: Player defense dashboard (test with first player)
    test_player = TEST_PLAYERS[0]
    dashboard_dfs = test_player_defense_dashboard(test_player)
    results['defense_dashboard'] = dashboard_dfs is not None
    
    time.sleep(1)  # Rate limit
    
    # Test 3: Player tracking defense (test with first player)
    tracking_df = test_player_tracking_defense(test_player)
    results['tracking_defense'] = tracking_df is not None
    
    # Test 4: Compare endpoints
    compare_defensive_endpoints()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(results.values())
    total_tests = len(results)
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} endpoints successful\n")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test_name.replace('_', ' ').title()}")
    
    if success_count == total_tests:
        print("\n🎉 All defensive stats endpoints working!")
        print("\n💡 Next Steps:")
        print("   1. Use LeagueDashPlayerStats for basic defensive metrics")
        print("   2. Use LeagueDashPtDefend for opponent shooting analysis")
        print("   3. Create implementation script: scripts/fetch_nba_defensive_stats.py")
        print("   4. Integrate with betting strategy (e.g., matchup analysis)")
    elif success_count > 0:
        print("\n⚠️  Some endpoints working, others failed")
        print("   Check error messages above for details")
    else:
        print("\n❌ All tests failed")
        print("   Possible issues:")
        print("   - Season 2025-26 not yet available in API")
        print("   - API endpoint changes")
        print("   - Network/SSL issues")
    
    print()


if __name__ == '__main__':
    main()

