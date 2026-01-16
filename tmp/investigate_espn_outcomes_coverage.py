"""
Investigate ESPN Outcomes Coverage

Check if ESPN API is only returning games for certain teams (e.g., top 25, Power 5).

This will help diagnose why we only have 978 games instead of ~5,000+ for a full season.

Usage:
    # Analyze cached data
    python tmp/investigate_espn_outcomes_coverage.py --season 2024-25
    
    # Test ESPN API directly with different parameters
    python tmp/investigate_espn_outcomes_coverage.py --testing-api --date 2024-12-01

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import pandas as pd
import boto3
import requests
import urllib3
from pathlib import Path
from io import StringIO
import argparse
from datetime import datetime

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Find project root
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'tmp'))

from join_ncaab_outcomes_and_lines import SEASON_DATES, get_cache_path

# Power 5 conferences + Big East
POWER_CONFERENCES = {
    'SEC', 'Big Ten', 'ACC', 'Big 12', 'Pac-12', 'Big East'
}

# Known top programs (rough list)
TOP_PROGRAMS = {
    'Duke', 'North Carolina', 'Kansas', 'Kentucky', 'Gonzaga',
    'Villanova', 'Michigan State', 'UCLA', 'Arizona', 'Louisville',
    'Florida', 'Ohio State', 'Michigan', 'Syracuse', 'Indiana',
    'UConn', 'Maryland', 'Wisconsin', 'Texas', 'Purdue',
    'Auburn', 'Tennessee', 'Alabama', 'Houston', 'Baylor'
}


def analyze_outcomes_bias(season='2024-25'):
    """
    Analyze if ESPN outcomes have bias toward certain teams.
    
    Args:
        season: Season string (e.g., '2024-25')
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}")
    
    start_date, end_date = SEASON_DATES[season]
    
    # Check cache
    cache_path = get_cache_path('outcomes', start_date, end_date)
    
    if not cache_path.exists():
        print(f"❌ Cache not found. Run this first:")
        print(f"   python tmp/build_ncaab_team_name_mapping_v2.py --season {season}")
        return
    
    print(f"{'='*80}")
    print(f"ESPN OUTCOMES COVERAGE INVESTIGATION: {season}")
    print(f"{'='*80}")
    
    # Load outcomes
    df = pd.read_parquet(cache_path)
    
    print(f"\n📊 Overall Stats:")
    print(f"   Total games: {len(df):,}")
    print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Get all teams
    home_teams = df.groupby('HOME_TEAM').size()
    away_teams = df.groupby('AWAY_TEAM').size()
    total_games = (home_teams.add(away_teams, fill_value=0)).sort_values(ascending=False)
    
    print(f"   Unique teams: {len(total_games)}")
    print(f"   Avg games per team: {total_games.mean():.1f}")
    print(f"   Median games per team: {total_games.median():.1f}")
    print(f"   Max games: {total_games.max():.0f}")
    print(f"   Min games: {total_games.min():.0f}")
    
    # Distribution analysis
    print(f"\n📈 Games Distribution:")
    
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]
    distribution = pd.cut(total_games, bins=bins, right=False).value_counts().sort_index()
    
    for interval, count in distribution.items():
        pct = count / len(total_games) * 100
        print(f"   {str(interval):15} {count:3} teams ({pct:5.1f}%)")
    
    # Check for potential biases
    print(f"\n🔍 Potential Bias Analysis:")
    
    # Teams with many games (>20)
    many_games = total_games[total_games >= 20]
    print(f"\n   Teams with 20+ games: {len(many_games)} / {len(total_games)} ({len(many_games)/len(total_games)*100:.1f}%)")
    
    # Check if they're top programs
    many_games_list = [team for team in many_games.index]
    top_program_count = sum(1 for team in many_games_list if any(prog in team for prog in TOP_PROGRAMS))
    
    print(f"   Of these, {top_program_count} appear to be top programs")
    print(f"   This suggests: {'⚠️  BIAS toward top teams' if top_program_count > len(many_games) * 0.7 else '✅ No obvious bias'}")
    
    # Teams with few games (<5)
    few_games = total_games[total_games < 5]
    print(f"\n   Teams with <5 games: {len(few_games)} / {len(total_games)} ({len(few_games)/len(total_games)*100:.1f}%)")
    
    if len(few_games) > 0:
        print(f"\n   Sample teams with few games:")
        for team, games in few_games.head(20).items():
            print(f"      {team:<40} {int(games)} games")
    
    # Date distribution - check if certain dates have fewer games
    print(f"\n📅 Games by Date:")
    games_by_date = df.groupby('GAME_DATE').size().sort_index()
    
    print(f"   Dates with games: {len(games_by_date)}")
    print(f"   Avg games per date: {games_by_date.mean():.1f}")
    print(f"   Max games in a day: {games_by_date.max()}")
    print(f"   Min games in a day: {games_by_date.min()}")
    
    # Dates with very few games
    sparse_dates = games_by_date[games_by_date < 5]
    if len(sparse_dates) > 20:
        print(f"\n   ⚠️  {len(sparse_dates)} dates with <5 games (might be incomplete data)")
        print(f"   Sample sparse dates:")
        for date, count in sparse_dates.head(10).items():
            print(f"      {date}: {count} games")
    
    # Check specific date ranges
    print(f"\n📆 Coverage by Month:")
    df['month'] = pd.to_datetime(df['GAME_DATE']).dt.to_period('M')
    monthly = df.groupby('month').size()
    
    for month, count in monthly.items():
        print(f"   {month}: {count:4} games")
    
    # Final diagnosis
    print(f"\n{'='*80}")
    print("🔬 DIAGNOSIS:")
    print(f"{'='*80}")
    
    if total_games.mean() < 15:
        print("❌ PROBLEM: Average games per team is very low (<15)")
        print("   Likely causes:")
        print("   1. ESPN API only returning games for major/ranked teams")
        print("   2. Data fetch was incomplete or filtered")
        print("   3. ESPN API has limited historical data")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Check ESPN API response for a random 'small' team")
        print("   2. Try fetching with different parameters")
        print("   3. Consider alternative data source for mid-major teams")
    else:
        print("✅ Coverage looks reasonable")
        print(f"   Most teams have adequate game counts")
    
    print(f"\n{'='*80}")


def test_espn_api_parameters(test_date='2024-12-01'):
    """
    Test ESPN API with different parameters to see if we can get more games.
    
    Args:
        test_date: Date to test in YYYY-MM-DD format
    """
    print(f"{'='*80}")
    print(f"ESPN API PARAMETER TESTING: {test_date}")
    print(f"{'='*80}")
    
    # Convert date to YYYYMMDD format
    date_obj = datetime.strptime(test_date, '%Y-%m-%d')
    date_str = date_obj.strftime('%Y%m%d')
    
    base_url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
    
    # Test different parameter combinations
    test_configs = [
        {
            'name': 'Current Implementation (limit=300)',
            'params': {'dates': date_str, 'limit': 300}
        },
        {
            'name': 'Higher Limit (limit=500)',
            'params': {'dates': date_str, 'limit': 500}
        },
        {
            'name': 'No Limit Parameter',
            'params': {'dates': date_str}
        },
        {
            'name': 'With Groups Parameter (top25)',
            'params': {'dates': date_str, 'limit': 300, 'groups': 'top25'}
        },
        {
            'name': 'With Groups Parameter (50)',
            'params': {'dates': date_str, 'limit': 300, 'groups': '50'}
        },
        {
            'name': 'With seasontype=2 (regular season)',
            'params': {'dates': date_str, 'limit': 300, 'seasontype': '2'}
        },
        {
            'name': 'Calendar API (alternative endpoint)',
            'url': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams',
            'params': {}
        }
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_configs)}: {config['name']}")
        print(f"{'='*80}")
        
        # Use custom URL if provided, otherwise use base_url
        url = config.get('url', base_url)
        params = config['params']
        
        print(f"URL: {url}")
        print(f"Params: {params}")
        
        try:
            response = requests.get(url, params=params, timeout=15, verify=False)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if 'events' in data:
                events = data.get('events', [])
                num_games = len(events)
                
                print(f"✅ Success: Found {num_games} games")
                
                # Show sample teams
                if events:
                    print(f"\n   Sample games:")
                    for event in events[:5]:
                        comps = event.get('competitions', [{}])[0]
                        competitors = comps.get('competitors', [])
                        if len(competitors) == 2:
                            home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                            away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                            home_name = home.get('team', {}).get('displayName', '?')
                            away_name = away.get('team', {}).get('displayName', '?')
                            print(f"      {away_name} @ {home_name}")
                
                # Check for pagination indicators
                if 'page' in data:
                    print(f"\n   📄 Pagination info:")
                    page_info = data['page']
                    print(f"      Current page: {page_info.get('number', 'N/A')}")
                    print(f"      Total pages: {page_info.get('totalPages', 'N/A')}")
                    print(f"      Items per page: {page_info.get('size', 'N/A')}")
                    print(f"      Total items: {page_info.get('totalElements', 'N/A')}")
                
                results.append({
                    'test': config['name'],
                    'games': num_games,
                    'status': 'success'
                })
            elif 'teams' in data:
                teams = data.get('teams', [])
                print(f"✅ Success: Found {len(teams)} teams (alternative endpoint)")
                results.append({
                    'test': config['name'],
                    'games': f"{len(teams)} teams",
                    'status': 'success'
                })
            else:
                print(f"⚠️  Response structure unexpected")
                print(f"   Keys in response: {list(data.keys())}")
                results.append({
                    'test': config['name'],
                    'games': 'unknown structure',
                    'status': 'unexpected'
                })
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            results.append({
                'test': config['name'],
                'games': 0,
                'status': 'failed'
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'test': config['name'],
                'games': 0,
                'status': 'error'
            })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nResults for {test_date}:\n")
    
    for result in results:
        status_emoji = {'success': '✅', 'failed': '❌', 'unexpected': '⚠️', 'error': '❌'}
        emoji = status_emoji.get(result['status'], '❓')
        print(f"{emoji} {result['test']:<50} {result['games']}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success' and isinstance(r['games'], int)]
    if successful:
        best = max(successful, key=lambda x: x['games'])
        print(f"\n✅ Best configuration: {best['test']}")
        print(f"   Games found: {best['games']}")
        print(f"\n💡 Consider updating fetch_historical_game_results_espn_api.py to use this configuration")
    else:
        print(f"\n❌ No successful configurations found")
        print(f"   The ESPN API might have restrictions or require authentication")
    
    # Check if we're missing games
    print(f"\n📊 Expected vs Actual:")
    print(f"   On a typical NCAAB game day in December, there should be 50-150+ games")
    if successful and best['games'] < 50:
        print(f"   ⚠️  Only finding {best['games']} games suggests API is filtering results")
        print(f"   Possible causes:")
        print(f"      1. ESPN API only returns games for Division I teams")
        print(f"      2. API requires pagination for all games")
        print(f"      3. API has rate limits or filters by default")
        print(f"      4. Need to query by conference/group and aggregate")
    elif successful:
        print(f"   ✅ Finding {best['games']} games looks reasonable")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='2024-25',
                       help='Season to analyze (e.g., 2024-25)')
    parser.add_argument('--testing-api', action='store_true',
                       help='Test ESPN API with different parameters to find optimal configuration')
    parser.add_argument('--date', type=str, default='2024-12-01',
                       help='Date to test API with (YYYY-MM-DD format), only used with --testing-api')
    args = parser.parse_args()
    
    if args.testing_api:
        test_espn_api_parameters(test_date=args.date)
    else:
        analyze_outcomes_bias(season=args.season)

