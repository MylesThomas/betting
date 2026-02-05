"""
Analyze Lawler's Law using ESPN Play-by-Play Data

Context:
--------
Lawler's Law: "The first team to reach 100 points wins the game"
- Originally stated by Ralph Lawler (LA Clippers announcer)
- Held true ~95% of the time in 1970s-80s
- Question: Is 100 still the optimal threshold, or has it shifted to 105/110/115?

Goal:
-----
Test various point thresholds (90, 95, 100, 105, 110, 115, 120) to find:
1. Win rate when reaching threshold first
2. How often each threshold is reached in games
3. Which threshold is most predictive in modern NBA

Data Source:
-----------
ESPN Play-by-Play API:
- Endpoint: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}
- Contains: Play-by-play with timestamps, scores, and events
- Alternative: https://cdn.espn.com/core/nba/playbyplay?xhr=1&gameId={game_id}

Output:
-------
Table format:
Threshold | Win Rate (First to X) | Games Where X Hit | % of All Games
----------------------------------------------------------------------
100       | 82.3%                 | 2,847            | 94.2%
105       | 84.1%                 | 2,654            | 87.8%
110       | 86.7%                 | 2,398            | 79.4%
115       | 89.2%                 | 1,987            | 65.8%
120       | 91.4%                 | 1,523            | 50.4%

Usage:
------
# Test with single game
python tmp/analyze_lawler_law_espn_pbp.py --test-game 401704974

# Analyze recent completed games
python tmp/analyze_lawler_law_espn_pbp.py --recent --limit 50

# Analyze games from specific date
python tmp/analyze_lawler_law_espn_pbp.py --date 20260125 --limit 25

Finding Game IDs:
-----------------
Method 1: ESPN website
  - Go to game page: https://www.espn.com/nba/game/_/gameId/401704974
  - Game ID is in URL (e.g., 401704974)

Method 2: ESPN scoreboard API
  - Today: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  - Specific date: ...scoreboard?dates=20260125

Method 3: Use --recent flag to automatically fetch latest completed games

Author: Thomas Myles
Date: 2026-02-04
"""

import requests
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import time
import sys
import ssl
import urllib3

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))


# =============================================================================
# ESPN API FUNCTIONS
# =============================================================================

def get_espn_pbp_data(game_id):
    """
    Fetch play-by-play data for a single game from ESPN.
    
    Args:
        game_id: ESPN game ID (e.g., 401704974)
    
    Returns:
        Dictionary with game summary and play-by-play data
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching game {game_id}: {e}")
        return None


def get_recent_games(date=None, limit=25):
    """
    Get recent/today's games from ESPN scoreboard.
    
    Args:
        date: Date string in YYYYMMDD format (default: today)
        limit: Max games to return
    
    Returns:
        List of game IDs
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}&limit={limit}"
    
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        game_ids = []
        if 'events' in data:
            for event in data['events']:
                game_id = event['id']
                # Check if game is completed
                status = event.get('status', {}).get('type', {}).get('state', '')
                if status == 'post':  # Only completed games
                    game_ids.append(game_id)
        
        return game_ids
    except Exception as e:
        print(f"❌ Error fetching games for {date}: {e}")
        return []


def get_season_schedule(season):
    """
    Get all games for a season from ESPN.
    
    NOTE: ESPN API requires iterating through dates. For large-scale analysis,
    better to use nba_api or scrape from ESPN's schedule page.
    
    Args:
        season: NBA season year (e.g., 2026 for 2025-26 season)
    
    Returns:
        List of game IDs
    """
    # TODO: Implement date range iteration
    # For now, just get games from a sample of dates
    print("⚠️  Full season scraping not yet implemented")
    print("   Use --recent or --test-game for now")
    return []


# =============================================================================
# PLAY-BY-PLAY PARSING
# =============================================================================

def parse_game_info(game_data):
    """
    Extract game info (teams, scores, winner) from ESPN data.
    
    Args:
        game_data: ESPN game JSON data
    
    Returns:
        Dict with game metadata
    """
    if 'header' not in game_data or 'competitions' not in game_data['header']:
        return None
    
    competition = game_data['header']['competitions'][0]
    competitors = competition.get('competitors', [])
    
    if len(competitors) != 2:
        return None
    
    # Find home/away teams
    home = next((c for c in competitors if c['homeAway'] == 'home'), None)
    away = next((c for c in competitors if c['homeAway'] == 'away'), None)
    
    if not home or not away:
        return None
    
    return {
        'game_id': game_data.get('header', {}).get('id', ''),
        'home_team': home['team']['displayName'],
        'away_team': away['team']['displayName'],
        'home_score': int(home['score']),
        'away_score': int(away['score']),
        'winner': 'home' if home.get('winner', False) else 'away',
        'date': game_data.get('header', {}).get('competitions', [{}])[0].get('date', '')
    }


def parse_pbp_for_scoring(pbp_data):
    """
    Parse play-by-play data to track scoring progression.
    
    Args:
        pbp_data: ESPN play-by-play JSON data
    
    Returns:
        DataFrame with columns: period, clock, home_score, away_score, event_text
    """
    plays = []
    
    # Navigate ESPN's nested structure
    if 'plays' not in pbp_data:
        return pd.DataFrame()
    
    for play in pbp_data['plays']:
        # Extract relevant fields
        period = play.get('period', {}).get('number', 0)
        clock = play.get('clock', {}).get('displayValue', '0:00')
        home_score = play.get('homeScore', 0)
        away_score = play.get('awayScore', 0)
        text = play.get('text', '')
        
        plays.append({
            'period': period,
            'clock': clock,
            'home_score': home_score,
            'away_score': away_score,
            'event_text': text
        })
    
    return pd.DataFrame(plays)


def find_first_to_threshold(pbp_df, threshold, home_team, away_team, winner):
    """
    Determine which team hit the threshold first and if they won.
    
    Args:
        pbp_df: DataFrame with play-by-play data
        threshold: Point threshold (e.g., 100)
        home_team: Home team name
        away_team: Away team name
        winner: 'home' or 'away'
    
    Returns:
        Dict with: {
            'threshold': int,
            'first_team': 'home' or 'away',
            'first_team_won': bool,
            'threshold_reached': bool
        }
    """
    if pbp_df.empty:
        return None
    
    # Find first play where either team hit threshold
    home_hit = pbp_df[pbp_df['home_score'] >= threshold]
    away_hit = pbp_df[pbp_df['away_score'] >= threshold]
    
    # Neither team hit threshold
    if home_hit.empty and away_hit.empty:
        return {
            'threshold': threshold,
            'first_team': None,
            'first_team_won': None,
            'threshold_reached': False
        }
    
    # Both teams hit - find who hit first (play-by-play is chronological)
    if not home_hit.empty and not away_hit.empty:
        home_idx = home_hit.index[0]
        away_idx = away_hit.index[0]
        first_team = 'home' if home_idx < away_idx else 'away'
    elif not home_hit.empty:
        first_team = 'home'
    else:
        first_team = 'away'
    
    return {
        'threshold': threshold,
        'first_team': first_team,
        'first_team_won': (first_team == winner),
        'threshold_reached': True
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_games(game_results, thresholds):
    """
    Analyze all games to calculate win rates for each threshold.
    
    Args:
        game_results: List of game analysis results
        thresholds: List of point thresholds to analyze
    
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    for threshold in thresholds:
        # Filter to games where threshold was reached
        threshold_games = [
            g for g in game_results 
            if g.get(threshold) and g[threshold]['threshold_reached']
        ]
        
        if not threshold_games:
            results.append({
                'threshold': threshold,
                'win_rate': 0.0,
                'games_hit': 0,
                'total_games': len(game_results),
                'pct_of_games': 0.0
            })
            continue
        
        # Calculate win rate
        wins = sum(1 for g in threshold_games if g[threshold]['first_team_won'])
        games_hit = len(threshold_games)
        total_games = len(game_results)
        
        results.append({
            'threshold': threshold,
            'win_rate': wins / games_hit if games_hit > 0 else 0.0,
            'games_hit': games_hit,
            'total_games': total_games,
            'pct_of_games': games_hit / total_games if total_games > 0 else 0.0
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Lawler\'s Law using ESPN play-by-play data'
    )
    parser.add_argument(
        '--test-game',
        type=str,
        help='Test with single ESPN game ID (e.g., 401704974)'
    )
    parser.add_argument(
        '--recent',
        action='store_true',
        help='Analyze recent completed games'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Analyze games from specific date (YYYYMMDD format, e.g., 20260125)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Max games to analyze (default: 50)'
    )
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=int,
        default=[90, 95, 100, 105, 110, 115, 120],
        help='Point thresholds to test (default: 90 95 100 105 110 115 120)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏀 LAWLER'S LAW ANALYSIS - ESPN PLAY-BY-PLAY DATA")
    print("="*80)
    print()
    
    # Test mode: single game
    if args.test_game:
        print(f"🧪 Testing with game ID: {args.test_game}")
        print()
        
        data = get_espn_pbp_data(args.test_game)
        if not data:
            return
        
        print(f"✅ Successfully fetched game data")
        
        # Parse game info
        game_info = parse_game_info(data)
        if game_info:
            print()
            print(f"📋 Game Info:")
            print(f"   {game_info['away_team']} @ {game_info['home_team']}")
            print(f"   Final Score: {game_info['away_team']} {game_info['away_score']}, "
                  f"{game_info['home_team']} {game_info['home_score']}")
            print(f"   Winner: {game_info['winner'].upper()}")
        
        # Parse play-by-play
        pbp_df = parse_pbp_for_scoring(data)
        print(f"\n   📊 Parsed {len(pbp_df)} plays")
        
        if pbp_df.empty:
            print("   ❌ No plays found")
            return
        
        # Test threshold analysis
        print(f"\n🎯 Testing First-to-Threshold Analysis:")
        print(f"   Thresholds: {args.thresholds}")
        print()
        
        for threshold in args.thresholds:
            result = find_first_to_threshold(
                pbp_df, threshold, 
                game_info['home_team'], game_info['away_team'], 
                game_info['winner']
            )
            
            if result and result['threshold_reached']:
                first_team_name = game_info['home_team'] if result['first_team'] == 'home' else game_info['away_team']
                won_indicator = '✅ WON' if result['first_team_won'] else '❌ LOST'
                print(f"   {threshold:>3} pts: {first_team_name:30} (first to reach) {won_indicator}")
            else:
                print(f"   {threshold:>3} pts: Neither team reached this threshold")
        
        # Save raw data for inspection
        output_file = Path.home() / 'Downloads' / 'tmp' / f'espn_pbp_{args.test_game}.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n   💾 Saved raw JSON to: {output_file}")
        
        return
    
    # Recent games or specific date analysis
    if args.recent or args.date:
        date = args.date if args.date else None
        date_label = args.date if args.date else "recent"
        
        print(f"📅 Fetching games for: {date_label}")
        game_ids = get_recent_games(date=date, limit=args.limit)
        
        if not game_ids:
            print(f"❌ No completed games found")
            return
        
        print(f"✅ Found {len(game_ids)} completed games")
        print(f"🎯 Thresholds to test: {args.thresholds}")
        print()
        
        # Analyze each game
        game_results = []
        for i, game_id in enumerate(game_ids, 1):
            print(f"[{i}/{len(game_ids)}] Analyzing game {game_id}...", end=' ')
            
            data = get_espn_pbp_data(game_id)
            if not data:
                print("❌ Failed to fetch")
                continue
            
            game_info = parse_game_info(data)
            if not game_info:
                print("❌ Failed to parse game info")
                continue
            
            pbp_df = parse_pbp_for_scoring(data)
            if pbp_df.empty:
                print("❌ No plays found")
                continue
            
            # Analyze each threshold
            game_result = {
                'game_id': game_id,
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'winner': game_info['winner']
            }
            
            for threshold in args.thresholds:
                result = find_first_to_threshold(
                    pbp_df, threshold,
                    game_info['home_team'], game_info['away_team'],
                    game_info['winner']
                )
                game_result[threshold] = result
            
            game_results.append(game_result)
            print(f"✅ {game_info['away_team']} @ {game_info['home_team']}")
            
            # Rate limiting - be nice to ESPN's servers
            time.sleep(0.5)
        
        if not game_results:
            print("\n❌ No games successfully analyzed")
            return
        
        # Analyze results
        print("\n" + "="*80)
        print("📊 LAWLER'S LAW ANALYSIS RESULTS")
        print("="*80)
        print()
        
        results_df = analyze_games(game_results, args.thresholds)
        
        # Print table
        print(f"{'Threshold':<12} | {'Win Rate':<12} | {'Games Hit':<12} | {'% of Games':<12}")
        print("-" * 55)
        for _, row in results_df.iterrows():
            print(f"{row['threshold']:<12} | "
                  f"{row['win_rate']:>10.1%}  | "
                  f"{row['games_hit']:>10}  | "
                  f"{row['pct_of_games']:>10.1%}")
        
        print()
        print(f"Total games analyzed: {len(game_results)}")
        
        # Save results
        output_file = Path.home() / 'Downloads' / 'tmp' / f'lawler_law_analysis_{date_label}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"💾 Saved detailed results to: {output_file}")
        
        return
    
    print("❌ Please specify --test-game, --recent, or --date")
    parser.print_help()


if __name__ == '__main__':
    main()
