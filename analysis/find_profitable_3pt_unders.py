"""
Daily 3PT Under Betting Opportunity Finder

Based on backtesting results from 2024-25 season:
- Lines 2.5+: PROFITABLE (+1% to +4% ROI)
- Lines 0.5-1.5: UNPROFITABLE (-4% to -7% ROI)

This script:
1. Fetches today's NBA games
2. Gets 3PT player prop odds
3. Filters for profitable opportunities (lines >= 2.5)
4. Shows best under bets with odds and expected value

Usage:
    python find_profitable_3pt_unders.py
    python find_profitable_3pt_unders.py --date 2024-11-24
    python find_profitable_3pt_unders.py --min-line 2.5 --min-odds -150

Author: Myles Thomas
Date: 2024-11-24
"""

import os
import sys
import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.odds_utils import calculate_bet_amount, calculate_profit, odds_to_implied_probability


# ============================================================================
# CONFIG
# ============================================================================

# API Configuration
API_KEY = os.environ.get('ODDS_API_KEY')
if not API_KEY:
    print("⚠️  WARNING: ODDS_API_KEY environment variable not set")
    print("   Set it with: export ODDS_API_KEY='your_key_here'")
    print()

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Betting Configuration
TARGET_WIN = 100  # Bet to win $100
MAX_ODDS_THRESHOLD = -300  # Skip odds worse than this (bookmaker traps)

# Strategy Configuration (from backtest results)
PROFITABLE_LINE_THRESHOLD = 2.5  # Lines >= 2.5 showed positive ROI
EXPECTED_WIN_RATES = {
    # From backtest results
    0.5: 0.452,  # 45.2% win rate (UNPROFITABLE)
    1.5: 0.511,  # 51.1% win rate (UNPROFITABLE)
    2.5: 0.538,  # 53.8% win rate (+1.01% ROI) ✅
    3.5: 0.564,  # 56.4% win rate (+3.95% ROI) ✅✅
    4.5: 0.570,  # 57.0% win rate (+4.31% ROI) ✅✅
}

# Default to 53% win rate for lines not in our backtest
DEFAULT_WIN_RATE = 0.53


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_todays_nba_games(target_date=None):
    """
    Fetch today's NBA games from the-odds-api.
    
    Args:
        target_date: Optional datetime object for specific date
    
    Returns:
        List of game dictionaries
    """
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events"
    
    params = {
        'apiKey': API_KEY,
    }
    
    if target_date:
        params['commenceTimeFrom'] = target_date.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
        params['commenceTimeTo'] = target_date.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        games = response.json()
        print(f"✅ Found {len(games)} NBA games")
        return games
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching games: {e}")
        return []


def get_player_props_for_game(event_id):
    """
    Fetch 3PT player props for a specific game.
    
    Args:
        event_id: The odds API event ID
    
    Returns:
        Dictionary with prop data
    """
    url = f"{ODDS_API_BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'player_threes',
        'oddsFormat': 'american',
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error fetching props for event {event_id}: {e}")
        return None


def parse_player_props(props_data):
    """
    Parse player prop data into structured format.
    
    Returns:
        List of prop dictionaries with player, line, over_odds, under_odds
    """
    if not props_data or 'bookmakers' not in props_data:
        return []
    
    # Aggregate props across all bookmakers
    player_props = defaultdict(lambda: {'over_odds': [], 'under_odds': []})
    
    for bookmaker in props_data['bookmakers']:
        bookmaker_name = bookmaker['key']
        
        for market in bookmaker['markets']:
            if market['key'] != 'player_threes':
                continue
            
            for outcome in market['outcomes']:
                player_name = outcome['description']
                line = outcome['point']
                odds = outcome['price']
                outcome_type = outcome['name']  # 'Over' or 'Under'
                
                key = (player_name, line)
                
                if outcome_type == 'Over':
                    player_props[key]['over_odds'].append((bookmaker_name, odds))
                elif outcome_type == 'Under':
                    player_props[key]['under_odds'].append((bookmaker_name, odds))
    
    # Convert to list format with best odds
    props_list = []
    for (player, line), odds_data in player_props.items():
        if odds_data['over_odds'] and odds_data['under_odds']:
            # Get best odds for each side
            best_over = max(odds_data['over_odds'], key=lambda x: x[1])
            best_under = max(odds_data['under_odds'], key=lambda x: x[1])
            
            props_list.append({
                'player': player,
                'line': line,
                'over_best_odds': best_over[1],
                'over_best_book': best_over[0],
                'under_best_odds': best_under[1],
                'under_best_book': best_under[0],
                'num_books': len(odds_data['over_odds']),
            })
    
    return props_list


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_expected_value(line, odds, win_rate=None):
    """
    Calculate expected value of a bet.
    
    Args:
        line: The prop line (e.g., 2.5)
        odds: American odds
        win_rate: Override win rate (otherwise use backtest data)
    
    Returns:
        Expected value per $100 bet
    """
    if win_rate is None:
        # Use closest line from backtest
        if line <= 0.5:
            win_rate = EXPECTED_WIN_RATES[0.5]
        elif line <= 1.5:
            win_rate = EXPECTED_WIN_RATES[1.5]
        elif line <= 2.5:
            win_rate = EXPECTED_WIN_RATES[2.5]
        elif line <= 3.5:
            win_rate = EXPECTED_WIN_RATES[3.5]
        elif line <= 4.5:
            win_rate = EXPECTED_WIN_RATES[4.5]
        else:
            win_rate = DEFAULT_WIN_RATE
    
    bet_amount = calculate_bet_amount(odds, TARGET_WIN)
    
    # EV = (win_rate * profit_if_win) + (lose_rate * loss_if_lose)
    profit_if_win = TARGET_WIN
    loss_if_lose = -bet_amount
    
    ev = (win_rate * profit_if_win) + ((1 - win_rate) * loss_if_lose)
    
    return ev


def filter_profitable_opportunities(props, min_line=2.5, min_odds=-300, max_odds=None):
    """
    Filter props for profitable betting opportunities.
    
    Args:
        props: List of prop dictionaries
        min_line: Minimum line value (default 2.5, from backtest)
        min_odds: Minimum acceptable odds (avoid bookmaker traps)
        max_odds: Maximum acceptable odds (optional, e.g., +150)
    
    Returns:
        Filtered list sorted by expected value
    """
    opportunities = []
    
    for prop in props:
        line = prop['line']
        under_odds = prop['under_best_odds']
        
        # Apply filters
        if line < min_line:
            continue
        
        if under_odds <= min_odds:
            # Bookmaker trap (e.g., -500 odds)
            continue
        
        if max_odds and under_odds > max_odds:
            # Too underdog (optional filter)
            continue
        
        # Calculate metrics
        implied_prob = odds_to_implied_probability(under_odds)
        bet_amount = calculate_bet_amount(under_odds, TARGET_WIN)
        ev = calculate_expected_value(line, under_odds)
        
        # Get expected win rate from backtest
        if line <= 0.5:
            expected_wr = EXPECTED_WIN_RATES[0.5]
        elif line <= 1.5:
            expected_wr = EXPECTED_WIN_RATES[1.5]
        elif line <= 2.5:
            expected_wr = EXPECTED_WIN_RATES[2.5]
        elif line <= 3.5:
            expected_wr = EXPECTED_WIN_RATES[3.5]
        elif line <= 4.5:
            expected_wr = EXPECTED_WIN_RATES[4.5]
        else:
            expected_wr = DEFAULT_WIN_RATE
        
        opportunities.append({
            **prop,
            'bet_amount': bet_amount,
            'implied_prob': implied_prob * 100,
            'expected_win_rate': expected_wr * 100,
            'edge': (expected_wr * 100) - (implied_prob * 100),
            'expected_value': ev,
        })
    
    # Sort by expected value (highest first)
    opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
    
    return opportunities


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_opportunities(opportunities, game_info=None):
    """
    Display betting opportunities in a nice format.
    """
    if not opportunities:
        print("❌ No profitable opportunities found with current filters")
        return
    
    print()
    print("="*120)
    print("🎯 PROFITABLE 3PT UNDER OPPORTUNITIES")
    print("="*120)
    
    if game_info:
        print(f"\n📅 {game_info['home_team']} vs {game_info['away_team']}")
        print(f"   Start: {game_info['commence_time']}")
    
    print()
    print(f"{'#':<4} {'Player':<25} {'Line':<6} {'Odds':<8} {'Book':<20} {'Bet':<10} {'Impl%':<8} {'Exp%':<8} {'Edge':<8} {'EV':<10}")
    print("-"*120)
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i:<4} {opp['player']:<25} {opp['line']:<6.1f} {opp['under_best_odds']:<8} "
              f"{opp['under_best_book']:<20} ${opp['bet_amount']:<9.2f} "
              f"{opp['implied_prob']:<7.1f}% {opp['expected_win_rate']:<7.1f}% "
              f"{opp['edge']:+7.1f}% ${opp['expected_value']:+8.2f}")
    
    print()
    print(f"Total opportunities: {len(opportunities)}")
    print(f"Total expected value: ${sum(o['expected_value'] for o in opportunities):+.2f}")
    print()


def display_summary(all_opportunities):
    """
    Display summary statistics.
    """
    if not all_opportunities:
        return
    
    total_ev = sum(o['expected_value'] for o in all_opportunities)
    total_stake = sum(o['bet_amount'] for o in all_opportunities)
    avg_edge = sum(o['edge'] for o in all_opportunities) / len(all_opportunities)
    
    positive_ev = [o for o in all_opportunities if o['expected_value'] > 0]
    
    print()
    print("="*120)
    print("📊 SUMMARY")
    print("="*120)
    print()
    print(f"Total Opportunities: {len(all_opportunities)}")
    print(f"Positive EV Bets: {len(positive_ev)}")
    print(f"Total Stake: ${total_stake:.2f}")
    print(f"Total Expected Value: ${total_ev:+.2f}")
    print(f"Average Edge: {avg_edge:+.2f}%")
    
    if total_stake > 0:
        print(f"Expected ROI: {(total_ev / total_stake * 100):+.2f}%")
    
    print()
    
    # Line breakdown
    line_breakdown = defaultdict(list)
    for o in all_opportunities:
        line_breakdown[o['line']].append(o)
    
    print("Breakdown by Line:")
    print(f"{'Line':<8} {'Count':<8} {'Avg EV':<12} {'Total EV':<12}")
    print("-"*50)
    for line in sorted(line_breakdown.keys()):
        opps = line_breakdown[line]
        avg_ev = sum(o['expected_value'] for o in opps) / len(opps)
        total_ev_line = sum(o['expected_value'] for o in opps)
        print(f"{line:<8.1f} {len(opps):<8} ${avg_ev:<11.2f} ${total_ev_line:+11.2f}")
    
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find profitable 3PT under betting opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find opportunities for today
  python %(prog)s
  
  # Specific date
  python %(prog)s --date 2024-11-24
  
  # Adjust filters
  python %(prog)s --min-line 2.5 --min-odds -200
  python %(prog)s --min-line 3.5  # Only highest lines
        """
    )
    
    parser.add_argument('--date', type=str,
                       help='Date to check (YYYY-MM-DD), default is today')
    parser.add_argument('--min-line', type=float, default=2.5,
                       help='Minimum line value (default: 2.5)')
    parser.add_argument('--min-odds', type=int, default=-300,
                       help='Minimum odds threshold (default: -300)')
    parser.add_argument('--max-odds', type=int,
                       help='Maximum odds threshold (optional)')
    
    args = parser.parse_args()
    
    # Parse date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("   Use YYYY-MM-DD format")
            return
    else:
        target_date = datetime.now()
    
    print()
    print("="*120)
    print("🏀 NBA 3PT UNDER OPPORTUNITY FINDER")
    print("="*120)
    print()
    print(f"Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Strategy: Bet UNDER on lines >= {args.min_line} (backtest shows +1% to +4% ROI)")
    print(f"Filters: Odds between {args.min_odds} and {args.max_odds if args.max_odds else 'no limit'}")
    print()
    
    # Check API key
    if not API_KEY:
        print("❌ Cannot proceed without ODDS_API_KEY")
        return
    
    # Fetch games
    print("Fetching games...")
    games = get_todays_nba_games(target_date)
    
    if not games:
        print("❌ No games found for this date")
        return
    
    print()
    
    # Process each game
    all_opportunities = []
    
    for game in games:
        game_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        
        print(f"⏳ Fetching props for {away_team} @ {home_team}...")
        
        props_data = get_player_props_for_game(game_id)
        
        if not props_data:
            print(f"   ⚠️  No prop data available")
            continue
        
        props = parse_player_props(props_data)
        print(f"   ✅ Found {len(props)} player props")
        
        # Filter for opportunities
        opportunities = filter_profitable_opportunities(
            props,
            min_line=args.min_line,
            min_odds=args.min_odds,
            max_odds=args.max_odds
        )
        
        if opportunities:
            game_info = {
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time.strftime('%Y-%m-%d %I:%M %p'),
            }
            display_opportunities(opportunities, game_info)
            all_opportunities.extend(opportunities)
    
    # Display summary
    if all_opportunities:
        display_summary(all_opportunities)
        
        print()
        print("💡 NEXT STEPS:")
        print("   1. Review the opportunities above")
        print("   2. Place bets on your preferred sportsbook")
        print("   3. Track results for ongoing validation")
        print()
    else:
        print()
        print("❌ No profitable opportunities found today")
        print("   Try adjusting filters or check back closer to game time")
        print()


if __name__ == "__main__":
    main()

