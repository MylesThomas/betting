"""
NFL Player Props Arbitrage Finder

WHAT IT DOES:
    Finds arbitrage opportunities in NFL player prop markets.
    Uses event-by-event fetching to capture ALL available lines.
    
ARBITRAGE EXPLAINED:
    When you can bet both sides (SAME LINE) and guarantee profit regardless of outcome.
    Example:
        - Bet Over 250.5 passing yards at +115 (BookA)
        - Bet Under 250.5 passing yards at +105 (BookB)
        - Over probability: 46.5%, Under probability: 48.8%
        - Combined probability: 95.3% (< 100% = guaranteed profit!)
    
    IMPORTANT: We only compare Over/Under for the SAME player + SAME line.
               (e.g., "Mahomes Over 275.5" vs "Mahomes Under 275.5" - NOT mixing lines)
        
HOW IT WORKS (3 steps):
    1. Get all NFL games for today/this week (ET timezone)
    2. For each game, fetch all player prop markets
    3. For each (player, line) pair, find best Over/Under across bookmakers
    
USAGE:
    # Run tests (no API calls)
    python scripts/find_nfl_arb_opportunities.py --test
    
    # See demo flow (no API calls)
    python scripts/find_nfl_arb_opportunities.py --demo
    
    # Default: just passing yards for today's games
    python scripts/find_nfl_arb_opportunities.py
    
    # All games this week, just passing yards (default market)
    python scripts/find_nfl_arb_opportunities.py --week
    
    # ALL MARKETS for all games this week (passing, rushing, receiving, TDs, kicking, defense)
    python scripts/find_nfl_arb_opportunities.py --week --all-markets
    
    # Test with 1 game (auto-enables week mode to find upcoming games)
    python scripts/find_nfl_arb_opportunities.py --limit 1
    
    # Test with 1 game, all markets
    python scripts/find_nfl_arb_opportunities.py --limit 1 --all-markets
    
    # Specific markets only
    python scripts/find_nfl_arb_opportunities.py --markets player_pass_yds,player_rush_yds
    
OUTPUT EXAMPLE:
    🎯 ARBITRAGE OPPORTUNITIES FOUND: 1
    
    🏈 Patrick Mahomes - 275.5 Passing Yards
       💰 PROFIT: 1.85%
       📊 Total Probability: 98.19% (< 100% = ARB!)
       
       📊 Odds & Implied Probabilities:
          Over 275.5: +140 (betmgm) → 41.67%
          Under 275.5: -130 (betonlineag) → 56.52%
       
       💵 Betting Strategy (for $100 total):
          Bet $42.44 on Over at betmgm
          Bet $57.56 on Under at betonlineag
          Guaranteed Profit: $1.85

OUTPUT FILES:
    - data/01_input/the-odds-api/nfl/all_markets/raw_YYYYMMDD_HHMMSS.csv (raw props with timestamp)
    - data/04_output/nfl/arbs/arb_output_YYYYMMDD_HHMMSS.csv (arb results with timestamp)
    
SETUP:
    1. Get API key from https://the-odds-api.com/
    2. Add to .env file: ODDS_API_KEY=your_key_here
    3. Run it!
    
API COST:
    - Events list: 1 credit
    - Per game odds: ~15 credits
    - 16 games: ~241 credits per week (Sunday slate)
    
NFL SCHEDULE:
    - Thursday Night: 1 game
    - Sunday: 13-14 games (main slate)
    - Monday Night: 1 game

Author: Myles Thomas
Date: 2025-12-06
"""

import argparse
import ssl
import urllib3
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add src to path for config_loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config_loader import get_data_path

# Fix SSL certificate issues on macOS (common with pyenv)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
API_BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'americanfootball_nfl'
REGIONS = 'us'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# Timezone Configuration
TIMEZONE = 'America/New_York'  # ET timezone

# NFL Player Prop Markets (verified common markets from The Odds API)
# Passing
PASSING_MARKETS = 'player_pass_yds,player_pass_tds,player_pass_completions,player_pass_attempts,player_pass_interceptions'
# Rushing
RUSHING_MARKETS = 'player_rush_yds,player_rush_attempts,player_rush_longest'
# Receiving
RECEIVING_MARKETS = 'player_receptions,player_reception_yds,player_reception_longest'
# Touchdowns
TD_MARKETS = 'player_anytime_td,player_1st_td,player_last_td,player_pass_rush_reception_tds'
# Kicking
KICKING_MARKETS = 'player_kicking_points,player_field_goals'
# Defense
DEFENSE_MARKETS = 'player_tackles_assists,player_sacks'

# All prop markets combined (used with --all-markets flag)
ALL_PROP_MARKETS = ','.join([
    PASSING_MARKETS,
    RUSHING_MARKETS,
    RECEIVING_MARKETS,
    TD_MARKETS,
    KICKING_MARKETS,
    DEFENSE_MARKETS
])

# Default Markets - just passing yards (use --all-markets for everything)
DEFAULT_MARKETS = 'player_pass_yds'

# Arbitrage Thresholds
MIN_ARB_PROFIT_PCT = 0.0  # Any profit > 0 is technically an arb
CLOSE_OPPORTUNITY_MIN = 0.98  # 98% probability
CLOSE_OPPORTUNITY_MAX = 1.00  # 100% probability

# Betting Configuration
BASE_WAGER_AMOUNT = 100  # Default total stake for bet calculations ($)

# Output Configuration
OUTPUT_ARB_DIR = 'data/04_output/nfl/arbs'  # Arb results
OUTPUT_RAW_DIR = 'data/01_input/the-odds-api/nfl/all_markets'  # Raw props
DEFAULT_TOTAL_STAKE = 100.0
SAMPLE_NON_ARBS_TO_SHOW = 5

# Market Display Names
MARKET_DISPLAY_NAMES = {
    # Passing
    'player_pass_yds': 'Passing Yards',
    'player_pass_tds': 'Passing TDs',
    'player_pass_completions': 'Completions',
    'player_pass_attempts': 'Pass Attempts',
    'player_pass_interceptions': 'Interceptions',
    # Rushing
    'player_rush_yds': 'Rushing Yards',
    'player_rush_attempts': 'Rush Attempts',
    'player_rush_longest': 'Longest Rush',
    # Receiving
    'player_receptions': 'Receptions',
    'player_reception_yds': 'Receiving Yards',
    'player_reception_longest': 'Longest Reception',
    # Touchdowns
    'player_anytime_td': 'Anytime TD',
    'player_1st_td': 'First TD',
    'player_last_td': 'Last TD',
    'player_pass_rush_reception_tds': 'Total TDs',
    # Kicking
    'player_kicking_points': 'Kicking Points',
    'player_field_goals': 'Field Goals',
    # Defense
    'player_tackles_assists': 'Tackles + Assists',
    'player_sacks': 'Sacks',
}

# ============================================================================


def american_to_probability(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_arb_profit(over_odds, under_odds):
    """
    Calculate if there's an arbitrage opportunity
    
    Returns:
        dict with 'is_arb', 'expected_profit_pct', 'over_prob', 'under_prob', 'total_prob'
    """
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    total_prob = over_prob + under_prob
    
    is_arb = total_prob < 1.0
    # Expected profit/loss percentage (positive = arb, negative = bookmaker edge)
    expected_profit_pct = ((1 / total_prob) - 1) * 100
    
    return {
        'is_arb': is_arb,
        'expected_profit_pct': expected_profit_pct,
        'over_prob': over_prob,
        'under_prob': under_prob,
        'total_prob': total_prob
    }


def calculate_bet_amounts(over_odds, under_odds, total_stake=100):
    """Calculate optimal bet amounts to guarantee equal profit"""
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    
    over_stake = (over_prob / (over_prob + under_prob)) * total_stake
    under_stake = (under_prob / (over_prob + under_prob)) * total_stake
    
    if over_odds > 0:
        over_return = over_stake * (1 + over_odds / 100)
    else:
        over_return = over_stake * (1 + 100 / abs(over_odds))
    
    if under_odds > 0:
        under_return = under_stake * (1 + under_odds / 100)
    else:
        under_return = under_stake * (1 + 100 / abs(under_odds))
    
    profit = min(over_return, under_return) - total_stake
    
    return {
        'over_stake': round(over_stake, 2),
        'under_stake': round(under_stake, 2),
        'over_return': round(over_return, 2),
        'under_return': round(under_return, 2),
        'guaranteed_profit': round(profit, 2)
    }


def get_nfl_events(api_key, days_ahead=7):
    """Get all NFL events for the upcoming week
    
    Args:
        api_key: API key
        days_ahead: Number of days ahead to look (default 7 for full week)
        
    Returns:
        tuple: (events_list, usage_dict with 'remaining' and 'used')
    """
    url = f"{API_BASE_URL}/sports/{SPORT}/events"
    
    params = {'apiKey': api_key}
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')
    
    usage = {'remaining': remaining, 'used': used}
    
    events = response.json()
    
    # Filter for games within the next N days
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    cutoff = now + timedelta(days=days_ahead)
    
    upcoming_events = []
    for event in events:
        event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        event_time_local = event_time_utc.astimezone(tz)
        
        if event_time_local <= cutoff:
            upcoming_events.append(event)
    
    return upcoming_events, usage


def get_todays_nfl_events(api_key):
    """Get NFL events happening today (in configured timezone)
    
    Returns:
        tuple: (events_list, usage_dict with 'remaining' and 'used')
    """
    url = f"{API_BASE_URL}/sports/{SPORT}/events"
    
    params = {'apiKey': api_key}
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')
    
    usage = {'remaining': remaining, 'used': used}
    
    events = response.json()
    
    # Filter for today's games in configured timezone
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()
    
    todays_events = []
    for event in events:
        event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        event_time_local = event_time_utc.astimezone(tz)
        
        if event_time_local.date() == today:
            todays_events.append(event)
    
    return todays_events, usage


def get_event_odds(api_key, event_id, markets=DEFAULT_MARKETS, historical_date=None):
    """Get odds for a specific event
    
    Args:
        api_key: API key
        event_id: Event ID
        markets: Markets to fetch
        historical_date: Optional ISO datetime string for historical odds
    
    Returns:
        tuple: (odds_data, usage_dict with 'remaining' and 'used')
    """
    if historical_date:
        url = f"{API_BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
        params = {
            'api_key': api_key,
            'date': historical_date,
            'regions': REGIONS,
            'markets': markets,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT
        }
    else:
        url = f"{API_BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
        params = {
            'apiKey': api_key,
            'regions': REGIONS,
            'markets': markets,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT
        }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')
    usage = {'remaining': remaining, 'used': used}
    
    data = response.json()
    
    if historical_date and 'data' in data:
        return data, usage
    
    return data, usage


def parse_event_props_to_df(event_data):
    """Parse event odds data into DataFrame"""
    props_list = []
    
    # Historical endpoint wraps data in 'data' key
    if 'data' in event_data:
        event_data = event_data['data']
    
    game_info = f"{event_data['away_team']} @ {event_data['home_team']}"
    game_time = event_data.get('commence_time')
    event_id = event_data.get('id')
    
    for bookmaker in event_data.get('bookmakers', []):
        bookmaker_name = bookmaker['key']
        
        for market in bookmaker.get('markets', []):
            market_key = market['key']
            
            player_line_props = {}
            
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', 'Unknown')
                line = outcome.get('point')
                odds = outcome.get('price')
                bet_type = outcome.get('name')
                
                key = (player, line)
                
                if key not in player_line_props:
                    player_line_props[key] = {
                        'event_id': event_id,
                        'player': player,
                        'market': market_key,
                        'line': line,
                        'bookmaker': bookmaker_name,
                        'game': game_info,
                        'game_time': game_time
                    }
                
                if bet_type == 'Over':
                    player_line_props[key]['over_odds'] = odds
                elif bet_type == 'Under':
                    player_line_props[key]['under_odds'] = odds
            
            props_list.extend(player_line_props.values())
    
    return pd.DataFrame(props_list)


def find_best_odds_per_player(props_df):
    """
    Find the best Over and Under odds for each market/player/line combination
    """
    if props_df.empty:
        return pd.DataFrame()
    
    best_odds = []
    
    for (market, player, line), group in props_df.groupby(['market', 'player', 'line']):
        over_bets = group[group['over_odds'].notna()].copy()
        if not over_bets.empty:
            best_over_idx = over_bets['over_odds'].idxmax()
            best_over = over_bets.loc[best_over_idx]
        else:
            best_over = None
        
        under_bets = group[group['under_odds'].notna()].copy()
        if not under_bets.empty:
            best_under_idx = under_bets['under_odds'].idxmax()
            best_under = under_bets.loc[best_under_idx]
        else:
            best_under = None
        
        if best_over is not None and best_under is not None:
            arb_calc = calculate_arb_profit(best_over['over_odds'], best_under['under_odds'])
            
            if arb_calc['is_arb']:
                bet_calc = calculate_bet_amounts(best_over['over_odds'], best_under['under_odds'], BASE_WAGER_AMOUNT)
                recommendation = f"Bet ${bet_calc['over_stake']:.2f} Over @ {best_over['bookmaker']}, ${bet_calc['under_stake']:.2f} Under @ {best_under['bookmaker']}"
                over_stake = bet_calc['over_stake']
                under_stake = bet_calc['under_stake']
                over_return = bet_calc['over_return']
                under_return = bet_calc['under_return']
                guaranteed_profit = bet_calc['guaranteed_profit']
            else:
                recommendation = "Don't bet - bookmaker has edge"
                over_stake = None
                under_stake = None
                over_return = None
                under_return = None
                guaranteed_profit = None
            
            best_odds.append({
                'player': player,
                'market': market,
                'line': line,
                'best_over_odds': best_over['over_odds'],
                'best_over_book': best_over['bookmaker'],
                'best_over_implied': arb_calc['over_prob'],
                'best_under_odds': best_under['under_odds'],
                'best_under_book': best_under['bookmaker'],
                'best_under_implied': arb_calc['under_prob'],
                'total_prob': arb_calc['total_prob'],
                'expected_profit_pct': arb_calc['expected_profit_pct'],
                'is_arb': arb_calc['is_arb'],
                'over_stake': over_stake,
                'under_stake': under_stake,
                'over_return': over_return,
                'under_return': under_return,
                'guaranteed_profit': guaranteed_profit,
                'total_wager': BASE_WAGER_AMOUNT if arb_calc['is_arb'] else None,
                'recommendation': recommendation,
                'game': group['game'].iloc[0],
                'game_time': group['game_time'].iloc[0],
                'num_bookmakers': len(group['bookmaker'].unique())
            })
    
    return pd.DataFrame(best_odds)


def display_arb_opportunities(arbs_df, min_profit_pct=0.0):
    """Display arbitrage opportunities"""
    if arbs_df.empty:
        print("❌ No arbitrage opportunities found")
        return
    
    arbs = arbs_df[arbs_df['expected_profit_pct'] > min_profit_pct].copy()
    arbs = arbs.sort_values('expected_profit_pct', ascending=False)
    
    if arbs.empty:
        print(f"❌ No arbitrage opportunities found with profit > {min_profit_pct}%")
        return
    
    print("\n" + "="*80)
    print(f"🎯 ARBITRAGE OPPORTUNITIES FOUND: {len(arbs)}")
    print("="*80 + "\n")
    
    for idx, row in arbs.iterrows():
        market_display = MARKET_DISPLAY_NAMES.get(row['market'], row['market'])
        print(f"🏈 {row['player']} - {row['line']} {market_display}")
        print(f"   Game: {row['game']}")
        print(f"   Time: {row['game_time']}")
        print(f"\n   💰 PROFIT: {row['expected_profit_pct']:.2f}%")
        print(f"   📊 Total Probability: {row['total_prob']:.2%} (< 100% = ARB!)")
        print(f"\n   📊 Odds & Implied Probabilities:")
        print(f"      Over {row['line']}: {row['best_over_odds']:+} ({row['best_over_book']}) → {row['best_over_implied']:.2%}")
        print(f"      Under {row['line']}: {row['best_under_odds']:+} ({row['best_under_book']}) → {row['best_under_implied']:.2%}")
        
        bet_calc = calculate_bet_amounts(row['best_over_odds'], row['best_under_odds'], BASE_WAGER_AMOUNT)
        print(f"\n   💵 Betting Strategy (for ${BASE_WAGER_AMOUNT} total):")
        print(f"      Bet ${bet_calc['over_stake']:.2f} on Over at {row['best_over_book']}")
        print(f"      Bet ${bet_calc['under_stake']:.2f} on Under at {row['best_under_book']}")
        print(f"      Guaranteed Profit: ${bet_calc['guaranteed_profit']:.2f}")
        
        print("\n" + "-"*80 + "\n")


def display_close_opportunities(all_odds_df, min_prob=0.98, max_prob=1.00, min_arb_profit=0.0):
    """Display close opportunities that might become arbs"""
    close = all_odds_df[
        (all_odds_df['total_prob'] >= min_prob) & 
        (all_odds_df['total_prob'] < max_prob) &
        (all_odds_df['expected_profit_pct'] <= min_arb_profit)
    ].copy()
    close = close.sort_values('total_prob')
    
    if close.empty:
        print(f"\n📊 No close opportunities found (between {min_prob:.1%} and {max_prob:.1%})")
        return
    
    print("\n" + "="*80)
    print(f"⚠️  CLOSE OPPORTUNITIES (might become arbs): {len(close)}")
    print("="*80 + "\n")
    
    for idx, row in close.head(10).iterrows():
        margin = (1 - row['total_prob']) * 100
        market_display = MARKET_DISPLAY_NAMES.get(row['market'], row['market'])
        
        print(f"🏈 {row['player']} - {row['line']} {market_display}")
        print(f"   Total Probability: {row['total_prob']:.2%} (margin to arb: {margin:.2f}%)")
        print(f"   Over {row['line']}: {row['best_over_odds']:+} ({row['best_over_book']}) → {row['best_over_implied']:.2%}")
        print(f"   Under {row['line']}: {row['best_under_odds']:+} ({row['best_under_book']}) → {row['best_under_implied']:.2%}")
        print()


def display_non_arbs(all_odds_df, sample_size=SAMPLE_NON_ARBS_TO_SHOW):
    """Display sample of NON-arb markets"""
    non_arbs = all_odds_df[all_odds_df['expected_profit_pct'] <= 0].copy()
    non_arbs = non_arbs.sort_values('total_prob')
    
    if non_arbs.empty:
        print("\n🎉 All markets are arbs! (Unlikely but congrats!)")
        return
    
    print("\n" + "="*80)
    print(f"📊 NON-ARB MARKETS (showing {min(sample_size, len(non_arbs))} of {len(non_arbs)})")
    print("="*80)
    print("These show how bookmakers maintain their edge\n")
    
    for idx, row in non_arbs.head(sample_size).iterrows():
        vig = (row['total_prob'] - 1.0) * 100
        market_display = MARKET_DISPLAY_NAMES.get(row['market'], row['market'])
        
        print(f"🏈 {row['player']} - {row['line']} {market_display}")
        print(f"   Total Probability: {row['total_prob']:.2%} (> 100% = Bookmaker edge)")
        print(f"   Expected loss: {row['expected_profit_pct']:.2f}%")
        print(f"   Bookmaker vig: {vig:.2f}%")
        print(f"   Over {row['line']}: {row['best_over_odds']:+} ({row['best_over_book']}) → {row['best_over_implied']:.2%}")
        print(f"   Under {row['line']}: {row['best_under_odds']:+} ({row['best_under_book']}) → {row['best_under_implied']:.2%}")
        print()


def run_tests():
    """Run unit tests for arbitrage calculations"""
    print("="*80)
    print("🧪 RUNNING UNIT TESTS")
    print("="*80 + "\n")
    
    # Test 1: Odds conversion
    print("TEST 1: Odds Conversion")
    test_cases = [
        (100, 0.5),
        (-110, 0.524),
        (150, 0.40),
        (-150, 0.60),
    ]
    
    for odds, expected in test_cases:
        result = american_to_probability(odds)
        status = '✅' if abs(result - expected) < 0.01 else '❌'
        print(f"  Odds {odds:+4}: {result:.3f} (expected ~{expected:.3f}) {status}")
    
    # Test 2: Arbitrage detection
    print("\nTEST 2: Arbitrage Detection")
    
    result = calculate_arb_profit(110, 105)
    assert result['is_arb'] == True
    print(f"  Clear arb (+110/+105): ✅ Detected, {result['expected_profit_pct']:.2f}% profit")
    
    result = calculate_arb_profit(-110, -110)
    assert result['is_arb'] == False
    assert result['expected_profit_pct'] < 0
    print(f"  No arb (-110/-110): ✅ Correctly rejected, {result['expected_profit_pct']:.2f}% expected loss")
    
    # Test 3: Bet sizing
    print("\nTEST 3: Bet Sizing")
    result = calculate_bet_amounts(110, 105, 100)
    profit = result['guaranteed_profit']
    assert profit > 0
    assert abs(result['over_stake'] + result['under_stake'] - 100) < 0.01
    print(f"  Optimal stakes: ${result['over_stake']:.2f} / ${result['under_stake']:.2f}")
    print(f"  Guaranteed profit: ${profit:.2f} ✅")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80 + "\n")


def run_demo():
    """Run demo showing the 3-step flow with mock data"""
    print("="*80)
    print("🏈 NFL PLAYER PROPS ARBITRAGE FINDER - DEMO")
    print(f"📅 {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("="*80 + "\n")
    
    print("STEP 1: Get this week's NFL events (ET timezone)")
    print("-"*80)
    print("Mock: 3 games found")
    print("  1. Kansas City Chiefs @ Buffalo Bills (Sunday 1:00 PM ET)")
    print("  2. Dallas Cowboys @ Philadelphia Eagles (Sunday 4:25 PM ET)")
    print("  3. San Francisco 49ers @ Seattle Seahawks (Sunday 8:20 PM ET)")
    
    print("\n" + "="*80 + "\n")
    print("STEP 2: For each game, fetch all player prop markets")
    print("-"*80)
    print("Mock: Fetched props for all games")
    print("  Game 1: 15 props (Mahomes pass yds, Kelce rec yds, etc.)")
    print("  Game 2: 12 props (Prescott pass yds, Lamb rec yds, etc.)")
    print("  Game 3: 10 props (Purdy pass yds, McCaffrey rush yds, etc.)")
    print("  Total: 37 prop bets from 5 bookmakers")
    
    print("\n" + "="*80 + "\n")
    print("STEP 3: Analyze each player-line for arbs")
    print("-"*80)
    
    mock_data = [
        ('Patrick Mahomes', 275.5, 115, -125, 'fanduel', 'betmgm', 'Passing Yards'),
        ('Travis Kelce', 65.5, -105, -110, 'fanduel', 'draftkings', 'Receiving Yards'),
    ]
    
    for player, line, over, under, over_book, under_book, market in mock_data:
        arb = calculate_arb_profit(over, under)
        print(f"\n🏈 {player} - {line} {market}")
        print(f"   Best Over: {over:+} ({over_book})")
        print(f"   Best Under: {under:+} ({under_book})")
        print(f"   Total probability: {arb['total_prob']:.2%}")
        if arb['is_arb']:
            print(f"   ✅ ARBITRAGE! Profit: {arb['expected_profit_pct']:.2f}%")
        else:
            vig = (arb['total_prob'] - 1.0) * 100
            print(f"   ❌ No arb. Bookmaker vig: {vig:.2f}%")
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("="*80)
    print("1. Event-by-event captures ALL lines (249.5, 274.5, 299.5)")
    print("   More lines = More opportunities to find arbs!")
    print()
    print("2. Each (player, line) is analyzed separately:")
    print("   ✅ Mahomes Over 275.5 vs Mahomes Under 275.5 (same line)")
    print("   ✅ Mahomes Over 299.5 vs Mahomes Under 299.5 (different market)")
    print("   ❌ NOT mixing: Mahomes Over 275.5 vs Mahomes Under 299.5")
    print("="*80 + "\n")


def main(markets=DEFAULT_MARKETS, limit=None, historical_date=None, historical_time="17:00:00", week_mode=False):
    """Main execution function
    
    Args:
        markets: Markets to fetch
        limit: Limit number of games (for testing)
        historical_date: Date object for historical mode (None for live)
        historical_time: Time string for historical snapshot in UTC
        week_mode: If True, fetch all games for the week instead of just today
    """
    is_historical = historical_date is not None
    
    if is_historical:
        display_date = historical_date
        from datetime import timezone
        time_obj = datetime.strptime(historical_time, "%H:%M:%S").time()
        target_datetime = datetime.combine(historical_date, time_obj, tzinfo=timezone.utc)
    else:
        tz = ZoneInfo(TIMEZONE)
        target_datetime = datetime.now(tz)
        display_date = target_datetime.date()
    
    print("="*80)
    print("🏈 NFL PROPS ARBITRAGE FINDER")
    if is_historical:
        et_tz = ZoneInfo(TIMEZONE)
        et_time = target_datetime.astimezone(et_tz)
        print(f"📅 HISTORICAL MODE: {display_date.strftime('%Y-%m-%d')} @ {historical_time} UTC ({et_time.strftime('%H:%M:%S %Z')})")
    else:
        print(f"📅 {target_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if week_mode:
        print("📆 WEEK MODE: Fetching all games for the week")
    print("="*80 + "\n")
    
    try:
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            print("❌ No valid API key found!")
            print("Get your API key at: https://the-odds-api.com/")
            print("Add it to .env file as: ODDS_API_KEY=your_key")
            return
        
        api_calls = []
        
        # Step 1: Get events
        if week_mode:
            print("🔍 Step 1: Fetching NFL events for the week...\n")
            events, initial_usage = get_nfl_events(api_key, days_ahead=7)
        else:
            print("🔍 Step 1: Fetching today's NFL events...\n")
            events, initial_usage = get_todays_nfl_events(api_key)
        
        api_calls.append({
            'call': 'events_list',
            'description': 'Fetch NFL events',
            'remaining': initial_usage['remaining'],
            'used_total': initial_usage['used']
        })
        
        print(f"💳 After events call - Remaining: {initial_usage['remaining']}, Used this period: {initial_usage['used']}\n")
        
        if not events:
            print("❌ No NFL games found")
            print("📝 Creating empty results files for dashboard...")
            
            empty_arb_df = pd.DataFrame(columns=[
                'player', 'market', 'line', 'best_over_odds', 'best_over_book', 
                'best_over_implied', 'best_under_odds', 'best_under_book', 
                'best_under_implied', 'total_prob', 'expected_profit_pct', 'is_arb',
                'over_stake', 'under_stake', 'over_return', 'under_return',
                'guaranteed_profit', 'total_wager', 'recommendation', 'game',
                'game_time', 'num_bookmakers'
            ])
            
            empty_raw_df = pd.DataFrame(columns=[
                'event_id', 'player', 'market', 'line', 'bookmaker', 'game',
                'game_time', 'over_odds', 'under_odds'
            ])
            
            if is_historical:
                timestamp = target_datetime.strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            arb_output_dir = Path(__file__).parent.parent / OUTPUT_ARB_DIR
            arb_output_dir.mkdir(exist_ok=True, parents=True)
            
            raw_output_dir = Path(__file__).parent.parent / OUTPUT_RAW_DIR
            raw_output_dir.mkdir(exist_ok=True, parents=True)
            
            output_file = arb_output_dir / f'arb_output_{timestamp}.csv'
            empty_arb_df.to_csv(output_file, index=False)
            print(f"💾 Empty arb results saved to: {output_file}")
            
            raw_output_file = raw_output_dir / f'raw_{timestamp}.csv'
            empty_raw_df.to_csv(raw_output_file, index=False)
            print(f"💾 Empty raw props saved to: {raw_output_file}")
            
            return
        
        if limit and limit < len(events):
            print(f"⚠️  Limiting to first {limit} game(s) for testing\n")
            events = events[:limit]
        
        print(f"✅ Found {len(events)} NFL games:\n")
        for i, event in enumerate(events, 1):
            event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            event_time_local = event_time_utc.astimezone(ZoneInfo(TIMEZONE))
            print(f"   {i}. {event['away_team']} @ {event['home_team']}")
            print(f"      {event_time_local.strftime('%A %b %d, %I:%M %p ET')}")
        
        print("\n" + "-"*80 + "\n")
        
        # Step 2: Get odds for each event
        markets_list = markets.split(',')
        markets_display = ', '.join([MARKET_DISPLAY_NAMES.get(m.strip(), m.strip()) for m in markets_list])
        print(f"🔍 Step 2: Fetching {markets_display} props for each game...\n")
        
        all_props = []
        
        for i, event in enumerate(events, 1):
            print(f"📥 Game {i}/{len(events)}: {event['away_team']} @ {event['home_team']}")
            
            try:
                historical_iso = None
                if is_historical:
                    historical_iso = target_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                event_odds, usage = get_event_odds(api_key, event['id'], markets=markets, historical_date=historical_iso)
                event_props_df = parse_event_props_to_df(event_odds)
                
                api_calls.append({
                    'call': f'event_odds_{i}',
                    'description': f"{event['away_team']} @ {event['home_team']}",
                    'remaining': usage['remaining'],
                    'used_total': usage['used']
                })
                
                if not event_props_df.empty:
                    all_props.append(event_props_df)
                    print(f"   ✅ Found {len(event_props_df)} prop bets")
                else:
                    print(f"   ⚠️  No props available")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
        
        if not all_props:
            print("❌ No prop data available")
            return
        
        props_df = pd.concat(all_props, ignore_index=True)
        
        print("-"*80 + "\n")
        print(f"✅ Total prop bets: {len(props_df)}")
        print(f"   Bookmakers: {len(props_df['bookmaker'].unique())}")
        print(f"   Markets: {len(props_df['market'].unique())}")
        print(f"   Players: {len(props_df['player'].unique())}")
        print(f"   Unique (market, player, line) combinations: {len(props_df.groupby(['market', 'player', 'line']))}")
        
        # Step 3: Find arbitrage
        print("\n" + "="*80)
        print("🔍 Step 3: Analyzing for arbitrage opportunities...")
        print("="*80 + "\n")
        
        best_odds_df = find_best_odds_per_player(props_df)
        
        if best_odds_df.empty:
            print("❌ Could not find any complete odds pairs")
            return
        
        print(f"✅ Analyzed {len(best_odds_df)} (market, player, line) combinations\n")
        
        display_arb_opportunities(best_odds_df, min_profit_pct=MIN_ARB_PROFIT_PCT)
        display_close_opportunities(best_odds_df, min_prob=CLOSE_OPPORTUNITY_MIN, max_prob=CLOSE_OPPORTUNITY_MAX, min_arb_profit=MIN_ARB_PROFIT_PCT)
        display_non_arbs(best_odds_df)
        
        # Summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"Total (market, player, line) combinations: {len(best_odds_df)}")
        print(f"Arbs (any profit > 0): {len(best_odds_df[best_odds_df['expected_profit_pct'] > 0])}")
        print(f"Non-arbs (bookmaker edge): {len(best_odds_df[best_odds_df['expected_profit_pct'] <= 0])}")
        print(f"Close to arb (98-100%): {len(best_odds_df[(best_odds_df['total_prob'] >= 0.98) & (best_odds_df['total_prob'] < 1.00) & (best_odds_df['expected_profit_pct'] <= 0)])}")
        
        if len(best_odds_df) > 0:
            avg_prob = best_odds_df['total_prob'].mean()
            min_prob = best_odds_df['total_prob'].min()
            max_profit = best_odds_df['expected_profit_pct'].max()
            
            print(f"Avg total probability: {avg_prob:.2%} (avg bookmaker edge: {(avg_prob-1)*100:.2f}%)")
            print(f"Lowest total probability: {min_prob:.2%}", end="")
            if min_prob < 1.0:
                print(f" → Best arb profit: {(1/min_prob - 1)*100:.2f}%")
            else:
                print(f" (no arbs, need < 100%)")
            
            if max_profit > 0:
                print(f"Best arb found: {max_profit:.2f}% profit")
        
        # Save results
        if is_historical:
            timestamp = target_datetime.strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        arb_output_dir = Path(__file__).parent.parent / OUTPUT_ARB_DIR
        arb_output_dir.mkdir(exist_ok=True, parents=True)
        
        raw_output_dir = Path(__file__).parent.parent / OUTPUT_RAW_DIR
        raw_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Sort by expected_profit_pct descending and save
        output_file = arb_output_dir / f'arb_output_{timestamp}.csv'
        best_odds_df = best_odds_df.sort_values('expected_profit_pct', ascending=False)
        best_odds_df.to_csv(output_file, index=False)
        print(f"\n💾 Arb results saved to: {output_file}")
        
        raw_output_file = raw_output_dir / f'raw_{timestamp}.csv'
        props_df.to_csv(raw_output_file, index=False)
        print(f"💾 Raw props saved to: {raw_output_file}")
        
        # Credit usage summary
        print("\n" + "="*80)
        print("💳 API CREDIT USAGE")
        print("="*80)
        
        if len(api_calls) > 0 and api_calls[0]['remaining'] != 'unknown':
            print(f"\n{'Call':<20} {'Description':<40} {'Remaining':<12} {'Credits Used'}")
            print("-" * 90)
            
            for i, call in enumerate(api_calls):
                remaining = int(float(call['remaining']))
                
                if i == 0:
                    credits_used = "N/A (first)"
                else:
                    prev_remaining = int(float(api_calls[i-1]['remaining']))
                    credits_used = prev_remaining - remaining
                    credits_used = f"{credits_used:,}" if credits_used > 0 else "0 (free!)"
                
                desc = call['description'][:38] + '..' if len(call['description']) > 40 else call['description']
                print(f"{call['call']:<20} {desc:<40} {call['remaining']:>10}   {credits_used}")
            
            print("-" * 90)
            
            first_remaining = int(float(api_calls[0]['remaining']))
            last_remaining = int(float(api_calls[-1]['remaining']))
            total_used = first_remaining - last_remaining
            
            print(f"\n📊 Summary:")
            print(f"   Started with (after 1st call): {first_remaining:,} requests")
            print(f"   Ending with: {last_remaining:,} requests")
            print(f"   Credits used in this run: {total_used:,} requests")
            
            if len(api_calls) > 1:
                per_game_avg = total_used / (len(api_calls) - 1)
                print(f"   📊 Avg credits per game: {per_game_avg:.1f}")
        
        print("\n" + "="*80)
        print(f"📄 ARB FILE: {output_file}")
        print("="*80)
        
        print("\n" + "="*80)
        print(f"📄 RAW PROPS FILE: {raw_output_file}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def valid_date(date_str):
    """Validate and parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: '{date_str}'. Expected format: YYYY-MM-DD.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NFL Props Arbitrage Finder')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--demo', action='store_true', help='Run demo with mock data')
    parser.add_argument('--markets', default=DEFAULT_MARKETS, 
                        help=f'Markets to fetch (default: {DEFAULT_MARKETS}). '
                             f'Use --all-markets for all available markets.')
    parser.add_argument('--all-markets', action='store_true', dest='all_markets',
                        help='Fetch ALL available prop markets (passing, rushing, receiving, TDs, kicking, defense)')
    parser.add_argument('--limit', type=int, help='Limit to first N games (automatically enables week mode)')
    parser.add_argument('--week', action='store_true', 
                        help='Fetch all games for the week (not just today)')
    
    # Historical arguments (for future use)
    parser.add_argument('--historical', action='store_true', 
                        help='Run in historical mode (requires --start and --end)')
    parser.add_argument('--start', type=valid_date, 
                        help='Start date for historical mode (YYYY-MM-DD)')
    parser.add_argument('--end', type=valid_date, 
                        help='End date for historical mode (YYYY-MM-DD)')
    parser.add_argument('--time', default="17:00:00",
                        help='Time for historical snapshot in UTC (HH:MM:SS)')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.demo:
        run_demo()
    elif args.historical:
        if not args.start or not args.end:
            print("❌ Error: --historical mode requires both --start and --end dates")
            sys.exit(1)
        # Historical mode not fully implemented yet for NFL
        print("⚠️  Historical mode for NFL coming soon!")
    else:
        # Use all markets if --all-markets flag is set
        markets = ALL_PROP_MARKETS if args.all_markets else args.markets
        # If --limit is set, automatically enable week mode (so we can find games to limit)
        week_mode = args.week or (args.limit is not None)
        main(markets=markets, limit=args.limit, week_mode=week_mode)

