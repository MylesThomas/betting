"""
NBA 3-Point Props Arbitrage Finder

WHAT IT DOES:
    Finds arbitrage opportunities in NBA player 3-point prop markets.
    Uses event-by-event fetching to capture ALL available lines (3.5, 4.5, 5.5, etc.)
    
ARBITRAGE EXPLAINED:
    When you can bet both sides (SAME LINE) and guarantee profit regardless of outcome.
    Example:
        - Bet Over 4.5 threes at +115 (BookA)
        - Bet Under 4.5 threes at +105 (BookB)
        - Over probability: 46.5%, Under probability: 48.8%
        - Combined probability: 95.3% (< 100% = guaranteed profit!)
    
    IMPORTANT: We only compare Over/Under for the SAME player + SAME line.
               (e.g., "Curry Over 4.5" vs "Curry Under 4.5" - NOT mixing 3.5 and 4.5)
        
HOW IT WORKS (3 steps):
    1. Get all NBA games today (ET timezone)
    2. For each game, fetch all 3-point prop markets
    3. For each (player, line) pair, find best Over/Under across bookmakers
    
USAGE:
    # Run tests (no API calls)
    python scripts/find_arb_opportunities.py --test
    
    # See demo flow (no API calls)
    python scripts/find_arb_opportunities.py --demo
    
    # Test with 1 game (uses ~15 credits)
    python scripts/find_arb_opportunities.py --limit 1
    
    # Find real arbs - all games (uses ~150 credits for 10 games)
    python scripts/find_arb_opportunities.py
    
    # Check other markets
    python scripts/find_arb_opportunities.py --markets player_points
    
    # Combine markets
    python scripts/find_arb_opportunities.py --markets player_threes,player_points,player_rebounds
    
OUTPUT EXAMPLE:
    🎯 ARBITRAGE OPPORTUNITIES FOUND: 1
    
    🏀 Michael Porter Jr - 3.5 Three-Pointers
       💰 PROFIT: 1.85%
       📊 Total Probability: 98.19% (< 100% = ARB!)
       
       📊 Odds & Implied Probabilities:
          Over 3.5: +140 (betmgm) → 41.67%
          Under 3.5: -130 (betonlineag) → 56.52%
       
       💵 Betting Strategy (for $100 total):
          Bet $42.44 on Over at betmgm
          Bet $57.56 on Under at betonlineag
          Guaranteed Profit: $1.85

OUTPUT FILES:
    - data/04_output/arbs/arb_3pt_props_YYYYMMDD.csv (main results, overwrites daily)
    - data/04_output/arbs/raw_3pt_props_YYYYMMDD_HHMMSS.csv (raw data with timestamp)
    
SETUP:
    1. Get API key from https://the-odds-api.com/
    2. Add to .env file: ODDS_API_KEY=your_key_here
    3. Run it!
    
API COST:
    - Events list: 1 credit
    - Per game odds: ~15 credits
    - 10 games: ~151 credits per day
    - 30 days: ~4,530 credits per month
    - 20,000 credits/month plan = 77% budget remaining
    
WHY EVENT-BY-EVENT?:
    - Bulk endpoint: Returns ~10 markets (main lines only)
    - Event-by-event: Returns ~40 markets (all alternate lines)
    - Alternate lines often have softer pricing
    - 4x more opportunities to find arbs
    - Worth the extra credits!
    
AUTOMATION:
    # Add to crontab for daily 9 AM runs
    0 9 * * * /path/to/scripts/run_daily_arb_check.sh >> /path/to/logs/arb.log 2>&1

OTHER MARKETS:
    Markets available:
    - player_points
    - player_rebounds
    - player_assists
    - player_threes (default)
    - player_blocks
    - player_steals
    - player_double_double
    - player_triple_double
    - player_points_rebounds_assists
    
    Use --markets flag to change or combine multiple markets:
    python scripts/find_arb_opportunities.py --markets player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists

SSL CERTIFICATE FIX:
    SSL verification is disabled by default (common issue with pyenv on macOS).
    This is safe for development with trusted APIs like The Odds API.
    See api_setup/fixing_ssl.md for details.

TODO / FUTURE ENHANCEMENTS:
    - Cross-line arbitrage (e.g., Over 3.5 + Under 4.5 for same player)
      Requires probability modeling to price the gap between lines
    - Multi-market parlays (e.g., points + rebounds combo)
    - Real-time monitoring with WebSocket feeds
    - Telegram/SMS notifications when arbs found
    - Track historical arb frequency by bookmaker/player/time
"""

import argparse
import ssl
import urllib3
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add src to path for config_loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config_loader import get_data_path

# Fix SSL certificate issues on macOS (common with pyenv)
# See api_setup/fixing_ssl.md for details
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
API_BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'basketball_nba'
REGIONS = 'us'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# Timezone Configuration
TIMEZONE = 'America/New_York'  # ET timezone for "today's games"

# Default Markets
DEFAULT_MARKETS = 'player_threes'
# All available prop markets (verified 2024-11-21 via scripts/test_available_markets.py)
ALL_PROP_MARKETS = 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists'
COMMON_PROP_MARKETS = 'player_threes,player_points,player_rebounds,player_assists'  # Most common markets

# Arbitrage Thresholds
MIN_ARB_PROFIT_PCT = 0.0  # Any profit > 0 is technically an arb
CLOSE_OPPORTUNITY_MIN = 0.98  # 98% probability
CLOSE_OPPORTUNITY_MAX = 1.00  # 100% probability

# Betting Configuration
BASE_WAGER_AMOUNT = 100  # Default total stake for bet calculations ($)

# Output Configuration
OUTPUT_DIR = str(get_data_path('output_arbs'))  # data/04_output/arbs
DEFAULT_TOTAL_STAKE = 100.0  # Default total amount to wager for bet sizing recommendations
SAMPLE_NON_ARBS_TO_SHOW = 5

# Market Display Names
MARKET_DISPLAY_NAMES = {
    'player_threes': 'Threes',
    'player_points': 'Points',
    'player_rebounds': 'Rebounds',
    'player_assists': 'Assists',
    'player_blocks': 'Blocks',
    'player_steals': 'Steals',
    'player_double_double': 'Double-Double',
    'player_triple_double': 'Triple-Double',
    'player_points_rebounds_assists': 'Pts+Reb+Ast'
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


def get_todays_nba_events(api_key):
    """Get all NBA events happening today (in configured timezone)
    
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


def get_event_odds(api_key, event_id, markets=DEFAULT_MARKETS):
    """Get odds for a specific event
    
    Returns:
        tuple: (odds_data, usage_dict with 'remaining' and 'used')
    """
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
    
    return response.json(), usage


def parse_event_props_to_df(event_data):
    """Parse event odds data into DataFrame"""
    props_list = []
    
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
    
    IMPORTANT: Groups by (market, player, line) tuple - only compares same market & line!
    Example: "player_threes, Curry, 4.5" Over vs "player_threes, Curry, 4.5" Under
    NOT mixing:
        - Different lines: 3.5 vs 4.5
        - Different markets: player_threes vs player_points
    
    For cross-line arbs (e.g., Over 3.5 + Under 4.5), we'd need probability
    modeling to price the gap - not implemented yet.
    """
    if props_df.empty:
        return pd.DataFrame()
    
    best_odds = []
    
    # Group by (market, player, line) - each market/line is analyzed separately
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
            
            # Calculate bet amounts if it's an arb
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
        over_prob = american_to_probability(row['best_over_odds'])
        under_prob = american_to_probability(row['best_under_odds'])
        
        market_display = MARKET_DISPLAY_NAMES.get(row['market'], row['market'])
        print(f"🏀 {row['player']} - {row['line']} {market_display}")
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
    """Display close opportunities that might become arbs (excludes actual arbs)"""
    close = all_odds_df[
        (all_odds_df['total_prob'] >= min_prob) & 
        (all_odds_df['total_prob'] < max_prob) &
        (all_odds_df['expected_profit_pct'] <= min_arb_profit)  # Exclude actual arbs
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
        
        print(f"🏀 {row['player']} - {row['line']} {market_display}")
        print(f"   Total Probability: {row['total_prob']:.2%} (margin to arb: {margin:.2f}%)")
        print(f"   Over {row['line']}: {row['best_over_odds']:+} ({row['best_over_book']}) → {row['best_over_implied']:.2%}")
        print(f"   Under {row['line']}: {row['best_under_odds']:+} ({row['best_under_book']}) → {row['best_under_implied']:.2%}")
        print()


def display_non_arbs(all_odds_df, sample_size=SAMPLE_NON_ARBS_TO_SHOW):
    """Display sample of NON-arb markets (bookmaker has edge)"""
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
        
        print(f"🏀 {row['player']} - {row['line']} {market_display}")
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
    
    # Clear arb
    result = calculate_arb_profit(110, 105)
    assert result['is_arb'] == True
    print(f"  Clear arb (+110/+105): ✅ Detected, {result['expected_profit_pct']:.2f}% profit")
    
    # No arb
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
    print("🏀 NBA 3-POINT PROPS ARBITRAGE FINDER - DEMO")
    print(f"📅 {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("="*80 + "\n")
    
    print("STEP 1: Get today's NBA events (ET timezone)")
    print("-"*80)
    print("Mock: 2 games found")
    print("  1. Boston Celtics @ Brooklyn Nets (06:30 PM ET)")
    print("  2. Golden State Warriors @ Los Angeles Lakers (10:00 PM ET)")
    
    print("\n" + "="*80 + "\n")
    print("STEP 2: For each game, fetch all 3-point prop markets")
    print("-"*80)
    print("Mock: Fetched props for both games")
    print("  Game 1: 7 props (Tatum 2 lines, Brown 1 line)")
    print("  Game 2: 7 props (Curry 2 lines, LeBron 1 line)")
    print("  Total: 14 prop bets from 3 bookmakers")
    
    print("\n" + "="*80 + "\n")
    print("STEP 3: Analyze each player-line for arbs")
    print("-"*80)
    
    mock_data = [
        ('Jayson Tatum', 3.5, 115, -125, 'fanduel', 'betmgm'),
        ('Stephen Curry', 4.5, -105, -110, 'fanduel', 'draftkings'),
    ]
    
    for player, line, over, under, over_book, under_book in mock_data:
        arb = calculate_arb_profit(over, under)
        print(f"\n🏀 {player} - {line} Threes")
        print(f"   Best Over: {over:+} ({over_book})")
        print(f"   Best Under: {under:+} ({under_book})")
        print(f"   Total probability: {arb['total_prob']:.2%}")
        if arb['is_arb']:
            print(f"   ✅ ARBITRAGE! Profit: {arb['profit_pct']:.2f}%")
        else:
            vig = (arb['total_prob'] - 1.0) * 100
            print(f"   ❌ No arb. Bookmaker vig: {vig:.2f}%")
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("="*80)
    print("1. Event-by-event captures ALL lines (3.5, 4.5, 5.5)")
    print("   More lines = More opportunities to find arbs!")
    print()
    print("2. Each (player, line) is analyzed separately:")
    print("   ✅ Tatum Over 3.5 vs Tatum Under 3.5 (same line)")
    print("   ✅ Tatum Over 4.5 vs Tatum Under 4.5 (different market)")
    print("   ❌ NOT mixing: Tatum Over 3.5 vs Tatum Under 4.5")
    print("      (Would need probability modeling for cross-line arbs)")
    print("="*80 + "\n")


def main(markets=DEFAULT_MARKETS, limit=None):
    """Main execution function"""
    print("="*80)
    print("🏀 NBA PROPS ARBITRAGE FINDER")
    print(f"📅 {datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("="*80 + "\n")
    
    try:
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            print("❌ No valid API key found!")
            print("Get your API key at: https://the-odds-api.com/")
            print("Add it to .env file as: ODDS_API_KEY=your_key")
            return
        
        # Track API usage for each call
        api_calls = []
        
        # Step 1: Get today's events
        print("🔍 Step 1: Fetching today's NBA events...\n")
        todays_events, initial_usage = get_todays_nba_events(api_key)
        
        api_calls.append({
            'call': 'events_list',
            'description': 'Fetch today\'s NBA events',
            'remaining': initial_usage['remaining'],
            'used_total': initial_usage['used']
        })
        
        print(f"💳 After events call - Remaining: {initial_usage['remaining']}, Used this period: {initial_usage['used']}\n")
        
        if not todays_events:
            print("❌ No NBA games found for today")
            print("⚠️  This might indicate an issue - exiting with warning code")
            import sys
            sys.exit(1)  # Exit with error so you get notified
        
        # Limit games if requested
        if limit and limit < len(todays_events):
            print(f"⚠️  Limiting to first {limit} game(s) for testing\n")
            todays_events = todays_events[:limit]
        
        print(f"✅ Found {len(todays_events)} NBA games today:\n")
        for i, event in enumerate(todays_events, 1):
            event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            event_time_local = event_time_utc.astimezone(ZoneInfo(TIMEZONE))
            print(f"   {i}. {event['away_team']} @ {event['home_team']}")
            print(f"      {event_time_local.strftime('%I:%M %p ET')}")
        
        print("\n" + "-"*80 + "\n")
        
        # Step 2: Get odds for each event
        markets_list = markets.split(',')
        markets_display = ', '.join([MARKET_DISPLAY_NAMES.get(m.strip(), m.strip()) for m in markets_list])
        print(f"🔍 Step 2: Fetching {markets_display} props for each game...\n")
        
        all_props = []
        
        for i, event in enumerate(todays_events, 1):
            print(f"📥 Game {i}/{len(todays_events)}: {event['away_team']} @ {event['home_team']}")
            
            try:
                event_odds, usage = get_event_odds(api_key, event['id'], markets=markets)
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
        today = datetime.now().strftime('%Y%m%d')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent.parent / OUTPUT_DIR
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine filename based on markets
        markets_str = markets.replace(',', '_').replace('player_', '')
        
        # Main results file with just date
        output_file = output_dir / f'arb_{markets_str}_{today}.csv'
        best_odds_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Raw props with full timestamp
        raw_output_file = output_dir / f'raw_{markets_str}_{timestamp}.csv'
        props_df.to_csv(raw_output_file, index=False)
        print(f"💾 Raw props saved to: {raw_output_file}")
        
        # Credit usage summary
        print("\n" + "="*80)
        print("💳 API CREDIT USAGE")
        print("="*80)
        
        if len(api_calls) > 0 and api_calls[0]['remaining'] != 'unknown':
            print(f"\n{'Call':<20} {'Description':<40} {'Remaining':<12} {'Credits Used'}")
            print("-" * 90)
            
            # Calculate credits used for each call
            for i, call in enumerate(api_calls):
                remaining = int(call['remaining'])
                
                if i == 0:
                    # First call - we don't know what we started with, so show as first call
                    credits_used = "N/A (first)"
                else:
                    prev_remaining = int(api_calls[i-1]['remaining'])
                    credits_used = prev_remaining - remaining
                    credits_used = f"{credits_used:,}" if credits_used > 0 else "0 (free!)"
                
                # Truncate description if too long
                desc = call['description'][:38] + '..' if len(call['description']) > 40 else call['description']
                print(f"{call['call']:<20} {desc:<40} {call['remaining']:>10}   {credits_used}")
            
            print("-" * 90)
            
            # Summary
            first_remaining = int(api_calls[0]['remaining'])
            last_remaining = int(api_calls[-1]['remaining'])
            total_used = first_remaining - last_remaining
            
            print(f"\n📊 Summary:")
            print(f"   Started with (after 1st call): {first_remaining:,} requests")
            print(f"   Ending with: {last_remaining:,} requests")
            print(f"   Credits used in this run: {total_used:,} requests")
            
            # Calculate per-game credit usage (excluding events list call)
            if len(api_calls) > 1:
                per_game_avg = total_used / (len(api_calls) - 1)
                print(f"   📊 Avg credits per game: {per_game_avg:.1f}")
                print(f"   💡 Note: The Odds API charges ~1 credit per market per game")
            
            if total_used == 0:
                print(f"   💡 No credits used - current odds may be free!")
        else:
            print("Unable to parse credit usage (API didn't return usage headers)")
        
        # Print arb file at the bottom for easy reference
        print("\n" + "="*80)
        print(f"📄 ARB FILE: {output_file}")
        print("="*80)

        # Print raw props file at the bottom for easy reference
        print("\n" + "="*80)
        print(f"📄 RAW PROPS FILE: {raw_output_file}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)  # Exit with error code so Lambda knows it failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA Props Arbitrage Finder')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--demo', action='store_true', help='Run demo with mock data')
    parser.add_argument('--markets', default=DEFAULT_MARKETS, 
                        help=f'Markets to fetch (default: {DEFAULT_MARKETS}). '
                             f'Available: {ALL_PROP_MARKETS}')
    parser.add_argument('--limit', type=int, help='Limit to first N games (for testing)')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.demo:
        run_demo()
    else:
        main(markets=args.markets, limit=args.limit)
