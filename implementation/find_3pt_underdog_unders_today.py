"""
Find Underdog-Unders Betting Opportunities for Today's NBA Games

STRATEGY: Underdog-Unders (Fade Hot Streaks)
============================================
Based on backtest: python backtesting/20251121_nba_3pt_prop_miss_streaks_24_25.py --underdog-unders

Bet UNDER with POSITIVE ODDS after hot streaks:
    1. Player is on a hot streak (hit their 3pt prop 2-10 consecutive games)
    2. Market is still pricing the UNDER as an underdog (positive odds like +110, +135)
    3. We bet UNDER expecting regression to the mean

Theory:
    - Public is reactive to streaks and overreacts
    - Hot players get overvalued (under becomes underdog)
    - Regression to the mean creates profitable fade opportunities
    - Basketball is a game of variance - hot streaks cool down

Backtest Performance (2024-25 season):
    Streak Length | Bets | Win% | ROI    | Verdict
    ------------- | ---- | ---- | ------ | --------
    2 games       | 241  | 46.5 | +5.82% | ✅ BEST
    3 games       | 155  | 48.4 | +5.40% | ✅ BEST  
    4 games       |  89  | 44.9 | -1.67% | ❌ Skip
    5 games       |  64  | 50.0 | +1.75% | ✅ OK

    BEST PERFORMANCE: 2-3 game hot streaks with positive under odds (+5-6% ROI)

QUICK START:
============
1. Build current season game logs (2025-26):
   $ python scripts/build_season_game_logs.py

2. Fetch and build historical props for season:
   $ python scripts/fetch_and_build_season_props.py --season 2025-26 --market player_threes
   (Creates: data/03_intermediate/consensus_props_2025_26_player_threes.csv)

3. Fetch today's live props:
   $ python scripts/fetch_historical_props.py
   (Creates: data/01_input/the-odds-api/nba/historical_props/props_YYYY-MM-DD_player_threes.csv)

4. Find today's opportunities:
   $ python implementation/find_underdog_unders_today.py

USAGE EXAMPLES:
==============
# Basic usage - find today's opportunities
python implementation/find_underdog_unders_today.py

# Show season statistics first (how often this pattern has occurred)
python implementation/find_underdog_unders_today.py --show-season-stats

# Focus on best-performing streaks (2-3 games)
python implementation/find_underdog_unders_today.py --min-streak 2 --max-streak 3

# Season analysis only (no today's bets)
python implementation/find_underdog_unders_today.py --season-stats-only

# Analyze previous season
python implementation/find_underdog_unders_today.py --season 2024-25 --season-stats-only

WHAT YOU'LL SEE:
===============
1. Season Statistics
   - How many players are currently on hot streaks (2-10 games)
   - Distribution by streak length
   - Estimated opportunities with positive under odds

2. Today's Opportunities (if any)
   - Player name and game details
   - Bet recommendation: UNDER X.X threes at +XXX odds
   - Hot streak details (length, recent performance, 3PA trend)
   - Confidence level (HIGH = 2-3 game streaks, best ROI)
   - Backtest stats for that specific streak length

3. CSV Export
   - Saved to: data/04_output/todays_plays/underdog_unders_YYYYMMDD.csv
   - Includes all bet details for tracking

EXAMPLE OUTPUT:
==============
🎯 FOUND 2 BETTING OPPORTUNITY(IES) FOR TODAY

OPPORTUNITY #1 - HIGH CONFIDENCE
=================================
🏀 Player:         Stephen Curry
🎯 Game:           Golden State Warriors @ Los Angeles Lakers
🕐 Time:           2024-11-24 22:00:00
📚 Bookmakers:     8 offering this market

📊 BET:            UNDER 4.5 three-pointers
💰 Best Odds:      +125 (under) vs -150 (over)
📈 Implied Prob:   44.4%
💵 Bet Amount:     $80.00 to win $100.00

🔥 HOT STREAK:     3 consecutive games HITTING OVER
   Recent dates:   2024-11-19, 2024-11-21, 2024-11-23
   Recent 3PM:     [5, 6, 5]
   Recent 3PA:     [10, 12, 11]
   Avg in streak:  5.3 makes on 11.0 attempts

📝 STRATEGY:       Bet UNDER - fade the hot streak, expect regression to mean
📊 BACKTEST:       3-game streaks: 155 bets, 48.4% win, +5.40% ROI ✅

DAILY WORKFLOW:
==============
Morning (before games):
    1. Update season game logs: python scripts/build_season_game_logs.py
    2. Fetch today's props: python api_setup/odds_api_efficient.py
    3. Find opportunities: python implementation/find_underdog_unders_today.py
    4. Review opportunities (check confidence, verify odds, check injuries)
    5. Place bets on your sportsbook
    6. Track results in spreadsheet

After games:
    - Update game logs to include tonight's results

IMPORTANT NOTES:
===============
⚠️ Risk Management:
   - Bet sizing: Script recommends betting to win $100 per play
   - Adjust based on your bankroll (1-3% per bet is standard)
   - Example: $5000 bankroll = $50-150 per bet
   - Only bet what you can afford to lose
   - Track EVERY bet to validate strategy
   - Need 50-100 bets to properly assess performance

📊 Expected Frequency:
   - 0-5 opportunities per day is normal
   - More on busy nights (10+ games)
   - Fewer on quiet nights (2-3 games)
   - Zero opportunities on some days is EXPECTED
   
   Why? Three specific conditions must align:
   1. Player on 2-10 game hot streak (rare)
   2. Under has positive odds (market mispricing)
   3. Player is playing TODAY

✅ Best Practices:
   - Focus on 2-3 game streaks (proven +5-6% ROI)
   - Verify odds haven't moved since fetch
   - Check injury reports before betting
   - Consider opponent's 3pt defense
   - Be patient - quality over quantity

❌ Avoid:
   - Chasing losses with bigger bets
   - Betting every opportunity blindly
   - Ignoring confidence levels (HIGH > MEDIUM > LOW)
   - Betting without verifying current data
   - Betting more than 1-3% of bankroll per play

📈 Performance Tracking:
   Track these in a spreadsheet:
   - Date, Player, Line, Odds, Streak Length
   - Actual Result (3PM made)
   - Won/Lost, Profit/Loss
   - Running Win Rate and ROI by streak length
   - Compare to backtest benchmarks

TROUBLESHOOTING:
===============
Error: "Season data not found"
    → Run: python scripts/build_season_game_logs.py --season 2025-26

Error: "No props found"
    → Run: python api_setup/odds_api_efficient.py

"No opportunities found today"
    → This is NORMAL! The strategy requires specific conditions.
    → Try again on busier game nights or wait for market to align.
    → Check back in a few hours as odds can change.

THEORY & VALIDATION:
===================
This strategy exploits two market inefficiencies:
1. Public overreaction to hot streaks (recency bias)
2. Market mispricing when under is still underdog despite hot streak

The backtest validates:
- 396 total bets over 2024-25 season
- 155 bets at 2-3 game streaks: +5.4-5.8% ROI
- Positive edge vs. implied probability
- Consistent performance across different months

Key Assumptions:
- Players regress to their mean after hot streaks
- Market is slow to adjust under odds despite hot streak
- Positive under odds indicate mispricing opportunity
- 2-3 game streaks are optimal (longer streaks = smaller edge)

Limitations:
- Past performance doesn't guarantee future results
- Market may adapt and eliminate edge over time
- Sample size is significant but not infinite
- Individual game variance is high (need volume)

RELATED SCRIPTS:
===============
- Backtest: backtesting/20251121_nba_3pt_prop_miss_streaks.py --underdog-unders
- Build Data: scripts/build_season_game_logs.py
- Fetch Props: api_setup/odds_api_efficient.py
- Odds Utils: src/odds_utils.py

Author: Myles Thomas
Date: 2024-11-24
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import binom
from math import floor, ceil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CURRENT_NBA_SEASON
from src.odds_utils import calculate_bet_amount, calculate_profit, odds_to_implied_probability
from src.player_name_utils import normalize_player_name


# ============================================================================
# CONFIG
# ============================================================================

# Bet Sizing (same as backtest)
TARGET_WIN = 100  # Target profit per bet ($100)

# Odds Filtering
MAX_ODDS_THRESHOLD = -300  # Exclude extreme odds (bookmaker traps)

# Streak Range (based on backtest results)
DEFAULT_MIN_STREAK = 2  # 2-3 streaks perform best
DEFAULT_MAX_STREAK = 5  # Don't go too long

# Prediction Model (from backtest)
LEAGUE_AVG_3PT_PCT = 0.35  # NBA league average ~35%

# Backtest Results from 2024-25 Season (Underdog-Unders Strategy)
# Format: streak_length -> {'all': (bets, win%, roi), 'filtered': (bets, win%, roi)}
BACKTEST_RESULTS = {
    2: {'all': (241, 46.5, 5.82), 'filtered': None},  # No filtered data yet
    3: {'all': (155, 48.4, 5.40), 'filtered': None},
    4: {'all': (89, 44.9, -1.67), 'filtered': None},
    5: {'all': (64, 50.0, 1.75), 'filtered': None},
    6: {'all': (49, 46.9, -0.85), 'filtered': None},
    7: {'all': (35, 45.7, -3.21), 'filtered': None},
    8: {'all': (25, 44.0, -5.12), 'filtered': None},
    9: {'all': (18, 38.9, -12.34), 'filtered': None},
    10: {'all': (12, 41.7, -8.67), 'filtered': None},
}

# Data Paths
SEASON_GAME_LOGS_DIR = PROJECT_ROOT / 'data' / '01_input' / 'nba_api' / 'season_game_logs'
INTERMEDIATE_DIR = PROJECT_ROOT / 'data' / '03_intermediate'
TODAYS_PROPS_DIR = PROJECT_ROOT / 'data' / '04_output' / 'arbs'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_season_game_logs(season=CURRENT_NBA_SEASON):
    """
    Load all player game logs for the current season.
    
    Uses individual player files from build_season_game_logs.py output.
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        DataFrame with all player game logs
    """
    season_dir = SEASON_GAME_LOGS_DIR / season.replace('-', '_')
    
    if not season_dir.exists():
        raise FileNotFoundError(
            f"Season data not found: {season_dir}\n"
            f"Run: python scripts/build_season_game_logs.py --season {season}"
        )
    
    # Load individual player files
    all_files = list(season_dir.glob('*.csv'))
    
    if not all_files:
        raise FileNotFoundError(f"No player game logs found in {season_dir}")
    
    print(f"Loading {len(all_files)} player game logs from {season}...")
    
    dfs = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Error loading {file_path.name}: {e}")
    
    if not dfs:
        raise ValueError("No valid player data found")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('date', ascending=False).reset_index(drop=True)
    
    print(f"  ✅ Loaded {len(combined):,} game logs for {combined['player'].nunique()} players")
    print(f"  📅 Date range: {combined['date'].min().strftime('%Y-%m-%d')} to {combined['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    return combined


def load_historical_props(season=CURRENT_NBA_SEASON, market='player_threes'):
    """
    Load historical props data for accurate streak detection.
    
    This is crucial - we need to know what the actual line was in each game.
    Loads from: data/03_intermediate/consensus_props_{season}_{market}.csv
    
    Args:
        season: Season string (e.g., '2025-26')
        market: Market name (e.g., 'player_threes')
    
    Returns:
        DataFrame with historical props (player, date, line, odds)
    """
    # Build filename with season and market
    season_clean = season.replace('-', '_')
    props_file = INTERMEDIATE_DIR / f'consensus_props_{season_clean}_{market}.csv'
    
    if not props_file.exists():
        print(f"  ⚠️  Historical props not found: {props_file}")
        print(f"     Run: python scripts/fetch_and_build_season_props.py --season {season} --market {market}")
        print(f"     This is OK - will estimate lines from player averages")
        return None
    
    print(f"Loading historical props from {season}...")
    df = pd.read_csv(props_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✅ Loaded {len(df):,} historical props")
    print(f"  📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    return df


def load_todays_props():
    """
    Load today's props from the most recent arb analysis.
    
    Loads from: data/04_output/arbs/arb_*_YYYYMMDD.csv
    
    Filters for: market == 'player_threes'
    
    This file already has best odds calculated per (player, line, market)!
    
    Returns:
        DataFrame with today's props (player, line, best_over_odds, best_over_book, 
                                      best_under_odds, best_under_book, game, game_time)
    """
    # Find most recent arb file
    arb_files = sorted(TODAYS_PROPS_DIR.glob('arb_*_[0-9]*.csv'))
    
    if not arb_files:
        raise FileNotFoundError(
            f"No arb files found in {TODAYS_PROPS_DIR}\n"
            f"These are created by the daily Lambda run (scripts/find_arb_opportunities.py)"
        )
    
    latest_file = arb_files[-1]
    
    print(f"Loading today's props...")
    print(f"  File: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    
    # Filter to only player_threes market
    df = df[df['market'] == 'player_threes'].copy()
    
    if len(df) == 0:
        raise ValueError(
            f"No player_threes props found in {latest_file.name}\n"
            f"The file may not contain 3-point props"
        )
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'best_over_odds': 'over_odds',
        'best_over_book': 'over_book',
        'best_under_odds': 'under_odds',
        'best_under_book': 'under_book',
    })
    
    print(f"  ✅ Loaded {len(df)} three-point props (player+line combinations)")
    print(f"  📊 {df['player'].nunique()} unique players")
    print()
    
    return df


# ============================================================================
# HOT STREAK DETECTION
# ============================================================================

def detect_hot_streak_with_props(player_name, game_logs, historical_props=None, 
                                 min_streak=2, max_streak=10, debug=False):
    """
    Detect if a player is on a hot streak (hitting their 3pt prop consecutively).
    
    Uses actual historical prop lines if available, otherwise estimates from performance.
    
    Args:
        player_name: Player name (normalized)
        game_logs: Season game logs DataFrame
        historical_props: Historical props DataFrame (optional)
        min_streak: Minimum consecutive hits
        max_streak: Maximum streak to check
        debug: If True, print detailed logging for this player
    
    Returns:
        Dict with streak info or None if no qualifying streak
    """
    if debug:
        print("\n" + "="*100)
        print(f"🔍 SANITY CHECK: Detailed Analysis for {player_name}")
        print("="*100)
    
    # Get player's games (sorted chronologically, most recent last)
    player_games = game_logs[game_logs['player'] == player_name].copy()
    
    if debug:
        print(f"\n1️⃣ PLAYER GAME HISTORY:")
        print(f"   Total games in game logs: {len(player_games)}")
    
    if len(player_games) < min_streak:
        if debug:
            print(f"   ❌ Not enough games with data ({len(player_games)} < {min_streak} required)")
            print(f"   → Player may have played but no game log data available")
        return None
    
    player_games = player_games.sort_values('date').reset_index(drop=True)
    
    # Determine hit/miss for each game
    if historical_props is not None:
        # Use actual prop lines
        player_props = historical_props[historical_props['player'] == player_name].copy()
        
        if debug:
            print(f"\n2️⃣ HISTORICAL PROPS DATA:")
            print(f"   Games with prop lines: {len(player_props)}")
        
        # Merge props with games
        merged = player_games.merge(
            player_props[['date', 'consensus_line']],
            on='date',
            how='left'
        )
        
        # For games with props, check if hit
        merged['has_prop'] = merged['consensus_line'].notna()
        merged['hit'] = merged['threes_made'] >= merged['consensus_line']
        
        if debug:
            print(f"   Games matched: {merged['has_prop'].sum()}")
    else:
        # Estimate line from season average
        season_avg = player_games['threes_made'].mean()
        typical_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        estimated_line = min(typical_lines, key=lambda x: abs(x - season_avg))
        
        if debug:
            print(f"\n2️⃣ ESTIMATED PROPS (no historical data):")
            print(f"   Season average: {season_avg:.1f} threes")
            print(f"   Estimated line: {estimated_line}")
        
        merged = player_games.copy()
        merged['consensus_line'] = estimated_line
        merged['has_prop'] = True
        merged['hit'] = merged['threes_made'] >= estimated_line
    
    if debug:
        print(f"\n3️⃣ GAME-BY-GAME BREAKDOWN (Last 10 games, most recent first):")
        print(f"   {'Date':<12} | {'3PM':<4} | {'Line':<5} | {'Hit?':<5} | {'3PA':<4}")
        print(f"   {'-'*50}")
        
        recent_games = merged.tail(10).sort_values('date', ascending=False)
        for _, game in recent_games.iterrows():
            if game['has_prop']:
                hit_str = "✅ YES" if game['hit'] else "❌ NO"
                print(f"   {game['date'].strftime('%Y-%m-%d'):<12} | {int(game['threes_made']):<4} | {game['consensus_line']:<5} | {hit_str:<5} | {int(game['threes_attempted']):<4}")
            else:
                print(f"   {game['date'].strftime('%Y-%m-%d'):<12} | {int(game['threes_made']):<4} | {'N/A':<5} | {'N/A':<5} | {int(game['threes_attempted']):<4}")
        
        # Count N/A lines
        n_prop_games = merged['has_prop'].sum()
        n_total_games = len(merged)
        if n_total_games > n_prop_games:
            print(f"\n   ℹ️  Note: 'N/A' means no historical prop line available for that date")
            print(f"      Only {n_prop_games}/{n_total_games} games have prop data. Streak detection uses ONLY games with props.")
    
    # Count consecutive hits from end (most recent games)
    current_streak = 0
    streak_games = []
    
    if debug:
        print(f"\n4️⃣ STREAK DETECTION (counting backwards from most recent):")
    
    for idx in reversed(merged.index):
        game = merged.loc[idx]
        
        # Only count games with props
        if not game['has_prop']:
            if debug and current_streak == 0:
                print(f"   Game {game['date'].strftime('%Y-%m-%d')}: No prop data, skipping")
            continue
        
        if game['hit']:
            current_streak += 1
            streak_games.insert(0, game)
            
            if debug:
                print(f"   ✅ Game {current_streak}: {game['date'].strftime('%Y-%m-%d')} - Made {int(game['threes_made'])} >= {game['consensus_line']} (HIT)")
            
            if current_streak >= max_streak:
                if debug:
                    print(f"   ⚠️  Reached max streak length ({max_streak}), stopping")
                break
        else:
            if debug:
                print(f"   ❌ Game: {game['date'].strftime('%Y-%m-%d')} - Made {int(game['threes_made'])} < {game['consensus_line']} (MISS) - Streak ends")
            break
    
    # Check if qualifies
    if debug:
        print(f"\n5️⃣ FINAL RESULT:")
        print(f"   Current streak: {current_streak} games")
        print(f"   Required range: {min_streak}-{max_streak} games")
    
    if current_streak >= min_streak:
        # Calculate prediction using mean 3PA from streak * league average
        streak_3pa = [g['threes_attempted'] for g in streak_games]
        mean_3pa = np.mean(streak_3pa)
        predicted_3pa = mean_3pa
        predicted_3pm = predicted_3pa * LEAGUE_AVG_3PT_PCT
        
        result = {
            'player': player_name,
            'streak_length': current_streak,
            'line': streak_games[-1]['consensus_line'] if len(streak_games) > 0 else None,
            'recent_makes': [int(g['threes_made']) for g in streak_games],
            'recent_attempts': [int(g['threes_attempted']) for g in streak_games],
            'recent_dates': [g['date'].strftime('%Y-%m-%d') for g in streak_games],
            'avg_3pm': np.mean([g['threes_made'] for g in streak_games]),
            'avg_3pa': mean_3pa,
            'predicted_3pa': predicted_3pa,
            'predicted_3pm': predicted_3pm,
        }
        
        if debug:
            print(f"   ✅ QUALIFIES! {current_streak}-game hot streak")
            print(f"   Streak games: {', '.join(result['recent_dates'])}")
            print(f"   Makes: {result['recent_makes']}")
            print(f"   Attempts: {result['recent_attempts']}")
            print(f"   Average: {result['avg_3pm']:.1f} makes on {result['avg_3pa']:.1f} attempts")
            print(f"\n   📊 PREDICTION MODEL:")
            print(f"   Mean 3PA from streak: {mean_3pa:.1f}")
            print(f"   Predicted 3PM (@ {LEAGUE_AVG_3PT_PCT:.0%}): {predicted_3pm:.2f}")
            print("="*100 + "\n")
        
        return result
    else:
        if debug:
            print(f"   ❌ DOES NOT QUALIFY (streak {current_streak} < minimum {min_streak})")
            print("="*100 + "\n")
        return None


# ============================================================================
# SEASON STATISTICS
# ============================================================================

def analyze_season_patterns(game_logs, historical_props=None, 
                           min_streak=2, max_streak=10):
    """
    Analyze how often the underdog-unders pattern has occurred this season.
    
    This gives context: "This pattern happened X times so far this season"
    
    Args:
        game_logs: Season game logs
        historical_props: Historical props (optional)
        min_streak: Min streak length
        max_streak: Max streak length
    
    Returns:
        Dict with season statistics
    """
    print("="*100)
    print(f"ANALYZING SEASON PATTERNS: {CURRENT_NBA_SEASON}")
    print("="*100)
    print()
    
    # Get unique players
    players = game_logs['player'].unique()
    
    print(f"Analyzing {len(players)} players for hot streak patterns...")
    print()
    
    # Track patterns by streak length
    patterns_by_streak = defaultdict(list)
    
    for player in players:
        # Detect current hot streak
        streak_info = detect_hot_streak_with_props(
            player, game_logs, historical_props,
            min_streak, max_streak
        )
        
        if streak_info:
            streak_len = streak_info['streak_length']
            patterns_by_streak[streak_len].append(streak_info)
    
    # Display statistics
    print("HOT STREAK PATTERNS FOUND:")
    print("-" * 100)
    print(f"{'Streak Length':<15} | {'Players':<10} | {'Details':<70}")
    print("-" * 100)
    
    total_patterns = 0
    for streak_len in sorted(patterns_by_streak.keys()):
        players_list = patterns_by_streak[streak_len]
        count = len(players_list)
        total_patterns += count
        
        # Show top 3 players
        top_players = [p['player'] for p in players_list[:3]]
        details = ', '.join(top_players)
        if count > 3:
            details += f" (+{count-3} more)"
        
        print(f"{streak_len} games{' '*7} | {count:<10} | {details:<70}")
    
    print("-" * 100)
    print(f"{'TOTAL':<15} | {total_patterns:<10} | Players currently on hot streaks")
    print()
    
    # Calculate how many would have positive under odds (estimate 30% based on backtest)
    estimated_underdog_unders = int(total_patterns * 0.30)
    
    print("📊 KEY INSIGHTS:")
    print(f"   • {total_patterns} players currently on hot streaks (2+ consecutive hits)")
    print(f"   • ~{estimated_underdog_unders} estimated to have positive under odds (30% of streaks)")
    print(f"   • Best performance: 2-3 game streaks (+5-6% ROI in backtest)")
    print()
    
    return {
        'patterns_by_streak': patterns_by_streak,
        'total_patterns': total_patterns,
        'estimated_opportunities': estimated_underdog_unders,
    }


# ============================================================================
# TODAY'S OPPORTUNITIES
# ============================================================================

def find_todays_opportunities(todays_props, game_logs, historical_props=None,
                             min_streak=2, max_streak=10):
    """
    Find underdog-under opportunities in today's games.
    
    Args:
        todays_props: Today's props DataFrame
        game_logs: Season game logs
        historical_props: Historical props (optional)
        min_streak: Min streak length
        max_streak: Max streak length
    
    Returns:
        DataFrame with opportunities
    """
    print("="*100)
    print("SCANNING TODAY'S GAMES FOR UNDERDOG-UNDER OPPORTUNITIES")
    print("="*100)
    print()
    
    opportunities = []
    first_player_checked = False  # Flag to debug first player
    
    # Each row in arb file is already a unique (player, line) combination with best odds calculated
    for idx, prop in todays_props.iterrows():
        player_raw = prop['player']
        player = normalize_player_name(player_raw)
        
        consensus_line = prop['line']
        under_best_odds = int(prop['under_odds'])
        over_best_odds = int(prop['over_odds'])
        under_best_book = prop['under_book']
        over_best_book = prop['over_book']
        
        # Game info
        game = prop['game']
        game_time = prop['game_time']
        num_bookmakers = int(prop['num_bookmakers'])
        
        # Show sanity check for first player (regardless of odds)
        debug_this_player = not first_player_checked
        if debug_this_player:
            first_player_checked = True
            print(f"🔍 Running detailed sanity check for first player: {player}")
            print(f"   Today's line: {consensus_line}")
            print(f"   Under odds: {under_best_odds:+d} ({under_best_book})")
            print(f"   Over odds: {over_best_odds:+d} ({over_best_book})")
            print(f"   Under is underdog? {under_best_odds > 0}")
            if under_best_book != over_best_book:
                print(f"   ⚠️  Note: Best odds from different bookmakers")
        
        # Check qualifications
        has_underdog_under = under_best_odds > 0
        
        # Check if player is on hot streak
        streak_info = detect_hot_streak_with_props(
            player, game_logs, historical_props,
            min_streak, max_streak, debug=debug_this_player
        )
        
        has_hot_streak = streak_info is not None
        qualifies = has_underdog_under and has_hot_streak
        
        # Debug output
        if debug_this_player:
            if not has_underdog_under:
                print(f"   ❌ Under is NOT underdog (odds {under_best_odds} <= 0)")
            if has_underdog_under and not has_hot_streak:
                print(f"   ✅ Under is underdog (+{under_best_odds})")
                print(f"   ❌ No qualifying hot streak ({min_streak}-{max_streak} games)")
            print()
        
        # Calculate disqualify reason
        if not has_underdog_under:
            disqualify_reason = 'Under not underdog'
        elif not has_hot_streak:
            disqualify_reason = 'No hot streak'
        else:
            disqualify_reason = None
        
        # Calculate all fields (only calculate signals if qualifies)
        if qualifies:
            # Calculate bet details
            implied_prob = odds_to_implied_probability(under_best_odds) * 100
            bet_amount = calculate_bet_amount(under_best_odds, TARGET_WIN)
            
            # Get streak data
            streak_len = streak_info['streak_length']
            streak_makes = streak_info['recent_makes']
            streak_attempts = streak_info['recent_attempts']
            streak_dates = streak_info['recent_dates']
            backtest_stats = BACKTEST_RESULTS.get(streak_len, None)
            streak_avg = streak_info['avg_3pm']
            predicted_3pm = streak_info['predicted_3pm']
            mean_3pa = streak_info['avg_3pa']
            
            # Signal 1: Regression - Is their hot streak avg ABOVE the line?
            regression_signal = streak_avg > consensus_line
            overperformance = streak_avg - consensus_line
            
            # Signal 2: Prediction model - Mean 3PA * league avg 3PT%
            prediction_signal = predicted_3pm < consensus_line
            
            # Signal 3: Binomial probability - P(makes < line) > break-even probability
            n_attempts = ceil(mean_3pa)  # Binomial requires integer n - round UP
            threshold = floor(consensus_line)
            prob_under = binom.cdf(threshold, n=n_attempts, p=LEAGUE_AVG_3PT_PCT)
            breakeven_prob = odds_to_implied_probability(under_best_odds)
            binomial_signal = prob_under > breakeven_prob
            
            # Total signals agreeing (0-3)
            signal_count = sum([regression_signal, prediction_signal, binomial_signal])
        else:
            # Doesn't qualify - all signals false, all values None/0
            implied_prob = None
            bet_amount = None
            streak_len = None
            streak_makes = None
            streak_attempts = None
            streak_dates = None
            backtest_stats = None
            streak_avg = None
            predicted_3pm = None
            mean_3pa = None
            regression_signal = False
            overperformance = None
            prediction_signal = False
            prob_under = None
            breakeven_prob = None
            binomial_signal = False
            signal_count = 0
        
        # Debug output for first player
        if debug_this_player and qualifies:
            is_play_debug = signal_count == 3
            
            print(f"{'✅ PLAY' if is_play_debug else '⚠️  NOT RECOMMENDED'}: {player} ({signal_count}/3 signals)")
            print(f"   ✅ Under is underdog (+{under_best_odds})")
            print(f"   ✅ On {streak_len}-game hot streak")
            print(f"   📊 BET: UNDER {consensus_line} at +{under_best_odds}")
            print(f"   📉 Regression: {'✅' if regression_signal else '❌'} (avg {streak_avg:.1f} vs line {consensus_line})")
            print(f"   🎯 Expected Value: {'✅' if prediction_signal else '❌'} (predicted {predicted_3pm:.2f} vs line {consensus_line})")
            print(f"   📊 Probability: {'✅' if binomial_signal else '❌'} (P(under)={prob_under*100:.1f}% vs {breakeven_prob*100:.1f}% breakeven)")
            print()
        
        # Build opportunity record (same structure for all rows)
        # All fields already set correctly in calculation block above
        opportunities.append({
            'player': player,
            'player_display': player_raw,
            'game': game,
            'game_time': game_time,
            'line': consensus_line,
            'under_odds': under_best_odds,
            'over_odds': over_best_odds,
            'under_book': under_best_book,
            'over_book': over_best_book,
            'num_bookmakers': num_bookmakers,
            'has_underdog_under': has_underdog_under,
            'has_hot_streak': has_hot_streak,
            'qualifies': qualifies,
            'disqualify_reason': disqualify_reason,
            'implied_probability': implied_prob,
            'bet_amount': bet_amount,
            'potential_profit': TARGET_WIN if qualifies else None,
            'streak_length': streak_len,
            'streak_makes': streak_makes,
            'streak_attempts': streak_attempts,
            'streak_dates': streak_dates,
            'avg_3pm_streak': streak_avg,
            'avg_3pa_streak': mean_3pa,
            'predicted_3pm': predicted_3pm,
            'prediction_signal': prediction_signal,
            'overperformance': overperformance,
            'regression_signal': regression_signal,
            'prob_under': prob_under,
            'breakeven_prob': breakeven_prob,
            'binomial_signal': binomial_signal,
            'signal_count': signal_count,
            'backtest_stats': backtest_stats,
        })
    
    df = pd.DataFrame(opportunities)
    
    if len(df) > 0:
        # Sort by qualifies, signal count, then backtest ROI
        def get_roi(backtest_stats):
            if pd.isna(backtest_stats):
                return -999
            if backtest_stats and isinstance(backtest_stats, dict) and 'all' in backtest_stats:
                return backtest_stats['all'][2]  # ROI is 3rd element
            return -999  # Put unknown at bottom
        
        df['roi'] = df['backtest_stats'].apply(get_roi)
        
        # Sort: qualifies first, then signal count, then ROI
        df = df.sort_values(['qualifies', 'signal_count', 'roi'], ascending=[False, False, False])
        df = df.drop('roi', axis=1)
    
    total = len(df)
    qualifies_count = df['qualifies'].sum()
    
    print(f"✅ Processed {total} player/line combinations")
    print(f"   {qualifies_count} qualify as underdog-unders (will be shown in detail)")
    print(f"   {total - qualifies_count} don't qualify (saved to CSV only)")
    print()
    
    return df


def display_opportunities(opportunities):
    """
    Display betting opportunities with full details.
    Only shows opportunities that QUALIFY (underdog + hot streak).
    
    Args:
        opportunities: DataFrame with opportunities
    """
    # Filter to only show qualifying opportunities
    if 'qualifies' in opportunities.columns:
        qualifying_opps = opportunities[opportunities['qualifies'] == True].copy()
    else:
        qualifying_opps = opportunities.copy()
    
    if len(qualifying_opps) == 0:
        print("="*100)
        print("NO QUALIFYING OPPORTUNITIES FOUND TODAY")
        print("="*100)
        print()
        print("No players met the basic criteria:")
        print("  ✓ On a hot streak (2-10 consecutive hits)")
        print("  ✓ UNDER odds are positive (underdog)")
        print("  ✓ Playing in today's games")
        print()
        print(f"Total player/line combinations checked: {len(opportunities)}")
        print("💡 All combinations saved to CSV for analysis")
        print()
        return
    
    opportunities = qualifying_opps
    
    # Show summary - only recommend plays with all 3 signals
    total_opps = len(opportunities)
    plays = len(opportunities[opportunities['signal_count'] == 3])
    signal_2 = len(opportunities[opportunities['signal_count'] == 2])
    signal_1 = len(opportunities[opportunities['signal_count'] == 1])
    signal_0 = len(opportunities[opportunities['signal_count'] == 0])
    
    print("="*100)
    print(f"🎯 FOUND {total_opps} UNDERDOG-UNDER OPPORTUNITIES")
    print("="*100)
    print(f"   ✅ {plays} PLAYS (all 3 signals agree)")
    print(f"   ⚠️  {signal_2 + signal_1 + signal_0} NOT RECOMMENDED (< 3 signals)")
    print()
    print(f"   Signal breakdown:")
    print(f"     3/3 signals: {plays} (PLAY)")
    print(f"     2/3 signals: {signal_2}")
    print(f"     1/3 signals: {signal_1}")
    print(f"     0/3 signals: {signal_0}")
    print()
    
    for idx, opp in opportunities.iterrows():
        # Only PLAY if all 3 signals agree
        is_play = opp['signal_count'] == 3
        play_emoji = "✅ PLAY" if is_play else "⚠️  NOT RECOMMENDED"
        signal_label = f"{opp['signal_count']}/3 signals"
        
        print("="*100)
        print(f"OPPORTUNITY #{idx + 1} - {play_emoji} ({signal_label})")
        print("="*100)
        print()
        
        print(f"🏀 Player:         {opp['player_display']}")
        print(f"🎯 Game:           {opp['game']}")
        print(f"🕐 Time:           {opp['game_time']}")
        print(f"📚 Bookmakers:     {opp['num_bookmakers']} offering this market")
        print()
        
        print(f"📊 BET:            UNDER {opp['line']:.1f} three-pointers")
        print(f"💰 Best Odds:")
        print(f"   Under: {opp['under_odds']:+d} ({opp['under_book']})")
        print(f"   Over:  {opp['over_odds']:+d} ({opp['over_book']})")
        if opp['under_book'] != opp['over_book']:
            print(f"   ⚠️  Note: Best odds from different bookmakers for this line")
        print(f"📈 Implied Prob:   {opp['implied_probability']:.1f}%")
        print(f"💵 Bet Amount:     ${opp['bet_amount']:.2f} to win ${opp['potential_profit']:.2f}")
        print()
        
        print(f"🔥 HOT STREAK:     {opp['streak_length']} consecutive games HITTING OVER")
        print(f"   Recent dates:   {', '.join(opp['streak_dates'][:5])}")
        print(f"   Recent 3PM:     {opp['streak_makes']}")
        print(f"   Recent 3PA:     {opp['streak_attempts']}")
        print(f"   Avg in streak:  {opp['avg_3pm_streak']:.1f} makes on {opp['avg_3pa_streak']:.1f} attempts")
        print()
        
        print(f"🎯 SIGNALS ({opp['signal_count']}/3):")
        print()
        
        # Signal 1: Regression
        regression_emoji = "✅" if opp['regression_signal'] else "❌"
        if opp['regression_signal']:
            print(f"   {regression_emoji} REGRESSION: Averaging {opp['avg_3pm_streak']:.1f} vs line {opp['line']:.1f} (+{opp['overperformance']:.1f} above)")
            print(f"                   Player overperforming during streak → Due for regression, but we are not seeing it yet")
        else:
            print(f"   {regression_emoji} REGRESSION: Averaging {opp['avg_3pm_streak']:.1f} vs line {opp['line']:.1f} ({opp['overperformance']:.1f})")
            print(f"                   Player barely hitting line → Less regression signal")
        print()
        
        # Signal 2: Expected Value
        prediction_emoji = "✅" if opp['prediction_signal'] else "❌"
        if opp['prediction_signal']:
            print(f"   {prediction_emoji} EXPECTED VALUE: Predicts {opp['predicted_3pm']:.2f} < {opp['line']:.1f} (UNDER)")
        else:
            print(f"   {prediction_emoji} EXPECTED VALUE: Predicts {opp['predicted_3pm']:.2f} ≥ {opp['line']:.1f} (OVER)")
        print(f"                      {opp['avg_3pa_streak']:.1f} attempts × 35% = {opp['predicted_3pm']:.2f} makes")
        print()
        
        # Signal 3: Binomial Probability
        binomial_emoji = "✅" if opp['binomial_signal'] else "❌"
        n_attempts_display = ceil(opp['avg_3pa_streak'])
        if opp['binomial_signal']:
            print(f"   {binomial_emoji} PROBABILITY: P(under) = {opp['prob_under']*100:.1f}% > {opp['breakeven_prob']*100:.1f}% break-even ✅")
            print(f"                 Binomial(n={n_attempts_display} from {opp['avg_3pa_streak']:.1f} mean, p=35%) → P(≤{int(opp['line'])} makes)")
            print(f"                 Edge: {(opp['prob_under'] - opp['breakeven_prob'])*100:+.1f}% probability advantage")
        else:
            print(f"   {binomial_emoji} PROBABILITY: P(under) = {opp['prob_under']*100:.1f}% < {opp['breakeven_prob']*100:.1f}% break-even ❌")
            print(f"                 Binomial(n={n_attempts_display} from {opp['avg_3pa_streak']:.1f} mean, p=35%) → P(≤{int(opp['line'])} makes)")
            print(f"                 Edge: {(opp['prob_under'] - opp['breakeven_prob'])*100:+.1f}% probability disadvantage")
        print()
        
        # Recommendation based on signal count
        is_play = opp['signal_count'] == 3
        if is_play:
            print(f"✅ RECOMMENDATION:  BET UNDER (all 3 signals agree)")
        else:
            print(f"⚠️  RECOMMENDATION:  PASS ({opp['signal_count']}/3 signals - need all 3 to play)")
        
        print(f"📝 STRATEGY:       Bet UNDER - fade the hot streak, expect regression to mean")
        
        # Show backtest stats with/without filter
        if opp['backtest_stats']:
            all_stats = opp['backtest_stats']['all']
            filtered_stats = opp['backtest_stats'].get('filtered')
            
            print(f"📊 BACKTEST (2024-25 Season):")
            print(f"   {opp['streak_length']}-game streaks (ALL underdog-unders):")
            print(f"      {all_stats[0]} bets | {all_stats[1]:.1f}% win | {all_stats[2]:+.2f}% ROI {'✅' if all_stats[2] > 0 else '❌'}")
            
            if filtered_stats:
                print(f"   {opp['streak_length']}-game streaks (PREDICTION-ALIGNED ONLY):")
                print(f"      {filtered_stats[0]} bets | {filtered_stats[1]:.1f}% win | {filtered_stats[2]:+.2f}% ROI {'✅' if filtered_stats[2] > 0 else '❌'}")
                roi_diff = filtered_stats[2] - all_stats[2]
                print(f"      Filter impact: {roi_diff:+.2f}% ROI improvement")
            else:
                print(f"   {opp['streak_length']}-game streaks (PREDICTION-ALIGNED): No data yet")
        else:
            print(f"📊 BACKTEST:       {opp['streak_length']}-game streaks - No historical data")
        
        print()
    
    print("="*100)
    print()


def save_opportunities(opportunities, output_dir='data/04_output/todays_plays'):
    """
    Save ALL opportunities to CSV with a 'play' column.
    
    Args:
        opportunities: DataFrame with opportunities
        output_dir: Output directory
    """
    if len(opportunities) == 0:
        print("💾 No opportunities to save")
        print()
        return
    
    # Add 'play' column - TRUE only if qualifies AND all 3 signals agree
    opportunities_to_save = opportunities.copy()
    opportunities_to_save['play'] = opportunities_to_save['qualifies'] & (opportunities_to_save['signal_count'] == 3)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV with today's date
    today_str = datetime.now().strftime('%Y%m%d')
    output_file = Path(output_dir) / f'3pt_underdog_unders_{today_str}.csv'
    
    opportunities_to_save.to_csv(output_file, index=False)
    
    # Summary
    total = len(opportunities_to_save)
    qualifies = opportunities_to_save['qualifies'].sum()
    plays = (opportunities_to_save['play'] == True).sum()
    
    print(f"💾 Saved {total} player/line combinations to: {output_file}")
    print(f"   ✅ {qualifies} qualify as underdog-unders (underdog + hot streak)")
    print(f"   ✅✅ {plays} marked as PLAY (qualifies + all 3 signals agree)")
    print(f"   ⚠️  {total - qualifies} don't qualify (filtered out)")
    print()


# ============================================================================
# SANITY CHECK
# ============================================================================

def sanity_check_player_line(player_name, line_value, historical_date=None, season=CURRENT_NBA_SEASON, 
                             min_streak=DEFAULT_MIN_STREAK, max_streak=DEFAULT_MAX_STREAK):
    """
    Detailed sanity check for a specific player+line combination.
    
    Shows:
    - Player's game history
    - Hot streak detection
    - All 3 signals with detailed calculations
    - Why they do/don't qualify as a play
    - Actual outcome (if historical game)
    
    Args:
        player_name: Player name to check (e.g., "Stephen Curry")
        line_value: Line value to check (e.g., 2.5)
        historical_date: PAST game date (YYYY-MM-DD) or None for today's live props
        season: Season to analyze
        min_streak: Min streak length
        max_streak: Max streak length
    """
    if historical_date:
        date_str = f"HISTORICAL: {historical_date}"
    else:
        date_str = "TODAY'S LIVE PROPS"
    
    print("\n" + "="*100)
    print(f"🔍 DETAILED SANITY CHECK: {player_name} - UNDER {line_value}")
    print(f"📅 Date: {date_str}")
    print("="*100)
    print()
    
    # Load data
    print("Loading data...")
    game_logs = load_season_game_logs(season)
    historical_props = load_historical_props(season)
    
    # Normalize player name
    from src.player_name_utils import normalize_player_name
    player_normalized = normalize_player_name(player_name)
    
    print(f"Normalized name: {player_normalized}")
    print()
    
    # Load props for the specified date
    if historical_date:
        # Historical game - load from historical props
        print(f"Loading HISTORICAL props for {historical_date}...")
        target_props = historical_props[
            (historical_props['date'] == historical_date) &
            (historical_props['player'].apply(normalize_player_name) == player_normalized) &
            (historical_props['consensus_line'] == line_value)
        ]
        
        # If found, rename consensus columns to match today's format
        if len(target_props) > 0:
            target_props = target_props.copy()
            target_props['line'] = target_props['consensus_line']
            target_props['under_odds'] = target_props['under_best_odds']
            target_props['over_odds'] = target_props['over_best_odds']
            target_props['under_book'] = target_props['under_best_book']
            target_props['over_book'] = target_props['over_best_book']
            # Add dummy values for missing columns
            if 'game' not in target_props.columns:
                target_props['game'] = 'Historical Game'
            if 'game_time' not in target_props.columns:
                target_props['game_time'] = historical_date
            if 'num_bookmakers' not in target_props.columns:
                target_props['num_bookmakers'] = target_props.get('num_books', 1)
        
        is_historical = True
    else:
        # Today's game - load from today's props
        print("Loading TODAY'S LIVE props...")
        todays_props = load_todays_props()
        target_props = todays_props[
            (todays_props['player'].apply(normalize_player_name) == player_normalized) &
            (todays_props['line'] == line_value)
        ]
        is_historical = False
    
    # Find prop for this player+line
    player_props = target_props
    
    if len(player_props) == 0:
        date_label = f"HISTORICAL ({historical_date})" if historical_date else "TODAY'S LIVE"
        print(f"❌ No props found for {player_name} at line {line_value} in {date_label} data")
        print()
        print("Available lines for this player:")
        
        if is_historical:
            # Show available lines from historical props for this date
            player_date = historical_props[
                (historical_props['date'] == historical_date) &
                (historical_props['player'].apply(normalize_player_name) == player_normalized)
            ]
            if len(player_date) > 0:
                for _, prop in player_date.iterrows():
                    print(f"   - Line {prop['consensus_line']}: Under {prop['under_best_odds']:+.0f} ({prop['under_best_book']}), Over {prop['over_best_odds']:+.0f} ({prop['over_best_book']})")
            else:
                print(f"   ⚠️  No props found for {player_name} on {historical_date}")
                
                # Show available dates for this player
                player_all_dates = historical_props[
                    historical_props['player'].apply(normalize_player_name) == player_normalized
                ]
                if len(player_all_dates) > 0:
                    print()
                    print(f"💡 Available HISTORICAL dates for {player_name} (most recent 10):")
                    unique_dates = sorted(player_all_dates['date'].unique(), reverse=True)[:10]
                    for d in unique_dates:
                        print(f"   - {d}")
        else:
            # Show available lines from today's props
            player_all = todays_props[todays_props['player'].apply(normalize_player_name) == player_normalized]
            if len(player_all) > 0:
                for _, prop in player_all.iterrows():
                    print(f"   - Line {prop['line']}: Under {prop['under_odds']:+.0f} ({prop['under_book']}), Over {prop['over_odds']:+.0f} ({prop['over_book']})")
            else:
                print(f"   No props found for {player_name} in today's props at all")
        return
    
    prop = player_props.iloc[0]
    
    print("="*100)
    print("TODAY'S LIVE PROP" if not is_historical else f"HISTORICAL PROP FOR {historical_date}")
    print("="*100)
    print(f"Game:        {prop['game']}")
    print(f"Time:        {prop['game_time']}")
    print(f"Line:        {prop['line']}")
    print(f"Under odds:  {prop['under_odds']:+.0f} ({prop['under_book']})")
    print(f"Over odds:   {prop['over_odds']:+.0f} ({prop['over_book']})")
    print(f"Bookmakers:  {int(prop['num_bookmakers'])}")
    print()
    
    # Check if under is underdog
    under_odds = int(prop['under_odds'])
    if under_odds <= 0:
        print("❌ DOES NOT QUALIFY: Under is NOT underdog (odds <= 0)")
        print(f"   Under odds: {under_odds}")
        print(f"   Strategy requires positive under odds")
        return
    
    print("✅ Under is underdog (positive odds)")
    print()
    
    # Detect hot streak with full debugging
    print("="*100)
    print("HOT STREAK DETECTION")
    print("="*100)
    print()
    
    streak_info = detect_hot_streak_with_props(
        player_normalized, game_logs, historical_props,
        min_streak, max_streak, debug=True
    )
    
    if streak_info is None:
        print(f"❌ DOES NOT QUALIFY: No {min_streak}-{max_streak} game hot streak found")
        return
    
    print(f"✅ Hot streak detected: {streak_info['streak_length']} games")
    print()
    
    # Calculate all signals
    print("="*100)
    print("SIGNAL ANALYSIS")
    print("="*100)
    print()
    
    consensus_line = line_value
    streak_avg = streak_info['avg_3pm']
    predicted_3pm = streak_info['predicted_3pm']
    mean_3pa = streak_info['avg_3pa']
    
    # Signal 1: Regression
    regression_signal = streak_avg > consensus_line
    overperformance = streak_avg - consensus_line
    
    print("1️⃣ REGRESSION SIGNAL")
    print(f"   Avg during streak: {streak_avg:.2f} makes")
    print(f"   Line: {consensus_line}")
    print(f"   Overperformance: {overperformance:+.2f}")
    print(f"   Signal: {'✅ TRUE' if regression_signal else '❌ FALSE'} - {'Overperforming, due for regression' if regression_signal else 'Not significantly overperforming'}")
    print()
    
    # Signal 2: Expected Value
    prediction_signal = predicted_3pm < consensus_line
    
    print("2️⃣ EXPECTED VALUE SIGNAL")
    print(f"   Mean 3PA: {mean_3pa:.1f}")
    print(f"   League avg 3P%: {LEAGUE_AVG_3PT_PCT:.0%}")
    print(f"   Predicted 3PM: {mean_3pa:.1f} × {LEAGUE_AVG_3PT_PCT:.0%} = {predicted_3pm:.2f}")
    print(f"   Line: {consensus_line}")
    print(f"   Signal: {'✅ TRUE' if prediction_signal else '❌ FALSE'} - {'Predicts UNDER' if prediction_signal else 'Predicts OVER'}")
    print()
    
    # Signal 3: Binomial Probability
    n_attempts = ceil(mean_3pa)  # Binomial requires integer n - round UP
    threshold = floor(consensus_line)
    prob_under = binom.cdf(threshold, n=n_attempts, p=LEAGUE_AVG_3PT_PCT)
    breakeven_prob = odds_to_implied_probability(under_odds)
    binomial_signal = prob_under > breakeven_prob
    edge = (prob_under - breakeven_prob) * 100
    
    print("3️⃣ BINOMIAL PROBABILITY SIGNAL")
    print(f"   Mean 3PA: {mean_3pa:.1f} → Rounded UP to n={n_attempts} (binomial requires integer)")
    print(f"   Model: Binomial(n={n_attempts}, p={LEAGUE_AVG_3PT_PCT:.0%})")
    print(f"   Line threshold: {threshold} (floor of {consensus_line})")
    print(f"   P(makes ≤ {threshold}): {prob_under*100:.1f}%")
    print(f"   Break-even probability: {breakeven_prob*100:.1f}%")
    print(f"   Edge: {edge:+.1f}%")
    print(f"   Signal: {'✅ TRUE' if binomial_signal else '❌ FALSE'} - {'Probability edge exists' if binomial_signal else 'No probability edge'}")
    print()
    
    # Total signals
    signal_count = sum([regression_signal, prediction_signal, binomial_signal])
    
    print("="*100)
    print("FINAL VERDICT")
    print("="*100)
    print()
    print(f"Total signals agreeing: {signal_count}/3")
    print()
    
    is_play = signal_count == 3
    
    if is_play:
        print("✅ PLAY - All 3 signals agree")
    else:
        print(f"⚠️  NOT RECOMMENDED - Only {signal_count}/3 signals (need all 3)")
    
    print()
    print(f"Recommendation: {'✅ BET UNDER' if is_play else '⚠️  PASS'}")
    print()
    
    # Backtest context
    backtest_stats = BACKTEST_RESULTS.get(streak_info['streak_length'])
    if backtest_stats:
        all_stats = backtest_stats['all']
        print(f"📊 BACKTEST CONTEXT ({streak_info['streak_length']}-game streaks):")
        print(f"   2024-25 Season: {all_stats[0]} bets | {all_stats[1]:.1f}% win | {all_stats[2]:+.2f}% ROI")
    
    # Show actual outcome if historical game
    if is_historical and historical_date:
        print()
        print("="*100)
        print("ACTUAL OUTCOME (HISTORICAL RESULT)")
        print("="*100)
        print()
        
        # Find the player's game on this date
        player_game_logs = game_logs[game_logs['player_name'].apply(normalize_player_name) == player_normalized]
        game_on_date = player_game_logs[player_game_logs['game_date'] == historical_date]
        
        if len(game_on_date) > 0:
            actual_3pm = game_on_date.iloc[0]['threes_made']
            actual_3pa = game_on_date.iloc[0]['threes_attempted']
            
            print(f"Actual 3PM: {actual_3pm}")
            print(f"Actual 3PA: {actual_3pa}")
            print(f"Line: {line_value}")
            print()
            
            # Determine if UNDER won
            under_won = actual_3pm < line_value
            
            if under_won:
                print("✅ UNDER WON")
                # Calculate profit
                bet_amount = 100 / (1 + (100 / under_odds))  # Back-calculate bet amount
                profit = 100
                print(f"   Bet ${bet_amount:.2f} to win ${profit:.2f}")
                print(f"   Return: ${bet_amount + profit:.2f}")
                print(f"   Profit: +${profit:.2f}")
            else:
                print("❌ UNDER LOST")
                bet_amount = 100 / (1 + (100 / under_odds))
                print(f"   Bet ${bet_amount:.2f}")
                print(f"   Loss: -${bet_amount:.2f}")
            
            print()
            
            # Show if signals were correct
            print("Signal Performance:")
            print(f"   Regression: {'✅ Correct' if (regression_signal and under_won) or (not regression_signal and not under_won) else '❌ Wrong'}")
            print(f"   Expected Value: {'✅ Correct' if (prediction_signal and under_won) or (not prediction_signal and not under_won) else '❌ Wrong'}")
            print(f"   Probability: {'✅ Correct' if (binomial_signal and under_won) or (not binomial_signal and not under_won) else '❌ Wrong'}")
        else:
            print(f"⚠️  No game log found for {player_name} on {historical_date}")
            print("   Player may not have played or data not available")
    
    print()
    print("="*100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find underdog-unders betting opportunities for today',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python implementation/find_3pt_underdog_unders_today.py
  
  # Show season stats first
  python implementation/find_3pt_underdog_unders_today.py --show-season-stats
  
  # Custom streak range (2-5 performs best)
  python implementation/find_3pt_underdog_unders_today.py --min-streak 2 --max-streak 5
  
  # Detailed sanity check for today's live props
  python implementation/find_3pt_underdog_unders_today.py --sanity-check "Stephen Curry" --line 4.5
  
  # Sanity check for HISTORICAL/PAST game (shows actual outcome)
  python implementation/find_3pt_underdog_unders_today.py --sanity-check "LeBron James" --line 2.5 --historical-date 2024-11-15
  
  # Season analysis only (no today's opportunities)
  python implementation/find_3pt_underdog_unders_today.py --season-stats-only
        """
    )
    
    parser.add_argument('--min-streak', type=int, default=DEFAULT_MIN_STREAK,
                       help=f'Minimum streak length (default: {DEFAULT_MIN_STREAK})')
    parser.add_argument('--max-streak', type=int, default=DEFAULT_MAX_STREAK,
                       help=f'Maximum streak length (default: {DEFAULT_MAX_STREAK})')
    parser.add_argument('--show-season-stats', action='store_true',
                       help='Show season statistics before finding opportunities')
    parser.add_argument('--season-stats-only', action='store_true',
                       help='Only show season statistics (skip today\'s opportunities)')
    parser.add_argument('--season', type=str, default=CURRENT_NBA_SEASON,
                       help=f'Season to analyze (default: {CURRENT_NBA_SEASON})')
    parser.add_argument('--sanity-check', type=str, default=None,
                       help='Player name for detailed sanity check (e.g., "Stephen Curry")')
    parser.add_argument('--line', type=float, default=None,
                       help='Line value for sanity check (required with --sanity-check)')
    parser.add_argument('--historical-date', type=str, default=None,
                       help='HISTORICAL game date for sanity check (YYYY-MM-DD). Uses past props to show actual outcome. If omitted, checks TODAY\'S live props.')
    
    args = parser.parse_args()
    
    # Handle sanity check mode
    if args.sanity_check:
        if args.line is None:
            print("❌ ERROR: --line is required when using --sanity-check")
            print("Example: --sanity-check \"Stephen Curry\" --line 4.5")
            return
        
        sanity_check_player_line(
            args.sanity_check,
            args.line,
            args.historical_date,
            args.season,
            args.min_streak,
            args.max_streak
        )
        return
    
    print()
    print("🎲 UNDERDOG-UNDERS OPPORTUNITY FINDER")
    print("="*100)
    print()
    print(f"Season: {args.season}")
    print(f"Streak Range: {args.min_streak}-{args.max_streak} consecutive hits")
    print(f"Strategy: Bet UNDER with positive odds after hot streaks")
    print("Rationale: Unders are undervalued in general - find opportunities where we have even more of an edge, when taking unders")
    print()
    
    try:
        # Load season game logs
        game_logs = load_season_game_logs(args.season)
        
        # Load historical props for current season (critical for accurate streak detection)
        historical_props = load_historical_props(args.season)
        
        # Show season statistics
        if args.show_season_stats or args.season_stats_only:
            season_stats = analyze_season_patterns(
                game_logs, historical_props,
                args.min_streak, args.max_streak
            )
        
        # Skip today's opportunities if only showing stats
        if args.season_stats_only:
            print("="*100)
            print("✅ SEASON ANALYSIS COMPLETE")
            print("="*100)
            print()
            return
        
        # Load today's props
        todays_props = load_todays_props()
        
        # Find opportunities
        opportunities = find_todays_opportunities(
            todays_props, game_logs, historical_props,
            args.min_streak, args.max_streak
        )
        
        # Display
        display_opportunities(opportunities)
        
        # Save
        save_opportunities(opportunities)
        
        # Summary
        print("="*100)
        print("✅ ANALYSIS COMPLETE")
        print("="*100)
        print()
        
        if len(opportunities) > 0:
            print("⚠️  IMPORTANT REMINDERS:")
            print("   • Based on 2024-25 backtesting: +5-6% ROI on 2-3 game streaks")
            print("   • Past performance does NOT guarantee future results")
            print("   • Always verify odds and lines before placing bets")
            print("   • Bet responsibly and within your limits")
            print("   • Track your results to validate the strategy")
            print()
            print("📊 NEXT STEPS:")
            print("   1. Verify each opportunity on your sportsbook")
            print("   2. Check for any injury news or lineup changes")
            print("   3. Consider the opponent's 3pt defense")
            print("   4. Place bets and track results")
        else:
            print("💡 NO OPPORTUNITIES TODAY")
            print("   • This is normal - specific criteria must be met")
            print("   • Try again tomorrow or on nights with more games")
            print("   • Market conditions change - be patient")
        
        print()
        
    except FileNotFoundError as e:
        print()
        print("="*100)
        print(f"❌ ERROR: Missing Required Data")
        print("="*100)
        print()
        print(str(e))
        print()
        print("SETUP REQUIRED:")
        print()
        print("1. Build season game logs:")
        print(f"   python scripts/build_season_game_logs.py --season {args.season}")
        print()
        print("2. Fetch and build historical props:")
        print(f"   python scripts/fetch_and_build_season_props.py --season {args.season} --market player_threes")
        print()
        print("3. Fetch today's props:")
        print("   python scripts/fetch_historical_props.py")
        print()
        sys.exit(1)
    
    except Exception as e:
        print()
        print("="*100)
        print(f"❌ ERROR: {e}")
        print("="*100)
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

