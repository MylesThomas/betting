"""
NBA 3PT Prop Streak Backtest - Multiple Analysis Options

Testing the theory that players who continously miss their 3pt prop are due for a regression to the mean.
- The public is very reactive to streaks, and will overreact to them. (e.g., start betting unders/against them when on an under streak, or betting overs for them when on an over streak.)
- This creates not only a market-based opportunity, but basketball is a game of runs, and players are due for a regression to the mean.

USAGE:
------
Run specific analyses using command-line flags:

  # Run only the main streak analysis
  python 20251121_nba_3pt_prop_miss_streaks.py --streak
  
  # Run only underdog unders analysis  
  python 20251121_nba_3pt_prop_miss_streaks.py --underdog-unders
  
  # Run multiple analyses
  python 20251121_nba_3pt_prop_miss_streaks.py --streak --underdog-unders
  
  # Run all analyses
  python 20251121_nba_3pt_prop_miss_streaks.py --all
  
  # Run with no flags = runs all (default)
  python 20251121_nba_3pt_prop_miss_streaks.py

Available flags:
  --streak              : Main streak analysis (OVERS + UNDERS backtest)
  --underdog-unders     : Underdog unders (positive odds only, streaks 2-10)
  --player-season       : Player season analysis (full season view)
  --blind-under         : Blind under betting analysis
  --blind-under-by-line : Blind under betting by line value
  --all                 : Run all analyses

STRATEGIES TESTED:
------------------
1. OVERS BACKTEST (Cold Streak Regression):
   - Find players who have MISSED their 3pt prop for X consecutive games
   - Bet OVER on the next game (theory: due to bounce back)
   
2. UNDERS BACKTEST (Hot Streak Regression):
   - Find players who have HIT their 3pt prop for X consecutive games  
   - Bet UNDER on the next game (theory: due to cool down)
   
3. UNDERDOG UNDERS (Positive Odds):
   - UNDER bets with positive odds after hot streaks (2-10 games)
   - Market mispricing: player is hot but under still underdog

Streak lengths tested: 2, 3, 4, 5, 6, 7 consecutive games (2-10 for underdog unders)

Optional filters:
- Verify betting odds movement (improving/worsening during streak)
- Track 3PA trend (shooting more/less during streak)

Assumptions:
- Each bet is sized to win TARGET_WIN ($100 by default).
    - If - odds are offered, bet more than $100 to win $100. (e.g., -150 odds = bet $150 to win $100)
    - If + odds are offered, bet less than $100 to win $100. (e.g., +150 odds = bet $66.67 to win $100)

Data Source:
- Uses cleaned and merged dataset: data/consensus_props_with_game_results_min10_2024_25.csv
- Created by: scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py
- Contains: Consensus prop lines + actual game results (3PM, 3PA, minutes)
- Pre-filtered: Only includes games where player played >= 10 minutes (removes garbage time/injury outliers)
- Match rate: 92.97% (11,267 props with game results)
- Player names normalized for accurate matching

Done:
- [X] Get the consensus props data (player threes line per game per player) for all games in the 2024-25 season.
- [X] Verify we have the actual game results data (3PA/3PM per player per game) - REQUIRED FOR BACKTESTING
- [X] Merge and clean prop data with actual game results
- [X] Filter out garbage time/injury appearances (< 10 min)

To do:
- [ ] Incorporate 3pt shots attempted into the analysis. (This is the #1 predictor of 3pt shooting performance/makes.)
- [ ] Incorporate a distribution so we can compare the games of a player whose over/under is moving down with the games of a player (Example: Game 1 was -110 to make 1.5 3's, Game 2 was +100 to make 1.5 3's, Game 3 is now -150 to make 0.5 ie. a DIFFERENT number of 3's)

Date: 2025-11-20
Updated: 2025-11-21
Author: Myles Thomas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.odds_utils import calculate_bet_amount, calculate_profit, odds_to_implied_probability


# ============================================================================
# CONFIG
# ============================================================================

# Bet Sizing
TARGET_WIN = 100  # Target profit per bet ($100)
# If + odds: bet less than $100 to win $100
# If - odds: bet more than $100 to win $100
# Example: +150 odds → bet $66.67 to win $100
# Example: -150 odds → bet $150 to win $100

# Odds Filtering
MAX_ODDS_THRESHOLD = -300  # Exclude odds worse than this (e.g., -300 means exclude -301, -400, etc.)
# Bookmakers sometimes post extreme odds on props that are essentially locks
# (e.g., centers who never shoot 3s). These are traps and should be excluded.

# League Averages (for prediction modeling)
LEAGUE_AVG_3PT_PCT = 0.35  # NBA league-wide 3P% (~35% as of 2024-25 season)
# Prediction approach:
#   1. Calculate mean 3PA from lookback period (e.g., 2-game streak = mean of those 2 games)
#   2. Predict next game 3PA = mean 3PA from lookback
#   3. Predict next game 3PM = predicted_3PA × LEAGUE_AVG_3PT_PCT
#   4. Compare prediction to line to get expected direction
# Note: These predictions are tracked but NOT yet used for betting decisions

# Trend Filters (set to None to disable filtering)
FILTER_3PA_TREND = None  # Set to 'up', 'down', or None to disable
FILTER_ODDS_TREND = None  # Set to 'up' (better odds), 'down' (worse odds), or None to disable

# Examples for OVER bets (cold streak regression):
# FILTER_3PA_TREND = 'up'     # Only bet when 3PA is increasing during streak
# FILTER_3PA_TREND = 'down'   # Only bet when 3PA is decreasing during streak
# FILTER_ODDS_TREND = 'down'  # Only bet when odds getting worse (market doubling down on under)
# FILTER_ODDS_TREND = 'up'    # Only bet when odds improving (market giving up on player)

# Examples for UNDER bets (hot streak regression):
# FILTER_3PA_TREND = 'down'   # Player cooling off (fewer attempts)
# FILTER_ODDS_TREND = 'down'  # Market overpricing the over (too confident player stays hot)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
# Note: Odds calculation functions moved to src/odds_utils.py


def calculate_trend(values):
    """
    Calculate trend direction for a list of values.
    
    Uses simple linear regression slope.
    Positive = increasing, Negative = decreasing
    
    Args:
        values: List of numeric values (ordered by time)
    
    Returns:
        Slope of the trend line
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = np.arange(n)  # 0, 1, 2, ... n-1
    y = np.array(values)
    
    # Linear regression: y = mx + b
    # slope m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return slope


# ============================================================================
# DATA LOADING
# ============================================================================

def load_clean_data():
    """
    Load cleaned and merged dataset with props + game results.
    
    This dataset was created by scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py
    and contains:
    - Consensus prop lines (from historical_props/consensus_props_player_threes.csv)
    - Actual game results (3PM, 3PA, minutes from NBA API)
    - Pre-filtered: Only games where player played >= 10 minutes
    - Player names normalized for accurate matching
    - Match rate: 92.97% (11,267 props with game results)
    
    Columns include:
    - player, date, game, game_time
    - consensus_line, num_bookmakers
    - over_best_odds, under_best_odds, over_avg_odds, under_avg_odds
    - threes_made, threes_attempted, minutes
    - team, matchup, opponent
    
    Returns:
        DataFrame with merged props and game results
    """
    clean_data_file = Path(__file__).parent.parent / 'data' / 'consensus_props_with_game_results_min10_2024_25.csv'
    
    if not clean_data_file.exists():
        raise FileNotFoundError(
            f"Clean data file not found: {clean_data_file}\n"
            f"Run scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py to generate it."
        )
    
    df = pd.read_csv(clean_data_file)
    
    # Filter out DNP situations (where player didn't play that game)
    # These show up as NULL in threes_made
    initial_count = len(df)
    df = df[df['threes_made'].notna()].copy()
    dnp_count = initial_count - len(df)
    
    if dnp_count > 0:
        print(f"   Filtered out {dnp_count:,} DNP situations (player didn't play)")
    
    # Sort by player and date for streak analysis
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    
    return df


# ============================================================================
# STREAK DETECTION
# ============================================================================

def detect_miss_streaks(df, min_streak_length=2, max_streak_length=10):
    """
    Detect consecutive games where player missed their prop.
    
    Args:
        df: Merged dataframe with props and game results
        min_streak_length: Minimum consecutive misses to qualify
        max_streak_length: Maximum streak length to track
    
    Returns:
        DataFrame with streak information
    """
    # Determine if prop was hit (actual >= line means OVER wins)
    df['hit_over'] = df['threes_made'] >= df['consensus_line']
    df['missed_over'] = ~df['hit_over']
    
    merged = df.copy()
    
    # Calculate consecutive miss streaks
    streaks = []
    
    for player in merged['player'].unique():
        player_games = merged[merged['player'] == player].sort_values('date').copy()
        
        current_streak = 0
        streak_start_idx = None
        
        for idx, row in player_games.iterrows():
            if row['missed_over']:
                current_streak += 1
                if streak_start_idx is None:
                    streak_start_idx = idx
            else:
                # Streak broken - record if it qualifies
                if current_streak >= min_streak_length:
                    streaks.append({
                        'player': player,
                        'streak_length': current_streak,
                        'streak_start_idx': streak_start_idx,
                        'streak_end_idx': idx - 1,
                        'next_game_idx': idx  # This is the game we'd bet on
                    })
                current_streak = 0
                streak_start_idx = None
        
        # Handle streak that extends to last game
        if current_streak >= min_streak_length:
            streaks.append({
                'player': player,
                'streak_length': current_streak,
                'streak_start_idx': streak_start_idx,
                'streak_end_idx': player_games.index[-1],
                'next_game_idx': None  # No next game to bet on
            })
    
    return pd.DataFrame(streaks)


# ============================================================================
# BETTING SIMULATION
# ============================================================================

def simulate_streak_betting(df, streak_lengths=[2, 3, 4, 5], 
                           check_odds_movement=True, bet_direction='over'):
    """
    Simulate betting on player OVERS (after miss streaks) or UNDERS (after hit streaks).
    
    Uses global TARGET_WIN config for bet sizing.
    
    Args:
        df: Merged dataframe with props and game results
        streak_lengths: List of streak lengths to test
        check_odds_movement: If True, only bet when odds are improving
        bet_direction: 'over' (bet overs after cold streaks) or 'under' (bet unders after hot streaks)
    
    Returns:
        Dictionary with results for each streak length
    """
    # Determine if prop was hit (actual >= line means OVER wins)
    merged = df.copy()
    merged['hit_over'] = merged['threes_made'] >= merged['consensus_line']
    merged['missed_over'] = ~merged['hit_over']
    
    # Determine which type of streak we're looking for
    if bet_direction == 'over':
        streak_column = 'missed_over'  # Looking for miss streaks to bet OVER
        bet_odds_column = 'over_best_odds'
        bet_type = 'OVER'
    else:  # 'under'
        streak_column = 'hit_over'  # Looking for hit streaks to bet UNDER
        bet_odds_column = 'under_best_odds'
        bet_type = 'UNDER'
    
    results = {}
    
    for min_streak in streak_lengths:
        bets = []
        
        for player in merged['player'].unique():
            player_games = merged[merged['player'] == player].sort_values('date').copy()
            player_games = player_games.reset_index(drop=True)
            
            current_streak = 0
            streak_games = []
            
            for i, row in player_games.iterrows():
                if row[streak_column]:
                    current_streak += 1
                    streak_games.append(row)
                    
                    # Check if we hit our streak threshold and have a next game
                    if current_streak == min_streak and i + 1 < len(player_games):
                        next_game = player_games.iloc[i + 1]
                        
                        # Check if line is consistent
                        if next_game['consensus_line'] == row['consensus_line']:
                            
                            # Optional: Check if odds improved
                            bet_approved = True
                            if check_odds_movement:
                                # Compare odds from first game to current
                                if bet_direction == 'over':
                                    first_odds = streak_games[0]['over_avg_odds']
                                    current_odds = row['over_avg_odds']
                                    next_odds = next_game['over_best_odds']
                                else:  # 'under'
                                    first_odds = streak_games[0]['under_avg_odds']
                                    current_odds = row['under_avg_odds']
                                    next_odds = next_game['under_best_odds']
                                
                                # Bet if odds improved (more positive or less negative)
                                bet_approved = next_odds > current_odds
                            
                            if bet_approved:
                                # Calculate trends during the streak
                                streak_3pa = [g['threes_attempted'] for g in streak_games]
                                
                                # Get odds based on bet direction
                                if bet_direction == 'over':
                                    streak_odds = [g['over_best_odds'] for g in streak_games]
                                else:  # 'under'
                                    streak_odds = [g['under_best_odds'] for g in streak_games]
                                
                                # Trend calculations
                                # Positive 3PA trend = taking more shots
                                trend_3pa = calculate_trend(streak_3pa)
                                
                                # Positive odds trend = odds getting better for us (more +, less -)
                                trend_odds = calculate_trend(streak_odds)
                                
                                # Calculate percentage changes for easier interpretation
                                pct_change_3pa = ((streak_3pa[-1] - streak_3pa[0]) / max(streak_3pa[0], 1)) * 100 if len(streak_3pa) >= 2 else 0
                                
                                # Calculate implied probability change (in percentage points)
                                if len(streak_odds) >= 2:
                                    first_prob = odds_to_implied_probability(streak_odds[0]) * 100
                                    last_prob = odds_to_implied_probability(streak_odds[-1]) * 100
                                    pct_change_odds = last_prob - first_prob  # Change in percentage points
                                else:
                                    pct_change_odds = 0
                                
                                # Apply trend filters if configured
                                passes_filters = True
                                
                                if FILTER_3PA_TREND == 'up' and trend_3pa <= 0:
                                    passes_filters = False
                                elif FILTER_3PA_TREND == 'down' and trend_3pa >= 0:
                                    passes_filters = False
                                
                                if FILTER_ODDS_TREND == 'up' and trend_odds <= 0:
                                    passes_filters = False
                                elif FILTER_ODDS_TREND == 'down' and trend_odds >= 0:
                                    passes_filters = False
                                
                                if not passes_filters:
                                    # Skip this bet - doesn't meet trend filters
                                    continue
                                
                                # Place bet on next game
                                if bet_direction == 'over':
                                    bet_odds = next_game['over_best_odds']
                                    bet_won = next_game['hit_over']
                                else:  # 'under'
                                    bet_odds = next_game['under_best_odds']
                                    bet_won = next_game['missed_over']  # Under wins when over misses
                                
                                # Skip bets with extreme odds (likely bookmaker traps)
                                if pd.notna(bet_odds) and bet_odds <= MAX_ODDS_THRESHOLD:
                                    continue
                                
                                bet_amount = calculate_bet_amount(bet_odds, TARGET_WIN)
                                
                                if bet_won:
                                    profit = calculate_profit(bet_odds, bet_amount)
                                else:
                                    profit = -bet_amount
                                
                                # Calculate prediction metrics using mean of lookback period
                                mean_3pa = np.mean(streak_3pa) if streak_3pa else 0
                                predicted_3pa = mean_3pa
                                predicted_3pm = predicted_3pa * LEAGUE_AVG_3PT_PCT
                                actual_3pa = next_game.get('threes_attempted', 0)
                                
                                # Check if prediction was directionally correct
                                prediction_correct = (
                                    (predicted_3pm >= next_game['consensus_line'] and next_game['threes_made'] >= next_game['consensus_line']) or
                                    (predicted_3pm < next_game['consensus_line'] and next_game['threes_made'] < next_game['consensus_line'])
                                )
                                
                                bets.append({
                                    'player': player,
                                    'bet_date': next_game['date'],
                                    'bet_direction': bet_direction.upper(),
                                    'streak_length': current_streak,
                                    'line': next_game['consensus_line'],
                                    'odds': bet_odds,
                                    'bet_amount': bet_amount,
                                    'actual_threes': next_game['threes_made'],
                                    'won': bet_won,
                                    'profit': profit,
                                    # Trend indicators
                                    'trend_3pa': trend_3pa,
                                    'trend_odds': trend_odds,
                                    'pct_change_3pa': pct_change_3pa,
                                    'pct_change_odds': pct_change_odds,
                                    'first_3pa': streak_3pa[0] if streak_3pa else None,
                                    'last_3pa': streak_3pa[-1] if streak_3pa else None,
                                    'first_odds': streak_odds[0] if streak_odds else None,
                                    'last_odds': streak_odds[-1] if streak_odds else None,
                                    'streak_type': 'MISS' if bet_direction == 'over' else 'HIT',
                                    # Prediction metrics (mean-based, not used for betting yet)
                                    'mean_3pa_lookback': mean_3pa,
                                    'predicted_3pa': predicted_3pa,
                                    'predicted_3pm': predicted_3pm,
                                    'actual_3pa': actual_3pa,
                                    'prediction_directionally_correct': prediction_correct,
                                    'streak_games': [g.to_dict() for g in streak_games],  # Save the streak history
                                    'bet_game': next_game.to_dict()  # Save the bet game
                                })
                else:
                    # Reset streak
                    current_streak = 0
                    streak_games = []
        
        # Calculate summary stats
        bets_df = pd.DataFrame(bets)
        
        if len(bets_df) > 0:
            total_bets = len(bets_df)
            wins = bets_df['won'].sum()
            losses = total_bets - wins
            win_rate = wins / total_bets * 100
            
            total_wagered = bets_df['bet_amount'].sum()
            total_profit = bets_df['profit'].sum()
            roi = (total_profit / total_wagered) * 100
            
            avg_odds = bets_df['odds'].mean()
            
            results[min_streak] = {
                'total_bets': total_bets,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_wagered': total_wagered,
                'total_profit': total_profit,
                'roi': roi,
                'avg_odds': avg_odds,
                'bets_detail': bets_df
            }
        else:
            results[min_streak] = {
                'total_bets': 0,
                'message': 'No qualifying bets found'
            }
    
    return results


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def american_odds_to_probability(odds):
    """Convert American odds to implied probability percentage"""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100) * 100
    else:
        return 100 / (odds + 100) * 100


def probability_to_american_odds(prob_pct):
    """
    Convert implied probability percentage back to American odds.
    Assumes negative odds (favorite) if prob > 50%, positive odds (underdog) if prob < 50%.
    """
    if prob_pct >= 50:
        # Negative odds (favorite)
        return -prob_pct / (100 - prob_pct) * 100
    else:
        # Positive odds (underdog)
        return (100 - prob_pct) / prob_pct * 100


def display_player_journey(bet_row):
    """Display a single player's journey through their streak with full details"""
    
    player = bet_row['player']
    streak_games = bet_row['streak_games']
    bet_game = bet_row['bet_game']
    won = bet_row['won']
    bet_direction = bet_row['bet_direction']
    streak_type = bet_row['streak_type']
    
    print(f"\n    Player: {player}")
    print(f"    Streak: {len(streak_games)} consecutive {streak_type.lower()}s")
    print(f"    {'─'*82}")
    print(f"    {'Game':12} | Date       | Line | 3PM  | 3PA  | Over Odds | Under Odds | Result")
    print(f"    {'─'*82}")
    
    # Collect 3PA for trend calculation
    streak_3pa = []
    
    # Collect odds during streak for trend analysis
    streak_odds = []
    
    # Show the streak
    for i, game in enumerate(streak_games, 1):
        date = game['date']
        line = game['consensus_line']
        actual = game['threes_made']
        attempts = game.get('threes_attempted', 0)
        over_odds = game.get('over_best_odds', 'N/A')
        under_odds = game.get('under_best_odds', 'N/A')
        
        streak_3pa.append(attempts)
        
        # Collect the odds we're betting (depends on bet direction)
        if bet_direction == 'OVER':
            if isinstance(over_odds, (int, float)):
                streak_odds.append(over_odds)
        else:  # 'UNDER'
            if isinstance(under_odds, (int, float)):
                streak_odds.append(under_odds)
        
        # Format odds properly
        over_str = f"{over_odds:>4}" if isinstance(over_odds, (int, float)) else str(over_odds)
        under_str = f"{under_odds:>4}" if isinstance(under_odds, (int, float)) else str(under_odds)
        
        # Label and result based on streak type
        streak_label = f"{streak_type} {i}:"
        
        if streak_type == 'MISS':
            result = f"❌ UNDER ({actual} < {line})"
        else:  # 'HIT'
            result = f"✅ OVER ({actual} ≥ {line})"
        
        print(f"    {streak_label:12} | {date} | {line:>4.1f} | {actual:>4.1f} | {attempts:>4.1f} | {over_str:>9} | {under_str:>10} | {result}")
    
    # Show trend analysis
    print(f"    {'─'*82}")
    print(f"    TREND ANALYSIS DURING STREAK:")
    print(f"    {'─'*82}")
    
    # 3PA trend
    trend_3pa = bet_row.get('trend_3pa', 0)
    pct_3pa = bet_row.get('pct_change_3pa', 0)
    first_3pa = bet_row.get('first_3pa', 0)
    last_3pa = bet_row.get('last_3pa', 0)
    
    # Calculate predicted 3PA using MEAN of lookback period (not extrapolation)
    # This is more stable than extrapolating trends
    if len(streak_3pa) >= 1:
        mean_3pa = np.mean(streak_3pa)
        min_3pa = np.min(streak_3pa)
        max_3pa = np.max(streak_3pa)
        predicted_3pa = mean_3pa
    else:
        mean_3pa = 0
        min_3pa = 0
        max_3pa = 0
        predicted_3pa = 0
    
    # Calculate predicted makes using league average 3P%
    predicted_3pm = predicted_3pa * LEAGUE_AVG_3PT_PCT
    
    # Get the line to compare prediction
    line = bet_game['consensus_line']
    
    # Determine if prediction suggests over or under
    if predicted_3pm >= line:
        prediction_signal = f"📈 OVER (pred: {predicted_3pm:.2f} ≥ line: {line:.1f})"
    else:
        prediction_signal = f"📉 UNDER (pred: {predicted_3pm:.2f} < line: {line:.1f})"
    
    # Format all 3PA values to show the full picture
    attempts_list_str = ", ".join([f"{a:.1f}" for a in streak_3pa])
    
    print(f"    3-Point Attempts:")
    print(f"      All games: [{attempts_list_str}]")
    print(f"      Range: {min_3pa:.1f} to {max_3pa:.1f}  |  Mean: {mean_3pa:.1f}  |  First→Last change: {pct_3pa:+.1f}%")
    print(f"      📊 Predicted 3PA for bet game: {predicted_3pa:.1f} (using mean)")
    print(f"      🎯 Predicted 3PM (@ {LEAGUE_AVG_3PT_PCT:.0%}): {predicted_3pm:.2f} makes → {prediction_signal}")
    print()
    
    # Odds trend analysis
    first_odds = bet_row.get('first_odds', 0)
    last_odds = bet_row.get('last_odds', 0)
    
    # Calculate implied probabilities for first and last
    first_prob = odds_to_implied_probability(first_odds) * 100 if first_odds != 0 else 0
    last_prob = odds_to_implied_probability(last_odds) * 100 if last_odds != 0 else 0
    prob_change = last_prob - first_prob
    odds_value_change = last_odds - first_odds
    
    # Format all odds values to show the full picture
    if len(streak_odds) > 0:
        odds_list_str = ", ".join([f"{o:+d}" for o in streak_odds])
        
        # Calculate implied probabilities for all games
        implied_probs = [odds_to_implied_probability(o) * 100 for o in streak_odds]
        mean_prob = np.mean(implied_probs)
        min_prob = np.min(implied_probs)
        max_prob = np.max(implied_probs)
    else:
        odds_list_str = "N/A"
        mean_prob = 0
        min_prob = 0
        max_prob = 0
    
    # Interpret market movement based on ODDS getting better/worse for bettor
    # Higher odds (more positive or less negative) = BETTER for bettor
    if odds_value_change > 0:
        odds_interpretation = "IMPROVING for bettor ✅"
        market_confidence = "Market becoming LESS confident"
    elif odds_value_change < 0:
        odds_interpretation = "WORSENING for bettor ⚠️"
        market_confidence = "Market becoming MORE confident"
    else:
        odds_interpretation = "STABLE"
        market_confidence = "No significant movement"
    
    # Calculate percentage change in implied probability
    prob_pct_change = ((prob_change / first_prob) * 100) if first_prob != 0 else 0
    
    print(f"    {bet_direction} Odds Movement:")
    print(f"      All games: [{odds_list_str}]")
    print(f"      Implied prob range: {min_prob:.1f}% to {max_prob:.1f}%  |  Mean: {mean_prob:.1f}%")
    print(f"      First→Last: {first_odds:+d} ({first_prob:.1f}%) → {last_odds:+d} ({last_prob:.1f}%)")
    print(f"      Total Change:")
    print(f"        • American odds: {odds_value_change:+d} points (e.g., {first_odds:+d} to {last_odds:+d})")
    print(f"        • Implied probability: {prob_change:+.1f} percentage points ({prob_pct_change:+.1f}% relative change)")
    print(f"      💭 {odds_interpretation} ({market_confidence})")
    print()
    
    # Show the bet game
    print(f"    {'─'*82}")
    print(f"    THE BET:")
    print(f"    {'─'*82}")
    
    date = bet_game['date']
    line = bet_game['consensus_line']
    actual = bet_game['threes_made']
    attempts = bet_game.get('threes_attempted', 0)
    profit = bet_row['profit']
    bet_amount = bet_row['bet_amount']
    
    # Get odds based on bet direction
    if bet_direction == 'OVER':
        bet_odds = bet_game.get('over_best_odds', 'N/A')
        alt_odds = bet_game.get('under_best_odds', 'N/A')
        over_str = f"{bet_odds:>4}" if isinstance(bet_odds, (int, float)) else str(bet_odds)
        under_str = f"{alt_odds:>4}" if isinstance(alt_odds, (int, float)) else str(alt_odds)
    else:  # 'UNDER'
        bet_odds = bet_game.get('under_best_odds', 'N/A')
        alt_odds = bet_game.get('over_best_odds', 'N/A')
        over_str = f"{alt_odds:>4}" if isinstance(alt_odds, (int, float)) else str(alt_odds)
        under_str = f"{bet_odds:>4}" if isinstance(bet_odds, (int, float)) else str(bet_odds)
    
    bet_desc = f"BET {bet_direction}:"
    
    # Determine result description
    actual_result = "OVER" if actual >= line else "UNDER"
    actual_emoji = "✅" if actual >= line else "❌"
    actual_desc = f"{actual_emoji} {actual_result} ({actual}/{attempts} made, line was {line})"
    
    print(f"    {bet_desc:12} | {date} | {line:>4.1f} | {actual:>4.1f} | {attempts:>4.1f} | {over_str:>9} | {under_str:>10} | {actual_desc}")
    
    print()
    
    # Compare prediction vs actual
    if len(streak_3pa) >= 1:
        # Compare 3PA prediction
        diff_3pa = attempts - predicted_3pa
        if abs(diff_3pa) <= 1:
            accuracy_3pa = "✅ Close!"
        elif diff_3pa > 0:
            accuracy_3pa = f"📈 {diff_3pa:+.1f} more than predicted"
        else:
            accuracy_3pa = f"📉 {abs(diff_3pa):.1f} fewer than predicted"
        
        # Compare 3PM prediction
        actual_3pm = actual
        diff_3pm = actual_3pm - predicted_3pm
        
        # Check if prediction was directionally correct
        prediction_correct = (predicted_3pm >= line and actual_3pm >= line) or (predicted_3pm < line and actual_3pm < line)
        direction_emoji = "✅" if prediction_correct else "❌"
        
        print(f"    PREDICTION ANALYSIS:")
        print(f"    ├─ 3PA: Predicted {predicted_3pa:.1f}, Actual {attempts:.1f} → {accuracy_3pa}")
        print(f"    ├─ 3PM: Predicted {predicted_3pm:.2f}, Actual {actual_3pm:.1f} → Diff: {diff_3pm:+.2f}")
        print(f"    └─ Direction vs Line ({line:.1f}): {direction_emoji} {'CORRECT' if prediction_correct else 'INCORRECT'}")
    
    print()
    
    # Final result
    if won:
        result_emoji = "✅ WIN"
    else:
        result_emoji = "❌ LOSS"
    
    print(f"    {result_emoji} | Wagered: ${bet_amount:.2f} @ {bet_odds:+d} odds | Profit: ${profit:>+7.2f}")
    print()


def display_results(results, bet_direction='over'):
    """
    Display backtesting results in a readable format.
    
    Args:
        results: Dictionary with results for each streak length
        bet_direction: 'over' or 'under' to determine strategy description
    """
    
    print("="*80)
    print("STREAK BETTING BACKTEST RESULTS")
    print("="*80)
    print()
    
    # Determine strategy description
    if bet_direction == 'over':
        strategy_desc = "Bet OVER after player MISSES X consecutive games (cold streak regression)"
        streak_type = "MISS"
    else:
        strategy_desc = "Bet UNDER after player HITS X consecutive games (hot streak regression)"
        streak_type = "HIT"
    
    print(f"Strategy: {strategy_desc}")
    print(f"Bet sizing: Each bet targets ${TARGET_WIN} profit")
    print(f"  (Variable stake based on odds: + odds = bet less, - odds = bet more)")
    print()
    
    # Show active filters
    filters_active = []
    if FILTER_3PA_TREND:
        filter_desc = "3PA Trend: " + FILTER_3PA_TREND.upper()
        if bet_direction == 'over':
            if FILTER_3PA_TREND == 'up':
                filter_desc += " (player shooting more during cold streak)"
            else:
                filter_desc += " (player shooting less during cold streak)"
        else:
            if FILTER_3PA_TREND == 'up':
                filter_desc += " (player shooting more during hot streak)"
            else:
                filter_desc += " (player shooting less during hot streak)"
        filters_active.append(filter_desc)
    
    if FILTER_ODDS_TREND:
        filter_desc = "Odds Trend: " + FILTER_ODDS_TREND.upper()
        if FILTER_ODDS_TREND == 'up':
            filter_desc += " (market giving better odds = less confident)"
        else:
            filter_desc += " (market giving worse odds = more confident)"
        filters_active.append(filter_desc)
    
    if filters_active:
        print("🔍 ACTIVE FILTERS:")
        for f in filters_active:
            print(f"   • {f}")
        print()
    else:
        print("🔍 NO FILTERS ACTIVE (all bets included)")
        print()
    
    for streak_length in sorted(results.keys()):
        result = results[streak_length]
        
        streak_type = "misses" if bet_direction == 'over' else "hits"
        
        print(f"\n{'─'*80}")
        print(f"STREAK LENGTH: {streak_length} consecutive {streak_type}")
        print(f"{'─'*80}")
        
        if 'message' in result:
            print(f"  {result['message']}")
            continue
        
        # Calculate proper odds statistics
        bets_df = result['bets_detail']
        median_odds = int(bets_df['odds'].median())
        
        # Calculate average implied probability (the RIGHT way to average odds)
        implied_probs = bets_df['odds'].apply(american_odds_to_probability)
        avg_implied_prob = implied_probs.mean()
        
        # Categorize odds and calculate avg implied prob for each
        favorites_mask = bets_df['odds'] < 0
        underdogs_mask = bets_df['odds'] > 0
        
        favorites = favorites_mask.sum()
        underdogs = underdogs_mask.sum()
        
        # Calculate average implied probability for favorites and underdogs
        if favorites > 0:
            fav_avg_impl_prob = bets_df[favorites_mask]['odds'].apply(american_odds_to_probability).mean()
        else:
            fav_avg_impl_prob = 0
        
        if underdogs > 0:
            dog_avg_impl_prob = bets_df[underdogs_mask]['odds'].apply(american_odds_to_probability).mean()
        else:
            dog_avg_impl_prob = 0
        
        print(f"  Total Bets:      {result['total_bets']:,}")
        print(f"  Wins:            {result['wins']:,} ({result['win_rate']:.1f}%)")
        print(f"  Losses:          {result['losses']:,}")
        print(f"  ")
        print(f"  Total Wagered:   ${result['total_wagered']:,.2f}")
        print(f"  Total Profit:    ${result['total_profit']:,.2f}")
        print(f"  ROI:             {result['roi']:.2f}%")
        print(f"  ")
        print(f"  Median Odds:              {median_odds:+d}")
        print(f"  Avg Implied Probability:  {avg_implied_prob:.1f}%")
        
        # Display favorites vs underdogs with implied probabilities
        fav_str = f"{favorites} (avg imp prob: {fav_avg_impl_prob:.1f}%)" if favorites > 0 else "0"
        dog_str = f"{underdogs} (avg imp prob: {dog_avg_impl_prob:.1f}%)" if underdogs > 0 else "0"
        print(f"  Favorites vs Underdogs:   {fav_str} vs {dog_str}")
        
        # Trend analysis
        print(f"  ")
        print(f"  TREND ANALYSIS (During Streak):")
        print(f"  {'─'*76}")
        
        # Overall trends
        avg_3pa_trend = bets_df['trend_3pa'].mean()
        avg_odds_trend = bets_df['trend_odds'].mean()
        avg_3pa_pct_change = bets_df['pct_change_3pa'].mean()
        avg_odds_change = bets_df['pct_change_odds'].mean()
        
        # Trends for winners vs losers
        winners_df = bets_df[bets_df['won'] == True]
        losers_df = bets_df[bets_df['won'] == False]
        
        win_3pa_trend = winners_df['trend_3pa'].mean() if len(winners_df) > 0 else 0
        win_odds_trend = winners_df['trend_odds'].mean() if len(winners_df) > 0 else 0
        win_3pa_pct = winners_df['pct_change_3pa'].mean() if len(winners_df) > 0 else 0
        win_odds_change = winners_df['pct_change_odds'].mean() if len(winners_df) > 0 else 0
        
        loss_3pa_trend = losers_df['trend_3pa'].mean() if len(losers_df) > 0 else 0
        loss_odds_trend = losers_df['trend_odds'].mean() if len(losers_df) > 0 else 0
        loss_3pa_pct = losers_df['pct_change_3pa'].mean() if len(losers_df) > 0 else 0
        loss_odds_change = losers_df['pct_change_odds'].mean() if len(losers_df) > 0 else 0
        
        # 3PA Trends
        print(f"  3-Point Attempts Trend:")
        print(f"    All Bets:     {avg_3pa_trend:>+7.3f} slope  |  {avg_3pa_pct_change:>+6.1f}% change")
        print(f"    Winners:      {win_3pa_trend:>+7.3f} slope  |  {win_3pa_pct:>+6.1f}% change")
        print(f"    Losers:       {loss_3pa_trend:>+7.3f} slope  |  {loss_3pa_pct:>+6.1f}% change")
        print(f"  ")
        
        # Odds Trends (higher = better for us, more positive odds)
        print(f"  Odds Trend (higher = better odds for bettor):")
        print(f"    All Bets:     {avg_odds_trend:>+7.3f} slope  |  {avg_odds_change:>+6.1f} pts change")
        print(f"    Winners:      {win_odds_trend:>+7.3f} slope  |  {win_odds_change:>+6.1f} pts change")
        print(f"    Losers:       {loss_odds_trend:>+7.3f} slope  |  {loss_odds_change:>+6.1f} pts change")
        print(f"  ")
        
        # Perfect storm analysis: 3PA up + odds improving
        perfect_storm = bets_df[(bets_df['trend_3pa'] > 0) & (bets_df['trend_odds'] > 0)]
        opposite = bets_df[(bets_df['trend_3pa'] < 0) & (bets_df['trend_odds'] < 0)]
        
        if len(perfect_storm) > 0:
            perfect_storm_wins = perfect_storm['won'].sum()
            perfect_storm_rate = (perfect_storm_wins / len(perfect_storm)) * 100
            print(f"  'Perfect Storm' (3PA ↑ + Odds ↑):")
            print(f"    Count: {len(perfect_storm)} bets  |  Win Rate: {perfect_storm_rate:.1f}%")
        
        if len(opposite) > 0:
            opposite_wins = opposite['won'].sum()
            opposite_rate = (opposite_wins / len(opposite)) * 100
            print(f"  'Worst Case' (3PA ↓ + Odds ↓):")
            print(f"    Count: {len(opposite)} bets  |  Win Rate: {opposite_rate:.1f}%")
        
        # Prediction accuracy analysis
        print(f"  ")
        print(f"  PREDICTION MODEL ACCURACY (Not used for betting yet):")
        print(f"  {'─'*76}")
        
        if 'predicted_3pm' in bets_df.columns:
            # Overall prediction accuracy
            pred_correct = bets_df['prediction_directionally_correct'].sum()
            pred_accuracy = (pred_correct / len(bets_df)) * 100
            
            # Average prediction errors
            avg_pred_3pa = bets_df['predicted_3pa'].mean()
            avg_actual_3pa = bets_df['actual_3pa'].mean()
            avg_pred_3pm = bets_df['predicted_3pm'].mean()
            avg_actual_3pm = bets_df['actual_threes'].mean()
            
            # MAE (Mean Absolute Error)
            mae_3pa = (bets_df['actual_3pa'] - bets_df['predicted_3pa']).abs().mean()
            mae_3pm = (bets_df['actual_threes'] - bets_df['predicted_3pm']).abs().mean()
            
            print(f"  Prediction Method: Mean 3PA from lookback × {LEAGUE_AVG_3PT_PCT:.0%} league avg")
            print(f"  Directional Accuracy: {pred_correct}/{len(bets_df)} ({pred_accuracy:.1f}%)")
            print(f"  ")
            print(f"  3PA Predictions:")
            print(f"    Avg Predicted: {avg_pred_3pa:.2f}  |  Avg Actual: {avg_actual_3pa:.2f}  |  MAE: {mae_3pa:.2f}")
            print(f"  ")
            print(f"  3PM Predictions:")
            print(f"    Avg Predicted: {avg_pred_3pm:.2f}  |  Avg Actual: {avg_actual_3pm:.2f}  |  MAE: {mae_3pm:.2f}")
        else:
            print(f"  Prediction data not available for this result set.")
        
        # Show example player journeys
        print(f"\n  {'─'*76}")
        print(f"  EXAMPLE PLAYER JOURNEY")
        print(f"  {'─'*76}")
        
        # Get one example (prefer a winner if available, otherwise take any)
        if len(bets_df) > 0:
            winners = bets_df[bets_df['won'] == True]
            if len(winners) > 0:
                example = winners.sample(1).iloc[0]
            else:
                example = bets_df.sample(1).iloc[0]
            
            display_player_journey(example)
    
    print(f"\n{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_streak_backtest(df, bet_direction='over', streak_lengths=[2, 3, 4, 5, 6, 7]):
    """
    Run streak-based backtest for either OVER or UNDER bets.
    
    Args:
        df: DataFrame with props and game results
        bet_direction: 'over' (bet overs after cold streaks) or 'under' (bet unders after hot streaks)
        streak_lengths: List of streak lengths to test
    
    Returns:
        Dictionary with results for each streak length
    """
    strategy_name = "MISS STREAK" if bet_direction == 'over' else "HIT STREAK"
    
    print("="*80)
    print(f"NBA 3PT PROP {strategy_name} BACKTEST")
    print("="*80)
    print()
    
    # Run simulation
    print("Running backtest simulation...")
    print(f"Bet Direction: {bet_direction.upper()}")
    if bet_direction == 'over':
        print("  Strategy: Bet OVER after MISS streaks (fade market on cold players)")
    else:
        print("  Strategy: Bet UNDER after HIT streaks (fade market on hot players)")
    print()
    
    results = simulate_streak_betting(
        df,
        streak_lengths=streak_lengths,
        check_odds_movement=True,
        bet_direction=bet_direction
    )
    
    # Display results
    display_results(results, bet_direction=bet_direction)
    
    return results


def overs_backtest(df):
    """
    Run backtest for OVER bets after cold streaks (miss streaks).
    
    Strategy: Bet OVER when player has missed their prop X games in a row.
    Theory: Regression to the mean after cold streak.
    
    Args:
        df: DataFrame with props and game results
    
    Returns:
        Dictionary with results for each streak length
    """
    return run_streak_backtest(df, bet_direction='over', streak_lengths=[2, 3, 4, 5, 6, 7])


def unders_backtest(df):
    """
    Run backtest for UNDER bets after hot streaks (hit streaks).
    
    Strategy: Bet UNDER when player has hit their prop X games in a row.
    Theory: Regression to the mean after hot streak.
    
    Args:
        df: DataFrame with props and game results
    
    Returns:
        Dictionary with results for each streak length
    """
    return run_streak_backtest(df, bet_direction='under', streak_lengths=[2, 3, 4, 5, 6, 7])


def filter_by_prediction_alignment(bets_df, bet_direction):
    """
    Filter bets to only include those where prediction aligns with strategy.
    
    For UNDER bets: Keep only where predicted_3pm < line
    For OVER bets: Keep only where predicted_3pm >= line
    
    Args:
        bets_df: DataFrame of bets
        bet_direction: 'over' or 'under'
    
    Returns:
        Filtered DataFrame
    """
    if 'predicted_3pm' not in bets_df.columns:
        return bets_df
    
    if bet_direction == 'under':
        # Under bets: keep where prediction says under (predicted < line)
        return bets_df[bets_df['predicted_3pm'] < bets_df['line']].copy()
    else:  # 'over'
        # Over bets: keep where prediction says over (predicted >= line)
        return bets_df[bets_df['predicted_3pm'] >= bets_df['line']].copy()


def calculate_filtered_stats(bets_df):
    """Calculate statistics for a set of bets"""
    if len(bets_df) == 0:
        return None
    
    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets * 100
    
    total_wagered = bets_df['bet_amount'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (total_profit / total_wagered) * 100
    
    avg_odds = bets_df['odds'].mean()
    
    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'roi': roi,
        'avg_odds': avg_odds,
        'bets_detail': bets_df
    }


def display_streak_examples(unfilt_result, filt_result, streak_length):
    """
    Display 4 example types for a streak length:
    1. Unfiltered bet that LOST
    2. Unfiltered bet that WON
    3. Filtered bet that WON (kept by prediction)
    4. Skipped bet (filtered out by prediction)
    """
    
    if 'bets_detail' not in unfilt_result or len(unfilt_result['bets_detail']) == 0:
        print(f"  No bets available for {streak_length}-hit streak")
        print()
        return
    
    unfilt_df = unfilt_result['bets_detail']
    filt_df = filt_result.get('bets_detail', pd.DataFrame()) if 'bets_detail' in filt_result else pd.DataFrame()
    
    # Find skipped bets (in unfiltered but not in filtered)
    if len(filt_df) > 0:
        filt_indices = set(filt_df.index)
        skipped_df = unfilt_df[~unfilt_df.index.isin(filt_indices)]
    else:
        skipped_df = unfilt_df.copy()
    
    # Get examples
    unfilt_loss = unfilt_df[unfilt_df['won'] == False].head(1) if len(unfilt_df[unfilt_df['won'] == False]) > 0 else pd.DataFrame()
    unfilt_win = unfilt_df[unfilt_df['won'] == True].head(1) if len(unfilt_df[unfilt_df['won'] == True]) > 0 else pd.DataFrame()
    filt_win = filt_df[filt_df['won'] == True].head(1) if len(filt_df) > 0 and len(filt_df[filt_df['won'] == True]) > 0 else pd.DataFrame()
    skipped = skipped_df.head(1) if len(skipped_df) > 0 else pd.DataFrame()
    
    def format_bet_example(row, category, is_skipped=False):
        if len(row) == 0:
            print(f"  {category}: None available")
            return
        
        bet = row.iloc[0]
        player = bet['player']
        line = bet['line']
        pred_3pm = bet.get('predicted_3pm', 0)
        actual_3pm = bet['actual_threes']
        odds = bet['odds']
        won = bet['won']
        profit = bet['profit']
        bet_direction = bet.get('bet_direction', 'UNDER')  # Default to UNDER for this analysis
        
        # Prediction signal
        if pred_3pm < line:
            pred_signal = f"UNDER (pred: {pred_3pm:.2f} < {line:.1f})"
            pred_emoji = "📉"
        else:
            pred_signal = f"OVER (pred: {pred_3pm:.2f} ≥ {line:.1f})"
            pred_emoji = "📈"
        
        # Actual result
        if actual_3pm < line:
            actual_signal = f"UNDER ({actual_3pm:.1f} < {line:.1f})"
            actual_emoji = "📉"
        else:
            actual_signal = f"OVER ({actual_3pm:.1f} ≥ {line:.1f})"
            actual_emoji = "📈"
        
        # Win/Loss - different formatting for skipped bets
        if is_skipped:
            if won:
                result_emoji = "✅ Would have WON"
                result_desc = f"(but filtered out - prediction conflicted)"
            else:
                result_emoji = "❌ Would have LOST"
                result_desc = f"(correctly filtered out!)"
        else:
            result_emoji = "✅ WIN" if won else "❌ LOSS"
            result_desc = ""
        
        print(f"  {category}:")
        print(f"    Player: {player} | Bet {bet_direction} {line:.1f} | Odds: {odds:+d}")
        print(f"    Prediction: {pred_emoji} {pred_signal}")
        print(f"    Actual: {actual_emoji} {actual_signal}")
        if is_skipped:
            print(f"    {result_emoji} | Would have profited: ${profit:+.2f} {result_desc}")
        else:
            print(f"    Result: {result_emoji} | Profit: ${profit:+.2f}")
    
    # Display all 4 examples
    format_bet_example(unfilt_loss, "1️⃣  UNFILTERED BET - LOST", is_skipped=False)
    print()
    format_bet_example(unfilt_win, "2️⃣  UNFILTERED BET - WON", is_skipped=False)
    print()
    format_bet_example(filt_win, "3️⃣  FILTERED BET - WON (Kept: prediction aligned)", is_skipped=False)
    print()
    format_bet_example(skipped, "4️⃣  SKIPPED BET (Filtered out: prediction conflicted)", is_skipped=True)
    print()
    print()


def underdog_unders_analysis(df):
    """
    Specialized analysis: UNDER bets with POSITIVE ODDS after hit streaks.
    
    Shows TWO comparisons:
    1. ALL positive odds bets (no prediction filter)
    2. PREDICTION-ALIGNED bets (only when prediction agrees with strategy)
    
    This filters for the sweet spot scenarios where:
    - Player is on a hot streak (hitting their over)
    - Market is still pricing the UNDER as an underdog (+ odds)
    - We bet UNDER expecting regression to the mean
    
    Tests streak lengths 2-10.
    
    Args:
        df: DataFrame with props and game results
    
    Returns:
        Tuple of (unfiltered_results, filtered_results)
    """
    print("="*80)
    print("UNDERDOG UNDERS ANALYSIS - WITH & WITHOUT PREDICTION FILTER")
    print("="*80)
    print()
    print("Strategy: Bet UNDER with POSITIVE ODDS after hit streaks")
    print("  - Only includes bets where under odds are positive (e.g., +100, +135)")
    print("  - Player is hot (hitting their over consecutively)")
    print("  - Market still favors the over (pricing under as underdog)")
    print("  - Testing streak lengths: 2-10 consecutive hits")
    print()
    print("📊 Comparing TWO approaches:")
    print("  1. ALL BETS (positive odds only)")
    print("  2. PREDICTION-ALIGNED BETS (positive odds + prediction says under)")
    print()
    
    # Run simulation for all streak lengths
    all_results = simulate_streak_betting(
        df,
        streak_lengths=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        check_odds_movement=True,
        bet_direction='under'
    )
    
    # Filter to only positive odds (unfiltered version)
    unfiltered_results = {}
    # Filter to positive odds + prediction alignment (filtered version)
    prediction_filtered_results = {}
    
    for streak_length, result in all_results.items():
        if 'bets_detail' in result:
            bets_df = result['bets_detail']
            
            # UNFILTERED: Only positive odds
            positive_odds_bets = bets_df[bets_df['odds'] > 0].copy()
            
            if len(positive_odds_bets) > 0:
                unfiltered_results[streak_length] = calculate_filtered_stats(positive_odds_bets)
            else:
                unfiltered_results[streak_length] = {
                    'total_bets': 0,
                    'message': 'No bets with positive odds found'
                }
            
            # PREDICTION-FILTERED: Positive odds + prediction alignment
            aligned_bets = filter_by_prediction_alignment(positive_odds_bets, 'under')
            
            if len(aligned_bets) > 0:
                prediction_filtered_results[streak_length] = calculate_filtered_stats(aligned_bets)
            else:
                prediction_filtered_results[streak_length] = {
                    'total_bets': 0,
                    'message': 'No prediction-aligned bets found'
                }
    
    # Display comparison summary
    print("\n" + "="*120)
    print("COMPARISON: ALL BETS vs PREDICTION-ALIGNED BETS")
    print("="*120)
    print()
    print(f"{'Streak':<8} | {'ALL BETS (No Filter)':<52} | {'PREDICTION-ALIGNED (Filtered)':<52}")
    print(f"{'Length':<8} | {'Count | Win% | Avg Odds | ROI':<52} | {'Count | Win% | Avg Odds | ROI':<52}")
    print("─" * 120)
    
    for streak_len in sorted(unfiltered_results.keys()):
        unfilt = unfiltered_results[streak_len]
        filt = prediction_filtered_results[streak_len]
        
        # Format unfiltered stats
        if 'roi' in unfilt:
            unfilt_str = f"{unfilt['total_bets']:>4} | {unfilt['win_rate']:>5.1f}% | {unfilt['avg_odds']:>+6.1f} | {unfilt['roi']:>+7.2f}%"
            if unfilt['roi'] > 0:
                unfilt_str += " ✅"
            else:
                unfilt_str += " ❌"
        else:
            unfilt_str = "No bets"
        
        # Format filtered stats
        if 'roi' in filt:
            filt_str = f"{filt['total_bets']:>4} | {filt['win_rate']:>5.1f}% | {filt['avg_odds']:>+6.1f} | {filt['roi']:>+7.2f}%"
            if filt['roi'] > 0:
                filt_str += " ✅"
            else:
                filt_str += " ❌"
            
            # Highlight if filtered is better
            if 'roi' in unfilt and filt['roi'] > unfilt['roi']:
                filt_str += " 📈"
        else:
            filt_str = "No bets"
        
        print(f"{streak_len:>2} hits  | {unfilt_str:<52} | {filt_str:<52}")
        
        # Show examples for this streak length
        print()
        print(f"  EXAMPLES FOR {streak_len}-HIT STREAK:")
        print(f"  {'─'*116}")
        
        display_streak_examples(unfilt, filt, streak_len)
    
    print()
    print("="*120)
    print()
    
    # Calculate overall totals
    print("OVERALL TOTALS:")
    print("─" * 120)
    
    # Unfiltered totals
    all_unfilt_bets = pd.concat([r['bets_detail'] for r in unfiltered_results.values() if 'bets_detail' in r])
    if len(all_unfilt_bets) > 0:
        unfilt_total_stats = calculate_filtered_stats(all_unfilt_bets)
        unfilt_profit = unfilt_total_stats['total_profit']
        print(f"ALL BETS: {unfilt_total_stats['total_bets']} bets | {unfilt_total_stats['win_rate']:.1f}% win | ROI: {unfilt_total_stats['roi']:+.2f}% | Total P&L: ${unfilt_profit:+,.2f}")
    
    # Filtered totals
    all_filt_bets = pd.concat([r['bets_detail'] for r in prediction_filtered_results.values() if 'bets_detail' in r])
    if len(all_filt_bets) > 0:
        filt_total_stats = calculate_filtered_stats(all_filt_bets)
        filt_profit = filt_total_stats['total_profit']
        print(f"PREDICTION-ALIGNED: {filt_total_stats['total_bets']} bets | {filt_total_stats['win_rate']:.1f}% win | ROI: {filt_total_stats['roi']:+.2f}% | Total P&L: ${filt_profit:+,.2f}")
        
        # Show improvement
        if len(all_unfilt_bets) > 0:
            roi_diff = filt_total_stats['roi'] - unfilt_total_stats['roi']
            profit_diff = filt_profit - unfilt_profit
            bets_filtered_out = unfilt_total_stats['total_bets'] - filt_total_stats['total_bets']
            pct_filtered = (bets_filtered_out / unfilt_total_stats['total_bets']) * 100
            print()
            print(f"📊 IMPACT OF PREDICTION FILTER:")
            print(f"   • Filtered out: {bets_filtered_out} bets ({pct_filtered:.1f}% of all bets)")
            print(f"   • ROI Improvement: {roi_diff:+.2f} percentage points")
            print(f"   • P&L Improvement: ${profit_diff:+,.2f} (from ${unfilt_profit:+,.2f} to ${filt_profit:+,.2f})")
            if roi_diff > 0:
                print(f"   • ✅ Prediction filter IMPROVES performance")
            else:
                print(f"   • ❌ Prediction filter HURTS performance")
    
    print()
    print("="*120)
    print()
    
    return unfiltered_results, prediction_filtered_results


def initial_analysis(df=None):
    """
    Run both OVERS and UNDERS streak backtests.
    
    Args:
        df: Optional pre-loaded DataFrame. If None, will load data.
        
    Returns:
        Tuple of (overs_results, unders_results)
    """
    
    print("="*80)
    print("NBA 3PT PROP STREAK BACKTEST - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print()
    
    # Load data if not provided
    if df is None:
        print("Loading cleaned data...")
        print("Source: data/consensus_props_with_game_results_min10_2024_25.csv")
        print("Created by: scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py")
        print()
        
        try:
            df = load_clean_data()
            print(f"✅ Loaded {len(df):,} props with game results")
            print(f"   Season: 2024-25")
            print(f"   Filtered: Games with >= 10 minutes played")
            print(f"   DNPs removed: Rows with NULL threes_made filtered out")
            print()
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            print("\nNext steps:")
            print("1. Run scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py")
            print("2. This will create the merged dataset with props + game results")
            return None, None
    
    # Run OVERS backtest
    print("\n" + "="*80)
    print("RUNNING: OVERS BACKTEST (Cold Streak Regression)")
    print("="*80 + "\n")
    overs_results = overs_backtest(df)
    
    # Run UNDERS backtest
    print("\n" * 3)
    print("="*80)
    print("RUNNING: UNDERS BACKTEST (Hot Streak Regression)")
    print("="*80 + "\n")
    unders_results = unders_backtest(df)
    
    # Display comparison summary
    print("\n" * 3)
    print("="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print()
    
    print("OVERS (Bet Over after Miss Streaks):")
    for streak_len in sorted(overs_results.keys()):
        result = overs_results[streak_len]
        if 'roi' in result:
            roi_emoji = "✅" if result['roi'] > 0 else "❌"
            print(f"  {streak_len} misses: {result['total_bets']:>4} bets | {result['win_rate']:>5.1f}% win | {result['roi']:>+7.2f}% ROI {roi_emoji}")
    
    print()
    print("UNDERS (Bet Under after Hit Streaks):")
    for streak_len in sorted(unders_results.keys()):
        result = unders_results[streak_len]
        if 'roi' in result:
            roi_emoji = "✅" if result['roi'] > 0 else "❌"
            print(f"  {streak_len} hits:   {result['total_bets']:>4} bets | {result['win_rate']:>5.1f}% win | {result['roi']:>+7.2f}% ROI {roi_emoji}")
    
    print()
    print("="*80)
    print()
    
    return overs_results, unders_results


def analyze_blind_under_betting_by_line(df):
    """
    Analyze blind under betting broken down by line value (0.5, 1.0, 1.5, ..., 5.0+).
    Shows monthly ROI, win rate, odds, and edge for each line value.
    
    Uses global TARGET_WIN config for bet sizing.
    """
    print("\n")
    print("="*120)
    print("🎯 BLIND UNDER BETTING - BREAKDOWN BY LINE VALUE")
    print("="*120)
    print()
    print("For each line value, showing monthly performance of betting UNDER on every prop.")
    print()
    
    # Define line value bins
    line_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    for line in line_values:
        # Filter to this line value AND exclude extreme odds
        df_line = df[
            (df['consensus_line'] == line) &
            (df['under_best_odds'].notna()) &
            (df['under_best_odds'] > MAX_ODDS_THRESHOLD)
        ].copy()
        
        if len(df_line) == 0:
            continue
        
        print("="*120)
        print(f"📊 LINE VALUE: {line}")
        print("="*120)
        print()
        
        # Calculate under results using TARGET_WIN for bet sizing
        df_line['under_wins'] = df_line['threes_made'] < df_line['consensus_line']
        df_line['bet_amount'] = df_line['under_best_odds'].apply(lambda odds: calculate_bet_amount(odds, TARGET_WIN))
        df_line['profit'] = df_line.apply(
            lambda row: calculate_profit(row['under_best_odds'], row['bet_amount']) if row['under_wins']
            else -row['bet_amount'],
            axis=1
        )
        
        # Overall stats for this line
        total_bets = len(df_line)
        total_wins = df_line['under_wins'].sum()
        total_profit = df_line['profit'].sum()
        total_wagered = df_line['bet_amount'].sum()
        win_rate = (total_wins / total_bets * 100)
        roi = (total_profit / total_wagered * 100)
        
        # Calculate average implied probability the RIGHT way
        df_line['impl_prob'] = df_line['under_best_odds'].apply(american_odds_to_probability)
        avg_impl_prob = df_line['impl_prob'].mean()
        avg_odds = probability_to_american_odds(avg_impl_prob)
        edge = win_rate - avg_impl_prob
        
        print(f"Overall Stats for Line {line}:")
        print(f"  Total Bets: {total_bets:,}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  ROI: {roi:+.2f}%")
        print(f"  Avg Odds: {int(avg_odds):+d}")
        print(f"  Avg Impl Prob: {avg_impl_prob:.1f}%")
        print(f"  Edge: {edge:+.1f}%")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print()
        
        # Monthly breakdown for this line
        df_line['month'] = pd.to_datetime(df_line['date']).dt.to_period('M')
        
        monthly_stats = []
        for month in sorted(df_line['month'].unique()):
            month_df = df_line[df_line['month'] == month]
            
            bets = len(month_df)
            wins = month_df['under_wins'].sum()
            total_profit_month = month_df['profit'].sum()
            total_wagered_month = month_df['bet_amount'].sum()
            win_rate_month = (wins / bets * 100)
            roi_month = (total_profit_month / total_wagered_month * 100)
            
            # Calculate average implied probability the RIGHT way
            month_df['impl_prob_month'] = month_df['under_best_odds'].apply(american_odds_to_probability)
            avg_impl_prob_month = month_df['impl_prob_month'].mean()
            avg_odds_month = probability_to_american_odds(avg_impl_prob_month)
            
            monthly_stats.append({
                'Month': str(month),
                'ROI': roi_month,
                'Win Rate': win_rate_month,
                'Bets': bets,
                'Avg Odds': avg_odds_month,
                'Avg Impl Prob': avg_impl_prob_month,
                'Edge': win_rate_month - avg_impl_prob_month
            })
        
        monthly_df = pd.DataFrame(monthly_stats)
        
        # Pretty print
        print(f"{'Month':<12} | {'ROI':<10} | {'Win Rate':<10} | {'Bets':<6} | {'Avg Odds':<10} | {'Avg Impl Prob':<15} | {'Edge':<8}")
        print("-" * 90)
        
        for _, row in monthly_df.iterrows():
            roi_str = f"{row['ROI']:>+6.2f}%"
            if row['ROI'] > 0:
                roi_str += " ✅"
            else:
                roi_str += " ❌"
            
            win_rate_str = f"{row['Win Rate']:>5.1f}%"
            odds_str = f"{int(row['Avg Odds']):>+4d}"
            impl_prob_str = f"{row['Avg Impl Prob']:>6.1f}%"
            edge_str = f"{row['Edge']:>+5.1f}%"
            
            print(f"{row['Month']:<12} | {roi_str:<12} | {win_rate_str:<10} | {row['Bets']:<6} | {odds_str:<10} | {impl_prob_str:<15} | {edge_str:<8}")
        
        print()
        print()


def analyze_player_season(df, player_name=None, min_games=70):
    """
    Analyze a single player's complete season to see betting opportunities in context.
    
    Args:
        df: Merged dataframe with props and game results
        player_name: Specific player to analyze (or None to find one with most games)
        min_games: Minimum games with prop data to qualify
    
    Returns:
        Player name analyzed
    """
    print("="*80)
    print("FULL SEASON PLAYER ANALYSIS")
    print("="*80)
    print()
    
    # Determine if prop was hit
    df['hit_over'] = df['threes_made'] >= df['consensus_line']
    df['missed_over'] = ~df['hit_over']
    
    # Find player with most complete season
    if player_name is None:
        player_games = df.groupby('player').size().sort_values(ascending=False)
        # Find player with at least min_games
        qualifying_players = player_games[player_games >= min_games]
        
        if len(qualifying_players) == 0:
            print(f"❌ No players found with >= {min_games} games")
            return None
        
        player_name = qualifying_players.index[0]
        print(f"Selected player with most complete data: {player_name}")
        print(f"Games with prop data: {qualifying_players.iloc[0]}")
    else:
        player_games_count = len(df[df['player'] == player_name])
        print(f"Analyzing: {player_name}")
        print(f"Games with prop data: {player_games_count}")
    
    print()
    
    # Get player's season
    player_df = df[df['player'] == player_name].sort_values('date').reset_index(drop=True)
    
    if len(player_df) == 0:
        print(f"❌ No data found for {player_name}")
        return None
    
    print("="*80)
    print(f"{player_name.upper()}'S 2024-25 SEASON")
    print("="*80)
    print()
    
    # Season stats
    total_games = len(player_df)
    total_overs = player_df['hit_over'].sum()
    total_unders = total_games - total_overs
    over_pct = (total_overs / total_games) * 100
    
    avg_line = player_df['consensus_line'].mean()
    avg_3pm = player_df['threes_made'].mean()
    avg_3pa = player_df['threes_attempted'].mean()
    
    print(f"Season Overview:")
    print(f"  Total Games: {total_games}")
    print(f"  Overs: {total_overs} ({over_pct:.1f}%)")
    print(f"  Unders: {total_unders} ({100-over_pct:.1f}%)")
    print(f"  Avg Line: {avg_line:.2f}")
    print(f"  Avg 3PM: {avg_3pm:.2f}")
    print(f"  Avg 3PA: {avg_3pa:.2f}")
    print()
    
    # Track streaks and betting opportunities
    current_miss_streak = 0
    current_hit_streak = 0
    betting_opportunities = []
    
    print("="*120)
    print(f"{'Game':<5} | {'Date':<12} | {'Opponent':<12} | {'Line':<5} | {'3PM':<5} | {'3PA':<5} | {'Result':<18} | {'Streak':<20} | {'Betting Opportunity':<25}")
    print("="*120)
    
    for i, row in player_df.iterrows():
        game_num = i + 1
        date = row['date']
        opponent = row.get('opponent', 'N/A')
        line = row['consensus_line']
        threes_made = row['threes_made']
        threes_attempted = row['threes_attempted']
        hit = row['hit_over']
        
        # Result
        if hit:
            result = f"✅ OVER ({threes_made}/{threes_attempted})"
        else:
            result = f"❌ UNDER ({threes_made}/{threes_attempted})"
        
        # Update streaks
        if hit:
            current_hit_streak += 1
            current_miss_streak = 0
        else:
            current_miss_streak += 1
            current_hit_streak = 0
        
        # Format streak info
        if current_miss_streak > 0:
            streak_info = f"🔴 {current_miss_streak} miss"
            if current_miss_streak > 1:
                streak_info += "es"
        elif current_hit_streak > 0:
            streak_info = f"🟢 {current_hit_streak} hit"
            if current_hit_streak > 1:
                streak_info += "s"
        else:
            streak_info = "Start"
        
        # Check for betting opportunities (show both OVER and UNDER opportunities)
        bet_opp = ""
        if current_miss_streak >= 2:
            # We'd bet OVER on the NEXT game after this one
            if i + 1 < len(player_df):
                bet_opp = f"→ BET OVER (after {current_miss_streak} misses)"
                betting_opportunities.append({
                    'game_num': game_num + 1,
                    'after_streak': current_miss_streak,
                    'type': 'OVER_AFTER_MISS'
                })
        elif current_hit_streak >= 2:
            # We'd bet UNDER on the NEXT game
            if i + 1 < len(player_df):
                bet_opp = f"→ BET UNDER (after {current_hit_streak} hits)"
                betting_opportunities.append({
                    'game_num': game_num + 1,
                    'after_streak': current_hit_streak,
                    'type': 'UNDER_AFTER_HIT'
                })
        
        print(f"{game_num:<5} | {date:<12} | {opponent:<12} | {line:<5.1f} | {threes_made:<5.1f} | {threes_attempted:<5.1f} | {result:<18} | {streak_info:<20} | {bet_opp:<25}")
    
    print("="*120)
    print()
    
    # Summary of betting opportunities
    print("="*80)
    print("BETTING OPPORTUNITIES SUMMARY")
    print("="*80)
    print()
    
    if len(betting_opportunities) > 0:
        print(f"Total betting opportunities: {len(betting_opportunities)}")
        print()
        
        # Group by type and streak length
        from collections import defaultdict
        type_counts = defaultdict(lambda: defaultdict(int))
        
        for opp in betting_opportunities:
            opp_type = opp['type']
            streak_len = opp['after_streak']
            type_counts[opp_type][streak_len] += 1
        
        # Display OVER opportunities (after miss streaks)
        if 'OVER_AFTER_MISS' in type_counts:
            print("OVER Betting Opportunities (after miss streaks):")
            for streak_len in sorted(type_counts['OVER_AFTER_MISS'].keys()):
                count = type_counts['OVER_AFTER_MISS'][streak_len]
                print(f"  After {streak_len} misses: {count} opportunities")
            print()
        
        # Display UNDER opportunities (after hit streaks)
        if 'UNDER_AFTER_HIT' in type_counts:
            print("UNDER Betting Opportunities (after hit streaks):")
            for streak_len in sorted(type_counts['UNDER_AFTER_HIT'].keys()):
                count = type_counts['UNDER_AFTER_HIT'][streak_len]
                print(f"  After {streak_len} hits: {count} opportunities")
    else:
        print("No betting opportunities found.")
    
    print()
    
    return player_name


def blind_under_betting_analysis(df):
    """
    Analyze profitability of blindly betting UNDER on every prop.
    
    Uses global TARGET_WIN config for bet sizing.
    This tests the hypothesis that unders are profitable in aggregate.
    """
    print("="*80)
    print("BLIND UNDER BETTING ANALYSIS")
    print("="*80)
    print()
    print(f"Strategy: Bet to win ${TARGET_WIN} on UNDER for EVERY prop posted")
    print("Using: Best under odds available")
    print()
    
    # Calculate results
    df = df.copy()
    df['hit_over'] = df['threes_made'] >= df['consensus_line']
    df['under_wins'] = ~df['hit_over']
    
    # Filter to only rows with valid odds AND exclude extreme outliers
    # Exclude odds worse than MAX_ODDS_THRESHOLD (unrealistic, typically centers who never shoot 3s)
    df_valid = df[
        df['under_best_odds'].notna() & 
        (df['under_best_odds'] > MAX_ODDS_THRESHOLD)
    ].copy()
    
    total_excluded = len(df[df['under_best_odds'].notna()]) - len(df_valid)
    
    print(f"Total props analyzed: {len(df_valid):,}")
    if total_excluded > 0:
        print(f"  (Excluded {total_excluded} props with extreme odds worse than {MAX_ODDS_THRESHOLD})")
    print()
    
    # Calculate bet outcomes using TARGET_WIN
    def calculate_under_profit(row):
        odds = row['under_best_odds']
        won = row['under_wins']
        bet_amount = calculate_bet_amount(odds, TARGET_WIN)
        
        if won:
            profit = calculate_profit(odds, bet_amount)
        else:
            profit = -bet_amount
        
        return profit
    
    df_valid['bet_amount'] = df_valid['under_best_odds'].apply(lambda odds: calculate_bet_amount(odds, TARGET_WIN))
    df_valid['profit'] = df_valid.apply(calculate_under_profit, axis=1)
    df_valid['cumulative_profit'] = df_valid['profit'].cumsum()
    
    # Overall stats
    total_bets = len(df_valid)
    total_wins = df_valid['under_wins'].sum()
    total_losses = total_bets - total_wins
    win_rate = (total_wins / total_bets) * 100
    
    total_wagered = df_valid['bet_amount'].sum()
    total_profit = df_valid['profit'].sum()
    roi = (total_profit / total_wagered) * 100
    
    # Calculate average odds properly (convert to probabilities first, then average)
    df_valid['implied_prob'] = df_valid['under_best_odds'].apply(american_odds_to_probability)
    avg_implied_prob = df_valid['implied_prob'].mean()
    median_odds = df_valid['under_best_odds'].median()
    median_implied_prob = american_odds_to_probability(median_odds)
    
    print("="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print()
    print(f"Total Bets:      {total_bets:,}")
    print(f"Wins:            {total_wins:,} ({win_rate:.2f}%)")
    print(f"Losses:          {total_losses:,} ({100-win_rate:.2f}%)")
    print()
    print(f"Total Wagered:   ${total_wagered:,.2f}")
    print(f"Total Profit:    ${total_profit:,.2f}")
    print(f"ROI:             {roi:.2f}%")
    print()
    print(f"Median Odds:     {median_odds:+.0f} (Impl. Prob: {median_implied_prob:.1f}%)")
    print(f"Avg Implied Prob: {avg_implied_prob:.2f}%")
    print()
    
    # Calculate edge
    print(f"Actual Win Rate: {win_rate:.2f}%")
    print(f"Edge: {win_rate - avg_implied_prob:+.2f}%")
    print()
    
    # Show profit curve over time
    print("="*80)
    print("PROFIT OVER TIME (First 20 Games)")
    print("="*80)
    print()
    print(f"{'Game':<6} | {'Date':<12} | {'Player':<20} | {'Line':<6} | {'Result':<18} | {'Profit':<10} | {'Cumulative':<12}")
    print("-" * 100)
    
    for i, row in df_valid.head(20).iterrows():
        game_num = i + 1
        date = row['date']
        player = row['player'][:20]  # Truncate long names
        line = row['consensus_line']
        actual = row['threes_made']
        profit = row['profit']
        cum_profit = row['cumulative_profit']
        
        if row['under_wins']:
            result = f"✅ U ({actual} < {line})"
        else:
            result = f"❌ O ({actual} ≥ {line})"
        
        print(f"{game_num:<6} | {date:<12} | {player:<20} | {line:<6.1f} | {result:<18} | ${profit:<9.2f} | ${cum_profit:<11.2f}")
    
    print()
    print(f"... ({len(df_valid) - 20:,} more games)")
    print()
    
    # Final cumulative
    print(f"Final Cumulative Profit: ${df_valid['cumulative_profit'].iloc[-1]:,.2f}")
    print()
    
    # Best and worst days
    print("="*80)
    print("BEST & WORST SINGLE BETS")
    print("="*80)
    print()
    
    best_bet = df_valid.loc[df_valid['profit'].idxmax()]
    worst_bet = df_valid.loc[df_valid['profit'].idxmin()]
    
    print("Best Win:")
    print(f"  {best_bet['player']} on {best_bet['date']}")
    print(f"  Line: {best_bet['consensus_line']:.1f}, Actual: {best_bet['threes_made']:.0f}")
    print(f"  Odds: {best_bet['under_best_odds']:+.0f}, Profit: ${best_bet['profit']:.2f}")
    print()
    
    print("Worst Loss:")
    print(f"  {worst_bet['player']} on {worst_bet['date']}")
    print(f"  Line: {worst_bet['consensus_line']:.1f}, Actual: {worst_bet['threes_made']:.0f}")
    print(f"  Odds: {worst_bet['under_best_odds']:+.0f}, Profit: ${worst_bet['profit']:.2f}")
    print()
    
    # Monthly breakdown
    print("="*80)
    print("MONTHLY BREAKDOWN - THE TREND")
    print("="*80)
    print()
    
    df_valid['month'] = pd.to_datetime(df_valid['date']).dt.to_period('M')
    
    # Calculate detailed monthly stats
    monthly_stats = []
    for month in df_valid['month'].unique():
        month_df = df_valid[df_valid['month'] == month]
        
        bets = len(month_df)
        wins = month_df['under_wins'].sum()
        total_profit = month_df['profit'].sum()
        total_wagered_month = month_df['bet_amount'].sum()
        win_rate = (wins / bets * 100)
        roi = (total_profit / total_wagered_month * 100)
        
        # Calculate average implied probability
        # Step 1: Convert each odds to implied probability
        month_df['impl_prob'] = month_df['under_best_odds'].apply(american_odds_to_probability)
        
        # Step 2: Take the mean of implied probabilities
        avg_implied_prob = month_df['impl_prob'].mean()
        
        # Step 3: Convert that back to American odds
        avg_odds = probability_to_american_odds(avg_implied_prob)
        
        monthly_stats.append({
            'Month': str(month),
            'ROI': roi,
            'Win Rate': win_rate,
            'Bets': bets,
            'Avg Odds': avg_odds,
            'Avg Impl Prob': avg_implied_prob,
            'Edge': win_rate - avg_implied_prob
        })
    
    monthly_df = pd.DataFrame(monthly_stats)
    # Sort by month chronologically
    monthly_df['month_sort'] = pd.to_datetime(monthly_df['Month'].astype(str))
    monthly_df = monthly_df.sort_values('month_sort')
    
    # Pretty print the table
    print(f"{'Month':<12} | {'ROI':<8} | {'Win Rate':<10} | {'Bets':<6} | {'Avg Odds':<10} | {'Avg Impl Prob':<15} | {'Edge':<8}")
    print("-" * 90)
    
    for _, row in monthly_df.iterrows():
        roi_str = f"{row['ROI']:>+6.2f}%"
        if row['ROI'] > 0:
            roi_str += " ✅"
        else:
            roi_str += " ❌"
        
        win_rate_str = f"{row['Win Rate']:>5.1f}%"
        odds_str = f"{int(row['Avg Odds']):>+4d}"
        impl_prob_str = f"{row['Avg Impl Prob']:>6.1f}%"
        edge_str = f"{row['Edge']:>+5.1f}%"
        
        print(f"{row['Month']:<12} | {roi_str:<10} | {win_rate_str:<10} | {row['Bets']:<6} | {odds_str:<10} | {impl_prob_str:<15} | {edge_str:<8}")
    
    print()
    
    # Summary observation
    print("📊 KEY OBSERVATION:")
    print()
    positive_months = monthly_df[monthly_df['ROI'] > 0]
    negative_months = monthly_df[monthly_df['ROI'] < 0]
    
    if len(positive_months) > 0 and len(negative_months) > 0:
        avg_roi_positive = positive_months['ROI'].mean()
        avg_roi_negative = negative_months['ROI'].mean()
        avg_edge_positive = positive_months['Edge'].mean()
        avg_edge_negative = negative_months['Edge'].mean()
        
        print(f"   Profitable months ({len(positive_months)}): Avg ROI = {avg_roi_positive:+.2f}%, Avg Edge = {avg_edge_positive:+.1f}%")
        print(f"   Losing months ({len(negative_months)}): Avg ROI = {avg_roi_negative:+.2f}%, Avg Edge = {avg_edge_negative:+.1f}%")
        print()
        
        # Find transition point chronologically
        if len(positive_months) >= 1:
            # Check if there's a clear transition from positive to negative
            profitable_period_ended = False
            last_profitable_month = None
            first_losing_month = None
            
            for i, row in monthly_df.iterrows():
                if row['ROI'] > 0:
                    last_profitable_month = row['Month']
                elif row['ROI'] < 0 and last_profitable_month is not None and not profitable_period_ended:
                    first_losing_month = row['Month']
                    profitable_period_ended = True
                    break
            
            if last_profitable_month and first_losing_month:
                first_profitable_month = positive_months.iloc[0]['Month']
                print(f"   💡 Strategy was profitable {first_profitable_month} through {last_profitable_month}")
                print(f"   ⚠️  Turned unprofitable starting {first_losing_month}")
                print(f"   → Market likely adjusted to the inefficiency!")
                print()
                
                # Additional insights
                avg_odds_early = monthly_df[monthly_df['ROI'] > 0]['Avg Odds'].mean()
                avg_odds_late = monthly_df[monthly_df['ROI'] < 0]['Avg Odds'].mean()
                
                print(f"   📈 Profitable months: Avg odds = {int(avg_odds_early):+d}")
                print(f"   📉 Losing months: Avg odds = {int(avg_odds_late):+d}")
                print(f"   🔍 Odds got {abs(avg_odds_late - avg_odds_early):.0f} points worse (books adjusted)")
    
    print()
    
    # Top/Bottom performers
    print("="*80)
    print("TOP 10 MOST PROFITABLE PLAYERS (For Betting Under)")
    print("="*80)
    print()
    
    player_stats = df_valid.groupby('player').agg({
        'profit': ['count', 'sum', 'mean'],
        'under_wins': 'sum'
    }).round(2)
    
    player_stats.columns = ['Bets', 'Total Profit', 'Avg Profit', 'Wins']
    player_stats['Win Rate'] = (player_stats['Wins'] / player_stats['Bets'] * 100).round(1)
    player_stats = player_stats[player_stats['Bets'] >= 10]  # At least 10 bets
    player_stats = player_stats.sort_values('Total Profit', ascending=False)
    
    print(f"{'Player':<25} | {'Bets':<6} | {'Wins':<6} | {'Win Rate':<10} | {'Total Profit':<15}")
    print("-" * 80)
    for player, row in player_stats.head(10).iterrows():
        print(f"{player:<25} | {row['Bets']:<6.0f} | {row['Wins']:<6.0f} | {row['Win Rate']:<10.1f}% | ${row['Total Profit']:<14.2f}")
    
    print()
    print("="*80)
    print("BOTTOM 10 PLAYERS (Worst for Betting Under)")
    print("="*80)
    print()
    
    print(f"{'Player':<25} | {'Bets':<6} | {'Wins':<6} | {'Win Rate':<10} | {'Total Profit':<15}")
    print("-" * 80)
    for player, row in player_stats.tail(10).iterrows():
        print(f"{player:<25} | {row['Bets']:<6.0f} | {row['Wins']:<6.0f} | {row['Win Rate']:<10.1f}% | ${row['Total Profit']:<14.2f}")
    
    print()
    
    return df_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NBA 3PT Prop Streak Backtest - Multiple Analysis Options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses
  python %(prog)s --all
  
  # Run only the main streak analysis (overs + unders)
  python %(prog)s --streak
  
  # Run only underdog unders analysis
  python %(prog)s --underdog-unders
  
  # Run multiple specific analyses
  python %(prog)s --streak --underdog-unders --player-season
  
  # Run only blind under analyses
  python %(prog)s --blind-under --blind-under-by-line
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses (default if no flags specified)')
    parser.add_argument('--streak', action='store_true',
                       help='Run main streak analysis (OVERS + UNDERS backtest)')
    parser.add_argument('--underdog-unders', action='store_true',
                       help='Run underdog unders analysis (positive odds only, streaks 2-10)')
    parser.add_argument('--player-season', action='store_true',
                       help='Run player season analysis (full season view with bet opportunities)')
    parser.add_argument('--blind-under', action='store_true',
                       help='Run blind under betting analysis (bet every under)')
    parser.add_argument('--blind-under-by-line', action='store_true',
                       help='Run blind under betting by line value (0.5, 1.0, 1.5, etc.)')
    
    args = parser.parse_args()
    
    # If no flags specified, run all
    if not any([args.all, args.streak, args.underdog_unders, args.player_season, 
                args.blind_under, args.blind_under_by_line]):
        args.all = True
    
    # Load data once
    print("Loading cleaned data...")
    print("Source: data/consensus_props_with_game_results_min10_2024_25.csv")
    print()
    
    df = load_clean_data()
    print(f"✅ Loaded {len(df):,} props with game results")
    print(f"   Season: 2024-25")
    print(f"   Filtered: Games with >= 10 minutes played")
    print()
    
    # Run requested analyses
    if args.all or args.streak:
        initial_analysis(df)
    
    if args.all or args.underdog_unders:
        if not (args.all or args.streak):
            # Add spacing if not first analysis
            print("\n" * 3)
        else:
            print("\n" * 3)
        underdog_unders_analysis(df)
    
    if args.all or args.player_season:
        print("\n" * 3)
        analyze_player_season(df)
    
    if args.all or args.blind_under:
        print("\n" * 3)
        blind_under_betting_analysis(df)
    
    if args.all or args.blind_under_by_line:
        print("\n" * 3)
        analyze_blind_under_betting_by_line(df)
    
    
    
    