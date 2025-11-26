"""
NBA 3-Point Prop Prediction Model - Beat the Books
===================================================

Date: 2025-11-24
Author: Myles Thomas

GOAL: Build a predictive model that is MORE ACCURATE than bookmakers at predicting
      whether a player will go over/under their 3PM line.

INITIAL FOCUS: 0.5 Lines Only (Low-Volume Shooters, Less Liquid Markets)
=========================================================================

Why start with 0.5 lines?
  1. LESS LIQUID MARKETS
     - These props are for players who rarely shoot 3s (bigs, defensive specialists)
     - Less sharp money, less efficient pricing
     - Books may be slower to adjust vs high-profile star player props
  
  2. SIMPLER MODELING
     - Binary outcome: 0 makes vs 1+ makes
     - Easier to predict than higher lines with more variance
     - Good starting point to validate approach before expanding
  
  3. MARKET INEFFICIENCY POTENTIAL
     - Low-volume props get less attention from professional bettors
     - Books may rely more on automated pricing vs manual adjustment
     - Historical data suggests these markets have exploitable patterns
  
  4. CLEAR SIGNAL
     - For players who attempt 0-2 threes per game, patterns are more stable
     - Less noise from game-to-game variance
     - Contextual factors (opponent, home/away) may matter more

Once model proves effective on 0.5 lines, expand to: [0.5, 1.0, 1.5, 2.0, etc.]

APPROACH: Multi-Stage Probabilistic Model
==========================================

STAGE 1: Predict 3-Point Attempts (3PA) Distribution
-----------------------------------------------------
Instead of predicting a single point estimate, model the DISTRIBUTION of attempts.

Input Features:
  • Recent games 3PA (rolling windows: 3, 5, 10, 15 games)
  • Home/Away splits
  • Opponent defensive rating vs 3PA
  • Back-to-back game indicator
  • Minutes played trends
  • Team pace of play
  • Injury status / lineup changes
  
Distribution Options:
  A. Normal Distribution: N(μ, σ²)
     - μ = weighted average of recent 3PA (more weight to recent)
     - σ = standard deviation of recent 3PA
  
  B. Poisson Distribution: Poisson(λ)
     - λ = expected 3PA based on rolling average
     - Better for count data (attempts are discrete)
  
  C. Negative Binomial: NB(r, p)
     - Allows for overdispersion (variance > mean)
     - More flexible than Poisson for high-variance players
  
  D. Empirical Bootstrap:
     - Resample from player's historical 3PA distribution
     - Non-parametric, captures actual distribution shape

STAGE 2: Predict 3-Point Makes (3PM) Given Attempts
----------------------------------------------------
For each possible attempt count, model the probability of makes.

Approach A: Binomial Model (Recommended)
  • For each 3PA value, 3PM ~ Binomial(n=3PA, p=3PT%)
  • p = player's SEASON 3PT% (with league avg prior), NOT recent games
  • KEY INSIGHT: Separate treatment for 3PA vs 3PT%
    - 3PA (from recent games) = captures current ROLE/OPPORTUNITY
    - 3PT% (from season data) = captures true SKILL/TALENT
    - Recent shooting % = pure NOISE for low-volume shooters
  • This naturally captures:
    - Volume (more attempts = higher variance)
    - Efficiency (player's actual shooting ability)
    - Shot distribution shape (binomial accounts for discrete makes)

Approach B: Bayesian Hierarchical Model
  • Prior: Player's career 3PT%
  • Likelihood: Recent games performance
  • Posterior: Updated shooting probability
  • Better for players with limited sample size

Features to Consider:
  • SEASON 3PT% (not recent games - too noisy for low-volume shooters)
  • Career 3PT% (potential future enhancement)
  • Opponent 3P% defense
  • Home/Away splits
  • Rest days
  • Shot quality metrics (if available)
  
Bayesian Shooting % Estimation:
  • Use Beta-Binomial conjugate prior model
  • Prior: Beta(α, β) where α/(α+β) = LEAGUE_AVG_3PT_PCT (35%)
  • Strength: α+β = PRIOR_STRENGTH (default: 200 virtual attempts)
  • Posterior: Beta(α + makes, β + misses) after observing season data
  • Higher strength = more conservative, prevents extreme predictions
  • TODO: Optimize PRIOR_STRENGTH empirically via backtesting

STAGE 3: Generate 3PM Distribution via Monte Carlo
---------------------------------------------------
Combine Stage 1 and Stage 2 to get full 3PM distribution:

1. Sample 3PA from distribution (e.g., 10,000 simulations)
2. For each 3PA sample, simulate 3PM ~ Binomial(3PA, player_3PT%)
3. Result: Distribution of possible 3PM outcomes

Example:
  Player with mean 6 3PA, 35% 3PT%:
  - Simulation 1: Sample 7 attempts → Binomial(7, 0.35) → 2 makes
  - Simulation 2: Sample 5 attempts → Binomial(5, 0.35) → 1 make
  - Simulation 3: Sample 8 attempts → Binomial(8, 0.35) → 3 makes
  - ... repeat 10,000 times
  - Result: P(0 makes), P(1 make), P(2 makes), ... P(8+ makes)

STAGE 4: Compare to Betting Line
---------------------------------
Given line (e.g., 2.5), calculate model probabilities:
  • P(Over) = P(3PM ≥ 2.5) = P(3PM ≥ 3)
  • P(Under) = P(3PM < 2.5) = P(3PM ≤ 2)

For 0.5 lines (CURRENT FOCUS):
  • P(Over 0.5) = P(3PM ≥ 1) = P(1 make) + P(2 makes) + P(3 makes) + ...
  • P(Under 0.5) = P(3PM = 0) = P(zero makes)
  • Simplified: This is just asking "will player make at least one 3?"
  
  Direct calculation without simulation:
    - P(0 makes) = (1 - p)^n  where p = 3PT%, n = attempts
    - But n is variable, so we still need the distribution approach
    - Our model estimates the probability better than books by modeling:
      1. Attempt distribution (how many shots will they take?)
      2. Make probability given attempts (what's their shooting %?)

Convert to implied odds:
  • If P(Over) = 60%, implied odds = -150
  • If P(Under) = 40%, implied odds = +150

STAGE 5: Find Edges vs Market
------------------------------
Compare model probability to market odds:

Edge Detection:
  • Market Odds: Convert bookmaker odds to implied probability
  • Model Probability: From our Monte Carlo simulation
  • Edge = Model_Prob - Market_Prob
  
Bet When:
  • Edge > Threshold (e.g., 5% edge)
  • Sample size sufficient (min games played)
  • Confidence interval doesn't overlap with market

VALIDATION METRICS
==================

Accuracy Metrics:
  1. Directional Accuracy: % of times we predict correct side (over/under)
     - Goal: Beat 52.4% breakeven (typical -110 odds)
     - Target: 55%+ for profitability
  
  2. Brier Score: Mean squared error of probability predictions
     - Measures calibration quality
     - Lower is better
     - Compare our Brier score to implied market probabilities
  
  3. Log Loss: Logarithmic loss for probability predictions
     - Penalizes confident wrong predictions heavily
     - Compare to market's log loss
  
  4. Calibration Plot:
     - Bin predictions by probability (0-10%, 10-20%, etc.)
     - Plot predicted prob vs actual frequency
     - Perfect calibration = diagonal line
  
  5. ROI by Confidence Level:
     - High confidence bets (>60% prob) should be more profitable
     - Track ROI for different probability thresholds

Comparison vs Books:
  • Side-by-side accuracy: Our model vs market favorite
  • Profit curves: Betting our model vs betting market favorite
  • Edge realization: Do our edges convert to profit?

DATA STRUCTURE
==============

Props Data (The Odds API - Historical Props):
  Source: data/01_input/the-odds-api/historical_props/props_YYYY-MM-DD_player_threes.csv
  
  Columns:
    - player: Player name (e.g., "Tyrese Maxey")
    - game: Game matchup (e.g., "Miami Heat @ Philadelphia 76ers")
    - game_time: Game start time (ISO format)
    - market: Always "player_threes"
    - line: The 3PM line (e.g., 2.5, 3.5)
    - bookmaker: Sportsbook name (e.g., "fanduel", "draftkings")
    - over_odds: American odds for over (e.g., -120, +150)
    - under_odds: American odds for under (e.g., -110, +130)
  
  Notes:
    - Multiple rows per player per game (one per bookmaker)
    - Need to aggregate to consensus line and best odds
    - Files span from October 2024 to November 2025 (full season)

Game Logs (NBA API - Season Game Logs):
  Source: data/01_input/nba_api/season_game_logs/2024_25/{Player_Name}.csv
  
  Columns:
    - player: Player name
    - date: Game date (YYYY-MM-DD)
    - matchup: Team vs/@ opponent (e.g., "DEN @ HOU")
    - minutes: Minutes played
    - threes_made: Actual 3PM in game
    - threes_attempted: Actual 3PA in game
    - three_pct: 3PT% for that game
    - home_away: "HOME" or "AWAY"
    - opponent: Opponent team abbreviation
    - plus_minus: +/- for the game
    - [Other stats: fgm, fga, pts, reb, ast, etc.]
  
  Notes:
    - One CSV file per player
    - One row per game
    - Chronologically ordered (most recent first)
    - Includes games with DNPs (minutes = 0)

Data Preparation Steps:
  1. Load all props files, aggregate to get:
     - Consensus line (median or mode across bookmakers)
     - Best over odds (most favorable for bettor)
     - Best under odds (most favorable for bettor)
  
  2. Load all game log CSVs into single dataframe
     - Filter out DNPs (minutes < 10)
     - Normalize player names to match props
  
  3. Merge props + game logs on (player, date)
     - Left join on props (each prop becomes a row)
     - Keep only games where we have both prop and result
  
  4. Sort by date chronologically
     - Critical for time-series modeling
     - Train on past, test on future
  
  5. Feature engineering per player:
     - Rolling 3PA averages (3, 5, 10, 15 game windows)
     - Rolling 3PT% (3, 5, 10, 15 game windows)
     - Rolling 3PA std dev (for distribution modeling)
     - Days rest since last game
     - Back-to-back indicator
     - Home/Away
     - [Future: opponent defense, pace, etc.]

IMPLEMENTATION PLAN
====================

EXISTING INFRASTRUCTURE (Already Built):
  ✓ Merged dataset exists: data/03_intermediate/consensus_props_with_game_results_min10_2024_25.csv
  ✓ Created by: scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py
  ✓ Contains: props + game results, normalized player names, filtered DNPs
  ✓ load_clean_data() function available in backtesting/20251121_nba_3pt_prop_miss_streaks_24_25.py

Phase 1: Baseline Model (SIMPLE - START HERE)
----------------------------------------------
Functions to implement:

1. load_clean_data()
   - Copy from 20251121 file or import
   - Returns merged props + game results

2. add_rolling_features(df, lookback_windows=[3, 5, 10, 15])
   - For each player, calculate:
     * rolling_3pa_mean_{window}
     * rolling_3pa_std_{window}
     * rolling_3pt_pct_{window}
     * rolling_3pm_{window}
   - Handle minimum sample size (need at least 3 games)
   - Return df with new columns

3. predict_3pa_distribution(player_row, distribution='normal')
   - Input: Single row with player's rolling features
   - Output: Distribution object (scipy.stats)
   - Normal: N(μ=rolling_3pa_mean_5, σ=rolling_3pa_std_5)
   - Could also try: Poisson, Negative Binomial

4. predict_3pm_distribution(attempts_dist, shooting_pct, n_simulations=10000)
   - Input: 3PA distribution, player's 3PT%
   - Monte Carlo simulation:
     a. Sample 3PA from distribution (10k times)
     b. For each 3PA, sample 3PM ~ Binomial(3PA, shooting_pct)
   - Output: Dictionary {0: prob, 1: prob, 2: prob, ...}
   
5. calculate_line_probabilities(makes_dist, line)
   - Input: 3PM distribution, betting line (e.g., 2.5)
   - Output: (prob_over, prob_under)
   - prob_over = sum(makes_dist[k] for k >= line)
   - prob_under = sum(makes_dist[k] for k < line)

6. predict_single_game(player_row, line, lookback=5)
   - Combines steps 3-5
   - Input: Player data, betting line
   - Output: Dictionary with:
     * predicted_3pa_mean
     * predicted_3pa_std
     * predicted_3pm_distribution
     * prob_over
     * prob_under
     * model_edge_over (vs market)
     * model_edge_under (vs market)

7. backtest_predictions(df, lookback=5, min_games=5)
   - For each prop in dataset:
     a. Skip if player has < min_games history
     b. Get rolling features from PRIOR games only (no lookahead bias!)
     c. Make prediction for this game
     d. Compare to actual result
     e. Track directional accuracy
   - Output: DataFrame with predictions + results

8. evaluate_model(predictions_df)
   - Calculate metrics:
     * Directional accuracy
     * Brier score
     * Log loss
     * Calibration plot data
     * ROI if we bet on model edges
   - Compare to market performance
   - Output: Summary statistics + plots

9. compare_to_market(predictions_df)
   - Market accuracy: How often does market favorite win?
   - Model accuracy: How often does our prediction win?
   - Edge realization: Do our edges convert to profit?
   - Output: Side-by-side comparison

Phase 2: Enhanced Features (AFTER BASELINE WORKS)
--------------------------------------------------
10. add_contextual_features(df)
    - Home/away indicator
    - Days rest (date diff from previous game)
    - Back-to-back indicator (0 days rest)
    - Minutes trend (rolling average minutes)
    - Usage patterns (attempts per minute)

11. weighted_rolling_features(df, decay=0.95)
    - Exponentially weighted averages (recent games matter more)
    - weight_i = decay^i (more recent = higher weight)

12. adjust_for_sample_size(prediction, n_games, regression_target=league_avg)
    - Regression to mean for small samples
    - If player only has 3 games, blend with league average
    - weight = n_games / (n_games + prior_strength)

Phase 3: Advanced Distributions (EXPERIMENTAL)
-----------------------------------------------
13. fit_poisson_3pa(player_history)
    - Poisson distribution: better for count data
    - λ = mean attempts

14. fit_negative_binomial_3pa(player_history)
    - Allows overdispersion (high variance players)
    - Fit using method of moments or MLE

15. bayesian_shooting_pct(player_history, prior_mean=0.35, prior_strength=20)
    - Prior: League average 3PT% with weight
    - Posterior: Weighted average of prior + player data
    - Better for players with limited data

Phase 4: Model Optimization
----------------------------
16. train_test_split_by_date(df, train_pct=0.7)
    - Split chronologically (not random!)
    - Train on first 70% of season
    - Test on last 30%

17. optimize_hyperparameters(df, param_grid)
    - Test different lookback windows
    - Test different distributions
    - Test different rolling window weights
    - Use validation set to pick best

18. ensemble_predictions(predictions_list)
    - Combine multiple models
    - Weighted average by historical accuracy

Phase 5: Betting Strategy (PROFIT MAXIMIZATION)
------------------------------------------------
19. kelly_bet_sizing(edge, odds, bankroll, kelly_fraction=0.25)
    - Optimal bet size based on edge
    - edge = model_prob - market_prob
    - Fractional Kelly for safety (25% of full Kelly)

20. filter_high_confidence_bets(predictions_df, min_edge=0.05, min_prob=0.55)
    - Only bet when edge > threshold
    - Only bet when confident (>55% probability)

21. simulate_betting_bankroll(predictions_df, starting_bankroll=10000)
    - Simulate season with bankroll management
    - Track growth over time
    - Calculate Sharpe ratio, max drawdown

SUCCESS CRITERIA (for 0.5 lines)
==================================

Minimum Viable Product:
  • Model accuracy > 52.4% (breakeven at typical -110 odds)
  • Positive ROI on backtest (any positive profit)
  • Brier score < market implied Brier score (better calibrated than books)
  • Sample size: At least 100 props to validate

Target Goals:
  • Model accuracy > 55% (strong profitability threshold)
  • ROI > 5% on backtest
  • Consistent month-to-month performance (not just one lucky month)
  • Edge holds up in out-of-sample test period (last 20% of season)

Stretch Goals (after 0.5 lines work):
  • Expand to other lines (1.0, 1.5, 2.0, etc.)
  • Accuracy > 55% across all lines
  • Works for different player archetypes:
    - Non-shooters (0-1 attempts/game)
    - Occasional shooters (1-3 attempts/game)  
    - Volume shooters (3+ attempts/game)
  • Build live betting tool for daily predictions

CURRENT STREAK-BASED LOGIC (To Compare Against)
================================================
The existing code in this file uses a simpler streak-based approach:
  • After X consecutive misses → bet OVER (regression to mean)
  • After X consecutive hits → bet UNDER (regression to mean)
  • Simple prediction: mean(recent_3PA) × league_avg_3PT%

The new model should BEAT this naive approach!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.odds_utils import (
    calculate_bet_amount, 
    calculate_profit, 
    odds_to_implied_probability,
    american_odds_to_percentage_probability,
    probability_to_american_odds
)


# ============================================================================
# CONFIG
# ============================================================================

# Line Filtering - FOCUS ON LESS LIQUID MARKETS
LST_LINES_OF_INTEREST = [0.5]  # Only analyze 0.5 lines (low-volume shooters, less liquid markets)
# Why focus on 0.5 lines?
#   - These are for players who rarely shoot 3s (typically big men, defensive specialists)
#   - Less liquid markets = less sharp action = potential market inefficiency
#   - Easier to model: binary outcome (0 makes vs 1+ makes)
#   - Books may be slower to adjust these lines vs high-volume star players
# To expand: LST_LINES_OF_INTEREST = [0.5, 1.0, 1.5] etc.

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

# Modeling Parameters
LEAGUE_AVG_3PT_PCT = 0.35  # NBA league-wide 3P% (~35% as of 2024-25 season)
                            # Used as BAYESIAN PRIOR for shooting % prediction
MIN_LOOKBACK_GAMES = 5  # Minimum games of history required to make a prediction
LOOKBACK_WINDOW = 10  # Number of recent games to use for rolling features
N_SIMULATIONS = 10000  # Monte Carlo simulations for 3PM distribution

# Bayesian Prior Parameters for Shooting % (Beta-Binomial Model)
# KEY: We use SEASON AVERAGE for 3PT%, not recent games
# Recent shooting % is noise; season average captures true skill
#
# Using Beta(α, β) prior where:
#   - Mean = α/(α+β) = LEAGUE_AVG_3PT_PCT
#   - Strength = α+β = total virtual attempts
#
# TODO: Optimize PRIOR_STRENGTH empirically via backtesting
#       Test values: [50, 100, 200, 500, 1000]
#       Find strength that maximizes accuracy/ROI
PRIOR_STRENGTH = 200  # Virtual attempts (high = conservative, prevents extreme predictions)
                      # This means prior has weight of ~200 attempts at league avg
                      # Higher strength = more regression to mean = safer predictions

# Old streak-based config (not used in new model, kept for compatibility)
FILTER_3PA_TREND = None  # Options: 'up', 'down', None
FILTER_ODDS_TREND = None  # Options: 'up', 'down', None

# Prediction approach for baseline model:
#   1. Calculate rolling mean 3PA from lookback period (e.g., last 10 games)
#   2. Calculate rolling 3PT% from lookback period
#   3. Model 3PA as Normal distribution: N(μ=mean_3pa, σ=std_3pa)
#   4. For each simulated 3PA, model 3PM as Binomial(n=3PA, p=player_3PT%)
#   5. Aggregate simulations to get 3PM probability distribution
#   6. Compare to line (e.g., 0.5) to get P(over) and P(under)
#   7. Find edge vs market odds

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

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
        implied_probs = bets_df['odds'].apply(american_odds_to_percentage_probability)
        avg_implied_prob = implied_probs.mean()
        
        # Categorize odds and calculate avg implied prob for each
        favorites_mask = bets_df['odds'] < 0
        underdogs_mask = bets_df['odds'] > 0
        
        favorites = favorites_mask.sum()
        underdogs = underdogs_mask.sum()
        
        # Calculate average implied probability for favorites and underdogs
        if favorites > 0:
            fav_avg_impl_prob = bets_df[favorites_mask]['odds'].apply(american_odds_to_percentage_probability).mean()
        else:
            fav_avg_impl_prob = 0
        
        if underdogs > 0:
            dog_avg_impl_prob = bets_df[underdogs_mask]['odds'].apply(american_odds_to_percentage_probability).mean()
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
        df_line['impl_prob'] = df_line['under_best_odds'].apply(american_odds_to_percentage_probability)
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
            month_df['impl_prob_month'] = month_df['under_best_odds'].apply(american_odds_to_percentage_probability)
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
    df_valid['implied_prob'] = df_valid['under_best_odds'].apply(american_odds_to_percentage_probability)
    avg_implied_prob = df_valid['implied_prob'].mean()
    median_odds = df_valid['under_best_odds'].median()
    median_implied_prob = american_odds_to_percentage_probability(median_odds)
    
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
        month_df['impl_prob'] = month_df['under_best_odds'].apply(american_odds_to_percentage_probability)
        
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


# ============================================================================
# NEW MODELING APPROACH - 3PA/3PM PREDICTION MODEL
# ============================================================================

def load_filled_player_data(season='2024_25'):
    """
    Load filled player data (includes DNPs) from fill_player_missing_games.py output.
    
    This data includes:
    - All games the player's team played (not just games they played)
    - DNP rows marked with dnp=True, minutes=0, threes_made=0, etc.
    - Allows for better modeling of player availability and streak patterns
    
    Args:
        season: Season to load (e.g., '2024_25')
    
    Returns:
        DataFrame with all player game logs (including DNPs)
    """
    filled_data_dir = Path(__file__).parent.parent / 'data' / '03_intermediate' / 'player_level_data_filled' / season
    
    if not filled_data_dir.exists():
        raise FileNotFoundError(
            f"Filled player data directory not found: {filled_data_dir}\n"
            f"Run scripts/fill_player_missing_games.py --all to generate it."
        )
    
    print(f"Loading filled player data from: {filled_data_dir}")
    
    # Load all player CSVs
    csv_files = list(filled_data_dir.glob("*.csv"))
    print(f"  Found {len(csv_files)} player files")
    
    all_data = []
    for csv_file in csv_files:
        player_df = pd.read_csv(csv_file)
        all_data.append(player_df)
    
    # Concatenate all players
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by player and date
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    
    print(f"  ✓ Loaded {len(df):,} total game logs")
    print(f"  ✓ {len(df[df['dnp'] == True]):,} DNP games")
    print(f"  ✓ {len(df[df['dnp'] != True]):,} games played")
    print()
    
    return df


def load_props_data():
    """
    Load consensus props data (betting lines + odds).
    
    Returns:
        DataFrame with props data
    """
    props_file = Path(__file__).parent.parent / 'data' / '03_intermediate' / 'consensus_props_with_game_results_min10_2024_25.csv'
    
    if not props_file.exists():
        raise FileNotFoundError(
            f"Props data file not found: {props_file}\n"
            f"Run scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py to generate it."
        )
    
    df = pd.read_csv(props_file)
    
    # Filter out rows where player didn't play
    df = df[df['threes_made'].notna()].copy()
    
    return df


def identify_half_point_line_players(props_df, threshold=0.70, min_props=10):
    """
    Identify players whose PRIMARY line is 0.5 (low-volume shooters).
    
    We want players who consistently get 0.5 lines, not stars with occasional 0.5 alt lines.
    
    Args:
        props_df: DataFrame with props data (must have 'player' and 'consensus_line' columns)
        threshold: Minimum % of props that must be at 0.5 (default: 70%)
        min_props: Minimum number of props to qualify (default: 10)
    
    Returns:
        List of player names who are "true" 0.5 line players
    """
    print("="*80)
    print("IDENTIFYING 0.5 LINE PLAYERS")
    print("="*80)
    print()
    print(f"Criteria:")
    print(f"  • At least {threshold:.0%} of props at 0.5 line")
    print(f"  • Minimum {min_props} total props")
    print()
    
    # Calculate % of props at 0.5 for each player
    player_line_stats = props_df.groupby('player').agg({
        'consensus_line': [
            'count',
            lambda x: (x == 0.5).sum(),
            lambda x: (x == 0.5).sum() / len(x)
        ]
    }).reset_index()
    
    player_line_stats.columns = ['player', 'total_props', 'props_at_half', 'pct_at_half']
    
    # Filter to players meeting criteria
    qualified_players = player_line_stats[
        (player_line_stats['pct_at_half'] >= threshold) &
        (player_line_stats['total_props'] >= min_props)
    ].copy()
    
    qualified_players = qualified_players.sort_values('pct_at_half', ascending=False)
    
    print(f"Found {len(qualified_players)} players meeting criteria:")
    print()
    print(f"{'Player':<30} | {'Total Props':<12} | {'Props @ 0.5':<12} | {'% @ 0.5':<10}")
    print("-" * 70)
    
    for _, row in qualified_players.head(20).iterrows():
        print(f"{row['player']:<30} | {row['total_props']:<12.0f} | {row['props_at_half']:<12.0f} | {row['pct_at_half']:<10.1%}")
    
    if len(qualified_players) > 20:
        print(f"... and {len(qualified_players) - 20} more players")
    
    print()
    print(f"✓ {len(qualified_players)} players identified as '0.5 line players'")
    print()
    
    return qualified_players['player'].tolist()


def merge_filled_data_with_props(filled_df, props_df, half_point_players=None):
    """
    Merge filled player data (with DNPs) with props data.
    
    This creates a complete dataset with:
    - Props data (lines, odds) for games where props were posted
    - Game results (3PM, 3PA, minutes) including DNPs
    - Only includes "true" 0.5 line players if specified
    
    Args:
        filled_df: DataFrame from load_filled_player_data()
        props_df: DataFrame from load_props_data()
        half_point_players: List of player names to include (or None for all)
    
    Returns:
        Merged DataFrame
    """
    print("="*80)
    print("MERGING FILLED DATA WITH PROPS")
    print("="*80)
    print()
    
    # Filter to half-point players if specified
    if half_point_players is not None:
        print(f"Filtering to {len(half_point_players)} identified 0.5 line players...")
        filled_df = filled_df[filled_df['player'].isin(half_point_players)].copy()
        props_df = props_df[props_df['player'].isin(half_point_players)].copy()
        print(f"  ✓ Filled data: {len(filled_df):,} rows")
        print(f"  ✓ Props data: {len(props_df):,} rows")
        print()
    
    # Filter props to only 0.5 lines
    props_df = props_df[props_df['consensus_line'] == 0.5].copy()
    print(f"Filtered props to 0.5 lines only: {len(props_df):,} props")
    print()
    
    # Merge on player + date
    # Left join on props (we want all props, with filled data providing context)
    merged = props_df.merge(
        filled_df,
        on=['player', 'date'],
        how='left',
        suffixes=('_prop', '_filled')
    )
    
    print(f"Merged dataset: {len(merged):,} rows")
    print(f"  • Players: {merged['player'].nunique()}")
    print(f"  • Date range: {merged['date'].min()} to {merged['date'].max()}")
    print()
    
    return merged


def load_clean_data():
    """
    Load cleaned and merged dataset with props + game results.
    
    This dataset was created by scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py
    and contains:
    - Consensus prop lines (from historical_props/consensus_props_player_threes.csv)
    - Actual game results (3PM, 3PA, minutes from NBA API)
    - Pre-filtered: Only games where player played >= 10 minutes
    - Player names normalized for accurate matching
    
    Returns:
        DataFrame with merged props and game results
    """
    clean_data_file = Path(__file__).parent.parent / 'data' / '03_intermediate' / 'consensus_props_with_game_results_min10_2024_25.csv'
    
    if not clean_data_file.exists():
        raise FileNotFoundError(
            f"Clean data file not found: {clean_data_file}\n"
            f"Run scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py to generate it."
        )
    
    df = pd.read_csv(clean_data_file)
    
    # Filter out DNP situations (where player didn't play that game)
    initial_count = len(df)
    df = df[df['threes_made'].notna()].copy()
    dnp_count = initial_count - len(df)
    
    if dnp_count > 0:
        print(f"   Filtered out {dnp_count:,} DNP situations (player didn't play)")
    
    # Filter to only lines of interest (0.5 for now)
    df = df[df['consensus_line'].isin(LST_LINES_OF_INTEREST)].copy()
    print(f"   Filtered to lines: {LST_LINES_OF_INTEREST} → {len(df):,} props")
    
    # Sort by player and date for time-series analysis
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    
    return df


def get_player_history_excluding_dnps(df, player_name, date, lookback_games=10):
    """
    Get player's recent game history EXCLUDING DNPs.
    
    This is for distribution fitting - we only want games where they actually played.
    
    Args:
        df: DataFrame with filled player data (includes DNP rows)
        player_name: Player name
        date: Current game date (we want history BEFORE this date)
        lookback_games: Number of games to look back
    
    Returns:
        DataFrame with player's recent games (DNPs excluded)
    """
    player_history = df[
        (df['player'] == player_name) &
        (df['date'] < date) &
        (df['dnp'] == False) &  # ← EXCLUDE DNPs
        (df['minutes'] > 0)      # Double-check they actually played
    ].tail(lookback_games).copy()
    
    return player_history


def add_rolling_features_excluding_dnps(df, lookback_windows=None):
    """
    Add rolling statistical features for each player, EXCLUDING DNP games.
    
    DNPs are excluded because:
    - We model "Given player plays, how many 3PA/3PM?"
    - DNPs with 0 attempts would artificially lower means
    - They would inflate variance (mixing 0s with real attempts)
    
    For each lookback window, calculates:
    - rolling_3pa_mean: Average attempts (games played only)
    - rolling_3pa_std: Standard deviation of attempts (games played only)
    - rolling_3pt_pct: Shooting percentage (games played only)
    - rolling_3pm_mean: Average makes (games played only)
    - rolling_zero_makes_pct: Frequency of 0 makes (for 0.5 lines)
    
    Args:
        df: DataFrame with player game data (must be sorted by player, date)
        lookback_windows: List of window sizes (default: [5, 10])
    
    Returns:
        DataFrame with added rolling feature columns
    """
    if lookback_windows is None:
        lookback_windows = [5, 10]
    
    df = df.copy()
    
    print(f"\nCalculating rolling features (excluding DNPs) for windows: {lookback_windows}")
    
    for window in lookback_windows:
        print(f"  Window {window}...", end=' ')
        
        # Create mask for non-DNP games
        played_mask = (df['dnp'] == False) & (df['minutes'] > 0)
        
        # For each player, calculate rolling stats ONLY over games they played
        def rolling_stat_excluding_dnps(group, stat_col, agg_func, min_periods=1):
            """Helper to calculate rolling stat excluding DNPs"""
            result = pd.Series(index=group.index, dtype=float)
            
            for idx in group.index:
                # Get games before this one where player actually played
                prior_games = group.loc[:idx].iloc[:-1]  # Exclude current game
                prior_played = prior_games[prior_games['dnp'] == False]
                
                if len(prior_played) >= min_periods:
                    recent = prior_played.tail(window)
                    result.loc[idx] = agg_func(recent[stat_col])
                else:
                    result.loc[idx] = np.nan
            
            return result
        
        # Apply rolling stats per player
        df[f'rolling_3pa_mean_{window}'] = df.groupby('player', group_keys=False).apply(
            lambda g: rolling_stat_excluding_dnps(g, 'threes_attempted', np.mean, min_periods=1)
        ).values
        
        df[f'rolling_3pa_std_{window}'] = df.groupby('player', group_keys=False).apply(
            lambda g: rolling_stat_excluding_dnps(g, 'threes_attempted', lambda x: np.std(x, ddof=1), min_periods=2)
        ).values
        
        df[f'rolling_3pm_mean_{window}'] = df.groupby('player', group_keys=False).apply(
            lambda g: rolling_stat_excluding_dnps(g, 'threes_made', np.mean, min_periods=1)
        ).values
        
        # Calculate 3PT% from recent games
        def calc_rolling_3pt_pct(group):
            result = pd.Series(index=group.index, dtype=float)
            for idx in group.index:
                prior_games = group.loc[:idx].iloc[:-1]
                prior_played = prior_games[prior_games['dnp'] == False]
                if len(prior_played) >= 1:
                    recent = prior_played.tail(window)
                    total_makes = recent['threes_made'].sum()
                    total_attempts = recent['threes_attempted'].sum()
                    result.loc[idx] = total_makes / total_attempts if total_attempts > 0 else np.nan
                else:
                    result.loc[idx] = np.nan
            return result
        
        df[f'rolling_3pt_pct_{window}'] = df.groupby('player', group_keys=False).apply(
            calc_rolling_3pt_pct
        ).values
        
        # For 0.5 lines: frequency of getting 0 makes (in games played)
        def calc_rolling_zero_pct(group):
            result = pd.Series(index=group.index, dtype=float)
            for idx in group.index:
                prior_games = group.loc[:idx].iloc[:-1]
                prior_played = prior_games[prior_games['dnp'] == False]
                if len(prior_played) >= 1:
                    recent = prior_played.tail(window)
                    zero_count = (recent['threes_made'] == 0).sum()
                    result.loc[idx] = zero_count / len(recent)
                else:
                    result.loc[idx] = np.nan
            return result
        
        df[f'rolling_zero_makes_pct_{window}'] = df.groupby('player', group_keys=False).apply(
            calc_rolling_zero_pct
        ).values
        
        print("✓")
    
    # Count of historical GAMES PLAYED (excluding DNPs)
    df['n_games_played_history'] = df.groupby('player').apply(
        lambda g: (g['dnp'] == False).cumsum()
    ).values
    
    print(f"  ✓ Rolling features calculated (DNPs excluded)")
    
    return df


def add_availability_features(df):
    """
    Add features that capture DNP patterns and availability.
    
    These features help model:
    - Is player coming back from injury?
    - Is player being load managed?
    - How fresh/tired is player?
    
    Features added:
    - days_since_last_played: Days between games PLAYED (not team games)
    - games_played_last_5: # of team games where player played (out of 5)
    - games_played_last_10: # of team games where player played (out of 10)
    - prev_game_dnp: Binary flag if previous team game was DNP
    - consecutive_games_played: Streak of consecutive games played
    
    Args:
        df: DataFrame with filled player data (includes DNP rows)
    
    Returns:
        DataFrame with added availability features
    """
    df = df.copy()
    
    print("\nCalculating availability features...")
    
    # Days since last game PLAYED
    def calc_days_since_last_played(group):
        result = pd.Series(index=group.index, dtype=float)
        group['date_dt'] = pd.to_datetime(group['date'])
        
        for idx in group.index:
            prior_games = group.loc[:idx].iloc[:-1]
            prior_played = prior_games[prior_games['dnp'] == False]
            
            if len(prior_played) > 0:
                last_played_date = prior_played.iloc[-1]['date_dt']
                current_date = group.loc[idx, 'date_dt']
                days_diff = (current_date - last_played_date).days
                result.loc[idx] = days_diff
            else:
                result.loc[idx] = np.nan
        
        return result
    
    df['days_since_last_played'] = df.groupby('player', group_keys=False).apply(
        calc_days_since_last_played
    ).values
    
    # Games played in last N team games
    for n in [5, 10]:
        def calc_games_played_last_n(group):
            result = pd.Series(index=group.index, dtype=float)
            for idx in group.index:
                prior_games = group.loc[:idx].iloc[:-1]
                if len(prior_games) >= 1:
                    recent = prior_games.tail(n)
                    played_count = (recent['dnp'] == False).sum()
                    result.loc[idx] = played_count
                else:
                    result.loc[idx] = np.nan
            return result
        
        df[f'games_played_last_{n}'] = df.groupby('player', group_keys=False).apply(
            calc_games_played_last_n
        ).values
    
    # Previous game was DNP?
    df['prev_game_dnp'] = df.groupby('player')['dnp'].shift(1).fillna(False).astype(bool)
    
    # Consecutive games played streak
    def calc_consecutive_streak(group):
        result = pd.Series(index=group.index, dtype=int)
        streak = 0
        for idx in group.index:
            prior_games = group.loc[:idx].iloc[:-1]
            if len(prior_games) == 0:
                result.loc[idx] = 0
            else:
                # Count backwards from most recent
                streak = 0
                for i in range(len(prior_games) - 1, -1, -1):
                    if prior_games.iloc[i]['dnp'] == False:
                        streak += 1
                    else:
                        break
                result.loc[idx] = streak
        return result
    
    df['consecutive_games_played'] = df.groupby('player', group_keys=False).apply(
        calc_consecutive_streak
    ).values
    
    print("  ✓ Availability features calculated")
    
    return df


def add_rolling_features(df, lookback_windows=None):
    """
    DEPRECATED: Use add_rolling_features_excluding_dnps() instead.
    
    This version includes DNPs which can corrupt distribution estimates.
    Kept for backwards compatibility with old code.
    """
    if lookback_windows is None:
        lookback_windows = [5, 10, 15]
    
    df = df.copy()
    
    print(f"\n⚠️  WARNING: Using old add_rolling_features() which includes DNPs")
    print(f"   Consider using add_rolling_features_excluding_dnps() instead")
    print(f"\nCalculating rolling features for windows: {lookback_windows}")
    
    for window in lookback_windows:
        print(f"  Window {window}...", end=' ')
        
        # Group by player and calculate rolling stats
        df[f'rolling_3pa_mean_{window}'] = df.groupby('player')['threes_attempted'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        df[f'rolling_3pa_std_{window}'] = df.groupby('player')['threes_attempted'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
        )
        
        df[f'rolling_3pm_mean_{window}'] = df.groupby('player')['threes_made'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Calculate 3PT% from rolling makes and attempts
        rolling_makes = df.groupby('player')['threes_made'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
        )
        rolling_attempts = df.groupby('player')['threes_attempted'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
        )
        
        df[f'rolling_3pt_pct_{window}'] = rolling_makes / rolling_attempts
        
        # For 0.5 lines, also track frequency of getting 0 vs 1+ makes
        df[f'rolling_zero_makes_pct_{window}'] = df.groupby('player')['threes_made'].transform(
            lambda x: (x.shift(1).rolling(window=window, min_periods=1).apply(lambda vals: (vals == 0).sum() / len(vals)))
        )
        
        print("✓")
    
    # Count of historical games available (for filtering minimum sample size)
    df['n_games_history'] = df.groupby('player').cumcount()
    
    return df


def fit_poisson_3pa(player_3pa_history):
    """
    Fit a Poisson distribution to player's 3PA history.
    
    Poisson assumes variance = mean (equidispersion).
    Good for players with consistent, stable attempt patterns.
    
    Args:
        player_3pa_history: Array-like of historical 3PA values
    
    Returns:
        Dictionary with:
        - 'distribution': scipy.stats.poisson object
        - 'mean': Expected 3PA (λ parameter)
        - 'variance': Variance (equal to mean for Poisson)
        - 'type': 'poisson'
        - 'lambda': λ parameter
    """
    from scipy import stats
    
    attempts = np.array(player_3pa_history)
    
    if len(attempts) == 0:
        return None
    
    mean_3pa = np.mean(attempts)
    
    # For Poisson, variance = mean (by definition)
    # But we still calculate empirical variance for comparison
    empirical_var = np.var(attempts, ddof=1) if len(attempts) > 1 else mean_3pa
    
    # Create Poisson distribution with λ = mean
    poisson_dist = stats.poisson(mu=mean_3pa)
    
    return {
        'distribution': poisson_dist,
        'mean': mean_3pa,
        'variance': mean_3pa,  # Theoretical variance (= λ)
        'empirical_variance': empirical_var,  # Actual variance from data
        'type': 'poisson',
        'lambda': mean_3pa
    }


def fit_negative_binomial_3pa(player_3pa_history):
    """
    Fit a Negative Binomial distribution to player's 3PA history.
    
    Negative Binomial is better than Poisson for count data with overdispersion
    (variance > mean), which is common in basketball shot attempts.
    
    Uses method of moments to estimate parameters:
    - r (number of successes)
    - p (probability of success)
    
    Args:
        player_3pa_history: Array-like of historical 3PA values
    
    Returns:
        Dictionary with:
        - 'distribution': scipy.stats.nbinom object
        - 'mean': Expected 3PA
        - 'variance': Variance of 3PA
        - 'r': NB parameter r
        - 'p': NB parameter p
    """
    from scipy import stats
    
    attempts = np.array(player_3pa_history)
    
    if len(attempts) == 0:
        return None
    
    mean_3pa = np.mean(attempts)
    var_3pa = np.var(attempts, ddof=1) if len(attempts) > 1 else mean_3pa
    
    # Handle edge case: if variance <= mean, use Poisson instead
    # (Negative Binomial requires variance > mean)
    if var_3pa <= mean_3pa or var_3pa == 0:
        # Fall back to Poisson
        return {
            'distribution': stats.poisson(mu=mean_3pa),
            'mean': mean_3pa,
            'variance': mean_3pa,
            'type': 'poisson_fallback',
            'lambda': mean_3pa,
            'note': 'Variance <= mean, used Poisson instead'
        }
    
    # Method of moments estimation for Negative Binomial
    # For NB: E[X] = r*p/(1-p), Var[X] = r*p/(1-p)^2
    # Solving: p = mean/var, r = mean^2/(var - mean)
    p = mean_3pa / var_3pa
    r = (mean_3pa ** 2) / (var_3pa - mean_3pa)
    
    # Create scipy negative binomial distribution
    # Note: scipy uses n (successes) and p (success probability)
    nb_dist = stats.nbinom(n=r, p=1-p)  # scipy uses different parameterization
    
    return {
        'distribution': nb_dist,
        'mean': mean_3pa,
        'variance': var_3pa,
        'type': 'negative_binomial',
        'r': r,
        'p': p
    }


def compare_distributions_single_player(filled_df, player_name, lookback_window=20, visualize=True):
    """
    Compare Poisson vs Negative Binomial distribution fits for a single player.
    
    Shows:
    - Histogram of actual 3PA (excluding DNPs)
    - Poisson fit overlay
    - Negative Binomial fit overlay
    - Goodness-of-fit metrics
    - Predicted 3PM distributions for 0.5 line
    
    Args:
        filled_df: DataFrame with filled player data
        player_name: Player name to analyze
        lookback_window: Number of games to use for fitting
        visualize: If True, show plots
    
    Returns:
        Dictionary with comparison results
    """
    print("="*80)
    print(f"DISTRIBUTION COMPARISON: {player_name}")
    print("="*80)
    print()
    
    # Get player's history (EXCLUDE DNPs)
    player_data = filled_df[
        (filled_df['player'] == player_name) &
        (filled_df['dnp'] == False) &
        (filled_df['minutes'] > 0)
    ].copy()
    
    if len(player_data) < 10:
        print(f"❌ Not enough data for {player_name} ({len(player_data)} games)")
        return None
    
    # Use recent games for fitting
    recent_data = player_data.tail(lookback_window)
    attempts_history = recent_data['threes_attempted'].values.astype(int)
    makes_history = recent_data['threes_made'].values.astype(int)
    
    print(f"Player: {player_name}")
    print(f"Games analyzed: {len(recent_data)} (last {lookback_window} games played)")
    print(f"Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
    print()
    
    # Calculate empirical stats
    mean_3pa = np.mean(attempts_history)
    var_3pa = np.var(attempts_history, ddof=1)
    mean_3pm = np.mean(makes_history)
    empirical_3pt_pct = makes_history.sum() / attempts_history.sum() if attempts_history.sum() > 0 else 0
    
    print(f"Empirical 3PA Stats:")
    print(f"  Mean: {mean_3pa:.2f}")
    print(f"  Variance: {var_3pa:.2f}")
    print(f"  Variance/Mean ratio: {var_3pa/mean_3pa:.2f} ({'overdispersed' if var_3pa > mean_3pa else 'underdispersed'})")
    print()
    
    print(f"Empirical 3PM Stats:")
    print(f"  Mean: {mean_3pm:.2f}")
    print(f"  3PT%: {empirical_3pt_pct:.1%}")
    print(f"  Zero makes: {(makes_history == 0).sum()}/{len(makes_history)} ({(makes_history == 0).sum()/len(makes_history):.1%})")
    print()
    
    # Fit distributions
    poisson_fit = fit_poisson_3pa(attempts_history)
    nb_fit = fit_negative_binomial_3pa(attempts_history)
    
    print(f"Distribution Fits:")
    print(f"  Poisson: λ = {poisson_fit['lambda']:.2f}")
    print(f"  Negative Binomial: r = {nb_fit['r']:.2f}, p = {nb_fit['p']:.3f}" if nb_fit['type'] == 'negative_binomial' else f"  Negative Binomial: {nb_fit['type']}")
    print()
    
    # Calculate goodness-of-fit metrics
    # Log-likelihood for each distribution
    poisson_loglik = poisson_fit['distribution'].logpmf(attempts_history).sum()
    nb_loglik = nb_fit['distribution'].logpmf(attempts_history).sum()
    
    # AIC (lower is better)
    # AIC = 2k - 2ln(L), where k = number of parameters
    poisson_aic = 2 * 1 - 2 * poisson_loglik  # 1 parameter (λ)
    nb_aic = 2 * 2 - 2 * nb_loglik  # 2 parameters (r, p)
    
    print(f"Goodness-of-Fit Metrics:")
    print(f"  Poisson Log-Likelihood: {poisson_loglik:.2f}")
    print(f"  Negative Binomial Log-Likelihood: {nb_loglik:.2f}")
    print(f"  Poisson AIC: {poisson_aic:.2f}")
    print(f"  Negative Binomial AIC: {nb_aic:.2f}")
    
    if nb_aic < poisson_aic:
        print(f"  → Negative Binomial fits better (lower AIC by {poisson_aic - nb_aic:.2f})")
    else:
        print(f"  → Poisson fits better (lower AIC by {nb_aic - poisson_aic:.2f})")
    print()
    
    # Visualize if requested
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{player_name} - Distribution Comparison (Last {len(recent_data)} Games)', fontsize=16, fontweight='bold')
        
        # Plot 1: 3PA Distribution
        ax1 = axes[0, 0]
        
        # Histogram of actual 3PA
        counts, bins, patches = ax1.hist(attempts_history, bins=range(int(attempts_history.max()) + 2), 
                                         alpha=0.7, color='lightblue', edgecolor='black', 
                                         density=True, label='Actual 3PA')
        
        # Overlay Poisson PMF
        x_range = np.arange(0, int(attempts_history.max()) + 5)
        poisson_pmf = poisson_fit['distribution'].pmf(x_range)
        ax1.plot(x_range, poisson_pmf, 'r-o', linewidth=2, markersize=6, label=f'Poisson (λ={poisson_fit["lambda"]:.2f})')
        
        # Overlay Negative Binomial PMF
        nb_pmf = nb_fit['distribution'].pmf(x_range)
        ax1.plot(x_range, nb_pmf, 'g-s', linewidth=2, markersize=6, label=f'Neg Binom (r={nb_fit["r"]:.2f})')
        
        ax1.set_xlabel('3-Point Attempts', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax1.set_title('3PA Distribution Comparison', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: 3PM Distribution (Actual vs Predicted)
        ax2 = axes[0, 1]
        
        # Histogram of actual 3PM
        ax2.hist(makes_history, bins=range(int(makes_history.max()) + 2), 
                alpha=0.7, color='lightgreen', edgecolor='black',
                density=True, label='Actual 3PM')
        
        # Predicted 3PM using Monte Carlo
        poisson_pred = predict_3pm_distribution_monte_carlo(poisson_fit, empirical_3pt_pct, n_simulations=10000)
        nb_pred = predict_3pm_distribution_monte_carlo(nb_fit, empirical_3pt_pct, n_simulations=10000)
        
        if poisson_pred and nb_pred:
            x_makes = np.arange(0, max(poisson_pred['prob_by_makes'].keys()) + 1)
            poisson_3pm_probs = [poisson_pred['prob_by_makes'].get(k, 0) for k in x_makes]
            nb_3pm_probs = [nb_pred['prob_by_makes'].get(k, 0) for k in x_makes]
            
            ax2.plot(x_makes, poisson_3pm_probs, 'r-o', linewidth=2, markersize=6, label='Poisson Prediction')
            ax2.plot(x_makes, nb_3pm_probs, 'g-s', linewidth=2, markersize=6, label='Neg Binom Prediction')
        
        ax2.set_xlabel('3-Point Makes', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_title('3PM Distribution (Actual vs Predicted)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: 0.5 Line Analysis
        ax3 = axes[1, 0]
        
        if poisson_pred and nb_pred:
            categories = ['Actual', 'Poisson\nPrediction', 'Neg Binom\nPrediction']
            actual_zero_pct = (makes_history == 0).sum() / len(makes_history) * 100
            poisson_zero_pct = poisson_pred['prob_under_0.5'] * 100
            nb_zero_pct = nb_pred['prob_under_0.5'] * 100
            
            under_probs = [actual_zero_pct, poisson_zero_pct, nb_zero_pct]
            over_probs = [100 - actual_zero_pct, 100 - poisson_zero_pct, 100 - nb_zero_pct]
            
            x_pos = np.arange(len(categories))
            ax3.bar(x_pos, under_probs, label='UNDER 0.5 (0 makes)', color='salmon', alpha=0.8)
            ax3.bar(x_pos, over_probs, bottom=under_probs, label='OVER 0.5 (1+ makes)', color='lightgreen', alpha=0.8)
            
            # Add percentage labels
            for i, (under, over) in enumerate(zip(under_probs, over_probs)):
                ax3.text(i, under/2, f'{under:.1f}%', ha='center', va='center', fontweight='bold', fontsize=10)
                ax3.text(i, under + over/2, f'{over:.1f}%', ha='center', va='center', fontweight='bold', fontsize=10)
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(categories)
            ax3.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
            ax3.set_title('0.5 Line: Over/Under Probabilities', fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.set_ylim(0, 100)
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Model Fit Comparison
        ax4 = axes[1, 1]
        
        # Compare predicted vs actual frequencies
        unique_attempts, actual_counts = np.unique(attempts_history, return_counts=True)
        actual_freqs = actual_counts / len(attempts_history)
        
        x_vals = np.arange(0, int(attempts_history.max()) + 2)
        poisson_freqs = [poisson_fit['distribution'].pmf(x) for x in x_vals]
        nb_freqs = [nb_fit['distribution'].pmf(x) for x in x_vals]
        
        width = 0.25
        x_positions = x_vals
        
        # Plot bars
        for i, (attempt, freq) in enumerate(zip(unique_attempts, actual_freqs)):
            ax4.bar(attempt - width, freq, width, color='lightblue', alpha=0.7, label='Actual' if i == 0 else '')
        
        ax4.bar(x_positions, poisson_freqs, width, color='red', alpha=0.6, label='Poisson')
        ax4.bar(x_positions + width, nb_freqs, width, color='green', alpha=0.6, label='Neg Binom')
        
        ax4.set_xlabel('3-Point Attempts', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title(f'Model Fit Quality (AIC: Poisson={poisson_aic:.1f}, NB={nb_aic:.1f})', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'player': player_name,
        'n_games': len(recent_data),
        'mean_3pa': mean_3pa,
        'var_3pa': var_3pa,
        'mean_3pm': mean_3pm,
        'shooting_pct': empirical_3pt_pct,
        'poisson_fit': poisson_fit,
        'nb_fit': nb_fit,
        'poisson_aic': poisson_aic,
        'nb_aic': nb_aic,
        'better_fit': 'negative_binomial' if nb_aic < poisson_aic else 'poisson',
        'poisson_pred': poisson_pred if visualize else None,
        'nb_pred': nb_pred if visualize else None
    }


def predict_3pm_distribution_monte_carlo(attempts_dist, shooting_pct, n_simulations=10000, max_3pa=15):
    """
    Predict 3PM distribution using Monte Carlo simulation.
    
    Process:
    1. Sample 3PA from the attempts distribution (10k times)
    2. For each 3PA sample, simulate 3PM ~ Binomial(3PA, shooting_pct)
    3. Aggregate to get probability distribution of makes
    
    Args:
        attempts_dist: Dict from fit_negative_binomial_3pa() with 'distribution' key
        shooting_pct: Player's 3PT% (e.g., 0.35)
        n_simulations: Number of Monte Carlo simulations
        max_3pa: Maximum attempts to consider (for efficiency)
    
    Returns:
        Dictionary with:
        - 'prob_by_makes': {0: prob, 1: prob, 2: prob, ...}
        - 'prob_over_0.5': P(1+ makes)
        - 'prob_under_0.5': P(0 makes)
        - 'mean_3pm': Expected makes
        - 'simulated_3pa': Array of simulated attempts
        - 'simulated_3pm': Array of simulated makes
    """
    from scipy import stats
    
    if attempts_dist is None or shooting_pct is None or np.isnan(shooting_pct):
        return None
    
    # Ensure shooting percentage is valid
    shooting_pct = np.clip(shooting_pct, 0.0, 1.0)
    
    dist = attempts_dist['distribution']
    
    # Sample 3PA from distribution
    simulated_3pa = dist.rvs(size=n_simulations)
    simulated_3pa = np.clip(simulated_3pa, 0, max_3pa)  # Cap at reasonable max
    
    # For each 3PA, simulate 3PM
    simulated_3pm = np.array([
        stats.binom.rvs(n=int(attempts), p=shooting_pct) if attempts > 0 else 0
        for attempts in simulated_3pa
    ])
    
    # Calculate probability distribution
    unique_makes, counts = np.unique(simulated_3pm, return_counts=True)
    prob_by_makes = {int(makes): count / n_simulations for makes, count in zip(unique_makes, counts)}
    
    # For 0.5 line
    prob_zero = prob_by_makes.get(0, 0)
    prob_one_plus = sum(prob for makes, prob in prob_by_makes.items() if makes >= 1)
    
    return {
        'prob_by_makes': prob_by_makes,
        'prob_over_0.5': prob_one_plus,
        'prob_under_0.5': prob_zero,
        'mean_3pm': np.mean(simulated_3pm),
        'simulated_3pa': simulated_3pa,
        'simulated_3pm': simulated_3pm
    }


def test_data_loading_and_filtering():
    """
    Test Steps 1 & 2: Load filled data and identify 0.5 line players.
    
    This is the foundation for the modeling pipeline.
    """
    print("="*80)
    print("STEP 1 & 2: DATA LOADING AND PLAYER FILTERING")
    print("="*80)
    print()
    
    # STEP 1: Load filled player data (with DNPs)
    print("STEP 1: Loading filled player data (includes DNPs)...")
    print("-"*80)
    filled_df = load_filled_player_data(season='2024_25')
    
    # STEP 2a: Load props data
    print("\nSTEP 2a: Loading props data (betting lines)...")
    print("-"*80)
    props_df = load_props_data()
    print(f"✓ Loaded {len(props_df):,} props")
    print(f"  • Players: {props_df['player'].nunique()}")
    print(f"  • Lines: {sorted(props_df['consensus_line'].unique())}")
    print()
    
    # STEP 2b: Identify "true" 0.5 line players
    print("\nSTEP 2b: Identifying 'true' 0.5 line players...")
    print("-"*80)
    half_point_players = identify_half_point_line_players(
        props_df, 
        threshold=0.70,  # 70% of props must be at 0.5
        min_props=10     # At least 10 props total
    )
    
    # STEP 2c: Merge datasets (only for 0.5 line players)
    print("\nSTEP 2c: Merging filled data with props...")
    print("-"*80)
    merged_df = merge_filled_data_with_props(
        filled_df, 
        props_df, 
        half_point_players=half_point_players
    )
    
    # STEP 2d: Add rolling features (excluding DNPs) and availability features
    print("\nSTEP 2d: Engineering features...")
    print("-"*80)
    
    # Get full history for these players (not just props, includes all team games)
    filled_half_point = filled_df[filled_df['player'].isin(half_point_players)].copy()
    
    # Add rolling features (excludes DNPs)
    filled_half_point = add_rolling_features_excluding_dnps(filled_half_point, lookback_windows=[5, 10])
    
    # Add availability features (uses DNPs as signals)
    filled_half_point = add_availability_features(filled_half_point)
    
    # Merge engineered features back into props dataset
    feature_cols = [col for col in filled_half_point.columns if 
                    col.startswith('rolling_') or 
                    col.startswith('days_') or 
                    col.startswith('games_played_') or
                    col.startswith('prev_game_') or
                    col.startswith('consecutive_') or
                    col == 'n_games_played_history']
    
    merged_with_features = merged_df.merge(
        filled_half_point[['player', 'date'] + feature_cols],
        on=['player', 'date'],
        how='left',
        suffixes=('', '_features')
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Identified {len(half_point_players)} 'true' 0.5 line players")
    print(f"✓ Merged dataset: {len(merged_with_features):,} props")
    print(f"✓ Features added:")
    print(f"  • Rolling stats (5, 10 games) - DNPs excluded")
    print(f"  • Availability features - using DNP patterns")
    print(f"✓ Ready for distribution fitting (Step 3)")
    print()
    
    # Show example of DNP handling for one player
    print("="*80)
    print("EXAMPLE: DNP HANDLING FOR A PLAYER")
    print("="*80)
    print()
    
    # Pick a player with some DNPs
    example_player = half_point_players[0]
    player_data = filled_half_point[filled_half_point['player'] == example_player].head(15)
    
    print(f"Player: {example_player}")
    print(f"Showing first 15 team games:")
    print()
    print(f"{'Date':<12} | {'DNP?':<6} | {'Min':<5} | {'3PA':<5} | {'Roll 3PA (5g)':<15} | {'Days Since Played':<18} | {'Games Played (L5)':<18}")
    print("-" * 110)
    
    for _, row in player_data.iterrows():
        dnp_str = "YES" if row.get('dnp', False) else "NO"
        minutes = row.get('minutes', 0)
        threes_att = row.get('threes_attempted', 0)
        roll_3pa = row.get('rolling_3pa_mean_5', np.nan)
        days_since = row.get('days_since_last_played', np.nan)
        games_5 = row.get('games_played_last_5', np.nan)
        
        roll_3pa_str = f"{roll_3pa:.2f}" if pd.notna(roll_3pa) else "N/A"
        days_str = f"{days_since:.0f}" if pd.notna(days_since) else "N/A"
        games_5_str = f"{games_5:.0f}/5" if pd.notna(games_5) else "N/A"
        
        print(f"{row['date']:<12} | {dnp_str:<6} | {minutes:<5.0f} | {threes_att:<5.0f} | {roll_3pa_str:<15} | {days_str:<18} | {games_5_str:<18}")
    
    print()
    print("KEY OBSERVATIONS:")
    print("  • Rolling 3PA EXCLUDES DNPs (only calculated from games played)")
    print("  • 'Days Since Played' tracks rest between ACTUAL games (not team games)")
    print("  • 'Games Played (L5)' shows availability rate (useful feature!)")
    print()
    
    return merged_with_features, half_point_players, filled_half_point


def consolidate_props_by_game(props_df, verbose=False):
    """
    Consolidate multiple props for same game into single row.
    
    Strategy:
    - Group by (player, date) - assumes we've already filtered to single line (e.g., 0.5)
    - Take BEST ODDS for bettor (highest over odds, highest under odds)
    - Keep consensus line (should be same across bookmakers)
    - Keep actual result
    
    This handles cases where multiple bookmakers post same prop.
    
    Args:
        props_df: DataFrame with potentially duplicate props per game
        verbose: If True, show detailed duplicate information
    
    Returns:
        DataFrame with one row per (player, date)
    """
    # Check for duplicates
    dup_check = props_df.groupby(['player', 'date']).size()
    duplicates = dup_check[dup_check > 1]
    
    if len(duplicates) > 0:
        print(f"  ⚠️  Found {len(duplicates)} games with multiple props (different bookmakers)")
        
        if verbose:
            print(f"\n  DUPLICATE EXAMPLES (first 5):")
            print(f"  {'Player':<25} | {'Date':<12} | {'Count':<6}")
            print("  " + "-" * 50)
            for (player, date), count in duplicates.head(5).items():
                print(f"  {player:<25} | {date:<12} | {count:<6}")
                
                # Show the odds for these duplicates
                dup_rows = props_df[(props_df['player'] == player) & (props_df['date'] == date)]
                for _, row in dup_rows.iterrows():
                    print(f"    → Over: {row['over_best_odds']:>+4.0f}, Under: {row['under_best_odds']:>+4.0f}, Line: {row['consensus_line']:.1f}")
            print()
        
        print(f"  → Consolidating by taking BEST ODDS for bettor...")
        
        # Build aggregation dict dynamically based on available columns
        agg_dict = {
            'consensus_line': 'first',  # Should be same
            'over_best_odds': 'max',    # Higher = better for bettor (e.g., +150 > +120)
            'under_best_odds': 'max',   # Higher = better for bettor
        }
        
        # Add columns if they exist
        for col in ['threes_made', 'threes_attempted', 'minutes', 'opponent', 'home_away', 
                    'matchup', 'result', 'player_id']:
            if col in props_df.columns:
                agg_dict[col] = 'first'
        
        # Group and aggregate
        consolidated = props_df.groupby(['player', 'date']).agg(agg_dict).reset_index()
        
        print(f"  ✓ Consolidated to {len(consolidated)} unique games")
        print()
        
        return consolidated
    else:
        print(f"  ✓ No duplicate props found (1 row per game)")
        print()
        return props_df


def log_detailed_prediction(player_name, date, history, shooting_pct, poisson_fit, prediction, 
                           line, over_odds, under_odds, actual_3pm, market_prob_over, market_prob_under,
                           prior_pct=None, prior_description=None, prior_strength=None, sample_pct=None,
                           season_makes=None, season_attempts=None):
    """
    Log detailed prediction for a single game to show all intermediate steps.
    
    This helps understand what's going into the model and what's coming out.
    """
    print("\n" + "="*100)
    print(f"🔍 DETAILED PREDICTION LOG: {player_name} on {date}")
    print("="*100)
    print()
    
    # STEP 1: Historical 3PA
    print("📊 STEP 1: HISTORICAL 3PA (Input to Poisson)")
    print("-"*100)
    attempts_history = history['threes_attempted'].values.astype(int)
    makes_history = history['threes_made'].values.astype(int)
    dates = history['date'].values
    
    print(f"  Lookback window: {len(attempts_history)} games")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print()
    print(f"  {'Date':<12} | {'3PM':<5} | {'3PA':<5} | {'3PT%':<8}")
    print("  " + "-"*40)
    
    for d, makes, att in zip(dates, makes_history, attempts_history):
        pct = (makes / att * 100) if att > 0 else 0
        print(f"  {d:<12} | {makes:<5} | {att:<5} | {pct:<8.1f}%")
    
    print()
    print(f"  Summary:")
    print(f"    • Mean 3PA: {np.mean(attempts_history):.2f}")
    print(f"    • Std Dev 3PA: {np.std(attempts_history, ddof=1):.2f}")
    print(f"    • Min/Max 3PA: {np.min(attempts_history)} / {np.max(attempts_history)}")
    print(f"    • Variance/Mean ratio: {np.var(attempts_history, ddof=1) / np.mean(attempts_history):.2f}")
    print()
    
    # STEP 2: Fitted Distribution
    print("📈 STEP 2: FITTED 3PA DISTRIBUTION")
    print("-"*100)
    print(f"  Distribution type: {poisson_fit['type'].upper()}")
    print(f"  Parameters:")
    print(f"    • λ (lambda): {poisson_fit['lambda']:.3f}")
    print(f"    • Mean: {poisson_fit['mean']:.3f}")
    print(f"    • Theoretical variance: {poisson_fit['variance']:.3f}")
    print(f"    • Empirical variance: {poisson_fit['empirical_variance']:.3f}")
    print()
    
    # Show PMF for likely values
    print(f"  Probability Mass Function (3PA):")
    print(f"    {'Attempts':<10} | {'Probability':<15} | {'Bar':<30}")
    print("    " + "-"*60)
    
    for attempts in range(0, int(poisson_fit['mean'] + 2*np.sqrt(poisson_fit['variance'])) + 2):
        prob = poisson_fit['distribution'].pmf(attempts)
        bar = "█" * int(prob * 100)
        print(f"    {attempts:<10} | {prob:<15.1%} | {bar}")
    print()
    
    # STEP 3: Shooting Percentage
    print("🎯 STEP 3: SHOOTING PERCENTAGE (SEASON AVERAGE)")
    print("-"*100)
    
    print(f"  💡 KEY INSIGHT: We use SEASON average for 3PT%, NOT recent games!")
    print(f"     • 3PA (attempts) = recent games → captures current ROLE")
    print(f"     • 3PT% (shooting) = season average → captures true SKILL")
    print(f"     • Why? Recent shooting % is pure NOISE for low-volume shooters")
    print()
    
    # Show recent vs season comparison
    recent_makes = makes_history.sum()
    recent_attempts = attempts_history.sum()
    recent_pct = recent_makes / recent_attempts if recent_attempts > 0 else 0
    
    print(f"  🔴 Recent Games (last {len(history)} games) - NOT USED:")
    print(f"    • Makes: {recent_makes}")
    print(f"    • Attempts: {recent_attempts}")
    print(f"    • Recent 3PT%: {recent_pct:.1%}")
    print(f"    • ⚠️  TOO SMALL SAMPLE - Pure variance, not skill")
    print()
    
    # Show season data (what we actually use)
    if season_makes is not None and season_attempts is not None:
        print(f"  ✅ Season Average (Used for Model):")
        print(f"    • Makes: {season_makes}")
        print(f"    • Attempts: {season_attempts}")
        print(f"    • Season 3PT%: {sample_pct:.1%}")
        print(f"    • ✓ LARGER SAMPLE - True skill estimate")
    else:
        print(f"  ✅ Season Average (Used for Model):")
        print(f"    • 3PT%: {sample_pct:.1%}")
    print()
    
    print(f"  🧠 Bayesian Prior (Beta-Binomial Model):")
    if prior_description:
        print(f"    • {prior_description}")
    else:
        print(f"    • Prior: Beta distribution centered at {LEAGUE_AVG_3PT_PCT:.1%}")
    
    if prior_strength:
        prior_alpha = prior_strength * LEAGUE_AVG_3PT_PCT
        prior_beta = prior_strength * (1 - LEAGUE_AVG_3PT_PCT)
        print(f"    • Prior strength: {prior_strength} virtual attempts")
        print(f"    • Beta({prior_alpha:.0f}, {prior_beta:.0f}) distribution")
        print(f"    • High strength = conservative (prevents wild predictions)")
    print(f"    • Why? Even season averages need regression to league mean")
    print(f"    • TODO: Optimize strength empirically (test [50, 100, 200, 500, 1000])")
    print()
    
    print(f"  🎯 Model Shooting % (Posterior):")
    print(f"    • Model prediction: {shooting_pct:.1%}")
    if prior_pct and prior_strength and season_attempts:
        prior_alpha = prior_strength * prior_pct
        prior_beta = prior_strength * (1 - prior_pct)
        posterior_alpha = prior_alpha + season_makes
        posterior_beta = prior_beta + (season_attempts - season_makes)
        print(f"    • Posterior: Beta({posterior_alpha:.0f}, {posterior_beta:.0f})")
        print(f"    • Formula: posterior_mean = α/(α+β)")
        print(f"    •         = {posterior_alpha:.0f}/({posterior_alpha:.0f}+{posterior_beta:.0f})")
        print(f"    •         = {shooting_pct:.1%}")
    print()
    
    # Show the comparison
    if sample_pct and abs(shooting_pct - sample_pct) > 0.01:
        diff = abs(shooting_pct - sample_pct)
        direction = "up" if shooting_pct > sample_pct else "down"
        pct_change = (diff / sample_pct * 100) if sample_pct > 0 else 0
        print(f"  📊 Prior Impact:")
        print(f"    • Raw season average: {sample_pct:.1%}")
        print(f"    • Bayesian posterior: {shooting_pct:.1%}")
        print(f"    • Pulled {direction} by {diff:.1%} ({pct_change:.0f}% change) toward {LEAGUE_AVG_3PT_PCT:.1%}")
        print(f"    • High prior strength keeps predictions conservative")
    print()
    
    # STEP 4: Monte Carlo Simulation
    print("🎲 STEP 4: MONTE CARLO SIMULATION (3PA → 3PM)")
    print("-"*100)
    print(f"  Simulations: 10,000")
    print(f"  Process:")
    print(f"    1. Sample 3PA from Poisson(λ={poisson_fit['lambda']:.2f})")
    print(f"    2. For each 3PA, sample 3PM ~ Binomial(n=3PA, p={shooting_pct:.3f})")
    print(f"    3. Aggregate results")
    print()
    
    # Show simulated 3PM distribution
    print(f"  Resulting 3PM Distribution:")
    print(f"    {'Makes':<10} | {'Probability':<15} | {'Cumulative':<15} | {'Bar':<30}")
    print("    " + "-"*75)
    
    cumulative = 0
    for makes in sorted(prediction['prob_by_makes'].keys()):
        prob = prediction['prob_by_makes'][makes]
        cumulative += prob
        bar = "█" * int(prob * 100)
        print(f"    {makes:<10} | {prob:<15.1%} | {cumulative:<15.1%} | {bar}")
        if cumulative > 0.99 and makes > 3:
            break
    print()
    
    # STEP 5: Line Probabilities
    print(f"📏 STEP 5: PROBABILITIES FOR LINE {line}")
    print("-"*100)
    print(f"  Model probabilities:")
    print(f"    • P(UNDER {line}) = P(0 makes): {prediction['prob_under_0.5']:.1%}")
    print(f"    • P(OVER {line}) = P(1+ makes): {prediction['prob_over_0.5']:.1%}")
    print()
    
    print(f"  Market implied probabilities:")
    print(f"    • P(UNDER {line}): {market_prob_under:.1%} (from {under_odds:+.0f} odds)")
    print(f"    • P(OVER {line}): {market_prob_over:.1%} (from {over_odds:+.0f} odds)")
    print()
    
    # STEP 6: Edges
    print("💰 STEP 6: EDGE CALCULATION")
    print("-"*100)
    
    edge_over = prediction['prob_over_0.5'] - market_prob_over
    edge_under = prediction['prob_under_0.5'] - market_prob_under
    
    # Calculate EV
    if over_odds > 0:
        over_profit_per_dollar = over_odds / 100
    else:
        over_profit_per_dollar = 100 / abs(over_odds)
    
    if under_odds > 0:
        under_profit_per_dollar = under_odds / 100
    else:
        under_profit_per_dollar = 100 / abs(under_odds)
    
    ev_over = (prediction['prob_over_0.5'] * over_profit_per_dollar) - ((1 - prediction['prob_over_0.5']) * 1)
    ev_under = (prediction['prob_under_0.5'] * under_profit_per_dollar) - ((1 - prediction['prob_under_0.5']) * 1)
    
    print(f"  Probability Edge (Model - Market):")
    print(f"    • OVER: {edge_over:+.2%} ({prediction['prob_over_0.5']:.1%} model vs {market_prob_over:.1%} market)")
    print(f"    • UNDER: {edge_under:+.2%} ({prediction['prob_under_0.5']:.1%} model vs {market_prob_under:.1%} market)")
    print()
    
    print(f"  Expected Value (per $1 bet):")
    print(f"    • OVER: ${ev_over:+.3f} ({ev_over*100:+.1f}% ROI)")
    print(f"    • UNDER: ${ev_under:+.3f} ({ev_under*100:+.1f}% ROI)")
    print()
    
    # Recommendation
    min_ev = 0.05
    if ev_over >= min_ev:
        print(f"  ✅ RECOMMENDATION: BET OVER (EV = ${ev_over:+.3f})")
    elif ev_under >= min_ev:
        print(f"  ✅ RECOMMENDATION: BET UNDER (EV = ${ev_under:+.3f})")
    else:
        print(f"  ❌ RECOMMENDATION: NO BET (insufficient edge)")
    print()
    
    # STEP 7: Actual Result
    print("📊 STEP 7: ACTUAL RESULT")
    print("-"*100)
    actual_result = 'OVER' if actual_3pm >= line else 'UNDER'
    
    print(f"  Actual 3PM: {actual_3pm:.0f}")
    print(f"  Result: {actual_result} {line}")
    print()
    
    # Did model predict correctly?
    model_prediction = 'OVER' if prediction['prob_over_0.5'] > 0.5 else 'UNDER'
    market_prediction = 'OVER' if market_prob_over > 0.5 else 'UNDER'
    
    model_correct = (model_prediction == actual_result)
    market_correct = (market_prediction == actual_result)
    
    print(f"  Model predicted: {model_prediction} {'✅' if model_correct else '❌'}")
    print(f"  Market predicted: {market_prediction} {'✅' if market_correct else '❌'}")
    print()
    
    print("="*100)
    print()


def analyze_player_market_edges(merged_df, filled_df, player_name, lookback_window=10, min_history=5, verbose=True, detailed_log_one=False):
    """
    Analyze a single player's props: compare Poisson predictions vs market odds.
    
    For each prop:
    - Fit Poisson to recent history
    - Predict P(Over 0.5) and P(Under 0.5)
    - Compare to market implied probabilities
    - Calculate edges
    
    Args:
        merged_df: Props data with features
        filled_df: Full filled player data
        player_name: Player to analyze
        lookback_window: Games to use for fitting
        min_history: Minimum games required
        verbose: Print summary output
        detailed_log_one: If True, log detailed steps for one random game
    
    Returns:
        DataFrame with prop-by-prop analysis
    """
    if verbose:
        print("="*80)
        print(f"MARKET EDGE ANALYSIS: {player_name}")
        print("="*80)
        print()
    
    # Get player's props
    player_props = merged_df[merged_df['player'] == player_name].copy()
    
    # Consolidate multiple props per game (if any)
    player_props = consolidate_props_by_game(player_props, verbose=verbose)
    
    if len(player_props) == 0:
        if verbose:
            print(f"❌ No props found for {player_name}")
        return None
    
    if verbose:
        print(f"Props analyzed: {len(player_props)}")
        print(f"Date range: {player_props['date'].min()} to {player_props['date'].max()}")
        print()
    
    # Get full player history (for fitting distributions)
    player_history = filled_df[
        (filled_df['player'] == player_name) &
        (filled_df['dnp'] == False) &
        (filled_df['minutes'] > 0)
    ].copy()
    
    results = []
    
    # If detailed logging requested, pick one random game to log
    logged_detailed = False
    random_idx_to_log = np.random.randint(0, len(player_props)) if detailed_log_one and len(player_props) > 0 else -1
    
    for prop_num, (idx, prop) in enumerate(player_props.iterrows()):
        date = prop['date']
        line = prop['consensus_line']
        over_odds = prop['over_best_odds']
        under_odds = prop['under_best_odds']
        actual_3pm = prop['threes_made']
        
        # Get history up to this date
        history = player_history[player_history['date'] < date].tail(lookback_window)
        
        if len(history) < min_history:
            continue
        
        # ============================================================================
        # KEY INSIGHT: Separate treatment for 3PA vs 3PT%
        # ============================================================================
        # 3PA (attempts) = RECENT GAMES → Captures current role/minutes/injuries
        # 3PT% (shooting) = SEASON AVERAGE → Captures true skill (not random variance)
        # ============================================================================
        
        attempts_history = history['threes_attempted'].values.astype(int)
        makes_history = history['threes_made'].values.astype(int)  # Not used for shooting % anymore
        
        # Calculate shooting % using FULL SEASON (not recent games)
        # Recent shooting % is pure noise for low-volume shooters
        # Going 1/12 vs 3/12 in last 10 games = random variance, not skill change
        
        # Get player's full season 3PT% (all games before this date)
        player_season_data = player_history[player_history['date'] < date]
        season_total_makes = player_season_data['threes_made'].sum()
        season_total_attempts = player_season_data['threes_attempted'].sum()
        
        # PROPER BAYESIAN: Beta-Binomial Conjugate Prior
        # Prior: Beta(α, β) where α/(α+β) = LEAGUE_AVG_3PT_PCT
        #                     α+β = PRIOR_STRENGTH
        prior_alpha = PRIOR_STRENGTH * LEAGUE_AVG_3PT_PCT
        prior_beta = PRIOR_STRENGTH * (1 - LEAGUE_AVG_3PT_PCT)
        
        # Posterior: Beta(α + makes, β + misses)
        posterior_alpha = prior_alpha + season_total_makes
        posterior_beta = prior_beta + (season_total_attempts - season_total_makes)
        
        # Posterior mean (our shooting % estimate)
        shooting_pct = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # For logging
        season_pct = season_total_makes / season_total_attempts if season_total_attempts > 0 else LEAGUE_AVG_3PT_PCT
        sample_pct = season_pct
        prior_pct = LEAGUE_AVG_3PT_PCT
        prior_strength = PRIOR_STRENGTH
        
        if season_total_attempts > 0:
            prior_description = f"Beta({prior_alpha:.0f}, {prior_beta:.0f}) + season data ({season_total_makes}/{season_total_attempts})"
        else:
            prior_description = f"Beta({prior_alpha:.0f}, {prior_beta:.0f}) - no season data (using prior mean)"
        
        # Reasonable bounds (no one shoots <15% or >50% from 3 in NBA)
        shooting_pct = np.clip(shooting_pct, 0.15, 0.50)
        
        # Fit Poisson
        poisson_fit = fit_poisson_3pa(attempts_history)
        
        if poisson_fit is None:
            continue
        
        # Predict 3PM distribution
        pred = predict_3pm_distribution_monte_carlo(
            poisson_fit, 
            shooting_pct,
            n_simulations=10000
        )
        
        if pred is None:
            continue
        
        # Model probabilities with FLOORS (no 0% or 100% predictions - unrealistic)
        model_prob_over = pred['prob_over_0.5']
        model_prob_under = pred['prob_under_0.5']
        
        # Apply minimum/maximum bounds (1% to 99%)
        # No prediction should be CERTAIN - there's always some chance
        MIN_PROB = 0.01
        MAX_PROB = 0.99
        
        model_prob_over = np.clip(model_prob_over, MIN_PROB, MAX_PROB)
        model_prob_under = np.clip(model_prob_under, MIN_PROB, MAX_PROB)
        
        # Renormalize so they sum to 1
        total = model_prob_over + model_prob_under
        model_prob_over = model_prob_over / total
        model_prob_under = model_prob_under / total
        
        # Market probabilities (with vig)
        market_prob_over = odds_to_implied_probability(over_odds)
        market_prob_under = odds_to_implied_probability(under_odds)
        
        # Calculate EXPECTED VALUE (mathematically correct)
        # EV = (probability of win × profit) - (probability of loss × stake)
        
        # For OVER bet: stake $X to win profit based on odds
        if over_odds > 0:
            # Positive odds: profit = stake × (odds/100)
            # Example: +250 means bet $100 to win $250
            over_profit_per_dollar = over_odds / 100
        else:
            # Negative odds: profit = stake / (|odds|/100)
            # Example: -150 means bet $150 to win $100, so profit per $1 bet = 100/150 = 0.667
            over_profit_per_dollar = 100 / abs(over_odds)
        
        ev_over = (model_prob_over * over_profit_per_dollar) - ((1 - model_prob_over) * 1)
        
        # For UNDER bet
        if under_odds > 0:
            under_profit_per_dollar = under_odds / 100
        else:
            under_profit_per_dollar = 100 / abs(under_odds)
        
        ev_under = (model_prob_under * under_profit_per_dollar) - ((1 - model_prob_under) * 1)
        
        # Also calculate "edge" as model prob - market prob (for comparison)
        edge_over = model_prob_over - market_prob_over
        edge_under = model_prob_under - market_prob_under
        
        # Actual result
        actual_result = 'OVER' if actual_3pm >= line else 'UNDER'
        
        # DETAILED LOGGING for one random game
        if detailed_log_one and prop_num == random_idx_to_log and not logged_detailed:
            log_detailed_prediction(
                player_name, date, history, shooting_pct, poisson_fit, pred,
                line, over_odds, under_odds, actual_3pm, 
                market_prob_over, market_prob_under,
                prior_pct=prior_pct, prior_description=prior_description, 
                prior_strength=prior_strength, sample_pct=sample_pct,
                season_makes=season_total_makes, season_attempts=season_total_attempts
            )
            logged_detailed = True
        
        # Recommended bet (based on EXPECTED VALUE, not edge)
        min_ev = 0.05  # Minimum 5% expected return per dollar bet
        if ev_over >= min_ev:
            recommended = 'BET OVER'
            edge_value = ev_over
        elif ev_under >= min_ev:
            recommended = 'BET UNDER'
            edge_value = ev_under
        else:
            recommended = 'NO BET'
            edge_value = max(ev_over, ev_under)
        
        results.append({
            'date': date,
            'line': line,
            'actual_3pm': actual_3pm,
            'actual_result': actual_result,
            'over_odds': over_odds,
            'under_odds': under_odds,
            'market_prob_over': market_prob_over,
            'market_prob_under': market_prob_under,
            'model_prob_over': model_prob_over,
            'model_prob_under': model_prob_under,
            'edge_over': edge_over,
            'edge_under': edge_under,
            'ev_over': ev_over,
            'ev_under': ev_under,
            'recommended': recommended,
            'ev_value': edge_value,  # This is actually EV now
            'mean_3pa': poisson_fit['mean'],
            'shooting_pct': shooting_pct
        })
    
    if len(results) == 0:
        if verbose:
            print("❌ Not enough history to analyze props")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Display summary with EXPECTED VALUE
    if verbose:
        print(f"{'Date':<12} | {'Line':<5} | {'Actual':<6} | {'Over Odds':<10} | {'Under Odds':<11} | {'Model O/U':<20} | {'Market O/U':<20} | {'EV O/U ($/1 bet)':<20} | {'Recommendation':<15}")
        print("-" * 150)
    
    if verbose:
        for _, row in results_df.head(10).iterrows():
            model_ou = f"{row['model_prob_over']:.1%}/{row['model_prob_under']:.1%}"
            market_ou = f"{row['market_prob_over']:.1%}/{row['market_prob_under']:.1%}"
            ev_ou = f"{row['ev_over']:+.3f}/{row['ev_under']:+.3f}"
            
            print(f"{row['date']:<12} | {row['line']:<5.1f} | {row['actual_3pm']:<6.0f} | {row['over_odds']:<10.0f} | {row['under_odds']:<11.0f} | {model_ou:<20} | {market_ou:<20} | {ev_ou:<20} | {row['recommended']:<15}")
        
        if len(results_df) > 10:
            print(f"... and {len(results_df) - 10} more props")
        
        print()
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print()
    
    # Overall stats
    avg_ev_over = results_df['ev_over'].mean()
    avg_ev_under = results_df['ev_under'].mean()
    avg_edge_over = results_df['edge_over'].mean()
    avg_edge_under = results_df['edge_under'].mean()
    
    if verbose:
        print(f"Average Expected Value (per $1 bet):")
        print(f"  Over: ${avg_ev_over:+.3f}")
        print(f"  Under: ${avg_ev_under:+.3f}")
        print()
        
        print(f"Average Probability Edge (Model - Market):")
        print(f"  Over: {avg_edge_over:+.2%}")
        print(f"  Under: {avg_edge_under:+.2%}")
        print()
    
    # Bet recommendations
    bet_over_count = (results_df['recommended'] == 'BET OVER').sum()
    bet_under_count = (results_df['recommended'] == 'BET UNDER').sum()
    no_bet_count = (results_df['recommended'] == 'NO BET').sum()
    
    if verbose:
        print(f"Recommendations (≥5% EV threshold):")
        print(f"  BET OVER: {bet_over_count} ({bet_over_count/len(results_df):.1%})")
        print(f"  BET UNDER: {bet_under_count} ({bet_under_count/len(results_df):.1%})")
        print(f"  NO BET: {no_bet_count} ({no_bet_count/len(results_df):.1%})")
        print()
    
    # Backtest accuracy with ROI calculation
    if bet_over_count + bet_under_count > 0:
        # Check if recommended bets won
        results_df['bet_won'] = (
            ((results_df['recommended'] == 'BET OVER') & (results_df['actual_result'] == 'OVER')) |
            ((results_df['recommended'] == 'BET UNDER') & (results_df['actual_result'] == 'UNDER'))
        )
        
        bets_made = results_df[results_df['recommended'] != 'NO BET'].copy()
        
        # Calculate profit for each bet (assume $100 bet to win $100)
        def calculate_bet_profit(row):
            if row['recommended'] == 'BET OVER':
                odds = row['over_odds']
            else:  # BET UNDER
                odds = row['under_odds']
            
            # Calculate bet amount for target win
            target_win = 100
            if odds > 0:
                bet_amount = target_win / (odds / 100)
            else:
                bet_amount = abs(odds) * target_win / 100
            
            # Calculate profit
            if row['bet_won']:
                profit = target_win
            else:
                profit = -bet_amount
            
            return profit, bet_amount
        
        bets_made[['profit', 'bet_amount']] = bets_made.apply(
            lambda row: pd.Series(calculate_bet_profit(row)), axis=1
        )
        
        wins = bets_made['bet_won'].sum()
        win_rate = wins / len(bets_made)
        total_wagered = bets_made['bet_amount'].sum()
        total_profit = bets_made['profit'].sum()
        roi = (total_profit / total_wagered) * 100
        
        if verbose:
            print(f"Backtest Results (if we bet on all edges ≥5% EV):")
            print(f"  Bets: {len(bets_made)}")
            print(f"  Wins: {wins}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Total Wagered: ${total_wagered:,.2f}")
            print(f"  Total Profit: ${total_profit:,.2f}")
            print(f"  ROI: {roi:+.2f}%")
            print()
            
            # Breakdown by bet type (OVER vs UNDER)
            print(f"Breakdown by Bet Type:")
            print(f"  {'─'*76}")
            
            over_bets = bets_made[bets_made['recommended'] == 'BET OVER']
            under_bets = bets_made[bets_made['recommended'] == 'BET UNDER']
            
            for bet_type, bets_df in [('OVER', over_bets), ('UNDER', under_bets)]:
                if len(bets_df) == 0:
                    print(f"  {bet_type}: No bets")
                    continue
                
                # Calculate stats
                bet_wins = bets_df['bet_won'].sum()
                bet_win_rate = bet_wins / len(bets_df)
                bet_wagered = bets_df['bet_amount'].sum()
                bet_profit = bets_df['profit'].sum()
                bet_roi = (bet_profit / bet_wagered) * 100
                
                # Get probabilities and calculate average odds CORRECTLY
                # Must convert to implied prob, average, then convert back
                if bet_type == 'OVER':
                    odds_series = bets_df['over_odds']
                    avg_market_prob = bets_df['market_prob_over'].mean() * 100
                    avg_model_prob = bets_df['model_prob_over'].mean() * 100
                else:
                    odds_series = bets_df['under_odds']
                    avg_market_prob = bets_df['market_prob_under'].mean() * 100
                    avg_model_prob = bets_df['model_prob_under'].mean() * 100
                
                # Convert odds to implied probs, average them, convert back
                implied_probs = odds_series.apply(american_odds_to_percentage_probability)
                avg_implied_prob = implied_probs.mean()
                avg_odds = probability_to_american_odds(avg_implied_prob)
                
                avg_edge = avg_model_prob - avg_market_prob
                
                print(f"  {bet_type} Bets:")
                print(f"    Count: {len(bets_df)}")
                print(f"    Wins: {bet_wins} ({bet_win_rate:.1%})")
                print(f"    Total Wagered: ${bet_wagered:,.2f}")
                print(f"    Total Profit: ${bet_profit:+,.2f}")
                print(f"    ROI: {bet_roi:+.2f}%")
                print(f"    Avg Odds: {avg_odds:+.0f}")
                print(f"    Avg Market Implied Prob: {avg_market_prob:.1f}%")
                print(f"    Avg Model Prob: {avg_model_prob:.1f}%")
                print(f"    Avg Edge (Model - Market): {avg_edge:+.1f}%")
                print()
            
            print(f"  {'─'*76}")
            print()
    
    # Model accuracy
    results_df['model_correct'] = (
        ((results_df['model_prob_over'] > 0.5) & (results_df['actual_result'] == 'OVER')) |
        ((results_df['model_prob_under'] > 0.5) & (results_df['actual_result'] == 'UNDER'))
    )
    
    model_accuracy = results_df['model_correct'].sum() / len(results_df)
    
    if verbose:
        print(f"Model Directional Accuracy:")
        print(f"  Correct: {results_df['model_correct'].sum()}/{len(results_df)} ({model_accuracy:.1%})")
        print()
    
    return results_df


def test_distribution_fitting(player_names=None):
    """
    Test Step 3: Compare Poisson vs Negative Binomial distributions.
    
    Args:
        player_names: List of player names to analyze (or None for examples)
    """
    print("="*80)
    print("STEP 3: DISTRIBUTION FITTING - POISSON VS NEGATIVE BINOMIAL")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    filled_df = load_filled_player_data(season='2024_25')
    props_df = load_props_data()
    
    # Identify 0.5 line players
    half_point_players = identify_half_point_line_players(
        props_df,
        threshold=0.70,
        min_props=10
    )
    
    # Filter to 0.5 line players
    filled_half_point = filled_df[filled_df['player'].isin(half_point_players)].copy()
    
    print("\n" + "="*80)
    print("ANALYZING DISTRIBUTIONS FOR SELECT PLAYERS")
    print("="*80)
    print()
    
    # Default players to analyze if none provided
    if player_names is None:
        # Pick interesting examples:
        # - Giannis (superstar big who rarely shoots 3s)
        # - Zach Edey (rookie center, ultra low volume)
        # - Alperen Sengun (modern big, occasional shooter)
        player_names = ['Giannis Antetokounmpo', 'Zach Edey', 'Alperen Sengun']
    
    results = []
    
    for player_name in player_names:
        if player_name not in half_point_players:
            print(f"⚠️  {player_name} is not a '0.5 line player', skipping...")
            print()
            continue
        
        result = compare_distributions_single_player(
            filled_half_point,
            player_name,
            lookback_window=20,
            visualize=True
        )
        
        if result:
            results.append(result)
        
        print("\n" + "-"*80 + "\n")
    
    # Summary
    print("="*80)
    print("SUMMARY OF DISTRIBUTION FITS")
    print("="*80)
    print()
    
    if len(results) > 0:
        print(f"{'Player':<30} | {'Mean 3PA':<10} | {'Var/Mean':<10} | {'Better Fit':<20} | {'AIC Diff':<10}")
        print("-" * 95)
        
        for r in results:
            var_mean_ratio = r['var_3pa'] / r['mean_3pa'] if r['mean_3pa'] > 0 else 0
            aic_diff = abs(r['poisson_aic'] - r['nb_aic'])
            
            print(f"{r['player']:<30} | {r['mean_3pa']:<10.2f} | {var_mean_ratio:<10.2f} | {r['better_fit']:<20} | {aic_diff:<10.2f}")
        
        print()
        
        # Count which fits better more often
        poisson_better = sum(1 for r in results if r['better_fit'] == 'poisson')
        nb_better = sum(1 for r in results if r['better_fit'] == 'negative_binomial')
        
        print(f"Poisson fits better: {poisson_better}/{len(results)} players")
        print(f"Negative Binomial fits better: {nb_better}/{len(results)} players")
        print()
        
        # Check correlation between overdispersion and better fit
        print("KEY INSIGHT:")
        if nb_better > poisson_better:
            print("  → Negative Binomial tends to fit better for these 0.5 line players")
            print("  → Low-volume shooters have HIGH VARIANCE (overdispersion)")
            print("  → NB captures this better than Poisson")
        else:
            print("  → Poisson tends to fit better for these 0.5 line players")
            print("  → Low-volume shooters have CONSISTENT attempts")
            print("  → Simple Poisson is sufficient")
    
    print()
    
    return results


def demo_modeling():
    """
    Demo: Load data and run basic negative binomial modeling on 0.5 lines.
    """
    print("="*80)
    print("NBA 3PT PROP MODELING - 0.5 LINES (DEMO)")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_clean_data()
    print(f"✓ Loaded {len(df):,} props (0.5 lines only)")
    print()
    
    # Add rolling features
    df = add_rolling_features(df, lookback_windows=[5, 10])
    print(f"✓ Added rolling features")
    print()
    
    # Filter to props with enough history
    df = df[df['n_games_history'] >= MIN_LOOKBACK_GAMES].copy()
    print(f"✓ Filtered to {len(df):,} props with >= {MIN_LOOKBACK_GAMES} games history")
    print()
    
    # Example: Predict for first few props
    print("="*80)
    print("EXAMPLE PREDICTIONS (First 5 props)")
    print("="*80)
    print()
    
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        
        print(f"Prop #{idx+1}: {row['player']} on {row['date']}")
        print(f"  Line: {row['consensus_line']}")
        print(f"  Market Odds: Over {row['over_best_odds']:+.0f} / Under {row['under_best_odds']:+.0f}")
        print(f"  Actual Result: {row['threes_made']:.0f} makes on {row['threes_attempted']:.0f} attempts")
        print()
        
        # Get player's historical 3PA (using 10-game lookback)
        player_history = df[(df['player'] == row['player']) & 
                           (df['date'] < row['date'])].tail(10)
        
        if len(player_history) >= 3:
            historical_3pa = player_history['threes_attempted'].values
            
            # Fit negative binomial
            nb_fit = fit_negative_binomial_3pa(historical_3pa)
            
            print(f"  Historical 3PA (last {len(historical_3pa)} games): {list(historical_3pa)}")
            print(f"  Fitted Distribution: {nb_fit['type'].upper()}")
            print(f"    Mean: {nb_fit['mean']:.2f}")
            print(f"    Variance: {nb_fit['variance']:.2f}")
            
            # Predict 3PM distribution
            shooting_pct = row['rolling_3pt_pct_10']
            if pd.notna(shooting_pct):
                prediction = predict_3pm_distribution_monte_carlo(
                    nb_fit, 
                    shooting_pct,
                    n_simulations=N_SIMULATIONS
                )
                
                if prediction:
                    print(f"  Shooting %: {shooting_pct:.1%}")
                    print(f"  Predicted 3PM Distribution:")
                    for makes in sorted(prediction['prob_by_makes'].keys())[:6]:
                        prob = prediction['prob_by_makes'][makes]
                        print(f"    {makes} makes: {prob:.1%}")
                    
                    print()
                    print(f"  Model Prediction:")
                    print(f"    P(Over 0.5): {prediction['prob_over_0.5']:.1%}")
                    print(f"    P(Under 0.5): {prediction['prob_under_0.5']:.1%}")
                    
                    # Compare to market
                    market_prob_over = odds_to_implied_probability(row['over_best_odds'])
                    market_prob_under = odds_to_implied_probability(row['under_best_odds'])
                    
                    edge_over = prediction['prob_over_0.5'] - market_prob_over
                    edge_under = prediction['prob_under_0.5'] - market_prob_under
                    
                    print(f"  Market Implied:")
                    print(f"    P(Over 0.5): {market_prob_over:.1%}")
                    print(f"    P(Under 0.5): {market_prob_under:.1%}")
                    
                    print(f"  Edge:")
                    print(f"    Over edge: {edge_over:+.1%}")
                    print(f"    Under edge: {edge_under:+.1%}")
                    
                    # Actual result
                    actual = "OVER" if row['threes_made'] >= 1 else "UNDER"
                    model_pred = "OVER" if prediction['prob_over_0.5'] > 0.5 else "UNDER"
                    market_pred = "OVER" if market_prob_over > 0.5 else "UNDER"
                    
                    print(f"  Actual: {actual}")
                    print(f"  Model predicted: {model_pred} {'✓' if model_pred == actual else '✗'}")
                    print(f"  Market predicted: {market_pred} {'✓' if market_pred == actual else '✗'}")
        
        print()
        print("-"*80)
        print()


def analyze_all_players_market_edges():
    """
    Run market edge analysis on ALL 38 0.5 line players.
    
    Aggregates results to show overall profitability.
    
    Returns:
        Tuple of (all_results_df, player_summary_df)
    """
    print("="*80)
    print("ANALYZING ALL 0.5 LINE PLAYERS")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    filled_df = load_filled_player_data(season='2024_25')
    props_df = load_props_data()
    
    # Identify 0.5 line players
    half_point_players = identify_half_point_line_players(
        props_df,
        threshold=0.70,
        min_props=10
    )
    
    # Filter props to 0.5 lines only
    props_df = props_df[props_df['consensus_line'] == 0.5].copy()
    props_df = props_df[props_df['player'].isin(half_point_players)].copy()
    
    print(f"\n✓ Analyzing {len(half_point_players)} players")
    print()
    
    # Run analysis on each player
    all_results = []
    player_summaries = []
    
    for i, player in enumerate(half_point_players):
        print(f"[{i+1}/{len(half_point_players)}] {player}...", end=' ')
        
        try:
            player_results = analyze_player_market_edges(
                props_df,
                filled_df,
                player,
                lookback_window=10,
                min_history=5,
                verbose=False  # Suppress verbose output when analyzing all players
            )
            
            if player_results is not None and len(player_results) > 0:
                # Add to full results
                all_results.append(player_results)
                
                # Calculate player summary
                player_summaries.append({
                    'player': player,
                    'total_props': len(player_results),
                    'avg_ev_over': player_results['ev_over'].mean(),
                    'avg_ev_under': player_results['ev_under'].mean(),
                    'bets_over': (player_results['recommended'] == 'BET OVER').sum(),
                    'bets_under': (player_results['recommended'] == 'BET UNDER').sum(),
                    'model_accuracy': player_results['model_correct'].sum() / len(player_results) if 'model_correct' in player_results.columns else 0,
                    'backtest_bets': len(player_results[player_results['recommended'] != 'NO BET']),
                    'backtest_wins': player_results['bet_won'].sum() if 'bet_won' in player_results.columns else 0,
                })
                
                print(f"✓ ({len(player_results)} props)")
            else:
                print("✗ (insufficient data)")
        except Exception as e:
            print(f"✗ (error: {e})")
    
    # Combine all results
    if len(all_results) > 0:
        all_results_df = pd.concat(all_results, ignore_index=True)
        player_summary_df = pd.DataFrame(player_summaries)
        player_summary_df['backtest_winrate'] = player_summary_df['backtest_wins'] / player_summary_df['backtest_bets']
    else:
        all_results_df = pd.DataFrame()
        player_summary_df = pd.DataFrame()
    
    return all_results_df, player_summary_df


def test_market_edges(player_name='Giannis Antetokounmpo', detailed_log=True):
    """
    Test Step 4: Compare Poisson predictions vs market odds.
    
    Shows prop-by-prop analysis with edges.
    
    Args:
        player_name: Player to analyze
        detailed_log: If True, show detailed logging for one random game
    """
    print("="*80)
    print("STEP 4: MARKET EDGE ANALYSIS - POISSON VS MARKET ODDS")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    filled_df = load_filled_player_data(season='2024_25')
    props_df = load_props_data()
    
    # Identify 0.5 line players
    half_point_players = identify_half_point_line_players(
        props_df,
        threshold=0.70,
        min_props=10
    )
    
    # Filter props to 0.5 lines only
    props_df = props_df[props_df['consensus_line'] == 0.5].copy()
    
    # Filter to half-point players only
    props_df = props_df[props_df['player'].isin(half_point_players)].copy()
    
    print(f"\n✓ Filtered to {len(props_df)} props for 0.5 line players")
    print()
    
    # Run market edge analysis
    results_df = analyze_player_market_edges(
        props_df,  # Use props_df directly (not merged)
        filled_df,
        player_name,
        lookback_window=10,
        min_history=5,
        verbose=True,
        detailed_log_one=detailed_log
    )
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA 3PT Prop Modeling - 0.5 Lines')
    parser.add_argument('--test-loading', action='store_true',
                       help='Test Steps 1 & 2: Load data and identify 0.5 line players')
    parser.add_argument('--test-distributions', action='store_true',
                       help='Test Step 3: Compare Poisson vs Negative Binomial (with plots)')
    parser.add_argument('--test-market-edges', action='store_true',
                       help='Test Step 4: Compare model predictions vs market odds')
    parser.add_argument('--all-players', action='store_true',
                       help='Run market edge analysis on ALL 38 0.5 line players')
    parser.add_argument('--players', nargs='+', 
                       help='Player names to analyze')
    parser.add_argument('--detailed-log', action='store_true',
                       help='Show detailed logging for 1 random game (shows all intermediate steps)')
    parser.add_argument('--demo', action='store_true',
                       help='Run full modeling demo')
    
    args = parser.parse_args()
    
    if args.test_loading:
        # Test Steps 1 & 2: Data loading and player filtering
        merged_df, half_point_players, filled_df = test_data_loading_and_filtering()
        
    elif args.test_distributions:
        # Test Step 3: Distribution fitting with visualizations
        player_names = args.players if args.players else None
        results = test_distribution_fitting(player_names=player_names)
        
    elif args.all_players:
        # Analyze all 38 players
        all_results_df, player_summary_df = analyze_all_players_market_edges()
        
        # TLDR Summary
        print("\n" + "="*80)
        print("🎯 TLDR - MARKET EDGE ANALYSIS (ALL 38 PLAYERS)")
        print("="*80)
        print()
        
        if len(all_results_df) > 0:
            # Overall stats
            total_props = len(all_results_df)
            avg_ev_over = all_results_df['ev_over'].mean()
            avg_ev_under = all_results_df['ev_under'].mean()
            
            bets_over = (all_results_df['recommended'] == 'BET OVER').sum()
            bets_under = (all_results_df['recommended'] == 'BET UNDER').sum()
            bets_total = bets_over + bets_under
            
            wins = all_results_df['bet_won'].sum() if 'bet_won' in all_results_df.columns else 0
            win_rate = wins / bets_total if bets_total > 0 else 0
            
            model_accuracy = all_results_df['model_correct'].sum() / len(all_results_df)
            
            print(f"📊 OVERALL RESULTS:")
            print(f"  • Total props analyzed: {total_props:,}")
            print(f"  • Players: {len(player_summary_df)}")
            print()
            
            print(f"💰 EXPECTED VALUE (per $1 bet):")
            print(f"  • OVER bets: ${avg_ev_over:+.3f} ({avg_ev_over*100:+.1f}% ROI)")
            print(f"  • UNDER bets: ${avg_ev_under:+.3f} ({avg_ev_under*100:+.1f}% ROI)")
            print()
            
            print(f"🎲 BETTING RECOMMENDATIONS (≥5% EV):")
            print(f"  • BET OVER: {bets_over} ({bets_over/total_props*100:.1f}%)")
            print(f"  • BET UNDER: {bets_under} ({bets_under/total_props*100:.1f}%)")
            print(f"  • NO BET: {total_props - bets_total} ({(total_props - bets_total)/total_props*100:.1f}%)")
            print()
            
            print(f"✅ BACKTEST PERFORMANCE:")
            print(f"  • Bets made: {bets_total}")
            print(f"  • Wins: {wins}")
            print(f"  • Win rate: {win_rate:.1%}")
            print(f"  • Model directional accuracy: {model_accuracy:.1%}")
            print()
            
            # Breakdown by bet type (OVER vs UNDER) for all players
            print(f"📊 BREAKDOWN BY BET TYPE (All Players Combined):")
            print(f"  {'─'*78}")
            
            over_bets_all = all_results_df[all_results_df['recommended'] == 'BET OVER']
            under_bets_all = all_results_df[all_results_df['recommended'] == 'BET UNDER']
            
            for bet_type, bets_df in [('OVER', over_bets_all), ('UNDER', under_bets_all)]:
                if len(bets_df) == 0:
                    continue
                
                # Calculate bet outcomes (assume $100 target win)
                def calc_profit(row):
                    odds = row['over_odds'] if bet_type == 'OVER' else row['under_odds']
                    target_win = 100
                    if odds > 0:
                        bet_amount = target_win / (odds / 100)
                    else:
                        bet_amount = abs(odds) * target_win / 100
                    
                    won = (row['actual_result'] == 'OVER' if bet_type == 'BET OVER' else row['actual_result'] == 'UNDER')
                    profit = target_win if won else -bet_amount
                    return pd.Series({'bet_amount': bet_amount, 'profit': profit, 'won': won})
                
                results = bets_df.apply(calc_profit, axis=1)
                
                bet_wins = results['won'].sum()
                bet_win_rate = bet_wins / len(bets_df)
                bet_wagered = results['bet_amount'].sum()
                bet_profit = results['profit'].sum()
                bet_roi = (bet_profit / bet_wagered) * 100
                
                # Get odds and probabilities (correctly!)
                if bet_type == 'OVER':
                    odds_series = bets_df['over_odds']
                    avg_market_prob = bets_df['market_prob_over'].mean() * 100
                    avg_model_prob = bets_df['model_prob_over'].mean() * 100
                else:
                    odds_series = bets_df['under_odds']
                    avg_market_prob = bets_df['market_prob_under'].mean() * 100
                    avg_model_prob = bets_df['model_prob_under'].mean() * 100
                
                # Convert odds to implied probs, average, convert back
                implied_probs = odds_series.apply(american_odds_to_percentage_probability)
                avg_implied_prob = implied_probs.mean()
                avg_odds = probability_to_american_odds(avg_implied_prob)
                
                avg_edge = avg_model_prob - avg_market_prob
                
                print(f"  {bet_type} Bets:")
                print(f"    • Count: {len(bets_df)}")
                print(f"    • Wins: {bet_wins} ({bet_win_rate:.1%})")
                print(f"    • Total Wagered: ${bet_wagered:,.2f}")
                print(f"    • Total Profit: ${bet_profit:+,.2f}")
                print(f"    • ROI: {bet_roi:+.2f}%")
                print(f"    • Avg Odds: {avg_odds:+.0f}")
                print(f"    • Avg Market Implied Prob: {avg_market_prob:.1f}%")
                print(f"    • Avg Model Prob: {avg_model_prob:.1f}%")
                print(f"    • Avg Edge (Model - Market): {avg_edge:+.1f}%")
                print()
            
            print(f"  {'─'*78}")
            print()
            
            # Top/Bottom players by EV
            player_summary_df = player_summary_df.sort_values('avg_ev_under', ascending=False)
            
            print(f"🏆 TOP 5 PLAYERS (Best UNDER EV):")
            for _, row in player_summary_df.head(5).iterrows():
                print(f"  • {row['player']:<25}: ${row['avg_ev_under']:+.3f}/bet, {row['backtest_winrate']:.1%} win rate ({row['backtest_bets']} bets)")
            print()
            
            print(f"⚠️  BOTTOM 5 PLAYERS (Worst UNDER EV):")
            for _, row in player_summary_df.tail(5).iterrows():
                print(f"  • {row['player']:<25}: ${row['avg_ev_under']:+.3f}/bet, {row['backtest_winrate']:.1%} win rate ({row['backtest_bets']} bets)")
            print()
            
            # Model confidence check
            print(f"🔬 MODEL CONFIDENCE CHECK:")
            high_conf_under = (all_results_df['model_prob_under'] > 0.90).sum()
            high_conf_over = (all_results_df['model_prob_over'] > 0.90).sum()
            print(f"  • Props with >90% UNDER confidence: {high_conf_under} ({high_conf_under/total_props*100:.1f}%)")
            print(f"  • Props with >90% OVER confidence: {high_conf_over} ({high_conf_over/total_props*100:.1f}%)")
            
            if high_conf_under > total_props * 0.3:
                print(f"  • ⚠️  MODEL MAY BE OVERCONFIDENT (>30% of predictions >90%)")
            print()
            
        print("="*80)
        
    elif args.test_market_edges:
        # Test Step 4: Market edge analysis
        player_name = args.players[0] if args.players else 'Giannis Antetokounmpo'
        results_df = test_market_edges(player_name=player_name, detailed_log=args.detailed_log)
        
    elif args.demo:
        # Run full demo
        demo_modeling()
        
    else:
        # Default: Show help
        print("="*80)
        print("NBA 3PT PROP MODELING - 0.5 LINES")
        print("="*80)
        print()
        print("Available commands:")
        print()
        print("  --test-loading")
        print("    Test Steps 1 & 2: Load data and identify 0.5 line players")
        print()
        print("  --test-distributions")
        print("    Test Step 3: Compare Poisson vs Negative Binomial distributions")
        print("    Shows plots comparing distribution fits for select players")
        print("    Optional: --players \"Player Name\" \"Another Player\"")
        print()
        print("  --test-market-edges")
        print("    Test Step 4: Compare Poisson predictions vs market odds")
        print("    Shows prop-by-prop edges and betting recommendations")
        print("    Optional: --players \"Player Name\"")
        print("    Optional: --detailed-log (shows ALL intermediate steps for 1 random game)")
        print()
        print("  --demo")
        print("    Run full modeling demo")
        print()
        print("Examples:")
        print("  python 20251124_nba_3pt_prop_modeling_ou_half_point_lines.py --test-distributions")
        print("  python 20251124_nba_3pt_prop_modeling_ou_half_point_lines.py --test-market-edges")
        print("  python 20251124_nba_3pt_prop_modeling_ou_half_point_lines.py --test-market-edges --players \"Zach Edey\"")
        print("  python 20251124_nba_3pt_prop_modeling_ou_half_point_lines.py --test-market-edges --detailed-log")
        print()
    
    
    
    