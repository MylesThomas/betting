"""
Monte Carlo simulation to analyze EV of middle betting strategy on Bam Adebayo points.

Context:
--------
User has two available bets:
- Bam Adebayo Under 20.5 at -129 odds
- Bam Adebayo Over 24.5 at +235 odds

Goal: Simulate his points distribution and calculate:
1. Probability of hitting under bet (≤20.5 points)
2. Probability of hitting over bet (≥25 points)
3. Probability of middle (21-24 points, lose both)
4. Expected Value (EV) of putting $20 on each bet

Approach:
---------
- Use Monte Carlo simulation with normal distribution for NBA player points
- Default parameters based on typical Bam Adebayo stats (~16-17 points avg, ~6 std dev)
- Calculate profit/loss scenarios and overall EV

Usage:
    cd betting
    python3 tmp/bam_adebayo_middle_ev_simulation.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import optimize
from scipy.stats import norm
import os
import sys
from pathlib import Path

# Add src to path for s3_utils
project_root = Path(__file__).resolve()
while not (project_root / '.gitignore').exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / 'src'))

from s3_utils import read_df_from_s3


@dataclass
class Bet:
    """Represents a single bet with American odds."""
    description: str
    threshold: float
    american_odds: int
    bet_type: str  # 'under' or 'over'
    stake: float
    
    def payout_if_win(self):
        """Calculate payout (including stake) if bet wins."""
        if self.american_odds > 0:
            # Positive odds: profit = stake * (odds / 100)
            profit = self.stake * (self.american_odds / 100)
        else:
            # Negative odds: profit = stake / (abs(odds) / 100)
            profit = self.stake / (abs(self.american_odds) / 100)
        return self.stake + profit
    
    def profit_if_win(self):
        """Calculate profit (excluding stake) if bet wins."""
        return self.payout_if_win() - self.stake
    
    def is_winner(self, points):
        """Check if bet wins given points scored."""
        if self.bet_type == 'under':
            return points <= self.threshold
        else:  # over
            return points >= self.threshold


def american_odds_to_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calibrate_distribution_to_market(under_threshold, under_odds, over_threshold, over_odds, initial_mean=17.9):
    """
    Calibrate normal distribution parameters to match market-implied probabilities.
    
    Uses optimization to find mean and std that best match both betting lines.
    
    Parameters:
    -----------
    under_threshold : float
        Under bet threshold (e.g., 20.5)
    under_odds : int
        American odds for under bet
    over_threshold : float
        Over bet threshold (e.g., 24.5)
    over_odds : int
        American odds for over bet
    initial_mean : float
        Starting guess for mean (e.g., season average)
    
    Returns:
    --------
    dict : Calibrated mean, std, and fit quality metrics
    """
    # Market-implied probabilities
    target_under_prob = american_odds_to_implied_prob(under_odds)
    target_over_prob = american_odds_to_implied_prob(over_odds)
    
    print(f"\n{'='*70}")
    print("CALIBRATING DISTRIBUTION TO MARKET ODDS")
    print(f"{'='*70}\n")
    print(f"Target probabilities from market:")
    print(f"  Under {under_threshold}: {target_under_prob:.2%} (from {under_odds:+d} odds)")
    print(f"  Over {over_threshold}:  {target_over_prob:.2%} (from {over_odds:+d} odds)")
    print(f"  Middle range: {1 - target_under_prob - target_over_prob:.2%}\n")
    
    def objective(params):
        """Minimize error between simulated and target probabilities."""
        mean, std = params
        if std <= 0:
            return 1e10  # Invalid std
        
        # Calculate probabilities using normal distribution
        sim_under_prob = norm.cdf(under_threshold, loc=mean, scale=std)
        sim_over_prob = 1 - norm.cdf(over_threshold - 0.01, loc=mean, scale=std)
        
        # Squared error
        error = (sim_under_prob - target_under_prob)**2 + (sim_over_prob - target_over_prob)**2
        return error
    
    # Initial guess: use provided mean, and estimate std
    initial_guess = [initial_mean, 8.0]
    
    # Optimize
    result = optimize.minimize(
        objective,
        initial_guess,
        method='Nelder-Mead',
        bounds=[(10, 30), (3, 15)]
    )
    
    calibrated_mean, calibrated_std = result.x
    
    # Verify fit
    sim_under_prob = norm.cdf(under_threshold, loc=calibrated_mean, scale=calibrated_std)
    sim_over_prob = 1 - norm.cdf(over_threshold - 0.01, loc=calibrated_mean, scale=calibrated_std)
    sim_middle_prob = 1 - sim_under_prob - sim_over_prob
    
    print(f"Calibrated distribution parameters:")
    print(f"  Mean:  {calibrated_mean:.2f} points")
    print(f"  Std:   {calibrated_std:.2f} points\n")
    
    print(f"Verification (how well calibrated distribution matches market):")
    print(f"  Under {under_threshold}:")
    print(f"    Target: {target_under_prob:.2%}")
    print(f"    Actual: {sim_under_prob:.2%}")
    print(f"    Error:  {abs(sim_under_prob - target_under_prob):.2%}\n")
    
    print(f"  Over {over_threshold}:")
    print(f"    Target: {target_over_prob:.2%}")
    print(f"    Actual: {sim_over_prob:.2%}")
    print(f"    Error:  {abs(sim_over_prob - target_over_prob):.2%}\n")
    
    print(f"  Middle ({under_threshold}-{over_threshold}):")
    print(f"    Implied: {sim_middle_prob:.2%}\n")
    
    print(f"{'='*70}\n")
    
    return {
        'mean': calibrated_mean,
        'std': calibrated_std,
        'fit_quality': {
            'under_error': abs(sim_under_prob - target_under_prob),
            'over_error': abs(sim_over_prob - target_over_prob)
        }
    }


def simulate_bam_points(n_simulations=100000, mean=16.5, std=6.0):
    """
    Simulate Bam Adebayo points using normal distribution.
    
    Parameters:
    -----------
    n_simulations : int
        Number of Monte Carlo simulations
    mean : float
        Mean points per game (default ~16.5 based on typical Bam stats)
    std : float
        Standard deviation (default ~6.0)
    
    Returns:
    --------
    np.ndarray : Simulated points values
    """
    # Use normal distribution, clipped at 0 (can't score negative points)
    points = np.random.normal(mean, std, n_simulations)
    points = np.maximum(points, 0)  # No negative points
    return points


def calculate_outcome(points, bet1, bet2):
    """
    Calculate profit/loss for a single simulation.
    
    Returns:
    --------
    tuple : (total_profit, outcome_type)
        outcome_type: 'both_win', 'under_only', 'over_only', 'both_lose'
    """
    bet1_wins = bet1.is_winner(points)
    bet2_wins = bet2.is_winner(points)
    
    profit = 0
    if bet1_wins:
        profit += bet1.profit_if_win()
    else:
        profit -= bet1.stake
    
    if bet2_wins:
        profit += bet2.profit_if_win()
    else:
        profit -= bet2.stake
    
    # Determine outcome type
    if bet1_wins and bet2_wins:
        outcome_type = 'both_win'
    elif bet1_wins and not bet2_wins:
        outcome_type = 'under_only'
    elif not bet1_wins and bet2_wins:
        outcome_type = 'over_only'
    else:
        outcome_type = 'both_lose'
    
    return profit, outcome_type


def run_simulation(bet1, bet2, n_simulations=100000, mean=16.5, std=6.0):
    """
    Run Monte Carlo simulation for middle betting strategy.
    
    Returns:
    --------
    dict : Results including probabilities, EV, and detailed breakdown
    """
    print(f"\n{'='*70}")
    print("BAM ADEBAYO MIDDLE BETTING SIMULATION")
    print(f"{'='*70}\n")
    
    # Generate simulated points
    points_samples = simulate_bam_points(n_simulations, mean, std)
    
    # Calculate outcomes for each simulation
    results = []
    for points in points_samples:
        profit, outcome_type = calculate_outcome(points, bet1, bet2)
        results.append({
            'points': points,
            'profit': profit,
            'outcome': outcome_type
        })
    
    df = pd.DataFrame(results)
    
    # Calculate probabilities
    prob_both_win = (df['outcome'] == 'both_win').mean()
    prob_under_only = (df['outcome'] == 'under_only').mean()
    prob_over_only = (df['outcome'] == 'over_only').mean()
    prob_both_lose = (df['outcome'] == 'both_lose').mean()
    
    # Calculate EV
    ev = df['profit'].mean()
    
    # Print bet details
    print("BET DETAILS:")
    print(f"  {bet1.description}")
    print(f"    • Threshold: {bet1.threshold} points")
    print(f"    • Odds: {bet1.american_odds:+d}")
    print(f"    • Implied Prob: {american_odds_to_implied_prob(bet1.american_odds):.2%}")
    print(f"    • Stake: ${bet1.stake:.2f}")
    print(f"    • Profit if win: ${bet1.profit_if_win():.2f}\n")
    
    print(f"  {bet2.description}")
    print(f"    • Threshold: {bet2.threshold} points")
    print(f"    • Odds: {bet2.american_odds:+d}")
    print(f"    • Implied Prob: {american_odds_to_implied_prob(bet2.american_odds):.2%}")
    print(f"    • Stake: ${bet2.stake:.2f}")
    print(f"    • Profit if win: ${bet2.profit_if_win():.2f}\n")
    
    total_stake = bet1.stake + bet2.stake
    print(f"  Total Stake: ${total_stake:.2f}\n")
    
    # Print simulation parameters
    print(f"SIMULATION PARAMETERS:")
    print(f"  • Number of simulations: {n_simulations:,}")
    print(f"  • Points distribution: Normal(μ={mean}, σ={std})")
    print(f"  • Middle range: {bet1.threshold} < points < {bet2.threshold}\n")
    
    # Print results
    print(f"{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}\n")
    
    print(f"OUTCOME PROBABILITIES:")
    print(f"  • Both bets WIN:    {prob_both_win:7.2%}  (≤{bet1.threshold} points)")
    print(f"  • Under only WIN:   {prob_under_only:7.2%}  (>{bet1.threshold}, <{bet2.threshold} points)")
    print(f"  • Over only WIN:    {prob_over_only:7.2%}  (≥{bet2.threshold} points)")
    print(f"  • Both bets LOSE:   {prob_both_lose:7.2%}  ({bet1.threshold} < points < {bet2.threshold})\n")
    
    # Calculate profit/loss for each outcome
    sample_under_win = df[df['outcome'] == 'both_win']['profit'].iloc[0] if len(df[df['outcome'] == 'both_win']) > 0 else 0
    sample_middle = df[df['outcome'] == 'both_lose']['profit'].iloc[0] if len(df[df['outcome'] == 'both_lose']) > 0 else -total_stake
    sample_over_win = df[df['outcome'] == 'over_only']['profit'].iloc[0] if len(df[df['outcome'] == 'over_only']) > 0 else 0
    
    print(f"PROFIT/LOSS BY OUTCOME:")
    print(f"  • Both WIN (≤{bet1.threshold}):     ${sample_under_win:+7.2f}")
    print(f"  • Under only WIN:    ${df[df['outcome'] == 'under_only']['profit'].iloc[0] if len(df[df['outcome'] == 'under_only']) > 0 else 0:+7.2f}")
    print(f"  • Over only WIN:     ${sample_over_win:+7.2f}")
    print(f"  • Both LOSE (middle): ${sample_middle:+7.2f}\n")
    
    print(f"EXPECTED VALUE (EV):")
    print(f"  • EV per bet set:    ${ev:+.2f}")
    print(f"  • EV as % of stake:  {(ev/total_stake)*100:+.2f}%")
    print(f"  • ROI:               {(ev/total_stake)*100:+.2f}%\n")
    
    # Profit/loss statistics
    print(f"PROFIT/LOSS STATISTICS:")
    print(f"  • Mean profit:       ${df['profit'].mean():+.2f}")
    print(f"  • Median profit:     ${df['profit'].median():+.2f}")
    print(f"  • Std dev:           ${df['profit'].std():.2f}")
    print(f"  • Min profit:        ${df['profit'].min():+.2f}")
    print(f"  • Max profit:        ${df['profit'].max():+.2f}\n")
    
    # Distribution stats
    print(f"POINTS DISTRIBUTION (from simulation):")
    print(f"  • Mean:              {points_samples.mean():.2f}")
    print(f"  • Median:            {np.median(points_samples):.2f}")
    print(f"  • Std dev:           {points_samples.std():.2f}")
    print(f"  • 25th percentile:   {np.percentile(points_samples, 25):.2f}")
    print(f"  • 75th percentile:   {np.percentile(points_samples, 75):.2f}\n")
    
    print(f"{'='*70}\n")
    
    return {
        'ev': ev,
        'roi': (ev/total_stake)*100,
        'prob_both_win': prob_both_win,
        'prob_under_only': prob_under_only,
        'prob_over_only': prob_over_only,
        'prob_both_lose': prob_both_lose,
        'total_stake': total_stake,
        'df': df,
        'points_samples': points_samples
    }


def test_all_stake_combinations(max_stake=20, mean=16.5, std=6.0, n_simulations=50000):
    """
    Test all combinations of stakes from $0 to max_stake for both bets.
    
    Parameters:
    -----------
    max_stake : int
        Maximum stake to test for each bet (tests $0 to $max_stake)
    mean : float
        Mean points for simulation
    std : float
        Standard deviation for simulation
    n_simulations : int
        Number of simulations per test
    
    Returns:
    --------
    pd.DataFrame : Results for all combinations
    """
    print(f"\n{'='*70}")
    print("2D STAKE ALLOCATION OPTIMIZATION")
    print(f"{'='*70}\n")
    print(f"Testing all combinations from $0-${max_stake} for both bets")
    print(f"Total combinations: {(max_stake+1)**2}")
    print(f"This may take a moment...\n")
    
    results_list = []
    
    # Test all combinations
    for under_stake in range(0, max_stake + 1):
        for over_stake in range(0, max_stake + 1):
            # Skip if both are 0
            if under_stake == 0 and over_stake == 0:
                continue
            
            bet1 = Bet(
                description="Bam Adebayo UNDER 20.5 points",
                threshold=20.5,
                american_odds=-129,
                bet_type='under',
                stake=float(under_stake)
            )
            
            bet2 = Bet(
                description="Bam Adebayo OVER 24.5 points",
                threshold=24.5,
                american_odds=+235,
                bet_type='over',
                stake=float(over_stake)
            )
            
            # Run simulation quietly
            points_samples = simulate_bam_points(n_simulations, mean, std)
            
            profits = []
            for points in points_samples:
                profit, _ = calculate_outcome(points, bet1, bet2)
                profits.append(profit)
            
            ev = np.mean(profits)
            total_stake = under_stake + over_stake
            roi = (ev / total_stake) * 100 if total_stake > 0 else 0
            
            results_list.append({
                'under_stake': under_stake,
                'over_stake': over_stake,
                'total_stake': total_stake,
                'ev': ev,
                'roi': roi,
                'std': np.std(profits),
                'sharpe': (ev / np.std(profits)) if np.std(profits) > 0 else 0
            })
    
    df_results = pd.DataFrame(results_list)
    
    # Find top 10 by EV
    top_10_ev = df_results.nlargest(10, 'ev')
    
    # Find top 10 by ROI
    top_10_roi = df_results.nlargest(10, 'roi')
    
    # Find top 10 by Sharpe ratio
    top_10_sharpe = df_results.nlargest(10, 'sharpe')
    
    print("="*70)
    print("TOP 10 ALLOCATIONS BY EXPECTED VALUE (EV)")
    print("="*70)
    print(f"{'Under':>6} | {'Over':>6} | {'Total':>6} | {'EV':>8} | {'ROI':>8} | {'Sharpe':>8}")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for _, row in top_10_ev.iterrows():
        print(f"${row['under_stake']:>5.0f} | ${row['over_stake']:>5.0f} | ${row['total_stake']:>5.0f} | "
              f"${row['ev']:>6.2f} | {row['roi']:>6.2f}% | {row['sharpe']:>7.4f}")
    
    print(f"\n{'='*70}")
    print("TOP 10 ALLOCATIONS BY ROI")
    print("="*70)
    print(f"{'Under':>6} | {'Over':>6} | {'Total':>6} | {'EV':>8} | {'ROI':>8} | {'Sharpe':>8}")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for _, row in top_10_roi.iterrows():
        print(f"${row['under_stake']:>5.0f} | ${row['over_stake']:>5.0f} | ${row['total_stake']:>5.0f} | "
              f"${row['ev']:>6.2f} | {row['roi']:>6.2f}% | {row['sharpe']:>7.4f}")
    
    print(f"\n{'='*70}")
    print("TOP 10 ALLOCATIONS BY SHARPE RATIO (Risk-Adjusted)")
    print("="*70)
    print(f"{'Under':>6} | {'Over':>6} | {'Total':>6} | {'EV':>8} | {'ROI':>8} | {'Sharpe':>8}")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for _, row in top_10_sharpe.iterrows():
        print(f"${row['under_stake']:>5.0f} | ${row['over_stake']:>5.0f} | ${row['total_stake']:>5.0f} | "
              f"${row['ev']:>6.2f} | {row['roi']:>6.2f}% | {row['sharpe']:>7.4f}")
    
    best_ev = top_10_ev.iloc[0]
    
    print(f"\n{'='*70}")
    print("ABSOLUTE BEST ALLOCATION (by EV)")
    print("="*70)
    print(f"  Under Stake: ${best_ev['under_stake']:.0f}")
    print(f"  Over Stake:  ${best_ev['over_stake']:.0f}")
    print(f"  Total Stake: ${best_ev['total_stake']:.0f}")
    print(f"  Expected EV: ${best_ev['ev']:+.2f}")
    print(f"  ROI:         {best_ev['roi']:+.2f}%")
    print(f"  Std Dev:     ${best_ev['std']:.2f}")
    print(f"  Sharpe:      {best_ev['sharpe']:.4f}\n")
    
    return df_results, best_ev


def test_stake_allocations(under_stake=20.0, mean=16.5, std=6.0, n_simulations=50000):
    """
    Test different stake allocations on the over bet while keeping under bet fixed.
    
    Parameters:
    -----------
    under_stake : float
        Fixed stake on the under bet
    mean : float
        Mean points for simulation
    std : float
        Standard deviation for simulation
    n_simulations : int
        Number of simulations per test
    
    Returns:
    --------
    pd.DataFrame : Results for each allocation
    """
    print(f"\n{'='*70}")
    print("STAKE ALLOCATION OPTIMIZATION (Fixed Under)")
    print(f"{'='*70}\n")
    print(f"Fixed under bet stake: ${under_stake:.2f}")
    print(f"Testing different over bet stakes...\n")
    
    # Test range of over stakes from $1 to $30
    over_stakes = [1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30]
    
    results_list = []
    
    for over_stake in over_stakes:
        bet1 = Bet(
            description="Bam Adebayo UNDER 20.5 points",
            threshold=20.5,
            american_odds=-129,
            bet_type='under',
            stake=under_stake
        )
        
        bet2 = Bet(
            description="Bam Adebayo OVER 24.5 points",
            threshold=24.5,
            american_odds=+235,
            bet_type='over',
            stake=over_stake
        )
        
        # Run simulation quietly
        points_samples = simulate_bam_points(n_simulations, mean, std)
        
        profits = []
        for points in points_samples:
            profit, _ = calculate_outcome(points, bet1, bet2)
            profits.append(profit)
        
        ev = np.mean(profits)
        total_stake = under_stake + over_stake
        roi = (ev / total_stake) * 100
        
        results_list.append({
            'under_stake': under_stake,
            'over_stake': over_stake,
            'total_stake': total_stake,
            'ev': ev,
            'roi': roi,
            'std': np.std(profits)
        })
    
    df_results = pd.DataFrame(results_list)
    
    # Find optimal allocation
    best_idx = df_results['ev'].idxmax()
    best_result = df_results.iloc[best_idx]
    
    print(f"{'Under Stake':>12} | {'Over Stake':>11} | {'Total Stake':>12} | {'EV':>8} | {'ROI':>8} | {'Std Dev':>8}")
    print(f"{'-'*12}-+-{'-'*11}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for _, row in df_results.iterrows():
        marker = " ← BEST" if row['ev'] == best_result['ev'] else ""
        print(f"${row['under_stake']:>10.2f} | ${row['over_stake']:>9.2f} | ${row['total_stake']:>10.2f} | "
              f"${row['ev']:>6.2f} | {row['roi']:>6.2f}% | ${row['std']:>6.2f}{marker}")
    
    print(f"\n{'='*70}")
    print("OPTIMAL ALLOCATION:")
    print(f"{'='*70}")
    print(f"  Under Stake: ${best_result['under_stake']:.2f}")
    print(f"  Over Stake:  ${best_result['over_stake']:.2f}")
    print(f"  Total Stake: ${best_result['total_stake']:.2f}")
    print(f"  Expected EV: ${best_result['ev']:+.2f}")
    print(f"  ROI:         {best_result['roi']:+.2f}%")
    print(f"  Std Dev:     ${best_result['std']:.2f}\n")
    
    return df_results, best_result


def load_bam_stats_from_s3():
    """
    Load Bam Adebayo's actual game stats from S3.
    
    Returns:
    --------
    dict : Stats including mean, std, and historical hit rates
    """
    print(f"\n{'='*70}")
    print("LOADING BAM ADEBAYO DATA FROM S3")
    print(f"{'='*70}\n")
    
    # Load from S3
    s3_path = "player_shot_charts/2025-26/Bam_Adebayo_1628389.csv"
    bucket = "nba-api-mt"
    
    print(f"Loading: s3://{bucket}/{s3_path}")
    
    try:
        df = read_df_from_s3(bucket, s3_path)
        print(f"Loaded {len(df)} rows\n")
        
        # Examine columns
        print(f"Available columns: {list(df.columns)[:10]}...\n")
        
        # Group by game to get points per game
        if 'GAME_ID' in df.columns and 'SHOT_MADE_FLAG' in df.columns:
            # Each row is a shot, need to aggregate
            game_stats = df.groupby('GAME_ID').agg({
                'SHOT_MADE_FLAG': 'sum',  # Total made shots
                'GAME_DATE': 'first'
            }).reset_index()
            
            # Made shots * 2 or 3 depending on shot type
            # This is simplified - need actual points
            print("Data structure: Shot-level data")
            print("Need to calculate points from shots...\n")
            
            # Check if we have shot values
            if 'SHOT_VALUE' in df.columns:
                game_points = df.groupby('GAME_ID').agg({
                    'SHOT_VALUE': lambda x: (df.loc[x.index, 'SHOT_MADE_FLAG'] * df.loc[x.index, 'SHOT_VALUE']).sum(),
                    'GAME_DATE': 'first'
                }).reset_index()
                game_points.columns = ['GAME_ID', 'PTS', 'GAME_DATE']
            else:
                print("⚠️  Shot value column not found, cannot calculate exact points")
                print("Falling back to manual recent game data\n")
                return None
                
        elif 'PTS' in df.columns:
            # Already aggregated by game
            game_points = df[['GAME_DATE', 'PTS']].copy()
            print("Data structure: Game-level data")
        else:
            print("⚠️  Could not find points column")
            print("Falling back to manual recent game data\n")
            return None
        
        # Sort by date
        game_points = game_points.sort_values('GAME_DATE', ascending=False)
        points_list = game_points['PTS'].tolist()
        
        print(f"Found {len(points_list)} games")
        print(f"Most recent 15 games: {points_list[:15]}\n")
        
        # Calculate statistics
        all_mean = np.mean(points_list)
        all_std = np.std(points_list, ddof=1)
        
        # Recent games (last 10)
        recent_games = points_list[:10]
        recent_mean = np.mean(recent_games)
        recent_std = np.std(recent_games, ddof=1)
        
        # Season stats
        print(f"Full season statistics ({len(points_list)} games):")
        print(f"  Mean:     {all_mean:.2f}")
        print(f"  Std Dev:  {all_std:.2f}")
        print(f"  Median:   {np.median(points_list):.2f}")
        print(f"  Min:      {min(points_list)}")
        print(f"  Max:      {max(points_list)}\n")
        
        print(f"Recent 10 games:")
        print(f"  Mean:     {recent_mean:.2f}")
        print(f"  Std Dev:  {recent_std:.2f}")
        print(f"  Median:   {np.median(recent_games):.2f}\n")
        
        # Hit rates vs thresholds
        under_20_5_count = sum(1 for pts in points_list if pts <= 20.5)
        over_24_5_count = sum(1 for pts in points_list if pts >= 25)
        middle_count = sum(1 for pts in points_list if 20.5 < pts < 25)
        
        print(f"Historical performance vs bet thresholds:")
        print(f"  ≤20.5 points (under wins): {under_20_5_count}/{len(points_list)} = {under_20_5_count/len(points_list):.1%}")
        print(f"  21-24 points (middle):     {middle_count}/{len(points_list)} = {middle_count/len(points_list):.1%}")
        print(f"  ≥25 points (over wins):    {over_24_5_count}/{len(points_list)} = {over_24_5_count/len(points_list):.1%}\n")
        
        print(f"{'='*70}\n")
        
        return {
            'all_games': {
                'mean': all_mean,
                'std': all_std,
                'data': points_list
            },
            'recent_games': {
                'mean': recent_mean,
                'std': recent_std,
                'data': recent_games
            },
            'season': {
                'mean': all_mean,
                'std': all_std
            },
            'hit_rates': {
                'under_20_5': under_20_5_count / len(points_list),
                'over_24_5': over_24_5_count / len(points_list),
                'middle': middle_count / len(points_list)
            }
        }
        
    except Exception as e:
        print(f"⚠️  Error loading from S3: {e}")
        print("Falling back to manual recent game data\n")
        return None


def analyze_actual_stats_manual():
    """
    Fallback: Analyze Bam Adebayo's manually entered recent game stats.
    
    Data from recent games (Jan 10-25, 2026):
    22, 26, 32, 25, 4, 30, 22, 29, 6, 13
    """
    print(f"\n{'='*70}")
    print("BAM ADEBAYO MANUAL STATS ANALYSIS")
    print(f"{'='*70}\n")
    
    # Recent 10 games
    recent_games = [22, 26, 32, 25, 4, 30, 22, 29, 6, 13]
    season_avg = 17.9
    
    print("Recent 10 games (Jan 10-25):")
    print(f"  {recent_games}\n")
    
    mean_recent = np.mean(recent_games)
    std_recent = np.std(recent_games, ddof=1)
    
    print(f"Statistics:")
    print(f"  Mean:     {mean_recent:.2f}")
    print(f"  Std Dev:  {std_recent:.2f}")
    print(f"  Season avg: {season_avg}\n")
    
    # Count how many times he hits each threshold
    under_20_5_count = sum(1 for pts in recent_games if pts <= 20.5)
    over_24_5_count = sum(1 for pts in recent_games if pts >= 25)
    middle_count = sum(1 for pts in recent_games if 20.5 < pts < 25)
    
    print(f"Recent performance vs bet thresholds:")
    print(f"  ≤20.5 points (under wins): {under_20_5_count}/10 = {under_20_5_count/10:.1%}")
    print(f"  21-24 points (middle):     {middle_count}/10 = {middle_count/10:.1%}")
    print(f"  ≥25 points (over wins):    {over_24_5_count}/10 = {over_24_5_count/10:.1%}\n")
    
    print(f"{'='*70}\n")
    
    return {
        'all_games': {
            'mean': mean_recent,
            'std': std_recent
        },
        'season': {
            'mean': season_avg,
            'std': std_recent
        }
    }


def main():
    """Main execution function."""
    
    # Try loading from S3 first
    stats = load_bam_stats_from_s3()
    
    # Fall back to manual if S3 fails
    if stats is None:
        stats = analyze_actual_stats_manual()
    
    # Calibrate distribution to match market odds
    calibration = calibrate_distribution_to_market(
        under_threshold=20.5,
        under_odds=-129,
        over_threshold=24.5,
        over_odds=+235,
        initial_mean=stats['season']['mean']
    )
    
    # Use market-calibrated parameters
    market_mean = calibration['mean']
    market_std = calibration['std']
    
    print(f"USING MARKET-CALIBRATED DISTRIBUTION: Mean={market_mean:.1f}, Std={market_std:.1f}\n")
    
    # First, run with equal stakes ($20 each) for baseline
    bet1 = Bet(
        description="Bam Adebayo UNDER 20.5 points",
        threshold=20.5,
        american_odds=-129,
        bet_type='under',
        stake=20.0
    )
    
    bet2 = Bet(
        description="Bam Adebayo OVER 24.5 points",
        threshold=24.5,
        american_odds=+235,
        bet_type='over',
        stake=20.0
    )
    
    print("BASELINE: Equal stakes ($20 each)")
    results = run_simulation(
        bet1=bet1,
        bet2=bet2,
        n_simulations=100000,
        mean=market_mean,
        std=market_std
    )
    
    # Test all combinations from $0-20
    print("\n" + "="*70)
    print("TESTING ALL COMBINATIONS ($0-$20 for each bet)")
    print("="*70)
    
    df_all_combos, best_combo = test_all_stake_combinations(
        max_stake=20,
        mean=market_mean,
        std=market_std,
        n_simulations=100000
    )
    
    # Run detailed simulation with optimal allocation
    print("\n" + "="*70)
    print("DETAILED RESULTS WITH OPTIMAL ALLOCATION (Market-Calibrated)")
    print("="*70 + "\n")
    
    bet1_optimal = Bet(
        description="Bam Adebayo UNDER 20.5 points",
        threshold=20.5,
        american_odds=-129,
        bet_type='under',
        stake=best_combo['under_stake']
    )
    
    bet2_optimal = Bet(
        description="Bam Adebayo OVER 24.5 points",
        threshold=24.5,
        american_odds=+235,
        bet_type='over',
        stake=best_combo['over_stake']
    )
    
    results_optimal = run_simulation(
        bet1=bet1_optimal,
        bet2=bet2_optimal,
        n_simulations=100000,
        mean=market_mean,
        std=market_std
    )
    
    # Compare with historical stats approach
    print("\n" + "="*70)
    print("COMPARISON: HISTORICAL STATS vs MARKET-IMPLIED")
    print("="*70 + "\n")
    
    hist_mean = stats['season']['mean']
    hist_std = stats['all_games']['std']
    
    print(f"If we believe Bam's historical performance (Mean={hist_mean:.1f}, Std={hist_std:.1f}):")
    print(f"  This implies the market may be mispriced\n")
    
    df_hist, best_hist = test_all_stake_combinations(
        max_stake=20,
        mean=hist_mean,
        std=hist_std,
        n_simulations=100000
    )
    
    # Run detailed simulation with historical optimal
    print("\n" + "="*70)
    print("DETAILED RESULTS WITH OPTIMAL ALLOCATION (Historical Stats)")
    print("="*70 + "\n")
    
    bet1_hist = Bet(
        description="Bam Adebayo UNDER 20.5 points",
        threshold=20.5,
        american_odds=-129,
        bet_type='under',
        stake=best_hist['under_stake']
    )
    
    bet2_hist = Bet(
        description="Bam Adebayo OVER 24.5 points",
        threshold=24.5,
        american_odds=+235,
        bet_type='over',
        stake=best_hist['over_stake']
    )
    
    results_hist = run_simulation(
        bet1=bet1_hist,
        bet2=bet2_hist,
        n_simulations=100000,
        mean=hist_mean,
        std=hist_std
    )


if __name__ == '__main__':
    main()
