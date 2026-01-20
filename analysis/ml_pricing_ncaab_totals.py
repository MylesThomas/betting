"""
NCAAB Game Totals Prediction Model

Goal: Predict game totals (over/under) more accurately than betting markets.

Context:
This is a companion to ml_pricing_ncaab_games_v2.py which focuses on spreads.
Same framework, different target: testing if totals markets are less efficient.

Model Inputs (per game):
- x1_home: Home team's average of last N games' Vegas-implied scores
- x1_away: Away team's average of last N games' Vegas-implied scores
- x2_home: Home team's current game Vegas-implied score
- x2_away: Away team's current game Vegas-implied score
- x3: Binary indicator if conference game (1) or non-conference (0)
- y: Actual total points (home + away) - TARGET

Market baseline: consensus_total from betting lines

Strategy:
1. Load team-level features from ml_pricing_ncaab_games_v2.py
2. Reconstruct game-level rows (merge home + away features)
3. Train model to predict total = home_score + away_score
4. Compare model MAE vs market MAE
5. Walk-forward validation by date

Usage:
    # Quick test on 2024-25 season with walk-forward validation
    python analysis/ml_pricing_ncaab_totals.py --season 2024-25 --use-cache --walk-forward
    
    # Compare model snapshots (train at each date, test on future)
    python analysis/ml_pricing_ncaab_totals.py --season 2024-25 --use-cache --compare-snapshots

Author: Thomas Myles
Date: 2026-01-18
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

# Add paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'tmp'))

from config_loader import get_config
from join_ncaab_outcomes_and_lines import (
    load_game_outcomes,
    load_game_lines,
    join_outcomes_and_lines,
    SEASON_DATES
)
from ncaab_conference_data import get_team_conference, is_conference_game

# Configuration
CONFIG = get_config()
PROJECT_ROOT = project_root
MODELING_DATASET_DIR = PROJECT_ROOT / 'data' / '03_intermediate'

# Load full conference mapping
CONFERENCE_MAPPING_PATH = PROJECT_ROOT / 'tmp' / 'ncaab_conference_mapping.csv'
if CONFERENCE_MAPPING_PATH.exists():
    _conf_df = pd.read_csv(CONFERENCE_MAPPING_PATH)
    FULL_CONFERENCE_MAP = dict(zip(_conf_df['team_name_espn'], _conf_df['conference']))
else:
    FULL_CONFERENCE_MAP = {}

MIN_GAMES_FOR_AVG = 10

# =============================================================================
# LINEAR REGRESSION (no sklearn)
# =============================================================================

def train_linear_regression(X, y):
    """Train linear regression using numpy."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    return coefficients


def predict_linear_regression(X, coefficients):
    """Make predictions using trained coefficients."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    return X_with_intercept @ coefficients


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data(season='2024-25', use_cache=True):
    """
    Load all data and prepare for totals modeling.
    
    Steps:
    1. Load game outcomes from S3/cache
    2. Load betting lines from S3/cache  
    3. Join with team name normalization
    4. Filter to D1 teams only
    
    Args:
        season: Season string (e.g., '2024-25')
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with joined game + line data
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}. Available: {list(SEASON_DATES.keys())}")
    
    start_date, end_date = SEASON_DATES[season]
    
    print(f"{'='*80}")
    print(f"LOADING DATA FOR TOTALS: {season}")
    print(f"{'='*80}\n")
    
    # Step 1: Load outcomes
    print("1️⃣  Loading game outcomes...")
    outcomes_df = load_game_outcomes(start_date, end_date, use_cache=use_cache)
    print(f"   ✅ Loaded {len(outcomes_df):,} games\n")
    
    # Step 2: Load lines
    print("2️⃣  Loading betting lines...")
    lines_df = load_game_lines(start_date, end_date, use_cache=use_cache)
    print(f"   ✅ Loaded {len(lines_df):,} game lines\n")
    
    # Step 3: Join (automatically filters to D1 teams)
    print("3️⃣  Joining outcomes + lines (with D1 filter)...")
    joined_df, stats = join_outcomes_and_lines(outcomes_df, lines_df, min_games=5)
    
    print(f"\n{'='*80}")
    print(f"DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Total games: {len(joined_df):,}")
    print(f"Games with lines: {stats['matched']:,} ({stats['coverage_pct']:.1f}%)")
    print(f"Games without lines: {stats['unmatched']:,}")
    print(f"Date range: {joined_df['GAME_DATE'].min()} to {joined_df['GAME_DATE'].max()}")
    print(f"{'='*80}\n")
    
    return joined_df


def calculate_market_implied_scores(spread, total):
    """
    Calculate market-implied scores from spread and total.
    
    Args:
        spread: Home team spread (negative = home favored)
        total: Over/under line
    
    Returns:
        tuple: (home_implied_score, away_implied_score)
    """
    if pd.isna(spread) or pd.isna(total):
        return None, None
    
    home_implied = (total - spread) / 2
    away_implied = (total + spread) / 2
    
    return home_implied, away_implied


def calculate_rolling_features(team_df, n=MIN_GAMES_FOR_AVG):
    """
    Calculate rolling average of implied scores (x1).
    Uses expanding window (all history).
    
    Args:
        team_df: DataFrame for single team (sorted by date)
        n: Minimum games for calculation
    
    Returns:
        DataFrame with x1_avg_implied column added
    """
    # Expanding average (all history)
    team_df['x1_avg_implied'] = team_df['implied_score'].expanding(min_periods=1).mean()
    
    # Set to NaN if less than n games
    team_df.loc[team_df.index < n, 'x1_avg_implied'] = np.nan
    
    return team_df


def build_team_level_features(joined_df):
    """
    Transform game-level data to team-level features.
    
    Each game becomes 2 rows (one per team) with:
    - x1: Rolling avg of implied scores (expanding window)
    - x2: Current game implied score
    - x3: Conference game indicator
    - y: Actual score
    
    Args:
        joined_df: DataFrame from join_outcomes_and_lines
    
    Returns:
        DataFrame with team-level features
    """
    rows = []
    
    # Transform each game to 2 rows
    for _, game in joined_df.iterrows():
        # Calculate implied scores
        if pd.notna(game.get('consensus_spread')) and pd.notna(game.get('consensus_total')):
            home_impl, away_impl = calculate_market_implied_scores(
                game['consensus_spread'], 
                game['consensus_total']
            )
        else:
            home_impl, away_impl = None, None
        
        # Conference info
        home_conf = FULL_CONFERENCE_MAP.get(game['HOME_TEAM'], None)
        away_conf = FULL_CONFERENCE_MAP.get(game['AWAY_TEAM'], None)
        is_conf_game = (home_conf is not None and away_conf is not None and home_conf == away_conf)
        
        # Home team row
        rows.append({
            'game_date': game['GAME_DATE'],
            'team': game['HOME_TEAM'],
            'opponent': game['AWAY_TEAM'],
            'is_home': True,
            'actual_score': game['HOME_SCORE'],
            'implied_score': home_impl,
            'is_conference_game': is_conf_game,
            'consensus_total': game.get('consensus_total'),
            'consensus_spread': game.get('consensus_spread')
        })
        
        # Away team row
        rows.append({
            'game_date': game['GAME_DATE'],
            'team': game['AWAY_TEAM'],
            'opponent': game['HOME_TEAM'],
            'is_home': False,
            'actual_score': game['AWAY_SCORE'],
            'implied_score': away_impl,
            'is_conference_game': is_conf_game,
            'consensus_total': game.get('consensus_total'),
            'consensus_spread': game.get('consensus_spread')
        })
    
    team_df = pd.DataFrame(rows)
    team_df['game_date'] = pd.to_datetime(team_df['game_date'])
    team_df = team_df.sort_values(['team', 'game_date']).reset_index(drop=True)
    
    # Calculate x1 for each team separately
    print("   Calculating x1 (rolling avg implied scores)...")
    team_dfs = []
    for team in team_df['team'].unique():
        team_subset = team_df[team_df['team'] == team].copy()
        team_subset = calculate_rolling_features(team_subset)
        team_dfs.append(team_subset)
    
    team_df = pd.concat(team_dfs, ignore_index=True)
    
    # x2 is current game implied score
    team_df['x2_current_implied'] = team_df['implied_score']
    
    return team_df


def build_game_level_features(team_df):
    """
    Merge home + away team features back into game-level format.
    
    Args:
        team_df: DataFrame with team-level features
    
    Returns:
        DataFrame with game-level features for totals prediction
    """
    # Separate home and away
    home_df = team_df[team_df['is_home'] == True].copy()
    away_df = team_df[team_df['is_home'] == False].copy()
    
    # Merge on game_date + teams
    games_df = pd.merge(
        home_df[['game_date', 'team', 'opponent', 'x1_avg_implied', 'x2_current_implied',
                 'actual_score', 'is_conference_game', 'consensus_total']],
        away_df[['game_date', 'team', 'opponent', 'x1_avg_implied', 'x2_current_implied', 'actual_score']],
        left_on=['game_date', 'team', 'opponent'],
        right_on=['game_date', 'opponent', 'team'],
        suffixes=('_home', '_away'),
        how='inner'
    )
    
    # Calculate actual total
    games_df['actual_total'] = games_df['actual_score_home'] + games_df['actual_score_away']
    
    # Conference game indicator
    games_df['x3_is_conf_game'] = games_df['is_conference_game'].astype(int)
    
    # Clean up column names
    games_df = games_df.rename(columns={
        'team_home': 'home_team',
        'opponent_home': 'away_team',
        'x1_avg_implied_home': 'x1_home',
        'x1_avg_implied_away': 'x1_away',
        'x2_current_implied_home': 'x2_home',
        'x2_current_implied_away': 'x2_away'
    })
    
    return games_df[['game_date', 'home_team', 'away_team', 'x1_home', 'x1_away',
                     'x2_home', 'x2_away', 'x3_is_conf_game', 'actual_total', 'consensus_total']]


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(games_df, min_train_games=50):
    """
    Walk-forward validation: Train on past, test on future.
    
    For each unique game date:
    1. Train on all games before this date (expanding window)
    2. Test on games on this date
    3. Track model MAE vs market MAE
    
    Args:
        games_df: Game-level DataFrame with features
        min_train_games: Minimum games needed to start testing
    
    Returns:
        dict with results
    """
    # Sort by date
    games_df = games_df.sort_values('game_date').reset_index(drop=True)
    
    # Get unique dates
    unique_dates = sorted(games_df['game_date'].unique())
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: TOTALS")
    print(f"{'='*80}")
    print(f"Total games: {len(games_df):,}")
    print(f"Unique dates: {len(unique_dates):,}")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"Min training games: {min_train_games}")
    print(f"{'='*80}\n")
    
    # Track results
    results = {
        'dates': [],
        'train_size': [],
        'test_size': [],
        'model_mae': [],
        'market_mae': [],
        'model_predictions': [],
        'market_predictions': [],
        'actuals': []
    }
    
    # For each date, train on past and test on current
    for i, test_date in enumerate(unique_dates):
        # Training set: all games before test_date
        train_mask = games_df['game_date'] < test_date
        test_mask = games_df['game_date'] == test_date
        
        train_df = games_df[train_mask]
        test_df = games_df[test_mask]
        
        # Skip if not enough training data
        if len(train_df) < min_train_games:
            continue
        
        # Skip if no test games
        if len(test_df) == 0:
            continue
        
        # Prepare features (drop rows with missing x1)
        train_clean = train_df.dropna(subset=['x1_home', 'x1_away', 'x2_home', 'x2_away', 'actual_total'])
        test_clean = test_df.dropna(subset=['x1_home', 'x1_away', 'x2_home', 'x2_away', 'actual_total', 'consensus_total'])
        
        if len(train_clean) == 0 or len(test_clean) == 0:
            continue
        
        # Features: x1_home, x1_away, x2_home, x2_away, x3_is_conf_game
        feature_cols = ['x1_home', 'x1_away', 'x2_home', 'x2_away', 'x3_is_conf_game']
        
        X_train = train_clean[feature_cols].values
        y_train = train_clean['actual_total'].values
        
        X_test = test_clean[feature_cols].values
        y_test = test_clean['actual_total'].values
        market_test = test_clean['consensus_total'].values
        
        # Train model
        try:
            coefs = train_linear_regression(X_train, y_train)
            model_preds = predict_linear_regression(X_test, coefs)
        except:
            continue
        
        # Calculate MAE
        model_mae = np.mean(np.abs(model_preds - y_test))
        market_mae = np.mean(np.abs(market_test - y_test))
        
        # Store results
        results['dates'].append(test_date)
        results['train_size'].append(len(train_clean))
        results['test_size'].append(len(test_clean))
        results['model_mae'].append(model_mae)
        results['market_mae'].append(market_mae)
        results['model_predictions'].extend(model_preds.tolist())
        results['market_predictions'].extend(market_test.tolist())
        results['actuals'].extend(y_test.tolist())
        
        # Progress update every 20 dates
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(unique_dates)} dates...")
    
    return results


def compare_model_snapshots(games_df, snapshot_freq=7, min_train_games=100):
    """
    Compare model performance at different points in the season.
    
    Train model at regular intervals (every N days), then test on ALL future games.
    Shows how model evolves over the season.
    
    Args:
        games_df: Game-level DataFrame
        snapshot_freq: Train model every N days
        min_train_games: Minimum games to start
    
    Returns:
        dict with snapshot results
    """
    games_df = games_df.sort_values('game_date').reset_index(drop=True)
    unique_dates = sorted(games_df['game_date'].unique())
    
    print(f"\n{'='*80}")
    print(f"MODEL SNAPSHOT COMPARISON: TOTALS")
    print(f"{'='*80}")
    print(f"Snapshot frequency: Every {snapshot_freq} days")
    print(f"Min training games: {min_train_games}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Sample dates for training
    snapshot_dates = unique_dates[::snapshot_freq]
    
    for train_date in snapshot_dates:
        # Train on all games up to train_date
        train_mask = games_df['game_date'] <= train_date
        test_mask = games_df['game_date'] > train_date
        
        train_df = games_df[train_mask]
        test_df = games_df[test_mask]
        
        if len(train_df) < min_train_games or len(test_df) == 0:
            continue
        
        # Clean data
        train_clean = train_df.dropna(subset=['x1_home', 'x1_away', 'x2_home', 'x2_away', 'actual_total'])
        test_clean = test_df.dropna(subset=['x1_home', 'x1_away', 'x2_home', 'x2_away', 'actual_total', 'consensus_total'])
        
        if len(train_clean) == 0 or len(test_clean) == 0:
            continue
        
        # Features
        feature_cols = ['x1_home', 'x1_away', 'x2_home', 'x2_away', 'x3_is_conf_game']
        
        X_train = train_clean[feature_cols].values
        y_train = train_clean['actual_total'].values
        
        X_test = test_clean[feature_cols].values
        y_test = test_clean['actual_total'].values
        market_test = test_clean['consensus_total'].values
        
        # Train and predict
        try:
            coefs = train_linear_regression(X_train, y_train)
            model_preds = predict_linear_regression(X_test, coefs)
        except:
            continue
        
        # Calculate MAE
        model_mae = np.mean(np.abs(model_preds - y_test))
        market_mae = np.mean(np.abs(market_test - y_test))
        
        results.append({
            'train_date': train_date,
            'train_size': len(train_clean),
            'test_size': len(test_clean),
            'model_mae': model_mae,
            'market_mae': market_mae,
            'mae_diff': model_mae - market_mae
        })
        
        print(f"   Snapshot {train_date}: Train={len(train_clean)}, Test={len(test_clean)}, "
              f"Model MAE={model_mae:.2f}, Market MAE={market_mae:.2f}")
    
    return pd.DataFrame(results)


# =============================================================================
# REPORTING
# =============================================================================

def print_walk_forward_results(results):
    """Print summary of walk-forward validation results."""
    if len(results['dates']) == 0:
        print("❌ No results to display")
        return
    
    # Overall MAE
    all_model_preds = np.array(results['model_predictions'])
    all_market_preds = np.array(results['market_predictions'])
    all_actuals = np.array(results['actuals'])
    
    overall_model_mae = np.mean(np.abs(all_model_preds - all_actuals))
    overall_market_mae = np.mean(np.abs(all_market_preds - all_actuals))
    
    # Win rate (games where model beats market)
    per_game_model_errors = np.abs(all_model_preds - all_actuals)
    per_game_market_errors = np.abs(all_market_preds - all_actuals)
    model_wins = (per_game_model_errors < per_game_market_errors).sum()
    total_games = len(all_actuals)
    win_rate = model_wins / total_games * 100
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD RESULTS: TOTALS")
    print(f"{'='*80}")
    print(f"Test dates: {len(results['dates'])}")
    print(f"Total test games: {total_games:,}")
    print(f"")
    print(f"📊 Overall Performance:")
    print(f"   Model MAE:  {overall_model_mae:.2f}")
    print(f"   Market MAE: {overall_market_mae:.2f}")
    print(f"   Difference: {overall_model_mae - overall_market_mae:+.2f} {'✅ Model wins' if overall_model_mae < overall_market_mae else '❌ Market wins'}")
    print(f"")
    print(f"🎯 Win Rate:")
    print(f"   Model beats market: {model_wins:,}/{total_games:,} games ({win_rate:.1f}%)")
    print(f"   Need >50% to be profitable")
    print(f"{'='*80}\n")


def print_snapshot_results(snapshots_df):
    """Print summary of model snapshot comparison."""
    if len(snapshots_df) == 0:
        print("❌ No snapshot results")
        return
    
    print(f"\n{'='*80}")
    print(f"MODEL SNAPSHOT SUMMARY: TOTALS")
    print(f"{'='*80}")
    print(f"Total snapshots: {len(snapshots_df)}")
    print(f"")
    print(f"📊 Aggregate Performance:")
    print(f"   Avg Model MAE:  {snapshots_df['model_mae'].mean():.2f}")
    print(f"   Avg Market MAE: {snapshots_df['market_mae'].mean():.2f}")
    print(f"   Avg Difference: {snapshots_df['mae_diff'].mean():+.2f}")
    print(f"")
    print(f"🔍 Best/Worst Snapshots:")
    best = snapshots_df.loc[snapshots_df['mae_diff'].idxmin()]
    worst = snapshots_df.loc[snapshots_df['mae_diff'].idxmax()]
    print(f"   Best:  {best['train_date']} (diff={best['mae_diff']:+.2f})")
    print(f"   Worst: {worst['train_date']} (diff={worst['mae_diff']:+.2f})")
    print(f"{'='*80}\n")
    
    # Show trend
    model_wins = (snapshots_df['mae_diff'] < 0).sum()
    print(f"✅ Model wins: {model_wins}/{len(snapshots_df)} snapshots ({model_wins/len(snapshots_df)*100:.1f}%)\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NCAAB Totals Prediction Model')
    parser.add_argument('--season', type=str, default='2024-25', help='Season (e.g., 2024-25)')
    parser.add_argument('--use-cache', action='store_true', help='Use cached data')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--compare-snapshots', action='store_true', help='Compare model snapshots')
    parser.add_argument('--snapshot-freq', type=int, default=7, help='Snapshot frequency (days)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"NCAAB TOTALS MODEL: {args.season}")
    print(f"{'='*80}\n")
    
    # Step 1: Load and join game data
    print("📊 Loading data...")
    joined_df = load_and_prepare_data(args.season, use_cache=args.use_cache)
    
    # Step 2: Build team-level features
    print("🔧 Building team-level features...")
    team_df = build_team_level_features(joined_df)
    print(f"   ✅ Created {len(team_df):,} team-game rows\n")
    
    # Step 3: Build game-level features
    print("🎯 Building game-level features for totals...")
    games_df = build_game_level_features(team_df)
    print(f"   ✅ Created {len(games_df):,} games\n")
    
    # Filter to games with totals
    games_with_totals = games_df.dropna(subset=['consensus_total'])
    print(f"📈 Games with totals: {len(games_with_totals):,} ({len(games_with_totals)/len(games_df)*100:.1f}%)\n")
    
    # Step 4: Run analysis
    if args.walk_forward:
        results = walk_forward_validation(games_with_totals)
        print_walk_forward_results(results)
    
    if args.compare_snapshots:
        snapshots_df = compare_model_snapshots(games_with_totals, snapshot_freq=args.snapshot_freq)
        print_snapshot_results(snapshots_df)
    
    if not args.walk_forward and not args.compare_snapshots:
        print("❓ No analysis selected. Use --walk-forward or --compare-snapshots")


if __name__ == '__main__':
    main()

