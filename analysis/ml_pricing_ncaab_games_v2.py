"""
NCAAB Team Points Prediction Model

Goal: Predict individual team scores more accurately than betting markets.

Model Inputs (per team per game):
- x1: Average of last N games' Vegas-implied scores for this team
- x2: Current game's Vegas-implied score (or season avg if missing)
- x3: Binary indicator if conference game (1) or non-conference (0)
- y: Actual points scored (target)

Strategy:
1. Load all game results and betting lines for season
2. Filter to D1 teams only (364 teams)
3. Join outcomes + lines with team name normalization
4. Add conference information to identify conference vs non-conference games
5. For missing lines: impute using team's season average implied score
6. Build time-series features (x1 = rolling avg of last N implied scores)
7. Train linear regression model to predict team scores

Usage:
    # Build dataset for 2024-25 season
    python analysis/ml_pricing_ncaab_games_v2.py --season 2024-25 --build-dataset
    
    # Test with single team (Wisconsin Badgers) - simple analysis
    python analysis/ml_pricing_ncaab_games_v2.py --season 2024-25 --use-cache --single-team "Wisconsin Badgers"
    
    # Walk-forward validation for single team (rolling window, last 10 games)
    python analysis/ml_pricing_ncaab_games_v2.py --season 2024-25 --use-cache --single-team "Wisconsin Badgers" --walk-forward
    
    # Walk-forward validation for single team (expanding window, all history)
    python analysis/ml_pricing_ncaab_games_v2.py --season 2024-25 --use-cache --single-team "Wisconsin Badgers" --walk-forward --use-all-history
    
    # Compare rolling vs expanding window across ALL D1 teams
    python analysis/ml_pricing_ncaab_games_v2.py --season 2024-25 --use-cache --compare-x1-methods
    
    # Train model on full dataset
    python analysis/ml_pricing_ncaab_games_v2.py --season 2024-25 --train --log-examples 10

Author: Thomas Myles
Date: 2026-01-16
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'tmp'))

from ncaab_conference_data import get_team_conference, is_conference_game

# Load full conference mapping from CSV (includes all 364 teams)
CONFERENCE_MAPPING_PATH = project_root / 'tmp' / 'ncaab_conference_mapping.csv'
if CONFERENCE_MAPPING_PATH.exists():
    _conf_df = pd.read_csv(CONFERENCE_MAPPING_PATH)
    FULL_CONFERENCE_MAP = dict(zip(_conf_df['team_name_espn'], _conf_df['conference']))
else:
    FULL_CONFERENCE_MAP = {}

# Linear regression using numpy (no sklearn needed)
def train_linear_regression(X, y):
    """
    Train linear regression using numpy.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    
    Returns:
        coefficients: [intercept, coef1, coef2, ...]
    """
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Solve using least squares: (X'X)^-1 X'y
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    return coefficients


def predict_linear_regression(X, coefficients):
    """
    Make predictions using trained coefficients.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        coefficients: [intercept, coef1, coef2, ...]
    
    Returns:
        predictions: Vector of predicted values
    """
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    return X_with_intercept @ coefficients

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

from config_loader import get_config
from join_ncaab_outcomes_and_lines import (
    load_game_outcomes,
    load_game_lines,
    join_outcomes_and_lines,
    SEASON_DATES
)

# Configuration
CONFIG = get_config()
MIN_GAMES_FOR_AVG = 10  # Minimum games to calculate x1 (avg last N)
MODELING_DATASET_DIR = PROJECT_ROOT / 'data' / '03_intermediate'
MODELING_DATASET_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_market_implied_scores(spread, total):
    """
    Calculate market-implied scores from spread and total.
    
    Args:
        spread: Home team spread (negative = home favored)
        total: Over/under line
    
    Returns:
        tuple: (home_implied_score, away_implied_score)
    
    Example:
        spread = -5.5 (home favored)
        total = 145.5
        home_implied = (145.5 - (-5.5)) / 2 = 75.5
        away_implied = (145.5 + (-5.5)) / 2 = 70.0
    """
    if pd.isna(spread) or pd.isna(total):
        return None, None
    
    home_implied = (total - spread) / 2
    away_implied = (total + spread) / 2
    
    return home_implied, away_implied


def build_team_game_rows(joined_df):
    """
    Transform game-level data to team-level rows.
    
    Each game becomes 2 rows (one per team) with:
    - team, opponent, is_home
    - actual_score, opp_actual_score
    - implied_score (from lines, or None if missing)
    - spread, total, num_books
    
    Args:
        joined_df: DataFrame from join_outcomes_and_lines
    
    Returns:
        DataFrame with one row per team per game
    """
    rows = []
    
    for _, game in joined_df.iterrows():
        # Calculate implied scores if lines available
        if pd.notna(game.get('consensus_spread')) and pd.notna(game.get('consensus_total')):
            home_impl, away_impl = calculate_market_implied_scores(
                game['consensus_spread'], 
                game['consensus_total']
            )
        else:
            home_impl, away_impl = None, None
        
        # Get conference information (defaults to None if not found)
        home_conf = FULL_CONFERENCE_MAP.get(game['HOME_TEAM'], None)
        away_conf = FULL_CONFERENCE_MAP.get(game['AWAY_TEAM'], None)
        
        # Determine if conference game
        is_conf_game = (home_conf is not None and away_conf is not None and home_conf == away_conf)
        
        # Home team row
        rows.append({
            'game_date': game['GAME_DATE'],
            'team': game['HOME_TEAM'],
            'opponent': game['AWAY_TEAM'],
            'team_conference': home_conf,
            'opponent_conference': away_conf,
            'is_conference_game': is_conf_game,
            'is_home': True,
            'actual_score': game['HOME_SCORE'],
            'opp_actual_score': game['AWAY_SCORE'],
            'implied_score': home_impl,
            'consensus_spread': game.get('consensus_spread'),
            'consensus_total': game.get('consensus_total'),
            'num_books': game.get('num_books_spread', 0)
        })
        
        # Away team row
        rows.append({
            'game_date': game['GAME_DATE'],
            'team': game['AWAY_TEAM'],
            'opponent': game['HOME_TEAM'],
            'team_conference': away_conf,
            'opponent_conference': home_conf,
            'is_conference_game': is_conf_game,
            'is_home': False,
            'actual_score': game['AWAY_SCORE'],
            'opp_actual_score': game['HOME_SCORE'],
            'implied_score': away_impl,
            'consensus_spread': -game.get('consensus_spread') if pd.notna(game.get('consensus_spread')) else None,
            'consensus_total': game.get('consensus_total'),
            'num_books': game.get('num_books_spread', 0)
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['team', 'game_date']).reset_index(drop=True)
    
    return df


def impute_missing_implied_scores(team_df):
    """
    Impute missing implied scores using team's season average.
    
    For games without betting lines, use the team's season average
    implied score from games that do have lines.
    
    Args:
        team_df: DataFrame for single team, sorted by game_date
    
    Returns:
        DataFrame with implied_score column filled
    """
    # Calculate season average from available data
    season_avg = team_df[team_df['implied_score'].notna()]['implied_score'].mean()
    
    # Fill missing with season average
    team_df['implied_score_filled'] = team_df['implied_score'].fillna(season_avg)
    
    return team_df


def calculate_rolling_features(team_df, n=MIN_GAMES_FOR_AVG, x1_window_type='rolling'):
    """
    Calculate x1 (rolling average of last N implied scores).
    
    Args:
        team_df: DataFrame for single team, sorted by game_date
        n: Number of previous games to average (default 10)
        use_all_history: If True, use all available history instead of last N games
    
    Returns:
        DataFrame with x1_avg_last_n column added
    """
    if x1_window_type == 'all':
        # Use expanding window - all games up to current
        team_df['x1_avg_last_n'] = team_df['implied_score_filled'].shift(1).expanding(min_periods=1).mean()
    else:
        # Use rolling window - last N games
        team_df['x1_avg_last_n'] = team_df['implied_score_filled'].shift(1).rolling(
            window=n, min_periods=1
        ).mean()
    
    return team_df


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_and_prepare_data(season='2024-25', use_cache=True):
    """
    Load all data and prepare for modeling.
    
    Steps:
    1. Load game outcomes from S3/cache
    2. Load betting lines from S3/cache
    3. Join with team name normalization
    4. Filter to D1 teams only (using min_games=5)
    
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
    print(f"LOADING DATA FOR {season}")
    print(f"{'='*80}\n")
    
    # Step 1: Load outcomes
    print("1️⃣  Loading game outcomes...")
    outcomes_df = load_game_outcomes(start_date, end_date, use_cache=use_cache)
    print(f"   ✅ Loaded {len(outcomes_df):,} games\n")
    
    # Step 2: Load lines
    print("2️⃣  Loading betting lines...")
    lines_df = load_game_lines(start_date, end_date, use_cache=use_cache)
    print(f"   ✅ Loaded {len(lines_df):,} game lines\n")
    
    # Step 3: Join (automatically filters to D1 teams with min_games=5)
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


def analyze_single_team_walk_forward(joined_df, team_name, x1_window_type='rolling', return_stats=False):
    """
    Walk-forward validation: train model before each game, predict, then update.
    
    For each game N:
    1. Train model on games 1 to N-1
    2. Show model coefficients (β₀ + β₁*x1 + β₂*x2 + β₃*x3)
    3. Predict game N score
    4. Show actual score and error
    5. Repeat for next game
    
    Args:
        joined_df: DataFrame from load_and_prepare_data
        team_name: Team name to analyze (e.g., "Wisconsin Badgers")
        use_all_history: If True, use all available games for x1 instead of last 10
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: {team_name}")
    print(f"{'='*80}\n")
    
    if x1_window_type == 'all':
        print("ℹ️  Using ALL game history for x1 (expanding window)")
    else:
        print(f"ℹ️  Using LAST {MIN_GAMES_FOR_AVG} games for x1 (rolling window)")
    print()
    
    # Build team-level rows
    team_df = build_team_game_rows(joined_df)
    
    # Filter to this team
    team_games = team_df[team_df['team'] == team_name].copy()
    
    if len(team_games) == 0:
        print(f"❌ No games found for {team_name}")
        return
    
    # Impute missing implied scores
    team_games = impute_missing_implied_scores(team_games)
    
    # Calculate rolling features with appropriate window
    team_games = calculate_rolling_features(team_games, n=MIN_GAMES_FOR_AVG, x1_window_type=x1_window_type)
    
    print(f"Total games: {len(team_games)}\n")
    print(f"{'='*80}")
    print("WALK-FORWARD TRAINING & PREDICTION")
    print(f"{'='*80}\n")
    
    # Track statistics
    model_errors = []
    market_errors = []
    model_wins = 0
    market_wins = 0
    ties = 0
    
    # Walk forward through games
    for idx in range(len(team_games)):
        game = team_games.iloc[idx]
        game_num = idx + 1
        
        print(f"{'='*80}")
        print(f"GAME {game_num}: {game['game_date']}")
        print(f"{'='*80}")
        location = "vs" if game['is_home'] else "@"
        print(f"{team_name} {location} {game['opponent']}")
        print(f"Actual Score: {int(game['actual_score'])}-{int(game['opp_actual_score'])}")
        
        # Can only predict if we have x1 (need at least 1 previous game)
        if idx == 0 or pd.isna(game['x1_avg_last_n']):
            # Show conference information
            team_conf = game.get('team_conference', 'Unknown')
            opp_conf = game.get('opponent_conference', 'Unknown')
            conf_game_str = "Conference Game" if game['is_conference_game'] else "Non-Conference"
            x3_val = float(game['is_conference_game'])
            
            print(f"{team_name} ({team_conf}) vs {game['opponent']} ({opp_conf}) - {conf_game_str}")
            
            if pd.notna(game['consensus_spread']) and pd.notna(game['consensus_total']):
                # Determine favorite
                if game['is_home']:
                    if game['consensus_spread'] < 0:
                        fav_str = f"{team_name} {abs(game['consensus_spread']):.1f}"
                    else:
                        fav_str = f"{game['opponent']} {abs(game['consensus_spread']):.1f}"
                else:
                    if game['consensus_spread'] < 0:
                        fav_str = f"{game['opponent']} {abs(game['consensus_spread']):.1f}"
                    else:
                        fav_str = f"{team_name} {abs(game['consensus_spread']):.1f}"
                print(f"Lines: {fav_str}, O/U {game['consensus_total']:.1f}")
            print("\n⏭️  No prediction (insufficient training data)")
            print(f"   x1: Not available (need previous games)")
            print(f"   x2: {game['implied_score_filled']:.1f}")
            print(f"   x3: {int(x3_val)}\n")
            continue
        
        # Prepare training data (all games before this one with valid features)
        train_data = team_games.iloc[:idx].copy()
        train_data = train_data[train_data['x1_avg_last_n'].notna()].copy()
        
        if len(train_data) == 0:
            print("\n⏭️  No prediction (no valid training games)\n")
            continue
        
        # Training features and target
        # Convert is_conference_game to float explicitly
        train_data_features = train_data[['x1_avg_last_n', 'implied_score_filled']].copy()
        train_data_features['is_conference_game'] = train_data['is_conference_game'].astype(float)
        X_train = train_data_features.values
        y_train = train_data['actual_score'].values
        
        # Train model
        coefficients = train_linear_regression(X_train, y_train)
        intercept = coefficients[0]
        coef_x1 = coefficients[1]
        coef_x2 = coefficients[2]
        coef_x3 = coefficients[3]
        
        # Current game features
        x1_val = game['x1_avg_last_n']
        x2_val = game['implied_score_filled']
        x3_val = float(game['is_conference_game'])  # Convert boolean to float (0.0 or 1.0)
        
        # Get the last N implied scores for x1 breakdown
        last_n_implied = team_games.iloc[:idx]['implied_score_filled'].tail(MIN_GAMES_FOR_AVG).values
        
        # Make prediction
        X_test = np.array([[x1_val, x2_val, x3_val]])
        prediction = predict_linear_regression(X_test, coefficients)[0]
        
        # Show conference information
        team_conf = game.get('team_conference', 'Unknown')
        opp_conf = game.get('opponent_conference', 'Unknown')
        conf_game_str = "Conference Game" if game['is_conference_game'] else "Non-Conference"
        
        # Show predicted score with spread/total right under header
        if pd.notna(game['consensus_spread']) and pd.notna(game['consensus_total']):
            # Calculate opponent's predicted score based on spread
            # consensus_spread is from team's perspective (negative = favorite, positive = underdog)
            # If team is -16.5 (favorite) and scores 82, opponent = 82 + (-16.5) = 65.5
            # If team is +5.0 (underdog) and scores 75, opponent = 75 + 5.0 = 80
            opponent_prediction = prediction + game['consensus_spread']
            
            # Determine favorite for display with proper +/- sign
            # Negative spread = team is favorite, Positive spread = team is underdog (opponent is favorite)
            if game['consensus_spread'] < 0:
                # Team is favorite
                fav_str = f"{team_name} {game['consensus_spread']:.1f}"
            else:
                # Opponent is favorite
                fav_str = f"{game['opponent']} -{game['consensus_spread']:.1f}"
            
            print(f"{team_name} ({team_conf}) vs {game['opponent']} ({opp_conf}) - {conf_game_str}")
            print(f"Predicted Score: {team_name} {prediction:.1f}, {game['opponent']} {opponent_prediction:.1f} ({fav_str}, O/U {game['consensus_total']:.1f})")
        else:
            print(f"{team_name} ({team_conf}) vs {game['opponent']} ({opp_conf}) - {conf_game_str}")
            print(f"Predicted Score: {team_name} {prediction:.1f} (no lines available)")
        
        print()
        
        # Calculate error
        actual = game['actual_score']
        error = abs(prediction - actual)
        
        # Print results
        print(f"📊 Model trained on {len(train_data)} previous games")
        print(f"\n   Model: y = {intercept:.2f} + {coef_x1:.3f}*x1 + {coef_x2:.3f}*x2 + {coef_x3:.3f}*x3")
        print(f"\n   Features (X):")
        # Dynamic label based on window type
        if x1_window_type == 'all':
            x1_label = f"x1 (avg of {len(last_n_implied)} available)"
        else:
            x1_label = f"x1 (avg last {MIN_GAMES_FOR_AVG})"
        
        print(f"      {x1_label}: {x1_val:.1f}")
        if len(last_n_implied) > 0:
            implied_str = ', '.join([f"{v:.1f}" for v in last_n_implied])
            print(f"         Last {len(last_n_implied)} implied scores: [{implied_str}]")
            print(f"         Average: {last_n_implied.mean():.1f}")
        print(f"      x2 (current implied):    {x2_val:.1f}")
        if pd.notna(game['consensus_spread']) and pd.notna(game['consensus_total']):
            print(f"         Spread: {game['consensus_spread']:.1f}, O/U: {game['consensus_total']:.1f}")
            print(f"         Calculation: ({game['consensus_total']:.1f} - {game['consensus_spread']:.1f}) / 2 = {x2_val:.1f}")
        else:
            print(f"         (Imputed: season average)")
        print(f"      x3 (is conference game): {int(x3_val)}")
        print(f"         {team_name} ({team_conf}) vs {game['opponent']} ({opp_conf})")
        print(f"\n   Calculation:")
        print(f"      y = {intercept:.2f} + {coef_x1:.3f}*{x1_val:.1f} + {coef_x2:.3f}*{x2_val:.1f} + {coef_x3:.3f}*{x3_val:.0f}")
        print(f"        = {intercept:.2f} + {coef_x1*x1_val:.2f} + {coef_x2*x2_val:.2f} + {coef_x3*x3_val:.2f}")
        print(f"        = {prediction:.1f}")
        
        # Calculate error and deltas
        error = abs(prediction - actual)
        market_error = abs(x2_val - actual)
        model_delta = actual - prediction
        market_delta = actual - x2_val
        
        # Determine which is better
        if error < market_error:
            better = "Model"
            model_wins += 1
        elif error > market_error:
            better = "Market"
            market_wins += 1
        else:
            better = "Tie"
            ties += 1
        
        print(f"\n   Results (Actual: {actual:.0f}):")
        print(f"      Model Predicted:  {prediction:.1f} (delta: {model_delta:+.1f}, error: {error:.1f})")
        print(f"      Market Predicted: {x2_val:.1f} (delta: {market_delta:+.1f}, error: {market_error:.1f})")
        print(f"      Better Prediction: {better}")
        
        # Track for summary statistics
        model_errors.append(error)
        market_errors.append(market_error)
        
        print()
    
    # Print summary statistics
    print(f"{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    if len(model_errors) > 0:
        model_mae = np.mean(model_errors)
        market_mae = np.mean(market_errors)
        model_rmse = np.sqrt(np.mean([e**2 for e in model_errors]))
        market_rmse = np.sqrt(np.mean([e**2 for e in market_errors]))
        
        total_predictions = len(model_errors)
        
        print(f"Games with predictions: {total_predictions}")
        print()
        print(f"Win/Loss/Tie Record:")
        print(f"   Model Wins:  {model_wins} ({100*model_wins/total_predictions:.1f}%)")
        print(f"   Market Wins: {market_wins} ({100*market_wins/total_predictions:.1f}%)")
        print(f"   Ties:        {ties} ({100*ties/total_predictions:.1f}%)")
        print()
        print(f"Mean Absolute Error (MAE):")
        print(f"   Model:  {model_mae:.2f} points")
        print(f"   Market: {market_mae:.2f} points")
        if model_mae < market_mae:
            print(f"   ✅ Model better by {market_mae - model_mae:.2f} points")
        else:
            print(f"   ❌ Market better by {model_mae - market_mae:.2f} points")
        print()
        print(f"Root Mean Squared Error (RMSE):")
        print(f"   Model:  {model_rmse:.2f} points")
        print(f"   Market: {market_rmse:.2f} points")
        if model_rmse < market_rmse:
            print(f"   ✅ Model better by {market_rmse - model_rmse:.2f} points")
        else:
            print(f"   ❌ Market better by {model_rmse - market_rmse:.2f} points")
        
        # Return stats if requested (for comparison mode)
        if return_stats:
            return {
                'n_games': total_predictions,
                'model_wins': model_wins,
                'market_wins': market_wins,
                'ties': ties,
                'model_mae': model_mae,
                'market_mae': market_mae,
                'model_rmse': model_rmse,
                'market_rmse': market_rmse
            }
    else:
        print("No predictions made (insufficient data)")
        if return_stats:
            return {
                'n_games': 0,
                'model_wins': 0,
                'market_wins': 0,
                'ties': 0,
                'model_mae': 0,
                'market_mae': 0,
                'model_rmse': 0,
                'market_rmse': 0
            }
    
    print(f"\n{'='*80}\n")


def analyze_single_team(joined_df, team_name):
    """
    Analyze sequence of games for a single team.
    
    Shows chronological game results with:
    - Game date, opponent, location
    - Actual score vs opponent score
    - Betting lines (spread, total)
    - Implied score (if available)
    - Missing data flags
    
    Args:
        joined_df: DataFrame from load_and_prepare_data
        team_name: Team name to analyze (e.g., "Wisconsin Badgers")
    """
    print(f"\n{'='*80}")
    print(f"SINGLE TEAM ANALYSIS: {team_name}")
    print(f"{'='*80}\n")
    
    # Build team-level rows
    team_df = build_team_game_rows(joined_df)
    
    # Filter to this team
    team_games = team_df[team_df['team'] == team_name].copy()
    
    if len(team_games) == 0:
        print(f"❌ No games found for {team_name}")
        print(f"\nAvailable teams (first 20):")
        available_teams = sorted(team_df['team'].unique())
        for i, t in enumerate(available_teams[:20], 1):
            print(f"   {i}. {t}")
        return
    
    # Impute missing implied scores
    team_games = impute_missing_implied_scores(team_games)
    
    # Calculate rolling features
    team_games = calculate_rolling_features(team_games, n=MIN_GAMES_FOR_AVG)
    
    print(f"Total games: {len(team_games)}")
    print(f"Home games: {team_games['is_home'].sum()}")
    print(f"Away games: {(~team_games['is_home']).sum()}")
    print(f"Games with lines: {team_games['implied_score'].notna().sum()}")
    print(f"Games without lines: {team_games['implied_score'].isna().sum()}")
    print(f"\n{'='*80}")
    print("GAME SEQUENCE")
    print(f"{'='*80}\n")
    
    # Display game by game
    for idx, (i, game) in enumerate(team_games.iterrows(), 1):
        location = "vs" if game['is_home'] else "@"
        has_lines = "✅" if pd.notna(game['implied_score']) else "❌"
        
        print(f"Game {idx}: {game['game_date']}")
        print(f"  {team_name} {location} {game['opponent']}")
        print(f"  Score: {int(game['actual_score'])}-{int(game['opp_actual_score'])}")
        print(f"  Lines: {has_lines}")
        
        if pd.notna(game['implied_score']):
            print(f"    Spread: {game['consensus_spread']:.1f}, Total: {game['consensus_total']:.1f}")
            print(f"    Implied score: {game['implied_score']:.1f}")
        else:
            print(f"    Imputed (season avg): {game['implied_score_filled']:.1f}")
        
        print(f"  Features:")
        if pd.notna(game['x1_avg_last_n']):
            print(f"    x1 (avg last {MIN_GAMES_FOR_AVG}): {game['x1_avg_last_n']:.1f}")
        else:
            print(f"    x1: Not enough games yet")
        print(f"    x2 (current implied): {game['implied_score_filled']:.1f}")
        print()
    
    print(f"{'='*80}\n")


def compare_x1_methods_all_teams(joined_df):
    """
    Compare rolling window vs expanding window for x1 across all D1 teams.
    Returns summary statistics for both methods.
    """
    print("="*80)
    print("COMPARING X1 CALCULATION METHODS ACROSS ALL TEAMS")
    print("="*80)
    print()
    print("Method 1: Rolling Window (last 10 games)")
    print("Method 2: Expanding Window (all available history)")
    print()
    
    # Get all unique teams (filter out NaN)
    all_teams = pd.concat([joined_df['home_team'], joined_df['away_team']]).dropna().unique()
    all_teams = sorted(all_teams)
    
    print(f"Found {len(all_teams)} unique teams")
    print()
    
    # Build team-level rows once
    print("🔨 Building team-level rows for all teams...")
    team_df = build_team_game_rows(joined_df)
    print(f"✅ Built {len(team_df)} team-game rows")
    print()
    
    # Store results for both methods
    rolling_results = []
    expanding_results = []
    
    teams_tested = 0
    teams_skipped = 0
    
    for team in all_teams:
        # Get team games from pre-built team_df
        team_games = team_df[team_df['team'] == team].copy()
        
        # Skip teams with too few games
        if len(team_games) < 15:
            teams_skipped += 1
            continue
        
        teams_tested += 1
        
        # Test both methods silently (suppress output)
        import sys
        from io import StringIO
        
        # Method 1: Rolling
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        rolling_stats = None
        try:
            rolling_stats = analyze_single_team_walk_forward(
                joined_df, team, x1_window_type='rolling', return_stats=True
            )
        except Exception as e:
            pass
        finally:
            sys.stdout = old_stdout
        
        # Method 2: Expanding
        sys.stdout = StringIO()
        expanding_stats = None
        try:
            expanding_stats = analyze_single_team_walk_forward(
                joined_df, team, x1_window_type='all', return_stats=True
            )
        except Exception as e:
            pass
        finally:
            sys.stdout = old_stdout
        
        # Only add to results if both methods succeeded
        if rolling_stats and expanding_stats:
            rolling_results.append({
                'team': team,
                'mae': rolling_stats['model_mae'],
                'rmse': rolling_stats['model_rmse'],
                'wins': rolling_stats['model_wins'],
                'games': rolling_stats['n_games']
            })
            expanding_results.append({
                'team': team,
                'mae': expanding_stats['model_mae'],
                'rmse': expanding_stats['model_rmse'],
                'wins': expanding_stats['model_wins'],
                'games': expanding_stats['n_games']
            })
        
        # Progress update
        if teams_tested % 50 == 0:
            print(f"   Tested {teams_tested} teams...")
    
    print(f"✅ Tested {teams_tested} teams (skipped {teams_skipped} with <15 games)")
    print(f"   Rolling results collected: {len(rolling_results)}")
    print(f"   Expanding results collected: {len(expanding_results)}")
    print()
    
    if len(rolling_results) == 0 or len(expanding_results) == 0:
        print("❌ No results collected - all teams failed validation")
        return
    
    # Calculate aggregate statistics
    rolling_df = pd.DataFrame(rolling_results)
    expanding_df = pd.DataFrame(expanding_results)
    
    print("="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    print()
    
    # Overall MAE
    rolling_mae_avg = rolling_df['mae'].mean()
    expanding_mae_avg = expanding_df['mae'].mean()
    mae_diff = rolling_mae_avg - expanding_mae_avg
    
    print(f"Average MAE across all teams:")
    print(f"   Rolling Window:   {rolling_mae_avg:.2f} points")
    print(f"   Expanding Window: {expanding_mae_avg:.2f} points")
    print(f"   Difference:       {mae_diff:+.2f} points {'✅ Expanding better' if mae_diff > 0 else '❌ Rolling better'}")
    print()
    
    # Overall RMSE
    rolling_rmse_avg = rolling_df['rmse'].mean()
    expanding_rmse_avg = expanding_df['rmse'].mean()
    rmse_diff = rolling_rmse_avg - expanding_rmse_avg
    
    print(f"Average RMSE across all teams:")
    print(f"   Rolling Window:   {rolling_rmse_avg:.2f} points")
    print(f"   Expanding Window: {expanding_rmse_avg:.2f} points")
    print(f"   Difference:       {rmse_diff:+.2f} points {'✅ Expanding better' if rmse_diff > 0 else '❌ Rolling better'}")
    print()
    
    # Win rate
    total_games = rolling_df['games'].sum()
    rolling_wins = rolling_df['wins'].sum()
    expanding_wins = expanding_df['wins'].sum()
    
    print(f"Model vs Market Win Rate:")
    print(f"   Rolling Window:   {rolling_wins}/{total_games} ({100*rolling_wins/total_games:.1f}%)")
    print(f"   Expanding Window: {expanding_wins}/{total_games} ({100*expanding_wins/total_games:.1f}%)")
    print()
    
    # Teams where each method won
    rolling_better = (rolling_df['mae'] < expanding_df['mae']).sum()
    expanding_better = (expanding_df['mae'] < rolling_df['mae']).sum()
    ties = (expanding_df['mae'] == rolling_df['mae']).sum()
    
    print(f"Head-to-Head by Team (based on MAE):")
    print(f"   Rolling better:   {rolling_better} teams ({100*rolling_better/teams_tested:.1f}%)")
    print(f"   Expanding better: {expanding_better} teams ({100*expanding_better/teams_tested:.1f}%)")
    print(f"   Ties:             {ties} teams")
    print()
    
    # Top 10 improvements
    comparison_df = pd.DataFrame({
        'team': rolling_df['team'],
        'rolling_mae': rolling_df['mae'],
        'expanding_mae': expanding_df['mae'],
        'improvement': rolling_df['mae'] - expanding_df['mae']
    })
    comparison_df = comparison_df.sort_values('improvement', ascending=False)
    
    print("Top 10 teams where Expanding Window helped most:")
    for i, row in comparison_df.head(10).iterrows():
        print(f"   {row['team']:<40} {row['improvement']:+.2f} pts ({row['rolling_mae']:.2f} → {row['expanding_mae']:.2f})")
    print()
    
    print("Top 10 teams where Rolling Window was better:")
    for i, row in comparison_df.tail(10).iterrows():
        print(f"   {row['team']:<40} {row['improvement']:+.2f} pts ({row['rolling_mae']:.2f} → {row['expanding_mae']:.2f})")
    print()
    
    print("="*80)
    print()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='NCAAB Team Points Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--season', type=str, default='2024-25',
                       help='Season to process (e.g., "2024-25")')
    parser.add_argument('--build-dataset', action='store_true',
                       help='Build modeling dataset')
    parser.add_argument('--single-team', type=str, default=None,
                       help='Analyze single team (e.g., "Wisconsin Badgers")')
    parser.add_argument('--walk-forward', action='store_true',
                       help='Use walk-forward validation (train before each game)')
    parser.add_argument('--use-all-history', action='store_true',
                       help='Use all available game history for x1 instead of last 10 games')
    parser.add_argument('--compare-x1-methods', action='store_true',
                       help='Compare rolling vs expanding window for x1 across all teams')
    parser.add_argument('--train', action='store_true',
                       help='Train model and evaluate')
    parser.add_argument('--log-examples', type=int, default=0,
                       help='Number of example predictions to show')
    parser.add_argument('--use-cache', action='store_true',
                       help='Use cached data (faster)')
    
    args = parser.parse_args()
    
    # Load data
    joined_df = load_and_prepare_data(season=args.season, use_cache=args.use_cache)
    
    # Compare x1 methods across all teams
    if args.compare_x1_methods:
        compare_x1_methods_all_teams(joined_df)
        return
    
    # Single team analysis
    if args.single_team:
        if args.walk_forward:
            x1_window = 'all' if args.use_all_history else 'rolling'
            analyze_single_team_walk_forward(joined_df, args.single_team, x1_window_type=x1_window)
        else:
            analyze_single_team(joined_df, args.single_team)
        return
    
    # Build dataset
    if args.build_dataset:
        print("📊 Building modeling dataset...")
        # TODO: Implement build_modeling_dataset
        print("   ⚠️  Not implemented yet")
        return
    
    # Train model
    if args.train:
        print("🤖 Training model...")
        # TODO: Implement train_and_evaluate
        print("   ⚠️  Not implemented yet")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()

