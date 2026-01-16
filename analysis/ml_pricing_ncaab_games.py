"""
NCAAB Team Points Prediction Model

Goal: Predict individual team scores more accurately than betting markets.

Model Inputs (per team per game):
- x1: Average of last N games' Vegas-implied scores for this team
- x2: Current game's Vegas-implied score (or season avg if missing)
- y: Actual points scored (target)

Strategy:
1. Load all game results and betting lines for season
2. Filter to D1 teams only (364 teams)
3. Join outcomes + lines with team name normalization
4. For missing lines: impute using team's season average implied score
5. Build time-series features (x1 = rolling avg of last N implied scores)
6. Train linear regression model to predict team scores

Usage:
    # Build dataset for 2024-25 season
    python analysis/ml_pricing_ncaab_games.py --season 2024-25 --build-dataset
    
    # Test with single team (Wisconsin Badgers)
    python analysis/ml_pricing_ncaab_games.py --season 2024-25 --single-team "Wisconsin Badgers"
    
    # Train model on full dataset
    python analysis/ml_pricing_ncaab_games.py --season 2024-25 --train --log-examples 10

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

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
    
    Errors:
        Our model error: |81.2 - 84| = 2.8
        Market error: |82.5 - 84| = 1.5
        Winner: Market (closer by 1.3 points)

Usage:
------
    # Build modeling dataset from raw data
    python analysis/ml_pricing_ncaab_games.py --build-dataset
    
    # Train model and evaluate (no examples)
    python analysis/ml_pricing_ncaab_games.py --train
    
    # Train and show 10 example predictions
    python analysis/ml_pricing_ncaab_games.py --train --log-examples 10
    
    # Train and show 99 examples (detailed analysis)
    python analysis/ml_pricing_ncaab_games.py --train --log-examples 99
    
    # Full pipeline: build dataset, train, evaluate, show examples
    python analysis/ml_pricing_ncaab_games.py --build-dataset --train --log-examples 5

Outputs:
--------
    1. Modeling Dataset:
       - Location: data/03_intermediate/ncaab_modeling_dataset.csv
       - Columns: game_date, team, opponent, is_home, 
                  x1_avg_last_10_implied, x2_current_implied, 
                  y_actual_score, consensus_spread, consensus_total, num_books
    
    2. Model Files:
       - data/04_output/ncaab_v0_model_home.pkl
       - data/04_output/ncaab_v0_model_away.pkl
    
    3. Evaluation Results:
       - data/04_output/ncaab_v0_evaluation.csv
       - Columns: model_type, rmse, mae, num_games
    
    4. Predictions CSV:
       - data/04_output/ncaab_v0_predictions.csv
       - Columns: game_date, team, opponent, is_home,
                  x1, x2, actual_score, model_prediction, market_prediction,
                  model_error, market_error

Expected Results:
-----------------
We expect the market to be more accurate overall (lower RMSE) because:
    - Markets aggregate information from many sources
    - Professional oddsmakers have proprietary models
    - Markets adjust for injuries, matchups, and other factors

However, we may find:
    - Our model performs competitively in certain scenarios
    - Market has systematic biases we can exploit
    - Combining both (ensemble) could improve accuracy
    - Certain team types or game situations favor our approach

Success Criteria for v0:
-------------------------
    - Model RMSE within 3-5 points of market RMSE = good baseline
    - Identify specific scenarios where model beats market = valuable insight
    - Understand feature importance (x1 vs x2) = foundation for v1
    - Clean, reproducible pipeline = ready for iteration

Next Steps (v1):
----------------
    - Add more features: opponent strength, home/away splits, tempo
    - Experiment with non-linear models (Random Forest, Gradient Boosting)
    - Incorporate team-specific adjustments
    - Add recency weighting to x1 (recent games matter more)
    - Ensemble: combine our model + market prediction

Context:
--------
User request: "i want to see if we can price NCAAB games better than the lines do"

This is the v0 baseline. The goal is to establish a reproducible pipeline and
baseline performance before building more sophisticated models.

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import boto3
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from io import StringIO

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

from config_loader import get_config

# Load config
CONFIG = get_config()

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 paths
S3_GAME_RESULTS_BUCKET = 'ncaab-betting-mt'
S3_GAME_RESULTS_PATH = 'data/01_input/historical_game_results/'
S3_BETTING_LINES_BUCKET = 'ncaab-betting-mt'
S3_BETTING_LINES_PATH = 'data/01_input/the-odds-api/ncaab/game_lines/'

# Local paths
INTERMEDIATE_DIR = PROJECT_ROOT / 'data' / '03_intermediate'
OUTPUT_DIR = PROJECT_ROOT / 'data' / '04_output'

MODELING_DATASET_PATH = INTERMEDIATE_DIR / 'ncaab_modeling_dataset.csv'
MODEL_HOME_PATH = OUTPUT_DIR / 'ncaab_v0_model_home.pkl'
MODEL_AWAY_PATH = OUTPUT_DIR / 'ncaab_v0_model_away.pkl'
EVALUATION_PATH = OUTPUT_DIR / 'ncaab_v0_evaluation.csv'
PREDICTIONS_PATH = OUTPUT_DIR / 'ncaab_v0_predictions.csv'

# Ensure directories exist
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
MIN_GAMES_FOR_AVG = 10  # Minimum games needed to calculate x1


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_game_results_from_s3():
    """
    Load all game results from S3.
    
    Returns:
        pd.DataFrame with columns: GAME_DATE, GAME_ID, HOME_TEAM, AWAY_TEAM, 
                                    HOME_SCORE, AWAY_SCORE
    """
    print("   Loading game results from S3...")
    
    s3_client = boto3.client('s3')
    
    # List all files in historical_game_results
    response = s3_client.list_objects_v2(
        Bucket=S3_GAME_RESULTS_BUCKET,
        Prefix=S3_GAME_RESULTS_PATH
    )
    
    all_dfs = []
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
    
    print(f"   Found {len(files)} game result files")
    
    for file_key in files:
        obj = s3_client.get_object(Bucket=S3_GAME_RESULTS_BUCKET, Key=file_key)
        df = pd.read_csv(obj['Body'])
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    print(f"   Loaded {len(df)} games")
    
    return df[['GAME_DATE', 'GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_SCORE', 'AWAY_SCORE']]


def load_betting_lines_from_s3():
    """
    Load all betting lines from S3.
    
    Returns:
        pd.DataFrame with columns: date, event_id, home_team, away_team,
                                    consensus_spread, consensus_total, num_books_spread, num_books_total
    """
    print("   Loading betting lines from S3...")
    
    s3_client = boto3.client('s3')
    
    # List all files in game_lines
    response = s3_client.list_objects_v2(
        Bucket=S3_BETTING_LINES_BUCKET,
        Prefix=S3_BETTING_LINES_PATH
    )
    
    all_dfs = []
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
    
    print(f"   Found {len(files)} betting line files")
    
    for file_key in files:
        obj = s3_client.get_object(Bucket=S3_BETTING_LINES_BUCKET, Key=file_key)
        df = pd.read_csv(obj['Body'])
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"   Loaded {len(df)} games with lines")
    
    return df[['date', 'event_id', 'home_team', 'away_team', 'consensus_spread', 'consensus_total', 
               'num_books_spread', 'num_books_total']]


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def calculate_market_implied_scores(spread, total):
    """
    Calculate market-implied scores from spread and total.
    
    Given:
        spread: Home team spread (negative = favorite)
        total: Over/Under line
    
    Returns:
        tuple: (home_implied_score, away_implied_score)
    
    Formula:
        home_implied = (total - spread) / 2
        away_implied = (total + spread) / 2
    
    Example:
        spread = -5.5 (home favored by 5.5)
        total = 145.5
        home_implied = (145.5 - (-5.5)) / 2 = 75.5
        away_implied = (145.5 + (-5.5)) / 2 = 70.0
    """
    if pd.isna(spread) or pd.isna(total):
        return None, None
    
    home_implied = (total - spread) / 2
    away_implied = (total + spread) / 2
    
    return home_implied, away_implied


def build_team_game_rows(games_df):
    """
    Transform game-level data into team-level rows.
    
    Each game becomes 2 rows:
        - Row 1: Home team perspective
        - Row 2: Away team perspective
    
    Args:
        games_df: DataFrame with game-level data
    
    Returns:
        pd.DataFrame with one row per team per game
    """
    rows = []
    
    for _, game in games_df.iterrows():
        # Home team row
        rows.append({
            'game_date': game['game_date'],
            'team': game['home_team'],
            'opponent': game['away_team'],
            'is_home': 1,
            'actual_score': game['home_score'],
            'opp_actual_score': game['away_score'],
            'x2_current_implied': game['home_implied'],
            'opp_x2_current_implied': game['away_implied'],
            'consensus_spread': game['consensus_spread'],
            'consensus_total': game['consensus_total'],
            'num_books': game['num_books_spread']
        })
        
        # Away team row
        rows.append({
            'game_date': game['game_date'],
            'team': game['away_team'],
            'opponent': game['home_team'],
            'is_home': 0,
            'actual_score': game['away_score'],
            'opp_actual_score': game['home_score'],
            'x2_current_implied': game['away_implied'],
            'opp_x2_current_implied': game['home_implied'],
            'consensus_spread': -game['consensus_spread'] if pd.notna(game['consensus_spread']) else None,
            'consensus_total': game['consensus_total'],
            'num_books': game['num_books_spread']
        })
    
    return pd.DataFrame(rows)


def calculate_x1_avg_last_n_implied(team_games_df, n=10):
    """
    Calculate x1 feature: Average of last N Vegas-implied scores for each team.
    
    For each team-game, look back at the previous N games and average the
    market-implied scores (x2) from those games.
    
    Args:
        team_games_df: DataFrame with one row per team per game (sorted by date)
        n: Number of previous games to average (default: 10)
    
    Returns:
        pd.DataFrame with x1_avg_last_N_implied column added
    """
    team_games_df = team_games_df.sort_values(['team', 'game_date']).reset_index(drop=True)
    
    # Calculate rolling average of x2 (current implied) for each team
    team_games_df['x1_avg_last_10_implied'] = team_games_df.groupby('team')['x2_current_implied'].transform(
        lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
    )
    
    return team_games_df


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_modeling_dataset():
    """
    Build modeling dataset from S3 data.
    
    Steps:
        1. Load game results from S3
        2. Load betting lines from S3
        3. Join results + lines
        4. Calculate market-implied scores (x2)
        5. Transform to team-level rows
        6. Calculate x1 (rolling avg of last 10 implied scores)
        7. Save to intermediate directory
    
    Output:
        data/03_intermediate/ncaab_modeling_dataset.csv
    """
    # Step 1: Load game results
    results_df = load_game_results_from_s3()
    
    if results_df.empty:
        print("   ❌ No game results found in S3")
        return
    
    # Step 2: Load betting lines
    lines_df = load_betting_lines_from_s3()
    
    if lines_df.empty:
        print("   ❌ No betting lines found in S3")
        return
    
    # Step 3: Join results + lines
    print("   Joining game results and betting lines...")
    
    # Merge on date and team names
    merged_df = pd.merge(
        results_df,
        lines_df,
        left_on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
        right_on=['date', 'home_team', 'away_team'],
        how='inner'
    )
    
    print(f"   Matched {len(merged_df)} games with both results and lines")
    
    if merged_df.empty:
        print("   ❌ No games matched between results and lines")
        print("   Tip: Check team name consistency between ESPN and The Odds API")
        return
    
    # Step 4: Calculate market-implied scores
    print("   Calculating market-implied scores...")
    
    merged_df['home_implied'], merged_df['away_implied'] = zip(
        *merged_df.apply(
            lambda row: calculate_market_implied_scores(row['consensus_spread'], row['consensus_total']),
            axis=1
        )
    )
    
    # Rename for clarity
    merged_df = merged_df.rename(columns={
        'GAME_DATE': 'game_date',
        'HOME_TEAM': 'home_team',
        'AWAY_TEAM': 'away_team',
        'HOME_SCORE': 'home_score',
        'AWAY_SCORE': 'away_score'
    })
    
    # Step 5: Transform to team-level rows
    print("   Transforming to team-level rows...")
    
    team_games_df = build_team_game_rows(merged_df)
    
    print(f"   Created {len(team_games_df)} team-game rows")
    
    # Step 6: Calculate x1 (rolling avg of last 10 implied scores)
    print("   Calculating x1 (avg last 10 implied scores)...")
    
    team_games_df = calculate_x1_avg_last_n_implied(team_games_df, n=MIN_GAMES_FOR_AVG)
    
    # Step 7: Clean up and save
    print("   Saving modeling dataset...")
    
    # Remove rows with missing values
    team_games_df = team_games_df.dropna(subset=['x2_current_implied', 'actual_score'])
    
    # Reorder columns
    output_cols = [
        'game_date', 'team', 'opponent', 'is_home',
        'x1_avg_last_10_implied', 'x2_current_implied',
        'actual_score', 'opp_actual_score',
        'consensus_spread', 'consensus_total', 'num_books'
    ]
    
    team_games_df = team_games_df[output_cols]
    
    # Save
    team_games_df.to_csv(MODELING_DATASET_PATH, index=False)
    
    print(f"   ✅ Saved {len(team_games_df)} rows to {MODELING_DATASET_PATH}")
    print()
    print(f"   Dataset Summary:")
    print(f"      Date range: {team_games_df['game_date'].min()} to {team_games_df['game_date'].max()}")
    print(f"      Unique teams: {team_games_df['team'].nunique()}")
    print(f"      Home games: {team_games_df['is_home'].sum()}")
    print(f"      Away games: {len(team_games_df) - team_games_df['is_home'].sum()}")
    print(f"      Rows with x1 (10+ games): {team_games_df['x1_avg_last_10_implied'].notna().sum()}")


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(log_examples=0):
    """
    Train models and evaluate against market baseline.
    
    Steps:
        1. Load modeling dataset
        2. Split train/test chronologically (80/20)
        3. Train separate models for home/away teams
        4. Make predictions on test set
        5. Calculate RMSE/MAE vs market baseline
        6. Save models and predictions
        7. Log examples if requested
    
    Args:
        log_examples: Number of example predictions to show (0 = none)
    """
    # Step 1: Load dataset
    print("   Loading modeling dataset...")
    
    if not MODELING_DATASET_PATH.exists():
        print(f"   ❌ Dataset not found: {MODELING_DATASET_PATH}")
        print("   Run with --build-dataset first")
        return
    
    df = pd.read_csv(MODELING_DATASET_PATH)
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    print(f"   Loaded {len(df)} rows")
    
    # Step 2: Split train/test chronologically
    print("   Splitting train/test (80/20 chronologically)...")
    
    df = df.sort_values('game_date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"   Train: {len(train_df)} rows ({train_df['game_date'].min()} to {train_df['game_date'].max()})")
    print(f"   Test:  {len(test_df)} rows ({test_df['game_date'].min()} to {test_df['game_date'].max()})")
    
    # Step 3: Train models
    print("   Training models...")
    
    # Separate home and away teams
    train_home = train_df[train_df['is_home'] == 1]
    train_away = train_df[train_df['is_home'] == 0]
    test_home = test_df[test_df['is_home'] == 1]
    test_away = test_df[test_df['is_home'] == 0]
    
    # Features: x1 (avg last 10 implied), x2 (current implied)
    # If x1 is missing, use only x2
    
    # Train home model
    train_home_with_x1 = train_home[train_home['x1_avg_last_10_implied'].notna()]
    train_home_no_x1 = train_home[train_home['x1_avg_last_10_implied'].isna()]
    
    X_train_home = train_home_with_x1[['x1_avg_last_10_implied', 'x2_current_implied']].values
    y_train_home = train_home_with_x1['actual_score'].values
    
    model_home = LinearRegression()
    model_home.fit(X_train_home, y_train_home)
    
    # Train away model
    train_away_with_x1 = train_away[train_away['x1_avg_last_10_implied'].notna()]
    train_away_no_x1 = train_away[train_away['x1_avg_last_10_implied'].isna()]
    
    X_train_away = train_away_with_x1[['x1_avg_last_10_implied', 'x2_current_implied']].values
    y_train_away = train_away_with_x1['actual_score'].values
    
    model_away = LinearRegression()
    model_away.fit(X_train_away, y_train_away)
    
    print(f"   Home model trained on {len(X_train_home)} rows")
    print(f"   Away model trained on {len(X_train_away)} rows")
    
    # Step 4: Make predictions on test set
    print("   Making predictions on test set...")
    
    # Home predictions
    test_home_with_x1 = test_home[test_home['x1_avg_last_10_implied'].notna()].copy()
    test_home_no_x1 = test_home[test_home['x1_avg_last_10_implied'].isna()].copy()
    
    if len(test_home_with_x1) > 0:
        X_test_home = test_home_with_x1[['x1_avg_last_10_implied', 'x2_current_implied']].values
        test_home_with_x1['model_prediction'] = model_home.predict(X_test_home)
    
    # For games without x1, use market prediction (x2 only)
    if len(test_home_no_x1) > 0:
        test_home_no_x1['model_prediction'] = test_home_no_x1['x2_current_implied']
    
    # Away predictions
    test_away_with_x1 = test_away[test_away['x1_avg_last_10_implied'].notna()].copy()
    test_away_no_x1 = test_away[test_away['x1_avg_last_10_implied'].isna()].copy()
    
    if len(test_away_with_x1) > 0:
        X_test_away = test_away_with_x1[['x1_avg_last_10_implied', 'x2_current_implied']].values
        test_away_with_x1['model_prediction'] = model_away.predict(X_test_away)
    
    if len(test_away_no_x1) > 0:
        test_away_no_x1['model_prediction'] = test_away_no_x1['x2_current_implied']
    
    # Combine
    test_df = pd.concat([test_home_with_x1, test_home_no_x1, test_away_with_x1, test_away_no_x1], ignore_index=True)
    test_df = test_df.sort_values('game_date').reset_index(drop=True)
    
    # Add market baseline prediction (just x2)
    test_df['market_prediction'] = test_df['x2_current_implied']
    
    # Calculate errors
    test_df['model_error'] = np.abs(test_df['model_prediction'] - test_df['actual_score'])
    test_df['market_error'] = np.abs(test_df['market_prediction'] - test_df['actual_score'])
    
    # Step 5: Evaluate
    print()
    print("   " + "=" * 76)
    print("   EVALUATION RESULTS")
    print("   " + "=" * 76)
    
    # Overall metrics
    model_mae = test_df['model_error'].mean()
    model_rmse = np.sqrt((test_df['model_error'] ** 2).mean())
    
    market_mae = test_df['market_error'].mean()
    market_rmse = np.sqrt((test_df['market_error'] ** 2).mean())
    
    print(f"   Test Set: {len(test_df)} predictions")
    print()
    print(f"   Our v0 Model:")
    print(f"      MAE:  {model_mae:.2f} points")
    print(f"      RMSE: {model_rmse:.2f} points")
    print()
    print(f"   Market Baseline:")
    print(f"      MAE:  {market_mae:.2f} points")
    print(f"      RMSE: {market_rmse:.2f} points")
    print()
    print(f"   Difference:")
    print(f"      MAE:  {model_mae - market_mae:+.2f} points (negative = we're better)")
    print(f"      RMSE: {model_rmse - market_rmse:+.2f} points")
    
    # Win rate
    model_wins = (test_df['model_error'] < test_df['market_error']).sum()
    ties = (test_df['model_error'] == test_df['market_error']).sum()
    market_wins = (test_df['market_error'] < test_df['model_error']).sum()
    
    print()
    print(f"   Head-to-Head:")
    print(f"      Our model closer: {model_wins} ({model_wins/len(test_df)*100:.1f}%)")
    print(f"      Market closer:    {market_wins} ({market_wins/len(test_df)*100:.1f}%)")
    print(f"      Ties:             {ties} ({ties/len(test_df)*100:.1f}%)")
    
    print("   " + "=" * 76)
    
    # Step 6: Save models and predictions
    print()
    print("   Saving models and predictions...")
    
    # Save models
    with open(MODEL_HOME_PATH, 'wb') as f:
        pickle.dump(model_home, f)
    
    with open(MODEL_AWAY_PATH, 'wb') as f:
        pickle.dump(model_away, f)
    
    print(f"   ✅ Saved home model to {MODEL_HOME_PATH}")
    print(f"   ✅ Saved away model to {MODEL_AWAY_PATH}")
    
    # Save predictions
    test_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"   ✅ Saved predictions to {PREDICTIONS_PATH}")
    
    # Save evaluation metrics
    eval_df = pd.DataFrame([
        {'model_type': 'our_v0_model', 'mae': model_mae, 'rmse': model_rmse, 'num_games': len(test_df)},
        {'model_type': 'market_baseline', 'mae': market_mae, 'rmse': market_rmse, 'num_games': len(test_df)}
    ])
    eval_df.to_csv(EVALUATION_PATH, index=False)
    print(f"   ✅ Saved evaluation to {EVALUATION_PATH}")
    
    # Step 7: Log examples if requested
    if log_examples > 0:
        print()
        print("   " + "=" * 76)
        print(f"   EXAMPLE PREDICTIONS (showing {min(log_examples, len(test_df))} examples)")
        print("   " + "=" * 76)
        
        # Sample examples (mix of good and bad predictions)
        sample_df = test_df.sample(n=min(log_examples, len(test_df)), random_state=42)
        
        for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
            print()
            print(f"   Example {i}/{len(sample_df)}:")
            print("   " + "-" * 76)
            print(f"   Game Date: {row['game_date'].date()}")
            print(f"   Team:      {row['team']}")
            print(f"   Opponent:  {row['opponent']}")
            print(f"   Location:  {'Home' if row['is_home'] == 1 else 'Away'}")
            print()
            print(f"   Features (X):")
            if pd.notna(row['x1_avg_last_10_implied']):
                print(f"      x1 (avg last 10 Vegas implied): {row['x1_avg_last_10_implied']:.1f}")
            else:
                print(f"      x1 (avg last 10 Vegas implied): N/A (fewer than 10 games)")
            print(f"      x2 (current game consensus):    {row['x2_current_implied']:.1f}")
            print()
            print(f"   Target (y):")
            print(f"      Actual score: {row['actual_score']:.0f}")
            print()
            print(f"   Predictions:")
            print(f"      Our model:  {row['model_prediction']:.1f}")
            print(f"      Market:     {row['market_prediction']:.1f}")
            print()
            print(f"   Market Data:")
            print(f"      Consensus spread: {row['consensus_spread']:+.1f}")
            print(f"      Consensus total:  {row['consensus_total']:.1f}")
            print(f"      Num books:        {row['num_books']:.0f}")
            print()
            print(f"   Errors:")
            print(f"      Our model: |{row['model_prediction']:.1f} - {row['actual_score']:.0f}| = {row['model_error']:.1f}")
            print(f"      Market:    |{row['market_prediction']:.1f} - {row['actual_score']:.0f}| = {row['market_error']:.1f}")
            
            if row['model_error'] < row['market_error']:
                winner = "Our model"
                diff = row['market_error'] - row['model_error']
            elif row['market_error'] < row['model_error']:
                winner = "Market"
                diff = row['model_error'] - row['market_error']
            else:
                winner = "Tie"
                diff = 0
            
            print(f"      Winner:    {winner}{f' (closer by {diff:.1f} points)' if diff > 0 else ''}")
        
        print("   " + "=" * 76)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='NCAAB Score Prediction - Market-Based v0 Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --build-dataset
  %(prog)s --train --log-examples 10
  %(prog)s --build-dataset --train --log-examples 5
        """
    )
    
    parser.add_argument('--build-dataset', action='store_true',
                       help='Build modeling dataset from S3 data')
    parser.add_argument('--train', action='store_true',
                       help='Train models and evaluate')
    parser.add_argument('--log-examples', type=int, default=0,
                       help='Number of example predictions to show (0 = none)')
    
    args = parser.parse_args()
    
    # Default: if no flags, show help
    if not args.build_dataset and not args.train:
        parser.print_help()
        return
    
    print("=" * 80)
    print("NCAAB SCORE PREDICTION - Market-Based v0 Model")
    print("=" * 80)
    print()
    
    # Step 1: Build dataset
    if args.build_dataset:
        print("📊 Building modeling dataset...")
        build_modeling_dataset()
        print()
    
    # Step 2: Train and evaluate
    if args.train:
        print("🤖 Training models...")
        train_and_evaluate(log_examples=args.log_examples)
        print()
    
    print("=" * 80)
    print("✅ Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

