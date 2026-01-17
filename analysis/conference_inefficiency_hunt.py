"""
Hunt for conference-specific market inefficiencies.

Test if we can beat the market on specific conference games.
Start with A-10, but make it easy to test any conference.

Strategy:
1. Train model on ALL games (pooled)
2. Test ONLY on target conference games
3. Compare model vs market on this subset
4. Look for patterns the market is missing
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'tmp'))

from ncaab_conference_data import get_team_conference, NCAAB_CONFERENCE_MAPPING_2025_26

MIN_GAMES_FOR_AVG = 10


def train_linear_regression(X, y):
    """Train linear regression using numpy."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    return coefficients


def predict_linear_regression(X, coefficients):
    """Make predictions using trained coefficients."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    return X_with_intercept @ coefficients


# Import functions from main modeling script
sys.path.append(str(project_root / 'analysis'))
from ml_pricing_ncaab_games_v2 import (
    load_and_prepare_data,
    build_team_game_rows,
    impute_missing_implied_scores
)


def analyze_conference(joined_df, target_conference='Atlantic 10', x1_window_type='all'):
    """
    Analyze if we can beat market on specific conference games.
    
    Strategy:
    - Train on ALL games (full dataset)
    - Test ONLY on target conference games
    - See if model beats market on this subset
    """
    print("="*80)
    print(f"CONFERENCE INEFFICIENCY HUNT: {target_conference}")
    print("="*80)
    print()
    print(f"Strategy: Train on ALL games, test on {target_conference} games only")
    print(f"x1 method: {x1_window_type}")
    print()
    
    # Build team-level rows
    print("🔨 Building team-level rows...")
    team_df = build_team_game_rows(joined_df)
    print(f"✅ Built {len(team_df)} team-game rows")
    
    # Impute missing implied scores
    team_df = impute_missing_implied_scores(team_df)
    print(f"✅ Imputed missing implied scores")
    
    # Add conference info using direct dictionary lookup (using 2025-26 mapping as proxy for 2024-25)
    team_df['team_conference'] = team_df['team'].map(NCAAB_CONFERENCE_MAPPING_2025_26)
    team_df['opponent_conference'] = team_df['opponent'].map(NCAAB_CONFERENCE_MAPPING_2025_26)
    team_df['is_conference_game'] = (
        (team_df['team_conference'].notna()) & 
        (team_df['opponent_conference'].notna()) & 
        (team_df['team_conference'] == team_df['opponent_conference'])
    )
    print(f"✅ Added conference information")
    print()
    
    # Filter to target conference games (for testing)
    conf_games = team_df[team_df['team_conference'] == target_conference].copy()
    print(f"🎯 Found {len(conf_games)} {target_conference} team-games")
    print(f"   Teams in conference: {conf_games['team'].nunique()}")
    print(f"   Conference games only: {conf_games['is_conference_game'].sum()}")
    print(f"   Non-conference games: {(~conf_games['is_conference_game']).sum()}")
    print()
    
    if len(conf_games) == 0:
        print(f"❌ No games found for {target_conference}")
        return None
    
    # Get unique dates
    dates = sorted(team_df['game_date'].unique())
    print(f"📅 Season has {len(dates)} game dates")
    print()
    
    # Walk-forward validation
    predictions = []
    actuals = []
    market_preds = []
    game_details = []
    
    print("🔄 Running walk-forward validation...")
    print()
    
    for date_idx, current_date in enumerate(dates):
        # Train on all games before this date
        train_df = team_df[team_df['game_date'] < current_date].copy()
        
        # Test on target conference games on this date
        test_df = conf_games[conf_games['game_date'] == current_date].copy()
        
        if len(test_df) == 0 or len(train_df) < 100:
            continue
        
        # Prepare training features
        train_with_features = []
        for team in train_df['team'].unique():
            team_games = train_df[train_df['team'] == team].sort_values('game_date').copy()
            
            for idx in range(len(team_games)):
                if idx == 0:
                    continue
                
                history = team_games.iloc[:idx]
                
                if x1_window_type == 'rolling':
                    x1_val = history['implied_score_filled'].tail(MIN_GAMES_FOR_AVG).mean()
                else:
                    x1_val = history['implied_score_filled'].mean()
                
                if pd.isna(x1_val):
                    continue
                
                game_copy = team_games.iloc[idx].copy()
                game_copy['x1_avg_last_n'] = x1_val
                game_copy['x2_implied'] = game_copy['implied_score_filled']
                game_copy['x3_conf'] = float(game_copy['is_conference_game'])
                
                train_with_features.append(game_copy)
        
        if len(train_with_features) == 0:
            continue
        
        train_features_df = pd.DataFrame(train_with_features)
        train_features_df = train_features_df.dropna(subset=['x1_avg_last_n', 'x2_implied', 'x3_conf', 'actual_score'])
        
        if len(train_features_df) < 50:
            continue
        
        # Train model
        X_train = train_features_df[['x1_avg_last_n', 'x2_implied', 'x3_conf']].values
        y_train = train_features_df['actual_score'].values
        
        coefficients = train_linear_regression(X_train, y_train)
        
        # Prepare test features
        test_with_features = []
        for _, test_game in test_df.iterrows():
            team = test_game['team']
            
            # Get team history (all games before this date)
            team_history = team_df[(team_df['team'] == team) & (team_df['game_date'] < current_date)].copy()
            
            if len(team_history) == 0:
                continue
            
            if x1_window_type == 'rolling':
                x1_val = team_history['implied_score_filled'].tail(MIN_GAMES_FOR_AVG).mean()
            else:
                x1_val = team_history['implied_score_filled'].mean()
            
            if pd.isna(x1_val):
                continue
            
            test_game_copy = test_game.copy()
            test_game_copy['x1_avg_last_n'] = x1_val
            test_game_copy['x2_implied'] = test_game_copy['implied_score_filled']
            test_game_copy['x3_conf'] = float(test_game_copy['is_conference_game'])
            
            test_with_features.append(test_game_copy)
        
        if len(test_with_features) == 0:
            continue
        
        test_features_df = pd.DataFrame(test_with_features)
        test_features_df = test_features_df.dropna(subset=['x1_avg_last_n', 'x2_implied', 'x3_conf', 'actual_score'])
        
        if len(test_features_df) == 0:
            continue
        
        # Make predictions
        X_test = test_features_df[['x1_avg_last_n', 'x2_implied', 'x3_conf']].values
        preds = predict_linear_regression(X_test, coefficients)
        
        predictions.extend(preds)
        actuals.extend(test_features_df['actual_score'].values)
        market_preds.extend(test_features_df['x2_implied'].values)
        
        # Store game details
        for idx, pred in enumerate(preds):
            game = test_features_df.iloc[idx]
            game_details.append({
                'date': game['game_date'],
                'team': game['team'],
                'opponent': game['opponent'],
                'is_conf_game': game['is_conference_game'],
                'actual': game['actual_score'],
                'predicted': pred,
                'market': game['x2_implied'],
                'model_error': abs(pred - game['actual_score']),
                'market_error': abs(game['x2_implied'] - game['actual_score'])
            })
        
        if (date_idx + 1) % 20 == 0:
            print(f"   Processed {date_idx + 1}/{len(dates)} dates, {len(predictions)} predictions")
    
    print(f"✅ Total predictions: {len(predictions)}")
    print()
    
    # Calculate results
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    market_preds = np.array(market_preds)
    
    model_errors = np.abs(predictions - actuals)
    market_errors = np.abs(market_preds - actuals)
    
    model_mae = np.mean(model_errors)
    market_mae = np.mean(market_errors)
    model_rmse = np.sqrt(np.mean(model_errors ** 2))
    market_rmse = np.sqrt(np.mean(market_errors ** 2))
    
    model_wins = np.sum(model_errors < market_errors)
    market_wins = np.sum(model_errors > market_errors)
    ties = np.sum(model_errors == market_errors)
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(f"Total {target_conference} games predicted: {len(predictions)}")
    print()
    print(f"MAE:")
    print(f"   Model:  {model_mae:.2f} points")
    print(f"   Market: {market_mae:.2f} points")
    if model_mae < market_mae:
        print(f"   ✅ Model better by {market_mae - model_mae:.2f} points")
    else:
        print(f"   ❌ Market better by {model_mae - market_mae:.2f} points")
    print()
    print(f"RMSE:")
    print(f"   Model:  {model_rmse:.2f} points")
    print(f"   Market: {market_rmse:.2f} points")
    if model_rmse < market_rmse:
        print(f"   ✅ Model better by {market_rmse - model_rmse:.2f} points")
    else:
        print(f"   ❌ Market better by {model_rmse - market_rmse:.2f} points")
    print()
    print(f"Win Rate:")
    print(f"   Model wins:  {model_wins} ({100*model_wins/len(predictions):.1f}%)")
    print(f"   Market wins: {market_wins} ({100*market_wins/len(predictions):.1f}%)")
    print(f"   Ties:        {ties} ({100*ties/len(predictions):.1f}%)")
    print()
    
    # Break down by conference vs non-conference
    details_df = pd.DataFrame(game_details)
    
    conf_only = details_df[details_df['is_conf_game']]
    non_conf = details_df[~details_df['is_conf_game']]
    
    if len(conf_only) > 0:
        print(f"Conference games only ({len(conf_only)} games):")
        print(f"   Model MAE:  {conf_only['model_error'].mean():.2f}")
        print(f"   Market MAE: {conf_only['market_error'].mean():.2f}")
        print(f"   Difference: {conf_only['market_error'].mean() - conf_only['model_error'].mean():+.2f}")
        print()
    
    if len(non_conf) > 0:
        print(f"Non-conference games ({len(non_conf)} games):")
        print(f"   Model MAE:  {non_conf['model_error'].mean():.2f}")
        print(f"   Market MAE: {non_conf['market_error'].mean():.2f}")
        print(f"   Difference: {non_conf['market_error'].mean() - non_conf['model_error'].mean():+.2f}")
        print()
    
    # Save details
    output_path = Path.home() / 'Downloads' / 'tmp' / f'{target_conference.replace(" ", "_")}_predictions.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(output_path, index=False)
    print(f"💾 Saved detailed predictions to: {output_path}")
    print()
    print("="*80)
    print()
    
    return details_df


def main():
    parser = argparse.ArgumentParser(description='Hunt for conference-specific market inefficiencies')
    parser.add_argument('--season', type=str, default='2024-25', help='Season to analyze')
    parser.add_argument('--conference', type=str, default='Atlantic 10', help='Conference to analyze')
    parser.add_argument('--use-cache', action='store_true', help='Use cached data')
    parser.add_argument('--use-all-history', action='store_true', help='Use all history for x1')
    parser.add_argument('--list-conferences', action='store_true', help='List all available conferences')
    
    args = parser.parse_args()
    
    if args.list_conferences:
        print("\n📋 Available Conferences:")
        conferences = sorted(set(NCAAB_CONFERENCE_MAPPING_2025_26.values()))
        for conf in conferences:
            teams = [t for t, c in NCAAB_CONFERENCE_MAPPING_2025_26.items() if c == conf]
            print(f"   {conf}: {len(teams)} teams")
        print()
        return
    
    # Load data
    joined_df = load_and_prepare_data(season=args.season, use_cache=args.use_cache)
    
    # Analyze conference
    x1_window = 'all' if args.use_all_history else 'rolling'
    analyze_conference(joined_df, target_conference=args.conference, x1_window_type=x1_window)


if __name__ == '__main__':
    main()

