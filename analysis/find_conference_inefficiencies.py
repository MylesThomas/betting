"""
Find Conference-Specific Market Inefficiencies

Test hypothesis: The market is less efficient at pricing specific conferences.

Strategy:
1. Train separate models for each conference
2. Compare model vs market performance on conference games
3. Identify which conferences have the most exploitable inefficiency

Author: Thomas Myles
Date: 2026-01-16
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
sys.path.append(str(project_root / 'analysis'))

from ncaab_conference_data import get_team_conference, is_conference_game
from ml_pricing_ncaab_games_v2 import (
    build_team_game_rows, 
    impute_missing_implied_scores,
    train_linear_regression,
    predict_linear_regression,
    load_and_prepare_data
)

# Constants
MIN_GAMES_FOR_AVG = 10


def analyze_conference(joined_df, conference_name, x1_window_type='all'):
    """
    Analyze model vs market performance for a specific conference.
    
    Strategy:
    1. Filter to games where BOTH teams are in the conference
    2. Train model on all data (not conference-specific)
    3. Test on conference games only
    4. Compare to market performance
    
    Args:
        joined_df: Full dataset
        conference_name: Conference to analyze (e.g., "Atlantic 10")
        x1_window_type: 'rolling' or 'all'
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {conference_name}")
    print(f"{'='*80}\n")
    
    # Build team-level rows
    team_df = build_team_game_rows(joined_df)
    team_df = impute_missing_implied_scores(team_df)
    
    # Add conference info (use 2025-26 mapping since teams rarely change conferences)
    team_df['team_conference'] = team_df['team'].apply(lambda x: get_team_conference(x, season='2025-26'))
    team_df['opponent_conference'] = team_df['opponent'].apply(lambda x: get_team_conference(x, season='2025-26'))
    
    # Filter to conference games (both teams in same conference)
    conf_games = team_df[
        (team_df['team_conference'] == conference_name) & 
        (team_df['opponent_conference'] == conference_name)
    ].copy()
    
    if len(conf_games) == 0:
        print(f"❌ No games found for {conference_name}")
        return None
    
    print(f"Found {len(conf_games)} {conference_name} conference games")
    print(f"Teams: {sorted(conf_games['team'].unique())}")
    print()
    
    # Prepare features for all conference games
    conf_with_features = []
    
    for team in conf_games['team'].unique():
        team_games = team_df[team_df['team'] == team].sort_values('game_date').copy()
        
        for idx, game in team_games.iterrows():
            # Only include conference games in results
            if game['team_conference'] != conference_name or game['opponent_conference'] != conference_name:
                continue
            
            # Get history up to this game (from ALL games, not just conference)
            history = team_games[team_games['game_date'] < game['game_date']]
            
            if len(history) == 0:
                continue
            
            # Calculate x1
            if x1_window_type == 'rolling':
                x1_val = history['implied_score_filled'].tail(MIN_GAMES_FOR_AVG).mean()
            else:
                x1_val = history['implied_score_filled'].mean()
            
            if pd.isna(x1_val):
                continue
            
            game_copy = game.copy()
            game_copy['x1_avg_last_n'] = x1_val
            game_copy['x2_implied'] = game_copy['implied_score_filled']
            game_copy['x3_conf'] = 1.0  # All conference games
            
            conf_with_features.append(game_copy)
    
    if len(conf_with_features) == 0:
        print("❌ No valid conference games with features")
        return None
    
    conf_features_df = pd.DataFrame(conf_with_features)
    conf_features_df = conf_features_df.dropna(subset=['x1_avg_last_n', 'x2_implied', 'x3_conf', 'actual_score'])
    
    print(f"Conference games with features: {len(conf_features_df)}")
    print()
    
    # Walk-forward validation on conference games
    dates = sorted(conf_features_df['game_date'].unique())
    
    model_predictions = []
    market_predictions = []
    actuals = []
    game_info = []
    
    for current_date in dates:
        # Train on ALL data before this date (including non-conference)
        train_df = team_df[team_df['game_date'] < current_date].copy()
        
        # Test on conference games on this date
        test_df = conf_features_df[conf_features_df['game_date'] == current_date].copy()
        
        if len(test_df) == 0:
            continue
        
        # Prepare training data with features
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
        
        if len(train_with_features) < 50:
            continue
        
        train_features_df = pd.DataFrame(train_with_features)
        train_features_df = train_features_df.dropna(subset=['x1_avg_last_n', 'x2_implied', 'x3_conf', 'actual_score'])
        
        # Train model
        X_train = train_features_df[['x1_avg_last_n', 'x2_implied', 'x3_conf']].values
        y_train = train_features_df['actual_score'].values
        
        coefficients = train_linear_regression(X_train, y_train)
        
        # Predict on conference games
        X_test = test_df[['x1_avg_last_n', 'x2_implied', 'x3_conf']].values
        predictions = predict_linear_regression(X_test, coefficients)
        
        model_predictions.extend(predictions)
        market_predictions.extend(test_df['x2_implied'].values)
        actuals.extend(test_df['actual_score'].values)
        
        for _, game in test_df.iterrows():
            game_info.append({
                'date': game['game_date'],
                'team': game['team'],
                'opponent': game['opponent'],
                'is_home': game['is_home']
            })
    
    if len(model_predictions) == 0:
        print("❌ No predictions made")
        return None
    
    # Calculate metrics
    model_preds = np.array(model_predictions)
    market_preds = np.array(market_predictions)
    actuals_arr = np.array(actuals)
    
    model_errors = np.abs(model_preds - actuals_arr)
    market_errors = np.abs(market_preds - actuals_arr)
    
    model_mae = np.mean(model_errors)
    market_mae = np.mean(market_errors)
    model_rmse = np.sqrt(np.mean(model_errors ** 2))
    market_rmse = np.sqrt(np.mean(market_errors ** 2))
    
    model_wins = np.sum(model_errors < market_errors)
    market_wins = np.sum(model_errors > market_errors)
    ties = np.sum(model_errors == market_errors)
    
    improvement = market_mae - model_mae
    win_rate = model_wins / (model_wins + market_wins) * 100
    
    # Print results
    print(f"RESULTS:")
    print(f"   Total predictions: {len(model_predictions)}")
    print()
    print(f"   Model MAE:  {model_mae:.2f} points")
    print(f"   Market MAE: {market_mae:.2f} points")
    if improvement > 0:
        print(f"   ✅ Model better by {improvement:.2f} points ({100*improvement/market_mae:.1f}% improvement)")
    else:
        print(f"   ❌ Market better by {-improvement:.2f} points")
    print()
    print(f"   Model RMSE:  {model_rmse:.2f} points")
    print(f"   Market RMSE: {market_rmse:.2f} points")
    print()
    print(f"   Model vs Market Win Rate: {win_rate:.1f}% ({model_wins}W-{market_wins}L-{ties}T)")
    print()
    
    return {
        'conference': conference_name,
        'n_games': len(model_predictions),
        'model_mae': model_mae,
        'market_mae': market_mae,
        'improvement': improvement,
        'model_rmse': model_rmse,
        'market_rmse': market_rmse,
        'model_wins': model_wins,
        'market_wins': market_wins,
        'ties': ties,
        'win_rate': win_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description='Find conference-specific market inefficiencies'
    )
    
    parser.add_argument('--season', type=str, default='2024-25',
                       help='Season to analyze')
    parser.add_argument('--conference', type=str, default=None,
                       help='Specific conference to analyze (e.g., "Atlantic 10")')
    parser.add_argument('--all-conferences', action='store_true',
                       help='Test all conferences')
    parser.add_argument('--use-cache', action='store_true',
                       help='Use cached data')
    parser.add_argument('--use-all-history', action='store_true',
                       help='Use all history for x1 instead of rolling window')
    
    args = parser.parse_args()
    
    # Load data
    print("="*80)
    print(f"LOADING DATA FOR {args.season}")
    print("="*80)
    print()
    
    joined_df = load_and_prepare_data(season=args.season, use_cache=args.use_cache)
    
    x1_window = 'all' if args.use_all_history else 'rolling'
    
    if args.conference:
        # Analyze single conference
        analyze_conference(joined_df, args.conference, x1_window_type=x1_window)
    
    elif args.all_conferences:
        # Analyze all conferences
        print("\n" + "="*80)
        print("ANALYZING ALL CONFERENCES")
        print("="*80)
        
        # Get list of conferences
        from src.ncaab_conference_data import NCAAB_CONFERENCE_MAP
        conferences = sorted(set(NCAAB_CONFERENCE_MAP.values()))
        
        results = []
        for conf in conferences:
            result = analyze_conference(joined_df, conf, x1_window_type=x1_window)
            if result:
                results.append(result)
        
        # Summary
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('improvement', ascending=False)
            
            print("\n" + "="*80)
            print("CONFERENCE RANKING BY MODEL IMPROVEMENT")
            print("="*80)
            print()
            
            for i, row in results_df.iterrows():
                status = "✅" if row['improvement'] > 0 else "❌"
                print(f"{status} {row['conference']:<25} | {row['n_games']:>4} games | Model: {row['model_mae']:>5.2f} | Market: {row['market_mae']:>5.2f} | Improvement: {row['improvement']:>+5.2f} | Win Rate: {row['win_rate']:>4.1f}%")
            
            print()
            
            # Save results
            output_path = Path.home() / 'Downloads' / 'tmp' / f'conference_inefficiencies_{x1_window}.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"💾 Saved results to: {output_path}")
            print()
    
    else:
        print("❌ Specify --conference or --all-conferences")
        return


if __name__ == '__main__':
    main()

