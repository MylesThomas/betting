"""
NFL ML Model: Final Clean Version

INPUTS (3 features):
  - prev_spread (last week's spread)
  - prev_luck (last week's luck)  
  - curr_spread (this week's spread - contains all market knowledge)

OUTPUT:
  - predicted_margin (predicted game outcome)

PHILOSOPHY:
  - Vegas line contains 95%+ of knowable information
  - Luck is our edge - does it predict future performance?
  - Model should make small adjustments from Vegas based on luck patterns

Usage:
    python backtesting/20251129_nfl_final_model.py --test-week 12
    python backtesting/20251129_nfl_final_model.py --all-weeks
    python backtesting/20251129_nfl_final_model.py --all-weeks --show-trees
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--test-week', type=int, help='Single week to test on')
parser.add_argument('--all-weeks', action='store_true', help='Test on all weeks 2-12')
parser.add_argument('--show-trees', action='store_true', help='Show tree structure')
parser.add_argument('--cap', type=float, default=14.0, help='Prediction cap')
args = parser.parse_args()

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'nfl_final_model_{timestamp}.log'

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(log_file)

print("=" * 120)
print("NFL ML MODEL: FINAL CLEAN VERSION")
print("=" * 120)
print("")
print("INPUTS (3 features):")
print("  • prev_spread     (last week's spread)")
print("  • prev_luck       (last week's luck = actual - expected margin)")
print("  • curr_spread     (this week's spread = all market knowledge)")
print("")
print("OUTPUT:")
print("  • predicted_margin (predicted game outcome)")
print("")
print("PHILOSOPHY:")
print("  Vegas line = 95%+ of information")
print("  Luck = our potential edge")
print("  Model makes small adjustments based on luck patterns")
print("")
print(f"Prediction Cap: ±{args.cap}")
print(f"Log file: {log_file}")
print("")

# Load data
intermediate_dir = Path(__file__).parent.parent / 'data' / '03_intermediate'
data_path = intermediate_dir / "nfl_games_with_spreads_and_results.csv"

df_games = pd.read_csv(data_path)
df_games = df_games.sort_values(['week', 'game_time']).reset_index(drop=True)

print("Loading data...")

# Create team-level data
team_game_rows = []
for idx, game in df_games.iterrows():
    team_game_rows.append({
        'game_id': game['game_id'], 'week': game['week'], 'season': game['season'],
        'team': game['away_abbr'], 'opponent': game['home_abbr'], 'is_home': False,
        'spread': game['consensus_spread'], 'actual_margin': game['actual_margin'],
        'covered': game['away_covered'],
        'team_adj_score': game['away_adj_score'], 'opp_adj_score': game['home_adj_score'],
    })
    team_game_rows.append({
        'game_id': game['game_id'], 'week': game['week'], 'season': game['season'],
        'team': game['home_abbr'], 'opponent': game['away_abbr'], 'is_home': True,
        'spread': -game['consensus_spread'], 'actual_margin': -game['actual_margin'],
        'covered': game['home_covered'],
        'team_adj_score': game['home_adj_score'], 'opp_adj_score': game['away_adj_score'],
    })

df_team_games = pd.DataFrame(team_game_rows)

# Build features
features_list = []
for team in df_team_games['team'].unique():
    team_data = df_team_games[df_team_games['team'] == team].sort_values('week').reset_index(drop=True)
    
    for i in range(len(team_data)):
        if i == 0:
            continue
        
        current_game = team_data.iloc[i]
        prev_game = team_data.iloc[i-1]
        
        prev_adj_margin = prev_game['team_adj_score'] - prev_game['opp_adj_score']
        prev_luck = prev_game['actual_margin'] - prev_adj_margin
        
        features = {
            'team': current_game['team'],
            'opponent': current_game['opponent'],
            'week': int(current_game['week']),
            'prev_spread': prev_game['spread'],
            'prev_luck': prev_luck,
            'curr_spread': current_game['spread'],
            'target_margin': current_game['actual_margin'],
            'covered': current_game['covered'],
        }
        features_list.append(features)

df_ml = pd.DataFrame(features_list)

# FINAL FEATURE SET (only 3 features)
feature_cols = ['prev_spread', 'prev_luck', 'curr_spread']

print(f"✅ Built {len(df_ml)} team-games with {len(feature_cols)} features")
print(f"   Weeks: {df_ml['week'].min():.0f} - {df_ml['week'].max():.0f}")
print("")

# Define models
models = {
    'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                      min_samples_split=20, min_samples_leaf=10, random_state=42),
    'RF': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20,
                                 min_samples_leaf=10, random_state=42, n_jobs=-1),
    'Tree': DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=42)
}

if args.all_weeks:
    # Walk-forward validation on all weeks
    weekly_results = []
    test_weeks = range(2, 13)
    
    print("=" * 120)
    print("WALK-FORWARD VALIDATION: All Weeks")
    print("=" * 120)
    print("")
    
    for test_week in test_weeks:
        print(f"Week {test_week}: ", end='')
        
        train_mask = df_ml['week'] < test_week
        test_mask = df_ml['week'] == test_week
        
        X_train = df_ml.loc[train_mask, feature_cols]
        y_train = df_ml.loc[train_mask, 'target_margin']
        X_test = df_ml.loc[test_mask, feature_cols]
        y_test = df_ml.loc[test_mask, 'target_margin']
        df_test = df_ml[test_mask].copy()
        
        if len(X_test) == 0:
            print(f"No test games")
            continue
        
        week_results = {'week': test_week, 'test_size': len(X_test)}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_test), -args.cap, args.cap)
            mae = mean_absolute_error(y_test, preds)
            bet_correct = (preds > -df_test['curr_spread']) == df_test['covered']
            accuracy = bet_correct.mean()
            
            # Calculate avg distance from Vegas
            avg_diff_from_vegas = np.mean(np.abs(preds - (-df_test['curr_spread'])))
            
            week_results[f'{name}_mae'] = mae
            week_results[f'{name}_acc'] = accuracy
            week_results[f'{name}_correct'] = bet_correct.sum()
            week_results[f'{name}_vegas_diff'] = avg_diff_from_vegas
            
            print(f"{name}={accuracy*100:.0f}% ", end='')
        
        print(f"({len(X_test)} games)")
        weekly_results.append(week_results)
    
    # Summary
    df_results = pd.DataFrame(weekly_results)
    total_test_games = df_results['test_size'].sum()
    
    print("")
    print("=" * 120)
    print("OVERALL PERFORMANCE")
    print("=" * 120)
    print("")
    print(f"{'Model':<10s} {'Avg MAE':<12s} {'Accuracy':<12s} {'Correct':<15s} {'ROI':<10s} {'Avg Diff':<10s}")
    print("-" * 69)
    
    for name in ['GBM', 'RF', 'Tree']:
        avg_mae = df_results[f'{name}_mae'].mean()
        total_correct = df_results[f'{name}_correct'].sum()
        overall_acc = total_correct / total_test_games
        roi = ((overall_acc * 1.909) - 1) * 100
        avg_vegas_diff = df_results[f'{name}_vegas_diff'].mean()
        
        print(f"{name:<10s} {avg_mae:<12.2f} {overall_acc*100:<12.1f} {total_correct}/{total_test_games} {'':8s} {roi:+<10.1f} {avg_vegas_diff:<10.2f}")
    
    print("")
    print("Note: 'Avg Diff' = Average distance from Vegas prediction (lower = closer to Vegas)")
    
    # Compare to baseline (spread-only)
    print("")
    print("=" * 120)
    print("BASELINE COMPARISON")
    print("=" * 120)
    print("")
    
    baseline_preds = -df_ml['curr_spread']
    baseline_mae = mean_absolute_error(df_ml['target_margin'], baseline_preds)
    baseline_acc = ((baseline_preds > -df_ml['curr_spread']) == df_ml['covered']).mean()
    
    print(f"If we just use Vegas spread as prediction:")
    print(f"  MAE:      {baseline_mae:.2f} points")
    print(f"  Accuracy: {baseline_acc*100:.1f}% (by definition ~50%)")
    print("")
    
    best_model_mae = df_results['Tree_mae'].mean()
    best_model_acc = df_results['Tree_correct'].sum() / total_test_games
    
    print(f"Best ML model (Tree):")
    print(f"  MAE:      {best_model_mae:.2f} points")
    print(f"  Accuracy: {best_model_acc*100:.1f}%")
    print(f"  Improvement: {baseline_mae - best_model_mae:+.2f} points MAE, {(best_model_acc - baseline_acc)*100:+.1f}% accuracy")
    
    if args.show_trees:
        print("")
        print("=" * 120)
        print("TREE STRUCTURE (trained on all data)")
        print("=" * 120)
        print("")
        
        X_all = df_ml[feature_cols]
        y_all = df_ml['target_margin']
        
        tree = DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=42)
        tree.fit(X_all, y_all)
        
        print("Feature Importance:")
        for feat, imp in zip(feature_cols, tree.feature_importances_):
            bar = '█' * int(imp * 50)
            print(f"  {feat:<15s} {imp:>6.1%} {bar}")
        
        print("\nTree Rules:")
        print(export_text(tree, feature_names=feature_cols, decimals=1))

else:
    # Single week test
    test_week = args.test_week if args.test_week else 12
    
    print("=" * 120)
    print(f"TESTING WEEK {test_week}")
    print("=" * 120)
    print("")
    
    train_mask = df_ml['week'] < test_week
    test_mask = df_ml['week'] == test_week
    
    X_train = df_ml.loc[train_mask, feature_cols]
    y_train = df_ml.loc[train_mask, 'target_margin']
    X_test = df_ml.loc[test_mask, feature_cols]
    y_test = df_ml.loc[test_mask, 'target_margin']
    df_test = df_ml[test_mask].copy()
    
    print(f"Train: Weeks {df_ml.loc[train_mask, 'week'].min():.0f}-{df_ml.loc[train_mask, 'week'].max():.0f} ({len(X_train)} games)")
    print(f"Test:  Week {test_week} ({len(X_test)} games)")
    print("")
    
    print(f"{'Model':<15s} {'MAE':<10s} {'Accuracy':<12s} {'Correct':<15s} {'Avg Diff':<10s}")
    print("-" * 62)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), -args.cap, args.cap)
        mae = mean_absolute_error(y_test, preds)
        bet_correct = (preds > -df_test['curr_spread']) == df_test['covered']
        accuracy = bet_correct.mean()
        correct_count = bet_correct.sum()
        
        avg_diff_from_vegas = np.mean(np.abs(preds - (-df_test['curr_spread'])))
        
        print(f"{name:<15s} {mae:<10.2f} {accuracy*100:<12.1f} {correct_count}/{len(X_test):<12s} {avg_diff_from_vegas:<10.2f}")
    
    if args.show_trees:
        print("")
        print("=" * 120)
        print("TREE STRUCTURE")
        print("=" * 120)
        print("")
        
        tree = models['Tree']
        
        print("Feature Importance:")
        for feat, imp in zip(feature_cols, tree.feature_importances_):
            bar = '█' * int(imp * 50)
            print(f"  {feat:<15s} {imp:>6.1%} {bar}")
        
        print("\nTree Rules:")
        print(export_text(tree, feature_names=feature_cols, decimals=1))

print("")
print("=" * 120)
print("✅ ANALYSIS COMPLETE")
print("=" * 120)
print(f"\nFull log: {log_file}")

