"""
NFL ML Model with CATEGORICAL Luck Inputs

Based on find_nfl_luck_regression_plays_both_teams.py approach

INPUTS:
  - prev_luck_cat (Lucky/Neutral/Unlucky - one-hot encoded)
  - curr_spread (this week's line)

OUTPUT:
  - predicted_margin OR spread_cover probability

Luck Categories (threshold = 7 by default):
  - Lucky: luck >= +threshold
  - Neutral: -threshold < luck < +threshold  
  - Unlucky: luck <= -threshold

Usage:
    python backtesting/20251203_nfl_categorical_luck_model.py --all-weeks
    python backtesting/20251203_nfl_categorical_luck_model.py --all-weeks --threshold 5
    python backtesting/20251203_nfl_categorical_luck_model.py --test-week 12
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from config import NFL_LUCK_THRESHOLD_DEFAULT

parser = argparse.ArgumentParser()
parser.add_argument('--test-week', type=int, help='Single week to test on')
parser.add_argument('--all-weeks', action='store_true', help='Test on all weeks 2-12')
parser.add_argument('--threshold', type=float, default=NFL_LUCK_THRESHOLD_DEFAULT,
                    help=f'Luck threshold (default: {NFL_LUCK_THRESHOLD_DEFAULT})')
parser.add_argument('--show-trees', action='store_true', help='Show tree structure')
args = parser.parse_args()

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = PROJECT_ROOT / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'nfl_cat_luck_model_{timestamp}.log'

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

threshold = args.threshold

print("=" * 120)
print("NFL ML MODEL: CATEGORICAL LUCK APPROACH")
print("=" * 120)
print("")
print(f"Luck Categories (threshold = ±{threshold}):")
print(f"  Lucky:   luck >= +{threshold}")
print(f"  Neutral: -{threshold} < luck < +{threshold}")
print(f"  Unlucky: luck <= -{threshold}")
print("")
print("INPUTS:")
print("  • prev_luck_cat (Lucky/Neutral/Unlucky - one-hot encoded)")
print("  • curr_spread (this week's line)")
print("")
print("OUTPUT:")
print("  • predicted_margin (game outcome)")
print("")
print(f"Log file: {log_file}")
print("")


def categorize_luck(luck_value, threshold):
    """Categorize luck into Lucky/Neutral/Unlucky."""
    if luck_value is None or pd.isna(luck_value):
        return 'Unknown'
    if luck_value >= threshold:
        return 'Lucky'
    elif luck_value <= -threshold:
        return 'Unlucky'
    else:
        return 'Neutral'


# Load data
intermediate_dir = PROJECT_ROOT / 'data' / '03_intermediate'
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

# Build features with categorical luck
features_list = []
for team in df_team_games['team'].unique():
    team_data = df_team_games[df_team_games['team'] == team].sort_values('week').reset_index(drop=True)
    
    for i in range(len(team_data)):
        if i == 0:
            continue
        
        current_game = team_data.iloc[i]
        prev_game = team_data.iloc[i-1]
        
        # Calculate previous luck
        prev_adj_margin = prev_game['team_adj_score'] - prev_game['opp_adj_score']
        prev_luck = prev_game['actual_margin'] - prev_adj_margin
        
        # Categorize luck
        prev_luck_cat = categorize_luck(prev_luck, threshold)
        
        features = {
            'team': current_game['team'],
            'opponent': current_game['opponent'],
            'week': int(current_game['week']),
            'prev_luck_raw': prev_luck,  # Keep raw for reference
            'prev_luck_cat': prev_luck_cat,
            'curr_spread': current_game['spread'],
            'target_margin': current_game['actual_margin'],
            'covered': current_game['covered'],
        }
        features_list.append(features)

df_ml = pd.DataFrame(features_list)

# One-hot encode luck categories
df_ml['is_lucky'] = (df_ml['prev_luck_cat'] == 'Lucky').astype(int)
df_ml['is_unlucky'] = (df_ml['prev_luck_cat'] == 'Unlucky').astype(int)
df_ml['is_neutral'] = (df_ml['prev_luck_cat'] == 'Neutral').astype(int)

# Feature columns (one-hot + spread)
feature_cols = ['is_lucky', 'is_unlucky', 'curr_spread']
# Note: is_neutral is redundant (1 - is_lucky - is_unlucky)

print(f"✅ Built {len(df_ml)} team-games with categorical features")
print(f"   Weeks: {df_ml['week'].min():.0f} - {df_ml['week'].max():.0f}")
print("")

# Show distribution of luck categories
print("Luck Category Distribution:")
for cat in ['Lucky', 'Neutral', 'Unlucky']:
    count = (df_ml['prev_luck_cat'] == cat).sum()
    pct = count / len(df_ml) * 100
    bar = '█' * int(pct / 2)
    print(f"  {cat:<10s} {count:>4d} ({pct:>5.1f}%) {bar}")
print("")

# Define models
models_reg = {
    'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                      min_samples_split=20, min_samples_leaf=10, random_state=42),
    'RF': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20,
                                 min_samples_leaf=10, random_state=42, n_jobs=-1),
    'Tree': DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
}

cap = 14.0

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
        
        for name, model in models_reg.items():
            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_test), -cap, cap)
            mae = mean_absolute_error(y_test, preds)
            bet_correct = (preds > -df_test['curr_spread']) == df_test['covered']
            accuracy = bet_correct.mean()
            
            week_results[f'{name}_mae'] = mae
            week_results[f'{name}_acc'] = accuracy
            week_results[f'{name}_correct'] = bet_correct.sum()
            
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
    print(f"{'Model':<10s} {'Avg MAE':<12s} {'Accuracy':<12s} {'Correct':<15s} {'ROI':<10s}")
    print("-" * 59)
    
    for name in ['GBM', 'RF', 'Tree']:
        avg_mae = df_results[f'{name}_mae'].mean()
        total_correct = df_results[f'{name}_correct'].sum()
        overall_acc = total_correct / total_test_games
        roi = ((overall_acc * 1.909) - 1) * 100
        
        print(f"{name:<10s} {avg_mae:<12.2f} {overall_acc*100:<12.1f} {total_correct}/{total_test_games} {'':<8s} {roi:+.1f}%")
    
    # Baseline comparison
    print("")
    print("=" * 120)
    print("BASELINE: Just Use Spread (50/50)")
    print("=" * 120)
    baseline_mae = mean_absolute_error(df_ml['target_margin'], -df_ml['curr_spread'])
    print(f"  MAE: {baseline_mae:.2f} points")
    print(f"  Accuracy: ~50%")
    
    # Show tree structure
    if args.show_trees:
        print("")
        print("=" * 120)
        print("TREE STRUCTURE (trained on all data)")
        print("=" * 120)
        print("")
        
        X_all = df_ml[feature_cols]
        y_all = df_ml['target_margin']
        
        tree = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
        tree.fit(X_all, y_all)
        
        print("Feature Importance:")
        for feat, imp in zip(feature_cols, tree.feature_importances_):
            bar = '█' * int(imp * 50)
            print(f"  {feat:<15s} {imp:>6.1%} {bar}")
        
        print("\nTree Rules:")
        print(export_text(tree, feature_names=feature_cols, decimals=1))
        
        # Human-readable interpretation
        print("\n" + "=" * 120)
        print("HUMAN-READABLE INTERPRETATION")
        print("=" * 120)
        print("")
        print("The tree learns:")
        print("  1. curr_spread is dominant - Vegas line is the baseline")
        print("  2. is_lucky and is_unlucky provide adjustments")
        print("")
        print("Expected patterns:")
        print("  • If team was LUCKY last week → predict worse than spread")
        print("  • If team was UNLUCKY last week → predict better than spread")
        print("  • This matches the regression-to-mean theory!")

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
    
    # Show test set with predictions
    print("=" * 120)
    print("PREDICTIONS VS ACTUAL")
    print("=" * 120)
    print("")
    
    tree = models_reg['Tree']
    tree.fit(X_train, y_train)
    df_test['pred'] = np.clip(tree.predict(X_test), -cap, cap)
    df_test['vegas_pred'] = -df_test['curr_spread']
    df_test['pred_cover'] = df_test['pred'] > -df_test['curr_spread']
    df_test['bet_correct'] = df_test['pred_cover'] == df_test['covered']
    
    print(f"{'Team':<6s} {'Luck':<10s} {'Spread':>8s} {'Vegas':>8s} {'Pred':>8s} {'Actual':>8s} {'Cover?':>8s} {'Bet':>6s}")
    print("-" * 72)
    
    for _, row in df_test.iterrows():
        luck_cat = row['prev_luck_cat']
        luck_symbol = "🍀" if luck_cat == "Lucky" else ("😔" if luck_cat == "Unlucky" else "➖")
        bet_result = "✅" if row['bet_correct'] else "❌"
        cover_result = "Y" if row['covered'] else "N"
        
        print(f"{row['team']:<6s} {luck_symbol} {luck_cat:<7s} {row['curr_spread']:>+8.1f} {row['vegas_pred']:>+8.1f} {row['pred']:>+8.1f} {row['target_margin']:>+8.1f} {cover_result:>8s} {bet_result:>6s}")
    
    print("")
    accuracy = df_test['bet_correct'].mean()
    correct = df_test['bet_correct'].sum()
    print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{len(df_test)})")
    
    # Show tree
    if args.show_trees:
        print("")
        print("=" * 120)
        print("TREE STRUCTURE")
        print("=" * 120)
        print("")
        
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

