# %% [markdown]
# # Strategy Returns Over Time (2023-24 through 2025-26)
# 
# For each strategy with ROI >= 5.0% in 2025-26 training data:
# - Track cumulative returns across all 3 seasons (2023-24, 2024-25, 2025-26)
# - Plot each strategy as a separate line
# - 2025-26 data comes from live daily tracking (actual plays this season)
# - Shows if "winning" strategies in training actually made money historically
# - Separate graphs for 2D and 3D strategies

# %%
import pandas as pd
import boto3
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# %%
# Setup
s3_client = boto3.client('s3')
bucket = 'nba-betting-mt'

# %% [markdown]
# ## Step 1: Load 2025-26 Training Strategies (to identify winning strategies)

# %%
def load_training_strategies(strategy_type):
    """Load 2025-26 training strategies from S3"""
    if strategy_type == '2d':
        s3_key = 'data/03_intermediate/points_by_role_gamespread_strategies_2025-26.json'
    else:  # 3d
        s3_key = 'data/03_intermediate/points_by_role_gamespread_6feet_strategies_2025-26_rim40.json'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Extract strategies list
        if 'strategies' in data:
            strategies = data['strategies']
        else:
            strategies = data
        
        # Convert to DataFrame for easier filtering
        if isinstance(strategies, dict):
            strategies = list(strategies.values())
        
        df = pd.DataFrame(strategies)
        print(f"✅ Loaded {len(df)} {strategy_type.upper()} training strategies from 2025-26")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {strategy_type.upper()} training strategies: {e}")
        return pd.DataFrame()

# Load both strategy types
training_2d = load_training_strategies('2d')
training_3d = load_training_strategies('3d')

# Filter to winning strategies (ROI >= 5.0%)
winning_2d = training_2d[training_2d['roi'] >= 5.0].copy()
winning_3d = training_3d[training_3d['roi'] >= 5.0].copy()

print(f"\n📊 Winning Strategies (ROI >= 5.0%):")
print(f"  2D: {len(winning_2d)} strategies")
print(f"  3D: {len(winning_3d)} strategies")

# %% [markdown]
# ## Step 2: Load Backtest Results (per-play data for all seasons)

# %%
def load_backtest_plays(strategy_type, seasons=['2023-24', '2024-25', '2025-26']):
    """Load detailed play-by-play backtest results from S3"""
    all_plays = []
    
    for season in seasons:
        s3_key = f'data/04_output/backtests/{strategy_type}/{season}/plays.csv'
        
        try:
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
            df['season'] = season
            all_plays.append(df)
            print(f"✅ Loaded {len(df)} {strategy_type.upper()} plays from {season} (backtest)")
        except Exception as e:
            print(f"⚠️  Skipping {season} {strategy_type.upper()}: {e}")
    
    if all_plays:
        return pd.concat(all_plays, ignore_index=True)
    else:
        return pd.DataFrame()

# Load backtest plays (all 3 seasons)
plays_2d = load_backtest_plays('2d')
plays_3d = load_backtest_plays('3d')

print(f"\n📊 Total Plays Loaded (All Seasons):")
print(f"  2D: {len(plays_2d):,} plays")
print(f"  3D: {len(plays_3d):,} plays")

# %% [markdown]
# ## Step 3: Calculate Cumulative Returns for Each Winning Strategy

# %%
def calculate_cumulative_returns(plays_df, winning_strategies_df, strategy_type):
    """
    Calculate cumulative returns over time for each winning strategy.
    
    Assumes:
    - $100 per bet
    - -100 odds (even money: win = +$100, loss = -$100)
    """
    results = []
    
    for idx, strat in winning_strategies_df.iterrows():
        # Create a unique identifier for this strategy
        strat_id = f"{strat['line_tier']}|{strat['spread_bin']}|{strat['bet_side']}"
        if strategy_type == '3d' and 'scorer_type' in strat:
            strat_id += f"|{strat['scorer_type']}"
        
        # Filter plays for this specific strategy
        strategy_plays = plays_df[
            (plays_df['line_tier'] == strat['line_tier']) &
            (plays_df['spread_bin'] == strat['spread_bin']) &
            (plays_df['bet_side'] == strat['bet_side'])
        ].copy()
        
        # For 3D, also match scorer_type
        if strategy_type == '3d' and 'scorer_type' in strat:
            strategy_plays = strategy_plays[
                strategy_plays['scorer_type'] == strat['scorer_type']
            ]
        
        if len(strategy_plays) == 0:
            continue
        
        # Sort by date
        strategy_plays = strategy_plays.sort_values('game_date')
        
        # Profit is already calculated in the plays CSV
        # Calculate cumulative return
        strategy_plays['cumulative_return'] = strategy_plays['profit'].cumsum()
        
        # Add strategy info
        strategy_plays['strategy_id'] = strat_id
        strategy_plays['training_roi'] = strat['roi']
        
        results.append(strategy_plays)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

# Calculate cumulative returns
returns_2d = calculate_cumulative_returns(plays_2d, winning_2d, '2d')
returns_3d = calculate_cumulative_returns(plays_3d, winning_3d, '3d')

print(f"\n📈 Cumulative Returns Calculated:")
print(f"  2D: {len(returns_2d):,} plays across {returns_2d['strategy_id'].nunique() if len(returns_2d) > 0 else 0} strategies")
print(f"  3D: {len(returns_3d):,} plays across {returns_3d['strategy_id'].nunique() if len(returns_3d) > 0 else 0} strategies")

# %% [markdown]
# ## Step 4: Plot Cumulative Returns (2D Strategies)

# %%
if len(returns_2d) > 0:
    plt.figure(figsize=(14, 8))
    
    for strategy_id in returns_2d['strategy_id'].unique():
        strategy_data = returns_2d[returns_2d['strategy_id'] == strategy_id].copy()
        
        # Convert date to numeric for plotting
        strategy_data['game_date'] = pd.to_datetime(strategy_data['game_date'])
        
        # Plot this strategy's cumulative return
        training_roi = strategy_data['training_roi'].iloc[0]
        label = f"{strategy_id} (Train: {training_roi:+.1f}%)"
        
        plt.plot(strategy_data['game_date'], 
                strategy_data['cumulative_return'],
                marker='o',
                markersize=2,
                alpha=0.7,
                label=label)
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return ($)', fontsize=12)
    plt.title('2D Strategy Cumulative Returns Over Time (2023-24 through 2025-26)\nStrategies with ROI >= 5.0% in 2025-26 training', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/thomasmyles/dev/betting/tmp/2d_strategy_returns_over_time.png', dpi=300, bbox_inches='tight')
    print("✅ Saved 2D strategy returns plot: tmp/2d_strategy_returns_over_time.png")
    plt.show()
else:
    print("⚠️  No 2D strategy data to plot")

# %% [markdown]
# ## Step 5: Plot Cumulative Returns (3D Strategies)

# %%
if len(returns_3d) > 0:
    plt.figure(figsize=(14, 8))
    
    for strategy_id in returns_3d['strategy_id'].unique():
        strategy_data = returns_3d[returns_3d['strategy_id'] == strategy_id].copy()
        
        # Convert date to numeric for plotting
        strategy_data['game_date'] = pd.to_datetime(strategy_data['game_date'])
        
        # Plot this strategy's cumulative return
        training_roi = strategy_data['training_roi'].iloc[0]
        label = f"{strategy_id.replace('|', ' | ')} (Train: {training_roi:+.1f}%)"
        
        plt.plot(strategy_data['game_date'], 
                strategy_data['cumulative_return'],
                marker='o',
                markersize=2,
                alpha=0.7,
                label=label)
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return ($)', fontsize=12)
    plt.title('3D Strategy Cumulative Returns Over Time (2023-24 through 2025-26)\nStrategies with ROI >= 5.0% in 2025-26 training', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/thomasmyles/dev/betting/tmp/3d_strategy_returns_over_time.png', dpi=300, bbox_inches='tight')
    print("✅ Saved 3D strategy returns plot: tmp/3d_strategy_returns_over_time.png")
    plt.show()
else:
    print("⚠️  No 3D strategy data to plot")

# %% [markdown]
# ## Step 6: Summary Statistics

# %%
def print_strategy_summary(returns_df, strategy_type):
    """Print summary statistics for each strategy"""
    print(f"\n{'='*80}")
    print(f"📊 {strategy_type.upper()} STRATEGY SUMMARY")
    print(f"{'='*80}")
    
    if len(returns_df) == 0:
        print("No data available")
        return
    
    for strategy_id in returns_df['strategy_id'].unique():
        strategy_data = returns_df[returns_df['strategy_id'] == strategy_id]
        
        final_return = strategy_data['cumulative_return'].iloc[-1]
        total_plays = len(strategy_data)
        wins = (strategy_data['result'] == 'WIN').sum()
        losses = (strategy_data['result'] == 'LOSS').sum()
        pushes = (strategy_data['result'] == 'PUSH').sum()
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        training_roi = strategy_data['training_roi'].iloc[0]
        
        print(f"\n{strategy_id}")
        print(f"  Training ROI: {training_roi:+.1f}%")
        print(f"  Total Plays: {total_plays}")
        print(f"  W-L-P: {wins}-{losses}-{pushes}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Final Return: ${final_return:,.2f}")

print_strategy_summary(returns_2d, '2d')
print_strategy_summary(returns_3d, '3d')

# %%

