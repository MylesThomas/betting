# %% [markdown]
# # Analysis: Profitable Backtest Strategies
# 
# Find strategies where backtest performance >= training performance

# %%
import pandas as pd
import boto3
from io import StringIO

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# %%
# Setup
s3_client = boto3.client('s3')
bucket = 'nba-betting-mt'

def find_latest_backtest(s3_client, bucket, strategy_type):
    """
    Find the most recent backtest results for a strategy type
    by listing all files and picking the latest timestamp
    """
    prefix = 'data/04_output/backtests/'
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return None
        
        # Find all matching files
        matching_files = [
            obj['Key'] for obj in response['Contents']
            if f'{strategy_type}_per_strategy_performance.csv' in obj['Key']
        ]
        
        if not matching_files:
            return None
        
        # Sort by key (timestamp in filename) and take the latest
        matching_files.sort(reverse=True)
        return matching_files[0]
        
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return None

# Try to find latest backtest results dynamically
print("🔍 Finding latest backtest results...\n")

backtests = []
for season in ['2024-25', '2023-24']:
    for strategy in ['2d', '3d']:
        path = find_latest_backtest(s3_client, bucket, strategy)
        
        if path:
            # Filter to correct season based on path content
            # For now, we'll use the manually specified paths as fallback
            backtests.append({
                'season': season,
                'strategy': strategy,
                'path': path
            })

# Use the new structured paths (organized by strategy type and season)
backtests = [
    {
        'season': '2024-25', 
        'strategy': '2d', 
        'path': 'data/04_output/backtests/2d/2024-25/per_strategy.csv'
    },
    {
        'season': '2024-25', 
        'strategy': '3d', 
        'path': 'data/04_output/backtests/3d/2024-25/per_strategy.csv'
    },
    {
        'season': '2023-24', 
        'strategy': '2d', 
        'path': 'data/04_output/backtests/2d/2023-24/per_strategy.csv'
    },
    {
        'season': '2023-24', 
        'strategy': '3d', 
        'path': 'data/04_output/backtests/3d/2023-24/per_strategy.csv'
    },
]

# %% [markdown]
# ## Load All Backtest Results

# %%
all_results = []

for backtest in backtests:
    try:
        response = s3_client.get_object(Bucket=bucket, Key=backtest['path'])
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        df['season'] = backtest['season']
        df['strategy_type'] = backtest['strategy']
        all_results.append(df)
        print(f"✅ Loaded {backtest['season']} {backtest['strategy'].upper()}: {len(df)} strategies")
    except Exception as e:
        print(f"⚠️  Skipping {backtest['season']} {backtest['strategy'].upper()}: file not found")
        # print(f"   Path: {backtest['path']}")  # Uncomment to debug

if not all_results:
    raise ValueError("❌ No backtest results found! Make sure backtests have been run.")

df_all = pd.concat(all_results, ignore_index=True)
print(f"\n📊 Total strategies analyzed: {len(df_all)}")
print(f"📊 From {len(all_results)} backtest file(s)")

# %% [markdown]
# ## Helper Function: Analyze Winners

# %%
def analyze_winners(df, title):
    """Analyze profitable strategies that generalized well"""
    winners = df[(df['backtest_roi'] > 0) & (df['roi_delta'] >= 0)].copy()
    
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print(f"{'='*80}")
    
    if len(winners) == 0:
        print(f"❌ No profitable strategies found that generalized well")
        return winners
    
    print(f"✅ Profitable strategies: {len(winners)} / {len(df)} ({len(winners)/len(df)*100:.1f}%)")
    print(f"💰 Total profit: ${winners['backtest_profit'].sum():,.2f}")
    print(f"📈 Average ROI: {winners['backtest_roi'].mean():+.2f}%")
    print(f"🎯 Average win rate: {winners['backtest_win_rate'].mean():.1f}%")
    print(f"🎲 Total plays: {winners['backtest_plays'].sum():.0f}")
    
    # By strategy type
    if len(winners) > 0:
        strategy_summary = winners.groupby('strategy_type').agg({
            'strategy_idx': 'count',
            'backtest_profit': 'sum',
            'backtest_roi': 'mean',
            'backtest_win_rate': 'mean',
            'backtest_plays': 'sum'
        }).rename(columns={'strategy_idx': 'num_strategies'})
        
        print(f"\nBy Strategy Type:")
        for strat_type in strategy_summary.index:
            row = strategy_summary.loc[strat_type]
            print(f"  {strat_type.upper()}: {row['num_strategies']:.0f} strategies | "
                  f"${row['backtest_profit']:,.2f} profit | "
                  f"{row['backtest_roi']:+.1f}% ROI | "
                  f"{row['backtest_win_rate']:.1f}% win rate")
    
    # Top 3 strategies
    if len(winners) > 0:
        print(f"\n🏆 Top 3 Strategies:")
        top_3 = winners.nlargest(3, 'backtest_roi')
        for idx, row in top_3.iterrows():
            print(f"\n  #{idx}: {row['line_tier']} | {row['spread_bin']} | {row['bet_side']}")
            if row['scorer_type'] != 'N/A':
                print(f"      Scorer: {row['scorer_type']}")
            print(f"      Training: {row['training_roi']:+.1f}% ROI | Backtest: {row['backtest_roi']:+.1f}% ROI (+{row['roi_delta']:.1f}%)")
            print(f"      {row['backtest_plays']:.0f} plays | ${row['backtest_profit']:,.2f} profit")
    
    return winners

# %% [markdown]
# ## Section 1: 2023-24 Season Analysis

# %%
df_2023 = df_all[df_all['season'] == '2023-24'].copy()
winners_2023 = analyze_winners(df_2023, "2023-24 SEASON ANALYSIS")

# %% [markdown]
# ## Section 2: 2024-25 Season Analysis

# %%
df_2024 = df_all[df_all['season'] == '2024-25'].copy()
winners_2024 = analyze_winners(df_2024, "2024-25 SEASON ANALYSIS")

# %% [markdown]
# ## Section 3: Combined Analysis (All Seasons)

# %%
winners_all = analyze_winners(df_all, "COMBINED ANALYSIS (ALL SEASONS)")

# %% [markdown]
# ## Final Summary: Compare All 3 Analyses

# %%
print(f"\n{'='*80}")
print(f"📈 FINAL SUMMARY: BACKTEST RESULTS")
print(f"{'='*80}")

print(f"\n2023-24 Season:")
print(f"  Total Strategies: {len(df_2023)}")
print(f"  Winners: {len(winners_2023)} ({len(winners_2023)/len(df_2023)*100:.1f}%)")
if len(winners_2023) > 0:
    print(f"  Profit: ${winners_2023['backtest_profit'].sum():,.2f}")
    print(f"  Avg ROI: {winners_2023['backtest_roi'].mean():+.1f}%")
    print(f"  Best Strategy: {winners_2023.nlargest(1, 'backtest_roi')['backtest_roi'].values[0]:+.1f}% ROI")

print(f"\n2024-25 Season:")
print(f"  Total Strategies: {len(df_2024)}")
print(f"  Winners: {len(winners_2024)} ({len(winners_2024)/len(df_2024)*100:.1f}%)")
if len(winners_2024) > 0:
    print(f"  Profit: ${winners_2024['backtest_profit'].sum():,.2f}")
    print(f"  Avg ROI: {winners_2024['backtest_roi'].mean():+.1f}%")
    print(f"  Best Strategy: {winners_2024.nlargest(1, 'backtest_roi')['backtest_roi'].values[0]:+.1f}% ROI")

print(f"\nCombined (All Seasons):")
print(f"  Total Strategies: {len(df_all)}")
print(f"  Winners: {len(winners_all)} ({len(winners_all)/len(df_all)*100:.1f}%)")
if len(winners_all) > 0:
    print(f"  Total Profit: ${winners_all['backtest_profit'].sum():,.2f}")
    print(f"  Avg ROI: {winners_all['backtest_roi'].mean():+.1f}%")
    print(f"  Best Strategy: {winners_all.nlargest(1, 'backtest_roi')['backtest_roi'].values[0]:+.1f}% ROI")

print(f"\n{'='*80}")
print(f"🎯 KEY INSIGHTS")
print(f"{'='*80}")

# Compare seasons
if len(winners_2023) > len(winners_2024):
    print(f"✅ 2023-24 had MORE successful strategies ({len(winners_2023)} vs {len(winners_2024)})")
elif len(winners_2024) > len(winners_2023):
    print(f"✅ 2024-25 had MORE successful strategies ({len(winners_2024)} vs {len(winners_2023)})")
else:
    print(f"⚖️  Both seasons had equal number of successful strategies ({len(winners_2023)})")

# Check if any strategies worked across both seasons
if len(winners_all) > 0:
    # Group by strategy characteristics to see if same setups worked in both seasons
    for season in ['2023-24', '2024-25']:
        season_winners = winners_all[winners_all['season'] == season]
        if len(season_winners) > 0:
            print(f"\n{season}:")
            for idx, row in season_winners.nlargest(3, 'backtest_roi').iterrows():
                print(f"  • {row['line_tier']} | {row['spread_bin']} | {row['bet_side']} ({row['strategy_type'].upper()})")
                print(f"    → {row['backtest_roi']:+.1f}% ROI | ${row['backtest_profit']:,.2f} profit")

# %% [markdown]
# ## Export Results

# %%
# Save full results
if len(winners_all) > 0:
    winners_sorted = winners_all.sort_values('backtest_roi', ascending=False)
    output_file = 'profitable_strategies_summary.csv'
    winners_sorted.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(winners_sorted)} profitable strategies to: {output_file}")
    
    # Show a preview
    display_cols = [
        'season', 'strategy_type', 'line_tier', 'spread_bin', 'bet_side', 'scorer_type',
        'training_roi', 'backtest_roi', 'roi_delta', 
        'backtest_win_rate', 'backtest_plays', 'backtest_profit'
    ]
    print(f"\nPreview:")
    print(winners_sorted[display_cols].head(20))
else:
    print(f"\n❌ No profitable strategies to export")

# %%

