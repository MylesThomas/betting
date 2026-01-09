# %% [markdown]
# # Find Resilient Strategies (Profitable in 2+ Seasons)
# 
# Identifies strategies that show consistent profitability across multiple seasons:
# - Profitable in at least 2 out of 3 seasons
# - Positive overall cumulative returns
# - Prioritizes strategies profitable in 2023-24 AND 2025-26 (even if 2024-25 was bad)
# - These may represent real edges that are resilient to market shifts

# %%
import pandas as pd
import boto3
from io import StringIO
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# %%
# Setup
s3_client = boto3.client('s3')
bucket = 'nba-betting-mt'

# %% [markdown]
# ## Step 1: Load Training Strategies and Backtest Plays

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
        
        if 'strategies' in data:
            strategies = data['strategies']
        else:
            strategies = data
        
        if isinstance(strategies, dict):
            strategies = list(strategies.values())
        
        df = pd.DataFrame(strategies)
        print(f"✅ Loaded {len(df)} {strategy_type.upper()} training strategies from 2025-26")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {strategy_type.upper()} training strategies: {e}")
        return pd.DataFrame()

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
            print(f"✅ Loaded {len(df)} {strategy_type.upper()} plays from {season}")
        except Exception as e:
            print(f"⚠️  Skipping {season} {strategy_type.upper()}: {e}")
    
    if all_plays:
        return pd.concat(all_plays, ignore_index=True)
    else:
        return pd.DataFrame()

# Load training strategies (for ROI >= 5% filter)
training_2d = load_training_strategies('2d')
training_3d = load_training_strategies('3d')

winning_2d = training_2d[training_2d['roi'] >= 5.0].copy()
winning_3d = training_3d[training_3d['roi'] >= 5.0].copy()

print(f"\n📊 Winning Strategies (ROI >= 5.0%):")
print(f"  2D: {len(winning_2d)} strategies")
print(f"  3D: {len(winning_3d)} strategies")

# Load backtest plays
plays_2d = load_backtest_plays('2d')
plays_3d = load_backtest_plays('3d')

print(f"\n📊 Total Plays Loaded:")
print(f"  2D: {len(plays_2d):,} plays")
print(f"  3D: {len(plays_3d):,} plays")

# %% [markdown]
# ## Step 2: Calculate Per-Season Performance for Each Strategy

# %%
def analyze_strategy_resilience(plays_df, winning_strategies_df, strategy_type):
    """
    Analyze each strategy's performance by season to find resilient ones.
    """
    results = []
    
    for idx, strat in winning_strategies_df.iterrows():
        # Create strategy identifier
        strat_id = f"{strat['line_tier']}|{strat['spread_bin']}|{strat['bet_side']}"
        if strategy_type == '3d' and 'scorer_type' in strat:
            strat_id += f"|{strat['scorer_type']}"
        
        # Filter plays for this specific strategy
        strategy_plays = plays_df[
            (plays_df['line_tier'] == strat['line_tier']) &
            (plays_df['spread_bin'] == strat['spread_bin']) &
            (plays_df['bet_side'] == strat['bet_side'])
        ].copy()
        
        if strategy_type == '3d' and 'scorer_type' in strat:
            strategy_plays = strategy_plays[
                strategy_plays['scorer_type'] == strat['scorer_type']
            ]
        
        if len(strategy_plays) == 0:
            continue
        
        # Calculate performance BY SEASON
        season_stats = {}
        for season in ['2023-24', '2024-25', '2025-26']:
            season_plays = strategy_plays[strategy_plays['season'] == season]
            
            if len(season_plays) > 0:
                wins = (season_plays['result'] == 'WIN').sum()
                losses = (season_plays['result'] == 'LOSS').sum()
                total_profit = season_plays['profit'].sum()
                
                season_stats[season] = {
                    'plays': len(season_plays),
                    'wins': wins,
                    'losses': losses,
                    'profit': total_profit,
                    'profitable': total_profit > 0
                }
            else:
                season_stats[season] = {
                    'plays': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0,
                    'profitable': False
                }
        
        # Calculate overall stats
        total_plays = len(strategy_plays)
        total_wins = (strategy_plays['result'] == 'WIN').sum()
        total_losses = (strategy_plays['result'] == 'LOSS').sum()
        total_profit = strategy_plays['profit'].sum()
        
        # Count profitable seasons
        profitable_seasons = sum(1 for s in season_stats.values() if s['profitable'])
        
        # Check if profitable in 2023-24 AND 2025-26
        profitable_bookends = (season_stats['2023-24']['profitable'] and 
                               season_stats['2025-26']['profitable'])
        
        results.append({
            'strategy_id': strat_id,
            'strategy_type': strategy_type,
            'line_tier': strat['line_tier'],
            'spread_bin': strat['spread_bin'],
            'bet_side': strat['bet_side'],
            'scorer_type': strat.get('scorer_type', 'N/A'),
            'training_roi': strat['roi'],
            
            # Overall stats
            'total_plays': total_plays,
            'total_profit': total_profit,
            'win_rate': (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0,
            
            # Season-by-season
            'profit_2023_24': season_stats['2023-24']['profit'],
            'profit_2024_25': season_stats['2024-25']['profit'],
            'profit_2025_26': season_stats['2025-26']['profit'],
            'plays_2023_24': season_stats['2023-24']['plays'],
            'plays_2024_25': season_stats['2024-25']['plays'],
            'plays_2025_26': season_stats['2025-26']['plays'],
            
            # Resilience metrics
            'profitable_seasons_count': profitable_seasons,
            'profitable_in_bookends': profitable_bookends,  # 2023-24 AND 2025-26
            'overall_profitable': total_profit > 0
        })
    
    return pd.DataFrame(results)

print("\n📊 Analyzing strategy resilience across seasons...")
resilience_2d = analyze_strategy_resilience(plays_2d, winning_2d, '2d')
resilience_3d = analyze_strategy_resilience(plays_3d, winning_3d, '3d')

# %% [markdown]
# ## Step 3: Find Resilient Strategies

# %%
def find_resilient_strategies(resilience_df):
    """
    Find strategies that are:
    1. Profitable in 2+ seasons
    2. Have positive overall returns
    """
    resilient = resilience_df[
        (resilience_df['profitable_seasons_count'] >= 2) &
        (resilience_df['overall_profitable'] == True)
    ].copy()
    
    # Sort by: bookends first, then by total profit
    resilient['sort_key'] = resilient['profitable_in_bookends'].astype(int) * 1000000 + resilient['total_profit']
    resilient = resilient.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    
    return resilient

resilient_2d = find_resilient_strategies(resilience_2d)
resilient_3d = find_resilient_strategies(resilience_3d)

print(f"\n{'='*80}")
print(f"🏆 RESILIENT STRATEGIES FOUND")
print(f"{'='*80}")
print(f"\n2D Strategies: {len(resilient_2d)} resilient strategies found")
print(f"3D Strategies: {len(resilient_3d)} resilient strategies found")

# %% [markdown]
# ## Step 4: Display Resilient Strategies

# %%
def display_resilient_strategies(resilient_df, strategy_type):
    """Display resilient strategies with detailed season breakdown"""
    
    if len(resilient_df) == 0:
        print(f"\n⚠️  No resilient {strategy_type.upper()} strategies found")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 {strategy_type.upper()} RESILIENT STRATEGIES")
    print(f"{'='*80}")
    
    for idx, row in resilient_df.iterrows():
        bookend_flag = "⭐ BOOKENDS" if row['profitable_in_bookends'] else ""
        
        print(f"\n{bookend_flag}")
        print(f"Strategy: {row['line_tier']} | {row['spread_bin']} | {row['bet_side']}")
        if row['scorer_type'] != 'N/A':
            print(f"Scorer Type: {row['scorer_type']}")
        print(f"Training ROI: {row['training_roi']:+.1f}%")
        
        print(f"\n  Season-by-Season Performance:")
        print(f"    2023-24: ${row['profit_2023_24']:>8,.2f} ({row['plays_2023_24']:>3} plays) {'✅' if row['profit_2023_24'] > 0 else '❌'}")
        print(f"    2024-25: ${row['profit_2024_25']:>8,.2f} ({row['plays_2024_25']:>3} plays) {'✅' if row['profit_2024_25'] > 0 else '❌'}")
        print(f"    2025-26: ${row['profit_2025_26']:>8,.2f} ({row['plays_2025_26']:>3} plays) {'✅' if row['profit_2025_26'] > 0 else '❌'}")
        
        print(f"\n  Overall: ${row['total_profit']:,.2f} profit | {row['total_plays']} plays | {row['win_rate']:.1f}% win rate")
        print(f"  Profitable in {row['profitable_seasons_count']}/3 seasons")

display_resilient_strategies(resilient_2d, '2d')
display_resilient_strategies(resilient_3d, '3d')

# %% [markdown]
# ## Step 5: Summary Statistics

# %%
print(f"\n{'='*80}")
print(f"📊 SUMMARY")
print(f"{'='*80}")

# Count bookend strategies (profitable in 2023-24 AND 2025-26)
bookend_2d = len(resilient_2d[resilient_2d['profitable_in_bookends']])
bookend_3d = len(resilient_3d[resilient_3d['profitable_in_bookends']])

print(f"\n🎯 Strategies Profitable in 2023-24 AND 2025-26 (Bookends):")
print(f"  2D: {bookend_2d} strategies")
print(f"  3D: {bookend_3d} strategies")

if bookend_2d > 0 or bookend_3d > 0:
    print(f"\n💡 These 'bookend' strategies are especially interesting because:")
    print(f"   - They worked in 2 different seasons with different market conditions")
    print(f"   - 2024-25 might have been an anomaly (injuries, rule changes, etc.)")
    print(f"   - They represent potentially real edges that are resilient over time")

# Calculate total profit from resilient strategies
total_profit_2d = resilient_2d['total_profit'].sum() if len(resilient_2d) > 0 else 0
total_profit_3d = resilient_3d['total_profit'].sum() if len(resilient_3d) > 0 else 0

print(f"\n💰 Total Profit from Resilient Strategies:")
print(f"  2D: ${total_profit_2d:,.2f}")
print(f"  3D: ${total_profit_3d:,.2f}")
print(f"  Combined: ${total_profit_2d + total_profit_3d:,.2f}")

# %% [markdown]
# ## Step 6: Export Results

# %%
# Combine and export
if len(resilient_2d) > 0 or len(resilient_3d) > 0:
    all_resilient = pd.concat([resilient_2d, resilient_3d], ignore_index=True)
    
    output_file = 'resilient_strategies_summary.csv'
    all_resilient.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(all_resilient)} resilient strategies to: {output_file}")
    
    # Show preview
    display_cols = [
        'strategy_type', 'line_tier', 'spread_bin', 'bet_side', 'scorer_type',
        'profitable_in_bookends', 'profitable_seasons_count',
        'profit_2023_24', 'profit_2024_25', 'profit_2025_26',
        'total_profit', 'total_plays', 'win_rate'
    ]
    print(f"\nPreview:")
    print(all_resilient[display_cols].head(20))
else:
    print(f"\n⚠️  No resilient strategies found to export")

# %%

