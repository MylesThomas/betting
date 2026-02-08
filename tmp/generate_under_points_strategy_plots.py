"""
Generate Strategy Performance Plots - Standalone Test Script

This script generates 4-panel performance plots for all 15 strategies in enhanced_unders_v5.json.
It's a standalone version that can be run locally to test plot generation before integrating into Lambda.

Usage:
    python src/pbp_data/tmp/generate_strategy_plots.py

Output:
    - Generates 15 PNG files in /tmp/strategy_plots/
    - Each PNG has 4 panels showing win rate over time
    - Prints summary of plots generated

After validation:
    - Upload plots to S3
    - Include S3 URLs in email (like lambda_function_track_game_line_movements.py does)
"""

import pandas as pd
import boto3
from io import StringIO, BytesIO
from pathlib import Path
import json

# Matplotlib setup (non-interactive backend)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
BACKTEST_PREFIX = 'data/04_output/backtests'
SEASONS = ['2023-24', '2024-25', '2025-26']
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp' / 'strategy_plots'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# LOAD CONFIG FROM S3
# =============================================================================

def load_strategy_config():
    """Load enhanced_unders_v5.json from S3."""
    s3_client = boto3.client('s3')
    
    try:
        s3_key = 'strategies/enhanced_unders_v5.json'
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        print(f"✅ Loaded config: {config['name']}")
        print(f"   Version: {config['version']}")
        print(f"   Strategies: {len(config['strategies'])}\n")
        
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None


# =============================================================================
# LOAD BACKTEST PLAYS
# =============================================================================

def load_plays_for_strategy(strategy, strategy_type):
    """
    Load historical plays for a specific strategy across all seasons.
    
    Args:
        strategy: Strategy dict with line_tier, spread_bin, bet_side, scorer_type
        strategy_type: '2d' or '3d'
    
    Returns:
        DataFrame with plays across all seasons
    """
    s3_client = boto3.client('s3')
    dfs = []
    
    for season in SEASONS:
        s3_key = f'{BACKTEST_PREFIX}/{strategy_type}/{season}/plays.csv'
        
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
            
            # Filter to this specific strategy
            mask = (
                (df['line_tier'] == strategy['line_tier']) &
                (df['spread_bin'] == strategy['spread_bin']) &
                (df['bet_side'] == strategy['bet_side'])
            )
            
            # For 3D strategies, also filter by scorer_type
            if strategy_type == '3d' and 'scorer_type' in strategy:
                mask = mask & (df['scorer_type'] == strategy['scorer_type'])
            
            df_strat = df[mask].copy()
            df_strat['season'] = season
            
            if len(df_strat) > 0:
                dfs.append(df_strat)
                
        except Exception as e:
            print(f"   ⚠️  Could not load {season} plays: {e}")
            continue
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# GENERATE 4-PANEL PLOT
# =============================================================================

def generate_performance_plot(strategy, strategy_type, output_path):
    """
    Generate 4-panel plot showing strategy performance over time.
    
    Panels:
    1. 2023-24 season: Date vs Win Rate
    2. 2024-25 season: Date vs Win Rate
    3. 2025-26 season: Date vs Win Rate
    4. Overall: All seasons combined
    
    Args:
        strategy: Strategy dict
        strategy_type: '2d' or '3d'
        output_path: Where to save the PNG
    
    Returns:
        bool: True if successful
    """
    # Load plays data
    df = load_plays_for_strategy(strategy, strategy_type)
    
    if df is None or len(df) == 0:
        print(f"   ⚠️  No data for strategy")
        return False
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    # Calculate win indicator
    df['is_win'] = (df['result'] == 'WIN').astype(int)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Build title
    title = f"Strategy Performance: {strategy['line_tier']} | {strategy['spread_bin']} | {strategy['bet_side']}"
    if strategy_type == '3d' and 'scorer_type' in strategy:
        title += f" | {strategy['scorer_type']}"
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Season colors
    season_colors = {
        '2023-24': '#1f77b4',  # Blue
        '2024-25': '#ff7f0e',  # Orange
        '2025-26': '#2ca02c'   # Green
    }
    
    # Plot each season individually (panels 1-3)
    for idx, season in enumerate(SEASONS[:3]):
        ax = axes[idx // 2, idx % 2]
        
        df_season = df[df['season'] == season].copy()
        
        if len(df_season) == 0:
            ax.text(0.5, 0.5, f'No data for {season}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 100)
            ax.set_title(f'{season}', fontsize=14, fontweight='bold')
            continue
        
        # Calculate cumulative win rate for this season
        df_season = df_season.sort_values('game_date')
        df_season['cumulative_wins'] = df_season['is_win'].cumsum()
        df_season['cumulative_plays'] = range(1, len(df_season) + 1)
        df_season['win_rate'] = (df_season['cumulative_wins'] / df_season['cumulative_plays'] * 100)
        
        # Plot
        ax.plot(df_season['game_date'], df_season['win_rate'], 
               color=season_colors.get(season, 'blue'), linewidth=2, label=season)
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Baseline')
        
        # Format
        ax.set_title(f'{season}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Win Rate (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add final stats
        final_wr = df_season['win_rate'].iloc[-1]
        total_plays = len(df_season)
        total_wins = int(df_season['cumulative_wins'].iloc[-1])
        total_losses = total_plays - total_wins
        ax.text(0.02, 0.98, f'{total_wins}W-{total_losses}L | {final_wr:.1f}%', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 4: Overall (all seasons combined)
    ax = axes[1, 1]
    
    # Reset cumulative across all seasons
    df_overall = df.copy()
    df_overall = df_overall.sort_values('game_date')
    df_overall['cumulative_wins'] = df_overall['is_win'].cumsum()
    df_overall['cumulative_plays'] = range(1, len(df_overall) + 1)
    df_overall['win_rate'] = (df_overall['cumulative_wins'] / df_overall['cumulative_plays'] * 100)
    
    # Plot with season-colored segments
    for season in SEASONS:
        df_season_segment = df_overall[df_overall['season'] == season]
        if len(df_season_segment) > 0:
            ax.plot(df_season_segment['game_date'], df_season_segment['win_rate'],
                   color=season_colors.get(season, 'black'), linewidth=2, label=season)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Baseline')
    
    # Format
    ax.set_title('Overall (All Seasons)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add final stats
    final_wr = df_overall['win_rate'].iloc[-1]
    total_plays = len(df_overall)
    total_wins = int(df_overall['cumulative_wins'].iloc[-1])
    total_losses = total_plays - total_wins
    ax.text(0.02, 0.98, f'{total_wins}W-{total_losses}L | {final_wr:.1f}%', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all strategy plots."""
    print("="*80)
    print("📊 GENERATING STRATEGY PERFORMANCE PLOTS")
    print("="*80)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Seasons: {', '.join(SEASONS)}\n")
    
    # Load config
    config = load_strategy_config()
    if not config:
        print("❌ Failed to load config")
        return
    
    strategies = config['strategies']
    plots_generated = 0
    plots_failed = 0
    
    # Generate plots for each strategy
    for i, strat in enumerate(strategies, 1):
        strategy_name = strat['strategy_name']
        strategy_type = strat['strategy_type']
        
        print(f"\n[{i}/{len(strategies)}] {strategy_name}")
        print(f"   Config: {strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}", end='')
        
        if strategy_type == '3d' and 'scorer_type' in strat:
            print(f" | {strat['scorer_type']}")
        else:
            print()
        
        # Generate filename
        plot_filename = f"{strategy_name}.png"
        output_path = OUTPUT_DIR / plot_filename
        
        # Generate plot
        try:
            success = generate_performance_plot(strat, strategy_type, str(output_path))
            
            if success:
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"   ✅ Generated: {plot_filename} ({file_size:.1f} KB)")
                plots_generated += 1
            else:
                print(f"   ❌ Failed: {plot_filename}")
                plots_failed += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            plots_failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Generated: {plots_generated}/{len(strategies)} plots")
    print(f"❌ Failed: {plots_failed}/{len(strategies)} plots")
    print(f"📂 Location: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    # List all generated files
    if plots_generated > 0:
        print("Generated files:")
        for f in sorted(OUTPUT_DIR.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name} ({size_kb:.1f} KB)")
        
        # Open all plots
        print(f"\n{'='*80}")
        print("Opening all plots...")
        print(f"{'='*80}\n")
        import subprocess
        subprocess.run(['open'] + [str(f) for f in sorted(OUTPUT_DIR.glob("*.png"))])
        print(f"✅ Opened {plots_generated} plots in default viewer")


if __name__ == '__main__':
    main()
