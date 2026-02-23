"""
Generate Daily Email for Role-Spread Points Model (2D/3D Strategies)

Combines yesterday's performance results with today's plays into a formatted email.
Supports both 2D (tier × spread) and 3D (tier × spread × scorer_type) strategies.
Outputs to stdout or sends via AWS SNS.

Context:
This is Step 8 of the daily workflow. It loads:
1. Yesterday's results from S3 (win/loss tracking from Step 7)
2. Today's plays from S3 (betting recommendations from Step 6)

And generates a formatted email containing both sections.

Usage:
    # Generate text email for both strategies (default)
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26
    
    # Only 2D strategy
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --strategy 2d
    
    # Only 3D strategy
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --strategy 3d
    
    # Generate HTML email
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --format html
    
    # Send via AWS SNS
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --sns-topic arn:aws:sns:us-east-2:232692785472:nba-props-alerts
    
    # Specify dates (defaults to today for plays, yesterday for results)
    python scripts/generate_role_spread_points_model_daily_email.py --plays-date 2026-01-05 --results-date 2026-01-04 --strategy both

Output:
    - Console: Formatted email text or HTML
    - SNS: Published to specified topic (if --sns-topic provided)

Author: Thomas Myles
Date: 2026-01-06
"""

import pandas as pd
import boto3
from io import StringIO
import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import Kelly Criterion calculator and odds utilities
from kelly_criterion import calculate_kelly_criterion, kelly_bet_size
from odds_utils import odds_to_implied_probability

# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'error': '❌',
    'push': '🟰',
    'unknown': '❓',
    'fire': '🔥',
    'target': '🎯',
    'chart': '📊',
    'calendar': '📅',
    'clock': '⏰',
    'basketball': '🏀',
    'arrow_up': '📈',
    'arrow_down': '📉',
    'moneybag': '💰',
    'warning': '⚠️',
}

# =============================================================================
# KELLY CRITERION CONSTANTS (defaults - will be overridden by S3 config)
# =============================================================================

DEFAULT_BANKROLL = 10000  # $10,000 default bankroll (fallback if S3 config not available)
DEFAULT_FRACTIONAL_KELLY = 1.0    # 1.0 = full Kelly, 0.5 = half Kelly, 0.25 = quarter Kelly
DEFAULT_MAX_KELLY = 0.10          # Cap Kelly at 10% max single bet
S3_KELLY_CONFIG_PATH = 'config/kelly_bankroll_tracker.json'

# =============================================================================
# S3 PATHS
# =============================================================================

STRATEGY_NAME = 'role_spread_points_model'
S3_BUCKET = 'nba-betting-mt'
S3_PREFIX_PLAYS = f'data/04_output/plays/{STRATEGY_NAME}'
S3_PREFIX_RESULTS = f'data/04_output/results/{STRATEGY_NAME}'

ET_TZ = ZoneInfo('America/New_York')


# =============================================================================
# STRATEGY CONFIG LOADING & ANALYSIS
# =============================================================================

def load_strategy_config_from_s3(config_name='enhanced_unders_v5.json'):
    """
    Load strategy configuration from S3.
    
    Args:
        config_name: Name of config file in S3 strategies folder
    
    Returns:
        dict: Config data with strategies list
    """
    s3_client = boto3.client('s3')
    
    try:
        s3_key = f'strategies/{config_name}'
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        print(f"✅ Loaded strategy config from S3: {config_name}")
        print(f"   Config: {config['name']}")
        print(f"   Strategies: {len(config['strategies'])}")
        
        return config
    except Exception as e:
        print(f"⚠️  Could not load strategy config from S3: {e}")
        return None


def load_backtest_plays_for_strategy(s3_client, strategy, strategy_type, seasons):
    """
    Load historical plays data for a specific strategy across multiple seasons.
    
    Args:
        s3_client: Boto3 S3 client
        strategy: Strategy dict with line_tier, spread_bin, bet_side, scorer_type
        strategy_type: '2d' or '3d'
        seasons: List of seasons to load (e.g., ['2023-24', '2024-25', '2025-26'])
    
    Returns:
        DataFrame with all plays for this strategy across all seasons
    """
    dfs = []
    
    for season in seasons:
        s3_key = f'data/04_output/backtests/{strategy_type}/{season}/plays.csv'
        
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


def generate_strategy_performance_plot(strategy, strategy_type, seasons, output_path):
    """
    Generate 4-panel plot showing strategy performance over time.
    
    Panels:
    1. 2023-24 season: Date vs Win Rate
    2. 2024-25 season: Date vs Win Rate
    3. 2025-26 season: Date vs Win Rate
    4. Overall: All seasons combined
    
    Args:
        strategy: Strategy dict with line_tier, spread_bin, bet_side, scorer_type
        strategy_type: '2d' or '3d'
        seasons: List of seasons (e.g., ['2023-24', '2024-25', '2025-26'])
        output_path: Where to save the PNG
    
    Returns:
        bool: True if successful
    """
    try:
        import logging as _log
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        _log.getLogger('matplotlib.font_manager').setLevel(_log.WARNING)
    except ImportError:
        print(f"   ⚠️  matplotlib not available - cannot generate plot")
        return False
    
    # Load plays data
    s3_client = boto3.client('s3')
    df = load_backtest_plays_for_strategy(s3_client, strategy, strategy_type, seasons)
    
    if df is None or len(df) == 0:
        print(f"   ⚠️  No plays data found for strategy")
        return False
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    # Calculate cumulative win rate over time
    df['is_win'] = (df['result'] == 'WIN').astype(int)
    df['cumulative_wins'] = df.groupby('season')['is_win'].cumsum()
    df['cumulative_plays'] = df.groupby('season').cumsum().index + 1
    df['win_rate'] = (df['cumulative_wins'] / df['cumulative_plays'] * 100)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Strategy Performance: {strategy['line_tier']} | {strategy['spread_bin']} | {strategy['bet_side']}", 
                 fontsize=16, fontweight='bold')
    
    # Define season colors
    season_colors = {
        '2023-24': '#1f77b4',  # Blue
        '2024-25': '#ff7f0e',  # Orange
        '2025-26': '#2ca02c'   # Green
    }
    
    # Plot each season individually (panels 1-3)
    for idx, season in enumerate(seasons[:3]):
        ax = axes[idx // 2, idx % 2]
        
        df_season = df[df['season'] == season].copy()
        
        if len(df_season) == 0:
            ax.text(0.5, 0.5, f'No data for {season}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 100)
            continue
        
        # Reset cumulative for this season
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
        total_wins = df_season['cumulative_wins'].iloc[-1]
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
    for season in seasons:
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
    
    print(f"   ✅ Generated plot: {output_path}")
    return True


def format_strategy_config_analysis(config, plots_dir='/tmp/strategy_plots'):
    """
    Generate strategy config analysis section for email.
    Creates performance plots for each strategy and formats text summary.
    
    Args:
        config: Strategy config dict from S3
        plots_dir: Where to save plot PNGs
    
    Returns:
        str: Formatted text for email
    """
    if not config or 'strategies' not in config:
        return ""
    
    # Create plots directory
    Path(plots_dir).mkdir(exist_ok=True, parents=True)
    
    text = f"""
{'='*80}
{EMOJI['chart']} STRATEGY PORTFOLIO ANALYSIS - {config['name'].upper()}
{'='*80}
Config Version: {config['version']}
Last Updated: {config['last_updated']}
Total Strategies: {len(config['strategies'])}

Description: {config['description']}
{'='*80}

"""
    
    seasons = ['2023-24', '2024-25', '2025-26']
    s3_client = boto3.client('s3')
    
    # Group strategies by tier
    strategies_by_tier = {}
    for strat in config['strategies']:
        tier = strat.get('tier', 'unknown')
        if tier not in strategies_by_tier:
            strategies_by_tier[tier] = []
        strategies_by_tier[tier].append(strat)
    
    # Process each tier
    for tier, strategies in sorted(strategies_by_tier.items()):
        text += f"\n{'─'*80}\n"
        text += f"TIER: {tier.upper()} ({len(strategies)} strategies)\n"
        text += f"{'─'*80}\n\n"
        
        for strat in strategies:
            strategy_name = strat['strategy_name']
            strategy_type = strat['strategy_type']
            
            text += f"{EMOJI['fire']} Strategy: {strategy_name}\n"
            text += f"   Config: {strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
            
            if strategy_type == '3d' and 'scorer_type' in strat:
                text += f" | {strat['scorer_type']}"
            
            text += "\n"
            
            # Show stats if available
            if 'stats' in strat:
                stats = strat['stats']
                text += f"   Performance: {stats['wins']}W-{stats['losses']}L-{stats.get('ties', 0)}T "
                text += f"| Hit Rate: {stats['hit_rate']:.1f}% | ROI: {stats['roi']:+.1f}% | Edge: {stats['edge']:+.1f}%\n"
                text += f"   Sample Size: {stats['sample_size']} games\n"
            
            # Generate plot
            plot_filename = f"{strategy_name}.png"
            plot_path = Path(plots_dir) / plot_filename
            
            text += f"   {EMOJI['chart']} Performance Plot: {plot_filename}\n"
            
            success = generate_strategy_performance_plot(
                strat, 
                strategy_type, 
                seasons, 
                str(plot_path)
            )
            
            if success:
                # Upload to S3
                s3_key = f'data/04_output/strategy_plots/{config["version"]}/{plot_filename}'
                try:
                    s3_client.upload_file(str(plot_path), S3_BUCKET, s3_key)
                    s3_url = f's3://{S3_BUCKET}/{s3_key}'
                    text += f"   {EMOJI['success']} Uploaded to: {s3_url}\n"
                except Exception as e:
                    text += f"   {EMOJI['warning']} Upload failed: {e}\n"
            else:
                text += f"   {EMOJI['warning']} Plot generation failed\n"
            
            text += "\n"
    
    text += f"{'='*80}\n\n"
    
    return text


# =============================================================================
# KELLY BANKROLL LOADING
# =============================================================================

def load_kelly_config_from_s3():
    """
    Load Kelly configuration from S3 including bankroll, fractional_kelly, and max_kelly.
    Falls back to defaults if config not found.
    
    Returns:
        dict with keys: bankroll, fractional_kelly, max_kelly
    """
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KELLY_CONFIG_PATH)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        bankroll = config.get('current_bankroll', DEFAULT_BANKROLL)
        fractional_kelly = config.get('fractional_kelly', DEFAULT_FRACTIONAL_KELLY)
        max_kelly = config.get('max_kelly', DEFAULT_MAX_KELLY)
        
        print(f"✅ Loaded Kelly config from S3:")
        print(f"   Bankroll: ${bankroll:,.2f}")
        print(f"   Fractional Kelly: {fractional_kelly*100:.0f}% ({_kelly_label(fractional_kelly)})")
        print(f"   Max Kelly Cap: {max_kelly*100:.0f}%")
        
        return {
            'bankroll': bankroll,
            'fractional_kelly': fractional_kelly,
            'max_kelly': max_kelly
        }
    except Exception as e:
        print(f"⚠️  Could not load Kelly config from S3 (using defaults): {e}")
        return {
            'bankroll': DEFAULT_BANKROLL,
            'fractional_kelly': DEFAULT_FRACTIONAL_KELLY,
            'max_kelly': DEFAULT_MAX_KELLY
        }


def _kelly_label(fractional_kelly):
    """Return human-readable label for fractional Kelly value"""
    if fractional_kelly == 1.0:
        return "Full Kelly"
    elif fractional_kelly == 0.5:
        return "Half Kelly"
    elif fractional_kelly == 0.25:
        return "Quarter Kelly"
    else:
        return f"{fractional_kelly:.2f}x Kelly"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_skipped_players_from_s3(date_str, strategy='both'):
    """
    Load skipped players metadata from S3
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        strategy: '2d', '3d', or 'both'
    
    Returns:
        dict with keys '2d' and/or '3d', each containing list of skipped players
    """
    s3 = boto3.client('s3')
    results = {}
    
    strategies_to_load = []
    if strategy == 'both':
        strategies_to_load = ['2d', '3d']
    else:
        strategies_to_load = [strategy]
    
    for strat in strategies_to_load:
        key = f"data/04_output/metadata/role_spread_points_model/{strat}/{date_str}_skipped.json"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj['Body'].read().decode('utf-8'))
            results[strat] = data['skipped_players']
            print(f"   ✅ Loaded {len(data['skipped_players'])} {strat.upper()} skipped players for {date_str}", file=sys.stderr)
        except s3.exceptions.NoSuchKey:
            print(f"   ℹ️  No {strat.upper()} skipped players found for {date_str} (normal if all players had bookmakers)", file=sys.stderr)
        except Exception as e:
            print(f"   ⚠️  Error loading {strat.upper()} skipped players: {e}", file=sys.stderr)
    
    return results if results else None


def load_plays_from_s3(date_str, strategy='both', plays_suffix=''):
    """
    Load today's plays from S3 (supports 2d/, 3d/, or both)
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        strategy: '2d', '3d', or 'both'
        plays_suffix: Suffix for filename (e.g., '_top3')
    
    Returns:
        DataFrame with plays (combined if strategy='both'), with overlap detection
    """
    s3 = boto3.client('s3')
    results = {}
    
    strategies_to_load = []
    if strategy in ['2d', 'both']:
        strategies_to_load.append('2d')
    if strategy in ['3d', 'both']:
        strategies_to_load.append('3d')
    
    for strat in strategies_to_load:
        key = f"{S3_PREFIX_PLAYS}/{strat}/{date_str}{plays_suffix}.csv"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df['strategy_dimension'] = strat.upper()  # Tag with 2D or 3D
            results[strat] = df
            print(f"   ✅ Loaded {len(df)} {strat.upper()} plays for {date_str}", file=sys.stderr)
        except s3.exceptions.NoSuchKey:
            print(f"   ⚠️  No {strat.upper()} plays found for {date_str}", file=sys.stderr)
            print(f"      Expected: s3://{S3_BUCKET}/{key}", file=sys.stderr)
        except Exception as e:
            print(f"   ❌ Error loading {strat.upper()} plays: {e}", file=sys.stderr)
    
    if not results:
        return None
    
    # If we have both 2D and 3D, detect overlaps
    if '2d' in results and '3d' in results:
        df_2d = results['2d']
        df_3d = results['3d']
        
        # Find overlaps (same player, bet_side, and line)
        overlaps = pd.merge(
            df_2d[['player', 'bet_side', 'line']],
            df_3d[['player', 'bet_side', 'line']],
            on=['player', 'bet_side', 'line'],
            how='inner'
        )
        
        if len(overlaps) > 0:
            print(f"   🔄 Found {len(overlaps)} overlapping plays between 2D and 3D", file=sys.stderr)
            
            # Mark overlaps in both dataframes
            for _, overlap in overlaps.iterrows():
                mask = (
                    (df_2d['player'] == overlap['player']) &
                    (df_2d['bet_side'] == overlap['bet_side']) &
                    (df_2d['line'] == overlap['line'])
                )
                df_2d.loc[mask, 'strategy_dimension'] = '2D AND 3D'
                
                mask = (
                    (df_3d['player'] == overlap['player']) &
                    (df_3d['bet_side'] == overlap['bet_side']) &
                    (df_3d['line'] == overlap['line'])
                )
                df_3d.loc[mask, 'strategy_dimension'] = '2D AND 3D'
            
            # For overlapping plays, keep only the 3D version (higher ROI)
            # Remove overlaps from 2D
            for _, overlap in overlaps.iterrows():
                mask = (
                    (df_2d['player'] == overlap['player']) &
                    (df_2d['bet_side'] == overlap['bet_side']) &
                    (df_2d['line'] == overlap['line'])
                )
                df_2d = df_2d[~mask]
            
            # Update results
            results['2d'] = df_2d
            results['3d'] = df_3d
    
    # Combine all results
    return pd.concat(list(results.values()), ignore_index=True)


def load_results_from_s3(date_str, strategy='both', tracking_suffix=''):
    """
    Load yesterday's results from S3 (supports 2d/, 3d/, or both)
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        strategy: '2d', '3d', or 'both'
        tracking_suffix: Suffix for filename (e.g., '_top3')
    
    Returns:
        DataFrame with results (combined if strategy='both')
    """
    s3 = boto3.client('s3')
    results = []
    
    strategies_to_load = []
    if strategy in ['2d', 'both']:
        strategies_to_load.append('2d')
    if strategy in ['3d', 'both']:
        strategies_to_load.append('3d')
    
    for strat in strategies_to_load:
        key = f"{S3_PREFIX_RESULTS}/{strat}/{date_str}{tracking_suffix}.csv"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df['strategy_dimension'] = strat.upper()  # Tag with 2D or 3D
            results.append(df)
            print(f"   ✅ Loaded {len(df)} {strat.upper()} results for {date_str}", file=sys.stderr)
        except s3.exceptions.NoSuchKey:
            print(f"   ⚠️  No {strat.upper()} results found for {date_str}", file=sys.stderr)
            print(f"      Expected: s3://{S3_BUCKET}/{key}", file=sys.stderr)
        except Exception as e:
            print(f"   ❌ Error loading {strat.upper()} results: {e}", file=sys.stderr)
    
    if not results:
        return None
    
    # Combine all results
    df_combined = pd.concat(results, ignore_index=True)
    
    # Calculate profit if not already present
    # Standard -110 odds: Stake $110 to win $100
    # WIN: +$100 profit
    # LOSS: -$110 (lose stake)
    # PUSH: $0
    if 'result' in df_combined.columns and 'profit' not in df_combined.columns:
        df_combined['profit'] = df_combined['result'].str.upper().map({
            'WIN': 100.0,    # Stake $110, win $100 profit
            'LOSS': -110.0,  # Lose your $110 stake
            'PUSH': 0.0,     # Get your money back
            'DNP': 0.0       # Did not play
        })
        # Handle any unexpected values
        df_combined['profit'] = df_combined['profit'].fillna(0.0)
    
    return df_combined


def load_season_ytd_results(season, tracking_suffix='_top3'):
    """
    Load ALL tracking results for the season to calculate YTD stats.
    Uses parallel workers for fast loading of 100+ CSV files.
    
    Args:
        season: NBA season (e.g., '2025-26')
        tracking_suffix: Suffix for tracking files (e.g., '_top3')
    
    Returns:
        dict with overall and per-strategy stats, or None if no data
    """
    print(f"📊 Loading season YTD stats (suffix: '{tracking_suffix}')...", file=sys.stderr)
    
    s3 = boto3.client('s3')
    
    # List all tracking result files for the season
    all_files = []
    for dimension in ['2d', '3d']:
        prefix = f"{S3_PREFIX_RESULTS}/{dimension}/"
        
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if 'Contents' in response:
            files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].endswith(f'{tracking_suffix}.csv')
            ]
            all_files.extend(files)
    
    if not all_files:
        print(f"   ⚠️  No YTD tracking files found", file=sys.stderr)
        return None
    
    print(f"   Found {len(all_files)} tracking files to load", file=sys.stderr)
    
    # Parallel load function
    def load_single_file(key):
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            return df
        except Exception as e:
            print(f"   ⚠️  Failed to load {key}: {e}", file=sys.stderr)
            return None
    
    # Load all files in parallel (100 workers for speed)
    all_data = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(load_single_file, key): key for key in all_files}
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    
    if not all_data:
        print(f"   ⚠️  No valid YTD data loaded", file=sys.stderr)
        return None
    
    # Combine all tracking results
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"   ✅ Loaded {len(df_all)} total plays", file=sys.stderr)
    
    # Check if we have any results yet
    if 'result' not in df_all.columns:
        print(f"   ⚠️  No results found in YTD tracking files (suffix: '{tracking_suffix}')", file=sys.stderr)
        return None
    
    # Calculate profit if not already present
    # Standard -110 odds: Stake $110 to win $100
    # WIN: +$100 profit
    # LOSS: -$110 (lose stake)
    # PUSH: $0
    if 'profit' not in df_all.columns:
        print(f"   💰 Calculating profit column...", file=sys.stderr)
        df_all['profit'] = df_all['result'].str.upper().map({
            'WIN': 100.0,    # Stake $110, win $100 profit
            'LOSS': -110.0,  # Lose your $110 stake
            'PUSH': 0.0,     # Get your money back
            'DNP': 0.0       # Did not play
        })
        # Handle any unexpected values
        df_all['profit'] = df_all['profit'].fillna(0.0)
    
    # Calculate overall stats (use uppercase for result values)
    wins = (df_all['result'].str.upper() == 'WIN').sum()
    losses = (df_all['result'].str.upper() == 'LOSS').sum()
    total = wins + losses
    win_pct = (wins / total * 100) if total > 0 else 0
    total_profit = df_all['profit'].sum()
    
    overall_stats = {
        'wins': wins,
        'losses': losses,
        'total': total,
        'win_pct': win_pct,
        'profit': total_profit
    }
    
    # Calculate per-strategy stats (if strategy_name column exists)
    strategy_stats = []
    if 'strategy_name' in df_all.columns:
        for strategy_name in df_all['strategy_name'].dropna().unique():
            df_strat = df_all[df_all['strategy_name'] == strategy_name]
            strat_wins = (df_strat['result'].str.upper() == 'WIN').sum()
            strat_losses = (df_strat['result'].str.upper() == 'LOSS').sum()
            strat_total = strat_wins + strat_losses
            strat_win_pct = (strat_wins / strat_total * 100) if strat_total > 0 else 0
            strat_profit = df_strat['profit'].sum()
            
            strategy_stats.append({
                'name': strategy_name,
                'wins': strat_wins,
                'losses': strat_losses,
                'total': strat_total,
                'win_pct': strat_win_pct,
                'profit': strat_profit
            })
        
        # Sort by profit descending
        strategy_stats = sorted(strategy_stats, key=lambda x: x['profit'], reverse=True)
    
    print(f"   ✅ YTD: {wins}-{losses} ({win_pct:.1f}%) | ${total_profit:,.2f} profit", file=sys.stderr)
    
    return {
        'overall': overall_stats,
        'strategies': strategy_stats
    }


# =============================================================================
# TEXT FORMATTING
# =============================================================================

def format_skipped_players(skipped_dict, date_str):
    """Format skipped players warning for email"""
    if not skipped_dict:
        return ""
    
    # Count total skipped across strategies
    total_skipped = sum(len(players) for players in skipped_dict.values())
    
    if total_skipped == 0:
        return ""
    
    text = f"""
{'='*80}
{EMOJI['warning']} DATA QUALITY NOTE ({date_str})
{'='*80}

{total_skipped} players were skipped due to no bookmakers offering lines near
their consensus (median) line. This may indicate:
  1. Bookmakers offering very different lines (wide spread)
  2. Limited market availability for these players
  3. Potential arbitrage opportunities (investigate manually)

"""
    
    # Show skipped players by strategy
    for strategy, players in sorted(skipped_dict.items()):
        if not players:
            continue
        
        text += f"\n{strategy.upper()} Strategy - {len(players)} skipped:\n"
        for i, p in enumerate(players[:10], 1):  # Show first 10 per strategy
            lines_str = ', '.join(str(l) for l in p.get('lines_offered', []))
            books_str = ', '.join(p.get('bookmakers', []))
            text += f"  {i}. {p['player']} (consensus: {p['consensus_line']}, offered: {lines_str}, books: {books_str})\n"
        
        if len(players) > 10:
            text += f"  ... and {len(players) - 10} more\n"
    
    text += "\n"
    return text


def format_ytd_stats(ytd_stats):
    """Format YTD season stats for email"""
    if not ytd_stats:
        return ""
    
    overall = ytd_stats['overall']
    strategies = ytd_stats['strategies']
    
    text = f"""
{'='*80}
{EMOJI['chart']} 2025-26 SEASON PERFORMANCE (YTD)
{'='*80}

Overall: {overall['wins']}-{overall['losses']} ({overall['win_pct']:.1f}%) | ${overall['profit']:,.2f} profit

"""
    
    if strategies:
        text += "Strategy Breakdown:\n"
        for i, strat in enumerate(strategies, 1):
            text += f"  {i}. {strat['name']:30s} {strat['wins']:3d}-{strat['losses']:2d} ({strat['win_pct']:4.1f}%) | ${strat['profit']:>10,.2f}\n"
        text += "\n"
    
    text += f"{'='*80}\n\n"
    
    return text


def format_results_text(df_results, date_str):
    """Format yesterday's results as text"""
    if df_results is None or df_results.empty:
        return f"""
{'='*80}
{EMOJI['chart']} YESTERDAY'S RESULTS ({date_str})
{'='*80}

No results available for yesterday.
This might be your first day, or yesterday had no plays.

"""
    
    # Calculate summary stats
    total = len(df_results)
    wins = (df_results['result'] == 'WIN').sum()
    losses = (df_results['result'] == 'LOSS').sum()
    pushes = (df_results['result'] == 'PUSH').sum()
    dnp = (df_results['result'] == 'DNP').sum()
    
    win_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Calculate ROI using profit column
    total_wagered = (wins + losses) * 110  # Total amount staked (standard -110 odds)
    profit = df_results['profit'].sum()
    actual_roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    expected_roi = df_results['expected_roi'].mean()
    
    # Build text
    text = f"""
{'='*80}
{EMOJI['chart']} YESTERDAY'S RESULTS ({date_str})
{'='*80}

{EMOJI['success']} {wins} WINS | {EMOJI['error']} {losses} LOSSES"""
    
    if pushes > 0:
        text += f" | {EMOJI['push']} {pushes} PUSHES"
    
    text += f"""
Win Rate: {win_pct:.1f}% | Actual ROI: {actual_roi:+.1f}% | Expected ROI: {expected_roi:+.1f}%

"""
    
    # Per-strategy breakdown (individual strategy names)
    if 'strategy_name' in df_results.columns:
        text += "BREAKDOWN BY STRATEGY:\n"
        text += "─" * 80 + "\n"
        
        strategy_summary = df_results.groupby('strategy_name').apply(
            lambda x: pd.Series({
                'wins': (x['result'] == 'WIN').sum(),
                'losses': (x['result'] == 'LOSS').sum(),
                'profit': x['profit'].sum()
            })
        ).reset_index()
        
        # Calculate win percentage and sort by total plays
        strategy_summary['total_plays'] = strategy_summary['wins'] + strategy_summary['losses']
        strategy_summary['win_pct'] = (
            strategy_summary['wins'] / strategy_summary['total_plays'] * 100
        ).fillna(0)
        strategy_summary = strategy_summary.sort_values('total_plays', ascending=False)
        
        for _, row in strategy_summary.iterrows():
            text += f"{row['strategy_name']:30s}: {int(row['wins'])}-{int(row['losses'])} ({row['win_pct']:.1f}%) | Profit: ${row['profit']:+.2f}\n"
        
        text += "\n"
    
    # Strategy dimension breakdown (if both 2D and 3D present)
    if 'strategy_dimension' in df_results.columns:
        dimensions = df_results['strategy_dimension'].unique()
        if len(dimensions) > 1:
            text += "BREAKDOWN BY DIMENSION:\n"
            text += "─" * 80 + "\n"
            
            for dim in sorted(dimensions):
                dim_data = df_results[df_results['strategy_dimension'] == dim]
                dim_wins = (dim_data['result'] == 'WIN').sum()
                dim_losses = (dim_data['result'] == 'LOSS').sum()
                dim_win_pct = (dim_wins / (dim_wins + dim_losses) * 100) if (dim_wins + dim_losses) > 0 else 0
                dim_profit = dim_data['profit'].sum()
                
                text += f"{dim} Strategy: {dim_wins}-{dim_losses} ({dim_win_pct:.1f}%) | Profit: ${dim_profit:+.2f}\n"
            
            text += "\n"
    
    # Individual bets
    text += "INDIVIDUAL BETS:\n"
    text += "─" * 80 + "\n"
    
    for _, row in df_results.iterrows():
        if row['result'] == 'WIN':
            emoji = EMOJI['success']
        elif row['result'] == 'LOSS':
            emoji = EMOJI['error']
        elif row['result'] == 'PUSH':
            emoji = EMOJI['push']
        else:
            emoji = EMOJI['unknown']
        
        # Format strategy label with dimension and strategy name
        if 'strategy_dimension' in row:
            strat_name = row.get('strategy_name', '')
            strat_label = f"[{row['strategy_dimension']} - {strat_name}]"
        else:
            strat_label = ""
        text += f"{emoji} {row['result']} {strat_label}: {row['player']} {row['bet_side']} {row['line']} pts\n"
        text += f"   Actual: {row['actual_pts']:.0f} pts | Margin: {row['margin']:+.1f}\n"
        text += f"   {row['team']} vs {row['opponent']} | Expected ROI: {row['expected_roi']:+.1f}%\n\n"
    
    return text


def get_best_odds_from_bookmakers(bookmaker_details_json, default_odds=-110):
    """
    Extract best odds from bookmaker details JSON.
    
    "Best" means most favorable for the bettor:
    - For positive odds: highest value (e.g., +150 > +120)
    - For negative odds: closest to 0 (e.g., -105 > -110)
    
    Args:
        bookmaker_details_json: JSON string with bookmaker details
        default_odds: Default odds if no bookmakers available (default -110)
    
    Returns:
        Best American odds as integer
    """
    try:
        details = json.loads(bookmaker_details_json)
        if not details:
            return default_odds
        
        # Extract all odds
        all_odds = [book['odds'] for book in details]
        
        # Find best odds for bettor
        # Positive odds: higher is better (+150 > +120)
        # Negative odds: closer to 0 is better (-105 > -110)
        positive_odds = [o for o in all_odds if o > 0]
        negative_odds = [o for o in all_odds if o < 0]
        
        if positive_odds:
            # If any positive odds, take the highest
            return max(positive_odds)
        elif negative_odds:
            # All negative, take closest to 0 (least negative)
            return max(negative_odds)
        else:
            return default_odds
    except (json.JSONDecodeError, KeyError, TypeError):
        return default_odds


def format_plays_text(df_plays, date_str, kelly_config=None):
    """Format today's plays as text"""
    # Use provided kelly_config or load from S3
    if kelly_config is None:
        kelly_config = load_kelly_config_from_s3()
    
    bankroll = kelly_config['bankroll']
    fractional_kelly = kelly_config['fractional_kelly']
    max_kelly = kelly_config['max_kelly']
    
    if df_plays is None or df_plays.empty:
        return f"""
{'='*80}
{EMOJI['target']} TODAY'S PLAYS ({date_str})
{'='*80}

No plays found for today.
Either no games match our strategies, or plays haven't been generated yet.

"""
    
    # Calculate summary
    total = len(df_plays)
    avg_roi = df_plays['expected_roi'].mean()
    
    # Track Kelly data for summary
    kelly_data = []
    
    text = f"""
{'='*80}
{EMOJI['target']} TODAY'S PLAYS ({date_str})
{'='*80}

Total Plays: {total} | Avg Expected ROI: {avg_roi:+.1f}%

"""
    
    # Strategy dimension breakdown (if both 2D and 3D present)
    if 'strategy_dimension' in df_plays.columns:
        dimensions = df_plays['strategy_dimension'].unique()
        if len(dimensions) > 1 or '2D AND 3D' in dimensions:
            text += "BREAKDOWN BY DIMENSION:\n"
            text += "─" * 80 + "\n"
            
            # Show 2D Only, 3D Only, and Both separately
            if '2D' in dimensions or '2D AND 3D' in dimensions:
                dim_2d_only = df_plays[df_plays['strategy_dimension'] == '2D']
                if len(dim_2d_only) > 0:
                    dim_avg_roi = dim_2d_only['expected_roi'].mean()
                    text += f"2D Only: {len(dim_2d_only)} plays | Avg Expected ROI: {dim_avg_roi:+.1f}%\n"
            
            if '3D' in dimensions or '2D AND 3D' in dimensions:
                dim_3d_only = df_plays[df_plays['strategy_dimension'] == '3D']
                if len(dim_3d_only) > 0:
                    dim_avg_roi = dim_3d_only['expected_roi'].mean()
                    text += f"3D Only: {len(dim_3d_only)} plays | Avg Expected ROI: {dim_avg_roi:+.1f}%\n"
            
            if '2D AND 3D' in dimensions:
                dim_both = df_plays[df_plays['strategy_dimension'] == '2D AND 3D']
                if len(dim_both) > 0:
                    dim_avg_roi = dim_both['expected_roi'].mean()
                    text += f"Both (2D+3D): {len(dim_both)} plays | Avg Expected ROI: {dim_avg_roi:+.1f}%\n"
            
            # Calculate total unique plays
            total_unique = len(df_plays)
            text += f"Total Unique: {total_unique} plays\n"
            
            text += "\n"
    
    # Strategy breakdown (by strategy_name if available)
    if 'strategy_name' in df_plays.columns:
        text += "BREAKDOWN BY STRATEGY:\n"
        text += "─" * 80 + "\n"
        
        # Get unique strategy names and their stats
        for strategy_name in sorted(df_plays['strategy_name'].unique()):
            strat_plays = df_plays[df_plays['strategy_name'] == strategy_name]
            strat_count = len(strat_plays)
            strat_avg_roi = strat_plays['expected_roi'].mean()
            text += f"{strategy_name}: {strat_count} {'play' if strat_count == 1 else 'plays'} | Avg Expected ROI: {strat_avg_roi:+.1f}%\n"
        
        text += "─" * 80 + "\n"
        text += f"Total Plays: {total} | Avg Expected ROI: {avg_roi:+.1f}%\n"
        text += "\n"
    
    # Kelly Criterion Summary (will be populated as we process plays)
    # This will be added after we've calculated all Kelly values
    kelly_summary_placeholder = "KELLY_SUMMARY_PLACEHOLDER"
    text += kelly_summary_placeholder + "\n"
    
    # Group plays by game (team + opponent)
    # Create a sortable game identifier - prefer away@home if available
    has_away_home = 'away_team' in df_plays.columns and 'home_team' in df_plays.columns
    if has_away_home:
        df_plays['game_key'] = df_plays.apply(
            lambda r: (r['away_team'], r['home_team']),
            axis=1
        )
    else:
        # Fallback to sorted tuple
        df_plays['game_key'] = df_plays.apply(
            lambda r: tuple(sorted([r['team'], r['opponent']])), axis=1
        )
    
    # Get unique games and their first occurrence for sorting
    games = df_plays.groupby('game_key').first().reset_index()
    
    # Sort games by game_time (chronological order)
    games['game_time_parsed'] = pd.to_datetime(games['game_time'])
    games = games.sort_values('game_time_parsed')
    
    game_num = 1
    for _, game in games.iterrows():
        game_teams = game['game_key']
        
        # Determine team names for display
        if has_away_home and isinstance(game_teams, tuple) and len(game_teams) == 2:
            away_team, home_team = game_teams
            team1, team2 = away_team, home_team
        else:
            team1, team2 = game_teams if isinstance(game_teams, tuple) else (game['team'], game['opponent'])
        
        # Get all plays for this game
        game_plays = df_plays[df_plays['game_key'] == game_teams].copy()
        
        # Sort by ROI descending
        game_plays = game_plays.sort_values('expected_roi', ascending=False)
        
        # Format game time if available
        game_time_str = ""
        if 'game_time' in game.index and pd.notna(game['game_time']):
            try:
                # Parse game_time - it might be a string or datetime
                if isinstance(game['game_time'], str):
                    game_time_dt = pd.to_datetime(game['game_time'])
                else:
                    game_time_dt = game['game_time']
                
                # Ensure it's timezone-aware (ET)
                if game_time_dt.tzinfo is None:
                    game_time_dt = game_time_dt.tz_localize(ET_TZ)
                else:
                    game_time_dt = game_time_dt.astimezone(ET_TZ)
                
                # Format as "6pm ET" (no minutes if on the hour)
                if game_time_dt.minute == 0:
                    time_formatted = game_time_dt.strftime('%I%p ET').lstrip('0').lower()
                else:
                    time_formatted = game_time_dt.strftime('%I:%M%p ET').lstrip('0').lower()
                game_time_str = f" ({time_formatted})"
            except Exception:
                # If parsing fails, just skip the time
                pass
        
        # Use @ if we have away/home, otherwise use vs
        game_separator = '@' if has_away_home else 'vs'
        
        text += f"""{'─'*80}
{EMOJI['basketball']} GAME {game_num}: {team1} {game_separator} {team2}{game_time_str}
{'─'*80}

"""
        
        for _, play in game_plays.iterrows():
            # Format strategy label with dimension and strategy name
            if 'strategy_dimension' in play:
                strat_name = play.get('strategy_name', '')
                strat_label = f"[{play['strategy_dimension']} - {strat_name}]"
            else:
                strat_label = ""
            # Calculate win-loss record from hit rate
            games = play['games_in_sample']
            hit_rate = play['hit_rate'] / 100
            wins = int(games * hit_rate)
            losses = games - wins
            
            text += f"{EMOJI['fire']} {strat_label} {play['bet_side']}: {play['player']} {play['line']} pts\n"
            text += f"   Team: {play['team']} (Spread: {play['spread']:+.1f})\n"
            text += f"   Strategy: {play['strategy_name']}\n"
            text += f"   Expected ROI: {play['expected_roi']:+.1f}% | Hit Rate: {play['hit_rate']:.1f}% (n={games}, {wins}-{losses})\n"
            text += f"   Edge vs Baseline: {play['edge_vs_baseline']:+.1f}% | Edge vs Breakeven: {play['edge_vs_breakeven']:+.1f}%\n"
            
            # Calculate Kelly Criterion
            win_prob = play['hit_rate'] / 100  # Convert to decimal
            bet_side = play['bet_side']
            bookmaker_details = play['bookmaker_details_over'] if bet_side == 'OVER' else play['bookmaker_details_under']
            best_odds = get_best_odds_from_bookmakers(bookmaker_details)
            
            kelly_result = calculate_kelly_criterion(win_prob, best_odds, max_kelly=max_kelly)
            kelly_pct = kelly_result['kelly_pct']
            
            # Apply fractional Kelly
            fractional_kelly_pct = kelly_pct * fractional_kelly
            kelly_pct_display = fractional_kelly_pct  # Display the fractional Kelly value
            kelly_dollars = kelly_bet_size(fractional_kelly_pct, bankroll)
            is_capped = kelly_result['capped']
            
            # Store Kelly data for summary
            kelly_data.append({
                'player': play['player'],
                'kelly_pct': fractional_kelly_pct,  # Store fractional Kelly for summary
                'kelly_dollars': kelly_dollars,
                'odds': best_odds
            })
            
            # Calculate implied probability and edge for display
            from odds_utils import odds_to_implied_probability
            implied_prob = odds_to_implied_probability(best_odds)
            breakeven_prob = 0.5238  # 52.38% breakeven for -110 odds
            edge_vs_breakeven = win_prob - breakeven_prob
            edge_vs_implied = win_prob - implied_prob
            
            # Calculate b (net profit per dollar) for Kelly formula explanation
            if best_odds > 0:
                b = best_odds / 100  # Positive odds: profit per $1
            else:
                b = 100 / abs(best_odds)  # Negative odds: profit per $1
            
            # Kelly formula components
            q = 1 - win_prob  # Lose probability
            kelly_numerator = (b * win_prob) - q
            kelly_before_cap = kelly_numerator / b if b > 0 else 0
            
            # Display Kelly info with detailed breakdown
            capped_warning = f" {EMOJI['warning']} CAPPED" if is_capped else ""
            
            text += f"   {EMOJI['moneybag']} Kelly Analysis:\n"
            text += f"      Win Prob: {win_prob*100:.1f}% | Implied Prob @ {best_odds:+d}: {implied_prob*100:.1f}% | Edge: {edge_vs_implied*100:+.1f}%\n"
            text += f"      Kelly Formula: ({b:.3f} × {win_prob:.3f} - {q:.3f}) / {b:.3f} = {kelly_before_cap*100:.1f}%\n"
            if is_capped:
                text += f"      Full Kelly: {kelly_before_cap*100:.1f}% → Capped at {max_kelly*100:.0f}% = {kelly_pct*100:.1f}%\n"
            if fractional_kelly < 1.0:
                text += f"      Fractional ({fractional_kelly*100:.0f}%): {kelly_pct*100:.1f}% × {fractional_kelly:.2f} = {fractional_kelly_pct*100:.1f}%\n"
            text += f"      → Bet Size: {fractional_kelly_pct*100:.1f}% of bankroll = ${kelly_dollars:.0f}{capped_warning}\n"
            
            # Show bookmakers offering this line (detailed format with BOTH sides for context)
            details_over = json.loads(play['bookmaker_details_over'])
            details_under = json.loads(play['bookmaker_details_under'])
            
            # Show the side we're betting
            bet_side = play['bet_side']
            details_bet_side = details_over if bet_side == 'OVER' else details_under
            details_other_side = details_under if bet_side == 'OVER' else details_over
            other_side_name = 'UNDER' if bet_side == 'OVER' else 'OVER'
            
            num_books = len(details_bet_side)
            text += f"   Books ({num_books}): "
            
            if num_books == 0:
                text += f"⚠️  No books offering {bet_side} at this line\n"
            else:
                # Format bookmakers for our bet side
                book_strs = []
                for book_info in details_bet_side:
                    bookmaker = book_info['bookmaker']
                    line = book_info['line']
                    odds = book_info['odds']
                    odds_str = f"{odds:+d}"
                    book_strs.append(f"{bookmaker} ({line} @ {odds_str})")
                text += ', '.join(book_strs) + "\n"
            
            # Show other side for context (if available)
            if details_other_side:
                text += f"   {other_side_name} available at: "
                other_book_strs = []
                for book_info in details_other_side[:3]:  # Show first 3 for brevity
                    bookmaker = book_info['bookmaker']
                    line = book_info['line']
                    odds = book_info['odds']
                    odds_str = f"{odds:+d}"
                    other_book_strs.append(f"{bookmaker} ({line} @ {odds_str})")
                if len(details_other_side) > 3:
                    other_book_strs.append(f"... +{len(details_other_side) - 3} more")
                text += ', '.join(other_book_strs) + "\n"
            
            text += "\n"
        
        game_num += 1
    
    # Generate Kelly Criterion Summary
    if kelly_data:
        total_kelly_pct = sum([k['kelly_pct'] for k in kelly_data])
        avg_kelly_pct = total_kelly_pct / len(kelly_data) if kelly_data else 0
        max_kelly_play = max(kelly_data, key=lambda k: k['kelly_pct'])
        total_kelly_dollars = sum([k['kelly_dollars'] for k in kelly_data])
        
        # Calculate what full Kelly would be (for comparison)
        total_full_kelly_dollars = total_kelly_dollars / fractional_kelly if fractional_kelly > 0 else total_kelly_dollars
        
        kelly_label = _kelly_label(fractional_kelly)
        
        kelly_summary = f"""KELLY CRITERION BETTING SUMMARY ({kelly_label}):
{'─'*80}
{EMOJI['moneybag']} Current Bankroll: ${bankroll:,.2f}
{EMOJI['chart']} Using: {fractional_kelly*100:.0f}% Kelly ({kelly_label})
{EMOJI['moneybag']} Total Kelly: {total_kelly_pct*100:.1f}% of bankroll (Sum of all {len(kelly_data)} plays)
{EMOJI['chart']} Avg Kelly per play: {avg_kelly_pct*100:.1f}% of bankroll
{EMOJI['fire']} Max single Kelly: {max_kelly_play['kelly_pct']*100:.1f}% ({max_kelly_play['player']})

Recommended Total Risk (using {kelly_label}):
  • ${total_kelly_dollars:.0f} total across {len(kelly_data)} bets

Comparison by Kelly Fraction:
  • Full Kelly (100%): ${total_full_kelly_dollars:.0f} total
  • Half Kelly (50%): ${total_full_kelly_dollars/2:.0f} total
  • Quarter Kelly (25%): ${total_full_kelly_dollars/4:.0f} total
  • Current ({fractional_kelly*100:.0f}%): ${total_kelly_dollars:.0f} total ← YOU ARE HERE

Standard Fixed-Size Comparison:
  • Fixed $110 per bet: ${110 * len(kelly_data):,.0f} total risked
"""
        
        # Add warning if total Kelly > 100%
        if total_kelly_pct > 1.0:
            kelly_summary += f"\n{EMOJI['warning']} WARNING: Total Kelly ({total_kelly_pct*100:.1f}%) > 100% suggests correlation between bets.\n"
            kelly_summary += f"   Consider reducing fractional Kelly (currently {fractional_kelly*100:.0f}%) to manage risk.\n"
        
        kelly_summary += "\n"
        
        # Replace placeholder with actual summary
        text = text.replace(kelly_summary_placeholder + "\n", kelly_summary)
    else:
        # Remove placeholder if no Kelly data
        text = text.replace(kelly_summary_placeholder + "\n", "")
    
    return text


def generate_plays_summary(df_plays):
    """
    Generate a concise summary of plays grouped by game, sorted by game start time.
    Format: AWAY @ HOME (time)
    
    Returns:
        str: Formatted summary text
    """
    if df_plays is None or len(df_plays) == 0:
        return ""
    
    # Check if we have away_team and home_team columns
    has_away_home = 'away_team' in df_plays.columns and 'home_team' in df_plays.columns
    
    if has_away_home:
        # Create game_key using away @ home format
        df_plays['game_key'] = df_plays.apply(
            lambda r: (r['away_team'], r['home_team']),
            axis=1
        )
    else:
        # Fallback to old method if columns not present
        df_plays['game_key'] = df_plays.apply(
            lambda r: tuple(sorted([r['team'], r['opponent']])),
            axis=1
        )
    
    # Group plays by game
    agg_dict = {
        'game_time': 'first',
        'team': 'first',
        'opponent': 'first'
    }
    
    if has_away_home:
        agg_dict['away_team'] = 'first'
        agg_dict['home_team'] = 'first'
    
    games = df_plays.groupby('game_key').agg(agg_dict).reset_index()
    
    # Sort by game_time
    games['game_time_parsed'] = pd.to_datetime(games['game_time'])
    games = games.sort_values('game_time_parsed')
    
    # Get day of week for header
    if len(games) > 0:
        first_game_time = games.iloc[0]['game_time_parsed']
        if first_game_time.tzinfo is None:
            first_game_time = first_game_time.tz_localize(ET_TZ)
        else:
            first_game_time = first_game_time.astimezone(ET_TZ)
        day_of_week = first_game_time.strftime('%A').lower()
    else:
        day_of_week = 'today'
    
    summary_lines = [
        f"nba {day_of_week}!",
        ""
    ]
    
    for _, game in games.iterrows():
        # Determine away and home team
        if has_away_home and pd.notna(game.get('away_team')) and pd.notna(game.get('home_team')):
            away_team = game['away_team']
            home_team = game['home_team']
        else:
            # Fallback: use alphabetical order
            game_teams = game['game_key']
            if isinstance(game_teams, tuple) and len(game_teams) == 2:
                away_team, home_team = game_teams
            else:
                away_team, home_team = game['team'], game['opponent']
        
        # Format game time
        game_time_str = ""
        if pd.notna(game['game_time']):
            try:
                if isinstance(game['game_time'], str):
                    game_time_dt = pd.to_datetime(game['game_time'])
                else:
                    game_time_dt = game['game_time']
                
                if game_time_dt.tzinfo is None:
                    game_time_dt = game_time_dt.tz_localize(ET_TZ)
                else:
                    game_time_dt = game_time_dt.astimezone(ET_TZ)
                
                # Format as "7:10pm et" (lowercase, include minutes)
                game_time_str = game_time_dt.strftime('%I:%M%p et').lstrip('0').lower()
            except Exception:
                pass
        
        # Get all plays for this game
        game_plays = df_plays[df_plays['game_key'] == game['game_key']].copy()
        game_plays = game_plays.sort_values('expected_roi', ascending=False)
        
        # Game header: AWAY @ HOME (time)
        summary_lines.append(f"{away_team} @ {home_team} ({game_time_str})")
        
        # Add each play
        for _, play in game_plays.iterrows():
            bet_side_lower = play['bet_side'].lower()[0]  # 'o' or 'u'
            summary_lines.append(f"- {play['player']} {bet_side_lower}{play['line']}")
        
        summary_lines.append("")
    
    return "\n".join(summary_lines)


def generate_email_text(df_results, results_date, df_plays, plays_date, custom_title=None, ytd_stats=None, skipped_players=None, include_strategy_analysis=False):
    """Generate complete email body in text format"""
    
    if custom_title:
        subject = f"{custom_title} - {plays_date}"
    else:
        subject = f"NBA Role-Spread Model: Plays for {plays_date}"
        if df_results is not None:
            subject += f" + {results_date} Results"
    
    body = f"""
{'='*80}
{EMOJI['basketball']} NBA ROLE-SPREAD POINTS MODEL - DAILY UPDATE
{'='*80}
{EMOJI['calendar']} Generated: {datetime.now(ET_TZ).strftime('%Y-%m-%d %I:%M %p ET')}
{'='*80}
"""
    
    # Load Kelly config (includes bankroll, fractional_kelly, max_kelly)
    kelly_config = load_kelly_config_from_s3()
    
    # Add strategy portfolio analysis (if requested)
    if include_strategy_analysis:
        config = load_strategy_config_from_s3('enhanced_unders_v5.json')
        if config:
            body += format_strategy_config_analysis(config)
    
    # Add YTD stats first (if provided)
    if ytd_stats:
        body += format_ytd_stats(ytd_stats)
    
    # Add results (yesterday's performance)
    body += format_results_text(df_results, results_date)
    
    # Add today's plays (with Kelly config)
    body += format_plays_text(df_plays, plays_date, kelly_config=kelly_config)
    
    # Add skipped players warning (if any)
    if skipped_players:
        body += format_skipped_players(skipped_players, plays_date)
    
    # Add plays summary at the end (before Kelly explanation)
    body += generate_plays_summary(df_plays)
    
    # Add Kelly explanation footnote
    body += f"""
{'='*80}
📚 KELLY CRITERION EXPLAINED
{'='*80}

The Kelly Criterion calculates optimal bet size based on your edge and the odds.

Formula: Kelly% = (bp - q) / b

Where:
  • b = net profit per dollar wagered
       At -110 odds: You risk $110 to win $100
       So b = 100/110 = 0.909 (profit per dollar risked)
  
  • p = win probability (from our model's historical hit rate)
       Example: 57.2% = 0.572
  
  • q = lose probability (1 - p)
       Example: 1 - 0.572 = 0.428

Example Calculation (57.2% win rate @ -110 odds):
  Kelly = (0.909 × 0.572 - 0.428) / 0.909
        = (0.520 - 0.428) / 0.909
        = 0.092 / 0.909
        = 10.1% of bankroll

Fractional Kelly:
  Most bettors use fractional Kelly to reduce variance:
  • Quarter Kelly (0.25x): 10.1% → 2.5% bet size (conservative)
  • Half Kelly (0.50x): 10.1% → 5.1% bet size (moderate)
  • Full Kelly (1.00x): 10.1% bet size (aggressive)

Current Setting: {kelly_config['fractional_kelly']*100:.0f}% Kelly ({_kelly_label(kelly_config['fractional_kelly'])})

{'='*80}
Strategy: Role-Spread Points Model (Detailed Granularity)
Generated by: /betting/scripts/generate_role_spread_points_model_daily_email.py
{'='*80}
"""
    
    return subject, body


# =============================================================================
# HTML FORMATTING (Optional)
# =============================================================================

def generate_email_html(df_results, results_date, df_plays, plays_date, custom_title=None, ytd_stats=None, skipped_players=None, include_strategy_analysis=False):
    """Generate complete email body in HTML format"""
    # TODO: Implement HTML formatting if needed
    # For now, just wrap text in <pre> tags
    subject, text_body = generate_email_text(df_results, results_date, df_plays, plays_date, custom_title, ytd_stats, skipped_players, include_strategy_analysis)
    html_body = f"<html><body><pre>{text_body}</pre></body></html>"
    return subject, html_body


# =============================================================================
# SNS PUBLISHING
# =============================================================================

def publish_to_sns(subject, body, topic_arn, format='text'):
    """Publish email to AWS SNS topic"""
    # Extract region from ARN: arn:aws:sns:REGION:ACCOUNT:TOPIC
    region = topic_arn.split(':')[3] if ':' in topic_arn else 'us-east-2'
    sns = boto3.client('sns', region_name=region)
    
    try:
        response = sns.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=body,
            MessageStructure='string'
        )
        print(f"{EMOJI['success']} Published to SNS: {topic_arn}", file=sys.stderr)
        print(f"   Message ID: {response['MessageId']}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"{EMOJI['error']} Failed to publish to SNS: {e}", file=sys.stderr)
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate daily email for Role-Spread Points Model (2D/3D)'
    )
    parser.add_argument('--season', type=str, default='2025-26',
                       help='NBA season (e.g., 2025-26)')
    parser.add_argument('--plays-date', type=str, default=None,
                       help='Date for plays (YYYY-MM-DD). Defaults to today (ET).')
    parser.add_argument('--results-date', type=str, default=None,
                       help='Date for results (YYYY-MM-DD). Defaults to yesterday (ET).')
    parser.add_argument('--strategy', type=str, default='both', choices=['2d', '3d', 'both'],
                       help='Which strategy to include: 2d, 3d, or both (default: both)')
    parser.add_argument('--format', choices=['text', 'html'], default='text',
                       help='Email format (default: text)')
    parser.add_argument('--sns-topic', type=str, default=None,
                       help='AWS SNS topic ARN to publish to (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save email body to file (optional)')
    parser.add_argument('--plays-suffix', type=str, default='',
                       help='Suffix for plays filename (e.g., "_top3")')
    parser.add_argument('--tracking-suffix', type=str, default='',
                       help='Suffix for tracking filename (e.g., "_top3")')
    parser.add_argument('--email-title', type=str, default='NBA Daily Props Report',
                       help='Custom email subject line')
    parser.add_argument('--load-ytd', action='store_true', default=False,
                       help='Load and display YTD season stats (default: False)')
    parser.add_argument('--include-strategy-plots', action='store_true', default=False,
                       help='Generate and include strategy performance plots (default: False)')
    
    args = parser.parse_args()
    
    # Calculate dates in ET
    now_et = datetime.now(ET_TZ)
    
    if args.plays_date:
        plays_date = args.plays_date
    else:
        plays_date = now_et.strftime('%Y-%m-%d')
    
    if args.results_date:
        results_date = args.results_date
    else:
        yesterday_et = now_et - timedelta(days=1)
        results_date = yesterday_et.strftime('%Y-%m-%d')
    
    print(f"{EMOJI['calendar']} Generating email...", file=sys.stderr)
    print(f"   Plays date: {plays_date}", file=sys.stderr)
    print(f"   Results date: {results_date}", file=sys.stderr)
    print(f"   Strategy: {args.strategy}", file=sys.stderr)
    print(f"   Format: {args.format}", file=sys.stderr)
    print(f"   Season: {args.season}\n", file=sys.stderr)
    
    # Load data
    df_plays = load_plays_from_s3(plays_date, strategy=args.strategy, plays_suffix=args.plays_suffix)
    df_results = load_results_from_s3(results_date, strategy=args.strategy, tracking_suffix=args.tracking_suffix)
    skipped_players = load_skipped_players_from_s3(plays_date, strategy=args.strategy)
    
    if df_plays is None:
        print(f"{EMOJI['error']} Warning: No plays found for {plays_date} (strategy: {args.strategy})", file=sys.stderr)
        print(f"   Expected: s3://{S3_BUCKET}/{S3_PREFIX_PLAYS}/{{2d,3d}}/{plays_date}.csv\n", file=sys.stderr)
    else:
        print(f"{EMOJI['success']} Loaded {len(df_plays)} plays for {plays_date}", file=sys.stderr)
    
    if df_results is None:
        print(f"{EMOJI['error']} Warning: No results found for {results_date} (strategy: {args.strategy})", file=sys.stderr)
        print(f"   Expected: s3://{S3_BUCKET}/{S3_PREFIX_RESULTS}/{{2d,3d}}/{results_date}.csv\n", file=sys.stderr)
    else:
        print(f"{EMOJI['success']} Loaded {len(df_results)} results for {results_date}\n", file=sys.stderr)
    
    # Load YTD stats if requested
    ytd_stats = None
    if args.load_ytd:
        ytd_stats = load_season_ytd_results(args.season, tracking_suffix=args.tracking_suffix)
    
    # Generate email
    if args.format == 'html':
        subject, body = generate_email_html(df_results, results_date, df_plays, plays_date, args.email_title, ytd_stats, skipped_players, args.include_strategy_plots)
    else:
        subject, body = generate_email_text(df_results, results_date, df_plays, plays_date, args.email_title, ytd_stats, skipped_players, args.include_strategy_plots)
    
    # Output
    if args.output:
        # Save to file
        with open(args.output, 'w') as f:
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
        print(f"{EMOJI['success']} Saved email to: {args.output}\n", file=sys.stderr)
    
    if args.sns_topic:
        # Publish to SNS
        publish_to_sns(subject, body, args.sns_topic, args.format)
    
    # Always print to stdout
    print(f"Subject: {subject}\n")
    print(body)


if __name__ == '__main__':
    main()

