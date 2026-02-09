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
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Matplotlib setup (non-interactive backend)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 buckets
S3_BUCKET = 'nba-betting-mt'
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_NBA = 'nba-api-mt'
BACKTEST_PREFIX = 'data/04_output/backtests'

# Seasons to analyze
SEASONS = ['2023-24', '2024-25', '2025-26']

# Output directory
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp' / 'strategy_plots'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Cache directory (for downloaded data)
CACHE_DIR = Path.home() / 'Downloads' / 'tmp' / 'cache'
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Config S3 key
CONFIG_S3_KEY = 'strategies/enhanced_unders_v5.json'

# Team name mappings: Odds API → NBA API (NBA API is source of truth)
# Different APIs use different team name formats, so we normalize Odds API names
# to match NBA API format for consistent joins across all data sources
ODDS_TO_NBA_TEAM_MAP = {
    'Los Angeles Clippers': 'LA Clippers',  # Odds API → NBA API (source of truth)
    # Note: Lakers already match, no mapping needed
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_today_et():
    """Get today's date in ET timezone (for cache naming)."""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    et_tz = ZoneInfo('America/New_York')
    return datetime.now(et_tz).strftime('%Y-%m-%d')


def bin_points_line(line):
    """Bin player points line into tiers."""
    if pd.isna(line):
        return 'Unknown'
    if line < 5:
        return '<5 (Deep Bench)'
    elif line < 10:
        return '5-10 (Bench)'
    elif line < 15:
        return '10-15 (Role Player)'
    elif line < 20:
        return '15-20 (High Role)'
    elif line < 25:
        return '20-25 (Star)'
    elif line < 30:
        return '25-30 (High Star)'
    elif line < 35:
        return '30-35 (Superstar)'
    elif line < 40:
        return '35-40 (Elite)'
    else:
        return '40+ (MVP)'


def bin_team_spread(spread):
    """Bin team spread into categories."""
    if pd.isna(spread):
        return 'Unknown'
    if spread < -15:
        return '15+ Fav'
    elif spread < -10:
        return '10-15 Fav'
    elif spread < -6:
        return '6-10 Fav'
    elif spread < -2:
        return '2-6 Fav'
    elif spread <= 2:
        return "Pick'em (-2 to +2)"
    elif spread <= 6:
        return '2-6 Dog'
    elif spread <= 10:
        return '6-10 Dog'
    elif spread <= 15:
        return '10-15 Dog'
    else:
        return '15+ Dog'


def get_cache_path(season):
    """Get cache file path for a season (unified for 2D and 3D)."""
    today_et = get_today_et()
    cache_filename = f"plays_{season}_{today_et}.parquet"
    return CACHE_DIR / cache_filename


def load_from_cache(season):
    """Load plays from cache if exists for today."""
    cache_path = get_cache_path(season)
    if cache_path.exists():
        print(f"   💾 Loading from cache: {cache_path.name}")
        df = pd.read_parquet(cache_path)
        return df
    return None


def save_to_cache(df, season):
    """Save plays to cache."""
    cache_path = get_cache_path(season)
    df.to_parquet(cache_path, index=False)
    print(f"   💾 Saved to cache: {cache_path.name}")
    
    # Clean up old cache files for this season
    pattern = f"plays_{season}_*.parquet"
    for old_file in CACHE_DIR.glob(pattern):
        if old_file != cache_path:
            old_file.unlink()
            print(f"   🗑️  Removed old cache: {old_file.name}")

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
# LOAD BACKTEST PLAYS (WITH CACHE AND SELF-CONTAINED JOIN)
# =============================================================================

def load_and_join_raw_data(season):
    """
    Load and join raw data from source S3 buckets (self-contained).
    Always loads complete dataset including scorer_type for both 2D and 3D use.
    
    Returns DataFrame with plays for this season.
    """
    print(f"   🔄 Loading and joining fresh data from source buckets for {season}...")
    
    s3_client = boto3.client('s3')
    
    # Import player_name_utils
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from player_name_utils import normalize_player_name
    
    # Step 1: Load player props
    print(f"   📊 Loading props...")
    prefix = f"nba/historical_player_props/{season}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"   ❌ No props files found")
        return None
    
    all_props = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv'):
            try:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_PROPS, Key=obj['Key'])
                df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                filename = obj['Key'].split('/')[-1]
                date_str = filename.replace('.csv', '')
                df['game_date'] = date_str
                all_props.append(df)
            except Exception:
                continue
    
    if not all_props:
        return None
    
    df_props = pd.concat(all_props, ignore_index=True)
    df_props['player_normalized'] = df_props['player'].apply(normalize_player_name)
    print(f"      ✅ {len(df_props):,} prop rows ({df_props['game_date'].min()} to {df_props['game_date'].max()})")
    
    # Step 2: Load game logs
    print(f"   🏀 Loading game logs...")
    prefix = f"player_game_logs/{season}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
    
    if 'Contents' not in response:
        return None
    
    all_game_logs = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv'):
            try:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_NBA, Key=obj['Key'])
                df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                all_game_logs.append(df)
            except Exception:
                continue
    
    if not all_game_logs:
        return None
    
    df_games = pd.concat(all_game_logs, ignore_index=True)
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
    df_games['game_date'] = df_games['GAME_DATE'].dt.date.astype(str)
    df_games['player_normalized'] = df_games['PLAYER_NAME'].apply(normalize_player_name)
    df_games = df_games[df_games['MIN'].notna() & (df_games['MIN'] > 0)].copy()
    print(f"      ✅ {len(df_games):,} player-game rows ({df_games['game_date'].min()} to {df_games['game_date'].max()})")
    
    # Step 3: Load shot charts (ALWAYS load for complete dataset)
    print(f"   🎯 Loading shot charts...")
    prefix = f"player_shot_charts/{season}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
    
    df_shots = None
    if 'Contents' in response:
        all_shot_data = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                try:
                    file_name = obj['Key'].split('/')[-1].replace('.csv', '')
                    parts = file_name.split('_')
                    if len(parts) >= 2:
                        player_name_raw = ' '.join(parts[:-1])
                        player_normalized = normalize_player_name(player_name_raw)
                        
                        obj_data = s3_client.get_object(Bucket=S3_BUCKET_NBA, Key=obj['Key'])
                        df_shots_player = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                        
                        rim_shots = df_shots_player[df_shots_player['SHOT_DISTANCE'] <= 6]
                        rim_makes = rim_shots['SHOT_MADE_FLAG'].sum() if not rim_shots.empty else 0
                        rim_points = rim_makes * 2
                        
                        all_shot_data.append({
                            'player_normalized': player_normalized,
                            'rim_season_points': rim_points
                        })
                except Exception:
                    continue
        
        if all_shot_data:
            df_shots = pd.DataFrame(all_shot_data)
            print(f"      ✅ {len(df_shots):,} player shot charts")
    
    # Step 4: Load game lines
    print(f"   📈 Loading game lines...")
    prefix = f"nba/historical_game_lines/{season}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
    
    df_lines = None
    if 'Contents' in response:
        all_lines = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv') and 'nba_game_lines' in obj['Key']:
                try:
                    obj_data = s3_client.get_object(Bucket=S3_BUCKET_PROPS, Key=obj['Key'])
                    df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                    filename = obj['Key'].split('/')[-1]
                    date_str = filename.replace('nba_game_lines_', '').replace('.csv', '')
                    df['game_date'] = date_str
                    all_lines.append(df)
                except Exception:
                    continue
        
        if all_lines:
            df_lines = pd.concat(all_lines, ignore_index=True)
            
            # =====================================================================
            # NORMALIZE TEAM NAMES: ODDS API → NBA API (SOURCE OF TRUTH)
            # =====================================================================
            # Odds API uses "Los Angeles Clippers", NBA API uses "LA Clippers"
            # NBA API is the source of truth for all game data, so we standardize
            # the Odds API team names to match NBA API format.
            # =====================================================================
            df_lines['home_team'] = df_lines['home_team'].replace(ODDS_TO_NBA_TEAM_MAP)
            df_lines['away_team'] = df_lines['away_team'].replace(ODDS_TO_NBA_TEAM_MAP)
            
            consensus = df_lines.groupby(['game_id', 'game_date', 'away_team', 'home_team', 'market']).agg({
                'away_line': 'mean',
                'home_line': 'mean'
            }).reset_index()
            spread = consensus[consensus['market'] == 'spread'][['game_id', 'game_date', 'away_team', 'home_team', 'away_line', 'home_line']]
            spread.columns = ['game_id', 'game_date', 'away_team', 'home_team', 'away_spread', 'home_spread']
            df_lines = spread
            print(f"      ✅ {len(df_lines):,} games with spreads")
    
    # Step 5: Join
    print(f"   🔗 Joining datasets...")
    
    # Filter props to player_points and aggregate
    df_props = df_props[df_props['market'] == 'player_points'].copy()
    props_agg = df_props.groupby(['player_normalized', 'game_date']).agg({
        'prop_line': 'mean'
    }).reset_index()
    props_agg.columns = ['player_normalized', 'game_date', 'points_line']
    
    df_merged = df_games.copy()
    df_merged = df_merged.merge(props_agg, on=['player_normalized', 'game_date'], how='left')
    
    # Join game lines
    if df_lines is not None:
        df_merged['is_home'] = ~df_merged['MATCHUP'].str.contains('@')
        df_merged_home = df_merged[df_merged['is_home']].copy()
        df_merged_away = df_merged[~df_merged['is_home']].copy()
        
        df_merged_home = df_merged_home.merge(
            df_lines[['game_date', 'home_team', 'home_spread']],
            left_on=['game_date', 'TEAM_NAME'],
            right_on=['game_date', 'home_team'],
            how='left'
        )
        df_merged_home['team_spread'] = df_merged_home['home_spread']
        
        df_merged_away = df_merged_away.merge(
            df_lines[['game_date', 'away_team', 'away_spread']],
            left_on=['game_date', 'TEAM_NAME'],
            right_on=['game_date', 'away_team'],
            how='left'
        )
        df_merged_away['team_spread'] = df_merged_away['away_spread']
        
        df_merged = pd.concat([df_merged_home, df_merged_away], ignore_index=True)
    
    # Join shot charts (ALWAYS, for complete dataset)
    if df_shots is not None:
        player_season_points = df_games.groupby('player_normalized').agg({'PTS': 'sum'}).reset_index()
        player_season_points.columns = ['player_normalized', 'total_pts_season']
        
        df_shots = df_shots.merge(player_season_points, on='player_normalized', how='left')
        df_shots['pts_0_6_pct'] = (df_shots['rim_season_points'] / df_shots['total_pts_season'] * 100).fillna(0)
        df_shots['scorer_type'] = df_shots['pts_0_6_pct'].apply(
            lambda x: 'Rim Attacker (≥40.0%)' if x >= 40.0 else 'Perimeter (<40.0%)'
        )
        
        df_merged = df_merged.merge(
            df_shots[['player_normalized', 'scorer_type']],
            on='player_normalized',
            how='left'
        )
    
    # Filter to rows with props only
    df_merged = df_merged[df_merged['points_line'].notna()].copy()
    df_merged['season'] = season
    
    print(f"      ✅ {len(df_merged):,} records with props and actuals")
    
    return df_merged


def load_plays_for_strategy(strategy, strategy_type):
    """
    Load historical plays for a specific strategy across all seasons.
    Uses unified cache (includes scorer_type for both 2D and 3D).
    """
    s3_client = boto3.client('s3')
    dfs = []
    
    for season in SEASONS:
        # Try cache first (unified cache for both 2D and 3D)
        df_cached = load_from_cache(season)
        
        if df_cached is not None:
            # CRITICAL: Validate no Unknown spread_bin values exist
            unknown_spread_count = (df_cached['spread_bin'] == 'Unknown').sum()
            if unknown_spread_count > 0:
                total_plays = len(df_cached)
                unknown_pct = (unknown_spread_count / total_plays) * 100
                unknown_dates = sorted(df_cached[df_cached['spread_bin'] == 'Unknown']['game_date'].unique())
                
                if unknown_pct < 1.0:
                    # Under 1% - acceptable, just filter out and warn
                    print(f"   ⚠️  Found {unknown_spread_count} plays ({unknown_pct:.2f}%) with spread_bin='Unknown' in {season}")
                    print(f"   Dates ({len(unknown_dates)}): {', '.join(unknown_dates)}")
                    print(f"   These plays will be EXCLUDED (under 1% threshold)")
                    
                    # Filter out Unknown plays
                    df_cached = df_cached[df_cached['spread_bin'] != 'Unknown'].copy()
                else:
                    # Over 1% - this is a problem, raise error
                    # Format date display
                    if len(unknown_dates) <= 10:
                        dates_display = ', '.join(unknown_dates)
                    else:
                        first_5 = ', '.join(unknown_dates[:5])
                        last_5 = ', '.join(unknown_dates[-5:])
                        dates_display = f"{first_5} ... {last_5}"
                    
                    error_msg = (
                        f"\n❌ FATAL ERROR: Found {unknown_spread_count} plays ({unknown_pct:.1f}%) with spread_bin='Unknown' in {season} cache!\n"
                        f"   This exceeds the 1% tolerance threshold.\n"
                        f"   Date range: {unknown_dates[0]} to {unknown_dates[-1]} ({len(unknown_dates)} dates)\n"
                        f"   Dates: {dates_display}\n\n"
                        f"   This means game lines are MISSING for these dates.\n\n"
                        f"   FIX:\n"
                        f"   1. Fetch missing game lines:\n"
                        f"      python scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season {season}\n"
                        f"   2. Delete cache to regenerate:\n"
                        f"      rm ~/Downloads/tmp/cache/plays_{season}_*.parquet\n"
                        f"   3. Re-run this script\n"
                    )
                    raise ValueError(error_msg)
            
            # Filter cached data to this strategy
            mask = (
                (df_cached['line_tier'] == strategy['line_tier']) &
                (df_cached['spread_bin'] == strategy['spread_bin'])
            )
            if strategy_type == '3d' and 'scorer_type' in strategy:
                mask = mask & (df_cached['scorer_type'] == strategy['scorer_type'])
            
            df_strat = df_cached[mask].copy()
            
            # Calculate results for this bet_side
            if len(df_strat) > 0:
                bet_side = strategy['bet_side']
                actual_points = df_strat['PTS']
                line = df_strat['points_line']
                
                # Calculate WIN/LOSS/PUSH
                if bet_side == 'OVER':
                    df_strat['result'] = df_strat.apply(
                        lambda row: 'WIN' if pd.notna(row['PTS']) and row['PTS'] > row['points_line']
                        else 'LOSS' if pd.notna(row['PTS']) and row['PTS'] < row['points_line']
                        else 'PUSH' if pd.notna(row['PTS'])
                        else 'NO_DATA',
                        axis=1
                    )
                else:  # UNDER
                    df_strat['result'] = df_strat.apply(
                        lambda row: 'WIN' if pd.notna(row['PTS']) and row['PTS'] < row['points_line']
                        else 'LOSS' if pd.notna(row['PTS']) and row['PTS'] > row['points_line']
                        else 'PUSH' if pd.notna(row['PTS'])
                        else 'NO_DATA',
                        axis=1
                    )
                
                # Add columns for compatibility
                df_strat['bet_side'] = bet_side
                df_strat['player_name'] = df_strat['player_normalized']  # Use normalized name
                df_strat['actual_points'] = df_strat['PTS']
                
                dfs.append(df_strat)
            continue
        
        # Not in cache - load and join from source (ALWAYS includes scorer_type)
        print(f"\n   🔄 Cache miss for {season} - loading from source...")
        df_merged = load_and_join_raw_data(season)
        
        if df_merged is None or df_merged.empty:
            print(f"   ⚠️  No data for {season}")
            continue
        
        # Bin the data
        df_merged['line_tier'] = df_merged['points_line'].apply(bin_points_line)
        df_merged['spread_bin'] = df_merged['team_spread'].apply(bin_team_spread)
        
        # Save to cache for future use (unified cache)
        save_to_cache(df_merged, season)
        
        # Filter to this specific strategy
        mask = (
            (df_merged['line_tier'] == strategy['line_tier']) &
            (df_merged['spread_bin'] == strategy['spread_bin'])
        )
        if strategy_type == '3d' and 'scorer_type' in strategy:
            mask = mask & (df_merged['scorer_type'] == strategy['scorer_type'])
        
        df_strat = df_merged[mask].copy()
        
        # Calculate results for this bet_side
        if len(df_strat) > 0:
            bet_side = strategy['bet_side']
            
            # Calculate WIN/LOSS/PUSH
            if bet_side == 'OVER':
                df_strat['result'] = df_strat.apply(
                    lambda row: 'WIN' if pd.notna(row['PTS']) and row['PTS'] > row['points_line']
                    else 'LOSS' if pd.notna(row['PTS']) and row['PTS'] < row['points_line']
                    else 'PUSH' if pd.notna(row['PTS'])
                    else 'NO_DATA',
                    axis=1
                )
            else:  # UNDER
                df_strat['result'] = df_strat.apply(
                    lambda row: 'WIN' if pd.notna(row['PTS']) and row['PTS'] < row['points_line']
                    else 'LOSS' if pd.notna(row['PTS']) and row['PTS'] > row['points_line']
                    else 'PUSH' if pd.notna(row['PTS'])
                    else 'NO_DATA',
                    axis=1
                )
            
            # Add columns for compatibility
            df_strat['bet_side'] = bet_side
            df_strat['player_name'] = df_strat['player_normalized']  # Use normalized name
            df_strat['actual_points'] = df_strat['PTS']
            
            dfs.append(df_strat)
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Show comprehensive stats for this strategy
    if 'game_date' in combined.columns:
        combined['game_date'] = pd.to_datetime(combined['game_date'])
        min_date = combined['game_date'].min().strftime('%Y-%m-%d')
        max_date = combined['game_date'].max().strftime('%Y-%m-%d')
        
        # Calculate stats
        total_plays = len(combined)
        wins = (combined['result'] == 'WIN').sum()
        losses = (combined['result'] == 'LOSS').sum()
        pushes = (combined['result'] == 'PUSH').sum()
        
        if (wins + losses) > 0:
            win_rate = (wins / (wins + losses) * 100)
        else:
            win_rate = 0.0
        
        print(f"   📊 Strategy Results:")
        print(f"      Date Range: {min_date} to {max_date}")
        print(f"      Record: {wins}W-{losses}L-{pushes}P ({total_plays} total plays)")
        print(f"      Win Rate: {win_rate:.1f}%")
    
    return combined


# =============================================================================
# GENERATE 4-PANEL PLOT
# =============================================================================

def generate_performance_plot(strategy, strategy_type, output_path, recent_plays_n=10):
    """
    Generate 5-panel plot showing strategy performance over time + recent plays table.
    
    Panels:
    1. 2023-24 season: Date vs Win Rate
    2. 2024-25 season: Date vs Win Rate
    3. 2025-26 season: Date vs Win Rate
    4. Overall: All seasons combined
    5. Recent plays table: Last N plays with details
    
    Args:
        strategy: Strategy dict
        strategy_type: '2d' or '3d'
        output_path: Where to save the PNG
        recent_plays_n: Number of recent plays to show in table (default: 10)
    
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
    
    # Create 3 rows x 2 cols layout (6 panels, using 5)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # Build title
    title = f"Strategy Performance: {strategy['line_tier']} | {strategy['spread_bin']} | {strategy['bet_side']}"
    if strategy_type == '3d' and 'scorer_type' in strategy:
        title += f" | {strategy['scorer_type']}"
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Season colors
    season_colors = {
        '2023-24': '#1f77b4',  # Blue
        '2024-25': '#ff7f0e',  # Orange
        '2025-26': '#2ca02c'   # Green
    }
    
    # Panels 1-3: Individual seasons
    for idx, season in enumerate(SEASONS[:3]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
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
    
    # Panel 4: Overall (use play number instead of date)
    ax = fig.add_subplot(gs[1, 1])
    
    # Reset cumulative across all seasons
    df_overall = df.copy()
    df_overall = df_overall.sort_values('game_date')
    df_overall['cumulative_wins'] = df_overall['is_win'].cumsum()
    df_overall['play_number'] = range(1, len(df_overall) + 1)
    df_overall['win_rate'] = (df_overall['cumulative_wins'] / df_overall['play_number'] * 100)
    
    # Plot each season with different colors
    for season in SEASONS:
        df_season_segment = df_overall[df_overall['season'] == season]
        if len(df_season_segment) > 0:
            ax.plot(df_season_segment['play_number'], df_season_segment['win_rate'],
                   color=season_colors.get(season, 'black'), linewidth=2, label=season)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Format
    ax.set_title('Overall (All Seasons)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Play Number', fontsize=11)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add final stats
    final_wr = df_overall['win_rate'].iloc[-1]
    total_plays = len(df_overall)
    total_wins = int(df_overall['cumulative_wins'].iloc[-1])
    total_losses = total_plays - total_wins
    ax.text(0.02, 0.98, f'{total_wins}W-{total_losses}L | {final_wr:.1f}%', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 5: Recent plays table (bottom row, spans both columns)
    ax_table = fig.add_subplot(gs[2, :])  # Row 3, span both columns
    ax_table.axis('off')
    
    # Get most recent N plays (sorted descending - most recent first)
    df_recent = df_overall.tail(recent_plays_n).sort_values('game_date', ascending=False).copy()
    
    # Format table data
    table_data = []
    for _, row in df_recent.iterrows():
        date_str = row['game_date'].strftime('%m/%d/%y')
        player = row.get('player_name', 'Unknown')[:18]  # Truncate long names
        line = f"{row['points_line']:.1f}"
        actual_pts = f"{int(row['actual_points'])}" if pd.notna(row['actual_points']) else 'N/A'
        
        # Calculate Diff and Diff %
        if pd.notna(row['actual_points']) and pd.notna(row['points_line']):
            diff = row['actual_points'] - row['points_line']
            diff_pct = (diff / row['points_line'] * 100) if row['points_line'] != 0 else 0
            diff_str = f"{diff:+.1f}"  # e.g., "+5.0" or "-10.4"
            diff_pct_str = f"{diff_pct:+.1f}%"  # e.g., "+14.1%" or "-29.4%"
        else:
            diff_str = 'N/A'
            diff_pct_str = 'N/A'
        
        # What we bet (from strategy config)
        our_bet = strategy['bet_side']  # e.g. 'UNDER' or 'OVER'
        
        # Determine what actually happened in the market
        if pd.notna(row['actual_points']):
            if row['actual_points'] > row['points_line']:
                actual_result = 'OVER'
            elif row['actual_points'] < row['points_line']:
                actual_result = 'UNDER'
            else:
                actual_result = 'PUSH'
        else:
            actual_result = 'N/A'
        
        # Strategy result (did OUR bet win?)
        strat_result = row['result']
        if strat_result == 'WIN':
            result_text = 'WIN'
        elif strat_result == 'LOSS':
            result_text = 'LOSS'
        else:
            result_text = 'PUSH'
        
        # Get metadata fields
        line_tier = row.get('line_tier', 'Unknown')
        spread_bin = row.get('spread_bin', 'Unknown')
        scorer_type = row.get('scorer_type', '')
        # Shorten scorer_type for display
        if scorer_type == 'Rim Attacker (≥40.0%)':
            scorer_type_display = 'Rim'
        elif scorer_type == 'Perimeter (<40.0%)':
            scorer_type_display = 'Perimeter'
        else:
            scorer_type_display = ''
        
        table_data.append([
            date_str,
            player,
            line,
            actual_pts,
            diff_str,
            diff_pct_str,
            line_tier,
            spread_bin,
            scorer_type_display,
            our_bet,
            actual_result,
            result_text
        ])
    
    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Date', 'Player', 'Line', 'Scored', 'Diff', 'Diff %', 'Line Tier', 'Spread', 'Scorer', 'Our Bet', 'Actual', 'Result'],
        loc='center',
        cellLoc='left',
        colWidths=[0.07, 0.14, 0.05, 0.05, 0.05, 0.06, 0.10, 0.09, 0.08, 0.07, 0.06, 0.10]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Header styling
    for i in range(12):  # Now 12 columns
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors and color-code results
    for i in range(1, len(table_data) + 1):
        for j in range(12):  # Now 12 columns
            cell = table[(i, j)]
            
            # Base row color (alternating)
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Color-code Diff % column (column 5) with gradient
            if j == 5:  # Diff % column
                diff_pct_str = table_data[i-1][5]
                if diff_pct_str != 'N/A':
                    try:
                        # Parse percentage value
                        diff_pct_val = float(diff_pct_str.replace('%', ''))
                        
                        # Normalize to 0-1 range (clamped between -100 and +100)
                        normalized = max(-100, min(100, diff_pct_val)) / 100.0
                        
                        # Generate color gradient
                        if normalized < 0:
                            # Red gradient for negative (under)
                            intensity = abs(normalized)
                            r = 1.0
                            g = 1.0 - (intensity * 0.7)  # Fade from white to red
                            b = 1.0 - (intensity * 0.7)
                        else:
                            # Green gradient for positive (over)
                            intensity = normalized
                            r = 1.0 - (intensity * 0.7)  # Fade from white to green
                            g = 1.0
                            b = 1.0 - (intensity * 0.7)
                        
                        cell.set_facecolor((r, g, b))
                    except ValueError:
                        pass
            
            # Color-code result column (column 11)
            if j == 11:  # Strategy Result column
                result_text = table_data[i-1][11]
                if result_text == 'WIN':
                    cell.set_facecolor('#C6EFCE')  # Light green
                    cell.set_text_props(weight='bold', color='#006100')
                elif result_text == 'LOSS':
                    cell.set_facecolor('#FFC7CE')  # Light red
                    cell.set_text_props(weight='bold', color='#9C0006')
                elif result_text == 'PUSH':
                    cell.set_facecolor('#FFEB9C')  # Light yellow
                    cell.set_text_props(weight='bold', color='#9C5700')
    
    # Title for table panel
    ax_table.set_title(f'Most Recent {recent_plays_n} Plays', fontsize=14, fontweight='bold', pad=20)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


# =============================================================================
# GENERATE V5 YESTERDAY SUMMARY PLOT
# =============================================================================

def generate_v5_yesterday_summary_plot(config, output_path):
    """
    Generate a 2-panel plot for yesterday's plays across all 15 v5 strategies.
    
    Top panel: Cumulative win rate over time (current season only, all v5 strategies)
    Bottom panel: Table of yesterday's plays across all 15 strategies
    
    Args:
        config: Strategy config dict
        output_path: Where to save the PNG
    
    Returns:
        bool: True if successful
    """
    yesterday = get_yesterday_et()
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    print(f"\n📊 Generating V5 Yesterday Summary Plot...")
    print(f"   Yesterday: {yesterday_str}")
    
    # Load cache for current season
    df_cached = load_from_cache('2025-26')
    if df_cached is None:
        print("   ⚠️  No cache found for 2025-26")
        return False
    
    # Collect all v5 strategy plays
    all_v5_plays = []
    
    for v5_strat in config['strategies']:
        strategy_type = v5_strat['strategy_type']
        
        # Filter cache to this strategy
        mask = (
            (df_cached['line_tier'] == v5_strat['line_tier']) &
            (df_cached['spread_bin'] == v5_strat['spread_bin'])
        )
        if strategy_type == '3d' and 'scorer_type' in v5_strat:
            mask = mask & (df_cached['scorer_type'] == v5_strat['scorer_type'])
        
        df_strat = df_cached[mask].copy()
        
        if len(df_strat) > 0:
            # Calculate results for this bet_side
            bet_side = v5_strat['bet_side']
            
            if bet_side == 'OVER':
                df_strat['result'] = df_strat.apply(
                    lambda row: 'WIN' if pd.notna(row['PTS']) and row['PTS'] > row['points_line']
                    else 'LOSS' if pd.notna(row['PTS']) and row['PTS'] < row['points_line']
                    else 'PUSH',
                    axis=1
                )
            else:  # UNDER
                df_strat['result'] = df_strat.apply(
                    lambda row: 'WIN' if pd.notna(row['PTS']) and row['PTS'] < row['points_line']
                    else 'LOSS' if pd.notna(row['PTS']) and row['PTS'] > row['points_line']
                    else 'PUSH',
                    axis=1
                )
            
            df_strat['strategy_name'] = v5_strat['strategy_name']
            df_strat['bet_side'] = bet_side
            all_v5_plays.append(df_strat)
    
    if not all_v5_plays:
        print("   ⚠️  No plays found for v5 strategies")
        return False
    
    # Combine all plays
    df_all = pd.concat(all_v5_plays, ignore_index=True)
    df_all['game_date'] = pd.to_datetime(df_all['game_date'])
    df_all = df_all.sort_values('game_date')
    
    # Filter to yesterday
    df_yesterday = df_all[df_all['game_date'] == yesterday_str].copy()
    
    print(f"   📅 Found {len(df_yesterday)} plays from yesterday")
    
    if len(df_yesterday) == 0:
        print("   ⚠️  No plays yesterday")
        return False
    
    # Create figure with 2 panels (top: win rate chart, bottom: table)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
    
    # Panel 1: Cumulative win rate over time (current season only)
    ax_chart = fig.add_subplot(gs[0])
    
    df_all['is_win'] = (df_all['result'] == 'WIN').astype(int)
    df_all['cumulative_wins'] = df_all['is_win'].cumsum()
    df_all['play_number'] = range(1, len(df_all) + 1)
    df_all['win_rate'] = (df_all['cumulative_wins'] / df_all['play_number'] * 100)
    
    ax_chart.plot(df_all['play_number'], df_all['win_rate'], 
                  color='#2ca02c', linewidth=2.5, label='All V5 Strategies')
    ax_chart.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax_chart.set_title(f"V5 Strategies - Cumulative Win Rate (2025-26 Season)", 
                       fontsize=14, fontweight='bold')
    ax_chart.set_xlabel('Play Number', fontsize=11)
    ax_chart.set_ylabel('Win Rate (%)', fontsize=11)
    ax_chart.set_ylim(0, 100)
    ax_chart.grid(True, alpha=0.3)
    ax_chart.legend(loc='best')
    
    # Add stats
    total_wins = int(df_all['cumulative_wins'].iloc[-1])
    total_plays = len(df_all)
    final_wr = df_all['win_rate'].iloc[-1]
    ax_chart.text(0.02, 0.98, f'{total_wins}W-{total_plays - total_wins}L | {final_wr:.1f}%',
                  transform=ax_chart.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Yesterday's plays table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    # Sort by strategy name for consistent display
    df_yesterday = df_yesterday.sort_values(['strategy_name', 'player_normalized'])
    
    # Format table data
    table_data = []
    for _, row in df_yesterday.iterrows():
        date_str = row['game_date'].strftime('%m/%d/%y')
        player = row.get('player_normalized', 'Unknown')[:18]
        line = f"{row['points_line']:.1f}"
        actual_pts = f"{int(row['PTS'])}" if pd.notna(row['PTS']) else 'N/A'
        
        # Calculate Diff and Diff %
        if pd.notna(row['PTS']) and pd.notna(row['points_line']):
            diff = row['PTS'] - row['points_line']
            diff_pct = (diff / row['points_line'] * 100) if row['points_line'] != 0 else 0
            diff_str = f"{diff:+.1f}"
            diff_pct_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = 'N/A'
            diff_pct_str = 'N/A'
        
        # Strategy metadata
        line_tier = row.get('line_tier', 'Unknown')
        spread_bin = row.get('spread_bin', 'Unknown')
        scorer_type = row.get('scorer_type', '')
        
        # Shorten scorer_type
        if pd.notna(scorer_type) and isinstance(scorer_type, str):
            if 'Rim' in scorer_type:
                scorer_type_display = 'Rim'
            elif 'Perimeter' in scorer_type:
                scorer_type_display = 'Perim'
            else:
                scorer_type_display = ''
        else:
            scorer_type_display = ''
        
        # Bet info
        our_bet = row['bet_side']
        
        # Result
        if row['PTS'] > row['points_line']:
            actual_result = 'OVER'
        elif row['PTS'] < row['points_line']:
            actual_result = 'UNDER'
        else:
            actual_result = 'PUSH'
        
        result_text = row['result']
        
        table_data.append([
            date_str,
            player,
            line,
            actual_pts,
            diff_str,
            diff_pct_str,
            line_tier,
            spread_bin,
            scorer_type_display,
            our_bet,
            actual_result,
            result_text
        ])
    
    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Date', 'Player', 'Line', 'Scored', 'Diff', 'Diff %', 
                   'Line Tier', 'Spread', 'Scorer', 'Our Bet', 'Actual', 'Result'],
        loc='center',
        cellLoc='left',
        colWidths=[0.07, 0.14, 0.05, 0.05, 0.05, 0.06, 0.10, 0.09, 0.08, 0.07, 0.06, 0.10]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Header styling
    for i in range(12):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Row styling
    for i in range(1, len(table_data) + 1):
        for j in range(12):
            cell = table[(i, j)]
            
            # Alternating row colors
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Color-code Diff % column
            if j == 5:
                diff_pct_str = table_data[i-1][5]
                if diff_pct_str != 'N/A':
                    try:
                        diff_pct_val = float(diff_pct_str.replace('%', ''))
                        normalized = max(-100, min(100, diff_pct_val)) / 100.0
                        
                        if normalized < 0:
                            intensity = abs(normalized)
                            r, g, b = 1.0, 1.0 - (intensity * 0.7), 1.0 - (intensity * 0.7)
                        else:
                            intensity = normalized
                            r, g, b = 1.0 - (intensity * 0.7), 1.0, 1.0 - (intensity * 0.7)
                        
                        cell.set_facecolor((r, g, b))
                    except ValueError:
                        pass
            
            # Color-code result column
            if j == 11:
                result_text = table_data[i-1][11]
                if result_text == 'WIN':
                    cell.set_facecolor('#C6EFCE')
                    cell.set_text_props(weight='bold', color='#006100')
                elif result_text == 'LOSS':
                    cell.set_facecolor('#FFC7CE')
                    cell.set_text_props(weight='bold', color='#9C0006')
                elif result_text == 'PUSH':
                    cell.set_facecolor('#FFEB9C')
                    cell.set_text_props(weight='bold', color='#9C5700')
    
    # Title
    wins = (df_yesterday['result'] == 'WIN').sum()
    losses = (df_yesterday['result'] == 'LOSS').sum()
    ax_table.set_title(f"Yesterday's Plays ({yesterday_str}) - {wins}W-{losses}L", 
                       fontsize=14, fontweight='bold', pad=20)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Generated: {output_path}")
    return True


# =============================================================================
# DATA FETCH & VALIDATION
# =============================================================================

def get_yesterday_et():
    """Get yesterday's date in ET timezone."""
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    et_tz = ZoneInfo('America/New_York')
    today_et = datetime.now(et_tz).date()
    yesterday_et = today_et - timedelta(days=1)
    return yesterday_et


def fetch_latest_data():
    """Run fetch script to get latest props, game results, and game lines."""
    import subprocess
    
    print("="*80)
    print("🔄 FETCHING LATEST DATA")
    print("="*80)
    print("Running: python scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season 2025-26")
    print("This fetches: props, game results, AND game lines\n")
    
    # Run the fetch script
    result = subprocess.run(
        ['python3', 'scripts/fetch_nba_player_props.py', '--mode', '2', '--fetch-games', '--s3', '--season', '2025-26'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print(f"❌ Fetch script failed with exit code {result.returncode}")
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        raise RuntimeError("Failed to fetch latest data")
    
    print("✅ Fetch script completed successfully\n")


def validate_data_freshness():
    """Validate all 3 data sources have data up to yesterday."""
    import boto3
    
    yesterday = get_yesterday_et()
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    print("="*80)
    print("✅ VALIDATING DATA FRESHNESS")
    print("="*80)
    print(f"Today (ET): {yesterday + timedelta(days=1)}")
    print(f"Yesterday (ET): {yesterday}")
    print(f"Checking for data up to: {yesterday_str}\n")
    
    s3_client = boto3.client('s3')
    
    # Define data sources to check
    data_sources = {
        'props': {
            'bucket': 'the-odds-api-mt',
            'prefix': f'nba/historical_player_props/2025-26/{yesterday_str}.csv'
        },
        'game_results': {
            'bucket': 'nba-api-mt',
            'prefix': f'player_game_logs/2025-26/{yesterday_str}.csv'
        },
        'game_lines': {
            'bucket': 'the-odds-api-mt',
            'prefix': f'nba/historical_game_lines/2025-26/nba_game_lines_{yesterday_str}.csv'
        }
    }
    
    all_valid = True
    for source_name, config in data_sources.items():
        bucket = config['bucket']
        key = config['prefix']
        
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"✅ {source_name:15s}: s3://{bucket}/{key}")
        except:
            print(f"❌ {source_name:15s}: MISSING - s3://{bucket}/{key}")
            all_valid = False
    
    print()
    
    if not all_valid:
        error_msg = (
            f"\n❌ VALIDATION FAILED: Missing data for {yesterday_str}\n\n"
            f"One or more data sources are missing files for yesterday.\n"
            f"The fetch script should have created these files.\n\n"
            f"Possible reasons:\n"
            f"1. No games on {yesterday_str} (check NBA schedule)\n"
            f"2. Fetch script had errors (check logs above)\n"
            f"3. NBA API delay - game results take 12+ hours after games finish\n\n"
            f"If there were games yesterday, wait and re-run, or check the fetch script logs.\n"
        )
        raise RuntimeError(error_msg)
    
    print(f"✅ All 3 data sources have fresh data up to {yesterday_str}\n")


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
    
    # Step 1: Fetch latest data
    fetch_latest_data()
    
    # Step 2: Validate data freshness
    from datetime import timedelta
    validate_data_freshness()
    
    # Step 3: Load config and generate plots
    config = load_strategy_config()
    if not config:
        print("❌ Failed to load config")
        return
    
    # Step 4: Show yesterday's plays summary for debugging
    yesterday = get_yesterday_et()
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    print("="*80)
    print(f"📅 YESTERDAY'S PLAYS SUMMARY ({yesterday_str})")
    print("="*80)
    
    # Load cache for 2025-26 season
    df_cached = load_from_cache('2025-26')
    if df_cached is not None:
        df_yesterday = df_cached[df_cached['game_date'] == yesterday_str].copy()
        
        if len(df_yesterday) > 0:
            print(f"Total plays: {len(df_yesterday)}")
            print(f"\nBreakdown by strategy filters:")
            
            # Group by line_tier + spread_bin
            for (line_tier, spread_bin), group_df in df_yesterday.groupby(['line_tier', 'spread_bin']):
                print(f"   {line_tier:30s} | {spread_bin:20s}: {len(group_df):2d} plays")
                
                # Show player names for this combo
                players = sorted(group_df['player_normalized'].unique())
                for player in players:
                    player_plays = group_df[group_df['player_normalized'] == player]
                    for _, row in player_plays.iterrows():
                        print(f"      - {player:25s} | Line: {row['points_line']:5.1f} | Scored: {int(row['PTS']):2d} | Spread: {row['team_spread']:+6.1f}")
        else:
            print(f"⚠️  No plays found for {yesterday_str}")
    else:
        print(f"⚠️  No cache found for 2025-26 season")
    
    print()
    
    strategies = config['strategies']
    plots_generated = 0
    plots_failed = 0
    
    # Generate V5 yesterday summary plot FIRST
    print(f"\n{'='*80}")
    print("GENERATING V5 YESTERDAY SUMMARY PLOT")
    print(f"{'='*80}")
    
    yesterday_plot_path = OUTPUT_DIR / f"v5_yesterday_summary_{yesterday_str}.png"
    try:
        success = generate_v5_yesterday_summary_plot(config, str(yesterday_plot_path))
        if success:
            file_size = yesterday_plot_path.stat().st_size / 1024
            print(f"✅ Generated: {yesterday_plot_path.name} ({file_size:.1f} KB)")
        else:
            print(f"⚠️  Skipped (no yesterday plays)")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate individual strategy plots
    print(f"\n{'='*80}")
    print("GENERATING INDIVIDUAL STRATEGY PLOTS")
    print(f"{'='*80}")
    
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
        import sys
        subprocess.run(['open'] + [str(f) for f in sorted(OUTPUT_DIR.glob("*.png"))])
        print(f"✅ Opened {plots_generated} plots in default viewer\n")
        sys.stdout.flush()  # Ensure output is visible


if __name__ == '__main__':
    main()
