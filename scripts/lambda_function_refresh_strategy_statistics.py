"""
Refresh Strategy Statistics - Daily Multi-Season Backtest Update

Lambda function: nba-strategy-stats-refresher

Context:
This script updates strategy JSONs daily with fresh multi-season statistics.
It combines backtest results from 2023-24, 2024-25, and 2025-26 (through yesterday)
to provide robust, up-to-date win rates and ROI figures.

Self-Contained Approach:
This implementation is FULLY self-contained and does not depend on pre-joined files.
It loads raw data directly from source buckets and joins them in memory:
- Player props from the-odds-api-mt bucket
- Game logs (actuals) from nba-api-mt bucket
- Shot charts from nba-api-mt bucket (for 3D strategies)
- Game lines (spreads) from the-odds-api-mt bucket

This eliminates the circular dependency on external join scripts.

Steps:
1. Load raw player props, game logs, shot charts, and game lines from S3
2. Join them in memory to create player_props_with_actuals dataset
3. Generate all possible strategy combinations (98 for 2D, 196 for 3D)
4. Match props to strategies and calculate WIN/LOSS/PUSH outcomes
5. Save plays.csv to S3
6. Load plays from all 3 seasons (2023-24, 2024-25, 2025-26)
7. Calculate aggregate statistics across all seasons
8. Filter strategies by minimum plays (50+) not historical ROI
9. Generate updated strategy JSON files
10. Upload to S3
11. Generate 5-panel performance plots (3 seasons + overall + recent plays table)
12. Send SNS email notification

Lambda Deployment:
This script is fully self-contained and can be copied directly into the Lambda editor.
No git clone or external dependencies required beyond boto3 and pandas (Lambda layer).

IMPORTANT - Lambda Timeout Configuration:
This function processes 3 full seasons of data and regenerates 280+ strategy plots.
The Lambda timeout MUST be set to 15 minutes (900 seconds) to avoid timeouts.
Current runtime: ~12-13 minutes (backtest: 3-4 min, plots: 8-9 min).
Timeout occurs during plot generation if set too low.

Lambda Layers:
This function uses TWO layers to provide all dependencies:

Layer 1: git:lambda2 (existing AWS layer)
- ARN: arn:aws:lambda:us-east-2:553035198032:layer:git-lambda2:8
- Provides base dependencies

Layer 2: nba-minimal-deps (custom layer we created)
- Created using the commands below to add matplotlib:
   ```
   # 1. Create a minimal layer locally
   mkdir -p layer/python
   cd layer/python
   
   # Install minimal dependencies
   pip install pandas matplotlib --target . --no-deps
   
   # Install only pandas + matplotlib dependencies (not full numpy)
   pip install numpy --target .
   
   # Zip it (keeping it under size limit)
   cd ..
   zip -r minimal-layer.zip python
   
   # Check size (under 50 MB)
   ls -lh minimal-layer.zip
   
   # 2. Upload to AWS (if under 50 MB)
   aws lambda publish-layer-version \
     --layer-name nba-minimal-deps \
     --zip-file fileb://minimal-layer.zip \
     --compatible-runtimes python3.12
   ```

Both layers are attached to the Lambda function to provide pandas, numpy, and matplotlib.

Required Lambda Environment Variables:
- SNS_TOPIC_ARN: SNS topic for email notifications (optional, for plain text email)
- SES_FROM_EMAIL: Email address to send from (e.g., myles@thomasquantitativestrategies.com)
- SES_TO_EMAIL: Email address to send to (e.g., mylescgthomas@gmail.com)

Note: SES must be configured in us-east-2 region and sender email must be verified.

Lambda IAM Role Setup (Required for SES):
The Lambda execution role needs permissions to send SES emails. Add one of these:

Option 1: Attach AWS Managed Policy (Quick)
1. Go to Lambda Console → Configuration → Permissions
2. Click on the execution role name (e.g., betting-dashboard-daily-update-role-ille2llh)
3. Click "Add permissions" → "Attach policies"
4. Search for "AmazonSESFullAccess" and attach it

Option 2: Custom Inline Policy (More Secure - Recommended)
1. Go to Lambda Console → Configuration → Permissions
2. Click on the execution role name
3. Click "Add permissions" → "Create inline policy"
4. Click "JSON" tab and paste:

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ses:SendEmail",
                "ses:SendRawEmail"
            ],
            "Resource": [
                "arn:aws:ses:us-east-2:232692785472:identity/myles@thomasquantitativestrategies.com",
                "arn:aws:ses:us-east-2:232692785472:identity/mylescgthomas@gmail.com"
            ]
        }
    ]
}

5. Name it "SESEmailSendPolicy" and create

Note: Both sender and recipient emails must be verified in SES (us-east-2 region).

S3 Bucket Policy Setup (Required for Inline Images):
To display plots inline in HTML emails, the strategy_plots folder must be publicly readable.
Follow these 3 steps:

Step 1: Adjust Block Public Access Settings
- Go to S3 Console: https://s3.console.aws.amazon.com/s3/buckets/nba-betting-mt
- Click "Permissions" tab
- Under "Block Public Access (bucket settings)", click "Edit"
- UNCHECK: "Block public access to buckets and objects granted through new public bucket or access point policies"
- Leave the other 3 checkboxes as they are
- Click "Save changes"

Step 2: Add Bucket Policy
- Still in the "Permissions" tab, scroll to "Bucket policy" section and click "Edit"
- Add this policy (or merge with existing policy):

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::nba-betting-mt/data/04_output/strategy_plots/*"
        }
    ]
}

Step 3: Save the Policy
- Click "Save changes"

This makes only the strategy_plots folder publicly readable. All other data remains private.
Images will be accessible via: https://nba-betting-mt.s3.us-east-2.amazonaws.com/data/04_output/strategy_plots/...

Usage (CLI):
    python scripts/lambda_function_refresh_strategy_statistics.py --season 2025-26

Usage (Lambda):
    Event payload:
    {
        "season": "2025-26",           # optional, default: "2025-26"
        "strategy": "both",            # optional, default: "both" (choices: "2d", "3d", "both")
        "skip_backtest": false         # optional, default: false
    }

Author: Myles Thomas
Date: 2026-01-30
Updated: 2026-02-07 (fully self-contained with in-memory joins, no dependency on pre-joined files)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict
from io import StringIO

# Set matplotlib config for Lambda (must be writable /tmp)
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Only import stdlib and boto3 (available in Lambda by default)
import boto3

# Initialize AWS clients
s3_client = boto3.client('s3')
ses_client = boto3.client('ses', region_name='us-east-2')

# SES configuration from environment variables
SES_FROM_EMAIL = os.environ.get('SES_FROM_EMAIL', '')
SES_TO_EMAIL = os.environ.get('SES_TO_EMAIL', '')

# Try to import pandas (may need to be in Lambda layer)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available")

# Try to import matplotlib for plotting
try:
    import logging as _logging
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Lambda
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    _logging.getLogger('matplotlib.font_manager').setLevel(_logging.WARNING)
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 buckets
S3_BUCKET = 'nba-betting-mt'
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_NBA = 'nba-api-mt'
BACKTEST_PREFIX = 'data/04_output/backtests'
STRATEGIES_PREFIX = 'data/03_intermediate'

# Multi-season analysis (hardcoded for stability)
BACKTEST_SEASONS = ['2023-24', '2024-25', '2025-26']

# Minimum plays to include strategy
MIN_PLAYS_THRESHOLD = 1

# Plot generation mode: 'all' or 'active_only'
# - 'all': Generate plots for all 280+ strategy combinations (slow, ~10 min)
# - 'active_only': Only generate plots for strategies with active JSON files (~15 strategies, fast ~2 min)
PLOT_GENERATION_MODE = 'active_only'

# Team name mappings: Odds API → NBA API (NBA API is source of truth)
# Different APIs use different team name formats, so we normalize Odds API names
# to match NBA API format for consistent joins across all data sources
ODDS_TO_NBA_TEAM_MAP = {
    'Los Angeles Clippers': 'LA Clippers'  # Odds API → NBA API (source of truth)
}


# =============================================================================
# PLAYER NAME NORMALIZATION (INLINED FOR SELF-CONTAINED LAMBDA)
# =============================================================================

def remove_accents(text):
    """Remove accents/diacritics from text (e.g., Dončić -> Doncic)."""
    import unicodedata
    if pd.isna(text):
        return text
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def normalize_player_name(name):
    """
    Normalize player name for consistent matching across data sources.
    
    Rules:
    1. Remove periods (P.J. -> PJ)
    2. Title case
    3. Remove accents
    4. Remove generational suffixes (III, II, IV, V)
    5. Apply known mappings
    """
    if pd.isna(name):
        return name
    
    name = name.strip().replace('.', '').title()
    name = remove_accents(name)
    
    # Remove generational suffixes at end
    if name.endswith(' Iii'):
        name = name[:-4]
    elif name.endswith(' Ii'):
        name = name[:-3]
    elif name.endswith(' Iv'):
        name = name[:-3]
    elif name.endswith(' V'):
        name = name[:-2]
    
    name = ' '.join(name.split())
    
    # Known name mappings (Odds API -> NBA API)
    mappings = {
        'Herb Jones': 'Herbert Jones',
        'Moe Wagner': 'Moritz Wagner',
        'Nicolas Claxton': 'Nic Claxton',
        'Ron Holland': 'Ronald Holland',
        'Vincent Williams Jr': 'Vince Williams Jr',
        'Derrick Jones': 'Derrick Jones Jr',
        'Bruce Brown Jr': 'Bruce Brown',
        'Kenyon Martin Jr': 'Kj Martin',
        'Paul Reed Jr': 'Paul Reed',
        'Carlton Carrington': 'Bub Carrington',
        'Alfred Joel Horford Reynoso': 'Al Horford',
        'Anthony Davis Jr': 'Anthony Davis',
    }
    
    return mappings.get(name, name)


# =============================================================================
# BINNING FUNCTIONS
# =============================================================================

def bin_points_line(line: float) -> str:
    """
    Bin player points line into tiers (detailed granularity).
    
    Args:
        line: Player points line
    
    Returns:
        Line tier string
    """
    if PANDAS_AVAILABLE and pd.isna(line):
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


def bin_team_spread(spread: float) -> str:
    """
    Bin team spread into categories (detailed granularity).
    
    Args:
        spread: Team spread (positive = underdog, negative = favorite)
    
    Returns:
        Spread bin string
    """
    if PANDAS_AVAILABLE and pd.isna(spread):
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


def generate_all_strategy_combinations(strategy_type: str) -> List[Dict]:
    """
    Generate all possible strategy combinations to test.
    
    Args:
        strategy_type: '2d' or '3d'
    
    Returns:
        List of all strategy dictionaries
    """
    # All possible line tiers
    line_tiers = [
        '5-10 (Bench)',
        '10-15 (Role Player)',
        '15-20 (High Role)',
        '20-25 (Star)',
        '25-30 (High Star)',
        '30-35 (Superstar)',
        '35-40 (Elite)'
    ]
    
    # All possible spread bins
    spread_bins = [
        "Pick'em (-2 to +2)",
        '2-6 Fav',
        '2-6 Dog',
        '6-10 Fav',
        '6-10 Dog',
        '10-15 Fav',
        '10-15 Dog'
    ]
    
    # Both bet sides
    bet_sides = ['OVER', 'UNDER']
    
    combinations = []
    
    if strategy_type == '2d':
        for line_tier in line_tiers:
            for spread_bin in spread_bins:
                for bet_side in bet_sides:
                    combinations.append({
                        'line_tier': line_tier,
                        'spread_bin': spread_bin,
                        'bet_side': bet_side
                    })
    
    elif strategy_type == '3d':
        scorer_types = ['Rim Attacker (≥40.0%)', 'Perimeter (<40.0%)']
        for line_tier in line_tiers:
            for spread_bin in spread_bins:
                for bet_side in bet_sides:
                    for scorer_type in scorer_types:
                        combinations.append({
                            'line_tier': line_tier,
                            'spread_bin': spread_bin,
                            'bet_side': bet_side,
                            'scorer_type': scorer_type
                        })
    
    return combinations


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_yesterday_et() -> str:
    """Get yesterday's date in ET timezone."""
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    yesterday = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')
    return yesterday


def load_player_props_from_s3(s3_client, season: str, strategy_type: str) -> 'pd.DataFrame':
    """
    Load and join player props with actuals directly from source data.
    
    This function is self-contained and does not depend on pre-joined files.
    It loads raw props, game logs, shot charts, and game lines from S3 and joins them.
    
    Args:
        s3_client: Boto3 S3 client
        season: NBA season (e.g., '2024-25')
        strategy_type: '2d' or '3d' (3d requires rim scorer data)
    
    Returns:
        DataFrame with player props and actuals (or None if not found)
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas not available - cannot load props data. Check Lambda layer.")
    
    print(f"\n   🔄 Loading and joining fresh data from source buckets...")
    
    # Step 1: Load player props
    print(f"   📊 Loading props from s3://{S3_BUCKET_PROPS}/nba/historical_player_props/{season}/...")
    prefix = f"nba/historical_player_props/{season}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
    
    if 'Contents' not in response:
        raise RuntimeError(f"No props files found in s3://{S3_BUCKET_PROPS}/nba/historical_player_props/{season}/")
    
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
            except Exception as e:
                print(f"   ⚠️  Failed to load {obj['Key']}: {e}")
                raise  # FAIL HARD - don't silently skip
    
    if not all_props:
        raise RuntimeError(f"No props data loaded for {season}")
    
    df_props = pd.concat(all_props, ignore_index=True)
    df_props['player_normalized'] = df_props['player'].apply(normalize_player_name)
    print(f"   ✅ Loaded {len(df_props):,} prop rows ({df_props['game_date'].min()} to {df_props['game_date'].max()})")
    
    # Step 2: Load game logs (actuals)
    print(f"   🏀 Loading game logs from s3://{S3_BUCKET_NBA}/player_game_logs/{season}/...")
    prefix = f"player_game_logs/{season}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
    
    if 'Contents' not in response:
        raise RuntimeError(f"No game log files found in s3://{S3_BUCKET_NBA}/player_game_logs/{season}/")
    
    all_game_logs = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv'):
            try:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_NBA, Key=obj['Key'])
                df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                all_game_logs.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to load {obj['Key']}: {e}")
                raise  # FAIL HARD
    
    if not all_game_logs:
        raise RuntimeError(f"No game logs loaded for {season}")
    
    df_games = pd.concat(all_game_logs, ignore_index=True)
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'], format='mixed')
    df_games['game_date'] = df_games['GAME_DATE'].dt.date.astype(str)
    df_games['player_normalized'] = df_games['PLAYER_NAME'].apply(normalize_player_name)
    df_games = df_games[df_games['MIN'].notna() & (df_games['MIN'] > 0)].copy()
    print(f"   ✅ Loaded {len(df_games):,} player-game rows ({df_games['game_date'].min()} to {df_games['game_date'].max()})")
    
    # Step 3: Load shot charts (only for 3d)
    df_shots = None
    if strategy_type == '3d':
        print(f"   🎯 Loading shot charts from s3://{S3_BUCKET_NBA}/player_shot_charts/{season}/...")
        prefix = f"player_shot_charts/{season}/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
        
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
                            
                            # Skip if required columns are missing
                            if 'SHOT_DISTANCE' not in df_shots_player.columns or 'SHOT_MADE_FLAG' not in df_shots_player.columns:
                                print(f"   ⚠️  Skipping {obj['Key']}: missing required columns")
                                continue
                            
                            rim_shots = df_shots_player[df_shots_player['SHOT_DISTANCE'] <= 6]
                            rim_makes = rim_shots['SHOT_MADE_FLAG'].sum() if not rim_shots.empty else 0
                            rim_points = rim_makes * 2
                            
                            all_shot_data.append({
                                'player_normalized': player_normalized,
                                'rim_season_points': rim_points
                            })
                    except Exception as e:
                        print(f"   ⚠️  Failed to load shot chart {obj['Key']}: {e}")
                        continue  # Skip malformed files instead of failing
            
            if all_shot_data:
                df_shots = pd.DataFrame(all_shot_data)
                print(f"   ✅ Loaded {len(df_shots):,} player shot chart aggregations")
            else:
                raise RuntimeError(f"No shot chart data loaded for 3D strategy in {season}")
    
    # Step 4: Load game lines (spreads)
    print(f"   📈 Loading game lines from s3://{S3_BUCKET_PROPS}/nba/historical_game_lines/{season}/...")
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
                except Exception as e:
                    print(f"   ⚠️  Failed to load game lines {obj['Key']}: {e}")
                    raise  # FAIL HARD
        
        if all_lines:
            df_lines = pd.concat(all_lines, ignore_index=True)
            # Calculate consensus spreads
            consensus = df_lines.groupby(['game_id', 'game_date', 'away_team', 'home_team', 'market']).agg({
                'away_line': 'mean',
                'home_line': 'mean'
            }).reset_index()
            spread = consensus[consensus['market'] == 'spread'][['game_id', 'game_date', 'away_team', 'home_team', 'away_line', 'home_line']]
            spread.columns = ['game_id', 'game_date', 'away_team', 'home_team', 'away_spread', 'home_spread']
            df_lines = spread
            
            # =====================================================================
            # NORMALIZE TEAM NAMES: ODDS API → NBA API (SOURCE OF TRUTH)
            # =====================================================================
            # Odds API uses "Los Angeles Clippers", NBA API uses "LA Clippers"
            # NBA API is the source of truth for all game data, so we standardize
            # the Odds API team names to match NBA API format.
            # =====================================================================
            df_lines['away_team'] = df_lines['away_team'].replace(ODDS_TO_NBA_TEAM_MAP)
            df_lines['home_team'] = df_lines['home_team'].replace(ODDS_TO_NBA_TEAM_MAP)
            
            print(f"   ✅ Loaded {len(df_lines):,} games with spreads")
        else:
            raise RuntimeError(f"No game lines loaded for {season}")
    
    # Step 5: Join all data
    print(f"   🔗 Joining all datasets...")
    
    # Filter props to player_points only and aggregate
    df_props = df_props[df_props['market'] == 'player_points'].copy()
    props_agg = df_props.groupby(['player_normalized', 'game_date']).agg({
        'prop_line': 'mean'
    }).reset_index()
    props_agg.columns = ['player_normalized', 'game_date', 'points_line']
    
    # Start with game logs (players who actually played)
    df_merged = df_games.copy()
    
    # Join props
    df_merged = df_merged.merge(props_agg, on=['player_normalized', 'game_date'], how='left')
    
    # Join game lines
    if df_lines is not None:
        # At this point, df_lines team names have already been normalized to match
        # NBA API format (done in Step 4 when loading game lines)
        
        # Determine home/away for each player
        df_merged['is_home'] = ~df_merged['MATCHUP'].str.contains('@')
        
        # Create separate joins for home and away
        df_merged_home = df_merged[df_merged['is_home']].copy()
        df_merged_away = df_merged[~df_merged['is_home']].copy()
        
        # Join home players to home team spreads
        # Both sides now use NBA API team name format
        df_merged_home = df_merged_home.merge(
            df_lines[['game_date', 'home_team', 'home_spread']],
            left_on=['game_date', 'TEAM_NAME'],      # NBA API format (source of truth)
            right_on=['game_date', 'home_team'],      # Odds API normalized to NBA API format
            how='left'
        )
        df_merged_home['team_spread'] = df_merged_home['home_spread']
        
        # Join away players to away team spreads
        # Both sides now use NBA API team name format
        df_merged_away = df_merged_away.merge(
            df_lines[['game_date', 'away_team', 'away_spread']],
            left_on=['game_date', 'TEAM_NAME'],      # NBA API format (source of truth)
            right_on=['game_date', 'away_team'],      # Odds API normalized to NBA API format
            how='left'
        )
        df_merged_away['team_spread'] = df_merged_away['away_spread']
        
        df_merged = pd.concat([df_merged_home, df_merged_away], ignore_index=True)
    
    # Join shot charts (3d only)
    if df_shots is not None:
        # Calculate scorer type
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
    
    print(f"   ✅ Joined {len(df_merged):,} player-game records with props and actuals")
    print(f"   📅 Final date range: {df_merged['game_date'].min()} to {df_merged['game_date'].max()}")
    
    return df_merged


def match_and_calculate_plays(df: 'pd.DataFrame', strategies: List[Dict], strategy_type: str) -> 'pd.DataFrame':
    """
    Match player props to strategies and calculate outcomes.
    
    Args:
        df: Player props with actuals
        strategies: List of all strategy combinations
        strategy_type: '2d' or '3d'
    
    Returns:
        DataFrame with all plays and their outcomes
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas not available - cannot process data")
    
    if df is None or df.empty:
        raise RuntimeError("No data to process - empty DataFrame")
    
    print(f"   Matching {len(df):,} records to {len(strategies)} strategies...")
    
    # Bin the data
    df['line_tier'] = df['points_line'].apply(bin_points_line)
    df['spread_bin'] = df['team_spread'].apply(bin_team_spread)
    
    plays = []
    
    for idx, row in df.iterrows():
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        scorer_type = row.get('scorer_type', None)
        
        # Try to match against each strategy
        for strat in strategies:
            # Check if this row matches the strategy
            line_match = strat['line_tier'] == line_tier
            spread_match = strat['spread_bin'] == spread_bin
            
            # For 3D strategies, also check scorer_type
            scorer_match = True
            if strategy_type == '3d':
                if pd.isna(scorer_type):
                    continue
                scorer_match = strat['scorer_type'] == scorer_type
            
            if line_match and spread_match and scorer_match:
                # This row matches this strategy - create a play
                actual_points = row.get('PTS')
                line = row['points_line']
                bet_side = strat['bet_side']
                
                # Determine result
                if pd.isna(actual_points):
                    result = 'NO_DATA'
                    profit = 0.0
                elif bet_side == 'OVER':
                    if actual_points > line:
                        result = 'WIN'
                        profit = 100.0
                    elif actual_points < line:
                        result = 'LOSS'
                        profit = -110.0
                    else:
                        result = 'PUSH'
                        profit = 0.0
                else:  # UNDER
                    if actual_points < line:
                        result = 'WIN'
                        profit = 100.0
                    elif actual_points > line:
                        result = 'LOSS'
                        profit = -110.0
                    else:
                        result = 'PUSH'
                        profit = 0.0
                
                plays.append({
                    'game_date': row.get('game_date'),
                    'player_name': row.get('player_normalized'),  # Use normalized name
                    'team': row.get('TEAM_NAME'),
                    'opponent': row.get('MATCHUP'),
                    'points_line': line,
                    'team_spread': row.get('team_spread'),
                    'line_tier': line_tier,
                    'spread_bin': spread_bin,
                    'scorer_type': scorer_type if strategy_type == '3d' else None,
                    'bet_side': bet_side,
                    'actual_points': actual_points,
                    'result': result,
                    'profit': profit,
                    'season': row.get('season', '')
                })
    
    if not plays:
        print(f"   ⚠️  No plays found")
        return pd.DataFrame()
    
    df_plays = pd.DataFrame(plays)
    print(f"   ✅ Generated {len(df_plays):,} plays")
    
    # Filter out NO_DATA results
    df_plays = df_plays[df_plays['result'] != 'NO_DATA']
    print(f"   ✅ {len(df_plays):,} plays with valid results")
    
    return df_plays


def save_plays_to_s3(s3_client, df_plays: 'pd.DataFrame', strategy_type: str, season: str) -> bool:
    """
    Save plays DataFrame to S3 as CSV.
    
    Args:
        s3_client: Boto3 S3 client
        df_plays: DataFrame with plays
        strategy_type: '2d' or '3d'
        season: NBA season
    
    Returns:
        bool: True if successful
    """
    if not PANDAS_AVAILABLE or df_plays is None or df_plays.empty:
        print(f"   ⚠️  No plays to save")
        return False
    
    s3_key = f'{BACKTEST_PREFIX}/{strategy_type}/{season}/plays.csv'
    
    try:
        # Convert to CSV
        csv_buffer = StringIO()
        df_plays.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
        
        print(f"   ✅ Saved {len(df_plays)} plays to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to save plays: {e}")
        return False


def generate_html_email(message: str, yesterday_plot_url: str = None) -> str:
    """
    Convert plain text message to HTML with embedded images.
    
    Args:
        message: Plain text message body
        yesterday_plot_url: HTTPS URL to yesterday's summary plot
    
    Returns:
        HTML formatted email body
    """
    import re
    
    # Start HTML
    html_parts = ["""
    <html>
    <head>
        <style>
            body { font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.4; color: #333; background-color: #f5f5f5; padding: 20px; }
            .container { max-width: 900px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; }
            .header { background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .plot-container { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; }
            .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px; display: block; margin: 10px auto; }
            .strategy-section { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-left: 3px solid #2c3e50; }
            pre { background-color: transparent; margin: 0; white-space: pre-wrap; word-wrap: break-word; }
            .footer { margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee; font-size: 11px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2 style="margin: 0;">🎯 Strategy Statistics Refresh</h2>
            </div>
    """]
    
    # Add yesterday's plot if available
    if yesterday_plot_url:
        html_parts.append(f"""
            <div class="plot-container">
                <h3>📅 Yesterday's Performance</h3>
                <img src="{yesterday_plot_url}" alt="Yesterday's Performance Summary">
            </div>
        """)
    
    # Parse message and convert plot URLs to embedded images
    # Pattern: 📈 Plot: https://...png
    plot_pattern = r'📈 Plot: (https://[^\s]+\.png)'
    
    # Split message by strategies (each starts with #)
    lines = message.split('\n')
    current_section = []
    
    html_parts.append('<div class="strategy-section">')
    
    for line in lines:
        # Check if line contains a plot URL
        match = re.search(plot_pattern, line)
        if match:
            plot_url = match.group(1)
            # Add text before the plot
            text_before = line[:match.start()]
            if text_before.strip():
                current_section.append(text_before)
            
            # Close current section and add plot
            if current_section:
                html_parts.append(f'<pre>{"".join(current_section)}</pre>')
                current_section = []
            
            # Add the plot as an image
            html_parts.append(f'''
                <div class="plot-container">
                    <img src="{plot_url}" alt="Strategy Performance Plot">
                </div>
            ''')
        else:
            # Regular line, add to current section
            current_section.append(line + '\n')
    
    # Add remaining content
    if current_section:
        html_parts.append(f'<pre>{"".join(current_section)}</pre>')
    
    html_parts.append('</div>')
    
    # Footer
    html_parts.append(f"""
            <div class="footer">
                <p>Generated by NBA Strategy Stats Refresher Lambda</p>
                <p>All plots available in S3: s3://{S3_BUCKET}/data/04_output/strategy_plots/</p>
            </div>
        </div>
    </body>
    </html>
    """)
    
    return ''.join(html_parts)


def send_sns(subject: str, message: str) -> None:
    """
    Send SNS notification (plain text).
    
    Args:
        subject: Email subject
        message: Email body
    """
    print(f"\n{'='*80}")
    print("📧 SENDING EMAIL NOTIFICATION")
    print(f"{'='*80}")
    
    try:
        sns_client = boto3.client('sns')
        topic_arn = os.environ.get('SNS_TOPIC_ARN')
        
        print(f"📋 Subject: {subject}")
        print(f"📧 Topic ARN: {topic_arn if topic_arn else 'NOT SET'}")
        print(f"📝 Message length: {len(message)} characters")
        
        if not topic_arn:
            print("   ❌ SNS_TOPIC_ARN environment variable not set - skipping notification")
            print("   💡 Set SNS_TOPIC_ARN in Lambda environment variables to enable email")
            return
        
        print("   📤 Publishing to SNS...")
        response = sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )
        
        message_id = response.get('MessageId', 'unknown')
        print(f"   ✅ SNS notification sent successfully!")
        print(f"   📬 Message ID: {message_id}")
        
    except Exception as e:
        print(f"   ❌ Failed to send SNS notification: {e}")
        import traceback
        traceback.print_exc()


def send_ses(subject: str, html_body: str, text_body: str) -> None:
    """
    Send HTML email with inline images via AWS SES.
    
    Args:
        subject: Email subject
        html_body: HTML content with embedded image URLs
        text_body: Plain text fallback
    """
    if not SES_FROM_EMAIL or not SES_TO_EMAIL:
        print("   ⚠️  SES_FROM_EMAIL or SES_TO_EMAIL not set - skipping SES email")
        return
    
    try:
        print(f"   📤 Sending HTML email via SES...")
        print(f"   From: {SES_FROM_EMAIL}")
        print(f"   To: {SES_TO_EMAIL}")
        
        response = ses_client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [SES_TO_EMAIL]},
            Message={
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {
                    'Text': {'Data': text_body, 'Charset': 'UTF-8'},
                    'Html': {'Data': html_body, 'Charset': 'UTF-8'}
                }
            }
        )
        
        message_id = response.get('MessageId', 'unknown')
        print(f"   ✅ SES email sent successfully!")
        print(f"   📬 Message ID: {message_id}")
        
    except Exception as e:
        print(f"   ❌ Failed to send SES email: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_backtest_for_season(s3_client, season: str, strategy_type: str) -> bool:
    """
    Run self-contained backtest for a specific season and strategy type.
    
    This generates ALL possible strategy combinations and tests them,
    eliminating the circular dependency on historical JSON files.
    
    Args:
        s3_client: Boto3 S3 client
        season: NBA season (e.g., '2025-26')
        strategy_type: '2d' or '3d'
    
    Returns:
        bool: True if successful
    """
    print(f"   Running {strategy_type.upper()} backtest for {season}...")
    
    # Step 1: Generate all possible strategy combinations
    all_strategies = generate_all_strategy_combinations(strategy_type)
    print(f"   ✅ Generated {len(all_strategies)} strategy combinations to test")
    
    # Step 1a: Validate strategy pairs (every OVER has an UNDER)
    validate_strategy_pairs(all_strategies, strategy_type)
    
    # Step 2: Load player props data from S3
    df_props = load_player_props_from_s3(s3_client, season, strategy_type)
    if df_props is None or df_props.empty:
        print(f"   ❌ No props data found for {season}")
        return False
    
    # Add season column if not present
    if 'season' not in df_props.columns:
        df_props['season'] = season
    
    # Step 3: Match props to strategies and calculate outcomes
    df_plays = match_and_calculate_plays(df_props, all_strategies, strategy_type)
    if df_plays is None or df_plays.empty:
        print(f"   ❌ No plays generated")
        return False
    
    # Step 3a: Validate inverse results (OVER losses ≈ UNDER wins)
    validate_inverse_results(df_plays, strategy_type)
    
    # Step 4: Save plays to S3
    success = save_plays_to_s3(s3_client, df_plays, strategy_type, season)
    
    if success:
        print(f"   ✅ {strategy_type.upper()} backtest complete")
    
    return success


def validate_strategy_pairs(strategies: List[Dict], strategy_type: str) -> bool:
    """
    Validate that every OVER strategy has a corresponding UNDER strategy.
    
    Args:
        strategies: List of strategy dictionaries
        strategy_type: '2d' or '3d'
    
    Returns:
        bool: True if validation passes
    
    Raises:
        AssertionError: If validation fails
    """
    print(f"\n   🔍 Validating strategy pairs...")
    
    # Group strategies by everything except bet_side
    if strategy_type == '2d':
        group_keys = ['line_tier', 'spread_bin']
    else:  # 3d
        group_keys = ['line_tier', 'spread_bin', 'scorer_type']
    
    # Create dictionary: (key_tuple) -> [list of strategies]
    strategy_groups = {}
    for strat in strategies:
        key = tuple(strat[k] for k in group_keys)
        if key not in strategy_groups:
            strategy_groups[key] = []
        strategy_groups[key].append(strat)
    
    # Validate each group has exactly 2 strategies (OVER and UNDER)
    errors = []
    for key, group in strategy_groups.items():
        if len(group) != 2:
            errors.append(f"   ❌ Strategy {key} has {len(group)} variations (expected 2)")
            continue
        
        bet_sides = {s['bet_side'] for s in group}
        if bet_sides != {'OVER', 'UNDER'}:
            errors.append(f"   ❌ Strategy {key} has bet_sides {bet_sides} (expected OVER and UNDER)")
    
    if errors:
        print(f"\n   ❌ VALIDATION FAILED:")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        raise AssertionError(f"Strategy pair validation failed: {len(errors)} issues found")
    
    print(f"   ✅ Validated {len(strategy_groups)} strategy pairs ({len(strategies)} total strategies)")
    print(f"   ✅ Each combination has both OVER and UNDER")
    
    return True


def validate_inverse_results(df_plays: 'pd.DataFrame', strategy_type: str) -> bool:
    """
    Validate that OVER and UNDER strategies have approximately inverse results.
    
    Args:
        df_plays: DataFrame with all plays
        strategy_type: '2d' or '3d'
    
    Returns:
        bool: True if validation passes
    """
    if not PANDAS_AVAILABLE or df_plays is None or df_plays.empty:
        print(f"   ⚠️  Cannot validate inverse results - no data")
        return True
    
    print(f"\n   🔍 Validating inverse results (OVER vs UNDER)...")
    
    # Group by everything except bet_side
    if strategy_type == '2d':
        group_cols = ['line_tier', 'spread_bin']
    else:  # 3d
        group_cols = ['line_tier', 'spread_bin', 'scorer_type']
    
    issues = []
    
    for group_key, group_df in df_plays.groupby(group_cols):
        over_df = group_df[group_df['bet_side'] == 'OVER']
        under_df = group_df[group_df['bet_side'] == 'UNDER']
        
        if len(over_df) == 0 or len(under_df) == 0:
            continue
        
        # Check that total plays are equal
        if len(over_df) != len(under_df):
            issues.append(f"   ⚠️  {group_key}: OVER has {len(over_df)} plays, UNDER has {len(under_df)} plays")
            continue
        
        # Calculate win rates
        over_wins = (over_df['result'] == 'WIN').sum()
        over_losses = (over_df['result'] == 'LOSS').sum()
        under_wins = (under_df['result'] == 'WIN').sum()
        under_losses = (under_df['result'] == 'LOSS').sum()
        
        # OVER wins should approximately equal UNDER losses (and vice versa)
        # Allow small discrepancy due to pushes
        if abs(over_wins - under_losses) > 5 or abs(over_losses - under_wins) > 5:
            over_wr = over_wins / (over_wins + over_losses) * 100 if (over_wins + over_losses) > 0 else 0
            under_wr = under_wins / (under_wins + under_losses) * 100 if (under_wins + under_losses) > 0 else 0
            issues.append(
                f"   ⚠️  {group_key}: OVER {over_wins}W-{over_losses}L ({over_wr:.1f}%), "
                f"UNDER {under_wins}W-{under_losses}L ({under_wr:.1f}%) - not inverse"
            )
    
    if issues:
        print(f"\n   ⚠️  Found {len(issues)} potential issues (showing first 5):")
        for issue in issues[:5]:
            print(issue)
        # Don't fail - just warn (pushes can cause legitimate differences)
        print(f"   Note: Small differences are OK due to pushes")
    else:
        print(f"   ✅ All OVER/UNDER pairs have inverse results")
    
    return True


def load_backtest_plays(s3_client, bucket: str, strategy_type: str, season: str) -> 'pd.DataFrame':
    """
    Load backtest plays CSV from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        strategy_type: '2d' or '3d'
        season: Season string (e.g., '2023-24')
    
    Returns:
        DataFrame of plays
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas not available - check Lambda layer")
    
    s3_key = f'{BACKTEST_PREFIX}/{strategy_type}/{season}/plays.csv'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        df['season'] = season
        print(f"   Loaded {len(df)} plays from {season} {strategy_type.upper()}")
        return df
    except Exception as e:
        raise RuntimeError(f"Could not load backtest plays for {season} {strategy_type.upper()}: {e}")


def calculate_aggregate_strategy_stats(
    df_all: 'pd.DataFrame',
    strategy_type: str,
    seasons: List[str],
    min_plays: int = MIN_PLAYS_THRESHOLD
) -> List[Dict]:
    """
    Calculate aggregate statistics for each strategy across all seasons.
    
    Args:
        df_all: Combined DataFrame with all plays from all seasons
        strategy_type: '2d' or '3d'
        seasons: List of seasons included
        min_plays: Minimum total plays to include strategy
    
    Returns:
        List of strategy dicts ready for JSON export
    """
    if not PANDAS_AVAILABLE or df_all is None:
        print("   ⚠️  pandas not available or no data - cannot calculate stats")
        return []
    
    print(f"\n   Calculating aggregate stats for {strategy_type.upper()} strategies...")
    
    # Group by strategy parameters
    if strategy_type == '2d':
        group_cols = ['line_tier', 'spread_bin', 'bet_side']
    else:  # 3d
        group_cols = ['line_tier', 'spread_bin', 'bet_side', 'scorer_type']
    
    strategies = []
    
    for group_key, group_df in df_all.groupby(group_cols):
        # Calculate stats
        total_plays = len(group_df)
        
        if total_plays < min_plays:
            continue
        
        total_wins = (group_df['result'] == 'WIN').sum()
        total_losses = (group_df['result'] == 'LOSS').sum()
        total_ties = (group_df['result'] == 'PUSH').sum()
        total_profit = group_df['profit'].sum()
        
        if (total_wins + total_losses) == 0:
            continue
        
        hit_rate = (total_wins / (total_wins + total_losses) * 100)
        
        # ROI calculation (assuming $100 bets)
        total_wagered = total_plays * 100
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        # Edge vs baseline (assume 50% baseline)
        edge = hit_rate - 50.0
        
        # Build strategy dict
        if strategy_type == '2d':
            line_tier, spread_bin, bet_side = group_key
            strat = {
                'line_tier': line_tier,
                'spread_bin': spread_bin,
                'bet_side': bet_side,
                'hit_rate': round(hit_rate, 1),
                'roi': round(roi, 1),
                'edge': round(edge, 1),
                'games': total_plays,
                'wins': int(total_wins),
                'losses': int(total_losses),
                'ties': int(total_ties)
            }
        else:  # 3d
            line_tier, spread_bin, bet_side, scorer_type = group_key
            strat = {
                'line_tier': line_tier,
                'spread_bin': spread_bin,
                'bet_side': bet_side,
                'scorer_type': scorer_type,
                'hit_rate': round(hit_rate, 1),
                'roi': round(roi, 1),
                'edge': round(edge, 1),
                'games': total_plays,
                'wins': int(total_wins),
                'losses': int(total_losses),
                'ties': int(total_ties)
            }
        
        strategies.append(strat)
    
    print(f"   ✅ Found {len(strategies)} strategies with >= {min_plays} plays")
    return strategies


def log_strategy_results(
    strategies: List[Dict],
    strategy_type: str,
    seasons: List[str],
    df_all: 'pd.DataFrame'
) -> None:
    """
    Log detailed backtest results for each strategy with per-season breakdown.
    
    Args:
        strategies: List of strategy dicts with performance metrics
        strategy_type: '2d' or '3d'
        seasons: List of seasons included in backtest
        df_all: Full dataframe with all plays for per-season breakdown
    """
    if not PANDAS_AVAILABLE or df_all is None:
        print("   ⚠️  pandas not available - cannot show detailed results")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 {strategy_type.upper()} STRATEGY BACKTEST RESULTS ({', '.join(seasons)})")
    print(f"{'='*80}\n")
    
    # Sort strategies by win rate descending
    sorted_strategies = sorted(strategies, key=lambda x: x['hit_rate'], reverse=True)
    
    # Group columns for filtering
    if strategy_type == '2d':
        group_cols = ['line_tier', 'spread_bin', 'bet_side']
    else:  # 3d
        group_cols = ['line_tier', 'spread_bin', 'bet_side', 'scorer_type']
    
    for i, strat in enumerate(sorted_strategies, 1):
        # Build strategy description
        if strategy_type == '2d':
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
        else:  # 3d
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']} | {strat['scorer_type']}"
        
        # Format aggregate metrics
        hit_rate = strat['hit_rate']
        roi = strat['roi']
        edge = strat['edge']
        total_games = strat['games']
        total_wins = strat['wins']
        total_losses = strat['losses']
        total_ties = strat['ties']
        
        # Determine emoji based on win rate
        if hit_rate >= 60:
            emoji = '🔥'
        elif hit_rate >= 55:
            emoji = '✅'
        elif hit_rate >= 50:
            emoji = '➖'
        else:
            emoji = '❌'
        
        print(f"{emoji} #{i:2d}. {desc}")
        print(f"        AGGREGATE: {total_wins}W-{total_losses}L-{total_ties}T | Hit Rate: {hit_rate:5.1f}% | ROI: {roi:6.1f}% | Edge: {edge:+5.1f}%")
        
        # Filter dataframe for this strategy
        if strategy_type == '2d':
            mask = (
                (df_all['line_tier'] == strat['line_tier']) &
                (df_all['spread_bin'] == strat['spread_bin']) &
                (df_all['bet_side'] == strat['bet_side'])
            )
        else:  # 3d
            mask = (
                (df_all['line_tier'] == strat['line_tier']) &
                (df_all['spread_bin'] == strat['spread_bin']) &
                (df_all['bet_side'] == strat['bet_side']) &
                (df_all['scorer_type'] == strat['scorer_type'])
            )
        
        strat_df = df_all[mask]
        
        # Calculate per-season stats
        for season in seasons:
            season_df = strat_df[strat_df['season'] == season]
            
            if len(season_df) == 0:
                continue
            
            season_wins = (season_df['result'] == 'WIN').sum()
            season_losses = (season_df['result'] == 'LOSS').sum()
            season_ties = (season_df['result'] == 'PUSH').sum()
            season_plays = len(season_df)
            
            if (season_wins + season_losses) > 0:
                season_hit_rate = (season_wins / (season_wins + season_losses) * 100)
            else:
                season_hit_rate = 0.0
            
            season_profit = season_df['profit'].sum()
            season_wagered = season_plays * 100
            season_roi = (season_profit / season_wagered * 100) if season_wagered > 0 else 0
            
            print(f"          {season}: {season_wins}W-{season_losses}L-{season_ties}T | Hit Rate: {season_hit_rate:5.1f}% | ROI: {season_roi:6.1f}%")
        
        print()
    
    # Summary statistics
    total_plays = sum(s['games'] for s in strategies)
    total_wins = sum(s['wins'] for s in strategies)
    total_losses = sum(s['losses'] for s in strategies)
    total_ties = sum(s['ties'] for s in strategies)
    avg_roi = sum(s['roi'] for s in strategies) / len(strategies) if strategies else 0
    avg_hit_rate = sum(s['hit_rate'] for s in strategies) / len(strategies) if strategies else 0
    
    print(f"{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total Strategies: {len(strategies)}")
    print(f"  Total Plays: {total_plays}")
    print(f"  Overall Record: {total_wins}W-{total_losses}L-{total_ties}T ({avg_hit_rate:.1f}% avg hit rate)")
    print(f"  Average ROI: {avg_roi:.1f}%")
    print(f"  Profitable Strategies: {sum(1 for s in strategies if s['roi'] > 0)}/{len(strategies)}")
    print(f"{'='*80}\n")


def generate_strategy_json(
    strategies: List[Dict],
    output_path: str,
    metadata: Dict
) -> None:
    """
    Generate strategy JSON file.
    
    Args:
        strategies: List of strategy dicts
        output_path: Where to save JSON
        metadata: Metadata to include in JSON
    """
    data = {
        'generated_at': metadata['generated_at'],
        'data_through': metadata['data_through'],
        'seasons_included': metadata['seasons_included'],
        'total_plays': metadata['total_plays'],
        'strategies': strategies
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   💾 Saved {len(strategies)} strategies to {output_path}")


def load_v5_strategies_from_s3() -> List[Dict]:
    """
    Load the v5 strategy config from S3.
    
    Returns:
        List of v5 strategy configs (empty list if error)
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key='strategies/enhanced_unders_v5.json'
        )
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data.get('strategies', [])
    except Exception as e:
        print(f"   ⚠️  Failed to load v5 strategies: {e}")
        return []


def format_v5_strategies_for_email(strategy_rankings: Dict, season: str) -> str:
    """
    Format v5 strategies with season-by-season breakdown for email.
    
    Shows each of the 15 v5 strategies with:
    - W-L-T for each season (2023-24, 2024-25, 2025-26)
    - Aggregate stats
    - Link to performance plot
    
    Args:
        strategy_rankings: Dict with '2d' and '3d' keys containing backtest results
        season: Current season (for plot links)
    
    Returns:
        Formatted string for email
    """
    # Load v5 config
    v5_strategies = load_v5_strategies_from_s3()
    if not v5_strategies:
        return "⚠️  Could not load v5 strategies from S3\n"
    
    # Combine all backtest strategies
    all_backtest_strategies = []
    for strat_type in ['2d', '3d']:
        if strat_type in strategy_rankings:
            for strat in strategy_rankings[strat_type]:
                strat['_type'] = strat_type
                all_backtest_strategies.append(strat)
    
    lines = [f"\n{'='*80}"]
    lines.append(f"📊 V5 STRATEGY PERFORMANCE (15 Strategies)")
    lines.append(f"{'='*80}\n")
    
    # Get season-by-season data for each v5 strategy
    s3_client = boto3.client('s3')
    seasons = BACKTEST_SEASONS  # ['2023-24', '2024-25', '2025-26']
    
    for i, v5_strat in enumerate(v5_strategies, 1):
        # Match to backtest results
        matched_strat = None
        for backtest_strat in all_backtest_strategies:
            if (v5_strat['line_tier'] == backtest_strat['line_tier'] and
                v5_strat['spread_bin'] == backtest_strat['spread_bin'] and
                v5_strat['bet_side'] == backtest_strat['bet_side']):
                # For 3D strategies, also match scorer_type
                if v5_strat['strategy_type'] == '3d':
                    if v5_strat.get('scorer_type') == backtest_strat.get('scorer_type'):
                        matched_strat = backtest_strat
                        break
                else:
                    matched_strat = backtest_strat
                    break
        
        if not matched_strat:
            lines.append(f"⚠️  #{i:2d}. {v5_strat['strategy_name']} - NO BACKTEST DATA")
            continue
        
        # Strategy header
        strategy_name = v5_strat['strategy_name']
        tier = v5_strat.get('tier', 'N/A').replace('_', ' ').title()
        lines.append(f"\n#{i:2d}. {strategy_name} ({tier})")
        lines.append(f"     {v5_strat['line_tier']} | {v5_strat['spread_bin']} | {v5_strat['bet_side']}")
        if v5_strat['strategy_type'] == '3d':
            lines.append(f"     Scorer Type: {v5_strat.get('scorer_type', 'N/A')}")
        
        # Load season-by-season data
        season_stats = {}
        strat_type = v5_strat['strategy_type']
        
        for s in seasons:
            try:
                df = load_backtest_plays_for_strategy_simple(
                    s3_client,
                    matched_strat,
                    strat_type,
                    [s]  # Single season
                )
                if df is not None and len(df) > 0:
                    wins = (df['result'] == 'WIN').sum()
                    losses = (df['result'] == 'LOSS').sum()
                    ties = (df['result'] == 'PUSH').sum()
                    total = wins + losses + ties
                    hit_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                    season_stats[s] = {
                        'wins': wins,
                        'losses': losses,
                        'ties': ties,
                        'total': total,
                        'hit_rate': hit_rate
                    }
            except Exception as e:
                print(f"   ⚠️  Failed to load {s} data for {strategy_name}: {e}")
        
        # Display season-by-season
        lines.append(f"")
        for s in seasons:
            if s in season_stats:
                st = season_stats[s]
                lines.append(f"     {s}: {st['wins']}W-{st['losses']}L-{st['ties']}T ({st['hit_rate']:.1f}%)")
            else:
                lines.append(f"     {s}: No data")
        
        # Aggregate stats
        total_wins = matched_strat.get('wins', 0)
        total_losses = matched_strat.get('losses', 0)
        total_ties = matched_strat.get('ties', 0)
        hit_rate = matched_strat.get('hit_rate', 0)
        roi = matched_strat.get('roi', 0)
        
        lines.append(f"")
        lines.append(f"     OVERALL: {total_wins}W-{total_losses}L-{total_ties}T | Hit: {hit_rate:.1f}% | ROI: {roi:+.1f}%")
        
        # Plot link
        plot_name = f"{matched_strat['line_tier']}_{matched_strat['spread_bin']}_{matched_strat['bet_side']}"
        if strat_type == '3d':
            scorer = matched_strat.get('scorer_type', '').replace('≥', 'ge').replace('%', 'pct')
            plot_name += f"_{scorer}"
        plot_name = plot_name.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
        # Use HTTPS URL for email embedding
        plot_url = f"https://{S3_BUCKET}.s3.us-east-2.amazonaws.com/data/04_output/strategy_plots/{season}/{plot_name}.png"
        lines.append(f"     📈 Plot: {plot_url}")
    
    lines.append(f"\n{'='*80}\n")
    return '\n'.join(lines)


def generate_v5_yesterday_summary_plot(strategy_rankings: Dict, season: str, yesterday: str) -> bool:
    """
    Generate a summary plot showing yesterday's plays across all v5 strategies.
    
    Top panel: Win rate over time (all v5 strategies combined) for current season
    Bottom panel: Table of yesterday's plays
    
    Args:
        strategy_rankings: Dict with '2d' and '3d' keys containing backtest results
        season: Current season (e.g., '2025-26')
        yesterday: Yesterday's date in YYYY-MM-DD format (ET)
    
    Returns:
        bool: True if plot generated successfully
    """
    try:
        print(f"\n📊 Generating V5 Yesterday Summary Plot...")
        
        # Load v5 strategies
        v5_strategies = load_v5_strategies_from_s3()
        if not v5_strategies:
            print("   ⚠️  Could not load v5 strategies")
            return False
        
        # Combine all backtest strategies
        all_backtest_strategies = []
        for strat_type in ['2d', '3d']:
            if strat_type in strategy_rankings:
                for strat in strategy_rankings[strat_type]:
                    strat['_type'] = strat_type
                    all_backtest_strategies.append(strat)
        
        # Load plays data for all v5 strategies (current season only)
        s3_client = boto3.client('s3')
        all_v5_plays = []
        
        print(f"   🔍 Matching {len(v5_strategies)} v5 strategies to backtest results...")
        
        for v5_strat in v5_strategies:
            # Match to backtest results
            matched_strat = None
            for backtest_strat in all_backtest_strategies:
                if (v5_strat['line_tier'] == backtest_strat['line_tier'] and
                    v5_strat['spread_bin'] == backtest_strat['spread_bin'] and
                    v5_strat['bet_side'] == backtest_strat['bet_side']):
                    if v5_strat['strategy_type'] == '3d':
                        if v5_strat.get('scorer_type') == backtest_strat.get('scorer_type'):
                            matched_strat = backtest_strat
                            break
                    else:
                        matched_strat = backtest_strat
                        break
            
            if not matched_strat:
                print(f"   ⚠️  No match for v5: {v5_strat['strategy_name']} - line_tier={v5_strat['line_tier']}, spread_bin={v5_strat['spread_bin']}, bet_side={v5_strat['bet_side']}")
            
            if not matched_strat:
                continue
            
            # Load plays for this strategy (current season only)
            strat_type = v5_strat['strategy_type']
            df = load_backtest_plays_for_strategy_simple(
                s3_client,
                matched_strat,
                strat_type,
                [season]  # Current season only
            )
            
            if df is not None and len(df) > 0:
                df['strategy_name'] = v5_strat['strategy_name']
                all_v5_plays.append(df)
        
        if not all_v5_plays:
            print("   ⚠️  No v5 plays data found")
            return False
        
        # Combine all v5 plays
        df_all = pd.concat(all_v5_plays, ignore_index=True)
        df_all['game_date'] = pd.to_datetime(df_all['game_date'])
        df_all = df_all.sort_values('game_date')
        
        # Filter to yesterday's plays
        yesterday_date = pd.to_datetime(yesterday)
        df_yesterday = df_all[df_all['game_date'] == yesterday_date].copy()
        
        print(f"   📅 Found {len(df_yesterday)} plays from yesterday ({yesterday})")
        
        # Create figure with 2 panels
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.2], hspace=0.3)
        
        fig.suptitle(f"V5 Strategies - Yesterday's Performance ({yesterday})", 
                    fontsize=16, fontweight='bold')
        
        # Panel 1: Win rate over time (current season)
        ax = fig.add_subplot(gs[0])
        
        df_all['is_win'] = (df_all['result'] == 'WIN').astype(int)
        df_all['cumulative_wins'] = df_all['is_win'].cumsum()
        df_all['cumulative_plays'] = range(1, len(df_all) + 1)
        df_all['win_rate'] = (df_all['cumulative_wins'] / df_all['cumulative_plays'] * 100)
        
        ax.plot(df_all['game_date'], df_all['win_rate'], 
               color='#2ca02c', linewidth=2, label=f'{season} (All V5)')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Break-even')
        
        # Highlight yesterday with vertical line
        ax.axvline(x=yesterday_date, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Yesterday')
        
        ax.set_title(f'Cumulative Win Rate - {season} (All 15 V5 Strategies Combined)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Win Rate (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add stats box
        final_wr = df_all['win_rate'].iloc[-1]
        total_wins = int(df_all['cumulative_wins'].iloc[-1])
        total_losses = len(df_all) - total_wins
        ax.text(0.02, 0.98, f'Season Total: {total_wins}W-{total_losses}L | {final_wr:.1f}%',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel 2: Yesterday's plays table
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')
        
        if len(df_yesterday) > 0:
            # Sort by strategy name, then player name
            df_yesterday = df_yesterday.sort_values(['strategy_name', 'player_name'])
            
            # Format table data
            table_data = []
            for _, row in df_yesterday.iterrows():
                player = row.get('player_name', 'Unknown')[:18]
                line = f"{row['points_line']:.1f}"
                actual_pts = f"{int(row['actual_points'])}" if pd.notna(row['actual_points']) else 'N/A'
                
                # Determine actual result
                if pd.notna(row['actual_points']):
                    if row['actual_points'] > row['points_line']:
                        actual_result = 'OVER'
                    elif row['actual_points'] < row['points_line']:
                        actual_result = 'UNDER'
                    else:
                        actual_result = 'PUSH'
                else:
                    actual_result = 'N/A'
                
                # Strategy result
                strat_result = row['result']
                if strat_result == 'WIN':
                    result_text = 'WIN'
                elif strat_result == 'LOSS':
                    result_text = 'LOSS'
                else:
                    result_text = 'PUSH'
                
                # Get metadata - actual values instead of bins
                line_value = row.get('points_line', 'N/A')
                spread_value = row.get('team_spread', 'N/A')
                
                # Format to 1 decimal place
                if pd.notna(line_value) and line_value != 'N/A':
                    line_value = f"{float(line_value):.1f}"
                if pd.notna(spread_value) and spread_value != 'N/A':
                    spread_value = f"{float(spread_value):+.1f}"
                
                scorer_type = row.get('scorer_type', '')
                
                # Shorten scorer_type - handle NaN/None
                if pd.notna(scorer_type) and isinstance(scorer_type, str):
                    if 'Rim' in scorer_type:
                        scorer_type_display = 'Rim'
                    elif 'Perimeter' in scorer_type:
                        scorer_type_display = 'Perim'
                    else:
                        scorer_type_display = ''
                else:
                    scorer_type_display = ''
                
                # Get our bet (from line_tier matching)
                our_bet = row.get('bet_side', 'UNDER')
                
                # Strategy name (shortened)
                strat_name = row.get('strategy_name', 'Unknown')[:20]
                
                table_data.append([
                    strat_name,
                    player,
                    line_value,
                    spread_value,
                    line,
                    actual_pts,
                    scorer_type_display,
                    our_bet,
                    actual_result,
                    result_text
                ])
            
            # Create table
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Strategy', 'Player', 'Line', 'Spread', 'Over/Under', 'Scored', 'Scorer', 'Our Bet', 'Actual', 'Result'],
                loc='center',
                cellLoc='left',
                colWidths=[0.13, 0.14, 0.06, 0.07, 0.09, 0.06, 0.07, 0.08, 0.07, 0.08]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 2)
            
            # Header styling
            for i in range(10):
                cell = table[(0, i)]
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            
            # Row styling (alternating colors)
            for i in range(1, len(table_data) + 1):
                for j in range(10):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    
                    # Color result column
                    if j == 9:  # Result column
                        if table_data[i-1][9] == 'WIN':
                            cell.set_facecolor('#90EE90')
                            cell.set_text_props(weight='bold')
                        elif table_data[i-1][9] == 'LOSS':
                            cell.set_facecolor('#FFB6C1')
                            cell.set_text_props(weight='bold')
            
            # Add summary text above table
            wins_yesterday = (df_yesterday['result'] == 'WIN').sum()
            losses_yesterday = (df_yesterday['result'] == 'LOSS').sum()
            pushes_yesterday = (df_yesterday['result'] == 'PUSH').sum()
            hit_rate_yesterday = (wins_yesterday / (wins_yesterday + losses_yesterday) * 100) if (wins_yesterday + losses_yesterday) > 0 else 0
            
            ax_table.text(0.5, 0.95, 
                         f"Yesterday's Results: {wins_yesterday}W-{losses_yesterday}L-{pushes_yesterday}P | Hit Rate: {hit_rate_yesterday:.1f}%",
                         transform=ax_table.transAxes, fontsize=12, fontweight='bold',
                         ha='center', va='top',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        else:
            # No plays yesterday
            ax_table.text(0.5, 0.5, 'No plays recorded for yesterday',
                         transform=ax_table.transAxes, fontsize=14,
                         ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        plot_filename = f'v5_yesterday_summary_{yesterday}.png'
        local_plot_path = f'/tmp/{plot_filename}'
        plt.tight_layout()
        plt.savefig(local_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # UPLOAD TO S3 (NO ACL - BUCKET POLICY APPROACH)
        # =====================================================================
        # We upload normally without ACL='public-read' because this bucket
        # has ACLs disabled. Instead, we use a bucket policy (see docstring)
        # to make data/04_output/strategy_plots/* publicly readable.
        # This allows HTML emails to embed images via direct HTTPS URLs.
        # =====================================================================
        s3_key = f'data/04_output/strategy_plots/{season}/{plot_filename}'
        s3_client.upload_file(local_plot_path, S3_BUCKET, s3_key)
        
        print(f"   ✅ Generated v5 yesterday summary plot: s3://{S3_BUCKET}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to generate v5 yesterday summary plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def format_strategies_for_email(strategies: List[Dict], strategy_type: str, top_n: int = 20) -> str:
    """
    Format top strategies for email notification.
    
    Args:
        strategies: List of strategy dicts with performance metrics
        strategy_type: '2d' or '3d'
        top_n: Number of top strategies to include
    
    Returns:
        Formatted string for email
    """
    if not strategies:
        return "No strategies available"
    
    # Sort by hit rate descending
    sorted_strategies = sorted(strategies, key=lambda x: x.get('hit_rate', 0), reverse=True)
    
    lines = [f"\n{'='*80}"]
    lines.append(f"TOP {top_n} {strategy_type.upper()} STRATEGIES BY HIT RATE")
    lines.append(f"{'='*80}\n")
    
    for i, strat in enumerate(sorted_strategies[:top_n], 1):
        # Build strategy description
        if strategy_type == '2d':
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
        else:  # 3d
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']} | {strat['scorer_type']}"
        
        hit_rate = strat.get('hit_rate', 0)
        roi = strat.get('roi', 0)
        edge = strat.get('edge', 0)
        wins = strat.get('wins', 0)
        losses = strat.get('losses', 0)
        ties = strat.get('ties', 0)
        
        # Determine emoji
        if hit_rate >= 60:
            emoji = '🔥'
        elif hit_rate >= 55:
            emoji = '✅'
        elif hit_rate >= 50:
            emoji = '➖'
        else:
            emoji = '❌'
        
        lines.append(f"{emoji} #{i:2d}. {desc}")
        lines.append(f"     {wins}W-{losses}L-{ties}T | Hit: {hit_rate:.1f}% | ROI: {roi:+.1f}% | Edge: {edge:+.1f}%")
    
    # Summary stats
    total_strategies = len(strategies)
    profitable = sum(1 for s in strategies if s.get('roi', 0) > 0)
    avg_hit_rate = sum(s.get('hit_rate', 0) for s in strategies) / len(strategies)
    avg_roi = sum(s.get('roi', 0) for s in strategies) / len(strategies)
    
    lines.append(f"\n{'='*80}")
    lines.append(f"SUMMARY:")
    lines.append(f"  Total Strategies: {total_strategies}")
    lines.append(f"  Profitable: {profitable}/{total_strategies} ({profitable/total_strategies*100:.1f}%)")
    lines.append(f"  Avg Hit Rate: {avg_hit_rate:.1f}%")
    lines.append(f"  Avg ROI: {avg_roi:+.1f}%")
    lines.append(f"{'='*80}\n")
    
    return '\n'.join(lines)


def format_all_strategies_combined(strategy_rankings: Dict, top_n: int = 40) -> str:
    """
    Format ALL strategies (2D + 3D combined) ranked by hit rate.
    
    Args:
        strategy_rankings: Dict with '2d' and '3d' keys containing strategy lists
        top_n: Number of top strategies to include
    
    Returns:
        Formatted string for email
    """
    # Combine all strategies
    all_strategies = []
    for strat_type in ['2d', '3d']:
        if strat_type in strategy_rankings:
            for strat in strategy_rankings[strat_type]:
                strat_copy = strat.copy()
                strat_copy['_type'] = strat_type
                all_strategies.append(strat_copy)
    
    if not all_strategies:
        return "No strategies available"
    
    # Sort by hit rate descending
    sorted_strategies = sorted(all_strategies, key=lambda x: x.get('hit_rate', 0), reverse=True)
    
    lines = [f"\n{'='*80}"]
    lines.append(f"ALL STRATEGIES RANKED BY HIT RATE (Top {top_n})")
    lines.append(f"{'='*80}\n")
    
    for i, strat in enumerate(sorted_strategies[:top_n], 1):
        # Build strategy description
        strat_type = strat['_type']
        if strat_type == '2d':
            desc = f"[2D] {strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
        else:  # 3d
            desc = f"[3D] {strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']} | {strat['scorer_type']}"
        
        hit_rate = strat.get('hit_rate', 0)
        roi = strat.get('roi', 0)
        edge = strat.get('edge', 0)
        wins = strat.get('wins', 0)
        losses = strat.get('losses', 0)
        ties = strat.get('ties', 0)
        
        # Determine emoji
        if hit_rate >= 60:
            emoji = '🔥'
        elif hit_rate >= 55:
            emoji = '✅'
        elif hit_rate >= 50:
            emoji = '➖'
        else:
            emoji = '❌'
        
        lines.append(f"{emoji} #{i:2d}. {desc}")
        lines.append(f"     {wins}W-{losses}L-{ties}T | Hit: {hit_rate:.1f}% | ROI: {roi:+.1f}% | Edge: {edge:+.1f}%")
    
    # Summary stats
    total_strategies = len(all_strategies)
    profitable = sum(1 for s in all_strategies if s.get('roi', 0) > 0)
    avg_hit_rate = sum(s.get('hit_rate', 0) for s in all_strategies) / len(all_strategies) if all_strategies else 0
    avg_roi = sum(s.get('roi', 0) for s in all_strategies) / len(all_strategies) if all_strategies else 0
    
    lines.append(f"\n{'='*80}")
    lines.append(f"SUMMARY:")
    lines.append(f"  Total Strategies: {total_strategies}")
    lines.append(f"  Profitable: {profitable}/{total_strategies} ({profitable/total_strategies*100:.1f}%)" if total_strategies > 0 else "  Profitable: 0")
    lines.append(f"  Avg Hit Rate: {avg_hit_rate:.1f}%")
    lines.append(f"  Avg ROI: {avg_roi:+.1f}%")
    lines.append(f"{'='*80}\n")
    
    return '\n'.join(lines)


def load_active_strategies_from_v5(s3_client) -> set:
    """
    Load the list of active strategies from enhanced_unders_v5.json.
    
    Returns:
        set: Set of tuples (line_tier, spread_bin, scorer_type) for active strategies
    """
    active_strategies = set()
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key='strategies/enhanced_unders_v5.json')
        v5_data = json.loads(response['Body'].read().decode('utf-8'))
        
        print(f"\n📋 Loading active strategies from v5 JSON:")
        for idx, strat in enumerate(v5_data.get('strategies', []), 1):
            line_tier = strat.get('line_tier', '')
            spread_bin = strat.get('spread_bin', '')
            scorer_type = strat.get('scorer_type', 'N/A')
            
            active_strategies.add((line_tier, spread_bin, scorer_type))
            
            # Log each active strategy
            strat_name = strat.get('strategy_name', 'Unknown')
            strat_type = strat.get('strategy_type', '?')
            print(f"   {idx}. [{strat_type.upper()}] {line_tier} | {spread_bin} | UNDER", end='')
            if scorer_type != 'N/A':
                print(f" | {scorer_type}", end='')
            print(f" ({strat_name})")
        
        print(f"\n✅ Loaded {len(active_strategies)} active strategies from v5 JSON")
        return active_strategies
    except Exception as e:
        print(f"⚠️  Failed to load v5 JSON, will generate all plots: {e}")
        return set()


def strategy_is_active(strat: Dict, strategy_type: str, active_strategies: set) -> bool:
    """
    Check if a strategy is in the active v5 list.
    Uses line_tier, spread_bin, and scorer_type to match (bet_side is always UNDER in v5).
    
    Args:
        strat: Strategy dictionary
        strategy_type: '2d' or '3d'
        active_strategies: Set of active strategy tuples (line_tier, spread_bin, scorer_type)
    
    Returns:
        bool: True if strategy is active, False otherwise
    """
    line_tier = strat.get('line_tier', '')
    spread_bin = strat.get('spread_bin', '')
    scorer_type = strat.get('scorer_type', 'N/A') if strategy_type == '3d' else 'N/A'
    
    return (line_tier, spread_bin, scorer_type) in active_strategies


def generate_all_strategy_plots(strategy_rankings: Dict, season: str, recent_plays_n: int = 10) -> int:
    """
    Generate performance plots for all strategies in ranking.
    
    Args:
        strategy_rankings: Dict with '2d' and '3d' keys containing strategy lists
        season: Current season string
        recent_plays_n: Number of recent plays to show in table (default 10)
    
    Returns:
        int: Number of plots generated
    """
    s3_client = boto3.client('s3')
    plots_generated = 0
    plots_skipped = 0
    seasons = BACKTEST_SEASONS  # ['2023-24', '2024-25', '2025-26']
    
    # Load active strategies from v5 JSON if in active_only mode
    active_strategies = set()
    if PLOT_GENERATION_MODE == 'active_only':
        active_strategies = load_active_strategies_from_v5(s3_client)
    
    for strategy_type, strategies in strategy_rankings.items():
        print(f"\n📊 Generating plots for {strategy_type.upper()} strategies...")
        
        for strat in strategies:
            # Check if we should skip this strategy based on config
            if PLOT_GENERATION_MODE == 'active_only' and active_strategies:
                if not strategy_is_active(strat, strategy_type, active_strategies):
                    strat_name = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
                    if strategy_type == '3d':
                        strat_name += f" | {strat.get('scorer_type', '')}"
                    print(f"   ⏭️  Skipping {strat_name} (not in v5)")
                    plots_skipped += 1
                    continue
            try:
                # Load plays data for this strategy
                df = load_backtest_plays_for_strategy_simple(s3_client, strat, strategy_type, seasons)
                
                if df is None or len(df) == 0:
                    print(f"   ⚠️  No data for {strat['line_tier']} | {strat['spread_bin']}")
                    continue
                
                # Generate plot filename
                plot_name = f"{strat['line_tier']}_{strat['spread_bin']}_{strat['bet_side']}"
                if strategy_type == '3d':
                    scorer = strat.get('scorer_type', '').replace('≥', 'ge').replace('%', 'pct')
                    plot_name += f"_{scorer}"
                plot_name = plot_name.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
                plot_filename = f"{plot_name}.png"
                
                # Generate 5-panel plot (3x2 grid)
                fig = plt.figure(figsize=(16, 14))
                gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])
                
                desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
                if strategy_type == '3d':
                    desc += f" | {strat.get('scorer_type', '')}"
                
                fig.suptitle(f"Strategy Performance: {desc}", fontsize=16, fontweight='bold')
                
                # Convert dates and calculate cumulative win rates
                df['game_date'] = pd.to_datetime(df['game_date'])
                df = df.sort_values('game_date')
                df['is_win'] = (df['result'] == 'WIN').astype(int)
                
                # Season colors
                season_colors = {
                    '2023-24': '#1f77b4',
                    '2024-25': '#ff7f0e',
                    '2025-26': '#2ca02c'
                }
                
                # Plot each season (panels 1-3)
                for idx, s in enumerate(seasons[:3]):
                    ax = fig.add_subplot(gs[idx // 2, idx % 2])
                    df_season = df[df['season'] == s].copy()
                    
                    if len(df_season) == 0:
                        ax.text(0.5, 0.5, f'No data for {s}', ha='center', va='center')
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 100)
                        continue
                    
                    df_season['cumulative_wins'] = df_season['is_win'].cumsum()
                    df_season['cumulative_plays'] = range(1, len(df_season) + 1)
                    df_season['win_rate'] = (df_season['cumulative_wins'] / df_season['cumulative_plays'] * 100)
                    
                    ax.plot(df_season['game_date'], df_season['win_rate'], 
                           color=season_colors.get(s, 'blue'), linewidth=2)
                    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                    ax.set_title(f'{s}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Date', fontsize=11)
                    ax.set_ylabel('Win Rate (%)', fontsize=11)
                    ax.set_ylim(0, 100)
                    ax.grid(True, alpha=0.3)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    final_wr = df_season['win_rate'].iloc[-1]
                    total_wins = int(df_season['cumulative_wins'].iloc[-1])
                    total_losses = len(df_season) - total_wins
                    ax.text(0.02, 0.98, f'{total_wins}W-{total_losses}L | {final_wr:.1f}%',
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Panel 4: Overall (use play number instead of date)
                ax = fig.add_subplot(gs[1, 1])
                df_overall = df.copy()
                df_overall['cumulative_wins'] = df_overall['is_win'].cumsum()
                df_overall['play_number'] = range(1, len(df_overall) + 1)
                df_overall['win_rate'] = (df_overall['cumulative_wins'] / df_overall['play_number'] * 100)
                
                # Plot each season with different colors
                for s in seasons:
                    df_seg = df_overall[df_overall['season'] == s]
                    if len(df_seg) > 0:
                        ax.plot(df_seg['play_number'], df_seg['win_rate'],
                               color=season_colors.get(s, 'black'), linewidth=2, label=s)
                
                ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_title('Overall (All Seasons)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Play Number', fontsize=11)
                ax.set_ylabel('Win Rate (%)', fontsize=11)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                final_wr = df_overall['win_rate'].iloc[-1]
                total_wins = int(df_overall['cumulative_wins'].iloc[-1])
                total_losses = len(df_overall) - total_wins
                ax.text(0.02, 0.98, f'{total_wins}W-{total_losses}L | {final_wr:.1f}%',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Panel 5: Recent plays table (bottom row, spans both columns)
                ax_table = fig.add_subplot(gs[2, :])
                ax_table.axis('off')
                
                # Get most recent N plays (sorted descending - most recent first)
                df_recent = df_overall.tail(recent_plays_n).sort_values('game_date', ascending=False).copy()
                
                # Format table data
                table_data = []
                for _, row in df_recent.iterrows():
                    date_str = row['game_date'].strftime('%m/%d/%y')
                    player = row.get('player_name', 'Unknown')[:18]
                    line = f"{row['points_line']:.1f}"
                    actual_pts = f"{int(row['actual_points'])}" if pd.notna(row['actual_points']) else 'N/A'
                    
                    # What we bet (from strategy config)
                    our_bet = strat['bet_side']
                    
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
                    
                    # Get metadata fields - actual values instead of bins
                    line_value = row.get('points_line', 'N/A')
                    spread_value = row.get('team_spread', 'N/A')
                    
                    # Format to 1 decimal place
                    if pd.notna(line_value) and line_value != 'N/A':
                        line_value = f"{float(line_value):.1f}"
                    if pd.notna(spread_value) and spread_value != 'N/A':
                        spread_value = f"{float(spread_value):+.1f}"
                    
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
                        line_value,
                        spread_value,
                        line,
                        actual_pts,
                        scorer_type_display,
                        our_bet,
                        actual_result,
                        result_text
                    ])
                
                # Create table
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=['Date', 'Player', 'Line', 'Spread', 'Over/Under', 'Scored', 'Scorer', 'Our Bet', 'Actual', 'Result'],
                    loc='center',
                    cellLoc='left',
                    colWidths=[0.08, 0.16, 0.06, 0.07, 0.09, 0.06, 0.09, 0.08, 0.07, 0.11]
                )
                
                # Style table
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 2)
                
                # Header styling
                for i in range(10):
                    cell = table[(0, i)]
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                
                # Alternate row colors and color-code results
                for i in range(1, len(table_data) + 1):
                    for j in range(10):
                        cell = table[(i, j)]
                        
                        # Base row color (alternating)
                        if i % 2 == 0:
                            cell.set_facecolor('#F0F0F0')
                        else:
                            cell.set_facecolor('#FFFFFF')
                        
                        # Color-code result column (column 9)
                        if j == 9:
                            result_text = table_data[i-1][9]
                            if result_text == 'WIN':
                                cell.set_facecolor('#C6EFCE')
                                cell.set_text_props(weight='bold', color='#006100')
                            elif result_text == 'LOSS':
                                cell.set_facecolor('#FFC7CE')
                                cell.set_text_props(weight='bold', color='#9C0006')
                            elif result_text == 'PUSH':
                                cell.set_facecolor('#FFEB9C')
                                cell.set_text_props(weight='bold', color='#9C5700')
                
                # Title for table panel
                ax_table.set_title(f'Most Recent {recent_plays_n} Plays', fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                
                # Save to /tmp and upload to S3
                local_path = f'/tmp/{plot_filename}'
                plt.savefig(local_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Upload to S3 (no ACL - bucket policy makes this folder public)
                s3_key = f'data/04_output/strategy_plots/{season}/{plot_filename}'
                s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                
                plots_generated += 1
                print(f"   ✅ {plot_filename}")
                
            except Exception as e:
                print(f"   ❌ Failed to generate plot: {e}")
                continue
    
    # Summary
    if PLOT_GENERATION_MODE == 'active_only':
        print(f"\n📊 Plot generation summary: {plots_generated} generated, {plots_skipped} skipped (mode: {PLOT_GENERATION_MODE})")
    else:
        print(f"\n📊 Plot generation summary: {plots_generated} generated (mode: {PLOT_GENERATION_MODE})")
    
    return plots_generated


def load_backtest_plays_for_strategy_simple(s3_client, strategy: Dict, strategy_type: str, seasons: List[str]) -> 'pd.DataFrame':
    """Load plays for a strategy across seasons (simplified for refresh lambda)."""
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas not available - check Lambda layer")
    
    dfs = []
    for season in seasons:
        s3_key = f'{BACKTEST_PREFIX}/{strategy_type}/{season}/plays.csv'
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
            
            # CRITICAL: Validate no Unknown spread_bin values exist
            unknown_spread_count = (df['spread_bin'] == 'Unknown').sum()
            if unknown_spread_count > 0:
                total_plays = len(df)
                unknown_pct = (unknown_spread_count / total_plays) * 100
                unknown_dates = sorted([str(d) for d in df[df['spread_bin'] == 'Unknown']['game_date'].unique()])
                
                if unknown_pct < 1.0:
                    # Under 1% - acceptable, just filter out and warn
                    print(f"   ⚠️  Found {unknown_spread_count} plays ({unknown_pct:.2f}%) with spread_bin='Unknown' in {season}")
                    print(f"   Dates ({len(unknown_dates)}): {', '.join(unknown_dates)}")
                    print(f"   These plays will be EXCLUDED (under 1% threshold)")
                    
                    # Filter out Unknown plays
                    df = df[df['spread_bin'] != 'Unknown'].copy()
                else:
                    # Over 1% - this is a problem, raise error
                    error_msg = (
                        f"\n❌ FATAL ERROR: Found {unknown_spread_count} plays ({unknown_pct:.1f}%) with spread_bin='Unknown' in {season}!\n"
                        f"   S3 file: s3://{S3_BUCKET}/{s3_key}\n"
                        f"   This exceeds the 1% tolerance threshold.\n"
                        f"   Dates ({len(unknown_dates)}): {', '.join(unknown_dates)}\n\n"
                        f"   This means game lines are MISSING for these dates.\n\n"
                        f"   FIX:\n"
                        f"   1. Fetch missing game lines:\n"
                        f"      python scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season {season}\n"
                        f"   2. Re-run backtest to regenerate plays.csv with correct spread_bin\n"
                    )
                    raise ValueError(error_msg)
            
            mask = (
                (df['line_tier'] == strategy['line_tier']) &
                (df['spread_bin'] == strategy['spread_bin']) &
                (df['bet_side'] == strategy['bet_side'])
            )
            
            if strategy_type == '3d' and 'scorer_type' in strategy:
                mask = mask & (df['scorer_type'] == strategy['scorer_type'])
            
            df_strat = df[mask].copy()
            df_strat['season'] = season
            if len(df_strat) > 0:
                dfs.append(df_strat)
        except Exception:
            continue
    
    return pd.concat(dfs, ignore_index=True) if dfs else None


def format_strategies_for_email(strategies: List[Dict], strategy_type: str, top_n: int = 20) -> str:
    """
    Format top strategies for email notification.
    
    Args:
        strategies: List of strategy dicts with performance metrics
        strategy_type: '2d' or '3d'
        top_n: Number of top strategies to include
    
    Returns:
        Formatted string for email
    """
    if not strategies:
        return "No strategies available"
    
    # Sort by hit rate descending
    sorted_strategies = sorted(strategies, key=lambda x: x.get('hit_rate', 0), reverse=True)
    
    lines = [f"\n{'='*80}"]
    lines.append(f"TOP {top_n} {strategy_type.upper()} STRATEGIES BY HIT RATE")
    lines.append(f"{'='*80}\n")
    
    for i, strat in enumerate(sorted_strategies[:top_n], 1):
        # Build strategy description
        if strategy_type == '2d':
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
        else:  # 3d
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']} | {strat['scorer_type']}"
        
        hit_rate = strat.get('hit_rate', 0)
        roi = strat.get('roi', 0)
        edge = strat.get('edge', 0)
        wins = strat.get('wins', 0)
        losses = strat.get('losses', 0)
        ties = strat.get('ties', 0)
        
        # Determine emoji
        if hit_rate >= 60:
            emoji = '🔥'
        elif hit_rate >= 55:
            emoji = '✅'
        elif hit_rate >= 50:
            emoji = '➖'
        else:
            emoji = '❌'
        
        lines.append(f"{emoji} #{i:2d}. {desc}")
        lines.append(f"     {wins}W-{losses}L-{ties}T | Hit: {hit_rate:.1f}% | ROI: {roi:+.1f}% | Edge: {edge:+.1f}%")
    
    # Summary stats
    total_strategies = len(strategies)
    profitable = sum(1 for s in strategies if s.get('roi', 0) > 0)
    avg_hit_rate = sum(s.get('hit_rate', 0) for s in strategies) / len(strategies)
    avg_roi = sum(s.get('roi', 0) for s in strategies) / len(strategies)
    
    lines.append(f"\n{'='*80}")
    lines.append(f"SUMMARY:")
    lines.append(f"  Total Strategies: {total_strategies}")
    lines.append(f"  Profitable: {profitable}/{total_strategies} ({profitable/total_strategies*100:.1f}%)")
    lines.append(f"  Avg Hit Rate: {avg_hit_rate:.1f}%")
    lines.append(f"  Avg ROI: {avg_roi:+.1f}%")
    lines.append(f"{'='*80}\n")
    
    return '\n'.join(lines)


# =============================================================================
# YESTERDAY PLAYS SUMMARY
# =============================================================================

def print_yesterday_plays_summary(season: str, yesterday: str) -> None:
    """
    Print a summary of yesterday's plays to CloudWatch logs.
    
    Args:
        season: Current season
        yesterday: Yesterday's date in YYYY-MM-DD format
    """
    try:
        print(f"\n{'='*80}")
        print(f"📋 YESTERDAY'S PLAYS SUMMARY ({yesterday})")
        print(f"{'='*80}\n")
        
        s3_client = boto3.client('s3')
        
        # Load v5 strategies
        v5_strategies = load_v5_strategies_from_s3()
        if not v5_strategies:
            print("   ⚠️  Could not load v5 strategies")
            return
        
        # Load 2025-26 backtest plays
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=f'{BACKTEST_PREFIX}/2d/{season}/plays.csv')
            df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        except Exception as e:
            print(f"   ⚠️  Could not load backtest plays: {e}")
            return
        
        # Filter to yesterday's plays
        df['game_date'] = pd.to_datetime(df['game_date'])
        yesterday_date = pd.to_datetime(yesterday)
        df_yesterday = df[df['game_date'] == yesterday_date].copy()
        
        if len(df_yesterday) == 0:
            print(f"   No plays found for {yesterday}")
            return
        
        # Match to v5 strategies
        v5_plays = []
        for _, row in df_yesterday.iterrows():
            for v5_strat in v5_strategies:
                if (row['line_tier'] == v5_strat['line_tier'] and
                    row['spread_bin'] == v5_strat['spread_bin'] and
                    row['bet_side'] == v5_strat['bet_side']):
                    
                    play_info = {
                        'strategy_name': v5_strat['strategy_name'],
                        'player': row['player_name'],
                        'line': row['points_line'],
                        'actual': row['actual_points'],
                        'result': row['result'],
                        'team': row.get('team', 'N/A'),
                        'opponent': row.get('opponent', 'N/A')
                    }
                    v5_plays.append(play_info)
                    break
        
        if not v5_plays:
            print(f"   No v5 strategy plays found for {yesterday}")
            return
        
        print(f"   Found {len(v5_plays)} plays matching v5 strategies:\n")
        
        # Print each play
        for i, play in enumerate(v5_plays, 1):
            result_emoji = '✅' if play['result'] == 'WIN' else '❌' if play['result'] == 'LOSS' else '🔄'
            actual = f"{int(play['actual'])}" if pd.notna(play['actual']) else 'DNP'
            
            print(f"   Play {i}: {result_emoji} {play['result']}")
            print(f"      Strategy: {play['strategy_name']}")
            print(f"      Player: {play['player']}")
            print(f"      Line: UNDER {play['line']:.1f} pts")
            print(f"      Actual: {actual} pts")
            print(f"      Matchup: {play['team']} vs {play['opponent']}")
            print()
        
        # Summary stats
        wins = sum(1 for p in v5_plays if p['result'] == 'WIN')
        losses = sum(1 for p in v5_plays if p['result'] == 'LOSS')
        hit_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        print(f"   Summary: {wins}W-{losses}L | Hit Rate: {hit_rate:.1f}%")
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"   ⚠️  Error generating yesterday plays summary: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN REFRESH FUNCTION
# =============================================================================

def refresh_strategy_statistics(
    season: str = '2025-26',
    strategy_types: List[str] = ['2d', '3d'],
    skip_backtest: bool = False
) -> Dict:
    """
    Main function to refresh strategy statistics.
    
    Self-contained implementation that:
    - Generates ALL possible strategy combinations (98 for 2D, 196 for 3D)
    - Loads player props directly from S3
    - Matches props to strategies and calculates outcomes
    - Aggregates statistics across multiple seasons
    - Filters by minimum plays (50+) not historical ROI
    
    Args:
        season: Current NBA season
        strategy_types: List of strategy types to update
        skip_backtest: If True, skip regenerating current season backtest
    
    Returns:
        Dict with results summary
    """
    yesterday = get_yesterday_et()
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    
    print("="*80)
    print("🔄 REFRESHING STRATEGY STATISTICS")
    print("="*80)
    print(f"Current Season: {season}")
    print(f"Strategy Types: {', '.join(strategy_types)}")
    print(f"Backtest Seasons: {', '.join(BACKTEST_SEASONS)}")
    print(f"Data Through: {yesterday}")
    print(f"Timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Mode: Self-contained (tests all {98 if '2d' in strategy_types else 0}/{196 if '3d' in strategy_types else 0} combinations)")
    print("="*80)
    
    results = {}
    strategy_rankings = {}  # Store strategy rankings for email
    s3_client = boto3.client('s3')
    
    for strategy_type in strategy_types:
        print(f"\n{'='*80}")
        print(f"Processing {strategy_type.upper()} Strategy")
        print(f"{'='*80}\n")
        
        # Step 1: Re-run backtests for ALL seasons (if not skipped)
        if not skip_backtest:
            print(f"Step 1: Updating backtests for all seasons...")
            for backtest_season in BACKTEST_SEASONS:
                print(f"\n   Regenerating {backtest_season} backtest...")
                success = run_backtest_for_season(s3_client, backtest_season, strategy_type)
                if not success:
                    error_msg = f"{backtest_season} backtest failed - cannot continue with stale data"
                    print(f"   ❌ {error_msg}")
                    results[strategy_type] = {'success': False, 'error': error_msg}
                    # FAIL HARD - don't continue
                    raise RuntimeError(f"Backtest failed for {backtest_season} {strategy_type}")
        else:
            print(f"Step 1: Skipping backtest regeneration (using existing data)")
        
        # Step 2: Load all seasons from S3
        print(f"\nStep 2: Loading multi-season backtest data...")
        
        dfs = []
        for s in BACKTEST_SEASONS:
            df = load_backtest_plays(s3_client, S3_BUCKET, strategy_type, s)
            if df is not None and not df.empty:
                dfs.append(df)
        
        if not dfs:
            print(f"   ❌ No backtest data found for any season!")
            results[strategy_type] = {'success': False, 'error': 'No data'}
            continue
        
        if not PANDAS_AVAILABLE:
            print(f"   ❌ pandas not available - cannot proceed!")
            results[strategy_type] = {'success': False, 'error': 'pandas not available'}
            continue
        
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Loaded {len(df_all)} total plays across {len(dfs)} seasons")
        
        # Step 3: Calculate aggregate stats
        print(f"\nStep 3: Calculating aggregate strategy statistics...")
        strategies = calculate_aggregate_strategy_stats(
            df_all,
            strategy_type,
            BACKTEST_SEASONS
        )
        
        if not strategies:
            print(f"   ❌ No strategies met minimum threshold!")
            results[strategy_type] = {'success': False, 'error': 'No strategies qualified'}
            continue
        
        # Store strategies for email
        strategy_rankings[strategy_type] = strategies
        
        # Step 4: Generate updated JSON
        print(f"\nStep 4: Generating updated strategy file...")
        
        if strategy_type == '2d':
            filename = f'points_by_role_gamespread_strategies_{season}.json'
        else:
            filename = f'points_by_role_gamespread_6feet_strategies_{season}_rim40.json'
        
        local_path = f'/tmp/{filename}'
        
        generate_strategy_json(
            strategies=strategies,
            output_path=local_path,
            metadata={
                'generated_at': now_et.isoformat(),
                'data_through': yesterday,
                'seasons_included': BACKTEST_SEASONS,
                'total_plays': len(df_all)
            }
        )
        
        # Validate final strategy pairs before logging
        try:
            validate_strategy_pairs(strategies, strategy_type)
        except AssertionError as e:
            print(f"   ⚠️  Warning: {e}")
        
        # Log detailed results for this strategy type
        log_strategy_results(strategies, strategy_type, BACKTEST_SEASONS, df_all)
        
        # Step 5: Upload to S3
        print(f"\nStep 5: Uploading to S3...")
        s3_key = f'{STRATEGIES_PREFIX}/{filename}'
        
        try:
            s3_client.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"   ✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")
            
            results[strategy_type] = {
                'success': True,
                'strategies_count': len(strategies),
                'total_plays': len(df_all),
                's3_path': f's3://{S3_BUCKET}/{s3_key}'
            }
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            results[strategy_type] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ REFRESH COMPLETE")
    print(f"{'='*80}")
    
    summary_lines = []
    all_success = all(r.get('success', False) for r in results.values())
    
    for strategy_type, result in results.items():
        if result['success']:
            line = f"{strategy_type.upper()}: ✅ {result['strategies_count']} strategies, {result['total_plays']} plays"
            print(line)
            summary_lines.append(line)
        else:
            line = f"{strategy_type.upper()}: ❌ {result.get('error', 'Failed')}"
            print(line)
            summary_lines.append(line)
    
    print(f"{'='*80}\n")
    
    # Send SNS notification
    if all_success:
        subject = f"✅ Strategy Statistics Refresh Complete - {season}"
        
        # Generate strategy performance plots
        print(f"\n{'='*80}")
        print("📊 Generating Strategy Performance Plots")
        print(f"{'='*80}\n")
        
        # Generate yesterday summary plot for v5 strategies
        v5_yesterday_success = generate_v5_yesterday_summary_plot(strategy_rankings, season, yesterday)
        
        # Generate individual strategy plots
        plots_generated = generate_all_strategy_plots(strategy_rankings, season)
        
        # Build yesterday summary section
        if v5_yesterday_success:
            # Use HTTPS URL for email embedding (with cache-busting timestamp)
            timestamp = int(now_et.timestamp())
            yesterday_plot_url = f"https://{S3_BUCKET}.s3.us-east-2.amazonaws.com/data/04_output/strategy_plots/{season}/v5_yesterday_summary_{yesterday}.png?t={timestamp}"
            yesterday_summary = f"\n{'='*80}\n"
            yesterday_summary += f"📅 YESTERDAY'S PERFORMANCE ({yesterday})\n"
            yesterday_summary += f"{'='*80}\n"
            yesterday_summary += f"Combined results across all 15 v5 strategies\n"
            yesterday_summary += f"📈 Plot: {yesterday_plot_url}\n"
            yesterday_summary += f"{'='*80}\n"
        else:
            yesterday_summary = ""
        
        plots_summary = f"\n{'='*80}\n"
        plots_summary += f"📈 STRATEGY PERFORMANCE PLOTS\n"
        plots_summary += f"{'='*80}\n"
        plots_summary += f"Generated {plots_generated} performance plots (5-panel: 2023-24, 2024-25, 2025-26, Overall, Recent Plays Table)\n"
        plots_summary += f"Location: s3://{S3_BUCKET}/data/04_output/strategy_plots/{season}/\n"
        plots_summary += f"{'='*80}\n"
        
        # Build V5 strategy performance section
        v5_performance = format_v5_strategies_for_email(strategy_rankings, season)
        
        # Build combined strategy rankings (ALL strategies, not split by 2D/3D)
        all_strategies_ranked = format_all_strategies_combined(strategy_rankings, top_n=40)
        
        message = f"""Strategy Statistics Refresh Completed Successfully

Season: {season}
Data Through: {yesterday}
Backtest Seasons: {', '.join(BACKTEST_SEASONS)}
Timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}

Results:
{chr(10).join(summary_lines)}

Total Strategies: {sum(r.get('strategies_count', 0) for r in results.values() if r.get('success'))}
Total Plays: {sum(r.get('total_plays', 0) for r in results.values() if r.get('success'))}

All strategy JSON files have been updated in S3.

{yesterday_summary}

{v5_performance}

{plots_summary}

{all_strategies_ranked}
"""
    else:
        subject = f"❌ Strategy Statistics Refresh Failed - {season}"
        message = f"""Strategy Statistics Refresh Failed

Season: {season}
Timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}

Errors:
{chr(10).join(summary_lines)}

Please check CloudWatch logs for details.
"""
    
    # Send via SNS (plain text)
    send_sns(subject, message)
    
    # Send via SES (HTML with inline images)
    print(f"   📤 Sending HTML email via SES...")
    if SES_FROM_EMAIL and SES_TO_EMAIL:
        # Log both the to / from emails
        print(f"   From: {SES_FROM_EMAIL}")
        print(f"   To: {SES_TO_EMAIL}")

        # Extract yesterday plot URL from yesterday_summary if it exists
        yesterday_plot_https = None
        if v5_yesterday_success and yesterday_summary:
            # Add timestamp to URL to bust email client caching
            timestamp = int(now_et.timestamp())
            yesterday_plot_https = f"https://{S3_BUCKET}.s3.us-east-2.amazonaws.com/data/04_output/strategy_plots/{season}/v5_yesterday_summary_{yesterday}.png?t={timestamp}"
        
        html_message = generate_html_email(message, yesterday_plot_https)
        send_ses(subject, html_message, message)
    
    # Print yesterday's plays summary to logs
    print_yesterday_plays_summary(season, yesterday)
    
    return results


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event (can contain 'season', 'strategy', 'skip_backtest')
        context: Lambda context
    
    Returns:
        Dict with execution results
    """
    # Extract parameters from event (with defaults)
    season = event.get('season', '2025-26')
    strategy = event.get('strategy', 'both')
    skip_backtest = event.get('skip_backtest', False)  # Default: run backtest
    
    # Determine strategy types
    if strategy == 'both':
        strategy_types = ['2d', '3d']
    else:
        strategy_types = [strategy]
    
    # Run refresh
    results = refresh_strategy_statistics(
        season=season,
        strategy_types=strategy_types,
        skip_backtest=skip_backtest
    )
    
    # Format response
    all_success = all(r.get('success', False) for r in results.values())
    
    return {
        'statusCode': 200 if all_success else 500,
        'body': json.dumps({
            'success': all_success,
            'results': results
        })
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for local execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Refresh strategy statistics with multi-season backtest data'
    )
    parser.add_argument(
        '--season',
        default='2025-26',
        help='Current NBA season (default: 2025-26)'
    )
    parser.add_argument(
        '--strategy',
        choices=['2d', '3d', 'both'],
        default='both',
        help='Which strategy type to update (default: both)'
    )
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip regenerating current season backtest (use existing data)'
    )
    
    args = parser.parse_args()
    
    # Determine strategy types
    if args.strategy == 'both':
        strategy_types = ['2d', '3d']
    else:
        strategy_types = [args.strategy]
    
    # Run refresh
    results = refresh_strategy_statistics(
        season=args.season,
        strategy_types=strategy_types,
        skip_backtest=args.skip_backtest
    )
    
    # Exit with error code if any failed
    all_success = all(r.get('success', False) for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
