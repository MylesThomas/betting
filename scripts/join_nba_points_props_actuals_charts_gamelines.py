"""
NBA Player Props Data Loader & Joiner

Loads and joins player props, game logs, and shot charts from S3.

Usage:
    # Load and display join stats
    python analysis/analyze_nba_player_props_coverage.py --season 2025-26
    
    # Save locally
    python analysis/analyze_nba_player_props_coverage.py --season 2025-26 --save data/merged_props_actuals.csv
    
    # Upload to S3 (s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_2025-26.csv)
    python analysis/analyze_nba_player_props_coverage.py --season 2025-26 --s3
    
    # Both
    python analysis/analyze_nba_player_props_coverage.py --season 2025-26 --save data/merged.csv --s3

Author: Thomas Myles
Date: 2026-01-05
"""

import pandas as pd
import boto3
from io import StringIO
import sys
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from player_name_utils import normalize_player_name

# S3 buckets
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_NBA = 'nba-api-mt'
S3_BUCKET_OUTPUT = 'nba-betting-mt'  # For merged output
S3_PREFIX_OUTPUT = 'data/03_intermediate'


def normalize_team_name(name):
    """
    Normalize team names for matching between NBA API and Odds API
    
    Args:
        name: Team name string
        
    Returns:
        Normalized name (Odds API format)
    """
    if pd.isna(name):
        return name
    
    # Team name mappings (NBA API -> Odds API format)
    team_mappings = {
        'LA Clippers': 'Los Angeles Clippers',
        'LA Lakers': 'Los Angeles Lakers',
    }
    
    return team_mappings.get(name, name)


def load_all_props(season):
    """Load ALL player props for the season from S3"""
    print(f"📊 Loading props from s3://{S3_BUCKET_PROPS}/nba/historical_player_props/{season}/...")
    
    s3_client = boto3.client('s3')
    prefix = f"nba/historical_player_props/{season}/"
    
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"❌ No props files found")
        return pd.DataFrame()
    
    all_props = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv'):
            try:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_PROPS, Key=obj['Key'])
                df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                
                # Extract date from filename (ET-based, more reliable than parsing UTC game_time)
                # Filename format: YYYY-MM-DD.csv
                filename = obj['Key'].split('/')[-1]
                date_str = filename.replace('.csv', '')
                df['game_date'] = date_str
                
                all_props.append(df)
            except Exception as e:
                print(f"  ⚠️  Error loading {obj['Key']}: {e}")
    
    df_props = pd.concat(all_props, ignore_index=True)
    
    # Add normalized name
    df_props['player_normalized'] = df_props['player'].apply(normalize_player_name)
    
    print(f"✅ Loaded {len(df_props):,} prop rows from {len(all_props)} files")
    print(f"   Dates: {df_props['game_date'].min()} to {df_props['game_date'].max()}")
    print(f"   Unique players: {df_props['player'].nunique():,}")
    print(f"   Markets: {df_props['market'].unique().tolist()}")
    
    return df_props


def load_all_game_logs(season):
    """Load ALL player game logs for the season from S3"""
    print(f"\n🏀 Loading game logs from s3://{S3_BUCKET_NBA}/player_game_logs/{season}/...")
    
    s3_client = boto3.client('s3')
    prefix = f"player_game_logs/{season}/"
    
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"❌ No game log files found")
        return pd.DataFrame()
    
    all_game_logs = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv'):
            try:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_NBA, Key=obj['Key'])
                df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                all_game_logs.append(df)
            except Exception as e:
                print(f"  ⚠️  Error loading {obj['Key']}: {e}")
    
    df_games = pd.concat(all_game_logs, ignore_index=True)
    
    # Parse game date
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'], format='mixed')
    df_games['game_date'] = df_games['GAME_DATE'].dt.date.astype(str)
    
    # Add normalized names - keep both for debugging
    df_games['PLAYER_NAME_NORMALIZED'] = df_games['PLAYER_NAME'].apply(normalize_player_name)
    df_games['player_normalized'] = df_games['PLAYER_NAME_NORMALIZED']  # Used for joining with props
    
    # Normalize team names for joining with game lines
    df_games['TEAM_NAME'] = df_games['TEAM_NAME'].apply(normalize_team_name)
    
    # Filter to players who actually played
    df_games = df_games[df_games['MIN'].notna() & (df_games['MIN'] > 0)].copy()
    
    print(f"✅ Loaded {len(df_games):,} player-game rows from {len(all_game_logs)} files")
    print(f"   Dates: {df_games['game_date'].min()} to {df_games['game_date'].max()}")
    print(f"   Unique players: {df_games['PLAYER_NAME'].nunique():,}")
    
    return df_games


def load_all_shot_charts(season):
    """
    Load ALL shot charts for the season from S3 (aggregated by player)
    
    Calculates season-long stats including:
    - Total shots and rim shots (0-6 feet)
    - Rim FG%
    - Total points scored within 6 feet (rim_season_points)
    """
    print(f"\n🎯 Loading shot charts from s3://{S3_BUCKET_NBA}/player_shot_charts/{season}/...")
    
    s3_client = boto3.client('s3')
    prefix = f"player_shot_charts/{season}/"
    
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"❌ No shot chart files found")
        return pd.DataFrame()
    
    all_shot_data = []
    
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv'):
            try:
                # Extract player name from filename
                file_name = obj['Key'].split('/')[-1].replace('.csv', '')
                parts = file_name.split('_')
                
                if len(parts) >= 2:
                    player_name_raw = ' '.join(parts[:-1])  # Everything except player ID
                    player_normalized = normalize_player_name(player_name_raw)
                    
                    # Load shot data
                    obj_data = s3_client.get_object(Bucket=S3_BUCKET_NBA, Key=obj['Key'])
                    df_shots = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                    
                    # Aggregate shot stats (0-6 feet = rim/close range)
                    rim_shots = df_shots[df_shots['SHOT_DISTANCE'] <= 6]
                    
                    # Calculate points from 0-6 feet (all are 2-pointers)
                    rim_makes = rim_shots['SHOT_MADE_FLAG'].sum() if not rim_shots.empty else 0
                    rim_points = rim_makes * 2  # All shots within 6 feet are 2-pointers
                    
                    all_shot_data.append({
                        'player_normalized': player_normalized,
                        'total_season_shots': len(df_shots),
                        'rim_season_shots': len(rim_shots),
                        'rim_season_makes': rim_makes,
                        'rim_season_points': rim_points,  # NEW: Points scored 0-6 feet
                        'rim_fg_pct': (rim_shots['SHOT_MADE_FLAG'].mean() * 100) if not rim_shots.empty else 0
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading {obj['Key']}: {e}")
    
    df_shots = pd.DataFrame(all_shot_data)
    
    print(f"✅ Loaded {len(df_shots):,} player shot chart aggregations")
    print(f"   Average rim FG%: {df_shots['rim_fg_pct'].mean():.1f}%")
    print(f"   Average rim points per player: {df_shots['rim_season_points'].mean():.1f}")
    
    return df_shots


def load_all_game_lines(season):
    """Load ALL game lines (spreads, moneylines) for the season from S3"""
    print(f"\n📈 Loading game lines from s3://{S3_BUCKET_PROPS}/nba/historical_game_lines/{season}/...")
    
    s3_client = boto3.client('s3')
    prefix = f"nba/historical_game_lines/{season}/"
    
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"❌ No game line files found")
        return pd.DataFrame()
    
    all_lines = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv') and 'nba_game_lines' in obj['Key']:
            try:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_PROPS, Key=obj['Key'])
                df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                
                # Extract date from filename (ET-based, more reliable than parsing UTC game_time)
                # Filename format: nba_game_lines_YYYY-MM-DD.csv
                filename = obj['Key'].split('/')[-1]
                date_str = filename.replace('nba_game_lines_', '').replace('.csv', '')
                df['game_date'] = date_str
                
                # Convert game_time from UTC to ET for reference
                from zoneinfo import ZoneInfo
                game_time_parsed = pd.to_datetime(df['game_time'])
                if game_time_parsed.dt.tz is None:
                    # If tz-naive, assume UTC
                    game_time_parsed = game_time_parsed.dt.tz_localize('UTC')
                df['gametime_et'] = game_time_parsed.dt.tz_convert(ZoneInfo('America/New_York'))
                
                all_lines.append(df)
            except Exception as e:
                print(f"  ⚠️  Error loading {obj['Key']}: {e}")
    
    df_lines = pd.concat(all_lines, ignore_index=True)
    
    # Calculate consensus by averaging across bookmakers for each game/market
    consensus = df_lines.groupby(['game_id', 'game_date', 'away_team', 'home_team', 'market']).agg({
        'away_line': 'mean',
        'away_odds': 'mean',
        'home_line': 'mean',
        'home_odds': 'mean'
    }).reset_index()
    
    # Pivot to get spread and moneyline as separate columns
    spread = consensus[consensus['market'] == 'spread'][['game_id', 'game_date', 'away_team', 'home_team', 'away_line', 'away_odds', 'home_line', 'home_odds']]
    spread.columns = ['game_id', 'game_date', 'away_team', 'home_team', 'away_spread', 'away_spread_odds', 'home_spread', 'home_spread_odds']
    
    moneyline = consensus[consensus['market'] == 'moneyline'][['game_id', 'game_date', 'away_team', 'home_team', 'away_odds', 'home_odds']]
    moneyline.columns = ['game_id', 'game_date', 'away_team', 'home_team', 'away_moneyline', 'home_moneyline']
    
    # Merge spread and moneyline
    df_consensus = spread.merge(moneyline, on=['game_id', 'game_date', 'away_team', 'home_team'], how='outer')
    
    print(f"✅ Loaded {len(df_consensus):,} unique games with consensus lines")
    print(f"   Dates: {df_consensus['game_date'].min()} to {df_consensus['game_date'].max()}")
    print(f"   Unique games: {df_consensus['game_id'].nunique():,}")
    
    return df_consensus


def calculate_scorer_type(df_shots, df_games, rim_scorer_pct=50):
    """
    Calculate scorer type (rim attacker vs perimeter) by joining shot charts with game logs
    
    Args:
        df_shots: Shot chart aggregations (has rim_season_points)
        df_games: Game logs (has PTS per game)
        rim_scorer_pct: Percentage threshold to classify as rim attacker (default: 50)
    
    Returns:
        Enhanced shot chart DataFrame with pts_0_6_pct and scorer_type classification
    """
    print(f"\n🎯 Calculating scorer type classification (0-6 feet vs total points, rim_scorer_pct={rim_scorer_pct}%)...")
    
    # Aggregate game logs to get total season points per player
    player_season_points = df_games.groupby('player_normalized').agg({
        'PTS': 'sum'  # Total points across all games
    }).reset_index()
    player_season_points.columns = ['player_normalized', 'total_pts_season']
    
    print(f"   Players with game logs: {len(player_season_points):,}")
    
    # Join shot charts with season points
    df_shots_enhanced = df_shots.merge(
        player_season_points,
        on='player_normalized',
        how='left'
    )
    
    # Calculate percentage of points from 0-6 feet
    df_shots_enhanced['pts_0_6_pct'] = (
        df_shots_enhanced['rim_season_points'] / df_shots_enhanced['total_pts_season'] * 100
    ).fillna(0)
    
    # Classify as rim attacker or perimeter based on rim_scorer_pct
    rim_label = f'Rim Attacker (≥{rim_scorer_pct}%)'
    perimeter_label = f'Perimeter (<{rim_scorer_pct}%)'
    
    df_shots_enhanced['scorer_type'] = df_shots_enhanced['pts_0_6_pct'].apply(
        lambda x: rim_label if x >= rim_scorer_pct else perimeter_label
    )
    
    # Add threshold as column for reproducibility
    df_shots_enhanced['rim_scorer_threshold'] = rim_scorer_pct
    
    # Stats
    rim_attackers = (df_shots_enhanced['scorer_type'] == rim_label).sum()
    perimeter = (df_shots_enhanced['scorer_type'] == perimeter_label).sum()
    
    print(f"✅ Scorer type classification complete:")
    print(f"   {rim_label} from 0-6 ft: {rim_attackers} ({rim_attackers/len(df_shots_enhanced)*100:.1f}%)")
    print(f"   {perimeter_label} from 0-6 ft: {perimeter} ({perimeter/len(df_shots_enhanced)*100:.1f}%)")
    print(f"   Average pts from 0-6 ft: {df_shots_enhanced['pts_0_6_pct'].mean():.1f}%")
    
    return df_shots_enhanced


def join_all_data(season, rim_scorer_pct=None):
    """
    Load and join all 4 datasets (props, game logs, shot charts, game lines)
    
    Args:
        season: NBA season (e.g., '2025-26')
        rim_scorer_pct: Percentage threshold to classify as rim attacker (default: None, no classification)
    
    Returns:
        Merged DataFrame
    """
    print(f"\n{'='*80}")
    print(f"LOADING ALL DATA FOR {season}")
    print(f"{'='*80}\n")
    
    # Load all 4 datasets
    df_props = load_all_props(season)
    df_games = load_all_game_logs(season)
    df_shots = load_all_shot_charts(season)
    df_lines = load_all_game_lines(season)
    
    # Calculate scorer type classification (rim attacker vs perimeter)
    if rim_scorer_pct is not None and not df_shots.empty and not df_games.empty:
        # Full classification with specific threshold
        df_shots = calculate_scorer_type(df_shots, df_games, rim_scorer_pct)
    elif not df_shots.empty:
        # Add columns with NULL values to maintain consistent schema
        print(f"\n🎯 Adding scorer type columns (NULL) for schema consistency...")
        df_shots['total_pts_season'] = None
        df_shots['pts_0_6_pct'] = None
        df_shots['scorer_type'] = None
        df_shots['rim_scorer_threshold'] = None
        print(f"   ✅ Schema columns added (values are NULL)")
    
    print(f"\n{'='*80}")
    print(f"JOINING DATA")
    print(f"{'='*80}\n")
    
    # Start with game logs (players who actually played)
    print("Starting with game logs as base...")
    df_merged = df_games.copy()
    
    # Filter to player_points only
    if not df_props.empty:
        print("Filtering to player_points market only...")
        df_props = df_props[df_props['market'] == 'player_points'].copy()
        print(f"   {len(df_props):,} player_points prop rows")
        
        # Aggregate props by player/date
        print("Aggregating props by player/date...")
        props_agg = df_props.groupby(['player_normalized', 'game_date']).agg({
            'prop_line': 'mean',      # Average line across bookmakers
            'over_odds': 'median',    # Median odds (can't average American odds!)
            'under_odds': 'median',   # Median odds (can't average American odds!)
            'bookmaker': 'count'
        }).reset_index()
        
        props_agg.columns = ['player_normalized', 'game_date', 'points_line', 'points_over_odds', 'points_under_odds', 'num_bookmakers']
        
        # Left join props
        print("Left joining player_points props to game logs...")
        df_merged = df_merged.merge(
            props_agg,
            on=['player_normalized', 'game_date'],
            how='left'
        )
        
        print(f"✅ Player points props joined")
    
    # Left join shot charts (now includes pts_0_6_pct and scorer_type)
    if not df_shots.empty:
        print("Left joining shot charts with scorer type classification...")
        df_merged = df_merged.merge(
            df_shots,
            on='player_normalized',
            how='left'
        )
        
        print(f"✅ Shot charts joined (includes pts_0_6_pct and scorer_type)")
    
    # Join game lines (need to match team name to away/home)
    if not df_lines.empty:
        print("Joining game lines...")
        
        # Create lookup for away teams
        away_lines = df_lines[['game_date', 'away_team', 'away_spread', 'away_spread_odds', 'away_moneyline', 'home_team']].copy()
        away_lines.columns = ['game_date', 'team_name', 'team_spread', 'team_spread_odds', 'team_moneyline', 'opponent_team']
        away_lines['is_home'] = False
        
        # Create lookup for home teams
        home_lines = df_lines[['game_date', 'home_team', 'home_spread', 'home_spread_odds', 'home_moneyline', 'away_team']].copy()
        home_lines.columns = ['game_date', 'team_name', 'team_spread', 'team_spread_odds', 'team_moneyline', 'opponent_team']
        home_lines['is_home'] = True
        
        # Combine
        team_lines = pd.concat([away_lines, home_lines], ignore_index=True)
        
        # Add is_favorite flag (negative spread means favorite)
        # Note: Only use spread, not moneyline, as spread is the reliable indicator
        team_lines['is_favorite'] = team_lines['team_spread'] < 0
        
        # Join to merged data
        df_merged = df_merged.merge(
            team_lines,
            left_on=['game_date', 'TEAM_NAME'],
            right_on=['game_date', 'team_name'],
            how='left'
        )
        
        # Drop duplicate team_name column
        df_merged = df_merged.drop('team_name', axis=1)
        
        print(f"✅ Game lines joined")
    
    # Reorder columns for better readability
    # Put PLAYER_NAME_NORMALIZED right after PLAYER_NAME
    cols = df_merged.columns.tolist()
    if 'PLAYER_NAME_NORMALIZED' in cols and 'PLAYER_NAME' in cols:
        # Remove PLAYER_NAME_NORMALIZED from its current position
        cols.remove('PLAYER_NAME_NORMALIZED')
        # Insert it right after PLAYER_NAME
        player_name_idx = cols.index('PLAYER_NAME')
        cols.insert(player_name_idx + 1, 'PLAYER_NAME_NORMALIZED')
        df_merged = df_merged[cols]
    
    print(f"\n{'='*80}")
    print(f"FINAL MERGED DATASET")
    print(f"{'='*80}")
    print(f"Total rows: {len(df_merged):,}")
    print(f"Total columns: {len(df_merged.columns)}")
    print(f"Date range: {df_merged['game_date'].min()} to {df_merged['game_date'].max()}")
    
    # Show scorer type distribution per unique player
    if 'scorer_type' in df_merged.columns:
        print(f"\n📊 Scorer Type Distribution:")
        
        # Per player (unique)
        player_scorer_type = df_merged[['player_normalized', 'scorer_type']].drop_duplicates()
        scorer_dist_players = player_scorer_type['scorer_type'].value_counts()
        total_players = len(player_scorer_type)
        
        print(f"   Per Player ({total_players:,} unique players):")
        for scorer_type, count in scorer_dist_players.items():
            pct = count / total_players * 100
            print(f"      {scorer_type}: {count:,} players ({pct:.1f}%)")
        
        # Per player-game (for context)
        scorer_dist_games = df_merged['scorer_type'].value_counts()
        print(f"\n   Per Player-Game ({len(df_merged):,} total games):")
        for scorer_type, count in scorer_dist_games.items():
            pct = count / len(df_merged) * 100
            print(f"      {scorer_type}: {count:,} games ({pct:.1f}%)")
    
    return df_merged


def upload_merged_to_s3(df, season, rim_scorer_pct=None):
    """
    Upload merged DataFrame to S3.
    
    Args:
        df: Merged DataFrame
        season: Season string (e.g., '2025-26')
        rim_scorer_pct: Rim scorer percentage threshold (e.g., 40 or 50), optional
    
    Returns:
        True if successful, False otherwise
    """
    # Use the rim scorer threshold parameter that was passed
    # (not inferred from data, which could be NULL for first row)
    rim_threshold = int(rim_scorer_pct) if rim_scorer_pct is not None else None
    
    # Build filename with threshold suffix only if threshold was actually used
    if rim_threshold is not None:
        filename = f"player_props_with_actuals_{season}_rim{rim_threshold}.csv"
    else:
        filename = f"player_props_with_actuals_{season}.csv"
    
    s3_key = f"{S3_PREFIX_OUTPUT}/{filename}"
    
    try:
        s3_client = boto3.client('s3')
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET_OUTPUT,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"\n💾 Uploaded merged data to S3: s3://{S3_BUCKET_OUTPUT}/{s3_key}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        if rim_threshold is not None:
            print(f"   Rim scorer threshold: {rim_threshold}%")
        return True
        
    except Exception as e:
        print(f"\n⚠️  S3 upload failed: {e}")
        return False


def calculate_null_percentages(df):
    """Calculate and display null percentages for all columns"""
    print(f"\n{'='*80}")
    print(f"NULL PERCENTAGE ANALYSIS")
    print(f"{'='*80}\n")
    
    null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    
    # Show columns with any nulls
    cols_with_nulls = null_pcts[null_pcts > 0]
    
    if len(cols_with_nulls) > 0:
        print(f"Columns with NULL values ({len(cols_with_nulls)} of {len(df.columns)}):\n")
        for col, pct in cols_with_nulls.items():
            print(f"  {col:.<60} {pct:>6.2f}%")
    else:
        print("✅ No NULL values in any column!")
    
    print(f"\n{'='*80}")
    print(f"Columns with 0% NULL ({len(null_pcts[null_pcts == 0])} of {len(df.columns)})")
    print(f"{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Load and join NBA player props data')
    parser.add_argument('--season', default='2025-26', help='NBA season (e.g., 2025-26)')
    parser.add_argument('--save', help='Save merged data to local CSV file (provide path)')
    parser.add_argument('--s3', action='store_true', help='Upload merged data to S3')
    parser.add_argument('--rim-scorer-pct', type=float, default=None, 
                        help='Percentage threshold to classify as rim attacker (optional, e.g., 40 or 50)')
    
    args = parser.parse_args()
    
    # Load and join all data
    df_merged = join_all_data(args.season, args.rim_scorer_pct)
    
    # Show null percentages
    calculate_null_percentages(df_merged)
    
    # Upload to S3 if requested
    if args.s3 and not df_merged.empty:
        upload_merged_to_s3(df_merged, args.season, args.rim_scorer_pct)
    
    # Save locally if requested
    if args.save and not df_merged.empty:
        from pathlib import Path
        save_path = Path(args.save).resolve()
        print(f"\n💾 Saving locally to {save_path}...")
        df_merged.to_csv(save_path, index=False)
        print(f"✅ Saved {len(df_merged):,} rows")
        print(f"   Location: {save_path}")


if __name__ == '__main__':
    main()

