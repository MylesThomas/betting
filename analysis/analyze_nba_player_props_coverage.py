"""
NBA Player Props Data Loader & Joiner

Loads and joins player props, game logs, and shot charts from S3.

Usage:
    python analysis/analyze_nba_player_props_coverage.py --season 2025-26
    python analysis/analyze_nba_player_props_coverage.py --season 2025-26 --save merged_data.csv

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
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
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
    """Load ALL shot charts for the season from S3 (aggregated by player)"""
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
                    
                    # Aggregate shot stats
                    rim_shots = df_shots[df_shots['SHOT_DISTANCE'] <= 6]
                    
                    all_shot_data.append({
                        'player_normalized': player_normalized,
                        'total_season_shots': len(df_shots),
                        'rim_season_shots': len(rim_shots),
                        'rim_season_makes': rim_shots['SHOT_MADE_FLAG'].sum() if not rim_shots.empty else 0,
                        'rim_fg_pct': (rim_shots['SHOT_MADE_FLAG'].mean() * 100) if not rim_shots.empty else 0
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading {obj['Key']}: {e}")
    
    df_shots = pd.DataFrame(all_shot_data)
    
    print(f"✅ Loaded {len(df_shots):,} player shot chart aggregations")
    print(f"   Average rim FG%: {df_shots['rim_fg_pct'].mean():.1f}%")
    
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


def join_all_data(season):
    """
    Load and join all 4 datasets (props, game logs, shot charts, game lines)
    
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
    
    # Left join shot charts
    if not df_shots.empty:
        print("Left joining shot charts...")
        df_merged = df_merged.merge(
            df_shots,
            on='player_normalized',
            how='left'
        )
        
        print(f"✅ Shot charts joined")
    
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
        
        # Add is_favorite flag (negative spread or moneyline means favorite)
        team_lines['is_favorite'] = (team_lines['team_spread'] < 0) | (team_lines['team_moneyline'] < 0)
        
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
    
    return df_merged


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
    
    args = parser.parse_args()
    
    # Load and join all data
    df_merged = join_all_data(args.season)
    
    # Show null percentages
    calculate_null_percentages(df_merged)
    
    # Save if requested
    if args.save and not df_merged.empty:
        from pathlib import Path
        save_path = Path(args.save).resolve()
        print(f"\n💾 Saving to {save_path}...")
        df_merged.to_csv(save_path, index=False)
        print(f"✅ Saved {len(df_merged):,} rows")
        print(f"   Location: {save_path}")


if __name__ == '__main__':
    main()

