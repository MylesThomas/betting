"""
Team-Specific ATS Analysis with Game-by-Game Logging

Analyzes a specific team's ATS performance with detailed game-by-game breakdowns.
Separates regular season vs playoffs to see how team performed against book expectations.

Usage:
    python tmp/analyze_ats_records.py --sport nba --team "Indiana Pacers" --season 2024-25 --start-date 2024-10-24 --end-date 2025-06-22 --full-logging-by-game

Context:
    Created to analyze playoff teams' ATS performance vs regular season.
    Helps identify if books properly adjusted lines for playoff performance.
    
Author: Thomas Myles
Date: 2026-01-19
"""

import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import os
import argparse
from pathlib import Path

# Add src to path for config
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from season_utils import parse_season_to_years, season_to_underscore

# Constants
S3_BUCKET_ODDS = 'the-odds-api-mt'
S3_BUCKET_NBA_API = 'nba-api-mt'

# Team name mapping (Odds API → nba_api)
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers',
}

# NBA Playoff start dates (approximate - play-in starts mid-April)
NBA_PLAYOFF_START_DATES = {
    '2024-25': '2025-04-15',  # Play-in tournament starts
    '2023-24': '2024-04-16',
    '2022-23': '2023-04-15',
}


def normalize_team_name(team_name, reverse=False):
    """Normalize team names between Odds API and nba_api"""
    if reverse:
        reverse_map = {v: k for k, v in TEAM_NAME_MAP.items()}
        return reverse_map.get(team_name, team_name)
    else:
        return TEAM_NAME_MAP.get(team_name, team_name)


def load_all_game_lines(season):
    """Load all NBA game lines for specified season from S3 (S3: the-odds-api-mt/nba/historical_game_lines/{season}/)"""
    s3_prefix = f'nba/historical_game_lines/{season}/'
    
    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET_ODDS, Prefix=s3_prefix)
    
    all_dfs = []
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        
        if not key.endswith('.csv') or 'failed' in key.lower():
            continue
        
        try:
            response = s3.get_object(Bucket=S3_BUCKET_ODDS, Key=key)
            df = pd.read_csv(BytesIO(response['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        raise ValueError("No game lines found in S3")
    
    df = pd.concat(all_dfs, ignore_index=True)
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['odds_pull_time'] = pd.to_datetime(df['odds_pull_time'])
    
    return df


def get_nba_scores(season):
    """Load NBA game scores from S3 ESPN game results"""
    # ESPN API stores game results in a different format
    s3_prefix = 'data/01_input/historical_game_results/'
    s3_bucket = 'nba-betting-mt'
    
    s3 = boto3.client('s3')
    
    print(f"   📂 Looking for ESPN game results: s3://{s3_bucket}/{s3_prefix}")
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    
    if 'Contents' not in response:
        raise ValueError(f"No game results found in S3")
    
    all_dfs = []
    file_count = 0
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        
        if not key.endswith('.csv'):
            continue
        
        try:
            obj_response = s3.get_object(Bucket=s3_bucket, Key=key)
            df = pd.read_csv(BytesIO(obj_response['Body'].read()))
            
            # Skip empty files (no games on that date)
            if df.empty:
                continue
                
            all_dfs.append(df)
            file_count += 1
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        raise ValueError(f"No valid game results found")
    
    print(f"   ✅ Loaded {file_count} daily ESPN game result files")
    
    # Combine all ESPN data
    results_df = pd.concat(all_dfs, ignore_index=True)
    results_df['GAME_DATE'] = pd.to_datetime(results_df['GAME_DATE'])
    
    # Convert ESPN format to our expected format
    # Need to create records for both home and away teams
    team_games = []
    
    for _, row in results_df.iterrows():
        # Home team record
        team_games.append({
            'team': row['HOME_TEAM'],
            'game_date': row['GAME_DATE'],
            'matchup': f"vs {row['AWAY_TEAM']}",  # Home game
            'win_loss': row['HOME_WL'],
            'points': row['HOME_SCORE']
        })
        
        # Away team record
        team_games.append({
            'team': row['AWAY_TEAM'],
            'game_date': row['GAME_DATE'],
            'matchup': f"@ {row['HOME_TEAM']}",  # Away game
            'win_loss': row['AWAY_WL'],
            'points': row['AWAY_SCORE']
        })
    
    team_games_df = pd.DataFrame(team_games)
    return team_games_df


def calculate_team_ats_records(lines_df, scores_df, team_name):
    """Calculate ATS records for a specific team"""
    spreads = lines_df[lines_df['market'] == 'spread'].copy()
    spreads = spreads[spreads['bookmaker_key'] == 'draftkings']
    
    if spreads.empty:
        bookmaker = lines_df[lines_df['market'] == 'spread']['bookmaker_key'].iloc[0]
        spreads = lines_df[(lines_df['market'] == 'spread') & 
                          (lines_df['bookmaker_key'] == bookmaker)].copy()
        print(f"ℹ️  Using {bookmaker} lines (DraftKings not available)")
    
    games = spreads.groupby('game_id').first().reset_index()
    
    et_tz = ZoneInfo('America/New_York')
    
    results = []
    
    for _, game in games.iterrows():
        game_time_utc = pd.to_datetime(game['game_time'])
        if game_time_utc.tzinfo is None:
            game_time_utc = game_time_utc.tz_localize('UTC')
        game_time_et = game_time_utc.tz_convert(et_tz)
        game_date = game_time_et.date()
        
        away_team = normalize_team_name(game['away_team'])
        home_team = normalize_team_name(game['home_team'])
        away_spread = game['away_line']
        home_spread = game['home_line']
        
        # Skip if this game doesn't involve our team
        if team_name not in [away_team, home_team]:
            continue
        
        away_score_row = scores_df[
            (scores_df['team'] == away_team) & 
            (scores_df['game_date'].dt.date == game_date)
        ]
        
        home_score_row = scores_df[
            (scores_df['team'] == home_team) & 
            (scores_df['game_date'].dt.date == game_date)
        ]
        
        if away_score_row.empty or home_score_row.empty:
            continue
        
        away_points = away_score_row.iloc[0]['points']
        home_points = home_score_row.iloc[0]['points']
        
        actual_margin = away_points - home_points
        
        # Determine which team is ours and calculate ATS result
        if team_name == away_team:
            team_spread = away_spread
            team_location = 'Away'
            opponent = home_team
            team_points = away_points
            opp_points = home_points
            team_margin = actual_margin
            ats_margin = actual_margin + away_spread
        else:  # team_name == home_team
            team_spread = home_spread
            team_location = 'Home'
            opponent = away_team
            team_points = home_points
            opp_points = away_points
            team_margin = -actual_margin
            ats_margin = -actual_margin + home_spread
        
        if ats_margin > 0:
            ats_result = 'W'
        elif ats_margin < 0:
            ats_result = 'L'
        else:
            ats_result = 'P'
        
        fav_dog = 'Favorite' if team_spread < 0 else 'Underdog' if team_spread > 0 else 'Pick'
        
        results.append({
            'team': team_name,
            'game_date': game_date,
            'opponent': opponent,
            'location': team_location,
            'spread': team_spread,
            'actual_margin': team_margin,
            'ats_margin': ats_margin,
            'ats_result': ats_result,
            'fav_dog': fav_dog,
            'team_points': team_points,
            'opp_points': opp_points,
            'game_time': game_time_et
        })
    
    return pd.DataFrame(results)


def print_game_by_game(df, team_name, start_date=None, end_date=None):
    """Print detailed game-by-game ATS results"""
    if df.empty:
        print(f"\n❌ No games found for {team_name}")
        return
    
    # Sort by date
    df = df.sort_values('game_date').copy()
    
    print(f"\n{'='*140}")
    print(f"📋 GAME-BY-GAME ATS RESULTS: {team_name}")
    if start_date or end_date:
        print(f"{'='*140}")
        print(f"Date Range: {start_date or 'Season Start'} to {end_date or 'Season End'}")
    print(f"{'='*140}\n")
    
    print(f"{'Date':<12} {'Opp':<25} {'Loc':<6} {'Score':<12} {'Margin':<8} "
          f"{'Spread':<8} {'ATS Margin':<12} {'Result':<8} {'Role':<12}")
    print(f"{'-'*140}")
    
    for _, row in df.iterrows():
        date_str = row['game_date'].strftime('%Y-%m-%d')
        
        # Format score
        if row['location'] == 'Home':
            score = f"{int(row['team_points'])}-{int(row['opp_points'])}"
        else:
            score = f"{int(row['team_points'])}-{int(row['opp_points'])}"
        
        # Color code result
        if row['ats_result'] == 'W':
            result_icon = '✅ W'
        elif row['ats_result'] == 'L':
            result_icon = '❌ L'
        else:
            result_icon = '➖ P'
        
        print(f"{date_str:<12} {row['opponent']:<25} {row['location']:<6} {score:<12} "
              f"{row['actual_margin']:+.1f}{'':<7} {row['spread']:+.1f}{'':<7} "
              f"{row['ats_margin']:+.1f}{'':<11} {result_icon:<8} {row['fav_dog']:<12}")


def print_summary_stats(df, label):
    """Print summary statistics for a set of games"""
    if df.empty:
        print(f"\n{label}: No games")
        return
    
    wins = len(df[df['ats_result'] == 'W'])
    losses = len(df[df['ats_result'] == 'L'])
    pushes = len(df[df['ats_result'] == 'P'])
    total = wins + losses
    
    win_pct = wins / total if total > 0 else 0
    avg_ats_margin = df['ats_margin'].mean()
    avg_spread = df['spread'].mean()
    avg_game_margin = df['actual_margin'].mean()
    
    # Favorite/Underdog splits
    fav_games = df[df['fav_dog'] == 'Favorite']
    dog_games = df[df['fav_dog'] == 'Underdog']
    
    fav_wins = len(fav_games[fav_games['ats_result'] == 'W'])
    fav_total = len(fav_games[fav_games['ats_result'] != 'P'])
    fav_pct = fav_wins / fav_total if fav_total > 0 else 0
    
    dog_wins = len(dog_games[dog_games['ats_result'] == 'W'])
    dog_total = len(dog_games[dog_games['ats_result'] != 'P'])
    dog_pct = dog_wins / dog_total if dog_total > 0 else 0
    
    print(f"\n{'='*100}")
    print(f"📊 {label}")
    print(f"{'='*100}")
    print(f"ATS Record:        {wins}-{losses}-{pushes} ({win_pct:.1%})")
    print(f"Avg Spread:        {avg_spread:+.1f}")
    print(f"Avg Game Margin:   {avg_game_margin:+.1f}")
    print(f"Avg ATS Margin:    {avg_ats_margin:+.1f}")
    print(f"\nAs Favorite:       {fav_wins}-{fav_total - fav_wins} ({fav_pct:.1%}) in {fav_total} games")
    print(f"As Underdog:       {dog_wins}-{dog_total - dog_wins} ({dog_pct:.1%}) in {dog_total} games")
    
    # Actual win/loss record
    actual_wins = len(df[df['team_points'] > df['opp_points']])
    actual_losses = len(df[df['team_points'] < df['opp_points']])
    actual_pct = actual_wins / (actual_wins + actual_losses) if (actual_wins + actual_losses) > 0 else 0
    print(f"\nActual Record:     {actual_wins}-{actual_losses} ({actual_pct:.1%})")
    
    # Value indicator
    value_diff = win_pct - actual_pct
    if value_diff > 0.05:
        print(f"\n💎 BETTING VALUE: {value_diff:.1%} better ATS than actual record")
    elif value_diff < -0.05:
        print(f"\n⚠️  BETTING TRAP: {abs(value_diff):.1%} worse ATS than actual record")


def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(
        description='Analyze team-specific ATS performance with game-by-game details',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pacers full season with game logs
  python tmp/analyze_ats_records.py --sport nba --team "Indiana Pacers" --season 2024-25 --start-date 2024-10-24 --end-date 2025-06-22 --full-logging-by-game
  
  # Pacers regular season only
  python tmp/analyze_ats_records.py --sport nba --team "Indiana Pacers" --season 2024-25 --start-date 2024-10-24 --end-date 2025-04-13
  
  # Pacers playoffs only
  python tmp/analyze_ats_records.py --sport nba --team "Indiana Pacers" --season 2024-25 --start-date 2025-04-15 --end-date 2025-06-22
        """
    )
    parser.add_argument('--sport', type=str, required=True, choices=['nba'], help='Sport (currently only nba supported)')
    parser.add_argument('--team', type=str, required=True, help='Team name (e.g., "Indiana Pacers")')
    parser.add_argument('--season', type=str, required=True, help='Season in YYYY-YY format (e.g., 2024-25)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--full-logging-by-game', action='store_true', help='Show game-by-game breakdown')
    
    args = parser.parse_args()
    
    print(f"\n{'='*100}")
    print(f"🏀 LOADING {args.team.upper()} ATS DATA")
    print(f"{'='*100}")
    print(f"Season: {args.season}")
    if args.start_date or args.end_date:
        print(f"Date Range: {args.start_date or 'Season Start'} to {args.end_date or 'Season End'}")
    print(f"{'='*100}")
    
    # Load betting lines from S3
    print("\n📥 Loading betting lines from S3... (S3: the-odds-api-mt/nba/historical_game_lines/{season}/)")
    lines_df = load_all_game_lines(args.season)
    print(f"✅ Loaded {len(lines_df):,} betting lines")
    
    # Load game results
    print("\n📥 Loading game results from ESPN API... (S3: nba-betting-mt/data/01_input/historical_game_results/)")
    scores_df = get_nba_scores(args.season)
    print(f"✅ Loaded results for {len(scores_df):,} team-games")
    
    # Calculate ATS records for this team
    print(f"\n📊 Calculating ATS records for {args.team}...")
    ats_df = calculate_team_ats_records(lines_df, scores_df, args.team)
    
    if ats_df.empty:
        print(f"\n❌ No games found for {args.team}")
        return
    
    print(f"✅ Found {len(ats_df):,} games for {args.team}")
    
    # Filter by date range if specified
    if args.start_date or args.end_date:
        original_count = len(ats_df)
        
        if args.start_date:
            start_date = pd.to_datetime(args.start_date).date()
            ats_df = ats_df[ats_df['game_date'] >= start_date]
        
        if args.end_date:
            end_date = pd.to_datetime(args.end_date).date()
            ats_df = ats_df[ats_df['game_date'] <= end_date]
        
        filtered_count = len(ats_df)
        print(f"   🗓️  {filtered_count} games in date range (filtered out {original_count - filtered_count})")
    
    # Determine playoff start date
    playoff_start = NBA_PLAYOFF_START_DATES.get(args.season)
    
    if playoff_start:
        playoff_start_date = pd.to_datetime(playoff_start).date()
        regular_season_df = ats_df[ats_df['game_date'] < playoff_start_date]
        playoff_df = ats_df[ats_df['game_date'] >= playoff_start_date]
    else:
        regular_season_df = ats_df
        playoff_df = pd.DataFrame()
    
    # Show game-by-game if requested
    if args.full_logging_by_game:
        if not regular_season_df.empty:
            print(f"\n{'='*140}")
            print(f"🏀 REGULAR SEASON GAMES")
            print(f"{'='*140}")
            print_game_by_game(regular_season_df, args.team, args.start_date, args.end_date)
        
        if not playoff_df.empty:
            print(f"\n{'='*140}")
            print(f"🏆 PLAYOFF GAMES")
            print(f"{'='*140}")
            print_game_by_game(playoff_df, args.team, args.start_date, args.end_date)
    
    # Print summary stats
    if not regular_season_df.empty:
        print_summary_stats(regular_season_df, f"REGULAR SEASON SUMMARY: {args.team}")
    
    if not playoff_df.empty:
        print_summary_stats(playoff_df, f"PLAYOFF SUMMARY: {args.team}")
    
    # Overall summary
    print_summary_stats(ats_df, f"OVERALL SUMMARY: {args.team}")
    
    print(f"\n{'='*100}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
