"""
Fetch NBA team game results for current season using nba_api

This script fetches complete team-level game results (scores, W/L, stats) 
for the specified NBA season. Essential for calculating ATS records.

Saves to:
- Local: data/01_input/nba_api/historical/nba_games_{season}.csv
- S3: nba-results-mt/team_games/{season}/nba_games_{season}.csv

Author: Thomas Myles
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder
from datetime import datetime
import time
import sys
import os
from pathlib import Path
import argparse
import boto3
from io import StringIO
import ssl
import urllib3
import requests

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)

requests.Session.request = patched_request

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Constants
RATE_LIMIT_DELAY = 0.6  # NBA API rate limit
S3_BUCKET = 'nba-results-mt'  # New bucket for game results


def fetch_season_games(season_year):
    """
    Fetch all team games for a season
    
    Args:
        season_year: Year season starts (e.g., 2025 for 2025-26 season)
    
    Returns:
        DataFrame with all team games
    """
    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    
    print(f"\n📡 Fetching games for {season_str} season from nba_api...")
    print(f"   (This may take a minute, rate-limited to avoid API blocks)")
    
    try:
        # Use LeagueGameFinder to get all games for the season
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable='00',  # NBA
            season_type_nullable='Regular Season'
        )
        
        games = gamefinder.get_data_frames()[0]
        
        # Add playoff games
        time.sleep(RATE_LIMIT_DELAY)
        
        playoff_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable='00',
            season_type_nullable='Playoffs'
        )
        
        playoff_games = playoff_finder.get_data_frames()[0]
        
        # Combine
        if not playoff_games.empty:
            games = pd.concat([games, playoff_games], ignore_index=True)
            print(f"   ✅ Fetched {len(games)} team-games (including playoffs)")
        else:
            print(f"   ✅ Fetched {len(games)} team-games (regular season only)")
        
        return games
        
    except Exception as e:
        print(f"   ❌ Error fetching data: {e}")
        return pd.DataFrame()


def clean_game_data(df):
    """
    Clean and standardize game data
    
    Args:
        df: Raw games DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    # Convert game date to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Sort by date
    df = df.sort_values('GAME_DATE')
    
    # Add season year column
    df['SEASON_YEAR'] = df['SEASON_ID'].astype(str).str[1:5]
    
    return df


def save_to_local(df, season_year):
    """
    Save games data to local CSV
    
    Args:
        df: Games DataFrame
        season_year: Season start year
    
    Returns:
        Path to saved file
    """
    season_str = f"{season_year}_{str(season_year + 1)[-2:]}"
    
    output_dir = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'historical'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'nba_games_{season_str}.csv'
    
    df.to_csv(output_path, index=False)
    
    return output_path


def save_to_s3(df, season_year):
    """
    Save games data to S3
    
    Args:
        df: Games DataFrame
        season_year: Season start year
    
    Returns:
        S3 key
    """
    season_str = f"{season_year}_{str(season_year + 1)[-2:]}"
    s3_key = f"team_games/{season_str}/nba_games_{season_str}.csv"
    
    try:
        s3_client = boto3.client('s3')
        
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        return s3_key
        
    except Exception as e:
        print(f"⚠️  Failed to save to S3: {e}")
        print(f"   (This is okay if bucket doesn't exist yet)")
        return None


def print_summary(df, season_year):
    """
    Print summary statistics
    
    Args:
        df: Games DataFrame
        season_year: Season start year
    """
    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY - {season_str} Season")
    print(f"{'='*80}")
    
    # Basic stats
    num_games = len(df) // 2  # Divide by 2 since each game has 2 rows (home & away)
    num_teams = df['TEAM_NAME'].nunique()
    date_range = f"{df['GAME_DATE'].min().strftime('%Y-%m-%d')} to {df['GAME_DATE'].max().strftime('%Y-%m-%d')}"
    
    print(f"Games: {num_games:,}")
    print(f"Teams: {num_teams}")
    print(f"Date range: {date_range}")
    print(f"Total team-games: {len(df):,}")
    
    # Team breakdown
    print(f"\nGames per team:")
    team_counts = df.groupby('TEAM_NAME').size().sort_values(ascending=False)
    
    for team, count in team_counts.head(5).items():
        print(f"  {team}: {count} games")
    
    if len(team_counts) > 5:
        print(f"  ... and {len(team_counts) - 5} more teams")
    
    print(f"\n{'='*80}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Fetch NBA team game results for current/past seasons'
    )
    parser.add_argument(
        '--season',
        type=int,
        required=True,
        help='Season start year (e.g., 2025 for 2025-26 season)'
    )
    parser.add_argument(
        '--s3',
        action='store_true',
        help='Also save to S3 (requires bucket to exist)'
    )
    
    args = parser.parse_args()
    
    season_year = args.season
    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    
    print(f"\n{'='*80}")
    print(f"🏀 FETCHING NBA TEAM GAME RESULTS")
    print(f"{'='*80}")
    print(f"Season: {season_str}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print(f"{'='*80}")
    
    # Fetch data
    df = fetch_season_games(season_year)
    
    if df.empty:
        print("\n❌ No data retrieved. Exiting.")
        return
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df = clean_game_data(df)
    print(f"   ✅ Cleaned {len(df):,} team-games")
    
    # Save locally
    print("\n💾 Saving to local file...")
    local_path = save_to_local(df, season_year)
    print(f"   ✅ Saved to: {local_path}")
    
    # Save to S3 if requested
    if args.s3:
        print("\n☁️  Saving to S3...")
        s3_key = save_to_s3(df, season_year)
        if s3_key:
            print(f"   ✅ Saved to: s3://{S3_BUCKET}/{s3_key}")
    
    # Print summary
    print_summary(df, season_year)
    
    print(f"\n✅ COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

