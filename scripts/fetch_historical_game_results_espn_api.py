"""
Fetch Historical Game Results from ESPN API (All Sports)

Fetches completed game scores from ESPN's public scoreboard API for:
- NFL (professional football)
- NBA (professional basketball) 
- NCAAF (college football)
- NCAAB (college basketball)

Data Format:
Each CSV contains game-level results with columns:
- GAME_DATE: Date of the game (YYYY-MM-DD)
- GAME_ID: ESPN game ID
- HOME_TEAM: Full home team name
- HOME_ABBR: Home team abbreviation
- HOME_SCORE: Home team score
- HOME_WL: Home team W/L
- AWAY_TEAM: Full away team name
- AWAY_ABBR: Away team abbreviation
- AWAY_SCORE: Away team score
- AWAY_WL: Away team W/L
- POINT_DIFF: Point differential (home - away)
- WINNER: Winning team name
- MARGIN: Victory margin

S3 Upload Paths:
- NFL: s3://nfl-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
- NBA: s3://nba-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
- NCAAF: s3://ncaaf-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
- NCAAB: s3://ncaab-betting-mt/data/01_input/historical_game_results/{timestamp}.csv

Usage:
    # Fetch specific date (both formats supported)
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --date 20250105 --s3
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --date 2025-01-05 --s3
    
    # Test mode (save to ~/Downloads/tmp for review)
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --date 2026-01-01 --test
    python scripts/fetch_historical_game_results_espn_api.py --sport nba --date 2025-01-14 --test
    
    # Backfill season - INTERACTIVE MODE (will prompt for start/end dates)
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --s3
    # Prompts:
    #   Enter START date (YYYY-MM-DD) or press ENTER for default [2025-09-05]: 2025-09-05
    #   Enter END date (YYYY-MM-DD) or press ENTER for today [2026-01-15]: <ENTER>
    
    # Backfill season - NON-INTERACTIVE (provide dates via args)
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --start-date 2025-09-05 --s3
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --start-date 2025-09-05 --end-date 2026-02-01 --s3
    python scripts/fetch_historical_game_results_espn_api.py --sport nba --start-date 2025-10-22 --s3
    
    # Resume interrupted backfill (skip dates already in S3)
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --start-date 2025-09-05 --s3 --skip-existing
    
    # Console preview (no save)
    python scripts/fetch_historical_game_results_espn_api.py --sport nfl --date 2026-01-01

Default Season Start Dates (Regular Season Only - NO PRESEASON):
- NFL: September 5 (Week 1 typically starts first Thursday of Sept)
- NBA: October 22 (Regular season typically starts ~Oct 22)
- NCAAF: August 23 (Week 0 starts ~Aug 23-24)
- NCAAB: November 3 (Season starts ~Nov 3-4)

Context:
User request - "make this fetcher scalable please, like it can do all 4 sports"
Added interactive date prompts to avoid fetching preseason games.
Created to support line movement analysis across NFL/NBA/NCAAF/NCAAB by providing
standardized game results data from ESPN API.

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import argparse
import requests
import pandas as pd
import boto3
import urllib3
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Find project root
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config_loader import get_config

# Load config
CONFIG = get_config()


def get_current_season(sport):
    """
    Get current season for a sport from config.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
    
    Returns:
        str: Season string (e.g., '2025' for NFL, '2025-26' for NBA)
    """
    espn_seasons = CONFIG.get('espn_seasons', {})
    season = espn_seasons.get(sport)
    
    if not season:
        raise ValueError(f"No season configured for sport: {sport}")
    
    # Convert season format based on sport
    if sport in ['nfl', 'ncaaf']:
        # NFL/NCAAF uses single year (2025)
        return str(season)
    else:
        # NBA/NCAAB uses YYYY-YY format (2026 → 2025-26)
        start_year = season - 1
        end_year_suffix = str(season)[-2:]
        return f"{start_year}-{end_year_suffix}"

# ESPN API Configuration
ESPN_SCOREBOARD_BASE = 'https://site.api.espn.com/apis/site/v2/sports'

SPORT_CONFIG = {
    'nfl': {
        'espn_sport': 'football',
        'espn_league': 'nfl',
        'season_format': 'YYYY',  # 2025 season = Sept 2025 - Feb 2026
        's3_bucket': 'nfl-betting-mt',
        's3_path_template': 'data/01_input/historical_game_results/',
    },
    'nba': {
        'espn_sport': 'basketball',
        'espn_league': 'nba',
        'season_format': 'YYYY-YY',  # 2025-26 season = Oct 2025 - June 2026
        's3_bucket': 'nba-betting-mt',
        's3_path_template': 'data/01_input/historical_game_results/',
    },
    'ncaaf': {
        'espn_sport': 'football',
        'espn_league': 'college-football',
        'season_format': 'YYYY',  # 2025 season
        's3_bucket': 'ncaaf-betting-mt',
        's3_path_template': 'data/01_input/historical_game_results/',
    },
    'ncaab': {
        'espn_sport': 'basketball',
        'espn_league': 'mens-college-basketball',
        'season_format': 'YYYY-YY',  # 2025-26 season
        's3_bucket': 'ncaab-betting-mt',
        's3_path_template': 'data/01_input/historical_game_results/',
    }
}


def fetch_espn_scoreboard(sport, date_str):
    """
    Fetch scoreboard data from ESPN API for a specific date.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        date_str: Date string in YYYYMMDD format
    
    Returns:
        dict: ESPN API response JSON
    """
    config = SPORT_CONFIG[sport]
    espn_sport = config['espn_sport']
    espn_league = config['espn_league']
    
    url = f"{ESPN_SCOREBOARD_BASE}/{espn_sport}/{espn_league}/scoreboard"
    params = {
        'dates': date_str,
        'limit': 500  # Increased from 300 to ensure we get all games
    }
    
    # For NCAAB, add groups=50 to get ALL games (not just featured game)
    # Testing showed this increases coverage from 1 game/day to 100+ games/day
    # ESPN with groups=50 returns MORE games than The Odds API (106-117% coverage)
    if sport == 'ncaab':
        params['groups'] = '50'
    
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching from ESPN API: {e}")
        return None


def parse_espn_games(espn_data, sport):
    """
    Parse ESPN scoreboard JSON into game-level results format.
    
    Args:
        espn_data: ESPN API response JSON
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
    
    Returns:
        pd.DataFrame with game-level data (one row per game, both teams' info)
    """
    if not espn_data or 'events' not in espn_data:
        return pd.DataFrame()
    
    rows = []
    
    for event in espn_data.get('events', []):
        # Parse game date (ISO format from ESPN in UTC, convert to ET)
        game_date_str = event.get('date', '')
        if game_date_str:
            # Parse as UTC datetime
            game_datetime_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
            # Convert to Eastern Time
            game_datetime_et = game_datetime_utc.astimezone(ZoneInfo('America/New_York'))
            # Extract date in ET
            game_date = game_datetime_et.date()
        else:
            continue
        
        # Get ESPN game ID
        game_id = event.get('id', 'unknown')
        
        # Get competition data
        competitions = event.get('competitions', [])
        if not competitions:
            continue
        
        comp = competitions[0]
        competitors = comp.get('competitors', [])
        
        if len(competitors) != 2:
            continue
        
        # Separate home and away teams
        home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        
        if not home_comp or not away_comp:
            continue
        
        # Parse home team
        home_team_info = home_comp.get('team', {})
        home_team_name = home_team_info.get('displayName', 'Unknown')
        home_team_abbr = home_team_info.get('abbreviation', '??')
        home_score_obj = home_comp.get('score', {})
        home_score = int(home_score_obj.get('value', 0)) if isinstance(home_score_obj, dict) else int(home_score_obj) if home_score_obj else 0
        home_winner = home_comp.get('winner', False)
        home_wl = 'W' if home_winner else 'L'
        
        # Parse away team
        away_team_info = away_comp.get('team', {})
        away_team_name = away_team_info.get('displayName', 'Unknown')
        away_team_abbr = away_team_info.get('abbreviation', '??')
        away_score_obj = away_comp.get('score', {})
        away_score = int(away_score_obj.get('value', 0)) if isinstance(away_score_obj, dict) else int(away_score_obj) if away_score_obj else 0
        away_winner = away_comp.get('winner', False)
        away_wl = 'W' if away_winner else 'L'
        
        # Calculate point differential (home - away perspective)
        point_diff = home_score - away_score
        
        # Determine winner
        if home_winner:
            winner_name = home_team_name
            margin = abs(point_diff)
        elif away_winner:
            winner_name = away_team_name
            margin = abs(point_diff)
        else:
            winner_name = 'TIE'
            margin = 0
        
        rows.append({
            'GAME_DATE': game_date,
            'GAME_ID': game_id,
            'HOME_TEAM': home_team_name,
            'HOME_ABBR': home_team_abbr,
            'HOME_SCORE': home_score,
            'HOME_WL': home_wl,
            'AWAY_TEAM': away_team_name,
            'AWAY_ABBR': away_team_abbr,
            'AWAY_SCORE': away_score,
            'AWAY_WL': away_wl,
            'POINT_DIFF': point_diff,  # Positive = home won, negative = away won
            'WINNER': winner_name,
            'MARGIN': margin
        })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    
    return df


def save_to_local(df, sport, date_str):
    """
    Save game results to local file (test mode).
    Saves file with headers even if no games found.
    
    Args:
        df: DataFrame with game results
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        date_str: Date string in YYYYMMDD format
    """
    # Create output directory
    output_dir = Path.home() / 'Downloads' / 'tmp'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename using game date (convert YYYYMMDD to YYYY-MM-DD)
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    filename = f"{sport}_game_results_{date_formatted}.csv"
    output_path = output_dir / filename
    
    # Save to CSV (even if empty, with headers)
    try:
        if df.empty:
            # Create empty dataframe with expected columns
            empty_df = pd.DataFrame(columns=[
                'GAME_DATE', 'GAME_ID', 'HOME_TEAM', 'HOME_ABBR', 'HOME_SCORE', 'HOME_WL',
                'AWAY_TEAM', 'AWAY_ABBR', 'AWAY_SCORE', 'AWAY_WL', 'POINT_DIFF', 'WINNER', 'MARGIN'
            ])
            empty_df.to_csv(output_path, index=False)
            print(f"   💾 Saved empty file (headers only) to: {output_path}")
        else:
            df.to_csv(output_path, index=False)
            print(f"   💾 Saved locally to: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ Local save failed: {e}")
        return False


def check_s3_file_exists(sport, date_str):
    """
    Check if a file already exists in S3 for a given date.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        date_str: Date string in YYYYMMDD format
    
    Returns:
        bool: True if file exists, False otherwise
    """
    config = SPORT_CONFIG[sport]
    bucket = config['s3_bucket']
    s3_path = config['s3_path_template']
    
    # Create filename
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    filename = f"{date_formatted}.csv"
    s3_key = f"{s3_path}{filename}"
    
    try:
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except:
        return False


def upload_to_s3(df, sport, date_str, skip_if_exists=False):
    """
    Upload game results to S3.
    Uploads file with headers even if no games found.
    
    Args:
        df: DataFrame with game results
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        date_str: Date string in YYYYMMDD format (Eastern Time)
        skip_if_exists: If True, skip upload if file already exists in S3
    
    Returns:
        bool: True if uploaded, False if skipped
    """
    config = SPORT_CONFIG[sport]
    bucket = config['s3_bucket']
    s3_path = config['s3_path_template']
    
    # Create filename using game date (convert YYYYMMDD to YYYY-MM-DD)
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    filename = f"{date_formatted}.csv"
    s3_key = f"{s3_path}{filename}"
    
    # Check if file exists and skip if requested
    if skip_if_exists:
        if check_s3_file_exists(sport, date_str):
            print(f"   ⏭️  Skipped (already exists): s3://{bucket}/{s3_key}")
            return False
    
    # Convert to CSV (even if empty, with headers)
    csv_buffer = StringIO()
    if df.empty:
        # Create empty dataframe with expected columns
        empty_df = pd.DataFrame(columns=[
            'GAME_DATE', 'GAME_ID', 'HOME_TEAM', 'HOME_ABBR', 'HOME_SCORE', 'HOME_WL',
            'AWAY_TEAM', 'AWAY_ABBR', 'AWAY_SCORE', 'AWAY_WL', 'POINT_DIFF', 'WINNER', 'MARGIN'
        ])
        empty_df.to_csv(csv_buffer, index=False)
    else:
        df.to_csv(csv_buffer, index=False)
    
    # Upload
    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        if df.empty:
            print(f"   ✅ Uploaded empty file (headers only) to s3://{bucket}/{s3_key}")
        else:
            print(f"   ✅ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"   ❌ S3 upload failed: {e}")
        return False


def get_season_date_range(sport, season, start_date_override=None, end_date_override=None):
    """
    Get start and end dates for a season (REGULAR SEASON ONLY by default).
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        season: Season string (e.g., '2025' or '2025-26')
        start_date_override: Optional start date (YYYY-MM-DD format)
        end_date_override: Optional end date (YYYY-MM-DD format)
    
    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """
    # Handle manual overrides first
    if start_date_override:
        start_date = datetime.strptime(start_date_override, '%Y-%m-%d').date()
    else:
        if sport == 'nfl':
            # NFL REGULAR SEASON runs ~Sept 5 -> early Feb (Week 18 + playoffs)
            year = int(season)
            start_date = datetime(year, 9, 5).date()  # Week 1 typically starts first Thursday of September
        elif sport == 'nba':
            # NBA REGULAR SEASON runs ~Oct 22 -> mid-April (82 games)
            # Playoffs: mid-April -> June
            year = int(season.split('-')[0])
            start_date = datetime(year, 10, 22).date()  # Regular season typically starts ~Oct 22
        elif sport == 'ncaaf':
            # College football runs late Aug/early Sept -> Jan (bowls/playoff)
            year = int(season)
            start_date = datetime(year, 8, 23).date()  # Week 0 starts ~Aug 23-24
        elif sport == 'ncaab':
            # College basketball runs early Nov -> early Apr
            year = int(season.split('-')[0])
            start_date = datetime(year, 11, 3).date()  # Season starts ~Nov 3-4
        else:
            raise ValueError(f"Unknown sport: {sport}")
    
    if end_date_override:
        end_date = datetime.strptime(end_date_override, '%Y-%m-%d').date()
    else:
        if sport == 'nfl':
            year = int(season)
            end_date = datetime(year + 1, 2, 28).date()  # Includes playoffs
        elif sport == 'nba':
            year = int(season.split('-')[0])
            end_date = datetime(year + 1, 6, 30).date()  # Includes playoffs
        elif sport == 'ncaaf':
            year = int(season)
            end_date = datetime(year + 1, 1, 31).date()  # Includes bowls/playoff
        elif sport == 'ncaab':
            year = int(season.split('-')[0])
            end_date = datetime(year + 1, 4, 30).date()  # Includes March Madness
    
    # Don't fetch future dates
    today = datetime.now().date()
    if end_date > today:
        end_date = today
    
    return start_date, end_date


def normalize_date_string(date_input):
    """
    Normalize date input to YYYYMMDD format.
    
    Args:
        date_input: Date string in YYYYMMDD or YYYY-MM-DD format
    
    Returns:
        str: Date in YYYYMMDD format
    """
    # Remove dashes if present
    date_str = date_input.replace('-', '')
    
    # Validate it's 8 digits
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date format: {date_input}. Expected YYYYMMDD or YYYY-MM-DD")
    
    return date_str


def fetch_single_date(sport, date_str, upload_s3=False, test_mode=False, skip_existing=False):
    """
    Fetch game results for a single date.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        date_str: Date string in YYYYMMDD format
        upload_s3: Whether to upload to S3
        test_mode: Whether to save locally to ~/Downloads/tmp
        skip_existing: If True, skip if file already exists in S3
    
    Returns:
        pd.DataFrame with game results (or None if skipped)
    """
    # Check if date is in the past (not today)
    # Only save results for completed days to avoid saving incomplete game data
    date_obj = datetime.strptime(date_str, '%Y%m%d').date()
    today = datetime.now(ZoneInfo('America/New_York')).date()
    
    if date_obj >= today:
        print(f"\n⚠️  Skipping {date_str} - date is today or in the future (games may not be complete)")
        print(f"   Today: {today}, Requested: {date_obj}")
        print(f"   💡 This script only saves data for dates in the past to ensure completeness")
        return None
    
    # Check if already exists and should skip
    if skip_existing and upload_s3:
        if check_s3_file_exists(sport, date_str):
            print(f"\n⏭️  Skipping {date_str} (already exists in S3)")
            return None
    
    print(f"\n📥 Fetching {sport.upper()} games for {date_str}...")
    
    # Fetch from ESPN
    espn_data = fetch_espn_scoreboard(sport, date_str)
    
    if not espn_data:
        print("   ❌ Failed to fetch from ESPN API")
        return pd.DataFrame()
    
    # Parse games
    df = parse_espn_games(espn_data, sport)
    
    if df.empty:
        print("   ℹ️  No games found for this date")
    else:
        print(f"   ✅ Found {len(df)} games")
        
        # Show sample
        print(f"\n   Sample games:")
        for _, row in df.head(3).iterrows():
            print(f"     {row['AWAY_ABBR']} @ {row['HOME_ABBR']}: {row['AWAY_SCORE']}-{row['HOME_SCORE']} (Winner: {row['WINNER']}, Margin: {row['MARGIN']})")
    
    # Save results
    if upload_s3:
        upload_to_s3(df, sport, date_str, skip_if_exists=False)  # Already checked above
    
    if test_mode:
        save_to_local(df, sport, date_str)
    
    return df


def get_interactive_date_range(sport, season):
    """
    Prompt user for start and end dates interactively.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        season: Season string
    
    Returns:
        tuple: (start_date_str, end_date_str) in YYYY-MM-DD format
    """
    # Get default date range
    default_start, default_end = get_season_date_range(sport, season)
    
    print(f"\n📅 Season Date Range Setup for {sport.upper()} {season}")
    print("=" * 80)
    print(f"Default regular season start: {default_start}")
    print(f"Default end date: {default_end} (today)")
    print()
    
    # Prompt for start date
    start_input = input(f"Enter START date (YYYY-MM-DD) or press ENTER for default [{default_start}]: ").strip()
    start_date_str = start_input if start_input else str(default_start)
    
    # Validate start date format
    try:
        datetime.strptime(start_date_str, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {start_date_str}. Using default: {default_start}")
        start_date_str = str(default_start)
    
    # Prompt for end date
    end_input = input(f"Enter END date (YYYY-MM-DD) or press ENTER for today [{default_end}]: ").strip()
    end_date_str = end_input if end_input else str(default_end)
    
    # Validate end date format
    try:
        datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {end_date_str}. Using default: {default_end}")
        end_date_str = str(default_end)
    
    print(f"\n✅ Using date range: {start_date_str} to {end_date_str}")
    print("=" * 80)
    
    return start_date_str, end_date_str


def backfill_season(sport, season, upload_s3=False, test_mode=False, start_date_override=None, end_date_override=None, skip_existing=False):
    """
    Backfill all games for a season.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        season: Season string
        upload_s3: Whether to upload to S3
        test_mode: Whether to save locally to ~/Downloads/tmp
        start_date_override: Optional start date (YYYY-MM-DD)
        end_date_override: Optional end date (YYYY-MM-DD)
        skip_existing: If True, skip dates that already exist in S3
    
    Returns:
        pd.DataFrame with all game results
    """
    print(f"\n🔄 Backfilling {sport.upper()} {season} season...")
    
    start_date, end_date = get_season_date_range(sport, season, start_date_override, end_date_override)
    
    print(f"   Date range: {start_date} to {end_date}")
    if skip_existing:
        print(f"   Mode: Skip existing files in S3")
    
    all_dfs = []
    dates_processed = 0
    dates_skipped = 0
    games_found = 0
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        
        # Fetch for this date
        df = fetch_single_date(sport, date_str, upload_s3=upload_s3, test_mode=test_mode, skip_existing=skip_existing)
        
        if df is None:
            # Skipped due to existing file
            dates_skipped += 1
        elif not df.empty:
            all_dfs.append(df)
            games_found += len(df)
        
        dates_processed += 1
        current_date += timedelta(days=1)
        
        # Progress update every 30 days
        if dates_processed % 30 == 0:
            skip_msg = f", {dates_skipped} skipped" if skip_existing else ""
            print(f"   📊 Progress: {dates_processed} dates processed{skip_msg}, {games_found} games found")
    
    print(f"\n✅ Backfill complete!")
    print(f"   Total dates processed: {dates_processed}")
    if skip_existing:
        print(f"   Dates skipped (already in S3): {dates_skipped}")
        print(f"   Dates fetched: {dates_processed - dates_skipped}")
    print(f"   Total games found: {games_found}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Fetch historical game results from ESPN API')
    parser.add_argument('--sport', type=str, choices=['nfl', 'nba', 'ncaaf', 'ncaab'], required=True,
                       help='Sport to fetch (nfl, nba, ncaaf, ncaab)')
    parser.add_argument('--season', type=str, default=None,
                       help='Season (e.g., 2025 for NFL/NCAAF, 2025-26 for NBA/NCAAB). Defaults to current season.')
    parser.add_argument('--date', type=str, default=None,
                       help='Specific date to fetch (YYYYMMDD or YYYY-MM-DD). If not provided, will prompt for date range.')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backfill (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backfill (YYYY-MM-DD format). Defaults to today.')
    parser.add_argument('--week', type=int, default=None,
                       help='Fetch specific week (NFL only, 1-18)')
    parser.add_argument('--backfill', action='store_true',
                       help='Backfill entire season (will prompt for date range if not provided)')
    parser.add_argument('--s3', action='store_true',
                       help='Upload results to S3')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip dates that already exist in S3 (useful for resuming large backfills)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files in S3 (opposite of --skip-existing)')
    parser.add_argument('--test', action='store_true',
                       help='Save results locally to ~/Downloads/tmp for review')
    
    args = parser.parse_args()
    
    # Handle conflicting flags
    if args.skip_existing and args.overwrite:
        print("❌ Error: Cannot use both --skip-existing and --overwrite")
        return None
    
    # Determine season
    if args.season:
        season = args.season
    else:
        # Get current season from season_utils
        season = get_current_season(args.sport)
    
    print("=" * 80)
    print(f"ESPN GAME RESULTS FETCHER - {args.sport.upper()}")
    print("=" * 80)
    print(f"Sport: {args.sport.upper()}")
    print(f"Season: {season}")
    print(f"S3 Upload: {'✅ Enabled' if args.s3 else '❌ Disabled'}")
    
    # Determine skip/overwrite behavior
    if args.skip_existing:
        print(f"Mode: Skip existing files (no overwrite)")
    elif args.overwrite:
        print(f"Mode: Overwrite existing files")
    else:
        print(f"Mode: Overwrite existing files (default)")
    
    print(f"Test Mode: {'✅ Enabled (saving to ~/Downloads/tmp)' if args.test else '❌ Disabled'}")
    
    # Determine mode: backfill (date range) or single date
    if args.backfill or not args.date:
        # Backfill mode - need date range
        
        # Check if dates provided via command line args
        if args.start_date or args.end_date:
            start_date_str = args.start_date
            end_date_str = args.end_date
            print(f"\nMode: Backfill season using command-line dates")
        else:
            # Prompt interactively for dates
            start_date_str, end_date_str = get_interactive_date_range(args.sport, season)
        
        df = backfill_season(
            args.sport, 
            season, 
            upload_s3=args.s3, 
            test_mode=args.test,
            start_date_override=start_date_str,
            end_date_override=end_date_str,
            skip_existing=args.skip_existing
        )
        
        if not df.empty:
            print(f"\n📊 Total Games: {len(df)}")
            print(f"\n📋 Sample of results:")
            print(df.head(10).to_string(index=False))
        
        return df
    
    # Single date mode (only if --date explicitly provided)
    # Normalize date to YYYYMMDD format
    date_str = normalize_date_string(args.date)
    
    df = fetch_single_date(args.sport, date_str, upload_s3=args.s3, test_mode=args.test, skip_existing=args.skip_existing)
    
    if not df.empty:
        print(f"\n📋 Results:")
        print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ Fetch complete!")
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    main()

