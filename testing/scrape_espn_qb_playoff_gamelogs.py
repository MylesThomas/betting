"""
Scrape ESPN.com for NFL QB playoff game-by-game logs.

Success! ESPN.com provides individual playoff game logs going back to at least 2001.
This script scrapes game-by-game playoff data (1 row per game) from ESPN player pages.

Data Structure:
  year | game_date | opponent | result | comp | att | yds | tds | ints | ...
  (1 row per game)

Storage:
  Test Mode:       ~/Downloads/tmp/ (local only)
  Production:      s3://nfl-betting-mt/data/01_input/espn_web/playoffs/qb/gamelogs/ (S3 only)
  Local Override:  data/01_input/espn_web/playoffs/qb/gamelogs/ (with --local-only)

Usage:
    # Test mode - saves to ~/Downloads/tmp
    python3 testing/scrape_espn_qb_playoff_gamelogs.py --qb "Tom Brady" --start-year 2001 --export --test-mode
    
    # Production - saves to S3 only
    python3 testing/scrape_espn_qb_playoff_gamelogs.py --qb "Tom Brady" --start-year 2001 --export
    
    # Save to local instead of S3
    python3 testing/scrape_espn_qb_playoff_gamelogs.py --qb "Tom Brady" --start-year 2001 --export --local-only
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import argparse
import time
import ssl
import urllib3
from io import StringIO
from pathlib import Path
from datetime import datetime
import os

# Disable SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# S3 configuration
S3_BUCKET = 'nfl-betting-mt'
S3_PREFIX = 'data/01_input/espn_web/playoffs/qb/gamelogs'

# Local data directories
LOCAL_DATA_DIR = 'data/01_input/espn_web/playoffs/qb/gamelogs'
TEST_DATA_DIR = os.path.expanduser('~/Downloads/tmp')

# Known QBs
KNOWN_QBS = {
    'Patrick Mahomes': 3139477,
    'Josh Allen': 3918298,
    'Tom Brady': 2330,
    'Peyton Manning': 1428,
}


def scrape_espn_playoff_gamelogs(athlete_id, athlete_name, year):
    """
    Scrape playoff game logs for a QB for a specific year from ESPN.com.
    
    Args:
        athlete_id: ESPN athlete ID
        athlete_name: QB name
        year: Season year
        
    Returns:
        pd.DataFrame: Game-by-game playoff stats (None if no games or error)
    """
    url = f"https://www.espn.com/nfl/player/gamelog/_/id/{athlete_id}/type/nfl/year/{year}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        # Find postseason table
        for table in tables:
            try:
                df = pd.read_html(StringIO(str(table)))[0]
                
                if len(df.columns) == 0:
                    continue
                
                # Check if postseason table
                first_col = df.columns[0]
                if isinstance(first_col, tuple) and 'Postseason' in str(first_col[0]):
                    # Flatten column names
                    new_cols = []
                    for c in df.columns:
                        if isinstance(c, tuple):
                            # Keep only the stat name, prefix with category if needed
                            category, stat = c
                            if category in ['Passing', 'Rushing']:
                                new_cols.append(f"{category.lower()}_{stat.lower()}")
                            else:
                                new_cols.append(stat)
                        else:
                            new_cols.append(c)
                    
                    df.columns = new_cols
                    
                    # Remove header rows (like "AFC Championship")
                    df = df[~df['Date'].str.contains('Championship|Divisional|Wild Card', na=False, case=False)]
                    df = df[df['Date'].str.contains(r'\d', na=False)]  # Keep only rows with dates
                    
                    # Add metadata columns
                    df['season'] = year
                    df['athlete'] = athlete_name
                    df['athlete_id'] = athlete_id
                    
                    # Rename columns for consistency
                    column_mapping = {
                        'Date': 'date',
                        'OPP': 'opponent',
                        'Result': 'result',
                        'passing_cmp': 'completions',
                        'passing_att': 'attempts',
                        'passing_yds': 'passing_yards',
                        'passing_cmp%': 'completion_pct',
                        'passing_avg': 'yards_per_attempt',
                        'passing_td': 'passing_tds',
                        'passing_int': 'interceptions',
                        'passing_lng': 'longest_pass',
                        'passing_sack': 'sacks',
                        'passing_rtg': 'passer_rating',
                        'passing_qbr': 'qbr',
                        'rushing_car': 'rushing_attempts',
                        'rushing_yds': 'rushing_yards',
                        'rushing_avg': 'rushing_avg',
                        'rushing_td': 'rushing_tds',
                        'rushing_lng': 'longest_rush',
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Reorder columns
                    base_cols = ['athlete', 'athlete_id', 'season', 'date', 'opponent', 'result']
                    passing_cols = ['completions', 'attempts', 'passing_yards', 'completion_pct', 
                                  'yards_per_attempt', 'passing_tds', 'interceptions', 'longest_pass',
                                  'sacks', 'passer_rating', 'qbr']
                    rushing_cols = ['rushing_attempts', 'rushing_yards', 'rushing_avg', 
                                  'rushing_tds', 'longest_rush']
                    
                    # Only include columns that exist
                    ordered_cols = []
                    for col in base_cols + passing_cols + rushing_cols:
                        if col in df.columns:
                            ordered_cols.append(col)
                    
                    df = df[ordered_cols]
                    
                    return df if len(df) > 0 else None
                    
            except Exception as e:
                continue
        
        return None
        
    except Exception as e:
        print(f"   ⚠️  Error scraping {year}: {e}")
        return None


def get_all_playoff_gamelogs(athlete_id, athlete_name, start_year=2017, end_year=2025):
    """
    Get all playoff game logs for a QB across multiple years.
    
    Args:
        athlete_id: ESPN athlete ID
        athlete_name: QB name
        start_year: First year to scrape
        end_year: Last year to scrape
        
    Returns:
        pd.DataFrame: All playoff games across years
    """
    print(f"\n{'='*80}")
    print(f"Scraping playoff game logs: {athlete_name} (ID: {athlete_id})")
    print(f"Years: {start_year} - {end_year}")
    print(f"{'='*80}")
    
    all_games = []
    
    for year in range(end_year, start_year - 1, -1):
        print(f"\n  Scraping {year}...", end=" ")
        
        df = scrape_espn_playoff_gamelogs(athlete_id, athlete_name, year)
        
        if df is not None and len(df) > 0:
            all_games.append(df)
            print(f"✅ {len(df)} playoff games")
        else:
            print(f"❌ No playoff games")
        
        time.sleep(0.5)  # Be nice to ESPN servers
    
    if all_games:
        combined_df = pd.concat(all_games, ignore_index=True)
        print(f"\n  {'='*80}")
        print(f"  Total playoff games: {len(combined_df)}")
        print(f"  {'='*80}")
        return combined_df
    else:
        print(f"\n  ℹ️  No playoff games found")
        return None


def sanitize_filename(name):
    """Sanitize name for filename."""
    return name.lower().replace(' ', '_').replace('-', '_')


def save_to_local(df, filename, test_mode=False):
    """Save DataFrame to local filesystem."""
    if test_mode:
        save_dir = Path(TEST_DATA_DIR)
    else:
        save_dir = Path(LOCAL_DATA_DIR)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename
    df.to_csv(file_path, index=False)
    
    return file_path


def save_to_s3(df, filename, bucket=S3_BUCKET, prefix=S3_PREFIX):
    """Save DataFrame to S3."""
    try:
        import boto3
        from io import StringIO
        
        s3_client = boto3.client('s3')
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_key = f"{prefix}/{filename}"
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        return f"s3://{bucket}/{s3_key}"
        
    except ImportError:
        raise ImportError("boto3 required for S3 uploads")
    except Exception as e:
        raise Exception(f"S3 upload failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scrape ESPN for NFL QB playoff game logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--qb', type=str, required=True, help='QB name')
    parser.add_argument('--id', type=int, help='ESPN athlete ID')
    parser.add_argument('--start-year', type=int, default=2017, help='Start year (default: 2017)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year (default: 2025)')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    parser.add_argument('--test-mode', action='store_true', help='Test mode: save to ~/Downloads/tmp only')
    parser.add_argument('--local-only', action='store_true', help='Save to local data/01_input/... instead of S3')
    
    args = parser.parse_args()
    
    # Get QB ID
    qb_name = args.qb
    if args.id:
        qb_id = args.id
    elif qb_name in KNOWN_QBS:
        qb_id = KNOWN_QBS[qb_name]
        print(f"Using known ID for {qb_name}: {qb_id}")
    else:
        print(f"❌ Error: QB '{qb_name}' not in known list. Please provide --id")
        return
    
    # Scrape data
    df = get_all_playoff_gamelogs(qb_id, qb_name, args.start_year, args.end_year)
    
    if df is not None and len(df) > 0:
        print(f"\n{'='*80}")
        print(f"SAMPLE DATA")
        print(f"{'='*80}")
        print(df[['season', 'date', 'opponent', 'result', 'completions', 'attempts', 
                  'passing_yards', 'passing_tds', 'interceptions']].head(10))
        
        # Export if requested
        if args.export:
            date_str = datetime.now().strftime('%Y%m%d')
            qb_sanitized = sanitize_filename(qb_name)
            filename = f"nfl_qb_playoff_gamelogs_{qb_sanitized}_{date_str}.csv"
            
            print(f"\n{'='*80}")
            print(f"EXPORTING")
            print(f"{'='*80}")
            
            # Test mode: save to ~/Downloads/tmp only
            if args.test_mode:
                local_path = save_to_local(df, filename, test_mode=True)
                print(f"💾 Local (test mode): {local_path.absolute()}")
                print(f"ℹ️  Skipping S3 (test mode)")
            
            # Production mode: S3 only (unless --local-only specified)
            elif args.local_only:
                local_path = save_to_local(df, filename, test_mode=False)
                print(f"💾 Local (production): {local_path.absolute()}")
                print(f"ℹ️  Skipping S3 (local-only)")
            
            # Default production: S3 only
            else:
                try:
                    s3_uri = save_to_s3(df, filename)
                    print(f"☁️  S3: {s3_uri}")
                    print(f"ℹ️  No local save (production default is S3-only)")
                except Exception as e:
                    print(f"⚠️  S3 upload failed: {e}")
        
        print(f"{'='*80}")


if __name__ == '__main__':
    main()

