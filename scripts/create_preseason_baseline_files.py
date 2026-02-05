"""
Create preseason baseline CSV files from config and upload to S3.

This is a one-time migration script to move preseason odds from config to S3.

Usage:
    python3 scripts/create_preseason_baseline_files.py

Outputs:
    - Creates CSV files matching the format of fetch_championship_futures.py
    - Uploads to S3:
        - s3://the-odds-api-mt/nba/futures/preseason_2024-25.csv
        - s3://the-odds-api-mt/ncaab/futures/preseason_2024-25.csv
"""

import sys
import yaml
import pandas as pd
import boto3
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability


def load_config():
    """Load futures config."""
    config_path = repo_root / 'config' / 'futures_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_preseason_csv(sport_name, preseason_config):
    """
    Create a CSV dataframe from preseason config.
    
    Format matches fetch_championship_futures.py output:
    - sport, bookmaker, team, odds, implied_prob, record
    """
    odds_dict = preseason_config['odds']
    date = preseason_config['date']
    
    rows = []
    for team, odds in odds_dict.items():
        rows.append({
            'sport': sport_name.upper(),
            'bookmaker': 'preseason_baseline',  # Placeholder bookmaker
            'team': team,
            'odds': odds,
            'implied_prob': odds_to_implied_probability(odds),
            'record': ''  # No records for preseason baseline
        })
    
    df = pd.DataFrame(rows)
    return df, date


def upload_to_s3(df, s3_path):
    """Upload DataFrame to S3 as CSV."""
    # Parse S3 path
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    path = s3_path[5:]  # Remove 's3://'
    bucket, key = path.split('/', 1)
    
    # Convert to CSV
    csv_data = df.to_csv(index=False)
    
    # Upload
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_data,
        ContentType='text/csv'
    )
    
    print(f"   ✅ Uploaded to {s3_path}")


def main():
    config = load_config()
    
    print("=" * 80)
    print("CREATING PRESEASON BASELINE FILES")
    print("=" * 80 + "\n")
    
    # Process NBA
    print("🏀 NBA Preseason Baseline")
    nba_config = config['sports']['nba']['historical_odds']['preseason']
    nba_df, nba_date = create_preseason_csv('nba', nba_config)
    
    # Determine season year from date (2024-10-22 → 2024-25 season)
    year = nba_date.split('-')[0]
    next_year = str(int(year) + 1)[-2:]  # Get last 2 digits
    season = f"{year}-{next_year}"
    
    nba_s3_path = f"s3://the-odds-api-mt/nba/futures/preseason_{season}.csv"
    
    print(f"   Date: {nba_date}")
    print(f"   Teams: {len(nba_df)}")
    print(f"   Uploading to: {nba_s3_path}")
    
    upload_to_s3(nba_df, nba_s3_path)
    print()
    
    # Process NCAAB
    print("🏀 NCAAB Preseason Baseline")
    ncaab_config = config['sports']['ncaab']['historical_odds']['preseason']
    ncaab_df, ncaab_date = create_preseason_csv('ncaab', ncaab_config)
    
    # Determine season year from date
    year = ncaab_date.split('-')[0]
    next_year = str(int(year) + 1)[-2:]
    season = f"{year}-{next_year}"
    
    ncaab_s3_path = f"s3://the-odds-api-mt/ncaab/futures/preseason_{season}.csv"
    
    print(f"   Date: {ncaab_date}")
    print(f"   Teams: {len(ncaab_df)}")
    print(f"   Uploading to: {ncaab_s3_path}")
    
    upload_to_s3(ncaab_df, ncaab_s3_path)
    print()
    
    # Summary
    print("=" * 80)
    print("✅ PRESEASON BASELINE FILES CREATED")
    print("=" * 80)
    print("\nYou can now use these S3 paths:")
    print(f"  NBA:   {nba_s3_path}")
    print(f"  NCAAB: {ncaab_s3_path}")
    print("\nNext steps:")
    print("  1. Verify files exist in S3")
    print("  2. Update analyze_futures.py to use --preseason argument")
    print("  3. Remove preseason odds from config (keep date for reference)")


if __name__ == '__main__':
    main()
