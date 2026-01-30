"""
Quick test script to debug ESPN game results loading
"""
import pandas as pd
import boto3
from io import BytesIO
from pathlib import Path
import yaml

# Load season dates
SEASON_DATES_PATH = Path(__file__).parent.parent / 'config' / 'season_dates.yaml'
with open(SEASON_DATES_PATH, 'r') as f:
    SEASON_DATES = yaml.safe_load(f)

S3_BUCKET_BETTING = 'nba-betting-mt'
season = '2025-26'

# Get season date range from config
season_dates = SEASON_DATES['nba']
season_start = pd.to_datetime(season_dates[season]['season_start']).date()
season_end = pd.to_datetime(season_dates[season]['playoff_end']).date()

print(f"Season: {season}")
print(f"Date range: {season_start} to {season_end}")
print()

s3_prefix = 'data/01_input/historical_game_results/'
s3 = boto3.client('s3')

# List all CSV files (handle pagination)
print(f"Looking for files in: s3://{S3_BUCKET_BETTING}/{s3_prefix}")

all_files = []
continuation_token = None

while True:
    if continuation_token:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET_BETTING, 
            Prefix=s3_prefix,
            ContinuationToken=continuation_token
        )
    else:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_BETTING, Prefix=s3_prefix)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                all_files.append(obj)
    
    # Check if there are more results
    if response.get('IsTruncated'):
        continuation_token = response.get('NextContinuationToken')
        print(f"  Fetching more... (got {len(all_files)} so far)")
    else:
        break

print(f"  Total files retrieved: {len(all_files)}")
in_range_files = []
skipped_files = []

for obj in all_files:
    key = obj['Key']
    
    # Extract date from filename
    try:
        filename = key.split('/')[-1]
        date_str = filename.replace('.csv', '')
        file_date = pd.to_datetime(date_str).date()
        
        # Filter by season date range
        if file_date < season_start or file_date > season_end:
            skipped_files.append((key, file_date, 'out of range'))
        else:
            in_range_files.append((key, file_date))
            
    except Exception as e:
        skipped_files.append((key, None, f'parse error: {e}'))

print(f"\nTotal CSV files: {len(all_files)}")
print(f"In range: {len(in_range_files)}")
print(f"Skipped: {len(skipped_files)}")

# Show date range of all files
if skipped_files:
    dates_with_reasons = [(date, reason) for key, date, reason in skipped_files if date is not None]
    if dates_with_reasons:
        dates = [d for d, r in dates_with_reasons]
        print(f"\nDate range in S3: {min(dates)} to {max(dates)}")
        print(f"Looking for: {season_start} to {season_end}")
        
        # Show files close to our range
        print(f"\nFiles near season start ({season_start}):")
        for key, date, reason in skipped_files:
            if date and date >= pd.to_datetime('2025-10-01').date():
                print(f"  {key} -> {date} ({reason})")
                if date >= pd.to_datetime('2025-11-01').date():
                    break

if in_range_files:
    print(f"\nFirst 5 in-range files:")
    for key, date in in_range_files[:5]:
        print(f"  {key} -> {date}")
    
    print(f"\nLast 5 in-range files:")
    for key, date in in_range_files[-5:]:
        print(f"  {key} -> {date}")
    
    # Try loading one file
    print(f"\nTrying to load first file: {in_range_files[0][0]}")
    try:
        obj_response = s3.get_object(Bucket=S3_BUCKET_BETTING, Key=in_range_files[0][0])
        df = pd.read_csv(BytesIO(obj_response['Body'].read()))
        print(f"  Success! Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\n  First row:")
        print(df.head(1))
    except Exception as e:
        print(f"  Error: {e}")
else:
    print("\nNo files in range! Showing skipped files:")
    for key, date, reason in skipped_files[:10]:
        print(f"  {key} -> {date} ({reason})")
