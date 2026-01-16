"""
Migrate NFL Game Lines from Local to S3

Context:
- Moves NFL betting lines from local storage to S3
- Mirrors the NBA historical_game_lines structure for consistency
- Organizes by season: s3://the-odds-api-mt/nfl/historical_game_lines/{season}/

Current State:
- Local: data/01_input/the-odds-api/nfl/game_lines/historical/nfl_game_lines_*.csv
- 922 CSV files (2020-2025 seasons + London games)

After Migration:
- S3: s3://the-odds-api-mt/nfl/historical_game_lines/{season}/nfl_game_lines_{date}.csv
- Organized by season folders (2020/, 2021/, 2022/, 2023/, 2024/, 2025/)

Usage:
    # Dry run (preview what would be uploaded)
    python tmp/migrate_nfl_lines_to_s3.py --dry-run
    
    # Upload all files
    python tmp/migrate_nfl_lines_to_s3.py
    
    # Upload specific season
    python tmp/migrate_nfl_lines_to_s3.py --season 2025
    
    # Skip files that already exist in S3
    python tmp/migrate_nfl_lines_to_s3.py --skip-existing
"""

import boto3
import pandas as pd
from pathlib import Path
from io import StringIO
from datetime import datetime
import argparse
import sys

# Find project root
project_root = Path(__file__).resolve().parent.parent
while not (project_root / '.gitignore').exists():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent

# Paths
LOCAL_HISTORICAL_DIR = project_root / "data/01_input/the-odds-api/nfl/game_lines/historical"
S3_BUCKET = 'the-odds-api-mt'
S3_BASE_PATH = 'nfl/historical_game_lines'

# Season date ranges (Sept 1 - Feb 28)
SEASON_RANGES = {
    2020: ('2020-09-01', '2021-02-28'),
    2021: ('2021-09-01', '2022-02-28'),
    2022: ('2022-09-01', '2023-02-28'),
    2023: ('2023-09-01', '2024-02-28'),
    2024: ('2024-09-01', '2025-02-28'),
    2025: ('2025-09-01', '2026-02-28'),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_season_from_date(date_str: str) -> int:
    """
    Determine NFL season from game date.
    NFL season spans Sept-Feb, so games in Sept-Dec are current year,
    games in Jan-Feb are previous year's season.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        Season year (e.g., 2025 for 2025-26 season)
    """
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month
    
    # Jan-Feb games belong to previous year's season
    if month <= 2:
        return year - 1
    # Sept-Dec games belong to current year's season
    else:
        return year


def get_s3_key(season: int, date_str: str) -> str:
    """Get S3 key for a game lines file."""
    return f"{S3_BASE_PATH}/{season}/nfl_game_lines_{date_str}.csv"


def check_s3_exists(s3_client, s3_key: str) -> bool:
    """Check if file exists in S3."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except:
        return False


def upload_file_to_s3(s3_client, local_file: Path, s3_key: str, dry_run: bool = False) -> bool:
    """
    Upload a CSV file to S3.
    
    Args:
        s3_client: Boto3 S3 client
        local_file: Path to local CSV file
        s3_key: S3 key (path) where file should be uploaded
        dry_run: If True, skip actual upload
    
    Returns:
        True if uploaded (or would be uploaded in dry_run), False otherwise
    """
    if dry_run:
        print(f"   [DRY RUN] Would upload: {local_file.name} -> {s3_key}")
        return True
    
    try:
        # Read CSV
        df = pd.read_csv(local_file)
        
        # Convert to CSV buffer
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        return True
    except Exception as e:
        print(f"   ❌ Error uploading {local_file.name}: {e}")
        return False


def find_local_files(season: int = None) -> list:
    """
    Find all local NFL game line CSV files.
    
    Args:
        season: Optional season filter (2020-2025)
    
    Returns:
        List of (Path, season, date_str) tuples
    """
    files = []
    
    # Regular game files: nfl_game_lines_YYYY-MM-DD.csv
    for csv_file in sorted(LOCAL_HISTORICAL_DIR.glob("nfl_game_lines_*.csv")):
        # Extract date from filename
        date_str = csv_file.stem.replace('nfl_game_lines_', '')
        
        # Determine season
        file_season = determine_season_from_date(date_str)
        
        # Filter by season if specified
        if season and file_season != season:
            continue
        
        files.append((csv_file, file_season, date_str))
    
    # London games files: {season}_game_lines_london.csv
    for london_file in sorted(LOCAL_HISTORICAL_DIR.glob("*_game_lines_london.csv")):
        file_season = int(london_file.stem.split('_')[0])
        
        # Filter by season if specified
        if season and file_season != season:
            continue
        
        # Use special identifier for London games
        date_str = f"{file_season}_london_games"
        files.append((london_file, file_season, date_str))
    
    return files


# =============================================================================
# MAIN MIGRATION FUNCTION
# =============================================================================

def migrate_files(season: int = None, dry_run: bool = False, skip_existing: bool = False):
    """
    Migrate NFL game line files from local to S3.
    
    Args:
        season: Optional season filter (2020-2025)
        dry_run: If True, preview changes without uploading
        skip_existing: If True, skip files that already exist in S3
    """
    print("=" * 100)
    print("NFL GAME LINES MIGRATION: Local -> S3")
    print("=" * 100)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be uploaded")
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Find local files
    print(f"\n📂 Scanning local files...")
    files = find_local_files(season)
    
    if not files:
        print("❌ No files found to migrate")
        return
    
    print(f"   Found {len(files)} files")
    
    if season:
        print(f"   Filtered to season: {season}")
    
    # Group by season for summary
    season_counts = {}
    for _, file_season, _ in files:
        season_counts[file_season] = season_counts.get(file_season, 0) + 1
    
    print(f"\n📊 Files by season:")
    for s in sorted(season_counts.keys()):
        print(f"   {s}: {season_counts[s]} files")
    
    # Process files
    print(f"\n🚀 Starting migration...")
    print(f"   Target: s3://{S3_BUCKET}/{S3_BASE_PATH}/")
    
    uploaded = 0
    skipped_exists = 0
    errors = 0
    
    for local_file, file_season, date_str in files:
        # Generate S3 key
        s3_key = get_s3_key(file_season, date_str)
        
        # Check if exists in S3
        if skip_existing and not dry_run:
            if check_s3_exists(s3_client, s3_key):
                skipped_exists += 1
                print(f"   ⏭️  {local_file.name} (already exists)")
                continue
        
        # Upload
        success = upload_file_to_s3(s3_client, local_file, s3_key, dry_run)
        
        if success:
            uploaded += 1
            if not dry_run:
                print(f"   ✅ {local_file.name} -> {s3_key}")
        else:
            errors += 1
    
    # Summary
    print(f"\n{'=' * 100}")
    print("MIGRATION SUMMARY")
    print("=" * 100)
    print(f"Total files found:     {len(files)}")
    print(f"Successfully uploaded: {uploaded}")
    
    if skip_existing:
        print(f"Skipped (exists):      {skipped_exists}")
    
    if errors > 0:
        print(f"Errors:                {errors}")
    
    if dry_run:
        print(f"\n💡 This was a dry run. Add --no-dry-run to actually upload files.")
    else:
        print(f"\n✅ Migration complete!")
        print(f"\nNext steps:")
        print(f"1. Verify uploads in S3 console:")
        print(f"   https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET}?prefix={S3_BASE_PATH}/")
        print(f"2. Update src/nfl_luck_utils.py to use load_nfl_betting_lines_from_s3()")
        print(f"3. Test analysis scripts with new S3 loading")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Migrate NFL game lines from local storage to S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration (dry run)
  python tmp/migrate_nfl_lines_to_s3.py --dry-run
  
  # Upload all files
  python tmp/migrate_nfl_lines_to_s3.py
  
  # Upload specific season
  python tmp/migrate_nfl_lines_to_s3.py --season 2025
  
  # Skip files that already exist
  python tmp/migrate_nfl_lines_to_s3.py --skip-existing
        """
    )
    
    parser.add_argument(
        '--season',
        type=int,
        choices=[2020, 2021, 2022, 2023, 2024, 2025],
        help='Migrate specific season only'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without uploading'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist in S3'
    )
    
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Confirm if not dry run
    if not args.dry_run and not args.yes:
        print(f"⚠️  This will upload files to S3 bucket: {S3_BUCKET}")
        
        if args.season:
            print(f"   Season: {args.season}")
        else:
            print(f"   All seasons (2020-2025)")
        
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("❌ Migration cancelled")
            return
    
    # Run migration
    migrate_files(
        season=args.season,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()

