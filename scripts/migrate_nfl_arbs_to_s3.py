"""
Migrate NFL Arbitrage Data from Local Git to S3

PURPOSE:
    One-time migration script to upload existing NFL arb files from local 
    storage (data/04_output/nfl/arbs/) to S3 (betting-nfl-arbs).

WHAT IT DOES:
    1. Finds all CSV files in data/04_output/nfl/arbs/
    2. Parses timestamp from filename (arb_output_20251206_125404.csv)
    3. Uploads to S3 with date-partitioned structure:
       - Local:  data/04_output/nfl/arbs/arb_output_20251206_125404.csv
       - S3:     s3://betting-nfl-arbs/nfl/arbs/2025-12-06/arb_output_20251206_125404.csv
    4. Shows progress and summary

S3 STRUCTURE (matches Lambda function):
    s3://betting-nfl-arbs/
    └── nfl/
        └── arbs/
            ├── 2025-12-06/
            │   └── arb_output_20251206_125404.csv
            ├── 2025-12-22/
            │   └── arb_output_20251222_170102.csv
            └── ...

USAGE:
    # Dry run (no upload, just show what would happen)
    python scripts/migrate_nfl_arbs_to_s3.py --dry-run
    
    # Actually upload
    python scripts/migrate_nfl_arbs_to_s3.py
    
    # Force upload (overwrite if already exists)
    python scripts/migrate_nfl_arbs_to_s3.py --force

ENVIRONMENT VARIABLES:
    AWS_ACCESS_KEY_ID       - Your AWS access key
    AWS_SECRET_ACCESS_KEY   - Your AWS secret key
    AWS_DEFAULT_REGION      - AWS region (default: us-east-2)
    S3_BUCKET_NAME_NFL      - S3 bucket name (default: betting-nfl-arbs)

NOTES:
    - Uses ThreadPoolExecutor for parallel uploads (fast!)
    - Skips files already in S3 (unless --force is used)
    - Safe to re-run (idempotent)
    - Progress bar shows upload status
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / '.env')

# Configuration
LOCAL_ARB_DIR = PROJECT_ROOT / 'data' / '04_output' / 'nfl' / 'arbs'
S3_BUCKET = os.getenv('S3_BUCKET_NAME_NFL', 'betting-nfl-arbs')
S3_PREFIX = 'nfl/arbs'
MAX_WORKERS = 10  # Parallel upload threads


def parse_timestamp_from_filename(filename: str) -> datetime:
    """
    Parse timestamp from filename.
    
    Example: arb_output_20251206_125404.csv -> 2025-12-06 12:54:04
    
    Args:
        filename: CSV filename
    
    Returns:
        datetime object
    
    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Remove .csv extension and prefix
    # arb_output_20251206_125404.csv -> 20251206_125404
    parts = filename.replace('arb_output_', '').replace('.csv', '')
    date_part, time_part = parts.split('_')
    
    # Parse: 20251206 -> 2025-12-06, 125404 -> 12:54:04
    year = date_part[:4]
    month = date_part[4:6]
    day = date_part[6:8]
    hour = time_part[:2]
    minute = time_part[2:4]
    second = time_part[4:6]
    
    timestamp_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')


def get_s3_key_for_file(filename: str) -> str:
    """
    Generate S3 key from filename.
    
    Example: arb_output_20251206_125404.csv 
          -> nfl/arbs/2025-12-06/arb_output_20251206_125404.csv
    
    Args:
        filename: CSV filename
    
    Returns:
        S3 key (path)
    """
    timestamp = parse_timestamp_from_filename(filename)
    date_str = timestamp.strftime('%Y-%m-%d')
    return f"{S3_PREFIX}/{date_str}/{filename}"


def check_file_exists_in_s3(s3_client, bucket: str, key: str) -> bool:
    """
    Check if file already exists in S3.
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 key (path)
    
    Returns:
        True if exists, False otherwise
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Other error (permissions, etc.)
            raise


def upload_file_to_s3(s3_client, local_path: Path, bucket: str, key: str, 
                     force: bool = False, dry_run: bool = False) -> dict:
    """
    Upload single file to S3.
    
    Args:
        s3_client: boto3 S3 client
        local_path: Path to local file
        bucket: S3 bucket name
        key: S3 key (path)
        force: Overwrite if already exists
        dry_run: Don't actually upload
    
    Returns:
        dict with status and metadata
    """
    filename = local_path.name
    
    try:
        # Check if already exists
        exists = check_file_exists_in_s3(s3_client, bucket, key)
        
        if exists and not force:
            return {
                'filename': filename,
                'status': 'skipped',
                'reason': 'already exists',
                's3_key': key
            }
        
        if dry_run:
            return {
                'filename': filename,
                'status': 'dry_run',
                'reason': 'would upload' if not exists else 'would overwrite',
                's3_key': key
            }
        
        # Upload file
        s3_client.upload_file(str(local_path), bucket, key)
        
        return {
            'filename': filename,
            'status': 'uploaded',
            'reason': 'new file' if not exists else 'overwritten',
            's3_key': key
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'status': 'failed',
            'reason': str(e),
            's3_key': key
        }


def migrate_nfl_arbs_to_s3(dry_run: bool = False, force: bool = False, max_workers: int = MAX_WORKERS):
    """
    Migrate all NFL arb files from local to S3.
    
    Args:
        dry_run: Don't actually upload, just show what would happen
        force: Overwrite files that already exist in S3
        max_workers: Number of parallel upload threads
    """
    print("=" * 80)
    print("🚀 NFL ARBITRAGE DATA MIGRATION TO S3")
    print("=" * 80)
    print(f"Local directory:  {LOCAL_ARB_DIR}")
    print(f"S3 bucket:        {S3_BUCKET}")
    print(f"S3 prefix:        {S3_PREFIX}")
    print(f"Dry run:          {dry_run}")
    print(f"Force overwrite:  {force}")
    print(f"Parallel workers: {max_workers}")
    print("=" * 80)
    
    # Find all CSV files
    csv_files = sorted(LOCAL_ARB_DIR.glob('arb_output_*.csv'))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in: {LOCAL_ARB_DIR}")
        return
    
    print(f"\n✅ Found {len(csv_files)} files to migrate")
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Upload files in parallel
    results = []
    
    print(f"\n{'📤' if not dry_run else '🔍'} {'Uploading' if not dry_run else 'Analyzing'} files...\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_file = {
            executor.submit(
                upload_file_to_s3,
                s3_client,
                csv_file,
                S3_BUCKET,
                get_s3_key_for_file(csv_file.name),
                force,
                dry_run
            ): csv_file
            for csv_file in csv_files
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_file), 1):
            result = future.result()
            results.append(result)
            
            # Print progress
            status_emoji = {
                'uploaded': '✅',
                'skipped': '⏭️',
                'dry_run': '🔍',
                'failed': '❌'
            }.get(result['status'], '❓')
            
            print(f"  [{i:2d}/{len(csv_files)}] {status_emoji} {result['filename']:<40} -> {result['reason']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 MIGRATION SUMMARY")
    print("=" * 80)
    
    status_counts = {}
    for result in results:
        status_counts[result['status']] = status_counts.get(result['status'], 0) + 1
    
    print(f"Total files:     {len(csv_files)}")
    print(f"Uploaded:        {status_counts.get('uploaded', 0)}")
    print(f"Skipped:         {status_counts.get('skipped', 0)}")
    print(f"Failed:          {status_counts.get('failed', 0)}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN COMPLETE - No files were actually uploaded")
        print(f"   Run without --dry-run to perform the migration")
    else:
        print(f"\n✅ MIGRATION COMPLETE")
        print(f"   Files are now available at: s3://{S3_BUCKET}/{S3_PREFIX}/")
    
    # Show failed uploads
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print("\n❌ FAILED UPLOADS:")
        for result in failed:
            print(f"  - {result['filename']}: {result['reason']}")
    
    print("=" * 80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate NFL arb files from local storage to S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python scripts/migrate_nfl_arbs_to_s3.py --dry-run
  
  # Actually migrate
  python scripts/migrate_nfl_arbs_to_s3.py
  
  # Force overwrite existing files
  python scripts/migrate_nfl_arbs_to_s3.py --force
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite files that already exist in S3'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=MAX_WORKERS,
        help=f'Number of parallel upload threads (default: {MAX_WORKERS})'
    )
    
    args = parser.parse_args()
    
    # Check AWS credentials (boto3 will use default credential chain)
    try:
        s3_test = boto3.client('s3')
        # Quick test to verify credentials work
        s3_test.list_buckets()
    except Exception as e:
        print("❌ AWS credentials not configured or invalid")
        print(f"   Error: {e}")
        print("\n   Set up AWS credentials:")
        print("   1. Create .env file with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, OR")
        print("   2. Run: aws configure")
        sys.exit(1)
    
    # Check local directory exists
    if not LOCAL_ARB_DIR.exists():
        print(f"❌ Local directory not found: {LOCAL_ARB_DIR}")
        sys.exit(1)
    
    # Run migration
    migrate_nfl_arbs_to_s3(
        dry_run=args.dry_run,
        force=args.force,
        max_workers=args.workers
    )


if __name__ == '__main__':
    main()

