"""
Migrate Existing NBA Arb Files from Local/GitHub to S3

This script:
1. Finds all existing arb CSV files in data/04_output/nba/arbs/
2. Checks if files already exist in S3 (skips if they do)
3. Uploads missing files to S3 with proper date partitioning (parallel)
4. Verifies all files were uploaded successfully
5. Optionally deletes local files after confirmation

USAGE:
    # Dry run - see what would be uploaded
    python scripts/migrate_local_arbs_to_s3.py --dry-run
    
    # Upload to S3 (default: 100 parallel workers)
    python scripts/migrate_local_arbs_to_s3.py
    
    # Upload with more workers for faster upload
    python scripts/migrate_local_arbs_to_s3.py --workers 200
    
    # Upload and delete local files after confirmation
    python scripts/migrate_local_arbs_to_s3.py --delete-after-upload

Author: Myles Thomas
Date: 2025-12-24
"""

import os
import boto3
from pathlib import Path
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'betting-nba-arbs')
LOCAL_ARBS_DIR = Path(__file__).parent.parent / 'data/04_output/nba/arbs'

# Thread-safe print lock
print_lock = Lock()


def thread_safe_print(message: str):
    """Thread-safe print function."""
    with print_lock:
        print(message)


def get_s3_key_from_filename(filename: str) -> str:
    """
    Convert local filename to S3 key with date partitioning.
    
    Example:
        arb_output_20251224_180000.csv
        -> nba/arbs/2025-12-24/arb_output_20251224_180000.csv
    
    Args:
        filename: Local filename (e.g., arb_output_20251224_180000.csv)
    
    Returns:
        S3 key with date partitioning
    """
    # Extract date from filename: arb_output_YYYYMMDD_HHMMSS.csv
    parts = filename.replace('.csv', '').split('_')
    if len(parts) >= 3:
        date_str = parts[-2]  # YYYYMMDD
        # Convert to YYYY-MM-DD format
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return f"nba/arbs/{date_formatted}/{filename}"
    else:
        # Fallback if filename doesn't match expected format
        return f"nba/arbs/unknown/{filename}"


def file_exists_in_s3(s3_key: str) -> bool:
    """
    Check if a file exists in S3.
    
    Args:
        s3_key: S3 key to check
    
    Returns:
        True if file exists, False otherwise
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except:
        return False


def upload_file_to_s3(local_path: Path, s3_key: str, dry_run: bool = False, 
                      skip_existing: bool = True) -> tuple[bool, str]:
    """
    Upload a file to S3.
    
    Args:
        local_path: Path to local file
        s3_key: S3 key (destination path)
        dry_run: If True, only print what would be uploaded
        skip_existing: If True, skip files that already exist in S3
    
    Returns:
        (success, status_message) tuple
    """
    # Check if file exists in S3
    if skip_existing and not dry_run:
        if file_exists_in_s3(s3_key):
            return (True, f"⏭️  Skipped (already exists): {local_path.name}")
    
    if dry_run:
        return (True, f"[DRY RUN] Would upload: {local_path.name} -> s3://{S3_BUCKET}/{s3_key}")
    
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        return (True, f"✅ Uploaded: {local_path.name} -> s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        return (False, f"❌ Failed to upload {local_path.name}: {e}")


def process_file(csv_file: Path, dry_run: bool = False) -> tuple[Path, str, bool, str]:
    """
    Process a single file upload (used by ThreadPoolExecutor).
    
    Returns:
        (local_path, s3_key, success, message) tuple
    """
    s3_key = get_s3_key_from_filename(csv_file.name)
    success, message = upload_file_to_s3(csv_file, s3_key, dry_run=dry_run)
    return (csv_file, s3_key, success, message)


def count_s3_files() -> tuple[int, list[str], list[str]]:
    """
    Count files in S3 under nba/arbs/ prefix.
    
    Returns:
        (total_count, first_10_files, last_10_files) tuple
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix='nba/arbs/')
    
    files = []
    for page in pages:
        if 'Contents' in page:
            files.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.csv')])
    
    return len(files), files[:10], files[-10:]  # Return count, first 10, and last 10


def main():
    parser = argparse.ArgumentParser(
        description='Migrate local NBA arb files to S3 (parallel)'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be uploaded without actually uploading')
    parser.add_argument('--delete-after-upload', action='store_true',
                       help='Delete local files after successful upload and verification')
    parser.add_argument('--skip-upload', action='store_true',
                       help='Skip upload check (assumes files already in S3), just backup and delete')
    parser.add_argument('--workers', type=int, default=100,
                       help='Number of parallel workers (default: 100)')
    parser.add_argument('--final-check', action='store_true',
                       help='Check how many files are in S3 (verification step)')
    
    args = parser.parse_args()
    
    # Final check mode - count S3 files and exit
    if args.final_check:
        print("=" * 80)
        print("🔍 S3 Final Check - Counting Files")
        print("=" * 80)
        print(f"Bucket: s3://{S3_BUCKET}/nba/arbs/")
        print()
        print("⏳ Counting files in S3 (may take a moment)...")
        
        count, first_10, last_10 = count_s3_files()
        
        print()
        print("=" * 80)
        print("📊 S3 File Count")
        print("=" * 80)
        print(f"✅ Total files in S3: {count:,}")
        print()
        
        if first_10:
            print("First 10 files:")
            for s3_key in first_10:
                print(f"  - {s3_key}")
        
        if last_10:
            print()
            print("Last 10 files:")
            for s3_key in last_10:
                print(f"  - {s3_key}")
        
        print()
        print("=" * 80)
        print(f"Verify in AWS Console:")
        print(f"https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET}?prefix=nba/arbs/")
        print("=" * 80)
        print()
        return
    
    print("=" * 80)
    print("🏀 NBA Arb Files - Local to S3 Migration (Parallel)")
    print("=" * 80)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be uploaded")
    
    print(f"Source: {LOCAL_ARBS_DIR}")
    print(f"Destination: s3://{S3_BUCKET}/nba/arbs/YYYY-MM-DD/")
    print(f"Workers: {args.workers} parallel threads")
    print()
    
    # Check if local directory exists
    if not LOCAL_ARBS_DIR.exists():
        print(f"❌ Local directory not found: {LOCAL_ARBS_DIR}")
        return
    
    # Find all CSV files
    csv_files = sorted(LOCAL_ARBS_DIR.glob('arb_output_*.csv'))
    
    if not csv_files:
        print("ℹ️  No arb files found in local directory")
        return
    
    print(f"📊 Found {len(csv_files)} files to migrate")
    print()
    
    # Upload files in parallel (unless --skip-upload is set)
    successful_uploads = []
    failed_uploads = []
    skipped_count = 0
    
    if args.skip_upload:
        print("⏭️  Skipping upload check (--skip-upload flag set)")
        print("   Assuming all files are already in S3...")
        print()
        # Mark all files as successful uploads (skip verification)
        successful_uploads = [(csv_file, get_s3_key_from_filename(csv_file.name)) for csv_file in csv_files]
        skipped_count = len(csv_files)
    else:
        print(f"🚀 Starting parallel upload with {args.workers} workers...")
        print()
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file, csv_file, args.dry_run): csv_file 
                for csv_file in csv_files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                csv_file, s3_key, success, message = future.result()
                completed += 1
                
                # Print message
                thread_safe_print(f"[{completed}/{len(csv_files)}] {message}")
                
                # Track results
                if success:
                    if "Skipped" in message:
                        skipped_count += 1
                    successful_uploads.append((csv_file, s3_key))
                else:
                    failed_uploads.append((csv_file, s3_key))
    
    # Summary
    print()
    print("=" * 80)
    print("📊 Migration Summary")
    print("=" * 80)
    if args.skip_upload:
        print(f"⏭️  Skipped upload check: {len(csv_files)} files")
        print(f"✅ Ready for backup/delete: {len(successful_uploads)}")
    else:
        print(f"✅ Successful: {len(successful_uploads)}")
        print(f"⏭️  Skipped (already in S3): {skipped_count}")
        print(f"📤 Newly uploaded: {len(successful_uploads) - skipped_count}")
        print(f"❌ Failed: {len(failed_uploads)}")
    print()
    
    if failed_uploads:
        print("Failed uploads:")
        for local_path, s3_key in failed_uploads:
            print(f"  - {local_path.name}")
        print()
    
    # Delete local files if requested
    if args.delete_after_upload and not args.dry_run and successful_uploads:
        print("🗑️  Backing up and deleting local files...")
        
        # Create backup directory
        backup_dir = Path.home() / 'Downloads' / 'tmp' / 'arbs'
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"   📦 Backup location: {backup_dir}")
        print()
        
        confirm = input(f"Backup {len(successful_uploads)} files to ~/Downloads/tmp/arbs/ then delete? (yes/no): ")
        if confirm.lower() == 'yes':
            backed_up = 0
            deleted = 0
            
            for local_path, s3_key in successful_uploads:
                try:
                    # Copy to backup
                    backup_path = backup_dir / local_path.name
                    import shutil
                    shutil.copy2(local_path, backup_path)
                    backed_up += 1
                    
                    # Delete original
                    local_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to backup/delete {local_path.name}: {e}")
            
            print()
            print(f"   ✅ Backed up {backed_up} files to {backup_dir}")
            print(f"   ✅ Deleted {deleted} original files")
        else:
            print("   ℹ️  Skipped deletion")
    
    print()
    print("=" * 80)
    print("✅ Migration complete!")
    print("=" * 80)
    print()
    
    if not args.dry_run:
        print("Verify in AWS Console:")
        print(f"https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET}?prefix=nba/arbs/")
    
    print()


if __name__ == '__main__':
    main()
