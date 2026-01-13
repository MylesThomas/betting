"""
Clean up local futures files after S3 migration.

Context:
All futures files have been uploaded to S3 archive.
Now we can safely delete them locally and remove from git tracking.

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 tmp/cleanup_futures_local.py [--dry-run]
"""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

FILES_TO_DELETE = [
    # Futures input directories
    'data/01_input/the-odds-api/nfl/futures',
    'data/01_input/the-odds-api/nba/futures',
    'data/01_input/the-odds-api/ncaaf/futures',
    'data/01_input/the-odds-api/ncaab/futures',
    
    # Analysis outputs
    'data/04_output/nfl/nfl_championship_fair_odds.csv',
    'data/04_output/nfl/nfl_championship_metadata.csv',
    'data/04_output/nba/nba_championship_fair_odds.csv',
    'data/04_output/nba/nba_championship_metadata.csv',
    'data/04_output/ncaaf/ncaaf_championship_fair_odds.csv',
    'data/04_output/ncaaf/ncaaf_championship_metadata.csv',
    'data/04_output/ncaab/ncaab_championship_fair_odds.csv',
    'data/04_output/ncaab/ncaab_championship_metadata.csv',
    
    # Visualizations
    'content/viz/nfl/nfl_futures_vig_single.png',
    'content/viz/nba/nba_futures_vig_single.png',
    'content/viz/ncaaf/ncaaf_futures_vig_single.png',
    'content/viz/ncaaf/ncaaf_futures_vig_single_temp.png',
    'content/viz/ncaab/ncaab_futures_vig_single.png',
]


def delete_files(dry_run=False):
    """Delete all futures-related files"""
    print("=" * 80)
    print("CLEANUP FUTURES FILES")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    
    deleted_count = 0
    skipped_count = 0
    
    for rel_path in FILES_TO_DELETE:
        full_path = REPO_ROOT / rel_path
        
        if not full_path.exists():
            print(f"⏭️  Skip (doesn't exist): {rel_path}")
            skipped_count += 1
            continue
        
        if dry_run:
            if full_path.is_dir():
                file_count = len(list(full_path.glob('*')))
                print(f"[DRY RUN] Would delete directory: {rel_path} ({file_count} files)")
            else:
                print(f"[DRY RUN] Would delete file: {rel_path}")
            deleted_count += 1
        else:
            try:
                if full_path.is_dir():
                    file_count = len(list(full_path.glob('*')))
                    shutil.rmtree(full_path)
                    print(f"✅ Deleted directory: {rel_path} ({file_count} files)")
                else:
                    full_path.unlink()
                    print(f"✅ Deleted file: {rel_path}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {rel_path}: {e}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Would delete' if dry_run else 'Deleted'}: {deleted_count}")
    print(f"Skipped (not found): {skipped_count}")
    print()
    
    if not dry_run and deleted_count > 0:
        print("=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print()
        print("1. Check git status:")
        print("   git status")
        print()
        print("2. Stage deletions:")
        print("   git add -A")
        print()
        print("3. Commit:")
        print("   git commit -m 'futures: remove local files, migrated to S3 archive'")
        print()
        print("4. Verify files are in S3:")
        print("   aws s3 ls s3://the-odds-api-mt/nfl/archive/")
        print("   aws s3 ls s3://the-odds-api-mt/nba/archive/")
        print("   aws s3 ls s3://the-odds-api-mt/ncaaf/archive/")
        print("   aws s3 ls s3://the-odds-api-mt/ncaab/archive/")


def main():
    import sys
    dry_run = '--dry-run' in sys.argv
    
    delete_files(dry_run)


if __name__ == "__main__":
    main()


