"""
Migrate local futures files to S3 archive.

Context:
Move all futures-related files (data, analysis, visualizations) to S3
for long-term storage before removing from git history.

Purpose:
- Upload all futures CSVs from data/01_input/the-odds-api/{sport}/futures/
- Upload all analysis outputs from data/04_output/{sport}/
- Upload all visualizations from content/viz/{sport}/
- Organize in S3 as {sport}/archive/ for easy access

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 tmp/migrate_futures_to_s3.py [--dry-run]
    
Output S3 structure:
    s3://the-odds-api-mt/
      nfl/archive/futures/...
      nfl/archive/analysis/...
      nfl/archive/viz/...
      nba/archive/futures/...
      ncaaf/archive/futures/...
      ncaab/archive/futures/...
"""

import boto3
import os
from pathlib import Path
from datetime import datetime

# Configuration
S3_BUCKET = 'the-odds-api-mt'
AWS_REGION = os.getenv('AWS_REGION_NAME', 'us-east-2')
REPO_ROOT = Path(__file__).parent.parent

SPORTS = ['nfl', 'nba', 'ncaaf', 'ncaab']


def get_futures_files():
    """Get all futures-related files to migrate"""
    files_to_migrate = []
    
    for sport in SPORTS:
        sport_lower = sport.lower()
        
        # 1. Input futures CSVs
        futures_dir = REPO_ROOT / f'data/01_input/the-odds-api/{sport_lower}/futures'
        if futures_dir.exists():
            for csv_file in futures_dir.glob('*.csv'):
                s3_key = f"{sport_lower}/archive/futures/{csv_file.name}"
                files_to_migrate.append({
                    'local_path': csv_file,
                    's3_key': s3_key,
                    'type': 'input_data',
                    'sport': sport_lower
                })
        
        # 2. Analysis outputs
        analysis_files = [
            REPO_ROOT / f'data/04_output/{sport_lower}/{sport_lower}_championship_fair_odds.csv',
            REPO_ROOT / f'data/04_output/{sport_lower}/{sport_lower}_championship_metadata.csv',
        ]
        for analysis_file in analysis_files:
            if analysis_file.exists():
                s3_key = f"{sport_lower}/archive/analysis/{analysis_file.name}"
                files_to_migrate.append({
                    'local_path': analysis_file,
                    's3_key': s3_key,
                    'type': 'analysis',
                    'sport': sport_lower
                })
        
        # 3. Visualizations
        viz_dir = REPO_ROOT / f'content/viz/{sport_lower}'
        if viz_dir.exists():
            for viz_file in viz_dir.glob('*futures*.png'):
                s3_key = f"{sport_lower}/archive/viz/{viz_file.name}"
                files_to_migrate.append({
                    'local_path': viz_file,
                    's3_key': s3_key,
                    'type': 'visualization',
                    'sport': sport_lower
                })
    
    return files_to_migrate


def upload_to_s3(local_path, s3_key, dry_run=False):
    """Upload file to S3"""
    if dry_run:
        return True, f"[DRY RUN] Would upload: {local_path.name} -> s3://{S3_BUCKET}/{s3_key}"
    
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        return True, f"✅ Uploaded: {local_path.name} -> s3://{S3_BUCKET}/{s3_key}"
    except Exception as e:
        return False, f"❌ Failed: {local_path.name} - {e}"


def main():
    """Main migration function"""
    import sys
    
    # Check for flags
    dry_run = '--dry-run' in sys.argv
    skip_confirm = '--yes' in sys.argv or '-y' in sys.argv
    
    print("=" * 80)
    print("MIGRATE FUTURES FILES TO S3")
    print("=" * 80)
    print(f"Bucket: s3://{S3_BUCKET}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    
    # Get all files
    files = get_futures_files()
    
    if not files:
        print("⚠️  No futures files found to migrate")
        return
    
    # Group by sport and type
    by_sport = {}
    for file in files:
        sport = file['sport']
        if sport not in by_sport:
            by_sport[sport] = {'input_data': [], 'analysis': [], 'visualization': []}
        by_sport[sport][file['type']].append(file)
    
    # Show summary
    print("📊 FILES TO MIGRATE:")
    print()
    for sport in SPORTS:
        sport_lower = sport.lower()
        if sport_lower in by_sport:
            print(f"🏈 {sport.upper()}:")
            print(f"   - Input data:      {len(by_sport[sport_lower]['input_data'])} files")
            print(f"   - Analysis:        {len(by_sport[sport_lower]['analysis'])} files")
            print(f"   - Visualizations:  {len(by_sport[sport_lower]['visualization'])} files")
            print()
    
    print(f"Total files: {len(files)}")
    print()
    
    if dry_run:
        print("=" * 80)
        print("DRY RUN - Showing what would be uploaded:")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print("UPLOADING TO S3...")
        print("=" * 80)
        print()
        if not skip_confirm:
            response = input("Continue with upload? (y/n): ")
            if response.lower() != 'y':
                print("❌ Cancelled")
                return
            print()
    
    # Upload files
    success_count = 0
    fail_count = 0
    
    for file in files:
        success, message = upload_to_s3(file['local_path'], file['s3_key'], dry_run)
        print(message)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed:  {fail_count}")
    print()
    
    if not dry_run and success_count > 0:
        print("📋 FILES NOW IN S3:")
        print()
        for sport in SPORTS:
            sport_lower = sport.lower()
            if sport_lower in by_sport:
                print(f"   s3://{S3_BUCKET}/{sport_lower}/archive/")
        print()
        print("=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print()
        print("1. Verify files in S3:")
        for sport in SPORTS:
            sport_lower = sport.lower()
            if sport_lower in by_sport:
                print(f"   aws s3 ls s3://{S3_BUCKET}/{sport_lower}/archive/ --recursive")
        print()
        print("2. Delete local files:")
        print("   rm -rf data/01_input/the-odds-api/*/futures/")
        print("   rm -f data/04_output/*/{nfl,nba,ncaaf,ncaab}_championship_*.csv")
        print("   rm -f content/viz/*/{nfl,nba,ncaaf,ncaab}_futures_*.png")
        print()
        print("3. Remove from git history:")
        print("   git filter-branch --tree-filter 'rm -rf data/01_input/the-odds-api/*/futures/' HEAD")
        print("   (or use git-filter-repo for better performance)")


if __name__ == "__main__":
    main()

