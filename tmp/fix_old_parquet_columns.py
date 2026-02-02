"""
Fix Column Names in Old Live Odds Parquet Files

Downloads old files with 'away_moneyline'/'home_moneyline', 
renames columns to 'away_ml'/'home_ml', and re-uploads.

Author: Thomas Myles
Date: 2026-02-01
"""

import boto3
import pandas as pd
from pathlib import Path
import tempfile


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
ODDS_PREFIX = 'data/01_input/live_odds/the-odds-api/'


# =============================================================================
# MAIN
# =============================================================================

def fix_old_parquet_files():
    """Download, fix column names, and re-upload old parquet files."""
    
    s3_client = boto3.client('s3')
    
    # List all parquet files
    print("\n📥 Listing files from S3...")
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=ODDS_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.parquet'):
                    files.append(key)
    
    print(f"   Found {len(files)} parquet files\n")
    
    files_fixed = 0
    files_skipped = 0
    
    for i, key in enumerate(files, 1):
        filename = Path(key).name
        print(f"[{i}/{len(files)}] Processing {filename}...", end=" ")
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            temp_path = Path(tmp.name)
            s3_client.download_file(S3_BUCKET, key, str(temp_path))
        
        # Read parquet
        df = pd.read_parquet(temp_path)
        
        # Check if it has the old column names
        if 'away_moneyline' in df.columns or 'home_moneyline' in df.columns:
            # Rename columns
            rename_map = {}
            if 'away_moneyline' in df.columns:
                rename_map['away_moneyline'] = 'away_ml'
            if 'home_moneyline' in df.columns:
                rename_map['home_moneyline'] = 'home_ml'
            
            df = df.rename(columns=rename_map)
            
            # Write back to temp file
            df.to_parquet(temp_path, index=False)
            
            # Upload back to S3
            s3_client.upload_file(str(temp_path), S3_BUCKET, key)
            
            print(f"✅ FIXED (renamed {', '.join(rename_map.keys())})")
            files_fixed += 1
        else:
            print(f"⏭️  OK (already has away_ml/home_ml)")
            files_skipped += 1
        
        # Clean up temp file
        temp_path.unlink()
    
    print(f"\n{'='*80}")
    print(f"✅ Done!")
    print(f"   Files fixed: {files_fixed}")
    print(f"   Files skipped: {files_skipped}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    fix_old_parquet_files()
