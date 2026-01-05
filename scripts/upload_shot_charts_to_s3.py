"""
Upload Shot Charts to S3

Uploads all shot chart data from local storage to S3.

SOURCE:
    Local: data/01_input/nba_api/shot_charts/{season}/
    
DESTINATION:
    S3: s3://nba-api-mt/player_shot_charts/{season}/

USAGE:
    cd /Users/thomasmyles/dev/betting
    python3 scripts/upload_shot_charts_to_s3.py

WHAT IT DOES:
    1. Finds all seasons in data/01_input/nba_api/shot_charts/
    2. Uploads each season's CSV files to S3
    3. Shows progress every 50 files
    4. Logs to: logs/upload_shot_charts_YYYYMMDD_HHMMSS.log
    
VERIFY UPLOAD:
    aws s3 ls s3://nba-api-mt/player_shot_charts/
    aws s3 ls s3://nba-api-mt/player_shot_charts/2025-26/

EXPECTED OUTPUT:
    Found 12 seasons of shot charts
    Uploading 2014-15 (424 files)...
      Progress: 50/424 files (11.8%)
      Progress: 100/424 files (23.6%)
      ...
      ✅ 2014-15: 424/424 files uploaded
    
    ✅ SUCCESS: Uploaded 5,643 shot chart files to S3
"""

import boto3
from pathlib import Path
import sys
import logging
from datetime import datetime

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'upload_shot_charts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def upload_shot_charts():
    """Upload all shot charts to S3"""
    
    logging.info("="*80)
    logging.info("UPLOAD SHOT CHARTS TO S3")
    logging.info("="*80)
    logging.info(f"Log file: {log_file}")
    
    try:
        s3 = boto3.client('s3')
        bucket = 'nba-api-mt'
        
        # Find all shot chart seasons
        shot_charts_base = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'shot_charts'
        
        if not shot_charts_base.exists():
            logging.error(f"Shot charts directory not found: {shot_charts_base}")
            return
        
        seasons = [d for d in shot_charts_base.iterdir() if d.is_dir()]
        
        logging.info(f"Found {len(seasons)} seasons of shot charts")
        logging.info(f"Seasons: {[s.name for s in seasons]}")
        logging.info(f"Bucket: s3://{bucket}")
        logging.info("")
        
        total_uploaded = 0
        total_errors = 0
        
        for season_dir in sorted(seasons):
            season = season_dir.name.replace('_', '-')
            files = list(season_dir.glob('*.csv'))
            
            if not files:
                logging.warning(f"{season}: No CSV files found, skipping")
                continue
            
            logging.info(f"Uploading {season} ({len(files)} files)...")
            
            season_uploaded = 0
            for i, file in enumerate(files, 1):
                s3_key = f'player_shot_charts/{season}/{file.name}'
                
                try:
                    s3.upload_file(str(file), bucket, s3_key)
                    season_uploaded += 1
                    
                    # Progress every 50 files
                    if i % 50 == 0:
                        logging.info(f"  Progress: {i}/{len(files)} files ({i/len(files)*100:.1f}%)")
                        
                except Exception as e:
                    logging.error(f"  Error uploading {file.name}: {e}")
                    total_errors += 1
                    # Continue instead of exit - try to upload remaining files
            
            logging.info(f"  ✅ {season}: {season_uploaded}/{len(files)} files uploaded")
            logging.info(f"     Location: s3://{bucket}/player_shot_charts/{season}/")
            logging.info("")
            total_uploaded += season_uploaded
        
        logging.info("="*80)
        if total_errors == 0:
            logging.info(f"✅ SUCCESS: Uploaded {total_uploaded} shot chart files to S3")
        else:
            logging.warning(f"⚠️  COMPLETED WITH ERRORS: {total_uploaded} uploaded, {total_errors} failed")
        logging.info(f"Location: s3://{bucket}/player_shot_charts/")
        logging.info("="*80)
        logging.info("")
        logging.info("Verify with:")
        logging.info(f"  aws s3 ls s3://{bucket}/player_shot_charts/")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    upload_shot_charts()

