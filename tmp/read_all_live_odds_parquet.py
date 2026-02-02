"""
Read and Combine All Live Odds Parquet Files from S3

This script:
1. Downloads all parquet files from S3 (using boto3)
2. Combines them into single parquet files using DuckDB
3. Queries the combined data

Author: Thomas Myles
Date: 2026-02-01
"""

import boto3
import duckdb
import pandas as pd
from pathlib import Path
import tempfile
import shutil


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
ODDS_PREFIX = 'data/01_input/live_odds/the-odds-api/'
ESPN_PREFIX = 'data/01_input/live_odds/espn/'


# =============================================================================
# FUNCTIONS
# =============================================================================

def download_and_combine_s3_parquets(bucket: str, prefix: str, output_name: str) -> Path:
    """
    Download all parquet files from S3 and combine them into one local parquet.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix/path
        output_name: Name for combined parquet file
        
    Returns:
        Path to combined parquet file
    """
    s3_client = boto3.client('s3')
    
    # Create temp directory for downloads
    temp_dir = Path(tempfile.mkdtemp())
    
    print(f"\n📥 Downloading from s3://{bucket}/{prefix}")
    
    # List all parquet files
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.parquet'):
                    files.append(key)
    
    print(f"   Found {len(files)} parquet files")
    
    if not files:
        print("   No files found!")
        return None
    
    # Download each file
    downloaded = []
    for i, key in enumerate(files, 1):
        filename = Path(key).name
        local_path = temp_dir / filename
        print(f"   Downloading {i}/{len(files)}: {filename}")
        s3_client.download_file(bucket, key, str(local_path))
        downloaded.append(local_path)
    
    # Combine all files into one using DuckDB
    print(f"\n🔄 Combining {len(downloaded)} files with DuckDB...")
    
    con = duckdb.connect()
    
    # Create combined parquet in user's home directory
    output_dir = Path.home() / 'Downloads' / 'tmp'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    
    query = f"""
        COPY (
            SELECT * FROM read_parquet([{','.join(f"'{p}'" for p in downloaded)}], union_by_name=true)
        ) TO '{output_path}' (FORMAT PARQUET)
    """
    
    con.execute(query)
    con.close()
    
    # Clean up temp files
    shutil.rmtree(temp_dir)
    
    print(f"   ✅ Combined into {output_path}")
    
    # Count rows
    con = duckdb.connect()
    row_count = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
    con.close()
    print(f"   📊 Total rows: {row_count:,}")
    
    return output_path


def get_live_odds_summary(odds_path: Path, espn_path: Path):
    """
    Get summary of what's currently being tracked live.
    
    Args:
        odds_path: Path to combined odds parquet
        espn_path: Path to combined ESPN parquet
    """
    print(f"\n🔥 LIVE TRACKING SUMMARY")
    print("="*80)
    
    con = duckdb.connect()
    
    # First, check what columns actually exist
    odds_columns = con.execute(f"DESCRIBE SELECT * FROM '{odds_path}'").df()
    espn_columns = con.execute(f"DESCRIBE SELECT * FROM '{espn_path}'").df()
    
    print(f"\n📋 Odds columns: {list(odds_columns['column_name'])}")
    print(f"📋 ESPN columns: {list(espn_columns['column_name'])}")
    
    # Get total counts
    odds_count = con.execute(f"SELECT COUNT(*) FROM '{odds_path}'").fetchone()[0]
    espn_count = con.execute(f"SELECT COUNT(*) FROM '{espn_path}'").fetchone()[0]
    
    print(f"\n📊 Total Records:")
    print(f"   Odds: {odds_count:,} rows")
    print(f"   ESPN: {espn_count:,} rows")
    
    # Get unique games
    unique_games = con.execute(f"SELECT COUNT(DISTINCT game_id) FROM '{odds_path}'").fetchone()[0]
    unique_books = con.execute(f"SELECT COUNT(DISTINCT bookmaker) FROM '{odds_path}'").fetchone()[0]
    
    print(f"\n📈 Odds Data:")
    print(f"   Unique games: {unique_games}")
    print(f"   Unique bookmakers: {unique_books}")
    
    # Show current live games from ESPN
    live_games_query = f"""
        SELECT 
            away_team_espn,
            home_team_espn,
            away_score,
            home_score,
            game_status
        FROM '{espn_path}'
        WHERE game_status = 'in'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY espn_game_id ORDER BY collection_timestamp DESC) = 1
    """
    
    try:
        live_games = con.execute(live_games_query).df()
        
        if len(live_games) > 0:
            print(f"\n🔥 CURRENT LIVE GAMES ({len(live_games)}):")
            for _, game in live_games.iterrows():
                score = f"[{int(game['away_score']) if pd.notna(game['away_score']) else '?'}-{int(game['home_score']) if pd.notna(game['home_score']) else '?'}]"
                print(f"   {game['away_team_espn']} @ {game['home_team_espn']} {score}")
        else:
            print(f"\n📊 No live games at the moment")
    except Exception as e:
        print(f"\n⚠️ Could not fetch live games: {e}")
    
    con.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Download, combine, and query all live odds data from S3."""
    print("\n" + "="*80)
    print("📦 LIVE ODDS DATA READER (Download + DuckDB)")
    print("="*80)
    
    # Download and combine odds data
    odds_path = download_and_combine_s3_parquets(S3_BUCKET, ODDS_PREFIX, 'combined_odds.parquet')
    
    # Download and combine ESPN data
    espn_path = download_and_combine_s3_parquets(S3_BUCKET, ESPN_PREFIX, 'combined_espn.parquet')
    
    if not odds_path or not espn_path:
        print("\n❌ Failed to download files")
        return None, None
    
    # Show live summary
    get_live_odds_summary(odds_path, espn_path)
    
    # Load into pandas for interactive use
    print("\n" + "="*80)
    print("📊 Loading full data into memory...")
    print("="*80)
    
    con = duckdb.connect()
    odds_df = con.execute(f"SELECT * FROM '{odds_path}'").df()
    espn_df = con.execute(f"SELECT * FROM '{espn_path}'").df()
    con.close()
    
    print(f"   Odds data: {len(odds_df):,} rows, {len(odds_df.columns)} columns")
    print(f"   ESPN data: {len(espn_df):,} rows, {len(espn_df.columns)} columns")
    
    if len(odds_df) > 0:
        print(f"\n📈 Odds Data Sample:")
        print(f"   Columns: {list(odds_df.columns)}")
        print(odds_df.head())
    
    if len(espn_df) > 0:
        print(f"\n🏀 ESPN Data Sample:")
        print(f"   Columns: {list(espn_df.columns)}")
        print(espn_df.head())
    
    print("\n✅ DataFrames loaded! Available as odds_df and espn_df")
    print(f"   Combined files saved to:")
    print(f"   - {odds_path}")
    print(f"   - {espn_path}")
    print("\n" + "="*80 + "\n")
    
    return odds_df, espn_df


if __name__ == '__main__':
    odds_df, espn_df = main()
