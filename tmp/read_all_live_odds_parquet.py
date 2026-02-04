"""
Read and Combine All Live Odds Parquet Files from S3

This script:
1. Downloads all parquet files from S3 (using boto3)
2. Combines them into single parquet files using DuckDB
3. Queries the combined data
4. Optionally plots ML and score changes over time

Usage:
    python tmp/read_all_live_odds_parquet.py
    python tmp/read_all_live_odds_parquet.py --recent-n-snapshots 100
    python tmp/read_all_live_odds_parquet.py --date 2026-02-02
    python tmp/read_all_live_odds_parquet.py --plot recent --date 2026-02-02
    python tmp/read_all_live_odds_parquet.py --plot live --date 2026-02-02

Plot modes:
    --plot recent: Plots the most recent game with sufficient data
    --plot live: Plots all games that are currently live (status='in')

Note: All timestamps are in ET (America/New_York) timezone

Author: Thomas Myles
Date: 2026-02-01
"""

import argparse
import boto3
import duckdb
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess
import platform
import sys
import requests
from io import BytesIO
from PIL import Image
import urllib3

# Disable SSL warnings for logo downloads
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from nba_team_colors import get_team_color


# =============================================================================
# TEAM LOGOS
# =============================================================================

def get_team_logos():
    """Get NBA team logos from ESPN - all 30 teams"""
    logo_map = {
        'Atlanta Hawks': 'https://a.espncdn.com/i/teamlogos/nba/500/atl.png',
        'Boston Celtics': 'https://a.espncdn.com/i/teamlogos/nba/500/bos.png',
        'Brooklyn Nets': 'https://a.espncdn.com/i/teamlogos/nba/500/bkn.png',
        'Charlotte Hornets': 'https://a.espncdn.com/i/teamlogos/nba/500/cha.png',
        'Chicago Bulls': 'https://a.espncdn.com/i/teamlogos/nba/500/chi.png',
        'Cleveland Cavaliers': 'https://a.espncdn.com/i/teamlogos/nba/500/cle.png',
        'Dallas Mavericks': 'https://a.espncdn.com/i/teamlogos/nba/500/dal.png',
        'Denver Nuggets': 'https://a.espncdn.com/i/teamlogos/nba/500/den.png',
        'Detroit Pistons': 'https://a.espncdn.com/i/teamlogos/nba/500/det.png',
        'Golden State Warriors': 'https://a.espncdn.com/i/teamlogos/nba/500/gs.png',
        'Houston Rockets': 'https://a.espncdn.com/i/teamlogos/nba/500/hou.png',
        'Indiana Pacers': 'https://a.espncdn.com/i/teamlogos/nba/500/ind.png',
        'LA Clippers': 'https://a.espncdn.com/i/teamlogos/nba/500/lac.png',
        'Los Angeles Clippers': 'https://a.espncdn.com/i/teamlogos/nba/500/lac.png',
        'Los Angeles Lakers': 'https://a.espncdn.com/i/teamlogos/nba/500/lal.png',
        'Memphis Grizzlies': 'https://a.espncdn.com/i/teamlogos/nba/500/mem.png',
        'Miami Heat': 'https://a.espncdn.com/i/teamlogos/nba/500/mia.png',
        'Milwaukee Bucks': 'https://a.espncdn.com/i/teamlogos/nba/500/mil.png',
        'Minnesota Timberwolves': 'https://a.espncdn.com/i/teamlogos/nba/500/min.png',
        'New Orleans Pelicans': 'https://a.espncdn.com/i/teamlogos/nba/500/no.png',
        'New York Knicks': 'https://a.espncdn.com/i/teamlogos/nba/500/ny.png',
        'Oklahoma City Thunder': 'https://a.espncdn.com/i/teamlogos/nba/500/okc.png',
        'Orlando Magic': 'https://a.espncdn.com/i/teamlogos/nba/500/orl.png',
        'Philadelphia 76ers': 'https://a.espncdn.com/i/teamlogos/nba/500/phi.png',
        'Phoenix Suns': 'https://a.espncdn.com/i/teamlogos/nba/500/phx.png',
        'Portland Trail Blazers': 'https://a.espncdn.com/i/teamlogos/nba/500/por.png',
        'Sacramento Kings': 'https://a.espncdn.com/i/teamlogos/nba/500/sac.png',
        'San Antonio Spurs': 'https://a.espncdn.com/i/teamlogos/nba/500/sa.png',
        'Toronto Raptors': 'https://a.espncdn.com/i/teamlogos/nba/500/tor.png',
        'Utah Jazz': 'https://a.espncdn.com/i/teamlogos/nba/500/utah.png',
        'Washington Wizards': 'https://a.espncdn.com/i/teamlogos/nba/500/wsh.png',
    }
    return logo_map


def download_team_logo(team_name: str):
    """
    Download team logo from ESPN.
    
    Args:
        team_name: NBA team name
        
    Returns:
        PIL Image object or None if download fails
    """
    logo_map = get_team_logos()
    logo_url = logo_map.get(team_name)
    
    if not logo_url:
        return None
    
    try:
        response = requests.get(logo_url, timeout=5, verify=False)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
    except Exception as e:
        print(f"   ⚠️  Failed to download logo for {team_name}: {e}")
    
    return None


def add_team_logos_to_figure(fig, away_team: str, home_team: str, away_color: str, home_color: str):
    """
    Add team logos to the figure title area.
    
    Args:
        fig: matplotlib figure
        away_team: Away team name
        home_team: Home team name
        away_color: Away team color (for fallback)
        home_color: Home team color (for fallback)
    """
    # Download logos
    away_logo = download_team_logo(away_team)
    home_logo = download_team_logo(home_team)
    
    if away_logo and home_logo:
        # Add logos to the figure
        # Position: left logo at 0.05, right logo at 0.95
        logo_size = 0.08  # Size relative to figure
        
        # Away logo (left)
        ax_away_logo = fig.add_axes([0.02, 0.935, logo_size, logo_size], anchor='NW')
        ax_away_logo.imshow(away_logo)
        ax_away_logo.axis('off')
        
        # Home logo (right)
        ax_home_logo = fig.add_axes([0.90, 0.935, logo_size, logo_size], anchor='NE')
        ax_home_logo.imshow(home_logo)
        ax_home_logo.axis('off')
        
        return True
    
    return False


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
ODDS_PREFIX = 'data/01_input/live_odds/the-odds-api/'
ESPN_PREFIX = 'data/01_input/live_odds/espn/'


# =============================================================================
# FUNCTIONS
# =============================================================================

def open_file(file_path: Path):
    """
    Open a file with the system's default application.
    
    Args:
        file_path: Path to the file to open
    """
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open', str(file_path)], check=True)
        elif system == 'Windows':
            subprocess.run(['start', str(file_path)], shell=True, check=True)
        elif system == 'Linux':
            subprocess.run(['xdg-open', str(file_path)], check=True)
        else:
            print(f"   ⚠️ Cannot auto-open on {system} - file saved at: {file_path}")
    except Exception as e:
        print(f"   ⚠️ Could not auto-open file: {e}")


def open_files_batch(file_paths: list[Path]):
    """
    Open multiple files with a single command.
    
    Args:
        file_paths: List of paths to open
    """
    if not file_paths:
        return
    
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open'] + [str(p) for p in file_paths], check=True)
        elif system == 'Windows':
            for path in file_paths:
                subprocess.run(['start', str(path)], shell=True, check=True)
        elif system == 'Linux':
            subprocess.run(['xdg-open'] + [str(p) for p in file_paths], check=True)
        else:
            print(f"   ⚠️ Cannot auto-open on {system}")
    except Exception as e:
        print(f"   ⚠️ Could not auto-open files: {e}")


def list_s3_parquet_files(bucket: str, prefix: str) -> list[str]:
    """
    List all parquet files in an S3 prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix/path
        
    Returns:
        List of S3 keys (file paths)
    """
    s3_client = boto3.client('s3')
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.parquet'):
                    files.append(key)
    
    return files


def get_common_timestamps(odds_files: list[str], espn_files: list[str], n_recent: int = None, date_filter: str = None) -> tuple[list[str], list[str]]:
    """
    Find files with matching timestamps from both sources and optionally limit to N most recent or specific date.
    
    Args:
        odds_files: List of S3 keys from odds prefix
        espn_files: List of S3 keys from espn prefix
        n_recent: If provided, only return the N most recent common timestamps
        date_filter: If provided (format: YYYY-MM-DD), only return files from that date in ET timezone
        
    Returns:
        Tuple of (filtered_odds_files, filtered_espn_files)
    """
    # Extract timestamps from filenames (format: YYYYMMDD_HHMMSS.parquet)
    def extract_timestamp(s3_key: str) -> str:
        filename = Path(s3_key).name
        return filename.replace('.parquet', '')
    
    odds_timestamps = {extract_timestamp(f): f for f in odds_files}
    espn_timestamps = {extract_timestamp(f): f for f in espn_files}
    
    # Find common timestamps
    common = set(odds_timestamps.keys()) & set(espn_timestamps.keys())
    
    if not common:
        print("   ⚠️ No common timestamps found between odds and ESPN data!")
        return odds_files, espn_files
    
    # Filter by date if specified
    if date_filter:
        from datetime import datetime, timedelta
        
        # Get target date and next day prefixes
        date_obj = datetime.strptime(date_filter, '%Y-%m-%d')
        next_date_obj = date_obj + timedelta(days=1)
        
        date_prefix = date_filter.replace('-', '')  # e.g., '20260203'
        next_date_prefix = next_date_obj.strftime('%Y%m%d')  # e.g., '20260204'
        
        # Filter files
        filtered_common = set()
        for ts in common:
            # Include all files from target date
            if ts.startswith(date_prefix):
                filtered_common.add(ts)
            # Include only early morning files (00:00-06:00) from next day
            elif ts.startswith(next_date_prefix):
                try:
                    # Extract hour from timestamp (format: YYYYMMDD_HHMMSS)
                    if '_' in ts:
                        hour_str = ts.split('_')[1][:2]  # Get 'HH' part
                        hour = int(hour_str)
                        if hour <= 6:  # Include files from 00:00 to 06:59
                            filtered_common.add(ts)
                except (IndexError, ValueError):
                    # Skip malformed timestamps
                    pass
        
        common = filtered_common
        print(f"   📅 Filtering to {date_filter} + next day 00:00-06:00 (for late games)")
        
        if not common:
            print(f"   ⚠️ No files found for date {date_filter}")
            return [], []
    
    # Sort timestamps (newest first)
    sorted_common = sorted(common, reverse=True)
    
    # Limit to N most recent if specified
    if n_recent:
        sorted_common = sorted_common[:n_recent]
        print(f"   📌 Limiting to {n_recent} most recent common snapshots")
    
    print(f"   ✅ Found {len(sorted_common)} common timestamps")
    
    # Filter files to only include common timestamps
    filtered_odds = [odds_timestamps[ts] for ts in sorted_common]
    filtered_espn = [espn_timestamps[ts] for ts in sorted_common]
    
    return filtered_odds, filtered_espn


def download_and_combine_s3_parquets(bucket: str, prefix: str, output_name: str, files_to_download: list[str] = None) -> Path:
    """
    Download all parquet files from S3 and combine them into one local parquet.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix/path
        output_name: Name for combined parquet file
        files_to_download: Optional list of specific S3 keys to download
        
    Returns:
        Path to combined parquet file
    """
    s3_client = boto3.client('s3')
    
    # Create temp directory for downloads
    temp_dir = Path(tempfile.mkdtemp())
    
    print(f"\n📥 Downloading from s3://{bucket}/{prefix}")
    
    # If specific files not provided, list all files
    if files_to_download is None:
        files_to_download = list_s3_parquet_files(bucket, prefix)
    
    print(f"   Found {len(files_to_download)} parquet files")
    
    if not files_to_download:
        print("   No files found!")
        return None
    
    # Download each file
    downloaded = []
    for i, key in enumerate(files_to_download, 1):
        filename = Path(key).name
        local_path = temp_dir / filename
        print(f"   Downloading {i}/{len(files_to_download)}: {filename}")
        s3_client.download_file(bucket, key, str(local_path))
        downloaded.append(local_path)
    
    # Combine all files into one using DuckDB
    print(f"\n🔄 Combining {len(downloaded)} files with DuckDB...")
    
    # Create combined parquet in user's home directory
    output_dir = Path.home() / 'Downloads' / 'tmp'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    
    # Remove existing file to avoid lock issues
    if output_path.exists():
        output_path.unlink()
    
    con = duckdb.connect()
    
    try:
        query = f"""
            COPY (
                SELECT * FROM read_parquet([{','.join(f"'{p}'" for p in downloaded)}], union_by_name=true)
            ) TO '{output_path}' (FORMAT PARQUET)
        """
        
        con.execute(query)
    finally:
        con.close()
    
    # Clean up temp files
    shutil.rmtree(temp_dir)
    
    print(f"   ✅ Combined into {output_path}")
    
    # Count rows with a fresh connection
    con = duckdb.connect()
    try:
        row_count = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
        print(f"   📊 Total rows: {row_count:,}")
    finally:
        con.close()
    
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
    print(f"\n📋 ESPN columns: {list(espn_columns['column_name'])}")
    
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


def find_most_recent_game_with_data(odds_path: Path, espn_path: Path) -> tuple[str, str]:
    """
    Find the most recent game that has both odds and score data.
    
    Args:
        odds_path: Path to combined odds parquet
        espn_path: Path to combined ESPN parquet
        
    Returns:
        Tuple of (away_team, home_team) or (None, None)
    """
    con = duckdb.connect()
    
    query = f"""
        SELECT 
            o.away_team,
            o.home_team,
            MAX(o.fetched_at) as latest_snapshot,
            COUNT(DISTINCT o.fetched_at) as num_snapshots
        FROM '{odds_path}' o
        INNER JOIN '{espn_path}' e
            ON o.away_team = e.away_team_espn
            AND o.home_team = e.home_team_espn
            AND o.fetched_at = e.collection_timestamp
        WHERE e.game_status IS NOT NULL
        GROUP BY o.away_team, o.home_team
        HAVING COUNT(DISTINCT o.fetched_at) >= 5
        ORDER BY latest_snapshot DESC
        LIMIT 1
    """
    
    try:
        result = con.execute(query).df()
        if len(result) > 0:
            away = result['away_team'].iloc[0]
            home = result['home_team'].iloc[0]
            print(f"\n📍 Most recent game with data: {away} @ {home}")
            print(f"   Snapshots: {result['num_snapshots'].iloc[0]}")
            return away, home
    except Exception as e:
        print(f"\n⚠️ Error finding recent game: {e}")
    finally:
        con.close()
    
    return None, None


def find_all_live_games(odds_path: Path, espn_path: Path) -> list[tuple[str, str, int]]:
    """
    Find all games that are currently live (most recent snapshot has status='in').
    
    Args:
        odds_path: Path to combined odds parquet
        espn_path: Path to combined ESPN parquet
        
    Returns:
        List of tuples: (away_team, home_team, num_snapshots)
    """
    con = duckdb.connect()
    
    # First, let's see all games with status='in' and their snapshot counts
    debug_query = f"""
        WITH latest_status AS (
            SELECT 
                e.away_team_espn as away_team,
                e.home_team_espn as home_team,
                e.game_status,
                e.collection_timestamp,
                ROW_NUMBER() OVER (PARTITION BY e.away_team_espn, e.home_team_espn ORDER BY e.collection_timestamp DESC) as rn
            FROM '{espn_path}' e
        ),
        game_stats AS (
            SELECT 
                o.away_team,
                o.home_team,
                COUNT(DISTINCT o.fetched_at) as num_snapshots
            FROM '{odds_path}' o
            INNER JOIN '{espn_path}' e
                ON o.away_team = e.away_team_espn
                AND o.home_team = e.home_team_espn
                AND o.fetched_at = e.collection_timestamp
            WHERE e.game_status IS NOT NULL
            GROUP BY o.away_team, o.home_team
        )
        SELECT 
            ls.away_team,
            ls.home_team,
            COALESCE(gs.num_snapshots, 0) as num_snapshots,
            ls.game_status,
            ls.collection_timestamp as latest_snapshot
        FROM latest_status ls
        LEFT JOIN game_stats gs
            ON ls.away_team = gs.away_team
            AND ls.home_team = gs.home_team
        WHERE ls.rn = 1
          AND ls.game_status = 'in'
        ORDER BY ls.collection_timestamp DESC
    """
    
    try:
        debug_result = con.execute(debug_query).df()
        
        if len(debug_result) > 0:
            print(f"\n🔍 Debug: Found {len(debug_result)} live games (status='in'):")
            for _, row in debug_result.iterrows():
                print(f"   • {row['away_team']} @ {row['home_team']}: {row['num_snapshots']} snapshots")
        
        # Now get games with sufficient data (lowered threshold to 3)
        query = f"""
            WITH latest_status AS (
                SELECT 
                    e.away_team_espn as away_team,
                    e.home_team_espn as home_team,
                    e.game_status,
                    e.collection_timestamp,
                    ROW_NUMBER() OVER (PARTITION BY e.away_team_espn, e.home_team_espn ORDER BY e.collection_timestamp DESC) as rn
                FROM '{espn_path}' e
            ),
            game_stats AS (
                SELECT 
                    o.away_team,
                    o.home_team,
                    COUNT(DISTINCT o.fetched_at) as num_snapshots
                FROM '{odds_path}' o
                INNER JOIN '{espn_path}' e
                    ON o.away_team = e.away_team_espn
                    AND o.home_team = e.home_team_espn
                    AND o.fetched_at = e.collection_timestamp
                WHERE e.game_status IS NOT NULL
                GROUP BY o.away_team, o.home_team
            )
            SELECT 
                ls.away_team,
                ls.home_team,
                gs.num_snapshots,
                ls.collection_timestamp as latest_snapshot
            FROM latest_status ls
            INNER JOIN game_stats gs
                ON ls.away_team = gs.away_team
                AND ls.home_team = gs.home_team
            WHERE ls.rn = 1
              AND ls.game_status = 'in'
              AND gs.num_snapshots >= 1
            ORDER BY ls.collection_timestamp DESC
        """
        
        result = con.execute(query).df()
        
        if len(result) == 0:
            print("\n⚠️ No live games found with sufficient data (need >= 1 snapshot with both odds and score)")
            return []
        
        print(f"\n🔴 Found {len(result)} live game(s) with sufficient data:")
        live_games = []
        for _, row in result.iterrows():
            away = row['away_team']
            home = row['home_team']
            snapshots = row['num_snapshots']
            latest = row['latest_snapshot']
            print(f"   • {away} @ {home} ({snapshots} snapshots, latest: {latest})")
            live_games.append((away, home, snapshots))
        
        return live_games
        
    except Exception as e:
        print(f"\n⚠️ Error finding live games: {e}")
        return []
    finally:
        con.close()


def find_all_games_with_data(odds_path: Path, espn_path: Path, target_date: str, min_snapshots: int = 5) -> list[tuple[str, str, int]]:
    """
    Find all games from a specific date (by game_time) that have sufficient data.
    
    Args:
        odds_path: Path to combined odds parquet
        espn_path: Path to combined ESPN parquet
        target_date: Target date in YYYY-MM-DD format (ET timezone)
        min_snapshots: Minimum number of snapshots required (default: 5)
        
    Returns:
        List of tuples: (away_team, home_team, num_snapshots)
    """
    con = duckdb.connect()
    
    try:
        # Find all games scheduled on target date with sufficient data
        # NOTE: game_time is in UTC, so we convert to ET timezone (UTC-5) before extracting date
        query = f"""
            WITH game_stats AS (
                SELECT 
                    o.away_team,
                    o.home_team,
                    MIN(o.game_time) as scheduled_tipoff,
                    COUNT(DISTINCT o.fetched_at) as num_snapshots
                FROM '{odds_path}' o
                INNER JOIN '{espn_path}' e
                    ON o.away_team = e.away_team_espn
                    AND o.home_team = e.home_team_espn
                    AND o.fetched_at = e.collection_timestamp
                WHERE e.game_status IS NOT NULL
                  AND DATE(o.game_time::TIMESTAMP - INTERVAL 5 HOURS) = '{target_date}'
                GROUP BY o.away_team, o.home_team
                HAVING COUNT(DISTINCT o.fetched_at) >= {min_snapshots}
            )
            SELECT 
                away_team,
                home_team,
                num_snapshots,
                scheduled_tipoff
            FROM game_stats
            ORDER BY scheduled_tipoff
        """
        
        result = con.execute(query).df()
        
        if len(result) == 0:
            print(f"\n⚠️ No games found on {target_date} with >= {min_snapshots} snapshots")
            return []
        
        print(f"\n📅 Found {len(result)} game(s) on {target_date} with >= {min_snapshots} snapshots:")
        all_games = []
        for _, row in result.iterrows():
            away = row['away_team']
            home = row['home_team']
            snapshots = row['num_snapshots']
            tipoff = row['scheduled_tipoff']
            print(f"   • {away} @ {home} ({snapshots} snapshots, tipoff: {tipoff})")
            all_games.append((away, home, snapshots))
        
        return all_games
        
    except Exception as e:
        print(f"\n⚠️ Error finding games for {target_date}: {e}")
        return []
    finally:
        con.close()


def plot_ml_and_score_movement(odds_path: Path, espn_path: Path, away_team: str, home_team: str):
    """
    Create a plot showing ML odds and score changes over time.
    
    Args:
        odds_path: Path to combined odds parquet
        espn_path: Path to combined ESPN parquet
        away_team: Away team name
        home_team: Home team name
        
    Returns:
        Path to the generated plot file
    """
    con = duckdb.connect()
    
    try:
        # Get ML, spread, and score movement over time
        query = f"""
            SELECT 
                o.fetched_at as timestamp,
                o.game_time as scheduled_tipoff,
                MEDIAN(o.away_ml) as away_ml,
                MEDIAN(o.home_ml) as home_ml,
                MEDIAN(o.away_ml_implied_prob) as away_implied_prob,
                MEDIAN(o.home_ml_implied_prob) as home_implied_prob,
                MEDIAN(o.away_spread) as away_spread,
                MEDIAN(o.home_spread) as home_spread,
                e.away_score,
                e.home_score,
                e.game_status,
                e.period
            FROM '{odds_path}' o
            LEFT JOIN '{espn_path}' e
                ON o.fetched_at = e.collection_timestamp
                AND o.away_team = e.away_team_espn
                AND o.home_team = e.home_team_espn
            WHERE o.away_team = '{away_team}'
              AND o.home_team = '{home_team}'
            GROUP BY o.fetched_at, o.game_time, e.away_score, e.home_score, e.game_status, e.period
            ORDER BY o.fetched_at
        """
        
        df = con.execute(query).df()
    finally:
        con.close()
    
    if len(df) == 0:
        print("\n❌ No data found for plotting")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter to data where scores exist
    df_with_scores = df[df['away_score'].notna() & df['home_score'].notna()].copy()
    
    if len(df_with_scores) == 0:
        print("\n❌ No score data available for plotting")
        return
    
    # Detect true game start: find 0-0 score or first basket
    print(f"\n🔍 Detecting true game start...")
    zero_zero_time = None
    first_basket_time = None
    
    # First, try to find a 0-0 score
    for idx, row in df_with_scores.iterrows():
        away_score = row['away_score']
        home_score = row['home_score']
        
        if away_score == 0 and home_score == 0:
            zero_zero_time = row['timestamp']
            print(f"   0️⃣  Found 0-0 at: {zero_zero_time}")
        
        # Find first basket
        if first_basket_time is None and (away_score > 0 or home_score > 0):
            first_basket_time = row['timestamp']
            print(f"   🏀 First basket at: {first_basket_time} (score: {int(away_score)}-{int(home_score)})")
            break
    
    # Determine the start time for our plot
    if zero_zero_time:
        game_start_time = zero_zero_time
        print(f"   ✅ Using 0-0 timestamp as game start")
    elif first_basket_time:
        # No 0-0 found, use 10 seconds before first basket
        game_start_time = first_basket_time - pd.Timedelta(seconds=10)
        print(f"   ⚠️  No 0-0 found, using 10s before first basket as game start")
    else:
        print("   ⚠️ No baskets scored yet (game at 0-0), using first score appearance")
        game_start_time = df_with_scores['timestamp'].min()
    
    # Filter data to start from game start
    df_filtered = df[df['timestamp'] >= game_start_time].copy()
    df_with_scores_game = df_with_scores[df_with_scores['timestamp'] >= game_start_time].copy()
    
    if len(df_with_scores_game) == 0:
        print("\n❌ No score data after game start")
        return
    
    # Detect game end (when game_status changes from 'in' to 'post')
    print(f"\n🏁 Detecting game end...")
    game_end_time = None
    df_with_status = df_filtered[df_filtered['game_status'].notna()].copy()
    
    if len(df_with_status) > 0:
        # Find the first occurrence of 'post' status
        post_status_rows = df_with_status[df_with_status['game_status'] == 'post']
        if len(post_status_rows) > 0:
            game_end_time = post_status_rows['timestamp'].min()
            print(f"   ✅ Game ended at: {game_end_time}")
            
            # Filter out all data after game end
            df_filtered = df_filtered[df_filtered['timestamp'] <= game_end_time].copy()
            df_with_scores_game = df_with_scores_game[df_with_scores_game['timestamp'] <= game_end_time].copy()
            print(f"   📊 Filtered data to game end (removed post-game odds movement)")
        else:
            print(f"   ⚠️  Game still in progress (no 'post' status found)")
    
    # Determine pregame favorite based on opening ML odds
    # Lower (more negative) ML = favorite, higher (more positive) ML = underdog
    opening_away_ml = df_filtered['away_ml'].dropna().iloc[0] if len(df_filtered['away_ml'].dropna()) > 0 else 0
    opening_home_ml = df_filtered['home_ml'].dropna().iloc[0] if len(df_filtered['home_ml'].dropna()) > 0 else 0
    
    # More negative ML = favorite
    if opening_away_ml < opening_home_ml:
        favorite_team = away_team
        underdog_team = home_team
        favorite_is_away = True
        print(f"   ⭐ Pregame favorite: {favorite_team} ({int(opening_away_ml):+d} ML)")
        print(f"   🐶 Pregame underdog: {underdog_team} ({int(opening_home_ml):+d} ML)")
    else:
        favorite_team = home_team
        underdog_team = away_team
        favorite_is_away = False
        print(f"   ⭐ Pregame favorite: {favorite_team} ({int(opening_home_ml):+d} ML)")
        print(f"   🐶 Pregame underdog: {underdog_team} ({int(opening_away_ml):+d} ML)")
    
    # Calculate score differential (Favorite - Underdog, so positive means favorite is winning)
    if favorite_is_away:
        df_with_scores_game['score_diff'] = df_with_scores_game['away_score'] - df_with_scores_game['home_score']
    else:
        df_with_scores_game['score_diff'] = df_with_scores_game['home_score'] - df_with_scores_game['away_score']
    
    # Convert timestamps to minutes since game start for easier reading
    df_filtered['minutes_since_start'] = (df_filtered['timestamp'] - game_start_time).dt.total_seconds() / 60
    df_with_scores_game['minutes_since_start'] = (df_with_scores_game['timestamp'] - game_start_time).dt.total_seconds() / 60
    
    # Add a 0,0 point at game start time (0 minutes)
    first_row = pd.DataFrame({
        'timestamp': [game_start_time],
        'minutes_since_start': [0],
        'away_ml': [df_filtered['away_ml'].iloc[0] if len(df_filtered) > 0 else None],
        'home_ml': [df_filtered['home_ml'].iloc[0] if len(df_filtered) > 0 else None],
        'away_implied_prob': [df_filtered['away_implied_prob'].iloc[0] if len(df_filtered) > 0 else None],
        'home_implied_prob': [df_filtered['home_implied_prob'].iloc[0] if len(df_filtered) > 0 else None],
        'away_spread': [df_filtered['away_spread'].iloc[0] if len(df_filtered) > 0 else None],
        'home_spread': [df_filtered['home_spread'].iloc[0] if len(df_filtered) > 0 else None],
        'score_diff': [0]
    })
    
    df_with_scores_full = pd.concat([first_row, df_with_scores_game[['timestamp', 'minutes_since_start', 'score_diff']]], ignore_index=True)
    df_ml_full = pd.concat([first_row[['timestamp', 'minutes_since_start', 'away_ml', 'home_ml', 'away_implied_prob', 'home_implied_prob', 'away_spread', 'home_spread']], 
                            df_filtered[['timestamp', 'minutes_since_start', 'away_ml', 'home_ml', 'away_implied_prob', 'home_implied_prob', 'away_spread', 'home_spread']]], ignore_index=True)
    
    # Remove duplicates (in case first_score_time already had data)
    df_with_scores_full = df_with_scores_full.drop_duplicates(subset=['timestamp'], keep='last')
    df_ml_full = df_ml_full.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Get team colors
    away_color = get_team_color(away_team, 'primary')
    home_color = get_team_color(home_team, 'primary')
    favorite_color = away_color if favorite_is_away else home_color
    underdog_color = home_color if favorite_is_away else away_color
    print(f"   🎨 Using colors: {away_team}={away_color}, {home_team}={home_color}")
    
    # Detect quarter endings and starts by looking for period changes
    print(f"\n🔍 Detecting quarter transitions...")
    quarter_end_times = []
    quarter_start_times = []
    df_with_period = df_filtered[df_filtered['period'].notna()].copy()
    
    if len(df_with_period) > 0:
        # Look for transitions between periods
        prev_period = None
        for idx, row in df_with_period.iterrows():
            current_period = row['period']
            if prev_period is not None and current_period != prev_period and current_period > prev_period:
                # Period changed - this timestamp is when we first saw the new period
                # So the previous period ended and the new period started
                transition_time = row['timestamp']
                quarter_end_times.append((prev_period, transition_time))
                quarter_start_times.append((current_period, transition_time))
                print(f"   📍 End of Q{int(prev_period)} / Start of Q{int(current_period)}: {transition_time}")
            prev_period = current_period
    
    # Get current time (latest timestamp in data)
    current_time = df_filtered['timestamp'].max() if len(df_filtered) > 0 else df['timestamp'].max()
    print(f"   🕐 Current time (latest data): {current_time}")
    
    # Get scheduled tipoff time for title
    scheduled_tipoff = df['scheduled_tipoff'].iloc[0] if 'scheduled_tipoff' in df.columns and len(df) > 0 else None
    tipoff_str = ""
    if pd.notna(scheduled_tipoff):
        tipoff_dt = pd.to_datetime(scheduled_tipoff)
        if tipoff_dt.tzinfo is None:
            tipoff_dt = tipoff_dt.tz_localize('UTC')
        tipoff_et = tipoff_dt.astimezone(ZoneInfo('America/New_York'))
        tipoff_str = f" ({tipoff_et.strftime('%Y-%m-%d')} | Tip: {tipoff_et.strftime('%I:%M %p ET')})"
    
    # Create figure with four subplots (ML, Spread, Win %, Score Differential)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'{away_team} @ {home_team}{tipoff_str}\nML Odds, Spread, Win Probability, and Score Differential Over Time', 
                 fontsize=16, fontweight='bold')
    
    # Add team logos to the figure
    print(f"\n🖼️  Adding team logos...")
    logos_added = add_team_logos_to_figure(fig, away_team, home_team, away_color, home_color)
    if logos_added:
        print(f"   ✅ Team logos added to title")
    else:
        print(f"   ⚠️  Could not add team logos")
    
    # Plot 1: ML Odds with consistent team colors and background shading
    # Get opening and current ML values for legend
    away_ml_open = df_ml_full['away_ml'].dropna().iloc[0] if len(df_ml_full['away_ml'].dropna()) > 0 else None
    away_ml_current = df_ml_full['away_ml'].dropna().iloc[-1] if len(df_ml_full['away_ml'].dropna()) > 0 else None
    home_ml_open = df_ml_full['home_ml'].dropna().iloc[0] if len(df_ml_full['home_ml'].dropna()) > 0 else None
    home_ml_current = df_ml_full['home_ml'].dropna().iloc[-1] if len(df_ml_full['home_ml'].dropna()) > 0 else None
    
    # Format ML values with +/- signs
    def format_ml(val):
        if val is None or pd.isna(val):
            return "N/A"
        return f"{int(val):+d}"
    
    away_label = f'{away_team} (At open: {format_ml(away_ml_open)}; Currently: {format_ml(away_ml_current)})'
    home_label = f'{home_team} (At open: {format_ml(home_ml_open)}; Currently: {format_ml(home_ml_current)})'
    
    ax1.plot(df_ml_full['minutes_since_start'], df_ml_full['away_ml'], 'o-', label=away_label, 
             color=away_color, linewidth=2, markersize=4)
    ax1.plot(df_ml_full['minutes_since_start'], df_ml_full['home_ml'], 's-', label=home_label, 
             color=home_color, linewidth=2, markersize=4)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax1.set_ylabel('Moneyline Odds', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Moneyline Movement', fontsize=14)
    
    # Set symmetric y-axis range for ML using percentiles to avoid outlier issues
    all_ml_values = pd.concat([df_ml_full['away_ml'], df_ml_full['home_ml']]).dropna()
    if len(all_ml_values) > 0:
        # Use 98th percentile to ignore extreme outliers
        ml_max = max(abs(all_ml_values.quantile(0.02)), abs(all_ml_values.quantile(0.98)))
        # Ensure minimum range of 500 for readability, cap at 2000 to avoid excessive zoom-out
        ml_max = min(max(ml_max, 500), 2000)
        ax1.set_ylim(-ml_max * 1.1, ml_max * 1.1)  # Add 10% padding
        
        # Add background shading for favorite/underdog zones
        ax1.axhspan(-ml_max * 1.1, 0, alpha=0.1, color='green', label='Favorite Zone')
        ax1.axhspan(0, ml_max * 1.1, alpha=0.1, color='red', label='Underdog Zone')
        
        # Check if there are outliers and add a note
        outliers = all_ml_values[(all_ml_values < -ml_max * 1.1) | (all_ml_values > ml_max * 1.1)]
        if len(outliers) > 0:
            print(f"   ⚠️  {len(outliers)} ML outlier(s) excluded from y-axis range (values: {outliers.tolist()})")
    
    # Plot 2: Spread Movement with consistent team colors
    # Get opening and current spread values for legend
    away_spread_open = df_ml_full['away_spread'].dropna().iloc[0] if len(df_ml_full['away_spread'].dropna()) > 0 else None
    away_spread_current = df_ml_full['away_spread'].dropna().iloc[-1] if len(df_ml_full['away_spread'].dropna()) > 0 else None
    home_spread_open = df_ml_full['home_spread'].dropna().iloc[0] if len(df_ml_full['home_spread'].dropna()) > 0 else None
    home_spread_current = df_ml_full['home_spread'].dropna().iloc[-1] if len(df_ml_full['home_spread'].dropna()) > 0 else None
    
    # Format spread values with +/- signs
    def format_spread(val):
        if val is None or pd.isna(val):
            return "N/A"
        return f"{val:+.1f}"
    
    away_spread_label = f'{away_team} (At open: {format_spread(away_spread_open)}; Currently: {format_spread(away_spread_current)})'
    home_spread_label = f'{home_team} (At open: {format_spread(home_spread_open)}; Currently: {format_spread(home_spread_current)})'
    
    ax2.plot(df_ml_full['minutes_since_start'], df_ml_full['away_spread'], 'o-', label=away_spread_label, 
             color=away_color, linewidth=2, markersize=4)
    ax2.plot(df_ml_full['minutes_since_start'], df_ml_full['home_spread'], 's-', label=home_spread_label, 
             color=home_color, linewidth=2, markersize=4)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax2.set_ylabel('Spread', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Spread Movement', fontsize=14)
    
    # Set symmetric y-axis range for spread
    all_spread_values = pd.concat([df_ml_full['away_spread'], df_ml_full['home_spread']]).dropna()
    if len(all_spread_values) > 0:
        spread_max = max(abs(all_spread_values.min()), abs(all_spread_values.max()))
        spread_max = max(spread_max, 5)  # Ensure minimum range of 5 for readability
        ax2.set_ylim(-spread_max * 1.2, spread_max * 1.2)  # Add 20% padding
    
    # Plot 3: Implied Win Probability - only show the favored team (>= 50%)
    # Create a column for favored team's probability
    df_ml_full['favored_prob'] = df_ml_full.apply(
        lambda row: max(row['away_implied_prob'], row['home_implied_prob']) * 100 
        if pd.notna(row['away_implied_prob']) and pd.notna(row['home_implied_prob']) 
        else None, 
        axis=1
    )
    
    # Plot line segments with appropriate colors based on who's favored
    # When away team is favored (away_prob >= 50)
    away_favored_mask = df_ml_full['away_implied_prob'] * 100 >= 50
    if away_favored_mask.any():
        ax3.plot(df_ml_full.loc[away_favored_mask, 'minutes_since_start'], 
                df_ml_full.loc[away_favored_mask, 'favored_prob'],
                'o-', color=away_color, linewidth=2, markersize=4, label=f'{away_team} Favored')
    
    # When home team is favored (home_prob >= 50)
    home_favored_mask = df_ml_full['home_implied_prob'] * 100 >= 50
    if home_favored_mask.any():
        ax3.plot(df_ml_full.loc[home_favored_mask, 'minutes_since_start'], 
                df_ml_full.loc[home_favored_mask, 'favored_prob'],
                's-', color=home_color, linewidth=2, markersize=4, label=f'{home_team} Favored')
    
    # Add shading for which team is favored
    ax3.fill_between(df_ml_full['minutes_since_start'], 
                     df_ml_full['favored_prob'], 
                     50,
                     where=(df_ml_full['away_implied_prob'] * 100 >= 50),
                     alpha=0.3, color=away_color, interpolate=True)
    ax3.fill_between(df_ml_full['minutes_since_start'], 
                     df_ml_full['favored_prob'], 
                     50,
                     where=(df_ml_full['home_implied_prob'] * 100 >= 50),
                     alpha=0.3, color=home_color, interpolate=True)
    
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.set_ylabel('Implied Win Probability (%)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    # No legend needed - implied from plot 1 and team colors/shading
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Win Probability Movement', fontsize=14)
    
    # Plot 4: Score Differential (Favorite - Underdog)
    ax4.plot(df_with_scores_full['minutes_since_start'], df_with_scores_full['score_diff'], 
             'o-', linewidth=2, markersize=6, color='#2ca02c')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.fill_between(df_with_scores_full['minutes_since_start'], 
                     df_with_scores_full['score_diff'], 
                     0,
                     where=(df_with_scores_full['score_diff'] >= 0),
                     alpha=0.3, color=favorite_color, label=f'{favorite_team} Leading')
    ax4.fill_between(df_with_scores_full['minutes_since_start'], 
                     df_with_scores_full['score_diff'], 
                     0,
                     where=(df_with_scores_full['score_diff'] < 0),
                     alpha=0.3, color=underdog_color, label=f'{underdog_team} Leading')
    
    # Set symmetric y-axis range for score differential
    score_max = max(abs(df_with_scores_full['score_diff'].max()), abs(df_with_scores_full['score_diff'].min()))
    ax4.set_ylim(-score_max * 1.1, score_max * 1.1)  # Add 10% padding
    
    # Get score for title with FINAL/CURRENT label
    if len(df_with_scores_game) > 0:
        final_row = df_with_scores_game.iloc[-1]
        final_away_score = int(final_row['away_score'])
        final_home_score = int(final_row['home_score'])
        score_label = 'FINAL SCORE' if game_end_time is not None else 'CURRENT SCORE'
        score_title = f'Score Differential - Favorite vs Underdog ({score_label}: {away_team} {final_away_score} - {home_team} {final_home_score})'
    else:
        score_title = 'Score Differential - Favorite vs Underdog (starts at 0)'
    
    ax4.set_ylabel('Score Differential\n(Favorite - Underdog)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Minutes Since Tipoff', fontsize=12, fontweight='bold')
    # No legend needed - team colors/shading make it clear
    ax4.grid(True, alpha=0.3)
    ax4.set_title(score_title, fontsize=14)
    
    # Add reference lines to all four plots
    print(f"\n🎨 Adding reference lines...")
    
    # Convert key timestamps to minutes since start
    game_start_minutes = 0
    current_time_minutes = (current_time - game_start_time).total_seconds() / 60
    
    # 1. Game start (dotted red) - always at 0
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=game_start_minutes, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Game Start')
    print(f"   🟥 Game start line at {game_start_minutes} minutes")
    
    # 2. Game end (solid red line if game finished)
    if game_end_time is not None:
        game_end_minutes = (game_end_time - game_start_time).total_seconds() / 60
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(x=game_end_minutes, color='darkred', linestyle='-', linewidth=2.5, alpha=0.8, label='Game End')
        print(f"   🏁 Game end line at {game_end_minutes:.1f} minutes")
    
    # 3. Current time (dotted red) - only if game is still in progress
    if game_end_time is None:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(x=current_time_minutes, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Current Time')
        print(f"   🟥 Current time line at {current_time_minutes:.1f} minutes")
    
    # 4. Quarter endings (dashed gray lines)
    for period, end_time in quarter_end_times:
        end_minutes = (end_time - game_start_time).total_seconds() / 60
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(x=end_minutes, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.text(end_minutes, ax.get_ylim()[1] * 0.95, f'Q{int(period)} End', 
                   rotation=90, verticalalignment='top', fontsize=9, alpha=0.7)
        print(f"   ⬜ Q{int(period)} end line at {end_minutes:.1f} minutes")
    
    # 5. Quarter starts (dashed blue lines)
    for period, start_time in quarter_start_times:
        start_minutes = (start_time - game_start_time).total_seconds() / 60
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(x=start_minutes, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.text(start_minutes, ax.get_ylim()[0] * 0.95, f'Q{int(period)} Start', 
                   rotation=90, verticalalignment='bottom', fontsize=9, alpha=0.7, color='blue')
        print(f"   🟦 Q{int(period)} start line at {start_minutes:.1f} minutes")
    
    # Format x-axis
    ax4.set_xlim(left=0)  # Start at 0 minutes
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with timestamp (no spaces in filename)
    timestamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')
    away_clean = away_team.replace(' ', '_')
    home_clean = home_team.replace(' ', '_')
    output_path = Path.home() / 'Downloads' / 'tmp' / f'ml_score_movement_{away_clean}_{home_clean}_{timestamp}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Close the plot to free memory
    plt.close()
    
    return output_path
    
    # Show plot
    plt.show()
    
    print(f"\n📊 Data Summary:")
    print(f"   Total snapshots: {len(df)}")
    print(f"   Snapshots with scores: {len(df_with_scores)}")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if len(df) > 0:
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        print(f"\n📈 ML Movement:")
        if pd.notna(first_row['away_ml']) and pd.notna(last_row['away_ml']):
            away_ml_change = last_row['away_ml'] - first_row['away_ml']
            print(f"   {away_team}: {int(first_row['away_ml']):+d} → {int(last_row['away_ml']):+d} (Δ {int(away_ml_change):+d})")
        
        if pd.notna(first_row['home_ml']) and pd.notna(last_row['home_ml']):
            home_ml_change = last_row['home_ml'] - first_row['home_ml']
            print(f"   {home_team}: {int(first_row['home_ml']):+d} → {int(last_row['home_ml']):+d} (Δ {int(home_ml_change):+d})")
    
    if len(df_with_scores) > 0:
        last_score_row = df_with_scores.iloc[-1]
        print(f"\n🏀 Final Score: {away_team} {int(last_score_row['away_score'])} - {home_team} {int(last_score_row['home_score'])}")
    
    print()



# =============================================================================
# MAIN
# =============================================================================

def main():
    """Download, combine, and query all live odds data from S3."""
    parser = argparse.ArgumentParser(
        description='Download and combine live odds data from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--recent-n-snapshots',
        type=int,
        default=None,
        help='Only download the N most recent snapshots that exist in both odds and ESPN data'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Filter to snapshots from a specific date (format: YYYY-MM-DD, ET timezone)'
    )
    parser.add_argument(
        '--plot',
        type=str,
        choices=['recent', 'live', 'all'],
        default=None,
        help='Create plots: "recent" = most recent game with data, "live" = all currently live games, "all" = all games from date'
    )
    parser.add_argument(
        '--min-snapshots',
        type=int,
        default=5,
        help='Minimum number of snapshots required for a game when using --plot all (default: 5)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📦 LIVE ODDS DATA READER (Download + DuckDB)")
    print("="*80)
    
    # If limiting to recent snapshots or filtering by date, find common timestamps first
    if args.recent_n_snapshots or args.date:
        if args.recent_n_snapshots:
            print(f"\n🔍 Finding {args.recent_n_snapshots} most recent common snapshots...")
        if args.date:
            print(f"\n🔍 Finding snapshots for date {args.date}...")
        
        all_odds_files = list_s3_parquet_files(S3_BUCKET, ODDS_PREFIX)
        all_espn_files = list_s3_parquet_files(S3_BUCKET, ESPN_PREFIX)
        
        print(f"   Total odds files: {len(all_odds_files)}")
        print(f"   Total ESPN files: {len(all_espn_files)}")
        
        odds_files, espn_files = get_common_timestamps(
            all_odds_files, 
            all_espn_files, 
            args.recent_n_snapshots,
            args.date
        )
    else:
        odds_files = None
        espn_files = None
    
    # Download and combine odds data
    odds_path = download_and_combine_s3_parquets(S3_BUCKET, ODDS_PREFIX, 'combined_odds.parquet', odds_files)
    
    # Download and combine ESPN data
    espn_path = download_and_combine_s3_parquets(S3_BUCKET, ESPN_PREFIX, 'combined_espn.parquet', espn_files)
    
    if not odds_path or not espn_path:
        print("\n❌ Failed to download files")
        return None, None
    
    # Show live summary
    get_live_odds_summary(odds_path, espn_path)
    
    # Create plot if requested
    plot_paths = []
    if args.plot:
        print("\n" + "="*80)
        print("📊 CREATING PLOT(S)")
        print("="*80)
        
        if args.plot == 'recent':
            print("\n🎯 Mode: Plotting most recent game with data")
            away, home = find_most_recent_game_with_data(odds_path, espn_path)
            
            if away and home:
                plot_path = plot_ml_and_score_movement(odds_path, espn_path, away, home)
                if plot_path:
                    plot_paths.append(plot_path)
            else:
                print("\n❌ Could not find a game with sufficient data for plotting")
        
        elif args.plot == 'live':
            print("\n🔴 Mode: Plotting all currently live games")
            live_games = find_all_live_games(odds_path, espn_path)
            
            if live_games:
                print(f"\n📊 Generating {len(live_games)} plot(s)...")
                for i, (away, home, snapshots) in enumerate(live_games, 1):
                    print(f"\n--- Plot {i}/{len(live_games)}: {away} @ {home} ---")
                    try:
                        plot_path = plot_ml_and_score_movement(odds_path, espn_path, away, home)
                        if plot_path:
                            plot_paths.append(plot_path)
                            print(f"✅ Plot {i} complete")
                    except Exception as e:
                        print(f"❌ Error plotting {away} @ {home}: {e}")
                
                print(f"\n✅ Successfully generated {len(plot_paths)}/{len(live_games)} plot(s)")
            else:
                print("\n❌ No live games found to plot")
        
        elif args.plot == 'all':
            # Require date when using --plot all
            if not args.date:
                print("\n❌ Error: --date required when using --plot all")
                print("   Example: python tmp/read_all_live_odds_parquet.py --plot all --date 2026-02-03")
                return None, None
            
            print(f"\n🎯 Mode: Plotting all games from {args.date}")
            print(f"   Minimum snapshots: {args.min_snapshots}\n")
            
            # Find all games from the date
            all_games = find_all_games_with_data(
                odds_path, 
                espn_path, 
                args.date,
                args.min_snapshots
            )
            
            if not all_games:
                print(f"\n❌ No games found on {args.date} with >= {args.min_snapshots} snapshots")
                return None, None
            
            # Generate plots for all games
            print(f"\n📊 Generating {len(all_games)} plot(s)...\n")
            
            # Track success/failure for summary
            successful_plots = []
            failed_plots = []
            
            for i, (away, home, snapshots) in enumerate(all_games, 1):
                print(f"\n{'='*80}")
                print(f"📊 Plot {i}/{len(all_games)}: {away} @ {home} ({snapshots} snapshots)")
                print(f"{'='*80}")
                
                try:
                    plot_path = plot_ml_and_score_movement(odds_path, espn_path, away, home)
                    if plot_path:
                        plot_paths.append(plot_path)
                        successful_plots.append((away, home, snapshots, plot_path))
                        print(f"✅ Plot {i} complete: {plot_path.name}")
                except Exception as e:
                    failed_plots.append((away, home, snapshots, str(e)))
                    print(f"❌ Error plotting {away} @ {home}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Summary Statistics
            print(f"\n{'='*80}")
            print(f"📊 PLOT GENERATION SUMMARY")
            print(f"{'='*80}")
            print(f"\n📅 Date: {args.date}")
            print(f"🎯 Min snapshots threshold: {args.min_snapshots}")
            print(f"\n📈 Results:")
            print(f"   • Total games found: {len(all_games)}")
            print(f"   • Successfully plotted: {len(successful_plots)}")
            print(f"   • Failed: {len(failed_plots)}")
            
            if successful_plots:
                print(f"\n✅ Successful plots ({len(successful_plots)}):")
                for away, home, snapshots, path in successful_plots:
                    print(f"   • {away} @ {home} ({snapshots} snapshots)")
                    print(f"     → {path.name}")
            
            if failed_plots:
                print(f"\n❌ Failed plots ({len(failed_plots)}):")
                for away, home, snapshots, error in failed_plots:
                    print(f"   • {away} @ {home} ({snapshots} snapshots)")
                    print(f"     → Error: {error}")
            
            print(f"\n{'='*80}\n")
        
        # Open all plots with a single command
        if plot_paths:
            print(f"\n📂 Opening {len(plot_paths)} plot(s)...")
            open_files_batch(plot_paths)
            
            # Print copy-paste summary
            if len(plot_paths) > 1:
                print(f"\n{'='*80}")
                print("📋 COPY-PASTE SUMMARY")
                print(f"{'='*80}\n")
                print("# Single command to open all plots:")
                print(f"open {' '.join([f'\"{p}\"' for p in plot_paths])}")
                print("\n# Or copy all paths:")
                for path in plot_paths:
                    print(path)
                print(f"\n{'='*80}\n")
    
    # Load into pandas for interactive use
    print("\n" + "="*80)
    print("📊 Loading full data into memory...")
    print("="*80)
    
    con = duckdb.connect()
    try:
        odds_df = con.execute(f"SELECT * FROM '{odds_path}'").df()
        espn_df = con.execute(f"SELECT * FROM '{espn_path}'").df()
    finally:
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
    
    if plot_paths:
        print(f"\n📊 Plot file(s) generated ({len(plot_paths)}):")
        for plot_path in plot_paths:
            print(f"   - {plot_path}")
        
        # Add copy-paste friendly summary
        print("\n" + "="*80)
        print("📋 COPY-PASTE SUMMARY")
        print("="*80)
        print("\n# Single command to open all plots:")
        quoted_paths = [f'"{p}"' for p in plot_paths]
        print(f"open {' '.join(quoted_paths)}")
        print("\n# Or copy all paths:")
        print(" ".join([str(p) for p in plot_paths]))
        print("\n" + "="*80)
    
    print("\n" + "="*80 + "\n")
    
    return odds_df, espn_df


if __name__ == '__main__':
    odds_df, espn_df = main()
