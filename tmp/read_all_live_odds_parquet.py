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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from nba_team_colors import get_team_color


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
        # Convert date string to YYYYMMDD format
        date_prefix = date_filter.replace('-', '')
        common = {ts for ts in common if ts.startswith(date_prefix)}
        print(f"   📅 Filtering to date: {date_filter} (ET timezone)")
        
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
    
    # First, get the most recent snapshot timestamp for each game
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
          AND gs.num_snapshots >= 5
        ORDER BY ls.collection_timestamp DESC
    """
    
    try:
        result = con.execute(query).df()
        
        if len(result) == 0:
            print("\n⚠️ No live games found with sufficient data")
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
        # Get ML movement with scores over time
        query = f"""
            SELECT 
                o.fetched_at as timestamp,
                o.game_time as scheduled_tipoff,
                MEDIAN(o.away_ml) as away_ml,
                MEDIAN(o.home_ml) as home_ml,
                MEDIAN(o.away_ml_implied_prob) as away_implied_prob,
                MEDIAN(o.home_ml_implied_prob) as home_implied_prob,
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
    
    # Calculate score differential (away - home, so positive means away is winning)
    df_with_scores_game['score_diff'] = df_with_scores_game['away_score'] - df_with_scores_game['home_score']
    
    # Add a 0,0 point at game start time
    first_row = pd.DataFrame({
        'timestamp': [game_start_time],
        'away_ml': [df_filtered['away_ml'].iloc[0] if len(df_filtered) > 0 else None],
        'home_ml': [df_filtered['home_ml'].iloc[0] if len(df_filtered) > 0 else None],
        'away_implied_prob': [df_filtered['away_implied_prob'].iloc[0] if len(df_filtered) > 0 else None],
        'home_implied_prob': [df_filtered['home_implied_prob'].iloc[0] if len(df_filtered) > 0 else None],
        'score_diff': [0]
    })
    
    df_with_scores_full = pd.concat([first_row, df_with_scores_game[['timestamp', 'score_diff']]], ignore_index=True)
    df_ml_full = pd.concat([first_row[['timestamp', 'away_ml', 'home_ml', 'away_implied_prob', 'home_implied_prob']], 
                            df_filtered[['timestamp', 'away_ml', 'home_ml', 'away_implied_prob', 'home_implied_prob']]], ignore_index=True)
    
    # Remove duplicates (in case first_score_time already had data)
    df_with_scores_full = df_with_scores_full.drop_duplicates(subset=['timestamp'], keep='last')
    df_ml_full = df_ml_full.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Get team colors
    away_color = get_team_color(away_team, 'primary')
    home_color = get_team_color(home_team, 'primary')
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
    current_time = df['timestamp'].max()
    print(f"   🕐 Current time (latest data): {current_time}")
    
    # Get scheduled tipoff time for title
    scheduled_tipoff = df['scheduled_tipoff'].iloc[0] if 'scheduled_tipoff' in df.columns and len(df) > 0 else None
    tipoff_str = ""
    if pd.notna(scheduled_tipoff):
        tipoff_dt = pd.to_datetime(scheduled_tipoff)
        if tipoff_dt.tzinfo is None:
            tipoff_dt = tipoff_dt.tz_localize('UTC')
        tipoff_et = tipoff_dt.astimezone(ZoneInfo('America/New_York'))
        tipoff_str = f" (Tip: {tipoff_et.strftime('%I:%M %p ET')})"
    
    # Create figure with three subplots (ML, Win %, Score Differential)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{away_team} @ {home_team}{tipoff_str}\nML Odds, Win Probability, and Score Differential Over Time', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: ML Odds with conditional coloring (red for underdog/+odds, green for favorite/-odds)
    # Away team
    for i in range(len(df_ml_full)-1):
        color = 'red' if df_ml_full['away_ml'].iloc[i] > 0 else 'green'
        ax1.plot(df_ml_full['timestamp'].iloc[i:i+2], df_ml_full['away_ml'].iloc[i:i+2], 
                'o-', color=color, linewidth=2, markersize=4, alpha=0.7)
    # Add legend entry
    ax1.plot([], [], 'o-', color='red', label=f'{away_team} ML (underdog)', linewidth=2, markersize=4)
    ax1.plot([], [], 'o-', color='green', label=f'{away_team} ML (favorite)', linewidth=2, markersize=4)
    
    # Home team
    for i in range(len(df_ml_full)-1):
        color = 'red' if df_ml_full['home_ml'].iloc[i] > 0 else 'green'
        ax1.plot(df_ml_full['timestamp'].iloc[i:i+2], df_ml_full['home_ml'].iloc[i:i+2], 
                's-', color=color, linewidth=2, markersize=4, alpha=0.7)
    # Add legend entry
    ax1.plot([], [], 's-', color='red', label=f'{home_team} ML (underdog)', linewidth=2, markersize=4)
    ax1.plot([], [], 's-', color='green', label=f'{home_team} ML (favorite)', linewidth=2, markersize=4)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Moneyline Odds', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Moneyline Movement (Red=Underdog, Green=Favorite)', fontsize=14)
    
    # Add favorite/underdog regions
    ax1.axhspan(-1000, 0, alpha=0.05, color='green')
    ax1.axhspan(0, 1000, alpha=0.05, color='red')
    
    # Plot 2: Implied Win Probability with shading for who's ahead
    ax2.plot(df_ml_full['timestamp'], df_ml_full['away_implied_prob'] * 100, 
             'o-', label=f'{away_team} Win %', color=away_color, linewidth=2, markersize=4)
    ax2.plot(df_ml_full['timestamp'], df_ml_full['home_implied_prob'] * 100, 
             's-', label=f'{home_team} Win %', color=home_color, linewidth=2, markersize=4)
    
    # Add shading for which team has higher win probability
    ax2.fill_between(df_ml_full['timestamp'], 
                     df_ml_full['away_implied_prob'] * 100, 
                     50,
                     where=(df_ml_full['away_implied_prob'] * 100 >= 50),
                     alpha=0.3, color=away_color, label=f'{away_team} Favored')
    ax2.fill_between(df_ml_full['timestamp'], 
                     df_ml_full['home_implied_prob'] * 100, 
                     50,
                     where=(df_ml_full['home_implied_prob'] * 100 >= 50),
                     alpha=0.3, color=home_color, label=f'{home_team} Favored')
    
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel('Implied Win Probability (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Win Probability Movement', fontsize=14)
    
    # Plot 3: Score Differential (away - home)
    ax3.plot(df_with_scores_full['timestamp'], df_with_scores_full['score_diff'], 
             'o-', linewidth=2, markersize=6, color='#2ca02c')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.fill_between(df_with_scores_full['timestamp'], 
                     df_with_scores_full['score_diff'], 
                     0,
                     where=(df_with_scores_full['score_diff'] >= 0),
                     alpha=0.3, color=away_color, label=f'{away_team} Leading')
    ax3.fill_between(df_with_scores_full['timestamp'], 
                     df_with_scores_full['score_diff'], 
                     0,
                     where=(df_with_scores_full['score_diff'] < 0),
                     alpha=0.3, color=home_color, label=f'{home_team} Leading')
    
    ax3.set_ylabel('Score Differential\n(Away - Home)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Score Differential (starts at 0)', fontsize=14)
    
    # Add reference lines to all three plots
    print(f"\n🎨 Adding reference lines...")
    
    # 1. Game start (dotted red)
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=game_start_time, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Game Start')
    print(f"   🟥 Game start line at {game_start_time}")
    
    # 2. Current time (dotted red)
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=current_time, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Current Time')
    print(f"   🟥 Current time line at {current_time}")
    
    # 3. Quarter endings (dashed gray lines)
    for period, end_time in quarter_end_times:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=end_time, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.text(end_time, ax.get_ylim()[1] * 0.95, f'Q{int(period)} End', 
                   rotation=90, verticalalignment='top', fontsize=9, alpha=0.7)
        print(f"   ⬜ Q{int(period)} end line at {end_time}")
    
    # 4. Quarter starts (dashed blue lines)
    for period, start_time in quarter_start_times:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=start_time, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.text(start_time, ax.get_ylim()[0] * 0.95, f'Q{int(period)} Start', 
                   rotation=90, verticalalignment='bottom', fontsize=9, alpha=0.7, color='blue')
        print(f"   🟦 Q{int(period)} start line at {start_time}")
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
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
        choices=['recent', 'live'],
        default=None,
        help='Create plots: "recent" = most recent game with data, "live" = all currently live games'
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
        
        # Open all plots with a single command
        if plot_paths:
            print(f"\n📂 Opening {len(plot_paths)} plot(s)...")
            open_files_batch(plot_paths)
    
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
