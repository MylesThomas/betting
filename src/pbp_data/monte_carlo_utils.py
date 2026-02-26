"""
Monte Carlo Utilities for NBA Player Props

Shared functions for Monte Carlo simulation, player profiling,
and visualization across multiple scripts.

Usage:
    from src.pbp_data.monte_carlo_utils import (
        load_player_profile,
        monte_carlo_simulate_bet,
        create_ggplot
    )
"""

import duckdb
import pandas as pd
import numpy as np
import random
import json
import subprocess
import tempfile
import os
import math
import requests
import urllib3
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disk-based image cache directory (persists across script runs)
_IMAGE_CACHE_DIR = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation" / "image_cache"
_IMAGE_CACHE_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# PATHS - Use functions to get paths relative to caller
# =============================================================================

def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_paths():
    """Get standard data paths."""
    project_root = get_project_root()
    return {
        'data_dir': project_root / "data",
        'minute_by_minute': project_root / "data" / "minute_by_minute.parquet",
        'pbp_data_dir': Path.home() / "Downloads" / "tmp" / "player_points_monte_carlo" / "pbp_data",
        'player_props_dir': Path.home() / "Downloads" / "tmp" / "player_props_raw" / "2025-26",
    }


# =============================================================================
# TEAM LOGOS (ESPN CDN)
# =============================================================================

TEAM_LOGOS = {
    "Atlanta Hawks": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
    "Boston Celtics": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
    "Brooklyn Nets": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
    "Charlotte Hornets": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
    "Chicago Bulls": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
    "Cleveland Cavaliers": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
    "Dallas Mavericks": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
    "Denver Nuggets": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
    "Detroit Pistons": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
    "Golden State Warriors": "https://a.espncdn.com/i/teamlogos/nba/500/gsw.png",
    "Houston Rockets": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
    "Indiana Pacers": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
    "LA Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
    "Los Angeles Lakers": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
    "Memphis Grizzlies": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
    "Miami Heat": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
    "Milwaukee Bucks": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
    "Minnesota Timberwolves": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
    "New Orleans Pelicans": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
    "New York Knicks": "https://a.espncdn.com/i/teamlogos/nba/500/ny.png",
    "Oklahoma City Thunder": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
    "Orlando Magic": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
    "Philadelphia 76ers": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
    "Phoenix Suns": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
    "Portland Trail Blazers": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
    "Sacramento Kings": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
    "San Antonio Spurs": "https://a.espncdn.com/i/teamlogos/nba/500/sa.png",
    "Toronto Raptors": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
    "Utah Jazz": "https://a.espncdn.com/i/teamlogos/nba/500/utah.png",
    "Washington Wizards": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png",
}


# =============================================================================
# CALIBRATION MAP (v9 empirical bias corrections)
# =============================================================================
# Hardcoded bias correction factors from v9 validation results
# Format: (period, bin_center) -> correction_factor
# correction_factor = avg_predicted_prob - actual_hit_rate
# To calibrate: calibrated_prob = raw_prob - correction_factor
#
# Generated from v9 validation (659,249 predictions across 710 games):
# - Query: See duckdb query in load_v9_bias_correction_table()
# - Verified: Each value matches v9/predictions.parquet within 0.001 tolerance
# - Version: v10 introduces this empirical calibration

# =============================================================================
# CONSERVATIVE BIAS (v12)
# =============================================================================
# Apply a final conservative multiplier to all probabilities
# Rationale: Err on the side of underpredicting (lower confidence)
# This reduces risk of overconfidence while maintaining relative rankings
CONSERVATIVE_FACTOR = 0.97  # Reduces all probabilities by 3%

CALIBRATION_MAP_V9 = {
    # Q1 - v9 data
    ('Q1', 0.05): 0.08,
    ('Q1', 0.15): 0.07,
    ('Q1', 0.25): 0.08,
    ('Q1', 0.35): 0.07,
    ('Q1', 0.45): 0.11,
    ('Q1', 0.55): 0.10,
    ('Q1', 0.65): 0.09,
    ('Q1', 0.75): 0.12,
    ('Q1', 0.85): 0.11,
    ('Q1', 0.95): 0.0,  # No data (n<10 in v9)
    
    # Q2 - v9 data
    ('Q2', 0.05): 0.03,
    ('Q2', 0.15): 0.02,
    ('Q2', 0.25): 0.0,
    ('Q2', 0.35): -0.01,
    ('Q2', 0.45): 0.0,
    ('Q2', 0.55): 0.03,
    ('Q2', 0.65): 0.10,
    ('Q2', 0.75): 0.13,
    ('Q2', 0.85): 0.08,
    ('Q2', 0.95): 0.02,
    
    # Q3 - v9 data
    ('Q3', 0.05): 0.02,
    ('Q3', 0.15): 0.04,
    ('Q3', 0.25): 0.06,
    ('Q3', 0.35): 0.09,
    ('Q3', 0.45): 0.07,
    ('Q3', 0.55): 0.04,
    ('Q3', 0.65): 0.05,
    ('Q3', 0.75): 0.04,
    ('Q3', 0.85): 0.09,
    ('Q3', 0.95): 0.06,
    
    # Q4 - v9 data
    ('Q4', 0.05): -0.01,
    ('Q4', 0.15): -0.01,
    ('Q4', 0.25): -0.02,
    ('Q4', 0.35): -0.03,
    ('Q4', 0.45): -0.05,
    ('Q4', 0.55): -0.04,
    ('Q4', 0.65): 0.07,
    ('Q4', 0.75): 0.10,
    ('Q4', 0.85): 0.20,
    ('Q4', 0.95): 0.51,
    
    # OT1 - v9 data
    ('OT1', 0.05): -0.03,
    ('OT1', 0.15): -0.35,
    ('OT1', 0.25): -0.32,
    ('OT1', 0.35): -0.20,
    ('OT1', 0.45): 0.31,
    ('OT1', 0.55): 0.08,
    ('OT1', 0.65): 0.36,
    ('OT1', 0.75): 0.77,
    ('OT1', 0.85): 0.72,
    ('OT1', 0.95): 0.67,
    
    # OT2 - sparse data, fill with 0.0 where no data
    ('OT2', 0.05): 0.01,   # v9 data (n=289)
    ('OT2', 0.15): 0.0,    # No data
    ('OT2', 0.25): 0.0,    # No data
    ('OT2', 0.35): 0.0,    # No data
    ('OT2', 0.45): 0.0,    # No data
    ('OT2', 0.55): 0.0,    # No data
    ('OT2', 0.65): 0.0,    # No data
    ('OT2', 0.75): 0.0,    # No data
    ('OT2', 0.85): -0.15,  # v9 data (n=12)
    ('OT2', 0.95): -0.07,  # v9 data (n=47)
    
    # OT3 - no data, use 0.0 (no correction)
    ('OT3', 0.05): 0.0,
    ('OT3', 0.15): 0.0,
    ('OT3', 0.25): 0.0,
    ('OT3', 0.35): 0.0,
    ('OT3', 0.45): 0.0,
    ('OT3', 0.55): 0.0,
    ('OT3', 0.65): 0.0,
    ('OT3', 0.75): 0.0,
    ('OT3', 0.85): 0.0,
    ('OT3', 0.95): 0.0,
    
    # OT4+ - no data, use 0.0 (no correction)
    ('OT4+', 0.05): 0.0,
    ('OT4+', 0.15): 0.0,
    ('OT4+', 0.25): 0.0,
    ('OT4+', 0.35): 0.0,
    ('OT4+', 0.45): 0.0,
    ('OT4+', 0.55): 0.0,
    ('OT4+', 0.65): 0.0,
    ('OT4+', 0.75): 0.0,
    ('OT4+', 0.85): 0.0,
    ('OT4+', 0.95): 0.0,
}


# =============================================================================
# IMAGE DOWNLOADING
# =============================================================================

def download_team_logo(team_name, size=(100, 100)):
    """Download team logo from ESPN CDN and return cached file path. Uses disk cache to persist across runs."""
    # Check disk cache first
    cache_filename = f"team_{team_name}_{size[0]}x{size[1]}.png"
    cache_path = _IMAGE_CACHE_DIR / cache_filename
    
    if cache_path.exists():
        return str(cache_path)
    
    if team_name not in TEAM_LOGOS:
        print(f"⚠️  Team '{team_name}' not found in TEAM_LOGOS dict")
        return create_blank_image(size)
    
    url = TEAM_LOGOS[team_name]
    
    try:
        response = requests.get(url, timeout=5, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(cache_path, "PNG")
        
        return str(cache_path)
    except Exception as e:
        print(f"⚠️  Failed to download logo for {team_name}: {e}")
        return create_blank_image(size)


def download_player_headshot(player_id, size=(120, 120)):
    """Download player headshot from ESPN/NBA CDN and return cached file path. Uses disk cache to persist across runs."""
    # Check disk cache first
    cache_filename = f"player_{player_id}_{size[0]}x{size[1]}.png"
    cache_path = _IMAGE_CACHE_DIR / cache_filename
    
    if cache_path.exists():
        print(f"   📦 Using cached headshot (player_id: {player_id})")
        return str(cache_path)
    
    # Try ESPN CDN first (has better quality actual photos)
    url_espn = f"https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{player_id}.png"
    try:
        response = requests.get(url_espn, timeout=5, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        # Check if it's a placeholder (very small file size)
        if len(response.content) > 50000:  # Real photos are > 50KB
            img = img.convert("RGBA")
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(str(cache_path), "PNG")
            
            print(f"   ✅ Downloaded headshot from ESPN CDN (player_id: {player_id})")
            print(f"      💾 Cached to: {cache_path}")
            return str(cache_path)
        else:
            raise Exception("ESPN returned placeholder image")
            
    except Exception as e1:
        # Try NBA CDN as fallback
        url_nba = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        try:
            response = requests.get(url_nba, timeout=5, verify=False)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            
            # Check if it's a placeholder
            if len(response.content) > 50000:  # Real photos are > 50KB
                img = img.convert("RGBA")
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(str(cache_path), "PNG")
                
                print(f"   ✅ Downloaded headshot from NBA CDN (player_id: {player_id})")
                print(f"      💾 Cached to: {cache_path}")
                return str(cache_path)
            else:
                raise Exception("NBA CDN returned placeholder image")
                
        except Exception as e2:
            print(f"   ⚠️  Failed to download headshot for player_id {player_id}")
            print(f"      ESPN CDN error: {e1}")
            print(f"      NBA CDN error: {e2}")
            return create_blank_image(size)


def create_blank_image(size=(100, 100)):
    """Create a blank transparent PNG."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    img.save(temp_file.name, "PNG")
    return temp_file.name


# =============================================================================
# LOAD PLAYER PROFILE
# =============================================================================

# Table name used when preloading minute_by_minute into DuckDB for fast per-player SELECT.
MBM_TABLE_NAME = "mbm"


def get_minute_by_minute_with_normalized_names(minute_by_minute_path=None):
    """
    Load minute_by_minute parquet with DuckDB, add normalized_name, register in DuckDB.
    Returns a DuckDB connection with table 'mbm' (normalized_name included).
    Call once per iteration; pass the returned con to load_player_profile(..., preloaded_duckdb_con=con).
    Caller must close() the connection when done.
    """
    if minute_by_minute_path is None:
        minute_by_minute_path = get_data_paths()['minute_by_minute']
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from player_team_history.name_normalization import normalize_from_nba_api
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{minute_by_minute_path}')").df()
    df['normalized_name'] = df['player_name'].apply(normalize_from_nba_api)
    con.register(MBM_TABLE_NAME, df)
    return con


def load_player_profile(player_name, minute_by_minute_path=None, preloaded_duckdb_con=None):
    """
    Load player's historical data (quarterly distributions) using DuckDB + pandas.
    
    Strategy:
    - If preloaded_duckdb_con: run DuckDB SELECT ... WHERE normalized_name = ? (fast).
    - Else: read parquet with DuckDB, add normalized_name, filter to player, then aggregate.
    
    Args:
        player_name: Normalized player name (e.g., "Lebron James", "Pj Washington")
        minute_by_minute_path: Optional path to minute_by_minute.parquet (ignored if preloaded_duckdb_con set)
        preloaded_duckdb_con: Optional DuckDB connection with table 'mbm' (has normalized_name); use SELECT WHERE.
    
    Returns:
        dict with quarterly distributions (lists) and player_id
    """
    if preloaded_duckdb_con is not None:
        player_df = preloaded_duckdb_con.execute(
            f"SELECT * FROM {MBM_TABLE_NAME} WHERE normalized_name = ?", [player_name]
        ).df()
    else:
        if minute_by_minute_path is None:
            minute_by_minute_path = get_data_paths()['minute_by_minute']
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from player_team_history.name_normalization import normalize_from_nba_api
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM read_parquet('{minute_by_minute_path}')").df()
        con.close()
        df['normalized_name'] = df['player_name'].apply(normalize_from_nba_api)
        player_df = df[df['normalized_name'] == player_name].copy()
    
    if len(player_df) == 0:
        raise ValueError(f"Player {player_name} not found")
    
    # STEP 4: Calculate game-level stats
    game_stats = player_df.groupby(['game_id', 'game_date', 'player_id', 'player_name']).agg({
        'playing_seconds': 'max',
        'cumulative_points': 'max'
    }).reset_index()
    
    game_stats['total_minutes'] = game_stats['playing_seconds'] / 60.0
    game_stats['total_points'] = game_stats['cumulative_points']
    game_stats['points_per_minute'] = game_stats.apply(
        lambda row: row['total_points'] / row['total_minutes'] if row['total_minutes'] > 0 else 0,
        axis=1
    )
    
    # STEP 5: Calculate quarterly splits for each game
    quarter_data = []
    for game_id in player_df['game_id'].unique():
        game_data = player_df[player_df['game_id'] == game_id].copy()
        game_date = game_data['game_date'].iloc[0]
        player_id = game_data['player_id'].iloc[0]
        original_name = game_data['player_name'].iloc[0]
        
        # Get cumulative values at end of each quarter
        q1_data = game_data[game_data['minute'] <= 11]
        q2_data = game_data[game_data['minute'] <= 23]
        q3_data = game_data[game_data['minute'] <= 35]
        q4_data = game_data[game_data['minute'] <= 47]
        
        # Q1 stats (0-11 minutes)
        q1_seconds = q1_data['playing_seconds'].max() if len(q1_data) > 0 else 0
        q1_points = q1_data['cumulative_points'].max() if len(q1_data) > 0 else 0
        q1_minutes = q1_seconds / 60.0
        
        # Q2 stats (12-23 minutes) - incremental
        q2_end_seconds = q2_data['playing_seconds'].max() if len(q2_data) > 0 else 0
        q2_end_points = q2_data['cumulative_points'].max() if len(q2_data) > 0 else 0
        q2_minutes = (q2_end_seconds - q1_seconds) / 60.0
        q2_points = q2_end_points - q1_points
        
        # Q3 stats (24-35 minutes) - incremental
        q3_end_seconds = q3_data['playing_seconds'].max() if len(q3_data) > 0 else 0
        q3_end_points = q3_data['cumulative_points'].max() if len(q3_data) > 0 else 0
        q3_minutes = (q3_end_seconds - q2_end_seconds) / 60.0
        q3_points = q3_end_points - q2_end_points
        
        # Q4 stats (36-47 minutes) - incremental
        q4_end_seconds = q4_data['playing_seconds'].max() if len(q4_data) > 0 else 0
        q4_end_points = q4_data['cumulative_points'].max() if len(q4_data) > 0 else 0
        q4_minutes = (q4_end_seconds - q3_end_seconds) / 60.0
        q4_points = q4_end_points - q3_end_points
        
        # Calculate PPM for each quarter
        q1_ppm = q1_points / q1_minutes if q1_minutes > 0 else 0
        q2_ppm = q2_points / q2_minutes if q2_minutes > 0 else 0
        q3_ppm = q3_points / q3_minutes if q3_minutes > 0 else 0
        q4_ppm = q4_points / q4_minutes if q4_minutes > 0 else 0
        
        quarter_data.append({
            'game_id': game_id,
            'game_date': game_date,
            'player_id': player_id,
            'player_name': original_name,
            'q1_minutes': q1_minutes,
            'q2_minutes': q2_minutes,
            'q3_minutes': q3_minutes,
            'q4_minutes': q4_minutes,
            'q1_ppm': q1_ppm,
            'q2_ppm': q2_ppm,
            'q3_ppm': q3_ppm,
            'q4_ppm': q4_ppm,
        })
    
    quarter_df = pd.DataFrame(quarter_data)
    
    # STEP 6: Merge game stats with quarter stats
    full_df = game_stats.merge(quarter_df, on=['game_id', 'game_date', 'player_id', 'player_name'])
    
    # Sort by game_date descending (most recent first)
    full_df = full_df.sort_values('game_date', ascending=False)
    
    # STEP 7: Build profile dict (same format as before)
    profile = {
        'player_id': full_df['player_id'].iloc[0],
        'player_name': full_df['player_name'].iloc[0],  # Original unnormalized name
        'num_games': len(full_df),
        'avg_points_per_game': full_df['total_points'].mean(),
        'avg_minutes_per_game': full_df['total_minutes'].mean(),
        'q1_minutes_history': full_df['q1_minutes'].tolist(),
        'q2_minutes_history': full_df['q2_minutes'].tolist(),
        'q3_minutes_history': full_df['q3_minutes'].tolist(),
        'q4_minutes_history': full_df['q4_minutes'].tolist(),
        'q4_points_per_minute_history': full_df['q4_ppm'].tolist(),
        'points_per_minute_history': full_df['points_per_minute'].tolist(),
    }
    
    return profile


# =============================================================================
# LOAD PROP LINES
# =============================================================================

def get_consensus_prop_line(player_name, pregame_props_lookup=None, **kwargs):
    """
    Get consensus (median) prop line from preloaded lookup dict.
    
    Args:
        player_name: Player name (e.g., "Luka Doncic")
        pregame_props_lookup: Dict mapping normalized_player_name -> pregame_line
        **kwargs: Ignored (for backwards compatibility with old signature)
    
    Returns:
        float: Consensus prop line (median), or None if not found
    """
    if pregame_props_lookup is None:
        return None
    
    # Normalize player name to match the lookup dict keys
    from player_team_history.name_normalization import normalize_from_odds_api
    normalized_name = normalize_from_odds_api(player_name)
    
    return pregame_props_lookup.get(normalized_name)


# =============================================================================
# LOAD PLAY-BY-PLAY DATA
# =============================================================================

def load_play_by_play(game_id, player_name):
    """
    Load play-by-play data from cached JSON and extract relevant plays.
    
    Args:
        game_id: ESPN game ID
        player_name: Player name to track
    
    Returns:
        DataFrame with columns: [play_id, quarter, game_minute, description, 
                                 away_score, home_score, cumulative_points]
        game_metadata: dict with away_team, home_team, game_date, commence_time_et
    """
    paths = get_data_paths()
    pbp_data_dir = paths['pbp_data_dir']
    
    # Find the JSON file
    json_files = list(pbp_data_dir.glob(f"*_{game_id}.json"))
    
    if not json_files:
        raise FileNotFoundError(f"PBP data not found for game_id {game_id}")
    
    json_file = json_files[0]
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    boxscore = data['boxscore']
    teams = boxscore['teams']
    away_team = teams[0]['team']['displayName']
    home_team = teams[1]['team']['displayName']
    
    # Extract game date from header
    header = data.get('header', {})
    game_date_str = header.get('competitions', [{}])[0].get('date', '')
    
    if game_date_str:
        # Parse UTC time and convert to ET
        game_dt_utc = pd.to_datetime(game_date_str)
        game_dt_et = game_dt_utc.tz_convert("America/New_York")
        game_date = game_dt_et.strftime("%Y-%m-%d")
        commence_time_et = game_dt_et
    else:
        # Fallback: parse from filename
        date_from_file = json_file.stem.split('_')[0]
        game_date = f"{date_from_file[:4]}-{date_from_file[4:6]}-{date_from_file[6:8]}"
        commence_time_et = None
    
    game_metadata = {
        'away_team': away_team,
        'home_team': home_team,
        'game_date': game_date,
        'commence_time_et': commence_time_et,
    }
    
    # Parse plays - TWO PASS: first extract, then sort, then calculate cumulative
    plays = data['plays']
    play_data = []
    
    # PASS 1: Extract play info and check if player scored
    for play_idx, play in enumerate(plays):
        play_id = play.get('id')
        quarter = play.get('period', {}).get('number', 1)
        
        # Calculate game minute from clock
        clock_display = play.get('clock', {}).get('displayValue', '12:00')
        try:
            # Handle both "MM:SS" string format and numeric values
            if isinstance(clock_display, (int, float)):
                time_left_in_quarter = float(clock_display) / 60.0
            elif ':' in str(clock_display):
                mins, secs = map(int, str(clock_display).split(':'))
                time_left_in_quarter = mins + secs / 60.0
            else:
                # Fallback: try to convert to float
                time_left_in_quarter = float(clock_display) / 60.0
            
            # Game minute = start of quarter + time elapsed in quarter
            quarter_start = (quarter - 1) * 12
            game_minute = quarter_start + (12 - time_left_in_quarter)
        except Exception:
            # Fallback if clock parsing fails
            game_minute = (quarter - 1) * 12
        
        description = play.get('text', '')
        away_score = play.get('awayScore', 0)
        home_score = play.get('homeScore', 0)
        
        # Check if player scored on THIS play (store points, don't accumulate yet)
        points_this_play = 0
        if player_name in description:
            if 'makes' in description.lower() or 'free throw' in description.lower():
                if '3-pt' in description.lower() or 'three point' in description.lower():
                    points_this_play = 3
                elif '2-pt' in description.lower() or 'two point' in description.lower():
                    points_this_play = 2
                elif 'free throw' in description.lower() and 'makes' in description.lower():
                    points_this_play = 1
        
        play_data.append({
            'play_id': play_id,
            'espn_index': play_idx,  # Store original ESPN order
            'quarter': quarter,
            'game_minute': game_minute,
            'description': description,
            'away_score': away_score,
            'home_score': home_score,
            'points_this_play': points_this_play,
        })
    
    # PASS 2: Sort by game_minute ONLY (ignore ESPN order)
    df = pd.DataFrame(play_data)
    df = df.sort_values('game_minute').reset_index(drop=True)
    
    # PASS 3: Calculate cumulative points in correct chronological order
    df['cumulative_points'] = df['points_this_play'].cumsum()
    
    # Clean up temporary columns (keep description for debugging)
    df = df.drop(columns=['espn_index', 'points_this_play'])
    
    return df, game_metadata


# =============================================================================
# MONTE CARLO SIMULATION WITH VEGAS ADJUSTMENT
# =============================================================================

def get_game_state(current_minute):
    """
    Determine game state including OT detection.
    
    ESPN's Play-by-Play Minute System (with 7-minute gaps between periods):
    - Regulation Q1-Q4: 0-48
    - OT1 (Q5): 55-60
    - OT2 (Q6): 67-72
    - OT3 (Q7): 79-84
    
    Args:
        current_minute: Current game minute
    
    Returns:
        dict with: quarter, time_remaining, is_ot, ot_period
    """
    if current_minute < 12:
        return {
            'quarter': 1,
            'time_remaining': 12 - current_minute,
            'is_ot': False,
            'ot_period': 0
        }
    elif current_minute < 24:
        return {
            'quarter': 2,
            'time_remaining': 24 - current_minute,
            'is_ot': False,
            'ot_period': 0
        }
    elif current_minute < 36:
        return {
            'quarter': 3,
            'time_remaining': 36 - current_minute,
            'is_ot': False,
            'ot_period': 0
        }
    elif current_minute < 48:
        return {
            'quarter': 4,
            'time_remaining': 48 - current_minute,
            'is_ot': False,
            'ot_period': 0
        }
    elif current_minute >= 55 and current_minute < 60:
        # OT1 (Quarter 5): 55-60
        return {
            'quarter': 5,
            'time_remaining': 60 - current_minute,
            'is_ot': True,
            'ot_period': 1
        }
    elif current_minute >= 67 and current_minute < 72:
        # OT2 (Quarter 6): 67-72
        return {
            'quarter': 6,
            'time_remaining': 72 - current_minute,
            'is_ot': True,
            'ot_period': 2
        }
    elif current_minute >= 79 and current_minute < 84:
        # OT3 (Quarter 7): 79-84 (capped at 3OT as requested)
        return {
            'quarter': 7,
            'time_remaining': 84 - current_minute,
            'is_ot': True,
            'ot_period': 3
        }
    else:
        # Gap between periods or past 3OT - treat as end of previous period
        if current_minute >= 84:
            # Past 3OT
            return {
                'quarter': 7,
                'time_remaining': 0,
                'is_ot': True,
                'ot_period': 3
            }
        elif current_minute >= 72:
            # Gap after OT2
            return {
                'quarter': 6,
                'time_remaining': 0,
                'is_ot': True,
                'ot_period': 2
            }
        elif current_minute >= 60:
            # Gap after OT1
            return {
                'quarter': 5,
                'time_remaining': 0,
                'is_ot': True,
                'ot_period': 1
            }
        else:
            # Gap after regulation (48-55)
            return {
                'quarter': 4,
                'time_remaining': 0,
                'is_ot': False,
                'ot_period': 0
            }


def estimate_ot_probability(current_minute, ot_period=0, score_differential=None):
    """
    Estimate probability of next OT period.
    
    Uses ESPN's minute system:
    - OT1 (Q5): 55-60
    - OT2 (Q6): 67-72
    - OT3 (Q7): 79-84
    
    Args:
        current_minute: Current game minute
        ot_period: Current OT period (0 = regulation, 1 = OT1, etc.)
        score_differential: Point differential (optional, improves estimate)
    
    Returns:
        Probability of next OT period (0-1)
    """
    # Determine time left in current period and base rate
    if ot_period == 0:
        # End of Q4 (regulation)
        if current_minute < 47:
            return 0.0  # Too early
        time_left = 48 - current_minute
        base_ot_rate = 0.06  # NBA average ~6%
    elif ot_period == 1:
        # In OT1, check for OT2 (minute 55-60)
        if current_minute < 59:
            return 0.0  # Too early in OT1
        time_left = 60 - current_minute
        base_ot_rate = 0.20  # ~20% of OT games go to 2OT
    elif ot_period == 2:
        # In OT2, check for OT3 (minute 67-72)
        if current_minute < 71:
            return 0.0
        time_left = 72 - current_minute
        base_ot_rate = 0.15  # ~15% go to 3OT
    else:
        # Cap at 3OT as requested
        return 0.0
    
    # Adjust based on time remaining
    if time_left > 2.0:
        return 0.0
    elif time_left > 1.0:
        time_multiplier = 0.5  # 1-2 min left
    elif time_left > 0.5:
        time_multiplier = 1.0  # 30sec-1min left
    else:
        time_multiplier = 1.5  # Final 30 seconds
    
    # Adjust based on score differential if available
    if score_differential is not None:
        abs_diff = abs(score_differential)
        if abs_diff <= 3:
            score_multiplier = 4.0  # One possession game
        elif abs_diff <= 5:
            score_multiplier = 2.5  # Two possession game
        elif abs_diff <= 8:
            score_multiplier = 1.5  # Three possession game
        else:
            score_multiplier = 0.2  # Unlikely
    else:
        # Conservative assumption of moderately close game
        score_multiplier = 2.0
    
    final_prob = base_ot_rate * time_multiplier * score_multiplier
    
    # Cap at reasonable maximum
    return min(final_prob, 0.40)


def project_ot_points(player_profile, vegas_adjustment=1.0, proportion=1.0):
    """
    Project points in OT period using Q4 stats.
    
    Rationale (v9 update):
    - Games that go to OT are close → key players don't sit
    - Filter Q4 history to exclude blowout-sits (minutes > 3)
    - Use Q4 minutes/PPM for more conservative OT projection
    - Q4 better represents late-game/clutch scenarios
    - Q4 is 12 minutes, OT is 5 minutes (ratio: 5/12 ≈ 0.417)
    
    Args:
        player_profile: Player's historical stats
        vegas_adjustment: PPM multiplier
        proportion: Proportion of OT period to project (default 1.0 = full 5 min)
    
    Returns:
        Projected points for this OT portion
    """
    # v9: Filter Q4 history to exclude blowout-sits
    # Games that go to OT are close → player actually plays
    q4_minutes_history = player_profile['q4_minutes_history']
    q4_ppm_history = player_profile['q4_points_per_minute_history']
    
    if not q4_minutes_history or not q4_ppm_history:
        return 0  # No Q4 data at all
    
    # Filter: only use Q4 games where player actually played (> 3 minutes)
    # Excludes: DNPs, garbage time sits, blowout benching
    filtered_minutes = [m for m in q4_minutes_history if m > 3]
    filtered_ppm = [ppm for m, ppm in zip(q4_minutes_history, q4_ppm_history) if m > 3]
    
    if not filtered_minutes:
        return 0  # No valid Q4 games with significant playing time
    
    # Sample from filtered distributions (OT games = close games = player plays)
    typical_q4_minutes = random.choice(filtered_minutes)
    ot_ppm = random.choice(filtered_ppm) * vegas_adjustment
    
    # Scale to OT length: 5 min vs 12 min Q4
    ot_length = 5.0
    q4_length = 12.0
    scale_factor = (ot_length / q4_length) * proportion
    
    projected_ot_minutes = typical_q4_minutes * scale_factor
    
    # Calculate OT points
    ot_points = ot_ppm * projected_ot_minutes
    
    return ot_points


def load_v9_bias_correction_table():
    """
    Load v9 bias correction table with optional verification.
    
    Returns hardcoded CALIBRATION_MAP_V9 and optionally verifies against
    v9/predictions.parquet if it exists.
    
    Returns:
        dict: (period, bin_center) -> correction_factor
    
    Raises:
        AssertionError: If v9 data exists and doesn't match hardcoded values
    """
    # Always return hardcoded map
    calibration_map = CALIBRATION_MAP_V9.copy()
    
    # Optional verification if v9 predictions exist
    v9_predictions_path = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation" / "versions" / "v9" / "predictions.parquet"
    
    if v9_predictions_path.exists():
        try:
            # Query v9 data to verify hardcoded values
            query = """
            WITH bucketed_predictions AS (
                SELECT 
                    CASE 
                        WHEN quarter <= 4 THEN 'Q' || quarter::VARCHAR
                        WHEN quarter = 5 THEN 'OT1'
                        WHEN quarter = 6 THEN 'OT2'
                        WHEN quarter = 7 THEN 'OT3'
                        ELSE 'OT4+'
                    END as period,
                    CASE WHEN result = 'HIT' THEN 1 ELSE 0 END as actual_outcome,
                    prob_over,
                    FLOOR(prob_over * 10) / 10.0 as bin_start
                FROM read_parquet(?)
                WHERE prob_over < 1.0
            ),
            calibration_stats AS (
                SELECT 
                    period,
                    ROUND(bin_start + 0.05, 2) as bin_center,
                    COUNT(*) as n_predictions,
                    ROUND(AVG(prob_over) - AVG(actual_outcome), 2) as bias
                FROM bucketed_predictions
                GROUP BY period, bin_center
            )
            SELECT period, bin_center, bias, n_predictions
            FROM calibration_stats
            WHERE n_predictions >= 10
            ORDER BY period, bin_center
            """
            
            conn = duckdb.connect()
            result = conn.execute(query, [str(v9_predictions_path)]).fetchall()
            conn.close()
            
            # Verify each value matches (within tolerance)
            tolerance = 0.001
            for period, bin_center, bias, n_preds in result:
                key = (period, bin_center)
                if key in calibration_map:
                    expected = calibration_map[key]
                    diff = abs(expected - bias)
                    assert diff <= tolerance, (
                        f"Calibration mismatch for {key}: "
                        f"hardcoded={expected:.3f}, v9_data={bias:.3f}, "
                        f"diff={diff:.4f} (n={n_preds})"
                    )
            
            print("✅ Calibration table verified against v9 data")
        
        except Exception as e:
            print(f"⚠️ Could not verify calibration table: {e}")
            print("   Using hardcoded values (this is fine if v9 data unavailable)")
    
    return calibration_map


def apply_calibration(raw_prob, quarter, calibration_map=None):
    """
    Apply period-specific calibration to raw Monte Carlo probability.
    
    v10 feature: Empirical calibration learned from v9 validation results.
    Corrects systematic overconfidence/underconfidence by period and probability bin.
    
    Args:
        raw_prob: Uncalibrated probability from Monte Carlo (0.0 to 1.0)
        quarter: Current quarter (1-4 for regulation, 5+ for OT)
        calibration_map: Optional override (defaults to CALIBRATION_MAP_V9)
    
    Returns:
        Calibrated probability (0.0 to 1.0, clamped)
    """
    if calibration_map is None:
        calibration_map = CALIBRATION_MAP_V9
    
    # Map quarter to period
    if quarter <= 4:
        period = f'Q{quarter}'
    elif quarter == 5:
        period = 'OT1'
    elif quarter == 6:
        period = 'OT2'
    elif quarter == 7:
        period = 'OT3'
    else:
        period = 'OT4+'
    
    # Find nearest bin center (0.05, 0.15, 0.25, ..., 0.95)
    # Bins are [0-0.1) -> 0.05, [0.1-0.2) -> 0.15, etc.
    # raw_prob in [0.95, 1.0] yields bin_center 1.05 which is not in the map; clamp to 0.95.
    bin_start = math.floor(raw_prob * 10) / 10.0
    bin_center = round(bin_start + 0.05, 2)
    if bin_center > 0.95:
        bin_center = 0.95

    # Lookup correction factor (no fallback - should always exist)
    key = (period, bin_center)
    correction_factor = calibration_map[key]  # Will raise KeyError if missing
    
    # Apply correction: calibrated = raw - bias
    # (If we predict 0.85 but actually hit 0.75, bias=+0.10, so we subtract 0.10)
    calibrated_prob = raw_prob - correction_factor
    
    # Clamp to [0, 1]
    calibrated_prob = max(0.0, min(1.0, calibrated_prob))
    
    return calibrated_prob


def monte_carlo_simulate_bet(
    player_profile,
    current_minute,
    current_points,
    prop_line,
    n_simulations=10000,
    vegas_adjustment=1.0,
    score_differential=None,
    debug=False
):
    """
    Run Monte Carlo simulation for remaining game with OT support.
    
    Methodology (v12):
    - Sample minutes from quarter-specific history (captures blowout risk)
    - Sample PPM from quarter-specific history (Q4/OT use Q4 stats)
    - Filter Q4 zeros for OT projections (close game assumption)
    - Apply vegas_adjustment to PPM (one-time calibration at game start)
    - Apply confidence limits (quarter-based caps)
    - Apply empirical calibration (period × probability bin corrections from v9)
    - Apply conservative bias (3% reduction to err on underpredicting)
    
    Args:
        player_profile: dict with quarterly distributions
        current_minute: Current game minute (0-48+, OT possible)
        current_points: Points scored so far
        prop_line: Target line (e.g., 30.5)
        n_simulations: Number of simulations
        vegas_adjustment: PPM multiplier (default 1.0 = no adjustment)
        score_differential: Point differential for OT estimation (optional)
        debug: If True, print first 5 simulations
    
    Returns:
        prob_over: Final probability of hitting over (0.0 to 1.0)
    """
    # Quick check: already hit
    if current_points > prop_line:
        return 1.0
    
    # Get game state (handles OT detection)
    game_state = get_game_state(current_minute)
    
    # If past 3OT and didn't hit, return 0
    if game_state['is_ot'] and game_state['ot_period'] >= 3 and game_state['time_remaining'] <= 0:
        return 0.0
    
    hits = 0
    
    for sim_num in range(n_simulations):
        projected_final_points = current_points
        
        # 1. Project remainder of current quarter/OT period
        if game_state['time_remaining'] > 0:
            if not game_state['is_ot']:
                # Regular quarter projection
                current_quarter = game_state['quarter']
                time_remaining = game_state['time_remaining']
                minutes_key = f'q{current_quarter}_minutes_history'
                
                minutes_history = player_profile.get(minutes_key, [])
                
                # v8 change: Use quarter-specific PPM for Q4 (no fallback - fail if missing)
                if current_quarter == 4:
                    ppm_key = 'q4_points_per_minute_history'
                    ppm_history = player_profile[ppm_key]  # Will fail if missing
                else:
                    ppm_history = player_profile['points_per_minute_history']
                
                # v8: Keep zeros for Q4 (DNP/benched scenarios), filter for Q1-Q3
                if current_quarter == 4:
                    # Keep zeros - player might not play Q4
                    minutes_history_filtered = minutes_history if minutes_history else []
                    ppm_history_filtered = ppm_history if ppm_history else []
                else:
                    # Filter out zeros for Q1-Q3 (players typically play earlier quarters)
                    minutes_history_filtered = [m for m in minutes_history if m > 0]
                    ppm_history_filtered = [p for p in ppm_history if p > 0]
                
                if minutes_history_filtered and ppm_history_filtered:
                    typical_minutes = random.choice(minutes_history_filtered)
                    
                    # If sampled 0 minutes in Q4, skip projection
                    if typical_minutes == 0:
                        continue
                    
                    quarter_length = 12.0
                    proportion_remaining = time_remaining / quarter_length
                    projected_minutes = typical_minutes * proportion_remaining
                    ppm = random.choice(ppm_history_filtered) * vegas_adjustment
                    projected_final_points += ppm * projected_minutes
            else:
                # In OT - project remainder using Q4 stats (v8 change)
                time_remaining = game_state['time_remaining']
                ot_length = 5.0
                proportion_remaining = time_remaining / ot_length
                ot_points = project_ot_points(player_profile, vegas_adjustment, proportion_remaining)
                projected_final_points += ot_points
        
        # 2. Project future quarters (if in regulation)
        if not game_state['is_ot'] and game_state['quarter'] < 4:
            for future_quarter in range(game_state['quarter'] + 1, 5):
                minutes_key = f'q{future_quarter}_minutes_history'
                minutes_history = player_profile.get(minutes_key, [])
                
                # v8 change: Use Q4 PPM for Q4 projections (no fallback - fail if missing)
                if future_quarter == 4:
                    ppm_key = 'q4_points_per_minute_history'
                    ppm_history = player_profile[ppm_key]  # Will fail if missing
                else:
                    ppm_history = player_profile['points_per_minute_history']
                
                # v8: Keep zeros for Q4, filter for Q1-Q3
                if future_quarter == 4:
                    # Keep zeros - player might not play Q4
                    minutes_history_filtered = minutes_history if minutes_history else []
                    ppm_history_filtered = ppm_history if ppm_history else []
                else:
                    # Filter out zeros for Q1-Q3
                    minutes_history_filtered = [m for m in minutes_history if m > 0]
                    ppm_history_filtered = [p for p in ppm_history if p > 0]
                
                if minutes_history_filtered and ppm_history_filtered:
                    future_minutes = random.choice(minutes_history_filtered)
                    
                    # If sampled 0 minutes in Q4, skip projection
                    if future_minutes == 0:
                        continue
                    
                    future_ppm = random.choice(ppm_history_filtered) * vegas_adjustment
                    projected_final_points += future_ppm * future_minutes
        
        # v8: No future OT projection at all
        # Rationale: Don't add OT1 points while in Q4 - too speculative
        # Only project remaining minutes if ALREADY IN OT (handled in step 1 above)
        # Probability will naturally be low in late Q4, then spike when OT actually starts
        
        # Check if bet hits
        if projected_final_points > prop_line:
            hits += 1
    
    prob_over = hits / n_simulations
    
    # Apply minimum probability floor (never exactly 0% until past 3OT)
    MIN_PROB = 0.001
    if prob_over < MIN_PROB and current_minute < 63:  # 48 + 15 (3 OT periods)
        prob_over = MIN_PROB
    
    # Apply confidence limits (quarter-based caps + deterministic overrides)
    prob_over_limited = apply_confidence_limits(prob_over, current_minute, current_points, prop_line)
    
    # v10: Apply empirical calibration (period-specific bias correction)
    prob_calibrated = apply_calibration(prob_over_limited, game_state['quarter'])
    
    # v12: Apply conservative bias (reduce all probabilities by 3%)
    # Rationale: Err on the side of underpredicting to avoid overconfidence
    prob_final = prob_calibrated * CONSERVATIVE_FACTOR
    
    return prob_final


def monte_carlo_get_distribution(
    player_profile,
    current_minute,
    current_points,
    n_simulations=10000,
    vegas_adjustment=1.0,
    score_differential=None,
    debug=False
):
    """
    Run Monte Carlo simulation and return full distribution of projected final points.
    
    This is more efficient than calling monte_carlo_simulate_bet multiple times
    for different lines - we can calculate probabilities for all lines from one distribution.
    
    Args:
        player_profile: dict with quarterly distributions
        current_minute: Current game minute (0-48+, OT possible)
        current_points: Points scored so far
        n_simulations: Number of simulations
        vegas_adjustment: PPM multiplier (default 1.0 = no adjustment)
        score_differential: Point differential for OT estimation (optional)
        debug: If True, print first 5 simulations
    
    Returns:
        List of simulated final points (length = n_simulations)
    """
    # Get game state (handles OT detection)
    game_state = get_game_state(current_minute)
    
    # If past 3OT, return current points for all simulations
    if game_state['is_ot'] and game_state['ot_period'] >= 3 and game_state['time_remaining'] <= 0:
        return [current_points] * n_simulations
    
    simulated_finals = []
    
    for sim_num in range(n_simulations):
        projected_final_points = current_points
        
        # 1. Project remainder of current quarter/OT period
        if game_state['time_remaining'] > 0:
            if not game_state['is_ot']:
                # Regular quarter projection
                current_quarter = game_state['quarter']
                time_remaining = game_state['time_remaining']
                minutes_key = f'q{current_quarter}_minutes_history'
                
                minutes_history = player_profile.get(minutes_key, [])
                
                # Use quarter-specific PPM for Q4
                if current_quarter == 4:
                    ppm_key = 'q4_points_per_minute_history'
                    ppm_history = player_profile[ppm_key]
                else:
                    ppm_history = player_profile['points_per_minute_history']
                
                # Keep zeros for Q4, filter for Q1-Q3
                if current_quarter == 4:
                    minutes_history_filtered = minutes_history if minutes_history else []
                    ppm_history_filtered = ppm_history if ppm_history else []
                else:
                    minutes_history_filtered = [m for m in minutes_history if m > 0]
                    ppm_history_filtered = [p for p in ppm_history if p > 0]
                
                if minutes_history_filtered and ppm_history_filtered:
                    typical_minutes = random.choice(minutes_history_filtered)
                    
                    if typical_minutes > 0:
                        quarter_length = 12.0
                        proportion_remaining = time_remaining / quarter_length
                        projected_minutes = typical_minutes * proportion_remaining
                        ppm = random.choice(ppm_history_filtered) * vegas_adjustment
                        projected_final_points += ppm * projected_minutes
            else:
                # In OT - project remainder using Q4 stats
                time_remaining = game_state['time_remaining']
                ot_length = 5.0
                proportion_remaining = time_remaining / ot_length
                ot_points = project_ot_points(player_profile, vegas_adjustment, proportion_remaining)
                projected_final_points += ot_points
        
        # 2. Project future quarters (if in regulation)
        if not game_state['is_ot'] and game_state['quarter'] < 4:
            for future_quarter in range(game_state['quarter'] + 1, 5):
                minutes_key = f'q{future_quarter}_minutes_history'
                minutes_history = player_profile.get(minutes_key, [])
                
                # Use Q4 PPM for Q4 projections
                if future_quarter == 4:
                    ppm_key = 'q4_points_per_minute_history'
                    ppm_history = player_profile[ppm_key]
                else:
                    ppm_history = player_profile['points_per_minute_history']
                
                # Keep zeros for Q4, filter for Q1-Q3
                if future_quarter == 4:
                    minutes_history_filtered = minutes_history if minutes_history else []
                    ppm_history_filtered = ppm_history if ppm_history else []
                else:
                    minutes_history_filtered = [m for m in minutes_history if m > 0]
                    ppm_history_filtered = [p for p in ppm_history if p > 0]
                
                if minutes_history_filtered and ppm_history_filtered:
                    future_minutes = random.choice(minutes_history_filtered)
                    
                    if future_minutes > 0:
                        future_ppm = random.choice(ppm_history_filtered) * vegas_adjustment
                        projected_final_points += future_ppm * future_minutes
        
        # Store this simulation's result
        simulated_finals.append(projected_final_points)
        
        if debug and sim_num < 5:
            print(f"Simulation {sim_num + 1}: {projected_final_points:.1f} points")
    
    return simulated_finals


def find_vegas_adjustment(player_profile, prop_line, n_simulations=10000):
    """
    Use binary search to find a PPM adjustment that makes the starting
    probability exactly 50% (game minute 0, 0 points scored).

    This calibrates the model to market efficiency (Vegas consensus).

    Args:
        player_profile: dict with quarterly distributions
        prop_line: Target prop line
        n_simulations: Number of simulations for calibration

    Returns:
        float: PPM adjustment factor (multiplier)
    """
    low, high = 0.5, 1.5
    tolerance = 0.01  # Target 50% ± 1%
    max_iterations = 20

    for iteration in range(max_iterations):
        mid = (low + high) / 2

        prob = monte_carlo_simulate_bet(
            player_profile,
            current_minute=0,
            current_points=0,
            prop_line=prop_line,
            n_simulations=n_simulations,
            vegas_adjustment=mid,
            debug=False
        )

        if abs(prob - 0.5) < tolerance:
            return mid
        elif prob < 0.5:
            low = mid
        else:
            high = mid

    return (low + high) / 2


# Old calibration system removed in v10 - replaced with period-specific CALIBRATION_MAP_V9


def apply_confidence_limits(prob_over, current_minute, current_points, prop_line):
    """
    Apply confidence limits to reduce overconfidence.
    
    Methodology:
    1. Deterministic override: If already hit, return 100%
    2. Asymmetric dampening: Only dampen over-predictions, not under-predictions
    3. Quarter-based caps: Additional limits based on game progress
    
    Rationale:
    - Model systematically over-predicts, so only dampen high probabilities
    - Early in game, even high MC confidence should be tempered
    - Late in game, allow more confidence as uncertainty decreases
    
    Args:
        prob_over: Raw Monte Carlo probability (0-1)
        current_minute: Current game minute (0-48+)
        current_points: Points scored so far
        prop_line: Prop bet line (e.g., 29.5)
    
    Returns:
        Limited probability (0-1)
    """
    # Override: Already hit the line
    if current_points > prop_line:
        return 1.0
    
    # Asymmetric dampening: Only dampen overconfidence (prob > 0.5)
    # Leave underconfidence (prob < 0.5) alone - those are often correct
    if prob_over > 0.5:
        # Quarter-based dampening strength
        if current_minute <= 12:
            dampening = 0.20  # Q1: Dampen 20% toward 50%
        elif current_minute <= 24:
            dampening = 0.15  # Q2: Dampen 15%
        elif current_minute <= 36:
            dampening = 0.10  # Q3: Dampen 10%
        elif current_minute < 42:
            dampening = 0.05  # Q4 early: Light dampening
        else:
            dampening = 0.0  # Q4 late: No dampening
        
        # Apply dampening (pull toward 50%)
        prob_over = 0.5 + (prob_over - 0.5) * (1 - dampening)
    
    # Hard caps to prevent extreme predictions early
    if current_minute <= 12:
        max_prob = 0.85  # Q1
    elif current_minute <= 24:
        max_prob = 0.90  # Q2
    elif current_minute <= 36:
        max_prob = 0.93  # Q3
    elif current_minute < 42:
        max_prob = 0.95  # Q4 early
    elif current_minute < 46:
        max_prob = 0.97  # Q4 mid
    elif current_minute < 48:
        max_prob = 0.98  # Q4 late
    else:
        max_prob = 0.99  # OT/End
    
    # Apply hard cap (only on high side)
    if prob_over > max_prob:
        return max_prob
    else:
        return prob_over


# =============================================================================
# VISUALIZATION (R + GGPLOT2)
# =============================================================================

def create_ggplot(df, prop_line, player_name, player_id, game_id, game_date, 
                  away_team, home_team, final_points, result, plot_dir=None, 
                  bet_placement_minute=None, current_game_minute=None):
    """
    Create publication-quality plot using R/ggplot2.
    
    Args:
        df: DataFrame with game_minute, cumulative_points, prob_over columns
        prop_line: Prop line (e.g., 30.5)
        player_name: Player name
        player_id: ESPN player ID for headshot
        game_id: ESPN game ID
        game_date: Game date (YYYY-MM-DD)
        away_team: Away team name
        home_team: Home team name
        final_points: Final points scored
        result: "HIT" or "MISS" or "IN PROGRESS"
        plot_dir: Optional custom plot directory (uses project default if None)
        bet_placement_minute: Optional minute to mark model comparison point (e.g., 13.18)
        current_game_minute: Optional minute to mark current game state (e.g., 20.5)
    
    Returns:
        str: Path to saved plot, or None if failed
    """
    if plot_dir is None:
        project_root = get_project_root()
        plot_dir = project_root / "src" / "pbp_data" / "tmp" / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Save df to temp CSV for R
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    # Prepare output files
    player_name_clean = player_name.replace(" ", "_")
    plot_file = plot_dir / f"monte_carlo_pbp_{player_name_clean}_{game_id}_{game_date}.png"
    
    # Download team logos and player headshot (with caching)
    away_logo_path = download_team_logo(away_team, size=(100, 100))
    home_logo_path = download_team_logo(home_team, size=(100, 100))
    player_headshot_path = download_player_headshot(player_id, size=(120, 120))
    
    # Prepare R code
    result_label = "HIT ✅" if result == "HIT" else "MISS ❌"
    if result == "IN PROGRESS":
        result_label = "IN PROGRESS 🔄"
    prop_line_ceil = math.ceil(prop_line)
    # Format prop line: show "22" instead of "22.0", but keep "22.5" as "22.5"
    prop_line_display = f"{prop_line:.1f}".rstrip('0').rstrip('.')
    
    # Prepare observation marker if provided
    observation_marker_code = ""
    if bet_placement_minute is not None:
        # Calculate quarter and time remaining
        obs_minute = bet_placement_minute
        if obs_minute < 12:
            obs_quarter = 1
            time_left = 12 - obs_minute
        elif obs_minute < 24:
            obs_quarter = 2
            time_left = 24 - obs_minute
        elif obs_minute < 36:
            obs_quarter = 3
            time_left = 36 - obs_minute
        else:
            obs_quarter = 4
            time_left = 48 - obs_minute
        
        # Format time as MM:SS
        time_mins = int(time_left)
        time_secs = int((time_left - time_mins) * 60)
        time_str = f"Q{obs_quarter} {time_mins}:{time_secs:02d}"
        
        # Get probability at that minute
        obs_prob = df[df['game_minute'].apply(lambda x: abs(x - obs_minute) < 0.1)]['prob_over'].iloc[0] if len(df[df['game_minute'].apply(lambda x: abs(x - obs_minute) < 0.1)]) > 0 else 0.5
        obs_prob_pct = obs_prob * 100
        
        observation_marker_code = f'''
  # Model comparison point marker
  geom_vline(xintercept = {obs_minute}, color = "red", linetype = "dotted", linewidth = 1.2) +
  annotate("text", x = {obs_minute}, y = 95, 
           label = "Model Comparison\\n{time_str}\\n{obs_prob_pct:.1f}% prob",
           color = "red", size = 3.5, hjust = -0.1, fontface = "bold") +
'''
    
    # Add current game state marker if provided
    current_marker_code = ""
    if current_game_minute is not None:
        # Calculate quarter and time remaining for current state
        curr_minute = current_game_minute
        if curr_minute < 12:
            curr_quarter = 1
            curr_time_left = 12 - curr_minute
        elif curr_minute < 24:
            curr_quarter = 2
            curr_time_left = 24 - curr_minute
        elif curr_minute < 36:
            curr_quarter = 3
            curr_time_left = 36 - curr_minute
        else:
            curr_quarter = 4
            curr_time_left = 48 - curr_minute
        
        # Format time as MM:SS
        curr_time_mins = int(curr_time_left)
        curr_time_secs = int((curr_time_left - curr_time_mins) * 60)
        curr_time_str = f"Q{curr_quarter} {curr_time_mins}:{curr_time_secs:02d}"
        
        # Get probability at current minute
        curr_prob = df[df['game_minute'].apply(lambda x: abs(x - curr_minute) < 0.1)]['prob_over'].iloc[0] if len(df[df['game_minute'].apply(lambda x: abs(x - curr_minute) < 0.1)]) > 0 else 0.5
        curr_prob_pct = curr_prob * 100
        
        current_marker_code = f'''
  # Current game state marker
  geom_vline(xintercept = {curr_minute}, color = "red", linetype = "dotted", linewidth = 1.2) +
  annotate("text", x = {curr_minute}, y = 10, 
           label = "Current\\n{curr_time_str}\\n{curr_prob_pct:.1f}% prob",
           color = "red", size = 3.5, hjust = -0.1, fontface = "bold") +
'''
    
    r_code = f'''
library(ggplot2)
library(dplyr)
library(zoo)
library(patchwork)
library(png)
library(grid)

# Read data
df <- read.csv("{temp_csv.name}")

# Calculate smoothed probability and fill NAs with raw values
df <- df %>%
  arrange(game_minute) %>%
  mutate(
    prob_over_smooth_raw = rollmean(prob_over, k=20, fill=NA, align="center"),
    prob_over_smooth = ifelse(is.na(prob_over_smooth_raw), prob_over, prob_over_smooth_raw),
    pace_line = {prop_line_ceil} * game_minute / 48,
    ahead_of_pace = cumulative_points >= pace_line
  )

# Top plot: Probability
p1 <- ggplot(df) +
  # Shaded regions (over/under 50%)
  geom_ribbon(aes(x = game_minute, ymin = 50, ymax = prob_over_smooth * 100,
                  fill = ifelse(prob_over_smooth * 100 >= 50, "Prob > 50%", "Prob < 50%")),
              alpha = 0.3) +
  
  # Lines
  geom_line(aes(x = game_minute, y = prob_over * 100, linetype = "Raw MC"), 
            color = "blue", alpha = 0.4, linewidth = 0.5) +
  geom_line(aes(x = game_minute, y = prob_over_smooth * 100, linetype = "Smoothed (20-play avg)"), 
            color = "blue", linewidth = 1.5) +
  geom_hline(aes(yintercept = 50, linetype = "50% baseline"), 
             color = "gray40", linewidth = 0.8) +
  {observation_marker_code}{current_marker_code}
  # Styling
  scale_fill_manual(values = c("Prob > 50%" = "green", "Prob < 50%" = "red"), 
                    name = "", na.translate = FALSE, guide = "none") +
  scale_linetype_manual(values = c("Raw MC" = "solid", 
                                   "Smoothed (20-play avg)" = "solid",
                                   "50% baseline" = "dashed"),
                        name = "") +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_x_continuous(limits = c(0, 48), breaks = seq(0, 48, 12), expand = c(0, 0)) +
  labs(title = paste0("{player_name} - Monte Carlo (Over {prop_line_display} pts)"),
       subtitle = paste0("Game: {away_team} @ {home_team} on {game_date} | Vegas-Adjusted (starts at 50%)"),
       y = "Probability (%)") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    plot.subtitle = element_text(size = 12, color = "gray30", hjust = 0),
    axis.title.x = element_blank(),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    legend.position = "top",
    legend.justification = "left",
    legend.box = "horizontal",
    legend.margin = margin(0,0,5,0),
    plot.margin = margin(t = 100, r = 10, b = 10, l = 10)
  )

# Bottom plot: Points vs Pace Line (colored by ahead/behind pace)
p2 <- ggplot(df, aes(x = game_minute)) +
  # Filled area (green when ahead of pace, red when behind)
  geom_ribbon(aes(ymin = pace_line, ymax = cumulative_points, fill = ahead_of_pace),
              alpha = 0.3) +
  
  # Points line (royal blue like MC smoothed line)
  geom_line(aes(y = cumulative_points, linetype = "Actual Points"), 
            color = "#4169E1", linewidth = 1.5) +
  
  # Pace line (straight line from 0 to ceil(prop_line))
  geom_line(aes(y = pace_line, linetype = "Pace Line"),
            color = "black", linewidth = 1.2) +
  
  # Prop line (for reference)
  geom_hline(aes(yintercept = {prop_line}, linetype = "Prop Line"), 
             color = "gray50", linewidth = 1.0) +
  
  # Styling
  scale_fill_manual(values = c("TRUE" = "green", "FALSE" = "red"),
                    labels = c("TRUE" = "Ahead of pace", "FALSE" = "Behind pace"),
                    name = "") +
  scale_linetype_manual(values = c("Actual Points" = "solid",
                                   "Pace Line" = "solid",
                                   "Prop Line" = "dashed"),
                        name = "") +
  scale_x_continuous(limits = c(0, 48), breaks = seq(0, 48, 12), expand = c(0, 0)) +
  labs(subtitle = paste0("Target: {prop_line_ceil} pts | Final: {final_points} pts ({result_label})"),
       x = "Game Time (minutes)",
       y = "Points Scored") +
  theme_minimal(base_size = 14) +
  theme(
    plot.subtitle = element_text(face = "bold", size = 13, hjust = 0),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    legend.position = "top",
    legend.justification = "left",
    legend.box = "horizontal"
  )

# Combine plots
combined <- p1 / p2 + plot_layout(heights = c(1.2, 1))

# Open PNG device
png("{plot_file}", width = 14, height = 11, units = "in", res = 150, bg = "white")

# Create viewport for entire plot
grid.newpage()
pushViewport(viewport(width = 1, height = 1))

# Draw the combined ggplot
print(combined)

# Add logos and headshot at top (clean spacing)
grid.raster(readPNG("{away_logo_path}"), x = 0.12, y = 0.92, width = 0.06, height = 0.06, just = c("center", "top"))
grid.raster(readPNG("{home_logo_path}"), x = 0.88, y = 0.92, width = 0.06, height = 0.06, just = c("center", "top"))
grid.raster(readPNG("{player_headshot_path}"), x = 0.5, y = 0.92, width = 0.07, height = 0.07, just = c("center", "top"))

dev.off()

cat("✅ Plot saved to {plot_file}\\n")
'''
    
    # Execute R code
    try:
        result = subprocess.run(
            ['Rscript', '-e', r_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"   ❌ R plotting failed:")
            print(result.stderr)
            return None
        
        # Clean up temp CSV (keep cached images for reuse)
        os.unlink(temp_csv.name)
        
        return str(plot_file)
        
    except Exception as e:
        print(f"   ❌ Error running R: {e}")
        return None


# =============================================================================
# MODULE INITIALIZATION - Verify calibration on import
# =============================================================================

# Verify calibration map on module import (silent if v9 data not available)
try:
    load_v9_bias_correction_table()
except Exception:
    pass  # Silently continue if verification fails (v9 data may not exist)
