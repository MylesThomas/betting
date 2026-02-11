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

def load_player_profile(player_name, minute_by_minute_path=None):
    """
    Load player's historical data (quarterly distributions) using DuckDB.
    
    Args:
        player_name: Player name
        minute_by_minute_path: Optional path to minute_by_minute.parquet
    
    Returns:
        dict with quarterly distributions (lists) and player_id
    """
    if minute_by_minute_path is None:
        minute_by_minute_path = get_data_paths()['minute_by_minute']
    
    con = duckdb.connect()
    
    # Build player profiles from minute_by_minute data
    con.execute(f"""
        -- Step 1: Game-level stats
        CREATE OR REPLACE TEMP TABLE game_level_stats AS
        SELECT 
            game_id,
            game_date,
            player_id,
            player_name,
            MAX(playing_seconds) / 60.0 AS total_minutes,
            MAX(cumulative_points) AS total_points
        FROM '{minute_by_minute_path}'
        GROUP BY game_id, game_date, player_id, player_name;
        
        CREATE OR REPLACE TEMP TABLE game_stats_with_ppm AS
        SELECT 
            *,
            CASE 
                WHEN total_minutes > 0 THEN total_points / total_minutes 
                ELSE 0 
            END AS points_per_minute
        FROM game_level_stats;
        
        -- Step 2: Quarterly splits
        CREATE OR REPLACE TEMP TABLE quarter_splits AS
        SELECT 
            game_id,
            game_date,
            player_id,
            player_name,
            
            -- Q1 (minutes 0-11)
            MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END) / 60.0 AS q1_minutes,
            MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END) AS q1_points,
            
            -- Q2 (minutes 12-23)
            (MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END) - 
             MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END)) / 60.0 AS q2_minutes,
            (MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END) - 
             MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END)) AS q2_points,
            
            -- Q3 (minutes 24-35)
            (MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END) - 
             MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END)) / 60.0 AS q3_minutes,
            (MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END) - 
             MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END)) AS q3_points,
            
            -- Q4 (minutes 36-47)
            (MAX(CASE WHEN minute <= 47 THEN playing_seconds ELSE 0 END) - 
             MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END)) / 60.0 AS q4_minutes,
            (MAX(CASE WHEN minute <= 47 THEN cumulative_points ELSE 0 END) - 
             MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END)) AS q4_points
            
        FROM '{minute_by_minute_path}'
        GROUP BY game_id, game_date, player_id, player_name;
        
        CREATE OR REPLACE TEMP TABLE quarter_splits_with_ppm AS
        SELECT 
            *,
            CASE WHEN q1_minutes > 0 THEN q1_points / q1_minutes ELSE 0 END AS q1_ppm,
            CASE WHEN q2_minutes > 0 THEN q2_points / q2_minutes ELSE 0 END AS q2_ppm,
            CASE WHEN q3_minutes > 0 THEN q3_points / q3_minutes ELSE 0 END AS q3_ppm,
            CASE WHEN q4_minutes > 0 THEN q4_points / q4_minutes ELSE 0 END AS q4_ppm
        FROM quarter_splits;
    """)
    
    # Get player profile with full game PPM history
    query = """
    SELECT 
        g.player_id,
        g.player_name,
        COUNT(*) AS num_games,
        
        -- Summary stats
        AVG(g.total_points) AS avg_points_per_game,
        AVG(g.total_minutes) AS avg_minutes_per_game,
        
        -- Q1 minutes distribution
        LIST(q.q1_minutes ORDER BY q.game_date DESC) AS q1_minutes_history,
        
        -- Q2 minutes distribution
        LIST(q.q2_minutes ORDER BY q.game_date DESC) AS q2_minutes_history,
        
        -- Q3 minutes distribution
        LIST(q.q3_minutes ORDER BY q.game_date DESC) AS q3_minutes_history,
        
        -- Q4 minutes distribution
        LIST(q.q4_minutes ORDER BY q.game_date DESC) AS q4_minutes_history,
        
        -- Full game PPM history (used for sampling PPM)
        LIST(g.points_per_minute ORDER BY q.game_date DESC) AS points_per_minute_history
        
    FROM game_stats_with_ppm g
    LEFT JOIN quarter_splits_with_ppm q 
        ON g.game_id = q.game_id 
        AND g.player_id = q.player_id
    WHERE g.player_name = ?
    GROUP BY g.player_id, g.player_name
    """
    
    result = con.execute(query, [player_name]).fetchone()
    
    if not result:
        con.close()
        raise ValueError(f"Player {player_name} not found")
    
    profile = {
        'player_id': result[0],
        'player_name': result[1],
        'num_games': result[2],
        'avg_points_per_game': result[3],
        'avg_minutes_per_game': result[4],
        'q1_minutes_history': result[5],
        'q2_minutes_history': result[6],
        'q3_minutes_history': result[7],
        'q4_minutes_history': result[8],
        'points_per_minute_history': result[9],
    }
    
    con.close()
    
    return profile


# =============================================================================
# LOAD PROP LINES
# =============================================================================

def get_consensus_prop_line(player_name, game_date, market="player_points"):
    """
    Get consensus (median) prop line from locally synced S3 data.
    
    Args:
        player_name: Player name (e.g., "Luka Doncic")
        game_date: Game date as string "YYYY-MM-DD"
        market: Market type (default: "player_points")
    
    Returns:
        float: Consensus prop line (median), or None if not found
    """
    paths = get_data_paths()
    player_props_dir = paths['player_props_dir']
    
    try:
        # Convert game_date to ET datetime for matching
        game_dt = pd.to_datetime(game_date).tz_localize("America/New_York")
        
        # Search for prop files on the game date
        date_str = game_dt.strftime("%Y%m%d")
        
        files = list(player_props_dir.glob(f"*{date_str}*.parquet"))
        
        if not files:
            return None
        
        # Load all files for this date
        con = duckdb.connect()
        
        for file in files:
            try:
                df = con.execute(f"""
                    SELECT 
                        player_name,
                        market,
                        point,
                        commence_time_et
                    FROM '{file}'
                    WHERE player_name = ?
                    AND market = ?
                """, [player_name, market]).df()
                
                if len(df) > 0:
                    # Calculate consensus: median, but if median is average of two values,
                    # take the higher of the two (more conservative for betting)
                    points = sorted(df['point'].values)
                    n = len(points)
                    
                    if n % 2 == 1:
                        # Odd number: median is the middle value
                        consensus = points[n // 2]
                    else:
                        # Even number: take the higher of the two middle values
                        consensus = points[n // 2]  # This is the higher one after sorting
                    
                    con.close()
                    return float(consensus)
            except Exception:
                continue
        
        con.close()
        return None
        
    except Exception as e:
        print(f"⚠️  Error loading prop line: {e}")
        return None


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
    Project points in OT period using Q1 starter stats.
    
    Rationale:
    - OT predominantly uses best players (like Q1 starters)
    - Q1 is 12 minutes, OT is 5 minutes (ratio: 5/12 ≈ 0.417)
    - Q1 captures high-usage starter minutes
    
    Args:
        player_profile: Player's historical stats
        vegas_adjustment: PPM multiplier
        proportion: Proportion of OT period to project (default 1.0 = full 5 min)
    
    Returns:
        Projected points for this OT portion
    """
    # Use Q1 minutes as proxy for OT (best players)
    q1_minutes_history = player_profile.get('q1_minutes_history', [])
    q1_minutes_history = [m for m in q1_minutes_history if m > 0]
    
    if not q1_minutes_history:
        # Fallback: use Q4 (starters in close games)
        q1_minutes_history = player_profile.get('q4_minutes_history', [])
        q1_minutes_history = [m for m in q1_minutes_history if m > 0]
    
    if not q1_minutes_history:
        return 0  # Player doesn't play meaningful minutes
    
    # Sample Q1 minutes
    typical_q1_minutes = random.choice(q1_minutes_history)
    
    # Scale to OT length: 5 min vs 12 min Q1
    ot_length = 5.0
    q1_length = 12.0
    scale_factor = (ot_length / q1_length) * proportion
    
    projected_ot_minutes = typical_q1_minutes * scale_factor
    
    # Get PPM from full game history
    ppm_history = player_profile.get('points_per_minute_history', [])
    ppm_history = [p for p in ppm_history if p > 0]
    
    if not ppm_history:
        return 0
    
    ot_ppm = random.choice(ppm_history) * vegas_adjustment
    
    # Calculate OT points
    ot_points = ot_ppm * projected_ot_minutes
    
    return ot_points


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
    
    Methodology:
    - Sample minutes from quarter-specific history (captures blowout risk)
    - Sample PPM from full game history (assumes aggressive scoring when playing)
    - Apply vegas_adjustment to PPM (one-time calibration at game start)
    - Model OT probability and project OT points using Q1 starter stats
    
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
        prob_over: Probability of hitting over
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
                ppm_history = player_profile.get('points_per_minute_history', [])
                
                # Filter out zeros
                minutes_history = [m for m in minutes_history if m > 0]
                ppm_history = [p for p in ppm_history if p > 0]
                
                if minutes_history and ppm_history:
                    typical_minutes = random.choice(minutes_history)
                    quarter_length = 12.0
                    proportion_remaining = time_remaining / quarter_length
                    projected_minutes = typical_minutes * proportion_remaining
                    ppm = random.choice(ppm_history) * vegas_adjustment
                    projected_final_points += ppm * projected_minutes
            else:
                # In OT - project remainder using Q1 stats
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
                ppm_history = player_profile.get('points_per_minute_history', [])
                
                # Filter out zeros
                minutes_history = [m for m in minutes_history if m > 0]
                ppm_history = [p for p in ppm_history if p > 0]
                
                if minutes_history and ppm_history:
                    future_minutes = random.choice(minutes_history)
                    future_ppm = random.choice(ppm_history) * vegas_adjustment
                    projected_final_points += future_ppm * future_minutes
        
        # 3. Model OT probability and project OT periods
        current_ot_period = game_state['ot_period']
        
        # Check for OT1 (if ending Q4)
        if not game_state['is_ot'] and game_state['quarter'] == 4:
            ot_prob = estimate_ot_probability(current_minute, 0, score_differential)
            if random.random() < ot_prob:
                # Add OT1
                projected_final_points += project_ot_points(player_profile, vegas_adjustment, 1.0)
                current_ot_period = 1
        
        # Check for OT2 (if in/finished OT1)
        if current_ot_period == 1:
            ot_prob = estimate_ot_probability(current_minute, 1, score_differential)
            if random.random() < ot_prob:
                # Add OT2
                projected_final_points += project_ot_points(player_profile, vegas_adjustment, 1.0)
                current_ot_period = 2
        
        # Check for OT3 (if in/finished OT2)
        if current_ot_period == 2:
            ot_prob = estimate_ot_probability(current_minute, 2, score_differential)
            if random.random() < ot_prob:
                # Add OT3 (capped here as requested)
                projected_final_points += project_ot_points(player_profile, vegas_adjustment, 1.0)
        
        # Check if bet hits
        if projected_final_points > prop_line:
            hits += 1
    
    prob_over = hits / n_simulations
    
    # Apply minimum probability floor (never exactly 0% until past 3OT)
    MIN_PROB = 0.001
    if prob_over < MIN_PROB and current_minute < 63:  # 48 + 15 (3 OT periods)
        prob_over = MIN_PROB
    
    # Step 1: Apply confidence limits (quarter-based caps + deterministic overrides)
    prob_over_limited = apply_confidence_limits(prob_over, current_minute, current_points, prop_line)
    
    # Step 2: Apply empirical calibration (v7 bias correction)
    prob_over_calibrated = apply_calibration(prob_over_limited)
    
    return prob_over_calibrated


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
    # Binary search bounds
    low, high = 0.5, 1.5
    tolerance = 0.01  # Target 50% ± 1%
    max_iterations = 20
    
    for iteration in range(max_iterations):
        mid = (low + high) / 2
        
        # Run simulation at game start with this adjustment
        prob = monte_carlo_simulate_bet(
            player_profile,
            current_minute=0,
            current_points=0,
            prop_line=prop_line,
            n_simulations=n_simulations,
            vegas_adjustment=mid,
            debug=False
        )
        
        # Adjust bounds
        if abs(prob - 0.5) < tolerance:
            return mid
        elif prob < 0.5:
            low = mid  # Need more points → increase PPM
        else:
            high = mid  # Need fewer points → decrease PPM
    
    # Return best guess
    return (low + high) / 2


# =============================================================================
# CALIBRATION DATA (from v7 validation - 2026-02-10)
# =============================================================================
# Source: v7 validation with 659,249 predictions across 710 games
# Query used to generate this data:
#
# duckdb -c "
# WITH bucketed_predictions AS (
#     SELECT 
#         prob_over,
#         CASE WHEN result = 'HIT' THEN 1 ELSE 0 END as actual_outcome,
#         FLOOR(prob_over * 20) / 20.0 as bucket_start,
#         FLOOR(prob_over * 20) / 20.0 + 0.05 as bucket_end
#     FROM '~/Downloads/tmp/monte_carlo_validation/versions/v7/predictions.parquet'
# ),
# calibration_stats AS (
#     SELECT 
#         bucket_start,
#         bucket_end,
#         ROUND((bucket_start + bucket_end) / 2.0, 3) as bucket_midpoint,
#         COUNT(*) as n_predictions,
#         ROUND(AVG(prob_over), 4) as avg_predicted_prob,
#         ROUND(AVG(actual_outcome), 4) as actual_hit_rate,
#         ROUND(AVG(prob_over) - AVG(actual_outcome), 4) as bias,
#         ROUND(AVG(POWER(prob_over - actual_outcome, 2)), 4) as avg_brier
#     FROM bucketed_predictions
#     GROUP BY bucket_start, bucket_end
#     ORDER BY bucket_start
# )
# SELECT 
#     bucket_start || '-' || ROUND(bucket_end * 100, 0) || '%' as bucket_range,
#     avg_predicted_prob,
#     actual_hit_rate,
#     n_predictions
# FROM calibration_stats
# WHERE bucket_midpoint < 1.0;"
#
# Raw query results stored below for reproducibility

V7_RAW_CALIBRATION_DATA = [
    # (bucket_range, avg_predicted_prob, actual_hit_rate, n_predictions)
    ('0.0-5.0%',     0.0089,  0.041,   150058),
    ('0.05-10.0%',   0.0737,  0.1322,   32921),
    ('0.1-15.0%',    0.1237,  0.1678,   27767),
    ('0.15-20.0%',   0.1743,  0.1942,   25147),
    ('0.2-25.0%',    0.2245,  0.2402,   23296),
    ('0.25-30.0%',   0.2744,  0.2718,   22729),
    ('0.3-35.0%',    0.3246,  0.2996,   23473),
    ('0.35-40.0%',   0.3749,  0.3233,   24974),
    ('0.4-45.0%',    0.4248,  0.3472,   26254),
    ('0.45-50.0%',   0.4737,  0.3752,   25760),
    ('0.5-55.0%',    0.5239,  0.414,    23796),
    ('0.55-60.0%',   0.575,   0.4642,   22400),
    ('0.6-65.0%',    0.6247,  0.4991,   21549),
    ('0.65-70.0%',   0.6746,  0.5325,   20291),
    ('0.7-75.0%',    0.7246,  0.5541,   19745),
    ('0.75-80.0%',   0.7751,  0.5856,   19465),
    ('0.8-85.0%',    0.825,   0.6141,   20754),
    ('0.85-90.0%',   0.8674,  0.7018,   26152),
    ('0.9-95.0%',    0.918,   0.7953,   37268),
    ('0.95-100.0%',  0.9619,  0.5824,    5938),
]

# Derived calibration mapping: (avg_predicted_prob → actual_hit_rate)
# Maps what v7 predicted to what actually happened
V7_CALIBRATION_MAPPING = [
    (pred, actual) 
    for _, pred, actual, _ in V7_RAW_CALIBRATION_DATA
]


def apply_calibration(prob_over):
    """
    Apply empirical calibration correction using v7 validation data.
    
    Maps raw MC probabilities to calibrated probabilities based on observed
    discrepancies in v7. Uses linear interpolation between calibration points
    for smooth transitions.
    
    Example: When v7 predicted 92%, it actually hit 79.5% of the time.
             So we map 0.92 → 0.795 instead.
    
    Key insight: High predictions (>70%) are systematically overconfident,
                 low predictions (<25%) are systematically underconfident.
    
    Args:
        prob_over: Raw probability from MC simulation (after confidence limits)
    
    Returns:
        Calibrated probability based on v7 empirical performance
    """
    # Handle edge cases
    if prob_over >= 0.999:
        return 0.999
    if prob_over <= 0.001:
        return 0.001
    
    # Find surrounding calibration points for linear interpolation
    for i in range(len(V7_CALIBRATION_MAPPING) - 1):
        pred1, actual1 = V7_CALIBRATION_MAPPING[i]
        pred2, actual2 = V7_CALIBRATION_MAPPING[i + 1]
        
        if pred1 <= prob_over <= pred2:
            # Linear interpolation
            t = (prob_over - pred1) / (pred2 - pred1)
            calibrated = actual1 + t * (actual2 - actual1)
            return max(0.001, min(0.999, calibrated))
    
    # Edge cases: beyond calibration range
    if prob_over < V7_CALIBRATION_MAPPING[0][0]:
        # Below lowest calibration point (~0.9%)
        return V7_CALIBRATION_MAPPING[0][1]
    else:
        # Above 96% - cap at last observed hit rate
        # (v7 showed 96%+ predictions only hit 58% - very unreliable!)
        return V7_CALIBRATION_MAPPING[-1][1]


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
                  away_team, home_team, final_points, result, plot_dir=None):
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
        result: "HIT" or "MISS"
        plot_dir: Optional custom plot directory (uses project default if None)
    
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
    prop_line_ceil = math.ceil(prop_line)
    # Format prop line: show "22" instead of "22.0", but keep "22.5" as "22.5"
    prop_line_display = f"{prop_line:.1f}".rstrip('0').rstrip('.')
    
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
