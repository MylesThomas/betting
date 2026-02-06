"""
Demo: Monte Carlo simulation for NBA player props with play-by-play updates.

Goal:
1. Load player's historical data (quarterly distributions)
2. Load a game's play-by-play data
3. For each play in the game, run Monte Carlo simulations
4. Plot:
   - Top: Probability of covering Over prop line over time (with smoothing)
   - Bottom: Actual points scored over time (colored by pace line)

Features:
- Uses real consensus prop lines from S3 data
- Vegas adjustment to start at 50% probability
- R/ggplot2 for publication-quality plots
- Team logos and player headshots
- Pace line visualization (green when ahead, red when behind)

Usage:
    # Single game
    python src/pbp_data/tmp/demo_monte_carlo_pbp.py --player-name "Luka Doncic" --game-id 401809820 --n-sims 1000
    
    # All games
    python src/pbp_data/tmp/demo_monte_carlo_pbp.py --player-name "LeBron James" --game-id all --n-sims 1000
    
    # With consensus prop lines
    python src/pbp_data/tmp/demo_monte_carlo_pbp.py --player-name "Luka Doncic" --game-id all --n-sims 1000 --use-consensus
"""

import duckdb
import pandas as pd
import numpy as np
import random
import argparse
import json
import subprocess
import sys
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


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MINUTE_BY_MINUTE = DATA_DIR / "minute_by_minute.parquet"
PBP_DATA_DIR = Path.home() / "Downloads" / "tmp" / "player_points_monte_carlo" / "pbp_data"
PLAYER_PROPS_DIR = Path.home() / "Downloads" / "tmp" / "player_props_raw" / "2025-26"
PLOT_DIR = PROJECT_ROOT / "src" / "pbp_data" / "tmp" / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)


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
    """Download team logo from ESPN CDN and return temp file path."""
    if team_name not in TEAM_LOGOS:
        print(f"⚠️  Team '{team_name}' not found in TEAM_LOGOS dict")
        return create_blank_image(size)
    
    url = TEAM_LOGOS[team_name]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    
    try:
        response = requests.get(url, timeout=5, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(temp_file.name, "PNG")
        
        return temp_file.name
    except Exception as e:
        print(f"⚠️  Failed to download logo for {team_name}: {e}")
        return create_blank_image(size)


def download_player_headshot(player_id, size=(120, 120)):
    """Download player headshot from NBA CDN and return temp file path."""
    url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    
    try:
        response = requests.get(url, timeout=5, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(temp_file.name, "PNG")
        
        return temp_file.name
    except Exception as e:
        print(f"⚠️  Failed to download headshot for player_id {player_id}: {e}")
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

def load_player_profile(player_name):
    """
    Load player's historical data (quarterly distributions) using DuckDB.
    
    Returns:
        dict with quarterly distributions (lists) and player_id
    """
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
        FROM '{MINUTE_BY_MINUTE}'
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
            
        FROM '{MINUTE_BY_MINUTE}'
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
    query = f"""
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
        LIST(g.points_per_minute ORDER BY g.game_date DESC) AS points_per_minute_history
        
    FROM game_stats_with_ppm g
    LEFT JOIN quarter_splits_with_ppm q 
        ON g.game_id = q.game_id 
        AND g.player_id = q.player_id
    WHERE g.player_name = '{player_name}'
    GROUP BY g.player_id, g.player_name
    """
    
    result = con.execute(query).fetchone()
    
    if not result:
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
    try:
        # Convert game_date to ET datetime for matching
        game_dt = pd.to_datetime(game_date).tz_localize("America/New_York")
        
        # Search for prop files on the game date
        date_str = game_dt.strftime("%Y%m%d")
        pattern = PLAYER_PROPS_DIR / f"*{date_str}*.parquet"
        
        files = list(PLAYER_PROPS_DIR.glob(f"*{date_str}*.parquet"))
        
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
                    WHERE player_name = '{player_name}'
                    AND market = '{market}'
                """).df()
                
                if len(df) > 0:
                    # Calculate median (consensus)
                    consensus = df['point'].median()
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
    
    Returns:
        DataFrame with columns: [play_id, quarter, game_minute, description, 
                                 away_score, home_score, player_points]
        game_metadata: dict with away_team, home_team, game_date, commence_time_et
    """
    # Find the JSON file
    json_files = list(PBP_DATA_DIR.glob(f"*_{game_id}.json"))
    
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
    
    # Parse plays
    plays = data['plays']
    play_data = []
    player_points = 0
    
    for play in plays:
        play_id = play.get('id')
        quarter = play.get('period', {}).get('number', 1)
        
        # Calculate game minute from clock
        clock_display = play.get('clock', {}).get('displayValue', '12:00')
        try:
            mins, secs = map(int, clock_display.split(':'))
            time_left_in_quarter = mins + secs / 60.0
            
            # Game minute = start of quarter + time elapsed in quarter
            quarter_start = (quarter - 1) * 12
            game_minute = quarter_start + (12 - time_left_in_quarter)
        except Exception:
            # Fallback if clock parsing fails
            game_minute = (quarter - 1) * 12
        
        description = play.get('text', '')
        away_score = play.get('awayScore', 0)
        home_score = play.get('homeScore', 0)
        
        # Check if player scored
        if player_name in description:
            # Check for scoring plays
            if 'makes' in description.lower() or 'free throw' in description.lower():
                # Extract points (e.g., "makes 2-pt", "makes 3-pt", "free throw")
                if '3-pt' in description.lower() or 'three point' in description.lower():
                    player_points += 3
                elif '2-pt' in description.lower() or 'two point' in description.lower():
                    player_points += 2
                elif 'free throw' in description.lower() and 'makes' in description.lower():
                    player_points += 1
        
        play_data.append({
            'play_id': play_id,
            'quarter': quarter,
            'game_minute': game_minute,
            'description': description,
            'away_score': away_score,
            'home_score': home_score,
            'cumulative_points': player_points,
        })
    
    df = pd.DataFrame(play_data)
    df = df.sort_values('game_minute').reset_index(drop=True)
    
    return df, game_metadata


# =============================================================================
# MONTE CARLO SIMULATION WITH VEGAS ADJUSTMENT
# =============================================================================

def find_vegas_adjustment(player_profile, prop_line, n_simulations=10000):
    """
    Use binary search to find a PPM adjustment that makes the starting 
    probability exactly 50% (game minute 0, 0 points scored).
    
    This calibrates the model to market efficiency (Vegas consensus).
    
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


def monte_carlo_simulate_bet(
    player_profile,
    current_minute,
    current_points,
    prop_line,
    n_simulations=10000,
    vegas_adjustment=1.0,
    debug=False
):
    """
    Run Monte Carlo simulation for remaining game.
    
    Methodology:
    - Sample minutes from quarter-specific history (captures blowout risk)
    - Sample PPM from full game history (assumes aggressive scoring when playing)
    - Apply vegas_adjustment to PPM (one-time calibration at game start)
    
    Args:
        player_profile: dict with quarterly distributions
        current_minute: Current game minute (0-47)
        current_points: Points scored so far
        prop_line: Target line (e.g., 30.5)
        n_simulations: Number of simulations
        vegas_adjustment: PPM multiplier (default 1.0 = no adjustment)
        debug: If True, print first 5 simulations
    
    Returns:
        prob_over: Probability of hitting over
    """
    # Determine current quarter and time remaining
    if current_minute < 12:
        current_quarter = 1
        time_remaining_in_quarter = 12 - current_minute
    elif current_minute < 24:
        current_quarter = 2
        time_remaining_in_quarter = 24 - current_minute
    elif current_minute < 36:
        current_quarter = 3
        time_remaining_in_quarter = 36 - current_minute
    else:
        current_quarter = 4
        time_remaining_in_quarter = 48 - current_minute
    
    # If game is over, return deterministic result
    if time_remaining_in_quarter <= 0 and current_quarter >= 4:
        return 1.0 if current_points > prop_line else 0.0
    
    hits = 0
    
    for sim_num in range(n_simulations):
        projected_final_points = current_points
        
        # Current quarter (partial) - project remaining time
        if time_remaining_in_quarter > 0:
            minutes_key = f'q{current_quarter}_minutes_history'
            
            minutes_history = player_profile[minutes_key]
            ppm_history = player_profile['points_per_minute_history']
            
            # Filter out zeros
            minutes_history = [m for m in minutes_history if m > 0]
            ppm_history = [p for p in ppm_history if p > 0]
            
            if minutes_history and ppm_history:
                # Sample minutes from quarter-specific history
                typical_minutes_this_quarter = random.choice(minutes_history)
                
                # Scale by proportion of quarter remaining
                quarter_length = 12.0
                proportion_remaining = time_remaining_in_quarter / quarter_length
                projected_minutes_remaining = typical_minutes_this_quarter * proportion_remaining
                
                # Sample PPM from full game history
                current_q_ppm = random.choice(ppm_history) * vegas_adjustment
                
                # Calculate points
                remaining_quarter_points = current_q_ppm * projected_minutes_remaining
                projected_final_points += remaining_quarter_points
        
        # Future quarters
        for future_quarter in range(current_quarter + 1, 5):
            minutes_key = f'q{future_quarter}_minutes_history'
            
            minutes_history = player_profile[minutes_key]
            ppm_history = player_profile['points_per_minute_history']
            
            # Filter out zeros
            minutes_history = [m for m in minutes_history if m > 0]
            ppm_history = [p for p in ppm_history if p > 0]
            
            if minutes_history and ppm_history:
                # Sample minutes from quarter-specific history
                future_q_minutes = random.choice(minutes_history)
                
                # Sample PPM from full game history
                future_q_ppm = random.choice(ppm_history) * vegas_adjustment
                
                future_quarter_points = future_q_ppm * future_q_minutes
                projected_final_points += future_quarter_points
        
        # Check if bet hits
        if projected_final_points > prop_line:
            hits += 1
    
    prob_over = hits / n_simulations
    return prob_over


# =============================================================================
# VISUALIZATION (R + GGPLOT2)
# =============================================================================

def create_ggplot(df, prop_line, player_name, player_id, game_id, game_date, 
                  away_team, home_team, final_points, result):
    """
    Create publication-quality plot using R/ggplot2.
    
    - Top panel: Probability over time (with smoothing, shaded areas)
    - Bottom panel: Points vs pace line (green when ahead, red when behind)
    """
    # Save df to temp CSV for R
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    # Prepare output files
    player_name_clean = player_name.replace(" ", "_")
    plot_file = PLOT_DIR / f"monte_carlo_pbp_{player_name_clean}_{game_id}_{game_date}.png"
    
    # Download team logos and player headshot
    print("   📥 Downloading images...")
    away_logo_path = download_team_logo(away_team, size=(100, 100))
    home_logo_path = download_team_logo(home_team, size=(100, 100))
    player_headshot_path = download_player_headshot(player_id, size=(120, 120))
    
    # Prepare R code
    result_label = "HIT ✅" if result == "HIT" else "MISS ❌"
    prop_line_ceil = math.ceil(prop_line)
    
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
  labs(title = paste0("{player_name} - Monte Carlo (Over {prop_line} pts)"),
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
    plot.margin = margin(t = 60, r = 10, b = 10, l = 10)
  )

# Bottom plot: Points vs Pace Line (colored by ahead/behind pace)
p2 <- ggplot(df, aes(x = game_minute)) +
  # Filled area (green when ahead of pace, red when behind)
  geom_ribbon(aes(ymin = pace_line, ymax = cumulative_points, fill = ahead_of_pace),
              alpha = 0.3) +
  
  # Points line
  geom_line(aes(y = cumulative_points, linetype = "Actual Points"), 
            color = "darkgreen", linewidth = 1.5) +
  
  # Pace line (straight line from 0 to ceil(prop_line))
  geom_line(aes(y = pace_line, linetype = "Pace Line"),
            color = "black", linewidth = 1.2) +
  
  # Prop line (for reference)
  geom_hline(aes(yintercept = {prop_line}, linetype = "Prop Line"), 
             color = "gray50", linewidth = 1.0) +
  
  # Annotations
  annotate("text", x = 45, y = {prop_line_ceil} + 2, 
           label = paste0("Pace: {prop_line_ceil}"), 
           color = "black", size = 4, fontface = "bold") +
  annotate("text", x = 45, y = {prop_line} - 2, 
           label = paste0("Prop: {prop_line}"), 
           color = "gray50", size = 4, fontface = "bold") +
  
  # Styling
  scale_fill_manual(values = c("TRUE" = "green", "FALSE" = "red"),
                    labels = c("TRUE" = "Ahead of pace", "FALSE" = "Behind pace"),
                    name = "") +
  scale_linetype_manual(values = c("Actual Points" = "solid",
                                   "Pace Line" = "solid",
                                   "Prop Line" = "dashed"),
                        name = "") +
  scale_x_continuous(limits = c(0, 48), breaks = seq(0, 48, 12), expand = c(0, 0)) +
  labs(subtitle = paste0("Final: {final_points} pts ({result_label})"),
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

# Add logos and headshot on top
grid.raster(readPNG("{away_logo_path}"), x = 0.12, y = 0.97, width = 0.08, height = 0.08, just = c("center", "top"))
grid.raster(readPNG("{home_logo_path}"), x = 0.88, y = 0.97, width = 0.08, height = 0.08, just = c("center", "top"))
grid.raster(readPNG("{player_headshot_path}"), x = 0.5, y = 0.97, width = 0.09, height = 0.09, just = c("center", "top"))

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
        
        # Clean up temp files
        os.unlink(temp_csv.name)
        os.unlink(away_logo_path)
        os.unlink(home_logo_path)
        os.unlink(player_headshot_path)
        
        return str(plot_file)
        
    except Exception as e:
        print(f"   ❌ Error running R: {e}")
        return None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_game(player_name, game_id, n_sims, use_consensus):
    """Process a single game."""
    print(f"\n{'='*80}")
    print(f"Game: {game_id}")
    print(f"{'='*80}")
    
    # Load player profile
    player_profile = load_player_profile(player_name)
    
    # Load play-by-play data
    print(f"📥 Loading play-by-play data...")
    pbp_df, metadata = load_play_by_play(game_id, player_name)
    
    away_team = metadata['away_team']
    home_team = metadata['home_team']
    game_date = metadata['game_date']
    commence_time_et = metadata['commence_time_et']
    
    print(f"   ✅ {away_team} @ {home_team}")
    if commence_time_et:
        print(f"   🕐 Tipoff: {commence_time_et.strftime('%Y-%m-%d %I:%M %p %Z')}")
    
    # Get prop line
    if use_consensus:
        prop_line = get_consensus_prop_line(player_name, game_date)
        if prop_line:
            print(f"   📊 Consensus prop line: {prop_line}")
        else:
            print(f"   ⚠️  No consensus prop line found, using average: {player_profile['avg_points_per_game']:.1f}")
            prop_line = player_profile['avg_points_per_game']
    else:
        prop_line = player_profile['avg_points_per_game']
        print(f"   📊 Using player average as prop line: {prop_line:.1f}")
    
    # Find Vegas adjustment (one-time, at game start)
    print(f"   🎲 Calibrating Vegas adjustment...")
    vegas_adjustment = find_vegas_adjustment(player_profile, prop_line, n_simulations=n_sims)
    print(f"   ✅ Vegas adjustment: {vegas_adjustment:.4f}")
    
    # Run Monte Carlo for each play
    print(f"   🎲 Running Monte Carlo simulation ({n_sims:,} iterations per play)...")
    
    results = []
    for idx, row in pbp_df.iterrows():
        game_minute = row['game_minute']
        current_points = row['cumulative_points']
        
        prob_over = monte_carlo_simulate_bet(
            player_profile,
            game_minute,
            current_points,
            prop_line,
            n_simulations=n_sims,
            vegas_adjustment=vegas_adjustment,
            debug=False
        )
        
        results.append({
            'game_minute': game_minute,
            'quarter': row['quarter'],
            'cumulative_points': current_points,
            'prob_over': prob_over,
        })
    
    results_df = pd.DataFrame(results)
    
    # Determine result
    final_points = int(pbp_df.iloc[-1]['cumulative_points'])
    result = "HIT" if final_points > prop_line else "MISS"
    
    # Save CSV
    player_name_clean = player_name.replace(" ", "_")
    csv_file = PLOT_DIR / f"monte_carlo_pbp_{player_name_clean}_{game_id}_{game_date}.csv"
    
    save_df = results_df.copy()
    save_df.insert(0, 'player_name', player_name)
    save_df.insert(1, 'game_id', game_id)
    save_df.insert(2, 'game_date', game_date)
    save_df.insert(3, 'prop_line', prop_line)
    save_df['final_points'] = final_points
    save_df['result'] = result
    
    save_df.to_csv(csv_file, index=False, float_format='%.4f')
    
    # Create plot
    print(f"   📊 Generating plot...")
    plot_file = create_ggplot(
        results_df,
        prop_line,
        player_name,
        player_profile['player_id'],
        game_id,
        game_date,
        away_team,
        home_team,
        final_points,
        result
    )
    
    if plot_file:
        print(f"   💾 Plot saved: {plot_file}")
    
    print(f"   💾 CSV saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Result: {final_points} pts ({result})")
    print(f"   Starting prob: {results_df.iloc[0]['prob_over']:.1%}")
    print(f"   Final prob: {results_df.iloc[-1]['prob_over']:.1%}")
    
    return {
        'game_id': game_id,
        'game_date': game_date,
        'final_points': final_points,
        'prop_line': prop_line,
        'result': result,
        'starting_prob': results_df.iloc[0]['prob_over'],
        'final_prob': results_df.iloc[-1]['prob_over'],
        'num_plays': len(results_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulation for NBA player props")
    parser.add_argument("--player-name", type=str, required=True, help="Player name (e.g., 'Luka Doncic')")
    parser.add_argument("--game-id", type=str, required=True, help="Game ID or 'all'")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of Monte Carlo simulations per play")
    parser.add_argument("--use-consensus", action="store_true", help="Use consensus prop lines from S3 data")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"MONTE CARLO SIMULATION - {args.player_name}")
    print("=" * 80)
    print()
    
    # Load player profile to get games
    player_profile = load_player_profile(args.player_name)
    
    # Get games
    if args.game_id == "all":
        con = duckdb.connect()
        games_df = con.execute(f"""
            SELECT DISTINCT game_id, game_date
            FROM '{MINUTE_BY_MINUTE}'
            WHERE player_name = '{args.player_name}'
            ORDER BY game_date DESC
        """).df()
        con.close()
        
        game_ids = games_df['game_id'].tolist()
        print(f"📊 Found {len(game_ids)} games for {args.player_name}")
    else:
        game_ids = [args.game_id]
    
    # Process each game
    summaries = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"\n[{i}/{len(game_ids)}]")
        
        try:
            summary = process_game(args.player_name, game_id, args.n_sims, args.use_consensus)
            summaries.append(summary)
        except Exception as e:
            print(f"   ❌ Error processing game {game_id}: {e}")
            continue
    
    # Save summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_file = PLOT_DIR / "monte_carlo_summary.csv"
        summary_df.to_csv(summary_file, index=False, float_format='%.3f')
        print(f"\n💾 Summary saved: {summary_file}")
    
    print(f"\n{'='*80}")
    print("✅ ALL GAMES COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
