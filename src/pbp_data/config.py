"""
Configuration for PBP data collection.
"""

import os
from pathlib import Path

# Cache directory
CACHE_DIR = Path.home() / "Downloads" / "tmp" / "player_points_monte_carlo"
GAME_IDS_DIR = CACHE_DIR / "game_ids"
PBP_DATA_DIR = CACHE_DIR / "pbp_data"
BOXSCORE_DATA_DIR = CACHE_DIR / "boxscore_data"
PROGRESS_DIR = CACHE_DIR / "progress"

# Output directory (final parquet files)
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"

# Ensure directories exist
GAME_IDS_DIR.mkdir(parents=True, exist_ok=True)
PBP_DATA_DIR.mkdir(parents=True, exist_ok=True)
BOXSCORE_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ESPN API endpoints
ESPN_SCOREBOARD_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# Rate limiting (seconds)
RATE_LIMIT_BETWEEN_DATES = 0.2
RATE_LIMIT_BETWEEN_GAMES = 0.5

# Season config
SEASON = "2025-26"
SEASON_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "season_dates.yaml"
