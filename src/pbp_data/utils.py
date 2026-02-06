"""
Utility functions for PBP data collection.
"""

import json
import requests
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import (
    ESPN_SCOREBOARD_URL,
    ESPN_SUMMARY_URL,
    PROGRESS_DIR,
)


def load_season_dates(config_path: Path, season: str = "2025-26") -> dict:
    """Load season start/end dates from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['nba'][season]


def date_range(start_date: str, end_date: str):
    """
    Generate dates between start and end (inclusive).
    
    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
    
    Yields:
        datetime.date objects
    """
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def get_games_on_date(date) -> list:
    """
    Get all games on a specific date from ESPN API.
    
    Args:
        date: datetime.date object
    
    Returns:
        List of game dictionaries with id, home_team, away_team
    """
    date_str = date.strftime('%Y%m%d')
    url = f"{ESPN_SCOREBOARD_URL}?dates={date_str}"
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    games = []
    
    for event in data.get('events', []):
        competition = event['competitions'][0]
        competitors = competition['competitors']
        
        games.append({
            'game_id': event['id'],
            'home_team': competitors[0]['team']['displayName'],
            'away_team': competitors[1]['team']['displayName'],
            'date': date.isoformat()
        })
    
    return games


def get_play_by_play(game_id: str) -> Optional[dict]:
    """
    Get play-by-play data for a specific game from ESPN API.
    
    Args:
        game_id: ESPN game ID
    
    Returns:
        Full game data dictionary or None if failed
    """
    url = f"{ESPN_SUMMARY_URL}?event={game_id}"
    
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    
    data = response.json()
    
    if 'plays' not in data:
        return None
    
    return data


def load_progress(filename: str) -> set:
    """
    Load completed items from progress file.
    
    Args:
        filename: Progress file name (e.g., 'game_ids_progress.json')
    
    Returns:
        Set of completed item IDs
    """
    progress_file = PROGRESS_DIR / filename
    
    if not progress_file.exists():
        return set()
    
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    return set(data.get('completed', []))


def save_progress(filename: str, completed: set):
    """
    Save completed items to progress file.
    
    Args:
        filename: Progress file name
        completed: Set of completed item IDs
    """
    progress_file = PROGRESS_DIR / filename
    
    data = {
        'completed': list(completed),
        'last_updated': datetime.now().isoformat()
    }
    
    with open(progress_file, 'w') as f:
        json.dump(data, f, indent=2)


def add_to_progress(filename: str, item_id: str):
    """
    Add single item to progress file.
    
    Args:
        filename: Progress file name
        item_id: Item to mark as completed
    """
    completed = load_progress(filename)
    completed.add(item_id)
    save_progress(filename, completed)
