"""
Utility to fetch NBA schedule and map teams to GAME_ID for live scoring.
"""

import logging
import ssl
import urllib3

import pandas as pd
import requests

# ============================================================================
# SSL FIX FOR MACOS
# ============================================================================
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests with timeout and no verify
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 10)
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

from nba_api.stats.endpoints import scoreboardv2


def get_schedule_for_date(date_str: str) -> pd.DataFrame:
    """
    Fetch the NBA schedule for a given date (YYYY-MM-DD).
    Returns a DataFrame with GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, etc.
    """
    logging.info(f"Fetching NBA schedule for {date_str}...")
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = sb.get_data_frames()[0]
        return games
    except Exception as e:
        logging.error(f"Failed to fetch schedule for {date_str}: {e}")
        return pd.DataFrame()


def get_team_id_mapping() -> dict:
    """
    Get mapping of team abbreviation to team ID.
    Using nba_api static data.
    """
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    return {team['abbreviation']: team['id'] for team in nba_teams}


def get_team_name_to_abbr_mapping() -> dict:
    """
    Get mapping of full team name to abbreviation.
    """
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    return {team['full_name']: team['abbreviation'] for team in nba_teams}


def resolve_game_id(home_team_name: str, away_team_name: str, date_str: str, schedule_df: pd.DataFrame = None) -> str:
    """
    Resolve GAME_ID from team names and date.
    """
    if schedule_df is None or schedule_df.empty:
        schedule_df = get_schedule_for_date(date_str)
        if schedule_df.empty:
            return None

    name_to_abbr = get_team_name_to_abbr_mapping()
    team_to_id = get_team_id_mapping()

    # The Odds API names might need normalization (e.g., "LA Clippers" -> "Los Angeles Clippers")
    from src.player_team_history.team_normalization import normalize_team_name_from_odds_api
    
    home_canonical = normalize_team_name_from_odds_api(home_team_name)
    away_canonical = normalize_team_name_from_odds_api(away_team_name)

    home_abbr = name_to_abbr.get(home_canonical)
    away_abbr = name_to_abbr.get(away_canonical)

    if not home_abbr or not away_abbr:
        logging.warning(f"Could not resolve abbreviations for {home_team_name} or {away_team_name}")
        return None

    home_id = team_to_id.get(home_abbr)
    away_id = team_to_id.get(away_abbr)

    # Find the game in the schedule
    # schedule_df has HOME_TEAM_ID and VISITOR_TEAM_ID
    match = schedule_df[
        (schedule_df['HOME_TEAM_ID'] == home_id) | 
        (schedule_df['VISITOR_TEAM_ID'] == home_id) |
        (schedule_df['HOME_TEAM_ID'] == away_id) |
        (schedule_df['VISITOR_TEAM_ID'] == away_id)
    ]

    if not match.empty:
        # Just take the first match since teams play once a day
        return match.iloc[0]['GAME_ID']
    
    return None
