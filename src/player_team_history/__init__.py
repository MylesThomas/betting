"""
Player team history - track player team assignments over time.

This package provides infrastructure to correctly join player props data
with team info using game dates, accounting for trades throughout the season.

Main Functions:
===============
- add_team_from_history(): Add team column to props DataFrame (MAIN FUNCTION)
- get_player_team_at_date(): Get a player's team on a specific date
- load_team_history(): Load the full history DataFrame
- build_team_history(): Build/rebuild the history from game logs

Quick Start:
============
    from src.player_team_history import add_team_from_history
    
    # Add team column based on game dates
    props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')

Build History:
==============
    python -m src.player_team_history.builder
"""

from .utils import (
    load_team_history,
    get_player_team_at_date,
    add_team_from_history,
    get_team_history_for_player
)

from .builder import build_team_history

__all__ = [
    'load_team_history',
    'get_player_team_at_date',
    'add_team_from_history',
    'get_team_history_for_player',
    'build_team_history'
]
