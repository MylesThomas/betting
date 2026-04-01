"""
Shared parser for The Odds API player props.
Extracted from scripts/fetch_nba_player_props.py to keep ingestion thin and reusable.
"""

from typing import Dict, List, Any


def parse_player_props(odds_data: Dict[str, Any], target_market: str = None) -> List[Dict[str, Any]]:
    """
    Parse player props from odds data.
    
    Args:
        odds_data: JSON response from The Odds API (either single event or list of events).
                   If single event, expects {'data': {...}} or just the event dict.
                   If list of events, expects a list of event dicts.
        target_market: Optional market key to filter by (e.g., 'player_rebounds').
        
    Returns:
        List of dictionaries, one per player/market/line/bookmaker.
    """
    props_list = []
    
    # Handle both single event (with or without 'data' wrapper) and list of events
    events = []
    if isinstance(odds_data, list):
        events = odds_data
    elif isinstance(odds_data, dict):
        if 'data' in odds_data:
            events = odds_data['data']
            if isinstance(events, dict):
                events = [events]
        else:
            events = [odds_data]
            
    for event_data in events:
        away_team = event_data.get('away_team')
        home_team = event_data.get('home_team')
        game_time = event_data.get('commence_time')
        event_id = event_data.get('id')
        
        for bookmaker in event_data.get('bookmakers', []):
            bookmaker_name = bookmaker['key']
            bookmaker_last_update = bookmaker.get('last_update')
            
            for market in bookmaker.get('markets', []):
                market_key = market['key']
                
                if target_market and market_key != target_market:
                    continue
                    
                market_last_update = market.get('last_update')
                
                # Group outcomes by player, market, and line
                player_line_props = {}
                for outcome in market.get('outcomes', []):
                    player = outcome.get('description', 'Unknown')
                    line = outcome.get('point')
                    odds = outcome.get('price')
                    bet_type = outcome.get('name')
                    
                    # Key by player, market, and line
                    key = (player, market_key, line)
                    
                    if key not in player_line_props:
                        player_line_props[key] = {
                            'odds_api_event_id': event_id,
                            'player': player,
                            'away_team': away_team,
                            'home_team': home_team,
                            'game_time': game_time,
                            'market': market_key,
                            'prop_line': line,
                            'bookmaker': bookmaker_name,
                            'bookmaker_last_update': bookmaker_last_update,
                            'market_last_update': market_last_update
                        }
                    
                    if bet_type == 'Over':
                        player_line_props[key]['over_odds'] = odds
                    elif bet_type == 'Under':
                        player_line_props[key]['under_odds'] = odds
                
                props_list.extend(player_line_props.values())
        
    return props_list
