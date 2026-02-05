"""
Scrape Bill Simmons's NBA Trade Value Rankings from The Ringer.

Context:
- User wants the top 81 players from Bill Simmons's 2026 trade value rankings
- Extract player_name, group, team, position, age, height, weighted, drafted, salary, stats
- Output to CSV with salary as list of strings and stats as JSON hashmap
- URL: https://nbarankings.theringer.com/trade-value
- Updated Feb 4, 2026

The site is a Next.js app with data embedded in __NEXT_DATA__ script tag.
Structure:
  - props.pageProps.content.processedPlayers.tradeValue: list of player IDs in rank order
  - props.pageProps.content.processedPlayers.playerData: dict of player details by ID

Created: 2026-02-04
"""

import csv
import json
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# =============================================================================
# CONSTANTS
# =============================================================================

URL = "https://nbarankings.theringer.com/trade-value"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fetch_html(url):
    """Fetch HTML content from URL."""
    print(f"Fetching {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    print(f"✓ Retrieved {len(response.text)} characters")
    return response.text


def extract_next_data(html):
    """Extract __NEXT_DATA__ JSON from Next.js HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    script_tag = soup.find('script', id='__NEXT_DATA__', type='application/json')
    
    if not script_tag:
        raise ValueError("Could not find __NEXT_DATA__ script tag")
    
    return json.loads(script_tag.string)


def parse_player_data(next_data):
    """
    Parse player data from Next.js data structure.
    
    Returns list of dicts with player info in rank order (top 81 only).
    """
    content = next_data['props']['pageProps']['content']
    processed_players = content['processedPlayers']
    
    # Get ordered list of player IDs from trade value ranking
    trade_value_ids = processed_players['tradeValue']
    
    # Get full player data
    player_data_dict = processed_players['playerData']
    
    players = []
    
    for rank, player_id in enumerate(trade_value_ids, start=1):
        if player_id not in player_data_dict:
            print(f"⚠️  Warning: Player ID {player_id} not found in playerData")
            continue
        
        player_raw = player_data_dict[player_id]
        
        # Extract relevant fields
        player = {
            'rank': rank,
            'player_name': player_raw.get('title', ''),
            'group': player_raw.get('trade_value_group', ''),
            'team': player_raw.get('meta', {}).get('team', ''),
            'position': player_raw.get('position_label', ''),
            'age': player_raw.get('meta', {}).get('age', ''),
            'height': player_raw.get('meta', {}).get('height', ''),
            'weighted': player_raw.get('meta', {}).get('weight', ''),  # Assuming "weighted" means weight
            'drafted': player_raw.get('draft_info', ''),  # May not exist
        }
        
        # Extract salary data if available
        salary_data = player_raw.get('salary', [])
        if isinstance(salary_data, list):
            salary_list = []
            for sal_item in salary_data:
                if isinstance(sal_item, dict) and 'salary' in sal_item:
                    sal_info = sal_item['salary']
                    years = sal_info.get('years', '')
                    numbers = sal_info.get('numbers', '')
                    salary_list.append(f"{years} {numbers}".strip())
            player['salary'] = salary_list
        else:
            player['salary'] = []
        
        # Extract stats as hashmap
        stats_raw = player_raw.get('stats', {}).get('stat', {})
        stats = {}
        if isinstance(stats_raw, dict):
            for stat_key, stat_val in stats_raw.items():
                if isinstance(stat_val, dict):
                    # Store both value and detail
                    value = stat_val.get('value', '')
                    detail = stat_val.get('detail', '')
                    stats[stat_key] = {'value': value, 'detail': detail}
                else:
                    stats[stat_key] = stat_val
        
        player['stats'] = stats
        
        # Stop after top 81 (exclude honorable mentions and omissions)
        if rank <= 81 and player['group'] not in ['honorable', 'omissions', '']:
            players.append(player)
    
    # Filter to top 81 by checking order_trade-value field
    filtered_players = []
    for player_id in trade_value_ids:
        if player_id not in player_data_dict:
            continue
        
        player_raw = player_data_dict[player_id]
        order = player_raw.get('order_trade-value', 9999)
        
        # Only include players ranked 1-81
        if order <= 81:
            # Extract data
            player = {
                'rank': order,
                'player_name': player_raw.get('title', ''),
                'group': player_raw.get('trade_value_group', ''),
                'team': player_raw.get('meta', {}).get('team', ''),
                'position': player_raw.get('position_label', ''),
                'age': player_raw.get('meta', {}).get('age', ''),
                'height': player_raw.get('meta', {}).get('height', ''),
                'weighted': player_raw.get('meta', {}).get('weight', ''),
                'drafted': player_raw.get('draft_info', ''),
            }
            
            # Extract salary
            salary_data = player_raw.get('salary', [])
            salary_list = []
            if isinstance(salary_data, list):
                for sal_item in salary_data:
                    if isinstance(sal_item, dict) and 'salary' in sal_item:
                        sal_info = sal_item['salary']
                        years = sal_info.get('years', '')
                        numbers = sal_info.get('numbers', '')
                        salary_list.append(f"{years} {numbers}".strip())
            player['salary'] = salary_list
            
            # Extract stats
            stats_raw = player_raw.get('stats', {}).get('stat', {})
            stats = {}
            if isinstance(stats_raw, dict):
                for stat_key, stat_val in stats_raw.items():
                    if isinstance(stat_val, dict):
                        value = stat_val.get('value', '')
                        detail = stat_val.get('detail', '')
                        stats[stat_key] = {'value': value, 'detail': detail}
                    else:
                        stats[stat_key] = stat_val
            player['stats'] = stats
            
            filtered_players.append(player)
    
    # Sort by rank
    filtered_players.sort(key=lambda x: x['rank'])
    
    return filtered_players


def save_to_csv(players, output_path):
    """Save player data to CSV."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(players)} players to {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'rank', 'player_name', 'group', 'team', 'position', 
            'age', 'height', 'weighted', 'drafted', 
            'salary', 'stats'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for player in players:
            # Convert lists/dicts to JSON strings
            row = player.copy()
            row['salary'] = json.dumps(player['salary'])
            row['stats'] = json.dumps(player['stats'])
            writer.writerow(row)
    
    print(f"✓ Saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    output_path = '~/Downloads/tmp/nba_trade_value_rankings_2026.csv'
    debug_json_path = '~/Downloads/tmp/trade_value_data.json'
    
    print("=" * 80)
    print("NBA Trade Value Rankings Scraper")
    print("=" * 80)
    
    # Fetch HTML
    html = fetch_html(URL)
    
    # Extract Next.js data
    print("\nExtracting Next.js data...")
    next_data = extract_next_data(html)
    
    # Save full JSON for debugging
    debug_path = Path(debug_json_path).expanduser()
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, 'w', encoding='utf-8') as f:
        json.dump(next_data, f, indent=2)
    print(f"✓ Saved full JSON to {debug_path}")
    
    # Parse player data
    print("\nParsing player data...")
    players = parse_player_data(next_data)
    print(f"✓ Parsed {len(players)} players")
    
    # Show first few players
    print("\nFirst 3 players:")
    for player in players[:3]:
        print(f"  {player['rank']}. {player['player_name']} ({player['team']}) - Group: {player['group']}")
    
    # Save to CSV
    save_to_csv(players, output_path)
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
