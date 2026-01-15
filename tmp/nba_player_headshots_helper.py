"""
NBA Player Headshots Helper

Purpose:
Shows how to join NBA players with their headshots for R gt visualizations.
This is a reference file for future NBA player visualizations (MVP odds, player props, etc.).

Headshot Sources:
1. NBA CDN (PRIMARY - 100% success rate)
   - URL: https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png
   - Resolution: 1040x760 (highest quality, 790K pixels)
   - Always available for all active players

2. ESPN (BACKUP - varies)
   - URL: https://a.espncdn.com/i/headshots/nba/players/full/{player_id}.png
   - Quality varies

Best Practices (from viz_config.yaml):
- Download at FULL RESOLUTION (1040x760 for NBA)
- Convert directly to base64 without resizing in Python
- Let R/gtExtras handle scaling for maximum sharpness
- Smaller display size (20-25px) = sharper appearance
"""

import pandas as pd
from pathlib import Path
import sys

# Get repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))


def get_nba_headshot_url(player_id, source='nba_cdn'):
    """
    Get NBA player headshot URL
    
    Args:
        player_id: NBA player ID (integer)
        source: 'nba_cdn' (primary) or 'espn' (backup)
    
    Returns:
        URL string or None if player_id is invalid
    """
    if pd.isna(player_id):
        return None
    
    player_id = int(player_id)
    
    if source == 'nba_cdn':
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    elif source == 'espn':
        return f"https://a.espncdn.com/i/headshots/nba/players/full/{player_id}.png"
    else:
        raise ValueError(f"Unknown source: {source}")


def add_headshots_to_df(df, player_id_column='player_id', source='nba_cdn'):
    """
    Add headshot URLs to a DataFrame
    
    Args:
        df: DataFrame with player data
        player_id_column: Name of column containing NBA player IDs
        source: 'nba_cdn' or 'espn'
    
    Returns:
        DataFrame with new 'headshot_url' column
    """
    df = df.copy()
    df['headshot_url'] = df[player_id_column].apply(
        lambda x: get_nba_headshot_url(x, source=source)
    )
    return df


def example_usage():
    """
    Example: Add headshots to a player stats DataFrame
    """
    # Example player data (from paint scoring analysis)
    players_data = {
        'player': ['Payton Pritchard', 'Luka Doncic', 'Anthony Edwards'],
        'player_id': [1630202, 1629029, 1630162],
        'paint_fg_pct': [67.1, 72.4, 68.4],
        'height': ['6\'1"', '6\'7"', '6\'4"']
    }
    
    df = pd.DataFrame(players_data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*80 + "\n")
    
    # Add headshot URLs
    df = add_headshots_to_df(df, player_id_column='player_id', source='nba_cdn')
    
    print("DataFrame with Headshots:")
    print(df)
    print("\n" + "="*80 + "\n")
    
    print("Headshot URLs:")
    for _, row in df.iterrows():
        print(f"  {row['player']}: {row['headshot_url']}")


def get_player_id_from_name(player_name):
    """
    Get NBA player ID from name using nba_api
    
    Args:
        player_name: Full player name (e.g., "Payton Pritchard")
    
    Returns:
        Player ID (integer) or None if not found
    """
    from nba_api.stats.static import players
    
    all_players = players.find_players_by_full_name(player_name)
    if not all_players:
        print(f"Player '{player_name}' not found")
        return None
    
    return all_players[0]['id']


def bulk_get_player_ids(player_names):
    """
    Get player IDs for a list of player names
    
    Args:
        player_names: List of full player names
    
    Returns:
        DataFrame with player names and IDs
    """
    results = []
    
    for name in player_names:
        player_id = get_player_id_from_name(name)
        results.append({
            'player': name,
            'player_id': player_id,
            'headshot_url': get_nba_headshot_url(player_id) if player_id else None
        })
    
    return pd.DataFrame(results)


# =============================================================================
# R VISUALIZATION INTEGRATION
# =============================================================================

def r_gt_table_with_headshots_example():
    """
    Example R code for gt table with headshots
    
    This shows the R code pattern used in viz_payton_pritchard_paint_scoring_r.py
    """
    r_code_template = '''
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    # Assuming you have 'player_data' DataFrame in R with 'headshot_url' column
    
    table <- player_data %>%
      gt() %>%
      
      # Add player headshots (key step!)
      gt_img_rows(columns = headshot_url, height = 25) %>%
      
      # Hide the headshot column header
      cols_label(headshot_url = "") %>%
      
      # Rest of table styling...
      tab_header(
        title = md("**Your Title Here**"),
        subtitle = md("Your subtitle")
      ) %>%
      
      # Column widths
      cols_width(
        headshot_url ~ px(45),  # Fixed width for headshot column
        player ~ px(180),        # Adjust other columns as needed
        # ... other columns
      )
    
    # Save as PNG
    gtsave(table, "output.png", vwidth = 1400, vheight = 1800)
    '''
    
    print("R Code Template for gt tables with headshots:")
    print("="*80)
    print(r_code_template)
    print("="*80)


# =============================================================================
# COMMON USE CASES
# =============================================================================

def mvp_odds_example():
    """Example: MVP odds table with headshots"""
    print("\nUSE CASE 1: MVP Odds Table")
    print("="*80)
    print("File: analysis/viz_nba_mvp_gt.py")
    print("Pattern:")
    print("  1. Load MVP odds data with player names")
    print("  2. Get player IDs using nba_api")
    print("  3. Add headshot URLs")
    print("  4. Create R gt table with gt_img_rows()")
    print("="*80)


def player_props_example():
    """Example: Player props comparison with headshots"""
    print("\nUSE CASE 2: Player Props Comparison")
    print("="*80)
    print("File: analysis/analyze_payton_pritchard_paint_scoring.py")
    print("Pattern:")
    print("  1. Fetch shot chart data (includes PLAYER_ID)")
    print("  2. Calculate player stats")
    print("  3. Add headshot URLs using player_id column")
    print("  4. Create R gt table with headshots")
    print("="*80)


def paint_scoring_example():
    """Example: Paint scoring analysis (current file)"""
    print("\nUSE CASE 3: Paint Scoring Analysis (THIS FILE)")
    print("="*80)
    print("Files:")
    print("  - analysis/analyze_payton_pritchard_paint_scoring.py")
    print("  - analysis/viz_payton_pritchard_paint_scoring_r.py")
    print("\nSteps:")
    print("  1. Fetch shot charts for multiple guards")
    print("  2. Analyze paint scoring (shots ≤6 feet)")
    print("  3. Save with player_id column")
    print("  4. Load in viz script, add headshot URLs")
    print("  5. Create R gt table with headshots + color gradients")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("NBA PLAYER HEADSHOTS HELPER")
    print("="*80 + "\n")
    
    # Run example
    example_usage()
    
    print("\n" + "="*80)
    print("COMMON USE CASES")
    print("="*80)
    
    mvp_odds_example()
    player_props_example()
    paint_scoring_example()
    
    print("\n" + "="*80)
    print("R INTEGRATION")
    print("="*80 + "\n")
    
    r_gt_table_with_headshots_example()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("\n1. Always use NBA CDN for headshots (1040x760 resolution)")
    print("2. Player IDs come from nba_api or shot chart data")
    print("3. Add 'headshot_url' column to DataFrame before passing to R")
    print("4. Use gt_img_rows(columns = headshot_url, height = 25) in R")
    print("5. Hide headshot column header with cols_label(headshot_url = \"\")")
    print("\n" + "="*80)

