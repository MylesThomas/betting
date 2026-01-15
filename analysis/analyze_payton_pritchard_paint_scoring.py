"""
Analyze Payton Pritchard's Paint Scoring vs Other Guards

Context:
User wants to create a post about how good Payton Pritchard is at scoring 
in the paint for a small guard (6'1"). This script:

1. Fetches shot chart data for Pritchard and comparable guards
2. Analyzes paint scoring (0-6 feet) efficiency
3. Compares to other guards by height/position
4. Creates a visualization showing how elite he is for his size

Key Questions:
- What % of his shots come in the paint?
- What's his FG% in the paint?
- How does he compare to other sub-6'3" guards?
- What types of shots is he taking in the paint? (layups, floaters, etc.)

Output:
- CSV with comparison data
- Visualization (gt table) ready for social media post
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import sys
import os
import ssl
import urllib3
import requests

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Fix SSL issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players


# =============================================================================
# CONFIGURATION
# =============================================================================

CURRENT_SEASON = "2025-26"
PAINT_DISTANCE = 6  # feet
MIN_PAINT_ATTEMPTS = 25  # Minimum paint attempts to qualify

# Guards to compare (mix of heights) - expanded list
# Format: (name, height, note)
COMPARISON_GUARDS = [
    # Small guards (≤6'2")
    ("Payton Pritchard", "6'1\"", "Target Player"),
    ("Fred VanVleet", "6'0\"", "Small Guard All-Star"),
    ("Chris Paul", "6'0\"", "Small Guard HOF"),
    ("Tyrese Maxey", "6'2\"", "Explosive Small Guard"),
    ("Jalen Brunson", "6'2\"", "Small Guard Star"),
    ("Darius Garland", "6'1\"", "Small Guard Star"),
    ("Trae Young", "6'1\"", "Small Guard Star"),
    ("Damian Lillard", "6'2\"", "Star Guard"),
    ("Stephen Curry", "6'2\"", "All-Time Great"),
    ("Jamal Murray", "6'3\"", "Star Guard"),
    
    # Athletic guards (6'3"-6'4")
    ("De'Aaron Fox", "6'3\"", "Athletic Guard"),
    ("Immanuel Quickley", "6'3\"", "Athletic Guard"),
    ("Donovan Mitchell", "6'3\"", "Star Guard"),
    ("Ja Morant", "6'3\"", "Elite Finisher"),
    ("Jordan Poole", "6'4\"", "Athletic Guard"),
    ("Anfernee Simons", "6'3\"", "Athletic Guard"),
    ("Collin Sexton", "6'1\"", "Athletic Guard"),
    ("Tyler Herro", "6'5\"", "Scoring Guard"),
    ("CJ McCollum", "6'3\"", "Scoring Guard"),
    ("Devin Booker", "6'5\"", "Star Guard"),
    
    # Bigger guards (6'5"+) for context  
    ("Luka Doncic", "6'7\"", "Jumbo Guard"),
    ("LaMelo Ball", "6'7\"", "Jumbo Guard"),
    ("Anthony Edwards", "6'4\"", "Big Athletic Guard"),
    ("Shai Gilgeous-Alexander", "6'6\"", "Star Guard"),
    ("Cade Cunningham", "6'6\"", "Jumbo Guard"),
]

# Data paths
SHOT_CHART_DIR = repo_root / 'data/01_input/nba_api/shot_charts' / CURRENT_SEASON.replace('-', '_')
OUTPUT_DIR = repo_root / 'data/04_output/nba'
VIZ_DIR = repo_root / 'content/viz/nba'


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_player_id(player_name):
    """Get NBA player ID from name"""
    all_players = players.find_players_by_full_name(player_name)
    if not all_players:
        print(f"   ❌ Player '{player_name}' not found")
        return None
    return all_players[0]['id']


def fetch_player_shot_chart(player_name, season=CURRENT_SEASON):
    """
    Fetch shot chart data for a player
    
    Returns:
        DataFrame with shot data or None
    """
    print(f"   🏀 Fetching {player_name}...")
    
    player_id = get_player_id(player_name)
    if not player_id:
        return None
    
    try:
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star='Regular Season',
            context_measure_simple='FGA'
        )
        
        shots_df = shot_chart.get_data_frames()[0]
        
        if shots_df.empty:
            print(f"   ⚠️  No shot data for {player_name}")
            return None
        
        print(f"      ✅ {len(shots_df)} shots")
        return shots_df
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def load_or_fetch_shot_data(player_name, season=CURRENT_SEASON):
    """
    Load shot data from CSV if exists, otherwise fetch from API
    """
    # Try to load from file first
    player_file = SHOT_CHART_DIR / f"{player_name.replace(' ', '_')}_{get_player_id(player_name)}.csv"
    
    if player_file.exists():
        print(f"   📂 Loading from file: {player_name}")
        return pd.read_csv(player_file)
    
    # Fetch from API
    shots_df = fetch_player_shot_chart(player_name, season)
    
    if shots_df is not None:
        # Save for future use
        SHOT_CHART_DIR.mkdir(parents=True, exist_ok=True)
        shots_df.to_csv(player_file, index=False)
        print(f"      💾 Saved to: {player_file.name}")
    
    # Rate limit
    time.sleep(0.6)
    
    return shots_df


# =============================================================================
# PAINT SCORING ANALYSIS
# =============================================================================

def analyze_paint_scoring(shots_df, player_name, max_distance=PAINT_DISTANCE):
    """
    Analyze paint scoring metrics for a player
    
    Returns:
        Dict with paint scoring stats
    """
    if shots_df is None or shots_df.empty:
        return None
    
    # Get player ID from shots data
    player_id = shots_df['PLAYER_ID'].iloc[0] if 'PLAYER_ID' in shots_df.columns else None
    
    # Overall stats
    total_fga = len(shots_df)
    total_fgm = shots_df['SHOT_MADE_FLAG'].sum()
    total_fg_pct = (total_fgm / total_fga * 100) if total_fga > 0 else 0
    
    # Paint shots (0-X feet)
    paint_shots = shots_df[shots_df['SHOT_DISTANCE'] <= max_distance].copy()
    
    if paint_shots.empty:
        return None
    
    paint_fga = len(paint_shots)
    paint_fgm = paint_shots['SHOT_MADE_FLAG'].sum()
    paint_fg_pct = (paint_fgm / paint_fga * 100) if paint_fga > 0 else 0
    paint_rate = (paint_fga / total_fga * 100) if total_fga > 0 else 0
    
    # Shot type breakdown in paint
    shot_types = paint_shots['ACTION_TYPE'].value_counts().head(3).to_dict()
    
    # Points per game equivalent (assuming 82 games)
    games_played = len(paint_shots['GAME_ID'].unique())
    paint_ppg = (paint_fgm * 2) / games_played if games_played > 0 else 0
    
    return {
        'player': player_name,
        'player_id': player_id,
        'total_fga': total_fga,
        'total_fg_pct': round(total_fg_pct, 1),
        'paint_fga': paint_fga,
        'paint_fgm': int(paint_fgm),
        'paint_fg_pct': round(paint_fg_pct, 1),
        'paint_rate': round(paint_rate, 1),
        'paint_ppg': round(paint_ppg, 1),
        'games': games_played,
        'top_shot_types': shot_types
    }


def compare_guards_paint_scoring(guard_list):
    """
    Compare paint scoring for multiple guards
    
    Args:
        guard_list: List of (name, height, note) tuples
    
    Returns:
        DataFrame with comparison
    """
    print("\n" + "="*80)
    print("FETCHING GUARD SHOT DATA")
    print("="*80 + "\n")
    
    results = []
    
    for player_name, height, note in guard_list:
        shots_df = load_or_fetch_shot_data(player_name)
        
        if shots_df is not None:
            analysis = analyze_paint_scoring(shots_df, player_name)
            
            if analysis:
                analysis['height'] = height
                analysis['note'] = note
                results.append(analysis)
        
        time.sleep(0.1)  # Small delay between players
    
    if not results:
        print("❌ No data collected")
        return None
    
    df = pd.DataFrame(results)
    
    # Filter out players with insufficient attempts
    print(f"\n📊 Filtering: Keeping players with ≥{MIN_PAINT_ATTEMPTS} paint attempts")
    df_filtered = df[df['paint_fga'] >= MIN_PAINT_ATTEMPTS].copy()
    print(f"   Players before filter: {len(df)}")
    print(f"   Players after filter: {len(df_filtered)}")
    
    # Sort by paint FG% (descending - best at top)
    df_filtered = df_filtered.sort_values('paint_fg_pct', ascending=False).reset_index(drop=True)
    
    # Add rank
    df_filtered['rank'] = range(1, len(df_filtered) + 1)
    
    df = df_filtered
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_paint_scoring_viz(df):
    """
    Create gt table visualization for paint scoring comparison
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        import subprocess
        import platform
        
    except ImportError as e:
        print(f"❌ Error: rpy2 not installed")
        print(f"   Install: pip install rpy2")
        return
    
    # Prepare display dataframe
    df_viz = df.copy()
    
    # Create display columns
    df_viz['paint_stats'] = df_viz.apply(
        lambda row: f"{row['paint_fgm']}/{row['paint_fga']} ({row['paint_fg_pct']}%)",
        axis=1
    )
    
    df_viz['overall_fg'] = df_viz['total_fg_pct'].apply(lambda x: f"{x}%")
    df_viz['paint_rate_display'] = df_viz['paint_rate'].apply(lambda x: f"{x}%")
    
    # Highlight Pritchard
    df_viz['is_pritchard'] = df_viz['player'].apply(
        lambda x: '⭐' if x == 'Payton Pritchard' else ''
    )
    
    # Select and reorder columns
    table_df = df_viz[[
        'rank', 'player', 'height', 'games',
        'paint_fgm', 'paint_fga', 'paint_fg_pct', 'paint_rate', 'paint_ppg',
        'total_fg_pct'
    ]].copy()
    
    # Rename for display
    table_df.columns = [
        'Rank', 'Player', 'Height', 'Games',
        'Paint Makes', 'Paint Attempts', 'Paint FG%', 'Paint Rate', 'Paint PPG',
        'Overall FG%'
    ]
    
    # Convert to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(table_df)
    
    ro.globalenv['paint_data'] = r_df
    
    # Output path
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VIZ_DIR / f'payton_pritchard_paint_scoring_{CURRENT_SEASON.replace("-", "_")}.png'
    
    print(f"\n   💾 Output: {output_path.name}\n")
    
    # R code for visualization
    r_code = f'''
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    # Create table
    table <- paint_data %>%
      gt() %>%
      
      # Title and subtitle
      tab_header(
        title = md("**Payton Pritchard is ELITE at Scoring in the Paint for a 6'1\" Guard**"),
        subtitle = md("Paint shots (≤6 feet) | {CURRENT_SEASON} NBA Season")
      ) %>%
      
      # Column alignment
      cols_align(align = "center", columns = everything()) %>%
      cols_align(align = "left", columns = c(Player)) %>%
      
      # Color gradients
      data_color(
        columns = `Paint FG%`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(40, 80),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Paint Rate`,
        method = "numeric",
        palette = c("#e8e8e8", "#cce5ff", "#66b3ff", "#0066cc"),
        domain = c(0, 50),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Paint PPG`,
        method = "numeric",
        palette = c("#e8e8e8", "#ffe6cc", "#ffcc80", "#ff9933"),
        domain = c(0, 10),
        na_color = "#e8e8e8"
      ) %>%
      
      # Column widths
      cols_width(
        Rank ~ px(50),
        Player ~ px(180),
        Height ~ px(70),
        Games ~ px(70),
        `Paint Makes` ~ px(90),
        `Paint Attempts` ~ px(100),
        `Paint FG%` ~ px(90),
        `Paint Rate` ~ px(90),
        `Paint PPG` ~ px(90),
        `Overall FG%` ~ px(90)
      ) %>%
      
      # Style headers
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px(12)),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Bold player names
      tab_style(
        style = cell_text(weight = "600"),
        locations = cells_body(columns = c(Player))
      ) %>%
      
      # Highlight Pritchard row
      tab_style(
        style = list(
          cell_fill(color = "#ffffcc"),
          cell_text(weight = "bold")
        ),
        locations = cells_body(rows = Player == "Payton Pritchard")
      ) %>%
      
      # Zebra striping
      opt_row_striping(row_striping = TRUE) %>%
      
      # Table options
      tab_options(
        table.font.names = "Arial",
        table.font.size = px(11),
        heading.title.font.size = px(20),
        heading.subtitle.font.size = px(14),
        heading.padding = px(8),
        column_labels.padding = px(4),
        data_row.padding = px(3),
        table.border.bottom.width = px(2),
        table.border.bottom.color = "#2c3e50",
        column_labels.border.bottom.width = px(2),
        column_labels.border.bottom.color = "#2c3e50",
        table.background.color = "#f8f9fa",
        row.striping.background_color = "#f0f0f0"
      ) %>%
      
      # Footer
      tab_source_note(
        source_note = md("**Paint = shots within 6 feet | Data:** NBA API | {datetime.now().strftime('%B %d, %Y')}")
      )
    
    # Save
    gtsave(table, "{str(output_path)}", vwidth = 1400, vheight = 1600)
    '''
    
    print("   🔧 Executing R code...\n")
    
    try:
        ro.r(r_code)
        print(f"   ✅ Visualization saved!\n")
        print(f"   🖼️  {output_path}\n")
        
        # Auto-open
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(output_path)])
                print("   📂 Opening PNG...\n")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error creating visualization:")
        print(f"   {e}")
        print("\n💡 Install R packages:")
        print("   R -e 'install.packages(c(\"gt\", \"gtExtras\", \"dplyr\", \"webshot2\"))'")


# =============================================================================
# MAIN
# =============================================================================

def print_insights(df):
    """
    Print key insights about Pritchard's paint scoring
    """
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR POST")
    print("="*80 + "\n")
    
    pritchard = df[df['player'] == 'Payton Pritchard'].iloc[0]
    
    print(f"📊 PAYTON PRITCHARD PAINT SCORING ({CURRENT_SEASON})")
    print(f"   Height: {pritchard['height']}")
    print(f"   Paint FG%: {pritchard['paint_fg_pct']}%")
    print(f"   Rank: #{pritchard['rank']} of {len(df)} guards analyzed")
    print(f"   Paint Rate: {pritchard['paint_rate']}% of all shots")
    print(f"   Paint PPG: {pritchard['paint_ppg']}")
    print(f"   Paint Makes: {pritchard['paint_fgm']}/{pritchard['paint_fga']}")
    
    # Compare to other small guards
    small_guards = df[df['height'].isin(['6\'0"', '6\'1"', '6\'2"'])]
    pp_rank_small = (small_guards['paint_fg_pct'] >= pritchard['paint_fg_pct']).sum()
    
    print(f"\n🎯 AMONG GUARDS 6'2\" AND UNDER:")
    print(f"   Rank: #{pp_rank_small} of {len(small_guards)}")
    
    # Top shot types
    print(f"\n🏀 TOP SHOT TYPES IN PAINT:")
    for i, (shot_type, count) in enumerate(pritchard['top_shot_types'].items(), 1):
        print(f"   {i}. {shot_type}: {count} attempts")
    
    # Comparison to league leaders
    top_3 = df.head(3)
    print(f"\n⭐ TOP 3 PAINT SCORERS:")
    for _, row in top_3.iterrows():
        print(f"   {row['rank']}. {row['player']} ({row['height']}): {row['paint_fg_pct']}%")
    
    print("\n" + "="*80)


def main(top_n=None):
    """
    Main analysis pipeline
    
    Args:
        top_n: Number of top players to keep (None = keep all)
    """
    
    print("\n" + "="*80)
    print("PAYTON PRITCHARD PAINT SCORING ANALYSIS")
    print("="*80)
    
    # Compare all guards
    df = compare_guards_paint_scoring(COMPARISON_GUARDS)
    
    if df is None:
        print("❌ Analysis failed - no data collected")
        return
    
    # Filter to top N if specified
    if top_n:
        print(f"\n📊 Filtering to top {top_n} players by paint FG%")
        df = df.head(top_n)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / f'payton_pritchard_paint_scoring_{CURRENT_SEASON.replace("-", "_")}.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved results: {output_csv}")
    
    # Print insights
    print_insights(df)
    
    # Note: Run viz_payton_pritchard_paint_scoring_r.py separately for visualization
    # (it has the updated base64 headshots implementation)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📊 Next step: Run visualization script")
    print("   python3 analysis/viz_payton_pritchard_paint_scoring_r.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Payton Pritchard paint scoring vs other guards')
    parser.add_argument('--top-n', type=int, default=None, 
                       help='Number of top players to keep (default: all qualifying players)')
    
    args = parser.parse_args()
    main(top_n=args.top_n)

