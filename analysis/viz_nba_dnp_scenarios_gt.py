"""
Create a FiveThirtyEight-style side-by-side visualization of NBA late-scratch performance using R's gt package.

Purpose:
Visualize how players perform when a high-usage teammate scratches late - showing
the top performers who exceed expectations and those who underperform.

Context:
Side-by-side layout using R's gt + magick packages:
- Left table: Top N players (best PPG over expectation)
- Right table: Bottom N players (worst PPG over expectation)
Using rpy2 for publication-quality side-by-side tables.

Architecture:
1. Load and prepare data in Python (pandas)
   - Read parquet cache from detect_nba_dnp_scenarios.py
   - Split into top N and bottom N players
   - Format display columns (PPG_OVER_EXP gradient is KEY)
   
2. Create TWO separate gt tables in R
   - Left table: Top N players with full title/subtitle
   - Right table: Bottom N players with simpler header
   - Both use 538-style formatting
   - Gradient coloring on PPG_OVER_EXP column
   
3. Combine horizontally using magick
   - Save left.png and right.png
   - Use magick::image_append() to combine
   - Clean up intermediate files
   
4. Export as high-resolution PNG
   - Output ready for Twitter/social media

Key Metrics:
- PPG_OVER_EXP: Actual PPG - Projected PPG
  - Positive = Player exceeded expectations (GOOD)
  - Negative = Player underperformed (BAD)
  - Gradient: Green (high positive) -> White (neutral) -> Red (negative)
- COVER_RATE: % of games player covered their line
- GAMES: Number of games in late-scratch scenarios

Installation:
    # 1. Install R (if not already installed)
    # macOS:
    brew install r
    
    # 2. Install required R packages
    R
    install.packages("gt")
    install.packages("gtExtras")
    install.packages("tidyverse")
    install.packages("webshot2")
    install.packages("magick")
    quit()
    
    # 3. Install Python packages
    pip install rpy2 pandas

Requirements:
    - R (with gt, gtExtras, tidyverse, webshot2, magick packages)
    - Python: rpy2, pandas, duckdb

Usage:
    cd /Users/thomasmyles/dev/betting
    
    # Use default cache (2025-26 season)
    python3 analysis/viz_nba_dnp_scenarios_gt.py
    
    # Show top/bottom 20 instead of 15
    python3 analysis/viz_nba_dnp_scenarios_gt.py --n 20
    
    # Use 3-season cache
    python3 analysis/viz_nba_dnp_scenarios_gt.py --seasons 2025-26 2024-25 2023-24
    
    # Custom gradient bounds (default: -8 to 10)
    python3 analysis/viz_nba_dnp_scenarios_gt.py --gradient-min -10 --gradient-max 12

Author: Thomas Myles
Date: 2026-02-04
"""

import pandas as pd
from pathlib import Path
import sys
import subprocess
import platform
from datetime import datetime
import argparse
import duckdb
import requests
import base64
from io import BytesIO
from PIL import Image
import ssl
import urllib3

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
TWITTER_HANDLE = "@TQSLabs"
CACHE_DIR = Path.home() / 'Downloads' / 'tmp' / 'dnp_scenarios_cache'

# -----------------------------------------------------------------------------
# Titles and Footer Text
# -----------------------------------------------------------------------------
# Main title (both tables)
MAIN_TITLE = "NBA Late-Game Scratch Performance"

# Left table subtitle
TITLE_LEFT = "Players Who Exceed Expectations"
SUBTITLE_LEFT = "When a high-usage teammate scratches late"

# Right table subtitle
TITLE_RIGHT = "Players Who Underperform"
SUBTITLE_RIGHT = "When a high-usage teammate scratches late"

# Left side footer - data citation & attribution
FOOTER_NOTES_LEFT = """
**Data**: The Odds API, NBA API ({date}) | **Analysis**: {handle}  
&nbsp;  
&nbsp;
"""

# Right side footer - key definitions (right-aligned)
# Note: min_games filter will be dynamically inserted
FOOTER_NOTES_RIGHT_TEMPLATE = """
**Consensus Line**: Median prop line across all bookmakers  
**PPG Over Exp**: Actual PPG minus consensus line (green = exceeded, red = underperformed)  
**Cover Rate**: Percentage of games player covered their line  
**Filters**: Minimum {min_games} games in late-scratch scenarios (2+ teammates projected 20+ pts)
"""
FOOTER_DATA_DATE = datetime.now().strftime("%B %d, %Y")

# -----------------------------------------------------------------------------
# Output Settings (Image Dimensions & Quality)
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp'
OUTPUT_FILENAME = "nba_dnp_scenarios.png"
LEFT_FILENAME = "nba_dnp_scenarios_left.png"
RIGHT_FILENAME = "nba_dnp_scenarios_right.png"

# Side-by-side: 16:9 aspect ratio when combined (matching defensive_disruptors)
TABLE_WIDTH = 1400   # pixels per table (matching defensive_disruptors)
TABLE_HEIGHT = 1200  # pixels per table (matching defensive_disruptors for proper aspect ratio)
OUTPUT_DPI = 300

# -----------------------------------------------------------------------------
# Color Palette (PPG_OVER_EXP Gradient)
# -----------------------------------------------------------------------------
# Green (high positive) -> White (neutral) -> Red (negative)
# Positive is GOOD (exceeded expectations), negative is BAD (underperformed)
# Use 5-color palette with white in the middle for proper anchoring at 0 and 50%
COLOR_PALETTE = ["#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"]  # red -> white -> green
# Default gradient bounds
DEFAULT_GRADIENT_MIN = -8.0  # Lower bound (red end - underperformed)
DEFAULT_GRADIENT_MAX = 10.0  # Upper bound (green end - exceeded)

# -----------------------------------------------------------------------------
# Typography (matching defensive_disruptors for proper proportions)
# -----------------------------------------------------------------------------
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 24
SUBTITLE_FONT_SIZE = 14
HEADER_FONT_SIZE = 13
BODY_FONT_SIZE = 12
FOOTER_FONT_SIZE = 10

# -----------------------------------------------------------------------------
# Spacing & Padding (matching defensive_disruptors for compact rows)
# -----------------------------------------------------------------------------
HEADER_PADDING_PX = 1
DATA_ROW_PADDING_PX = 0.5    # Minimal padding for compact rows (matching defensive_disruptors)
HEADING_PADDING_PX = 6

# -----------------------------------------------------------------------------
# Column Widths (pixels) - compact like defensive_disruptors
# -----------------------------------------------------------------------------
COL_WIDTH_RANK = 60
COL_WIDTH_PLAYER = 180
COL_WIDTH_HEADSHOT = 55
COL_WIDTH_GAMES = 70
COL_WIDTH_CONSENSUS_LINE = 110
COL_WIDTH_AVG_ACTUAL = 90
COL_WIDTH_PPG_OVER_EXP = 110
COL_WIDTH_COVER_RATE = 100

HEADSHOT_HEIGHT = 35  # Height of player headshots in pixels (matching defensive_disruptors)

# -----------------------------------------------------------------------------
# Player ID Mapping (for headshots from NBA CDN)
# -----------------------------------------------------------------------------
def load_player_id_map():
    """Load comprehensive player ID map from JSON file."""
    import json
    map_file = Path.home() / 'Downloads' / 'tmp' / 'nba_player_id_map.json'
    
    if not map_file.exists():
        print(f"   ⚠️  Player ID map not found at {map_file}")
        print("   ⚠️  Run: python3 analysis/build_nba_player_id_map.py")
        return {}
    
    with open(map_file, 'r') as f:
        player_map = json.load(f)
    
    # Convert player IDs to integers
    return {name: int(player_id) for name, player_id in player_map.items()}

# Load player ID map at module level
PLAYER_ID_MAP = load_player_id_map()


# =============================================================================
# HEADSHOT FUNCTIONS
# =============================================================================

def download_and_convert_to_base64(url, max_size=(300, 300)):
    """
    Download image and convert to base64 data URI.
    
    Args:
        url: Image URL
        max_size: Tuple of (width, height) to resize to
        
    Returns:
        base64 data URI string or None if failed
    """
    try:
        response = requests.get(url, verify=False, timeout=5)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        data_uri = f"data:image/png;base64,{img_base64}"
        
        return data_uri
    except:
        return None


def add_player_headshots(df):
    """
    Add player headshot data URIs to dataframe.
    
    Downloads images from NBA CDN and converts to base64 data URIs.
    Uses normalized + case-insensitive lookup for player ID map.
    
    Args:
        df: DataFrame with player names
        
    Returns:
        DataFrame with headshot_url column added (as base64 data URI)
    """
    from player_name_utils import normalize_player_name
    
    # Create normalized lookup map
    # This handles: accents (Dončić → Doncic), case (CJ → cj), generational suffixes (III removed)
    player_id_map_normalized = {
        normalize_player_name(k).lower(): v 
        for k, v in PLAYER_ID_MAP.items()
    }
    
    def get_headshot_data_uri(player_name):
        # Try exact match first
        player_id = PLAYER_ID_MAP.get(player_name)
        
        # If not found, try normalized + case-insensitive lookup
        if player_id is None:
            player_id = player_id_map_normalized.get(player_name.lower())
        
        if player_id is None:
            # Return transparent 1x1 pixel placeholder
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        url = f'https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png'
        data_uri = download_and_convert_to_base64(url, max_size=(300, 300))
        # Return placeholder if conversion fails
        return data_uri if data_uri else "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    df['headshot_url'] = df['player'].apply(get_headshot_data_uri)
    
    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_player_stats_from_cache(cache_key, min_games=3):
    """
    Load player statistics from cached parquet file.
    
    Args:
        cache_key: Cache key string
        min_games: Minimum games to include player
    
    Returns:
        DataFrame with player stats
    """
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}\n"
            f"Run detect_nba_dnp_scenarios.py with --use-cache first"
        )
    
    print(f"📂 Loading from cache: {cache_file}")
    
    # Read with DuckDB
    con = duckdb.connect(':memory:')
    df = con.execute(f"SELECT * FROM '{cache_file}' WHERE played = true").df()
    con.close()
    
    # Calculate player-level stats
    player_stats = df.groupby('player').agg({
        'actual_points': ['mean', 'count'],
        'projection': 'mean'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['player', 'avg_actual', 'games', 'avg_projection']
    
    # Calculate metrics
    player_stats['ppg_over_exp'] = player_stats['avg_actual'] - player_stats['avg_projection']
    
    # Calculate cover rate
    df['covered'] = df['actual_points'] >= df['projection']
    covers = df.groupby('player')['covered'].mean().reset_index(name='cover_rate')
    player_stats = player_stats.merge(covers, on='player')
    
    # Filter to min games
    player_stats = player_stats[player_stats['games'] >= min_games]
    
    # Sort by PPG over expectation (primary), then cover rate (secondary)
    player_stats = player_stats.sort_values(['ppg_over_exp', 'cover_rate'], ascending=[False, False])
    
    print(f"✅ Loaded {len(player_stats)} players with {min_games}+ games\n")
    
    return player_stats


def prepare_display_data(player_stats, n=15, is_top=True):
    """
    Prepare data for display in gt table.
    
    Args:
        player_stats: DataFrame with player statistics
        n: Number of players to show
        is_top: True for top performers, False for bottom
    
    Returns:
        DataFrame formatted for display
    """
    if is_top:
        df = player_stats.head(n).copy()
    else:
        df = player_stats.tail(n).copy()
        df = df.sort_values('ppg_over_exp', ascending=True)  # Still worst to best visually
    
    # Add rank column
    df['rank'] = range(1, len(df) + 1)
    
    # Add player headshots
    print(f"   🖼️  Converting player headshots to base64 data URIs...")
    df = add_player_headshots(df)
    success_count = df['headshot_url'].apply(lambda x: len(x) > 200).sum()
    print(f"      ✅ Converted {success_count}/{len(df)} headshots successfully")
    
    # Format columns for display
    df['display_games'] = df['games'].astype(int)
    df['display_ppg_over_exp'] = df['ppg_over_exp'].apply(lambda x: f"{x:+.1f}")
    df['display_cover_rate'] = df['cover_rate'].apply(lambda x: f"{x:.0%}")
    df['display_avg_actual'] = df['avg_actual'].apply(lambda x: f"{x:.1f}")
    df['display_avg_projection'] = df['avg_projection'].apply(lambda x: f"{x:.1f}")
    
    # Select and rename columns for display (headshot after player)
    display_df = df[[
        'rank', 'player', 'headshot_url', 'display_games', 'display_avg_projection',
        'display_avg_actual', 'display_ppg_over_exp', 'display_cover_rate',
        'ppg_over_exp',  # Keep for gradient coloring
        'cover_rate'     # Keep for gradient coloring
    ]].copy()
    
    display_df.columns = [
        'RANK', 'PLAYER', 'headshot_url', 'GAMES', 'CONSENSUS LINE',
        'AVG PPG', 'PPG OVER EXP', 'COVER RATE', 'ppg_over_exp_value', 'cover_rate_value'
    ]
    
    return display_df


# =============================================================================
# R GT TABLE GENERATION
# =============================================================================

def create_gt_table_r(df, title, subtitle, footer_notes, is_left, 
                      ppg_gradient_min, ppg_gradient_max,
                      cover_rate_gradient_min, cover_rate_gradient_max):
    """
    Create gt table using R via rpy2.
    
    Args:
        df: DataFrame to display
        title: Table title
        subtitle: Table subtitle  
        footer_notes: Footer text
        is_left: True if left table (for filename)
        ppg_gradient_min: Min value for PPG OVER EXP gradient
        ppg_gradient_max: Max value for PPG OVER EXP gradient
        cover_rate_gradient_min: Min value for COVER RATE gradient
        cover_rate_gradient_max: Max value for COVER RATE gradient
    
    Returns:
        Path to saved PNG file
    """
    try:
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        
        # Convert pandas DataFrame to R using context manager
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(df)
        
        # Pass data to R environment
        robjects.globalenv['gt_tbl_data'] = r_df
        
        # Determine output filename with full path
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = str(OUTPUT_DIR / (LEFT_FILENAME if is_left else RIGHT_FILENAME))
        
        # Build R code as string for easier debugging
        # Note: User library path set in R code so packages load from ~/R/library
        user_lib_path = str(Path.home() / 'R' / 'library')
        print(f"   📁 Using R library path: {user_lib_path}")
        
        r_code = f"""
        # Set library path to user library
        .libPaths(c("{user_lib_path}", .libPaths()))
        
        library(gt)
        library(gtExtras)
        library(webshot2)
        
        # Create gt table
        tbl <- gt_tbl_data %>%
          gt() %>%
          
          # Add player headshots using gtExtras
          gt_img_rows(columns = headshot_url, height = {HEADSHOT_HEIGHT}) %>%
          
          # Add title and subtitle
          tab_header(
            title = md("**{title}**"),
            subtitle = md("{subtitle}")
          ) %>%
          
          # Format PPG OVER EXP column with gradient (using numeric value column)
          # 5-color palette with white in middle: red -> light red -> white -> light green -> green
          data_color(
            columns = ppg_over_exp_value,
            target_columns = `PPG OVER EXP`,
            method = "numeric",
            palette = c("{COLOR_PALETTE[0]}", "{COLOR_PALETTE[1]}", "{COLOR_PALETTE[2]}", 
                       "{COLOR_PALETTE[3]}", "{COLOR_PALETTE[4]}"),
            domain = c({ppg_gradient_min}, {ppg_gradient_max})
          ) %>%
          
          # Format COVER RATE column with gradient (using numeric value column)
          # 5-color palette with white in middle, anchored at 50%
          data_color(
            columns = cover_rate_value,
            target_columns = `COVER RATE`,
            method = "numeric",
            palette = c("{COLOR_PALETTE[0]}", "{COLOR_PALETTE[1]}", "{COLOR_PALETTE[2]}", 
                       "{COLOR_PALETTE[3]}", "{COLOR_PALETTE[4]}"),
            domain = c({cover_rate_gradient_min}, {cover_rate_gradient_max})
          ) %>%
          
          # Column alignments
          cols_align(
            align = "center",
            columns = c(RANK, GAMES, `CONSENSUS LINE`, `AVG PPG`, `PPG OVER EXP`, `COVER RATE`)
          ) %>%
          cols_align(
            align = "left",
            columns = PLAYER
          ) %>%
          
          # Column widths (headshot between player and games)
          cols_width(
            RANK ~ px({COL_WIDTH_RANK}),
            PLAYER ~ px({COL_WIDTH_PLAYER}),
            headshot_url ~ px({COL_WIDTH_HEADSHOT}),
            GAMES ~ px({COL_WIDTH_GAMES}),
            `CONSENSUS LINE` ~ px({COL_WIDTH_CONSENSUS_LINE}),
            `AVG PPG` ~ px({COL_WIDTH_AVG_ACTUAL}),
            `PPG OVER EXP` ~ px({COL_WIDTH_PPG_OVER_EXP}),
            `COVER RATE` ~ px({COL_WIDTH_COVER_RATE})
          ) %>%
          
          # Rename headshot_url column header to empty
          cols_label(
            headshot_url = ""
          ) %>%
          
          # Hide the numeric gradient columns
          cols_hide(columns = c(ppg_over_exp_value, cover_rate_value)) %>%
          
          # Add footer
          tab_source_note(source_note = md("{footer_notes}")) %>%
          
          # Apply 538 theme
          gt_theme_538() %>%
          
          # Additional styling
          tab_options(
            table.font.size = px({BODY_FONT_SIZE}),
            heading.title.font.size = px({TITLE_FONT_SIZE}),
            heading.subtitle.font.size = px({SUBTITLE_FONT_SIZE}),
            column_labels.font.size = px({HEADER_FONT_SIZE}),
            source_notes.font.size = px({FOOTER_FONT_SIZE}),
            table.width = px({TABLE_WIDTH}),
            data_row.padding = px({DATA_ROW_PADDING_PX}),
            heading.padding = px({HEADING_PADDING_PX})
          )
        
        # Save as PNG
        gtsave(tbl, "{output_file}", vwidth = {TABLE_WIDTH}, vheight = {TABLE_HEIGHT})
        """
        
        # Execute R code
        robjects.r(r_code)
        
        print(f"✅ Created {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating gt table: {e}")
        print(f"   Make sure R and required packages are installed")
        print(f"   Run: R -e \"install.packages(c('gt', 'gtExtras', 'webshot2', 'magick'))\"")
        raise


def combine_tables_r(left_file, right_file, output_file):
    """
    Combine left and right tables side-by-side using R's magick package.
    Resizes to achieve proper 16:9 aspect ratio.
    
    Args:
        left_file: Path to left table PNG
        right_file: Path to right table PNG
        output_file: Path for combined output PNG
    """
    try:
        from rpy2 import robjects
        from rpy2.robjects.packages import importr
        
        # Import magick
        magick = importr('magick')
        
        # R code to combine images (matching defensive_disruptors - NO resizing!)
        r_code = f"""
        library(magick)
        
        # Read images
        left <- image_read("{left_file}")
        right <- image_read("{right_file}")
        
        # Get individual table dimensions
        left_info <- image_info(left)
        right_info <- image_info(right)
        
        cat(sprintf("Left table: %dx%d\\n", left_info$width, left_info$height))
        cat(sprintf("Right table: %dx%d\\n", right_info$width, right_info$height))
        
        # Combine side-by-side (horizontal append)
        combined <- image_append(c(left, right))
        
        # Verify final dimensions
        final_info <- image_info(combined)
        cat(sprintf("Final image: %dx%d (aspect ratio: %.3f)\\n", 
                   final_info$width, final_info$height, 
                   final_info$width / final_info$height))
        
        # Save directly without resizing (let gt render at natural size)
        image_write(combined, "{output_file}", format = "png", quality = 100)
        """
        
        robjects.r(r_code)
        
        print(f"✅ Combined tables saved to: {output_file}")
        
        # Clean up intermediate files
        if Path(left_file).exists():
            Path(left_file).unlink()
        if Path(right_file).exists():
            Path(right_file).unlink()
        print(f"🧹 Cleaned up intermediate files")
        
    except Exception as e:
        print(f"❌ Error combining tables: {e}")
        raise


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create side-by-side visualization of NBA late-scratch performance'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=15,
        help='Number of top/bottom players to show (default: 15)'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2025-26'],
        help='NBA seasons (must match cache key) (default: 2025-26)'
    )
    parser.add_argument(
        '--teams',
        nargs='+',
        default=None,
        help='Team filter (must match cache key)'
    )
    parser.add_argument(
        '--points-threshold',
        type=float,
        default=20.0,
        help='Points threshold (must match cache key) (default: 20)'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=10,
        help='Minimum games to include player (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏀 NBA LATE-GAME SCRATCH VISUALIZATION")
    print("="*80)
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Top/Bottom N: {args.n}")
    print(f"Min Games: {args.min_games}")
    print()
    
    # Generate cache key
    seasons_str = '_'.join(sorted(args.seasons))
    teams_str = '_'.join(sorted(args.teams)) if args.teams else 'all'
    cache_key = f"{seasons_str}_teams_{teams_str}_threshold_{args.points_threshold}"
    
    # Load data
    player_stats = load_player_stats_from_cache(cache_key, min_games=args.min_games)
    
    if len(player_stats) < args.n * 2:
        print(f"⚠️  Warning: Only {len(player_stats)} players available, need {args.n * 2} for top/bottom {args.n}")
        print(f"   Showing all available players")
        n = len(player_stats) // 2
    else:
        n = args.n
    
    # Prepare display data
    top_players = prepare_display_data(player_stats, n=n, is_top=True)
    bottom_players = prepare_display_data(player_stats, n=n, is_top=False)
    
    print(f"📊 Top {n} players:")
    print(f"   Best: {top_players.iloc[0]['PLAYER']} ({top_players.iloc[0]['PPG OVER EXP']})")
    print(f"   {n}th: {top_players.iloc[-1]['PLAYER']} ({top_players.iloc[-1]['PPG OVER EXP']})")
    print()
    print(f"📊 Bottom {n} players:")
    print(f"   Worst: {bottom_players.iloc[0]['PLAYER']} ({bottom_players.iloc[0]['PPG OVER EXP']})")
    print(f"   {n}th: {bottom_players.iloc[-1]['PLAYER']} ({bottom_players.iloc[-1]['PPG OVER EXP']})")
    print()
    
    # Calculate dynamic gradient ranges based on the plotted data
    # Combine both datasets to get full range
    all_ppg_values = list(top_players['ppg_over_exp_value']) + list(bottom_players['ppg_over_exp_value'])
    all_cover_rate_values = list(top_players['cover_rate_value']) + list(bottom_players['cover_rate_value'])
    
    # For PPG OVER EXP: use symmetric range around 0
    ppg_max_abs = max(abs(min(all_ppg_values)), abs(max(all_ppg_values)))
    ppg_gradient_min = -ppg_max_abs
    ppg_gradient_max = ppg_max_abs
    
    # For COVER RATE: use symmetric range around 0.5 (50%)
    # First center the values around 0
    cover_rate_centered = [v - 0.5 for v in all_cover_rate_values]
    cover_rate_max_abs = max(abs(min(cover_rate_centered)), abs(max(cover_rate_centered)))
    cover_rate_gradient_min = 0.5 - cover_rate_max_abs
    cover_rate_gradient_max = 0.5 + cover_rate_max_abs
    
    print(f"📊 Dynamic gradient ranges:")
    print(f"   PPG OVER EXP: [{ppg_gradient_min:.1f}, {ppg_gradient_max:.1f}]")
    print(f"   COVER RATE: [{cover_rate_gradient_min:.1%}, {cover_rate_gradient_max:.1%}]")
    print()
    
    # Format footer notes
    footer_left = FOOTER_NOTES_LEFT.format(
        date=FOOTER_DATA_DATE,
        handle=TWITTER_HANDLE
    )
    footer_right = FOOTER_NOTES_RIGHT_TEMPLATE.format(min_games=args.min_games)
    
    # Create left table (top performers)
    print("🎨 Creating left table (top performers)...")
    left_file = create_gt_table_r(
        top_players,
        TITLE_LEFT,
        SUBTITLE_LEFT,
        footer_left,
        is_left=True,
        ppg_gradient_min=ppg_gradient_min,
        ppg_gradient_max=ppg_gradient_max,
        cover_rate_gradient_min=cover_rate_gradient_min,
        cover_rate_gradient_max=cover_rate_gradient_max
    )
    
    # Create right table (bottom performers)
    print("🎨 Creating right table (bottom performers)...")
    right_file = create_gt_table_r(
        bottom_players,
        TITLE_RIGHT,
        SUBTITLE_RIGHT,
        footer_right,
        is_left=False,
        ppg_gradient_min=ppg_gradient_min,
        ppg_gradient_max=ppg_gradient_max,
        cover_rate_gradient_min=cover_rate_gradient_min,
        cover_rate_gradient_max=cover_rate_gradient_max
    )
    
    # Combine tables
    print("🔗 Combining tables...")
    final_output = str(OUTPUT_DIR / OUTPUT_FILENAME)
    combine_tables_r(left_file, right_file, final_output)
    
    print()
    print("="*80)
    print(f"✅ Visualization complete: {final_output}")
    print("="*80)


if __name__ == '__main__':
    main()
