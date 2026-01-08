"""
Create a FiveThirtyEight-style side-by-side visualization of NBA defensive disruptors using R's gt package.

Purpose:
Visualize the most impactful defensive players in the NBA - those who make their teams
significantly better (or worse) defensively when on the court.

Context:
Side-by-side layout using R's gt + magick packages:
- Left table: Top N players (biggest positive defensive impact)
- Right table: Bottom N players (biggest negative defensive impact / hurt team)
Using rpy2 for publication-quality side-by-side tables.

Architecture:
1. Load and prepare data in Python (pandas)
   - Read CSV from analyze_defensive_disruptors.py
   - Split into top N and bottom N players
   - Format display columns (DEF_IMPACT gradient is KEY)
   
2. Create TWO separate gt tables in R
   - Left table: Top N players with full title/subtitle
   - Right table: Bottom N players with simpler header
   - Both use 538-style formatting
   - Gradient coloring on DEF_IMPACT column
   
3. Combine horizontally using magick
   - Save left.png and right.png
   - Use magick::image_append() to combine
   - Clean up intermediate files
   
4. Export as high-resolution PNG
   - Output ready for Twitter/social media

Key Metrics:
- DEF_IMPACT: Team DEF_RATING - Player DEF_RATING
  - Positive = Player improves team defense (GOOD)
  - Negative = Player hurts team defense (BAD)
  - Gradient: Green (high positive) -> White (neutral) -> Red (negative)

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
    - Python: rpy2, pandas

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/viz_nba_defensive_disruptors_gt.py
    
    # Show top/bottom 20 instead of 15
    python3 analysis/viz_nba_defensive_disruptors_gt.py --n 20
    
    # Custom gradient bounds (default: -20 to 20)
    python3 analysis/viz_nba_defensive_disruptors_gt.py --gradient-min -10 --gradient-max 15
"""

import pandas as pd
from pathlib import Path
import sys
import subprocess
import platform
from datetime import datetime
import argparse
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
CURRENT_NBA_SEASON = "2025-26"

# -----------------------------------------------------------------------------
# Titles and Footer Text
# -----------------------------------------------------------------------------
# Main title (both tables)
MAIN_TITLE = "NBA's Most Disruptive Defenders"

# Left table subtitle
TITLE_LEFT = "Players Who Elevate Team Defense"
SUBTITLE_LEFT = ""

# Right table subtitle
TITLE_RIGHT = "Players Who Hurt Team Defense"
SUBTITLE_RIGHT = ""

# Left side footer - data citation & attribution (like other gt visualizations)
# Note: Extra line breaks added to match right footer height for aligned bottom borders
FOOTER_NOTES_LEFT = """
**Data**: NBA API (stats.nba.com) ({date}) | **Analysis**: {handle}  
&nbsp;  
&nbsp;
"""

# Right side footer - key definitions (right-aligned)
FOOTER_NOTES_RIGHT = """
**Defensive Rating**: Points allowed per 100 possessions (lower is better)  
**Defensive Impact**: Team DEF_RATING - Player DEF_RATING  
**Filters**: Minimum 100 minutes played
"""
FOOTER_DATA_SOURCE = "NBA API (stats.nba.com)"
FOOTER_DATA_DATE = datetime.now().strftime("%B %d, %Y")

# -----------------------------------------------------------------------------
# Output Settings (Image Dimensions & Quality)
# -----------------------------------------------------------------------------
OUTPUT_FILENAME = "nba_defensive_disruptors.png"
LEFT_FILENAME = "nba_defensive_disruptors_left.png"
RIGHT_FILENAME = "nba_defensive_disruptors_right.png"

# Side-by-side: 16:9 aspect ratio, wider tables
TABLE_WIDTH = 1400   # pixels per table (wider for 16:9)
TABLE_HEIGHT = 1200  # pixels per table (shorter for 16:9)
OUTPUT_DPI = 300

# -----------------------------------------------------------------------------
# Color Palette (DEF_IMPACT Gradient)
# -----------------------------------------------------------------------------
# Green (high positive) -> White (neutral) -> Red (negative impact)
# Positive impact is GOOD (helps defense), negative is BAD (hurts defense)
COLOR_PALETTE = ["#d62728", "#ff9999", "#ffcccc", "#ffffff", "#90EE90", "#00b300"]  # red -> white -> green
# Default gradient bounds (will be overridden by command-line args)
DEFAULT_GRADIENT_MIN = -20.0  # Lower bound (red end - hurts team)
DEFAULT_GRADIENT_MAX = 20.0   # Upper bound (green end - helps team)

# -----------------------------------------------------------------------------
# Typography
# -----------------------------------------------------------------------------
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 24
SUBTITLE_FONT_SIZE = 14
HEADER_FONT_SIZE = 13
BODY_FONT_SIZE = 12
FOOTER_FONT_SIZE = 10

# -----------------------------------------------------------------------------
# Spacing & Padding
# -----------------------------------------------------------------------------
HEADER_PADDING_PX = 1
DATA_ROW_PADDING_PX = 0.5    # Minimal padding for compact rows
HEADING_PADDING_PX = 6

# -----------------------------------------------------------------------------
# Column Widths (pixels)
# -----------------------------------------------------------------------------
COL_WIDTH_RANK = 60
COL_WIDTH_PLAYER = 180
COL_WIDTH_TEAM = 60
COL_WIDTH_MIN = 60
COL_WIDTH_DEF_RATING = 90
COL_WIDTH_TEAM_DEF_RATING = 100
COL_WIDTH_DEF_IMPACT = 100
COL_WIDTH_STL = 55
COL_WIDTH_BLK = 55
COL_WIDTH_DEF_WS = 70
COL_WIDTH_HEADSHOT = 55  # Width for player headshot column
HEADSHOT_HEIGHT = 35      # Height of player headshots in pixels (smaller for compact rows)


def download_and_convert_to_base64(url, max_size=(60, 60)):
    """
    Download image and convert to base64 data URI.
    
    Args:
        url: Image URL
        max_size: Tuple of (width, height) to resize to
        
    Returns:
        base64 data URI string or None if failed
    """
    try:
        response = requests.get(url, verify=False, timeout=10)
        if response.status_code != 200:
            return None
        
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
    This avoids HTTPS loading issues in webshot2/R rendering.
    
    Args:
        df: DataFrame with PLAYER_ID column
        
    Returns:
        DataFrame with headshot_url column added (as base64 data URI)
    """
    print("   🖼️  Converting player headshots to base64 data URIs...")
    
    def get_headshot_data_uri(player_id):
        url = f'https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png'
        data_uri = download_and_convert_to_base64(url, max_size=(60, 60))
        # Return placeholder if conversion fails
        return data_uri if data_uri else "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    df['headshot_url'] = df['PLAYER_ID'].apply(get_headshot_data_uri)
    
    success_count = df['headshot_url'].apply(lambda x: len(x) > 200).sum()
    print(f"      ✅ Converted {success_count}/{len(df)} headshots successfully\n")
    
    return df


def prepare_data_for_visualization(df, n=15):
    """
    Prepare the dataframe with all display columns.
    
    Args:
        df: Raw dataframe from defensive_disruptors_2025_26.csv
        n: Number of players to show on each side (top N and bottom N)
        
    Returns:
        Tuple of (top_n_df, bottom_n_df) ready for visualization
    """
    print(f"📊 Preparing data for visualization (top/bottom {n})...\n")
    
    df_display = df.copy()
    
    # Add player headshots
    df_display = add_player_headshots(df_display)
    
    # Sort by DEF_IMPACT (descending - highest positive impact first)
    df_display = df_display.sort_values('DEF_IMPACT', ascending=False)
    
    # Add rank column
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Format numeric columns for display
    df_display['MIN_PG_str'] = df_display['MIN_PG'].round(1).astype(str)
    df_display['DEF_RATING_str'] = df_display['DEF_RATING'].round(1).astype(str)
    df_display['TEAM_DEF_RATING_str'] = df_display['TEAM_DEF_RATING'].round(1).astype(str)
    df_display['DEF_IMPACT_str'] = df_display['DEF_IMPACT'].apply(
        lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
    )
    df_display['STL_PG_str'] = df_display['STL_PG'].round(1).astype(str)
    df_display['BLK_PG_str'] = df_display['BLK_PG'].round(1).astype(str)
    df_display['DEF_WS_str'] = df_display['DEF_WS'].round(1).astype(str)
    
    # Split into top N and bottom N
    top_n = df_display.head(n).copy()
    bottom_n = df_display.tail(n).copy()
    
    # Reverse bottom_n so worst is at top
    bottom_n = bottom_n.sort_values('DEF_IMPACT', ascending=True)
    
    print(f"   ✅ Top {n} players: DEF_IMPACT range {top_n['DEF_IMPACT'].min():.1f} to {top_n['DEF_IMPACT'].max():.1f}")
    print(f"   ✅ Bottom {n} players: DEF_IMPACT range {bottom_n['DEF_IMPACT'].min():.1f} to {bottom_n['DEF_IMPACT'].max():.1f}\n")
    
    return top_n, bottom_n


def create_gt_table_with_r(top_n, bottom_n, n=15, gradient_min=0.0, gradient_max=20.0):
    """
    Create a side-by-side publication-quality table using R's gt + magick packages via rpy2.
    
    Creates two tables:
    - Left: Top N defensive disruptors (biggest positive impact)
    - Right: Bottom N defenders (biggest negative impact)
    
    Args:
        top_n: DataFrame with top N players
        bottom_n: DataFrame with bottom N players
        n: Number of players per side
        gradient_min: Lower bound for color gradient (default: -20.0)
        gradient_max: Upper bound for color gradient (default: 20.0)
        
    Returns:
        Path to saved combined PNG file
    """
    print(f"🎨 Creating SIDE-BY-SIDE tables with R's gt + magick packages...\n")
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
        
        print("   ✅ rpy2 loaded successfully")
        
    except ImportError as e:
        print(f"❌ Error: rpy2 not installed or R not found")
        print(f"   {e}")
        print("\n📖 Installation instructions:")
        print("   1. Install R: brew install r")
        print("   2. Install Python package: pip install rpy2")
        print("   3. Install R packages: R -e 'install.packages(c(\"gt\", \"gtExtras\", \"tidyverse\", \"webshot2\", \"magick\"))'")
        sys.exit(1)
    
    # Select columns for display (headshot between Player and Team)
    cols_to_keep = [
        'rank', 'PLAYER_NAME', 'headshot_url', 'TEAM_ABBREVIATION', 'MIN_PG',
        'DEF_RATING', 'TEAM_DEF_RATING', 'DEF_IMPACT',
        'STL_PG', 'BLK_PG', 'DEF_WS'
    ]
    
    table_left = top_n[cols_to_keep].copy()
    table_right = bottom_n[cols_to_keep].copy()
    
    # Rename columns for display
    table_left.columns = ['Rank', 'Player', 'headshot_url', 'Team', 'MPG', 'Player DEF', 'Team DEF', 'Impact', 'STL', 'BLK', 'DEF WS']
    table_right.columns = ['Rank', 'Player', 'headshot_url', 'Team', 'MPG', 'Player DEF', 'Team DEF', 'Impact', 'STL', 'BLK', 'DEF WS']
    
    print(f"   📋 Left table dimensions: {table_left.shape}")
    print(f"   📋 Right table dimensions: {table_right.shape}\n")
    
    # Convert both to R dataframes
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_left = ro.conversion.py2rpy(table_left)
        r_df_right = ro.conversion.py2rpy(table_right)
    
    ro.globalenv['nba_data_left'] = r_df_left
    ro.globalenv['nba_data_right'] = r_df_right
    
    # Output paths
    output_dir = repo_root / 'content/viz/nba'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left_path = output_dir / LEFT_FILENAME
    right_path = output_dir / RIGHT_FILENAME
    final_path = output_dir / OUTPUT_FILENAME
    
    left_path_str = str(left_path)
    right_path_str = str(right_path)
    final_path_str = str(final_path)
    
    print(f"   💾 Left output: {LEFT_FILENAME}")
    print(f"   💾 Right output: {RIGHT_FILENAME}")
    print(f"   💾 Combined output: {OUTPUT_FILENAME}\n")
    
    # Format footer notes with current date and handle
    footer_left_formatted = FOOTER_NOTES_LEFT.format(date=FOOTER_DATA_DATE, handle=TWITTER_HANDLE)
    
    # R code to create TWO tables and combine them side-by-side
    r_code = f"""
    # Set library path to user library
    .libPaths(c("~/R/library", .libPaths()))
    
    # Configure for HTTPS image loading
    options(download.file.method = "libcurl")
    options(download.file.extra = "-k")
    Sys.setenv(CHROMOTE_CHROME = chromote::find_chrome())
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    library(magick)
    library(webshot2)
    
    # ==========================================================================
    # LEFT TABLE (Top {n} players) - Full title/subtitle + footer
    # ==========================================================================
    table_left <- nba_data_left %>%
      gt() %>%
      
      # Add player headshots using gtExtras
      gt_img_rows(columns = headshot_url, height = {HEADSHOT_HEIGHT}) %>%
      
      # Title on left side only
      tab_header(
        title = md("**{MAIN_TITLE}**"),
        subtitle = md("{TITLE_LEFT}")
      ) %>%
      
      # Column alignment
      cols_align(
        align = "center",
        columns = everything()
      ) %>%
      cols_align(
        align = "left",
        columns = c(Player)
      ) %>%
      
      # Format numeric columns
      fmt_number(
        columns = c(MPG, `Player DEF`, `Team DEF`, STL, BLK, `DEF WS`),
        decimals = 1
      ) %>%
      
      # Format Impact column with + sign for positive values
      fmt(
        columns = Impact,
        fns = function(x) {{
          ifelse(x >= 0, paste0("+", sprintf("%.1f", x)), sprintf("%.1f", x))
        }}
      ) %>%
      
      # Column widths (headshot between Player and Team)
      cols_width(
        Rank ~ px({COL_WIDTH_RANK}),
        Player ~ px({COL_WIDTH_PLAYER}),
        headshot_url ~ px({COL_WIDTH_HEADSHOT}),
        Team ~ px({COL_WIDTH_TEAM}),
        MPG ~ px({COL_WIDTH_MIN}),
        `Player DEF` ~ px({COL_WIDTH_DEF_RATING}),
        `Team DEF` ~ px({COL_WIDTH_TEAM_DEF_RATING}),
        Impact ~ px({COL_WIDTH_DEF_IMPACT}),
        STL ~ px({COL_WIDTH_STL}),
        BLK ~ px({COL_WIDTH_BLK}),
        `DEF WS` ~ px({COL_WIDTH_DEF_WS})
      ) %>%
      
      # Rename headshot_url column header to empty
      cols_label(
        headshot_url = ""
      ) %>%
      
      # Style headers - bold and larger
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px({HEADER_FONT_SIZE}), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Style title - 538 aesthetic (left-aligned)
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}",
          size = px({TITLE_FONT_SIZE}),
          weight = "bold",
          color = "#2c3e50",
          align = "left"
        ),
        locations = cells_title(groups = "title")
      ) %>%
      
      # Style subtitle (left-aligned)
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}", 
          size = px({SUBTITLE_FONT_SIZE}),
          color = "#555555",
          align = "left"
        ),
        locations = cells_title(groups = "subtitle")
      ) %>%
      
      # Conditional formatting for Impact column (KEY GRADIENT!)
      # Red (negative/bad) -> White (neutral) -> Green (positive/good)
      data_color(
        columns = Impact,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({gradient_min}, {gradient_max})
      ) %>%
      
      # Make rank column bold
      tab_style(
        style = cell_text(weight = "bold", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Rank)
      ) %>%
      
      # Make player names bold
      tab_style(
        style = cell_text(weight = "600", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Player)
      ) %>%
      
      # Zebra striping - subtle
      opt_row_striping(row_striping = TRUE) %>%
      
      # Table options - 538 aesthetic
      tab_options(
        table.font.names = "{FONT_FAMILY}",
        table.font.size = px({BODY_FONT_SIZE}),
        heading.title.font.size = px({TITLE_FONT_SIZE}),
        heading.subtitle.font.size = px({SUBTITLE_FONT_SIZE}),
        heading.title.font.weight = "bold",
        heading.align = "left",  # Left-align title
        heading.padding = px({HEADING_PADDING_PX}),
        table.border.top.style = "hidden",
        table.border.bottom.style = "solid",
        table.border.bottom.width = px(2),
        table.border.bottom.color = "#2c3e50",
        column_labels.border.top.style = "hidden",
        column_labels.border.bottom.width = px(3),
        column_labels.border.bottom.color = "#2c3e50",
        column_labels.padding = px({HEADER_PADDING_PX}),
        data_row.padding = px({DATA_ROW_PADDING_PX}),
        table.background.color = "white",
        row.striping.background_color = "#f8f8f8",
        source_notes.font.size = px({FOOTER_FONT_SIZE}),
        source_notes.padding = px(10)
      ) %>%
      
      # Footer notes - left side (citation on one line with pipe separator, plus spacing to match right footer height)
      tab_source_note(
        source_note = md("{footer_left_formatted}")
      )
    
    # ==========================================================================
    # RIGHT TABLE (Bottom {n} players) - Simpler header + right-aligned
    # ==========================================================================
    table_right <- nba_data_right %>%
      gt() %>%
      
      # Add player headshots using gtExtras
      gt_img_rows(columns = headshot_url, height = {HEADSHOT_HEIGHT}) %>%
      
      # Right side - invisible title for alignment, subtitle on right
      tab_header(
        title = md("**{MAIN_TITLE}**"),  # Same title but will be styled invisible below
        subtitle = md("{TITLE_RIGHT}")
      ) %>%
      
      # Column alignment
      cols_align(
        align = "center",
        columns = everything()
      ) %>%
      cols_align(
        align = "left",
        columns = c(Player)
      ) %>%
      
      # Format numeric columns
      fmt_number(
        columns = c(MPG, `Player DEF`, `Team DEF`, STL, BLK, `DEF WS`),
        decimals = 1
      ) %>%
      
      # Format Impact column with + sign for positive values
      fmt(
        columns = Impact,
        fns = function(x) {{
          ifelse(x >= 0, paste0("+", sprintf("%.1f", x)), sprintf("%.1f", x))
        }}
      ) %>%
      
      # Column widths (same as left table - headshot between Player and Team)
      cols_width(
        Rank ~ px({COL_WIDTH_RANK}),
        Player ~ px({COL_WIDTH_PLAYER}),
        headshot_url ~ px({COL_WIDTH_HEADSHOT}),
        Team ~ px({COL_WIDTH_TEAM}),
        MPG ~ px({COL_WIDTH_MIN}),
        `Player DEF` ~ px({COL_WIDTH_DEF_RATING}),
        `Team DEF` ~ px({COL_WIDTH_TEAM_DEF_RATING}),
        Impact ~ px({COL_WIDTH_DEF_IMPACT}),
        STL ~ px({COL_WIDTH_STL}),
        BLK ~ px({COL_WIDTH_BLK}),
        `DEF WS` ~ px({COL_WIDTH_DEF_WS})
      ) %>%
      
      # Rename headshot_url column header to empty
      cols_label(
        headshot_url = ""
      ) %>%
      
      # Style headers - bold and larger
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px({HEADER_FONT_SIZE}), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Make title invisible (white text) on right side
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}",
          size = px({TITLE_FONT_SIZE}),
          weight = "bold",
          color = "white",  # White text on white background = invisible
          align = "right"
        ),
        locations = cells_title(groups = "title")
      ) %>%
      
      # Style subtitle on right (right-aligned)
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}", 
          size = px({SUBTITLE_FONT_SIZE}),
          color = "#555555",
          align = "right"
        ),
        locations = cells_title(groups = "subtitle")
      ) %>%
      
      # Conditional formatting for Impact column (SAME GRADIENT)
      # Red (negative/bad) -> White (neutral) -> Green (positive/good)
      data_color(
        columns = Impact,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({gradient_min}, {gradient_max})
      ) %>%
      
      # Make rank column bold
      tab_style(
        style = cell_text(weight = "bold", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Rank)
      ) %>%
      
      # Make player names bold
      tab_style(
        style = cell_text(weight = "600", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Player)
      ) %>%
      
      # Zebra striping - subtle
      opt_row_striping(row_striping = TRUE) %>%
      
      # LEFT BORDER for right table (visual separator)
      tab_style(
        style = cell_borders(
          sides = "left",
          color = "black",
          weight = px(3)
        ),
        locations = list(
          cells_body(columns = 1),
          cells_column_labels(1)
        )
      ) %>%
      
      # Table options - 538 aesthetic with RIGHT-ALIGNED heading
      tab_options(
        table.font.names = "{FONT_FAMILY}",
        table.font.size = px({BODY_FONT_SIZE}),
        heading.title.font.size = px({TITLE_FONT_SIZE}),
        heading.subtitle.font.size = px({SUBTITLE_FONT_SIZE}),
        heading.title.font.weight = "bold",
        heading.align = "right",  # RIGHT-ALIGNED!
        heading.padding = px({HEADING_PADDING_PX}),
        table.border.top.style = "hidden",
        table.border.bottom.style = "solid",
        table.border.bottom.width = px(2),
        table.border.bottom.color = "#2c3e50",
        column_labels.border.top.style = "hidden",
        column_labels.border.bottom.width = px(3),
        column_labels.border.bottom.color = "#2c3e50",
        column_labels.padding = px({HEADER_PADDING_PX}),
        data_row.padding = px({DATA_ROW_PADDING_PX}),
        table.background.color = "white",
        row.striping.background_color = "#f8f8f8",
        source_notes.font.size = px({FOOTER_FONT_SIZE}),
        source_notes.padding = px(10)
      ) %>%
      
      # Footer notes - right side (key definitions) - RIGHT ALIGNED
      tab_source_note(
        source_note = md("{FOOTER_NOTES_RIGHT}")
      ) %>%
      
      # Right-align the footer on right table
      tab_style(
        style = cell_text(align = "right"),
        locations = cells_source_notes()
      )
    
    # ==========================================================================
    # SAVE BOTH TABLES (with delay for image loading)
    # ==========================================================================
    gtsave(table_left, "{left_path_str}", vwidth = {TABLE_WIDTH}, vheight = {TABLE_HEIGHT}, delay = 1)
    gtsave(table_right, "{right_path_str}", vwidth = {TABLE_WIDTH}, vheight = {TABLE_HEIGHT}, delay = 1)
    
    print("✅ Left table saved!")
    print("✅ Right table saved!")
    
    # ==========================================================================
    # COMBINE HORIZONTALLY WITH MAGICK
    # ==========================================================================
    img1 <- magick::image_read("{left_path_str}")
    img2 <- magick::image_read("{right_path_str}")
    img3 <- magick::image_append(c(img1, img2))  # Horizontal append
    
    magick::image_write(
      image = img3,
      path = "{final_path_str}",
      format = 'png'
    )
    
    print("✅ Combined side-by-side image saved!")
    """
    
    print("   🔧 Executing R code (creating 2 tables + combining)...\n")
    
    try:
        ro.r(r_code)
        print(f"\n   ✅ Side-by-side tables created and combined!\n")
        return final_path
        
    except Exception as e:
        print(f"❌ Error creating tables in R:")
        print(f"   {e}")
        print("\n💡 Make sure R packages are installed:")
        print("   R -e 'install.packages(c(\"gt\", \"gtExtras\", \"dplyr\", \"webshot2\", \"magick\"))'")
        sys.exit(1)


def main():
    """Main visualization function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize NBA defensive disruptors')
    parser.add_argument('--n', type=int, default=15, 
                       help='Number of players to show on each side (default: 15)')
    parser.add_argument('--gradient-min', type=float, default=DEFAULT_GRADIENT_MIN,
                       help=f'Lower bound for color gradient (default: {DEFAULT_GRADIENT_MIN})')
    parser.add_argument('--gradient-max', type=float, default=DEFAULT_GRADIENT_MAX,
                       help=f'Upper bound for color gradient (default: {DEFAULT_GRADIENT_MAX})')
    args = parser.parse_args()
    
    n = args.n
    gradient_min = args.gradient_min
    gradient_max = args.gradient_max
    
    print("="*80)
    print(f"NBA DEFENSIVE DISRUPTORS VISUALIZATION (Top/Bottom {n})")
    print("="*80)
    print(f"Gradient bounds: {gradient_min} to {gradient_max}\n")
    
    # Read the CSV from analyze_defensive_disruptors.py
    csv_file = repo_root / 'data/04_output/nba/defensive_disruptors_2025_26.csv'
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Run analyze_defensive_disruptors.py first!")
        return
    
    print(f"📂 Reading: {csv_file.name}\n")
    df = pd.read_csv(csv_file)
    
    print(f"   📊 Loaded {len(df)} players")
    print(f"   📊 Columns: {list(df.columns)}\n")
    
    # Prepare data
    top_n, bottom_n = prepare_data_for_visualization(df, n=n)
    
    # Create table using R's gt package
    output_path = create_gt_table_with_r(top_n, bottom_n, n=n, 
                                         gradient_min=gradient_min, 
                                         gradient_max=gradient_max)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n🖼️  Output: {output_path}")
    print(f"\n🐦 Ready to post to Twitter!")
    print(f"   - Top {n} players who ELEVATE team defense")
    print(f"   - Bottom {n} players who HURT team defense")
    print(f"   - Gradient shows defensive impact\n")
    
    # Auto-open the PNG
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(output_path)])
            print("📂 Opening PNG...\n")
        elif platform.system() == 'Windows':
            subprocess.run(['start', str(output_path)], shell=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(output_path)])
    except Exception as e:
        print(f"⚠️  Could not auto-open: {e}")
        print(f"   Run: open {output_path}\n")


if __name__ == "__main__":
    main()

