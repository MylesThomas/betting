"""
Create a FiveThirtyEight-style side-by-side visualization of NFL championship futures using R's gt package.

Context:
Side-by-side layout using R's gt + magick packages:
- Left table: Top 16 teams (best odds)
- Right table: Bottom 16 teams (longest shots)
Using rpy2 for publication-quality side-by-side tables.

Architecture:
1. Load and prepare data in Python (pandas)
   - Read CSV with fair odds calculations
   - Split into top 16 and bottom 16 teams
   - Format display columns (best odds, fair odds, vig %, etc.)
   
2. Create TWO separate gt tables in R
   - Left table: Top 16 teams with full title/subtitle
   - Right table: Bottom 16 teams with simpler header
   - Both use 538-style formatting
   
3. Combine horizontally using magick
   - Save left.png and right.png
   - Use magick::image_append() to combine
   - Clean up intermediate files
   
4. Export as high-resolution PNG
   - Output ready for social media sharing

Purpose:
- Create side-by-side comparison of NFL futures
- Show both favorites and longshots in single image
- Use R's gt + gtExtras + magick packages

Key Configuration Notes:
- LOGO_HEIGHT controls the row height (larger logos = taller rows)
- DATA_ROW_PADDING_PX controls spacing within rows (smaller = more compact)
- Each table shows 16 teams for balanced layout

Installation:
    # 1. Install R (if not already installed)
    # macOS:
    brew install r
    
    # 2. Install required R packages
    # Open R console and run:
    R
    install.packages("gt")
    install.packages("gtExtras")
    install.packages("tidyverse")
    install.packages("webshot2")
    quit()
    
    # 3. Install Python packages
    pip install rpy2 pandas

Requirements:
    - R (with gt, gtExtras, tidyverse, webshot2 packages)
    - Python: rpy2, pandas

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/viz_nfl_futures_gt_sidebyside.py
"""

import pandas as pd
from pathlib import Path
import sys
import subprocess
import platform
from datetime import datetime

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
CURRENT_NFL_SEASON = 2025
CURRENT_NFL_WEEK = 16
CURRENT_NFL_DATE = datetime.now()

# -----------------------------------------------------------------------------
# Titles and Footer Text
# -----------------------------------------------------------------------------
# Left table (top 16)
TITLE_LEFT = "NFL Super Bowl Futures: True Odds vs. What Books Charge"
SUBTITLE_LEFT = "Bookmakers charge an *average 18.5% vig* on championship futures (vs. 4-5% on game lines)"

# Right table (bottom 16)
TITLE_RIGHT = "Bottom 16 Teams"
SUBTITLE_RIGHT = ""

FOOTER_NOTES = """
1. 'Implied %' includes bookmaker vig. 'Fair %' is the true probability with vig removed (fair probabilities sum to exactly 100%).  
2. Color indicates vig level: green = low vig, red = high vig, yellow = negative vig (bettor advantage).  
3. 19 of 32 teams have odds available — bookmakers no longer offer odds on eliminated/longshot teams.
"""
FOOTER_DATA_SOURCE = "The Odds API"
FOOTER_DATA_DATE = datetime.now().strftime("%B %d, %Y")  # Auto-generate today's date

# -----------------------------------------------------------------------------
# Output Settings (Image Dimensions & Quality)
# -----------------------------------------------------------------------------
OUTPUT_FILENAME = "futures_vig_sidebyside.png"
LEFT_FILENAME = "futures_vig_left.png"
RIGHT_FILENAME = "futures_vig_right.png"

# Side-by-side: each table is narrower, combine horizontally
TABLE_WIDTH = 1200   # pixels per table
TABLE_HEIGHT = 1800  # pixels per table (16 rows each)
OUTPUT_DPI = 300

# -----------------------------------------------------------------------------
# Color Palette (Vig % Gradient)
# -----------------------------------------------------------------------------
# Conditional coloring:
# - Negative vig (< 0): Yellow (bettor advantage!)
# - Positive/zero vig (>= 0): Green -> White -> Red gradient
COLOR_PALETTE = ["#90EE90", "#ffffff", "#ffcccc", "#ff9999", "#d62728"]  # green -> white -> red
VIG_COLOR_DOMAIN_MIN = 0.0    # Start gradient at 0%
VIG_COLOR_DOMAIN_MAX = 5.0    # End gradient at 5%
NEGATIVE_VIG_COLOR = "#ffeb3b"  # Bright yellow for negative vig (override) 
...
# -----------------------------------------------------------------------------
# Typography
# -----------------------------------------------------------------------------
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 28
SUBTITLE_FONT_SIZE = 16
HEADER_FONT_SIZE = 14
BODY_FONT_SIZE = 13
FOOTER_FONT_SIZE = 11

# -----------------------------------------------------------------------------
# Spacing & Padding
# -----------------------------------------------------------------------------
# NOTE: LOGO_HEIGHT is the PRIMARY control for row height!
# Larger logos = taller rows. Adjust this first for compactness.
LOGO_HEIGHT = 21           # pixels - controls row height (images drive row size)
HEADER_PADDING_PX = 1      # Padding around column headers
DATA_ROW_PADDING_PX = 1    # Padding around data rows (smaller = more compact)
HEADING_PADDING_PX = 10    # Padding around title/subtitle

# -----------------------------------------------------------------------------
# Column Widths (pixels)
# -----------------------------------------------------------------------------
COL_WIDTH_RANK = 70
COL_WIDTH_TEAM = 200
COL_WIDTH_LOGO = 60
COL_WIDTH_RECORD = 90
COL_WIDTH_AVG_ODDS = 110
COL_WIDTH_IMPLIED_PCT = 110
COL_WIDTH_FAIR_ODDS = 120
COL_WIDTH_FAIR_PCT = 100
COL_WIDTH_VIG_PCT = 100
COL_WIDTH_BEST_BOOK = 140
COL_WIDTH_BEST_ODDS = 110
COL_WIDTH_BEST_VIG_PCT = 100


def get_team_logos():
    """Get NFL team logos from ESPN - all 32 teams"""
    logo_map = {
        # Teams with odds
        'Los Angeles Rams': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
        'Seattle Seahawks': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
        'Denver Broncos': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
        'Buffalo Bills': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
        'Philadelphia Eagles': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
        'Houston Texans': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
        'Green Bay Packers': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
        'New England Patriots': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
        'Jacksonville Jaguars': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
        'San Francisco 49ers': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
        'Baltimore Ravens': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
        'Los Angeles Chargers': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
        'Detroit Lions': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
        'Chicago Bears': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
        'Tampa Bay Buccaneers': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
        'Pittsburgh Steelers': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
        'Carolina Panthers': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
        'Indianapolis Colts': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
        'Dallas Cowboys': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
        # Teams without odds (eliminated/longshots)
        'Kansas City Chiefs': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
        'Minnesota Vikings': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
        'Washington Commanders': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
        'Atlanta Falcons': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
        'Arizona Cardinals': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
        'Miami Dolphins': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
        'Cincinnati Bengals': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
        'New Orleans Saints': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
        'New York Jets': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
        'Cleveland Browns': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
        'Tennessee Titans': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
        'Las Vegas Raiders': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
        'New York Giants': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
    }
    return logo_map


def prepare_data_for_visualization(df, logo_map):
    """
    Prepare the dataframe with all display columns (same logic as v2).
    
    Args:
        df: Raw dataframe from nfl_championship_fair_odds.csv
        
    Returns:
        DataFrame ready for visualization with formatted columns
    """
    print("📊 Preparing data for visualization...\n")
    
    df_display = df.copy()
    
    # Determine which teams have odds available
    df_display['has_odds'] = df_display['num_books'] > 0
    
    # Best odds string (American odds format)
    df_display['best_odds_str'] = df_display.apply(
        lambda row: '-' if not row['has_odds'] 
        else (f"+{int(row['best_odds'])}" if row['best_odds'] > 0 else str(int(row['best_odds']))),
        axis=1
    )
    
    # Calculate average odds from implied_prob_avg
    from odds_utils import probability_to_american_odds, american_odds_to_percentage_probability
    df_display['avg_odds'] = df_display.apply(
        lambda row: 100000 if not row['has_odds']
        else probability_to_american_odds(row['implied_prob_avg'] * 100),
        axis=1
    )
    
    # Average odds string
    df_display['avg_odds_str'] = df_display.apply(
        lambda row: '-' if not row['has_odds']
        else (f"+{int(row['avg_odds'])}" if row['avg_odds'] > 0 else str(int(row['avg_odds']))),
        axis=1
    )
    
    # Calculate Best Vig % (vig on the best odds specifically)
    df_display['best_vig_diff'] = df_display.apply(
        lambda row: None if not row['has_odds']
        else (american_odds_to_percentage_probability(row['best_odds']) / 100 - row['fair_prob']) * 100,
        axis=1
    )
    
    # Fair odds string (American odds format)
    df_display['fair_odds_str'] = df_display.apply(
        lambda row: '+100000' if not row['has_odds']
        else (f"+{int(row['fair_odds'])}" if row['fair_odds'] > 0 else str(int(row['fair_odds']))),
        axis=1
    )
    
    # Implied % (bookmaker's price including vig)
    df_display['implied_pct'] = df_display.apply(
        lambda row: 0.0 if not row['has_odds'] else (row['implied_prob_avg'] * 100),
        axis=1
    ).round(1)
    
    # Fair % (true probability with vig removed)
    df_display['fair_pct'] = df_display.apply(
        lambda row: 0.0 if not row['has_odds'] else (row['fair_prob'] * 100),
        axis=1
    ).round(1)
    
    df_display['fair_pct_str'] = df_display.apply(
        lambda row: '<0.1' if not row['has_odds'] else str(round(row['fair_prob'] * 100, 1)),
        axis=1
    )
    
    # Calculate vig difference (the "tax" bookmakers charge)
    df_display['vig_diff'] = df_display.apply(
        lambda row: None if not row['has_odds'] 
        else (row['implied_prob_avg'] - row['fair_prob']) * 100,
        axis=1
    )
    
    # Format vig_diff for display
    df_display['vig_diff_str'] = df_display.apply(
        lambda row: '-' if pd.isna(row['vig_diff']) or row['vig_diff'] is None
        else (f"+{row['vig_diff']:.1f}" if row['vig_diff'] > 0 else f"{row['vig_diff']:.1f}"),
        axis=1
    )
    
    # Best book display
    df_display['best_book_display'] = df_display['best_book'].fillna('-')
    
    # Add rank column
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Format implied_pct for display
    df_display['implied_pct_str'] = df_display.apply(
        lambda row: '0.0' if not row['has_odds'] else f"{row['implied_pct']:.1f}",
        axis=1
    )
    
    # Add team logo URLs
    df_display['logo_url'] = df_display['team'].map(logo_map)
    
    print(f"   ✅ Prepared {len(df_display)} teams")
    print(f"   ✅ {df_display['has_odds'].sum()} teams have odds available")
    print(f"   ✅ {(~df_display['has_odds']).sum()} teams eliminated/no odds")
    print(f"   ✅ {df_display['logo_url'].notna().sum()} teams have logo URLs\n")
    
    return df_display


def create_gt_table_with_r(df_display):
    """
    Create a side-by-side publication-quality table using R's gt + magick packages via rpy2.
    
    Creates two tables:
    - Left: Top 16 teams (best odds) with full title/subtitle
    - Right: Bottom 16 teams (longest shots) with simpler header
    
    Args:
        df_display: Prepared dataframe with all display columns (32 rows)
        
    Returns:
        Path to saved combined PNG file
    """
    print("🎨 Creating SIDE-BY-SIDE tables with R's gt + magick packages...\n")
    
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
    
    # Select columns for display (reordered: Rank, Team, W-L, Fair Odds/%, Avg Odds/Implied%/Vig%, Best Book/Odds/Vig%)
    # Keep vig_diff and best_vig_diff for both display AND coloring
    table_df = df_display[[
        'rank', 'team', 'logo_url', 'record', 
        'fair_odds_str', 'fair_pct_str',
        'avg_odds_str', 'implied_pct_str', 'vig_diff',
        'best_book_display', 'best_odds_str', 'best_vig_diff',
        'has_odds'
    ]].copy()
    
    # Rename columns for display
    table_df.columns = [
        'Rank', 'Team', 'logo_url', 'W-L', 
        'Fair Odds', 'Fair %',
        'Avg Odds', 'Implied %', 'Vig %',
        'Best Book', 'Best Odds', 'Best Vig %',
        'has_odds_flag'
    ]
    
    print(f"   📋 Full table dimensions: {table_df.shape}")
    print(f"   📋 Columns: {list(table_df.columns)}\n")
    
    # Split into top 16 and bottom 16
    table_df_left = table_df.iloc[:16].copy()
    table_df_right = table_df.iloc[16:].copy()
    
    print(f"   📊 Left table (top 16): {table_df_left.shape}")
    print(f"   📊 Right table (bottom 16): {table_df_right.shape}\n")
    
    # Convert both to R dataframes
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_left = ro.conversion.py2rpy(table_df_left)
        r_df_right = ro.conversion.py2rpy(table_df_right)
    
    ro.globalenv['nfl_data_left'] = r_df_left
    ro.globalenv['nfl_data_right'] = r_df_right
    
    # Output paths
    output_dir = repo_root / 'content/viz/nfl'
    left_path = output_dir / LEFT_FILENAME
    right_path = output_dir / RIGHT_FILENAME
    final_path = output_dir / OUTPUT_FILENAME
    
    left_path_str = str(left_path)
    right_path_str = str(right_path)
    final_path_str = str(final_path)
    
    print(f"   💾 Left output: {LEFT_FILENAME}")
    print(f"   💾 Right output: {RIGHT_FILENAME}")
    print(f"   💾 Combined output: {OUTPUT_FILENAME}\n")
    
    # R code to create TWO tables and combine them side-by-side
    r_code = f"""
    # Set library path to user library
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    library(magick)
    
    # ==========================================================================
    # LEFT TABLE (Top 16 teams) - Full title/subtitle + footer
    # ==========================================================================
    table_left <- nfl_data_left %>%
      select(-has_odds_flag) %>%
      gt() %>%
      
      # Add team logos using gtExtras
      gt_img_rows(columns = logo_url, height = {LOGO_HEIGHT}) %>%
      
      # Title and subtitle - 538 style
      tab_header(
        title = md("**{TITLE_LEFT}**"),
        subtitle = md("{SUBTITLE_LEFT}")
      ) %>%
      
      # Column alignment
      cols_align(
        align = "center",
        columns = everything()
      ) %>%
      cols_align(
        align = "left",
        columns = c(Team)
      ) %>%
      
      # Format Vig % column (add + sign for positive, % symbol, handle NA)
      fmt(
        columns = `Vig %`,
        fns = function(x) {{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}
      ) %>%
      
      # Format Best Vig % column (add + sign for positive, % symbol, handle NA)
      fmt(
        columns = `Best Vig %`,
        fns = function(x) {{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}
      ) %>%
      
      # Column widths (reordered columns)
      cols_width(
        Rank ~ px({COL_WIDTH_RANK}),
        Team ~ px({COL_WIDTH_TEAM}),
        logo_url ~ px({COL_WIDTH_LOGO}),
        `W-L` ~ px({COL_WIDTH_RECORD}),
        `Fair Odds` ~ px({COL_WIDTH_FAIR_ODDS}),
        `Fair %` ~ px({COL_WIDTH_FAIR_PCT}),
        `Avg Odds` ~ px({COL_WIDTH_AVG_ODDS}),
        `Implied %` ~ px({COL_WIDTH_IMPLIED_PCT}),
        `Vig %` ~ px({COL_WIDTH_VIG_PCT}),
        `Best Book` ~ px({COL_WIDTH_BEST_BOOK}),
        `Best Odds` ~ px({COL_WIDTH_BEST_ODDS}),
        `Best Vig %` ~ px({COL_WIDTH_BEST_VIG_PCT})
      ) %>%
      
      # Rename logo_url column header to empty
      cols_label(
        logo_url = ""
      ) %>%
      
      # Style headers - bold and larger
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px({HEADER_FONT_SIZE}), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Style title - 538 aesthetic
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}",
          size = px({TITLE_FONT_SIZE}),
          weight = "bold",
          color = "#2c3e50"
        ),
        locations = cells_title(groups = "title")
      ) %>%
      
      # Style subtitle
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}", 
          size = px({SUBTITLE_FONT_SIZE}),
          color = "#555555"
        ),
        locations = cells_title(groups = "subtitle")
      ) %>%
      
      # Conditional formatting for Vig % column (average vig)
      # Green -> White -> Red gradient for positive/zero vig
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({VIG_COLOR_DOMAIN_MIN}, {VIG_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Conditional formatting for Best Vig % column
      # Green -> White -> Red gradient for positive/zero vig
      data_color(
        columns = `Best Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({VIG_COLOR_DOMAIN_MIN}, {VIG_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Override negative vig values with YELLOW (bettor advantage!)
      # Apply to Vig % column independently
      tab_style(
        style = cell_fill(color = "{NEGATIVE_VIG_COLOR}"),
        locations = cells_body(
          columns = `Vig %`,
          rows = `Vig %` < 0
        )
      ) %>%
      
      # Apply to Best Vig % column independently
      tab_style(
        style = cell_fill(color = "{NEGATIVE_VIG_COLOR}"),
        locations = cells_body(
          columns = `Best Vig %`,
          rows = `Best Vig %` < 0
        )
      ) %>%
      
      # Make rank column bold
      tab_style(
        style = cell_text(weight = "bold", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Rank)
      ) %>%
      
      # Make team names bold
      tab_style(
        style = cell_text(weight = "600", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Team)
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
        table.background.color = "#f8f9fa",
        row.striping.background_color = "#f0f0f0",
        source_notes.font.size = px({FOOTER_FONT_SIZE}),
        source_notes.padding = px(10)
      ) %>%
      
      # Footer notes - consolidated into single block with tight spacing
      tab_source_note(
        source_note = md("{FOOTER_NOTES}")
      ) %>%
      tab_source_note(
        source_note = md("**Data:** {FOOTER_DATA_SOURCE} ({FOOTER_DATA_DATE}) | **Analysis:** {TWITTER_HANDLE}")
      )
    
    # ==========================================================================
    # RIGHT TABLE (Bottom 16 teams) - Simpler header + right-aligned + left border
    # ==========================================================================
    table_right <- nfl_data_right %>%
      select(-has_odds_flag) %>%
      gt() %>%
      
      # Add team logos using gtExtras
      gt_img_rows(columns = logo_url, height = {LOGO_HEIGHT}) %>%
      
      # Simpler title for right side
      tab_header(
        title = md("**{TITLE_RIGHT}**"),
        subtitle = md("{SUBTITLE_RIGHT}")
      ) %>%
      
      # Column alignment
      cols_align(
        align = "center",
        columns = everything()
      ) %>%
      cols_align(
        align = "left",
        columns = c(Team)
      ) %>%
      
      # Format Vig % column (add + sign for positive, % symbol, handle NA)
      fmt(
        columns = `Vig %`,
        fns = function(x) {{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}
      ) %>%
      
      # Format Best Vig % column (add + sign for positive, % symbol, handle NA)
      fmt(
        columns = `Best Vig %`,
        fns = function(x) {{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}
      ) %>%
      
      # Column widths (same as left table)
      cols_width(
        Rank ~ px({COL_WIDTH_RANK}),
        Team ~ px({COL_WIDTH_TEAM}),
        logo_url ~ px({COL_WIDTH_LOGO}),
        `W-L` ~ px({COL_WIDTH_RECORD}),
        `Fair Odds` ~ px({COL_WIDTH_FAIR_ODDS}),
        `Fair %` ~ px({COL_WIDTH_FAIR_PCT}),
        `Avg Odds` ~ px({COL_WIDTH_AVG_ODDS}),
        `Implied %` ~ px({COL_WIDTH_IMPLIED_PCT}),
        `Vig %` ~ px({COL_WIDTH_VIG_PCT}),
        `Best Book` ~ px({COL_WIDTH_BEST_BOOK}),
        `Best Odds` ~ px({COL_WIDTH_BEST_ODDS}),
        `Best Vig %` ~ px({COL_WIDTH_BEST_VIG_PCT})
      ) %>%
      
      # Rename logo_url column header to empty
      cols_label(
        logo_url = ""
      ) %>%
      
      # Style headers - bold and larger
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px({HEADER_FONT_SIZE}), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Style title - 538 aesthetic
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}",
          size = px({TITLE_FONT_SIZE}),
          weight = "bold",
          color = "#2c3e50"
        ),
        locations = cells_title(groups = "title")
      ) %>%
      
      # Style subtitle
      tab_style(
        style = cell_text(
          font = "{FONT_FAMILY}", 
          size = px({SUBTITLE_FONT_SIZE}),
          color = "#555555"
        ),
        locations = cells_title(groups = "subtitle")
      ) %>%
      
      # Conditional formatting for Vig % column (average vig)
      # Green -> White -> Red gradient for positive/zero vig
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({VIG_COLOR_DOMAIN_MIN}, {VIG_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Conditional formatting for Best Vig % column
      # Green -> White -> Red gradient for positive/zero vig
      data_color(
        columns = `Best Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({VIG_COLOR_DOMAIN_MIN}, {VIG_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Override negative vig values with YELLOW (bettor advantage!)
      # Apply to Vig % column independently
      tab_style(
        style = cell_fill(color = "{NEGATIVE_VIG_COLOR}"),
        locations = cells_body(
          columns = `Vig %`,
          rows = `Vig %` < 0
        )
      ) %>%
      
      # Apply to Best Vig % column independently
      tab_style(
        style = cell_fill(color = "{NEGATIVE_VIG_COLOR}"),
        locations = cells_body(
          columns = `Best Vig %`,
          rows = `Best Vig %` < 0
        )
      ) %>%
      
      # Make rank column bold
      tab_style(
        style = cell_text(weight = "bold", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Rank)
      ) %>%
      
      # Make team names bold
      tab_style(
        style = cell_text(weight = "600", size = px({BODY_FONT_SIZE})),
        locations = cells_body(columns = Team)
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
        table.background.color = "#f8f9fa",
        row.striping.background_color = "#f0f0f0",
        source_notes.font.size = px({FOOTER_FONT_SIZE}),
        source_notes.padding = px(10),
        column_labels.background.color = "white",
        column_labels.font.weight = "bold"
      )
    
    # ==========================================================================
    # SAVE BOTH TABLES
    # ==========================================================================
    gtsave(table_left, "{left_path_str}", vwidth = {TABLE_WIDTH}, vheight = {TABLE_HEIGHT})
    gtsave(table_right, "{right_path_str}", vwidth = {TABLE_WIDTH}, vheight = {TABLE_HEIGHT})
    
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
    """Main visualization function - V4 SIDE-BY-SIDE"""
    
    print("="*80)
    print("NFL CHAMPIONSHIP FUTURES - V4 (SIDE-BY-SIDE LAYOUT)")
    print("="*80 + "\n")
    
    # Read the CSV (same as v2)
    csv_file = repo_root / 'data/04_output/nfl/nfl_championship_fair_odds.csv'
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Run analyze_nfl_championship_futures_vig.py first!")
        return
    
    print(f"📂 Reading: {csv_file.name}\n")
    df = pd.read_csv(csv_file)
    
    print(f"   📊 Loaded {len(df)} teams")
    print(f"   📊 Columns: {list(df.columns)}\n")
    
    # Get team logos
    logo_map = get_team_logos()
    print(f"   🏈 Loaded {len(logo_map)} team logos\n")
    
    # Prepare data (same logic as v2)
    df_display = prepare_data_for_visualization(df, logo_map)
    
    # Create table using R's gt package
    output_path = create_gt_table_with_r(df_display)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n🖼️  Output: {output_path}\n")
    
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

