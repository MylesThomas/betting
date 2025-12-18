"""
Create a FiveThirtyEight-style visualization of NFL championship futures using R's gt package.

Context:
Single table visualization (all 32 teams) using R's gt package via rpy2.
Publication-quality table with proper spacing and layout.

Architecture:
1. Load and prepare data in Python (pandas)
   - Read CSV with fair odds calculations
   - Format display columns (best odds, fair odds, vig %, etc.)
   - Calculate conditional formatting values
   
2. Convert Python DataFrame to R via rpy2
   - Use localconverter with pandas2ri for seamless conversion
   - Pass data to R environment
   
3. Create visualization in R using gt package
   - Apply 538-style formatting (fonts, colors, spacing)
   - Add conditional color formatting for Vig % column
   - Style headers, borders, and zebra striping
   
4. Export as high-resolution PNG
   - Save using gtsave() function
   - Output ready for social media sharing

Purpose:
- Read the fair odds CSV (same as v2)
- Prepare data in Python (same logic as v2)
- Use R's gt + gtExtras packages for table creation
- Export as high-quality PNG for social media

Key Configuration Notes:
- LOGO_HEIGHT controls the row height (larger logos = taller rows)
- DATA_ROW_PADDING_PX controls spacing within rows (smaller = more compact)
- RATIO_TYPE controls aspect ratio: "portrait" (9:16) or "landscape" (16:9)

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
    python3 analysis/viz_nfl_futures_gt_single.py
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
TITLE = "NFL Super Bowl Futures: True Odds vs. What Books Charge"
SUBTITLE = "Bookmakers charge an *average 18.5% vig* on championship futures (vs. 4-5% on game lines)"

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
OUTPUT_FILENAME = "futures_vig_single.png"

# Aspect ratio settings
RATIO_TYPE = "portrait"  # "portrait" (9:16, taller) or "landscape" (16:9, wider)
RATIO = 16/9 if RATIO_TYPE == "portrait" else 9/16  # portrait = 16/9 multiplier, landscape = 9/16 multiplier
OUTPUT_WIDTH = 1600   # pixels
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * RATIO)  # 1600 * (16/9) = 2844 for portrait
OUTPUT_DPI = 300

# Portrait 9:16 = good for tall tables (32 rows)
# Landscape 16:9 = good for wide tables (NOT recommended here)

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
    Create a publication-quality table using R's gt package via rpy2.
    
    Args:
        df_display: Prepared dataframe with all display columns
        
    Returns:
        Path to saved PNG file
    """
    print("🎨 Creating table with R's gt package...\n")
    
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
        print("   3. Install R packages: R -e 'install.packages(c(\"gt\", \"gtExtras\", \"tidyverse\", \"webshot2\"))'")
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
    
    print(f"   📋 Table dimensions: {table_df.shape}")
    print(f"   📋 Columns: {list(table_df.columns)}\n")
    
    # Convert to R dataframe using localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(table_df)
    
    ro.globalenv['nfl_data'] = r_df
    
    # Output path
    output_path = repo_root / 'content/viz/nfl' / OUTPUT_FILENAME
    output_path_str = str(output_path)
    
    print(f"   💾 Output path: {output_path.name}")
    
    # R code to create the table
    r_code = f"""
    # Set library path to user library
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    # Create gt table with 538-style formatting
    table <- nfl_data %>%
      select(-has_odds_flag) %>%
      gt() %>%
      
      # Add team logos using gtExtras
      gt_img_rows(columns = logo_url, height = {LOGO_HEIGHT}) %>%
      
      # Title and subtitle - 538 style
      tab_header(
        title = md("**{TITLE}**"),
        subtitle = md("{SUBTITLE}")
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
    
    # Save as PNG with higher resolution
    gtsave(table, "{output_path_str}", vwidth = {OUTPUT_WIDTH}, vheight = {OUTPUT_HEIGHT})
    
    print("✅ Table saved successfully!")
    """
    
    print("   🔧 Executing R code...\n")
    
    try:
        ro.r(r_code)
        print(f"\n   ✅ Table created and saved!\n")
        return output_path
        
    except Exception as e:
        print(f"❌ Error creating table in R:")
        print(f"   {e}")
        print("\n💡 Make sure R packages are installed:")
        print("   R -e 'install.packages(c(\"gt\", \"gtExtras\", \"dplyr\", \"webshot2\"))'")
        sys.exit(1)


def main():
    """Main visualization function"""
    
    print("="*80)
    print("NFL CHAMPIONSHIP FUTURES - V3 (R + GT PACKAGE)")
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

