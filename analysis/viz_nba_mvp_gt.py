"""
Create a FiveThirtyEight-style visualization of NBA MVP odds using R's gt package.

Context:
Adapted from championship futures visualization for MVP award.
Thomas wants a clean, professional table showing FanDuel odds vs fair odds.
Player headshots could be added later if desired.

Architecture:
1. Load and prepare data in Python (pandas)
   - Read CSV with fair odds calculations
   - Format display columns (FanDuel odds, fair odds, vig %, etc.)
   - Calculate conditional formatting values
   
2. Generate R script
   - Create gt table with 538-style formatting
   - Apply conditional color formatting for Vig % column
   - Style headers, borders, and zebra striping
   
3. Execute R script via subprocess
   - Save as high-resolution PNG
   - Output ready for social media sharing

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/viz_nba_mvp_gt.py

Input:
    data/04_output/nba/mvp/nba_mvp_fair_odds_YYYYMMDD_HHMMSS.csv (latest)

Output:
    content/viz/nba/nba_mvp_vig.png

Requirements:
    - R (with gt, dplyr, webshot2 packages)
    - Python: pandas
"""

import pandas as pd
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import glob
import platform

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
CURRENT_NBA_SEASON = "2024-25"

# -----------------------------------------------------------------------------
# Titles and Footer Text
# -----------------------------------------------------------------------------
TITLE = "NBA MVP Odds: True Odds vs. What FanDuel Charges"
# SUBTITLE is generated dynamically with calculated avg vig

FOOTER_NOTES = """
1. 'Implied %' includes bookmaker vig. 'Fair %' is the true probability with vig removed (fair probabilities sum to exactly 100%).  
2. Color indicates vig level: green = low vig (bettor advantage), red = high vig (house edge).
"""
FOOTER_DATA_SOURCE = "FanDuel Sportsbook"
FOOTER_DATA_DATE = "January 7, 2025"  # Update when you update odds

# -----------------------------------------------------------------------------
# Output Settings (Image Dimensions & Quality)
# -----------------------------------------------------------------------------
OUTPUT_FILENAME = "nba_mvp_vig.png"

# Smaller table (only ~12 players vs 30 teams)
OUTPUT_WIDTH = 1400   # pixels
OUTPUT_HEIGHT = 1200  # pixels
OUTPUT_DPI = 300

# -----------------------------------------------------------------------------
# Color Palette (Vig % Gradient)
# -----------------------------------------------------------------------------
# Green -> White -> Red gradient for vig
COLOR_PALETTE = ["#4CAF50", "#90EE90", "#ffffff", "#ffcccc", "#d62728"]  # green -> white -> red
VIG_COLOR_DOMAIN_MIN = 0.0    # Start gradient at 0%
VIG_COLOR_DOMAIN_MAX = 10.0   # End gradient at 10%

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
HEADER_PADDING_PX = 8
DATA_ROW_PADDING_PX = 6
HEADING_PADDING_PX = 10

# -----------------------------------------------------------------------------
# Column Widths (pixels)
# -----------------------------------------------------------------------------
COL_WIDTH_RANK = 70
COL_WIDTH_PLAYER = 250
COL_WIDTH_FANDUEL_ODDS = 120
COL_WIDTH_IMPLIED_PCT = 110
COL_WIDTH_FAIR_ODDS = 120
COL_WIDTH_FAIR_PCT = 100
COL_WIDTH_VIG_PCT = 100


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_latest_fair_odds():
    """Load the most recent fair odds CSV"""
    input_dir = repo_root / 'data/04_output/nba/mvp'
    
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Run analyze_nba_mvp_vig.py first!"
        )
    
    csv_files = list(input_dir.glob('nba_mvp_fair_odds_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No fair odds files found in {input_dir}\n"
            "Run analyze_nba_mvp_vig.py first!"
        )
    
    latest_file = max(csv_files, key=os.path.getmtime)
    
    print(f"📂 Loading: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file


def prepare_data_for_visualization(df):
    """
    Prepare the dataframe with all display columns.
    
    Args:
        df: Raw dataframe from nba_mvp_fair_odds.csv
        
    Returns:
        tuple: (DataFrame ready for visualization, average_vig_pct)
    """
    print("📊 Preparing data for visualization...\n")
    
    df_display = df.copy()
    
    # Add rank column
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Format odds as strings with + sign for positive
    df_display['fanduel_odds_str'] = df_display['fanduel_odds'].apply(
        lambda x: f"{int(x):+d}"
    )
    df_display['fair_odds_str'] = df_display['fair_odds'].apply(
        lambda x: f"{int(x):+d}"
    )
    
    # Format percentages
    df_display['implied_pct_str'] = df_display['fanduel_implied_prob'].apply(
        lambda x: f"{x*100:.1f}%"
    )
    df_display['fair_pct_str'] = df_display['fair_prob'].apply(
        lambda x: f"{x*100:.1f}%"
    )
    df_display['vig_pct_str'] = df_display['vig_pct'].apply(
        lambda x: f"{x:+.1f}%"
    )
    
    # Calculate average vig
    average_vig_pct = df_display['vig_pct'].mean()
    
    print(f"   ✅ Prepared {len(df_display)} players")
    print(f"   ✅ Average vig: {average_vig_pct:.1f}%\n")
    
    return df_display, average_vig_pct


def create_gt_table_with_r(df_display, average_vig_pct):
    """
    Create a publication-quality table using R's gt package.
    
    Args:
        df_display: Prepared dataframe with all display columns
        average_vig_pct: Calculated average vig percentage
        
    Returns:
        Path to saved PNG file
    """
    print("🎨 Creating table with R's gt package...\n")
    
    # Generate subtitle dynamically with calculated vig
    subtitle = f"FanDuel charges {average_vig_pct:.1f}% vig on the MVP market (vs. 4-5% on game lines)"
    
    # Select columns for display
    table_df = df_display[[
        'rank', 'player', 'fanduel_odds_str', 'implied_pct_str',
        'fair_odds_str', 'fair_pct_str', 'vig_pct', 'vig_pct_str'
    ]].copy()
    
    # Rename columns for display
    table_df.columns = [
        'Rank', 'Player', 'FanDuel Odds', 'Implied %',
        'Fair Odds', 'Fair %', 'vig_pct_num', 'Vig %'
    ]
    
    print(f"   📋 Table dimensions: {table_df.shape}")
    print(f"   📋 Columns: {list(table_df.columns)}\n")
    
    # Output path
    output_path = repo_root / 'content/viz/nba' / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_str = str(output_path)
    
    print(f"   💾 Output path: {output_path.name}")
    
    # Save table_df to temp CSV for R to read
    temp_csv = repo_root / 'temp_mvp_data.csv'
    table_df.to_csv(temp_csv, index=False)
    
    # R code to create the table
    r_code = f"""
    # Set library path to user library
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(dplyr)
    
    # Read data (check.names=FALSE preserves column names, stringsAsFactors=FALSE keeps strings as strings)
    mvp_data <- read.csv("{str(temp_csv)}", check.names=FALSE, stringsAsFactors=FALSE, colClasses=c(
      "Rank"="integer",
      "Player"="character",
      "FanDuel Odds"="character",
      "Implied %"="character",
      "Fair Odds"="character",
      "Fair %"="character",
      "vig_pct_num"="numeric",
      "Vig %"="character"
    ))
    
    # Create gt table with 538-style formatting
    table <- mvp_data %>%
      gt() %>%
      
      # Title and subtitle
      tab_header(
        title = md("**{TITLE}**"),
        subtitle = md("{subtitle}")
      ) %>%
      
      # Hide numeric vig column (used for coloring only)
      cols_hide(columns = c(vig_pct_num)) %>%
      
      # Column alignment
      cols_align(
        align = "center",
        columns = everything()
      ) %>%
      cols_align(
        align = "left",
        columns = c(Player)
      ) %>%
      
      # Column widths
      cols_width(
        Rank ~ px({COL_WIDTH_RANK}),
        Player ~ px({COL_WIDTH_PLAYER}),
        `FanDuel Odds` ~ px({COL_WIDTH_FANDUEL_ODDS}),
        `Implied %` ~ px({COL_WIDTH_IMPLIED_PCT}),
        `Fair Odds` ~ px({COL_WIDTH_FAIR_ODDS}),
        `Fair %` ~ px({COL_WIDTH_FAIR_PCT}),
        `Vig %` ~ px({COL_WIDTH_VIG_PCT})
      ) %>%
      
      # Style headers
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px({HEADER_FONT_SIZE}), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Style title
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
      
      # Conditional formatting for Vig % column
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({VIG_COLOR_DOMAIN_MIN}, {VIG_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
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
      
      # Zebra striping
      opt_row_striping(row_striping = TRUE) %>%
      
      # Table options
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
      
      # Footer notes
      tab_source_note(
        source_note = md("{FOOTER_NOTES}")
      ) %>%
      tab_source_note(
        source_note = md("**Data:** {FOOTER_DATA_SOURCE} ({FOOTER_DATA_DATE}) | **Analysis:** {TWITTER_HANDLE}")
      )
    
    # Save as PNG
    gtsave(table, "{output_path_str}", vwidth = {OUTPUT_WIDTH}, vheight = {OUTPUT_HEIGHT})
    
    cat("✅ Table saved successfully!\\n")
    """
    
    print("   🔧 Executing R code...\n")
    
    # Save R script to temp file
    temp_r_file = repo_root / 'temp_viz_mvp.R'
    with open(temp_r_file, 'w') as f:
        f.write(r_code)
    
    try:
        result = subprocess.run(
            ['Rscript', str(temp_r_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"\n   ✅ Table created and saved!\n")
            return output_path
        else:
            print(f"❌ Error creating table in R:")
            print(result.stderr)
            print("\n💡 Make sure R packages are installed:")
            print("   Rscript -e 'install.packages(c(\"gt\", \"dplyr\", \"webshot2\"), repos=\"https://cran.rstudio.com/\")'")
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("   ❌ R script timed out after 60 seconds")
        sys.exit(1)
    except FileNotFoundError:
        print("   ❌ Rscript not found. Is R installed?")
        print("      Install: brew install r")
        sys.exit(1)
    finally:
        # Clean up temp files
        if temp_r_file.exists():
            temp_r_file.unlink()
        if temp_csv.exists():
            temp_csv.unlink()


def main():
    """Main visualization function"""
    
    print("="*80)
    print("NBA MVP VIG VISUALIZATION (R + GT PACKAGE)")
    print("="*80 + "\n")
    
    # Load fair odds
    print("1️⃣ Loading fair odds data...")
    
    try:
        df, source_file = load_latest_fair_odds()
        print(f"   ✅ Loaded {len(df)} players from {source_file.name}\n")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nRun analyze_nba_mvp_vig.py first!")
        return
    
    # Prepare data
    df_display, average_vig_pct = prepare_data_for_visualization(df)
    
    # Create table using R's gt package
    output_path = create_gt_table_with_r(df_display, average_vig_pct)
    
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
    
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Share on X/Twitter")
    print("2. Update MVP odds weekly:")
    print("   - Edit CURRENT_MVP_ODDS in scripts/fetch_nba_mvp_odds_fanduel.py")
    print("   - Update FETCH_DATE to current date")
    print("   - Re-run: python3 scripts/fetch_nba_mvp_odds_fanduel.py")
    print("   - Re-run: python3 analysis/analyze_nba_mvp_vig.py")
    print("   - Re-run: python3 analysis/viz_nba_mvp_gt.py\n")


if __name__ == "__main__":
    main()

