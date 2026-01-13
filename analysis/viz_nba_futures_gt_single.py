"""
Create a FiveThirtyEight-style visualization of NBA Championship futures using R's gt package.

Context:
Adapted from NFL futures visualization to support NBA. Thomas wants weekly posts for both NFL and NBA.
Single table visualization (all 30 NBA teams) using R's gt package via rpy2.
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
- Read the fair odds CSV from analyze_nba_championship_futures_vig.py
- Prepare data in Python
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
    python3 analysis/viz_nba_futures_gt_single.py
    
    # Show only top 15 teams
    python3 analysis/viz_nba_futures_gt_single.py --top-n 15
"""

import pandas as pd
from pathlib import Path
import sys
import subprocess
import platform
import argparse
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
CURRENT_NBA_SEASON = "2024-25"
CURRENT_NBA_DATE = datetime.now()

# -----------------------------------------------------------------------------
# Titles and Footer Text
# -----------------------------------------------------------------------------
TITLE = "NBA Championship Futures: True Odds vs. What Books Charge"
# SUBTITLE is generated dynamically in prepare_data_for_visualization() with calculated avg vig

def generate_footer_notes(total_teams, top_n):
    """Generate footer notes with optional filtering message."""
    base_notes = """
1. 'Implied %' includes bookmaker vig. 'Fair %' is the true probability with vig removed (fair probabilities sum to exactly 100%).  
2. Color indicates vig level: green = low vig, red = high vig, yellow = negative vig (bettor advantage).  
3. All 30 NBA teams shown — some may have very long odds due to poor season performance."""
    
    if top_n < total_teams:
        base_notes += f"""  
4. Filtered to top {top_n} teams by fair probability."""
    
    return base_notes

FOOTER_DATA_SOURCE = "The Odds API & ESPN"
FOOTER_DATA_DATE = datetime.now().strftime("%B %d, %Y")  # Auto-generate today's date

# -----------------------------------------------------------------------------
# Output Settings (Image Dimensions & Quality)
# -----------------------------------------------------------------------------
OUTPUT_FILENAME = "nba_futures_vig_single.png"

# Aspect ratio settings
RATIO_TYPE = "portrait"  # "portrait" (9:16, taller) or "landscape" (16:9, wider)
RATIO = 16/9 if RATIO_TYPE == "portrait" else 9/16  # portrait = 16/9 multiplier, landscape = 9/16 multiplier
OUTPUT_WIDTH = 1600   # pixels
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * RATIO)  # 1600 * (16/9) = 2844 for portrait
OUTPUT_DPI = 300

# Portrait 9:16 = good for tall tables (30 rows)
# Landscape 16:9 = good for wide tables (NOT recommended here)

# -----------------------------------------------------------------------------
# Color Palette (Vig % Gradient)
# -----------------------------------------------------------------------------
# Conditional coloring:
# - Negative vig (< 0): Yellow (bettor advantage!)
# - Positive/zero vig (>= 0): Green -> White -> Red gradient
COLOR_PALETTE = ["#90EE90", "#ffffff", "#ffcccc", "#ff9999", "#d62728"]  # green -> white -> red
VIG_COLOR_DOMAIN_MIN = 0.0    # Start gradient at 0%
VIG_COLOR_DOMAIN_MAX = 10.0   # End gradient at 10%
NEGATIVE_VIG_COLOR = "#ffeb3b"  # Bright yellow for negative vig (override)

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
HEADING_PADDING_PX = 3     # Padding around title/subtitle (reduced from 10)

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
    """Get NBA team logos from ESPN - all 30 teams"""
    logo_map = {
        'Atlanta Hawks': 'https://a.espncdn.com/i/teamlogos/nba/500/atl.png',
        'Boston Celtics': 'https://a.espncdn.com/i/teamlogos/nba/500/bos.png',
        'Brooklyn Nets': 'https://a.espncdn.com/i/teamlogos/nba/500/bkn.png',
        'Charlotte Hornets': 'https://a.espncdn.com/i/teamlogos/nba/500/cha.png',
        'Chicago Bulls': 'https://a.espncdn.com/i/teamlogos/nba/500/chi.png',
        'Cleveland Cavaliers': 'https://a.espncdn.com/i/teamlogos/nba/500/cle.png',
        'Dallas Mavericks': 'https://a.espncdn.com/i/teamlogos/nba/500/dal.png',
        'Denver Nuggets': 'https://a.espncdn.com/i/teamlogos/nba/500/den.png',
        'Detroit Pistons': 'https://a.espncdn.com/i/teamlogos/nba/500/det.png',
        'Golden State Warriors': 'https://a.espncdn.com/i/teamlogos/nba/500/gs.png',
        'Houston Rockets': 'https://a.espncdn.com/i/teamlogos/nba/500/hou.png',
        'Indiana Pacers': 'https://a.espncdn.com/i/teamlogos/nba/500/ind.png',
        'Los Angeles Clippers': 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/nba/500/lac.png',
        'Los Angeles Lakers': 'https://a.espncdn.com/i/teamlogos/nba/500/lal.png',
        'Memphis Grizzlies': 'https://a.espncdn.com/i/teamlogos/nba/500/mem.png',
        'Miami Heat': 'https://a.espncdn.com/i/teamlogos/nba/500/mia.png',
        'Milwaukee Bucks': 'https://a.espncdn.com/i/teamlogos/nba/500/mil.png',
        'Minnesota Timberwolves': 'https://a.espncdn.com/i/teamlogos/nba/500/min.png',
        'New Orleans Pelicans': 'https://a.espncdn.com/i/teamlogos/nba/500/no.png',
        'New York Knicks': 'https://a.espncdn.com/i/teamlogos/nba/500/ny.png',
        'Oklahoma City Thunder': 'https://a.espncdn.com/i/teamlogos/nba/500/okc.png',
        'Orlando Magic': 'https://a.espncdn.com/i/teamlogos/nba/500/orl.png',
        'Philadelphia 76ers': 'https://a.espncdn.com/i/teamlogos/nba/500/phi.png',
        'Phoenix Suns': 'https://a.espncdn.com/i/teamlogos/nba/500/phx.png',
        'Portland Trail Blazers': 'https://a.espncdn.com/i/teamlogos/nba/500/por.png',
        'Sacramento Kings': 'https://a.espncdn.com/i/teamlogos/nba/500/sac.png',
        'San Antonio Spurs': 'https://a.espncdn.com/i/teamlogos/nba/500/sa.png',
        'Toronto Raptors': 'https://a.espncdn.com/i/teamlogos/nba/500/tor.png',
        'Utah Jazz': 'https://a.espncdn.com/i/teamlogos/nba/500/utah.png',
        'Washington Wizards': 'https://a.espncdn.com/i/teamlogos/nba/500/wsh.png',
    }
    return logo_map


def prepare_data_for_visualization(df, logo_map):
    """
    Prepare the dataframe with all display columns.
    
    Args:
        df: Raw dataframe from nba_championship_fair_odds.csv
        logo_map: Dictionary of team names to logo URLs
        
    Returns:
        tuple: (DataFrame ready for visualization, average_vig_pct)
    """
    print("📊 Preparing data for visualization...\n")
    
    from odds_utils import probability_to_american_odds, american_odds_to_percentage_probability
    
    df_display = df.copy()
    
    # Add team logo URLs
    df_display['logo_url'] = df_display['team'].map(logo_map)
    
    # Best odds string (American odds format)
    df_display['best_odds_str'] = df_display.apply(
        lambda row: (f"+{int(row['best_odds'])}" if row['best_odds'] > 0 else str(int(row['best_odds']))),
        axis=1
    )
    
    # Calculate average odds from implied_prob_avg
    df_display['avg_odds'] = df_display.apply(
        lambda row: probability_to_american_odds(row['implied_prob_avg'] * 100),
        axis=1
    )
    
    # Average odds string
    df_display['avg_odds_str'] = df_display.apply(
        lambda row: (f"+{int(row['avg_odds'])}" if row['avg_odds'] > 0 else str(int(row['avg_odds']))),
        axis=1
    )
    
    # Calculate Best Vig % (vig on the best odds specifically)
    df_display['best_vig_diff'] = df_display.apply(
        lambda row: (american_odds_to_percentage_probability(row['best_odds']) / 100 - row['fair_prob']) * 100,
        axis=1
    )
    
    # Fair odds string (American odds format)
    df_display['fair_odds_str'] = df_display.apply(
        lambda row: (f"+{int(row['fair_odds'])}" if row['fair_odds'] > 0 else str(int(row['fair_odds']))),
        axis=1
    )
    
    # Implied % (bookmaker's price including vig)
    df_display['implied_pct'] = (df_display['implied_prob_avg'] * 100).round(1)
    
    # Fair % (true probability with vig removed)
    df_display['fair_pct'] = (df_display['fair_prob'] * 100).round(1)
    
    df_display['fair_pct_str'] = df_display['fair_pct'].apply(lambda x: str(round(x, 1)))
    
    # Calculate vig difference (the "tax" bookmakers charge)
    df_display['vig_diff'] = (df_display['implied_prob_avg'] - df_display['fair_prob']) * 100
    
    # Calculate average vig across all teams (for subtitle)
    average_vig_pct = df_display['vig_diff'].mean()
    
    # Format vig_diff for display
    df_display['vig_diff_str'] = df_display.apply(
        lambda row: (f"+{row['vig_diff']:.1f}" if row['vig_diff'] > 0 else f"{row['vig_diff']:.1f}"),
        axis=1
    )
    
    # Best book display
    df_display['best_book_display'] = df_display['best_book'].fillna('-')
    
    # Add rank column
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Format implied_pct for display
    df_display['implied_pct_str'] = df_display['implied_pct'].apply(lambda x: f"{x:.1f}")
    
    # Handle missing records
    df_display['record'] = df_display['record'].fillna('-')
    
    print(f"   ✅ Prepared {len(df_display)} teams")
    print(f"   ✅ {df_display['logo_url'].notna().sum()} teams have logo URLs")
    print(f"   ✅ Average vig: {average_vig_pct:.1f}%\n")
    
    return df_display, average_vig_pct


def create_gt_table_with_r(df_display, average_vig_pct, total_teams, top_n):
    """
    Create a publication-quality table using R's gt package via rpy2.
    
    Args:
        df_display: Prepared dataframe with all display columns
        average_vig_pct: Calculated average vig percentage across all teams
        total_teams: Total number of teams before filtering
        top_n: Number of teams to display (for footer note)
        
    Returns:
        Path to saved PNG file
    """
    print("🎨 Creating table with R's gt package...\n")
    
    # Generate subtitle dynamically with calculated vig
    subtitle = f"Bookmakers charge an *average {average_vig_pct:.1f}% vig* on championship futures (vs. 4-5% on game lines)"
    
    # Generate footer notes with optional filtering message
    FOOTER_NOTES = generate_footer_notes(total_teams, top_n)
    
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
    
    # Select columns for display
    table_df = df_display[[
        'rank', 'team', 'logo_url', 'record', 
        'fair_odds_str', 'fair_pct_str',
        'avg_odds_str', 'implied_pct_str', 'vig_diff',
        'best_book_display', 'best_odds_str', 'best_vig_diff'
    ]].copy()
    
    # Rename columns for display
    table_df.columns = [
        'Rank', 'Team', 'logo_url', 'W-L', 
        'Fair Odds', 'Fair %',
        'Avg Odds', 'Implied %', 'Vig %',
        'Best Book', 'Best Odds', 'Best Vig %'
    ]
    
    print(f"   📋 Table dimensions: {table_df.shape}")
    print(f"   📋 Columns: {list(table_df.columns)}\n")
    
    # Convert to R dataframe using localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(table_df)
    
    ro.globalenv['nba_data'] = r_df
    
    # Output path
    output_path = repo_root / 'content/viz/nba' / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    table <- nba_data %>%
      gt() %>%
      
      # Add team logos using gtExtras
      gt_img_rows(columns = logo_url, height = {LOGO_HEIGHT}) %>%
      
      # Title and subtitle - 538 style
      tab_header(
        title = md("**{TITLE}**"),
        subtitle = md("{subtitle}")
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
      
      # Column widths
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
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create NBA championship futures visualization')
    parser.add_argument('--top-n', type=int, default=9999,
                        help='Number of top teams to show (default: 9999 = all teams)')
    args = parser.parse_args()
    
    print("="*80)
    print("NBA CHAMPIONSHIP FUTURES VISUALIZATION (R + GT PACKAGE)")
    print("="*80 + "\n")
    
    if args.top_n < 9999:
        print(f"📊 Limiting to top {args.top_n} teams\n")
    
    # Read the CSV
    csv_file = repo_root / 'data/04_output/nba/nba_championship_fair_odds.csv'
    metadata_file = repo_root / 'data/04_output/nba/nba_championship_metadata.csv'
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Run analyze_nba_championship_futures_vig.py first!")
        return
    
    print(f"📂 Reading: {csv_file.name}\n")
    df = pd.read_csv(csv_file)
    
    # Store total teams before filtering
    total_teams = len(df)
    
    # Limit to top N teams (already sorted by fair_prob descending from analysis script)
    if args.top_n < total_teams:
        df = df.head(args.top_n)
        print(f"   ⚠️  Showing top {args.top_n} of {total_teams} teams")
    
    print(f"   📊 Loaded {len(df)} teams")
    print(f"   📊 Columns: {list(df.columns)}\n")
    
    # Read metadata to get actual average vig across bookmakers
    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)
        avg_vig_pct = metadata_df['avg_vig_pct'].iloc[0]
        print(f"   📊 Average market vig: {avg_vig_pct:.2f}%\n")
    else:
        print(f"   ⚠️  Metadata file not found, calculating vig from team data")
        avg_vig_pct = ((df['implied_prob_avg'] - df['fair_prob']) * 100).mean()
    
    # Get team logos
    logo_map = get_team_logos()
    print(f"   🏀 Loaded {len(logo_map)} team logos\n")
    
    # Prepare data
    df_display, _ = prepare_data_for_visualization(df, logo_map)
    
    # Create table using R's gt package with correct vig
    output_path = create_gt_table_with_r(df_display, avg_vig_pct, total_teams, args.top_n)
    
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

