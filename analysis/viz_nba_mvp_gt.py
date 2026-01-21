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

from odds_utils import odds_to_implied_probability
from s3_utils import get_latest_file_from_s3, read_df_from_s3


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
TITLE = "NBA MVP Odds: Preseason to Now"
# SUBTITLE is generated dynamically

FOOTER_NOTES = """
1. 'Fair Odds' = true odds with vig removed. 'Vig %' = FanDuel's edge (green = low, red = high).  
2. 'Difference' = change in implied probability from preseason to now (green = improved, red = worsened). Players with '-' = removed from board.
"""
FOOTER_DATA_SOURCE = "FanDuel Sportsbook"
# FOOTER_DATA_DATE is now generated dynamically from the CSV file

# -----------------------------------------------------------------------------
# Output Settings (Image Dimensions & Quality)
# -----------------------------------------------------------------------------
OUTPUT_FILENAME = "nba_mvp_vig.png"

# Larger table now (showing all ~20 players + more columns)
OUTPUT_WIDTH = 1600   # pixels (increased for more columns)
OUTPUT_HEIGHT = 1800  # pixels (increased for more rows)
OUTPUT_DPI = 300

# -----------------------------------------------------------------------------
# Color Palettes
# -----------------------------------------------------------------------------
# Difference column: Green = positive change (improved), Red = negative change (worsened)
DIFF_COLOR_PALETTE = ["#d62728", "#ffcccc", "#ffffff", "#90EE90", "#4CAF50"]  # red -> white -> green
DIFF_COLOR_DOMAIN_MIN = -50.0   # Large negative change (red)
DIFF_COLOR_DOMAIN_MAX = 50.0    # Large positive change (green)

# Vig % column: Green = low vig (good), Red = high vig (bad)
VIG_COLOR_PALETTE = ["#4CAF50", "#90EE90", "#ffffff", "#ffcccc", "#d62728"]  # green -> white -> red
VIG_COLOR_DOMAIN_MIN = 0.0      # No vig (green)
VIG_COLOR_DOMAIN_MAX = 5.0      # High vig (red)

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
HEADER_PADDING_PX = 1      # Padding around column headers (match futures viz)
DATA_ROW_PADDING_PX = 1    # Padding around data rows (smaller = more compact, sharper images)
HEADING_PADDING_PX = 3     # Padding around title/subtitle

# -----------------------------------------------------------------------------
# Column Widths (pixels)
# -----------------------------------------------------------------------------
COL_WIDTH_RANK = 60
COL_WIDTH_HEADSHOT = 40
COL_WIDTH_PLAYER = 150
COL_WIDTH_PRESEASON = 95
COL_WIDTH_PRESEASON_IMPLIED = 115
COL_WIDTH_CURRENT = 85
COL_WIDTH_CURRENT_IMPLIED = 105
COL_WIDTH_FAIR_ODDS = 90
COL_WIDTH_VIG_PCT = 75
COL_WIDTH_DIFFERENCE = 90

HEADSHOT_HEIGHT = 25  # Height of player headshots in pixels (smaller = sharper with tight padding)

# -----------------------------------------------------------------------------
# Player ID Mapping (for headshots)
# -----------------------------------------------------------------------------
# Map player names to NBA PLAYER_IDs for headshot URLs
PLAYER_ID_MAP = {
    'Shai Gilgeous-Alexander': 1628983,
    'Luka Doncic': 1629029,
    'Cade Cunningham': 1630595,
    'Jaylen Brown': 1627759,
    'Jalen Brunson': 1628973,
    'Anthony Edwards': 1630162,
    'Tyrese Maxey': 1630178,
    'Donovan Mitchell': 1628378,
    'Kawhi Leonard': 202695,
    'Stephen Curry': 201939,
    'Alperen Sengun': 1630578,
    'Kevin Durant': 201142,
    'Nikola Jokic': 203999,
    'Giannis Antetokounmpo': 203507,
    'LeBron James': 2544,
    'Joel Embiid': 203954,
    'Damian Lillard': 203081,
    'Devin Booker': 1626164,
    'Trae Young': 1629027,
    'Jayson Tatum': 1628369,
    'Victor Wembanyama': 1641705,
    'Evan Mobley': 1630596,
    'Paolo Banchero': 1631094,
    'Ja Morant': 1629630,
    'Anthony Davis': 203076,
    'Pascal Siakam': 1627783,
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_latest_fair_odds():
    """Load the most recent fair odds CSV from S3"""
    bucket = 'nba-betting-mt'
    prefix = 'data/04_output/nba/mvp/'
    
    # Get most recent file
    latest_key = get_latest_file_from_s3(bucket, prefix)
    
    if not latest_key:
        raise FileNotFoundError(
            f"No fair odds files found in s3://{bucket}/{prefix}\n"
            "Run analyze_nba_mvp_vig.py first!"
        )
    
    filename = latest_key.split('/')[-1]
    print(f"📂 Loading from S3: {filename}")
    
    df = read_df_from_s3(bucket, latest_key)
    
    # Extract fetch_date from the CSV for timestamp
    fetch_date = None
    if 'fetch_date' in df.columns:
        fetch_date = df['fetch_date'].iloc[0]
    
    return df, filename, fetch_date


def download_and_convert_to_base64(url):
    """
    Download image at FULL RESOLUTION and convert to base64 data URI.
    
    DO NOT thumbnail/resize in Python - let R/gtExtras handle scaling for best quality.
    
    Args:
        url: Image URL
        
    Returns:
        base64 data URI string or None if failed
    """
    try:
        response = requests.get(url, verify=False, timeout=10)
        if response.status_code != 200:
            return None
        
        # Convert directly to base64 WITHOUT resizing
        # This preserves maximum quality - R will scale it down
        img_base64 = base64.b64encode(response.content).decode('utf-8')
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
        df: DataFrame with 'player' column
        
    Returns:
        DataFrame with headshot_url column added (as base64 data URI)
    """
    print("   🖼️  Converting player headshots to base64 data URIs...")
    
    # Placeholder transparent pixel for missing headshots
    placeholder = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    def get_headshot_data_uri(player_name):
        player_id = PLAYER_ID_MAP.get(player_name)
        if not player_id:
            print(f"      ⚠️  No PLAYER_ID for {player_name}")
            return placeholder
        
        # Use NBA CDN 1040x760 (highest quality, 100% success rate)
        # Download at FULL RESOLUTION and let R/gtExtras scale for best quality
        nba_url = f'https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png'
        data_uri = download_and_convert_to_base64(nba_url)
        
        return data_uri if data_uri else placeholder
    
    df['headshot_url'] = df['player'].apply(get_headshot_data_uri)
    
    success_count = df['headshot_url'].apply(lambda x: len(x) > 200).sum()
    print(f"      ✅ Converted {success_count}/{len(df)} headshots successfully\n")
    
    return df


def prepare_data_for_visualization(df):
    """
    Prepare the dataframe with all display columns.
    
    Args:
        df: Raw dataframe from nba_mvp_fair_odds.csv
        
    Returns:
        tuple: (DataFrame ready for visualization, average_vig_pct, season_start_date)
    """
    print("📊 Preparing data for visualization...\n")
    
    df_display = df.copy()
    
    # Add rank column
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Add player headshots
    df_display = add_player_headshots(df_display)
    
    # Format preseason odds
    if 'season_start_odds' in df_display.columns:
        df_display['preseason_odds_str'] = df_display['season_start_odds'].apply(
            lambda x: f"{int(x):+d}" if pd.notna(x) else "NEW"
        )
        # Calculate preseason implied probability
        df_display['preseason_implied_prob'] = df_display['season_start_odds'].apply(
            lambda x: odds_to_implied_probability(x) if pd.notna(x) else None
        )
        df_display['preseason_implied_str'] = df_display['preseason_implied_prob'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
        )
        season_start_date = df_display['season_start_date'].iloc[0] if 'season_start_date' in df_display.columns else None
    else:
        df_display['preseason_odds_str'] = "N/A"
        df_display['preseason_implied_str'] = "N/A"
        season_start_date = None
    
    # Format current odds
    df_display['current_odds_str'] = df_display['fanduel_odds'].apply(
        lambda x: f"{int(x):+d}" if pd.notna(x) else "-"
    )
    
    # Format current implied probability (0.0% for removed players)
    df_display['current_implied_str'] = df_display['fanduel_implied_prob'].apply(
        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "0.0%"
    )
    
    # Calculate difference in implied probability (percentage points)
    # For ALL players (including removed ones where current = 0.0%)
    if 'preseason_implied_prob' in df_display.columns:
        df_display['difference'] = df_display.apply(
            lambda row: (row['fanduel_implied_prob'] - row['preseason_implied_prob']) * 100 
            if pd.notna(row['preseason_implied_prob'])
            else None,
            axis=1
        )
    
    # Format fair odds and percentages
    df_display['fair_odds_str'] = df_display['fair_odds'].apply(
        lambda x: f"{int(x):+d}" if pd.notna(x) else "-"
    )
    df_display['fair_pct_str'] = df_display['fair_prob'].apply(
        lambda x: f"{x*100:.1f}%" if pd.notna(x) and x > 0 else "-"
    )
    df_display['vig_pct_str'] = df_display['vig_pct'].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
    )
    
    # Calculate average vig
    average_vig_pct = df_display['vig_pct'].mean()
    
    print(f"   ✅ Prepared {len(df_display)} players")
    print(f"   ✅ Average vig: {average_vig_pct:.1f}%\n")
    
    return df_display, average_vig_pct, season_start_date


def create_gt_table_with_r(df_display, average_vig_pct, fetch_date, season_start_date):
    """
    Create a publication-quality table using R's gt package.
    
    Args:
        df_display: Prepared dataframe with all display columns
        average_vig_pct: Calculated average vig percentage
        fetch_date: Date when odds were fetched (YYYY-MM-DD format)
        season_start_date: Date of season start odds (YYYY-MM-DD format)
        
    Returns:
        Path to saved PNG file
    """
    print("🎨 Creating table with R's gt package...\n")
    
    # Generate subtitle dynamically
    # Count players on board vs removed
    on_board_count = df_display['fanduel_odds'].notna().sum()
    removed_count = df_display['fanduel_odds'].isna().sum()
    
    subtitle = f"{on_board_count} players currently on FanDuel's board"
    
    # Format fetch_date for display (convert YYYY-MM-DD to "Month DD, YYYY")
    if fetch_date:
        try:
            date_obj = datetime.strptime(fetch_date, '%Y-%m-%d')
            footer_date = date_obj.strftime('%B %d, %Y')
        except:
            footer_date = fetch_date
    else:
        footer_date = datetime.now().strftime('%B %d, %Y')
    
    # Select columns for display - preseason comparison + vig metrics
    # Keep difference and vig_pct as numeric for gradient coloring
    # Move Vig % to the far right (after Difference)
    columns_to_include = ['rank', 'headshot_url', 'player', 
                          'preseason_odds_str', 'preseason_implied_str',
                          'current_odds_str', 'current_implied_str', 
                          'fair_odds_str', 'difference', 'vig_pct']
    
    table_df = df_display[columns_to_include].copy()
    
    # Rename columns for display
    column_names = ['Rank', 'headshot_url', 'Player', 
                    'Preseason', 'Preseason Implied',
                    'Current', 'Current Implied', 
                    'Fair Odds', 'Difference', 'Vig %']
    
    table_df.columns = column_names
    
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
    
    # Determine if we have season start column
    has_season_start = any('Oct' in col or 'Nov' in col or 'Dec' in col or 'Jan' in col or 'Feb' in col or 'Start' in col for col in table_df.columns)
    
    # Build colClasses (Vig % moved to far right)
    col_classes = {
        'Rank': 'integer',
        'headshot_url': 'character',
        'Player': 'character',
        'Preseason': 'character',
        'Preseason Implied': 'character',
        'Current': 'character',
        'Current Implied': 'character',
        'Fair Odds': 'character',
        'Difference': 'numeric',
        'Vig %': 'numeric'
    }
    
    col_classes_str = ',\n      '.join([f'"{k}"="{v}"' for k, v in col_classes.items()])
    
    # Build column widths string (Vig % moved to far right)
    col_widths = [
        f"Rank ~ px({COL_WIDTH_RANK})",
        f"headshot_url ~ px({COL_WIDTH_HEADSHOT})",
        f"Player ~ px({COL_WIDTH_PLAYER})",
        f"`Preseason` ~ px({COL_WIDTH_PRESEASON})",
        f"`Preseason Implied` ~ px({COL_WIDTH_PRESEASON_IMPLIED})",
        f"`Current` ~ px({COL_WIDTH_CURRENT})",
        f"`Current Implied` ~ px({COL_WIDTH_CURRENT_IMPLIED})",
        f"`Fair Odds` ~ px({COL_WIDTH_FAIR_ODDS})",
        f"`Difference` ~ px({COL_WIDTH_DIFFERENCE})",
        f"`Vig %` ~ px({COL_WIDTH_VIG_PCT})"
    ]
    
    col_widths_str = ',\n        '.join(col_widths)
    
    # R code to create the table
    r_code = f"""
    # Set library path to user library
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    # Read data (check.names=FALSE preserves column names, stringsAsFactors=FALSE keeps strings as strings)
    mvp_data <- read.csv("{str(temp_csv)}", check.names=FALSE, stringsAsFactors=FALSE, colClasses=c(
      {col_classes_str}
    ))
    
    # Create gt table with 538-style formatting
    table <- mvp_data %>%
      gt() %>%
      
      # Title and subtitle
      tab_header(
        title = md("**{TITLE}**"),
        subtitle = md("{subtitle}")
      ) %>%
      
      # Add player headshots using gtExtras
      gt_img_rows(columns = headshot_url, height = {HEADSHOT_HEIGHT}) %>%
      
      # Format Difference column as percentage points with + sign (now before Vig %)
      fmt_number(
        columns = `Difference`,
        decimals = 1,
        pattern = "{{x}}pp",
        force_sign = TRUE
      ) %>%
      
      # Format Vig % column as percentage with + sign (moved to far right)
      fmt_number(
        columns = `Vig %`,
        decimals = 1,
        pattern = "{{x}}%",
        force_sign = TRUE
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
      
      # Column widths
      cols_width(
        {col_widths_str}
      ) %>%
      
      # Rename headshot_url column header to empty
      cols_label(
        headshot_url = ""
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
      
      # Conditional formatting for Difference column (now before Vig %)
      # Red -> White -> Green gradient (negative = worsened = red, positive = improved = green)
      data_color(
        columns = `Difference`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in DIFF_COLOR_PALETTE])}),
        domain = c({DIFF_COLOR_DOMAIN_MIN}, {DIFF_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Conditional formatting for Vig % column (moved to far right)
      # Green -> White -> Red gradient (low vig = green = good, high vig = red = bad)
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in VIG_COLOR_PALETTE])}),
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
        source_note = md("**Data:** {FOOTER_DATA_SOURCE} ({footer_date}) | **Analysis:** {TWITTER_HANDLE}")
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
        df, filename, fetch_date = load_latest_fair_odds()
        print(f"   ✅ Loaded {len(df)} players from {filename}")
        if fetch_date:
            print(f"   ✅ Fetch date: {fetch_date}\n")
        else:
            print(f"   ⚠️  No fetch_date found in CSV\n")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nRun analyze_nba_mvp_vig.py first!")
        return
    
    # Prepare data
    df_display, average_vig_pct, season_start_date = prepare_data_for_visualization(df)
    
    # Create table using R's gt package
    output_path = create_gt_table_with_r(df_display, average_vig_pct, fetch_date, season_start_date)
    
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

