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
2. 'Vig %' shows the bookmaker's edge in percentage points. Color indicates vig level: green = low vig (bettor advantage), red = high vig (house edge).
"""
FOOTER_DATA_SOURCE = "FanDuel Sportsbook"
# FOOTER_DATA_DATE is now generated dynamically from the CSV file

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
# Now using absolute difference (percentage points) like championship futures
COLOR_PALETTE = ["#4CAF50", "#90EE90", "#ffffff", "#ffcccc", "#d62728"]  # green -> white -> red
VIG_COLOR_DOMAIN_MIN = 0.0    # Start gradient at 0 percentage points
VIG_COLOR_DOMAIN_MAX = 5.0    # End gradient at 5 percentage points (MVPs have lower vig than futures)

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
COL_WIDTH_RANK = 70
COL_WIDTH_HEADSHOT = 45
COL_WIDTH_PLAYER = 220
COL_WIDTH_FANDUEL_ODDS = 120
COL_WIDTH_IMPLIED_PCT = 110
COL_WIDTH_FAIR_ODDS = 120
COL_WIDTH_FAIR_PCT = 100
COL_WIDTH_VIG_PCT = 100

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
}


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
    
    # Extract fetch_date from the CSV for timestamp
    fetch_date = None
    if 'fetch_date' in df.columns:
        fetch_date = df['fetch_date'].iloc[0]
    
    return df, latest_file, fetch_date


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
        tuple: (DataFrame ready for visualization, average_vig_pct)
    """
    print("📊 Preparing data for visualization...\n")
    
    df_display = df.copy()
    
    # Add rank column
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Add player headshots
    df_display = add_player_headshots(df_display)
    
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


def create_gt_table_with_r(df_display, average_vig_pct, fetch_date):
    """
    Create a publication-quality table using R's gt package.
    
    Args:
        df_display: Prepared dataframe with all display columns
        average_vig_pct: Calculated average vig percentage
        fetch_date: Date when odds were fetched (YYYY-MM-DD format)
        
    Returns:
        Path to saved PNG file
    """
    print("🎨 Creating table with R's gt package...\n")
    
    # Generate subtitle dynamically with market vig
    # Calculate total market vig from sum of implied probabilities
    if 'fanduel_implied_prob' in df_display.columns:
        total_implied = df_display['fanduel_implied_prob'].sum()
        market_vig_pct = (total_implied - 1.0) * 100
    else:
        # Fallback if column not found
        market_vig_pct = 5.5
    
    subtitle = f"FanDuel charges {market_vig_pct:.1f}% vig on the MVP market (vs. 4-5% on game lines)"
    
    # Format fetch_date for display (convert YYYY-MM-DD to "Month DD, YYYY")
    if fetch_date:
        try:
            date_obj = datetime.strptime(fetch_date, '%Y-%m-%d')
            footer_date = date_obj.strftime('%B %d, %Y')
        except:
            footer_date = fetch_date
    else:
        footer_date = datetime.now().strftime('%B %d, %Y')
    
    # Select columns for display (including headshot)
    # Note: Keep vig_pct as numeric for gradient coloring (will format in R)
    table_df = df_display[[
        'rank', 'headshot_url', 'player', 'fanduel_odds_str', 'implied_pct_str',
        'fair_odds_str', 'fair_pct_str', 'vig_pct'
    ]].copy()
    
    # Rename columns for display
    table_df.columns = [
        'Rank', 'headshot_url', 'Player', 'FanDuel Odds', 'Implied %',
        'Fair Odds', 'Fair %', 'Vig %'
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
    library(gtExtras)
    library(dplyr)
    
    # Read data (check.names=FALSE preserves column names, stringsAsFactors=FALSE keeps strings as strings)
    mvp_data <- read.csv("{str(temp_csv)}", check.names=FALSE, stringsAsFactors=FALSE, colClasses=c(
      "Rank"="integer",
      "headshot_url"="character",
      "Player"="character",
      "FanDuel Odds"="character",
      "Implied %"="character",
      "Fair Odds"="character",
      "Fair %"="character",
      "Vig %"="numeric"
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
      
      # Format Vig % column as percentage with + sign
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
      
      # Column widths (headshot between Rank and Player)
      cols_width(
        Rank ~ px({COL_WIDTH_RANK}),
        headshot_url ~ px({COL_WIDTH_HEADSHOT}),
        Player ~ px({COL_WIDTH_PLAYER}),
        `FanDuel Odds` ~ px({COL_WIDTH_FANDUEL_ODDS}),
        `Implied %` ~ px({COL_WIDTH_IMPLIED_PCT}),
        `Fair Odds` ~ px({COL_WIDTH_FAIR_ODDS}),
        `Fair %` ~ px({COL_WIDTH_FAIR_PCT}),
        `Vig %` ~ px({COL_WIDTH_VIG_PCT})
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
      
      # Conditional formatting for Vig % column (BEFORE formatting as text)
      # Green -> White -> Red gradient (low vig = green = good, high vig = red = bad)
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({', '.join([f'"{c}"' for c in COLOR_PALETTE])}),
        domain = c({VIG_COLOR_DOMAIN_MIN}, {VIG_COLOR_DOMAIN_MAX}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Override negative vig values with YELLOW (bettor advantage!)
      tab_style(
        style = cell_fill(color = "#ffeb3b"),
        locations = cells_body(
          columns = `Vig %`,
          rows = `Vig %` < 0
        )
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
        df, source_file, fetch_date = load_latest_fair_odds()
        print(f"   ✅ Loaded {len(df)} players from {source_file.name}")
        if fetch_date:
            print(f"   ✅ Fetch date: {fetch_date}\n")
        else:
            print(f"   ⚠️  No fetch_date found in CSV\n")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nRun analyze_nba_mvp_vig.py first!")
        return
    
    # Prepare data
    df_display, average_vig_pct = prepare_data_for_visualization(df)
    
    # Create table using R's gt package
    output_path = create_gt_table_with_r(df_display, average_vig_pct, fetch_date)
    
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

