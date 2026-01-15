"""
Create R gt visualization for Payton Pritchard Paint Scoring Analysis

Uses R's gt package with player headshots (similar to futures/MVP visualizations).
Follows the pattern from src/r_viz.py with NBA player headshots from CDN.
"""

import pandas as pd
import yaml
from pathlib import Path
import sys
import subprocess
import platform
import base64
import requests
import ssl
import urllib3

# Fix SSL issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Get repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Load config
viz_config_file = repo_root / 'config/viz_config.yaml'
with open(viz_config_file) as f:
    viz_config = yaml.safe_load(f)

# Paths
csv_file = repo_root / 'data/04_output/nba/payton_pritchard_paint_scoring_2025_26.csv'
output_dir = repo_root / 'content/viz/nba'
output_file = output_dir / 'payton_pritchard_paint_scoring_2025_26.png'


def download_and_convert_to_base64(url):
    """
    Download image at FULL RESOLUTION and convert to base64 data URI.
    
    DO NOT thumbnail/resize in Python - let R/gtExtras handle scaling for best quality.
    This avoids HTTPS loading issues in webshot2/R rendering.
    
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


def get_player_headshot_data_uri(player_id):
    """
    Get NBA player headshot as base64 data URI.
    
    Args:
        player_id: NBA player ID
        
    Returns:
        base64 data URI string or placeholder if failed
    """
    # Placeholder transparent pixel for missing headshots
    placeholder = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    if pd.isna(player_id):
        return placeholder
    
    # Use NBA CDN 1040x760 (highest quality, 100% success rate)
    # Download at FULL RESOLUTION and let R/gtExtras scale for best quality
    nba_url = f'https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png'
    data_uri = download_and_convert_to_base64(nba_url)
    
    return data_uri if data_uri else placeholder


def create_paint_scoring_table():
    """
    Create gt table with R, including player headshots
    """
    print("\n" + "="*80)
    print("CREATING R GT VISUALIZATION")
    print("="*80 + "\n")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} players\n")
    
    # Convert headshots to base64 data URIs
    print("   🖼️  Converting player headshots to base64 data URIs...")
    df['headshot_url'] = df['player_id'].apply(get_player_headshot_data_uri)
    
    # Check success rate
    success_count = df['headshot_url'].apply(lambda x: len(x) > 200).sum()
    print(f"      ✅ Converted {success_count}/{len(df)} headshots successfully\n")
    
    # Prepare display dataframe (no filtering - just use what's in CSV)
    df_viz = df.copy()
    
    # Create display columns
    df_viz['paint_stats'] = df_viz.apply(
        lambda row: f"{row['paint_fgm']}/{row['paint_fga']}",
        axis=1
    )
    
    df_viz['fg_pct_display'] = df_viz['paint_fg_pct'].apply(lambda x: f"{x}%")
    df_viz['paint_rate_display'] = df_viz['paint_rate'].apply(lambda x: f"{x}%")
    df_viz['paint_ppg_display'] = df_viz['paint_ppg'].apply(lambda x: f"{x:.1f}")
    
    # Select columns for table (headshot AFTER player name)
    table_df = df_viz[[
        'rank', 'player', 'headshot_url', 'height', 'games',
        'paint_stats', 'paint_fg_pct', 'paint_rate', 'paint_ppg'
    ]].copy()
    
    # Rename for display
    table_df.columns = [
        'Rank', 'Player', 'headshot_url', 'Height', 'GP',
        'Makes/Attempts', 'Paint FG%', 'Paint Rate', 'Paint PPG'
    ]
    
    # Import R/Python interface
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        
        print("   ✅ rpy2 loaded successfully\n")
        
    except ImportError as e:
        print(f"❌ Error: rpy2 not installed")
        print(f"   Install: pip install rpy2")
        print(f"   Also ensure R is installed: brew install r")
        return
    
    # Convert to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(table_df)
    
    ro.globalenv['paint_data'] = r_df
    
    # Output path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_str = str(output_file)
    
    print(f"   💾 Output: {output_file.name}\n")
    
    # Get config values
    headshot_config = viz_config['headshots']['nba']
    headshot_height = headshot_config['display_height_px']
    
    # Get current date
    from datetime import datetime
    data_date = datetime.now().strftime("%B %d, %Y")
    
    # R code for visualization
    r_code = f'''
    # Set library path
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    # Create table
    table <- paint_data %>%
      gt() %>%
      
      # Add player headshots
      gt_img_rows(columns = headshot_url, height = {headshot_height}) %>%
      
      # Title and subtitle
      tab_header(
        title = md("**Payton Pritchard is ELITE at Scoring in the Paint for a 6'1\\" Guard**"),
        subtitle = md("Paint shots (≤6 feet from basket) | 2025-26 NBA Season")
      ) %>%
      
      # Column alignment
      cols_align(align = "center", columns = everything()) %>%
      cols_align(align = "left", columns = c(Player)) %>%
      
      # Hide headshot column header
      cols_label(headshot_url = "") %>%
      
      # Format Paint FG% column
      fmt(
        columns = `Paint FG%`,
        fns = function(x) {{
          paste0(sprintf("%.1f", x), "%")
        }}
      ) %>%
      
      # Format Paint Rate column
      fmt(
        columns = `Paint Rate`,
        fns = function(x) {{
          paste0(sprintf("%.1f", x), "%")
        }}
      ) %>%
      
      # Format Paint PPG column
      fmt(
        columns = `Paint PPG`,
        fns = function(x) {{
          sprintf("%.1f", x)
        }}
      ) %>%
      
      # ==========================================================================
      # COLOR GRADIENTS (based on NBA distribution of 356 players)
      # ==========================================================================
      # Paint FG%: Red (below average) -> Green (elite)
      #   - Based on league-wide data (356 players, ≥25 paint attempts):
      #   - Median = 59.8% (white/center point)
      #   - Domain: 47% to 72.6% (symmetric around 59.8% median, ~12.8% on each side)
      #   - This ensures white appears at 59.8%, red below, green above
      data_color(
        columns = `Paint FG%`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(47, 72.6),
        na_color = "#e8e8e8"
      ) %>%
      
      # Paint Rate: Light blue (low) -> Dark blue (high)
      #   - Shows what % of their shots are in the paint
      #   - Dark blue (35%+) = Frequent paint attacker
      #   - Light blue (15%) = Occasional paint scorer
      #   - Domain: 0% to 50%
      data_color(
        columns = `Paint Rate`,
        method = "numeric",
        palette = c("#e8e8e8", "#cce5ff", "#66b3ff", "#0066cc"),
        domain = c(0, 50),
        na_color = "#e8e8e8"
      ) %>%
      
      # Paint PPG: Light orange (low) -> Dark orange (high)
      #   - Shows points per game scored in the paint
      #   - Dark orange (8-10) = High volume paint scorer
      #   - Light orange (2-4) = Low volume paint scorer
      #   - Domain: 0 to 10 points
      data_color(
        columns = `Paint PPG`,
        method = "numeric",
        palette = c("#e8e8e8", "#ffe6cc", "#ffcc80", "#ff9933"),
        domain = c(0, 10),
        na_color = "#e8e8e8"
      ) %>%
      
      # Column widths (headshot after player)
      cols_width(
        Rank ~ px(50),
        Player ~ px(180),
        headshot_url ~ px(45),
        Height ~ px(70),
        GP ~ px(60),
        `Makes/Attempts` ~ px(120),
        `Paint FG%` ~ px(90),
        `Paint Rate` ~ px(90),
        `Paint PPG` ~ px(90)
      ) %>%
      
      # Style headers
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px(13), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Style title
      tab_style(
        style = cell_text(
          font = "Arial",
          size = px(24),
          weight = "bold",
          color = "#2c3e50"
        ),
        locations = cells_title(groups = "title")
      ) %>%
      
      # Style subtitle
      tab_style(
        style = cell_text(
          font = "Arial",
          size = px(14),
          color = "#555555"
        ),
        locations = cells_title(groups = "subtitle")
      ) %>%
      
      # Bold player names and ranks
      tab_style(
        style = cell_text(weight = "600", size = px(12)),
        locations = cells_body(columns = c(Rank, Player))
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
        table.font.size = px(12),
        heading.title.font.size = px(24),
        heading.subtitle.font.size = px(14),
        heading.padding = px(3),
        column_labels.padding = px(6),
        data_row.padding = px(1),
        table.border.bottom.width = px(2),
        table.border.bottom.color = "#2c3e50",
        column_labels.border.bottom.width = px(2),
        column_labels.border.bottom.color = "#2c3e50",
        table.background.color = "#f8f9fa",
        row.striping.background_color = "#f0f0f0",
        source_notes.font.size = px(10),
        source_notes.padding = px(8)
      ) %>%
      
      # Footer notes
      tab_source_note(
        source_note = md("**Paint FG%** = field goal % on shots within 6 feet | **Paint Rate** = % of all shots taken in paint | **Paint PPG** = points per game in paint")
      ) %>%
      tab_source_note(
        source_note = md("**NBA Guard Medians (356 players):** Paint FG% = 59.8% | Paint Rate = 39.5% | Paint PPG = 4.0")
      ) %>%
      tab_source_note(
        source_note = md("**Data:** NBA API ({data_date}) | **Analysis:** @TQSLabs")
      )
    
    # Save as PNG (portrait format for social media)
    gtsave(table, "{output_path_str}", vwidth = 1400, vheight = 1800)
    
    print("✅ Table saved successfully!")
    '''
    
    print("   🔧 Executing R code...\n")
    
    try:
        ro.r(r_code)
        print(f"\n   ✅ Visualization saved!\n")
        print(f"   🖼️  {output_file}\n")
        
        # Auto-open on macOS
        try:
            if platform.system() == 'Darwin':
                subprocess.run(['open', str(output_file)])
                print("   📂 Opening PNG...\n")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error creating visualization:")
        print(f"   {e}")
        print("\n💡 Install R packages:")
        print("   R -e 'install.packages(c(\"gt\", \"gtExtras\", \"dplyr\", \"webshot2\"))'")
        sys.exit(1)


if __name__ == "__main__":
    create_paint_scoring_table()
    
    print("="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)

