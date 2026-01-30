"""
R Visualization for Championship Futures

Purpose:
Generate FiveThirtyEight-style tables using R's gt package for championship futures.
Consolidated from 4 sport-specific scripts (NFL, NBA, NCAAF, NCAAB).

Functions:
- generate_footer_notes: Create footer with optional filtering note
- create_futures_table: Main function to generate R gt table

Usage:
    from r_viz import create_futures_table
    
    output_path = create_futures_table(
        df_display=df,
        sport='nfl',
        sport_config=config['sports']['nfl'],
        viz_config=config['visualization'],
        average_vig_pct=7.5,
        total_teams=32,
        top_n=19
    )
"""

import sys
import boto3
from pathlib import Path
from typing import Dict, Any

# Get repo root
repo_root = Path(__file__).parent.parent


def generate_footer_notes(
    sport: str,
    teams_with_odds: int,
    total_teams: int,
    top_n: int,
    has_historical: bool = False
) -> str:
    """
    Generate footer notes with optional filtering message.
    
    Args:
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        teams_with_odds: Number of teams that have odds available
        total_teams: Total number of teams before filtering
        top_n: Number of teams displayed
        has_historical: Whether historical odds are included
        
    Returns:
        Formatted footer notes string
    """
    # Base notes (different for historical vs non-historical)
    if has_historical:
        note_1 = "'Implied %' includes bookmaker vig. 'Fair Odds' is the true probability with vig removed."
        note_2 = "'Difference' columns show change in implied probability (green = improved, red = worsened)."
    else:
        note_1 = "'Implied %' includes bookmaker vig. 'Fair %' is the true probability with vig removed (fair probabilities sum to exactly 100%)."
        note_2 = "Color indicates vig level: green = low vig, red = high vig, yellow = negative vig (bettor advantage)."
    
    # Sport-specific note 3 (pro sports only - college doesn't have elimination concept)
    note_count = 2
    base_notes = f"""
1. {note_1}  
2. {note_2}"""
    
    if sport == 'nfl':
        note_count = 3
        base_notes += f"""  
3. {teams_with_odds} of 32 teams have odds available — bookmakers no longer offer odds on eliminated/longshot teams."""
    elif sport == 'nba':
        note_count = 3
        base_notes += f"""  
3. {teams_with_odds} of 30 teams have odds available — some teams may have very long odds due to poor season performance."""
    # NCAAF and NCAAB don't have note 3 (no elimination concept in college)
    
    # Add filtering note if top_n was used
    if top_n < total_teams:
        note_count += 1
        base_notes += f"""  
{note_count}. Filtered to top {top_n} teams by fair probability."""
    
    return base_notes


def create_futures_table(
    df_display,
    sport: str,
    sport_config: Dict[str, Any],
    viz_config: Dict[str, Any],
    average_vig_pct: float,
    total_teams: int,
    top_n: int,
    save_locally: bool = False,
    s3_bucket: str = None,
    s3_path: str = None,
    season_start_date: str = None,
    season_start_label: str = None,
    last_week_date: str = None,
    last_week_label: str = None,
    diff_domain_max: float = 10.0
) -> Path:
    """
    Create a publication-quality futures table using R's gt package.
    
    This function consolidates R table generation for all 4 sports (NFL, NBA, NCAAF, NCAAB).
    
    Args:
        df_display: Prepared DataFrame with display columns
        sport: Sport key ('nfl', 'nba', 'ncaaf', 'ncaab')
        sport_config: Sport-specific config dict
        viz_config: Shared visualization config dict
        average_vig_pct: Average vig percentage across bookmakers
        total_teams: Total teams before filtering (for footer note)
        top_n: Number of teams displayed (for footer note)
        save_locally: If True, save to local filesystem
        s3_bucket: S3 bucket name for output (e.g., "nfl-betting-mt")
        s3_path: S3 path prefix (e.g., "viz")
        
    Returns:
        Path to saved PNG file (local or temp path if S3-only)
    """
    print("🎨 Creating table with R's gt package...\n")
    
    # Check if we have historical odds (must do this first)
    has_historical = (
        'preseason_odds_str' in df_display.columns and 
        df_display['preseason_odds_str'].notna().any() and
        df_display['preseason_odds_str'].ne('-').any()
    )
    
    # Generate subtitle with calculated vig
    subtitle = f"Bookmakers charge an *average {average_vig_pct:.1f}% vig* on championship futures (vs. 4-5% on game lines)"
    
    # Generate footer notes
    # Note: total_teams = count of teams with odds (before any top_n filtering)
    footer_notes = generate_footer_notes(sport, total_teams, total_teams, top_n, has_historical)
    
    # Import R/Python interface
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        
        print("   ✅ rpy2 loaded successfully")
        
    except ImportError as e:
        print(f"❌ Error: rpy2 not installed or R not found")
        print(f"   {e}")
        print("\n📖 Installation instructions:")
        print("   1. Install R: brew install r")
        print("   2. Install Python package: pip install rpy2")
        print("   3. Install R packages: R -e 'install.packages(c(\"gt\", \"gtExtras\", \"tidyverse\", \"webshot2\"))'")
        sys.exit(1)
    
    # Select and rename columns for display
    if has_historical:
        # Include historical odds columns
        table_df = df_display[[
            'rank', 'team', 'logo_url', 'record',
            'preseason_odds_str', 'preseason_implied_str',
            'last_week_odds_str', 'last_week_implied_str',
            'current_odds_str', 'current_implied_str',
            'fair_odds_str', 'diff_preseason', 'diff_last_week', 'vig_diff'
        ]].copy()
        
        table_df.columns = [
            'Rank', 'Team', 'logo_url', 'Record',
            'Preseason', 'Preseason Implied',
            'Last Week', 'Last Week Implied',
            'Current', 'Current Implied',
            'Fair Odds', 'Difference<br>(Pre → Current)', 'Difference<br>(LW → Current)', 'Vig %'
        ]
    else:
        # Original columns (no historical odds)
        table_df = df_display[[
            'rank', 'team', 'logo_url', 'record', 
            'fair_odds_str', 'fair_pct_str',
            'avg_odds_str', 'implied_pct_str', 'vig_diff',
            'best_book_display', 'best_odds_str', 'best_vig_diff'
        ]].copy()
        
        table_df.columns = [
            'Rank', 'Team', 'logo_url', 'Record', 
            'Fair Odds', 'Fair %',
            'Avg Odds', 'Implied %', 'Vig %',
            'Best Book', 'Best Odds', 'Best Vig %'
        ]
    
    print(f"   📋 Table dimensions: {table_df.shape}")
    print(f"   📋 Columns: {list(table_df.columns)}")
    print(f"   📋 Historical odds: {'Yes' if has_historical else 'No'}")
    print(f"   📋 Vig % column type: {table_df['Vig %'].dtype}")
    print(f"   📋 Vig % min/max: {table_df['Vig %'].min():.2f} to {table_df['Vig %'].max():.2f}")
    
    if has_historical:
        diff_pre_col = 'Difference<br>(Pre → Current)'
        diff_lw_col = 'Difference<br>(LW → Current)'
        if diff_pre_col in table_df.columns and table_df[diff_pre_col].notna().any():
            print(f"   📋 Diff (Pre→Current) min/max: {table_df[diff_pre_col].min():.2f} to {table_df[diff_pre_col].max():.2f}")
        if diff_lw_col in table_df.columns and table_df[diff_lw_col].notna().any():
            print(f"   📋 Diff (LW→Current) min/max: {table_df[diff_lw_col].min():.2f} to {table_df[diff_lw_col].max():.2f}")
    else:
        print(f"   📋 Best Vig % min/max: {table_df['Best Vig %'].min():.2f} to {table_df['Best Vig %'].max():.2f}")
    
    print()
    
    # Convert pandas DataFrame to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(table_df)
    
    ro.globalenv['futures_data'] = r_df
    
    # Output path
    output_dir = repo_root / sport_config['viz_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / sport_config['viz_filename']
    output_path_str = str(output_path)
    
    print(f"   💾 Output path: {output_path.name}")
    
    # Extract config values
    title = sport_config['title']
    logo_height = sport_config['logo_height']
    output_width = sport_config['output_width']
    output_height = sport_config['output_height']
    
    # Viz config
    font_family = viz_config['font_family']
    title_font_size = viz_config['title_font_size']
    subtitle_font_size = viz_config['subtitle_font_size']
    header_font_size = viz_config['header_font_size']
    body_font_size = viz_config['body_font_size']
    footer_font_size = viz_config['footer_font_size']
    heading_padding = viz_config['heading_padding_px']
    header_padding = viz_config['header_padding_px']
    data_row_padding = viz_config['data_row_padding_px']
    
    # Column widths (different for historical vs non-historical)
    if has_historical:
        col_widths = {
            'Rank': viz_config['col_width_rank'],
            'Team': viz_config['col_width_team'],
            'logo_url': viz_config['col_width_logo'],
            'Record': viz_config['col_width_record'],
            'Preseason': viz_config['col_width_preseason'],
            'Preseason Implied': viz_config['col_width_preseason_implied'],
            'Last Week': viz_config['col_width_last_week'],
            'Last Week Implied': viz_config['col_width_last_week_implied'],
            'Current': viz_config['col_width_current'],
            'Current Implied': viz_config['col_width_current_implied'],
            'Fair Odds': viz_config['col_width_fair_odds'],
            'Difference<br>(Pre → Current)': viz_config['col_width_diff_preseason'],
            'Difference<br>(LW → Current)': viz_config['col_width_diff_last_week'],
            'Vig %': viz_config['col_width_vig_pct'],
        }
    else:
        col_widths = {
            'Rank': viz_config['col_width_rank'],
            'Team': viz_config['col_width_team'],
            'logo_url': viz_config['col_width_logo'],
            'Record': viz_config['col_width_record'],
            'Fair Odds': viz_config['col_width_fair_odds'],
            'Fair %': viz_config['col_width_fair_pct'],
            'Avg Odds': viz_config['col_width_avg_odds'],
            'Implied %': viz_config['col_width_implied_pct'],
            'Vig %': viz_config['col_width_vig_pct'],
            'Best Book': viz_config['col_width_best_book'],
            'Best Odds': viz_config['col_width_best_odds'],
            'Best Vig %': viz_config['col_width_best_vig_pct'],
        }
    
    # Color settings
    vig_color_palette_str = ', '.join([f'"{c}"' for c in viz_config['vig_color_palette']])
    diff_color_palette_str = ', '.join([f'"{c}"' for c in viz_config['diff_color_palette']])
    vig_color_min = sport_config.get('vig_color_domain_min', 0.0)
    vig_color_max = sport_config.get('vig_color_domain_max', 5.0)
    negative_vig_color = viz_config['negative_vig_color']
    
    # Debug: Print color config
    print(f"   🎨 Vig color domain: {vig_color_min} to {vig_color_max}")
    print(f"   🎨 Vig color palette: {viz_config['vig_color_palette']}")
    print(f"   🎨 Diff color palette: {viz_config['diff_color_palette']}\n")
    # Footer
    data_source = "The Odds API"
    twitter_handle = viz_config['twitter_handle']
    
    # Get current date
    from datetime import datetime
    data_date = datetime.now().strftime("%B %d, %Y")
    
    # Build column widths string dynamically
    col_widths_r_list = []
    for col_name, width in col_widths.items():
        if col_name == 'logo_url':
            col_widths_r_list.append(f"{col_name} ~ px({width})")
        elif '<br>' in col_name:
            # Escape column names with HTML
            col_widths_r_list.append(f"`{col_name}` ~ px({width})")
        else:
            col_widths_r_list.append(f"`{col_name}` ~ px({width})")
    
    col_widths_str = ',\n        '.join(col_widths_r_list)
    
    # Generate formatting and coloring code based on has_historical
    if has_historical:
        # Historical odds table: Format difference columns + vig column
        format_code = f"""
      # Format Difference columns as percentage points with + sign
      fmt_number(
        columns = c(`Difference<br>(Pre → Current)`, `Difference<br>(LW → Current)`),
        decimals = 1,
        pattern = "{{{{x}}}}pp",
        force_sign = TRUE
      ) %>%
      
      # Format Vig % column
      fmt(
        columns = `Vig %`,
        fns = function(x) {{{{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}}}
      ) %>%"""
        
        color_code = f"""
      # Conditional formatting for Difference columns
      # Red -> White -> Green gradient (negative = worsened = red, positive = improved = green)
      data_color(
        columns = `Difference<br>(Pre → Current)`,
        method = "numeric",
        palette = c({diff_color_palette_str}),
        domain = c(-{diff_domain_max}, {diff_domain_max}),
        na_color = "#e8e8e8"
      ) %>%
      data_color(
        columns = `Difference<br>(LW → Current)`,
        method = "numeric",
        palette = c({diff_color_palette_str}),
        domain = c(-{diff_domain_max}, {diff_domain_max}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Color for Vig % column
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({vig_color_palette_str}),
        domain = c({vig_color_min}, {vig_color_max}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Override negative vig with yellow
      tab_style(
        style = cell_fill(color = "{negative_vig_color}"),
        locations = cells_body(columns = `Vig %`, rows = `Vig %` < 0)
      ) %>%"""
        
        label_code = """
      # Rename column headers with HTML line breaks
      cols_label(
        logo_url = "",
        `Difference<br>(Pre → Current)` = html("Difference<br>(Pre → Current)"),
        `Difference<br>(LW → Current)` = html("Difference<br>(LW → Current)")
      ) %>%"""
        
    else:
        # Original table: Format vig columns only
        format_code = f"""
      # Format Vig % columns (numeric → "+X.X%" strings)
      fmt(
        columns = `Vig %`,
        fns = function(x) {{{{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}}}
      ) %>%
      fmt(
        columns = `Best Vig %`,
        fns = function(x) {{{{
          ifelse(is.na(x), "-", 
                 ifelse(x >= 0, paste0("+", sprintf("%.1f", x), "%"),
                        paste0(sprintf("%.1f", x), "%")))
        }}}}
      ) %>%"""
        
        color_code = f"""
      # Apply color gradient to Vig % columns (reads numeric values)
      data_color(
        columns = `Vig %`,
        method = "numeric",
        palette = c({vig_color_palette_str}),
        domain = c({vig_color_min}, {vig_color_max}),
        na_color = "#e8e8e8"
      ) %>%
      data_color(
        columns = `Best Vig %`,
        method = "numeric",
        palette = c({vig_color_palette_str}),
        domain = c({vig_color_min}, {vig_color_max}),
        na_color = "#e8e8e8"
      ) %>%
      
      # Override negative vig with yellow (bettor advantage)
      tab_style(
        style = cell_fill(color = "{negative_vig_color}"),
        locations = cells_body(columns = `Vig %`, rows = `Vig %` < 0)
      ) %>%
      tab_style(
        style = cell_fill(color = "{negative_vig_color}"),
        locations = cells_body(columns = `Best Vig %`, rows = `Best Vig %` < 0)
      ) %>%"""
        
        label_code = """
      # Hide logo column header
      cols_label(logo_url = "") %>%"""
    
    # Generate R code for gt table
    r_code = f"""
    # Set library path
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    # Create gt table
    table <- futures_data %>%
      gt() %>%
      
      # Add logos
      gt_img_rows(columns = logo_url, height = {logo_height}) %>%
      
      # Title and subtitle
      tab_header(
        title = md("**{title}**"),
        subtitle = md("{subtitle}")
      ) %>%
      
      # Column alignment
      cols_align(align = "center", columns = everything()) %>%
      cols_align(align = "left", columns = c(Team)) %>%
      
      # Format columns (step 1)
      {format_code}
      
      # Column widths
      cols_width(
        {col_widths_str}
      ) %>%
      
      # Column labels
      {label_code}
      
      # Style headers
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px({header_font_size}), color = "#2c3e50"),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Style title
      tab_style(
        style = cell_text(
          font = "{font_family}",
          size = px({title_font_size}),
          weight = "bold",
          color = "#2c3e50"
        ),
        locations = cells_title(groups = "title")
      ) %>%
      
      # Style subtitle
      tab_style(
        style = cell_text(
          font = "{font_family}", 
          size = px({subtitle_font_size}),
          color = "#555555"
        ),
        locations = cells_title(groups = "subtitle")
      ) %>%
      
      # Apply color gradients (step 2)
      {color_code}
      
      # Bold rank and team names
      tab_style(
        style = cell_text(weight = "bold", size = px({body_font_size})),
        locations = cells_body(columns = Rank)
      ) %>%
      tab_style(
        style = cell_text(weight = "600", size = px({body_font_size})),
        locations = cells_body(columns = Team)
      ) %>%
      
      # Zebra striping
      opt_row_striping(row_striping = TRUE) %>%
      
      # Table options
      tab_options(
        table.font.names = "{font_family}",
        table.font.size = px({body_font_size}),
        heading.title.font.size = px({title_font_size}),
        heading.subtitle.font.size = px({subtitle_font_size}),
        heading.title.font.weight = "bold",
        heading.padding = px({heading_padding}),
        table.border.top.style = "hidden",
        table.border.bottom.style = "solid",
        table.border.bottom.width = px(2),
        table.border.bottom.color = "#2c3e50",
        column_labels.border.top.style = "hidden",
        column_labels.border.bottom.width = px(3),
        column_labels.border.bottom.color = "#2c3e50",
        column_labels.padding = px({header_padding}),
        data_row.padding = px({data_row_padding}),
        table.background.color = "#f8f9fa",
        row.striping.background_color = "#f0f0f0",
        source_notes.font.size = px({footer_font_size}),
        source_notes.padding = px(10)
      ) %>%
      
      # Footer notes
      tab_source_note(source_note = md("{footer_notes}")) %>%
      tab_source_note(
        source_note = md("**Data:** {data_source} ({data_date}) | **Analysis:** {twitter_handle}")
      )
    
    # Save as PNG
    gtsave(table, "{output_path_str}", vwidth = {output_width}, vheight = {output_height})
    
    print("✅ Table saved successfully!")
    """
    
    print("   🔧 Executing R code...\n")
    
    try:
        ro.r(r_code)
        print(f"\n   ✅ Table created and saved!\n")
        
        # Upload to S3 if requested
        if s3_bucket and s3_path:
            try:
                s3_client = boto3.client('s3')
                viz_filename = sport_config['viz_filename']
                s3_key = f"{s3_path}/{viz_filename}"
                
                with open(output_path, 'rb') as f:
                    s3_client.put_object(
                        Bucket=s3_bucket,
                        Key=s3_key,
                        Body=f.read(),
                        ContentType='image/png'
                    )
                print(f"☁️  Uploaded to s3://{s3_bucket}/{s3_key}\n")
                
            except Exception as e:
                print(f"⚠️  S3 upload failed: {e}\n")
        
        # Remove local file if not saving locally
        if not save_locally and output_path.exists():
            output_path.unlink()
            print(f"🗑️  Removed local file (S3-only mode)\n")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error creating table in R:")
        print(f"   {e}")
        print("\n💡 Make sure R packages are installed:")
        print("   R -e 'install.packages(c(\"gt\", \"gtExtras\", \"dplyr\", \"webshot2\"))'")
        sys.exit(1)
