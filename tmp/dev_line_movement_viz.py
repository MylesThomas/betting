"""
Line Movement Visualization Development Script

PURPOSE:
Develop and iterate on line movement visualizations with fake data
before integrating into the production email alert system.

CONTEXT (from request):
User wants to visualize line movements over time where:
- Y-axis: spread line value (inverted: favorites at bottom, underdogs at top)
- X-axis: time (left = first snapshot, right = current)
- Each bookmaker shows BOTH teams' lines (mirror images)
- Multiple books plotted on same graph to see how they diverge/converge

VISUALIZATION OPTIONS:
- Default (matplotlib): Fast iteration, good for testing
- --ggplot flag: Use R's ggplot2 for publication-quality charts

This script generates fake snapshot data to simulate what we'd pull from S3,
then creates visualizations that can be refined before deployment.

Usage:
python scripts/dev_line_movement_viz.py
python scripts/dev_line_movement_viz.py --ggplot
"""

import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import io
import base64
import argparse
import sys
from pathlib import Path
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# =============================================================================
# FAKE DATA GENERATION
# =============================================================================

def generate_fake_snapshots(game_id: str, away_team: str, home_team: str, 
                           num_snapshots: int = 24) -> pd.DataFrame:
    """
    Generate fake line movement data simulating hourly snapshots.
    
    Creates realistic-looking line movement with:
    - Different bookmakers moving at different times
    - Some steam moves (multiple books moving together)
    - Natural variance in pricing
    """
    bookmakers = [
        'draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers',
        'pointsbet', 'fanatics', 'bovada', 'mybookieag', 'betonlineag'
    ]
    
    # Start time (24 hours ago)
    start_time = datetime.now() - timedelta(hours=num_snapshots)
    
    rows = []
    
    for i in range(num_snapshots):
        timestamp = start_time + timedelta(hours=i)
        
        # Base line that drifts over time (simulate market movement)
        # Home team (Packers) are favorites: away_spread POSITIVE, moves from +3.0 to +5.0
        # This means Packers line goes from -3.0 to -5.0 (getting more favored)
        base_away_spread = 3.0 + (i / num_snapshots) * 2.0  # Ravens spread: +3 to +5 (underdogs)
        
        for book_idx, bookmaker in enumerate(bookmakers):
            # Each book has slight variance from base line
            # Some books are faster to move, some are slower
            if bookmaker in ['draftkings', 'fanduel', 'betmgm']:
                # Sharp books - move quickly
                book_line = base_away_spread + (0.1 * (book_idx % 3 - 1))
            elif bookmaker in ['caesars', 'betrivers']:
                # Medium speed books
                lag_factor = 0.7
                book_line = (3.0 + (i / num_snapshots) * 2.0 * lag_factor) + (0.15 * (book_idx % 3 - 1))
            else:
                # Slower books - lag behind market
                lag_factor = 0.5
                book_line = (3.0 + (i / num_snapshots) * 2.0 * lag_factor) + (0.2 * (book_idx % 3 - 1))
            
            # Add some random noise
            import random
            book_line += random.uniform(-0.15, 0.15)
            
            # Round to half-point
            book_line = round(book_line * 2) / 2
            
            # Price variance
            if book_line == base_away_spread:
                price = -110
            elif book_line > base_away_spread:
                price = random.choice([-105, -108, -110])
            else:
                price = random.choice([-110, -112, -115])
            
            # Match exact S3 structure
            rows.append({
                'game_id': game_id,
                'game_time': timestamp.isoformat(),
                'away_team': away_team,
                'home_team': home_team,
                'bookmaker': bookmaker,
                'away_spread': book_line,
                'away_price': price,
                'away_adjusted_spread': book_line,  # Simplified for fake data
                'home_spread': -book_line,
                'home_price': price,  # Prices should be the same structure
                'home_adjusted_spread': -book_line,
                'fetched_at': timestamp.isoformat(),
                'last_bookmaker_update': timestamp.isoformat(),
                'game_time_et': timestamp.strftime('%Y-%m-%d %H:%M:%S ET'),
                'fetched_at_et': timestamp.strftime('%Y-%m-%d %H:%M:%S ET'),
                'last_bookmaker_update_et': timestamp.strftime('%Y-%m-%d %H:%M:%S ET')
            })
    
    return pd.DataFrame(rows)


def generate_fake_crossed_zero_game() -> pd.DataFrame:
    """Generate fake data where favorite/underdog flips (crosses zero)."""
    bookmakers = ['draftkings', 'fanduel', 'betmgm', 'caesars']
    start_time = datetime.now() - timedelta(hours=24)
    
    rows = []
    
    for i in range(24):
        timestamp = start_time + timedelta(hours=i)
        
        # Line moves from +2 (home underdog) to -2 (home favorite)
        if i < 8:
            base_line = 2.0
        elif i < 12:
            base_line = 1.0  # Approaching zero
        elif i < 14:
            base_line = 0.0  # Pick'em
        elif i < 18:
            base_line = -1.0  # Just crossed
        else:
            base_line = -2.0  # Full flip
        
        for bookmaker in bookmakers:
            import random
            book_line = base_line + random.uniform(-0.25, 0.25)
            book_line = round(book_line * 2) / 2
            
            # Match exact S3 structure
            rows.append({
                'game_id': 'game_crossed_zero',
                'game_time': timestamp.isoformat(),
                'away_team': 'New York Giants',
                'home_team': 'Las Vegas Raiders',
                'bookmaker': bookmaker,
                'away_spread': book_line,
                'away_price': -110,
                'away_adjusted_spread': book_line,
                'home_spread': -book_line,
                'home_price': -110,
                'home_adjusted_spread': -book_line,
                'fetched_at': timestamp.isoformat(),
                'last_bookmaker_update': timestamp.isoformat(),
                'game_time_et': timestamp.strftime('%Y-%m-%d %H:%M:%S ET'),
                'fetched_at_et': timestamp.strftime('%Y-%m-%d %H:%M:%S ET'),
                'last_bookmaker_update_et': timestamp.strftime('%Y-%m-%d %H:%M:%S ET')
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_line_movement_chart(df: pd.DataFrame, title: str = None) -> bytes:
    """
    Create line movement chart for a single game.
    
    Args:
        df: DataFrame with columns: fetched_at, bookmaker, away_spread, away_team, home_team
        title: Chart title (defaults to game matchup)
    
    Returns:
        PNG image as bytes (for embedding in email)
    """
    if df.empty:
        return None
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    
    # Get game info
    away_team = df['away_team'].iloc[0]
    home_team = df['home_team'].iloc[0]
    
    if not title:
        title = f"{away_team} @ {home_team}"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color map for bookmakers
    bookmakers = df['bookmaker'].unique()
    colors = plt.cm.tab10(range(len(bookmakers)))
    color_map = dict(zip(bookmakers, colors))
    
    # Plot each bookmaker's line
    for bookmaker in bookmakers:
        book_df = df[df['bookmaker'] == bookmaker].copy()
        
        # Plot away spread (negative = away favored)
        ax.plot(
            book_df['timestamp'],
            book_df['away_spread'],
            label=bookmaker,
            color=color_map[bookmaker],
            marker='o',
            markersize=3,
            linewidth=2,
            alpha=0.8
        )
    
    # Add horizontal line at 0 (pick'em)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Pick\'em')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{away_team} Spread', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show time nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)
    
    # Tight layout
    plt.tight_layout()
    
    # Convert to bytes for email embedding
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)
    
    return img_bytes


def create_line_movement_chart_simple(df: pd.DataFrame, title: str = None) -> bytes:
    """
    Simplified version with cleaner styling and focus on major books.
    """
    if df.empty:
        return None
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    
    # Get game info
    away_team = df['away_team'].iloc[0]
    home_team = df['home_team'].iloc[0]
    
    # Get time range for subtitle
    first_snapshot = df['timestamp'].min()
    last_snapshot = df['timestamp'].max()
    time_range_hours = (last_snapshot - first_snapshot).total_seconds() / 3600
    
    # Format times for subtitle
    first_time_str = first_snapshot.strftime('%b %d %I:%M %p')
    last_time_str = last_snapshot.strftime('%b %d %I:%M %p ET')
    
    # Determine who is currently favored (most recent snapshot)
    latest_spread = df['away_spread'].iloc[-1]
    if latest_spread < 0:
        favorite = away_team
        underdog = home_team
    elif latest_spread > 0:
        favorite = home_team
        underdog = away_team
    else:
        favorite = "Pick'em"
        underdog = "Pick'em"
    
    if not title:
        title = f"{away_team} @ {home_team}"
    
    subtitle = f"{time_range_hours:.0f}h movement ({first_time_str} → {last_time_str})"
    if favorite != "Pick'em":
        subtitle += f" | Current Favorite: {favorite}"
    
    # Focus on major books
    major_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers']
    df_major = df[df['bookmaker'].isin(major_books)]
    
    # Create figure with clean style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color map for bookmakers (distinct, branded colors)
    book_colors = {
        'draftkings': '#53D337',  # DK green
        'fanduel': '#0E8FEF',     # FD blue
        'betmgm': '#BA9000',      # MGM gold
        'caesars': '#0033A0',     # Caesars blue
        'betrivers': '#00A4E4'    # BetRivers light blue
    }
    
    # Plot each bookmaker's line - BOTH TEAMS (mirror)
    for bookmaker in major_books:
        book_df = df_major[df_major['bookmaker'] == bookmaker].copy()
        if book_df.empty:
            continue
        
        color = book_colors.get(bookmaker, '#333333')
        
        # Plot away team spread (solid line)
        ax.plot(
            book_df['timestamp'],
            book_df['away_spread'],
            label=f"{bookmaker.upper()} ({away_team})",
            color=color,
            marker='o',
            markersize=5,
            linewidth=3,
            alpha=0.9,
            linestyle='-'
        )
        
        # Plot home team spread (dashed line - mirror)
        ax.plot(
            book_df['timestamp'],
            book_df['home_spread'],
            label=f"{bookmaker.upper()} ({home_team})",
            color=color,
            marker='s',
            markersize=5,
            linewidth=3,
            alpha=0.9,
            linestyle='--'
        )
    
    # Add horizontal line at 0 (pick'em)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Pick\'em', zorder=1)
    
    # Add shaded regions to show favorite/underdog zones
    y_min, y_max = ax.get_ylim()
    ax.axhspan(0, y_max, alpha=0.05, color='green', zorder=0)  # Underdog zone (top)
    ax.axhspan(y_min, 0, alpha=0.05, color='red', zorder=0)    # Favorite zone (bottom)
    
    # INVERT Y-AXIS: Favorites (negative) at bottom, underdogs (positive) at top
    ax.invert_yaxis()
    
    # Formatting
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spread (points)', fontsize=13, fontweight='bold')
    
    # Title and subtitle
    ax.set_title(title, fontsize=16, fontweight='bold', pad=35)
    ax.text(0.5, 1.03, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=11, style='italic', color='#555')
    
    # Add favorite zone labels on the chart
    # Top (negative lines) = Favorites, Bottom (positive lines) = Underdogs
    ax.text(0.02, 0.98, '← FAVORITES (negative lines)', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    ax.text(0.02, 0.02, '← UNDERDOGS (positive lines)', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend - show all books with both teams
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8, ncol=2, 
              title='Sportsbooks', title_fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)
    
    return img_bytes


def create_line_movement_chart_ggplot(df: pd.DataFrame, title: str = None) -> bytes:
    """
    Create line movement chart using R's ggplot2 for publication-quality output.
    
    Args:
        df: DataFrame with columns: fetched_at, bookmaker, away_spread, home_spread, away_team, home_team
        title: Chart title (defaults to game matchup)
    
    Returns:
        PNG image as bytes (for embedding in email)
    """
    if df.empty:
        return None
    
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
        print("   3. Install R packages: R -e 'install.packages(c(\"ggplot2\", \"dplyr\", \"scales\"))'")
        return None
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    
    # Get game info
    away_team = df['away_team'].iloc[0]
    home_team = df['home_team'].iloc[0]
    
    # Get time range for subtitle
    first_snapshot = df['timestamp'].min()
    last_snapshot = df['timestamp'].max()
    time_range_hours = (last_snapshot - first_snapshot).total_seconds() / 3600
    
    # Format times for subtitle
    first_time_str = first_snapshot.strftime('%b %d %I:%M %p')
    last_time_str = last_snapshot.strftime('%b %d %I:%M %p ET')
    
    # Determine current favorite
    latest_spread = df['away_spread'].iloc[-1]
    if latest_spread < 0:
        favorite = away_team
    elif latest_spread > 0:
        favorite = home_team
    else:
        favorite = "Pick'em"
    
    if not title:
        title = f"{away_team} @ {home_team}"
    
    subtitle = f"{time_range_hours:.0f}h movement ({first_time_str} → {last_time_str})"
    if favorite != "Pick'em":
        subtitle += f" | Current Favorite: {favorite}"
    
    # Focus on major books
    major_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers']
    df_major = df[df['bookmaker'].isin(major_books)].copy()
    
    # Reshape data for ggplot (need both away and home in long format)
    df_away = df_major[['timestamp', 'bookmaker', 'away_spread']].copy()
    df_away['team'] = away_team
    df_away['spread'] = df_away['away_spread']
    df_away['line_type'] = 'solid'
    
    df_home = df_major[['timestamp', 'bookmaker', 'home_spread']].copy()
    df_home['team'] = home_team
    df_home['spread'] = df_home['home_spread']
    df_home['line_type'] = 'dashed'
    
    plot_df = pd.concat([df_away, df_home], ignore_index=True)
    plot_df = plot_df[['timestamp', 'bookmaker', 'team', 'spread', 'line_type']]
    
    # Convert to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(plot_df)
    
    ro.globalenv['plot_data'] = r_df
    
    # Define output path
    output_path = Path('/Users/thomasmyles/dev/betting/data/99_tmp/ggplot_chart.png')
    
    # R code for ggplot2 visualization
    r_code = f"""
    # Set library path to user library first
    .libPaths(c("~/R/library", .libPaths()))

    # Check if libraries we need are installed, if not install them
    if (!requireNamespace("ggplot2", quietly = TRUE)) {{
        install.packages("ggplot2")
    }}
    if (!requireNamespace("dplyr", quietly = TRUE)) {{
        install.packages("dplyr")
    }}
    if (!requireNamespace("scales", quietly = TRUE)) {{
        install.packages("scales")
    }}
    if (!requireNamespace("teamcolors", quietly = TRUE)) {{
        install.packages("teamcolors")
    }}
    if (!requireNamespace("ggtext", quietly = TRUE)) {{
        install.packages("ggtext")
    }}
    if (!requireNamespace("ggforce", quietly = TRUE)) {{
        install.packages("ggforce")
    }}
    library(ggplot2)
    library(dplyr)
    library(scales)
    
    # Convert timestamp to POSIXct
    plot_data$timestamp <- as.POSIXct(plot_data$timestamp, format="%Y-%m-%d %H:%M:%S")
    
    # Define colors for bookmakers
    book_colors <- c(
        'draftkings' = '#53D337',
        'fanduel' = '#0E8FEF',
        'betmgm' = '#BA9000',
        'caesars' = '#0033A0',
        'betrivers' = '#00A4E4'
    )
    
    # Determine who is favorite/underdog from CURRENT data
    away_data <- plot_data[plot_data$team == "{away_team}", ]
    latest_away_spread <- tail(away_data$spread, 1)
    
    if (latest_away_spread < 0) {{
        favorite_team <- "{away_team}"
        underdog_team <- "{home_team}"
    }} else {{
        favorite_team <- "{home_team}"
        underdog_team <- "{away_team}"
    }}
    
    # Create the plot
    p <- ggplot(plot_data, aes(x = timestamp, y = spread, color = bookmaker, group = interaction(bookmaker, team))) +
        # Add lines
        geom_line(aes(linetype = line_type), size = 1.2, alpha = 0.9) +
        geom_point(aes(shape = line_type), size = 2.5, alpha = 0.9) +
        
        # Add horizontal line at 0 (pick'em)
        geom_hline(yintercept = 0, color = 'red', linetype = 'dashed', size = 1, alpha = 0.6) +
        
        # Add shaded regions (GREEN for favorites at top, RED for underdogs at bottom)
        annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0, 
                 alpha = 0.05, fill = "green") +  # Favorites zone
        annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = Inf, 
                 alpha = 0.05, fill = "red") +    # Underdogs zone
        
        # Scale y-axis (inverted)
        scale_y_reverse() +
        
        # Colors
        scale_color_manual(values = book_colors, name = "Sportsbooks") +
        
        # Line types
        scale_linetype_manual(values = c('solid' = 'solid', 'dashed' = 'dashed'), guide = 'none') +
        scale_shape_manual(values = c('solid' = 16, 'dashed' = 15), guide = 'none') +
        
        # Labels
        labs(
            title = "{title}",
            subtitle = "{subtitle}",
            x = "Time",
            y = "Spread (points)"
        ) +
        
        # Theme
        theme_minimal(base_size = 14) +
        theme(
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40", face = "italic"),
            axis.title = element_text(size = 13, face = "bold"),
            axis.text = element_text(size = 11),
            legend.position = "right",
            legend.title = element_text(face = "bold"),
            legend.text = element_text(size = 10),
            panel.grid.major = element_line(color = "gray80", size = 0.3),
            panel.grid.minor = element_line(color = "gray90", size = 0.2),
            plot.background = element_rect(fill = "white", color = NA),
            panel.background = element_rect(fill = "#f8f9fa", color = NA)
        ) +
        
        # Format x-axis for dates
        scale_x_datetime(date_labels = "%m/%d %H:%M", date_breaks = "6 hours")
    
    # Add zone labels with ACTUAL team names from data
    p <- p + 
        annotate("label", x = min(plot_data$timestamp), y = max(plot_data$spread) * 0.95,
                 label = paste(toupper(underdog_team), "- UNDERDOGS"), 
                 hjust = 0, vjust = 1, size = 3.5, fontface = "bold", 
                 color = "darkred", fill = "white", alpha = 0.8) +
        annotate("label", x = min(plot_data$timestamp), y = min(plot_data$spread) * 0.95,
                 label = paste(toupper(favorite_team), "- FAVORITES"), 
                 hjust = 0, vjust = 0, size = 3.5, fontface = "bold",
                 color = "darkgreen", fill = "white", alpha = 0.8)
    
    # Add time reference lines dynamically based on data range
    current_time <- max(plot_data$timestamp)
    earliest_time <- min(plot_data$timestamp)
    time_range_hours <- as.numeric(difftime(current_time, earliest_time, units = "hours"))
    
    # Always show NOW
    p <- p +
        geom_vline(xintercept = as.numeric(current_time), 
                   linetype = "solid", color = "blue", size = 0.8, alpha = 0.6) +
        annotate("text", x = current_time, y = min(plot_data$spread), 
                 label = "NOW", vjust = -0.5, hjust = 0.5, 
                 color = "blue", fontface = "bold", size = 3)
    
    # Add 1H AGO if we have at least 1.5 hours of data
    if (time_range_hours >= 1.5) {{
        time_1h <- current_time - 3600
        if (time_1h >= earliest_time) {{
            p <- p +
                geom_vline(xintercept = as.numeric(time_1h), 
                           linetype = "dashed", color = "orange", size = 0.6, alpha = 0.5) +
                annotate("text", x = time_1h, y = min(plot_data$spread), 
                         label = "1H AGO", vjust = -0.5, hjust = 0.5, 
                         color = "orange", fontface = "bold", size = 2.5)
        }}
    }}
    
    # Add 24H AGO if we have at least 24 hours of data
    if (time_range_hours >= 24) {{
        time_24h <- current_time - 86400
        if (time_24h >= earliest_time) {{
            p <- p +
                geom_vline(xintercept = as.numeric(time_24h), 
                           linetype = "dashed", color = "purple", size = 0.6, alpha = 0.5) +
                annotate("text", x = time_24h, y = min(plot_data$spread), 
                         label = "24H AGO", vjust = -0.5, hjust = 0.5, 
                         color = "purple", fontface = "bold", size = 2.5)
        }}
    }}
    
    # Add 1 WEEK AGO if we have at least 7 days of data
    if (time_range_hours >= 168) {{
        time_1week <- current_time - 604800
        if (time_1week >= earliest_time) {{
            p <- p +
                geom_vline(xintercept = as.numeric(time_1week), 
                           linetype = "dashed", color = "darkgreen", size = 0.6, alpha = 0.5) +
                annotate("text", x = time_1week, y = min(plot_data$spread), 
                         label = "1WK AGO", vjust = -0.5, hjust = 0.5, 
                         color = "darkgreen", fontface = "bold", size = 2.5)
        }}
    }}
    
    # Save
    ggsave("{str(output_path)}", p, width = 14, height = 7, dpi = 150, bg = "white")
    """
    
    print("   🔧 Executing R/ggplot2 code...")
    
    try:
        ro.r(r_code)
        print(f"   ✅ ggplot2 chart created!")
        
        # Read the saved PNG and return as bytes
        with open(output_path, 'rb') as f:
            img_bytes = f.read()
        
        return img_bytes
        
    except Exception as e:
        print(f"❌ Error creating ggplot2 chart:")
        print(f"   {e}")
        print("\n💡 Make sure R packages are installed:")
        print("   R -e 'install.packages(c(\"ggplot2\", \"dplyr\", \"scales\"))'")
        return None


def image_to_base64(img_bytes: bytes) -> str:
    """Convert image bytes to base64 string for HTML embedding."""
    return base64.b64encode(img_bytes).decode('utf-8')


# =============================================================================
# DEMO/TEST FUNCTIONS
# =============================================================================

def save_chart_to_file(img_bytes: bytes, filename: str):
    """Save chart to local file for inspection."""
    with open(filename, 'wb') as f:
        f.write(img_bytes)
    print(f"✅ Saved chart to: {filename}")


def demo_steam_move(use_ggplot: bool = False):
    """Demo: Normal steam move scenario."""
    viz_type = "ggplot2" if use_ggplot else "matplotlib"
    print(f"📊 Generating Steam Move Chart ({viz_type})...")
    
    df = generate_fake_snapshots(
        game_id='game_ravens_packers',
        away_team='Baltimore Ravens',
        home_team='Green Bay Packers',
        num_snapshots=24
    )
    
    # Show data sample
    print("\nSample data:")
    print(df.head(10))
    
    # Create chart
    if use_ggplot:
        img_bytes = create_line_movement_chart_ggplot(df)
    else:
        img_bytes = create_line_movement_chart_simple(df)
    
    if img_bytes:
        # Save to file
        save_chart_to_file(img_bytes, '/Users/thomasmyles/dev/betting/data/99_tmp/test_steam_move_chart.png')
    
    return img_bytes


def demo_crossed_zero(use_ggplot: bool = False):
    """Demo: Line crosses zero (favorite flips)."""
    viz_type = "ggplot2" if use_ggplot else "matplotlib"
    print(f"\n📊 Generating Crossed Zero Chart ({viz_type})...")
    
    df = generate_fake_crossed_zero_game()
    
    # Show data sample
    print("\nSample data:")
    print(df.head(10))
    
    # Create chart
    if use_ggplot:
        img_bytes = create_line_movement_chart_ggplot(df)
    else:
        img_bytes = create_line_movement_chart_simple(df)
    
    if img_bytes:
        # Save to file
        save_chart_to_file(img_bytes, '/Users/thomasmyles/dev/betting/data/99_tmp/test_crossed_zero_chart.png')
    
    return img_bytes


def demo_html_email_with_inline_image(use_ggplot: bool = False):
    """Demo: Generate HTML email with inline base64 image."""
    print("\n📧 Generating HTML Email Preview...")
    
    # Generate charts
    steam_img = demo_steam_move(use_ggplot)
    crossed_img = demo_crossed_zero(use_ggplot)
    
    if not steam_img or not crossed_img:
        print("⚠️  Skipping HTML generation due to chart errors")
        return None
    
    # Convert to base64
    steam_b64 = image_to_base64(steam_img)
    crossed_b64 = image_to_base64(crossed_img)
    
    # Create HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
            h2 {{ color: #0066cc; margin-top: 30px; }}
            .chart {{ margin: 20px 0; }}
            .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>🚨 Line Movement Alert</h1>
        <p><strong>Time:</strong> Dec 25, 2025 05:00 PM ET</p>
        
        <h2>📊 Large Moves (Last Hour)</h2>
        
        <div class="chart">
            <h3>Baltimore Ravens @ Green Bay Packers</h3>
            <p>Steam move detected: Line moved from -3.0 to -5.0 across all major books</p>
            <img src="data:image/png;base64,{steam_b64}" alt="Line Movement Chart">
        </div>
        
        <div class="chart">
            <h3>New York Giants @ Las Vegas Raiders 🚨</h3>
            <p>Line crossed zero! Favorite flipped from Raiders to Giants</p>
            <img src="data:image/png;base64,{crossed_b64}" alt="Crossed Zero Chart">
        </div>
    </body>
    </html>
    """
    
    # Save HTML to file for preview
    html_path = '/Users/thomasmyles/dev/betting/data/99_tmp/test_email_preview.html'
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Saved HTML email preview to: {html_path}")
    print(f"   Open this file in a browser to see how the email will look!")
    
    return html


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Line Movement Visualization Development',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (matplotlib)
  python scripts/dev_line_movement_viz.py
  
  # Use R's ggplot2 for publication-quality charts
  python scripts/dev_line_movement_viz.py --ggplot
        """
    )
    parser.add_argument(
        '--ggplot',
        action='store_true',
        help='Use R/ggplot2 instead of matplotlib (requires R and rpy2)'
    )
    
    args = parser.parse_args()
    
    viz_engine = "R/ggplot2" if args.ggplot else "matplotlib"
    
    print("=" * 80)
    print(f"LINE MOVEMENT VISUALIZATION DEVELOPMENT ({viz_engine.upper()})")
    print("=" * 80)
    
    # Run demos
    demo_steam_move(use_ggplot=args.ggplot)
    demo_crossed_zero(use_ggplot=args.ggplot)
    demo_html_email_with_inline_image(use_ggplot=args.ggplot)
    
    print("\n" + "=" * 80)
    print("✅ Complete! Check the generated files:")
    print("   - data/99_tmp/test_steam_move_chart.png")
    print("   - data/99_tmp/test_crossed_zero_chart.png")
    print("   - data/99_tmp/test_email_preview.html (open in browser)")
    print("=" * 80)

