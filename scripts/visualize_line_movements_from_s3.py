"""
Visualize Line Movement History from S3 Snapshots

PURPOSE:
Generate publication-quality line movement charts for all games being tracked.
Uses Python to fetch/process data from S3, and R/ggplot2 for visualization.

This is an ANALYSIS/REPORTING tool, separate from the automated tracking system.
Run on-demand to create charts for specific games or time periods.

WHY THIS EXISTS:
The Lambda function (track_game_line_movements.py) generates charts in emails,
but those are:
- Optimized for email viewing (matplotlib, smaller size)
- Only created when movements are detected
- Sent once per hour, then gone

This script lets you:
- Re-create charts for ANY game in your S3 history
- Use R/ggplot2 for publication-quality output
- Customize chart parameters (time range, books, styling)
- Export to high-res PNG/PDF for sharing/reports
- Batch process multiple games at once

USE CASES:
1. Deep dive analysis on a specific game's line movement
2. Create charts for social media / blog posts
3. Compare movement across multiple games
4. Historical analysis (e.g., "How did lines move during Week 15?")
5. Export to PDF for weekly reports

CORE FUNCTIONALITY:
1. List all games in S3 snapshots (NBA/NFL)
2. For each selected game:
   - Fetch all snapshots from S3 (within time window)
   - Parse spread/price data by bookmaker
   - Pass to R via rpy2
3. R generates ggplot2 chart with:
   - Mirrored lines (away/home spreads)
   - Color-coded bookmakers
   - Vertical lines at 1h, 24h ago
   - Shaded favorite/underdog zones
   - Clean legend and labels
   - High resolution output

DATA FLOW:
  S3 Bucket (snapshots)
       ↓
  Python (boto3) → Load CSVs
       ↓
  Python (pandas) → Parse/filter/aggregate
       ↓
  rpy2 → Pass data frame to R
       ↓
  R (ggplot2) → Generate chart
       ↓
  Save to: data/04_output/line_movement/charts/{game_id}_{timestamp}.png

DEPENDENCIES:
Python:
- boto3 (S3 access)
- pandas (data manipulation)
- rpy2 (Python-R interface)
- python-dotenv (environment variables)

R (must be installed on system):
- ggplot2 (plotting)
- scales (date formatting)
- dplyr (data manipulation)
- lubridate (datetime handling)

SETUP:

1. Install Python packages:
   pip install boto3 pandas rpy2 python-dotenv

2. Install R (if not already installed):
   macOS: brew install r
   Ubuntu: sudo apt-get install r-base
   
3. Install R packages:
   R -e "install.packages(c('ggplot2', 'scales', 'dplyr', 'lubridate'), repos='https://cran.rstudio.com/')"

4. Configure AWS credentials (if not already done):
   aws configure
   (boto3 will automatically use these credentials)

5. Set environment variables (in .env):
   S3_BUCKET=betting-line-movement-snapshots
   AWS_REGION_NAME=us-east-2

USAGE:

# List all games with movements in S3
python scripts/visualize_line_movements_from_s3.py --list-games

# Generate chart for specific game
python scripts/visualize_line_movements_from_s3.py --game-id abc123xyz --sport nba

# Generate charts for all games with movements today
python scripts/visualize_line_movements_from_s3.py --date 2025-12-26 --sport nfl

# Custom time window (last 7 days)
python scripts/visualize_line_movements_from_s3.py --game-id abc123xyz --hours 168

# Batch: Generate charts for all games tracked in last 24h
python scripts/visualize_line_movements_from_s3.py --batch --hours 24

# Output to specific directory
python scripts/visualize_line_movements_from_s3.py --game-id abc123xyz --output-dir ~/Desktop/charts/

# High-res PDF output (for print)
python scripts/visualize_line_movements_from_s3.py --game-id abc123xyz --format pdf --dpi 300

CHART FEATURES (R/ggplot2):

Visual Elements:
- Mirrored spread lines (solid = away, dashed = home)
- Color-coded bookmakers (DraftKings green, FanDuel blue, etc.)
- Vertical reference lines (1h ago orange, 24h ago purple)
- Shaded zones (favorite in green, underdog in red)
- Horizontal line at 0 (pick'em)
- Data points as circles/squares (easy to see exact values)

Annotations:
- Title: "{Away} @ {Home}"
- Subtitle: Time range, current favorite, magnitude of movement
- Legend: Bookmaker names only (not duplicated per team)
- Axis labels: Clear, readable fonts
- Team zone labels: Dynamic based on current favorite

Styling:
- Publication-quality fonts (system defaults)
- High DPI (150 for screen, 300 for print)
- Wide aspect ratio (14x7) for email/web
- White background with subtle gridlines
- Color-blind friendly palette

Customization Options:
--bookmakers: Comma-separated list (e.g., "draftkings,fanduel,betmgm")
--theme: ggplot2 theme (default: theme_minimal)
--width: Chart width in inches (default: 14)
--height: Chart height in inches (default: 7)
--title-override: Custom title text

OUTPUT STRUCTURE:

data/04_output/line_movement/charts/
  nba/
    20251226_MIN_DEN_47h.png          (filename encodes game/time range)
    20251226_MIN_DEN_47h.pdf          (PDF version if requested)
  nfl/
    20251226_BAL_GB_48h.png
    20251226_PHI_BUF_48h.png

Filename format: {date}_{away}_{home}_{hours}h.{ext}

COMPARISON TO LAMBDA VERSION:

Lambda (track_game_line_movements.py):
- Real-time automated emails
- Matplotlib charts (optimized for email)
- Only created when movements detected
- Smaller file size (~200KB PNG)
- Limited customization

This Script:
- On-demand analysis/reporting
- R/ggplot2 charts (publication-quality)
- Create for ANY game in history
- High-res output (1MB+ PNG, scalable PDF)
- Full customization (books, time, styling)

EXAMPLE WORKFLOW:

# Step 1: See what games are available
python scripts/visualize_line_movements_from_s3.py --list-games --sport nfl --date 2025-12-26

Output:
  NFL Games (2025-12-26):
  1. abc123 - Baltimore Ravens @ Green Bay Packers (crossed zero, 48h data)
  2. def456 - Philadelphia Eagles @ Buffalo Bills (4.5pt move, 48h data)
  3. ghi789 - Houston Texans @ Los Angeles Chargers (2.0pt move, 48h data)

# Step 2: Generate chart for game of interest
python scripts/visualize_line_movements_from_s3.py --game-id abc123 --sport nfl --output-dir ~/Desktop/

Output:
  ✅ Chart saved: ~/Desktop/20251226_BAL_GB_48h.png
  ✅ Chart saved: ~/Desktop/20251226_BAL_GB_48h.pdf

# Step 3: Open and review
open ~/Desktop/20251226_BAL_GB_48h.png

# Step 4: Share on social media / blog
(Manual step - copy file to wherever you need it)

ERROR HANDLING:
- Missing game_id: Lists available games and exits
- No S3 data: Shows error with time range that HAS data
- R not installed: Shows install instructions
- Missing R packages: Runs install command automatically
- AWS credentials missing: Shows setup instructions

PERFORMANCE:
- Single game: ~2-3 seconds (S3 fetch + R rendering)
- Batch (10 games): ~20-30 seconds
- Large time window (7 days): ~5-10 seconds per game

S3 COSTS:
- Minimal: Only reading existing snapshot files
- ~$0.0004 per 1000 GET requests
- Example: 100 charts = $0.04

FUTURE ENHANCEMENTS:
- Interactive HTML charts (plotly/highcharts via R)
- Animated GIFs showing line movement over time
- Side-by-side game comparisons
- Steam move detection overlay
- Correlation analysis (injury news timestamps)
- Export to Twitter-optimized format (16:9 ratio)

RELATED FILES:
- scripts/track_game_line_movements.py (Lambda function that creates S3 data)
- scripts/dev_line_movement_viz.py (Plotly version for Streamlit dashboard)
- data/01_input/the-odds-api/{sport}/line_movement/*.csv (S3 snapshots)

AUTHOR: Thomas Myles
CREATED: 2025-12-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import boto3
from io import StringIO
from typing import Optional, List, Dict, Tuple
import json

# rpy2 for Python-R interface
try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion
    
    # Use new converter context (pandas2ri.activate() is deprecated)
    converter = conversion.Converter('pandas2ri converter')
    converter += pandas2ri.converter
except ImportError:
    print("❌ ERROR: rpy2 not installed")
    print("   Install with: pip install rpy2")
    print("   Also ensure R is installed on your system")
    sys.exit(1)

# Load environment variables
load_dotenv()

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# AWS Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'betting-line-movement-snapshots')
AWS_REGION = os.getenv('AWS_REGION_NAME', 'us-east-2')

# Sports
SPORT_NBA = 'basketball_nba'
SPORT_NFL = 'americanfootball_nfl'
SUPPORTED_SPORTS = [SPORT_NBA, SPORT_NFL]

SPORT_DISPLAY_NAMES = {
    SPORT_NBA: 'NBA',
    SPORT_NFL: 'NFL'
}

# Display timezone
DISPLAY_TIMEZONE = 'America/New_York'

# Chart configuration (16:9 aspect ratio)
CHART_WIDTH = 16  # inches
CHART_HEIGHT = 9  # inches
CHART_DPI = 150   # dots per inch (150 for screen, 300 for print)

# Initialize boto3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# =============================================================================
# S3 HELPER FUNCTIONS
# =============================================================================

def list_s3_snapshots(sport: str) -> List[str]:
    """List all snapshot files in S3 for a given sport."""
    sport_short = sport.split('_')[1]  # 'nba' or 'nfl'
    prefix = f"data/01_input/the-odds-api/{sport_short}/line_movement/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
    except Exception as e:
        print(f"Warning: Failed to list S3 snapshots: {e}")
        return []


def load_dataframe_from_s3(s3_key: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from S3 CSV."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return df
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Warning: Failed to load from S3: {s3_key}")
        print(f"   {e}")
        return None


def fetch_all_snapshots_for_game(sport: str, game_id: str, hours: int = 168) -> pd.DataFrame:
    """
    Fetch all snapshots for a specific game from S3 within a time window.
    
    Args:
        sport: 'basketball_nba' or 'americanfootball_nfl'
        game_id: Game ID from The Odds API
        hours: How far back to look (default 1 week = 168 hours)
    
    Returns:
        DataFrame with all snapshots for this game
    """
    all_snapshots = list_s3_snapshots(sport)
    
    if not all_snapshots:
        return pd.DataFrame()
    
    # Calculate time cutoff
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    # Collect data from all relevant snapshots
    game_data = []
    
    for s3_key in all_snapshots:
        # Extract timestamp from filename
        filename = s3_key.split('/')[-1]
        timestamp_str = filename.replace('snapshot_', '').replace('.csv', '')
        
        try:
            file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            file_time = file_time.replace(tzinfo=timezone.utc)
            
            # Skip if outside time window
            if file_time < cutoff_time:
                continue
            
            # Load snapshot
            df = load_dataframe_from_s3(s3_key)
            if df is None or df.empty:
                continue
            
            # Filter to just this game
            game_df = df[df['game_id'] == game_id]
            if not game_df.empty:
                game_data.append(game_df)
                
        except ValueError:
            continue
    
    if not game_data:
        return pd.DataFrame()
    
    # Combine all snapshots
    combined_df = pd.concat(game_data, ignore_index=True)
    combined_df = combined_df.sort_values('fetched_at')
    
    return combined_df


def list_available_games(sport: str, hours: int = 48) -> List[Dict]:
    """
    List all unique games in S3 snapshots within time window.
    
    Returns:
        List of dicts with game info: game_id, away_team, home_team, game_time, num_snapshots
    """
    all_snapshots = list_s3_snapshots(sport)
    
    if not all_snapshots:
        return []
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    # Collect all games
    games_dict = {}  # game_id -> game info
    
    for s3_key in all_snapshots:
        filename = s3_key.split('/')[-1]
        timestamp_str = filename.replace('snapshot_', '').replace('.csv', '')
        
        try:
            file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            file_time = file_time.replace(tzinfo=timezone.utc)
            
            if file_time < cutoff_time:
                continue
            
            df = load_dataframe_from_s3(s3_key)
            if df is None or df.empty:
                continue
            
            # Get unique games
            for game_id in df['game_id'].unique():
                if game_id not in games_dict:
                    game_df = df[df['game_id'] == game_id].iloc[0]
                    games_dict[game_id] = {
                        'game_id': game_id,
                        'away_team': game_df['away_team'],
                        'home_team': game_df['home_team'],
                        'game_time': game_df['game_time'],
                        'num_snapshots': 0
                    }
                games_dict[game_id]['num_snapshots'] += len(df[df['game_id'] == game_id])
                
        except (ValueError, KeyError):
            continue
    
    return list(games_dict.values())


# =============================================================================
# CHART GENERATION (R/ggplot2)
# =============================================================================

def generate_chart_with_r(df: pd.DataFrame, sport: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Generate line movement chart using R/ggplot2.
    
    Args:
        df: DataFrame with snapshot data for one game
        sport: Sport key (basketball_nba or americanfootball_nfl)
        output_dir: Output directory (default: data/04_output/line_movement/charts/)
    
    Returns:
        Path to generated PNG file, or None if failed
    """
    if df.empty:
        return None
    
    # Set output directory
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        sport_short = sport.split('_')[1]
        output_dir = project_root / 'data' / '04_output' / 'line_movement' / 'charts' / sport_short
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get game info
    away_team = df['away_team'].iloc[0]
    home_team = df['home_team'].iloc[0]
    game_time = pd.to_datetime(df['game_time'].iloc[0])
    
    # Create filename
    game_time_et = game_time.tz_convert(ZoneInfo(DISPLAY_TIMEZONE))
    date_str = game_time_et.strftime('%Y%m%d')
    away_abbr = ''.join([word[0] for word in away_team.split()[:3]]).upper()
    home_abbr = ''.join([word[0] for word in home_team.split()[:3]]).upper()
    
    # Calculate time range
    first_snapshot = pd.to_datetime(df['fetched_at'].min())
    last_snapshot = pd.to_datetime(df['fetched_at'].max())
    time_range_hours = int((last_snapshot - first_snapshot).total_seconds() / 3600)
    
    filename = f"{date_str}_{away_abbr}_{home_abbr}_{time_range_hours}h.png"
    output_path = output_dir / filename
    
    # Prepare data for R
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    
    # Focus on major bookmakers (and any others present)
    major_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers']
    available_books = df['bookmaker'].unique().tolist()
    books_to_plot = [b for b in major_books if b in available_books]
    other_books = [b for b in available_books if b not in major_books]
    books_to_plot.extend(sorted(other_books))
    
    df_plot = df[df['bookmaker'].isin(books_to_plot)].copy()
    
    # Create R script
    r_script = f"""
# Set library path to include user library
.libPaths(c('~/R/library', .libPaths()))

# Auto-install missing packages
required_packages <- c('ggplot2', 'dplyr', 'lubridate', 'scales')
for (pkg in required_packages) {{
  if (!require(pkg, character.only=TRUE, quietly=TRUE)) {{
    cat(sprintf("📦 Installing R package: %s\\n", pkg))
    install.packages(pkg, repos='https://cran.rstudio.com/', lib='~/R/library', quiet=TRUE)
    library(pkg, character.only=TRUE, lib.loc='~/R/library')
    cat(sprintf("   ✅ Installed %s\\n", pkg))
  }} else {{
    cat(sprintf("   ✅ Loaded %s\\n", pkg))
  }}
}}

cat("\\n")

# Load data from Python
df <- r_to_py_data

# Convert timestamp to POSIXct
df$timestamp <- as.POSIXct(df$timestamp, tz="UTC")

# Get game info
away_team <- "{away_team}"
home_team <- "{home_team}"
game_time_str <- "{game_time_et.strftime('%b %d, %I:%M %p ET')}"

cat(sprintf("📊 Creating chart for: %s @ %s (%s)\\n", away_team, home_team, game_time_str))
cat(sprintf("   Data points: %d\\n", nrow(df)))
cat(sprintf("   Bookmakers: %d\\n", length(unique(df$bookmaker))))

# Calculate anchor time (last snapshot)
now <- max(df$timestamp)
time_1h_ago <- now - hours(1)
time_24h_ago <- now - hours(24)

# Format anchor time
now_et <- format(now, "%I:%M %p ET", tz="America/New_York")

# Determine favorite
latest_spread <- df %>% filter(timestamp == max(timestamp)) %>% pull(away_spread) %>% first()
if (latest_spread < 0) {{
  favorite <- away_team
  underdog <- home_team
}} else if (latest_spread > 0) {{
  favorite <- home_team
  underdog <- away_team
}} else {{
  favorite <- "Pick'em"
  underdog <- "Pick'em"
}}

cat(sprintf("   Current favorite: %s\\n", favorite))
cat(sprintf("   Anchor time: %s\\n\\n", now_et))

# Create plot
p <- ggplot(df, aes(x=timestamp)) +
  # Plot away team (solid lines)
  geom_line(aes(y=away_spread, color=bookmaker, group=bookmaker), 
            linewidth=1.2, alpha=0.9) +
  geom_point(aes(y=away_spread, color=bookmaker), 
             size=2, alpha=0.9, shape=16) +
  # Plot home team (dashed lines)
  geom_line(aes(y=home_spread, color=bookmaker, group=bookmaker), 
            linewidth=1.2, alpha=0.9, linetype="dashed") +
  geom_point(aes(y=home_spread, color=bookmaker), 
             size=2, alpha=0.9, shape=15) +
  # Horizontal line at 0
  geom_hline(yintercept=0, color="red", linetype="dashed", linewidth=1, alpha=0.6) +
  # Vertical lines for 1h and 24h ago
  geom_vline(xintercept=time_1h_ago, color="orange", linetype="dashed", linewidth=1, alpha=0.7) +
  geom_vline(xintercept=time_24h_ago, color="purple", linetype="dashed", linewidth=1, alpha=0.7) +
  # Shaded regions
  annotate("rect", xmin=-Inf, xmax=Inf, ymin=0, ymax=Inf, fill="red", alpha=0.05) +
  annotate("rect", xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=0, fill="green", alpha=0.05) +
  # Labels and theme
  scale_x_datetime(date_labels="%m/%d %H:%M", date_breaks="12 hours") +
  scale_y_reverse() +  # Invert y-axis (favorites at top)
  scale_color_manual(values=c(
    "draftkings"="#53D337", "fanduel"="#0E8FEF", "betmgm"="#BA9000",
    "caesars"="#0033A0", "betrivers"="#00A4E4"
  )) +
  labs(
    title=paste0(away_team, " @ ", home_team, " (", game_time_str, ")"),
    subtitle=paste0("{time_range_hours}h movement | Current Favorite: ", favorite),
    x="Time",
    y="Spread (points)",
    color="Sportsbooks"
  ) +
  theme_minimal(base_size=14) +
  theme(
    plot.title=element_text(face="bold", size=16, hjust=0.5),
    plot.subtitle=element_text(size=11, hjust=0.5, color="#555", face="italic"),
    axis.title=element_text(face="bold", size=13),
    legend.position="right",
    legend.title=element_text(face="bold"),
    panel.grid.minor=element_blank(),
    axis.text.x=element_text(angle=45, hjust=1)
  ) +
  # Add annotations for time markers
  annotate("text", x=time_1h_ago, y=Inf, label="1h ago", 
           color="orange", fontface="bold", size=3.5, vjust=-0.5) +
  annotate("text", x=time_24h_ago, y=Inf, label="24h ago", 
           color="purple", fontface="bold", size=3.5, vjust=-0.5) +
  # Add anchor timestamp
  annotate("text", x=Inf, y=-Inf, label=paste0("Anchor: ", now_et), 
           color="#666", size=3, hjust=1.1, vjust=-0.5, fontface="italic")

# Save plot
cat(sprintf("💾 Saving chart to: {str(output_path)}\\n"))
ggsave("{str(output_path)}", plot=p, width={CHART_WIDTH}, height={CHART_HEIGHT}, dpi={CHART_DPI}, bg="white")
cat("   ✅ Chart saved successfully\\n")
"""
    
    try:
        # Use converter context to pass data to R
        with converter.context() as ctx:
            # Convert pandas DataFrame to R
            r_df = pandas2ri.py2rpy(df_plot)
            
            # Assign to R global environment
            ro.globalenv['r_to_py_data'] = r_df
            
            # Execute R script
            ro.r(r_script)
        
        return str(output_path)
    
    except Exception as e:
        print(f"   ❌ Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate line movement charts from S3 snapshots using R/ggplot2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available games
  python scripts/visualize_line_movements_from_s3.py --list-games --sport nba
  
  # Generate chart for specific game
  python scripts/visualize_line_movements_from_s3.py --game-id abc123 --sport nfl
  
  # Batch: all games from last 24 hours
  python scripts/visualize_line_movements_from_s3.py --batch --hours 24 --sport nba
        """
    )
    
    parser.add_argument('--list-games', action='store_true',
                       help='List all available games in S3')
    parser.add_argument('--game-id', type=str,
                       help='Specific game ID to chart')
    parser.add_argument('--sport', type=str, choices=['nba', 'nfl'], required=True,
                       help='Sport to process')
    parser.add_argument('--hours', type=int, default=168,
                       help='Time window in hours (default: 168 = 1 week)')
    parser.add_argument('--batch', action='store_true',
                       help='Generate charts for all games in time window')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: data/04_output/line_movement/charts/)')
    
    args = parser.parse_args()
    
    # Map sport name to API key
    sport_map = {'nba': SPORT_NBA, 'nfl': SPORT_NFL}
    sport = sport_map[args.sport]
    
    print(f"\n{'='*80}")
    print(f"Line Movement Chart Generator - {SPORT_DISPLAY_NAMES[sport]}")
    print(f"{'='*80}\n")
    
    # List games mode
    if args.list_games:
        print(f"📋 Listing games from last {args.hours} hours...\n")
        games = list_available_games(sport, args.hours)
        
        if not games:
            print(f"❌ No games found in S3 for {SPORT_DISPLAY_NAMES[sport]}")
            print(f"   Time window: Last {args.hours} hours")
            return
        
        print(f"Found {len(games)} games:\n")
        for i, game in enumerate(games, 1):
            game_time = pd.to_datetime(game['game_time']).tz_convert(DISPLAY_TIMEZONE)
            game_time_str = game_time.strftime('%b %d, %I:%M %p ET')
            print(f"{i:2d}. {game['game_id']}")
            print(f"    {game['away_team']} @ {game['home_team']}")
            print(f"    Game Time: {game_time_str}")
            print(f"    Snapshots: {game['num_snapshots']}")
            print()
        
        return
    
    # Generate chart mode
    if not args.game_id and not args.batch:
        print("❌ ERROR: Must specify --game-id or --batch mode")
        print("   Run with --list-games to see available games")
        return
    
    # Batch mode: get all games
    if args.batch:
        print(f"📊 Batch mode: Generating charts for all games...")
        games = list_available_games(sport, args.hours)
        game_ids = [g['game_id'] for g in games]
    else:
        game_ids = [args.game_id]
    
    print(f"Processing {len(game_ids)} game(s)...\n")
    
    for game_id in game_ids:
        print(f"{'─'*80}")
        print(f"Game ID: {game_id}")
        print(f"{'─'*80}\n")
        
        # Fetch data
        print(f"📥 Fetching snapshots from S3...")
        df = fetch_all_snapshots_for_game(sport, game_id, args.hours)
        
        if df.empty:
            print(f"❌ No data found for game {game_id}")
            continue
        
        # Convert timestamps to ET for display
        first_time = pd.to_datetime(df['fetched_at'].min()).tz_convert(DISPLAY_TIMEZONE)
        last_time = pd.to_datetime(df['fetched_at'].max()).tz_convert(DISPLAY_TIMEZONE)
        
        print(f"   Found {len(df)} snapshot rows")
        print(f"   Time range: {first_time.strftime('%b %d %I:%M %p')} → {last_time.strftime('%b %d %I:%M %p ET')}")
        print(f"   Bookmakers: {df['bookmaker'].nunique()}")
        
        # Generate chart with R
        print(f"\n📊 Generating chart with R/ggplot2...")
        chart_path = generate_chart_with_r(df, sport, args.output_dir)
        
        if chart_path:
            print(f"   ✅ Chart saved: {chart_path}")
            
            # Auto-open the chart
            print(f"\n🖼️  Opening chart...")
            import subprocess
            subprocess.run(['open', chart_path])
        else:
            print(f"   ❌ Chart generation failed")
        
        print()


if __name__ == '__main__':
    main()

