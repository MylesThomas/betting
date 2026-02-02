"""
Plot Line Movement for a Specific Team's Game

PURPOSE:
Quick utility to visualize line movement for a specific team on a specific date.
Uses R/ggplot2 for publication-quality charts.

USAGE:
    # Default (today's date, full team name)
    python scripts/plot_team_line_movement.py --team "San Antonio Spurs"
    
    # Specific date
    python scripts/plot_team_line_movement.py --team "San Antonio Spurs" --date 2026-01-31
    
    # Partial team name (case-insensitive matching)
    python scripts/plot_team_line_movement.py --team "spurs" --date 2026-01-31
    
WHAT IT DOES:
1. Loads all NBA line movement snapshots from S3
2. Finds the game for the specified team on the specified date
3. Generates publication-quality plot using R/ggplot2
4. Saves to ~/Downloads/tmp/ and optionally displays

CHART FEATURES (R/ggplot2):
- Solid lines = away team spreads, points = circles
- Dashed lines = home team spreads, points = squares
- Different colors per sportsbook (brand colors)
- Horizontal lines show opening & current consensus
- Vertical lines at 24h ago, 1h ago, and now
- Inverted y-axis (favorites at top)
- Green zone = favorites, Red zone = underdogs
- High-resolution output (150 DPI)

AUTHOR: Thomas Myles
DATE: 2026-01-31
"""

import pandas as pd
import boto3
from io import BytesIO
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import argparse
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'betting-line-movement-snapshots'
SPORT = 'basketball_nba'
S3_PREFIX = f'data/01_input/the-odds-api/nba/line_movement/'
ET_TZ = ZoneInfo('America/New_York')

# =============================================================================
# S3 HELPERS
# =============================================================================

def list_s3_snapshots(days_back=7, hours_back=None):
    """
    List all snapshot files in S3 for NBA.
    
    Args:
        days_back: How many days back to load (default: 7)
        hours_back: How many hours back to load (overrides days_back if provided)
    """
    s3 = boto3.client('s3')
    
    # Use hours_back if provided, otherwise use days_back
    if hours_back is not None:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        time_desc = f"{hours_back} hours"
    else:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_back)
        time_desc = f"{days_back} days"
    
    cutoff_date_str = cutoff_time.strftime('%Y-%m-%d')
    
    print(f"📥 Loading NBA snapshots from S3...")
    print(f"   Bucket: {S3_BUCKET}")
    print(f"   Prefix: {S3_PREFIX}")
    print(f"   Look back: {time_desc}")
    print(f"   Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    except Exception as e:
        print(f"❌ Error accessing S3: {e}")
        sys.exit(1)
    
    if 'Contents' not in response:
        print(f"❌ No snapshots found in S3")
        sys.exit(1)
    
    # Filter by date/time
    files_to_load = []
    for obj in response['Contents']:
        key = obj['Key']
        if not key.endswith('.csv') or 'snapshot_' not in key:
            continue
        
        try:
            filename = key.split('/')[-1]
            if filename.startswith('snapshot_'):
                # Format: snapshot_YYYYMMDD_HHMMSS.csv (no dashes/colons)
                # Extract timestamp
                timestamp_str = filename.replace('snapshot_', '').replace('.csv', '')
                parts = timestamp_str.split('_')
                
                if len(parts) == 2:
                    date_part = parts[0]  # YYYYMMDD
                    time_part = parts[1]  # HHMMSS
                    
                    # Parse: YYYYMMDD -> YYYY-MM-DD, HHMMSS -> HH:MM:SS
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    
                    # Parse to datetime for hours-based filtering
                    file_time = datetime.strptime(f"{formatted_date} {formatted_time}", '%Y-%m-%d %H:%M:%S')
                    file_time = file_time.replace(tzinfo=timezone.utc)
                    
                    # Check if file is within cutoff window
                    if file_time >= cutoff_time:
                        files_to_load.append(key)
        except Exception:
            continue
    
    print(f"   Found {len(files_to_load)} snapshot files")
    return files_to_load


def load_snapshots_from_s3(s3_keys):
    """Load all snapshots from S3."""
    s3 = boto3.client('s3')
    all_dfs = []
    
    for key in s3_keys:
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(BytesIO(response['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        print(f"❌ No valid snapshot files found")
        sys.exit(1)
    
    df = pd.concat(all_dfs, ignore_index=True)
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
    
    print(f"   Loaded {len(df):,} line records")
    print(f"   Date range: {df['fetched_at'].min()} to {df['fetched_at'].max()}")
    print(f"   Unique games: {df['game_id'].nunique():,}")
    
    return df


def find_team_game(df, team_name, date_str):
    """
    Find the game for a specific team on a specific date.
    
    Args:
        df: DataFrame with all snapshots
        team_name: Team name (case-insensitive, can be partial)
        date_str: Date string in YYYY-MM-DD format (ET timezone)
    
    Returns:
        (game_id, game_df) or (None, None) if not found
    """
    # Parse target date
    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Add game_date_et column
    df['game_date_et'] = df['game_time'].dt.tz_convert(ET_TZ).dt.date
    
    # Filter to target date
    date_games = df[df['game_date_et'] == target_date].copy()
    
    if len(date_games) == 0:
        return None, None
    
    # Find team (case-insensitive partial match)
    team_lower = team_name.lower()
    
    matching_games = date_games[
        (date_games['away_team'].str.lower().str.contains(team_lower)) |
        (date_games['home_team'].str.lower().str.contains(team_lower))
    ]
    
    if len(matching_games) == 0:
        return None, None
    
    # Get unique game
    game_id = matching_games['game_id'].iloc[0]
    game_df = df[df['game_id'] == game_id].copy()
    
    return game_id, game_df


# =============================================================================
# CHART GENERATION (R/ggplot2 for publication quality)
# =============================================================================

def create_line_movement_chart_with_r(df, output_path=None):
    """
    Create line movement chart using R/ggplot2 (publication quality).
    
    Args:
        df: DataFrame with snapshot data for one game
        output_path: Where to save the chart (default: ~/Downloads/tmp/)
    
    Returns:
        Path to saved chart
    """
    if df.empty:
        return None
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    
    # Get game info
    away_team = df['away_team'].iloc[0]
    home_team = df['home_team'].iloc[0]
    game_time = pd.to_datetime(df['game_time'].iloc[0])
    game_time_et = game_time.tz_convert(ET_TZ)
    
    # Calculate time range
    first_snapshot = df['timestamp'].min()
    last_snapshot = df['timestamp'].max()
    time_range_hours = int((last_snapshot - first_snapshot).total_seconds() / 3600)
    
    # Select books to plot
    major_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers']
    available_books = df['bookmaker'].unique().tolist()
    books_to_plot = [b for b in major_books if b in available_books]
    other_books = [b for b in available_books if b not in major_books]
    books_to_plot.extend(sorted(other_books))
    
    df_plot = df[df['bookmaker'].isin(books_to_plot)].copy()
    
    print(f"📊 Plotting {len(books_to_plot)} bookmakers: {', '.join(books_to_plot)}")
    
    # Default output path: ~/Downloads/tmp/adhoc_line_movement_YYYYMMDD_TEAM1_TEAM2.png
    if output_path is None:
        output_dir = Path.home() / 'Downloads' / 'tmp'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = game_time_et.strftime('%Y%m%d')
        away_abbr = ''.join([word[0] for word in away_team.split()[:3]]).upper()
        home_abbr = ''.join([word[0] for word in home_team.split()[:3]]).upper()
        filename = f"adhoc_line_movement_{date_str}_{away_abbr}_vs_{home_abbr}.png"
        output_path = output_dir / filename
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            date_str = game_time_et.strftime('%Y%m%d')
            away_abbr = ''.join([word[0] for word in away_team.split()[:3]]).upper()
            home_abbr = ''.join([word[0] for word in home_team.split()[:3]]).upper()
            filename = f"adhoc_line_movement_{date_str}_{away_abbr}_vs_{home_abbr}.png"
            output_path = output_path / filename
    
    # Save data to temp CSV for R
    temp_csv = Path('/tmp/temp_line_movement_data.csv')
    df_plot.to_csv(temp_csv, index=False)
    
    # R script for visualization
    r_script = f"""
# Set library path
.libPaths(c("~/R/library", .libPaths()))

# Load packages
library(ggplot2)
library(dplyr)
library(lubridate)
library(scales)

# Read data
df <- read.csv("{str(temp_csv)}", stringsAsFactors=FALSE)
df$timestamp <- as.POSIXct(df$timestamp, tz="UTC")

# Game info
away_team <- "{away_team}"
home_team <- "{home_team}"
game_time_str <- "{game_time_et.strftime('%b %d, %I:%M %p ET')}"
game_time <- as.POSIXct("{game_time.strftime('%Y-%m-%d %H:%M:%S')}", tz="UTC")

# Calculate consensus
opening_spreads <- df %>% 
  filter(timestamp == min(timestamp)) %>% 
  pull(away_spread)
opening_consensus <- median(opening_spreads)

current_spreads <- df %>% 
  filter(timestamp == max(timestamp)) %>% 
  pull(away_spread)
current_consensus <- median(current_spreads)

# Calculate tipoff consensus (snapshot closest to but before game time)
# If no snapshot before tipoff, use first available snapshot
pre_tipoff_df <- df %>% filter(timestamp <= game_time)
if (nrow(pre_tipoff_df) > 0) {{
  tipoff_time <- max(pre_tipoff_df$timestamp)
  tipoff_spreads <- df %>% 
    filter(timestamp == tipoff_time) %>% 
    pull(away_spread)
  tipoff_consensus <- median(tipoff_spreads)
}} else {{
  # Game hasn't started yet, use most recent snapshot
  tipoff_consensus <- current_consensus
  tipoff_time <- max(df$timestamp)
}}

# Determine favorite
if (current_consensus < 0) {{
  favorite <- away_team
  underdog <- home_team
}} else if (current_consensus > 0) {{
  favorite <- home_team
  underdog <- away_team
}} else {{
  favorite <- "Pick'em"
  underdog <- "Pick'em"
}}

# Time markers
now <- max(df$timestamp)
time_1h_ago <- now - hours(1)
time_24h_ago <- now - hours(24)

# Create plot
p <- ggplot(df, aes(x=timestamp)) +
  # Away team (solid lines, circles)
  geom_line(aes(y=away_spread, color=bookmaker, group=bookmaker), 
            linewidth=1.2, alpha=0.9) +
  geom_point(aes(y=away_spread, color=bookmaker), 
             size=2, alpha=0.9, shape=16) +
  # Home team (dashed lines, squares)
  geom_line(aes(y=home_spread, color=bookmaker, group=bookmaker), 
            linewidth=1.2, alpha=0.9, linetype="dashed") +
  geom_point(aes(y=home_spread, color=bookmaker), 
             size=2, alpha=0.9, shape=15) +
  # Pick'em line
  geom_hline(yintercept=0, color="gray30", linetype="dashed", linewidth=1, alpha=0.6) +
  # Tipoff consensus line (RED)
  geom_hline(yintercept=tipoff_consensus, color="red", linetype="solid", linewidth=2, alpha=0.8) +
  # Time markers
  geom_vline(xintercept=time_24h_ago, color="purple", linetype="dashed", linewidth=1, alpha=0.7) +
  geom_vline(xintercept=time_1h_ago, color="orange", linetype="dashed", linewidth=1, alpha=0.7) +
  geom_vline(xintercept=now, color="darkgreen", linetype="solid", linewidth=1.5, alpha=0.8) +
  # Tipoff time marker (vertical red line)
  geom_vline(xintercept=game_time, color="red", linetype="solid", linewidth=1.5, alpha=0.8) +
  # Shaded regions
  annotate("rect", xmin=-Inf, xmax=Inf, ymin=0, ymax=Inf, fill="red", alpha=0.05) +
  annotate("rect", xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=0, fill="green", alpha=0.05) +
  # Scales
  scale_x_datetime(date_labels="%m/%d %H:%M", date_breaks="12 hours") +
  scale_y_reverse(breaks=seq(-14, 14, by=1)) +
  scale_color_manual(values=c(
    "draftkings"="#53D337", "fanduel"="#0E8FEF", "betmgm"="#BA9000",
    "caesars"="#0033A0", "betrivers"="#00A4E4", "betonlineag"="#FF5722",
    "betus"="#C00000", "bovada"="#8B0000", "fanatics"="#FF1493",
    "lowvig"="#9370DB", "mybookieag"="#008B8B", "williamhill_us"="#E91E63"
  )) +
  labs(
    title=paste0(away_team, " @ ", home_team, " (", game_time_str, ")"),
    subtitle=paste0("{time_range_hours}h movement | Opening: ", sprintf("%+.1f", opening_consensus), 
                   " → Current: ", sprintf("%+.1f", current_consensus)),
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
  # Time marker labels
  annotate("text", x=time_24h_ago, y=Inf, label="24h ago", 
           color="purple", fontface="bold", size=3.5, vjust=-0.5) +
  annotate("text", x=time_1h_ago, y=Inf, label="1h ago", 
           color="orange", fontface="bold", size=3.5, vjust=-0.5) +
  annotate("text", x=now, y=Inf, label="now", 
           color="darkgreen", fontface="bold", size=4, vjust=-0.5) +
  annotate("text", x=game_time, y=Inf, label="TIPOFF", 
           color="red", fontface="bold", size=4, vjust=-0.5) +
  # Tipoff consensus label
  annotate("text", x=min(df$timestamp), y=tipoff_consensus, 
           label=paste0("Tipoff Consensus: ", sprintf("%+.1f", tipoff_consensus)), 
           hjust=0, vjust=-0.5, fontface="bold", size=3.5, color="red",
           fill="white", alpha=0.9)

# Save
ggsave("{str(output_path)}", plot=p, width=16, height=9, dpi=150, bg="white")

cat("✅ Chart saved successfully\\n")
"""
    
    # Save R script to temp file
    temp_r_file = Path('/tmp/temp_viz_line_movement.R')
    with open(temp_r_file, 'w') as f:
        f.write(r_script)
    
    try:
        result = subprocess.run(
            ['Rscript', str(temp_r_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ Chart created with R/ggplot2")
            return str(output_path)
        else:
            print(f"❌ Error creating chart in R:")
            print(result.stderr)
            print("\n💡 Make sure R packages are installed:")
            print("   R -e 'install.packages(c(\"ggplot2\", \"dplyr\", \"lubridate\", \"scales\"), repos=\"https://cran.rstudio.com/\")'")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ R script timed out after 60 seconds")
        return None
    except FileNotFoundError:
        print("❌ Rscript not found. Is R installed?")
        print("   Install: brew install r")
        return None
    finally:
        # Clean up temp files
        if temp_r_file.exists():
            temp_r_file.unlink()
        if temp_csv.exists():
            temp_csv.unlink()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot line movement for a specific team on a specific date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Today's date, full team name
  python scripts/plot_team_line_movement.py --team "San Antonio Spurs"
  
  # Specific date with 72 hour look-back
  python scripts/plot_team_line_movement.py --team "San Antonio Spurs" --date 2026-01-31 --look-back 72h
  
  # Partial team name (case-insensitive)
  python scripts/plot_team_line_movement.py --team "spurs" --date 2026-01-31 --look-back 48h
        """
    )
    
    parser.add_argument('--team', type=str, required=True,
                       help='Team name (can be partial, case-insensitive)')
    parser.add_argument('--date', type=str,
                       help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--look-back', type=str,
                       help='Time window to load snapshots (e.g., "72h", "48h", "7d"). Overrides --days-back.')
    parser.add_argument('--days-back', type=int, default=7,
                       help='How many days back to load snapshots (default: 7, ignored if --look-back is set)')
    parser.add_argument('--no-open', action='store_true',
                       help='Do not auto-open the chart after generating')
    
    args = parser.parse_args()
    
    # Default to today if no date provided
    if args.date is None:
        args.date = datetime.now(ET_TZ).strftime('%Y-%m-%d')
    
    # Parse --look-back parameter
    hours_back = None
    if args.look_back:
        look_back_str = args.look_back.lower()
        try:
            if look_back_str.endswith('h'):
                hours_back = int(look_back_str[:-1])
            elif look_back_str.endswith('d'):
                hours_back = int(look_back_str[:-1]) * 24
            else:
                # Try parsing as integer (assume hours)
                hours_back = int(look_back_str)
        except ValueError:
            print(f"❌ Invalid --look-back format: {args.look_back}")
            print(f"   Use format like: 72h, 48h, 7d")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Team Line Movement Plotter")
    print(f"{'='*80}\n")
    print(f"Team: {args.team}")
    print(f"Date: {args.date} (ET timezone)")
    if hours_back:
        print(f"Look back: {hours_back} hours")
    else:
        print(f"Look back: {args.days_back} days")
    print()
    
    # Load snapshots
    s3_keys = list_s3_snapshots(days_back=args.days_back, hours_back=hours_back)
    df = load_snapshots_from_s3(s3_keys)
    
    # Find team's game
    print(f"\n🔍 Finding game for {args.team} on {args.date}...")
    game_id, game_df = find_team_game(df, args.team, args.date)
    
    if game_id is None:
        print(f"❌ No game found for '{args.team}' on {args.date}")
        print(f"\nAvailable games on {args.date}:")
        
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        df['game_date_et'] = df['game_time'].dt.tz_convert(ET_TZ).dt.date
        date_games = df[df['game_date_et'] == target_date]
        
        if len(date_games) == 0:
            print(f"   (No games found on {args.date})")
        else:
            unique_games = date_games[['game_id', 'away_team', 'home_team', 'game_time']].drop_duplicates('game_id')
            for _, game in unique_games.iterrows():
                game_time_et = game['game_time'].tz_convert(ET_TZ)
                print(f"   - {game['away_team']} @ {game['home_team']} ({game_time_et.strftime('%I:%M %p ET')})")
        
        sys.exit(1)
    
    # Show game info
    away = game_df['away_team'].iloc[0]
    home = game_df['home_team'].iloc[0]
    game_time = pd.to_datetime(game_df['game_time'].iloc[0]).tz_convert(ET_TZ)
    
    print(f"✅ Found game: {away} @ {home}")
    print(f"   Game time: {game_time.strftime('%b %d, %I:%M %p ET')}")
    print(f"   Snapshots: {len(game_df)}")
    print(f"   Bookmakers: {game_df['bookmaker'].nunique()}")
    print(f"   Time range: {game_df['fetched_at'].min()} to {game_df['fetched_at'].max()}")
    
    # Generate chart with R
    print(f"\n📊 Generating chart with R/ggplot2...")
    chart_path = create_line_movement_chart_with_r(game_df)
    
    if chart_path:
        print(f"✅ Chart saved: {chart_path}")
        
        # Auto-open
        if not args.no_open:
            print(f"\n🖼️  Opening chart...")
            import subprocess
            subprocess.run(['open', chart_path])
    else:
        print(f"❌ Chart generation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
