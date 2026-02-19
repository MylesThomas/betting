"""
Plot live odds movement for all games of a specified team.

Creates publication-quality plots with:
- Team colors for lines (not bookmaker colors)
- Team logos in top corners
- Separate lines for each bookmaker with alpha transparency
- Score progression with color-coded leader
- Saves locally to ~/Downloads/tmp/live_odds_plots/
- Uploads to S3 for backup

Usage:
    python scripts/tmp/plot_live_odds_by_team.py --team "Milwaukee Bucks"
    python scripts/tmp/plot_live_odds_by_team.py --team "Los Angeles Lakers"
"""

import duckdb
import boto3
import pandas as pd
import subprocess
import tempfile
import os
import argparse
import requests
import urllib3
from pathlib import Path
from PIL import Image
from io import BytesIO

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
S3_BUCKET = "nba-betting-mt"
S3_PREFIX = "data/04_output/live_odds_plots"

# Image cache directory
IMAGE_CACHE_DIR = Path.home() / "Downloads" / "tmp" / "live_odds_plots" / "logos"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# NBA Team Colors (primary colors)
TEAM_COLORS = {
    "Atlanta Hawks": "#E03A3E",
    "Boston Celtics": "#007A33",
    "Brooklyn Nets": "#000000",
    "Charlotte Hornets": "#1D1160",
    "Chicago Bulls": "#CE1141",
    "Cleveland Cavaliers": "#860038",
    "Dallas Mavericks": "#00538C",
    "Denver Nuggets": "#0E2240",
    "Detroit Pistons": "#C8102E",
    "Golden State Warriors": "#1D428A",
    "Houston Rockets": "#CE1141",
    "Indiana Pacers": "#002D62",
    "LA Clippers": "#C8102E",
    "Los Angeles Lakers": "#552583",
    "Memphis Grizzlies": "#5D76A9",
    "Miami Heat": "#98002E",
    "Milwaukee Bucks": "#00471B",
    "Minnesota Timberwolves": "#0C2340",
    "New Orleans Pelicans": "#0C2340",
    "New York Knicks": "#006BB6",
    "Oklahoma City Thunder": "#007AC1",
    "Orlando Magic": "#0077C0",
    "Philadelphia 76ers": "#006BB6",
    "Phoenix Suns": "#1D1160",
    "Portland Trail Blazers": "#E03A3E",
    "Sacramento Kings": "#5A2D81",
    "San Antonio Spurs": "#C4CED4",
    "Toronto Raptors": "#CE1141",
    "Utah Jazz": "#002B5C",
    "Washington Wizards": "#002B5C",
}

# Team Logos (ESPN CDN)
TEAM_LOGOS = {
    "Atlanta Hawks": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
    "Boston Celtics": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
    "Brooklyn Nets": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
    "Charlotte Hornets": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
    "Chicago Bulls": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
    "Cleveland Cavaliers": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
    "Dallas Mavericks": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
    "Denver Nuggets": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
    "Detroit Pistons": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
    "Golden State Warriors": "https://a.espncdn.com/i/teamlogos/nba/500/gsw.png",
    "Houston Rockets": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
    "Indiana Pacers": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
    "LA Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
    "Los Angeles Lakers": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
    "Memphis Grizzlies": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
    "Miami Heat": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
    "Milwaukee Bucks": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
    "Minnesota Timberwolves": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
    "New Orleans Pelicans": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
    "New York Knicks": "https://a.espncdn.com/i/teamlogos/nba/500/ny.png",
    "Oklahoma City Thunder": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
    "Orlando Magic": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
    "Philadelphia 76ers": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
    "Phoenix Suns": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
    "Portland Trail Blazers": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
    "Sacramento Kings": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
    "San Antonio Spurs": "https://a.espncdn.com/i/teamlogos/nba/500/sa.png",
    "Toronto Raptors": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
    "Utah Jazz": "https://a.espncdn.com/i/teamlogos/nba/500/utah.png",
    "Washington Wizards": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png",
}


def download_team_logo(team_name, size=(100, 100)):
    """Download team logo from ESPN CDN and return cached file path."""
    # Check cache first
    cache_filename = f"team_{team_name.replace(' ', '_')}_{size[0]}x{size[1]}.png"
    cache_path = IMAGE_CACHE_DIR / cache_filename
    
    if cache_path.exists():
        return str(cache_path)
    
    if team_name not in TEAM_LOGOS:
        print(f"⚠️  Team '{team_name}' not found in TEAM_LOGOS dict")
        return None
    
    url = TEAM_LOGOS[team_name]
    
    try:
        response = requests.get(url, timeout=5, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(cache_path, "PNG")
        
        print(f"   ✅ Downloaded logo: {team_name}")
        return str(cache_path)
    except Exception as e:
        print(f"⚠️  Failed to download logo for {team_name}: {e}")
        return None


def main(team_name):
    LOCAL_DIR = Path.home() / "Downloads" / "tmp" / "live_odds_plots"
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect()
    
    # Get AWS credentials
    session = boto3.Session()
    credentials = session.get_credentials()
    s3_client = boto3.client('s3')
    
    # Configure S3 access for DuckDB
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_region='us-east-2';")
    con.execute(f"SET s3_access_key_id='{credentials.access_key}';")
    con.execute(f"SET s3_secret_access_key='{credentials.secret_key}';")
    if credentials.token:
        con.execute(f"SET s3_session_token='{credentials.token}';")
    
    print("=" * 80)
    print(f"LIVE ODDS VISUALIZATION - {team_name}")
    print(f"Local directory: {LOCAL_DIR}")
    print(f"S3 destination: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print("=" * 80)
    
    # Step 1: Get all games for this team (each game_id is unique)
    print(f"\n🔍 Finding all {team_name} games...")
    games_df = con.execute(f"""
        WITH team_games AS (
            SELECT DISTINCT
                game_id,
                away_team,
                home_team,
                CASE 
                    WHEN away_team = '{team_name}' THEN 'away'
                    WHEN home_team = '{team_name}' THEN 'home'
                END as team_side,
                MIN(fetched_at::TIMESTAMP) as game_start,
                MAX(fetched_at::TIMESTAMP) as game_end,
                DATE_TRUNC('day', MIN(fetched_at::TIMESTAMP)) as game_date
            FROM 's3://nba-betting-mt/data/01_input/live_odds/consolidated/the-odds-api.parquet'
            WHERE away_team = '{team_name}' OR home_team = '{team_name}'
            GROUP BY game_id, away_team, home_team
            ORDER BY game_start
        )
        SELECT * FROM team_games
    """).df()
    
    print(f"✅ Found {len(games_df)} games")
    print(games_df[['game_date', 'away_team', 'home_team', 'team_side', 'game_start']].to_string(index=False))
    
    # Step 2: For each game, create a plot
    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        away_team = game['away_team']
        home_team = game['home_team']
        team_side = game['team_side']
        
        print(f"\n{'='*80}")
        print(f"📊 Game {idx+1}/{len(games_df)}: {away_team} @ {home_team}")
        print(f"{'='*80}")
        
        # Download team logos
        away_logo_path = download_team_logo(away_team, size=(100, 100))
        home_logo_path = download_team_logo(home_team, size=(100, 100))
        
        # Get team colors
        away_color = TEAM_COLORS.get(away_team, "#000000")
        home_color = TEAM_COLORS.get(home_team, "#666666")
        
        # Get odds data with all bookmakers (NOT aggregated)
        odds_df = con.execute(f"""
            SELECT 
                fetched_at,
                bookmaker,
                away_spread,
                home_spread,
                away_ml,
                home_ml
            FROM 's3://nba-betting-mt/data/01_input/live_odds/consolidated/the-odds-api.parquet'
            WHERE game_id = '{game_id}'
            ORDER BY fetched_at, bookmaker
        """).df()
        
        # Get ESPN score data
        espn_df = con.execute(f"""
            SELECT DISTINCT
                collection_timestamp as fetched_at,
                away_score,
                home_score,
                period,
                display_clock
            FROM 's3://nba-betting-mt/data/01_input/live_odds/consolidated/espn.parquet'
            WHERE away_team_espn = '{away_team}' 
              AND home_team_espn = '{home_team}'
            ORDER BY fetched_at
        """).df()
        
        if odds_df.empty:
            print(f"⚠️  No odds data found for this game, skipping...")
            continue
        
        # Convert timestamps
        odds_df['fetched_at'] = pd.to_datetime(odds_df['fetched_at'])
        espn_df['fetched_at'] = pd.to_datetime(espn_df['fetched_at'])
        
        # Calculate consensus (median) for each timestamp
        consensus_df = odds_df.groupby('fetched_at').agg({
            'away_spread': 'median',
            'home_spread': 'median',
            'away_ml': 'median',
            'home_ml': 'median'
        }).reset_index()
        
        # Save odds data to temp CSV for R
        temp_odds_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_odds.csv", mode='w')
        odds_df.to_csv(temp_odds_csv.name, index=False)
        temp_odds_csv.close()
        
        # Save consensus data to temp CSV for R
        temp_consensus_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_consensus.csv", mode='w')
        consensus_df.to_csv(temp_consensus_csv.name, index=False)
        temp_consensus_csv.close()
        
        # Save ESPN data to temp CSV for R
        temp_espn_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_espn.csv", mode='w')
        espn_df.to_csv(temp_espn_csv.name, index=False)
        temp_espn_csv.close()
        
        # Create R plot with game date in filename
        game_date_str = str(game['game_date'])[:10]  # YYYY-MM-DD
        safe_filename = f"{away_team.replace(' ', '_')}_at_{home_team.replace(' ', '_')}_{game_date_str}.png"
        plot_file = LOCAL_DIR / safe_filename
        
        # R code for publication-quality plots with team colors and logos
        r_code = f'''
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)
library(png)
library(grid)

# Read data
odds <- read.csv("{temp_odds_csv.name}")
consensus <- read.csv("{temp_consensus_csv.name}")
espn <- read.csv("{temp_espn_csv.name}")

# Convert timestamps
odds$fetched_at <- as.POSIXct(odds$fetched_at, format="%Y-%m-%d %H:%M:%OS")
consensus$fetched_at <- as.POSIXct(consensus$fetched_at, format="%Y-%m-%d %H:%M:%OS")
espn$fetched_at <- as.POSIXct(espn$fetched_at, format="%Y-%m-%d %H:%M:%OS")

# Team colors
away_color <- "{away_color}"
home_color <- "{home_color}"

# Define alpha levels for bookmakers (to distinguish them)
bookmaker_alphas <- c(
  "fanduel" = 0.9,
  "draftkings" = 0.8,
  "betmgm" = 0.7,
  "williamhill_us" = 0.6,
  "betrivers" = 0.9,
  "mybookieag" = 0.8,
  "fanatics" = 0.7,
  "pointsbetus" = 0.6,
  "bovada" = 0.9,
  "betonlineag" = 0.8
)

# Plot 1: Both teams spreads - colored by TEAM + CONSENSUS LINE
odds_spread_long <- odds %>%
  select(fetched_at, bookmaker, away_spread, home_spread) %>%
  pivot_longer(cols = c(away_spread, home_spread), 
               names_to = "team_side", 
               values_to = "spread") %>%
  mutate(
    team = ifelse(team_side == "away_spread", "{away_team}", "{home_team}"),
    team_color = ifelse(team_side == "away_spread", away_color, home_color),
    linetype_val = ifelse(team_side == "away_spread", "solid", "dashed"),
    alpha_val = bookmaker_alphas[bookmaker]
  )

# Prepare consensus for plotting
consensus_spread_long <- consensus %>%
  select(fetched_at, away_spread, home_spread) %>%
  pivot_longer(cols = c(away_spread, home_spread),
               names_to = "team_side",
               values_to = "spread") %>%
  mutate(
    team_color = ifelse(team_side == "away_spread", away_color, home_color),
    linetype_val = ifelse(team_side == "away_spread", "solid", "dashed")
  )

p1 <- ggplot() +
  # Individual bookmaker lines (thin, transparent)
  geom_line(data = odds_spread_long, 
            aes(x = fetched_at, y = spread, 
                color = team_color, 
                linetype = linetype_val,
                alpha = alpha_val,
                group = interaction(bookmaker, team)),
            linewidth = 0.8) +
  # Consensus lines (thick, opaque)
  geom_line(data = consensus_spread_long,
            aes(x = fetched_at, y = spread,
                color = team_color,
                linetype = linetype_val,
                group = team_side),
            linewidth = 2.5, alpha = 1) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "gray50", linewidth = 0.5) +
  scale_color_identity() +
  scale_linetype_identity() +
  scale_alpha_identity() +
  labs(
    title = paste0("{away_team} @ {home_team} - Live Odds Movement"),
    subtitle = "Spreads (Both Teams) - Thick line = Consensus",
    y = "Spread"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    plot.subtitle = element_text(size = 12, color = "gray30", hjust = 0),
    axis.title.x = element_blank(),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    plot.margin = margin(t = 80, r = 10, b = 10, l = 10)
  ) +
  # Manual legend
  annotate("text", x = Inf, y = Inf, label = "{away_team}", 
           color = away_color, hjust = 1.1, vjust = 2, size = 4, fontface = "bold") +
  annotate("text", x = Inf, y = Inf, label = "{home_team}", 
           color = home_color, hjust = 1.1, vjust = 3.5, size = 4, fontface = "bold") +
  annotate("text", x = Inf, y = Inf, label = "(solid = away, dashed = home)", 
           color = "gray40", hjust = 1.1, vjust = 5, size = 3)

# Plot 2: Both teams moneylines - colored by TEAM + CONSENSUS
odds_ml_long <- odds %>%
  filter(!is.na(away_ml) | !is.na(home_ml)) %>%
  select(fetched_at, bookmaker, away_ml, home_ml) %>%
  pivot_longer(cols = c(away_ml, home_ml), 
               names_to = "team_side", 
               values_to = "moneyline") %>%
  filter(!is.na(moneyline)) %>%
  mutate(
    team = ifelse(team_side == "away_ml", "{away_team}", "{home_team}"),
    team_color = ifelse(team_side == "away_ml", away_color, home_color),
    linetype_val = ifelse(team_side == "away_ml", "solid", "dashed"),
    alpha_val = bookmaker_alphas[bookmaker]
  )

consensus_ml_long <- consensus %>%
  filter(!is.na(away_ml) | !is.na(home_ml)) %>%
  select(fetched_at, away_ml, home_ml) %>%
  pivot_longer(cols = c(away_ml, home_ml),
               names_to = "team_side",
               values_to = "moneyline") %>%
  filter(!is.na(moneyline)) %>%
  mutate(
    team_color = ifelse(team_side == "away_ml", away_color, home_color),
    linetype_val = ifelse(team_side == "away_ml", "solid", "dashed")
  )

p2 <- ggplot() +
  # Individual bookmaker lines
  geom_line(data = odds_ml_long,
            aes(x = fetched_at, y = moneyline,
                color = team_color,
                linetype = linetype_val,
                alpha = alpha_val,
                group = interaction(bookmaker, team)),
            linewidth = 0.8) +
  # Consensus lines (thick)
  geom_line(data = consensus_ml_long,
            aes(x = fetched_at, y = moneyline,
                color = team_color,
                linetype = linetype_val,
                group = team_side),
            linewidth = 2.5, alpha = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  scale_color_identity() +
  scale_linetype_identity() +
  scale_alpha_identity() +
  scale_y_continuous(limits = c(-1500, 1500), breaks = seq(-1500, 1500, 500)) +
  labs(
    subtitle = "Moneylines (Both Teams) - Thick line = Consensus",
    y = "Moneyline"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.subtitle = element_text(face = "bold", size = 13, hjust = 0),
    axis.title.x = element_blank(),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90")
  )

# Plot 3: Score progression (if ESPN data available)
if (nrow(espn) > 0 && !all(is.na(espn$away_score))) {{
  p3 <- ggplot(espn, aes(x = fetched_at)) +
    geom_line(aes(y = away_score), color = away_color, linewidth = 1.5) +
    geom_line(aes(y = home_score), color = home_color, linewidth = 1.5) +
    geom_ribbon(aes(ymin = pmin(away_score, home_score), ymax = pmax(away_score, home_score),
                    fill = away_score > home_score), alpha = 0.2) +
    scale_fill_manual(values = c("TRUE" = away_color, "FALSE" = home_color),
                     labels = c("TRUE" = "{away_team} ahead", "FALSE" = "{home_team} ahead"),
                     name = "") +
    labs(
      subtitle = "Score Progression",
      x = "Time",
      y = "Score"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.subtitle = element_text(face = "bold", size = 13, hjust = 0),
      axis.text = element_text(size = 11),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      legend.position = "top"
    )
  
  # Combine all 3 plots
  combined <- p1 / p2 / p3 + plot_layout(heights = c(1.2, 1, 1))
}} else {{
  # No ESPN data - just show odds
  combined <- p1 / p2 + plot_layout(heights = c(1.2, 1))
}}

# Save plot
png("{plot_file}", width = 16, height = 12, units = "in", res = 150, bg = "white")

# Create viewport for entire plot
grid.newpage()
pushViewport(viewport(width = 1, height = 1))

# Draw the combined ggplot
print(combined)

# Add team logos at top (if available)'''
        
        # Add logo rendering code if logos exist
        if away_logo_path and home_logo_path:
            r_code += f'''
grid.raster(readPNG("{away_logo_path}"), x = 0.12, y = 0.95, width = 0.06, height = 0.06, just = c("center", "top"))
grid.raster(readPNG("{home_logo_path}"), x = 0.88, y = 0.95, width = 0.06, height = 0.06, just = c("center", "top"))'''
        
        r_code += '''

dev.off()

cat("✅ Plot saved\\n")
'''
        
        # Execute R code
        try:
            result = subprocess.run(
                ['Rscript', '-e', r_code],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"❌ R plotting failed:")
                print(result.stderr)
                # Clean up temp files
                os.unlink(temp_odds_csv.name)
                os.unlink(temp_espn_csv.name)
                continue
            
            print(f"💾 Saved locally: {plot_file}")
            
            # Upload to S3
            s3_key = f"{S3_PREFIX}/{safe_filename}"
            s3_client.upload_file(str(plot_file), S3_BUCKET, s3_key)
            print(f"☁️  Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
            
            # Clean up temp files
            os.unlink(temp_odds_csv.name)
            os.unlink(temp_consensus_csv.name)
            os.unlink(temp_espn_csv.name)
            
        except Exception as e:
            print(f"❌ Error creating plot: {e}")
            # Clean up temp files
            if os.path.exists(temp_odds_csv.name):
                os.unlink(temp_odds_csv.name)
            if os.path.exists(temp_consensus_csv.name):
                os.unlink(temp_consensus_csv.name)
            if os.path.exists(temp_espn_csv.name):
                os.unlink(temp_espn_csv.name)
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Done! Created {len(games_df)} plots")
    print(f"📂 Local: {LOCAL_DIR}")
    print(f"☁️  S3: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot live odds movement for a team")
    parser.add_argument("--team", type=str, default="Milwaukee Bucks", help="Team name (e.g., 'Milwaukee Bucks')")
    
    args = parser.parse_args()
    
    main(args.team)
