"""
Plot live odds movement for all games of a specified team.

Creates publication-quality plots with:
- Separate lines for each bookmaker (spread + moneyline)
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
from pathlib import Path

# Configuration
S3_BUCKET = "nba-betting-mt"
S3_PREFIX = "data/04_output/live_odds_plots"


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
        
        # Save odds data to temp CSV for R
        temp_odds_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_odds.csv", mode='w')
        odds_df.to_csv(temp_odds_csv.name, index=False)
        temp_odds_csv.close()
        
        # Save ESPN data to temp CSV for R
        temp_espn_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_espn.csv", mode='w')
        espn_df.to_csv(temp_espn_csv.name, index=False)
        temp_espn_csv.close()
        
        # Create R plot with game date in filename
        game_date_str = str(game['game_date'])[:10]  # YYYY-MM-DD
        safe_filename = f"{away_team.replace(' ', '_')}_at_{home_team.replace(' ', '_')}_{game_date_str}.png"
        plot_file = LOCAL_DIR / safe_filename
        
        # R code for publication-quality plots
        r_code = f'''
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)

# Read data
odds <- read.csv("{temp_odds_csv.name}")
espn <- read.csv("{temp_espn_csv.name}")

# Convert timestamps
odds$fetched_at <- as.POSIXct(odds$fetched_at, format="%Y-%m-%d %H:%M:%OS")
espn$fetched_at <- as.POSIXct(espn$fetched_at, format="%Y-%m-%d %H:%M:%OS")

# Define color palette for bookmakers (distinct colors)
bookmaker_colors <- c(
  "fanduel" = "#4169E1",      # Royal blue
  "draftkings" = "#53D337",   # Green
  "betmgm" = "#FFD700",       # Gold  
  "williamhill_us" = "#FF6347", # Tomato
  "betrivers" = "#9370DB",    # Medium purple
  "mybookieag" = "#FF1493",   # Deep pink
  "fanatics" = "#00CED1",     # Dark turquoise
  "pointsbetus" = "#FF8C00",  # Dark orange
  "bovada" = "#32CD32",       # Lime green
  "betonlineag" = "#DC143C"   # Crimson
)

# Plot 1: Both teams spreads (each bookmaker separate line)
# Reshape data to long format for both teams
odds_spread_long <- odds %>%
  select(fetched_at, bookmaker, away_spread, home_spread) %>%
  pivot_longer(cols = c(away_spread, home_spread), 
               names_to = "team", 
               values_to = "spread") %>%
  mutate(team = ifelse(team == "away_spread", "{away_team}", "{home_team}"),
         bookmaker_team = paste0(bookmaker, " (", team, ")"))

p1 <- ggplot(odds_spread_long, aes(x = fetched_at, y = spread, color = bookmaker, 
                                     linetype = team, group = bookmaker_team)) +
  geom_line(linewidth = 1.2, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "gray50") +
  scale_color_manual(values = bookmaker_colors, name = "Bookmaker") +
  scale_linetype_manual(values = c("solid", "dashed"), name = "Team") +
  labs(
    title = paste0("{away_team} @ {home_team} - Live Odds Movement"),
    subtitle = "Spreads (Both Teams)",
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
    legend.position = "right"
  )

# Plot 2: Both teams moneylines (each bookmaker separate line)
odds_ml_long <- odds %>%
  filter(!is.na(away_ml) | !is.na(home_ml)) %>%
  select(fetched_at, bookmaker, away_ml, home_ml) %>%
  pivot_longer(cols = c(away_ml, home_ml), 
               names_to = "team", 
               values_to = "moneyline") %>%
  filter(!is.na(moneyline)) %>%
  mutate(team = ifelse(team == "away_ml", "{away_team}", "{home_team}"),
         bookmaker_team = paste0(bookmaker, " (", team, ")"))

p2 <- ggplot(odds_ml_long, aes(x = fetched_at, y = moneyline, color = bookmaker,
                                 linetype = team, group = bookmaker_team)) +
  geom_line(linewidth = 1.2, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = bookmaker_colors, name = "Bookmaker") +
  scale_linetype_manual(values = c("solid", "dashed"), name = "Team") +
  scale_y_continuous(limits = c(-1500, 1500), breaks = seq(-1500, 1500, 500)) +
  labs(
    subtitle = "Moneylines (Both Teams)",
    y = "Moneyline"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.subtitle = element_text(face = "bold", size = 13, hjust = 0),
    axis.title.x = element_blank(),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    legend.position = "right"
  )

# Plot 3: Score progression (if ESPN data available)
if (nrow(espn) > 0 && !all(is.na(espn$away_score))) {{
  p3 <- ggplot(espn, aes(x = fetched_at)) +
    geom_line(aes(y = away_score), color = "#4169E1", linewidth = 1.5) +
    geom_line(aes(y = home_score), color = "#DC143C", linewidth = 1.5) +
    geom_ribbon(aes(ymin = pmin(away_score, home_score), ymax = pmax(away_score, home_score),
                    fill = away_score > home_score), alpha = 0.2) +
    scale_fill_manual(values = c("TRUE" = "#4169E1", "FALSE" = "#DC143C"),
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
  combined <- p1 / p2 / p3 + plot_layout(heights = c(1.2, 1, 1), guides = "collect") &
    theme(legend.position = "right")
}} else {{
  # No ESPN data - just show odds
  combined <- p1 / p2 + plot_layout(heights = c(1.2, 1), guides = "collect") &
    theme(legend.position = "right")
}}

# Save plot
ggsave("{plot_file}", combined, width = 16, height = 12, dpi = 150, bg = "white")

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
            os.unlink(temp_espn_csv.name)
            
        except Exception as e:
            print(f"❌ Error creating plot: {e}")
            # Clean up temp files
            if os.path.exists(temp_odds_csv.name):
                os.unlink(temp_odds_csv.name)
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
