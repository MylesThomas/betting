"""
Generate publication-quality Monte Carlo plot using R + ggplot2.

This creates a faceted plot with:
- Top panel: Probability over time (with smoothing)
- Bottom panel: Points scored over time
- Clean legend explaining all visual elements
- 538-style theme

Usage:
    cd /Users/thomasmyles/dev/betting
    python src/pbp_data/tmp/plot_monte_carlo_ggplot.py \\
        --csv src/pbp_data/tmp/plots/monte_carlo_pbp_401809820_2025-11-28.csv \\
        --player "Luka Doncic" \\
        --game-id 401809820 \\
        --game-date "2025-11-28" \\
        --prop-line 33.5 \\
        --final-points 35
"""

import subprocess
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def create_ggplot_monte_carlo(csv_file, player_name, game_id, game_date, prop_line, final_points, output_file):
    """
    Create Monte Carlo plot using R + ggplot2.
    
    Args:
        csv_file: Path to monte_carlo_pbp CSV
        player_name: Player name
        game_id: Game ID
        game_date: Game date (YYYY-MM-DD)
        prop_line: Prop line value
        final_points: Actual final points
        output_file: Output PNG path
    """
    result_label = "HIT" if final_points > prop_line else "MISS"
    
    # R code for ggplot2
    r_code = f'''
library(ggplot2)
library(dplyr)
library(tidyr)

# Read data
df <- read.csv("{csv_file}")

# Calculate smoothed probability (20-play rolling average)
df <- df %>%
  arrange(game_minute) %>%
  mutate(
    prob_over_smooth = zoo::rollmean(prob_over, k=20, fill=NA, align="center")
  )

# Create long format for probability lines (raw + smoothed)
prob_df <- df %>%
  select(game_minute, prob_over, prob_over_smooth) %>%
  pivot_longer(cols = c(prob_over, prob_over_smooth), 
               names_to = "line_type", 
               values_to = "probability")

# Top plot: Probability
p1 <- ggplot() +
  # Shaded regions (over/under 50%)
  geom_ribbon(data = df %>% filter(!is.na(prob_over_smooth)),
              aes(x = game_minute, 
                  ymin = 50, 
                  ymax = prob_over_smooth * 100,
                  fill = ifelse(prob_over_smooth * 100 >= 50, "over", "under")),
              alpha = 0.3) +
  
  # Raw MC probability (thin line)
  geom_line(data = df, 
            aes(x = game_minute, y = prob_over * 100),
            color = "blue", alpha = 0.3, linewidth = 0.5) +
  
  # Smoothed probability (thick line)
  geom_line(data = df %>% filter(!is.na(prob_over_smooth)),
            aes(x = game_minute, y = prob_over_smooth * 100),
            color = "blue", linewidth = 1.5) +
  
  # 50% baseline
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  
  # Quarter markers
  geom_vline(xintercept = c(0, 12, 24, 36), linetype = "dotted", color = "gray50", linewidth = 0.5) +
  
  # Styling
  scale_fill_manual(values = c("over" = "green", "under" = "red"),
                    labels = c("over" = "Prob > 50%", "under" = "Prob < 50%"),
                    name = "") +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_x_continuous(limits = c(0, 48), breaks = seq(0, 48, 12)) +
  labs(title = paste0("{player_name} - Monte Carlo (Over {prop_line} pts)"),
       subtitle = paste0("Game: {game_id} on {game_date} | Vegas-Adjusted (starts at 50%)"),
       y = "Probability (%)") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    plot.subtitle = element_text(size = 12, color = "gray30", hjust = 0),
    axis.title.x = element_blank(),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    legend.position = "top",
    legend.justification = "left",
    legend.box = "horizontal",
    legend.margin = margin(0,0,5,0)
  )

# Bottom plot: Points
p2 <- ggplot(df, aes(x = game_minute)) +
  # Filled area for points
  geom_ribbon(aes(ymin = 0, ymax = cumulative_points), 
              fill = "green", alpha = 0.3) +
  
  # Points line
  geom_line(aes(y = cumulative_points), 
            color = "darkgreen", linewidth = 1.5) +
  
  # Prop line
  geom_hline(yintercept = {prop_line}, linetype = "dashed", color = "red", linewidth = 1.2) +
  
  # Quarter markers
  geom_vline(xintercept = c(0, 12, 24, 36), linetype = "dotted", color = "gray50", linewidth = 0.5) +
  
  # Annotations
  annotate("text", x = 45, y = {prop_line} + 2, 
           label = paste0("Prop: {prop_line}"), 
           color = "red", size = 4, fontface = "bold") +
  
  # Styling
  scale_x_continuous(limits = c(0, 48), breaks = seq(0, 48, 12)) +
  labs(subtitle = paste0("Final: {final_points} pts ({result_label})"),
       x = "Game Time (minutes)",
       y = "Points Scored") +
  theme_minimal(base_size = 14) +
  theme(
    plot.subtitle = element_text(face = "bold", size = 13, hjust = 0),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90")
  )

# Combine plots
library(patchwork)
combined <- p1 / p2 + plot_layout(heights = c(1.2, 1))

# Save
ggsave("{output_file}", combined, width = 14, height = 10, dpi = 150, bg = "white")

cat("✅ Plot saved to {output_file}\\n")
'''
    
    # Write R script to temp file
    temp_r_file = PROJECT_ROOT / 'temp_plot_mc.R'
    with open(temp_r_file, 'w') as f:
        f.write(r_code)
    
    print(f"   🔧 Executing R script...")
    
    try:
        result = subprocess.run(
            ['Rscript', str(temp_r_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"   ✅ Plot created: {output_file}")
            return output_file
        else:
            print(f"❌ R error:")
            print(result.stderr)
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("❌ R script timed out")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Rscript not found. Install R:")
        print("   brew install r")
        sys.exit(1)
    finally:
        if temp_r_file.exists():
            temp_r_file.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to monte_carlo_pbp CSV')
    parser.add_argument('--player', type=str, required=True, help='Player name')
    parser.add_argument('--game-id', type=str, required=True, help='Game ID')
    parser.add_argument('--game-date', type=str, required=True, help='Game date (YYYY-MM-DD)')
    parser.add_argument('--prop-line', type=float, required=True, help='Prop line')
    parser.add_argument('--final-points', type=int, required=True, help='Final points')
    parser.add_argument('--output', type=str, default=None, help='Output PNG path')
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        output_file = Path(args.csv).parent / f"monte_carlo_ggplot_{args.game_id}_{args.game_date}.png"
    else:
        output_file = Path(args.output)
    
    print("="*80)
    print("MONTE CARLO PLOT (R + GGPLOT2)")
    print("="*80)
    print(f"\n📊 Creating plot for {args.player}...")
    print(f"   Game: {args.game_id} on {args.game_date}")
    print(f"   Prop: Over {args.prop_line}, Final: {args.final_points}")
    print()
    
    create_ggplot_monte_carlo(
        args.csv, 
        args.player, 
        args.game_id, 
        args.game_date, 
        args.prop_line, 
        args.final_points,
        str(output_file)
    )
    
    print(f"\n✅ Done! Open with: open {output_file}")


if __name__ == "__main__":
    main()
