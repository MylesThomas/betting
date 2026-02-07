#!/usr/bin/env Rscript

#' Demo: Player Headshot on Time Series Plot
#'
#' Purpose: Demonstrate how to overlay a player headshot image on a time series plot in R
#' Created: 2026-02-06
#' Context: User requested a tmp file with fake data showing how to load player headshots
#'          on time series plots, inspired by the Monte Carlo visualization
#'
#' Requirements:
#' - ggplot2: for plotting
#' - magick: for image processing
#' - png: for reading PNG files
#' - grid: for adding images to plots
#'
#' Install packages if needed:
#' install.packages(c("ggplot2", "magick", "png", "grid"))

library(ggplot2)
library(magick)
library(png)
library(grid)

# =============================================================================
# CONFIGURATION
# =============================================================================

# NBA player headshot URLs (these are example URLs - ESPN/NBA.com provide these)
PLAYER_HEADSHOTS <- list(
  "LeBron James" = "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/1966.png",
  "Luka Doncic" = "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3945274.png",
  "Stephen Curry" = "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3975.png"
)

OUTPUT_DIR <- "tmp/plots"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

create_fake_timeseries_data <- function(n_points = 48, seed = 42) {
  #' Create fake time series data for Monte Carlo probability
  #'
  #' @param n_points Number of time points (default 48 for minutes in game)
  #' @param seed Random seed for reproducibility
  #' @return data.frame with time and probability columns
  
  set.seed(seed)
  
  # Create realistic probability curve that starts at 50%, drops, then spikes at end
  time_minutes <- seq(0, 48, length.out = n_points)
  
  # Base trend: gradually decreasing
  base_prob <- 50 - (time_minutes * 0.8)
  
  # Add some noise
  noise <- rnorm(n_points, mean = 0, sd = 5)
  
  # Add late game spike (last 10 minutes)
  spike <- ifelse(time_minutes > 38, (time_minutes - 38) * 8, 0)
  
  # Combine and constrain to 0-100
  probability <- pmax(0, pmin(100, base_prob + noise + spike))
  
  # Create actual points scored (step function)
  points_scored <- cumsum(rbinom(n_points, 1, prob = 0.4)) * 0.5  # ~20 points total
  
  data.frame(
    time_minutes = time_minutes,
    probability = probability,
    points_scored = points_scored,
    prop_line = 22  # Over 22 points
  )
}

download_player_headshot <- function(player_name, output_dir = "tmp") {
  #' Download player headshot from URL
  #'
  #' @param player_name Name of the player
  #' @param output_dir Directory to save the image
  #' @return path to downloaded image file
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  url <- PLAYER_HEADSHOTS[[player_name]]
  
  if (is.null(url)) {
    stop(paste("No headshot URL found for", player_name))
  }
  
  output_file <- file.path(output_dir, paste0(gsub(" ", "_", player_name), "_headshot.png"))
  
  # Download using magick
  tryCatch({
    img <- image_read(url)
    image_write(img, output_file, format = "png")
    cat("Downloaded headshot for", player_name, "to", output_file, "\n")
    return(output_file)
  }, error = function(e) {
    warning(paste("Failed to download headshot:", e$message))
    return(NULL)
  })
}

create_plot_with_headshot <- function(df, player_name, headshot_path) {
  #' Create time series plot with player headshot overlay
  #'
  #' @param df data.frame with time_minutes, probability, points_scored
  #' @param player_name Name of the player
  #' @param headshot_path Path to the headshot image file
  #' @return ggplot object
  
  # Create main plot
  p <- ggplot(df, aes(x = time_minutes)) +
    
    # Shaded regions for over/under
    geom_ribbon(aes(ymin = 0, ymax = probability), 
                fill = "#FF6B6B", alpha = 0.3) +
    geom_ribbon(aes(ymin = probability, ymax = 100), 
                fill = "#4ECDC4", alpha = 0.3) +
    
    # Probability line
    geom_line(aes(y = probability), 
              color = "#2E5EAA", size = 1.5) +
    
    # Prop line
    geom_hline(yintercept = 50, 
               linetype = "dashed", 
               color = "#555555", 
               size = 1) +
    
    # Labels and theme
    labs(
      title = paste(player_name, "- Monte Carlo Probability (Over 22.0 pts)"),
      subtitle = "Game: Lakers vs Nets | 2026-02-03",
      x = "Game Time (minutes)",
      y = "Probability (%)",
      caption = "Fake data for demonstration purposes"
    ) +
    
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
      plot.caption = element_text(size = 9, color = "gray50"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(t = 80, r = 20, b = 20, l = 20)  # Extra top margin for headshot
    ) +
    
    scale_x_continuous(breaks = seq(0, 48, 12)) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25))
  
  # Add headshot if available
  if (!is.null(headshot_path) && file.exists(headshot_path)) {
    # Read the image
    img <- readPNG(headshot_path)
    g <- rasterGrob(img, interpolate = TRUE)
    
    # Add as annotation in top-right corner
    p <- p + annotation_custom(
      g,
      xmin = 36, xmax = 48,  # Right side of plot
      ymin = 85, ymax = 105   # Above the plot area
    )
  }
  
  return(p)
}

save_plot <- function(plot_obj, filename, width = 12, height = 8) {
  #' Save plot to file
  #'
  #' @param plot_obj ggplot object
  #' @param filename Output filename
  #' @param width Width in inches
  #' @param height Height in inches
  
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
  }
  
  output_path <- file.path(OUTPUT_DIR, filename)
  
  ggsave(
    filename = output_path,
    plot = plot_obj,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )
  
  cat("Plot saved to:", output_path, "\n")
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main <- function() {
  #' Main execution function
  
  cat("================================================================================\n")
  cat("DEMO: Player Headshot on Time Series Plot\n")
  cat("================================================================================\n\n")
  
  # Configuration
  player_name <- "LeBron James"
  
  # Step 1: Create fake data
  cat("1. Creating fake time series data...\n")
  df <- create_fake_timeseries_data(n_points = 48, seed = 42)
  cat("   - Generated", nrow(df), "time points\n\n")
  
  # Step 2: Download headshot
  cat("2. Downloading player headshot...\n")
  headshot_path <- download_player_headshot(player_name, output_dir = "tmp")
  cat("\n")
  
  # Step 3: Create plot
  cat("3. Creating plot with headshot overlay...\n")
  plot_obj <- create_plot_with_headshot(df, player_name, headshot_path)
  cat("   - Plot created successfully\n\n")
  
  # Step 4: Save plot
  cat("4. Saving plot...\n")
  plot_filename <- paste0("demo_headshot_timeseries_", gsub(" ", "_", player_name), ".png")
  save_plot(
    plot_obj, 
    filename = plot_filename,
    width = 12,
    height = 8
  )
  
  # Step 5: Open the plot
  cat("\n5. Opening plot...\n")
  plot_path <- file.path(OUTPUT_DIR, plot_filename)
  system(paste("open", shQuote(plot_path)))
  cat("   - Plot opened in default viewer\n")
  
  cat("\n================================================================================\n")
  cat("COMPLETE\n")
  cat("================================================================================\n")
  cat("\nNote: This is a demonstration with fake data.\n")
  cat("For production use, integrate with actual game data.\n")
}

# =============================================================================
# RUN
# =============================================================================

if (!interactive()) {
  main()
}
