# Test R image loading with gt and gtExtras
# Run this with: Rscript testing/test_r_image_loading.R

library(gt)
library(gtExtras)
library(dplyr)

# Test data with NBA headshot URLs
test_data <- data.frame(
  player = c("LeBron James", "Stephen Curry", "Giannis Antetokounmpo"),
  headshot_url = c(
    "https://cdn.nba.com/headshots/nba/latest/1040x760/2544.png",
    "https://cdn.nba.com/headshots/nba/latest/1040x760/201939.png",
    "https://cdn.nba.com/headshots/nba/latest/1040x760/203507.png"
  ),
  points = c(25.4, 26.8, 30.2)
)

# Create table with images
table <- test_data %>%
  gt() %>%
  gt_img_rows(columns = headshot_url, height = 50) %>%
  cols_label(headshot_url = "") %>%
  tab_header(title = "Test Player Headshots")

# Save to PNG
gtsave(table, "content/viz/nba/test_headshots.png", vwidth = 800, vheight = 400, delay = 2)

print("✅ Test table saved to content/viz/nba/test_headshots.png")
print("Check the file to see if images loaded properly")



