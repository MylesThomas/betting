#!/bin/bash
#
# Install required R packages for Monte Carlo visualization
#
# Usage:
#   bash scripts/install_r_packages.sh

echo "=================================================="
echo "Installing R packages for Monte Carlo plotting"
echo "=================================================="
echo ""

# List of required packages
PACKAGES=(
    "ggplot2"
    "dplyr"
    "zoo"
    "patchwork"
    "png"
    "grid"
)

echo "📦 Installing ${#PACKAGES[@]} packages..."
echo ""

# Build R command to install all packages
R_CMD="install.packages(c($(printf '"%s",' "${PACKAGES[@]}" | sed 's/,$//')),"
R_CMD="$R_CMD repos='https://cran.rstudio.com/', lib='~/R/library')"

# Run R command
Rscript -e "$R_CMD"

echo ""
echo "=================================================="
echo "✅ Installation complete!"
echo "=================================================="
echo ""
echo "Installed packages:"
for pkg in "${PACKAGES[@]}"; do
    echo "  • $pkg"
done
echo ""
