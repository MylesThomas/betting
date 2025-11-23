#!/bin/bash
#
# Daily Dashboard Update Script
# 
# This script:
# 1. Fetches latest NBA prop data and finds arbitrage opportunities
# 2. Commits and pushes to GitHub
# 3. Streamlit Cloud auto-deploys the updates
#
# Usage:
#   ./scripts/daily_update.sh
#
# Requirements:
#   - ODDS_API_KEY environment variable set (or in .env file)
#   - Git configured with your GitHub credentials
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🏀 TQS NBA Props Dashboard - Daily Update${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Get to project root
cd "$(dirname "$0")/.."

# Check if ODDS_API_KEY is set (either in env or .env file)
if [ -z "$ODDS_API_KEY" ] && [ ! -f .env ]; then
    echo -e "${RED}❌ Error: ODDS_API_KEY not found${NC}"
    echo "   Set it with: export ODDS_API_KEY='your_key_here'"
    echo "   Or create a .env file with: ODDS_API_KEY=your_key_here"
    exit 1
fi

# Step 1: Find arbitrage opportunities across all markets
echo -e "${GREEN}📊 Step 1: Fetching props and finding arbitrage opportunities...${NC}"
echo -e "${YELLOW}   Markets: points, rebounds, assists, threes, blocks, steals, double-double, triple-double, PRA${NC}"
echo ""

python scripts/find_arb_opportunities.py --markets player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to fetch arbitrage data${NC}"
    exit 1
fi

# Step 2: Commit and push
echo ""
echo -e "${GREEN}📤 Step 2: Committing and pushing to GitHub...${NC}"

# Get today's date
TODAY=$(date +%Y-%m-%d)

# Check if there are new arb files to commit
NEW_FILES=$(git status --porcelain data/arbs/arb_*.csv 2>/dev/null | wc -l | tr -d ' ')

if [ "$NEW_FILES" -eq 0 ]; then
    echo -e "${BLUE}ℹ️  No new arbitrage data to commit${NC}"
else
    # Add new arb files
    git add data/arbs/arb_*.csv
    
    # Commit
    git commit -m "Daily update: arbs for ${TODAY}"
    
    # Push
    git push
    
    echo -e "${GREEN}✅ Successfully pushed to GitHub!${NC}"
    echo -e "${BLUE}   Streamlit Cloud will auto-deploy in 1-2 minutes${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ Dashboard update complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${BLUE}View your dashboard at:${NC}"
echo -e "${BLUE}https://tqs-nba-props-dashboard.streamlit.app${NC}"
echo ""

