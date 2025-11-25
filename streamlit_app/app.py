"""
TQS NBA Props Dashboard

==============================================================================
SYSTEM ARCHITECTURE & FILE RELATIONSHIPS
==============================================================================

This dashboard is part of a complete NBA arbitrage betting system. Here's how
all the pieces fit together and the order they run:

1. CONFIGURATION (src/config.py)
   - Auto-calculates current NBA season based on today's date
   - Shared constants used across all scripts
   - Run: Automatically imported by other scripts

2. SETUP: Build Full Roster Cache (ONE-TIME or WEEKLY)
   Run: python scripts/build_full_roster_cache.py
   
   - Queries NBA API for all 525+ players across 30 teams
   - Uses CURRENT_NBA_SEASON from config.py
   - Creates: data/02_cache/nba_full_roster_cache.csv
   - Purpose: Baseline player-to-team mapping for the season
   - When: Run once at season start, then weekly to catch trades

3. DAILY: Find Arbitrage Opportunities (CRON at 7 AM ET)
   Run: python scripts/find_arb_opportunities.py --markets player_points,...
   
   - Queries The Odds API for live prop odds from all bookmakers
   - Finds arbitrage opportunities (combined probability < 100%)
   - Creates: data/04_output/arbs/arb_*_YYYYMMDD.csv (one per day)
   - Uses: ODDS_API_KEY environment variable
   - When: Daily at 7 AM ET (before games start)

4. TEAM MAPPING: Player-to-Team Cache (AUTOMATIC)
   File: data/02_cache/player_team_cache.csv
   
   - Auto-generated when dashboard loads
   - First load: Queries NBA API for each unique player (~3 min)
   - Subsequent loads: Instant (uses cached mappings)
   - Cache Duration: 7 days (trades don't happen daily)
   - Validation: Checks if player's team matches game matchup
   - Auto-refresh: If team doesn't match game, cache is invalidated and rebuilt
   - Priority: Uses nba_full_roster_cache.csv as baseline, then validates
     against tonight's games, then queries API for recent trades
   
   MANUAL REFRESH (For trades, signings, or missing teams):
   To rebuild the complete cache with all 30 teams and 525+ players:
   
     Step 1: Rebuild full roster from NBA API
     $ python scripts/build_full_roster_cache.py
     
     Step 2: Convert to player_team_cache format
     $ python3 << 'EOF'
     import pandas as pd
    from datetime import datetime
    full_roster = pd.read_csv('data/02_cache/nba_full_roster_cache.csv')
    player_team_cache = pd.DataFrame({
         'player_normalized': full_roster['player_normalized'],
         'team': full_roster['team'],
         'timestamp': datetime.now().isoformat()
     })
     player_team_cache = player_team_cache.drop_duplicates(subset=['player_normalized'], keep='first')
    player_team_cache = player_team_cache.sort_values('player_normalized')
    player_team_cache.to_csv('data/02_cache/player_team_cache.csv', index=False)
    print(f"✅ Updated player_team_cache.csv with {len(player_team_cache)} players")
     EOF
   
   Alternatively, use the "Invalidate Cache" button in the dashboard sidebar.

5. THIS DASHBOARD: View & Analyze (ALWAYS RUNNING)
   Run: streamlit run streamlit_app/app.py
   
   - Loads all historical arb files from data/04_output/arbs/
   - Adds team column using cached player-to-team mappings
   - Provides filtering, sorting, and analysis
   - When: Always running on port 8501

DATA FLOW:
---------
NBA API (rosters)
    ↓
build_full_roster_cache.py
    ↓
data/02_cache/nba_full_roster_cache.csv  ←─┐
    ↓                                        │
The Odds API (props)                         │
    ↓                                        │
find_arb_opportunities.py                    │
    ↓                                        │
data/04_output/arbs/arb_*_YYYYMMDD.csv      │
    ↓                                        │
Streamlit Dashboard (YOU ARE HERE)           │
    ↓                               │
Needs player teams? ────────────────┘
    ↓
Check data/player_team_cache.csv
    ↓
If cached & fresh (< 24hrs): Use it (INSTANT)
If not cached: Query NBA API, cache result
    ↓
Display with team column

==============================================================================
LOCAL DEVELOPMENT
==============================================================================

Current setup for local development and testing.
    
Usage:
    streamlit run streamlit_app/app.py
    
Features:
    - View today's arbitrage opportunities
    - Filter by profitable arbs only (is_arb=True)
    - Filter by market, player, team, and game date
    - Historical arb tracking (from saved files)
    - Key metrics and profitability stats
    - Download filtered results as CSV
    - Auto-refreshing team mappings

Performance:
    - First load: ~3 min (builds player-team cache via NBA API)
    - Subsequent loads: < 1 second (uses cached mappings)
    - Cache auto-refreshes after 24 hours

==============================================================================
FUTURE: AWS DEPLOYMENT (Option A - EC2 with Cron)
==============================================================================
    
Architecture:
    EC2 instance (t2.small, ~$15/month)
    ├── Cron job (runs arb finder at 7am ET daily)
    │   └── python scripts/find_arb_opportunities.py --markets player_points,player_rebounds,...
    ├── Streamlit app (always running on port 8501)
    │   └── streamlit run streamlit_app/app.py
    └── Data stored locally at /data/04_output/arbs/
    
Deployment Steps:
    1. Launch EC2 instance (Ubuntu)
    2. Install Python, dependencies
    3. Clone repo, set ODDS_API_KEY
    4. Build initial roster cache: python scripts/build_full_roster_cache.py
    5. Add cron job:
       0 7 * * * cd /home/ubuntu/betting && python scripts/find_arb_opportunities.py
    6. Run Streamlit as systemd service (auto-restart)
    7. Access via EC2 public IP:8501 (or set up domain + SSL)
    
Cost Estimate:
    - EC2 t2.small: ~$15/month (always-on)
    - Elastic IP: Free (if attached)
    - Data transfer: ~$1/month
    - Total: ~$16/month
    
Security:
    - Set up security group (allow only your IP on port 8501)
    - Use SSH key authentication
    - Store API key in environment variable (not in code)
    - Optional: Add nginx reverse proxy + SSL cert
    - No manual refresh button (prevents unauthorized API usage)

FUTURE ENHANCEMENT: Lambda + EventBridge (Option B)
    If we want to save money, we can split:
    - Lambda: Run arb finder at 7am (basically free)
    - EC2 micro: Just run Streamlit (~$8/month)
    - S3: Store arb data (pennies)
    
    But Option A is simpler for now - everything on one box.

==============================================================================
Author: Myles Thomas
Date: 2025-11-22
==============================================================================
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.team_utils_simple import add_team_column_simple
from src.team_utils import get_all_teams


def validate_team_game_mapping(df: pd.DataFrame) -> list:
    """
    Validate that each player's team appears in their game matchup.
    
    Args:
        df: DataFrame with 'player', 'team', and 'game' columns
        
    Returns:
        List of player names with invalid team mappings
    """
    from src.config import NBA_TEAMS
    
    # Create reverse mapping: full name -> abbreviation
    name_to_abbr = {v: k for k, v in NBA_TEAMS.items()}
    
    mismatches = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('team')) or pd.isna(row.get('game')):
            continue
            
        team_abbr = str(row['team']).strip()
        game = str(row['game']).strip()
        
        # Get full team name from abbreviation
        team_full = NBA_TEAMS.get(team_abbr, team_abbr)
        
        # Game format is "TEAM1 @ TEAM2" or "TEAM1 vs TEAM2"
        # Teams in game are full names
        if '@' in game:
            teams_in_game = [t.strip() for t in game.split('@')]
        elif ' vs ' in game:
            teams_in_game = [t.strip() for t in game.split(' vs ')]
        else:
            # Can't parse game format
            continue
        
        # Check if team's full name appears in either side of the game
        if team_full not in teams_in_game:
            mismatches.append(row['player'])
    
    return mismatches


def invalidate_player_team_cache():
    """
    Delete the player team cache file to force a refresh.
    """
    cache_path = Path(__file__).parent.parent / "data" / "02_cache" / "player_team_cache.csv"
    if cache_path.exists():
        cache_path.unlink()
        st.info("✅ Player team cache invalidated. Re-fetching from NBA API...")

# Page config
st.set_page_config(
    page_title="NBA Arbitrage Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants  
DATA_DIR = Path("data/04_output/arbs")
SCRIPT_PATH = "scripts/find_arb_opportunities.py"

# Helper functions
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_all_arbs():
    """Load all arbitrage opportunities from all files (for date filtering)."""
    if not DATA_DIR.exists():
        return None
    
    # Find all arb files
    arb_files = sorted(DATA_DIR.glob("arb_*.csv"))
    
    if not arb_files:
        return None
    
    all_dfs = []
    
    for arb_file in arb_files:
        try:
            df = pd.read_csv(arb_file)
            
            # Extract date from filename (e.g., arb_threes_20251121.csv)
            date_str = arb_file.stem.split('_')[-1]
            file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            
            # Add file_date column for tracking
            df['file_date'] = file_date
            df['source_file'] = arb_file.name
            
            all_dfs.append(df)
        except Exception as e:
            # Skip files that can't be loaded
            continue
    
    if not all_dfs:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    return combined_df


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_latest_arbs():
    """Load the most recent arbitrage opportunities file."""
    if not DATA_DIR.exists():
        return None, None
    
    # Find most recent arb file
    arb_files = sorted(DATA_DIR.glob("arb_*.csv"))
    
    if not arb_files:
        return None, None
    
    latest_file = arb_files[-1]
    
    try:
        df = pd.read_csv(latest_file)
        
        # Extract date from filename (e.g., arb_threes_20251121.csv)
        date_str = latest_file.stem.split('_')[-1]
        file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        
        return df, file_date
    except Exception as e:
        st.error(f"Error loading arb file: {e}")
        return None, None


@st.cache_data(ttl=60)
def get_arb_history():
    """Get list of all historical arb files."""
    if not DATA_DIR.exists():
        return []
    
    arb_files = sorted(DATA_DIR.glob("arb_*.csv"), reverse=True)
    
    history = []
    for file in arb_files:
        try:
            df = pd.read_csv(file)
            date_str = file.stem.split('_')[-1]
            file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            
            # Count total prop markets and actual arbs
            total_props = len(df)
            
            # Filter to only actual arbs for calculations
            arbs_df = df[df['is_arb'] == True] if 'is_arb' in df.columns else pd.DataFrame()
            arbs_count = len(arbs_df)
            
            # Calculate total wagered and profit (only for actual arbs)
            total_wagered = 0
            total_profit = 0
            avg_profit_pct = 0
            
            if len(arbs_df) > 0:
                if 'over_stake' in arbs_df.columns and 'under_stake' in arbs_df.columns:
                    total_wagered = (arbs_df['over_stake'].sum() + arbs_df['under_stake'].sum())
                if 'guaranteed_profit' in arbs_df.columns:
                    total_profit = arbs_df['guaranteed_profit'].sum()
                if 'expected_profit_pct' in arbs_df.columns:
                    avg_profit_pct = arbs_df['expected_profit_pct'].mean()
            
            history.append({
                'date': file_date,
                'file': file.name,
                'prop_markets': total_props,
                'arbs_found': arbs_count,
                'total_wagered': total_wagered,
                'total_profit': total_profit,
                'avg_profit': avg_profit_pct
            })
        except:
            continue
    
    return history




# Main app
def main():
    # Header
    st.title("🏀 TQS NBA Props Dashboard")
    st.markdown("---")
    
    # Show loading message
    with st.spinner("📊 Loading betting data..."):
        # Load ALL data (not just latest)
        df = load_all_arbs()
    
    if df is None or len(df) == 0:
        st.warning("⚠️ No arbitrage opportunities found yet.")
        st.info("Data will be updated automatically at 7:00 AM ET daily.")
        return
    
    # Add team column from cache (simple, read-only)
    with st.spinner("🏀 Loading player teams from cache..."):
        df = add_team_column_simple(df, player_col='player')
        
        # Count missing teams (unique players)
        missing_rows = df['team'].isna().sum()
        if missing_rows > 0:
            missing_players_list = sorted(df[df['team'].isna()]['player'].unique())
            missing_count = len(missing_players_list)
            
            # Format the message with player names
            if missing_count <= 5:
                players_str = ", ".join(missing_players_list)
                st.info(f"ℹ️ {missing_count} player(s) not in cache ({missing_rows} rows): **{players_str}**. Run `python scripts/build_full_roster_cache.py` to update.")
            else:
                players_str = ", ".join(missing_players_list[:5])
                st.info(f"ℹ️ {missing_count} players not in cache ({missing_rows} rows): **{players_str}** ... and {missing_count - 5} more. Run `python scripts/build_full_roster_cache.py` to update.")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Filters")
        
        # Profitable arb filter
        show_only_arbs = st.checkbox("Show Profitable Arb Opportunities Only", value=True, 
                                      help="Show only rows where is_arb=True")
        
        # Date filter (game date) - MOVED TO TOP
        if 'game_time' in df.columns:
            # Parse dates from game_time and convert to ET timezone
            try:
                # Parse game_time as UTC and convert to ET
                df['game_time_et'] = pd.to_datetime(df['game_time'], utc=True).dt.tz_convert('America/New_York')
                df['game_date_et'] = df['game_time_et'].dt.date
                
                all_dates = sorted(df['game_date_et'].unique().tolist(), reverse=True)
                
                # Default to most recent date (latest date in data)
                most_recent_date = all_dates[0] if all_dates else None
                
                # Find index of most recent date (it's at position 1 since 'All' is at position 0)
                default_index = 1 if most_recent_date else 0
                
                selected_date = st.selectbox(
                    "Game Date (ET)",
                    ['All'] + all_dates,
                    index=default_index,  # Default to most recent date, not 'All'
                    help="Filter by game date in Eastern Time (defaults to most recent)"
                )
            except:
                selected_date = 'All'
        else:
            selected_date = 'All'
        
        # Market filter
        if 'market' in df.columns:
            all_markets = ['All'] + sorted(df['market'].unique().tolist())
            selected_market = st.selectbox("Market", all_markets)
        else:
            selected_market = 'All'
        
        # Player filter
        if 'player' in df.columns:
            all_players = ['All'] + sorted(df['player'].unique().tolist())
            selected_player = st.selectbox("Player", all_players, 
                                          help="Search for specific player")
        else:
            selected_player = 'All'
        
        # Team filter
        if 'team' in df.columns:
            all_teams = ['All'] + sorted([t for t in df['team'].unique() if pd.notna(t)])
            selected_team = st.selectbox("Team", all_teams,
                                        help="Filter by NBA team")
        else:
            selected_team = 'All'
        
        st.markdown("---")
        st.header("📊 Additional Filters")
        
        # Sort by selector
        sort_options = {
            "Profit % (Desc.)": ("expected_profit_pct", False),
            "Profit % (Asc.)": ("expected_profit_pct", True),
            "Total Wager (Desc.)": ("total_wager", False),
            "Total Wager (Asc.)": ("total_wager", True),
            "Guaranteed Profit (Desc.)": ("guaranteed_profit", False),
            "Guaranteed Profit (Asc.)": ("guaranteed_profit", True),
            "Player Name (A-Z)": ("player", True),
            "Player Name (Z-A)": ("player", False),
            "Game Time (Earliest First)": ("game_time", True),
            "Game Time (Latest First)": ("game_time", False),
        }
        
        selected_sort = st.selectbox(
            "Sort By",
            list(sort_options.keys()),
            index=0,
            help="Choose how to sort the opportunities table"
        )
        
        sort_column, sort_ascending = sort_options[selected_sort]
        
        # Min profit filter
        min_profit = st.slider("Min Profit %", 0.0, 10.0, 0.0, 0.1)
        
        st.markdown("---")
        st.header("📝 Info")
        st.info("""
        **Scheduled run:** 12:00 PM ET (daily)
        
        **Markets monitored:**
        - Points, Rebounds, Assists
        - Threes, Blocks, Steals
        - Double/Triple Doubles
        - Combined stats
        
        **Data refreshes automatically after scheduled runs.**
        """)
    
    # Apply filters
    filtered_df = df.copy()
    
    # Filter by profitable arbs only
    if show_only_arbs and 'is_arb' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_arb'] == True]
    
    # Filter by market
    if selected_market != 'All' and 'market' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['market'] == selected_market]
    
    # Filter by player
    if selected_player != 'All' and 'player' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['player'] == selected_player]
    
    # Filter by team
    if selected_team != 'All' and 'team' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    
    # Filter by date
    if selected_date != 'All' and 'game_date_et' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['game_date_et'] == selected_date]
    
    # Filter by min profit
    if min_profit > 0 and 'expected_profit_pct' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['expected_profit_pct'] >= min_profit]
    
    # ========================================================================
    # OVERALL METRICS (All Historical Data) - AT THE TOP
    # ========================================================================
    st.subheader("📈 Overall Summary (All Time)")
    
    # Calculate overall metrics from ALL data (not filtered)
    all_arbs_df = df[df['is_arb'] == True] if 'is_arb' in df.columns else pd.DataFrame()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total prop markets across all data
        total_prop_markets = len(df)
        st.metric("🎯 Total Prop Markets", f"{total_prop_markets:,}", 
                 help="Total prop markets analyzed across all dates")
    
    with col2:
        # Total arbs found (only actual arbs, not all props)
        total_arbs = len(all_arbs_df)
        st.metric("✅ Total Arb Opportunities", f"{total_arbs:,}",
                 help="Total arbitrage opportunities found (is_arb=True)")
    
    with col3:
        # Calculate total wagered (only for actual arbs)
        total_wagered = 0
        if len(all_arbs_df) > 0 and 'over_stake' in all_arbs_df.columns and 'under_stake' in all_arbs_df.columns:
            total_wagered = (all_arbs_df['over_stake'].sum() + all_arbs_df['under_stake'].sum())
        
        st.metric("💰 Total Wagered", f"${total_wagered:,.2f}",
                 help="Total amount wagered across all arbs (assuming $100 stake)")
    
    with col4:
        # Calculate total profit (only for actual arbs)
        total_profit = 0
        if len(all_arbs_df) > 0 and 'guaranteed_profit' in all_arbs_df.columns:
            total_profit = all_arbs_df['guaranteed_profit'].sum()
        
        st.metric("💵 Total Profit", f"${total_profit:,.2f}",
                 help="Total guaranteed profit from all arbs")
    
    st.markdown("---")
    
    # ========================================================================
    # DAILY METRICS (Based on Game Date Filter)
    # ========================================================================
    # Show the selected date or "All"
    if selected_date == 'All':
        st.subheader("📊 Daily Summary (All Dates)")
        daily_df = df  # Use all data
    else:
        st.subheader(f"📊 Daily Summary ({selected_date})")
        # Filter to selected date
        daily_df = df[df['game_date_et'] == selected_date] if 'game_date_et' in df.columns else df
    
    # Count daily arbs (is_arb=True)
    daily_arbs_df = daily_df[daily_df['is_arb'] == True] if 'is_arb' in daily_df.columns else pd.DataFrame()
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns([
        .75,  # Games
        1.1,  # Prop Markets
        1.15,  # Arb Opportunities
        1,  # Avg Edge
        1,  # Best Arb
        1.25,  # Total Wagered
        1.5,  # Total Profit
    ])
    
    with col1:
        # Count unique games from selected date(s)
        unique_games = daily_df['game'].nunique() if 'game' in daily_df.columns else 0
        st.metric("🏀 Games", unique_games,
                 help="Number of NBA games for selected date(s)")
    
    with col2:
        # Total props from selected date(s)
        daily_total_props = len(daily_df)
        st.metric("🎯 Prop Markets", daily_total_props,
                 help=f"Total prop markets for selected date(s)")
    
    with col3:
        # Count ONLY arbs from selected date(s)
        daily_arbs_count = len(daily_arbs_df)
        st.metric("✅ Arb Opportunities", daily_arbs_count, 
                 help=f"Profitable arbitrage opportunities (is_arb=True)")
    
    with col4:
        # Calculate avg profit for selected date's arbs
        daily_avg_edge = daily_arbs_df['expected_profit_pct'].mean() if 'expected_profit_pct' in daily_arbs_df.columns and len(daily_arbs_df) > 0 else 0
        st.metric("📈 Avg Edge", f"{daily_avg_edge:.2f}%",
                 help="Average edge (profit %) for arbs")
    
    with col5:
        daily_max_profit = daily_arbs_df['expected_profit_pct'].max() if 'expected_profit_pct' in daily_arbs_df.columns and len(daily_arbs_df) > 0 else 0
        st.metric("🔥 Best Arb", f"{daily_max_profit:.2f}%",
                 help="Highest profit opportunity")
    
    with col6:
        # Calculate total wagered (only for arbs)
        daily_wagered = 0
        if len(daily_arbs_df) > 0 and 'over_stake' in daily_arbs_df.columns and 'under_stake' in daily_arbs_df.columns:
            daily_wagered = (daily_arbs_df['over_stake'].sum() + daily_arbs_df['under_stake'].sum())
        
        st.metric("💰 Total Wagered", f"${daily_wagered:,.2f}",
                 help="Total amount wagered on arbs (assuming $100 stake)")
    
    with col7:
        # Calculate total profit (only for arbs)
        daily_profit = 0
        if len(daily_arbs_df) > 0 and 'guaranteed_profit' in daily_arbs_df.columns:
            daily_profit = daily_arbs_df['guaranteed_profit'].sum()
        
        st.metric("💵 Total Profit", f"${daily_profit:,.2f}",
                 help="Total guaranteed profit from arbs")
    
    st.markdown("---")
    
    # Opportunities table
    st.subheader("🎰 Current Arbitrage Opportunities")
    
    if len(filtered_df) > 0:
        # Format the dataframe for display
        display_df = filtered_df.copy()
        
        # Convert game_time to ET and format nicely (this will replace game_time column)
        if 'game_time_et' in display_df.columns:
            display_df['game_time'] = display_df['game_time_et'].dt.strftime('%I:%M %p ET')
        
        # Drop redundant date/time columns (game_date_et, game_time_et, file_date, source_file)
        cols_to_drop = ['game_time_et', 'game_date_et', 'file_date', 'source_file']
        for col in cols_to_drop:
            if col in display_df.columns:
                display_df = display_df.drop(col, axis=1)
        
        # Reorder columns to put Team right after Player
        # Get all columns and reorder
        cols = display_df.columns.tolist()
        if 'team' in cols and 'player' in cols:
            # Remove team from wherever it is
            cols.remove('team')
            # Insert it right after player
            player_idx = cols.index('player')
            cols.insert(player_idx + 1, 'team')
            # Reorder the dataframe
            display_df = display_df[cols]
        
        # Convert implied probability decimals to percentages (0.55 -> 55.0)
        implied_cols = ['best_over_implied', 'best_under_implied']
        for col in implied_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col] * 100
        
        # Convert combined probability to percentage (0.9662 -> 96.62)
        if 'total_prob' in display_df.columns:
            display_df['total_prob'] = display_df['total_prob'] * 100
        
        # Round numeric columns
        numeric_cols = ['expected_profit_pct', 'guaranteed_profit', 'total_wager', 'over_stake', 'under_stake']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        # Apply sorting
        if sort_column in display_df.columns:
            display_df = display_df.sort_values(sort_column, ascending=sort_ascending)
        
        # Apply color gradient to Profit % column
        def color_profit_gradient(val):
            """
            Color code profit percentages:
            - 10% or higher: very green
            - 0%: white
            - -10% or lower: very red
            - Gradient in between
            """
            if pd.isna(val):
                return ''
            
            # Clamp values between -10 and 10 for color mapping
            clamped = max(-10, min(10, val))
            
            if clamped >= 0:
                # Green gradient from white (0%) to green (10%)
                intensity = int((clamped / 10) * 255)
                return f'background-color: rgb({255 - intensity}, 255, {255 - intensity})'
            else:
                # Red gradient from white (0%) to red (-10%)
                intensity = int((abs(clamped) / 10) * 255)
                return f'background-color: rgb(255, {255 - intensity}, {255 - intensity})'
        
        # Style the dataframe
        styled_df = display_df.style.applymap(
            color_profit_gradient,
            subset=['expected_profit_pct'] if 'expected_profit_pct' in display_df.columns else []
        )
        
        # Display with highlighting
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "player": st.column_config.TextColumn("Player"),
                "team": st.column_config.TextColumn("Team"),
                "market": st.column_config.TextColumn("Market"),
                "line": st.column_config.NumberColumn("Line", format="%.1f"),
                "best_over_odds": st.column_config.NumberColumn(
                    "Over Odds",
                    format="%+d"
                ),
                "best_over_book": st.column_config.TextColumn("Over Book"),
                "best_over_implied": st.column_config.NumberColumn(
                    "Over Implied %",
                    format="%.2f%%"
                ),
                "best_under_odds": st.column_config.NumberColumn(
                    "Under Odds",
                    format="%+d"
                ),
                "best_under_book": st.column_config.TextColumn("Under Book"),
                "best_under_implied": st.column_config.NumberColumn(
                    "Under Implied %",
                    format="%.2f%%"
                ),
                "total_prob": st.column_config.NumberColumn(
                    "Combined Probability",
                    format="%.2f%%",
                    help="Total implied probability. < 100% = guaranteed profit"
                ),
                "expected_profit_pct": st.column_config.NumberColumn(
                    "Profit %",
                    format="%.2f%%"
                ),
                "is_arb": st.column_config.CheckboxColumn(
                    "Is Arb?",
                    help="True when combined probability < 1.0"
                ),
                "over_stake": st.column_config.NumberColumn(
                    "Over Stake",
                    format="$%.2f"
                ),
                "under_stake": st.column_config.NumberColumn(
                    "Under Stake",
                    format="$%.2f"
                ),
                "over_return": st.column_config.NumberColumn(
                    "Over Return",
                    format="$%.2f"
                ),
                "under_return": st.column_config.NumberColumn(
                    "Under Return",
                    format="$%.2f"
                ),
                "guaranteed_profit": st.column_config.NumberColumn(
                    "Guaranteed Profit",
                    format="$%.2f"
                ),
                "total_wager": st.column_config.NumberColumn(
                    "Total Wager",
                    format="$%.2f"
                ),
                "recommendation": st.column_config.TextColumn("Recommendation"),
                "game": st.column_config.TextColumn("Game"),
                "game_time": st.column_config.TextColumn("Game Time (ET)"),
                "num_bookmakers": st.column_config.NumberColumn(
                    "# Bookmakers",
                    format="%d"
                )
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"arb_opportunities_filtered.csv",
            mime="text/csv"
        )
    else:
        st.info("No opportunities match your filter criteria.")
    
    st.markdown("---")
    
    # Historical performance
    st.subheader("📊 Historical Performance")
    
    history = get_arb_history()
    
    if history:
        history_df = pd.DataFrame(history)
        
        st.markdown("**Recent Runs:**")
        
        # Reorder columns to put file at the end
        column_order = ['date', 'prop_markets', 'arbs_found', 'total_wagered', 'total_profit', 'avg_profit', 'file']
        display_history = history_df[column_order].head(10)
        
        # Apply color gradient to avg_profit column in history
        def color_profit_gradient_history(val):
            """Color code profit percentages with gradient."""
            if pd.isna(val):
                return ''
            
            # Clamp values between -10 and 10 for color mapping
            clamped = max(-10, min(10, val))
            
            if clamped >= 0:
                # Green gradient from white (0%) to green (10%)
                intensity = int((clamped / 10) * 255)
                return f'background-color: rgb({255 - intensity}, 255, {255 - intensity})'
            else:
                # Red gradient from white (0%) to red (-10%)
                intensity = int((abs(clamped) / 10) * 255)
                return f'background-color: rgb(255, {255 - intensity}, {255 - intensity})'
        
        # Style the history dataframe
        styled_history = display_history.style.applymap(
            color_profit_gradient_history,
            subset=['avg_profit'] if 'avg_profit' in display_history.columns else []
        )
        
        st.dataframe(
            styled_history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.TextColumn("Date"),
                "prop_markets": st.column_config.NumberColumn("Prop Markets", format="%d"),
                "arbs_found": st.column_config.NumberColumn("Arbs Found", format="%d"),
                "total_wagered": st.column_config.NumberColumn("Total Wagered", format="$%.2f"),
                "total_profit": st.column_config.NumberColumn("Total Profit", format="$%.2f"),
                "avg_profit": st.column_config.NumberColumn("Avg Profit %", format="%.2f%%"),
                "file": st.column_config.TextColumn("File")
            }
        )
    else:
        st.info("No historical data available yet.")


if __name__ == "__main__":
    main()

