"""
TQS NBA Props Dashboard

NBA Player Props Arbitrage Dashboard - find and track arb opportunities.

DATA HANDLING:
    - Lambda runs every 15 minutes, saving snapshots to data/04_output/nba/arbs/
    - Each file contains arbs found at that moment (lines change frequently)
    - Dashboard loads ALL files for a date and DEDUPES by (player, market, line)
    - For duplicate combos, keeps the row with HIGHEST expected_profit_pct
    - This captures the BEST opportunity that appeared during the day

DEDUPLICATION:
    - Same player/market/line may appear in multiple 15-min snapshots
    - Odds fluctuate, so arb may be 3% at 10am but 5% at 2pm
    - We keep the 5% version (best opportunity)
    - Metrics (total arbs, profit, etc.) calculated on deduped data

LIVE PROP BEHAVIOR (observed):
    - Lines move frequently, especially close to game time
    - Arb opportunities appear and disappear within minutes
    - 15-min snapshots capture most opportunities
    - Best arbs often last < 30 minutes before books adjust
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.team_utils_simple import add_team_column_simple
from src.team_utils import get_all_teams
from src.config import NBA_TEAMS


def validate_team_game_mapping(df: pd.DataFrame) -> list:
    """
    Validate that each player's team appears in their game matchup.
    """
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
        if '@' in game:
            teams_in_game = [t.strip() for t in game.split('@')]
        elif ' vs ' in game:
            teams_in_game = [t.strip() for t in game.split(' vs ')]
        else:
            continue
        
        if team_full not in teams_in_game:
            mismatches.append(row['player'])
    
    return mismatches


def invalidate_player_team_cache():
    """Delete the player team cache file to force a refresh."""
    cache_path = Path(__file__).parent.parent.parent / "data" / "02_cache" / "player_team_cache.csv"
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

# Mobile-responsive CSS
st.markdown("""
<style>
    /* Mobile optimizations */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        
        [data-testid="metric-container"] {
            margin-bottom: 1rem !important;
        }
        
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
        }
        
        [data-testid="stCheckbox"] {
            min-height: 44px !important;
        }
        
        .stButton > button {
            width: 100% !important;
            padding: 0.75rem !important;
        }
        
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
        
        section[data-testid="stSidebar"] {
            display: none;
        }
    }
    
    @media (min-width: 769px) {
        /* Normal desktop styles */
    }
</style>
""", unsafe_allow_html=True)

# Mobile-first responsive design
st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container {
            padding: 0.5rem !important;
            max-width: 100% !important;
        }
        
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.25rem !important;
        }
        
        div[data-testid="stMetric"] {
            width: 100% !important;
            margin-bottom: 0.25rem !important;
            padding: 0.25rem !important;
        }
        
        h1 {
            font-size: 1.25rem !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.3 !important;
        }
        
        h2 {
            font-size: 1rem !important;
            margin-bottom: 0.4rem !important;
            margin-top: 0.4rem !important;
            line-height: 1.3 !important;
        }
        
        h3 {
            font-size: 0.9rem !important;
            margin-bottom: 0.3rem !important;
            line-height: 1.3 !important;
        }
        
        div[data-testid="stMetric"] label {
            font-size: 0.7rem !important;
            line-height: 1.2 !important;
        }
        
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
            line-height: 1.2 !important;
        }
        
        hr {
            margin: 0.5rem 0 !important;
        }
        
        div[data-testid="stDataFrame"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        div[data-testid="stDataFrame"] table {
            font-size: 0.7rem !important;
        }
        
        .stButton > button {
            width: 100% !important;
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
        }
        
        .stDownloadButton > button {
            width: 100% !important;
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
        }
        
        div[data-baseweb="select"] {
            font-size: 0.8rem !important;
        }
        
        .stInfo, .stWarning {
            font-size: 0.8rem !important;
            padding: 0.5rem !important;
        }
        
        .stMarkdown {
            font-size: 0.85rem !important;
        }
        
        .element-container {
            margin-bottom: 0.25rem !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .block-container {
            padding: 2rem !important;
        }
        
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        
        div[data-testid="stMetric"] {
            min-width: 45% !important;
        }
    }
    
    @media (max-width: 768px) {
        button, input, select, a {
            min-height: 44px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants  
DATA_DIR = Path(__file__).parent.parent.parent / "data/04_output/nba/arbs"


# Helper functions
@st.cache_data(ttl=60)
def load_all_arbs():
    """
    Load all arbitrage opportunities from all files.
    
    DEDUPLICATION STRATEGY:
    - Multiple files may exist per day (Lambda runs every 15 min)
    - Same player/market/line may appear in multiple files with different odds
    - We keep the BEST opportunity (highest expected_profit_pct) for each:
      - (file_date, player, market, line) combination
    - This captures the best historical opportunity even as lines move
    
    Returns:
        DataFrame with deduped arbs, keeping best expected_profit_pct per player/market/line/day
    """
    if not DATA_DIR.exists():
        return None
    
    arb_files = sorted(DATA_DIR.glob("arb_output_*.csv"))
    
    if not arb_files:
        return None
    
    all_dfs = []
    
    for arb_file in arb_files:
        try:
            df = pd.read_csv(arb_file)
            
            # Skip empty files
            if len(df) == 0:
                continue
            
            parts = arb_file.stem.split('_')
            date_str = parts[-2]
            time_str = parts[-1]
            
            file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            file_datetime_utc = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
            # Convert UTC to ET for display
            file_datetime_utc = file_datetime_utc.replace(tzinfo=ZoneInfo('UTC'))
            file_datetime_et = file_datetime_utc.astimezone(ZoneInfo('America/New_York'))
            
            df['file_date'] = file_date
            df['file_datetime'] = file_datetime_et
            df['source_file'] = arb_file.name
            
            all_dfs.append(df)
        except Exception as e:
            continue
    
    if not all_dfs:
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplicate: for each (file_date, player, market, line), keep the row with best expected_profit_pct
    if len(combined_df) > 0 and 'expected_profit_pct' in combined_df.columns:
        # Sort by expected_profit_pct descending, then take first per group
        combined_df = combined_df.sort_values('expected_profit_pct', ascending=False)
        combined_df = combined_df.drop_duplicates(
            subset=['file_date', 'player', 'market', 'line'],
            keep='first'
        )
        # Re-sort by file_date (desc) then expected_profit_pct (desc)
        combined_df = combined_df.sort_values(
            ['file_date', 'expected_profit_pct'], 
            ascending=[False, False]
        )
    
    return combined_df


@st.cache_data(ttl=60)
def get_arb_history():
    """
    Get historical summary by DATE (not by file).
    
    Multiple files may exist per day (Lambda runs every 15 min).
    This function groups by date and shows deduped metrics.
    """
    if not DATA_DIR.exists():
        return []
    
    arb_files = sorted(DATA_DIR.glob("arb_output_*.csv"), reverse=True)
    
    if not arb_files:
        return []
    
    # Group files by date
    files_by_date = {}
    for file in arb_files:
        try:
            parts = file.stem.split('_')
            date_str = parts[-2]
            file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            
            if file_date not in files_by_date:
                files_by_date[file_date] = []
            files_by_date[file_date].append(file)
        except:
            continue
    
    history = []
    for file_date, files in sorted(files_by_date.items(), reverse=True):
        try:
            # Load all files for this date
            day_dfs = []
            for file in files:
                try:
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        day_dfs.append(df)
                except:
                    continue
            
            if not day_dfs:
                continue
            
            combined = pd.concat(day_dfs, ignore_index=True)
            
            # Dedupe by player/market/line, keep best expected_profit_pct
            if 'expected_profit_pct' in combined.columns:
                combined = combined.sort_values('expected_profit_pct', ascending=False)
                deduped = combined.drop_duplicates(
                    subset=['player', 'market', 'line'],
                    keep='first'
                )
            else:
                deduped = combined.drop_duplicates(
                    subset=['player', 'market', 'line'],
                    keep='first'
                )
            
            # Calculate metrics on deduped data
            arbs_df = deduped[deduped['is_arb'] == True] if 'is_arb' in deduped.columns else pd.DataFrame()
            
            total_wagered = 0
            total_profit = 0
            avg_profit_pct = 0
            max_profit_pct = 0
            
            if len(arbs_df) > 0:
                if 'over_stake' in arbs_df.columns and 'under_stake' in arbs_df.columns:
                    total_wagered = (arbs_df['over_stake'].sum() + arbs_df['under_stake'].sum())
                if 'guaranteed_profit' in arbs_df.columns:
                    total_profit = arbs_df['guaranteed_profit'].sum()
                if 'expected_profit_pct' in arbs_df.columns:
                    avg_profit_pct = arbs_df['expected_profit_pct'].mean()
                    max_profit_pct = arbs_df['expected_profit_pct'].max()
            
            num_games = deduped['game'].nunique() if 'game' in deduped.columns else 0
            
            history.append({
                'date': file_date,
                'num_games': num_games,
                'num_snapshots': len(files),  # How many 15-min snapshots
                'prop_markets': len(deduped),  # Unique player/market/line combos
                'arbs_found': len(arbs_df),
                'avg_profit': avg_profit_pct,
                'max_profit': max_profit_pct,
                'total_wagered': total_wagered,
                'total_profit': total_profit
            })
        except:
            continue
    
    return history


# Main app
def main():
    # Header
    st.title("🏀 TQS NBA Props Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("📊 Loading betting data..."):
        df = load_all_arbs()
    
    if df is None:
        st.warning("⚠️ No data files found yet.")
        st.info("Data will be updated automatically at 12:00 PM ET daily.")
        return
    
    if len(df) == 0:
        st.info("ℹ️ No NBA games scheduled for the selected date(s).")
        st.markdown("---")
        
        st.subheader("📈 Overall Summary (All Time)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🎯 Total Prop Markets", "0")
        with col2:
            st.metric("✅ Total Arb Opportunities", "0")
        with col3:
            st.metric("💰 Total Wagered", "$0.00")
        with col4:
            st.metric("💵 Total Profit", "$0.00")
        with col5:
            st.metric("📊 IRR (Annualized)", "N/A")
        
        return
    
    # Add team column
    with st.spinner("🏀 Loading player teams from cache..."):
        df = add_team_column_simple(df, player_col='player')
        
        missing_rows = df['team'].isna().sum()
        if missing_rows > 0:
            missing_players_list = sorted(df[df['team'].isna()]['player'].unique())
            missing_count = len(missing_players_list)
            
            if missing_count <= 5:
                players_str = ", ".join(missing_players_list)
                st.info(f"ℹ️ {missing_count} player(s) not in cache ({missing_rows} rows): **{players_str}**. Run `python scripts/build_full_roster_cache.py` to update.")
            else:
                players_str = ", ".join(missing_players_list[:5])
                st.info(f"ℹ️ {missing_count} players not in cache ({missing_rows} rows): **{players_str}** ... and {missing_count - 5} more. Run `python scripts/build_full_roster_cache.py` to update.")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Filters")
        
        show_only_arbs = st.checkbox("Show Profitable Arb Opportunities Only", value=True, 
                                      help="Show only rows where is_arb=True")
        
        # Date filter
        if 'game_time' in df.columns and len(df) > 0:
            try:
                df['game_time_et'] = pd.to_datetime(df['game_time'], utc=True).dt.tz_convert('America/New_York')
                df['game_date_et'] = df['game_time_et'].dt.date
                
                all_dates = sorted(df['game_date_et'].unique().tolist(), reverse=True)
            except:
                all_dates = []
        else:
            all_dates = []
        
        today_et = datetime.now(ZoneInfo('America/New_York')).date()
        if today_et not in all_dates:
            all_dates.insert(0, today_et)
        
        try:
            default_index = all_dates.index(today_et) + 1
        except ValueError:
            default_index = 1 if all_dates else 0
        
        selected_date = st.selectbox(
            "Game Date (ET)",
            ['All'] + all_dates,
            index=default_index,
            help="Filter by game date in Eastern Time"
        )
        
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
        
        st.markdown("---")
        st.header("📊 IRR Calculation")
        st.info("""
        **IRR = (1 + ROI)^(365/days) - 1**
        
        Where:
        - ROI = Total Profit / Total Wagered
        - days = # of unique trading days
        
        *Example:* 3.6% ROI over 10 days → (1.036)^36.5 - 1 = **259%** annualized
        """)
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'game_time' in filtered_df.columns and len(filtered_df) > 0 and 'game_date_et' not in filtered_df.columns:
        try:
            filtered_df['game_time_et'] = pd.to_datetime(filtered_df['game_time'], utc=True).dt.tz_convert('America/New_York')
            filtered_df['game_date_et'] = filtered_df['game_time_et'].dt.date
        except:
            pass
    
    if show_only_arbs and 'is_arb' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_arb'] == True]
    
    if selected_market != 'All' and 'market' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['market'] == selected_market]
    
    if selected_player != 'All' and 'player' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['player'] == selected_player]
    
    if selected_team != 'All' and 'team' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    
    if selected_date != 'All':
        if 'game_date_et' in filtered_df.columns and len(filtered_df) > 0:
            filtered_df = filtered_df[filtered_df['game_date_et'] == selected_date]
        else:
            filtered_df = filtered_df.iloc[0:0]
    
    if min_profit > 0 and 'expected_profit_pct' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['expected_profit_pct'] >= min_profit]
    
    # Overall Metrics
    st.subheader("📈 Overall Summary (All Time)")
    
    all_arbs_df = df[df['is_arb'] == True] if 'is_arb' in df.columns else pd.DataFrame()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_prop_markets = len(df)
        st.metric("🎯 Total Prop Markets", f"{total_prop_markets:,}", 
                 help="Total prop markets analyzed across all dates")
    
    with col2:
        total_arbs = len(all_arbs_df)
        st.metric("✅ Total Arb Opportunities", f"{total_arbs:,}",
                 help="Total arbitrage opportunities found (is_arb=True)")
    
    with col3:
        total_wagered = 0
        if len(all_arbs_df) > 0 and 'over_stake' in all_arbs_df.columns and 'under_stake' in all_arbs_df.columns:
            total_wagered = (all_arbs_df['over_stake'].sum() + all_arbs_df['under_stake'].sum())
        
        st.metric("💰 Total Wagered", f"${total_wagered:,.0f}",
                 help="Total amount wagered across all arbs (assuming $100 stake)")
    
    with col4:
        total_profit = 0
        if len(all_arbs_df) > 0 and 'guaranteed_profit' in all_arbs_df.columns:
            total_profit = all_arbs_df['guaranteed_profit'].sum()
        
        st.metric("💵 Total Profit", f"${total_profit:,.0f}",
                 help="Total guaranteed profit from all arbs")
    
    with col5:
        # Calculate IRR using proper multi-day annualization
        # Formula: (1 + total_roi)^(365/days) - 1
        irr_str = "N/A"
        if total_wagered > 0 and total_profit > 0:
            # Count unique days with arb data
            num_days = 1
            if 'file_date' in all_arbs_df.columns:
                num_days = all_arbs_df['file_date'].nunique()
            elif 'game_date_et' in all_arbs_df.columns:
                num_days = all_arbs_df['game_date_et'].nunique()
            num_days = max(1, num_days)  # Avoid division by zero
            
            total_roi = total_profit / total_wagered
            annualized_irr = ((1 + total_roi) ** (365 / num_days) - 1) * 100
            
            if annualized_irr >= 1000:
                irr_str = f"{annualized_irr/1000:,.1f}K%"
            else:
                irr_str = f"{annualized_irr:,.0f}%"
        
        st.metric("📊 IRR (Annualized)", irr_str,
                 help="Internal Rate of Return annualized based on actual days of data. Formula: (1 + total_roi)^(365/days) - 1")
    
    st.markdown("---")
    
    # Daily Metrics
    if selected_date == 'All':
        st.subheader("📊 Daily Summary (All Dates)")
        daily_df = df
    else:
        st.subheader(f"📊 Daily Summary ({selected_date})")
        if 'game_date_et' in df.columns and len(df) > 0:
            daily_df = df[df['game_date_et'] == selected_date]
        else:
            daily_df = pd.DataFrame()
    
    daily_arbs_df = daily_df[daily_df['is_arb'] == True] if 'is_arb' in daily_df.columns and len(daily_df) > 0 else pd.DataFrame()
    
    unique_games = daily_df['game'].nunique() if 'game' in daily_df.columns else 0
    daily_total_props = len(daily_df)
    daily_arbs_count = len(daily_arbs_df)
    
    if len(daily_arbs_df) > 0 and 'expected_profit_pct' in daily_arbs_df.columns:
        daily_avg_edge = daily_arbs_df['expected_profit_pct'].mean()
        daily_avg_edge_str = f"{daily_avg_edge:.2f}%"
        daily_max_profit = daily_arbs_df['expected_profit_pct'].max()
        daily_max_profit_str = f"{daily_max_profit:.2f}%"
    else:
        daily_avg_edge_str = "N/A"
        daily_max_profit_str = "N/A"
    
    daily_wagered = 0
    if len(daily_arbs_df) > 0 and 'over_stake' in daily_arbs_df.columns and 'under_stake' in daily_arbs_df.columns:
        daily_wagered = (daily_arbs_df['over_stake'].sum() + daily_arbs_df['under_stake'].sum())
    
    daily_profit = 0
    if len(daily_arbs_df) > 0 and 'guaranteed_profit' in daily_arbs_df.columns:
        daily_profit = daily_arbs_df['guaranteed_profit'].sum()
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("🏀 Games", unique_games,
                 help="Number of NBA games for selected date(s)")
    with col2:
        st.metric("🎯 Prop Markets", daily_total_props,
                 help=f"Total prop markets for selected date(s)")
    with col3:
        st.metric("✅ Arb Opportunities", daily_arbs_count, 
                 help=f"Profitable arbitrage opportunities (is_arb=True)")
    with col4:
        st.metric("💰 Total Wagered", f"${daily_wagered:,.2f}",
                 help="Total amount wagered on arbs (assuming $100 stake)")
    with col5:
        st.metric("💵 Total Profit", f"${daily_profit:,.2f}",
                 help="Total guaranteed profit from arbs")
    with col6:
        st.metric("📈 Avg Edge", daily_avg_edge_str,
                 help="Average edge (profit %) for arbs")
    with col7:
        st.metric("🔥 Best Arb", daily_max_profit_str,
                 help="Highest profit opportunity")
    
    st.markdown("---")
    
    # Opportunities table
    st.subheader("🎰 Current Arbitrage Opportunities")
    
    if selected_date != 'All' and len(daily_df) == 0:
        st.info(f"ℹ️ No NBA games scheduled for {selected_date}")
    elif len(filtered_df) > 0:
        mobile_view = st.checkbox("📱 Mobile View (fewer columns)", value=False, help="Show simplified view with key columns only")
        
        display_df = filtered_df.copy()
        
        if 'game_time_et' in display_df.columns:
            display_df['game_time'] = display_df['game_time_et'].dt.strftime('%I:%M %p ET')
        
        # Format file_datetime to ET string
        if 'file_datetime' in display_df.columns:
            display_df['file_datetime'] = pd.to_datetime(display_df['file_datetime']).dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
        
        cols_to_drop = ['game_time_et', 'game_date_et', 'file_date', 'source_file']
        for col in cols_to_drop:
            if col in display_df.columns:
                display_df = display_df.drop(col, axis=1)
        
        if mobile_view:
            display_df = display_df.sort_values('expected_profit_pct', ascending=False)
            
            essential_cols = [
                'player', 'recommendation', 'market', 'line',
                'expected_profit_pct', 
                'best_over_odds', 'best_under_odds',
            ]
            mobile_cols = [col for col in essential_cols if col in display_df.columns]
            mobile_df = display_df[mobile_cols].copy()
            
            mobile_df = mobile_df.rename(columns={
                'expected_profit_pct': 'Profit %',
                'best_over_odds': 'Over',
                'best_under_odds': 'Under',
                'player': 'Player',
                'recommendation': 'Rec',
                'market': 'Market',
                'line': 'Line'
            })
            
            def color_profit_gradient(val):
                if pd.isna(val):
                    return ''
                clamped = max(-10, min(10, val))
                if clamped >= 0:
                    intensity = int((clamped / 10) * 255)
                    return f'background-color: rgb({255 - intensity}, 255, {255 - intensity})'
                else:
                    intensity = int((abs(clamped) / 10) * 255)
                    return f'background-color: rgb(255, {255 - intensity}, {255 - intensity})'
            
            styled_mobile_df = mobile_df.style.applymap(
                color_profit_gradient,
                subset=['Profit %'] if 'Profit %' in mobile_df.columns else []
            )
            
            st.dataframe(
                styled_mobile_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Line": st.column_config.NumberColumn("Line", format="%.1f"),
                    "Profit %": st.column_config.NumberColumn("Profit %", format="%.2f%%"),
                    "Over": st.column_config.NumberColumn("Over", format="%d"),
                    "Under": st.column_config.NumberColumn("Under", format="%d"),
                }
            )
            
            with st.expander("💡 How to bet these arbs"):
                st.markdown("""
                **For each opportunity:**
                1. Bet the **Over** at the listed odds on the best bookmaker
                2. Bet the **Under** at the listed odds on the best bookmaker
                3. Profit is guaranteed regardless of outcome!
                
                **Stake sizes** (for $100 total):
                - Calculate based on implied probabilities
                - Use an [arb calculator](https://www.arbitrage-calculator.com/) for exact amounts
                """)
        else:
            # Desktop view
            cols = display_df.columns.tolist()
            if 'team' in cols and 'player' in cols:
                cols.remove('team')
                player_idx = cols.index('player')
                cols.insert(player_idx + 1, 'team')
                display_df = display_df[cols]
            
            implied_cols = ['best_over_implied', 'best_under_implied']
            for col in implied_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] * 100
            
            if 'total_prob' in display_df.columns:
                display_df['total_prob'] = display_df['total_prob'] * 100
            
            numeric_cols = ['expected_profit_pct', 'guaranteed_profit', 'total_wager', 'over_stake', 'under_stake']
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            if sort_column in display_df.columns:
                display_df = display_df.sort_values(sort_column, ascending=sort_ascending)
        
        def color_profit_gradient(val):
            if pd.isna(val):
                return ''
            clamped = max(-10, min(10, val))
            if clamped >= 0:
                intensity = int((clamped / 10) * 255)
                return f'background-color: rgb({255 - intensity}, 255, {255 - intensity})'
            else:
                intensity = int((abs(clamped) / 10) * 255)
                return f'background-color: rgb(255, {255 - intensity}, {255 - intensity})'
        
        if not mobile_view:
            styled_df = display_df.style.applymap(
                color_profit_gradient,
                subset=['expected_profit_pct'] if 'expected_profit_pct' in display_df.columns else []
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "player": st.column_config.TextColumn("Player"),
                    "team": st.column_config.TextColumn("Team"),
                    "market": st.column_config.TextColumn("Market"),
                    "line": st.column_config.NumberColumn("Line", format="%.1f"),
                    "best_over_odds": st.column_config.NumberColumn("Over Odds", format="%+d"),
                    "best_over_book": st.column_config.TextColumn("Over Book"),
                    "best_over_implied": st.column_config.NumberColumn("Over Implied %", format="%.2f%%"),
                    "best_under_odds": st.column_config.NumberColumn("Under Odds", format="%+d"),
                    "best_under_book": st.column_config.TextColumn("Under Book"),
                    "best_under_implied": st.column_config.NumberColumn("Under Implied %", format="%.2f%%"),
                    "total_prob": st.column_config.NumberColumn("Combined Probability", format="%.2f%%"),
                    "expected_profit_pct": st.column_config.NumberColumn("Profit %", format="%.2f%%"),
                    "is_arb": st.column_config.CheckboxColumn("Is Arb?"),
                    "over_stake": st.column_config.NumberColumn("Over Stake", format="$%.2f"),
                    "under_stake": st.column_config.NumberColumn("Under Stake", format="$%.2f"),
                    "over_return": st.column_config.NumberColumn("Over Return", format="$%.2f"),
                    "under_return": st.column_config.NumberColumn("Under Return", format="$%.2f"),
                    "guaranteed_profit": st.column_config.NumberColumn("Guaranteed Profit", format="$%.2f"),
                    "total_wager": st.column_config.NumberColumn("Total Wager", format="$%.2f"),
                    "recommendation": st.column_config.TextColumn("Recommendation"),
                    "game": st.column_config.TextColumn("Game"),
                    "game_time": st.column_config.TextColumn("Game Time (ET)"),
                    "num_bookmakers": st.column_config.NumberColumn("# Bookmakers", format="%d")
                }
            )
        
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
        
        st.markdown("**Daily Summary (deduped by player/market/line, best profit kept):**")
        
        column_order = ['date', 'num_games', 'num_snapshots', 'prop_markets', 'arbs_found', 'avg_profit', 'max_profit', 'total_wagered', 'total_profit']
        display_history = history_df[column_order]
        
        def color_profit_gradient_history(val):
            if pd.isna(val):
                return ''
            clamped = max(-10, min(10, val))
            if clamped >= 0:
                intensity = int((clamped / 10) * 255)
                return f'background-color: rgb({255 - intensity}, 255, {255 - intensity})'
            else:
                intensity = int((abs(clamped) / 10) * 255)
                return f'background-color: rgb(255, {255 - intensity}, {255 - intensity})'
        
        profit_cols = []
        if 'avg_profit' in display_history.columns:
            profit_cols.append('avg_profit')
        if 'max_profit' in display_history.columns:
            profit_cols.append('max_profit')
        
        styled_history = display_history.style.applymap(
            color_profit_gradient_history,
            subset=profit_cols if profit_cols else []
        )
        
        st.dataframe(
            styled_history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.TextColumn("Date"),
                "num_games": st.column_config.NumberColumn("# Games", format="%d"),
                "num_snapshots": st.column_config.NumberColumn("Snapshots", format="%d", help="Number of 15-min data snapshots"),
                "prop_markets": st.column_config.NumberColumn("Unique Props", format="%d", help="Unique player/market/line combos"),
                "arbs_found": st.column_config.NumberColumn("Arbs Found", format="%d"),
                "avg_profit": st.column_config.NumberColumn("Avg Profit %", format="%.2f%%"),
                "max_profit": st.column_config.NumberColumn("Best Arb", format="%.2f%%"),
                "total_wagered": st.column_config.NumberColumn("Total Wagered", format="$%.2f"),
                "total_profit": st.column_config.NumberColumn("Total Profit", format="$%.2f")
            }
        )
    else:
        st.info("No historical data available yet.")


if __name__ == "__main__":
    main()

