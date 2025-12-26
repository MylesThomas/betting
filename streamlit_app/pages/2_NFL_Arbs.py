"""
TQS NFL Props Dashboard

NFL Player Props Arbitrage Dashboard - find and track arb opportunities.

DATA SOURCE: S3 (migrated from Git)
    - Lambda saves arb files to: s3://betting-nfl-arbs/nfl/arbs/YYYY-MM-DD/
    - Dashboard reads directly from S3
    - No more git clone/pull operations

AWS CREDENTIALS SETUP (Streamlit Cloud):
    1. Go to: https://share.streamlit.io/
    2. Click on your app → Settings (⚙️) → Secrets
    3. Add the following in TOML format:
       
       ```toml
       AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
       AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
       AWS_DEFAULT_REGION = "us-east-2"
       S3_BUCKET_NAME_NFL = "betting-nfl-arbs"
       ```
    
    4. Click "Save"
    5. App will auto-restart with new credentials
    
    HOW TO GET AWS CREDENTIALS:
    - Go to: https://console.aws.amazon.com/iam/
    - Click "Users" → Find your user
    - Click "Security credentials" tab
    - Scroll to "Access keys" → "Create access key"
    - Choose "Application running outside AWS"
    - Copy both keys (secret key only shown once!)
    - Paste into Streamlit Cloud Secrets (TOML format above)
    
    Note: These are stored securely in Streamlit Cloud and accessed via os.getenv()

LOCAL TESTING:
    If running locally, credentials are auto-loaded from ~/.aws/credentials
    (set up via `aws configure` command)

DATA HANDLING:
    - Lambda runs every 5 minutes during game days, saving snapshots to S3
    - Each file contains arbs found at that moment (lines change frequently)
    - Dashboard loads ALL files for a date and DEDUPES by (player, market, line)
    - For duplicate combos, keeps the row with HIGHEST expected_profit_pct
    - This captures the BEST opportunity that appeared during the day

DEDUPLICATION:
    - Same player/market/line may appear in multiple 5-min snapshots
    - Odds fluctuate, so arb may be 3% at 1pm but 5% at 2pm
    - We keep the 5% version (best opportunity)
    - Metrics (total arbs, profit, etc.) calculated on deduped data

LIVE PROP BEHAVIOR (observed):
    - NFL lines move frequently, especially during live games
    - Arb opportunities appear and disappear within minutes
    - 5-min snapshots capture most opportunities during game days
    - Best arbs often last < 30 minutes before books adjust
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import boto3
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.nfl_team_utils import NFL_TEAM_MAPPING, full_name_to_abbr
from src.config import get_current_nfl_week, get_nfl_week_range, get_nfl_week_for_date

# Add utils to path for shared formatters
sys.path.append(str(Path(__file__).parent.parent))
from utils.formatters import format_large_number

# S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME_NFL', 'betting-nfl-arbs')
s3_client = boto3.client('s3')

# Performance tuning for parallel S3 downloads
# Higher = faster loading but more memory/network usage
# 200 workers can handle hundreds of files in seconds
MAX_WORKERS = 200


# Page config
st.set_page_config(
    page_title="NFL Arbitrage Dashboard",
    page_icon="🏈",
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

# Constants - no longer needed, using S3
# Dashboard now reads directly from S3 bucket


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Note: format_large_number() now imported from utils.formatters


# Market display names
MARKET_DISPLAY_NAMES = {
    'player_pass_yds': 'Passing Yards',
    'player_pass_tds': 'Passing TDs',
    'player_pass_completions': 'Completions',
    'player_pass_attempts': 'Pass Attempts',
    'player_pass_interceptions': 'Interceptions',
    'player_rush_yds': 'Rushing Yards',
    'player_rush_attempts': 'Rush Attempts',
    'player_rush_longest': 'Longest Rush',
    'player_receptions': 'Receptions',
    'player_reception_yds': 'Receiving Yards',
    'player_reception_longest': 'Longest Reception',
    'player_anytime_td': 'Anytime TD',
    'player_1st_td': 'First TD',
    'player_last_td': 'Last TD',
    'player_pass_rush_reception_tds': 'Total TDs',
    'player_kicking_points': 'Kicking Points',
    'player_field_goals': 'Field Goals',
    'player_tackles_assists': 'Tackles + Assists',
    'player_sacks': 'Sacks',
}


# Helper functions
def load_single_s3_file(s3_key: str) -> pd.DataFrame:
    """
    Load a single S3 file (used for parallel processing).
    
    Args:
        s3_key: S3 key to load
    
    Returns:
        DataFrame with file metadata added, or None if failed
    """
    try:
        # Download file from S3
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        csv_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Skip empty files
        if len(df) == 0:
            return None
        
        # Extract date from S3 key: nfl/arbs/2025-12-24/arb_output_20251224_180000.csv
        parts = s3_key.split('/')
        if len(parts) >= 4:
            file_date = parts[2]  # YYYY-MM-DD
            filename = parts[-1]  # arb_output_20251224_180000.csv
            
            # Extract time from filename
            filename_parts = filename.replace('.csv', '').split('_')
            if len(filename_parts) >= 3:
                date_str = filename_parts[-2]  # YYYYMMDD
                time_str = filename_parts[-1]  # HHMMSS
                
                file_datetime_utc = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
                # Convert UTC to ET for display
                file_datetime_utc = file_datetime_utc.replace(tzinfo=ZoneInfo('UTC'))
                file_datetime_et = file_datetime_utc.astimezone(ZoneInfo('America/New_York'))
                
                df['file_date'] = file_date
                df['file_datetime'] = file_datetime_et
                df['source_file'] = filename
        
        return df
    except Exception as e:
        # Silently skip failed files (will be logged if needed)
        return None


@st.cache_data(ttl=60)
def load_all_arbs(max_workers: int = MAX_WORKERS):
    """
    Load all arbitrage opportunities from S3 (parallel loading for speed).
    
    PARALLEL LOADING:
    - Uses ThreadPoolExecutor with {MAX_WORKERS} workers by default
    - Loads files much faster than sequential loading
    - Each worker downloads and parses one file independently
    
    DEDUPLICATION STRATEGY:
    - Multiple files may exist per day (Lambda runs every 5 min during game days)
    - Same player/market/line may appear in multiple files with different odds
    - We keep the BEST opportunity (highest expected_profit_pct) for each:
      - (file_date, player, market, line) combination
    - This captures the best historical opportunity even as lines move
    
    Args:
        max_workers: Number of parallel download threads (default: 200)
    
    Returns:
        DataFrame with deduped arbs, keeping best expected_profit_pct per player/market/line/day
    """
    try:
        # List all files in S3 under nfl/arbs/ (with pagination for >1000 files)
        arb_files = []
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix='nfl/arbs/'
        )
        
        for page in page_iterator:
            if 'Contents' in page:
                # Filter for CSV files
                arb_files.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.csv')])
        
        if not arb_files:
            return None
        
        # Load files in parallel with ThreadPoolExecutor
        all_dfs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_key = {executor.submit(load_single_s3_file, s3_key): s3_key for s3_key in arb_files}
            
            # Collect results as they complete
            for future in as_completed(future_to_key):
                df = future.result()
                if df is not None:
                    all_dfs.append(df)
        
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
        
    except Exception as e:
        st.error(f"Error loading from S3: {e}")
        return None


@st.cache_data(ttl=60)
def get_arb_history(max_workers: int = MAX_WORKERS):
    """
    Get historical summary by DATE (not by file) from S3.
    
    Multiple files may exist per day (Lambda runs every 5 min during game days).
    This function groups by date and shows deduped metrics.
    
    OPTIMIZATION: Uses parallel loading with ThreadPoolExecutor for speed.
    
    Args:
        max_workers: Number of parallel download threads (default: 200)
    
    Returns:
        List of dictionaries with daily metrics
    """
    try:
        # List all files in S3 (with pagination for >1000 files)
        arb_files = []
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix='nfl/arbs/'
        )
        
        for page in page_iterator:
            if 'Contents' in page:
                arb_files.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.csv')])
        
        if not arb_files:
            return []
        
        # Load ALL files in parallel first (reuse load_single_s3_file helper)
        all_dfs_with_dates = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(load_single_s3_file, s3_key): s3_key for s3_key in arb_files}
            
            for future in as_completed(future_to_key):
                s3_key = future_to_key[future]
                df = future.result()
                if df is not None:
                    # Extract date from key
                    parts = s3_key.split('/')
                    if len(parts) >= 3:
                        file_date = parts[2]
                        all_dfs_with_dates.append((file_date, df))
        
        # Group loaded DataFrames by date
        dfs_by_date = {}
        for file_date, df in all_dfs_with_dates:
            if file_date not in dfs_by_date:
                dfs_by_date[file_date] = []
            dfs_by_date[file_date].append(df)
        
        # Calculate metrics for each date
        history = []
        for file_date, day_dfs in sorted(dfs_by_date.items(), reverse=True):
            try:
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
                arbs_df = deduped[deduped['is_arb'] == True] if 'is_arb' in deduped.columns else deduped
                
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
                num_snapshots = len([d for d in day_dfs])
                
                history.append({
                    'date': file_date,
                    'num_games': num_games,
                    'num_snapshots': num_snapshots,  # How many 5-min snapshots
                    'arbs_found': len(arbs_df),
                    'avg_profit': avg_profit_pct,
                    'max_profit': max_profit_pct,
                    'total_wagered': total_wagered,
                    'total_profit': total_profit
                })
            except:
                continue
        
        return history
        
    except Exception as e:
        st.error(f"Error loading history from S3: {e}")
        return []


def add_team_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team abbreviation column based on game matchup.
    
    For NFL, we parse the game column (e.g., "Kansas City Chiefs @ Buffalo Bills")
    and try to determine which team the player is on.
    """
    if df.empty or 'game' not in df.columns:
        df['team'] = None
        return df
    
    df = df.copy()
    df['team'] = None
    
    # For now, we don't have player-to-team mapping for NFL
    # This would need to be implemented similar to NBA
    # For now, just leave team as None
    
    return df


# Main app
def main():
    # Header
    st.title("🏈 TQS NFL Props Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("📊 Loading betting data..."):
        df = load_all_arbs()
    
    if df is None:
        st.warning("⚠️ No NFL arb data files found yet.")
        st.info("Data will be updated automatically during NFL game days (Thu/Sun/Mon).")
        
        # Show empty metrics
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
    
    if len(df) == 0:
        st.info("ℹ️ No NFL games scheduled for the selected date(s).")
        return
    
    # Add team column (placeholder for now)
    df = add_team_column(df)
    
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
            default_index = 0  # Show All by default for NFL
        
        selected_date = st.selectbox(
            "Game Date (ET)",
            ['All'] + all_dates,
            index=default_index,
            help="Filter by game date in Eastern Time"
        )
        
        # Market filter
        if 'market' in df.columns:
            all_markets = ['All'] + sorted(df['market'].unique().tolist())
            # Map to display names for the selectbox
            market_options = ['All'] + [MARKET_DISPLAY_NAMES.get(m, m) for m in sorted(df['market'].unique().tolist())]
            selected_market_display = st.selectbox("Market", market_options)
            
            # Map back to internal market name
            if selected_market_display == 'All':
                selected_market = 'All'
            else:
                # Find the original key
                reverse_map = {v: k for k, v in MARKET_DISPLAY_NAMES.items()}
                selected_market = reverse_map.get(selected_market_display, selected_market_display)
        else:
            selected_market = 'All'
        
        # Player filter
        if 'player' in df.columns:
            all_players = ['All'] + sorted(df['player'].unique().tolist())
            selected_player = st.selectbox("Player", all_players, 
                                          help="Search for specific player")
        else:
            selected_player = 'All'
        
        # Game filter (unique games)
        if 'game' in df.columns:
            all_games = ['All'] + sorted(df['game'].unique().tolist())
            selected_game = st.selectbox("Game", all_games,
                                        help="Filter by specific matchup")
        else:
            selected_game = 'All'
        
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
        
        # Show current week info
        current_week_num = get_current_nfl_week()
        wk_start, wk_end = get_nfl_week_range(current_week_num)
        st.success(f"**📅 NFL Week {current_week_num}**\n\n{wk_start.strftime('%a %b %d')} → {wk_end.strftime('%a %b %d')}")
        
        st.info("""
        **NFL Schedule:**
        - Thursday Night: 1 game
        - Sunday: 13-14 games
        - Monday Night: 1 game
        
        **Data Source:** S3 (betting-nfl-arbs)
        
        **Scheduled run:** Every 5 min during game days
        
        **Markets monitored:**
        - Passing: Yards, TDs, Completions
        - Rushing: Yards, Attempts
        - Receiving: Yards, Receptions
        - Touchdowns: Anytime TD
        - Kicking: Points, Field Goals
        - Defense: Tackles, Sacks
        
        **Data refreshes automatically from S3.**
        
        **AWS Setup (Streamlit Cloud):**
        Add to Secrets in Streamlit Cloud settings:
        ```
        AWS_ACCESS_KEY_ID = "your_key"
        AWS_SECRET_ACCESS_KEY = "your_secret"
        AWS_DEFAULT_REGION = "us-east-2"
        S3_BUCKET_NAME_NFL = "betting-nfl-arbs"
        ```
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
    
    if selected_game != 'All' and 'game' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['game'] == selected_game]
    
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
        st.metric("🎯 Total Prop Markets", format_large_number(total_prop_markets), 
                 help="Total prop markets analyzed across all dates")
    
    with col2:
        total_arbs = len(all_arbs_df)
        st.metric("✅ Total Arb Opportunities", format_large_number(total_arbs),
                 help="Total arbitrage opportunities found (is_arb=True)")
    
    with col3:
        total_wagered = 0
        if len(all_arbs_df) > 0 and 'over_stake' in all_arbs_df.columns and 'under_stake' in all_arbs_df.columns:
            total_wagered = (all_arbs_df['over_stake'].sum() + all_arbs_df['under_stake'].sum())
        
        st.metric("💰 Total Wagered", f"${format_large_number(total_wagered)}",
                 help="Total amount wagered across all arbs (assuming $100 stake)")
    
    with col4:
        total_profit = 0
        if len(all_arbs_df) > 0 and 'guaranteed_profit' in all_arbs_df.columns:
            total_profit = all_arbs_df['guaranteed_profit'].sum()
        
        st.metric("💵 Total Profit", f"${format_large_number(total_profit)}",
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
    
    # Weekly Metrics (NFL is a weekly sport)
    current_week = get_current_nfl_week()
    week_start, week_end = get_nfl_week_range(current_week)
    
    st.subheader(f"📊 Week {current_week} Summary ({week_start.strftime('%b %d')} → {week_end.strftime('%b %d')})")
    
    # Filter data for current NFL week
    if 'game_date_et' in df.columns and len(df) > 0:
        weekly_df = df[
            (df['game_date_et'] >= week_start) & 
            (df['game_date_et'] <= week_end)
        ]
    elif 'file_date' in df.columns and len(df) > 0:
        # Fallback: use file_date if game_date_et not available
        df['file_date_parsed'] = pd.to_datetime(df['file_date']).dt.date
        weekly_df = df[
            (df['file_date_parsed'] >= week_start) & 
            (df['file_date_parsed'] <= week_end)
        ]
    else:
        weekly_df = df
    
    weekly_arbs_df = weekly_df[weekly_df['is_arb'] == True] if 'is_arb' in weekly_df.columns and len(weekly_df) > 0 else pd.DataFrame()
    
    unique_games = weekly_df['game'].nunique() if 'game' in weekly_df.columns else 0
    weekly_total_props = len(weekly_df)
    weekly_arbs_count = len(weekly_arbs_df)
    
    if len(weekly_arbs_df) > 0 and 'expected_profit_pct' in weekly_arbs_df.columns:
        weekly_avg_edge = weekly_arbs_df['expected_profit_pct'].mean()
        weekly_avg_edge_str = f"{weekly_avg_edge:.2f}%"
        weekly_max_profit = weekly_arbs_df['expected_profit_pct'].max()
        weekly_max_profit_str = f"{weekly_max_profit:.2f}%"
    else:
        weekly_avg_edge_str = "N/A"
        weekly_max_profit_str = "N/A"
    
    weekly_wagered = 0
    if len(weekly_arbs_df) > 0 and 'over_stake' in weekly_arbs_df.columns and 'under_stake' in weekly_arbs_df.columns:
        weekly_wagered = (weekly_arbs_df['over_stake'].sum() + weekly_arbs_df['under_stake'].sum())
    
    weekly_profit = 0
    if len(weekly_arbs_df) > 0 and 'guaranteed_profit' in weekly_arbs_df.columns:
        weekly_profit = weekly_arbs_df['guaranteed_profit'].sum()
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("🏈 Games", unique_games,
                 help=f"Number of NFL games for Week {current_week}")
    with col2:
        st.metric("🎯 Prop Markets", weekly_total_props,
                 help=f"Total prop markets analyzed for Week {current_week}")
    with col3:
        st.metric("✅ Arb Opportunities", weekly_arbs_count, 
                 help=f"Profitable arbitrage opportunities (is_arb=True)")
    with col4:
        st.metric("💰 Total Wagered", f"${weekly_wagered:,.2f}",
                 help="Total amount wagered on arbs (assuming $100 stake)")
    with col5:
        st.metric("💵 Total Profit", f"${weekly_profit:,.2f}",
                 help="Total guaranteed profit from arbs")
    with col6:
        st.metric("📈 Avg Edge", weekly_avg_edge_str,
                 help="Average edge (profit %) for arbs")
    with col7:
        st.metric("🔥 Best Arb", weekly_max_profit_str,
                 help="Highest profit opportunity")
    
    st.markdown("---")
    
    # Opportunities table
    st.subheader(f"🎰 Week {current_week} Arbitrage Opportunities")
    
    if selected_date != 'All' and len(filtered_df) == 0:
        st.info(f"ℹ️ No NFL games scheduled for {selected_date}")
    elif len(filtered_df) > 0:
        mobile_view = st.checkbox("📱 Mobile View (fewer columns)", value=False, help="Show simplified view with key columns only")
        
        display_df = filtered_df.copy()
        
        # Convert market to display name
        if 'market' in display_df.columns:
            display_df['market'] = display_df['market'].map(lambda x: MARKET_DISPLAY_NAMES.get(x, x))
        
        if 'game_time_et' in display_df.columns:
            display_df['game_time'] = display_df['game_time_et'].dt.strftime('%a %I:%M %p ET')
        
        cols_to_drop = ['game_time_et', 'game_date_et', 'file_date', 'source_file', 'team']
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
            file_name=f"nfl_arb_opportunities_filtered.csv",
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
        
        column_order = ['date', 'num_games', 'num_snapshots', 'arbs_found', 'avg_profit', 'max_profit', 'total_wagered', 'total_profit']
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
                "num_snapshots": st.column_config.NumberColumn("Snapshots", format="%d", help="Number of 5-min data snapshots"),
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

