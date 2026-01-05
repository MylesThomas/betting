"""
Fetch Shot Chart Data for All NBA Players - 2024-25 Season

Purpose:
- Fetch detailed shot-by-shot data for every active NBA player in 2024-25 season
- Save individual CSV files per player with all shot data
- Track progress and allow resuming if interrupted
- Handle rate limiting and errors gracefully

Output:
- One CSV per player in data/01_input/nba_api/shot_charts/2024_25/
- Progress log to track which players are complete
- Summary statistics file

Data includes for each shot:
- SHOT_DISTANCE (feet)
- LOC_X, LOC_Y (court coordinates)
- SHOT_MADE_FLAG (1=made, 0=missed)
- ACTION_TYPE (Dunk, Layup, Jump Shot, etc.)
- GAME_DATE, PERIOD, time remaining
- Opponent teams

Runtime:
- Approximately 500+ active players per season
- ~0.6 second delay per player
- Single season: ~5-10 minutes
- All seasons (2014-15 to 2025-26): ~1-2 hours

Usage:
    # Fetch current season only (2025-26)
    python scripts/fetch_all_nba_shot_charts.py --auto
    
    # Fetch all available seasons (2014-15 to 2025-26)
    python scripts/fetch_all_nba_shot_charts.py --auto --all-seasons
    
    # Fetch specific seasons
    python scripts/fetch_all_nba_shot_charts.py --auto --seasons 2023-24,2024-25,2025-26
    
    # Interactive mode (prompts for confirmation)
    python scripts/fetch_all_nba_shot_charts.py
"""

import pandas as pd
import os
import time
from datetime import datetime
import ssl
import urllib3
import requests
import json

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

from nba_api.stats.endpoints import shotchartdetail, commonallplayers
from nba_api.stats.static import players

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get absolute path to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Available seasons in NBA API (shot chart data available from 2014-15 onwards)
AVAILABLE_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"
]

DEFAULT_SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
SHOT_CHARTS_BASE_DIR = os.path.join(REPO_ROOT, "data/01_input/nba_api/shot_charts")
RATE_LIMIT_DELAY = 0.6  # seconds between API calls
ERROR_RETRY_DELAY = 2.0  # seconds to wait before retrying on error

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_season_paths(season):
    """Get file paths for a specific season"""
    season_clean = season.replace('-', '_')
    return {
        'output_dir': os.path.join(SHOT_CHARTS_BASE_DIR, season_clean),
        'progress_file': os.path.join(SHOT_CHARTS_BASE_DIR, f"{season_clean}_progress.json"),
        'summary_file': os.path.join(SHOT_CHARTS_BASE_DIR, f"{season_clean}_summary.csv")
    }


def load_progress(progress_file):
    """Load progress from previous run"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_players': [], 'failed_players': []}


def save_progress(progress, progress_file):
    """Save progress to resume later"""
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def get_all_active_players():
    """Get list of all active NBA players (current season)"""
    print("\n🏀 Fetching list of all active NBA players...")
    all_players = players.get_active_players()
    print(f"✅ Found {len(all_players)} active players")
    return all_players


def get_players_for_season(season):
    """
    Get all players who were active in a specific season
    
    For current season: uses get_active_players()
    For past seasons: uses commonallplayers endpoint with season parameter
    
    Args:
        season: NBA season (e.g., "2024-25")
    
    Returns:
        List of player dicts with 'id' and 'full_name'
    """
    # For current season, use simple active players list
    if season == DEFAULT_SEASON:
        return get_all_active_players()
    
    # For past seasons, get players who were active that season
    print(f"\n🏀 Fetching players active in {season}...")
    
    try:
        # Use NBA API to get all players for specific season
        all_players_data = commonallplayers.CommonAllPlayers(
            is_only_current_season=0,  # Include all historical players
            league_id='00',
            season=season
        )
        
        players_df = all_players_data.get_data_frames()[0]
        
        # Filter to players who were active that season (ROSTERSTATUS = 1)
        # Convert to format matching get_active_players()
        active_players = []
        for _, row in players_df.iterrows():
            if row.get('ROSTERSTATUS') == 1:  # 1 = Active that season
                active_players.append({
                    'id': row['PERSON_ID'],
                    'full_name': row['DISPLAY_FIRST_LAST']
                })
        
        print(f"✅ Found {len(active_players)} players active in {season}")
        return active_players
        
    except Exception as e:
        print(f"⚠️  Error fetching players for {season}: {e}")
        print(f"   Falling back to current active players list...")
        return get_all_active_players()


def get_player_shot_chart(player_id, player_name, season):
    """
    Get shot chart data for a specific player
    
    Args:
        player_id: NBA player ID
        player_name: Player name (for logging)
        season: NBA season (e.g., "2024-25")
    
    Returns:
        DataFrame with shot data, or None if error/no data
    """
    try:
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star=SEASON_TYPE,
            context_measure_simple='FGA'
        )
        
        shots_df = shot_chart.get_data_frames()[0]
        
        if shots_df.empty:
            return None
        
        return shots_df
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def save_player_shot_chart(shots_df, player_name, player_id, output_dir):
    """
    Save player shot chart to CSV
    
    Args:
        shots_df: DataFrame with shot data
        player_name: Player name
        player_id: Player ID
        output_dir: Directory to save to
    
    Returns:
        Filepath where data was saved
    """
    if shots_df is None or shots_df.empty:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean player name for filename
    clean_name = player_name.replace(' ', '_').replace('.', '').replace("'", '')
    filename = f"{clean_name}_{player_id}.csv"
    filepath = os.path.join(output_dir, filename)
    
    shots_df.to_csv(filepath, index=False)
    return filepath


def analyze_player_shots(shots_df):
    """
    Quick analysis of player's shots
    
    Returns:
        Dict with summary stats
    """
    if shots_df is None or shots_df.empty:
        return None
    
    total_shots = len(shots_df)
    makes = shots_df['SHOT_MADE_FLAG'].sum()
    fg_pct = (makes / total_shots * 100) if total_shots > 0 else 0
    
    # Distance stats
    avg_distance = shots_df['SHOT_DISTANCE'].mean()
    
    # Close range (0-6 feet)
    close_range = shots_df[shots_df['SHOT_DISTANCE'] <= 6]
    close_range_attempts = len(close_range)
    close_range_makes = close_range['SHOT_MADE_FLAG'].sum() if not close_range.empty else 0
    close_range_pct = (close_range_makes / close_range_attempts * 100) if close_range_attempts > 0 else 0
    
    # Three pointers
    threes = shots_df[shots_df['SHOT_TYPE'] == '3PT Field Goal']
    three_attempts = len(threes)
    three_makes = threes['SHOT_MADE_FLAG'].sum() if not threes.empty else 0
    three_pct = (three_makes / three_attempts * 100) if three_attempts > 0 else 0
    
    return {
        'total_shots': total_shots,
        'fg_pct': fg_pct,
        'avg_distance': avg_distance,
        'close_range_attempts': close_range_attempts,
        'close_range_fg_pct': close_range_pct,
        'three_attempts': three_attempts,
        'three_pct': three_pct
    }


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def fetch_all_player_shot_charts(season=DEFAULT_SEASON, resume=True):
    """
    Fetch shot chart data for all active NBA players for a specific season
    
    Args:
        season: NBA season (e.g., "2024-25")
        resume: If True, skip players already completed
    """
    print("="*80)
    print(f"FETCHING SHOT CHARTS FOR ALL NBA PLAYERS - {season}")
    print("="*80)
    
    # Get paths for this season
    paths = get_season_paths(season)
    output_dir = paths['output_dir']
    progress_file = paths['progress_file']
    summary_file = paths['summary_file']
    
    # Check if this is a past season (not current)
    is_past_season = season != DEFAULT_SEASON
    
    print(f"\nSeason: {season}")
    print(f"Season Type: {SEASON_TYPE}")
    print(f"Output Directory: {output_dir}")
    print(f"Rate Limit Delay: {RATE_LIMIT_DELAY}s per player")
    
    if is_past_season:
        print(f"📁 Past season detected - will skip players with existing files")
    else:
        print(f"🔄 Current season - fetching all players for latest data")
    
    # Load progress if resuming
    progress = load_progress(progress_file) if resume else {'completed_players': [], 'failed_players': []}
    
    if resume and progress['completed_players']:
        print(f"\n📋 Resuming from previous run:")
        print(f"   Already completed: {len(progress['completed_players'])} players")
        print(f"   Previously failed: {len(progress['failed_players'])} players")
    
    # Get players who were active in this specific season
    all_players = get_players_for_season(season)
    
    # Filter out already completed players if resuming
    if resume:
        all_players = [p for p in all_players if p['id'] not in progress['completed_players']]
        print(f"\n🎯 Players remaining: {len(all_players)}")
    
    # Summary data
    summary_data = []
    skipped_count = 0
    
    # Process each player
    print(f"\n{'='*80}")
    print("FETCHING PLAYER DATA")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    for idx, player in enumerate(all_players, 1):
        player_id = player['id']
        player_name = player['full_name']
        
        print(f"[{idx}/{len(all_players)}] {player_name} (ID: {player_id})")
        
        # For past seasons, check if file already exists and skip
        if is_past_season:
            clean_name = player_name.replace(' ', '_').replace('.', '').replace("'", '')
            filename = f"{clean_name}_{player_id}.csv"
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                print(f"      ⏭️  File exists - skipping (past season data complete)")
                skipped_count += 1
                progress['completed_players'].append(player_id)
                save_progress(progress, progress_file)
                continue
        
        # For current season, always fetch (will overwrite with latest data)
        shots_df = get_player_shot_chart(player_id, player_name, season)
        
        if shots_df is not None and not shots_df.empty:
            # Save to file
            filepath = save_player_shot_chart(shots_df, player_name, player_id, output_dir)
            
            # Analyze shots
            stats = analyze_player_shots(shots_df)
            
            print(f"      ✅ Saved {len(shots_df)} shots")
            print(f"      📊 FG%: {stats['fg_pct']:.1f}% | Close Range: {stats['close_range_fg_pct']:.1f}% | 3PT: {stats['three_pct']:.1f}%")
            
            # Add to summary
            summary_data.append({
                'season': season,
                'player_id': player_id,
                'player_name': player_name,
                'total_shots': stats['total_shots'],
                'fg_pct': round(stats['fg_pct'], 1),
                'avg_shot_distance': round(stats['avg_distance'], 1),
                'close_range_attempts': stats['close_range_attempts'],
                'close_range_fg_pct': round(stats['close_range_fg_pct'], 1),
                'three_attempts': stats['three_attempts'],
                'three_pct': round(stats['three_pct'], 1),
                'filepath': filepath,
                'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Mark as completed
            progress['completed_players'].append(player_id)
            
        else:
            print(f"      ⚠️  No shot data (likely hasn't played this season)")
            progress['failed_players'].append(player_id)
        
        # Save progress after each player
        save_progress(progress, progress_file)
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        # Save summary every 50 players
        if idx % 50 == 0:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"\n      💾 Progress saved ({idx} players processed)\n")
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ FETCH COMPLETE!")
    print("="*80)
    print(f"\nTotal players processed: {len(all_players)}")
    print(f"Players with shot data: {len(summary_data)}")
    print(f"Players with no data: {len(progress['failed_players'])}")
    if skipped_count > 0:
        print(f"Players skipped (file exists): {skipped_count}")
    print(f"Total time: {duration/60:.1f} minutes")
    
    # Save final summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('total_shots', ascending=False)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved to: {summary_file}")
        
        # Show top 10 players by shot volume
        print("\n📊 Top 10 Players by Shot Volume:")
        print(summary_df[['player_name', 'total_shots', 'fg_pct', 'close_range_fg_pct', 'three_pct']].head(10).to_string(index=False))
    
    return summary_df


def fetch_multiple_seasons(seasons, resume=True):
    """
    Fetch shot chart data for multiple seasons
    
    Args:
        seasons: List of season strings (e.g., ["2023-24", "2024-25"])
        resume: If True, resume from previous progress
    """
    print("="*80)
    print("MULTI-SEASON SHOT CHART DATA COLLECTION")
    print("="*80)
    print(f"\nSeasons to fetch: {', '.join(seasons)}")
    print(f"Total seasons: {len(seasons)}")
    print(f"Estimated time: {len(seasons) * 10} minutes (assuming ~10 min per season)")
    
    overall_start = datetime.now()
    season_summaries = []
    
    for season_idx, season in enumerate(seasons, 1):
        print(f"\n\n{'='*80}")
        print(f"SEASON {season_idx}/{len(seasons)}: {season}")
        print(f"{'='*80}\n")
        
        try:
            summary_df = fetch_all_player_shot_charts(season=season, resume=resume)
            
            if summary_df is not None and not summary_df.empty:
                season_summaries.append({
                    'season': season,
                    'players_with_data': len(summary_df),
                    'total_shots': summary_df['total_shots'].sum(),
                    'avg_fg_pct': summary_df['fg_pct'].mean()
                })
        
        except Exception as e:
            print(f"\n❌ Error fetching season {season}: {e}")
            continue
    
    # Final multi-season summary
    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()
    
    print("\n\n" + "="*80)
    print("✅ MULTI-SEASON FETCH COMPLETE!")
    print("="*80)
    
    if season_summaries:
        summary_df = pd.DataFrame(season_summaries)
        print("\n📊 Season Summary:")
        print(summary_df.to_string(index=False))
        
        print(f"\n⏱️  Total time: {overall_duration/60:.1f} minutes")
        print(f"\nAll data saved to: {SHOT_CHARTS_BASE_DIR}")
    else:
        print("\n⚠️  No data collected")
    
    return season_summaries


def analyze_summary_stats(summary_df):
    """
    Analyze the summary statistics across all players
    """
    print("\n" + "="*80)
    print("LEAGUE-WIDE SHOT ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Overall Stats:")
    print(f"   Total players with shot data: {len(summary_df)}")
    print(f"   Total shots recorded: {summary_df['total_shots'].sum():,}")
    print(f"   Average shots per player: {summary_df['total_shots'].mean():.1f}")
    print(f"   Median shots per player: {summary_df['total_shots'].median():.1f}")
    
    print(f"\n🎯 Field Goal Percentages:")
    print(f"   League average FG%: {summary_df['fg_pct'].mean():.1f}%")
    print(f"   Close range (0-6 ft) FG%: {summary_df['close_range_fg_pct'].mean():.1f}%")
    print(f"   Three-point FG%: {summary_df['three_pct'].mean():.1f}%")
    
    print(f"\n📏 Shot Distance:")
    print(f"   Average shot distance: {summary_df['avg_shot_distance'].mean():.1f} feet")
    
    # Close range specialists (high volume + high efficiency)
    close_range_df = summary_df[summary_df['close_range_attempts'] >= 100].copy()
    close_range_df = close_range_df.sort_values('close_range_fg_pct', ascending=False)
    
    print(f"\n🏀 Best Close-Range Finishers (min 100 attempts):")
    print(close_range_df[['player_name', 'close_range_attempts', 'close_range_fg_pct']].head(10).to_string(index=False))
    
    # Three point shooters
    three_point_df = summary_df[summary_df['three_attempts'] >= 100].copy()
    three_point_df = three_point_df.sort_values('three_pct', ascending=False)
    
    print(f"\n🎯 Best Three-Point Shooters (min 100 attempts):")
    print(three_point_df[['player_name', 'three_attempts', 'three_pct']].head(10).to_string(index=False))


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("NBA SHOT CHART DATA COLLECTION")
    print("="*80)
    
    # Parse command line arguments
    args = sys.argv[1:]
    auto_run = '--auto' in args
    all_seasons = '--all-seasons' in args
    
    # Parse specific seasons if provided
    seasons_to_fetch = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--seasons='):
            # Format: --seasons=2023-24,2024-25
            seasons_str = arg.split('=')[1]
            seasons_to_fetch = [s.strip() for s in seasons_str.split(',')]
        elif arg == '--seasons' and i + 1 < len(args):
            # Format: --seasons 2023-24 or --seasons 2023-24,2024-25
            seasons_str = args[i + 1]
            seasons_to_fetch = [s.strip() for s in seasons_str.split(',')]
            i += 1  # Skip next arg since we used it
        i += 1
    
    # Determine which seasons to fetch
    if all_seasons:
        seasons_to_fetch = AVAILABLE_SEASONS
        print(f"\n🌐 Fetching ALL available seasons ({len(AVAILABLE_SEASONS)} total)")
        print(f"   Seasons: {', '.join(AVAILABLE_SEASONS)}")
        print(f"   Estimated time: {len(AVAILABLE_SEASONS)} hours")
    elif seasons_to_fetch:
        # Validate provided seasons
        invalid = [s for s in seasons_to_fetch if s not in AVAILABLE_SEASONS]
        if invalid:
            print(f"\n❌ Invalid seasons: {', '.join(invalid)}")
            print(f"   Available seasons: {', '.join(AVAILABLE_SEASONS)}")
            sys.exit(1)
        print(f"\n📅 Fetching {len(seasons_to_fetch)} specific seasons: {', '.join(seasons_to_fetch)}")
    else:
        seasons_to_fetch = [DEFAULT_SEASON]
        print(f"\n📅 Fetching current season: {DEFAULT_SEASON}")
        print(f"   (Use --all-seasons to fetch all available seasons)")
    
    print("\nYou can safely interrupt (Ctrl+C) and resume later with the same command.")
    
    # Check if auto-run or prompt user
    if auto_run:
        print("\n🤖 Running in automatic mode (--auto flag detected)\n")
        run_script = True
    else:
        try:
            user_input = input(f"\nContinue? (y/n): ").strip().lower()
            run_script = user_input == 'y'
        except EOFError:
            print("\n\n⚠️  No input detected (non-interactive mode)")
            print("💡 Use '--auto' flag to run without prompting")
            print(f"\nExamples:")
            print(f"  python scripts/fetch_all_nba_shot_charts.py --auto")
            print(f"  python scripts/fetch_all_nba_shot_charts.py --auto --all-seasons")
            print(f"  python scripts/fetch_all_nba_shot_charts.py --auto --seasons 2023-24")
            print(f"  python scripts/fetch_all_nba_shot_charts.py --auto --seasons=2023-24,2024-25")
            run_script = False
    
    if run_script:
        # Fetch shot charts
        if len(seasons_to_fetch) == 1:
            # Single season
            season = seasons_to_fetch[0]
            summary_df = fetch_all_player_shot_charts(season=season, resume=True)
            
            if summary_df is not None and not summary_df.empty:
                analyze_summary_stats(summary_df)
                
                paths = get_season_paths(season)
                print("\n" + "="*80)
                print("✅ ALL DONE!")
                print("="*80)
                print(f"\nShot chart files saved to: {paths['output_dir']}")
                print(f"Summary file: {paths['summary_file']}")
                print(f"\nYou can now analyze close-range shooting (0-6 feet) for any player!")
        else:
            # Multiple seasons
            season_summaries = fetch_multiple_seasons(seasons_to_fetch, resume=True)
            
            if season_summaries:
                print("\n" + "="*80)
                print("✅ ALL DONE!")
                print("="*80)
                print(f"\nShot chart files saved to: {SHOT_CHARTS_BASE_DIR}")
                print(f"\nYou can now analyze shot distance data across multiple seasons!")
    else:
        print("\n❌ Cancelled by user")

