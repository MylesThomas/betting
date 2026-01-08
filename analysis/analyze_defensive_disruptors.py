"""
Find the most disruptive defensive players in the NBA (2025-26 season).

Purpose:
Identify players who make their team significantly better defensively when they're on the court.
This is calculated by comparing:
- Player's on-court DEF_RATING (defensive rating when player is on the floor)
- Team's overall DEF_RATING (team's season average)

A positive "Defensive Impact" means the team is BETTER defensively when that player plays
(lower defensive rating = better defense).

Context:
- DEF_RATING = points allowed per 100 possessions
- Lower is better (you want to allow fewer points)
- If Player DEF_RATING (106.6) < Team DEF_RATING (112.0), impact = +5.4
- This means the team allows 5.4 fewer points per 100 possessions when this player plays

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/analyze_defensive_disruptors.py
    
Output:
- DataFrame with player defensive metrics joined with team averages
- Top defensive disruptors (players who improve team defense most)
- Bottom players (players who hurt team defense)
- CSV saved to: data/04_output/nba/defensive_disruptors_2025_26.csv
"""

import pandas as pd
import ssl
import urllib3
import requests
from pathlib import Path

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Import after SSL fixes
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

SEASON = "2025-26"
MIN_MINUTES = 100  # Minimum total minutes played to be included


def fetch_player_defensive_stats():
    """
    Fetch defensive stats for all players.
    
    Returns:
        DataFrame with player defensive metrics including on-court DEF_RATING
    """
    print("📡 Fetching player defensive stats...")
    print(f"   Season: {SEASON}")
    print(f"   Endpoint: LeagueDashPlayerStats(measure_type='Defense')\n")
    
    player_defense = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Defense",
        per_mode_detailed="Totals"  # Get totals to calculate MIN
    )
    
    df = player_defense.get_data_frames()[0]
    
    print(f"✅ Fetched {len(df)} players\n")
    
    # Select key columns
    cols_to_keep = [
        'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
        'GP', 'MIN', 'DEF_RATING',
        'STL', 'BLK', 'DREB', 'DEF_WS',
        'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT'
    ]
    
    # Only keep columns that exist
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    df = df[cols_to_keep].copy()
    
    # Filter for players with minimum minutes
    df = df[df['MIN'] >= MIN_MINUTES].copy()
    
    print(f"📊 Filtered to {len(df)} players with {MIN_MINUTES}+ minutes\n")
    
    return df


def fetch_team_defensive_stats():
    """
    Fetch defensive stats for all teams (season averages).
    
    Returns:
        DataFrame with team defensive metrics including overall DEF_RATING
    """
    print("📡 Fetching team defensive stats...")
    print(f"   Season: {SEASON}")
    print(f"   Endpoint: LeagueDashTeamStats(measure_type='Defense')\n")
    
    team_defense = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Defense",
        per_mode_detailed="PerGame"
    )
    
    df = team_defense.get_data_frames()[0]
    
    print(f"✅ Fetched {len(df)} teams\n")
    
    # Select key columns and rename for clarity
    # Check which columns are available
    cols_to_keep = ['TEAM_ID', 'DEF_RATING']
    if 'TEAM_NAME' in df.columns:
        cols_to_keep.append('TEAM_NAME')
    if 'TEAM_ABBREVIATION' in df.columns:
        cols_to_keep.append('TEAM_ABBREVIATION')
    
    df = df[cols_to_keep].copy()
    df = df.rename(columns={'DEF_RATING': 'TEAM_DEF_RATING'})
    
    return df


def calculate_defensive_impact(player_df, team_df):
    """
    Join player and team data, calculate defensive impact.
    
    Args:
        player_df: Player defensive stats
        team_df: Team defensive stats
        
    Returns:
        DataFrame with defensive impact calculated
    """
    print("🔗 Joining player and team data...")
    
    # Merge on team
    df = player_df.merge(
        team_df[['TEAM_ID', 'TEAM_DEF_RATING']], 
        on='TEAM_ID',
        how='left'
    )
    
    # Calculate defensive impact
    # Positive impact = team is better defensively when player is on court
    # (lower DEF_RATING = better, so team_rating - player_rating)
    df['DEF_IMPACT'] = df['TEAM_DEF_RATING'] - df['DEF_RATING']
    
    # Calculate per-game stats
    df['MIN_PG'] = df['MIN'] / df['GP']
    df['STL_PG'] = df['STL'] / df['GP']
    df['BLK_PG'] = df['BLK'] / df['GP']
    df['DREB_PG'] = df['DREB'] / df['GP']
    
    # Sort by defensive impact (highest first)
    df = df.sort_values('DEF_IMPACT', ascending=False)
    
    print(f"✅ Calculated defensive impact for {len(df)} players\n")
    
    return df


def display_top_disruptors(df, n=20):
    """
    Display the top defensive disruptors.
    
    Args:
        df: DataFrame with defensive impact
        n: Number of players to show
    """
    print("="*100)
    print(f"TOP {n} DEFENSIVE DISRUPTORS (2025-26 Season)")
    print("="*100)
    print("Players who make their team significantly BETTER defensively when on the court\n")
    
    top_n = df.head(n)
    
    # Create display dataframe
    display_df = top_n[[
        'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN_PG',
        'DEF_RATING', 'TEAM_DEF_RATING', 'DEF_IMPACT',
        'STL_PG', 'BLK_PG', 'DEF_WS'
    ]].copy()
    
    # Format columns
    display_df['MIN_PG'] = display_df['MIN_PG'].round(1)
    display_df['DEF_RATING'] = display_df['DEF_RATING'].round(1)
    display_df['TEAM_DEF_RATING'] = display_df['TEAM_DEF_RATING'].round(1)
    display_df['DEF_IMPACT'] = display_df['DEF_IMPACT'].round(1)
    display_df['STL_PG'] = display_df['STL_PG'].round(1)
    display_df['BLK_PG'] = display_df['BLK_PG'].round(1)
    display_df['DEF_WS'] = display_df['DEF_WS'].round(1)
    
    print(display_df.to_string(index=False))
    print()


def display_worst_defenders(df, n=20):
    """
    Display the worst defensive players (hurt their team).
    
    Args:
        df: DataFrame with defensive impact
        n: Number of players to show
    """
    print("="*100)
    print(f"BOTTOM {n} DEFENDERS (2025-26 Season)")
    print("="*100)
    print("Players who make their team WORSE defensively when on the court\n")
    
    bottom_n = df.tail(n).sort_values('DEF_IMPACT', ascending=True)
    
    # Create display dataframe
    display_df = bottom_n[[
        'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN_PG',
        'DEF_RATING', 'TEAM_DEF_RATING', 'DEF_IMPACT',
        'STL_PG', 'BLK_PG', 'DEF_WS'
    ]].copy()
    
    # Format columns
    display_df['MIN_PG'] = display_df['MIN_PG'].round(1)
    display_df['DEF_RATING'] = display_df['DEF_RATING'].round(1)
    display_df['TEAM_DEF_RATING'] = display_df['TEAM_DEF_RATING'].round(1)
    display_df['DEF_IMPACT'] = display_df['DEF_IMPACT'].round(1)
    display_df['STL_PG'] = display_df['STL_PG'].round(1)
    display_df['BLK_PG'] = display_df['BLK_PG'].round(1)
    display_df['DEF_WS'] = display_df['DEF_WS'].round(1)
    
    print(display_df.to_string(index=False))
    print()


def display_summary_stats(df):
    """Display summary statistics"""
    print("="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    print(f"\n📊 Defensive Impact Distribution:")
    print(f"   Mean: {df['DEF_IMPACT'].mean():.2f}")
    print(f"   Median: {df['DEF_IMPACT'].median():.2f}")
    print(f"   Std Dev: {df['DEF_IMPACT'].std():.2f}")
    print(f"   Min: {df['DEF_IMPACT'].min():.2f} ({df.loc[df['DEF_IMPACT'].idxmin(), 'PLAYER_NAME']})")
    print(f"   Max: {df['DEF_IMPACT'].max():.2f} ({df.loc[df['DEF_IMPACT'].idxmax(), 'PLAYER_NAME']})")
    
    # Count positive vs negative impact
    positive_impact = (df['DEF_IMPACT'] > 0).sum()
    negative_impact = (df['DEF_IMPACT'] < 0).sum()
    
    print(f"\n🎯 Impact Breakdown:")
    print(f"   Positive Impact (help team): {positive_impact} players ({positive_impact/len(df)*100:.1f}%)")
    print(f"   Negative Impact (hurt team): {negative_impact} players ({negative_impact/len(df)*100:.1f}%)")
    
    # Top teams by having elite defenders
    print(f"\n🏀 Teams with Most Elite Defenders (Top 50):")
    top_50 = df.head(50)
    team_counts = top_50['TEAM_ABBREVIATION'].value_counts().head(10)
    for team, count in team_counts.items():
        print(f"   {team}: {count} players")
    
    print()


def save_results(df, output_path):
    """Save results to CSV"""
    print(f"💾 Saving results to: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df)} players to CSV\n")


def main():
    """Main analysis function"""
    print("="*100)
    print("NBA DEFENSIVE DISRUPTORS ANALYSIS (2025-26 Season)")
    print("="*100)
    print(f"\nFinding players who make their teams better defensively when on the court")
    print(f"Minimum {MIN_MINUTES} minutes played\n")
    
    # Fetch data
    player_df = fetch_player_defensive_stats()
    team_df = fetch_team_defensive_stats()
    
    # Calculate impact
    results_df = calculate_defensive_impact(player_df, team_df)
    
    # Display results
    display_top_disruptors(results_df, n=25)
    display_worst_defenders(results_df, n=25)
    display_summary_stats(results_df)
    
    # Save results
    output_path = Path('data/04_output/nba/defensive_disruptors_2025_26.csv')
    save_results(results_df, output_path)
    
    print("="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\n💡 Key Insights:")
    print("   - DEF_IMPACT > 0: Player makes team better defensively (positive)")
    print("   - DEF_IMPACT < 0: Player makes team worse defensively (negative)")
    print("   - Higher DEF_IMPACT = More disruptive defender")
    print("\n📈 Use Cases:")
    print("   - Identify elite defenders for betting on team defense props")
    print("   - Analyze matchups when key defenders are out")
    print("   - Find undervalued defensive players")
    print("   - Predict team defensive performance based on lineup")
    print()


if __name__ == "__main__":
    main()

