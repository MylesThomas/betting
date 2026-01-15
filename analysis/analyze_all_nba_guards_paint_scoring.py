"""
Analyze Paint Scoring for ALL NBA Guards (2025-26)

Purpose:
Get distribution statistics (quartiles, percentiles) for paint scoring among ALL NBA guards.
Use this to set proper gradient ranges and filter criteria.

This will answer:
- What's the median paint FG% for NBA guards?
- What are the quartiles (25th, 50th, 75th percentile)?
- Where does Pritchard rank among ALL guards (not just our sample)?
- What threshold should we use to define "elite"?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import ssl
import urllib3
import requests

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats
from nba_api.stats.static import players

# Get repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

CURRENT_SEASON = "2025-26"
PAINT_DISTANCE = 6
MIN_PAINT_ATTEMPTS = 25
SHOT_CHART_DIR = repo_root / 'data/01_input/nba_api/shot_charts/2025_26'


def get_all_guards_this_season():
    """Get all guards who have played this season"""
    print("📊 Fetching all NBA players this season...")
    
    try:
        # Get all players with stats this season
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CURRENT_SEASON,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        
        df = stats.get_data_frames()[0]
        
        # Filter to guards (PLAYER_POSITION contains 'G')
        guards = df[df['PLAYER_NAME'].notna()].copy()
        
        print(f"   ✅ Found {len(guards)} players with stats\n")
        
        return guards
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def get_player_paint_stats(player_name, player_id):
    """Get paint scoring stats for a single player"""
    
    # Try to load from existing shot chart
    shot_chart_file = SHOT_CHART_DIR / f"{player_name.replace(' ', '_')}_{player_id}.csv"
    
    if shot_chart_file.exists():
        shots_df = pd.read_csv(shot_chart_file)
    else:
        # Fetch from API
        try:
            shot_chart = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=CURRENT_SEASON,
                season_type_all_star='Regular Season',
                context_measure_simple='FGA'
            )
            shots_df = shot_chart.get_data_frames()[0]
            
            if shots_df.empty:
                return None
                
            # Save for future
            SHOT_CHART_DIR.mkdir(parents=True, exist_ok=True)
            shots_df.to_csv(shot_chart_file, index=False)
            
            time.sleep(0.6)  # Rate limit
            
        except:
            return None
    
    if shots_df.empty:
        return None
    
    # Calculate paint stats
    paint_shots = shots_df[shots_df['SHOT_DISTANCE'] <= PAINT_DISTANCE].copy()
    
    if len(paint_shots) < MIN_PAINT_ATTEMPTS:
        return None
    
    paint_fga = len(paint_shots)
    paint_fgm = paint_shots['SHOT_MADE_FLAG'].sum()
    paint_fg_pct = (paint_fgm / paint_fga * 100) if paint_fga > 0 else 0
    
    total_fga = len(shots_df)
    paint_rate = (paint_fga / total_fga * 100) if total_fga > 0 else 0
    
    games = len(paint_shots['GAME_ID'].unique())
    paint_ppg = (paint_fgm * 2) / games if games > 0 else 0
    
    return {
        'player': player_name,
        'player_id': player_id,
        'paint_fga': paint_fga,
        'paint_fgm': int(paint_fgm),
        'paint_fg_pct': round(paint_fg_pct, 1),
        'paint_rate': round(paint_rate, 1),
        'paint_ppg': round(paint_ppg, 1),
        'games': games
    }


def analyze_all_guards():
    """Analyze paint scoring for all NBA guards"""
    
    print("="*80)
    print("ANALYZING ALL NBA GUARDS - PAINT SCORING")
    print("="*80 + "\n")
    
    # Get all active players
    all_players = players.get_active_players()
    
    print(f"📊 Total active players: {len(all_players)}")
    print(f"🎯 Min paint attempts: {MIN_PAINT_ATTEMPTS}\n")
    
    results = []
    
    for i, player_dict in enumerate(all_players):
        player_name = player_dict['full_name']
        player_id = player_dict['id']
        
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(all_players)} players...")
        
        stats = get_player_paint_stats(player_name, player_id)
        if stats:
            results.append(stats)
    
    print(f"\n✅ Analyzed {len(results)} players with ≥{MIN_PAINT_ATTEMPTS} paint attempts\n")
    
    return pd.DataFrame(results)


def calculate_distribution_stats(df):
    """Calculate distribution statistics for paint scoring"""
    
    print("="*80)
    print("PAINT SCORING DISTRIBUTION (ALL QUALIFYING PLAYERS)")
    print("="*80 + "\n")
    
    # Overall stats
    print("📊 PAINT FG% DISTRIBUTION:")
    print(f"   Count: {len(df)}")
    print(f"   Mean: {df['paint_fg_pct'].mean():.1f}%")
    print(f"   Median: {df['paint_fg_pct'].median():.1f}%")
    print(f"   Std Dev: {df['paint_fg_pct'].std():.1f}%\n")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95]
    print("📈 PERCENTILES:")
    for p in percentiles:
        value = np.percentile(df['paint_fg_pct'], p)
        print(f"   {p}th: {value:.1f}%")
    
    # Quartiles
    q1 = df['paint_fg_pct'].quantile(0.25)
    q2 = df['paint_fg_pct'].quantile(0.50)
    q3 = df['paint_fg_pct'].quantile(0.75)
    
    print(f"\n📊 QUARTILES:")
    print(f"   Q1 (25th): {q1:.1f}% - Bottom 25%")
    print(f"   Q2 (50th): {q2:.1f}% - Median")
    print(f"   Q3 (75th): {q3:.1f}% - Top 25%")
    
    # Find Pritchard's rank
    pritchard = df[df['player'] == 'Payton Pritchard']
    if not pritchard.empty:
        pp_pct = pritchard['paint_fg_pct'].iloc[0]
        pp_rank = (df['paint_fg_pct'] >= pp_pct).sum()
        pp_percentile = (1 - (pp_rank / len(df))) * 100
        
        print(f"\n⭐ PAYTON PRITCHARD:")
        print(f"   Paint FG%: {pp_pct}%")
        print(f"   Rank: #{pp_rank} of {len(df)}")
        print(f"   Percentile: {pp_percentile:.1f}th (top {100-pp_percentile:.1f}%)")
    
    return {
        'mean': df['paint_fg_pct'].mean(),
        'median': q2,
        'q1': q1,
        'q3': q3,
        'p10': np.percentile(df['paint_fg_pct'], 10),
        'p90': np.percentile(df['paint_fg_pct'], 90),
        'p95': np.percentile(df['paint_fg_pct'], 95),
    }


def recommend_gradient_and_filter(stats):
    """Recommend gradient range and filtering criteria"""
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR VISUALIZATION")
    print("="*80 + "\n")
    
    print("🎨 RECOMMENDED GRADIENT RANGE:")
    print(f"   Domain: [{stats['q1']:.1f}%, {stats['p95']:.1f}%]")
    print(f"   This spans from 25th percentile to 95th percentile")
    print(f"   Red (poor) = {stats['q1']:.1f}%")
    print(f"   White (average) = {stats['median']:.1f}%")
    print(f"   Green (elite) = {stats['p95']:.1f}%\n")
    
    print("🔍 RECOMMENDED FILTER:")
    print(f"   Only show players above median ({stats['median']:.1f}%)")
    print(f"   This focuses on 'above average' to 'elite' finishers")
    print(f"   Removes players who are genuinely bad at paint scoring\n")
    
    print("🎯 ALTERNATIVE: Show top 50%")
    print(f"   Filter: paint_fg_pct >= {stats['median']:.1f}%")
    print(f"   This would show ~{int(len(df)/2)} players\n")
    
    print("🎯 ALTERNATIVE: Show top 25% (elite only)")
    print(f"   Filter: paint_fg_pct >= {stats['q3']:.1f}%")
    print(f"   This would show ~{int(len(df)/4)} players\n")


def main():
    """Main analysis"""
    
    # Analyze all guards
    global df
    df = analyze_all_guards()
    
    if df.empty:
        print("❌ No data collected")
        return
    
    # Sort by paint FG%
    df = df.sort_values('paint_fg_pct', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Save results
    output_dir = repo_root / 'data/04_output/nba'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'all_nba_guards_paint_scoring_{CURRENT_SEASON.replace("-", "_")}.csv'
    df.to_csv(output_file, index=False)
    print(f"💾 Saved: {output_file}\n")
    
    # Calculate distribution stats
    stats = calculate_distribution_stats(df)
    
    # Recommendations
    recommend_gradient_and_filter(stats)
    
    # Show top 20
    print("\n" + "="*80)
    print("TOP 20 PAINT SCORERS")
    print("="*80 + "\n")
    print(df[['rank', 'player', 'paint_fg_pct', 'paint_fga', 'paint_ppg']].head(20).to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

