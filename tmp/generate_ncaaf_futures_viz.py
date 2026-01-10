"""
Generate hardcoded NCAAF Championship futures visualization.

Context:
Since the NCAAF Championship API data is sparse (playoff teams only),
this script creates a realistic futures dataset for the final 4 playoff teams
with multiple bookmakers, then runs the full vig analysis and visualization.

Purpose:
- Create hardcoded NCAAF futures data for College Football Playoff teams
- Save in same format as API data
- Run existing analysis and visualization pipeline
- Generate beautiful gt table visualization

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 tmp/generate_ncaaf_futures_viz.py
"""

import pandas as pd
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path for odds utils
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability

# NCAAF Championship Game futures (2025-26 season)
# Championship Game: Miami Hurricanes @ Indiana Hoosiers (Jan 19, 2026)
# Real odds from The Odds API (fetched January 10, 2026)
NCAAF_FUTURES_DATA = [
    # Indiana Hoosiers
    {'bookmaker': 'fanduel', 'team': 'Indiana Hoosiers', 'odds': -310, 'record': '15-0'},
    {'bookmaker': 'lowvig', 'team': 'Indiana Hoosiers', 'odds': -305, 'record': '15-0'},
    {'bookmaker': 'betonlineag', 'team': 'Indiana Hoosiers', 'odds': -305, 'record': '15-0'},
    {'bookmaker': 'draftkings', 'team': 'Indiana Hoosiers', 'odds': -310, 'record': '15-0'},
    {'bookmaker': 'betrivers', 'team': 'Indiana Hoosiers', 'odds': -335, 'record': '15-0'},
    {'bookmaker': 'betmgm', 'team': 'Indiana Hoosiers', 'odds': -300, 'record': '15-0'},
    {'bookmaker': 'williamhill_us', 'team': 'Indiana Hoosiers', 'odds': -320, 'record': '15-0'},
    {'bookmaker': 'fanatics', 'team': 'Indiana Hoosiers', 'odds': -310, 'record': '15-0'},
    {'bookmaker': 'bovada', 'team': 'Indiana Hoosiers', 'odds': -310, 'record': '15-0'},
    {'bookmaker': 'betus', 'team': 'Indiana Hoosiers', 'odds': -300, 'record': '15-0'},

    # Miami Hurricanes
    {'bookmaker': 'fanduel', 'team': 'Miami Hurricanes', 'odds': 250, 'record': '13-2'},
    {'bookmaker': 'lowvig', 'team': 'Miami Hurricanes', 'odds': 249, 'record': '13-2'},
    {'bookmaker': 'betonlineag', 'team': 'Miami Hurricanes', 'odds': 249, 'record': '13-2'},
    {'bookmaker': 'draftkings', 'team': 'Miami Hurricanes', 'odds': 250, 'record': '13-2'},
    {'bookmaker': 'betrivers', 'team': 'Miami Hurricanes', 'odds': 260, 'record': '13-2'},
    {'bookmaker': 'betmgm', 'team': 'Miami Hurricanes', 'odds': 250, 'record': '13-2'},
    {'bookmaker': 'williamhill_us', 'team': 'Miami Hurricanes', 'odds': 250, 'record': '13-2'},
    {'bookmaker': 'fanatics', 'team': 'Miami Hurricanes', 'odds': 245, 'record': '13-2'},
    {'bookmaker': 'bovada', 'team': 'Miami Hurricanes', 'odds': 255, 'record': '13-2'},
    {'bookmaker': 'betus', 'team': 'Miami Hurricanes', 'odds': 250, 'record': '13-2'},
]


def create_ncaaf_futures_csv():
    """Create NCAAF futures CSV from hardcoded data"""
    print("=" * 80)
    print("GENERATING NCAAF CHAMPIONSHIP FUTURES (HARDCODED DATA)")
    print("=" * 80)
    print()
    
    # Convert to DataFrame
    df = pd.DataFrame(NCAAF_FUTURES_DATA)
    
    # Add sport column
    df.insert(0, 'sport', 'NCAAF')
    
    # Calculate implied probabilities
    df['implied_prob'] = df['odds'].apply(odds_to_implied_probability)
    
    # Reorder columns to match API format
    df = df[['sport', 'bookmaker', 'team', 'odds', 'implied_prob', 'record']]
    
    print(f"📊 Generated {len(df)} odds entries")
    print(f"🏈 Teams: {df['team'].nunique()}")
    print(f"📚 Bookmakers: {df['bookmaker'].nunique()}")
    print()
    
    # Show summary by team
    print("Teams and best odds:")
    print("-" * 80)
    for team in df['team'].unique():
        team_df = df[df['team'] == team]
        best_row = team_df.loc[team_df['odds'].idxmax()]  # Max odds = best for bettor
        odds_str = f"+{int(best_row['odds'])}" if best_row['odds'] > 0 else f"{int(best_row['odds'])}"
        print(f"  {team:<35} {odds_str:>7}  ({best_row['implied_prob']*100:>5.1f}% @ {best_row['bookmaker']})")
    print()
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to CSV
    output_dir = repo_root / 'data/01_input/the-odds-api/ncaaf/futures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'ncaaf_championship_futures_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    print(f"💾 Saved to: {output_file}")
    print()
    
    return output_file


def run_analysis_pipeline():
    """Run the full analysis and visualization pipeline"""
    print("=" * 80)
    print("RUNNING ANALYSIS PIPELINE")
    print("=" * 80)
    print()
    
    # Change to repo root
    os.chdir(repo_root)
    
    # Run analysis
    print("📊 Running vig analysis...")
    result = subprocess.run(
        ['python3', 'analysis/analyze_futures.py', '--sport', 'ncaaf'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print("❌ Analysis failed!")
        return False
    
    print()
    
    # Run visualization
    print("🎨 Creating visualization...")
    result = subprocess.run(
        ['python3', 'analysis/viz_futures.py', '--sport', 'ncaaf'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print("❌ Visualization failed!")
        return False
    
    print()
    
    # Copy to _temp.png
    import shutil
    viz_path = repo_root / 'content/viz/ncaaf/ncaaf_futures_vig_single.png'
    temp_path = repo_root / 'content/viz/ncaaf/ncaaf_futures_vig_single_temp.png'
    
    if viz_path.exists():
        shutil.copy(viz_path, temp_path)
        print(f"💾 Copied to: {temp_path}")
        print(f"🖼️  Opening visualization: {temp_path}")
        subprocess.run(['open', str(temp_path)])
    else:
        print(f"⚠️  Visualization not found at: {viz_path}")
    
    return True


def main():
    """Main function"""
    
    # Create futures CSV
    output_file = create_ncaaf_futures_csv()
    
    # Run analysis pipeline
    success = run_analysis_pipeline()
    
    if success:
        print("=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print()
        print(f"📁 Data saved to: {output_file}")
        print(f"📁 Analysis output: data/04_output/ncaaf/")
        print(f"🖼️  Visualization: content/viz/ncaaf/ncaaf_futures_vig_single_temp.png")
        print()
        print("=" * 80)
        print("📝 NOTE: Moneyline odds from The Odds API (January 10, 2026)")
        print("   Game: Miami Hurricanes @ Indiana Hoosiers (Jan 19, 2026)")
        print("=" * 80)
    else:
        print("=" * 80)
        print("❌ PIPELINE FAILED")
        print("=" * 80)


if __name__ == "__main__":
    main()

