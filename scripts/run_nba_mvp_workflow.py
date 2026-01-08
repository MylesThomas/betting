"""
NBA MVP Odds Workflow - Complete Pipeline

Context:
Thomas wanted to track NBA MVP odds similar to championship futures.
The Odds API doesn't support MVP futures, so we hardcode FanDuel odds manually.

This script runs the complete workflow:
1. Fetch MVP odds from hardcoded FanDuel data
2. Calculate fair odds and vig
3. Generate visualization

Usage:
    python3 scripts/run_nba_mvp_workflow.py

Output:
    - data/01_input/fanduel/nba/mvp/nba_mvp_odds_YYYYMMDD_HHMMSS.csv
    - data/04_output/nba/mvp/nba_mvp_fair_odds_YYYYMMDD_HHMMSS.csv
    - content/viz/nba/nba_mvp_vig.png

To update MVP odds:
1. Go to FanDuel → NBA → Awards → MVP
2. Edit CURRENT_MVP_ODDS dict in scripts/fetch_nba_mvp_odds_fanduel.py
3. Update FETCH_DATE to today's date
4. Run this workflow script
"""

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=repo_root)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {' '.join(cmd)}")
        sys.exit(1)
    
    return True


def main():
    """Run complete NBA MVP workflow"""
    
    print("="*80)
    print("NBA MVP ODDS WORKFLOW")
    print("="*80)
    
    steps = [
        {
            'cmd': ['python3', 'scripts/fetch_nba_mvp_odds_fanduel.py'],
            'desc': '1️⃣  Fetching MVP odds from FanDuel (hardcoded)'
        },
        {
            'cmd': ['python3', 'analysis/analyze_nba_mvp_vig.py'],
            'desc': '2️⃣  Analyzing vig and calculating fair odds'
        },
        {
            'cmd': ['python3', 'analysis/viz_nba_mvp_gt.py'],
            'desc': '3️⃣  Generating visualization'
        }
    ]
    
    for step in steps:
        run_command(step['cmd'], step['desc'])
    
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE!")
    print("="*80)
    
    print("\nOutput files:")
    print(f"   - Latest odds CSV: data/01_input/fanduel/nba/mvp/")
    print(f"   - Fair odds CSV: data/04_output/nba/mvp/")
    print(f"   - Visualization: content/viz/nba/nba_mvp_vig.png")
    
    print("\nNext steps:")
    print("   1. View the visualization")
    print("   2. Share on X/Twitter")
    print("   3. Update weekly by editing CURRENT_MVP_ODDS in fetch_nba_mvp_odds_fanduel.py")


if __name__ == "__main__":
    main()

