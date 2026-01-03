#!/usr/bin/env python3
"""
Unified workflow for championship futures analysis (NFL + NBA).

Context:
Thomas wanted to:
1. Enhance NFL futures workflow to fetch team records from ESPN API
2. Extend to support NBA championship futures with weekly posts
3. Create a single unified workflow script for both sports

This script orchestrates the complete futures analysis pipeline:
1. Fetch championship odds from The Odds API
2. Fetch team records from ESPN API
3. Analyze vig and calculate fair odds
4. Generate publication-quality visualizations

Usage:
    # Both NFL and NBA (default)
    cd /Users/thomasmyles/dev/betting
    python3 scripts/run_championship_futures_workflow.py
    
    # NFL only
    python3 scripts/run_championship_futures_workflow.py --nfl
    
    # NBA only
    python3 scripts/run_championship_futures_workflow.py --nba

Output:
- data/01_input/the-odds-api/nfl/futures/nfl_super_bowl_futures_YYYYMMDD_HHMMSS.csv
- data/01_input/the-odds-api/nba/futures/nba_championship_futures_YYYYMMDD_HHMMSS.csv
- data/04_output/nfl/nfl_championship_fair_odds.csv
- data/04_output/nba/nba_championship_fair_odds.csv
- content/viz/nfl/futures_vig_single.png
- content/viz/nba/nba_futures_vig_single.png
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Get repo root
repo_root = Path(__file__).parent.parent


def run_command(cmd, description, cwd=None):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run (list or string)
        description: Description of what this command does
        cwd: Working directory (defaults to repo_root)
    """
    if cwd is None:
        cwd = repo_root
    
    print("\n" + "="*80)
    print(f"📌 {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during: {description}")
        print(f"   Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during: {description}")
        print(f"   {e}")
        return False


def run_nfl_workflow():
    """Run complete NFL championship futures workflow"""
    print("\n" + "="*80)
    print("🏈 NFL CHAMPIONSHIP FUTURES WORKFLOW")
    print("="*80)
    
    steps = [
        {
            'cmd': ['python3', 'scripts/fetch_nfl_nba_championship_futures.py'],
            'desc': 'Step 1/3: Fetch NFL Super Bowl odds + team records from ESPN API',
        },
        {
            'cmd': ['python3', 'analysis/analyze_nfl_championship_futures_vig.py'],
            'desc': 'Step 2/3: Analyze vig and calculate fair odds',
        },
        {
            'cmd': ['python3', 'analysis/viz_nfl_futures_gt_single.py'],
            'desc': 'Step 3/3: Generate publication-quality visualization',
        },
    ]
    
    for step in steps:
        success = run_command(step['cmd'], step['desc'])
        if not success:
            print(f"\n❌ NFL workflow failed at: {step['desc']}")
            return False
    
    print("\n" + "="*80)
    print("✅ NFL WORKFLOW COMPLETE!")
    print("="*80)
    print("\n📊 Outputs:")
    print(f"   - Latest odds CSV: data/01_input/the-odds-api/nfl/futures/")
    print(f"   - Fair odds CSV: data/04_output/nfl/nfl_championship_fair_odds.csv")
    print(f"   - Visualization: content/viz/nfl/futures_vig_single.png")
    
    return True


def run_nba_workflow():
    """Run complete NBA championship futures workflow"""
    print("\n" + "="*80)
    print("🏀 NBA CHAMPIONSHIP FUTURES WORKFLOW")
    print("="*80)
    
    steps = [
        {
            'cmd': ['python3', 'scripts/fetch_nfl_nba_championship_futures.py'],
            'desc': 'Step 1/3: Fetch NBA Championship odds + team records from ESPN API',
        },
        {
            'cmd': ['python3', 'analysis/analyze_nba_championship_futures_vig.py'],
            'desc': 'Step 2/3: Analyze vig and calculate fair odds',
        },
        {
            'cmd': ['python3', 'analysis/viz_nba_futures_gt_single.py'],
            'desc': 'Step 3/3: Generate publication-quality visualization',
        },
    ]
    
    for step in steps:
        success = run_command(step['cmd'], step['desc'])
        if not success:
            print(f"\n❌ NBA workflow failed at: {step['desc']}")
            return False
    
    print("\n" + "="*80)
    print("✅ NBA WORKFLOW COMPLETE!")
    print("="*80)
    print("\n📊 Outputs:")
    print(f"   - Latest odds CSV: data/01_input/the-odds-api/nba/futures/")
    print(f"   - Fair odds CSV: data/04_output/nba/nba_championship_fair_odds.csv")
    print(f"   - Visualization: content/viz/nba/nba_futures_vig_single.png")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run championship futures analysis workflow for NFL and/or NBA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both NFL and NBA workflows (default)
  python3 scripts/run_championship_futures_workflow.py
  
  # Run NFL only
  python3 scripts/run_championship_futures_workflow.py --nfl
  
  # Run NBA only
  python3 scripts/run_championship_futures_workflow.py --nba
        """
    )
    
    parser.add_argument(
        '--nfl',
        action='store_true',
        help='Run NFL workflow only'
    )
    
    parser.add_argument(
        '--nba',
        action='store_true',
        help='Run NBA workflow only'
    )
    
    args = parser.parse_args()
    
    # If no flags specified, run both
    run_both = not (args.nfl or args.nba)
    
    print("="*80)
    print("CHAMPIONSHIP FUTURES ANALYSIS WORKFLOW")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if run_both:
        print("\n🎯 Running both NFL and NBA workflows")
    else:
        if args.nfl:
            print("\n🏈 Running NFL workflow only")
        if args.nba:
            print("\n🏀 Running NBA workflow only")
    
    success = True
    
    # Run NFL workflow
    if run_both or args.nfl:
        nfl_success = run_nfl_workflow()
        success = success and nfl_success
    
    # Run NBA workflow
    if run_both or args.nba:
        nba_success = run_nba_workflow()
        success = success and nba_success
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL WORKFLOWS COMPLETED SUCCESSFULLY!")
    else:
        print("❌ SOME WORKFLOWS FAILED - CHECK OUTPUT ABOVE")
    print("="*80)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

