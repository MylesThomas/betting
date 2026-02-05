"""
STEP 2: Analyze player build failures and suggest fixes.

Run this AFTER running 01_build.py to understand what failed and why.
Provides actionable recommendations for fixing name normalization issues.

Usage:
    python src/player_team_history/02_analyze_failures.py

Input:
    ~/Downloads/tmp/player_team_history/failures.txt

Output:
    - Categorized failures (garbage data, reversed names, full legal names, etc.)
    - Suggested code to add to name_normalization.py
    - Actionable recommendations

Next Steps:
    1. Review the suggestions
    2. Add recommended mappings to name_normalization.py
    3. Test: python src/player_team_history/name_normalization.py
    4. Re-run: python src/player_team_history/01_build.py
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.config import EMOJI
from src.player_team_history.name_normalization import is_college_player

FAILURE_REPORT = Path.home() / 'Downloads' / 'tmp' / 'player_team_history' / 'failures.txt'


def parse_failure_report():
    """Parse the failure report file."""
    if not FAILURE_REPORT.exists():
        print(f"{EMOJI['error']} Failure report not found: {FAILURE_REPORT}")
        print("   Run 01_build.py first to generate the report.")
        return None
    
    with open(FAILURE_REPORT, 'r') as f:
        content = f.read()
    
    failures = {
        'not_found_in_nba': [],
        'no_game_logs': [],
        'no_history_created': [],
        'processing_errors': []
    }
    
    current_section = None
    
    for line in content.split('\n'):
        if 'NOT FOUND IN NBA API' in line:
            current_section = 'not_found_in_nba'
        elif 'NO GAME LOGS' in line:
            current_section = 'no_game_logs'
        elif 'NO HISTORY CREATED' in line:
            current_section = 'no_history_created'
        elif 'PROCESSING ERRORS' in line:
            current_section = 'processing_errors'
        elif line.strip().startswith('- ') and current_section:
            player_name = line.strip()[2:]
            failures[current_section].append(player_name)
    
    return failures


def categorize_not_found(player_names):
    """Categorize 'not found' players by issue type."""
    categories = {
        'college_players': [],
        'garbage_data': [],
        'reversed_names': [],
        'full_legal_names': [],
        'typos': [],
        'missing_mappings': []
    }
    
    for name in player_names:
        # College players (check FIRST - these are expected to not be in NBA API)
        # Don't add them to any other category
        if is_college_player(name):
            categories['college_players'].append(name)
            continue  # Skip other checks for college players
        
        # Garbage data patterns
        if any(x in name.lower() for x in ['total', 'alternate', 'over', 'under']):
            categories['garbage_data'].append(name)
        # Reversed names
        elif ' ' in name and len(name.split()) >= 2:
            words = name.split()
            if words[-1][0].isupper() and words[0][0].isupper():
                common_last_to_first = ['Caldwell', 'Grant', 'Love', 'Highsmith', 'Huerter', 'Murray', 'Portis', 'Wembanyama']
                if any(words[0].startswith(x) for x in common_last_to_first):
                    categories['reversed_names'].append(name)
                    continue
        
        # Full legal names (3+ words with middle names)
        words = name.split()
        if len(words) >= 3 and not any(x in name for x in ['Jr', 'Sr', 'II', 'III']):
            categories['full_legal_names'].append(name)
        # Single initial names (likely typos)
        elif len(words) >= 2 and len(words[0]) <= 2:
            categories['typos'].append(name)
        else:
            categories['missing_mappings'].append(name)
    
    return categories


def suggest_fixes(failures):
    """Suggest fixes for failures."""
    print("="*80)
    print(f"{EMOJI['chart']} FAILURE ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    print()
    
    # Analyze NOT FOUND IN NBA API
    if failures['not_found_in_nba']:
        print(f"{EMOJI['warning']} NOT FOUND IN NBA API ({len(failures['not_found_in_nba'])} players)")
        print("-"*80)
        
        categories = categorize_not_found(failures['not_found_in_nba'])
        
        if categories['college_players']:
            print(f"\n1. COLLEGE PLAYERS ({len(categories['college_players'])} players)")
            print("   These are college players, NOT expected in NBA API.")
            print("   Action: No fix needed - already in get_college_players() list")
            print()
            for name in sorted(categories['college_players']):
                print(f"      • {name}")
        
        if categories['garbage_data']:
            print(f"\n2. GARBAGE DATA ({len(categories['garbage_data'])} items)")
            print("   These are not real players - already filtered by name_normalization.py")
            print("   No action needed - filtering is working correctly.")
            for name in sorted(categories['garbage_data'])[:5]:
                print(f"      • {name}")
            if len(categories['garbage_data']) > 5:
                print(f"      ... and {len(categories['garbage_data']) - 5} more")
        
        if categories['reversed_names']:
            print(f"\n3. REVERSED NAMES ({len(categories['reversed_names'])} players)")
            print("   Add to fix_reversed_names() in name_normalization.py:")
            print()
            for name in sorted(categories['reversed_names']):
                words = name.split()
                fixed = f"{words[-1]} {' '.join(words[:-1])}"
                print(f"      '{name}': '{fixed}',")
        
        if categories['full_legal_names']:
            print(f"\n4. FULL LEGAL NAMES ({len(categories['full_legal_names'])} players)")
            print("   Add to get_odds_api_to_nba_mappings() in name_normalization.py:")
            print("   (Need to look up NBA API names manually)")
            print()
            for name in sorted(categories['full_legal_names']):
                print(f"      '{name}': 'TODO - lookup in NBA API',")
        
        if categories['typos']:
            print(f"\n5. TYPOS/ABBREVIATIONS ({len(categories['typos'])} players)")
            print("   These are malformed names from Odds API:")
            print()
            for name in sorted(categories['typos']):
                print(f"      • {name}")
        
        if categories['missing_mappings']:
            print(f"\n6. MISSING MAPPINGS ({len(categories['missing_mappings'])} players)")
            print("   Need manual investigation:")
            print()
            for name in sorted(categories['missing_mappings']):
                print(f"      • {name}")
    
    # Analyze NO GAME LOGS
    if failures['no_game_logs']:
        print(f"\n{EMOJI['info']} NO GAME LOGS ({len(failures['no_game_logs'])} players)")
        print("-"*80)
        print("These players exist in NBA API but have no game logs.")
        print("Common reasons: rookies not yet played, recently retired, etc.")
        print()
        for name in sorted(failures['no_game_logs'])[:10]:
            print(f"   • {name}")
        if len(failures['no_game_logs']) > 10:
            print(f"   ... and {len(failures['no_game_logs']) - 10} more")
    
    # Analyze NO HISTORY CREATED
    if failures['no_history_created']:
        print(f"\n{EMOJI['error']} NO HISTORY CREATED ({len(failures['no_history_created'])} players)")
        print("-"*80)
        print("Game logs fetched but team history could not be created.")
        print("This indicates a bug in create_team_history_from_gamelogs()")
        print()
        for name in sorted(failures['no_history_created']):
            print(f"   • {name}")
    
    # Summary
    print()
    print("="*80)
    print(f"{EMOJI['success']} SUMMARY")
    print("="*80)
    total_failures = sum(len(v) for v in failures.values())
    actionable = len(categories.get('reversed_names', [])) + len(categories.get('full_legal_names', []))
    expected = (
        len(categories.get('college_players', [])) + 
        len(categories.get('garbage_data', [])) + 
        len(failures.get('no_game_logs', []))
    )
    print(f"Total failures: {total_failures}")
    print(f"Actionable (need name mappings): {actionable}")
    print(f"Expected failures (college/garbage/no-logs): {expected}")
    print(f"  - College players: {len(categories.get('college_players', []))}")
    print(f"  - Garbage data: {len(categories.get('garbage_data', []))}")
    print(f"  - No game logs: {len(failures.get('no_game_logs', []))}")
    print()


def main():
    print("="*80)
    print(f"{EMOJI['test']} ANALYZING PLAYER BUILD FAILURES")
    print("="*80)
    print()
    
    failures = parse_failure_report()
    if not failures:
        return
    
    suggest_fixes(failures)


if __name__ == '__main__':
    main()
