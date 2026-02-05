"""
Test name normalization across all 3 APIs to identify players needing special treatment.

This script:
1. Gets players from S3 (The Odds API) - already normalized
2. For each player, queries NBA API to get their official name
3. For each player, queries ESPN API to get their name (if available)
4. Compares normalized names from all 3 sources
5. Reports which players DON'T match after normalization

This helps us identify which players need hardcoded mappings in name_normalization.py

Usage:
    python src/player_team_history/tmp/test_name_normalization_across_apis.py --sample 50

Author: Myles Thomas
Date: 2025-02-04
"""

import pandas as pd
from pathlib import Path
import sys
import time
import argparse
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

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.player_team_history.name_normalization import normalize_player_name
from src.player_team_history.discovery import discover_all_players
from src.config import EMOJI

try:
    from nba_api.stats.static import players as nba_players
except ImportError:
    print(f"{EMOJI['error']} nba_api not found. Install with: pip install nba_api")
    sys.exit(1)


def find_player_in_nba_api(normalized_name):
    """
    Find player in NBA API by normalized name.
    
    Returns:
        (player_id, nba_api_name, nba_api_name_normalized) or (None, None, None)
    """
    all_players = nba_players.get_players()
    
    # Try exact match on normalized names
    for p in all_players:
        nba_normalized = normalize_player_name(p['full_name'])
        if nba_normalized == normalized_name:
            return (p['id'], p['full_name'], nba_normalized)
    
    # Try partial match
    for p in all_players:
        nba_normalized = normalize_player_name(p['full_name'])
        if nba_normalized and normalized_name in nba_normalized:
            return (p['id'], p['full_name'], nba_normalized)
    
    return (None, None, None)


def test_normalization_across_apis(sample_size=50):
    """
    Test name normalization across APIs and report mismatches.
    
    Args:
        sample_size: Number of S3 files to sample for player discovery
    """
    print("="*80)
    print(f"{EMOJI['test']} NAME NORMALIZATION CROSS-API TEST")
    print("="*80)
    print()
    print("Testing player names from:")
    print("  1. The Odds API (S3 props data)")
    print("  2. NBA API (nba_api library)")
    print("  3. ESPN API (TODO)")
    print()
    
    # Get players from S3 (Odds API) - already normalized
    print(f"{EMOJI['chart']} Discovering players from S3...")
    odds_api_players = discover_all_players(s3_sample_size=sample_size, verbose=False)
    odds_api_players = sorted(list(odds_api_players))
    print(f"{EMOJI['success']} Found {len(odds_api_players)} unique players from Odds API\n")
    
    # Test each player across APIs
    results = []
    
    print(f"Testing {len(odds_api_players)} players across APIs...\n")
    
    for i, odds_name in enumerate(odds_api_players, 1):
        if i % 25 == 0:
            print(f"[{i}/{len(odds_api_players)}] Testing...")
        
        # Query NBA API
        nba_id, nba_raw_name, nba_normalized = find_player_in_nba_api(odds_name)
        
        # Check if names match after normalization
        match_status = "✅" if odds_name == nba_normalized else "❌"
        
        results.append({
            'odds_api_name': odds_name,
            'nba_api_raw': nba_raw_name,
            'nba_api_normalized': nba_normalized,
            'nba_api_found': nba_id is not None,
            'names_match': odds_name == nba_normalized,
        })
        
        # Rate limit
        time.sleep(0.1)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    
    total = len(df)
    found_in_nba = df['nba_api_found'].sum()
    names_match = df['names_match'].sum()
    not_found = total - found_in_nba
    found_but_mismatch = found_in_nba - names_match
    
    print(f"Total players from Odds API: {total}")
    print(f"Found in NBA API: {found_in_nba} ({found_in_nba/total*100:.1f}%)")
    print(f"Names match after normalization: {names_match} ({names_match/total*100:.1f}%)")
    print()
    print(f"{EMOJI['error']} Not found in NBA API: {not_found}")
    print(f"{EMOJI['warning']} Found but names don't match: {found_but_mismatch}")
    print()
    
    # Show players not found in NBA API
    if not_found > 0:
        print("="*80)
        print("PLAYERS NOT FOUND IN NBA API")
        print("="*80)
        not_found_df = df[~df['nba_api_found']].copy()
        for idx, row in not_found_df.iterrows():
            print(f"  {EMOJI['error']} {row['odds_api_name']}")
        print()
    
    # Show players with name mismatches
    if found_but_mismatch > 0:
        print("="*80)
        print("PLAYERS WITH NAME MISMATCHES (NEED HARDCODED MAPPINGS)")
        print("="*80)
        print()
        mismatch_df = df[df['nba_api_found'] & ~df['names_match']].copy()
        
        print("Add these to get_canonical_name_mappings() in name_normalization.py:")
        print()
        print("```python")
        for idx, row in mismatch_df.iterrows():
            odds = row['odds_api_name']
            nba = row['nba_api_normalized']
            print(f"    '{odds}': '{nba}',  # Odds API -> NBA API")
        print("```")
        print()
        
        print("Detailed mismatches:")
        print()
        for idx, row in mismatch_df.iterrows():
            print(f"  {EMOJI['warning']} Odds API: {row['odds_api_name']}")
            print(f"     NBA API (raw): {row['nba_api_raw']}")
            print(f"     NBA API (norm): {row['nba_api_normalized']}")
            print()
    
    if found_but_mismatch == 0 and not_found == 0:
        print(f"{EMOJI['success']} ALL PLAYERS MATCH PERFECTLY!")
        print()
    
    # Save detailed results
    output_dir = Path.home() / 'Downloads' / 'tmp'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'name_normalization_test_results.csv'
    df.to_csv(output_file, index=False)
    print(f"{EMOJI['save']} Detailed results saved to: {output_file}")
    print()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Test name normalization across APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sample', type=int, default=50,
                       help='Number of S3 files to sample (default: 50)')
    
    args = parser.parse_args()
    
    test_normalization_across_apis(sample_size=args.sample)


if __name__ == '__main__':
    main()
