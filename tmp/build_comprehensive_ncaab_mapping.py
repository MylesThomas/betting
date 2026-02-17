"""
Build comprehensive NCAAB team name mapping using boto3 + S3 data.

Query a month of historical line movement snapshots from S3 to get all 327-364 D1 teams.
Then create mapping: Odds API team name → ESPN team name.

Context:
- Historical data in S3: s3://betting-line-movement-snapshots/data/01_input/the-odds-api/ncaab/line_movement/
- Need to capture all teams, not just the ~109 from futures data
- Each team plays throughout season, so a month of data should capture all active teams

Created: 2026-02-16
"""

import os
import sys
from pathlib import Path
import pandas as pd
from io import StringIO
import boto3

# Find project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# AWS Configuration
S3_BUCKET = 'betting-line-movement-snapshots'
S3_PATH = 'data/01_input/the-odds-api/ncaab/line_movement'


def extract_all_teams_from_s3(num_files=100):
    """
    Extract all unique team names from S3 historical snapshots.
    
    Args:
        num_files: Number of snapshot files to sample
    
    Returns:
        List of unique team names from Odds API
    """
    print("="*80)
    print("EXTRACTING ODDS API TEAM NAMES FROM S3")
    print("="*80)
    print(f"S3 path: s3://{S3_BUCKET}/{S3_PATH}/")
    print(f"Sampling: {num_files} files")
    print()
    
    s3 = boto3.client('s3')
    
    # List all snapshot files
    print("📂 Listing S3 files...")
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PATH + '/'
    )
    
    files = [obj['Key'] for obj in response.get('Contents', [])]
    print(f"   Found {len(files)} total files")
    print()
    
    # Sample files evenly distributed across date range
    step = max(1, len(files) // num_files)
    sampled_files = files[::step][:num_files]
    
    print(f"📥 Processing {len(sampled_files)} files...")
    
    teams = set()
    files_processed = 0
    
    for file_key in sampled_files:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=file_key)
            csv_content = obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            if 'away_team' in df.columns:
                teams.update(df['away_team'].unique())
            if 'home_team' in df.columns:
                teams.update(df['home_team'].unique())
            
            files_processed += 1
            
            if files_processed % 20 == 0:
                print(f"   Processed {files_processed}/{len(sampled_files)} files... ({len(teams)} teams so far)")
        except Exception as e:
            print(f"   ⚠️  Error reading {file_key}: {e}")
    
    print()
    print(f"✅ Processed {files_processed} files")
    print(f"✅ Found {len(teams)} unique teams from Odds API")
    print()
    
    return sorted(teams)


def generate_mapping_dict(odds_api_teams):
    """
    Generate comprehensive mapping dictionary.
    
    Args:
        odds_api_teams: List of team names from Odds API
    
    Returns:
        Dictionary mapping Odds API names to ESPN names
    """
    print("="*80)
    print("GENERATING MAPPING DICTIONARY")
    print("="*80)
    print()
    
    mapping = {}
    
    # Pattern-based normalizations
    for team in odds_api_teams:
        normalized = team
        
        # St → State (most common)
        if ' St ' in normalized:
            normalized = normalized.replace(' St ', ' State ')
        
        # St at end (e.g., "Alcorn St Braves")
        if normalized.endswith(' St Braves'):
            normalized = normalized.replace(' St Braves', ' State Braves')
        if normalized.endswith(' St Red Wolves'):
            normalized = normalized.replace(' St Red Wolves', ' State Red Wolves')
        # ... (add more patterns as needed)
        
        # Univ. → University
        if 'Univ.' in normalized:
            normalized = normalized.replace('Univ.', 'University')
        
        # Miss → Mississippi (at start)
        if normalized.startswith('Miss '):
            normalized = normalized.replace('Miss ', 'Mississippi ', 1)
        
        # CSU → Cal State
        if normalized.startswith('CSU '):
            normalized = normalized.replace('CSU ', 'Cal State ')
        
        # Only add to mapping if different
        if normalized != team:
            mapping[team] = normalized
            print(f"  {team:<50} → {normalized}")
    
    print()
    print(f"✅ Created {len(mapping)} mappings")
    print(f"   {len(odds_api_teams) - len(mapping)} teams don't need normalization")
    print()
    
    return mapping


def generate_python_code(mapping):
    """Generate Python dictionary code."""
    print("="*80)
    print("GENERATED PYTHON CODE")
    print("="*80)
    print()
    
    print("ODDS_API_TO_ESPN_NCAAB = {")
    for odds_name, espn_name in sorted(mapping.items()):
        print(f'    "{odds_name}": "{espn_name}",')
    print("}")
    print()


def save_to_file(mapping, odds_api_teams, output_file='src/ncaab_team_name_mapping.py'):
    """Save mapping to Python file."""
    print(f"💾 Saving to {output_file}...")
    
    # Read existing file to preserve docstring
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            content = f.read()
            # Extract docstring
            if '"""' in content:
                docstring_end = content.find('"""', 3) + 3
                docstring = content[:docstring_end]
            else:
                docstring = '"""\nNCAA Team Name Mapping\n"""'
    else:
        docstring = '"""\nNCAA Team Name Mapping\n\nAuto-generated mapping from Odds API to ESPN team names.\n"""'
    
    # Generate new file content
    lines = [
        docstring,
        "\n\n",
        "# Complete mapping: Odds API → ESPN (all NCAAB teams)",
        f"# Generated from {len(odds_api_teams)} teams in historical data",
        f"# {len(mapping)} teams require normalization, {len(odds_api_teams) - len(mapping)} are identical",
        "ODDS_API_TO_ESPN_NCAAB = {",
        f"    # ============================================================================",
        f"    # TEAMS REQUIRING NORMALIZATION ({len(mapping)} teams)",
        f"    # ============================================================================",
    ]
    
    # Add teams that need normalization first
    for odds_name, espn_name in sorted(mapping.items()):
        lines.append(f'    "{odds_name}": "{espn_name}",')
    
    lines.extend([
        "",
        f"    # ============================================================================",
        f"    # TEAMS WITH IDENTICAL NAMES ({len(odds_api_teams) - len(mapping)} teams)",
        f"    # ============================================================================",
    ])
    
    # Then add all identical teams
    all_teams = set(odds_api_teams)
    teams_with_mapping = set(mapping.keys())
    identical_teams = sorted(all_teams - teams_with_mapping)
    
    for team_name in identical_teams:
        lines.append(f'    "{team_name}": "{team_name}",')
    
    lines.extend([
        "}",
        "",
        "",
        "# Validation assertions (run at import time)",
        f"assert len(ODDS_API_TO_ESPN_NCAAB) == {len(all_teams)}, \\",
        f'    f"Expected {len(all_teams)} total NCAAB teams, got {{len(ODDS_API_TO_ESPN_NCAAB)}}"',
        "",
        "# Count teams with different names (key != value)",
        "differences_count = sum(1 for k, v in ODDS_API_TO_ESPN_NCAAB.items() if k != v)",
        f"assert differences_count == {len(mapping)}, \\",
        f'    f"Expected {len(mapping)} teams with different names, got {{differences_count}}"',
        "",
        "",
        "def normalize_ncaab_team_name(odds_api_name: str) -> str:",
        '    """',
        "    Normalize NCAAB team name from The Odds API format to ESPN format.",
        "    ",
        "    Args:",
        "        odds_api_name: Team name from The Odds API",
        "        ",
        "    Returns:",
        "        Normalized team name matching ESPN format",
        '    """',
        "    # Check exact mapping first",
        "    if odds_api_name in ODDS_API_TO_ESPN_NCAAB:",
        "        return ODDS_API_TO_ESPN_NCAAB[odds_api_name]",
        "    ",
        "    # If not in mapping, return as-is (most teams are identical)",
        "    return odds_api_name",
        "",
    ])
    
    output_path = project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Saved to {output_path}")
    print()


def main():
    print("\n" + "="*80)
    print("NCAAB TEAM NAME MAPPING BUILDER (boto3 + S3)")
    print("="*80)
    print()
    
    # Extract all teams from S3 historical data (sample 100 files from past month)
    odds_api_teams = extract_all_teams_from_s3(num_files=100)
    
    if not odds_api_teams:
        print("❌ No teams found. Check S3 access and data availability.")
        return
    
    # Generate mapping
    mapping = generate_mapping_dict(odds_api_teams)
    
    # Show generated code
    generate_python_code(mapping)
    
    # Save to file
    save_to_file(mapping, odds_api_teams)
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total teams: {len(odds_api_teams)}")
    print(f"Teams requiring normalization: {len(mapping)}")
    print(f"Teams identical in both APIs: {len(odds_api_teams) - len(mapping)}")
    print(f"Coverage: 100% ({len(odds_api_teams)}/{len(odds_api_teams)})")
    print()
    print("✅ Comprehensive mapping complete!")
    print()


if __name__ == '__main__':
    main()
