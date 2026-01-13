"""
Generic Championship Futures Visualization

Creates FiveThirtyEight-style tables using R's gt package for NFL, NBA, NCAAF, or NCAAB.

Usage:
    python3 analysis/viz_futures.py --sport nfl
    python3 analysis/viz_futures.py --sport nba --top-n 20
    python3 analysis/viz_futures.py --sport ncaaf
    python3 analysis/viz_futures.py --sport ncaab --top-n 25

Output:
    - content/viz/{sport}/{sport}_futures_vig_single.png
"""

import sys
import argparse
import pandas as pd
import yaml
import subprocess
import platform
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import probability_to_american_odds
from r_viz import create_futures_table


def load_configs():
    """Load futures and viz configs."""
    futures_config_path = repo_root / 'config' / 'futures_config.yaml'
    viz_config_path = repo_root / 'config' / 'viz_config.yaml'
    
    with open(futures_config_path) as f:
        futures_config = yaml.safe_load(f)
    
    with open(viz_config_path) as f:
        viz_config = yaml.safe_load(f)
    
    return futures_config, viz_config


def get_team_logos_espn_nfl(team_names):
    """Get ESPN logo URLs for NFL teams."""
    ESPN_NFL_LOGO_BASE = "https://a.espncdn.com/i/teamlogos/nfl/500"
    
    # Map team names to ESPN abbreviations
    nfl_abbr_map = {
        'Los Angeles Rams': 'lar', 'Seattle Seahawks': 'sea', 'Denver Broncos': 'den',
        'Buffalo Bills': 'buf', 'Philadelphia Eagles': 'phi', 'Houston Texans': 'hou',
        'Green Bay Packers': 'gb', 'Tampa Bay Buccaneers': 'tb', 'Los Angeles Chargers': 'lac',
        'Washington Commanders': 'wsh', 'Pittsburgh Steelers': 'pit', 'New England Patriots': 'ne',
        'Tennessee Titans': 'ten', 'Arizona Cardinals': 'ari', 'Baltimore Ravens': 'bal',
        'Detroit Lions': 'det', 'Kansas City Chiefs': 'kc', 'Minnesota Vikings': 'min',
        'San Francisco 49ers': 'sf', 'Cincinnati Bengals': 'cin', 'Dallas Cowboys': 'dal',
        'Indianapolis Colts': 'ind', 'Jacksonville Jaguars': 'jax', 'Miami Dolphins': 'mia',
        'Atlanta Falcons': 'atl', 'Carolina Panthers': 'car', 'Chicago Bears': 'chi',
        'Cleveland Browns': 'cle', 'Las Vegas Raiders': 'lv', 'New Orleans Saints': 'no',
        'New York Giants': 'nyg', 'New York Jets': 'nyj'
    }
    
    logo_map = {}
    for team_name in team_names:
        if team_name in nfl_abbr_map:
            abbr = nfl_abbr_map[team_name]
            logo_map[team_name] = f"{ESPN_NFL_LOGO_BASE}/{abbr}.png"
        else:
            logo_map[team_name] = None
    
    return logo_map


def get_team_logos_espn_nba(team_names):
    """Get ESPN logo URLs for NBA teams."""
    ESPN_NBA_LOGO_BASE = "https://a.espncdn.com/i/teamlogos/nba/500"
    
    # Map team names to ESPN abbreviations
    nba_abbr_map = {
        'Atlanta Hawks': 'atl', 'Boston Celtics': 'bos', 'Brooklyn Nets': 'bkn',
        'Charlotte Hornets': 'cha', 'Chicago Bulls': 'chi', 'Cleveland Cavaliers': 'cle',
        'Dallas Mavericks': 'dal', 'Denver Nuggets': 'den', 'Detroit Pistons': 'det',
        'Golden State Warriors': 'gs', 'Houston Rockets': 'hou', 'Indiana Pacers': 'ind',
        'LA Clippers': 'lac', 'Los Angeles Clippers': 'lac', 'Los Angeles Lakers': 'lal',
        'Memphis Grizzlies': 'mem', 'Miami Heat': 'mia', 'Milwaukee Bucks': 'mil',
        'Minnesota Timberwolves': 'min', 'New Orleans Pelicans': 'no', 'New York Knicks': 'ny',
        'Oklahoma City Thunder': 'okc', 'Orlando Magic': 'orl', 'Philadelphia 76ers': 'phi',
        'Phoenix Suns': 'phx', 'Portland Trail Blazers': 'por', 'Sacramento Kings': 'sac',
        'San Antonio Spurs': 'sa', 'Toronto Raptors': 'tor', 'Utah Jazz': 'utah',
        'Washington Wizards': 'wsh'
    }
    
    logo_map = {}
    for team_name in team_names:
        if team_name in nba_abbr_map:
            abbr = nba_abbr_map[team_name]
            logo_map[team_name] = f"{ESPN_NBA_LOGO_BASE}/{abbr}.png"
        else:
            logo_map[team_name] = None
    
    return logo_map


def get_team_logos_ncaa(team_names):
    """Get local logo file paths for NCAAF/NCAAB teams."""
    from ncaa_team_utils import map_teams_to_logos
    
    logo_map = map_teams_to_logos(team_names, repo_root)
    return logo_map


def prepare_data_for_visualization(df, logo_map):
    """Prepare the dataframe with all display columns."""
    print("📊 Preparing data for visualization...\n")
    
    df_display = df.copy()
    
    # Add team logo URLs
    df_display['logo_url'] = df_display['team'].map(logo_map)
    
    # Add rank
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Format odds strings
    df_display['best_odds_str'] = df_display.apply(
        lambda row: (f"+{int(row['best_odds'])}" if row['best_odds'] > 0 else str(int(row['best_odds']))),
        axis=1
    )
    
    # Calculate average odds from implied_prob_avg
    df_display['avg_odds'] = df_display.apply(
        lambda row: probability_to_american_odds(row['implied_prob_avg'] * 100),
        axis=1
    )
    
    df_display['avg_odds_str'] = df_display.apply(
        lambda row: (f"+{int(row['avg_odds'])}" if row['avg_odds'] > 0 else str(int(row['avg_odds']))),
        axis=1
    )
    
    # Fair odds string
    df_display['fair_odds_str'] = df_display.apply(
        lambda row: (f"+{int(row['fair_odds'])}" if row['fair_odds'] > 0 else str(int(row['fair_odds']))),
        axis=1
    )
    
    # Fair %
    df_display['fair_pct_str'] = df_display['fair_prob'].apply(lambda p: f"{p*100:.1f}")
    
    # Implied %
    df_display['implied_pct_str'] = df_display['implied_prob_avg'].apply(lambda p: f"{p*100:.1f}")
    
    # Calculate vig difference
    df_display['vig_diff'] = (df_display['implied_prob_avg'] - df_display['fair_prob']) * 100
    
    # Calculate best vig
    best_implied_prob = df_display['implied_prob_min']
    df_display['best_vig_diff'] = (best_implied_prob - df_display['fair_prob']) * 100
    
    # Best book display (capitalize)
    df_display['best_book_display'] = df_display['best_book'].str.title()
    
    print(f"   ✅ Prepared {len(df_display)} teams")
    
    logos_with_urls = df_display['logo_url'].notna().sum()
    print(f"   ✅ {logos_with_urls}/{len(df_display)} teams have logos")
    
    avg_vig = df_display['vig_diff'].mean()
    print(f"   ✅ Average vig: {avg_vig:.1f}%\n")
    
    return df_display


def main():
    parser = argparse.ArgumentParser(
        description='Visualize championship futures for NFL, NBA, NCAAF, or NCAAB'
    )
    parser.add_argument(
        '--sport',
        type=str,
        required=True,
        choices=['nfl', 'nba', 'ncaaf', 'ncaab'],
        help='Sport to visualize (nfl, nba, ncaaf, ncaab)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=99999,
        help='Number of top teams to show (default: 99999 = all teams)'
    )
    args = parser.parse_args()
    
    sport = args.sport.lower()
    
    # Load configs
    futures_config, viz_config = load_configs()
    sport_config_futures = futures_config['sports'][sport]
    sport_config_viz = viz_config['sports'][sport]
    viz_settings = viz_config['visualization']
    
    # Merge sport configs (viz settings override futures settings)
    sport_config = {**sport_config_futures, **sport_config_viz}
    
    # Print header
    emoji = sport_config['emoji']
    display_name = sport_config['display_name']
    print("=" * 80)
    print(f"{emoji} {display_name.upper()} CHAMPIONSHIP FUTURES VISUALIZATION (R + GT PACKAGE)")
    print("=" * 80 + "\n")
    
    if args.top_n < 99999:
        print(f"📊 Limiting to top {args.top_n} teams\n")
    
    # Read the analysis CSV (from S3 if not saving locally)
    output_dir = repo_root / sport_config['output_dir']
    output_prefix = sport_config['output_prefix']
    csv_file = output_dir / f'{output_prefix}_fair_odds.csv'
    metadata_file = output_dir / f'{output_prefix}_metadata.csv'
    
    # Download from S3 if not saving locally
    save_locally = futures_config.get('save_locally', False)
    if not save_locally:
        try:
            import boto3
            s3_client = boto3.client('s3')
            s3_bucket = sport_config.get('s3_output_bucket')
            s3_analysis_path = sport_config.get('s3_analysis_path')
            
            # Download analysis CSV
            output_dir.mkdir(parents=True, exist_ok=True)
            s3_key = f"{s3_analysis_path}/{output_prefix}_fair_odds.csv"
            print(f"📥 Downloading from s3://{s3_bucket}/{s3_key}")
            s3_client.download_file(s3_bucket, s3_key, str(csv_file))
            
            # Download metadata
            metadata_s3_key = f"{s3_analysis_path}/{output_prefix}_metadata.csv"
            s3_client.download_file(s3_bucket, metadata_s3_key, str(metadata_file))
            print(f"   ✅ Downloaded analysis files\n")
            
        except Exception as e:
            print(f"❌ Failed to download from S3: {e}")
            print(f"💡 Run analysis first: python3 analysis/analyze_futures.py --sport {sport}")
            sys.exit(1)
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print(f"💡 Run analysis first: python3 analysis/analyze_futures.py --sport {sport}")
        sys.exit(1)
    
    print(f"📂 Reading: {csv_file.name}\n")
    df = pd.read_csv(csv_file)
    
    # Store total teams before filtering
    total_teams = len(df)
    
    # Limit to top N teams
    if args.top_n < total_teams:
        df = df.head(args.top_n)
        print(f"   ⚠️  Showing top {args.top_n} of {total_teams} teams")
    
    print(f"   📊 Loaded {len(df)} teams")
    print(f"   📊 Columns: {list(df.columns)}\n")
    
    # Read metadata
    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)
        avg_vig_pct = metadata_df['avg_vig_pct'].iloc[0]
        print(f"   📊 Average market vig: {avg_vig_pct:.2f}%\n")
    else:
        print(f"   ⚠️  Metadata file not found, calculating vig from team data")
        avg_vig_pct = ((df['implied_prob_avg'] - df['fair_prob']) * 100).mean()
    
    # Get team logos
    team_names = df['team'].tolist()
    logo_source = sport_config['logo_source']
    
    if logo_source == 'espn_cdn':
        if sport == 'nfl':
            logo_map = get_team_logos_espn_nfl(team_names)
        elif sport == 'nba':
            logo_map = get_team_logos_espn_nba(team_names)
        print(f"   {emoji} Loaded ESPN logos\n")
    elif logo_source == 'local':
        logo_map = get_team_logos_ncaa(team_names)
        
        # Get coverage stats
        from ncaa_team_utils import get_logo_coverage_stats
        stats = get_logo_coverage_stats(team_names, repo_root)
        print(f"   {emoji} Logo coverage: {stats['matched']}/{stats['total']} teams ({stats['coverage_pct']:.1f}%)")
        
        if stats['unmatched_teams']:
            print(f"   ⚠️  Teams without logos: {', '.join(stats['unmatched_teams'])}")
        print()
    
    # Prepare data
    df_display = prepare_data_for_visualization(df, logo_map)
    
    # Get save settings from config
    save_locally = futures_config.get('save_locally', False)
    s3_bucket = sport_config.get('s3_output_bucket')
    s3_path = sport_config.get('s3_viz_path')
    
    # Create table
    output_path = create_futures_table(
        df_display=df_display,
        sport=sport,
        sport_config=sport_config,
        viz_config=viz_settings,
        average_vig_pct=avg_vig_pct,
        total_teams=total_teams,
        top_n=args.top_n,
        save_locally=save_locally,
        s3_bucket=s3_bucket,
        s3_path=s3_path
    )
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    
    if save_locally:
        print(f"\n🖼️  Output: {output_path}\n")
    
    # Auto-open only if saved locally
    if save_locally:
        try:
            if platform.system() == 'Darwin':
                subprocess.run(['open', str(output_path)])
                print("📂 Opening PNG...\n")
            elif platform.system() == 'Windows':
                subprocess.run(['start', str(output_path)], shell=True)
            else:
                subprocess.run(['xdg-open', str(output_path)])
        except Exception as e:
            print(f"⚠️  Could not auto-open: {e}")


if __name__ == "__main__":
    main()
