"""
Create a visual table of NFL championship futures with team logos.

Purpose:
- Read the fair odds CSV from analyze_nfl_championship_futures_vig.py
- Get NFL team logos using nfl-data-py
- Create an interactive plotly table with logos
- Export as PNG for social media

Requirements:
    pip install plotly kaleido

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/visualize_nfl_futures_vig.py
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

def get_team_logos():
    """Get NFL team logos from ESPN"""
    print("📥 Loading NFL team logos...")
    
    # ESPN logo URLs - these are publicly available
    logo_map = {
        'Los Angeles Rams': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
            'color': '#003594'
        },
        'Seattle Seahawks': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
            'color': '#002244'
        },
        'Denver Broncos': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
            'color': '#FB4F14'
        },
        'Buffalo Bills': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
            'color': '#00338D'
        },
        'Philadelphia Eagles': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
            'color': '#004C54'
        },
        'Houston Texans': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
            'color': '#03202F'
        },
        'Green Bay Packers': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
            'color': '#203731'
        },
        'New England Patriots': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
            'color': '#002244'
        },
        'Jacksonville Jaguars': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
            'color': '#006778'
        },
        'San Francisco 49ers': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
            'color': '#AA0000'
        },
        'Baltimore Ravens': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
            'color': '#241773'
        },
        'Los Angeles Chargers': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
            'color': '#0080C6'
        },
        'Detroit Lions': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
            'color': '#0076B6'
        },
        'Chicago Bears': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
            'color': '#0B162A'
        },
        'Tampa Bay Buccaneers': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
            'color': '#D50A0A'
        },
        'Pittsburgh Steelers': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
            'color': '#FFB612'
        },
        'Carolina Panthers': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
            'color': '#0085CA'
        },
        'Indianapolis Colts': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
            'color': '#002C5F'
        },
        'Dallas Cowboys': {
            'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
            'color': '#003594'
        },
    }
    
    print(f"✅ Loaded {len(logo_map)} team logos\n")
    return logo_map


def match_team_to_logo(team_name, logo_map):
    """Match betting team names to logo data"""
    if team_name in logo_map:
        return logo_map[team_name]
    
    print(f"⚠️  Could not match logo for: {team_name}")
    return {'logo': '', 'color': '#003366'}


def create_plotly_table(df, logo_map):
    """Create an interactive plotly table with team logos"""
    
    print("🎨 Creating plotly visualization...")
    
    # Add logo URLs to dataframe
    df['logo_url'] = df['team'].apply(lambda x: match_team_to_logo(x, logo_map)['logo'] if match_team_to_logo(x, logo_map) else '')
    
    # Format odds columns
    df['best_odds_str'] = df['best_odds'].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))
    df['fair_odds_str'] = df['fair_odds'].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))
    
    # Format percentage columns
    df['implied_prob_str'] = df['implied_prob_avg'].apply(lambda x: f"{x*100:.2f}%")
    df['fair_prob_str'] = df['fair_prob'].apply(lambda x: f"{x*100:.2f}%")
    
    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Logo', 'Team', 'Best Book', 'Best Odds', 'Implied %', 'Fair Odds', 'Fair %'],
            fill_color='#003366',  # Dark blue
            font=dict(color='white', size=14, family='Arial Black'),
            align=['center', 'center', 'left', 'center', 'center', 'center', 'center', 'center'],
            height=40
        ),
        cells=dict(
            values=[
                list(range(1, len(df) + 1)),  # Rank
                [f'<img src="{url}" width="40" height="40">' for url in df['logo_url']],  # Logo
                df['team'],  # Team name
                df['best_book'],  # Best book
                df['best_odds_str'],  # Best odds
                df['implied_prob_str'],  # Implied %
                df['fair_odds_str'],  # Fair odds
                df['fair_prob_str']  # Fair %
            ],
            fill_color=[['#f8f9fa', '#ffffff'] * len(df)],  # Alternating row colors
            font=dict(color='#333333', size=12),
            align=['center', 'center', 'left', 'center', 'center', 'center', 'center', 'center'],
            height=50
        )
    )])
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'NFL Super Bowl Futures: Implied vs Fair Odds<br><sub>18.5% Average Vig Across All Books</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#003366', 'family': 'Arial Black'}
        },
        width=1400,
        height=1200,
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    return fig


def main():
    """Main visualization function"""
    
    print("="*80)
    print("NFL CHAMPIONSHIP FUTURES VISUALIZATION")
    print("="*80 + "\n")
    
    # Read the CSV from analyze_nfl_championship_futures_vig.py
    csv_file = repo_root / 'data/04_output/nfl/nfl_championship_fair_odds.csv'
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Run analyze_nfl_championship_futures_vig.py first!")
        return
    
    print(f"📂 Reading: {csv_file.name}\n")
    df = pd.read_csv(csv_file)
    
    # Get team logos
    logo_map = get_team_logos()
    
    # Create plotly table
    fig = create_plotly_table(df, logo_map)
    
    # Save as HTML (interactive)
    html_output = repo_root / 'data/04_output/nfl/nfl_championship_futures_viz.html'
    fig.write_html(str(html_output))
    print(f"✅ Saved interactive HTML: {html_output}")
    
    # Save as PNG (static image for social media)
    png_output = repo_root / 'data/04_output/nfl/nfl_championship_futures_viz.png'
    try:
        fig.write_image(str(png_output), width=1400, height=1200, scale=2)
        print(f"✅ Saved PNG image: {png_output}")
    except Exception as e:
        print(f"⚠️  Could not save PNG: {e}")
        print("Install kaleido for PNG export: pip install kaleido")
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Open the HTML file in your browser to view the interactive table")
    print(f"🖼️  Use the PNG file for social media posts")


if __name__ == "__main__":
    main()

