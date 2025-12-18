"""
Create a FiveThirtyEight-style visualization of NFL championship futures with team logos using matplotlib.

Purpose:
- Read the fair odds CSV
- Create a 538-style graphic with embedded logo images
- Export as high-quality PNG for social media

Implementation:
- Uses matplotlib for rendering (instead of R's gt package)
- Embeds team logos using OffsetImage and AnnotationBbox
- Custom color gradients for vig visualization

Requirements:
    pip install matplotlib pillow requests

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/viz_nfl_futures_matplotlib.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import urllib3
import subprocess
import platform
import numpy as np

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))


def get_team_logos():
    """Get NFL team logos from ESPN - all 32 teams"""
    logo_map = {
        # Teams with odds
        'Los Angeles Rams': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
        'Seattle Seahawks': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
        'Denver Broncos': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
        'Buffalo Bills': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
        'Philadelphia Eagles': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
        'Houston Texans': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
        'Green Bay Packers': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
        'New England Patriots': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
        'Jacksonville Jaguars': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
        'San Francisco 49ers': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
        'Baltimore Ravens': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
        'Los Angeles Chargers': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
        'Detroit Lions': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
        'Chicago Bears': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
        'Tampa Bay Buccaneers': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
        'Pittsburgh Steelers': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
        'Carolina Panthers': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
        'Indianapolis Colts': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
        'Dallas Cowboys': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
        # Teams without odds (eliminated/longshots)
        'Kansas City Chiefs': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
        'Minnesota Vikings': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
        'Washington Commanders': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
        'Atlanta Falcons': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
        'Arizona Cardinals': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
        'Miami Dolphins': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
        'Cincinnati Bengals': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
        'New Orleans Saints': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
        'New York Jets': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
        'Cleveland Browns': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
        'Tennessee Titans': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
        'Las Vegas Raiders': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
        'New York Giants': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
    }
    return logo_map


def download_logo(url, zoom=0.12):
    """Download and prepare logo for matplotlib with consistent sizing"""
    try:
        response = requests.get(url, timeout=5, verify=False)
        img = Image.open(BytesIO(response.content))
        
        # Force all logos to same size (50x50 pixels) for consistency
        img = img.resize((50, 50), Image.Resampling.LANCZOS)
        
        return OffsetImage(img, zoom=0.5)
    except Exception as e:
        print(f"Failed to download logo from {url}: {e}")
        return None


def create_538_table(df, logo_map):
    """Create a FiveThirtyEight-style table with logos"""
    
    print("🎨 Creating FiveThirtyEight-style visualization...\n")
    
    # Use matplotlib's built-in FiveThirtyEight theme
    plt.style.use('fivethirtyeight')
    
    # Create figure - fixed size for proper export
    num_rows = len(df)
    assert num_rows == 32, f"Expected 32 teams, got {num_rows}"
    fig_height = 15  # Taller canvas to fit everything
    fig_width = 16
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    # Title - centered and professional
    title_text = "NFL Super Bowl Futures: True Odds vs. What Books Charge"
    subtitle_text = "Bookmakers charge an average 18.5% vig on championship futures (vs 4-5% on game lines)"
    
    fig.text(0.5, 0.98, title_text, 
             fontsize=22, fontweight='bold', ha='center')
    fig.text(0.5, 0.96, subtitle_text,
             fontsize=11, style='italic', ha='center', color='#555555')
    
    # Prepare data - show all available teams
    df_display = df.copy()
    
    # Format columns based on whether team has odds
    df_display['has_odds'] = df_display['num_books'] > 0
    
    # Best odds string
    df_display['best_odds_str'] = df_display.apply(
        lambda row: '-' if not row['has_odds'] 
        else (f"+{int(row['best_odds'])}" if row['best_odds'] > 0 else str(int(row['best_odds']))),
        axis=1
    )
    
    # Calculate average odds from implied_prob_avg
    from odds_utils import probability_to_american_odds, american_odds_to_percentage_probability
    df_display['avg_odds'] = df_display.apply(
        lambda row: 100000 if not row['has_odds']
        else probability_to_american_odds(row['implied_prob_avg'] * 100),
        axis=1
    )
    
    # Average odds string
    df_display['avg_odds_str'] = df_display.apply(
        lambda row: '-' if not row['has_odds']
        else (f"+{int(row['avg_odds'])}" if row['avg_odds'] > 0 else str(int(row['avg_odds']))),
        axis=1
    )
    
    # Fair odds string
    df_display['fair_odds_str'] = df_display.apply(
        lambda row: '+100000' if not row['has_odds']
        else (f"+{int(row['fair_odds'])}" if row['fair_odds'] > 0 else str(int(row['fair_odds']))),
        axis=1
    )
    
    # Calculate Best Vig % (vig on the best odds specifically)
    df_display['best_vig_diff'] = df_display.apply(
        lambda row: None if not row['has_odds']
        else (american_odds_to_percentage_probability(row['best_odds']) / 100 - row['fair_prob']) * 100,
        axis=1
    )
    
    # Implied % and Fair %
    df_display['implied_pct'] = df_display.apply(
        lambda row: 0.0 if not row['has_odds'] else (row['implied_prob_avg'] * 100),
        axis=1
    ).round(1)
    
    df_display['fair_pct_str'] = df_display.apply(
        lambda row: '<0.1' if not row['has_odds'] else str(round(row['fair_prob'] * 100, 1)),
        axis=1
    )
    
    # Calculate vig difference (only for teams with odds)
    df_display['vig_diff'] = df_display.apply(
        lambda row: None if not row['has_odds'] 
        else (row['implied_prob_avg'] - row['fair_prob']) * 100,
        axis=1
    )
    
    # Best book display
    df_display['best_book_display'] = df_display['best_book'].fillna('-')
    
    # Create red-to-green colormap
    colors = ['#d62728', '#ff9999', '#ffffff', '#90EE90', '#2ca02c']  # red -> white -> green
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('vig_cmap', colors, N=n_bins)
    
    # Column headers - reorganized with 'Best' columns at the end
    headers = ['#', 'Team', 'W-L', 'Avg Odds', 'Implied %', 'Fair Odds', 'Fair %', 'Vig %', 'Best Book', 'Best Odds', 'Best Vig %']
    col_widths = [0.04, 0.15, 0.06, 0.09, 0.09, 0.10, 0.08, 0.08, 0.11, 0.09, 0.08]
    
    # Add thick horizontal line under title
    line_y = 0.90
    ax.plot([0.08, 0.92], [line_y, line_y], color='black', linewidth=2.5, transform=fig.transFigure)
    
    # ============================================================================
    # VERTICAL SPACING CALCULATION
    # ============================================================================
    # y=1.0 is top, y=0.0 is bottom
    # 
    x_start = 0.10
    y_header = 0.88        # Header row position (moved WAY down)
    y_footer = 0.05        # Where footer starts
    
    # Calculate row height to fit all 32 rows
    available_space = y_header - y_footer - 0.08  # More padding for footer
    row_height = available_space / 33  # 32 rows + spacing buffer
    
    print(f"\n📐 Layout Calculation:")
    print(f"   - Available space: {available_space:.3f} units")
    print(f"   - Rows to fit: 32")
    print(f"   - Calculated row_height: {row_height:.4f}")
    print(f"   - Expected table height: {32 * row_height:.3f} units")
    print(f"   - Header at: y = {y_header:.3f}")
    print(f"   - Footer at: y = {y_footer:.3f}\n")
    
    # Draw headers with better styling
    x_pos = x_start
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        # Center the header in its column
        header_x = x_pos + width/2
        
        # Left-align Team and W-L headers, center the rest
        if header in ['Team', 'W-L']:
            align = 'left'
            header_x_pos = x_pos + 0.01
        else:
            align = 'center'
            header_x_pos = header_x
        
        ax.text(header_x_pos, y_header, header,
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8E8E8', edgecolor='none', alpha=0.6),
                ha=align,
                va='center',
                color='#333333')
        x_pos += width
    
    # Add line under headers
    header_line_y = y_header - row_height/2
    ax.plot([x_start - 0.02, x_start + sum(col_widths) + 0.02], 
           [header_line_y, header_line_y], 
           color='black', linewidth=1.2, alpha=0.6, zorder=0)
    
    # Draw rows - start below header line
    y_pos = y_header - row_height * 1.5 # This controls where the column headers go
    # y_pos = 1
    
    print(f"   - First row starts at: y = {y_pos:.3f}")
    
    for idx, (_, row) in enumerate(df_display.iterrows()):
        # Alternate row background (light gray)
        if idx % 2 == 1:
            rect = mpatches.Rectangle((x_start - 0.01, y_pos - row_height/2), 
                                     0.88, row_height,
                                     facecolor='#f0f0f0', edgecolor='none', zorder=0, alpha=0.3)
            ax.add_patch(rect)
        
        # Rank
        ax.text(x_start + col_widths[0]/2, y_pos, str(idx + 1),
                fontsize=11, fontweight='bold',
                ha='center', va='center')
        
        # Team name (left aligned)
        team_x = x_start + col_widths[0]
        ax.text(team_x + 0.01, y_pos, row['team'],
                fontsize=10, ha='left', va='center',
                fontweight='600')
        
        # Logo (to the RIGHT of team name) - consistent size for all
        logo_url = logo_map.get(row['team'])
        if logo_url:
            # download_logo now returns fixed-size logos
            logo_img = download_logo(logo_url)
            if logo_img:
                # Position logo at the end of the team column
                logo_x = x_start + sum(col_widths[:2]) - 0.020
                ab = AnnotationBbox(logo_img, (logo_x, y_pos),
                                   frameon=False, zorder=10)
                ax.add_artist(ab)
        
        # W-L Record
        x_pos = x_start + sum(col_widths[:2])
        ax.text(x_pos + col_widths[2]/2, y_pos, row['record'],
                fontsize=9, ha='center', va='center', color='black',
                fontfamily='monospace')
        
        # Avg Odds - BLACK text
        x_pos = x_start + sum(col_widths[:3])
        ax.text(x_pos + col_widths[3]/2, y_pos, row['avg_odds_str'],
                fontsize=10, ha='center', va='center', color='black',
                fontfamily='monospace', fontweight='600')
        
        # Implied % (with vig) - BLACK text - based on AVG odds
        x_pos = x_start + sum(col_widths[:4])
        implied_display = '0%' if not row['has_odds'] else f"{row['implied_pct']}%"
        ax.text(x_pos + col_widths[4]/2, y_pos, implied_display,
                fontsize=9, ha='center', va='center', color='black',
                fontfamily='monospace')
        
        # Fair Odds - BLACK text, no background
        x_pos = x_start + sum(col_widths[:5])
        ax.text(x_pos + col_widths[5]/2, y_pos, row['fair_odds_str'],
                fontsize=10, ha='center', va='center', color='black',
                fontfamily='monospace', fontweight='bold')
        
        # Fair % - BLACK text, no background
        x_pos = x_start + sum(col_widths[:6])
        ax.text(x_pos + col_widths[6]/2, y_pos, f"{row['fair_pct_str']}%",
                fontsize=9, ha='center', va='center', color='black',
                fontfamily='monospace', fontweight='600')
        
        # Vig % (Average Vig Delta) - with red-to-green gradient background
        x_pos = x_start + sum(col_widths[:7])
        
        if pd.isna(row['vig_diff']) or row['vig_diff'] is None:
            # No odds - display '-'
            ax.text(x_pos + col_widths[8]/2, y_pos, '-',
                    fontsize=9, ha='center', va='center', color='black',
                    fontfamily='monospace', fontweight='bold')
        else:
            # Has odds - display with color gradient
            vig_diff = row['vig_diff']
            
            # Normalize to 0-1 range for colormap (-5 = 0 (green), +5 = 1 (red))
            # Flip it: higher vig = more red (worse for bettor)
            normalized_vig = np.clip((vig_diff + 5) / 10, 0, 1)
            cell_color = cmap(1 - normalized_vig)  # Invert so high vig = red
            
            color_rect = mpatches.Rectangle((x_pos, y_pos - row_height/2.2), 
                                           col_widths[7], row_height*0.9,
                                           facecolor=cell_color, edgecolor='none', zorder=1, alpha=0.7)
            ax.add_patch(color_rect)
            
            # Format vig with + sign for positive
            vig_str = f"+{vig_diff:.1f}%" if vig_diff > 0 else f"{vig_diff:.1f}%"
            ax.text(x_pos + col_widths[7]/2, y_pos, vig_str,
                    fontsize=9, ha='center', va='center', color='black',
                    fontfamily='monospace', fontweight='bold', zorder=2)
        
        # Best Book
        x_pos = x_start + sum(col_widths[:8])
        ax.text(x_pos + col_widths[8]/2, y_pos, row['best_book_display'],
                fontsize=9, ha='center', va='center', color='black')
        
        # Best Odds
        x_pos = x_start + sum(col_widths[:9])
        ax.text(x_pos + col_widths[9]/2, y_pos, row['best_odds_str'],
                fontsize=10, ha='center', va='center', color='black',
                fontfamily='monospace', fontweight='600')
        
        # Best Vig % - with red-to-green gradient background
        x_pos = x_start + sum(col_widths[:10])
        
        if pd.isna(row['best_vig_diff']) or row['best_vig_diff'] is None:
            # No odds - display '-'
            ax.text(x_pos + col_widths[10]/2, y_pos, '-',
                    fontsize=9, ha='center', va='center', color='black',
                    fontfamily='monospace', fontweight='bold')
        else:
            # Has odds - display with color gradient
            best_vig_diff = row['best_vig_diff']
            
            # Normalize to 0-1 range for colormap (-5 = 0 (green), +5 = 1 (red))
            normalized_vig = np.clip((best_vig_diff + 5) / 10, 0, 1)
            cell_color = cmap(1 - normalized_vig)  # Invert so high vig = red
            
            color_rect = mpatches.Rectangle((x_pos, y_pos - row_height/2.2), 
                                           col_widths[10], row_height*0.9,
                                           facecolor=cell_color, edgecolor='none', zorder=1, alpha=0.7)
            ax.add_patch(color_rect)
            
            # Format vig with + sign for positive
            best_vig_str = f"+{best_vig_diff:.1f}%" if best_vig_diff > 0 else f"{best_vig_diff:.1f}%"
            ax.text(x_pos + col_widths[10]/2, y_pos, best_vig_str,
                    fontsize=9, ha='center', va='center', color='black',
                    fontfamily='monospace', fontweight='bold', zorder=2)
        
        # Add horizontal grid line between rows (lighter, more subtle)
        line_y_pos = y_pos - row_height/2
        ax.plot([x_start - 0.02, x_start + sum(col_widths) + 0.02], 
               [line_y_pos, line_y_pos], 
               color='#CCCCCC', linewidth=0.8, alpha=0.5, zorder=0)
        
        y_pos -= row_height
    
    print(f"   - Last row ends at: y = {y_pos:.3f}")
    print(f"   - Actual table space used: {(y_header - row_height * 1.5) - y_pos:.3f} units\n")
    
    # Footer notes
    teams_with_odds = df_display['has_odds'].sum()
    footer_y = y_footer
    
    fig.text(0.12, footer_y, "Note: 'Implied %' includes bookmaker vig. 'Fair %' is the true probability with vig removed.",
             fontsize=9, style='italic')
    fig.text(0.12, footer_y - 0.025, f"{teams_with_odds} of 32 teams have odds available — bookmakers no longer offer odds on eliminated/longshot teams.",
             fontsize=8.5, style='italic', color='#666666')
    fig.text(0.12, footer_y - 0.050, "Data: The Odds API",
             fontsize=8, color='#888888')
    
    # plt.tight_layout(rect=[0, 0, 1, 1])
    
    return fig


def main():
    """Main visualization function"""
    
    print("="*80)
    print("NFL CHAMPIONSHIP FUTURES - 538 STYLE VISUALIZATION")
    print("="*80 + "\n")
    
    # Read the CSV
    csv_file = repo_root / 'data/04_output/nfl/nfl_championship_fair_odds.csv'
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Run analyze_nfl_championship_futures_vig.py first!")
        return
    
    print(f"📂 Reading: {csv_file.name}\n")
    df = pd.read_csv(csv_file)
    
    # Get team logos
    logo_map = get_team_logos()
    
    # Create 538-style visualization
    fig = create_538_table(df, logo_map)
    
    # Save as high-quality PNG (theme handles background color)
    png_output = repo_root / 'content/viz/nfl/futures_vig_matplotlib.png'
    fig.savefig(str(png_output), dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved PNG: {png_output}")
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n🖼️  Open: {png_output}")
    
    # Auto-open the PNG
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(png_output)])
        elif platform.system() == 'Windows':
            subprocess.run(['start', str(png_output)], shell=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(png_output)])
        print("📂 Opening PNG...")
    except Exception as e:
        print(f"⚠️  Could not auto-open: {e}")
        print(f"   Run: open {png_output}")


if __name__ == "__main__":
    main()

