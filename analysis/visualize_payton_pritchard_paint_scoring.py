"""
Create matplotlib visualization for Payton Pritchard paint scoring analysis

Simpler alternative to R/gt visualization - creates a clean, publication-ready 
horizontal bar chart showing paint FG% comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Configuration
repo_root = Path(__file__).parent.parent
csv_file = repo_root / 'data/04_output/nba/payton_pritchard_paint_scoring_2025_26.csv'
output_dir = repo_root / 'content/viz/nba'
output_file = output_dir / 'payton_pritchard_paint_scoring_2025_26.png'

# Load data
df = pd.read_csv(csv_file)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# Prepare data
y_positions = np.arange(len(df))
colors = ['#FFD700' if player == 'Payton Pritchard' else '#3498db' for player in df['player']]

# Create horizontal bars
bars = ax.barh(y_positions, df['paint_fg_pct'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Customize player labels with height
labels = [f"{row['player']} ({row['height']})" for _, row in df.iterrows()]
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=11, fontweight='600')

# Highlight Pritchard row
pritchard_idx = df[df['player'] == 'Payton Pritchard'].index[0]
ax.get_yticklabels()[pritchard_idx].set_fontsize(13)
ax.get_yticklabels()[pritchard_idx].set_color('#FF6B35')

# Add value labels on bars
for i, (idx, row) in enumerate(df.iterrows()):
    percentage = row['paint_fg_pct']
    makes = int(row['paint_fgm'])
    attempts = int(row['paint_fga'])
    
    # Position label inside bar if long enough, otherwise outside
    if percentage > 15:
        x_pos = percentage - 2
        ha = 'right'
        color = 'white'
    else:
        x_pos = percentage + 1
        ha = 'left'
        color = 'black'
    
    ax.text(x_pos, i, f'{percentage}% ({makes}/{attempts})', 
            va='center', ha=ha, fontsize=10, fontweight='bold', color=color)

# Add rank numbers
for i, (idx, row) in enumerate(df.iterrows()):
    rank = row['rank']
    ax.text(-3, i, f'#{int(rank)}', va='center', ha='right', 
            fontsize=10, fontweight='bold', color='#2c3e50')

# Formatting
ax.set_xlabel('Paint FG% (shots ≤6 feet)', fontsize=13, fontweight='bold')
ax.set_xlim(-5, max(df['paint_fg_pct']) + 8)
ax.set_ylim(-0.5, len(df) - 0.5)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Title
title_text = "Payton Pritchard is ELITE at Scoring in the Paint for a 6'1\" Guard"
subtitle_text = "Paint FG% (shots within 6 feet) | 2025-26 NBA Season"

plt.text(0.5, 0.98, title_text, transform=fig.transFigure, 
         fontsize=18, fontweight='bold', ha='center', va='top')
plt.text(0.5, 0.95, subtitle_text, transform=fig.transFigure,
         fontsize=12, ha='center', va='top', style='italic', color='#555555')

# Footer
footer_text = "Paint = shots within 6 feet | Data: NBA API | Analysis: @mylesinthomas"
plt.text(0.5, 0.02, footer_text, transform=fig.transFigure,
         fontsize=9, ha='center', va='bottom', color='#666666')

# Legend
legend_elements = [
    mpatches.Patch(color='#FFD700', label='Payton Pritchard'),
    mpatches.Patch(color='#3498db', label='Other Guards')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# Save
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
print(f"✅ Saved visualization: {output_file}")

# Display
plt.show()

# Print summary stats
print("\n" + "="*80)
print("KEY STATS FOR YOUR POST:")
print("="*80)
pritchard = df[df['player'] == 'Payton Pritchard'].iloc[0]
print(f"\n🏀 Payton Pritchard (6'1\") is #{int(pritchard['rank'])} among 17 guards analyzed")
print(f"   Paint FG%: {pritchard['paint_fg_pct']}%")
print(f"   Paint Makes: {int(pritchard['paint_fgm'])}/{int(pritchard['paint_fga'])}")
print(f"   Paint PPG: {pritchard['paint_ppg']}")
print(f"   Paint Rate: {pritchard['paint_rate']}% of all shots")

small_guards = df[df['height'].isin(['6\'0"', '6\'1"', '6\'2"'])]
print(f"\n🎯 Among guards 6'2\" and under: #{(small_guards['paint_fg_pct'] >= pritchard['paint_fg_pct']).sum()} of {len(small_guards)}")

top3 = df.head(3)
print(f"\n⭐ Top 3 Paint Scorers:")
for _, row in top3.iterrows():
    print(f"   {int(row['rank'])}. {row['player']} ({row['height']}): {row['paint_fg_pct']}%")

print("\n" + "="*80)

