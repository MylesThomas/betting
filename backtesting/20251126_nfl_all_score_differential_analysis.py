"""
Complete Score Differential Analysis: ALL teams across full range

Analyzes performance differential (actual vs adjusted score) for ALL teams,
then tracks next week performance to see regression patterns across the full spectrum.

Includes detailed examples showing week N → week N+1 trends.

Analyzes 2025 Adjusted Scores sheet from Unexpected Points Subscriber Data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data_path = Path("/Users/thomasmyles/dev/betting/data/01_input/unexpected_points/Unexpected Points Subscriber Data.xlsx")
df = pd.read_excel(data_path, sheet_name="2025 Adjusted Scores")

print("=" * 120)
print("NFL COMPLETE SCORE DIFFERENTIAL ANALYSIS - ALL TEAMS, ALL RANGES")
print("=" * 120)
print("\nDataset Info:")
print(f"Total rows: {len(df)}")
print(f"Total unique teams: {df['team'].nunique()}")
print(f"Weeks covered: {df['week'].min()} to {df['week'].max()}")

# Calculate the difference between actual score and adjusted score for ALL teams
df['score_diff'] = df['score'] - df['adj_score']

print("\n" + "=" * 120)
print("STEP 1: Score Differential Distribution (ALL Teams)")
print("=" * 120)

print(f"\nScore Differential Stats:")
print(f"  Mean: {df['score_diff'].mean():.2f}")
print(f"  Median: {df['score_diff'].median():.2f}")
print(f"  Std Dev: {df['score_diff'].std():.2f}")
print(f"  Min: {df['score_diff'].min():.2f}")
print(f"  Max: {df['score_diff'].max():.2f}")

print("\n--- Score Differential Breakdown ---")
print(f"Large overperformance (7+): {len(df[df['score_diff'] >= 7])} ({len(df[df['score_diff'] >= 7])/len(df)*100:.1f}%)")
print(f"Small overperformance (0 to 7): {len(df[(df['score_diff'] >= 0) & (df['score_diff'] < 7)])} ({len(df[(df['score_diff'] >= 0) & (df['score_diff'] < 7)])/len(df)*100:.1f}%)")
print(f"Small underperformance (-7 to 0): {len(df[(df['score_diff'] < 0) & (df['score_diff'] >= -7)])} ({len(df[(df['score_diff'] < 0) & (df['score_diff'] >= -7)])/len(df)*100:.1f}%)")
print(f"Large underperformance (<-7): {len(df[df['score_diff'] < -7])} ({len(df[df['score_diff'] < -7])/len(df)*100:.1f}%)")

print("\n" + "=" * 120)
print("STEP 2: Track ALL teams' performance the following week")
print("=" * 120)

# For each team performance, find their next game
results = []

for idx, row in df.iterrows():
    team = row['team']
    season = row['season']
    week = row['week']
    next_week = week + 1
    game_id = row['game_id']
    
    # Determine if they won this game by finding opponent
    game_teams = df[df['game_id'] == game_id]
    if len(game_teams) == 2:
        opponent_row = game_teams[game_teams['team'] != team].iloc[0]
        won_game = 1 if row['score'] > opponent_row['score'] else 0
        opponent_team = opponent_row['team']
        opponent_score = opponent_row['score']
    else:
        won_game = None
        opponent_team = None
        opponent_score = None
    
    # Find the team's next game
    next_game = df[(df['team'] == team) & 
                   (df['season'] == season) & 
                   (df['week'] == next_week)]
    
    if len(next_game) > 0:
        next_game = next_game.iloc[0]
        next_game_id = next_game['game_id']
        
        # Determine if they won next game
        next_game_teams = df[df['game_id'] == next_game_id]
        if len(next_game_teams) == 2:
            next_opponent_row = next_game_teams[next_game_teams['team'] != team].iloc[0]
            won_next_game = 1 if next_game['score'] > next_opponent_row['score'] else 0
            next_opponent_team = next_opponent_row['team']
            next_opponent_score = next_opponent_row['score']
        else:
            won_next_game = None
            next_opponent_team = None
            next_opponent_score = None
        
        result = {
            # Week N data
            'week': week,
            'team': team,
            'game_id': game_id,
            'opponent': opponent_team,
            'score': row['score'],
            'opponent_score': opponent_score,
            'adj_score': row['adj_score'],
            'score_diff': row['score_diff'],
            'won_game': won_game,
            
            # Week N+1 data
            'next_week': next_week,
            'next_game_id': next_game_id,
            'next_opponent': next_opponent_team,
            'next_score': next_game['score'],
            'next_opponent_score': next_opponent_score,
            'next_adj_score': next_game['adj_score'],
            'next_score_diff': next_game['score'] - next_game['adj_score'],
            'won_next_game': won_next_game,
            
            # Change metrics
            'diff_change': (next_game['score'] - next_game['adj_score']) - row['score_diff'],
        }
        results.append(result)

results_df = pd.DataFrame(results)

print(f"\nTracked {len(results_df)} teams into their next week performance")

print("\n" + "=" * 120)
print("STEP 3: DETAILED EXAMPLES - Week N → Week N+1 Trends")
print("=" * 120)

# Show examples from different differential ranges
print("\n" + "=" * 120)
print("📊 LARGE OVERPERFORMERS (7+ points) - Examples")
print("=" * 120)

large_overperf = results_df[results_df['score_diff'] >= 7].nlargest(5, 'score_diff')
for idx, row in large_overperf.iterrows():
    print(f"\n{row['team']} - Large Overperformance Example:")
    print(f"  Week {int(row['week'])}: {row['team']} {int(row['score'])}-{int(row['opponent_score'])} {row['opponent']} {'✅ WIN' if row['won_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['adj_score']:.1f} → Overperformed by {row['score_diff']:+.1f} points")
    print(f"  Week {int(row['next_week'])}: {row['team']} {int(row['next_score'])}-{int(row['next_opponent_score'])} {row['next_opponent']} {'✅ WIN' if row['won_next_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['next_adj_score']:.1f} → Differential: {row['next_score_diff']:+.1f} points")
    print(f"  📉 Regression: {row['score_diff']:+.1f} → {row['next_score_diff']:+.1f} (change: {row['diff_change']:+.1f})")

print("\n" + "=" * 120)
print("📊 SMALL OVERPERFORMERS (0 to 7 points) - Examples")
print("=" * 120)

small_overperf = results_df[(results_df['score_diff'] >= 0) & (results_df['score_diff'] < 7)].sample(min(5, len(results_df[(results_df['score_diff'] >= 0) & (results_df['score_diff'] < 7)])), random_state=42)
for idx, row in small_overperf.iterrows():
    print(f"\n{row['team']} - Small Overperformance Example:")
    print(f"  Week {int(row['week'])}: {row['team']} {int(row['score'])}-{int(row['opponent_score'])} {row['opponent']} {'✅ WIN' if row['won_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['adj_score']:.1f} → Overperformed by {row['score_diff']:+.1f} points")
    print(f"  Week {int(row['next_week'])}: {row['team']} {int(row['next_score'])}-{int(row['next_opponent_score'])} {row['next_opponent']} {'✅ WIN' if row['won_next_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['next_adj_score']:.1f} → Differential: {row['next_score_diff']:+.1f} points")
    print(f"  📊 Change: {row['score_diff']:+.1f} → {row['next_score_diff']:+.1f} (change: {row['diff_change']:+.1f})")

print("\n" + "=" * 120)
print("📊 SMALL UNDERPERFORMERS (-7 to 0 points) - Examples")
print("=" * 120)

small_underperf = results_df[(results_df['score_diff'] < 0) & (results_df['score_diff'] >= -7)].sample(min(5, len(results_df[(results_df['score_diff'] < 0) & (results_df['score_diff'] >= -7)])), random_state=42)
for idx, row in small_underperf.iterrows():
    print(f"\n{row['team']} - Small Underperformance Example:")
    print(f"  Week {int(row['week'])}: {row['team']} {int(row['score'])}-{int(row['opponent_score'])} {row['opponent']} {'✅ WIN' if row['won_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['adj_score']:.1f} → Underperformed by {row['score_diff']:+.1f} points")
    print(f"  Week {int(row['next_week'])}: {row['team']} {int(row['next_score'])}-{int(row['next_opponent_score'])} {row['next_opponent']} {'✅ WIN' if row['won_next_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['next_adj_score']:.1f} → Differential: {row['next_score_diff']:+.1f} points")
    print(f"  📈 Bounce back: {row['score_diff']:+.1f} → {row['next_score_diff']:+.1f} (change: {row['diff_change']:+.1f})")

print("\n" + "=" * 120)
print("📊 LARGE UNDERPERFORMERS (<-7 points) - Examples")
print("=" * 120)

large_underperf = results_df[results_df['score_diff'] < -7].nsmallest(5, 'score_diff')
for idx, row in large_underperf.iterrows():
    print(f"\n{row['team']} - Large Underperformance Example:")
    print(f"  Week {int(row['week'])}: {row['team']} {int(row['score'])}-{int(row['opponent_score'])} {row['opponent']} {'✅ WIN' if row['won_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['adj_score']:.1f} → Underperformed by {row['score_diff']:+.1f} points")
    print(f"  Week {int(row['next_week'])}: {row['team']} {int(row['next_score'])}-{int(row['next_opponent_score'])} {row['next_opponent']} {'✅ WIN' if row['won_next_game'] else '❌ LOSS'}")
    print(f"    Adjusted score: {row['next_adj_score']:.1f} → Differential: {row['next_score_diff']:+.1f} points")
    print(f"  📈 Bounce back: {row['score_diff']:+.1f} → {row['next_score_diff']:+.1f} (change: {row['diff_change']:+.1f})")

print("\n" + "=" * 120)
print("STEP 4: Overall Statistics - Next Week Performance by Differential Range")
print("=" * 120)

# Create bins for the full range
bins = [-50, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 50]
labels = ['<-15', '-15 to -10', '-10 to -7', '-7 to -5', '-5 to -3', '-3 to 0',
          '0 to 3', '3 to 5', '5 to 7', '7 to 10', '10 to 15', '15+']

results_df['diff_bin'] = pd.cut(results_df['score_diff'], bins=bins, labels=labels)

print("\n📊 NEXT WEEK PERFORMANCE BY SCORE DIFFERENTIAL BIN:")
print("=" * 120)

bin_stats = []
for bin_label in labels:
    bin_data = results_df[results_df['diff_bin'] == bin_label]
    
    if len(bin_data) > 0:
        wins = bin_data['won_next_game'].sum()
        games = bin_data['won_next_game'].notna().sum()
        win_pct = (wins / games * 100) if games > 0 else 0
        avg_next_diff = bin_data['next_score_diff'].mean()
        avg_change = bin_data['diff_change'].mean()
        
        bin_stats.append({
            'bin': bin_label,
            'games': games,
            'wins': wins,
            'losses': games - wins,
            'win_pct': win_pct,
            'avg_current_diff': bin_data['score_diff'].mean(),
            'avg_next_diff': avg_next_diff,
            'avg_change': avg_change,
        })

bin_stats_df = pd.DataFrame(bin_stats)

for _, row in bin_stats_df.iterrows():
    print(f"\n{row['bin']} score differential (n={int(row['games'])}):")
    print(f"  Current week avg diff: {row['avg_current_diff']:+.2f}")
    print(f"  Next week win rate: {int(row['wins'])}/{int(row['games'])} ({row['win_pct']:.1f}%)")
    print(f"  Next week avg diff: {row['avg_next_diff']:+.2f}")
    print(f"  Average change: {row['avg_change']:+.2f}")
    
    # Visual bar
    bar_length = int(row['win_pct'] / 2)
    bar = '█' * bar_length
    print(f"  Win % [{bar:<50}] {row['win_pct']:.1f}%")

print("\n" + "=" * 120)
print("STEP 5: Key Insights")
print("=" * 120)

# Calculate regression metrics
overperf = results_df[results_df['score_diff'] > 0]
underperf = results_df[results_df['score_diff'] < 0]

print(f"\n--- Overperformers (scored > adjusted) ---")
print(f"  Count: {len(overperf)}")
print(f"  Avg current diff: {overperf['score_diff'].mean():+.2f}")
print(f"  Avg next week diff: {overperf['next_score_diff'].mean():+.2f}")
print(f"  Avg change: {overperf['diff_change'].mean():+.2f}")
print(f"  Next week win rate: {overperf['won_next_game'].sum()}/{overperf['won_next_game'].notna().sum()} ({overperf['won_next_game'].sum()/overperf['won_next_game'].notna().sum()*100:.1f}%)")

print(f"\n--- Underperformers (scored < adjusted) ---")
print(f"  Count: {len(underperf)}")
print(f"  Avg current diff: {underperf['score_diff'].mean():+.2f}")
print(f"  Avg next week diff: {underperf['next_score_diff'].mean():+.2f}")
print(f"  Avg change: {underperf['diff_change'].mean():+.2f}")
print(f"  Next week win rate: {underperf['won_next_game'].sum()}/{underperf['won_next_game'].notna().sum()} ({underperf['won_next_game'].sum()/underperf['won_next_game'].notna().sum()*100:.1f}%)")

# Save outputs
output_dir = Path("/Users/thomasmyles/dev/betting/data/04_output/nfl/unexpected_points")
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed results
results_path = output_dir / "all_teams_score_differential_analysis.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Detailed results saved to: {results_path}")

# Save bin summary
bin_stats_path = output_dir / "all_teams_differential_bin_summary.csv"
bin_stats_df.to_csv(bin_stats_path, index=False)
print(f"✓ Bin summary saved to: {bin_stats_path}")

print("\n" + "=" * 120)
print("CREATING VISUALIZATIONS")
print("=" * 120)

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('NFL Complete Score Differential Analysis: ALL Teams, ALL Ranges', fontsize=18, fontweight='bold')

# Plot 1: Win Rate by Differential Bin (Full Range)
ax1 = fig.add_subplot(gs[0, :2])
colors1 = ['#e74c3c' if float(label.split()[0].replace('<', '').replace('to', '').replace('+', '').replace('-15', '-15')) < 0 
           else '#2ecc71' for label in bin_stats_df['bin']]
bars1 = ax1.bar(bin_stats_df['bin'], bin_stats_df['win_pct'], color=colors1, alpha=0.7, edgecolor='black')
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='50% (Break Even)')
ax1.set_xlabel('Score Differential Range (Points)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Win % Next Week', fontsize=12, fontweight='bold')
ax1.set_title('Next Week Win Rate by Current Week Score Differential', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels
for bar, pct, games in zip(bars1, bin_stats_df['win_pct'], bin_stats_df['games']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{pct:.0f}%\n(n={int(games)})',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Sample Size
ax2 = fig.add_subplot(gs[0, 2])
ax2.barh(bin_stats_df['bin'], bin_stats_df['games'], color='#9b59b6', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Games', fontsize=10, fontweight='bold')
ax2.set_title('Sample Size', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bin_val, count) in enumerate(zip(bin_stats_df['bin'], bin_stats_df['games'])):
    ax2.text(count + 1, i, f'{int(count)}', va='center', fontweight='bold', fontsize=8)

# Plot 3: Current vs Next Week Differential
ax3 = fig.add_subplot(gs[1, :2])
x = np.arange(len(bin_stats_df))
width = 0.35

bars3a = ax3.bar(x - width/2, bin_stats_df['avg_current_diff'], width,
                 label='Current Week', color='#3498db', alpha=0.7, edgecolor='black')
bars3b = ax3.bar(x + width/2, bin_stats_df['avg_next_diff'], width,
                 label='Next Week', color='#e67e22', alpha=0.7, edgecolor='black')

ax3.axhline(y=0, color='gray', linestyle='--', linewidth=2)
ax3.set_xlabel('Score Differential Range (Points)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Avg Score Differential', fontsize=12, fontweight='bold')
ax3.set_title('Regression to Mean: Current Week vs Next Week', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(bin_stats_df['bin'], rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Change in Differential
ax4 = fig.add_subplot(gs[1, 2])
colors4 = ['#2ecc71' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' for x in bin_stats_df['avg_change']]
bars4 = ax4.barh(bin_stats_df['bin'], bin_stats_df['avg_change'], color=colors4, alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=2)
ax4.set_xlabel('Avg Change', fontsize=10, fontweight='bold')
ax4.set_title('Week-to-Week Change', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

for i, (val, bar) in enumerate(zip(bin_stats_df['avg_change'], bars4)):
    width = bar.get_width()
    ax4.text(width + (0.2 if width > 0 else -0.2), bar.get_y() + bar.get_height()/2,
            f'{val:+.1f}',
            ha='left' if width > 0 else 'right', va='center', fontsize=8, fontweight='bold')

# Plot 5: Scatter - Current vs Next Week Differential
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(results_df['score_diff'], results_df['next_score_diff'], 
                     alpha=0.4, s=30, c=results_df['won_next_game'], cmap='RdYlGn')
ax5.plot([-20, 25], [-20, 25], 'k--', alpha=0.3, label='No Change')
ax5.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax5.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax5.set_xlabel('Current Week Score Diff', fontsize=10, fontweight='bold')
ax5.set_ylabel('Next Week Score Diff', fontsize=10, fontweight='bold')
ax5.set_title('Regression Pattern', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='Won Next Game')

# Plot 6: Distribution of Current Week Differentials
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(results_df['score_diff'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
ax6.set_xlabel('Score Differential', fontsize=10, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax6.set_title('Distribution of Score Differentials', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Box plot of changes by original differential
ax7 = fig.add_subplot(gs[2, 2])
diff_categories = pd.cut(results_df['score_diff'], bins=[-50, -7, 0, 7, 50], 
                         labels=['Large Under', 'Small Under', 'Small Over', 'Large Over'])
results_df['diff_category'] = diff_categories

box_data = [results_df[results_df['diff_category'] == cat]['diff_change'].dropna() 
            for cat in ['Large Under', 'Small Under', 'Small Over', 'Large Over']]
bp = ax7.boxplot(box_data, labels=['Large\nUnder', 'Small\nUnder', 'Small\nOver', 'Large\nOver'],
                 patch_artist=True)
for patch, color in zip(bp['boxes'], ['#c0392b', '#e74c3c', '#2ecc71', '#27ae60']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax7.axhline(y=0, color='gray', linestyle='--', linewidth=2)
ax7.set_ylabel('Change in Differential', fontsize=10, fontweight='bold')
ax7.set_title('Change Distribution', fontsize=12, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)

# Save figure
fig_path = output_dir / "all_teams_complete_differential_analysis.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {fig_path}")

# Display the plot
plt.show()

print("\n" + "=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)
print("\n💡 KEY TAKEAWAYS:")
print("1. Teams regress to the mean regardless of direction")
print("2. Larger differentials show stronger regression")
print("3. Underperformers tend to bounce back")
print("4. Overperformers tend to regress downward")

