"""
Game-Level Luck Analysis: When one team gets lucky vs the other

For each game, calculates:
- Actual score differential (winner's perspective)
- Adjusted score differential (what "should have" happened)
- Luck delta = actual_diff - adj_diff

Then tracks both teams' performance next week to see if:
- Lucky team regresses (fade them)
- Unlucky team bounces back (back them)

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

print("=" * 100)
print("NFL GAME-LEVEL LUCK ANALYSIS")
print("=" * 100)
print("\nDataset Info:")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Calculate score differential for each team
df['score_diff'] = df['score'] - df['adj_score']

print("\n" + "=" * 100)
print("STEP 1: Pair teams by game and calculate luck delta")
print("=" * 100)

# Group by game to get both teams
games = []
for game_id in df['game_id'].unique():
    game_data = df[df['game_id'] == game_id].copy()
    
    if len(game_data) == 2:
        team1 = game_data.iloc[0]
        team2 = game_data.iloc[1]
        
        # Actual score differential (from team1 perspective)
        actual_diff = team1['score'] - team2['score']
        
        # Adjusted score differential (from team1 perspective)
        adj_diff = team1['adj_score'] - team2['adj_score']
        
        # Luck delta - how much did the actual outcome differ from expected
        luck_delta = actual_diff - adj_diff
        
        # Determine lucky vs unlucky team
        if luck_delta > 0:
            # Team1 got luckier than team2
            lucky_team = team1['team']
            lucky_team_data = team1
            unlucky_team = team2['team']
            unlucky_team_data = team2
        else:
            # Team2 got luckier than team1
            lucky_team = team2['team']
            lucky_team_data = team2
            unlucky_team = team1['team']
            unlucky_team_data = team1
            luck_delta = abs(luck_delta)
        
        game_info = {
            'season': team1['season'],
            'week': team1['week'],
            'game_id': game_id,
            
            # Team 1 (as listed in data)
            'team1': team1['team'],
            'team1_score': team1['score'],
            'team1_adj_score': team1['adj_score'],
            
            # Team 2
            'team2': team2['team'],
            'team2_score': team2['score'],
            'team2_adj_score': team2['adj_score'],
            
            # Game differentials
            'actual_diff': actual_diff,
            'adj_diff': adj_diff,
            'luck_delta': luck_delta,
            
            # Lucky/unlucky designation
            'lucky_team': lucky_team,
            'lucky_team_score': lucky_team_data['score'],
            'lucky_team_adj_score': lucky_team_data['adj_score'],
            'lucky_team_overperf': lucky_team_data['score'] - lucky_team_data['adj_score'],
            
            'unlucky_team': unlucky_team,
            'unlucky_team_score': unlucky_team_data['score'],
            'unlucky_team_adj_score': unlucky_team_data['adj_score'],
            'unlucky_team_underperf': unlucky_team_data['score'] - unlucky_team_data['adj_score'],
        }
        
        games.append(game_info)

games_df = pd.DataFrame(games)

print(f"\nAnalyzed {len(games_df)} games")
print("\nSample games with luck deltas:")
print(games_df[['week', 'team1', 'team2', 'actual_diff', 'adj_diff', 'luck_delta']].head(10))

print("\n--- Biggest Luck Deltas ---")
print(games_df.nlargest(10, 'luck_delta')[['week', 'lucky_team', 'unlucky_team', 
                                             'actual_diff', 'adj_diff', 'luck_delta']])

print("\n" + "=" * 100)
print("STEP 2: Track next week performance for both lucky and unlucky teams")
print("=" * 100)

# For each game, track both teams next week
results = []

for idx, game in games_df.iterrows():
    season = game['season']
    week = game['week']
    next_week = week + 1
    
    # Track lucky team next week
    lucky_next = df[(df['team'] == game['lucky_team']) & 
                    (df['season'] == season) & 
                    (df['week'] == next_week)]
    
    # Track unlucky team next week
    unlucky_next = df[(df['team'] == game['unlucky_team']) & 
                      (df['season'] == season) & 
                      (df['week'] == next_week)]
    
    if len(lucky_next) > 0 and len(unlucky_next) > 0:
        lucky_next = lucky_next.iloc[0]
        unlucky_next = unlucky_next.iloc[0]
        
        # Get opponent scores to determine wins
        lucky_next_game = df[df['game_id'] == lucky_next['game_id']]
        unlucky_next_game = df[df['game_id'] == unlucky_next['game_id']]
        
        lucky_won = None
        unlucky_won = None
        
        if len(lucky_next_game) == 2:
            lucky_opp = lucky_next_game[lucky_next_game['team'] != game['lucky_team']].iloc[0]
            lucky_won = 1 if lucky_next['score'] > lucky_opp['score'] else 0
        
        if len(unlucky_next_game) == 2:
            unlucky_opp = unlucky_next_game[unlucky_next_game['team'] != game['unlucky_team']].iloc[0]
            unlucky_won = 1 if unlucky_next['score'] > unlucky_opp['score'] else 0
        
        result = {
            'original_week': week,
            'luck_delta': game['luck_delta'],
            
            'lucky_team': game['lucky_team'],
            'lucky_team_overperf': game['lucky_team_overperf'],
            'lucky_next_score': lucky_next['score'],
            'lucky_next_adj_score': lucky_next['adj_score'],
            'lucky_next_diff': lucky_next['score'] - lucky_next['adj_score'],
            'lucky_won_next': lucky_won,
            
            'unlucky_team': game['unlucky_team'],
            'unlucky_team_underperf': game['unlucky_team_underperf'],
            'unlucky_next_score': unlucky_next['score'],
            'unlucky_next_adj_score': unlucky_next['adj_score'],
            'unlucky_next_diff': unlucky_next['score'] - unlucky_next['adj_score'],
            'unlucky_won_next': unlucky_won,
        }
        
        results.append(result)

results_df = pd.DataFrame(results)

print(f"\nTracked {len(results_df)} game pairs into next week")
print("\nSample results:")
print(results_df[['luck_delta', 'lucky_team', 'lucky_won_next', 'unlucky_team', 'unlucky_won_next']].head(15))

print("\n" + "=" * 100)
print("STEP 3: Overall Statistics - Lucky vs Unlucky Teams Next Week")
print("=" * 100)

# Lucky teams next week
lucky_wins = results_df['lucky_won_next'].sum()
lucky_games = results_df['lucky_won_next'].notna().sum()
lucky_win_pct = (lucky_wins / lucky_games * 100) if lucky_games > 0 else 0

# Unlucky teams next week
unlucky_wins = results_df['unlucky_won_next'].sum()
unlucky_games = results_df['unlucky_won_next'].notna().sum()
unlucky_win_pct = (unlucky_wins / unlucky_games * 100) if unlucky_games > 0 else 0

print(f"\n--- NEXT WEEK WIN RATES ---")
print(f"Lucky teams: {lucky_wins}/{lucky_games} ({lucky_win_pct:.1f}%)")
print(f"Unlucky teams: {unlucky_wins}/{unlucky_games} ({unlucky_win_pct:.1f}%)")
print(f"Difference: {unlucky_win_pct - lucky_win_pct:+.1f} percentage points")

print(f"\n--- NEXT WEEK SCORE DIFFERENTIALS ---")
print(f"Lucky teams avg diff: {results_df['lucky_next_diff'].mean():.2f}")
print(f"Unlucky teams avg diff: {results_df['unlucky_next_diff'].mean():.2f}")
print(f"Difference: {results_df['unlucky_next_diff'].mean() - results_df['lucky_next_diff'].mean():+.2f}")

print("\n" + "=" * 100)
print("STEP 4: Binned Analysis by Luck Delta Magnitude")
print("=" * 100)

# Create bins for luck delta across full range
bins = [-50, -15, -10, -7, -3, 0, 3, 7, 10, 15, 50]
labels = ['-15+', '-15 to -10', '-10 to -7', '-7 to -3', '-3 to 0', 
          '0 to 3', '3 to 7', '7 to 10', '10 to 15', '15+']

results_df['luck_bin'] = pd.cut(results_df['luck_delta'], bins=bins, labels=labels)

print("\n📊 NEXT WEEK PERFORMANCE BY LUCK DELTA:")
print("=" * 100)

bin_stats = []
for bin_label in labels:
    bin_data = results_df[results_df['luck_bin'] == bin_label]
    
    if len(bin_data) > 0:
        # Lucky team stats
        lucky_wins_bin = bin_data['lucky_won_next'].sum()
        lucky_games_bin = bin_data['lucky_won_next'].notna().sum()
        lucky_win_pct_bin = (lucky_wins_bin / lucky_games_bin * 100) if lucky_games_bin > 0 else 0
        
        # Unlucky team stats
        unlucky_wins_bin = bin_data['unlucky_won_next'].sum()
        unlucky_games_bin = bin_data['unlucky_won_next'].notna().sum()
        unlucky_win_pct_bin = (unlucky_wins_bin / unlucky_games_bin * 100) if unlucky_games_bin > 0 else 0
        
        bin_stats.append({
            'luck_bin': bin_label,
            'games': len(bin_data),
            'lucky_wins': lucky_wins_bin,
            'lucky_games': lucky_games_bin,
            'lucky_win_pct': lucky_win_pct_bin,
            'lucky_avg_next_diff': bin_data['lucky_next_diff'].mean(),
            'unlucky_wins': unlucky_wins_bin,
            'unlucky_games': unlucky_games_bin,
            'unlucky_win_pct': unlucky_win_pct_bin,
            'unlucky_avg_next_diff': bin_data['unlucky_next_diff'].mean(),
        })

bin_stats_df = pd.DataFrame(bin_stats)

for _, row in bin_stats_df.iterrows():
    print(f"\nLuck Delta: {row['luck_bin']} (n={int(row['games'])})")
    print(f"  Lucky team next week:   {int(row['lucky_wins'])}/{int(row['lucky_games'])} wins ({row['lucky_win_pct']:.1f}%), avg diff: {row['lucky_avg_next_diff']:+.2f}")
    print(f"  Unlucky team next week: {int(row['unlucky_wins'])}/{int(row['unlucky_games'])} wins ({row['unlucky_win_pct']:.1f}%), avg diff: {row['unlucky_avg_next_diff']:+.2f}")
    print(f"  Win % edge (unlucky - lucky): {row['unlucky_win_pct'] - row['lucky_win_pct']:+.1f} percentage points")

# Save outputs
output_dir = Path("/Users/thomasmyles/dev/betting/data/04_output/nfl/unexpected_points")
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed results
results_path = output_dir / "game_level_luck_analysis.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Detailed results saved to: {results_path}")

# Save bin summary
bin_stats_path = output_dir / "luck_delta_bin_summary.csv"
bin_stats_df.to_csv(bin_stats_path, index=False)
print(f"✓ Bin summary saved to: {bin_stats_path}")

# Save original game data
games_path = output_dir / "all_games_with_luck_delta.csv"
games_df.to_csv(games_path, index=False)
print(f"✓ All games saved to: {games_path}")

print("\n" + "=" * 100)
print("CREATING VISUALIZATIONS")
print("=" * 100)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('NFL Game-Level Luck Analysis: Next Week Performance', fontsize=16, fontweight='bold')

# Plot 1: Win Rate Comparison by Luck Delta
ax1 = axes[0, 0]
x = np.arange(len(bin_stats_df))
width = 0.35

bars1 = ax1.bar(x - width/2, bin_stats_df['lucky_win_pct'], width, 
                label='Lucky Team', color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, bin_stats_df['unlucky_win_pct'], width,
                label='Unlucky Team', color='#2ecc71', alpha=0.7, edgecolor='black')

ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Luck Delta (Points)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Next Week Win %', fontsize=12, fontweight='bold')
ax1.set_title('Next Week Win Rate: Lucky vs Unlucky Teams', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(bin_stats_df['luck_bin'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 100)

# Plot 2: Score Differential Next Week
ax2 = axes[0, 1]
bars3 = ax2.bar(x - width/2, bin_stats_df['lucky_avg_next_diff'], width,
                label='Lucky Team', color='#e74c3c', alpha=0.7, edgecolor='black')
bars4 = ax2.bar(x + width/2, bin_stats_df['unlucky_avg_next_diff'], width,
                label='Unlucky Team', color='#2ecc71', alpha=0.7, edgecolor='black')

ax2.axhline(y=0, color='gray', linestyle='--', linewidth=2)
ax2.set_xlabel('Luck Delta (Points)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Avg Score Differential', fontsize=12, fontweight='bold')
ax2.set_title('Next Week Avg Score Diff: Lucky vs Unlucky Teams', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(bin_stats_df['luck_bin'], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Win % Edge (Unlucky - Lucky)
ax3 = axes[1, 0]
edge = bin_stats_df['unlucky_win_pct'] - bin_stats_df['lucky_win_pct']
colors3 = ['#2ecc71' if x > 0 else '#e74c3c' for x in edge]
bars5 = ax3.bar(bin_stats_df['luck_bin'], edge, color=colors3, alpha=0.7, edgecolor='black')

ax3.axhline(y=0, color='gray', linestyle='--', linewidth=2)
ax3.set_xlabel('Luck Delta (Points)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Win % Edge (Unlucky - Lucky)', fontsize=12, fontweight='bold')
ax3.set_title('Betting Edge: Unlucky Team Win % Advantage', fontsize=13, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars5, edge):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
            f'{val:+.1f}',
            ha='center', va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)

# Plot 4: Sample Size
ax4 = axes[1, 1]
ax4.bar(bin_stats_df['luck_bin'], bin_stats_df['games'], color='#9b59b6', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Luck Delta (Points)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
ax4.set_title('Sample Size by Luck Delta', fontsize=13, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

for i, (bin_val, count) in enumerate(zip(bin_stats_df['luck_bin'], bin_stats_df['games'])):
    ax4.text(i, count + 0.5, f'{int(count)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save figure
fig_path = output_dir / "game_level_luck_visualization.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {fig_path}")

# Display the plot
plt.show()

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print("\n💡 KEY TAKEAWAY:")
print("Fade teams that got lucky (actual result >> expected result)")
print("Back teams that got unlucky (actual result << expected result)")

