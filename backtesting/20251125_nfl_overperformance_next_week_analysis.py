"""
Analysis: Teams that overperformed by 7+ points (actual vs adjusted score)
and their performance the following week

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

print("=" * 80)
print("NFL OVERPERFORMANCE ANALYSIS")
print("=" * 80)
print("\nDataset Info:")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head(10))

# Calculate the difference between actual score and adjusted score
# Looking at the columns, it seems we have 'score' and 'adj_score'
df['score_diff'] = df['score'] - df['adj_score']

print("\n" + "=" * 80)
print("STEP 1: Find games where actual score exceeds adjusted score by 7+ points")
print("=" * 80)

# Filter for teams that overperformed by 7+ points
overperformers = df[df['score_diff'] >= 7].copy()
print(f"\nFound {len(overperformers)} instances where a team overperformed by 7+ points")
print("\nTop overperformances:")
print(overperformers.nlargest(10, 'score_diff')[['season', 'week', 'game_id', 'team', 'score', 'adj_score', 'score_diff']])

print("\n" + "=" * 80)
print("STEP 2: Track these teams' performance the following week")
print("=" * 80)

# For each overperformer, find their next game
results = []

for idx, row in overperformers.iterrows():
    team = row['team']
    season = row['season']
    week = row['week']
    next_week = week + 1
    game_id = row['game_id']
    
    # Determine if they won the overperformance game by finding opponent
    game_teams = df[df['game_id'] == game_id]
    if len(game_teams) == 2:
        opponent_row = game_teams[game_teams['team'] != team].iloc[0]
        won_overperf_game = 1 if row['score'] > opponent_row['score'] else 0
    else:
        won_overperf_game = None
    
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
        else:
            won_next_game = None
        
        result = {
            'overperf_week': week,
            'overperf_team': team,
            'overperf_game_id': game_id,
            'overperf_score': row['score'],
            'overperf_adj_score': row['adj_score'],
            'overperf_diff': row['score_diff'],
            'won_overperf_game': won_overperf_game,
            'next_week': next_week,
            'next_game_id': next_game_id,
            'next_score': next_game['score'],
            'next_adj_score': next_game['adj_score'],
            'next_score_diff': next_game['score'] - next_game['adj_score'],
            'won_next_game': won_next_game
        }
        results.append(result)

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(f"\nTracked {len(results_df)} teams into their next week performance")
    print("\nSample of follow-up performances:")
    print(results_df.head(15))
    
    print("\n" + "=" * 80)
    print("STEP 3: Summary Statistics for Next Week Performance")
    print("=" * 80)
    
    # Win rates
    print("\n--- WIN RATES ---")
    overperf_wins = results_df['won_overperf_game'].sum()
    overperf_total = results_df['won_overperf_game'].notna().sum()
    next_wins = results_df['won_next_game'].sum()
    next_total = results_df['won_next_game'].notna().sum()
    
    print(f"Win rate in overperformance game: {overperf_wins}/{overperf_total} ({overperf_wins/overperf_total*100:.1f}%)")
    print(f"Win rate in next week's game: {next_wins}/{next_total} ({next_wins/next_total*100:.1f}%)")
    
    # Score differential stats
    print("\n--- SCORE DIFFERENTIAL (Actual - Adjusted) ---")
    print(f"Average score difference in next game: {results_df['next_score_diff'].mean():.2f}")
    print(f"Median score difference in next game: {results_df['next_score_diff'].median():.2f}")
    print(f"Std dev of score difference in next game: {results_df['next_score_diff'].std():.2f}")
    
    # Distribution of next week performance
    print("\n--- Distribution of Next Week Score Differentials ---")
    print(f"Still overperformed (diff >= 0): {len(results_df[results_df['next_score_diff'] >= 0])} ({len(results_df[results_df['next_score_diff'] >= 0])/len(results_df)*100:.1f}%)")
    print(f"Underperformed (diff < 0): {len(results_df[results_df['next_score_diff'] < 0])} ({len(results_df[results_df['next_score_diff'] < 0])/len(results_df)*100:.1f}%)")
    print(f"Still overperformed by 7+ (diff >= 7): {len(results_df[results_df['next_score_diff'] >= 7])} ({len(results_df[results_df['next_score_diff'] >= 7])/len(results_df)*100:.1f}%)")
    print(f"Underperformed by 7+ (diff <= -7): {len(results_df[results_df['next_score_diff'] <= -7])} ({len(results_df[results_df['next_score_diff'] <= -7])/len(results_df)*100:.1f}%)")
    
    print("\n--- Comparison: Overperformance Week vs Next Week ---")
    print(f"Avg score diff (overperformance week): {results_df['overperf_diff'].mean():.2f}")
    print(f"Avg score diff (next week): {results_df['next_score_diff'].mean():.2f}")
    print(f"Change: {results_df['next_score_diff'].mean() - results_df['overperf_diff'].mean():.2f}")
    print(f"Regression to mean: {(1 - results_df['next_score_diff'].mean()/results_df['overperf_diff'].mean())*100:.1f}%")
    
    # Save detailed results to proper output directory
    output_dir = Path("/Users/thomasmyles/dev/betting/data/04_output/nfl/unexpected_points")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "overperformance_next_week_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")
    
    # Additional breakdown by magnitude of overperformance
    print("\n" + "=" * 80)
    print("STEP 4: Breakdown by Magnitude of Overperformance")
    print("=" * 80)
    
    bins = [7, 10, 13, 100]
    labels = ['7-9 points', '10-12 points', '13+ points']
    results_df['overperf_bucket'] = pd.cut(results_df['overperf_diff'], bins=bins, labels=labels, include_lowest=True)
    
    for bucket in labels:
        bucket_data = results_df[results_df['overperf_bucket'] == bucket]
        if len(bucket_data) > 0:
            bucket_wins = bucket_data['won_next_game'].sum()
            bucket_games = bucket_data['won_next_game'].notna().sum()
            win_rate = (bucket_wins / bucket_games * 100) if bucket_games > 0 else 0
            
            print(f"\n{bucket} overperformance (n={len(bucket_data)}):")
            print(f"  Next week win rate: {bucket_wins}/{bucket_games} ({win_rate:.1f}%)")
            print(f"  Next week avg score diff: {bucket_data['next_score_diff'].mean():.2f}")
            print(f"  Next week median score diff: {bucket_data['next_score_diff'].median():.2f}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("STEP 5: Visualizations - Win Rate by Overperformance Bins")
    print("=" * 80)
    
    # Create bins for overperformance
    bins = [7, 9, 11, 13, 15, 100]
    labels = ['7-8', '9-10', '11-12', '13-14', '15+']
    results_df['overperf_bin'] = pd.cut(results_df['overperf_diff'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate win rate for each bin
    bin_stats = []
    for bin_label in labels:
        bin_data = results_df[results_df['overperf_bin'] == bin_label]
        if len(bin_data) > 0:
            wins = bin_data['won_next_game'].sum()
            games = bin_data['won_next_game'].notna().sum()
            win_pct = (wins / games * 100) if games > 0 else 0
            avg_next_diff = bin_data['next_score_diff'].mean()
            
            bin_stats.append({
                'bin': bin_label,
                'games': games,
                'wins': wins,
                'losses': games - wins,
                'win_pct': win_pct,
                'avg_next_diff': avg_next_diff
            })
    
    bin_stats_df = pd.DataFrame(bin_stats)
    
    print("\n📊 WIN PERCENTAGE BY OVERPERFORMANCE BIN:")
    print("=" * 80)
    for _, row in bin_stats_df.iterrows():
        print(f"\n{row['bin']} points overperformance:")
        print(f"  Games: {int(row['games'])}")
        print(f"  Wins: {int(row['wins'])} | Losses: {int(row['losses'])}")
        print(f"  Win %: {row['win_pct']:.1f}%")
        print(f"  Avg next week diff: {row['avg_next_diff']:+.2f}")
        
        # Visual bar
        bar_length = int(row['win_pct'] / 2)
        bar = '█' * bar_length
        print(f"  [{bar:<50}] {row['win_pct']:.1f}%")
    
    # Create matplotlib visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NFL Overperformance Analysis: Next Week Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Win Rate by Bin
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if x >= 50 else '#e74c3c' for x in bin_stats_df['win_pct']]
    bars = ax1.bar(bin_stats_df['bin'], bin_stats_df['win_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='50% (Break Even)')
    ax1.set_xlabel('Overperformance Range (Points)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Win % Next Week', fontsize=12, fontweight='bold')
    ax1.set_title('Win Rate Next Week by Overperformance Magnitude', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, pct, games in zip(bars, bin_stats_df['win_pct'], bin_stats_df['games']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct:.1f}%\n(n={int(games)})',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Next Week Score Differential Box Plot
    ax2 = axes[0, 1]
    bin_data_for_box = [results_df[results_df['overperf_bin'] == label]['next_score_diff'].dropna() 
                        for label in labels if len(results_df[results_df['overperf_bin'] == label]) > 0]
    bp = ax2.boxplot(bin_data_for_box, labels=[l for l in labels if len(results_df[results_df['overperf_bin'] == l]) > 0],
                     patch_artist=True, notch=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Break Even (0)')
    ax2.set_xlabel('Overperformance Range (Points)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Next Week Score Differential', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Next Week Performance', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Sample Size by Bin
    ax3 = axes[1, 0]
    ax3.bar(bin_stats_df['bin'], bin_stats_df['games'], color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Overperformance Range (Points)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Occurrences', fontsize=12, fontweight='bold')
    ax3.set_title('Sample Size by Overperformance Magnitude', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (bin_val, count) in enumerate(zip(bin_stats_df['bin'], bin_stats_df['games'])):
        ax3.text(i, count + 0.5, f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Avg Next Week Differential by Bin
    ax4 = axes[1, 1]
    colors4 = ['#2ecc71' if x >= 0 else '#e74c3c' for x in bin_stats_df['avg_next_diff']]
    bars4 = ax4.bar(bin_stats_df['bin'], bin_stats_df['avg_next_diff'], color=colors4, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    ax4.set_xlabel('Overperformance Range (Points)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Avg Score Differential', fontsize=12, fontweight='bold')
    ax4.set_title('Average Next Week Score Differential', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars4, bin_stats_df['avg_next_diff']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.3 if height > 0 else -0.5),
                f'{val:+.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / "overperformance_next_week_visualization.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {fig_path}")
    
    # Display the plot
    plt.show()
    
    # Save bin stats summary
    bin_stats_path = output_dir / "overperformance_bin_summary.csv"
    bin_stats_df.to_csv(bin_stats_path, index=False)
    print(f"✓ Bin summary saved to: {bin_stats_path}")

else:
    print("\nNo follow-up data found for overperforming teams")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

