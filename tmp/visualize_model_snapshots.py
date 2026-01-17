"""
Visualize model snapshot comparison results.

Shows how model performance and coefficients change as we train on more data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
data_path = Path.home() / 'Downloads' / 'tmp' / 'model_snapshot_comparison_all.csv'
df = pd.read_csv(data_path)

# Convert date to datetime for better plotting
df['model_date'] = pd.to_datetime(df['model_date'])

print(f"Loaded {len(df)} models")
print(f"Date range: {df['model_date'].min()} to {df['model_date'].max()}")
print()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Snapshot Analysis: How Performance Changes Over Season', fontsize=16, fontweight='bold')

# ============================================================================
# Plot 1: Train/Test MAE Over Time
# ============================================================================
ax1 = axes[0, 0]

ax1.plot(df['model_date'], df['train_mae'], 'o-', label='Train MAE', alpha=0.7, markersize=3)
ax1.plot(df['model_date'], df['test_mae'], 's-', label='Test MAE', alpha=0.7, markersize=3)
ax1.plot(df['model_date'], df['test_market_mae'], '^-', label='Market MAE (test)', alpha=0.7, markersize=3, color='red')

ax1.axhline(y=df['test_mae'].mean(), color='blue', linestyle='--', alpha=0.3, label=f'Avg Test MAE: {df["test_mae"].mean():.2f}')
ax1.axhline(y=df['test_market_mae'].mean(), color='red', linestyle='--', alpha=0.3, label=f'Avg Market MAE: {df["test_market_mae"].mean():.2f}')

ax1.set_xlabel('Model Training Cutoff Date', fontsize=12)
ax1.set_ylabel('Mean Absolute Error (points)', fontsize=12)
ax1.set_title('Model Performance Over Season', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: Coefficient Evolution (How model weights change)
# ============================================================================
ax2 = axes[0, 1]

ax2.plot(df['model_date'], df['intercept'], 'o-', label='Intercept (β₀)', alpha=0.7, markersize=3)
ax2.plot(df['model_date'], df['coef_x1'], 's-', label='x1 coef (β₁: hist avg)', alpha=0.7, markersize=3)
ax2.plot(df['model_date'], df['coef_x2'], '^-', label='x2 coef (β₂: market)', alpha=0.7, markersize=3)
ax2.plot(df['model_date'], df['coef_x3'], 'd-', label='x3 coef (β₃: conf game)', alpha=0.7, markersize=3)

ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='y = x2 (pure market)')
ax2.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3)

ax2.set_xlabel('Model Training Cutoff Date', fontsize=12)
ax2.set_ylabel('Coefficient Value', fontsize=12)
ax2.set_title('Coefficient Evolution: Model Converges to Market', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Overfit Metric (test_mae - train_mae) vs Training Set Size
# ============================================================================
ax3 = axes[1, 0]

scatter = ax3.scatter(df['n_train'], df['overfit'], c=df['n_test'], 
                     cmap='viridis', alpha=0.6, s=50)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No overfit line')

# Add trend line
z = np.polyfit(df['n_train'], df['overfit'], 2)
p = np.poly1d(z)
x_smooth = np.linspace(df['n_train'].min(), df['n_train'].max(), 100)
ax3.plot(x_smooth, p(x_smooth), "r-", alpha=0.5, linewidth=2, label='Trend')

ax3.set_xlabel('Training Set Size (n_train)', fontsize=12)
ax3.set_ylabel('Overfit (test_mae - train_mae)', fontsize=12)
ax3.set_title('Negative Overfitting: Test is EASIER than Train!', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Test Set Size', rotation=270, labelpad=15)

# ============================================================================
# Plot 4: Model vs Market Win Rate Over Season
# ============================================================================
ax4 = axes[1, 1]

df['test_model_win_pct'] = df['test_model_wins'] / (df['test_model_wins'] + df['test_market_wins']) * 100

ax4.plot(df['model_date'], df['test_model_win_pct'], 'o-', alpha=0.7, markersize=3, color='purple')
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (coin flip)')
ax4.axhline(y=df['test_model_win_pct'].mean(), color='purple', linestyle='--', alpha=0.3, 
           label=f'Avg: {df["test_model_win_pct"].mean():.1f}%')

ax4.set_xlabel('Model Training Cutoff Date', fontsize=12)
ax4.set_ylabel('Model Win % vs Market', fontsize=12)
ax4.set_title('Model vs Market: Win Rate Over Season', fontsize=14, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([30, 70])

plt.tight_layout()

# Save figure
output_path = Path.home() / 'Downloads' / 'tmp' / 'model_snapshot_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"💾 Saved visualization to: {output_path}")

# Print key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print()

print("1. COEFFICIENT EVOLUTION:")
print(f"   Early season (first model):")
print(f"      Intercept: {df.iloc[0]['intercept']:.2f}")
print(f"      x1 (hist): {df.iloc[0]['coef_x1']:.3f}")
print(f"      x2 (market): {df.iloc[0]['coef_x2']:.3f}")
print(f"   Late season (last model):")
print(f"      Intercept: {df.iloc[-1]['intercept']:.2f}")
print(f"      x1 (hist): {df.iloc[-1]['coef_x1']:.3f}")
print(f"      x2 (market): {df.iloc[-1]['coef_x2']:.3f}")
print(f"   → Model learns to trust market (x2) and ignore history (x1)")
print()

print("2. OVERFITTING:")
print(f"   Average overfit: {df['overfit'].mean():.2f} points")
print(f"   Std dev: {df['overfit'].std():.2f} points")
print(f"   Min (most negative): {df['overfit'].min():.2f} points")
print(f"   Max (most positive): {df['overfit'].max():.2f} points")
print(f"   → NEGATIVE overfit = test is easier than train!")
print()

print("3. MODEL VS MARKET:")
print(f"   Average model win rate: {df['test_model_win_pct'].mean():.1f}%")
print(f"   Best win rate: {df['test_model_win_pct'].max():.1f}%")
print(f"   Worst win rate: {df['test_model_win_pct'].min():.1f}%")
print(f"   → Model barely beats coin flip (50%)")
print()

print("4. BEST MODEL BY TEST MAE:")
best_idx = df['test_mae'].idxmin()
best = df.iloc[best_idx]
print(f"   Date: {best['model_date']}")
print(f"   Training games: {best['n_train']:.0f}")
print(f"   Test games: {best['n_test']:.0f}")
print(f"   Test MAE: {best['test_mae']:.2f} vs Market: {best['test_market_mae']:.2f}")
print(f"   ⚠️  WARNING: Only {best['n_test']:.0f} test games - unreliable!")
print()

# Find best model with at least 500 test games
reliable_models = df[df['n_test'] >= 500]
if len(reliable_models) > 0:
    best_reliable_idx = reliable_models['test_mae'].idxmin()
    best_reliable = reliable_models.loc[best_reliable_idx]
    print("5. BEST RELIABLE MODEL (n_test ≥ 500):")
    print(f"   Date: {best_reliable['model_date']}")
    print(f"   Training games: {best_reliable['n_train']:.0f}")
    print(f"   Test games: {best_reliable['n_test']:.0f}")
    print(f"   Test MAE: {best_reliable['test_mae']:.2f} vs Market: {best_reliable['test_market_mae']:.2f}")
    print(f"   Win rate: {best_reliable['test_model_win_pct']:.1f}%")
print()

print("="*80)

plt.show()

