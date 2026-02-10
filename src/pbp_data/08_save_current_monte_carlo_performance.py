"""
Script 08: Save Current Monte Carlo Performance (Baseline Version)

Purpose:
- Archive current predictions and analysis results before making changes
- Creates versioned snapshot for comparison (parquet-based)
- Tracks all versions in a summary parquet for easy comparison

Usage:
    python src/pbp_data/08_save_current_monte_carlo_performance.py --version v1 --description "Baseline: no dampening, has quarter boundary bug"
    
Output:
    ~/Downloads/tmp/monte_carlo_validation/
        - versions_summary.parquet  (tracks all versions and metrics)
        - versions/v1/
            - predictions.parquet
            - metrics.parquet (detailed metrics by quarter/probability bucket)
            - analysis/ (plots)
            - metadata.json
"""

import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytz


# =============================================================================
# PATHS
# =============================================================================

VALIDATION_DIR = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation"
PREDICTIONS_FILE = VALIDATION_DIR / "predictions.parquet"
ANALYSIS_DIR = VALIDATION_DIR / "analysis"
VERSIONS_DIR = VALIDATION_DIR / "versions"
VERSIONS_SUMMARY_FILE = VALIDATION_DIR / "versions_summary.parquet"
VERSIONS_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_et_timestamp():
    """Get current timestamp in ET timezone."""
    et_tz = pytz.timezone('America/New_York')
    return datetime.now(et_tz)


# =============================================================================
# ARCHIVAL FUNCTIONS
# =============================================================================

def save_version(version_name, description, notes=None):
    """
    Save current Monte Carlo results as a versioned snapshot.
    
    Args:
        version_name: Version identifier (e.g., "v1", "v2_with_dampening")
        description: Short description of this version
        notes: Optional detailed notes
    """
    print("="*80)
    print(f"ARCHIVING MONTE CARLO PERFORMANCE - {version_name}")
    print("="*80)
    print()
    
    # Create version directory
    version_dir = VERSIONS_DIR / version_name
    version_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"📁 Version directory: {version_dir}")
    print()
    
    # 1. Copy predictions
    if PREDICTIONS_FILE.exists():
        print("📥 Copying predictions.parquet...")
        dest_predictions = version_dir / "predictions.parquet"
        shutil.copy2(PREDICTIONS_FILE, dest_predictions)
        
        # Get stats
        df = pd.read_parquet(PREDICTIONS_FILE)
        n_predictions = len(df)
        n_games = df['game_id'].nunique()
        n_players = df['player_name'].nunique()
        
        print(f"   ✅ Saved {n_predictions:,} predictions")
        print(f"      ({n_games} games, {n_players} players)")
    else:
        print("   ⚠️  predictions.parquet not found, skipping")
        n_predictions = n_games = n_players = 0
    print()
    
    # 2. Copy analysis directory
    if ANALYSIS_DIR.exists():
        print("📊 Copying analysis results...")
        dest_analysis = version_dir / "analysis"
        
        # Remove if exists and recreate
        if dest_analysis.exists():
            shutil.rmtree(dest_analysis)
        
        shutil.copytree(ANALYSIS_DIR, dest_analysis)
        
        # Count files
        plots = list(dest_analysis.glob("*.png"))
        csvs = list(dest_analysis.glob("*.csv"))
        txts = list(dest_analysis.glob("*.txt"))
        
        print(f"   ✅ Saved analysis:")
        print(f"      {len(plots)} plots")
        print(f"      {len(csvs)} CSV files")
        print(f"      {len(txts)} text reports")
    else:
        print("   ⚠️  analysis/ directory not found, skipping")
        plots = csvs = txts = []
    print()
    
    # 3. Load all metrics and save as parquet
    print("📊 Loading metrics...")
    
    # Overall Brier score
    brier_file = ANALYSIS_DIR / "brier_scores.csv"
    if brier_file.exists():
        brier_df = pd.read_csv(brier_file)
        overall_brier = float(brier_df['overall_brier_score'].iloc[0])
    else:
        overall_brier = None
    
    # Calibration data
    calibration_file = ANALYSIS_DIR / "calibration.csv"
    calibration_df = pd.read_csv(calibration_file) if calibration_file.exists() else None
    
    # Player-level Brier scores
    player_brier_file = ANALYSIS_DIR / "player_brier_scores.csv"
    player_brier_df = pd.read_csv(player_brier_file) if player_brier_file.exists() else None
    
    # Combine all metrics into single parquet
    metrics_data = []
    
    # Add overall metric
    if overall_brier is not None:
        metrics_data.append({
            'metric_type': 'overall',
            'metric_name': 'brier_score',
            'value': overall_brier,
            'breakdown': None,
        })
    
    # Add calibration metrics
    if calibration_df is not None:
        for idx, row in calibration_df.iterrows():
            metrics_data.append({
                'metric_type': 'calibration',
                'metric_name': f'prob_bin_{row["prob_bin"]}',
                'value': row['actual_freq'] - row['predicted_prob'],  # Calibration error
                'breakdown': {
                    'predicted_prob': row['predicted_prob'],
                    'actual_freq': row['actual_freq'],
                    'count': row['count'],
                }
            })
    
    # Add player-level metrics
    if player_brier_df is not None:
        for idx, row in player_brier_df.iterrows():
            metrics_data.append({
                'metric_type': 'player',
                'metric_name': row['player_name'],
                'value': row['brier_score'],
                'breakdown': {
                    'num_games': row['num_games'],
                    'hit_rate': row['hit_rate'],
                }
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = version_dir / "metrics.parquet"
        metrics_df.to_parquet(metrics_file, index=False)
        print(f"   ✅ Saved metrics.parquet ({len(metrics_df)} metrics)")
    else:
        print(f"   ⚠️  No metrics available")
    print()
    
    # 4. Update versions summary parquet
    print("📝 Updating versions summary...")
    
    timestamp_et = get_et_timestamp()
    
    version_summary = {
        'version': version_name,
        'description': description,
        'notes': notes,
        'timestamp_et': timestamp_et,
        'timestamp_unix': timestamp_et.timestamp(),
        'overall_brier_score': overall_brier,
        'n_predictions': n_predictions,
        'n_games': n_games,
        'n_players': n_players,
        'n_plots': len(plots),
    }
    
    # Load existing summary or create new
    if VERSIONS_SUMMARY_FILE.exists():
        summary_df = pd.read_parquet(VERSIONS_SUMMARY_FILE)
        # Remove existing entry for this version if it exists
        summary_df = summary_df[summary_df['version'] != version_name]
        # Append new entry
        summary_df = pd.concat([summary_df, pd.DataFrame([version_summary])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([version_summary])
    
    # Sort by timestamp
    summary_df = summary_df.sort_values('timestamp_unix').reset_index(drop=True)
    summary_df.to_parquet(VERSIONS_SUMMARY_FILE, index=False)
    
    print(f"   ✅ Updated versions_summary.parquet")
    print()
    
    # 5. Create metadata JSON (for human readability)
    print("📝 Creating metadata JSON...")
    metadata = {
        "version": version_name,
        "description": description,
        "notes": notes,
        "timestamp_et": timestamp_et.isoformat(),
        "dataset": {
            "n_predictions": n_predictions,
            "n_games": n_games,
            "n_players": n_players,
        },
        "performance": {
            "overall_brier_score": overall_brier,
        },
        "files": {
            "predictions": "predictions.parquet",
            "metrics": "metrics.parquet",
            "analysis_dir": "analysis/",
            "n_plots": len(plots),
            "n_csvs": len(csvs),
            "n_reports": len(txts),
        }
    }
    
    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved metadata.json")
    print()
    
    # 6. Create README
    print("📄 Creating README...")
    readme_lines = [
        f"# Monte Carlo Performance - {version_name}",
        "",
        f"**Date (ET)**: {timestamp_et.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Description",
        f"{description}",
        "",
    ]
    
    if notes:
        readme_lines.extend([
            "## Notes",
            f"{notes}",
            "",
        ])
    
    readme_lines.extend([
        "## Dataset Statistics",
        f"- Total predictions: {n_predictions:,}",
        f"- Unique games: {n_games}",
        f"- Unique players: {n_players}",
        "",
        "## Performance",
    ])
    
    if overall_brier is not None:
        readme_lines.append(f"- Overall Brier Score: {overall_brier:.4f}")
    else:
        readme_lines.append("- Overall Brier Score: Not available")
    
    readme_lines.extend([
        "",
        "## Files",
        "- `predictions.parquet` - All predictions (~200k rows)",
        "- `metrics.parquet` - Detailed metrics (overall, calibration, per-player)",
        "- `analysis/` - Analysis plots and CSV reports",
        "- `metadata.json` - Structured metadata",
        "",
        "## Comparing Versions",
        "```python",
        "# Load versions summary",
        "import pandas as pd",
        f"summary = pd.read_parquet('{VERSIONS_SUMMARY_FILE}')",
        "print(summary[['version', 'timestamp_et', 'overall_brier_score']])",
        "",
        "# Load metrics for this version",
        f"metrics = pd.read_parquet('{version_dir / 'metrics.parquet'}')",
        "```",
        "",
        "## Analysis Plots",
    ])
    
    for plot in sorted(plots):
        readme_lines.append(f"- `analysis/{plot.name}`")
    
    readme_file = version_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write('\n'.join(readme_lines))
    
    print(f"   ✅ Saved README.md")
    print()
    
    # Summary
    print("="*80)
    print("✅ VERSION ARCHIVED SUCCESSFULLY")
    print("="*80)
    print(f"\nVersion: {version_name}")
    print(f"Location: {version_dir}")
    if overall_brier is not None:
        print(f"Brier Score: {overall_brier:.4f}")
    print()
    print("Files saved:")
    print(f"  - predictions.parquet ({n_predictions:,} rows)")
    print(f"  - metrics.parquet")
    print(f"  - {len(plots)} plots")
    print(f"  - {len(csvs)} CSV reports")
    print(f"  - {len(txts)} text reports")
    print(f"  - metadata.json")
    print(f"  - README.md")
    print()
    print(f"📊 versions_summary.parquet updated at:")
    print(f"   {VERSIONS_SUMMARY_FILE}")
    print()


def list_versions():
    """List all archived versions from summary parquet."""
    print("="*80)
    print("ARCHIVED VERSIONS")
    print("="*80)
    print()
    
    if not VERSIONS_SUMMARY_FILE.exists():
        print("No versions found. Run with --version and --description to create first version.")
        return
    
    summary_df = pd.read_parquet(VERSIONS_SUMMARY_FILE)
    
    for idx, row in summary_df.iterrows():
        print(f"📦 {row['version']}")
        print(f"   Description: {row['description']}")
        print(f"   Date (ET): {pd.to_datetime(row['timestamp_et']).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if pd.notna(row['overall_brier_score']):
            print(f"   Brier Score: {row['overall_brier_score']:.4f}")
        print(f"   Dataset: {row['n_predictions']:,} predictions, {row['n_games']} games, {row['n_players']} players")
        print(f"   Location: {VERSIONS_DIR / row['version']}")
        print()
    
    print(f"\n💡 To compare versions programmatically:")
    print(f"   summary = pd.read_parquet('{VERSIONS_SUMMARY_FILE}')")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Archive Monte Carlo performance results")
    parser.add_argument("--version", type=str, help="Version name (e.g., 'v1', 'v2_dampening')")
    parser.add_argument("--description", type=str, help="Short description of this version")
    parser.add_argument("--notes", type=str, default=None, help="Optional detailed notes")
    parser.add_argument("--list", action="store_true", help="List all archived versions")
    
    args = parser.parse_args()
    
    if args.list:
        list_versions()
        return
    
    if not args.version or not args.description:
        print("❌ Error: --version and --description are required")
        print("\nExample:")
        print('  python src/pbp_data/08_save_current_monte_carlo_performance.py \\')
        print('    --version "v1" \\')
        print('    --description "Baseline: no dampening, has quarter boundary bug"')
        return
    
    save_version(args.version, args.description, args.notes)


if __name__ == "__main__":
    main()
