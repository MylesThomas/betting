from pathlib import Path

import pytest


def test_latest_run_contains_required_artifacts():
    runs_dir = Path("src/nba_three_point_modeling/03_backtesting/runs")
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.exists() else []
    if len(run_dirs) == 0:
        pytest.skip("No backtest runs found yet")

    latest = run_dirs[-1]
    required = [
        "config.yaml",
        "manifest.json",
        "predictions.parquet",
        "bets.parquet",
        "summary.json",
    ]
    for name in required:
        assert (latest / name).exists(), f"Missing {name} in {latest}"

