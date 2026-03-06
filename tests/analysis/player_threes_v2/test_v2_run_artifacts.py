import json
from pathlib import Path

import pytest
import yaml


def test_latest_v2_run_has_required_artifacts_and_summary_fields():
    runs_dir = Path("src/nba_three_point_modeling/03_backtesting/runs")
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.exists() else []
    if len(run_dirs) == 0:
        pytest.skip("No backtest runs found yet")

    v2_runs = [p for p in run_dirs if "v2_three_input_regression" in p.name]
    if len(v2_runs) == 0:
        pytest.skip("No v2 runs found yet")

    latest_v2 = v2_runs[-1]
    for name in [
        "config.yaml",
        "manifest.json",
        "predictions.parquet",
        "bets.parquet",
        "summary.json",
    ]:
        assert (latest_v2 / name).exists(), f"Missing {name} in {latest_v2}"

    config = yaml.safe_load((latest_v2 / "config.yaml").read_text())
    summary = json.loads((latest_v2 / "summary.json").read_text())
    assert config["mean_model_id"] == "v2_three_input_regression"
    assert summary["uncertainty_model_id"] == config["uncertainty_model_id"]
