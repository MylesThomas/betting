"""
Orchestrator for phased v5 3PM decomposition research workflow.

Context:
- Provides one modular CLI with a required --phase flag.
- Allows another agent to run one phase or the full sequence end-to-end using
  consistent file paths and deterministic seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse CLI args for orchestration."""
    parser = argparse.ArgumentParser(description="Run modular v5 workflow phases.")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=[
            "phase0",
            "phase1a",
            "phase1b",
            "phase1c",
            "phase2",
            "phase3",
            "phase4",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--tmp-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--cache-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--use-cache", type=str, default="true")
    parser.add_argument("--force-refresh-cache", type=str, default="false")
    return parser.parse_args()


def run_cmd(command: list[str]) -> None:
    """Run one phase command and fail fast on non-zero exit."""
    print(" ".join(command))
    subprocess.run(command, check=True)


def phase_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    """Build command map for each phase."""
    tmp_dir = str(Path(args.tmp_dir).expanduser())
    phase0 = [
        sys.executable,
        str(SCRIPT_DIR / "v5_build_eval_universe.py"),
        "--phase",
        "phase0",
        "--seed",
        str(args.seed),
        "--season",
        args.season,
        "--cache-dir",
        args.cache_dir,
        "--use-cache",
        args.use_cache,
        "--force-refresh-cache",
        args.force_refresh_cache,
        "--output-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--output-qc",
        f"{tmp_dir}/v5_eval_universe_qc.csv",
    ]
    phase1a = [
        sys.executable,
        str(SCRIPT_DIR / "v5_model_target_sweep.py"),
        "--phase",
        "phase1a",
        "--seed",
        str(args.seed),
        "--target",
        "min",
        "--input-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--selection-mode",
        "both",
        "--output-csv",
        f"{tmp_dir}/v5_min_models.csv",
        "--output-trace-csv",
        f"{tmp_dir}/v5_min_trace.csv",
        "--output-importance-csv",
        f"{tmp_dir}/v5_min_importance.csv",
    ]
    phase1b = [
        sys.executable,
        str(SCRIPT_DIR / "v5_model_target_sweep.py"),
        "--phase",
        "phase1b",
        "--seed",
        str(args.seed),
        "--target",
        "fga_per_min",
        "--input-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--selection-mode",
        "both",
        "--output-csv",
        f"{tmp_dir}/v5_fga_per_min_models.csv",
        "--output-trace-csv",
        f"{tmp_dir}/v5_fga_per_min_trace.csv",
        "--output-importance-csv",
        f"{tmp_dir}/v5_fga_per_min_importance.csv",
    ]
    phase1c = [
        sys.executable,
        str(SCRIPT_DIR / "v5_model_target_sweep.py"),
        "--phase",
        "phase1c",
        "--seed",
        str(args.seed),
        "--target",
        "fg3_pct",
        "--input-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--selection-mode",
        "both",
        "--output-csv",
        f"{tmp_dir}/v5_fg3_pct_models.csv",
        "--output-trace-csv",
        f"{tmp_dir}/v5_fg3_pct_trace.csv",
        "--output-importance-csv",
        f"{tmp_dir}/v5_fg3_pct_importance.csv",
        "--output-calibration-csv",
        f"{tmp_dir}/v5_fg3_pct_calibration.csv",
    ]
    phase2 = [
        sys.executable,
        str(SCRIPT_DIR / "v5_recompose_fg3m.py"),
        "--phase",
        "phase2",
        "--seed",
        str(args.seed),
        "--input-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--min-model-csv",
        f"{tmp_dir}/v5_min_models.csv",
        "--fga-per-min-model-csv",
        f"{tmp_dir}/v5_fga_per_min_models.csv",
        "--fg3-pct-model-csv",
        f"{tmp_dir}/v5_fg3_pct_models.csv",
        "--output-csv",
        f"{tmp_dir}/v5_fg3m_recompose_comparison.csv",
        "--output-predictions-csv",
        f"{tmp_dir}/v5_fg3m_recompose_predictions.csv",
        "--output-outliers-csv",
        f"{tmp_dir}/v5_fg3m_recompose_outliers.csv",
        "--output-memo-md",
        f"{tmp_dir}/v5_recommendation_memo.md",
    ]
    phase3 = [
        sys.executable,
        str(SCRIPT_DIR / "v5_segment_robustness.py"),
        "--phase",
        "phase3",
        "--seed",
        str(args.seed),
        "--input-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--input-predictions",
        f"{tmp_dir}/v5_fg3m_recompose_predictions.csv",
        "--input-comparison",
        f"{tmp_dir}/v5_fg3m_recompose_comparison.csv",
        "--output-segment-csv",
        f"{tmp_dir}/v5_robustness_segment_metrics.csv",
        "--output-stability-csv",
        f"{tmp_dir}/v5_model_stability_summary.csv",
    ]
    phase4 = [
        sys.executable,
        str(SCRIPT_DIR / "v5_prob_calibration.py"),
        "--phase",
        "phase4",
        "--seed",
        str(args.seed),
        "--input-predictions",
        f"{tmp_dir}/v5_fg3m_recompose_predictions.csv",
        "--input-comparison",
        f"{tmp_dir}/v5_fg3m_recompose_comparison.csv",
        "--input-universe",
        f"{tmp_dir}/v5_eval_universe.parquet",
        "--output-calibration-csv",
        f"{tmp_dir}/v5_prob_calibration.csv",
        "--output-edge-csv",
        f"{tmp_dir}/v5_edge_bucket_eval.csv",
    ]
    return {
        "phase0": phase0,
        "phase1a": phase1a,
        "phase1b": phase1b,
        "phase1c": phase1c,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
    }


def main() -> None:
    """Dispatch one phase or full sequence."""
    args = parse_args()
    commands = phase_commands(args)
    if args.phase == "all":
        order = ["phase0", "phase1a", "phase1b", "phase1c", "phase2", "phase3", "phase4"]
        for key in order:
            run_cmd(commands[key])
    else:
        run_cmd(commands[args.phase])


if __name__ == "__main__":
    main()

