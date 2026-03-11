"""
Orchestrate v6 spread-context research workflow.

Context:
- Provides one CLI entrypoint to run v6 phase build/model/review components.
- Keeps deterministic paths and arguments so agent handoffs can run partial or
  full spread-context experiments without manual command stitching.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v6 orchestration."""
    parser = argparse.ArgumentParser(description="Run modular v6 spread workflow phases.")
    parser.add_argument("--phase", type=str, required=True, choices=["build", "model", "review", "all"])
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--tmp-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--cache-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--use-cache", type=str, default="true")
    parser.add_argument("--force-refresh-cache", type=str, default="false")
    parser.add_argument("--include-optional-targets", type=str, default="true")
    parser.add_argument("--include-incremental-model", type=str, default="true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def run_cmd(command: list[str]) -> None:
    """Run one phase command and fail fast on non-zero exit."""
    print(" ".join(command))
    subprocess.run(command, check=True)


def phase_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    """Build command map for each v6 phase."""
    tmp_dir = str(Path(args.tmp_dir).expanduser())
    build = [
        sys.executable,
        str(SCRIPT_DIR / "v6_build_spread_universe.py"),
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
        f"{tmp_dir}/v6_spread_universe.parquet",
        "--output-qc",
        f"{tmp_dir}/v6_spread_universe_qc.csv",
    ]
    model = [
        sys.executable,
        str(SCRIPT_DIR / "v6_spread_target_sweep.py"),
        "--seed",
        str(args.seed),
        "--input-universe",
        f"{tmp_dir}/v6_spread_universe.parquet",
        "--include-optional-targets",
        args.include_optional_targets,
        "--include-incremental-model",
        args.include_incremental_model,
        "--test-fraction",
        str(args.test_fraction),
        "--output-summary-csv",
        f"{tmp_dir}/v6_spread_model_summary.csv",
        "--output-bin-effects-csv",
        f"{tmp_dir}/v6_spread_bin_effects.csv",
        "--output-ranked-targets-csv",
        f"{tmp_dir}/v6_spread_ranked_targets.csv",
    ]
    review = [
        sys.executable,
        str(SCRIPT_DIR / "v6_review_outputs_duckdb.py"),
        "--tmp-dir",
        tmp_dir,
        "--top-n",
        str(args.top_n),
    ]
    return {"build": build, "model": model, "review": review}


def main() -> None:
    """Dispatch one v6 phase or full sequence."""
    args = parse_args()
    commands = phase_commands(args)
    if args.phase == "all":
        for key in ["build", "model", "review"]:
            run_cmd(commands[key])
    else:
        run_cmd(commands[args.phase])


if __name__ == "__main__":
    main()
