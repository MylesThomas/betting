"""
Validate v1 runs by comparing current run vs naive baseline.

Reads artifacts in `03_backtesting/runs/*` and writes standardized outputs:
- validation_summary.json
- comparison_table.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
RUNS_DIR = MODULE_DIR.parent / "03_backtesting" / "runs"


def _load_run_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def main() -> None:
    run_dirs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])
    if len(run_dirs) == 0:
        raise ValueError("No run folders found in 03_backtesting/runs")

    summaries = [_load_run_summary(run_dir) for run_dir in run_dirs]
    table = pd.DataFrame(summaries).sort_values("run_id").reset_index(drop=True)
    current = table.iloc[-1].to_dict()

    baseline_ref = table.iloc[0].to_dict()
    deltas = {
        "roi_delta_vs_baseline": float(current["roi"] - baseline_ref["roi"]),
        "rmse_delta_vs_baseline": float(current["rmse"] - baseline_ref["rmse"]),
        "win_rate_delta_vs_baseline": float(current["win_rate"] - baseline_ref["win_rate"]),
        "signal_rate_delta_vs_baseline": float(current["signal_rate"] - baseline_ref["signal_rate"]),
    }
    summary = {"current_run": current["run_id"], "baseline_run": baseline_ref["run_id"], **deltas}

    current_run_dir = RUNS_DIR / current["run_id"]
    (current_run_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    table.to_csv(current_run_dir / "comparison_table.csv", index=False)
    print(f"Validation written to: {current_run_dir}")


if __name__ == "__main__":
    main()

