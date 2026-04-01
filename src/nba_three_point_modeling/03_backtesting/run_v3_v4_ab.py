"""
Run controlled v3 vs v4 backtest A/B with identical settings.

Context:
- Uses current_config.yaml as base configuration.
- Executes two runs differing only in mean_model_id:
  - v3_three_input_regression
  - v4_market_spread_regression
- Writes side-by-side comparison artifact for easy decisioning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd
import yaml


MODULE_DIR = Path(__file__).resolve().parent
RUNS_DIR = MODULE_DIR / "runs"
CONFIG_PATH = MODULE_DIR / "current_config.yaml"
VALIDATION_SCRIPT = MODULE_DIR.parent / "04_validation" / "validate_runs.py"
OUT_CSV = Path("~/Downloads/tmp/v3_v4_backtest_ab_comparison.csv").expanduser()


def parse_args() -> argparse.Namespace:
    """Parse CLI args for single-player v3/v4 A-B."""
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled single-player v3/v4 A-B with identical settings and "
            "write comparison artifacts."
        )
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(CONFIG_PATH),
        help="Backtest config path used as the base template.",
    )
    parser.add_argument(
        "--output-comparison-csv",
        type=str,
        default=str(OUT_CSV),
        help="Output CSV path for v3/v4 run-level metrics.",
    )
    parser.add_argument(
        "--spread-gate-mode",
        type=str,
        default="",
        choices=["", "strict", "relaxed", "off"],
        help="Optional gate mode override for both runs.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validate_runs.py after both backtests complete.",
    )
    return parser.parse_args()


def _run_with_model(
    base_config: dict[str, Any],
    config_path: Path,
    model_id: str,
    group_id: str,
    spread_gate_mode: str,
) -> Path:
    """Execute one backtest run with temporary model-specific config."""
    config = dict(base_config)
    model_tag = "v3" if model_id == "v3_three_input_regression" else "v4"
    config["mean_model_id"] = model_id
    if spread_gate_mode != "":
        config["spread_gate_mode"] = spread_gate_mode
    config["run_suffix"] = f"{base_config['run_suffix']}_ab_{group_id}_{model_tag}"
    before = {p.name for p in RUNS_DIR.iterdir() if p.is_dir()}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(["python", str(MODULE_DIR / "run_backtest.py")], check=True)
    after = {p.name for p in RUNS_DIR.iterdir() if p.is_dir()}
    created = sorted(list(after - before))
    if len(created) != 1:
        raise ValueError(f"Expected exactly one created run dir, found: {created}")
    return RUNS_DIR / created[0]


def _summary_row(run_dir: Path, model_id: str) -> dict[str, Any]:
    """Read summary.json and normalize fields for comparison."""
    summary = json.loads((run_dir / "summary.json").read_text())
    return {
        "run_id": summary["run_id"],
        "model": model_id,
        "rmse": float(summary["rmse"]),
        "win_rate": float(summary["win_rate"]),
        "roi": float(summary["roi"]),
        "n_bets": int(summary["n_bets"]),
        "signal_rate": float(summary["signal_rate"]),
        "spread_context_active_fg3m": int(summary.get("spread_context_active_fg3m", 0)),
    }


def main() -> None:
    """Run v3/v4 A-B backtests and write consolidated comparison CSV."""
    args = parse_args()
    config_path = Path(args.config_path).expanduser().resolve()
    output_csv = Path(args.output_comparison_csv).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    base_text = config_path.read_text()
    base_config = yaml.safe_load(base_text)
    group_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = config_path.with_suffix(".yaml.ab_backup")
    shutil.copy2(config_path, backup_path)
    try:
        run_v3 = _run_with_model(
            base_config=base_config,
            config_path=config_path,
            model_id="v3_three_input_regression",
            group_id=group_id,
            spread_gate_mode=args.spread_gate_mode,
        )
        run_v4 = _run_with_model(
            base_config=base_config,
            config_path=config_path,
            model_id="v4_market_spread_regression",
            group_id=group_id,
            spread_gate_mode=args.spread_gate_mode,
        )
        if not args.skip_validation:
            subprocess.run(["python", str(VALIDATION_SCRIPT)], check=True)
    finally:
        config_path.write_text(base_text)
        if backup_path.exists():
            backup_path.unlink()

    rows = [
        _summary_row(run_dir=run_v3, model_id="v3_three_input_regression"),
        _summary_row(run_dir=run_v4, model_id="v4_market_spread_regression"),
    ]
    out = pd.DataFrame(rows)
    v3_row = out[out["model"] == "v3_three_input_regression"].iloc[0]
    v4_row = out[out["model"] == "v4_market_spread_regression"].iloc[0]
    deltas = pd.DataFrame(
        [
            {
                "delta_rmse_v4_minus_v3": float(v4_row["rmse"] - v3_row["rmse"]),
                "delta_win_rate_v4_minus_v3": float(v4_row["win_rate"] - v3_row["win_rate"]),
                "delta_roi_v4_minus_v3": float(v4_row["roi"] - v3_row["roi"]),
                "delta_n_bets_v4_minus_v3": int(v4_row["n_bets"] - v3_row["n_bets"]),
                "delta_signal_rate_v4_minus_v3": float(v4_row["signal_rate"] - v3_row["signal_rate"]),
            }
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    deltas.to_csv(output_csv.with_name("v3_v4_backtest_ab_deltas.csv"), index=False)
    print(f"group_id={group_id}")
    print(f"comparison_csv={output_csv}")
    print(out.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
