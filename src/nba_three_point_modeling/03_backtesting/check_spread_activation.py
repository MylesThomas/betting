"""
Check spread activation state for the latest or specified run.

Context:
- Fast run audit utility to answer "is spread active?" without opening code.
- Reads summary/manifest/target_feature_promotion artifacts from one run folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _resolve_run_dir(runs_dir: Path, run_id: str | None) -> Path:
    if run_id is not None:
        run_dir = runs_dir / run_id
    else:
        run_dirs = sorted(
            [p for p in runs_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        run_dir = run_dirs[0]
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print spread gate mode, manifest feature activation, and FG3M gate row."
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id folder name.")
    args = parser.parse_args()

    runs_dir = Path(__file__).resolve().parent / "runs"
    run_dir = _resolve_run_dir(runs_dir=runs_dir, run_id=args.run_id)
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    promotion_path = run_dir / "target_feature_promotion.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary artifact: {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest artifact: {manifest_path}")
    if not promotion_path.exists():
        raise FileNotFoundError(f"Missing promotion artifact: {promotion_path}")

    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    promotion = pd.read_csv(promotion_path)
    fg3m = promotion[promotion["target"] == "FG3M"].iloc[0].to_dict()

    print(f"run_id={summary['run_id']}")
    print(f"spread_gate_mode={summary['spread_gate_mode']}")
    print(f"spread_context_enabled={summary['spread_context_enabled']}")
    print(f"spread_context_active_fg3m={summary['spread_context_active_fg3m']}")
    print("feature_manifest:")
    print(json.dumps(manifest["feature_manifest"], indent=2))
    print("fg3m_gate_row:")
    print(json.dumps(fg3m, indent=2, default=str))
    final_state = "active" if int(summary["spread_context_active_fg3m"]) == 1 else "inactive"
    print(f"final_activation_state={final_state}")


if __name__ == "__main__":
    main()
