"""
Rebuild the MLB Batter Home Runs feature spine on S3.

  1. Fetch latest Statcast batting data via pybaseball for current season
  2. Merge with market lines (from S3 market_raw)
  3. Recompute rolling features (hr_roll_*, ab_roll_*, opp_hr_rate_career)
  4. Save to S3

The saved logistic model is NOT retrained — it operates on the updated spine features.
Run via EventBridge at 8:30 AM ET daily (before the pipeline at 9:00 AM).

Usage:
  python src/mlb_batter_home_runs_modeling/scripts/update_spine.py
  python src/mlb_batter_home_runs_modeling/scripts/update_spine.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ET = ZoneInfo("America/New_York")


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    today   = datetime.now(ET).strftime("%Y-%m-%d")

    output_lines = []

    def log(msg: str) -> None:
        print(msg)
        output_lines.append(msg)

    log(f"MLB Batter Home Runs spine update | {today}")

    build_script = scripts / "20260801_build_spine.py"
    if build_script.exists() and not args.dry_run:
        log("Running spine rebuild (fetches Statcast + rebuilds rolling features)...")
        result = subprocess.run(
            [sys.executable, str(build_script)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        log(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode != 0:
            log(f"WARN: spine build exited {result.returncode}: {result.stderr[-500:]}")
    elif args.dry_run:
        log("Dry run — skipping spine rebuild")
    else:
        log("Spine build script not found — skipping rebuild")

    out = "\n".join(output_lines)
    return out


if __name__ == "__main__":
    main()
