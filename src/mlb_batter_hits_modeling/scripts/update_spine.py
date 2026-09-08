"""
Rebuild / update the MLB Batter Hits spine on S3.

  1. Fetch latest Statcast batting data via pybaseball for current season
  2. Merge with market lines (from S3 market_raw)
  3. Recompute rolling features
  4. Re-train OLS + logistic models on updated data
  5. Upload spine + models to S3

Run via EventBridge at 8:30 AM ET daily (before the pipeline at 9:00 AM).

Usage:
  python src/mlb_batter_hits_modeling/scripts/update_spine.py
  python src/mlb_batter_hits_modeling/scripts/update_spine.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ET          = ZoneInfo("America/New_York")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg     = _load_config()
    scripts = Path(__file__).resolve().parent
    today   = datetime.now(ET).strftime("%Y-%m-%d")

    output_lines = []

    def log(msg: str) -> None:
        print(msg)
        output_lines.append(msg)

    log(f"MLB Batter Hits spine update | {today}")

    # Step 1: Rebuild spine (runs build_spine.py with --rebuild flag if available)
    build_script = scripts / "20260727_build_spine.py"
    if build_script.exists():
        log("Running spine rebuild...")
        result = subprocess.run(
            [sys.executable, str(build_script)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        log(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode != 0:
            log(f"WARN: spine build exited {result.returncode}: {result.stderr[-500:]}")
    else:
        log("Spine build script not found — skipping rebuild")

    # Step 2: Re-train models and upload
    train_script = scripts / "train_model.py"
    if train_script.exists() and not args.dry_run:
        log("Re-training models and uploading to S3...")
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        log(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            log(f"WARN: train_model exited {result.returncode}: {result.stderr[-500:]}")

    log(f"Spine update complete: {today}")
    return "\n".join(output_lines)


if __name__ == "__main__":
    output = main()
    print(output)
