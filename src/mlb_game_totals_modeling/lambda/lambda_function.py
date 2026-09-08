"""
MLB Game Totals Lambda orchestrator.

Modes (set via EventBridge payload {"mode": "..."}):
  pipeline  — Fetch live totals, train model, score, email bets        (10:00am ET daily)
  settle    — Settle yesterday's bets, update cumulative history        (11:00am ET daily)
  combined  — Settle yesterday + run today's pipeline → ONE launch      (manually)

Strategy: UNDER 9.5 · edge > 0 (p_model_under > raw_prob_under)
OOS ROI: +8.45% benchmark (2025+2026, n=2,324) · +13.65% with model edge>0 (n=383)
Model: Ridge(alpha=50) regression + Method C per-line calibration · Re-trained daily

Env vars:
  ODDS_API_KEY           (required for pipeline / combined modes)
  SES_SOURCE             (verified SES sender)
  SES_TO                 (comma-separated recipients)
  SNS_TOPIC_ARN          (optional SNS notifications)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _run_script(script_name: str, args: list[str] | None = None) -> int:
    scripts_dir = Path(__file__).resolve().parent / "src" / "mlb_game_totals_modeling" / "scripts"
    script_path = scripts_dir / script_name
    cmd = [sys.executable, str(script_path)] + (args or [])
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def handler(event: dict, context) -> dict:
    mode    = event.get("mode", "pipeline")
    gameday = event.get("gameday") or datetime.now(ET).strftime("%Y-%m-%d")
    yesterday = (datetime.now(ET).date() - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"MLB Game Totals Lambda | mode={mode} | gameday={gameday}")

    if mode == "pipeline":
        rc = _run_script("run_pipeline.py", ["--gameday", gameday])
        return {"statusCode": 200 if rc == 0 else 500, "mode": mode, "gameday": gameday}

    elif mode == "settle":
        settle_day = event.get("gameday", yesterday)
        rc = _run_script("settle_game_totals.py", ["--gameday", settle_day])
        return {"statusCode": 200 if rc == 0 else 500, "mode": mode, "gameday": settle_day}

    elif mode == "combined":
        # Settle yesterday first, then run today's pipeline
        print(f"\n--- SETTLE {yesterday} ---")
        rc1 = _run_script("settle_game_totals.py", ["--gameday", yesterday])
        print(f"\n--- PIPELINE {gameday} ---")
        rc2 = _run_script("run_pipeline.py", ["--gameday", gameday])
        return {
            "statusCode": 200 if (rc1 == 0 and rc2 == 0) else 500,
            "mode": mode,
            "settle_gameday": yesterday,
            "pipeline_gameday": gameday,
        }

    else:
        print(f"Unknown mode: {mode}")
        return {"statusCode": 400, "error": f"Unknown mode: {mode}"}
