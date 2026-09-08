"""
Lambda handler for the MLB total bases prop snapshot pipeline.

Invokes snapshot_props.main() and returns its result dict.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add repo root to path so snapshot_props can be imported
LAMBDA_TASK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LAMBDA_TASK_ROOT.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mlb_total_bases_modeling.scripts import snapshot_props


def lambda_handler(event, context):
    result = snapshot_props.main()
    print(json.dumps(result))
    return result
