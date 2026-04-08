"""
Production entrypoint to run rebounds daily pipeline.

Context:
- Keeps production orchestration under src/nba_rebounds_modeling/00_research/scripts.
- Delegates to the canonical implementation in scripts/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    cmd = [sys.executable, "scripts/run_rebounds_daily_pipeline.py", *sys.argv[1:]]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
