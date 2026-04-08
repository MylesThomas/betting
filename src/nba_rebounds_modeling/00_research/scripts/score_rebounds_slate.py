"""
Score rebounds slate with trained models.

Context:
- Clean production-facing name.
- Delegates to existing scoring implementation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    cmd = [
        sys.executable,
        "src/nba_rebounds_modeling/00_research/scripts/prod_score_rebounds_slate.py",
        *sys.argv[1:],
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
