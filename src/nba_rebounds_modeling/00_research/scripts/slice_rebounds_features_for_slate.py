"""
Slice rebounds features for a slate date.

Context:
- Clean production-facing name.
- Delegates to existing implementation script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    cmd = [
        sys.executable,
        "src/nba_rebounds_modeling/00_research/scripts/prod_slice_rebounds_features.py",
        *sys.argv[1:],
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
