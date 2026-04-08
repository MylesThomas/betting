"""
Settle rebounds run artifacts with realized outcomes.

Context:
- Clean production-facing name.
- Delegates to existing settlement implementation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    cmd = [sys.executable, "scripts/rebounds_settle_runs.py", *sys.argv[1:]]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
