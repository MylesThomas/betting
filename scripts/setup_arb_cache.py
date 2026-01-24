"""
Quick-start script to build arb caches.

This script builds the initial cache for NBA and NFL arb dashboards.
Run this once, then schedule build_arb_cache.py to run daily.

Usage:
    python scripts/setup_arb_cache.py
"""

import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_cache_builder():
    """Run the cache builder for both sports."""
    print("=" * 70)
    print("  ARB CACHE SETUP")
    print("=" * 70)
    print()
    print("Building initial cache for NBA and NFL...")
    print()
    
    # Run cache builder with initial build flag
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_arb_cache.py"),
            "--sport", "all",
            "--file-type", "parquet",
            "--initial-cache-create", "true"
        ],
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print()
        print("❌ Cache build failed. Check error messages above.")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("  ✅ SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Start your dashboard:")
    print("     cd streamlit_app && streamlit run app.py")
    print()
    print("  2. Enjoy 10-50x faster loading times!")
    print()
    print("  3. Schedule daily cache rebuilds:")
    print("     - Option A: Add to cron (macOS/Linux):")
    print("       0 2 * * * cd {} && python scripts/build_arb_cache.py --sport all --file-type parquet --initial-cache-create false".format(PROJECT_ROOT))
    print()
    print("     - Option B: Run manually when needed:")
    print("       python scripts/build_arb_cache.py --sport all --file-type parquet --initial-cache-create false")
    print()
    print("     - Option C: Set up AWS Lambda")
    print()
    print("Documentation:")
    print("  See docstring in scripts/build_arb_cache.py for full details")
    print()


if __name__ == '__main__':
    run_cache_builder()
