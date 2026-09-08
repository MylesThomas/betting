"""
Quick test: verify a Levels.fyi Bearer token works against the salary API.

Usage:
  export LEVELS_BEARER="eyJhbGciOi..."   # paste your token here (no "Bearer " prefix)
  uv run python analysis/sentiment_snap/test_levels_auth.py
"""

import os
import json
import requests

token = os.environ.get("LEVELS_BEARER")
if not token:
    raise SystemExit(
        "Set LEVELS_BEARER env var.\n"
        "Get it from: DevTools → Network → any /v3/ request → Authorization header"
    )

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0",
    "Authorization": f"Bearer {token}",
    "Referer": "https://www.levels.fyi/companies/snap/salaries/",
    "Origin": "https://www.levels.fyi",
    "Accept": "application/json",
}

endpoints = [
    ("/v3/salary/search", {"company": "Snap", "limit": 3}),
    ("/v1/level-submissions",  {"company": "Snap", "limit": 3}),
    ("/v2/salary",             {"company": "Snap", "limit": 3}),
]

for path, params in endpoints:
    url = f"https://www.levels.fyi{path}"
    r = requests.get(url, headers=headers, params=params, timeout=15)
    print(f"\n{path}: {r.status_code}")
    if r.status_code == 200:
        try:
            data = r.json()
            print(json.dumps(data, indent=2)[:600])
        except Exception:
            print(r.text[:300])
    else:
        print(r.text[:200])
