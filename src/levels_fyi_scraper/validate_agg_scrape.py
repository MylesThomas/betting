"""
Validate aggregate overview scraping — one row per company per day.
Proves the data that goes to s3://levels-fyi-mt/overview/daily.parquet is correct.

Usage:
  uv run python src/levels_fyi_scraper/validate_agg_scrape.py
  uv run python src/levels_fyi_scraper/validate_agg_scrape.py --companies snap meta google
"""

import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
from scrapling.fetchers import Fetcher

SAVE_DIR = Path.home() / "Downloads" / "tmp" / "levels_fyi"

SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() != "false"
BASE = "https://www.levels.fyi"

COMPANIES = [
    "snap", "meta", "google", "microsoft", "nvidia", "apple", "amazon",
    "netflix", "uber", "lyft", "airbnb", "stripe", "palantir", "coinbase",
    "salesforce", "adobe", "oracle", "intel", "amd", "qualcomm",
    "linkedin", "pinterest", "reddit", "bytedance",
    "openai", "anthropic", "databricks", "snowflake", "cloudflare",
]


def fetch_overview(slug: str) -> dict | None:
    try:
        r = Fetcher.get(f"{BASE}/companies/{slug}/salaries/", verify=SSL_VERIFY)
        if r.status != 200:
            return None
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', str(r.html_content), re.DOTALL)
        if not m:
            return None
        props = json.loads(m.group(1))["props"]["pageProps"]
    except Exception as e:
        print(f"  {slug:<20} error: {e}")
        return None

    overview = props.get("overview", [])
    total = sum(sum(t.get("count", 0) for t in jf.get("titles", [])) for jf in overview)
    median = props.get("medianAcrossAllJobFamilies")
    highest = (props.get("highestPayingJobFamilyAndLevel") or {})
    lowest  = (props.get("lowestPayingJobFamilyAndLevel") or {})

    return {
        "company_slug":      slug,
        "total_submissions": total,
        "median_tc":         median,
        "num_job_families":  len(overview),
        "highest_role":      highest.get("jobFamily"),
        "highest_tc":        highest.get("totalCompensation"),
        "lowest_role":       lowest.get("jobFamily"),
        "lowest_tc":         lowest.get("totalCompensation"),
    }


def run(companies: list[str]) -> None:
    print(f"Aggregate scrape validation — {len(companies)} companies   SSL_VERIFY={SSL_VERIFY}\n")
    print(f"{'Company':<20} {'Subs':>6}  {'Median TC':>10}  {'Job Families':>13}  {'Highest Role':<30}  Status")
    print("─" * 100)

    passed = 0
    saved_rows = []
    for slug in companies:
        row = fetch_overview(slug)
        if not row or not row["total_submissions"] or not row["median_tc"]:
            print(f"  {slug:<20} ❌ FAIL")
            time.sleep(0.4)
            continue

        median_str   = f"${row['median_tc']/1000:.0f}K"
        highest_str  = row["highest_role"] or "—"
        print(f"  {slug:<20} {row['total_submissions']:>6}  {median_str:>10}  {row['num_job_families']:>13}  {highest_str:<30}  ✅")
        saved_rows.append(row)
        passed += 1
        time.sleep(0.4)

    print("─" * 100)
    print(f"\n{passed}/{len(companies)} passed")

    if saved_rows:
        out = SAVE_DIR / f"overview_{date.today().isoformat()}.parquet"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(saved_rows).to_parquet(out, index=False)
        print(f"Saved {len(saved_rows)} rows → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", default=COMPANIES)
    args = parser.parse_args()
    run(args.companies)
