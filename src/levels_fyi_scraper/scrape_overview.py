"""
Daily snapshot scraper for Levels.fyi company overview metrics.
No auth required. Run daily to build a time series.

Usage:
  uv run python analysis/levels_scraper/scrape_overview.py
  uv run python analysis/levels_scraper/scrape_overview.py --companies snap meta google

Output: data/levels_overview_daily.parquet
  One row per company per day. Skips if today already logged (idempotent).

Signals captured:
  total_submissions     — all-time submission count (hiring velocity proxy)
  median_tc             — median total comp across all roles
  num_job_families      — breadth of hiring
  highest_tc            — top-end comp (talent magnet signal)
  lowest_tc             — floor comp
"""

import argparse
import json
import os
import re
import time
import pandas as pd
from datetime import date
from pathlib import Path
from scrapling.fetchers import Fetcher, StealthyFetcher

SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() != "false"

OUT = Path(__file__).parent / "data" / "levels_overview_daily.parquet"

COMPANIES = [
    "snap", "meta", "google", "microsoft", "nvidia", "apple", "amazon",
    "netflix", "uber", "lyft", "airbnb", "stripe", "palantir", "coinbase",
    "salesforce", "adobe", "oracle", "intel", "amd", "qualcomm",
    "linkedin", "pinterest", "reddit", "bytedance",
    "openai", "anthropic", "databricks", "snowflake", "cloudflare",
]

def fetch_overview(slug: str, stealth: bool = False) -> dict | None:
    try:
        url = f"https://www.levels.fyi/companies/{slug}/salaries/"
        if stealth:
            r = StealthyFetcher.fetch(url, verify=SSL_VERIFY)
        else:
            r = Fetcher.get(url, verify=SSL_VERIFY)
        if r.status != 200:
            print(f"  {slug:<20} HTTP {r.status} — skipping")
            return None
        html = str(r.html_content)
    except Exception as e:
        print(f"  {slug:<20} request error: {e}")
        return None

    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        print(f"  {slug:<20} no __NEXT_DATA__ — skipping")
        return None

    try:
        props = json.loads(m.group(1))["props"]["pageProps"]
    except (json.JSONDecodeError, KeyError):
        print(f"  {slug:<20} JSON parse error — skipping")
        return None

    overview = props.get("overview", [])
    total_submissions = sum(
        sum(t.get("count", 0) for t in jf.get("titles", []))
        for jf in overview
    )

    highest = props.get("highestPayingJobFamilyAndLevel") or {}
    lowest  = props.get("lowestPayingJobFamilyAndLevel") or {}

    return {
        "scrape_date":       date.today().isoformat(),
        "company_slug":      slug,
        "total_submissions": total_submissions,
        "median_tc":         props.get("medianAcrossAllJobFamilies"),
        "num_job_families":  len(overview),
        "highest_role":      highest.get("jobFamily"),
        "highest_tc":        highest.get("totalCompensation"),
        "lowest_role":       lowest.get("jobFamily"),
        "lowest_tc":         lowest.get("totalCompensation"),
    }


def load_existing() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame()
    df = pd.read_parquet(OUT)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"]).dt.date.astype(str)
    return df


def run(companies: list[str], stealth: bool = False) -> None:
    today = date.today().isoformat()
    print(f"Levels.fyi overview scrape — {today} — {len(companies)} companies{'  stealth' if stealth else ''}\n")

    existing = load_existing()
    already_done = set()
    if not existing.empty:
        already_done = set(
            existing[existing["scrape_date"] == today]["company_slug"].tolist()
        )
        if already_done:
            print(f"Already have today's data for: {', '.join(sorted(already_done))}")
            print()

    new_rows = []
    for slug in companies:
        if slug in already_done:
            continue
        row = fetch_overview(slug, stealth=stealth)
        if row:
            new_rows.append(row)
            tc_str = f"  median ${row['median_tc']:,.0f}" if row["median_tc"] else ""
            print(f"  {slug:<20} {row['total_submissions']:>5} subs{tc_str}")
        time.sleep(0.4)

    if not new_rows:
        print("\nNothing new to save.")
        return

    new_df = pd.DataFrame(new_rows)

    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["scrape_date", "company_slug"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)

    print(f"\nSaved {len(new_rows)} new rows → {OUT}")
    print(f"Total rows in file: {len(combined)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", default=COMPANIES)
    parser.add_argument("--stealth", action="store_true",
                        help="Use StealthyFetcher (headless browser, local only — run 'scrapling install' first)")
    args = parser.parse_args()
    run(args.companies, stealth=args.stealth)
