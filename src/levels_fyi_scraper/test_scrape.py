"""
Scrapling integration test — validates that all 29 companies can be scraped.

Tests two layers:
  1. Overview  — total_submissions + median_tc from company salaries page
  2. Submissions — individual salary records from __NEXT_DATA__ samples (no auth needed)

Usage:
  uv run python src/levels_fyi_scraper/test_scrape.py
  uv run python src/levels_fyi_scraper/test_scrape.py --companies snap meta google
  uv run python src/levels_fyi_scraper/test_scrape.py --overview-only
"""

import argparse
import json
import os
import re
import time

from scrapling.fetchers import Fetcher

SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() != "false"

COMPANIES = [
    "snap", "meta", "google", "microsoft", "nvidia", "apple", "amazon",
    "netflix", "uber", "lyft", "airbnb", "stripe", "palantir", "coinbase",
    "salesforce", "adobe", "oracle", "intel", "amd", "qualcomm",
    "linkedin", "pinterest", "reddit", "bytedance",
    "openai", "anthropic", "databricks", "snowflake", "cloudflare",
]

BASE = "https://www.levels.fyi"


def fetch(url: str) -> dict | None:
    try:
        r = Fetcher.get(url, verify=SSL_VERIFY)
        if r.status != 200:
            return None
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', str(r.html_content), re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(1)).get("props", {}).get("pageProps", {})
    except Exception as e:
        print(f"  error: {e}")
        return None


def test_overview(companies: list[str]) -> list[dict]:
    print(f"\n{'='*70}")
    print(f"  OVERVIEW TEST — {len(companies)} companies")
    print(f"{'='*70}")

    results = []
    for slug in companies:
        props = fetch(f"{BASE}/companies/{slug}/salaries/")
        if not props:
            print(f"  {slug:<20} ❌ FAIL  (no data)")
            results.append({"slug": slug, "passed": False, "total_submissions": None, "median_tc": None})
            time.sleep(0.4)
            continue

        overview = props.get("overview", [])
        total = sum(sum(t.get("count", 0) for t in jf.get("titles", [])) for jf in overview)
        median = props.get("medianAcrossAllJobFamilies")

        ok = total > 0 and median is not None
        median_str = f"${median/1000:.0f}K" if median else "—"
        icon = "✅" if ok else "❌"
        print(f"  {slug:<20} {icon}  {total:>5} subs   median {median_str}")
        results.append({"slug": slug, "passed": ok, "total_submissions": total, "median_tc": median})
        time.sleep(0.4)

    n_pass = sum(r["passed"] for r in results)
    print(f"\n  Result: {n_pass}/{len(results)} passed")
    return results


def test_submissions(companies: list[str]) -> list[dict]:
    print(f"\n{'='*70}")
    print(f"  SUBMISSIONS TEST (no-auth samples) — {len(companies)} companies")
    print(f"{'='*70}")

    results = []
    for slug in companies:
        props = fetch(f"{BASE}/companies/{slug}/salaries/")
        if not props:
            print(f"  {slug:<20} ❌ FAIL  (no page data)")
            results.append({"slug": slug, "passed": False, "sample_count": 0})
            time.sleep(0.4)
            continue

        overview = props.get("overview", [])
        job_families = [{"name": jf.get("name", ""), "slug": jf.get("slug", "")} for jf in overview]

        total_samples = 0
        for jf in job_families[:3]:
            jf_props = fetch(f"{BASE}/companies/{slug}/salaries/{jf['slug']}/")
            if not jf_props:
                time.sleep(0.4)
                continue
            for bucket in jf_props.get("averages", []):
                total_samples += len(bucket.get("samples", []))
            time.sleep(0.4)

        ok = total_samples > 0
        icon = "✅" if ok else "❌"
        print(f"  {slug:<20} {icon}  {total_samples:>3} individual submission samples (first 3 job families)")
        results.append({"slug": slug, "passed": ok, "sample_count": total_samples})
        time.sleep(0.4)

    n_pass = sum(r["passed"] for r in results)
    print(f"\n  Result: {n_pass}/{len(results)} passed")
    return results


def main(companies: list[str], overview_only: bool) -> None:
    print(f"Scrapling test — {len(companies)} companies   SSL_VERIFY={SSL_VERIFY}")

    overview_results = test_overview(companies)

    if not overview_only:
        sub_results = test_submissions(companies)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    ov_pass = sum(r["passed"] for r in overview_results)
    print(f"  Overview:     {ov_pass}/{len(overview_results)} passed")
    if not overview_only:
        sub_pass = sum(r["passed"] for r in sub_results)
        print(f"  Submissions:  {sub_pass}/{len(sub_results)} passed")

    ov_fails = [r["slug"] for r in overview_results if not r["passed"]]
    if ov_fails:
        print(f"\n  Overview failures:    {', '.join(ov_fails)}")
    if not overview_only:
        sub_fails = [r["slug"] for r in sub_results if not r["passed"]]
        if sub_fails:
            print(f"  Submission failures:  {', '.join(sub_fails)}")

    all_passed = ov_pass == len(overview_results)
    if not overview_only:
        all_passed = all_passed and sub_pass == len(sub_results)
    print(f"\n  {'✅ ALL PASS' if all_passed else '❌ FAILURES — see above'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", default=COMPANIES)
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()
    main(args.companies, args.overview_only)
