"""
Validate individual submission scraping — one row per salary offer.
Proves we can get real submission records before building the submissions Lambda.

Uses no-auth mode (samples embedded in __NEXT_DATA__). Shows actual records:
  uuid, offer_date, level, location, base_salary, total_comp, yoe

Saves results to ~/Downloads/tmp/levels_fyi/submissions_YYYYMMDD.parquet

Usage:
  uv run python src/levels_fyi_scraper/validate_individual_submission_scrape.py
  uv run python src/levels_fyi_scraper/validate_individual_submission_scrape.py --companies snap meta
  uv run python src/levels_fyi_scraper/validate_individual_submission_scrape.py --show-rows 10
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SAVE_DIR = Path.home() / "Downloads" / "tmp" / "levels_fyi"

sys.path.insert(0, str(Path(__file__).parent))
from scrape_levels import scrape_company_noauth

SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() != "false"

COMPANIES = [
    "snap", "meta", "google", "microsoft", "nvidia", "apple", "amazon",
    "netflix", "uber", "lyft", "airbnb", "stripe", "palantir", "coinbase",
    "salesforce", "adobe", "oracle", "intel", "amd", "qualcomm",
    "linkedin", "pinterest", "reddit", "bytedance",
    "openai", "anthropic", "databricks", "snowflake", "cloudflare",
]

DISPLAY_COLS = ["company_slug", "job_family", "offer_date", "level", "location",
                "base_salary", "total_comp", "yoe"]


def run(companies: list[str], show_rows: int) -> None:
    print(f"Individual submission scrape validation — {len(companies)} companies   SSL_VERIFY={SSL_VERIFY}\n")

    all_rows = []
    summary = []

    for slug in companies:
        rows = scrape_company_noauth(slug)
        n = len(rows)
        ok = n > 0
        summary.append({"slug": slug, "n_records": n, "passed": ok})
        all_rows.extend(rows)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Company':<20} {'Records':>8}  Status")
    print(f"  {'─'*40}")
    for s in summary:
        icon = "✅" if s["passed"] else "❌"
        print(f"  {s['slug']:<20} {s['n_records']:>8}  {icon}")

    n_pass  = sum(s["passed"] for s in summary)
    total   = sum(s["n_records"] for s in summary)
    print(f"\n  {n_pass}/{len(summary)} companies returned records")
    print(f"  {total} individual submission records total")

    if not all_rows:
        print("\nNo records scraped.")
        return

    df = pd.DataFrame(all_rows)
    df["offer_date"] = pd.to_datetime(df["offer_date"], utc=True, errors="coerce")
    df = df.sort_values("offer_date", ascending=False)

    print(f"\n{'='*60}")
    print(f"  SAMPLE RECORDS (most recent {show_rows})")
    print(f"{'='*60}")

    display = df[DISPLAY_COLS].head(show_rows).copy()
    display["offer_date"] = display["offer_date"].dt.strftime("%Y-%m-%d")
    display["base_salary"] = display["base_salary"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")
    display["total_comp"]  = display["total_comp"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")
    display["yoe"]         = display["yoe"].apply(lambda v: f"{v:.0f}yr" if pd.notna(v) else "—")
    display["location"]    = display["location"].apply(lambda v: str(v)[:25] if pd.notna(v) else "—")

    print(display.to_string(index=False))

    print(f"\n{'='*60}")
    print("  FIELD COVERAGE")
    print(f"{'='*60}")
    for col in ["uuid", "offer_date", "level", "location", "base_salary", "total_comp", "yoe", "gender"]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:<20} {pct:>5.1f}% non-null")

    out = SAVE_DIR / f"submissions_{date.today().isoformat()}.parquet"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n  Saved {len(df)} rows → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", default=COMPANIES)
    parser.add_argument("--show-rows", type=int, default=20)
    args = parser.parse_args()
    run(args.companies, args.show_rows)
