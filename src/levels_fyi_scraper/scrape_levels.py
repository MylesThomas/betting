"""
Levels.fyi multi-company salary submission scraper.

Two modes:
  --mode noauth   Extract samples embedded in __NEXT_DATA__ (no credentials
                  needed, but only captures roles with low submission volume,
                  typically < ~100 submissions/yr per level).

  --mode auth     Call the Levels.fyi salary API with a Bearer token.
                  Gets full historical data for any company/role.
                  Requires LEVELS_EMAIL + LEVELS_PASSWORD env vars (or a
                  pre-cached token at data/.levels_token via levels_auth.py).

Usage:
  # No-auth daily scrape (Lambda-safe):
  uv run python analysis/levels_scraper/scrape_levels.py --companies snap meta google

  # Auth-based historical backfill (run locally):
  export LEVELS_EMAIL=you@email.com LEVELS_PASSWORD=yourpass
  uv run python analysis/levels_scraper/scrape_levels.py --mode auth --companies snap meta google

Output: data/levels_submissions.parquet (appended, deduped by uuid)
"""

import argparse
import json
import os
import re
import sys
import time
import requests
import pandas as pd
from datetime import date
from pathlib import Path
from scrapling.fetchers import Fetcher, StealthyFetcher

SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() != "false"

# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_COMPANIES = [
    "snap", "meta", "google", "microsoft", "nvidia", "apple", "amazon"
]

LEVELS_BASE = "https://www.levels.fyi"
API_SEARCH  = f"{LEVELS_BASE}/v3/salary/search"
API_SUBS    = f"{LEVELS_BASE}/v1/level-submissions"
AUTH_FILE   = Path(__file__).parent.parent / "sentiment_snap" / "data" / ".levels_token"
OUT         = Path(__file__).parent / "data" / "levels_submissions.parquet"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── shared ────────────────────────────────────────────────────────────────────

def normalize(raw: dict, company_slug: str, job_family: str) -> dict | None:
    uuid = raw.get("uuid") or raw.get("id")
    offer_date = raw.get("offerDate") or raw.get("offer_date") or raw.get("createdAt")
    if not uuid or not offer_date:
        return None
    return {
        "uuid":         uuid,
        "scraped_at":   date.today().isoformat(),
        "company_slug": company_slug,
        "job_family":   job_family,
        "offer_date":   offer_date,
        "level":        raw.get("level") or raw.get("levelName"),
        "location":     raw.get("location"),
        "base_salary":  raw.get("baseSalary"),
        "total_comp":   raw.get("totalCompensation") or raw.get("totalComp"),
        "stock_grant":  raw.get("avgAnnualStockGrantValue") or raw.get("stockGrant"),
        "bonus":        raw.get("avgAnnualBonusValue") or raw.get("annualBonus") or raw.get("bonus"),
        "yoe":          raw.get("yearsOfExperience") or raw.get("yoe"),
        "yac":          raw.get("yearsAtCompany") or raw.get("yac"),
        "gender":       raw.get("gender"),
        "focus_tag":    raw.get("focusTag"),
    }


def save(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        print("Nothing new to save.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["offer_date"] = pd.to_datetime(df["offer_date"], utc=True, errors="coerce")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        existing["offer_date"] = pd.to_datetime(existing["offer_date"], utc=True, errors="coerce")
        before = len(existing)
        combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset="uuid")
        new_rows = len(combined) - before
        print(f"\nMerged: {before} existing + {new_rows} net new = {len(combined)} total")
    else:
        combined = df
        print(f"\nNew file: {len(combined)} submissions")

    combined = combined.sort_values("offer_date").reset_index(drop=True)
    combined.to_parquet(OUT, index=False)
    print(f"Saved → {OUT}")
    _print_summary(combined)
    return combined


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print(f"\nDate range: {df['offer_date'].min()} → {df['offer_date'].max()}")
    print("\nSubmissions by company:")
    print(df.groupby("company_slug").size().sort_values(ascending=False).to_string())


# ── no-auth mode ──────────────────────────────────────────────────────────────

def fetch_next_data(url: str, stealth: bool = False) -> dict | None:
    try:
        if stealth:
            r = StealthyFetcher.fetch(url, verify=SSL_VERIFY)
        else:
            r = Fetcher.get(url, verify=SSL_VERIFY)
        if r.status != 200:
            print(f"    HTTP {r.status} {url}")
            return None
        html = str(r.html_content)
    except Exception as e:
        print(f"    GET failed {url}: {e}")
        return None
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def get_job_families_noauth(company_slug: str, stealth: bool = False) -> list[dict]:
    data = fetch_next_data(f"{LEVELS_BASE}/companies/{company_slug}/salaries/", stealth=stealth)
    if not data:
        return []
    overview = data.get("props", {}).get("pageProps", {}).get("overview", [])
    return [{"name": jf.get("name", jf.get("slug", "")), "slug": jf.get("slug", "")} for jf in overview]


def scrape_company_noauth(company_slug: str, stealth: bool = False) -> list[dict]:
    """Scrape embedded __NEXT_DATA__ samples for one company.
    Only captures roles with count_last_12_months < ~100."""
    print(f"\n[{company_slug.upper()}] (no-auth{'  stealth' if stealth else ''})")
    job_families = get_job_families_noauth(company_slug, stealth=stealth)
    if not job_families:
        print(f"  Could not fetch job families")
        return []
    print(f"  {len(job_families)} job families")

    all_rows = []
    for jf in job_families:
        url = f"{LEVELS_BASE}/companies/{company_slug}/salaries/{jf['slug']}/"
        data = fetch_next_data(url, stealth=stealth)
        if not data:
            time.sleep(0.5)
            continue

        averages = data.get("props", {}).get("pageProps", {}).get("averages", [])
        jf_rows = []
        for bucket in averages:
            for sample in bucket.get("samples", []):
                norm = normalize(sample, company_slug, jf["name"])
                if norm:
                    jf_rows.append(norm)

        if jf_rows:
            print(f"  {jf['slug']:<40} {len(jf_rows):>3} samples")
        all_rows.extend(jf_rows)
        time.sleep(0.4)

    print(f"  → {len(all_rows)} total samples")
    return all_rows


# ── auth mode ─────────────────────────────────────────────────────────────────

def get_token() -> str:
    """Get a valid Bearer token, using cache or Playwright login."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "sentiment_snap"))
    try:
        from levels_auth import get_token as _get_token
        return _get_token()
    except ImportError:
        raise RuntimeError("levels_auth.py not found — check path")


def auth_headers(token: str) -> dict:
    return {
        **HEADERS,
        "Authorization": f"Bearer {token}",
        "Origin": LEVELS_BASE,
        "Referer": f"{LEVELS_BASE}/companies/snap/salaries/",
        "Accept": "application/json",
    }


def fetch_submissions_page(token: str, company: str, offset: int, limit: int = 100) -> dict | None:
    """Fetch one page of submissions from the Levels.fyi API."""
    try:
        r = requests.get(
            API_SEARCH,
            headers=auth_headers(token),
            params={"company": company, "limit": limit, "offset": offset},
            timeout=20,
        )
        if r.status_code == 403:
            print(f"  Auth failed (403) — token may be expired")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  API error at offset={offset}: {e}")
        return None


def scrape_company_auth(company_slug: str, token: str) -> list[dict]:
    """Fetch ALL submissions for a company via the authenticated API."""
    # Levels.fyi uses display name in the API, not the slug
    company_display = company_slug.replace("-", " ").title()
    special = {
        "snap": "Snap", "meta": "Meta", "google": "Google",
        "microsoft": "Microsoft", "nvidia": "NVIDIA",
        "apple": "Apple", "amazon": "Amazon",
    }
    company_name = special.get(company_slug, company_display)

    print(f"\n[{company_slug.upper()}] (auth — querying as '{company_name}')")

    all_rows = []
    offset = 0
    limit  = 100

    while True:
        data = fetch_submissions_page(token, company_name, offset, limit)
        if not data:
            break

        # Handle different response shapes
        items = data if isinstance(data, list) else (
            data.get("submissions") or data.get("results") or data.get("data") or []
        )

        if not items:
            break

        batch = []
        for item in items:
            # job family may be nested
            jf = (item.get("jobFamily") or item.get("job_family") or
                  item.get("jobFamilyName") or "Unknown")
            norm = normalize(item, company_slug, jf)
            if norm:
                batch.append(norm)

        all_rows.extend(batch)
        print(f"  offset={offset:>5}  got {len(items):>3}  total so far: {len(all_rows)}")

        if len(items) < limit:
            break
        offset += limit
        time.sleep(0.3)

    print(f"  → {len(all_rows)} total submissions")
    return all_rows


# ── entrypoint ────────────────────────────────────────────────────────────────

def main(companies: list[str], mode: str, stealth: bool = False) -> None:
    print(f"Levels.fyi scraper — {date.today()} — mode={mode}{'  stealth' if stealth else ''} — companies: {', '.join(companies)}")

    all_rows = []

    if mode == "auth":
        token = get_token()
        for slug in companies:
            all_rows.extend(scrape_company_auth(slug, token))
    else:
        for slug in companies:
            all_rows.extend(scrape_company_noauth(slug, stealth=stealth))

    save(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", default=DEFAULT_COMPANIES)
    parser.add_argument("--mode", choices=["noauth", "auth"], default="noauth")
    parser.add_argument("--stealth", action="store_true",
                        help="Use StealthyFetcher (headless browser, local only — run 'scrapling install' first)")
    args = parser.parse_args()
    main(args.companies, args.mode, stealth=args.stealth)
