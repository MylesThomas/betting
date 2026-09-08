"""
Scrape individual Levels.fyi salary submissions for Snap — no auth required.

The __NEXT_DATA__ JSON embedded in each page contains individual salary
submissions with full timestamps (offerDate). We iterate through all job
families and levels to build a complete historical dataset.

What's available without auth:
  - Individual submissions with offerDate (via __NEXT_DATA__ page embedding)
  - All job families and levels for Snap
  - Historical coverage back as far as Levels.fyi records go

Strategy:
  1. Scrape overview page → get all job family slugs
  2. For each job family, scrape the main job family page → extract submissions
  3. For each job family, iterate through level pages → extract more submissions
  4. Deduplicate by uuid, save to parquet

Output:
  data/levels_snap_submissions.parquet  — one row per individual submission
  data/levels_snap_snapshot.json        — latest overview for inspection
"""

import json
import re
import time
import requests
import pandas as pd
from datetime import date
from pathlib import Path

COMPANY_SLUG = "snap"
BASE_URL = f"https://www.levels.fyi/companies/{COMPANY_SLUG}/salaries"

OUT = Path(__file__).parent / "data" / "levels_snap_submissions.parquet"
OUT_SNAPSHOT = Path(__file__).parent / "data" / "levels_snap_snapshot.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_next_data(url: str) -> dict | None:
    """Fetch a Levels.fyi page and extract the __NEXT_DATA__ JSON blob."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  GET error {url}: {e}")
        return None

    # Prefer the explicit __NEXT_DATA__ script tag
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError as e:
            print(f"  JSON parse error on {url}: {e}")
            return None

    # Fallback: largest <script> block
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", r.text, re.DOTALL)
    if not scripts:
        return None
    biggest = max(scripts, key=len)
    try:
        return json.loads(biggest)
    except json.JSONDecodeError:
        return None


def extract_submissions(next_data: dict) -> list[dict]:
    """Pull individual salary submissions out of a __NEXT_DATA__ blob."""
    submissions = []

    props = next_data.get("props", {}).get("pageProps", {})

    # Job-family page: submissions in "averages" array
    for key in ("averages", "samples", "salaries", "submissions"):
        items = props.get(key, [])
        if items and isinstance(items, list) and isinstance(items[0], dict):
            submissions.extend(items)

    # Level page: may nest under "levelData" or similar
    level_data = props.get("levelData", {})
    if isinstance(level_data, dict):
        for key in ("averages", "samples", "salaries"):
            items = level_data.get(key, [])
            if items and isinstance(items, list):
                submissions.extend(items)

    return submissions


def get_job_family_slugs() -> list[dict]:
    """Scrape the overview page and return all job family names + slugs."""
    url = f"{BASE_URL}/"
    data = fetch_next_data(url)
    if not data:
        return []

    props = data.get("props", {}).get("pageProps", {})
    overview = props.get("overview", [])

    result = []
    for jf in overview:
        slug = jf.get("slug") or jf.get("name", "").lower().replace(" ", "-")
        result.append({
            "name": jf.get("name", slug),
            "slug": slug,
            "titles": jf.get("titles", []),
        })
    return result


def get_level_slugs(jf_slug: str, jf_titles: list[dict]) -> list[str]:
    """
    Return level slugs for a job family. Try to read from titles data first;
    fall back to scraping the job family page for a levelBreakdown key.
    """
    # Title slugs often map to level page paths (e.g. "l3", "senior", etc.)
    slugs = []
    for t in jf_titles:
        slug = t.get("slug") or t.get("levelSlug")
        if slug:
            slugs.append(slug)

    if not slugs:
        # Try fetching the job family page and looking for levelBreakdown
        url = f"{BASE_URL}/{jf_slug}/"
        data = fetch_next_data(url)
        if data:
            props = data.get("props", {}).get("pageProps", {})
            for key in ("levelBreakdown", "levels"):
                for lvl in props.get(key, []):
                    s = lvl.get("slug") or lvl.get("levelSlug")
                    if s:
                        slugs.append(s)

    return slugs


def scrape_job_family(jf: dict) -> list[dict]:
    """Scrape main job-family page + all level sub-pages. Return raw submissions."""
    slug = jf["slug"]
    all_subs = []

    # 1. Job-family landing page
    url = f"{BASE_URL}/{slug}/"
    data = fetch_next_data(url)
    if data:
        subs = extract_submissions(data)
        print(f"    {slug}/ → {len(subs)} submissions")
        all_subs.extend(subs)
    time.sleep(0.5)

    # 2. Per-level pages
    level_slugs = get_level_slugs(slug, jf.get("titles", []))
    for lvl_slug in level_slugs:
        lvl_url = f"{BASE_URL}/{slug}/levels/{lvl_slug}/"
        lvl_data = fetch_next_data(lvl_url)
        if lvl_data:
            subs = extract_submissions(lvl_data)
            if subs:
                print(f"    {slug}/levels/{lvl_slug} → {len(subs)} submissions")
                all_subs.extend(subs)
        time.sleep(0.4)

    return all_subs


def normalize_submission(raw: dict, job_family: str, scraped_at: str) -> dict | None:
    """Flatten a raw submission dict into a consistent schema."""
    uuid = raw.get("uuid") or raw.get("id")
    offer_date = raw.get("offerDate") or raw.get("offer_date") or raw.get("createdAt")
    if not uuid or not offer_date:
        return None

    return {
        "uuid": uuid,
        "scraped_at": scraped_at,
        "job_family": job_family,
        "offer_date": offer_date,
        "company": raw.get("company", COMPANY_SLUG),
        "level": raw.get("level") or raw.get("levelName"),
        "location": raw.get("location"),
        "base_salary": raw.get("baseSalary"),
        "total_comp": raw.get("totalCompensation") or raw.get("totalComp"),
        "stock_grant": raw.get("avgAnnualStockGrantValue") or raw.get("stockGrant"),
        "bonus": raw.get("annualBonus") or raw.get("bonus"),
        "yoe": raw.get("yearsOfExperience") or raw.get("yoe"),
        "yac": raw.get("yearsAtCompany") or raw.get("yac"),
        "gender": raw.get("gender"),
    }


def main():
    today = date.today().isoformat()
    print(f"Scraping Levels.fyi / {COMPANY_SLUG} — {today}")
    print("Fetching job family list...")

    job_families = get_job_family_slugs()
    if not job_families:
        print("Could not retrieve job families — aborting.")
        return
    print(f"Found {len(job_families)} job families\n")

    all_raw: list[dict] = []

    for jf in job_families:
        print(f"  [{jf['name']}]")
        subs = scrape_job_family(jf)
        for raw in subs:
            norm = normalize_submission(raw, jf["name"], today)
            if norm:
                all_raw.append(norm)
        print()

    if not all_raw:
        print("No submissions extracted — check page structure.")
        return

    df = pd.DataFrame(all_raw)
    df["offer_date"] = pd.to_datetime(df["offer_date"], utc=True, errors="coerce")

    # Deduplicate by uuid
    before = len(df)
    df = df.drop_duplicates(subset="uuid").reset_index(drop=True)
    print(f"Deduped: {before} → {len(df)} unique submissions")

    # Merge with any existing data
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        existing["offer_date"] = pd.to_datetime(existing["offer_date"], utc=True, errors="coerce")
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="uuid").reset_index(drop=True)
        print(f"Merged with existing: {len(existing)} → {len(combined)} total rows")
    else:
        combined = df

    combined = combined.sort_values("offer_date").reset_index(drop=True)
    combined.to_parquet(OUT, index=False)
    print(f"\nSaved {len(combined):,} submissions → {OUT}")

    # Summary
    if "offer_date" in combined.columns and combined["offer_date"].notna().any():
        print(f"Date range: {combined['offer_date'].min().date()} → {combined['offer_date'].max().date()}")

    print(f"\nSubmissions by job family:")
    counts = combined.groupby("job_family").size().sort_values(ascending=False)
    for jf, n in counts.items():
        print(f"  {jf:<35} {n:>4}")

    # Save snapshot JSON for inspection
    OUT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SNAPSHOT, "w") as f:
        json.dump({"scraped_at": today, "total": len(combined), "job_families": counts.to_dict()}, f, indent=2)


if __name__ == "__main__":
    main()
