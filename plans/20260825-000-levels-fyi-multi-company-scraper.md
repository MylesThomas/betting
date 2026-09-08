LEVELS.FYI MULTI-COMPANY SALARY SCRAPER
========================================
Goal: build a historical + daily-forward time series of individual salary
submissions from Levels.fyi for major tech companies, stored in S3, fed by
a Lambda that runs daily.

HYPOTHESIS CONTEXT
------------------
Salary submission velocity and comp quality on Levels.fyi are forward-looking
signals on company health. High comp + rising submission count → talent
attraction is strong → precedes stock outperformance. Lagging comp → decay
signal. This is the Levels.fyi data leg of the legal-insider-trading project
(see plans/20260828-001-legal-insider-trading-reddit-scrape.md).

TARGET COMPANIES (pilot)
------------------------
SNAP, META, GOOGLE (alphabet), MICROSOFT, NVIDIA, APPLE, AMAZON

Levels.fyi slugs: snap, meta, google, microsoft, nvidia, apple, amazon

DATA AVAILABLE WITHOUT AUTH
----------------------------
Each job family page (e.g., /companies/snap/salaries/software-engineer/)
embeds a __NEXT_DATA__ JSON block containing:
  - averages[]: per-level buckets, each with samples[] of individual submissions
  - Each sample: uuid, offerDate, level, location, baseSalary,
                 totalCompensation, avgAnnualStockGrantValue, yearsOfExperience,
                 yearsAtCompany, gender, focusTag

Coverage per page: ~30 samples per level (rolling window). Full historical
reach — oldest submissions visible go back 5+ years. Per-level pages are
CSR-only (no server data extractable). Result: ~85% of all submissions
accessible without auth.

PIPELINE
--------

Step 1 — One-time historical scrape (run once locally)
  Script: analysis/levels_scraper/scrape_levels.py --companies all
  - For each company: fetch overview → get job family slugs
  - For each job family: fetch page → extract averages[i]["samples"]
  - Normalize to flat schema, dedup by uuid
  - Save: data/levels_submissions.parquet (all companies, all time)
  Expected volume: ~200–2,000 submissions per company

Step 2 — Daily Lambda (runs forever)
  Function: levels-fyi-daily-scraper
  - Same logic as Step 1
  - Reads existing s3://betting-prod/levels/submissions.parquet
  - Appends only new uuids (dedup before write)
  - Runtime: ~60–120s for 7 companies × ~15 job families × 1 page each
  - Trigger: EventBridge cron, daily 06:00 UTC
  - New submissions appear on-site within 24h → never miss one going forward

Step 3 — Correlation analysis (future)
  - Join levels_submissions + snap_price_weekly on week(offerDate)
  - Signals: weekly submission count, median TC, TC vs market percentile
  - Same lag/Granger analysis as Reddit sentiment leg

FILE STRUCTURE
--------------
analysis/levels_scraper/
  scrape_levels.py           — parameterized scraper (company slug arg)
  lambda_function.py         — AWS Lambda handler wrapping scraper
  requirements.txt
  data/
    levels_submissions.parquet   — local output (all companies)

S3 LAYOUT (prod)
----------------
s3://betting-prod/levels/
  submissions.parquet        — cumulative, all companies

SCHEMA (levels_submissions.parquet)
------------------------------------
uuid               str       — Levels.fyi submission ID (dedup key)
scraped_at         date      — date this script ran (not offerDate)
company_slug       str       — e.g. "snap", "meta"
job_family         str       — e.g. "Software Engineer"
offer_date         datetime  — when the offer was accepted (the signal date)
level              str       — e.g. "L5", "Senior SWE"
location           str
base_salary        int
total_comp         int
stock_grant        int
bonus              int
yoe                float     — years of experience
yac                float     — years at company
gender             str

LAMBDA SPECS
------------
Runtime:     Python 3.13
Memory:      256 MB
Timeout:     5 min
Layers:      requests, pandas, pyarrow  (or container image)
IAM:         s3:GetObject + s3:PutObject on betting-prod/levels/*
Trigger:     EventBridge — cron(0 6 * * ? *)
Env vars:    S3_BUCKET=betting-prod, COMPANIES=snap,meta,google,microsoft,nvidia,apple,amazon

BOT PROTECTION FINDINGS
-----------------------
Levels.fyi uses CloudFront + AWS WAF. Behaviour varies by company:
  - SNAP, META, GOOGLE: no WAF challenge — __NEXT_DATA__ scraping works
  - MICROSOFT, NVIDIA, APPLE, AMAZON: WAF CAPTCHA (x-amzn-waf-action: captcha)
    These require Playwright (headless browser) to bypass. Can scrape locally;
    Lambda will need a container image with Playwright or auth-based API call.

SAMPLE EMBEDDING RULE
----------------------
Levels.fyi only embeds individual samples in __NEXT_DATA__ when the level's
count_last_12_months is roughly 10–100. High-traffic roles (Meta SWE: 663/yr)
have samples=[] — they load client-side via the authenticated API.
Low-traffic roles (Snap SWE: 38/yr, Google PM: ~90/yr) get samples embedded.

CURRENT DATA (2026-08-25)
-------------------------
416 submissions total:
  google  200  (2022-01-29 → 2026-08-24)  SWE + PM only
  meta    159  (2021-10-31 → 2026-08-23)  Data Scientist + Project Manager
  snap     57  (2025-04-02 → 2026-08-22)  SWE only

STATUS
------
[x] Step 1: build scrape_levels.py (multi-company, averages[i]["samples"] extraction)
[x] Step 2: run historical pull locally — 416 rows in data/levels_submissions.parquet
[x] Step 3: build lambda_function.py
[ ] Step 4: get auth token (LEVELS_EMAIL + LEVELS_PASSWORD) for historical backfill
            of Microsoft/NVIDIA/Apple/Amazon + high-traffic roles at Meta/Google
[ ] Step 5: deploy Lambda to AWS, enable EventBridge cron (daily 06:00 UTC)
[ ] Step 6: after 30 days of daily runs, feed into correlate.py
