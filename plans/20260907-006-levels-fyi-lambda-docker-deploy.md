LEVELS.FYI LAMBDA — DOCKER DEPLOY (PENDING GOOD CONNECTION)
============================================================

STATUS: Blocked on connection. Everything is built. Just needs Docker + ECR.

CURRENT AWS STATE
-----------------
- S3 bucket: s3://levels-fyi-mt/  (may or may not exist — verify in Step 0)
- Lambda: levels-fyi-daily-scraper  (EXISTS but BROKEN — zip deploy, missing pyarrow)
- EventBridge rule: levels-fyi-daily-0700-est  (ENABLED, pointing at broken Lambda)

S3 LAYOUT
---------
s3://levels-fyi-mt/
  overview/daily.parquet
    grain:    one row per (company_slug, scrape_date)
    written:  overview Lambda, daily append + dedup on (company_slug, scrape_date)
    columns:
      scrape_date         str        ISO date, e.g. "2026-09-07"
      company_slug        str        e.g. "snap", "meta"
      total_submissions   int        all-time submission count (hiring velocity proxy)
      median_tc           float      median total comp across all roles ($)
      num_job_families    int        count of distinct job families on the page
      highest_role        str        job family name with highest TC
      highest_tc          float      that role's total comp ($)
      lowest_role         str        job family name with lowest TC
      lowest_tc           float      that role's total comp ($)

  submissions/submissions.parquet
    grain:    one row per salary submission, deduped by uuid
    written:  submissions Lambda (future — not yet built)
    columns:
      uuid                str        stable dedup key from Levels.fyi
      scraped_at          str        ISO date when row was fetched
      company_slug        str        e.g. "snap"
      job_family          str        e.g. "Software Engineering"
      offer_date          datetime   UTC — when the offer was made
      level               str        e.g. "L5", "Senior Engineer"
      location            str        e.g. "San Francisco, CA"
      base_salary         float      annual base ($)
      total_comp          float      total annual compensation ($)
      stock_grant         float      annualized equity grant ($)
      bonus               float      annual bonus ($)
      yoe                 float      years of experience
      yac                 float      years at this company
      gender              str        self-reported, nullable
      focus_tag           str        e.g. "backend", "ML", nullable

  validation/{date}_log.parquet      e.g. validation/2026-09-07_log.parquet
    grain:    one row per company per day
    written:  overview Lambda, same run that writes overview/daily.parquet
    columns:
      date                str        ISO date
      company_slug        str
      total_submissions   int        nullable if company not scraped
      median_tc           float      nullable if company not scraped
      submissions_delta   int        change vs yesterday's count (null on first day)
      check_present       bool       company appeared in today's scrape
      check_nonnull       bool       total_submissions > 0 and median_tc is not null
      check_monotone      bool       submissions_delta >= 0 (never went backward)
      check_tc_range      bool       median_tc in [$50K, $1.5M]
      passed              bool       all four checks true

DEPLOY STEPS
------------
Run the following on a good connection (home wifi, office, hotspot):

  # Step 0 — disable the EventBridge rule (avoids firing at nothing during the gap)
  aws events disable-rule --name levels-fyi-daily-0700-est --region us-east-2

  # Step 0.5 — create S3 bucket if it doesn't exist (idempotent if already exists)
  aws s3 mb s3://levels-fyi-mt --region us-east-2 || true

  # Step 1 — delete the broken zip Lambda (can't switch zip→container in-place)
  aws lambda delete-function --function-name levels-fyi-daily-scraper --region us-east-2

  # Step 2 — deploy via Docker (builds image, pushes to ECR, creates Lambda, re-enables EventBridge)
  cd /Users/thomasmyles/dev/betting
  ./src/levels_fyi_scraper/deploy.sh

deploy.sh is idempotent — it creates ECR repo if needed, creates or updates Lambda,
and does put-rule (ENABLED) + put-targets to wire EventBridge.

WHAT DEPLOY.SH DOES
--------------------
1. ECR login (private)
2. Create ECR repo levels-fyi-daily-scraper if it doesn't exist
3. docker buildx build --platform linux/amd64 → push to ECR
4. aws lambda create-function (container image, 512MB, 300s timeout)
5. EventBridge cron(0 12 * * ? *)  — 7am EST / 8am EDT
6. Lambda env vars: S3_BUCKET=levels-fyi-mt, SSL_VERIFY=true

DOCKERFILE
----------
File: src/levels_fyi_scraper/Dockerfile
Base: python:3.13-slim (Docker Hub, not ECR public — avoids TLS issues)
Deps: awslambdaric + requirements_lambda.txt (curl_cffi, pandas, pyarrow)

IAM ROLE
--------
betting-dashboard-daily-update-role-ille2llh — confirmed to have:
  - S3 write access to levels-fyi-mt
  - ses:SendEmail

SES EMAIL
---------
From: tqstrats@gmail.com  ✅ verified SES identity
To:   mylescgthomas@gmail.com  ✅ verified SES identity
Note: both identities confirmed — safe even if SES is still in sandbox mode.

VERIFY AFTER DEPLOY
-------------------
  aws lambda invoke \
    --function-name levels-fyi-daily-scraper \
    --region us-east-2 \
    --payload '{}' \
    --cli-binary-format raw-in-base64-out \
    /tmp/levels_response.json && cat /tmp/levels_response.json

  # Expect: {"statusCode": 200, "body": "{\"scraped\": 29, ...}"}

  # If it errors, check logs:
  aws logs tail /aws/lambda/levels-fyi-daily-scraper --region us-east-2 --since 5m

  # Check S3 wrote correctly:
  aws s3 ls s3://levels-fyi-mt/overview/
  aws s3 ls s3://levels-fyi-mt/validation/

AFTER 1 WEEK STABLE
-------------------
- Disable local cron: comment out the line in src/levels_fyi_scraper/run_scraper.sh
- Run /levels-validate to spot-check 7 companies against live site

FUTURE WORK — SUBMISSIONS LAMBDA
----------------------------------
scrape_levels.py (auth mode, full historical backfill) needs its own Lambda.
Will write to: s3://levels-fyi-mt/submissions/submissions.parquet
Not built yet. Tracked separately.

LAMBDA ARN (once deployed)
--------------------------
arn:aws:lambda:us-east-2:232692785472:function:levels-fyi-daily-scraper

EVENTBRIDGE RULE ARN
---------------------
arn:aws:events:us-east-2:232692785472:rule/levels-fyi-daily-0700-est
