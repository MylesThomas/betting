LEVELS.FYI SUBMISSIONS LAMBDA
==============================

Scrape individual salary submissions (one row per offer) for all 29 companies
and write to s3://levels-fyi-mt/submissions/submissions.parquet.

STATUS: Not started.

S3 TARGET
---------
  submissions/submissions.parquet
    grain:    one row per salary submission, deduped by uuid
    columns:  uuid, scraped_at, company_slug, job_family, offer_date,
              level, location, base_salary, total_comp, stock_grant,
              bonus, yoe, yac, gender, focus_tag

AUTH PROBLEM
------------
scrape_levels.py --mode auth uses Playwright to get a Bearer token
(logs in via headless Chrome, caches token to disk at data/.levels_token).
Playwright inside Lambda is heavy (~300MB). Solution: don't run Playwright in Lambda.

Auth strategy:
  1. Run levels_auth.py locally to get a fresh token
  2. Push token to SSM Parameter Store (SecureString, free tier)
  3. Lambda reads token from SSM at startup
  4. When token expires (403 from API), Lambda returns an error → you refresh locally and push again

SSM param name:  /levels-fyi/bearer-token  (us-east-2)

LAMBDA DESIGN
-------------
Function name:  levels-fyi-submissions-scraper
Same ECR repo:  232692785472.dkr.ecr.us-east-2.amazonaws.com/levels-fyi-submissions-scraper
  OR: same image as overview scraper, different handler (levels-fyi-daily-scraper has handler
      set to lambda_function.handler — could add submissions_handler.py to same image)

Simpler: separate image, separate Lambda, same Dockerfile pattern.

Handler logic (submissions_lambda.py):
  1. Read token from SSM
  2. For each company slug → fetch all pages from /v3/salary/search (offset/limit 100)
  3. Normalize each row using normalize() from scrape_levels.py
  4. Load existing submissions/submissions.parquet from S3
  5. Concat + dedup on uuid → write back

Timeout:    900s (15 min) — 29 companies × potentially 1,000s of submissions each
Memory:     512MB
Schedule:   weekly, Sunday 6am EST  cron(0 11 * * ? 1)

ESTIMATED VOLUME
----------------
Large companies (Google, Meta, Amazon, Microsoft) may have 5,000–10,000+ submissions each.
At 100/page that's 50–100 API calls per company, ~29 × 75 avg = ~2,175 total API calls.
At 0.3s sleep between calls → ~10 min. Fits in 15 min timeout.

BUILD STEPS
-----------
1. Add SSM push helper:
     src/levels_fyi_scraper/push_token_to_ssm.py
     - runs levels_auth.get_token() locally
     - pushes result to SSM /levels-fyi/bearer-token

2. Write src/levels_fyi_scraper/submissions_lambda.py
     - reads SSM token
     - paginates API per company
     - writes submissions/submissions.parquet

3. Write src/levels_fyi_scraper/Dockerfile.submissions
     - same base as overview (python:3.13-slim + awslambdaric)
     - requirements: boto3, curl_cffi (or requests), pandas, pyarrow

4. Write src/levels_fyi_scraper/deploy_submissions.sh
     - same pattern as deploy.sh
     - Lambda name: levels-fyi-submissions-scraper
     - handler: submissions_lambda.handler
     - add ssm:GetParameter to IAM role (or use same role — check if it has SSM access)

5. IAM: ensure betting-dashboard-daily-update-role-ille2llh has:
     - ssm:GetParameter on arn:aws:ssm:us-east-2:232692785472:parameter/levels-fyi/*

FIRST RUN FLOW
--------------
  # 1. Get token locally
  uv run python src/levels_fyi_scraper/push_token_to_ssm.py

  # 2. Deploy
  ./src/levels_fyi_scraper/deploy_submissions.sh

  # 3. Test invoke (will take ~10 min)
  aws lambda invoke \
    --function-name levels-fyi-submissions-scraper \
    --region us-east-2 \
    --payload '{}' \
    --cli-binary-format raw-in-base64-out \
    /tmp/sub_response.json && cat /tmp/sub_response.json

  # 4. Check S3
  aws s3 ls s3://levels-fyi-mt/submissions/

TOKEN REFRESH
-------------
When Lambda returns 403 / token-expired error:
  uv run python src/levels_fyi_scraper/push_token_to_ssm.py
  # re-invokes Lambda or waits for next weekly run
