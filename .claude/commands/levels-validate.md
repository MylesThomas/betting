# Levels.fyi Manual Site Validation

Spot-check scraped data against the live Levels.fyi site using Playwright.

## Prompt

For each of the following companies — snap, meta, google, microsoft, nvidia, apple, amazon — do the following:

1. Open `https://www.levels.fyi/companies/{slug}/salaries/` in the browser
2. Read the **total submission count** and **median total compensation** shown on the overview page
3. Compare these to the latest row for that company in `s3://levels-fyi-mt/overview/daily.parquet`

Flag any company where either value differs from the scraped data by more than 5%.

Load the parquet for comparison with:
```python
import boto3, io, pandas as pd
obj = boto3.client("s3").get_object(Bucket="levels-fyi-mt", Key="overview/daily.parquet")
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
latest = df[df["scrape_date"] == df["scrape_date"].max()].set_index("company_slug")
print(latest[["total_submissions", "median_tc"]])
```

## Pass criteria
- Submission count: scraped value within ±5% of live page count
- Median TC: scraped value within ±5% of live page median TC

## When to run
- After any change to the scraper
- When the daily email shows unexpected deltas
- Before enabling the EventBridge rule on a new deployment
