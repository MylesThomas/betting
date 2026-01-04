"""
Track Kalshi Prediction Market Order Books Hourly

Lambda function: kalshi-order-book-tracker

This script monitors high-volume Kalshi prediction markets by tracking order book
snapshots, detecting fills, calculating overreaction scores, and generating trading
signals based on order book microstructure analysis.

OVERVIEW
--------
Similar to track_game_line_movements.py but adapted for Kalshi prediction markets.
Instead of tracking betting line movements across sportsbooks, this tracks order book
dynamics within Kalshi's marketplace to identify mispricing and overreaction opportunities.

The script performs three core functions:
1. **Market Discovery**: Finds new high-volume, liquid, non-sports markets to monitor
2. **Order Book Tracking**: Saves hourly snapshots of order books to S3
3. **Signal Generation**: Analyzes order book movements to detect trading opportunities

WORKFLOW (3-STEP FLOW)
-----------------------
**Lambda Schedule:**
- Runs hourly (EventBridge trigger)
- Midnight ET (00:00): Full discovery run + daily status email (always sent)
- All other hours: Skip discovery, only track existing markets, only email on signals

TODO List:
[ ] 1. Load existing markets from S3 config
    - Read tracked_markets.json from S3
    - Parse list of active market tickers

[✓] 2. Discover and add new markets (midnight ET only)
    - Fetch all open markets from Kalshi API (with pagination)
    - Filter by volume (>= 100K contracts), liquidity (has yes_bid), non-sports
    - Add qualifying new markets to S3 config with metadata

[ ] 3. Track order book movements for all active markets
    - Fetch current order book for each active market
    - Save snapshot to S3 (JSON format)
    - Compare to 1h, 24h, and 168h historical snapshots
    - Calculate overreaction score (0-10 scale) using Phase 3 baseline
    - Check order book imbalance and depth ratio as independent signals
    - Generate trading signals: FADE, FOLLOW, or NEUTRAL
    - Send email alert via SES with actionable markets at top

MARKET SELECTION CRITERIA
--------------------------
Markets must meet ALL of the following to be added to tracking:

1. **High Volume**: >= 100,000 contracts traded
   - Ensures sufficient market depth and interest
   - Higher volume = more reliable price discovery

2. **Active Trading**: Has yes_bid (current best bid price)
   - Confirms market is actively traded, not stale
   - Required for order book analysis

3. **Non-Sports Markets**: Exclude sports betting markets
   - Sports markets behave like traditional sportsbooks (line movements, vig-driven)
   - We want markets that react to news/events differently
   - Excluded keywords: nba, nfl, mlb, nhl, basketball, football, hockey, soccer,
     baseball, super bowl, world cup, stanley cup, finals, championship, mvp, 
     rookie, playoff, division
   - Why? Sports markets are essentially sports betting with different mechanics.
     We want markets driven by fundamentals, news cycles, and behavioral biases,
     not game outcomes and injury reports.

4. **Open Status**: Market must be open for trading (not closed/settled)

RATIONALE: No Time-to-Expiration Constraint
--------------------------------------------
Unlike the original docstring draft, we do NOT filter by "expiring in N days" because:
- Order flow behavior is what matters, not time to expiration
- Some markets (e.g., 2028 elections) trade actively for years
- High volume + liquidity = exploitable inefficiencies regardless of expiration
- We care about WHEN the market MOVES (news/events), not when it expires

MARKET CONFIGURATION
--------------------
Tracked markets are stored in S3 at:
  s3://kalshi-order-book-snapshots/config/tracked_markets.json

Structure:
{
  "markets": [
    {
      "ticker": "KXGREENLAND-29",
      "date_added": "2024-12-25 00:30:00",  // Eastern Time
      "category": "politics",
      "initial_volume": 2017107,
      "initial_price": 0.21,
      "active": true,
      "consecutive_failures": 0  // Auto-deactivate after 3 consecutive API failures
    },
    ...
  ]
}

Why S3 Config File?
- Flexibility: Update market list without redeploying Lambda
- Version Control: S3 versioning tracks changes to market list
- Auditability: Can see when/why markets were added
- No Hardcoding: Avoids brittle market lists in code
- Auto-Deactivation: Markets with 3+ consecutive API failures are auto-deactivated

DATA STRUCTURE (S3)
-------------------
s3://kalshi-order-book-snapshots/
├── config/
│   ├── tracked_markets.json          # Market configuration
│   └── tracker_health.json           # Health check state
├── data/
│   ├── 01_input/
│   │   └── kalshi/
│   │       └── order_books/
│   │           ├── KXGREENLAND-29_20241225_010000.json
│   │           ├── KXGREENLAND-29_20241225_020000.json
│   │           └── ...
│   └── 04_output/
│       └── kalshi/
│           ├── market_baselines/
│           │   ├── KXGREENLAND-29_baseline.json  # 48h rolling stats
│           │   └── ...
│           └── movement_summary/
│               └── kalshi_movements_20241225.csv
└── email_alerts/
    └── kalshi_signals_20241225_010000.html

Order Book Snapshot JSON Format:
{
  "ticker": "KXGREENLAND-29",
  "timestamp": "2024-12-25T01:00:00Z",
  "yes_bid": 21.0,
  "yes_ask": 22.0,
  "no_bid": 78.0,
  "no_ask": 79.0,
  "yes_bid_size": 500,
  "yes_ask_size": 300,
  "no_bid_size": 450,
  "no_ask_size": 350,
  "volume": 2017107,
  "open_interest": 150000,
  "last_price": 21.5
}

Market Baseline JSON Format (Phase 3):
{
  "ticker": "KXGREENLAND-29",
  "baseline_start": "2024-12-23T01:00:00Z",
  "baseline_end": "2024-12-25T01:00:00Z",
  "hours_of_data": 48,
  "bid_imbalance": {
    "p15": 0.35,
    "p50": 0.50,
    "p85": 0.65
  },
  "depth_ratio": {
    "p15": 0.80,
    "p50": 1.10,
    "p85": 1.45
  },
  "overreaction_score": {
    "p15": 2.0,
    "p50": 5.0,
    "p85": 8.0
  }
}

Tracker Health Check JSON Format:
{
  "last_run": "2024-12-25T01:00:00Z",
  "last_run_duration_seconds": 45.2,
  "consecutive_runs_no_signals": 5,
  "last_signal_detected": "2024-12-24T18:00:00Z",
  "total_runs": 48,
  "total_signals_generated": 12
}

Why Health Check Tracking?
- If consecutive_runs_no_signals >= 24 (24 hours), send health alert email
- Track run duration to detect performance degradation over time
- Prevents silent failures (e.g., baselines stuck, API issues, threshold too strict)
- User gets either: (1) signal emails OR (2) daily health check if no signals
- Ensures system is running and working correctly
- Can alert if Lambda is approaching timeout (e.g., duration > 540s for 10min timeout)

Health Check Email Trigger Flow:
  Run → Start timer
      → Signals detected? 
        ├─ YES → Send signal email, reset counter to 0
        └─ NO → Increment counter
            └─ Counter >= 24? 
                ├─ YES → Send "24h no signals" health check email
                │        Include: markets processed, avg run time, system status
                │        Reset counter to 0
                └─ NO → Skip email, save updated counter
      → End timer, save duration to health.json

TIME WINDOWS FOR SIGNAL GENERATION
-----------------------------------
We compare current order book snapshot against 3 historical windows:

1. **1-Hour Lookback** (immediate reactions)
   - Detects fast-moving news events
   - Identifies panic buying/selling
   - Example: Trump announcement → Greenland market spikes 15 cents in 1 hour

2. **24-Hour Lookback** (daily trends)
   - Captures intraday trading patterns
   - Smooths out noise from 1h window
   - Example: Fed Chair nomination market grinds up 8 cents over 24h

3. **168-Hour Lookback** (weekly trends)
   - Identifies sustained momentum or reversals
   - Filters out short-term volatility
   - Example: 2028 election market drifts down 20 cents over a week

Why These Windows?
- 1h: Catches overreactions to breaking news
- 24h: Confirms sustained movement vs. noise
- 168h: Provides context for longer-term trends
- Combined: Allows multi-timeframe analysis for higher conviction signals

SIGNAL GENERATION LOGIC (PHASE 3 ONLY)
---------------------------------------
We use a **market-specific baseline** approach (Phase 3) rather than absolute thresholds.
For each market, we track 48 hours of historical data to establish percentile-based
thresholds. This accounts for each market's unique volatility and trading patterns.

Two Independent Signal Checks (CURRENT IMPLEMENTATION):

1. **Order Book Imbalance** (top-of-book, smart money indicator)
   Formula: BestYesBidSize / (BestYesBidSize + BestNoAskSize)
   - Measures whether "smart money" is accumulating or distributing
   - High imbalance = institutions building positions quietly
   - Based on Stoikov's market-making theory
   
   Thresholds (market-specific):
   - Imbalance >= p85 → FOLLOW_UP (smart money buying, follow them)
   - Imbalance <= p15 → FOLLOW_DOWN (smart money selling, follow them)
   - Otherwise → NEUTRAL

2. **Depth Ratio** (full book, hidden support/resistance)
   Formula: TotalYesBidDepth / TotalNoAskDepth
   - Measures liquidity asymmetry across entire order book
   - High ratio = strong bid support (hidden demand)
   - Low ratio = strong ask resistance (hidden supply)
   
   Thresholds (market-specific):
   - Depth ratio >= p85 → FOLLOW_UP (strong support, upward pressure)
   - Depth ratio <= p15 → FOLLOW_DOWN (strong resistance, downward pressure)
   - Otherwise → NEUTRAL

FUTURE ENHANCEMENT: Overreaction Score (0-10 scale)
   Would add a 3rd signal based on fill detection across time windows.
   Components: fill velocity, aggression ratio, spread widening, depth changes.
   See view_kalshi_order_book.py for reference implementation.

MULTI-SIGNAL ALIGNMENT AND CONFLICTS
-------------------------------------
After calculating signals (order book imbalance, depth ratio) for the current snapshot,
we categorize markets as:

**ACTIONABLE** (high-conviction signals):
- Both signals align (both FOLLOW_UP, or both FOLLOW_DOWN) → HIGH conviction
- One signal detected, other neutral → MODERATE conviction
- Example: Imbalance p90 FOLLOW_UP + Depth p88 FOLLOW_UP = HIGH conviction BULLISH

**CONFLICTING** (low conviction):
- Signals disagree (one UP, one DOWN)
- Example: Imbalance p90 FOLLOW_UP + Depth p12 FOLLOW_DOWN = CONFLICTING
- Interpretation: Hidden demand but smart money distributing, wait for clarity

**NEUTRAL** (no edge):
- Both signals are NEUTRAL (within p15-p85 range)
- No actionable opportunity detected

**EMAIL ALERT STRUCTURE**:
- **Top Section**: Actionable markets sorted by conviction (HIGH → MODERATE)
- **Bottom Section**: Neutral markets (collapsed list, for reference)

Why Phase 3 Only?
-----------------
Earlier versions considered "Phase 1" (absolute thresholds) and "Phase 2" (market-class
averages), but these fail to account for market-specific dynamics:

- Phase 1: Thresholds like "imbalance > 0.80" fail for naturally imbalanced markets
- Phase 2: Grouping by category (e.g., "politics") still misses individual market quirks
- Phase 3: Each market's OWN 48h history provides the best baseline for comparison

If a market lacks 48h of data, we skip signal generation and just save snapshots.

DEPLOYMENT (AWS LAMBDA)
-----------------------
**Lambda Function Configuration:**
- Runtime: Python 3.12
- Memory: 512 MB
- Timeout: 300 seconds (5 minutes)
- Trigger: EventBridge (CloudWatch Events) every hour

**Required Environment Variables:**
- S3_BUCKET_KALSHI: 'kalshi-order-book-snapshots' (separate from S3_BUCKET used by line movement tracker)
- SES_FROM_EMAIL: Verified sender email in AWS SES
- SES_TO_EMAIL: Recipient email for alerts

**Required IAM Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::kalshi-order-book-snapshots",
        "arn:aws:s3:::kalshi-order-book-snapshots/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ses:SendEmail",
        "ses:SendRawEmail"
      ],
      "Resource": "*"
    }
  ]
}
```

**Deployment Steps:**
1. Zip script + dependencies: `zip -r kalshi_tracker.zip kalshi_order_book_tracker.py`
2. Upload to Lambda console or via AWS CLI
3. Set environment variables
4. Create EventBridge rule: `rate(1 hour)`
5. Test with `--check-api` flag first (market discovery only)
6. Enable hourly production runs with `--prod-run` flag

SETUP & NEXT STEPS
------------------
**1. Create S3 Bucket:**
```bash
aws s3 mb s3://kalshi-order-book-snapshots

# Optional: Enable versioning for config file tracking
aws s3api put-bucket-versioning \
  --bucket kalshi-order-book-snapshots \
  --versioning-configuration Status=Enabled
```

**IMPORTANT: S3 Bucket Settings (via Console or CLI):**
When creating the bucket, configure for FUTURE email chart hosting from the start:

✅ **Object Ownership:** ACLs disabled (recommended)
   - Keep default. We'll use bucket policy, not ACLs.

🔓 **Block Public Access:** UNCHECK ALL 4 boxes (set up now for future email charts)
   - ☐ Block public access to buckets and objects granted through new access control lists (ACLs)
   - ☐ Block public access to buckets and objects granted through any access control lists (ACLs)
   - ☐ Block public access to buckets and objects granted through new public bucket or access point policies
   - ☐ Block public access to buckets and objects granted through any public bucket or access point policies
   
   Why: When we add SES email support later, Gmail needs to load chart images from public S3 URLs.
   We'll use a bucket policy to limit public access to ONLY the email-charts/* folder.

✅ **Bucket Versioning:** Enable (optional, useful for tracking config file changes)
   - Tracks changes to tracked_markets.json over time

✅ **Encryption:** Default encryption (SSE-S3) - automatic, no config needed

**Then add this Bucket Policy right away** (limits public access to only email charts):

Go to Permissions tab → Bucket policy → Edit → Paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::kalshi-order-book-snapshots/email-charts/*",
        "arn:aws:s3:::kalshi-order-book-snapshots/tmp/*"
      ]
    }
  ]
}

```

This makes ONLY `email-charts/*` and `tmp/*` publicly accessible - everything else stays private!
No need to reconfigure later when adding SES emails.

SETUP & NEXT STEPS
------------------
**1. Create S3 Bucket:**
```bash
aws s3 mb s3://kalshi-order-book-snapshots

# Optional: Enable versioning for config file tracking
aws s3api put-bucket-versioning \
  --bucket kalshi-order-book-snapshots \
  --versioning-configuration Status=Enabled
```

Then configure Block Public Access and Bucket Policy as shown above.

**2. Set Local Environment Variables:**
```bash
# Required
export S3_BUCKET_KALSHI=kalshi-order-book-snapshots
export SES_FROM_EMAIL=your-verified-sender@example.com
export SES_TO_EMAIL=your-recipient@example.com
```

**3. Verify SES Email Addresses:**
Before Lambda deployment, verify sender and recipient emails in AWS SES:
- Go to: https://console.aws.amazon.com/ses/ (us-east-2 region)
- Click "Identities" → "Create identity"  
- Verify both sender and recipient email addresses
- Check inbox and click verification links

**4. Test Locally:**
```bash
# Test market discovery (finds new high-volume markets)
python scripts/kalshi_order_book_tracker.py --check-api

# Test single market tracking (after markets added to config)
python scripts/kalshi_order_book_tracker.py --test-market KXGREENLAND-29

# Full dry run (discover + track + generate report, no email)
python scripts/kalshi_order_book_tracker.py --prod-run
```

**5. Create Lambda Function:**

Via AWS Console (recommended):
1. Go to: https://console.aws.amazon.com/lambda/
2. Click "Create function"
3. Configure:
   - Name: kalshi-order-book-tracker
   - Runtime: Python 3.12
   - Architecture: x86_64
   - Execution role: Create new role with basic Lambda permissions
4. Click "Create function"
5. Configuration → General configuration → Edit:
   - Memory: 512 MB
   - Timeout: 300 seconds (5 minutes)
6. Configuration → Permissions → Execution role → Add permissions:
   - Attach policies: AmazonS3FullAccess, AmazonSESFullAccess
   - (or use custom policies from IAM section above)
7. Save the Execution Role ARN for later:
   - Configuration → Permissions → Role name (click it)
   - Copy the ARN at top (format: arn:aws:iam::123456789:role/kalshi-order-book-tracker-role-xyz)

**6. Deploy Code to Lambda:**

Via Console (direct upload):
1. Lambda → Code tab → Paste code directly in inline editor
2. Click "Deploy"

Via CLI (alternative):
```bash
# Package dependencies
pip install -r requirements.txt -t ./package
cd package
zip -r ../kalshi_tracker.zip .
cd ..
zip -g kalshi_tracker.zip scripts/kalshi_order_book_tracker.py

# Upload
aws lambda update-function-code \
  --function-name kalshi-order-book-tracker \
  --zip-file fileb://kalshi_tracker.zip
```

**7. Add Lambda Layer (for dependencies):**

The script requires pandas, numpy, requests, and other packages not included in Lambda by default.

Via Console (recommended):
1. Lambda → Code tab → Scroll down to "Layers" section
2. Click "Add a layer"
3. Choose "AWS Layers"
4. Select "AWSSDKPandas-Python312" (AWS-provided layer with pandas, numpy, requests, boto3)
5. Select latest version
6. Click "Add"

Note: The AWSSDKPandas layer includes:
- pandas, numpy, scipy
- boto3, botocore (S3, SES)
- requests, urllib3
- pytz (for timezones)

Alternative - Create custom layer:
```bash
# If AWS layer doesn't work, create custom layer
mkdir python
pip install pandas numpy requests python-dotenv -t python/
zip -r layer.zip python
aws lambda publish-layer-version \
  --layer-name kalshi-dependencies \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.12
```

**8. Configure Environment Variables:**

Via Console (recommended):
1. Lambda → Configuration → Environment variables → Edit
2. Add these key-value pairs:
   - S3_BUCKET_KALSHI: kalshi-order-book-snapshots
   - SES_FROM_EMAIL: myles@thomasquantitativestrategies.com
   - SES_TO_EMAIL: mylescgthomas@gmail.com
   - AWS_REGION_NAME: us-east-2
3. Remove if present: SNS_TOPIC_ARN (deprecated)
4. Click "Save"

Via CLI (alternative):
```bash
aws lambda update-function-configuration \
  --function-name kalshi-order-book-tracker \
  --environment Variables="{S3_BUCKET_KALSHI=kalshi-order-book-snapshots,SES_FROM_EMAIL=myles@thomasquantitativestrategies.com,SES_TO_EMAIL=mylescgthomas@gmail.com,AWS_REGION_NAME=us-east-2}"
```

**9. Create EventBridge Hourly Trigger:**

Via Console:
1. Lambda → Configuration → Triggers → Add trigger
2. Select: EventBridge (CloudWatch Events)
3. Create new rule:
   - Rule name: kalshi-hourly-tracker
   - Schedule expression: rate(1 hour)
   - Alternative: cron(0 * * * ? *)
4. Click "Add"

Via CLI (alternative):
```bash
# Create rule
aws events put-rule \
  --name kalshi-hourly-tracker \
  --schedule-expression "rate(1 hour)"

# Add Lambda as target
aws events put-targets \
  --rule kalshi-hourly-tracker \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-2:YOUR_ACCOUNT_ID:function:kalshi-order-book-tracker"

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
  --function-name kalshi-order-book-tracker \
  --statement-id EventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-2:YOUR_ACCOUNT_ID:rule/kalshi-hourly-tracker
```

**10. Monitor:**

Via Console:
- CloudWatch Logs: Lambda → Monitor → View CloudWatch logs
- S3 Snapshots: Check s3://kalshi-order-book-snapshots/data/
- Email: Check inbox for SES alerts

Via CLI:
```bash
# View Lambda logs
aws logs tail /aws/lambda/kalshi-order-book-tracker --follow

# Check S3 snapshots
aws s3 ls s3://kalshi-order-book-snapshots/data/01_input/kalshi/order_books/

# View market baselines
aws s3 ls s3://kalshi-order-book-snapshots/data/04_output/kalshi/market_baselines/
```

**Important Notes:**
- Script needs 48 hours of baseline data per market before generating signals
- First 48 hours = calibration period (snapshots saved, no alerts)
- After 48h, market-specific percentiles enable Phase 3 signal detection
- Check S3 config regularly: `s3://kalshi-order-book-snapshots/config/tracked_markets.json`
- Markets can be deactivated by setting `"active": false` in config (no Lambda redeploy needed)
- **Markets auto-deactivate after 3 consecutive API failures** (expired/closed markets)
- **Lambda runs discovery daily at midnight ET** - rest of day skips discovery for speed

USAGE
-----
List tracked markets (from S3 config):
  python scripts/kalshi_order_book_tracker.py --list-markets

Local testing (market discovery):
  python scripts/kalshi_order_book_tracker.py --check-api

Local testing (single market):
  python scripts/kalshi_order_book_tracker.py --test-market KXGREENLAND-29

Local testing (skip discovery, track existing):
  python scripts/kalshi_order_book_tracker.py --skip-discovery

Production run (Lambda or manual):
  python scripts/kalshi_order_book_tracker.py --prod-run

Flags:
  --check-api      : Discover new markets, add to config, then exit (no tracking)
  --skip-discovery : Skip market discovery, only track existing markets
  --prod-run       : Full run (discover + track + alerts)

Lambda Handler:
  def lambda_handler(event, context):
      # Runs with --prod-run behavior
      main(skip_discovery=False, check_api_only=False)

NOTES
-----
- All timestamps are stored in UTC, displayed in ET
- S3 config file is loaded/saved on every run
- Markets can be manually deactivated by setting "active": false in config
- Baseline data refreshes every 48 hours (rolling window)
- Email alerts are only sent if actionable signals are found

SEE ALSO
--------
- scripts/track_game_line_movements.py (inspiration for structure/deployment)
- scripts/kalshi/view_kalshi_order_book.py (order book visualization)
- scripts/kalshi/OPERATIONAL_WORKFLOW.md (daily monitoring workflow)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import os
import requests
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import ssl
import urllib3
from typing import Optional, Tuple, Dict, List
import math
import boto3
from io import StringIO
import json
import statistics

# Load environment variables
load_dotenv()

# Fix SSL certificate issues (for API calls)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path by finding project root (look for .gitignore)
def find_project_root() -> Path:
    """Find project root by looking for .gitignore file."""
    # In Lambda, we don't need project root - everything is in /var/task
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        return Path('/var/task')
    
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    
    # Fallback to current directory if .gitignore not found
    return current

PROJECT_ROOT = find_project_root()
# Only add src to path if it exists (won't exist in Lambda)
src_path = PROJECT_ROOT / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# Display timezone
DISPLAY_TIMEZONE = 'America/New_York'  # Eastern Time for logging

# Kalshi API Configuration
KALSHI_API_BASE = 'https://api.elections.kalshi.com/trade-api/v2'

# Time windows for comparison
LOOKBACK_WINDOW_1H = timedelta(hours=1)
LOOKBACK_WINDOW_24H = timedelta(hours=24)
LOOKBACK_WINDOW_168H = timedelta(hours=168)  # 1 week
SNAPSHOT_TIME_TOLERANCE = timedelta(minutes=5)  # Allow 5min variance when finding snapshots

# Market selection criteria
MIN_VOLUME = 100000  # Minimum 100K contracts traded (be selective)

# Comprehensive sports keywords to exclude sports betting markets
SPORTS_KEYWORDS = [
    # Sports leagues/organizations
    'nba', 'nfl', 'mlb', 'nhl', 'mls', 'ncaa', 'uefa', 'fifa',
    # Sports categories
    'basketball', 'football', 'baseball', 'hockey', 'soccer',
    # Championships/events
    'super bowl', 'world cup', 'stanley cup', 'finals', 'championship',
    'playoff', 'division', 'conference',
    # Awards/positions
    'mvp', 'rookie', 'dpoy', 'coach of the year',
    # Specific identifiers
    'pro basketball', 'pro football', 'college football',
    # Teams (common prefixes that appear in sports markets)
    'fc ', 'fc-',  # Football club prefix
]

# Sports ticker prefixes to exclude (based on Kalshi naming conventions)
SPORTS_TICKER_PREFIXES = [
    'KXSB-',        # Super Bowl
    'KXNCAAF-',     # NCAA Football
    'KXNFL-',       # NFL
    'KXNBA-',       # NBA
    'KXMLB-',       # MLB
    'KXNHL-',       # NHL
    'KXMLS-',       # MLS
    'KXNCAAB-',     # NCAA Basketball
]

# AWS Configuration
S3_BUCKET = os.getenv('S3_BUCKET_KALSHI', 'kalshi-order-book-snapshots')  # Use separate var to avoid conflict with line movement tracker
SES_FROM_EMAIL = os.getenv('SES_FROM_EMAIL', '')
SES_TO_EMAIL = os.getenv('SES_TO_EMAIL', '')
IS_LAMBDA = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ

# ============================================================================
# IMPORTANT: Processing limit for testing
# ============================================================================
# Market Processing Limit
# ============================================================================
MAX_MARKETS_TO_PROCESS = 10000  # Process all markets (no limit in production)
# ============================================================================

# Initialize boto3 clients
s3_client = boto3.client('s3')
ses_client = boto3.client('ses', region_name='us-east-2') if IS_LAMBDA else None

# Signal thresholds (Phase 3 - market-relative percentiles)
BASELINE_HOURS_REQUIRED = 48  # Need 48h of data before generating signals
IMBALANCE_PERCENTILE_HIGH = 85  # Top 15% = extreme
IMBALANCE_PERCENTILE_LOW = 15   # Bottom 15% = extreme
DEPTH_RATIO_PERCENTILE_HIGH = 85
DEPTH_RATIO_PERCENTILE_LOW = 15

# Timestamp format for filenames
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# S3 Paths
MARKETS_CONFIG_KEY = 'config/tracked_markets.json'

# Order book analysis constants (from view_kalshi_order_book.py)
MIN_FILL_SIZE = 50  # Minimum contracts to consider a fill
FILL_THRESHOLD_PCT = 0.10  # 10% of total volume
FILL_VELOCITY_LOW = 300  # contracts/minute
FILL_VELOCITY_MODERATE = 1000
FILL_VELOCITY_HIGH = 2000
AGGRESSION_ORDERLY = 0.40  # 40% aggressive = orderly
AGGRESSION_PANIC = 0.75  # 75% = panic

# =============================================================================
# S3 HELPER FUNCTIONS
# =============================================================================

def get_s3_snapshot_key(market_ticker: str, timestamp: datetime) -> str:
    """Generate S3 key for order book snapshot."""
    timestamp_str = timestamp.strftime(TIMESTAMP_FORMAT)
    return f"data/01_input/kalshi/order_books/{market_ticker}_{timestamp_str}.json"


def get_s3_baseline_key(market_ticker: str) -> str:
    """Generate S3 key for market baseline."""
    return f"data/04_output/kalshi/market_baselines/{market_ticker}_baseline.json"


def list_s3_snapshots(market_ticker: str) -> List[str]:
    """
    List all snapshot files in S3 for a given market.
    
    Returns:
        List of S3 keys sorted by timestamp
    """
    prefix = f"data/01_input/kalshi/order_books/{market_ticker}_"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        return sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')])
    except Exception as e:
        print(f"Warning: Failed to list S3 snapshots for {market_ticker}: {e}")
        return []


def save_json_to_s3(data: dict, s3_key: str) -> bool:
    """Save JSON data to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(data, indent=2)
        )
        return True
    except Exception as e:
        print(f"Error: Failed to save to S3: {s3_key}")
        print(f"   {e}")
        return False


def load_json_from_s3(s3_key: str) -> Optional[dict]:
    """Load JSON data from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Warning: Failed to load from S3: {s3_key}")
        print(f"   {e}")
        return None


def find_snapshot_near_time_s3(market_ticker: str, target_time: datetime) -> Optional[str]:
    """
    Find snapshot S3 key closest to target_time within SNAPSHOT_TIME_TOLERANCE.
    
    Returns:
        S3 key (path) or None if not found
    """
    all_snapshots = list_s3_snapshots(market_ticker)
    
    if not all_snapshots:
        return None
    
    best_key = None
    best_diff = None
    
    for s3_key in all_snapshots:
        # Extract timestamp from key: data/.../TICKER_20241224_120000.json
        filename = s3_key.split('/')[-1]
        parts = filename.replace('.json', '').split('_')
        if len(parts) < 3:
            continue
            
        timestamp_str = f"{parts[-2]}_{parts[-1]}"
        
        try:
            file_time = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
            # Make timezone-aware
            if target_time.tzinfo:
                file_time = file_time.replace(tzinfo=timezone.utc)
            
            time_diff = abs(file_time - target_time)
            
            if time_diff <= SNAPSHOT_TIME_TOLERANCE:
                if best_diff is None or time_diff < best_diff:
                    best_diff = time_diff
                    best_key = s3_key
        except ValueError:
            continue
    
    return best_key


# =============================================================================
# MARKET CONFIG MANAGEMENT
# =============================================================================

def load_markets_config() -> dict:
    """
    Load tracked markets config from S3.
    
    Returns:
        Dict with 'markets' list and metadata, or empty structure if not found
    """
    config = load_json_from_s3(MARKETS_CONFIG_KEY)
    
    if config is None:
        # Initialize empty config
        config = {
            'markets': [],
            'last_updated': datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')
        }
        save_json_to_s3(config, MARKETS_CONFIG_KEY)
    
    return config


def save_markets_config(config: dict):
    """Save markets config to S3."""
    config['last_updated'] = datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')
    save_json_to_s3(config, MARKETS_CONFIG_KEY)


def add_market_to_config(market_ticker: str, category: str, initial_volume: int, initial_price: float):
    """
    Add a new market to tracked config (if not already present).
    
    Returns:
        True if added, False if already exists
    """
    config = load_markets_config()
    
    # Check if already tracked
    existing_tickers = {m['ticker'] for m in config['markets']}
    if market_ticker in existing_tickers:
        return False
    
    # Add new market
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    config['markets'].append({
        'ticker': market_ticker,
        'date_added': now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
        'category': category,
        'initial_volume': initial_volume,
        'initial_price': initial_price,
        'active': True,
        'consecutive_failures': 0
    })
    
    save_markets_config(config)
    return True


# =============================================================================
# KALSHI API FUNCTIONS
# =============================================================================

def fetch_kalshi_markets(limit: int = 100, cursor: str = None) -> Optional[dict]:
    """
    Fetch available markets from Kalshi API with pagination support.
    
    Returns:
        Dict with 'markets' list and 'cursor' for pagination, or None on error
    """
    url = f"{KALSHI_API_BASE}/markets"
    params = {
        'limit': limit,
        'status': 'open'
    }
    
    if cursor:
        params['cursor'] = cursor
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'markets': data.get('markets', []),
            'cursor': data.get('cursor')  # For pagination
        }
    except Exception as e:
        print(f"Error fetching Kalshi markets: {e}")
        return None


def fetch_order_book(market_ticker: str) -> Optional[dict]:
    """
    Fetch full order book for a specific market.
    
    Returns:
        Order book data or None on error
    """
    url = f"{KALSHI_API_BASE}/markets/{market_ticker}/orderbook"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching order book for {market_ticker}: {e}")
        return None


# =============================================================================
# ORDER BOOK ANALYSIS
# =============================================================================

def analyze_order_book(order_book_data: dict) -> Optional[dict]:
    """
    Calculate metrics from full order book distribution.
    
    Returns:
        Dict with imbalance, depth ratio, and other metrics, or None if invalid data
    """
    if not order_book_data or 'orderbook' not in order_book_data:
        return None
    
    book = order_book_data['orderbook']
    yes_orders = book.get('yes', [])  # Format: [[price_cents, size], ...]
    no_orders = book.get('no', [])
    
    if not yes_orders or not no_orders:
        return None
    
    # Best bid/ask (YES side = buying event, NO side = selling event)
    best_yes_price = yes_orders[-1][0] / 100  # Highest YES price
    best_yes_size = yes_orders[-1][1]
    best_no_price = no_orders[-1][0] / 100   # Highest NO price  
    best_no_size = no_orders[-1][1]
    
    # Mid price (YES + NO should ≈ 1.0)
    mid_price = best_yes_price
    spread = abs(1.0 - best_yes_price - best_no_price)
    
    # Top-of-book imbalance (Stoikov's key signal)
    total_top = best_yes_size + best_no_size
    bid_imbalance = best_yes_size / total_top if total_top > 0 else 0.5
    
    # Total depth across all levels
    total_yes_depth = sum(size for _, size in yes_orders)
    total_no_depth = sum(size for _, size in no_orders)
    total_depth = total_yes_depth + total_no_depth
    
    # Depth ratio (for detecting hidden support/resistance)
    depth_ratio = total_yes_depth / total_no_depth if total_no_depth > 0 else 1.0
    
    # Weighted average prices
    weighted_yes_price = sum(p/100 * s for p, s in yes_orders) / total_yes_depth if total_yes_depth > 0 else mid_price
    weighted_no_price = sum(p/100 * s for p, s in no_orders) / total_no_depth if total_no_depth > 0 else (1.0 - mid_price)
    
    return {
        'mid_price': round(mid_price, 4),
        'spread': round(spread, 4),
        'best_yes_price': round(best_yes_price, 4),
        'best_yes_size': best_yes_size,
        'best_no_price': round(best_no_price, 4),
        'best_no_size': best_no_size,
        'bid_imbalance': round(bid_imbalance, 4),
        'total_yes_depth': total_yes_depth,
        'total_no_depth': total_no_depth,
        'total_depth': total_depth,
        'depth_ratio': round(depth_ratio, 4),
        'weighted_yes_price': round(weighted_yes_price, 4),
        'weighted_no_price': round(weighted_no_price, 4),
        'num_yes_levels': len(yes_orders),
        'num_no_levels': len(no_orders),
    }


def save_order_book_snapshot(market_ticker: str, order_book_data: dict, metrics: dict, timestamp: datetime) -> bool:
    """
    Save order book snapshot to S3.
    
    Stores full order book + calculated metrics for later analysis.
    """
    s3_key = get_s3_snapshot_key(market_ticker, timestamp)
    
    snapshot = {
        'timestamp': timestamp.isoformat(),
        'market_ticker': market_ticker,
        'order_book': order_book_data,
        'metrics': metrics
    }
    
    return save_json_to_s3(snapshot, s3_key)


# =============================================================================
# BASELINE MANAGEMENT (Phase 3)
# =============================================================================

def load_baseline(market_ticker: str) -> Optional[dict]:
    """Load market-specific baseline if it exists."""
    s3_key = get_s3_baseline_key(market_ticker)
    return load_json_from_s3(s3_key)


def update_baseline(market_ticker: str, current_metrics: dict):
    """
    Update rolling 48h baseline for a market.
    
    Tracks percentiles of key metrics to enable market-relative signal detection.
    """
    s3_key = get_s3_baseline_key(market_ticker)
    baseline = load_json_from_s3(s3_key)
    
    if baseline is None:
        # Initialize new baseline
        baseline = {
            'market_ticker': market_ticker,
            'first_seen': datetime.now(timezone.utc).isoformat(),
            'samples': []
        }
    
    # Add current sample
    baseline['samples'].append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': {
            'bid_imbalance': current_metrics.get('bid_imbalance', 0.5),
            'depth_ratio': current_metrics.get('depth_ratio', 1.0),
            'spread': current_metrics.get('spread', 0.02),
            'total_depth': current_metrics.get('total_depth', 0),
        }
    })
    
    # Keep only last 48h
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    baseline['samples'] = [
        s for s in baseline['samples']
        if datetime.fromisoformat(s['timestamp']) > cutoff
    ]
    
    # Calculate statistics
    if baseline['samples']:
        # Calculate hours of data based on actual timestamps (not sample count)
        first_sample_time = datetime.fromisoformat(baseline['samples'][0]['timestamp'])
        last_sample_time = datetime.fromisoformat(baseline['samples'][-1]['timestamp'])
        hours_of_data = (last_sample_time - first_sample_time).total_seconds() / 3600
        
        baseline['hours_of_data'] = hours_of_data
        baseline['last_updated'] = datetime.now(timezone.utc).isoformat()
        baseline['ready_for_alerts'] = hours_of_data >= BASELINE_HOURS_REQUIRED
        
        # Calculate percentiles for each metric
        def calc_percentiles(values):
            if not values:
                return {}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if n > 1 else 0,
                'p15': sorted_vals[max(0, int(n * 0.15))] if n > 0 else 0,
                'p85': sorted_vals[min(n-1, int(n * 0.85))] if n > 0 else 0,
                'p10': sorted_vals[max(0, int(n * 0.10))] if n > 0 else 0,
                'p90': sorted_vals[min(n-1, int(n * 0.90))] if n > 0 else 0,
            }
        
        # Extract metric arrays
        imbalances = [s['metrics']['bid_imbalance'] for s in baseline['samples']]
        depth_ratios = [s['metrics']['depth_ratio'] for s in baseline['samples']]
        spreads = [s['metrics']['spread'] for s in baseline['samples']]
        depths = [s['metrics']['total_depth'] for s in baseline['samples']]
        
        baseline['metrics_stats'] = {
            'bid_imbalance': calc_percentiles(imbalances),
            'depth_ratio': calc_percentiles(depth_ratios),
            'spread': calc_percentiles(spreads),
            'total_depth': calc_percentiles(depths),
        }
    
    # Save updated baseline
    save_json_to_s3(baseline, s3_key)
    return baseline


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_signals(market_ticker: str, current_metrics: dict, baseline: dict) -> Optional[List[dict]]:
    """
    Check all signals for a market (Phase 3 only - requires baseline).
    
    Returns:
        List of detected signals, or None if insufficient baseline data
    """
    if not baseline or not baseline.get('ready_for_alerts', False):
        return None
    
    stats = baseline['metrics_stats']
    signals = []
    
    # 1. Order Book Imbalance Signal
    current_imbalance = current_metrics['bid_imbalance']
    imbalance_stats = stats['bid_imbalance']
    
    if current_imbalance >= imbalance_stats['p85']:
        signals.append({
            'type': 'SMART_MONEY_BID',
            'signal': 'FOLLOW_UP',
            'value': current_imbalance,
            'percentile': 85,
            'reason': f"Imbalance {current_imbalance:.2f} (top 15% for this market) - Smart money buying"
        })
    elif current_imbalance <= imbalance_stats['p15']:
        signals.append({
            'type': 'SMART_MONEY_ASK',
            'signal': 'FOLLOW_DOWN',
            'value': current_imbalance,
            'percentile': 15,
            'reason': f"Imbalance {current_imbalance:.2f} (bottom 15% for this market) - Smart money selling"
        })
    
    # 2. Depth Ratio Signal
    current_depth_ratio = current_metrics['depth_ratio']
    depth_stats = stats['depth_ratio']
    
    if current_depth_ratio >= depth_stats['p85']:
        signals.append({
            'type': 'DEEP_BID_SUPPORT',
            'signal': 'FOLLOW_UP',
            'value': current_depth_ratio,
            'percentile': 85,
            'reason': f"Depth ratio {current_depth_ratio:.2f}x (top 15%) - Strong bid support"
        })
    elif current_depth_ratio <= depth_stats['p15']:
        signals.append({
            'type': 'DEEP_ASK_RESISTANCE',
            'signal': 'FOLLOW_DOWN',
            'value': current_depth_ratio,
            'percentile': 15,
            'reason': f"Depth ratio {current_depth_ratio:.2f}x (bottom 15%) - Strong ask resistance"
        })
    
    return signals if signals else []


def analyze_signal_alignment(signals: List[dict]) -> dict:
    """
    Analyze how signals align (all bullish, all bearish, or conflicting).
    
    Returns:
        Dict with alignment type and conviction level
    """
    if not signals:
        return {'alignment': 'NONE', 'conviction': 'NONE'}
    
    # Count directional signals
    bullish = sum(1 for s in signals if 'UP' in s['signal'])
    bearish = sum(1 for s in signals if 'DOWN' in s['signal'])
    
    if bullish > 0 and bearish == 0:
        conviction = 'HIGH' if len(signals) >= 2 else 'MODERATE'
        return {'alignment': 'BULLISH', 'conviction': conviction}
    elif bearish > 0 and bullish == 0:
        conviction = 'HIGH' if len(signals) >= 2 else 'MODERATE'
        return {'alignment': 'BEARISH', 'conviction': conviction}
    elif bullish > 0 and bearish > 0:
        return {'alignment': 'CONFLICTING', 'conviction': 'LOW'}
    else:
        return {'alignment': 'NEUTRAL', 'conviction': 'NONE'}


# =============================================================================
# EMAIL FORMATTING
# =============================================================================

def format_market_signals_text(market_ticker: str, current_metrics: dict, signals: List[dict], alignment: dict) -> str:
    """Format a single market's signals as plain text for email."""
    lines = []
    
    # Market header
    lines.append(f"\n{market_ticker}")
    lines.append(f"  Price: {current_metrics['mid_price']:.2f}")
    lines.append(f"  Spread: {current_metrics['spread']:.4f}")
    lines.append("")
    
    # Show each signal
    for signal in signals:
        emoji = "🔺" if 'UP' in signal['signal'] else "🔻"
        lines.append(f"  {emoji} {signal['type']} (p{signal['percentile']})")
        lines.append(f"     {signal['reason']}")
    
    # Alignment analysis
    lines.append("")
    if alignment['alignment'] == 'BULLISH':
        lines.append(f"  💡 {alignment['conviction']} CONVICTION → FOLLOW UP")
        lines.append(f"     {len(signals)} signal(s) agree on upward direction")
    elif alignment['alignment'] == 'BEARISH':
        lines.append(f"  💡 {alignment['conviction']} CONVICTION → FOLLOW DOWN")
        lines.append(f"     {len(signals)} signal(s) agree on downward direction")
    elif alignment['alignment'] == 'CONFLICTING':
        lines.append(f"  ⚠️  CONFLICTING SIGNALS → WAIT FOR CLARITY")
        lines.append(f"     Signals disagree on direction")
    
    return "\n".join(lines)


def format_signals_email(actionable_markets: List[dict], neutral_markets: List[str], timestamp: datetime, markets_processed: int = 0, new_markets_added: int = 0, is_daily: bool = False) -> str:
    """
    Format all signals into plain text email.
    
    Args:
        actionable_markets: List of dicts with market_ticker, signals, metrics, alignment
        neutral_markets: List of market tickers with no signals
        timestamp: Current run time
        markets_processed: Total markets attempted (including calibrating)
        new_markets_added: Number of new markets discovered (daily run only)
        is_daily: Whether this is the daily discovery run
    """
    time_et = timestamp.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    time_str = time_et.strftime('%b %d, %Y %I:%M %p ET')
    
    lines = []
    lines.append("=" * 80)
    if is_daily and not actionable_markets:
        lines.append("📅 KALSHI DAILY STATUS")
    else:
        lines.append("🚨 KALSHI TRADING SIGNALS")
    lines.append("=" * 80)
    lines.append(f"Time: {time_str}")
    lines.append("")
    
    # Summary
    total_with_data = len(actionable_markets) + len(neutral_markets)
    lines.append(f"Markets processed: {markets_processed if markets_processed > 0 else total_with_data}")
    lines.append(f"Actionable signals: {len(actionable_markets)}")
    lines.append(f"Neutral markets: {len(neutral_markets)}")
    
    # Show discovery info on daily runs
    if is_daily:
        lines.append("")
        lines.append("🔍 MARKET DISCOVERY (Daily)")
        lines.append(f"   New markets added: {new_markets_added}")
    
    lines.append("")
    
    # Actionable markets
    if actionable_markets:
        lines.append("=" * 80)
        lines.append("📊 ACTIONABLE MARKETS")
        lines.append("=" * 80)
        
        # Sort by conviction (HIGH first)
        actionable_markets.sort(key=lambda x: (
            0 if x['alignment']['conviction'] == 'HIGH' else 
            1 if x['alignment']['conviction'] == 'MODERATE' else 2
        ))
        
        for market in actionable_markets:
            lines.append(format_market_signals_text(
                market['market_ticker'],
                market['metrics'],
                market['signals'],
                market['alignment']
            ))
            lines.append("-" * 80)
    else:
        lines.append("=" * 80)
        lines.append("✅ NO ACTIONABLE SIGNALS DETECTED")
        lines.append("=" * 80)
    
    # Neutral markets (collapsed)
    if neutral_markets:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"📋 NEUTRAL MARKETS ({len(neutral_markets)})")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Markets with no signals (all metrics within normal ranges):")
        for ticker in sorted(neutral_markets):
            lines.append(f"  • {ticker}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def send_email_via_ses(subject: str, html_body: str, text_body: str):
    """
    Send HTML email with inline images via AWS SES.
    
    Args:
        subject: Email subject
        html_body: HTML content with S3-hosted image URLs
        text_body: Plain text fallback
    """
    if not ses_client:
        print("Warning: SES client not initialized, skipping email")
        return
    
    if not SES_FROM_EMAIL or not SES_TO_EMAIL:
        print("Warning: SES_FROM_EMAIL or SES_TO_EMAIL not set, skipping email")
        return
    
    try:
        response = ses_client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={
                'ToAddresses': [SES_TO_EMAIL]
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Text': {
                        'Data': text_body,
                        'Charset': 'UTF-8'
                    },
                    'Html': {
                        'Data': html_body,
                        'Charset': 'UTF-8'
                    }
                }
            }
        )
        print(f"✅ Email sent via SES: {subject}")
        print(f"   Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"❌ Error: Failed to send email via SES: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN LOGIC
# =============================================================================

def discover_and_add_new_markets():
    """
    Discover high-volume Kalshi markets and add to tracking config.
    
    Fetches ALL pages to find high-volume markets (>100K volume).
    
    Returns:
        Number of new markets added
    """
    print("\n📡 Discovering new Kalshi markets...")
    print("   Fetching all pages from Kalshi API...")
    
    all_markets = []
    cursor = None
    pages_fetched = 0
    
    # Paginate through ALL pages
    while True:
        result = fetch_kalshi_markets(limit=200, cursor=cursor)
        if not result:
            if pages_fetched == 0:
                print("   ❌ Failed to fetch markets from Kalshi API")
                return 0
            break
        
        markets_batch = result['markets']
        if not markets_batch:
            break  # No more markets
            
        all_markets.extend(markets_batch)
        pages_fetched += 1
        
        cursor = result.get('cursor')
        if not cursor:
            break  # No more pages
        
        # Progress indicator
        if pages_fetched % 5 == 0:
            print(f"   ... fetched {len(all_markets)} markets so far ({pages_fetched} pages)")
        
        # Rate limiting: sleep between requests to avoid 429 errors
        time.sleep(0.2)  # 200ms delay between pages
    
    print(f"   ✅ Fetched {len(all_markets)} total markets from API ({pages_fetched} pages)")
    print(f"   Analyzing markets (min volume: {MIN_VOLUME:,})...")
    
    added_count = 0
    candidates = []
    
    for market in all_markets:
        ticker = market.get('ticker')
        volume = market.get('volume', 0)
        
        # Try multiple possible category fields
        category = (market.get('category') or 
                   market.get('series_category') or 
                   market.get('title', '')).lower()
        
        yes_bid = market.get('yes_bid')
        title = market.get('title', ticker)
        
        # Collect all markets with >10K volume for debugging
        if volume >= 10000:
            candidates.append({
                'ticker': ticker,
                'volume': volume,
                'category': category,
                'yes_bid': yes_bid,
                'title': title
            })
        
        # Skip if below volume threshold
        if volume < MIN_VOLUME:
            continue
        
        # Skip sports markets (check ticker prefix, category, and title)
        # Check ticker prefix first (fastest)
        is_sports_ticker = any(ticker.startswith(prefix) for prefix in SPORTS_TICKER_PREFIXES)
        
        if not is_sports_ticker:
            # If ticker doesn't match sports prefixes, check keywords in category/title
            category_lower = category.lower()
            title_lower = title.lower()
            is_sports = any(keyword in category_lower or keyword in title_lower 
                           for keyword in SPORTS_KEYWORDS)
        else:
            is_sports = True
        
        if is_sports:
            continue  # Skip without printing (too noisy)
        
        # Skip if no liquidity
        if yes_bid is None:
            print(f"   ⏭️  Skipped {ticker} (no yes_bid)")
            continue
        
        # Try to add (will skip if already tracked)
        # Use 'unknown' as fallback category if empty
        final_category = category if category else 'unknown'
        initial_price = yes_bid / 100
        
        if add_market_to_config(ticker, final_category, volume, initial_price):
            print(f"   ✅ Added {ticker} (volume: {volume:,}, price: {initial_price:.2f})")
            print(f"      Title: {title[:60]}")
            added_count += 1
    
    # Debug output
    print(f"\n   Markets with volume >= 10K: {len(candidates)}")
    print(f"   Markets with volume >= {MIN_VOLUME:,}: {added_count}")
    
    if added_count == 0:
        print(f"\n   ℹ️  No markets met high volume criteria (>= {MIN_VOLUME:,})")
        if candidates:
            print(f"   📊 Top markets by volume (showing top 10):")
            candidates.sort(key=lambda x: x['volume'], reverse=True)
            for i, m in enumerate(candidates[:10], 1):
                ticker = m['ticker']
                title_lower = m['title'].lower()
                
                # Check if sports via ticker prefix OR keywords
                is_sports_ticker = any(ticker.startswith(prefix) for prefix in SPORTS_TICKER_PREFIXES)
                is_sports_keyword = any(keyword in title_lower for keyword in SPORTS_KEYWORDS)
                is_sports = is_sports_ticker or is_sports_keyword
                
                sports_flag = " [SPORTS]" if is_sports else ""
                print(f"      {i}. {ticker}: {m['volume']:,} volume{sports_flag}")
                print(f"         {m['title'][:70]}")
    else:
        print(f"\n   ✅ Successfully added {added_count} high-volume market(s) to tracking")
    
    return added_count


def process_market(market_ticker: str, timestamp: datetime) -> Optional[dict]:
    """
    Process a single market: fetch order book, calculate signals.
    
    Returns:
        Dict with market data and signals, or None if skipped
    """
    # Fetch current order book
    order_book_data = fetch_order_book(market_ticker)
    if not order_book_data:
        return None
    
    # Analyze order book
    metrics = analyze_order_book(order_book_data)
    if not metrics:
        return None
    
    # Save snapshot
    save_order_book_snapshot(market_ticker, order_book_data, metrics, timestamp)
    
    # Update baseline
    baseline = update_baseline(market_ticker, metrics)
    
    # Check if ready for signals
    if not baseline.get('ready_for_alerts', False):
        hours = baseline.get('hours_of_data', 0)
        print(f"   ⏳ {market_ticker}: Calibrating baseline ({hours:.0f}h / {BASELINE_HOURS_REQUIRED}h)")
        return None
    
    # Detect signals
    signals = check_signals(market_ticker, metrics, baseline)
    
    if not signals:
        return {'market_ticker': market_ticker, 'signals': [], 'metrics': metrics}
    
    # Analyze signal alignment
    alignment = analyze_signal_alignment(signals)
    
    return {
        'market_ticker': market_ticker,
        'signals': signals,
        'metrics': metrics,
        'alignment': alignment
    }


def main(is_daily_run=False):
    """
    Main execution function.
    
    Args:
        is_daily_run: If True, sends email regardless of signals (daily status report)
    """
    parser = argparse.ArgumentParser(
        description='Track Kalshi order books and generate trading signals'
    )
    parser.add_argument('--prod-run', action='store_true',
                       help='Production mode (no prompts)')
    parser.add_argument('--check-api', action='store_true',
                       help='Check API connection and discover markets only')
    parser.add_argument('--skip-discovery', action='store_true',
                       help='Skip market discovery, only process tracked markets')
    parser.add_argument('--test-market', type=str,
                       help='Test tracking on a single market ticker (e.g., KXGREENLAND-29)')
    parser.add_argument('--list-markets', action='store_true',
                       help='Load and display tracked markets from S3 config')
    parser.add_argument('--health', action='store_true',
                       help='Display tracker health status from S3 (run duration, no-signal streak)')
    
    args = parser.parse_args()
    
    timestamp = datetime.now(timezone.utc)
    time_et = timestamp.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    
    print("=" * 80)
    print("KALSHI ORDER BOOK TRACKER")
    print("=" * 80)
    print(f"Time: {time_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("")
    
    # Health check mode
    if args.health:
        print("🏥 Loading tracker health status from S3...")
        health_key = "config/tracker_health.json"
        health = load_json_from_s3(health_key)
        
        if not health:
            print("   ❌ ERROR: Health tracking file not found in S3")
            print(f"   Expected: s3://{S3_BUCKET_KALSHI}/{health_key}")
            print("")
            print("   This means the tracker has never run successfully, or health")
            print("   tracking is not enabled. Run the tracker at least once to initialize.")
            return
        
        print("\n" + "=" * 80)
        print("TRACKER HEALTH STATUS")
        print("=" * 80)
        
        # Parse timestamps
        last_run = health.get('last_run')
        last_signal = health.get('last_signal_detected')
        
        if last_run:
            last_run_dt = datetime.fromisoformat(last_run).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
            print(f"\n📅 Last Run: {last_run_dt.strftime('%Y-%m-%d %I:%M %p ET')}")
            
            # Calculate time since last run
            time_since = datetime.now(timezone.utc) - datetime.fromisoformat(last_run)
            hours_since = time_since.total_seconds() / 3600
            if hours_since < 2:
                print(f"   ✅ {hours_since:.1f} hours ago (healthy)")
            elif hours_since < 4:
                print(f"   ⚠️  {hours_since:.1f} hours ago (slightly delayed)")
            else:
                print(f"   ❌ {hours_since:.1f} hours ago (stale - check Lambda)")
        
        # Run duration
        duration = health.get('last_run_duration_seconds', 0)
        print(f"\n⏱️  Last Run Duration: {duration:.1f} seconds")
        if duration > 540:
            print(f"   ⚠️  Approaching Lambda timeout (10min = 600s)")
        elif duration > 300:
            print(f"   ⚠️  Long run time - consider optimization")
        elif duration > 0:
            print(f"   ✅ Normal execution time")
        
        # Signal streak
        no_signal_runs = health.get('consecutive_runs_no_signals', 0)
        print(f"\n📊 Consecutive Runs with No Signals: {no_signal_runs}")
        if no_signal_runs >= 24:
            print(f"   🚨 {no_signal_runs} hours without signals - health check email should be sent")
        elif no_signal_runs >= 12:
            print(f"   ⚠️  {no_signal_runs} hours quiet - halfway to health check threshold")
        else:
            print(f"   ✅ Within normal range (health check triggers at 24)")
        
        # Last signal detected
        if last_signal:
            last_signal_dt = datetime.fromisoformat(last_signal).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
            print(f"\n🎯 Last Signal Detected: {last_signal_dt.strftime('%Y-%m-%d %I:%M %p ET')}")
            
            time_since_signal = datetime.now(timezone.utc) - datetime.fromisoformat(last_signal)
            hours_since_signal = time_since_signal.total_seconds() / 3600
            print(f"   ({hours_since_signal:.1f} hours ago)")
        else:
            print(f"\n🎯 Last Signal Detected: Never")
        
        # Totals
        total_runs = health.get('total_runs', 0)
        total_signals = health.get('total_signals_generated', 0)
        print(f"\n📈 Lifetime Stats:")
        print(f"   Total Runs: {total_runs}")
        print(f"   Total Signals: {total_signals}")
        if total_runs > 0:
            signal_rate = (total_signals / total_runs) * 100
            print(f"   Signal Rate: {signal_rate:.1f}% of runs")
        
        print("\n" + "=" * 80)
        return
    
    # List markets mode
    if args.list_markets:
        print("📋 Loading tracked markets from S3...")
        config = load_markets_config()
        markets = config.get('markets', [])
        
        if not markets:
            print("   ⚠️  No markets found in config")
            return
        
        # Separate active and inactive
        active = [m for m in markets if m.get('active', True)]
        inactive = [m for m in markets if not m.get('active', True)]
        
        print(f"\n✅ Total: {len(markets)} markets ({len(active)} active, {len(inactive)} inactive)")
        print("\n" + "=" * 80)
        print("ACTIVE MARKETS:")
        print("=" * 80)
        
        # Sort by date_added (most recent first)
        active_sorted = sorted(active, key=lambda x: x.get('date_added', ''), reverse=True)
        
        for i, market in enumerate(active_sorted, 1):
            ticker = market.get('ticker', 'unknown')
            date_added = market.get('date_added', 'unknown')
            category = market.get('category', 'unknown')
            volume = market.get('initial_volume', 0)
            price = market.get('initial_price', 0)
            failures = market.get('consecutive_failures', 0)
            
            print(f"\n{i}. {ticker}")
            print(f"   Added: {date_added}")
            print(f"   Category: {category}")
            print(f"   Initial Volume: {volume:,}")
            print(f"   Initial Price: {price:.2f}")
            if failures > 0:
                print(f"   ⚠️  Consecutive failures: {failures}/3")
        
        if inactive:
            print("\n" + "=" * 80)
            print(f"INACTIVE MARKETS ({len(inactive)}):")
            print("=" * 80)
            for market in inactive:
                failures = market.get('consecutive_failures', 0)
                reason = f"(auto-deactivated after {failures} failures)" if failures >= 3 else "(manually deactivated)"
                print(f"   • {market.get('ticker', 'unknown')} {reason}")
        
        return
    
    # Test single market mode
    if args.test_market:
        print(f"🔬 Testing single market: {args.test_market}")
        result = process_market(args.test_market, timestamp)
        if result:
            if result['signals']:
                alignment = result['alignment']
                print(f"\n✅ Detected {len(result['signals'])} signal(s):")
                print(f"   Alignment: {alignment['alignment']} ({alignment['conviction']} conviction)")
                for signal in result['signals']:
                    print(f"   - {signal['type']}: {signal['signal']}")
                    print(f"     {signal['reason']}")
            else:
                print(f"\n↔️  No signals detected (market trading normally)")
                print(f"   Price: {result['metrics']['mid_price']:.2f}")
                print(f"   Spread: {result['metrics']['spread']:.4f}")
        else:
            print(f"\n⏳ Market still calibrating or no data available")
        return
    
    # Step 1: Load existing markets config
    print("📋 Loading tracked markets...")
    config = load_markets_config()
    active_markets = [m for m in config['markets'] if m.get('active', True)]
    
    # Apply processing limit (for testing/performance)
    if len(active_markets) > MAX_MARKETS_TO_PROCESS:
        print(f"   ⚠️  Limiting to first {MAX_MARKETS_TO_PROCESS} markets (out of {len(active_markets)} total)")
        active_markets = active_markets[:MAX_MARKETS_TO_PROCESS]
    
    print(f"   Currently tracking: {len(active_markets)} markets")
    
    # Start timing for health tracking
    start_time = time.time()
    
    # Step 2: Discover new markets (unless skipped or check-api only)
    new_markets_added = 0
    if not args.skip_discovery:
        new_markets_added = discover_and_add_new_markets()
        if new_markets_added > 0:
            # Reload config
            config = load_markets_config()
            active_markets = [m for m in config['markets'] if m.get('active', True)]
            
            # Reapply limit after discovery
            if len(active_markets) > MAX_MARKETS_TO_PROCESS:
                print(f"   ⚠️  Limiting to first {MAX_MARKETS_TO_PROCESS} markets (out of {len(active_markets)} total)")
                active_markets = active_markets[:MAX_MARKETS_TO_PROCESS]
    
    if args.check_api:
        print("\n✅ API check complete")
        return
    
    # Step 3: Process each market
    print(f"\n📊 Processing {len(active_markets)} markets...")
    
    actionable_markets = []
    neutral_markets = []
    markets_to_update = []  # Track failure counts
    
    for i, market_config in enumerate(active_markets):
        ticker = market_config['ticker']
        print(f"\n   Processing {ticker}...")
        
        result = process_market(ticker, timestamp)
        
        if result is None:
            # API failure or no data - increment failure count
            consecutive_failures = market_config.get('consecutive_failures', 0) + 1
            market_config['consecutive_failures'] = consecutive_failures
            markets_to_update.append(market_config)
            
            if consecutive_failures >= 3:
                # Auto-deactivate after 3 failures
                if market_config.get('active', True):
                    print(f"   ❌ {ticker}: 3 consecutive failures - AUTO-DEACTIVATING")
                    market_config['active'] = False
            else:
                print(f"   ⚠️  {ticker}: API failure ({consecutive_failures}/3)")
            
            continue
        
        # Success - reset failure count
        if market_config.get('consecutive_failures', 0) > 0:
            market_config['consecutive_failures'] = 0
            markets_to_update.append(market_config)
        
        if result['signals']:
            actionable_markets.append(result)
            print(f"   ✅ {ticker}: {len(result['signals'])} signal(s) detected")
        else:
            neutral_markets.append(ticker)
            print(f"   ↔️  {ticker}: No signals (neutral)")
    
    # Update failure counts in S3 config if any changes
    if markets_to_update:
        config = load_markets_config()
        for updated_market in markets_to_update:
            # Find and update the market in config
            for market in config['markets']:
                if market['ticker'] == updated_market['ticker']:
                    market['consecutive_failures'] = updated_market.get('consecutive_failures', 0)
                    market['active'] = updated_market.get('active', True)
                    break
        save_markets_config(config)
        
        deactivated_count = sum(1 for m in markets_to_update if not m.get('active', True))
        if deactivated_count > 0:
            print(f"\n   🔄 Auto-deactivated {deactivated_count} market(s) due to repeated failures")
    
    # Step 4: Generate and send alerts
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Markets processed: {len(active_markets)}")
    print(f"Actionable signals: {len(actionable_markets)}")
    print(f"Neutral markets: {len(neutral_markets)}")
    
    # Send email: always on daily runs, only on signals for hourly runs
    should_send_email = actionable_markets or is_daily_run
    
    if should_send_email:
        # Generate plain text email
        text_body = format_signals_email(
            actionable_markets, 
            neutral_markets, 
            timestamp, 
            len(active_markets),
            new_markets_added,
            is_daily_run
        )
        
        # TODO: Generate HTML email with charts (future enhancement)
        html_body = f"<html><body><pre>{text_body}</pre></body></html>"  # Simple wrapper for now
        
        if is_daily_run and not actionable_markets:
            subject = f"📅 Kalshi Daily Status - {time_et.strftime('%b %d, %Y %I:%M %p ET')}"
            print("\n📧 Sending daily status email (midnight run)")
        else:
            subject = f"🚨 Kalshi Trading Signals - {time_et.strftime('%b %d, %Y %I:%M %p ET')}"
            print("\n📧 Sending signal alert email")
        
        # Send email via SES
        send_email_via_ses(subject, html_body, text_body)
    else:
        print("\n📧 No actionable signals - skipping email")
    
    # Print to console (local runs only)
    if not IS_LAMBDA and (actionable_markets or is_daily_run):
        text_body = format_signals_email(
            actionable_markets, 
            neutral_markets, 
            timestamp, 
            len(active_markets),
            new_markets_added,
            is_daily_run
        )
        print("\n" + text_body)
    
    # Step 5: Update health tracking
    duration = time.time() - start_time
    print(f"\n⏱️  Run duration: {duration:.1f} seconds")
    
    # Load existing health data
    health_key = "config/tracker_health.json"
    health = load_json_from_s3(health_key)
    
    if health is None:
        # Initialize health tracking
        health = {
            'consecutive_runs_no_signals': 0,
            'total_runs': 0,
            'total_signals_generated': 0
        }
        print("   🆕 Initializing health tracking")
    
    # Update health data
    health['last_run'] = timestamp.isoformat()
    health['last_run_duration_seconds'] = duration
    health['total_runs'] = health.get('total_runs', 0) + 1
    
    if actionable_markets:
        # Reset no-signal streak
        health['consecutive_runs_no_signals'] = 0
        health['last_signal_detected'] = timestamp.isoformat()
        health['total_signals_generated'] = health.get('total_signals_generated', 0) + len(actionable_markets)
        print(f"   📊 Signals detected: {len(actionable_markets)} (streak reset)")
    else:
        # Increment no-signal streak
        health['consecutive_runs_no_signals'] = health.get('consecutive_runs_no_signals', 0) + 1
        no_signal_count = health['consecutive_runs_no_signals']
        print(f"   📊 No signals: {no_signal_count} consecutive runs")
        
        # Send health check email if 24 hours without signals
        if no_signal_count >= 24:
            print(f"   🚨 24h without signals - sending health check email")
            
            health_subject = f"🏥 Kalshi Tracker Health Check - {time_et.strftime('%b %d, %Y %I:%M %p ET')}"
            health_text = f"""
================================================================================
🏥 KALSHI TRACKER HEALTH CHECK
================================================================================
Time: {time_et.strftime('%b %d, %Y %I:%M %p ET')}

⚠️  NO SIGNALS FOR {no_signal_count} HOURS

The tracker is running correctly but hasn't detected any actionable signals
in the past {no_signal_count} hours. This could mean:

1. Markets are trading normally (no extreme imbalances)
2. Thresholds may need adjustment
3. Market conditions are quiet

================================================================================
SYSTEM STATUS
================================================================================

✅ Tracker Status: Running normally
⏱️  Last Run Duration: {duration:.1f} seconds
📊 Markets Processed: {len(active_markets)}
📈 Total Runs (Lifetime): {health['total_runs']}
🎯 Total Signals (Lifetime): {health.get('total_signals_generated', 0)}
📅 Last Signal: {health.get('last_signal_detected', 'Never')}

================================================================================
"""
            health_html = f"<html><body><pre>{health_text}</pre></body></html>"
            send_email_via_ses(health_subject, health_html, health_text)
            
            # Reset streak after sending health check
            health['consecutive_runs_no_signals'] = 0
    
    # Save updated health data
    save_json_to_s3(health, health_key)
    print(f"   💾 Health tracking updated")
    
    print("\n✅ Complete")


# =============================================================================
# AWS LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Entry point when running in Lambda.
    
    Schedule:
    - Midnight ET (00:00): Full discovery run + daily status email (always sent)
    - All other hours: Skip discovery, only email on signals
    """
    try:
        print("Lambda function started")
        print(f"Event: {json.dumps(event)}")
        
        # Check current hour in ET to determine if this is the daily discovery run
        current_time_et = datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        current_hour_et = current_time_et.hour
        
        is_daily_run = (current_hour_et == 0)  # Midnight ET
        
        if is_daily_run:
            print(f"🌙 DAILY DISCOVERY RUN (Midnight ET)")
            print(f"   - Will discover new markets")
            print(f"   - Will send daily status email regardless of signals")
            sys.argv = ['kalshi_order_book_tracker.py']  # No flags = full discovery
        else:
            print(f"⏰ HOURLY RUN ({current_hour_et:02d}:00 ET)")
            print(f"   - Skipping market discovery")
            print(f"   - Will only email if signals detected")
            sys.argv = ['kalshi_order_book_tracker.py', '--skip-discovery']
        
        main(is_daily_run=is_daily_run)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Kalshi order book tracking complete'})
        }
    except Exception as e:
        print(f"Error in Lambda handler: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


if __name__ == '__main__':
    main()
