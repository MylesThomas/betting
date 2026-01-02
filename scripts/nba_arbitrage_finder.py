"""
AWS Lambda Function - NBA Arb Alert (S3 Version)

PURPOSE:
Automated arbitrage detection that runs every 15 minutes during game hours (6pm-midnight ET)
to find profitable betting opportunities across NBA player props markets. Captures real-time
odds discrepancies and sends instant email alerts for high-value arbs.

STORAGE: S3 (migrated from Git for better performance and reliability)

WHY THIS MATTERS:
Arb opportunities:
- Appear and disappear within minutes as odds update
- Require instant alerts to capitalize before lines move
- Can guarantee profit regardless of outcome when executed correctly
- Are most valuable during live games when odds are volatile

CORE FUNCTIONALITY:
1. Fetch live NBA props from The Odds API (today's games only, ET timezone)
2. Find ALL arbitrage opportunities (any profit > 0%)
3. Calculate optimal stake allocation for guaranteed profit
4. Track staleness (how old bookmaker odds are)
5. Save ALL arbs to S3 with timestamp (for dashboard tracking)
6. Send email alert via SNS ONLY for high-value arbs (MIN_PROFIT_PCT% threshold)
7. Filter out excluded bookmakers from email alerts (but keep in CSV)
8. Handle errors gracefully and send failure alerts

S3 STRUCTURE:
    s3://betting-nba-arbs/nba/arbs/YYYY-MM-DD/arb_output_YYYYMMDD_HHMMSS.csv
    
    Example paths:
    - s3://betting-nba-arbs/nba/arbs/2025-12-24/arb_output_20251224_180000.csv
    - s3://betting-nba-arbs/nba/arbs/2025-12-24/arb_output_20251224_181500.csv
    - s3://betting-nba-arbs/nba/arbs/2025-12-24/arb_output_20251224_183000.csv
    
    Files organized by date (YYYY-MM-DD) for:
    - Easy querying (read all files for a day)
    - Automatic partitioning (Athena-compatible)
    - S3 lifecycle rules (auto-delete old data)
    
    Each file contains:
    - Best arb per player/market/line at that snapshot (15-min intervals)
    - All bookmakers (including ones excluded from email alerts)
    - Staleness indicators (is_stale, max_staleness, stale_bookmaker)
    - Guaranteed profit calculations (over_stake, under_stake, guaranteed_profit)

DATA SCHEMA (CSV columns):
- player: Player name (e.g., "LeBron James")
- market: Market type (e.g., "player_points", "player_threes")
- line: Line value (e.g., 25.5)
- best_over_odds: Best Over odds found (American format, e.g., +105)
- best_over_book: Bookmaker with best Over odds
- best_over_implied: Implied probability for Over
- best_over_last_update: When Over bookmaker last updated (ET format)
- best_under_odds: Best Under odds found (e.g., -110)
- best_under_book: Bookmaker with best Under odds
- best_under_implied: Implied probability for Under
- best_under_last_update: When Under bookmaker last updated (ET format)
- total_prob: Sum of implied probabilities (< 1.0 = arb exists)
- expected_profit_pct: Expected profit % (e.g., 5.2 = 5.2% profit)
- is_arb: True (only arbs saved to file)
- over_stake: $ to bet on Over (out of $100 total)
- under_stake: $ to bet on Under (out of $100 total)
- over_return: $ returned if Over wins
- under_return: $ returned if Under wins
- guaranteed_profit: $ profit guaranteed (min of over_return, under_return - 100)
- total_wager: Total $ wagered (default 100)
- recommendation: Human-readable bet recommendation
- game: Game matchup (e.g., "Lakers @ Warriors")
- game_time: ISO timestamp of game start
- game_is_live: True if game currently in progress
- num_bookmakers: How many bookmakers offer this line
- over_staleness_minutes: Minutes since Over bookmaker updated
- under_staleness_minutes: Minutes since Under bookmaker updated
- max_staleness: Max of over/under staleness
- is_stale: True if max_staleness > MAX_STALENESS_MINUTES threshold
- stale_bookmaker: Which bookmaker(s) have stale odds
- fetch_time_et: When we fetched from API (ET timezone)

DEDUPLICATION STRATEGY:
    - Lambda: Saves snapshot every 15 min with best odds at that moment
    - Dashboard: Reads all files for the day, groups by (player, market, line),
      keeps row with HIGHEST expected_profit_pct across all snapshots
    - This captures the BEST opportunity even if lines move between snapshots

LIVE PROP MARKET BEHAVIOR:
    - Lines change frequently throughout the day (every few minutes)
    - Same player/market/line may have different odds each 15-min snapshot
    - By saving all snapshots, we capture when arbs appear/disappear
    - Dashboard shows the BEST historical opportunity for each player/market/line
    - Late in day (after games start/end), fewer arbs expected

STALENESS TRACKING:
    - Bookmakers include timestamp when they last updated odds
    - We compare bookmaker's last_update against our fetch_at time
    - staleness_minutes = (fetch_at - last_update) / 60
    - Stale lines (> MAX_STALENESS_MINUTES) may have changed since API cache
    - Stale arbs still saved to CSV (for tracking) but flagged in email

EMAIL FILTERING:
    1. All arbs saved to CSV (including stale, including excluded bookmakers)
    2. Email only sent if:
       - At least one arb has expected_profit_pct >= MIN_PROFIT_PCT
    3. Email excludes:
       - Arbs involving EXCLUDED_BOOKMAKERS (but still in CSV)
    4. Email sections:
       - Fresh high-value arbs (detailed with stakes)
       - Fresh other arbs (compact format)
       - Stale arbs (with warnings)

TIME WINDOW OPTIMIZATION:
    - Lambda runs every 15 minutes (EventBridge rate expression)
    - Code checks current time: only run 6pm-midnight ET (game hours)
    - Within game hours, code checks if games are live or upcoming (30 min)
    - Only fetches props for live/upcoming games (massive credit savings!)
    - Saves API credits: ~87% reduction vs fetching all games every run
    - Before (no live filter): 6hr × 4 runs/hr × 71 credits/run = 1,704 credits/day
    - After (live filter): ~960 credits/day (most runs use only 1 credit)
    - Savings: 744 credits/day (~22k credits/month)

CREDIT USAGE (The Odds API):
    CRITICAL OPTIMIZATION: We only fetch props for LIVE or UPCOMING games!
    
    - 1 credit: Get today's events (1 call per run)
    - ~5 credits: Get props for 1 game (only if live or upcoming within 30 min)
    
    Typical scenarios:
    1. No games live/upcoming (early afternoon):
       - Only 1 credit (event list check)
       - Happens ~50% of run times
    
    2. A few games live (evening games starting):
       - 1 credit (events) + (5 × num_live_games) credits
       - Example: 3 games live = 1 + 15 = 16 credits
    
    3. Multiple games live (peak evening hours):
       - 1 credit (events) + (5 × num_live_games) credits
       - Peak: 10-14 games live = 1 + 50-70 = 51-71 credits
    
    Daily estimate (realistic) with 6pm-midnight window:
    - 6pm-7pm: ~3 games starting × 4 runs = 16 runs × 16 credits = 256
    - 7pm-9pm: ~10 games live × 8 runs = 8 runs × 51 credits = 408
    - 9pm-10pm: ~8 games live × 4 runs = 4 runs × 41 credits = 164
    - 10pm-midnight: ~3 games finishing × 8 runs = 8 runs × 16 credits = 128
    - Off-hours (no games): ~4 runs × 1 credit = 4
    
    Total per day: ~960 credits (down from 1,104 without live filtering!)
    Monthly: ~29,000 credits (87% reduction from fetching all games every run!)
    
    30-minute pre-game window: Catches pre-game line movements without wasting credits
    on games that are hours away.
    
    OLD ESTIMATE (without live filtering):
    - With time window: 1,104 credits/day = ~33k/month
    - Without time window: 4,416 credits/day = ~132k/month

ENVIRONMENT VARIABLES (set in Lambda configuration):
Required:
- SECRET_NAME: betting-dashboard-secrets (AWS Secrets Manager secret name)
- AWS_REGION_NAME: us-east-2 (your AWS region)
- S3_BUCKET_NAME: betting-nba-arbs (S3 bucket for arb storage)

Optional:
- SNS_TOPIC_ARN: arn:aws:sns:us-east-2:ACCOUNT_ID:betting-arb-alerts
  (leave empty to disable email alerts)
- MIN_PROFIT_PCT: 5.0 (default 5.0 - only email for 5%+ arbs)
- MAX_STALENESS_MINUTES: 2.0 (default 2.0 - flag odds older than 2 min)
- EXCLUDED_BOOKMAKERS: 'bovada,mybookieag' (comma-separated, no spaces)
  (bookmakers to exclude from EMAIL alerts but keep in CSV)

SECRETS (AWS Secrets Manager - secret name: betting-dashboard-secrets):
- ODDS_API_KEY: Your Odds API key from https://the-odds-api.com/

LAMBDA CONFIGURATION:
- Runtime: Python 3.12
- Memory: 256 MB (reduced from 512 MB - no git clone needed)
- Timeout: 60 seconds (reduced from 120s - S3 is much faster than git)
- Ephemeral storage: 512 MB (default)

LAMBDA LAYERS:
- betting-dashboard-dependencies (provides pandas, requests)
- NOTE: git-lambda2 layer NO LONGER NEEDED (was needed for git operations)

IAM PERMISSIONS (Lambda execution role):
- secretsmanager:GetSecretValue (read API key from Secrets Manager)
- sns:Publish (send email alerts via SNS)
- s3:PutObject (write arb files to S3)
- s3:GetObject (read arb files from S3 - for future features)
- s3:ListBucket (list files in S3 - for cleanup/maintenance)
- logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents (CloudWatch Logs)

SCHEDULE (EventBridge):
Option 1 (recommended): Rate expression with code-level time check
  - Rate: rate(15 minutes)
  - Code checks time and skips if outside 6pm-midnight ET
  - Simpler to configure, flexible time window in code

Option 2: Cron expression for exact hours
  - Cron: cron(0/15 22-4 * * ? *)  # Every 15 min, 10pm-4am UTC (6pm-midnight ET)
  - More precise but requires UTC conversion
  - Harder to adjust time window (need to update EventBridge rule)

BENEFITS OF S3 vs GIT:
- ⚡ 10-20s faster per run (no git clone/push operations)
- 💰 Lower Lambda costs (50% less memory, 50% shorter timeout)
- 🔒 More reliable (no git push failures or merge conflicts)
- 📊 Easier querying (S3 Select, Athena, native CSV reading)
- 🗂️ Better organization (automatic date partitioning, S3 lifecycle rules)
- 🚀 Scales better (handles high-frequency updates without git overhead)
- 🔧 Simpler setup (no GitHub tokens, no git layer)

MIGRATION NOTES (Git → S3):
What changed:
- Removed: git clone, git commit, git push operations
- Removed: GITHUB_TOKEN, GITHUB_REPO_URL, GITHUB_USERNAME, GITHUB_EMAIL env vars
- Removed: git-lambda2 layer
- Removed: run_command() function
- Added: S3 helper functions (save_to_s3, list_s3_files, etc.)
- Added: S3_BUCKET_NAME env var
- Reduced: Memory from 512MB → 256MB
- Reduced: Timeout from 120s → 60s

What stayed the same:
- API fetching logic (identical)
- Arb detection algorithm (identical)
- Email formatting (identical)
- SNS alerts (identical)
- Dashboard compatibility (CSV schema unchanged)

AWS SETUP (if not already configured):
Most AWS resources already exist, just need to add S3 bucket.

1. Create S3 Bucket
   Go to: https://s3.console.aws.amazon.com/s3/
   
   a) Create bucket:
      - Region: us-east-2 (match your Lambda region)
      - Bucket name: 'betting-nba-arbs' (must be globally unique)
      - Block all public access: YES (keep data private)
      - Versioning: Disabled (not needed for arb data)
      - Encryption: Server-side encryption with Amazon S3 managed keys (SSE-S3)
      - Click "Create bucket"
   
   b) Folder structure (auto-created by Lambda on first run):
      s3://betting-nba-arbs/
        nba/
          arbs/
            2025-12-24/
              arb_output_20251224_180000.csv
              arb_output_20251224_181500.csv
              arb_output_20251224_183000.csv
            2025-12-25/
              arb_output_20251225_180000.csv
              ...
   
   c) OPTIONAL: Set up lifecycle rule (skip to keep all historical data):
      RECOMMENDED: Skip this step to keep all data forever
      - Cost: ~$1-2/month for years of historical data (S3 is cheap)
      - Benefit: Dashboard can show yearly trends and all historical arbs
      
      Alternative (if you want to auto-delete old files):
      - Go to bucket → "Management" tab → "Create lifecycle rule"
      - Rule name: 'delete-old-arbs'
      - Rule scope: Apply to all objects in bucket
        - Check box: 'I acknowledge this rule will apply to all objects'
      - Lifecycle rule actions: Check "Expire current versions of objects"
      - Days after object creation: 365 (keep 1 year)
      - Click "Create rule"

2. Update Lambda IAM Role (add S3 permissions)
   a) Go to Lambda → Configuration → Permissions
   b) Click the execution role name (opens in IAM)
   c) Click "Add permissions" → "Attach policies"
   d) Search and attach: "AmazonS3FullAccess"
      (Or create custom policy with only PutObject, GetObject, ListBucket)
   e) Click "Add permissions"

3. Update Lambda Environment Variables
   a) Go to Lambda → Configuration → Environment variables → Edit
   b) Remove (no longer needed):
      - GITHUB_TOKEN
      - GITHUB_REPO_URL
      - GITHUB_USERNAME
      - GITHUB_EMAIL
   c) Add:
      - S3_BUCKET_NAME: betting-nba-arbs
   d) Keep existing:
      - SECRET_NAME: betting-dashboard-secrets
      - AWS_REGION_NAME: us-east-2
      - SNS_TOPIC_ARN: (your SNS ARN)
      - MIN_PROFIT_PCT: 10.0
      - MAX_STALENESS_MINUTES: 2.0
      - EXCLUDED_BOOKMAKERS: (if any)
   e) Click "Save"

4. Update Lambda Configuration
   a) Go to Lambda → Configuration → General configuration → Edit
   b) Update:
      - Memory: 256 MB (reduced from 512 MB)
      - Timeout: 60 seconds (reduced from 120s)
   c) Click "Save"

5. Remove Git Layer (no longer needed)
   a) Go to Lambda → Code → Layers
   b) Remove: git-lambda2 layer (if present)
   c) Keep: betting-dashboard-dependencies layer

6. Deploy Updated Code
   a) Copy this entire file to Lambda code editor
   b) Deploy
   c) Test manually (Test button in Lambda console)
   d) Check CloudWatch Logs for any errors
   e) Verify files appear in S3 bucket

7. Update Dashboard (Streamlit)
   ...

LOCAL TESTING (before deploying to Lambda):
1. Install dependencies:
   pip install pandas requests python-dotenv boto3

2. Configure AWS CLI (if not already):
   Check if AWS is configured:
   ```bash
   aws configure list
   ```
   
   If you see credentials listed (with **** hiding most of it), you're good!
   Boto3 will automatically use these credentials.
   
   If NOT configured, run:
   ```bash
   aws configure
   ```
   Then enter:
   - AWS Access Key ID: (get from AWS Console → IAM → Users → Security credentials)
   - AWS Secret Access Key: (from same place)
   - Default region name: us-east-2
   - Default output format: json (or press Enter)
   
   This saves credentials to ~/.aws/credentials:
   ```
   [default]
   aws_access_key_id = AKIAIOSFODNN7EXAMPLE
   aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   ```
   
   And config to ~/.aws/config:
   ```
   [default]
   region = us-east-2
   output = json
   ```
   
   To get AWS credentials (if you don't have them):
   - Go to: https://console.aws.amazon.com/iam/
   - Click "Users" → Find your user (e.g., 'myles')
   - Click "Security credentials" tab
   - Scroll to "Access keys" → Click "Create access key"
   - Choose "Command Line Interface (CLI)"
   - Click "Next" → "Create access key"
   - Copy both keys (you won't see the secret again!)
   - Run `aws configure` and paste them in

3. Create .env file:
   ODDS_API_KEY=your_odds_api_key
   S3_BUCKET_NAME=betting-nba-arbs
   AWS_REGION_NAME=us-east-2
   MIN_PROFIT_PCT=10.0
   MAX_STALENESS_MINUTES=2.0
   
   Note: AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) will be 
   auto-loaded from ~/.aws/credentials - no need to add them to .env!

4. Test modes:
   
   # Test API connection + save to S3 tmp/ folder (for testing)
   python scripts/nba_arbitrage_finder.py --test
   
   # Run for real (saves to production S3 path)
   python scripts/nba_arbitrage_finder.py
   
   # Dry run (fetch arbs but don't save to S3)
   python scripts/nba_arbitrage_finder.py --dry-run

5. Verify S3 (test mode):
   aws s3 ls s3://betting-nba-arbs/tmp/nba/arbs/ --recursive
   
6. Verify S3 (production mode):
   aws s3 ls s3://betting-nba-arbs/nba/arbs/ --recursive

7. Deploy to Lambda once tested

WORKFLOW (similar to track_game_line_movements.py):
```bash
cd betting

# Test API + save to S3 tmp/ folder
python scripts/nba_arbitrage_finder.py --test

# Run for real - saves to production S3
python scripts/nba_arbitrage_finder.py

# Deploy to Lambda
# 1. Copy code to Lambda console
# 2. Deploy
# 3. Test (create test event if needed)
# 4. Check CloudWatch logs
# 5. Verify S3 files created
# 6. Check email alerts
```

ERROR HANDLING:
- Missing API key: Raise exception with clear message
- API rate limit: Log warning, skip this run
- No games today: Save empty CSV, return success
- API error: Log error, send failure email via SNS
- S3 write error: Raise exception (critical failure)
- SNS error: Log warning (non-critical, arbs still saved)
- Lambda timeout: Will NOT send email (set up CloudWatch alarm separately)
- Lambda out-of-memory: Will NOT send email (monitor CloudWatch metrics)

MONITORING:
- CloudWatch Logs: Check for errors after each run
- CloudWatch Metrics: Monitor Lambda duration, memory usage
- S3 Bucket: Verify files being created every 15 min
- Email Alerts: Should receive when high-value arbs found
- API Usage: Monitor at https://the-odds-api.com/account/

COST ESTIMATE (monthly):
- Lambda: $0 (free tier: 1M requests/month, we use ~4,320/month)
- S3: ~$0.10/month (storage for 90 days of CSVs)
- Secrets Manager: $0.40/month
- SNS: $0 (free tier: 1,000 emails/month)
- EventBridge: $0 (free tier)
- TOTAL: ~$0.50/month (vs ~$2-3/month with git operations)

RELATED FILES:
- docs/dashboard_s3_reader.py (Streamlit dashboard S3 integration)
- docs/s3_iam_policy.json (IAM policy template for S3 access)
- scripts/migrate_git_to_s3.py (one-time migration of historical data)

AUTHOR: Myles Thomas
CREATED: 2025-12-06
UPDATED: 2025-12-24 (migrated from Git to S3)
"""

import json
import os
import boto3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from botocore.exceptions import ClientError
from io import StringIO

# These come from Lambda layer (or pip install locally)
import requests
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'basketball_nba'
REGIONS = 'us'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'
TIMEZONE = 'America/New_York'

# S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'betting-nba-arbs')
assert S3_BUCKET == "betting-nba-arbs", "S3_BUCKET must be set to 'betting-nba-arbs'"
IS_LAMBDA = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ

# Markets to check
MARKETS = 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists'

# Bookmakers to exclude from EMAIL ALERTS (but still saved to CSV for dashboard)
# Set via environment variable: EXCLUDED_BOOKMAKERS='bovada,mybookieag'
# Leave empty to get alerts for all bookmakers
EXCLUDED_BOOKMAKERS = os.environ.get('EXCLUDED_BOOKMAKERS', '').split(',') if os.environ.get('EXCLUDED_BOOKMAKERS') else []

# Market display names
MARKET_DISPLAY_NAMES = {
    'player_threes': 'Threes',
    'player_points': 'Points',
    'player_rebounds': 'Rebounds',
    'player_assists': 'Assists',
    'player_blocks': 'Blocks',
    'player_steals': 'Steals',
    'player_double_double': 'Double-Double',
    'player_triple_double': 'Triple-Double',
    'player_points_rebounds_assists': 'Pts+Reb+Ast'
}


# ============================================================================
# S3 HELPER FUNCTIONS
# ============================================================================

def get_s3_key(timestamp: datetime, is_test: bool = False) -> str:
    """
    Generate S3 key (path) for arb output file.
    
    Args:
        timestamp: Datetime for filename (in ET timezone)
        is_test: If True, save to tmp/ folder for testing
    
    Returns:
        S3 key like: nba/arbs/2025-12-24/arb_output_20251224_180000.csv
        Or (test): tmp/nba/arbs/arb_output_20251224_180000.csv
    """
    filename = f"arb_output_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    
    if is_test:
        # Test mode - save to tmp/ folder
        return f"tmp/nba/arbs/{filename}"
    else:
        # Production mode - save to date-partitioned folder
        date_str = timestamp.strftime('%Y-%m-%d')
        return f"nba/arbs/{date_str}/{filename}"


def save_to_s3(df: pd.DataFrame, timestamp: datetime, is_test: bool = False, dry_run: bool = False) -> str:
    """
    Save DataFrame to S3 as CSV.
    
    Args:
        df: DataFrame with arb data
        timestamp: Datetime for filename (in ET timezone)
        is_test: If True, save to tmp/ folder for testing
        dry_run: If True, skip S3 upload (for local testing)
    
    Returns:
        S3 key where file was saved (or would be saved if dry_run)
    """
    s3_key = get_s3_key(timestamp, is_test)
    
    if dry_run:
        print(f"💡 DRY RUN - Would save to: s3://{S3_BUCKET}/{s3_key}")
        print(f"   Rows: {len(df)}")
        return s3_key
    
    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload to S3
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION_NAME', 'us-east-2'))
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        if is_test:
            print(f"🧪 TEST: Saved to s3://{S3_BUCKET}/{s3_key}")
            print(f"   (View at: https://s3.console.aws.amazon.com/s3/object/{S3_BUCKET}?prefix={s3_key})")
        else:
            print(f"💾 Saved to s3://{S3_BUCKET}/{s3_key}")
        
        return s3_key
    except Exception as e:
        print(f"❌ Error saving to S3: {e}")
        raise


def list_todays_arbs_s3(date: datetime = None) -> list:
    """
    List all arb files in S3 for a given date.
    
    Args:
        date: Date to list files for (defaults to today in ET)
    
    Returns:
        List of S3 keys
    """
    if date is None:
        date = datetime.now(ZoneInfo(TIMEZONE))
    
    date_str = date.strftime('%Y-%m-%d')
    prefix = f"nba/arbs/{date_str}/"
    
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION_NAME', 'us-east-2'))
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
    except Exception as e:
        print(f"Warning: Failed to list S3 files: {e}")
        return []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_secrets():
    """
    Fetch secrets from AWS Secrets Manager.
    
    Returns:
        dict: Contains ODDS_API_KEY
    """
    # For local testing, check environment variables first
    odds_key = os.environ.get('ODDS_API_KEY')
    
    if odds_key:
        return {'ODDS_API_KEY': odds_key}
    
    # Fetch from Secrets Manager
    secret_name = os.environ.get('SECRET_NAME', 'betting-dashboard-secrets')
    region_name = os.environ.get('AWS_REGION_NAME', 'us-east-2')
    
    client = boto3.client('secretsmanager', region_name=region_name)
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise Exception(f"Failed to retrieve secret: {e}")
    
    return json.loads(response['SecretString'])


def send_email(subject, message):
    """Send email via SNS."""
    topic_arn = os.environ.get('SNS_TOPIC_ARN')
    if not topic_arn:
        print("⚠️  No SNS_TOPIC_ARN - skipping email")
        return
    
    region = os.environ.get('AWS_REGION_NAME', 'us-east-2')
    sns = boto3.client('sns', region_name=region)
    
    response = sns.publish(
        TopicArn=topic_arn,
        Subject=subject[:100],  # SNS subject limit
        Message=message
    )
    print(f"✅ Email sent (MessageId: {response['MessageId']})")


def american_to_probability(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_arb(over_odds, under_odds):
    """Calculate arb metrics."""
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    total_prob = over_prob + under_prob
    
    is_arb = total_prob < 1.0
    profit_pct = ((1 / total_prob) - 1) * 100 if total_prob > 0 else 0
    
    return {
        'is_arb': is_arb,
        'profit_pct': profit_pct,
        'total_prob': total_prob
    }


def calculate_stakes(over_odds, under_odds, total=100):
    """Calculate optimal stake allocation."""
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    
    over_stake = (over_prob / (over_prob + under_prob)) * total
    under_stake = (under_prob / (over_prob + under_prob)) * total
    
    return round(over_stake, 2), round(under_stake, 2)


def is_game_live(game_time_str, current_time=None):
    """
    Determine if a game is currently live.
    
    Args:
        game_time_str: ISO format game start time (e.g., '2025-12-18T19:00:00Z')
        current_time: Optional datetime to compare against (defaults to now)
    
    Returns:
        bool: True if game is live, False if upcoming or finished
    """
    if not game_time_str:
        return False
    
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    try:
        game_start = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
        
        # Game is live if:
        # - Started (current time > start time)
        # - Not finished (less than 3 hours since start)
        time_since_start = (current_time - game_start).total_seconds() / 3600  # hours
        
        if time_since_start > 0 and time_since_start < 3:
            return True
        
        return False
    except:
        return False


def check_if_live(events, current_time=None):
    """
    Filter events to only those that are currently live or starting soon.
    
    This is a critical optimization: we only fetch props for games that are 
    actually in progress or about to start. Pre-game and post-game, we just 
    check event list (1 credit). During live games, we fetch props (5 credits per game).
    
    Args:
        events: List of event dicts from API (with 'commence_time' field)
        current_time: Optional datetime to compare against (defaults to now)
    
    Returns:
        tuple: (live_events, upcoming_events)
            - live_events: Events currently in progress
            - upcoming_events: Events starting within next 30 minutes
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    live_events = []
    upcoming_events = []
    
    for event in events:
        game_time_str = event.get('commence_time')
        
        if is_game_live(game_time_str, current_time):
            live_events.append(event)
        else:
            # Check if starting soon (within 30 minutes)
            try:
                game_start = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                time_until_start = (game_start - current_time).total_seconds() / 60  # minutes
                
                if 0 <= time_until_start <= 30:
                    upcoming_events.append(event)
            except:
                pass
    
    return live_events, upcoming_events


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_todays_events(api_key):
    """Get today's NBA events."""
    url = f"{API_BASE_URL}/sports/{SPORT}/events"
    response = requests.get(url, params={'apiKey': api_key}, verify=False)
    response.raise_for_status()
    
    events = response.json()
    
    # Filter for today
    tz = ZoneInfo(TIMEZONE)
    today = datetime.now(tz).date()
    
    todays = []
    for event in events:
        event_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        if event_time.astimezone(tz).date() == today:
            todays.append(event)
    
    return todays


def get_event_props(api_key, event_id):
    """Get props for a single event."""
    url = f"{API_BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT
    }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    return response.json()


def parse_props(event_data, api_fetch_time=None):
    """
    Parse event props into list of dicts with staleness tracking.
    
    Args:
        event_data: API response for a single event
        api_fetch_time: datetime when API was called (UTC)
    
    Returns:
        list of dicts with prop data including staleness info
    """
    props = []
    
    if api_fetch_time is None:
        api_fetch_time = datetime.now(timezone.utc)
    
    game = f"{event_data['away_team']} @ {event_data['home_team']}"
    game_time = event_data.get('commence_time')
    
    # Convert fetch time to ET for display
    et_tz = ZoneInfo(TIMEZONE)
    fetch_time_et = api_fetch_time.astimezone(et_tz).strftime('%Y-%m-%d %H:%M:%S ET')
    
    for bookmaker in event_data.get('bookmakers', []):
        book = bookmaker['key']
        bookmaker_last_update = bookmaker.get('last_update')  # Capture bookmaker timestamp
        
        for market in bookmaker.get('markets', []):
            market_key = market['key']
            market_last_update = market.get('last_update')  # Capture market timestamp
            
            # Calculate staleness (use market timestamp if available, else bookmaker)
            last_update_str = market_last_update or bookmaker_last_update
            staleness_minutes = None
            
            if last_update_str:
                try:
                    last_update_dt = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                    staleness_seconds = (api_fetch_time - last_update_dt).total_seconds()
                    staleness_minutes = staleness_seconds / 60.0
                except:
                    pass
            
            player_lines = {}
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', 'Unknown')
                line = outcome.get('point')
                odds = outcome.get('price')
                bet_type = outcome.get('name')
                
                key = (player, line)
                if key not in player_lines:
                    player_lines[key] = {
                        'player': player,
                        'market': market_key,
                        'line': line,
                        'bookmaker': book,
                        'game': game,
                        'game_time': game_time,
                        'bookmaker_last_update': bookmaker_last_update,
                        'market_last_update': market_last_update,
                        'staleness_minutes': staleness_minutes,
                        'api_fetch_time': api_fetch_time.isoformat(),
                        'fetch_time_et': fetch_time_et
                    }
                
                if bet_type == 'Over':
                    player_lines[key]['over_odds'] = odds
                elif bet_type == 'Under':
                    player_lines[key]['under_odds'] = odds
            
            props.extend(player_lines.values())
    
    return props


def find_arbs(all_props, min_profit_pct=0.0, total_stake=100.0, max_staleness_minutes=2.0):
    """
    Find all arbs with profit >= min_profit_pct (default 0 = all arbs).
    Includes staleness tracking for all bookmakers.
    
    Args:
        all_props: List of prop dicts with staleness info
        min_profit_pct: Minimum profit % threshold
        total_stake: Total wager amount for stake calculations
        max_staleness_minutes: Max minutes since last update (default 2.0)
    
    Returns:
        list of dicts with full arb info compatible with dashboard schema
    """
    if not all_props:
        return []
    
    df = pd.DataFrame(all_props)
    
    arbs = []
    
    for (market, player, line), group in df.groupby(['market', 'player', 'line']):
        overs = group[group['over_odds'].notna()]
        unders = group[group['under_odds'].notna()]
        
        if overs.empty or unders.empty:
            continue
        
        best_over_idx = overs['over_odds'].idxmax()
        best_under_idx = unders['under_odds'].idxmax()
        
        best_over = overs.loc[best_over_idx]
        best_under = unders.loc[best_under_idx]
        
        arb = calculate_arb(best_over['over_odds'], best_under['under_odds'])
        
        if arb['is_arb'] and arb['profit_pct'] >= min_profit_pct:
            over_stake, under_stake = calculate_stakes(
                best_over['over_odds'], 
                best_under['under_odds'],
                total=total_stake
            )
            
            # Calculate returns
            over_odds = best_over['over_odds']
            under_odds = best_under['under_odds']
            
            if over_odds > 0:
                over_return = over_stake * (1 + over_odds / 100)
            else:
                over_return = over_stake * (1 + 100 / abs(over_odds))
            
            if under_odds > 0:
                under_return = under_stake * (1 + under_odds / 100)
            else:
                under_return = under_stake * (1 + 100 / abs(under_odds))
            
            guaranteed_profit = min(over_return, under_return) - total_stake
            
            # Build recommendation string
            recommendation = f"Bet ${over_stake:.2f} Over @ {best_over['bookmaker']}, ${under_stake:.2f} Under @ {best_under['bookmaker']}"
            
            # Staleness tracking
            over_staleness = best_over.get('staleness_minutes', 0) or 0
            under_staleness = best_under.get('staleness_minutes', 0) or 0
            max_staleness = max(over_staleness, under_staleness)
            
            # Determine if stale and which bookmaker(s)
            is_stale = max_staleness > max_staleness_minutes
            stale_bookmakers = []
            if over_staleness > max_staleness_minutes:
                stale_bookmakers.append(best_over['bookmaker'])
            if under_staleness > max_staleness_minutes and best_under['bookmaker'] not in stale_bookmakers:
                stale_bookmakers.append(best_under['bookmaker'])
            
            stale_bookmaker = ', '.join(stale_bookmakers) if stale_bookmakers else None
            
            # Get last update times for bookmakers
            over_last_update = best_over.get('market_last_update') or best_over.get('bookmaker_last_update')
            under_last_update = best_under.get('market_last_update') or best_under.get('bookmaker_last_update')
            
            # Convert to ET timezone for display
            et_tz = ZoneInfo(TIMEZONE)
            over_update_et = None
            under_update_et = None
            
            if over_last_update:
                try:
                    over_dt = datetime.fromisoformat(over_last_update.replace('Z', '+00:00'))
                    over_update_et = over_dt.astimezone(et_tz).strftime('%I:%M:%S %p ET')
                except:
                    pass
            
            if under_last_update:
                try:
                    under_dt = datetime.fromisoformat(under_last_update.replace('Z', '+00:00'))
                    under_update_et = under_dt.astimezone(et_tz).strftime('%I:%M:%S %p ET')
                except:
                    pass
            
            # Determine if game is live
            game_time_str = group['game_time'].iloc[0]
            game_is_live = is_game_live(game_time_str)
            
            arbs.append({
                'player': player,
                'market': market,
                'line': line,
                'best_over_odds': int(over_odds),
                'best_over_book': best_over['bookmaker'],
                'best_over_implied': arb['total_prob'] - american_to_probability(under_odds),  # over prob
                'best_over_last_update': over_update_et,
                'best_under_odds': int(under_odds),
                'best_under_book': best_under['bookmaker'],
                'best_under_implied': american_to_probability(under_odds),
                'best_under_last_update': under_update_et,
                'total_prob': arb['total_prob'],
                'expected_profit_pct': arb['profit_pct'],
                'is_arb': arb['is_arb'],
                'over_stake': over_stake,
                'under_stake': under_stake,
                'over_return': round(over_return, 2),
                'under_return': round(under_return, 2),
                'guaranteed_profit': round(guaranteed_profit, 2),
                'total_wager': total_stake,
                'recommendation': recommendation,
                'game': group['game'].iloc[0],
                'game_time': group['game_time'].iloc[0],
                'game_is_live': game_is_live,
                'num_bookmakers': len(group['bookmaker'].unique()),
                'over_staleness_minutes': over_staleness,
                'under_staleness_minutes': under_staleness,
                'max_staleness': max_staleness,
                'is_stale': is_stale,
                'stale_bookmaker': stale_bookmaker,
                'fetch_time_et': best_over.get('fetch_time_et', '')
            })
    
    return sorted(arbs, key=lambda x: x['expected_profit_pct'], reverse=True)




# ============================================================================
# EMAIL FORMATTING
# ============================================================================

def format_arb_email(high_value_arbs, other_arbs, stale_arbs, max_staleness_minutes=2.0):
    """
    Format arbs into email body with staleness indicators.
    
    Args:
        high_value_arbs: Fresh high-value arbs (above threshold)
        other_arbs: Fresh arbs below threshold
        stale_arbs: Stale arbs (any profit level)
        max_staleness_minutes: Threshold for staleness in minutes
    """
    now = datetime.now(ZoneInfo(TIMEZONE))
    fresh_arbs = high_value_arbs + other_arbs
    total_arbs = len(fresh_arbs) + len(stale_arbs)
    
    # Header depends on whether we have high-value arbs
    if high_value_arbs:
        header = "🚨 high-value nba arbs found! 🚨"
        arb_summary = f"arbs found: {total_arbs} ({len(high_value_arbs)} high-value, {len(fresh_arbs)} fresh, {len(stale_arbs)} stale)"
    else:
        header = "📊 nba arb scan complete"
        arb_summary = f"arbs found: {total_arbs} ({len(fresh_arbs)} fresh, {len(stale_arbs)} stale)"
    
    lines = [
        header,
        "",
        f"time: {now.strftime('%Y-%m-%d %I:%M %p ET')}",
        arb_summary,
        "",
        "=" * 50,
        ""
    ]
    
    # High-value FRESH arbs (detailed format)
    if high_value_arbs:
        lines.extend([
            "✅ FRESH ARBS (NOT STALE):",
            ""
        ])
        
        for i, arb in enumerate(high_value_arbs, 1):
            market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
            
            # Build timestamp info
            fetch_time = arb.get('fetch_time_et', 'Unknown')
            over_update = arb.get('best_over_last_update', 'Unknown')
            under_update = arb.get('best_under_last_update', 'Unknown')
            
            # Staleness info
            max_staleness = arb.get('max_staleness', 0)
            is_stale = arb.get('is_stale', False)
            
            if max_staleness < max_staleness_minutes:
                staleness_status = f"⏱️  Staleness: {max_staleness:.1f} min < {max_staleness_minutes:.1f} min threshold ✅ NOT STALE"
            else:
                staleness_status = f"⏱️  Staleness: {max_staleness:.1f} min > {max_staleness_minutes:.1f} min threshold ⚠️ STALE"
            
            # Game status
            game_status = "🔴 LIVE" if arb.get('game_is_live', False) else "⏰ UPCOMING"
            
            lines.extend([
                f"#{i} - {arb['expected_profit_pct']:.2f}% PROFIT ✅",
                f"   Player: {arb['player']}",
                f"   Market: {market_display} {arb['line']}",
                f"   Game: {arb['game']}",
                "",
                f"   📈 OVER {arb['line']}: {arb['best_over_odds']:+d} @ {arb['best_over_book']}",
                f"      Line updated: {over_update}",
                f"   📉 UNDER {arb['line']}: {arb['best_under_odds']:+d} @ {arb['best_under_book']}",
                f"      Line updated: {under_update}",
                f"   🕐 Data pulled: {fetch_time}",
                f"   {staleness_status}",
                f"   {game_status}",
                "",
                f"   💰 Stake $100 total:",
                f"      → ${arb['over_stake']:.2f} on OVER @ {arb['best_over_book']}",
                f"      → ${arb['under_stake']:.2f} on UNDER @ {arb['best_under_book']}",
                f"      → Guaranteed profit: ${arb['guaranteed_profit']:.2f}",
                "",
                "-" * 50,
                ""
            ])
    
    # Other FRESH arbs (compact format)
    if other_arbs:
        if high_value_arbs:
            lines.extend([
                "",
                "=" * 50,
                "📋 other fresh arbs (below threshold):",
                "=" * 50,
                ""
            ])
        else:
            lines.extend([
                "✅ FRESH ARBS (NOT STALE):",
                ""
            ])
        
        for i, arb in enumerate(other_arbs, len(high_value_arbs) + 1):
            market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
            fetch_time = arb.get('fetch_time_et', 'Unknown')
            over_update = arb.get('best_over_last_update', 'Unknown')
            under_update = arb.get('best_under_last_update', 'Unknown')
            
            # Staleness info
            max_staleness = arb.get('max_staleness', 0)
            
            if max_staleness < max_staleness_minutes:
                staleness_status = f"⏱️  {max_staleness:.1f} min < {max_staleness_minutes:.1f} min threshold ✅ NOT STALE"
            else:
                staleness_status = f"⏱️  {max_staleness:.1f} min > {max_staleness_minutes:.1f} min threshold ⚠️ STALE"
            
            # Game status
            game_status = "🔴 LIVE" if arb.get('game_is_live', False) else "⏰ UPCOMING"
            
            lines.extend([
                f"#{i} - {arb['expected_profit_pct']:.2f}% | {arb['player']} | {market_display} {arb['line']} ✅",
                f"     Game: {arb['game']}",
                f"     Over {arb['best_over_odds']:+d} @ {arb['best_over_book']} (updated {over_update})",
                f"     Under {arb['best_under_odds']:+d} @ {arb['best_under_book']} (updated {under_update})",
                f"     Data pulled: {fetch_time}",
                f"     {staleness_status}",
                f"     {game_status}",
                ""
            ])
    
    # STALE arbs section
    if stale_arbs:
        lines.extend([
            "",
            "=" * 50,
            "⚠️  STALE ARBS (lines may have changed):",
            "=" * 50,
            ""
        ])
        
        for i, arb in enumerate(stale_arbs, len(fresh_arbs) + 1):
            market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
            staleness = arb.get('max_staleness', 0)
            stale_book = arb.get('stale_bookmaker', 'Unknown')
            fetch_time = arb.get('fetch_time_et', 'Unknown')
            over_update = arb.get('best_over_last_update', 'Unknown')
            under_update = arb.get('best_under_last_update', 'Unknown')
            
            # Staleness info
            staleness_status = f"⏱️  {staleness:.1f} min > {max_staleness_minutes:.1f} min threshold ⚠️ STALE"
            
            # Game status
            game_status = "🔴 LIVE" if arb.get('game_is_live', False) else "⏰ UPCOMING"
            
            lines.extend([
                f"#{i} - {arb['expected_profit_pct']:.2f}% | {arb['player']} | {market_display} {arb['line']} ⚠️ STALE",
                f"     Game: {arb['game']}",
                f"     Over {arb['best_over_odds']:+d} @ {arb['best_over_book']} (updated {over_update})",
                f"     Under {arb['best_under_odds']:+d} @ {arb['best_under_book']} (updated {under_update})",
                f"     Data pulled: {fetch_time}",
                f"     {staleness_status}",
                f"     {game_status}",
                f"     ⚠️  {stale_book} data is {staleness:.1f} min old - verify before betting!",
                ""
            ])
    
    lines.extend([
        "",
        "⚡ act fast - lines move quickly!",
        "✅ = fresh lines (safe to bet)",
        "⚠️  = stale lines (double-check before betting)",
        "",
        "Dashboard: https://tqs-props-dashboard.streamlit.app"
    ])
    
    return "\n".join(lines)


# ============================================================================
# LAMBDA HANDLER
# ============================================================================

def lambda_handler(event, context):
    """Main Lambda handler - fetches arbs, saves to S3, sends alerts."""
    now = datetime.now(ZoneInfo(TIMEZONE))
    
    # Check for test/dry-run modes (set by local CLI)
    is_test = os.environ.get('TEST_MODE') == 'true'
    dry_run = os.environ.get('DRY_RUN') == 'true'
    
    print("=" * 60)
    if is_test:
        print("🧪 TEST MODE - Saving to S3 tmp/ folder")
    elif dry_run:
        print("🔍 DRY RUN - No files will be saved")
    print("🏀 NBA Arb Alert Check (15-min)")
    print(f"Time: {now.strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)
    
    # ========================================================================
    # TIME CHECK: Only run during game hours (6pm-midnight ET)
    # ========================================================================
    force_run = os.environ.get('FORCE_RUN') == 'true'
    current_hour = now.hour
    
    # Skip if between midnight (0) and 6pm (18), unless --force flag is set
    if current_hour < 18 and not force_run:
        skip_msg = f"⏰ Skipping - outside game hours (current: {now.strftime('%I:%M %p ET')})"
        print(skip_msg)
        print("   Game hours: 6:00 PM - 11:59 PM ET")
        print("   This saves ~960 credits/day during off-hours!")
        print("   💡 Use --force flag to run anyway")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'skipped': True,
                'reason': 'outside_game_hours',
                'time': now.isoformat(),
                'message': skip_msg
            })
        }
    
    if force_run:
        print(f"⚡ FORCE RUN - Ignoring time check (current: {now.strftime('%I:%M %p ET')})")
    else:
        print(f"✅ Within game hours (6pm-midnight ET) - proceeding...")
    # ========================================================================
    
    min_profit = float(os.environ.get('MIN_PROFIT_PCT', '5.0'))
    print(f"Looking for arbs with {min_profit}%+ edge...")
    
    try:
        # Step 1: Get secrets
        print("\n📊 Step 1: Fetching secrets...")
        secrets = get_secrets()
        odds_api_key = secrets['ODDS_API_KEY']
        print("✅ Secrets retrieved")
        
        # Step 2: Fetch today's events (ET timezone)
        print("\n🔍 Step 2: Fetching today's NBA events...")
        events = get_todays_events(odds_api_key)
        print(f"Found {len(events)} games today (ET)")
        
        if not events:
            print("No games today - saving empty file")
            empty_df = pd.DataFrame(columns=[
                'player', 'market', 'line', 'best_over_odds', 'best_over_book',
                'best_over_implied', 'best_over_last_update', 'best_under_odds', 
                'best_under_book', 'best_under_implied', 'best_under_last_update',
                'total_prob', 'expected_profit_pct', 'is_arb',
                'over_stake', 'under_stake', 'over_return', 'under_return',
                'guaranteed_profit', 'total_wager', 'recommendation', 'game',
                'game_time', 'game_is_live', 'num_bookmakers', 'over_staleness_minutes',
                'under_staleness_minutes', 'max_staleness', 'is_stale',
                'stale_bookmaker', 'fetch_time_et'
            ])
            s3_key = save_to_s3(empty_df, now, is_test=is_test, dry_run=dry_run)
            return {'statusCode': 200, 'body': json.dumps({'message': 'No games today', 's3_key': s3_key})}
        
        # Step 3: Filter to only LIVE games (API credit optimization)
        print("\n🎯 Step 3: Filtering to live games only...")
        api_fetch_time = datetime.now(timezone.utc)
        live_events, upcoming_events = check_if_live(events, api_fetch_time)
        
        print(f"   Live games: {len(live_events)}")
        print(f"   Upcoming games (within 30 min): {len(upcoming_events)}")
        
        # Include upcoming games in prop fetch (lines are active)
        events_to_fetch = live_events + upcoming_events
        
        if not events_to_fetch:
            print("   No live or upcoming games - saving empty file")
            print("   💡 API credit savings: Only 1 credit used (event list only)!")
            empty_df = pd.DataFrame(columns=[
                'player', 'market', 'line', 'best_over_odds', 'best_over_book',
                'best_over_implied', 'best_over_last_update', 'best_under_odds', 
                'best_under_book', 'best_under_implied', 'best_under_last_update',
                'total_prob', 'expected_profit_pct', 'is_arb',
                'over_stake', 'under_stake', 'over_return', 'under_return',
                'guaranteed_profit', 'total_wager', 'recommendation', 'game',
                'game_time', 'game_is_live', 'num_bookmakers', 'over_staleness_minutes',
                'under_staleness_minutes', 'max_staleness', 'is_stale',
                'stale_bookmaker', 'fetch_time_et'
            ])
            s3_key = save_to_s3(empty_df, now, is_test=is_test, dry_run=dry_run)
            return {
                'statusCode': 200, 
                'body': json.dumps({
                    'message': 'No live/upcoming games',
                    's3_key': s3_key,
                    'total_games': len(events),
                    'live_games': 0,
                    'credits_used': 1
                })
            }
        
        print(f"   Fetching props for {len(events_to_fetch)} games (live + upcoming)")
        print(f"   💡 Skipping {len(events) - len(events_to_fetch)} games (not live/upcoming)")
        
        # Step 4: Fetch props for live/upcoming games only
        print("\n📥 Step 4: Fetching props for live/upcoming games...")
        all_props = []
        for event in events_to_fetch:
            try:
                props_data = get_event_props(odds_api_key, event['id'])
                props = parse_props(props_data, api_fetch_time=api_fetch_time)
                all_props.extend(props)
                
                # Show if game is live or upcoming
                is_live = event in live_events
                status = "🔴 LIVE" if is_live else "⏰ UPCOMING"
                print(f"  ✓ {event['away_team']} @ {event['home_team']}: {len(props)} props {status}")
            except Exception as e:
                print(f"  ✗ Error fetching {event['id']}: {e}")
        
        print(f"\nTotal props: {len(all_props)}")
        print(f"💡 API credits saved by skipping {len(events) - len(events_to_fetch)} non-live games: ~{(len(events) - len(events_to_fetch)) * 5} credits")
        
        # Step 5: Find arbs (includes ALL bookmakers for CSV/dashboard)
        print("\n🔍 Step 5: Finding arbitrage opportunities...")
        max_staleness_minutes = float(os.environ.get('MAX_STALENESS_MINUTES', '2.0'))
        all_arbs = find_arbs(all_props, min_profit_pct=0.0, max_staleness_minutes=max_staleness_minutes)
        
        print(f"\n{'='*60}")
        print(f"📊 ALL ARBS FOUND: {len(all_arbs)}")
        
        # Staleness summary
        if all_arbs:
            stale_arbs = [a for a in all_arbs if a.get('is_stale', False)]
            fresh_arbs = [a for a in all_arbs if not a.get('is_stale', False)]
            print(f"   ✅ Fresh lines (≤{max_staleness_minutes} min): {len(fresh_arbs)}")
            print(f"   ⚠️  Stale lines (>{max_staleness_minutes} min): {len(stale_arbs)}")
        
        print(f"{'='*60}")
        
        if all_arbs:
            for i, arb in enumerate(all_arbs, 1):
                market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
                alert_flag = "🚨" if arb['expected_profit_pct'] >= min_profit else "  "
                stale_flag = " [STALE]" if arb.get('is_stale', False) else ""
                print(f"{alert_flag} {i:2d}. {arb['expected_profit_pct']:5.2f}% | {arb['player']:<25s} | {market_display} {arb['line']}{stale_flag}")
                print(f"        Over {arb['best_over_odds']:+4d} @ {arb['best_over_book']:<12s} | Under {arb['best_under_odds']:+4d} @ {arb['best_under_book']}")
                if arb.get('is_stale', False):
                    staleness = arb.get('max_staleness', 0)
                    stale_book = arb.get('stale_bookmaker', 'Unknown')
                    print(f"        ⚠️  Stale: {stale_book} ({staleness:.1f} min old)")
        else:
            print("   No arbs found at all.")
        
        print(f"{'='*60}\n")
        
        # Step 6: Save output file to S3 (sorted by expected_profit_pct descending)
        # All arbs are saved to CSV (including stale ones for tracking)
        print("💾 Step 6: Saving output file to S3...")
        if all_arbs:
            arbs_df = pd.DataFrame(all_arbs)
            arbs_df = arbs_df.sort_values('expected_profit_pct', ascending=False)
        else:
            arbs_df = pd.DataFrame(columns=[
                'player', 'market', 'line', 'best_over_odds', 'best_over_book',
                'best_over_implied', 'best_over_last_update', 'best_under_odds', 
                'best_under_book', 'best_under_implied', 'best_under_last_update',
                'total_prob', 'expected_profit_pct', 'is_arb',
                'over_stake', 'under_stake', 'over_return', 'under_return',
                'guaranteed_profit', 'total_wager', 'recommendation', 'game',
                'game_time', 'game_is_live', 'num_bookmakers', 'over_staleness_minutes',
                'under_staleness_minutes', 'max_staleness', 'is_stale',
                'stale_bookmaker', 'fetch_time_et'
            ])
        
        s3_key = save_to_s3(arbs_df, now, is_test=is_test, dry_run=dry_run)
        print(f"   💡 Note: CSV includes all arbs (stale ones flagged with is_stale=True)")
        
        # Step 7: Send email alert if any arbs found (fresh or stale)
        # Filter out excluded bookmakers from EMAIL ALERTS (but they're already saved to CSV)
        if EXCLUDED_BOOKMAKERS:
            pre_filter_count = len(all_arbs)
            all_arbs = [
                a for a in all_arbs 
                if a['best_over_book'] not in EXCLUDED_BOOKMAKERS 
                and a['best_under_book'] not in EXCLUDED_BOOKMAKERS
            ]
            filtered_count = pre_filter_count - len(all_arbs)
            if filtered_count > 0:
                print(f"\n📧 EMAIL FILTERING:")
                print(f"   Filtered out {filtered_count} arbs involving: {', '.join(EXCLUDED_BOOKMAKERS)}")
                print(f"   (These are still saved to CSV for dashboard)")
        
        # Separate fresh from stale arbs
        fresh_arbs = [a for a in all_arbs if not a.get('is_stale', False)]
        stale_arbs = [a for a in all_arbs if a.get('is_stale', False)]
        
        high_value_arbs = [a for a in fresh_arbs if a['expected_profit_pct'] >= min_profit]
        other_arbs = [a for a in fresh_arbs if a['expected_profit_pct'] < min_profit]
        
        print(f"\n📊 EMAIL FILTERING:")
        print(f"   Total arbs found: {len(all_arbs)}")
        print(f"   Fresh arbs (≤{max_staleness_minutes} min): {len(fresh_arbs)}")
        print(f"   Stale arbs (included in email): {len(stale_arbs)}")
        print(f"   Fresh arbs with {min_profit}%+ edge: {len(high_value_arbs)}")
        
        # Send email if we have any HIGH-VALUE arbs (fresh or stale)
        high_value_arbs_all = [a for a in all_arbs if a['expected_profit_pct'] >= min_profit]
        
        if high_value_arbs_all:
            # Subject line indicates staleness status
            if high_value_arbs:
                # Use fresh arbs for subject (count and best profit)
                arb_count = len(high_value_arbs)
                best_profit = high_value_arbs[0]['expected_profit_pct']
                staleness_indicator = f" ({len(high_value_arbs)} FRESH, {len(stale_arbs)} STALE)" if stale_arbs else " (ALL FRESH)"
            else:
                # All high-value arbs are stale
                arb_count = len(high_value_arbs_all)
                best_profit = high_value_arbs_all[0]['expected_profit_pct']
                staleness_indicator = " (ALL STALE)"
            
            subject = f"🚨 {arb_count} NBA ARB(S)! {best_profit:.1f}%{staleness_indicator}"
            
            message = format_arb_email(high_value_arbs, other_arbs, stale_arbs, max_staleness_minutes)
            
            print("\n" + "=" * 60)
            print("📧 SENDING ALERT EMAIL (high-value arbs found)")
            print("=" * 60)
            
            send_email(subject, message)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'total_arbs': len(all_arbs),
                    'fresh_arbs': len(fresh_arbs),
                    'stale_arbs': len(stale_arbs),
                    'high_value_arbs': len(high_value_arbs_all),
                    'high_value_fresh': len(high_value_arbs),
                    'best_profit': best_profit,
                    'alert_sent': True,
                    's3_key': s3_key
                })
            }
        else:
            print("No high-value arbs found - no alert sent")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'total_arbs': len(all_arbs),
                    'fresh_arbs': len(fresh_arbs),
                    'stale_arbs': len(stale_arbs),
                    'high_value_arbs': 0,
                    'alert_sent': False,
                    's3_key': s3_key
                })
            }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Send failure alert via SNS
        error_msg = f"""❌ NBA Arb Alert Check FAILED

Time: {now.strftime('%Y-%m-%d %I:%M %p ET')}
Error: {str(e)}

Traceback:
{traceback.format_exc()[:500]}

Check CloudWatch logs for full details:
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups/log-group/$252Faws$252Flambda$252F{os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'unknown')}

NOTE: Lambda timeouts and out-of-memory errors won't trigger this email.
Check CloudWatch metrics and set up separate alarms for those.
"""
        send_email("❌ NBA Arb Alert Failed", error_msg)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# ============================================================================
# LOCAL TESTING CLI
# ============================================================================

def main():
    """
    Main function for local testing with command-line arguments.
    
    Usage:
        # Test API connection (saves to S3 tmp/ folder for testing)
        python scripts/nba_arbitrage_finder.py --test
        
        # Run for real (saves to production S3)
        python scripts/nba_arbitrage_finder.py
        
        # Run without saving to S3 (dry run - not implemented yet)
        # python scripts/nba_arbitrage_finder.py --dry-run
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NBA Arb Alert - Find arbitrage opportunities in NBA props'
    )
    parser.add_argument('--test', action='store_true',
                       help='Test mode - saves to S3 tmp/ folder instead of production path')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - fetch arbs but do not save to S3 (for testing)')
    parser.add_argument('--force', action='store_true',
                       help='Force run outside game hours (skip time check)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    if args.test:
        print("🧪 TEST MODE - Saving to S3 tmp/ folder")
    elif args.dry_run:
        print("🔍 DRY RUN - No files will be saved")
    else:
        print("🏀 PRODUCTION MODE - Saving to production S3")
    
    if args.force:
        print("⚡ FORCE MODE - Skipping time check")
    
    print("=" * 60)
    
    # Set environment flags
    if args.test:
        os.environ['TEST_MODE'] = 'true'
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
    if args.force:
        os.environ['FORCE_RUN'] = 'true'
    
    # Run lambda handler
    result = lambda_handler({}, None)
    
    print("\n" + "=" * 60)
    print("✅ Complete")
    print("=" * 60)
    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    # For local testing
    import ssl
    import urllib3
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run main CLI
    main()

