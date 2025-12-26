"""
Track Betting Line Movement for NBA/NFL Spreads

PURPOSE:
Automated line movement detection that runs hourly to capture spread changes
at the game/market/bookmaker level. Tracks both short-term (1-hour) and 
medium-term (24-hour) line movement to identify betting opportunities and 
market inefficiencies.

WHY THIS MATTERS:
Line movement signals:
- Sharp money (professionals moving the market)
- Injury news or lineup changes
- Public betting patterns
- Steam moves (rapid line changes across multiple books)
- Potential value bets (catching the right side of a move)

CORE FUNCTIONALITY:
1. Fetch current spreads from The Odds API every hour
2. Iterate through: League -> Game -> Bookmaker
   - For each NBA game, get line from FanDuel, DraftKings, BetMGM, etc.
   - For each NFL game, get line from FanDuel, DraftKings, BetMGM, etc.
   - Store each bookmaker's line as a separate row (NOT consensus/average)
3. Calculate vig-adjusted spread score for each bookmaker's line
4. Compare against snapshots from 1 hour ago and 24 hours ago
5. Detect changes at two granularity levels:
   - GAME level: Specific matchup (e.g., LAL vs BOS, tracking each book separately)
   - BOOKMAKER level: Individual book's line for that game (this is primary focus)
   
   Example: If LAL vs BOS game has 8 bookmakers, we track 8 separate lines.
   If FanDuel moves their line but DraftKings doesn't, we capture that.

6. Store historical snapshots with timestamps for comparison
7. Generate alerts/reports for significant movement (e.g., >= 0.5 points adjusted)

DATA STRUCTURE:
Stored as timestamped CSV files in data/01_input/the-odds-api/{sport}/line_movement/

CRITICAL: Each row is ONE bookmaker's line for ONE game (not aggregated/consensus).
We iterate: league -> game -> bookmaker, storing each book's line separately.

Each snapshot contains:
- game_id: Unique game identifier from API
- game_time: Scheduled game start time (UTC)
- away_team: Away team name
- home_team: Home team name
- bookmaker: Bookmaker key (e.g., 'fanduel', 'draftkings', 'betmgm')
- away_spread: Raw spread for away team (e.g., -3.5)
- away_price: American odds for away spread (e.g., -110, -130, +105)
- home_spread: Implied from away_spread (e.g., +3.5)
- home_price: American odds for home spread (e.g., -110, -105)
- away_adjusted_spread: Vig-adjusted score for away (see formula below)
- home_adjusted_spread: Vig-adjusted score for home (see formula below)
- fetched_at: When WE fetched from API (ISO format UTC)
- last_bookmaker_update: When bookmaker last updated this line (from API)

Example rows:
game_id, bookmaker, away_spread, away_price, away_adjusted_spread
abc123, fanduel, -2.5, -110, -2.50  (standard vig)
abc123, draftkings, -2.5, -130, -2.75  (high vig = line shaded up)
abc123, betmgm, -3.0, -105, -2.90  (low vig = line shaded down)

VIG-ADJUSTED SPREAD SCORING:
Instead of tracking raw spreads (e.g., -2.5), calculate a "true spread score" 
that accounts for the vig/odds. This reveals where the bookmaker ACTUALLY thinks 
the line is, not just the posted number.

WHY THIS MATTERS:
Books use vig to shade lines without moving the number. If you see:
  - DraftKings: -2.5 at -110
  - FanDuel: -2.5 at -130
These are NOT the same line! FanDuel thinks the true line is closer to -3.

CONCEPTUAL APPROACH:
- Baseline: -110 on both sides = true line is the posted number
- Higher price (worse odds like -130): Book thinks line should be higher/further
- Lower price (better odds like -105): Book thinks line should be lower/closer
- Vig encodes bookmaker's true belief about half-point increments

MATHEMATICAL FORMULA:
Linear approximation works for normal vig range (-120 to +100), but we need
a non-linear approach for extremes like -200 or +150.

Step 1: Calculate vig distance from baseline
  vig_cents = (price - (-110)) / 10  (in dimes)

Step 2: Apply tiered adjustment factors (handles extremes better)
  if vig_cents <= -3.0:  (price >= -140)
    # Extreme high vig: diminishing returns
    base_adjustment = -3.0 * 0.015 = -0.045
    excess = vig_cents - (-3.0)
    adjustment = base_adjustment + (excess * 0.025)  # steeper curve
  
  elif vig_cents >= 3.0:  (price <= +80)
    # Extreme low vig: diminishing returns  
    base_adjustment = 3.0 * 0.015 = 0.045
    excess = vig_cents - 3.0
    adjustment = base_adjustment + (excess * 0.025)  # steeper curve
  
  else:
    # Normal range: linear
    adjustment = vig_cents * 0.015

  adjusted_spread = raw_spread + adjustment

Alternative (simpler): Use square root to naturally handle extremes
  sign = +1 if vig_cents >= 0 else -1
  adjustment = sign * sqrt(abs(vig_cents)) * 0.025
  adjusted_spread = raw_spread + adjustment

EXAMPLES (using tiered approach):
  Raw: -2.5 at -110
    vig_cents = 0, adjustment = 0
    Adjusted: -2.50 (baseline)
  
  Raw: -2.5 at -130
    vig_cents = -2.0, adjustment = -2.0 * 0.015 = -0.030
    Adjusted: -2.53 (slight move toward -3)
  
  Raw: -2.5 at -200
    vig_cents = -9.0 (extreme!)
    base = -0.045, excess = -6.0 * 0.025 = -0.15
    adjustment = -0.195
    Adjusted: -2.695 (major move toward -3, almost at -2.7)
  
  Raw: -2.5 at -300
    vig_cents = -19.0 (very extreme!)
    base = -0.045, excess = -16.0 * 0.025 = -0.40
    adjustment = -0.445
    Adjusted: -2.945 (effectively at -3.0)
  
  Raw: -3.0 at +100
    vig_cents = +1.0, adjustment = +0.015
    Adjusted: -2.985 (slight move toward -2.5)
  
  Raw: -3.0 at +150
    vig_cents = +2.6 (favorable odds)
    adjustment = +2.6 * 0.015 = +0.039
    Adjusted: -2.96 (move toward -2.5)

ALTERNATIVE (sqrt approach) - may be more intuitive:
  Raw: -2.5 at -200
    vig_cents = -9.0
    adjustment = -1 * sqrt(9.0) * 0.025 = -3.0 * 0.025 = -0.075
    Adjusted: -2.575
  
  Raw: -2.5 at -300  
    vig_cents = -19.0
    adjustment = -1 * sqrt(19.0) * 0.025 = -4.36 * 0.025 = -0.109
    Adjusted: -2.609

LOOKUP TABLE (using tiered approach):
  Price  | Vig Cents | Adjustment | Example: -2.5 becomes | Notes
  -------|-----------|------------|-----------------------|------------------
  -400   | -29.0     | -0.695     | -3.195               | Extreme vig
  -300   | -19.0     | -0.445     | -2.945               | Very high vig
  -250   | -14.0     | -0.320     | -2.820               | High vig
  -200   | -9.0      | -0.195     | -2.695               | High vig
  -175   | -6.5      | -0.1325    | -2.6325              | High vig
  -150   | -4.0      | -0.0725    | -2.5725              | Moderate vig
  -140   | -3.0      | -0.045     | -2.545               | Threshold
  -135   | -2.5      | -0.0375    | -2.5375              | Normal range
  -130   | -2.0      | -0.030     | -2.530               | Normal range
  -125   | -1.5      | -0.0225    | -2.5225              | Normal range
  -120   | -1.0      | -0.015     | -2.515               | Normal range
  -115   | -0.5      | -0.0075    | -2.5075              | Normal range
  -110   |  0.0      |  0.00      | -2.50                | Baseline
  -105   | +0.5      | +0.0075    | -2.4925              | Normal range
  +100   | +1.0      | +0.015     | -2.485               | Normal range
  +105   | +1.5      | +0.0225    | -2.4775              | Normal range
  +110   | +2.0      | +0.030     | -2.470               | Normal range
  +115   | +2.5      | +0.0375    | -2.4625              | Normal range
  +120   | +3.0      | +0.045     | -2.455               | Threshold
  +130   | +4.0      | +0.0700    | -2.430               | Favorable odds
  +140   | +5.0      | +0.0950    | -2.405               | Favorable odds
  +150   | +6.0      | +0.1200    | -2.380               | Favorable odds
  +200   | +11.0     | +0.245     | -2.255               | Very favorable
  +300   | +21.0     | +0.495     | -2.005               | Extreme

NOTE: For spreads, extreme vig like -300 is rare but can happen when books
are very confident (injury news, weather, etc.). More common in props/futures.

MOVEMENT DETECTION:
For each league/game/bookmaker combination:
- Load current snapshot (t0)
- Load 1-hour-ago snapshot (t-1h)
- Load 24-hour-ago snapshot (t-24h)
- Calculate vig-adjusted spread scores for all snapshots
- Calculate movement using adjusted scores:
  * hourly_movement = adjusted_spread_current - adjusted_spread_1h_ago
  * daily_movement = adjusted_spread_current - adjusted_spread_24h_ago
- Flag significant moves (threshold: 0.5 points or more on adjusted scale)

CRITICAL: CROSSED ZERO DETECTION
Lines that move through zero are ALWAYS flagged as significant, regardless of 
threshold. This indicates a favorite/underdog flip.

We check BOTH time windows independently:
- crossed_zero_1h: Did line cross zero between 1h ago and now?
- crossed_zero_24h: Did line cross zero between 24h ago and now?

Examples:
  Hourly cross (1h ago -> now):
  - Away spread: -1.5 @ 12:00 -> +1.5 @ 13:00 (crossed_zero_1h = TRUE)
  
  Daily cross (24h ago -> now):
  - Away spread: +2.5 yesterday @ 13:00 -> -0.5 today @ 13:00 (crossed_zero_24h = TRUE)
  
  Both (line has been volatile):
  - Away spread: +3.0 (24h ago) -> -1.0 (1h ago) -> +2.0 (now)
    crossed_zero_24h = TRUE, crossed_zero_1h = TRUE

Detection logic:
  crossed_zero_1h = (prev_1h_spread < 0 and current_spread > 0) OR 
                    (prev_1h_spread > 0 and current_spread < 0) OR
                    (prev_1h_spread == 0 or current_spread == 0)
  
  crossed_zero_24h = (prev_24h_spread < 0 and current_spread > 0) OR 
                     (prev_24h_spread > 0 and current_spread < 0) OR
                     (prev_24h_spread == 0 or current_spread == 0)

This approach captures:
1. Traditional line moves (2.5 -> 3.0)
2. Vig moves (2.5 @ -110 -> 2.5 @ -130)
3. Combination moves (2.5 @ -110 -> 3.0 @ -105)
4. Zero-crossing moves (ALWAYS significant)

SPORTS COVERAGE:
- NBA: 'basketball_nba' (regular season + playoffs)
- NFL: 'americanfootball_nfl' (regular season + playoffs)

MARKETS:
- Spreads only (NOT moneylines)
- Focus on closing lines (as game time approaches)

GAME FILTERING (Future Games Only):
Only fetch lines for games that have NOT started yet:
- Games starting later today (after current hour)
- Games starting within next 14 days

API Parameters:
  commenceTimeFrom: now (current UTC time)
  commenceTimeTo: now + 14 days

Rationale:
- Past games: No lines available, waste of API credits
- Today (future hours): Yes, track closing line movement
- Next 14 days: Yes, track opening/middle line movement
- Beyond 14 days: Lines may not be posted yet, or too far out to be meaningful

NOTE: Test with --test mode first to verify we're not pulling unnecessary data

BOOKMAKERS (from config):
- Major US books: FanDuel, DraftKings, BetMGM, Caesars, etc.
- Regional books: Pinnacle, Circa, etc.
- Track all available in 'us' region from API

AUTOMATION:
Designed to run as a cron job (hourly):
0 * * * * cd /Users/thomasmyles/dev/betting && /path/to/venv/bin/python scripts/track_line_movement.py --prod-run >> logs/line_movement.log 2>&1

OR as AWS Lambda (EventBridge hourly trigger)

USAGE:
    # Hourly cron job (production, no prompts)
    python scripts/track_game_line_movements.py --prod-run
    
    # Manual run (with confirmations)
    python scripts/track_game_line_movements.py
    
    # Specific sport only
    python scripts/track_game_line_movements.py --sport nba
    python scripts/track_game_line_movements.py --sport nfl
    
    # Generate movement report from existing snapshots (no new fetch)
    python scripts/track_game_line_movements.py --report-only
    
    # Custom movement threshold
    python scripts/track_game_line_movements.py --movement-threshold 1.0
    
    # Check API usage without saving (uses real API, shows what you'd get)
    python scripts/track_game_line_movements.py --check-api-usage
    
    Note: Saves to S3 tmp/ folder for testing (won't interfere with production data)

TESTING STRATEGY:
Before running in production, test to avoid wasting API credits:

1. First, check what the API would return (dry run, uses 1 API call, saves to S3 tmp/):
   python scripts/track_game_line_movements.py --check-api-usage
   
   Output: Shows actual games, schedules, bookmakers, saves to S3 tmp/ folder

2. Then run for real with one sport:
   python scripts/track_game_line_movements.py --sport nba
   
   Confirms: Everything works, saves snapshot for future comparisons

3. If all good, enable both sports in production:
   python scripts/track_game_line_movements.py --prod-run

LOCAL TESTING WITH S3:
This script uses S3 for all storage (local and Lambda). To test locally:

1. Install dependencies:
   pip install pandas numpy requests python-dotenv boto3
   
2. Check if you already have AWS credentials configured:
   Run: aws configure list
   
   If you see access_key and secret_key (with *** hiding most of it), you're good!
   Boto3 will automatically use these credentials - no need to add to .env
   
   If NOT configured, set up AWS CLI:
   a) Run: aws configure
   b) Enter your AWS Access Key ID
   c) Enter your AWS Secret Access Key
   d) Enter region: us-east-2
   e) Enter output format: json (or leave blank)
   
   OR manually add to .env (less secure, only if AWS CLI not working):
   AWS_ACCESS_KEY_ID=your_aws_access_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
   AWS_DEFAULT_REGION=us-east-2
   
   To get AWS credentials (if you don't have them):
   - Go to AWS Console → IAM → Users → Your user
   - Security credentials → Access keys → Create access key
   - Choose "Command Line Interface (CLI)"
   - Save both keys securely! (as well as AWS_DEFAULT_REGION=us-east-2)

3. Add S3 permissions to your IAM user (required for S3 read/write):
   a) Go to: https://console.aws.amazon.com/iam/
   b) Click "Users" → Find your user (e.g., 'myles')
   c) Click "Add permissions" -> "Add permissions" -> "Attach policies directly"
   d) Search for: "AmazonS3FullAccess"
   e) Check the box next to it
   f) Click "Next" → "Add permissions"
   
   This allows your user to read/write to S3 buckets.

   Note: These are for us-east-1, us-east-2, and all other regions.

4. Set up .env file with minimal required variables:
   ODDS_API_KEY=your_odds_api_key_here
   S3_BUCKET=betting-line-movement-snapshots
   
   (AWS credentials from step 2 will be used automatically)
   
5. Test API connection (saves to S3 tmp/ folder for testing):
   python scripts/track_game_line_movements.py --check-api-usage
   
   Check S3: s3://betting-line-movement-snapshots/tmp/snapshots/
   
6. Test with S3 (saves snapshots to S3):
   python scripts/track_game_line_movements.py --sport nba
   python scripts/track_game_line_movements.py --sport nfl
   
7. Verify in S3 console:
   - Go to S3 bucket: betting-line-movement-snapshots
   - Check for files in: data/01_input/the-odds-api/nba/line_movement/
   - Check for files in: data/01_input/the-odds-api/nfl/line_movement/
   
8. Run again (should detect no movements on first comparison):
   python scripts/track_game_line_movements.py --sport nba
   python scripts/track_game_line_movements.py --sport nfl
   
9. After 1 hour, run again (should detect real movements if any):
   python scripts/track_game_line_movements.py --sport nba
   python scripts/track_game_line_movements.py --sport nfl

Note: Local runs save to S3 but do NOT send emails. Emails only sent when 
running in Lambda (auto-detected via AWS_LAMBDA_FUNCTION_NAME env var).

OUTPUT:
1. Timestamped snapshot files in S3 (CSV)
   S3 Paths:
   - s3://betting-line-movement-snapshots/data/01_input/the-odds-api/nba/line_movement/snapshot_20251224_120000.csv
   - s3://betting-line-movement-snapshots/data/01_input/the-odds-api/nfl/line_movement/snapshot_20251224_120000.csv
   
   Columns per snapshot:
   - game_id, game_time, away_team, home_team
   - bookmaker (one row per bookmaker per game)
   - away_spread, away_price, away_adjusted_spread
   - home_spread, home_price, home_adjusted_spread
   - fetched_at, last_bookmaker_update
   - game_time_et, fetched_at_et, last_bookmaker_update_et (string versions in ET)

2. Movement files in S3 (CSV)
   S3 Paths:
   - s3://betting-line-movement-snapshots/data/04_output/line_movement/nba_movements_20251224_120000.csv
   - s3://betting-line-movement-snapshots/data/04_output/line_movement/nfl_movements_20251224_120000.csv
   
   Columns:
   - game_id, away_team, home_team, game_time
   - bookmaker (individual book that moved)
   - side ('away' | 'home')
   - current_raw_spread, current_price, current_adjusted_spread
   - prev_1h_raw_spread, prev_1h_price, prev_1h_adjusted_spread
   - prev_24h_raw_spread, prev_24h_price, prev_24h_adjusted_spread
   - hourly_movement_raw, hourly_movement_adjusted
   - daily_movement_raw, daily_movement_adjusted
   - significant_hourly (bool), significant_daily (bool)
   - crossed_zero_1h (bool) - TRUE if line crossed 0 in last hour
   - crossed_zero_24h (bool) - TRUE if line crossed 0 in last 24 hours
   - movement_direction ('toward_favorite' | 'toward_underdog' | 'none')
   - movement_type ('line_only' | 'vig_only' | 'both')
   - timestamp

3. Email alerts (Lambda only, via SNS)
   Subject: 🚨 Line Movement Alert - Dec 24, 2025 13:00 ET
   Body: HTML email with two sections:
   - SIGNIFICANT MOVEMENTS (≥threshold or crossed zero)
   - ALL OTHER MOVEMENTS (below threshold)
   
   Each movement shows:
   - Game (Away @ Home)
   - Bookmaker
   - Side
   - Was (1h ago): spread/price
   - Was (24h ago): spread/price
   - Now: spread/price
   - Type: line_only, vig_only, both
   - 🚨 if crossed zero

4. Console/CloudWatch output (both local and Lambda)
   - Games tracked: N (NBA), M (NFL)
   - Total bookmaker-game combinations: X
   - Bookmakers tracked: [list]
   - Significant hourly moves: N (threshold >= 0.5 adjusted points)
   - Significant daily moves: M (threshold >= 0.5 adjusted points)
   - Steam moves detected: K (same direction across 3+ books)
   - Top movers by absolute adjusted movement
   - Books with most movement (by count)

DEPENDENCIES:
- The Odds API (requires ODDS_API_KEY in .env)
- pandas, numpy, requests for data manipulation and API calls
- boto3 for AWS S3 and SNS (included in Lambda runtime, install locally)
- python-dotenv for local .env file loading (not needed in Lambda)
- AWS S3 bucket for storing snapshots and movements
- AWS SNS topic for email alerts (Lambda only)

ERROR HANDLING:
- Missing API key: Exit with error message
- API rate limit hit: Log warning, use cached data
- No comparison data (first run): Log info, skip comparison
- Missing 1h snapshot: Skip hourly comparison, log info, continue with daily
- Missing 24h snapshot: Skip daily comparison, log info, continue with hourly
- Missing BOTH snapshots: Log warning, save current snapshot only (no movements)
- Missing games in comparison: Handle gracefully (new games added to schedule)
- Bookmaker dropped from API: Handle gracefully (book stopped offering)

MISSING SNAPSHOT LOGIC:
When comparing snapshots, handle missing data gracefully:

1. If prev_1h snapshot doesn't exist for a game/bookmaker:
   - hourly_movement_raw = None
   - hourly_movement_adjusted = None
   - significant_hourly = False
   - crossed_zero_1h = False
   - Log: "No 1h snapshot for {game_id} / {bookmaker}, skipping hourly comparison"
   - Continue to check 24h snapshot

2. If prev_24h snapshot doesn't exist for a game/bookmaker:
   - daily_movement_raw = None
   - daily_movement_adjusted = None
   - significant_daily = False
   - crossed_zero_24h = False
   - Log: "No 24h snapshot for {game_id} / {bookmaker}, skipping daily comparison"
   - Continue to check 1h snapshot

3. If BOTH missing (first time seeing this game/bookmaker combo):
   - Don't write to movement alerts CSV (nothing to compare)
   - Save to snapshot CSV for future comparisons
   - Log: "First snapshot for {game_id} / {bookmaker}, no comparison available"

This is expected behavior when:
- Script runs for the first time (no historical data)
- New games added to schedule mid-day
- Bookmaker starts offering a line they didn't have before
- Snapshot file was deleted/corrupted

DESIGN DECISIONS:
1. Store raw snapshots (not just movements) for flexibility
2. Keep 7 days of snapshots (delete older), compress if needed
3. Movement threshold configurable (default 0.5 points)
4. Run hourly but support ad-hoc runs for debugging
5. One snapshot file per sport per run (not combined)
6. Use ISO timestamps in filenames for sortability
7. Fail fast if API key missing (don't use fake data)
8. Compare using game_id + bookmaker as composite key
9. Handle timezone properly (store all times in UTC)
10. Don't commit large snapshot files (add to .gitignore)

FUTURE ENHANCEMENTS:
- Slack/Discord alerts for steam moves
- Compare against consensus vs specific books
- Track vig changes (spread odds movement)
- Detect reverse line movement (line moves opposite to bet%)
- Integration with injury reports API
- Machine learning to predict line movement
- Track closing line value (CLV) for our bets

------------------------------------------------------------------------------------------------

AWS LAMBDA AUTOMATION SETUP:
For fully automated hourly tracking, deploy to AWS Lambda with EventBridge trigger.

ARCHITECTURE:
  EventBridge (Scheduler) → Triggers hourly (e.g., every hour on the hour)
  Lambda Function → Runs this script with --prod-run
  S3 Bucket (optional) → Stores snapshots long-term
  SNS Topic → Sends email alerts for significant movements

SETUP STEPS:

1. AWS Secrets Manager - Store API key
   Go to: https://console.aws.amazon.com/secretsmanager/
   
   a) Store a new secret
      - Secret type: Other type of secret
      - Key/value pairs:
        * Key: ODDS_API_KEY | Value: [your_odds_api_key]
      - Encryption key: Default encryption key (aws/secretsmanager)
      - Click "Next"
   
   b) Name the secret
      - Secret name: 'line-movement-secrets'
      - Description: (leave blank)
      - Tags: (leave blank)
      - Click "Next"
   
   c) Configure rotation
      - Automatic rotation: Disabled
      - Click "Next"
   
   d) Review and store
      - Click "Store"
   
   e) Save the Secret ARN (for Lambda permissions)
      - After creation, click on the secret name
      - Copy the ARN (looks like: arn:aws:secretsmanager:us-east-2:123456789012:secret:line-movement-secrets-xxxxx)
      - You'll need this ARN in Step 5

2. Create S3 Bucket
   Go to: https://s3.console.aws.amazon.com/s3/
   
   a) Create bucket:
      - Region: us-east-2 (or your preferred region)
      - Bucket type: General purpose
      - Bucket name: 'betting-line-movement-snapshots' (must be globally unique)
      - Block all public access: YES (keep data private)
      - Versioning: Disabled (not needed)
      - Encryption: Server-side encryption with Amazon S3 managed keys (SSE-S3)
      - Bucket Key: Enable
      - Click "Create bucket"
   
   b) Set up folder structure (OPTIONAL - Lambda will create automatically)
      You can skip this step! Lambda will create folders automatically on first run.
      
      If you want to see the structure in advance, it will look like:
      
      s3://betting-line-movement-snapshots/
        data/
          01_input/
            the-odds-api/
              nba/
                line_movement/
                  snapshot_20251224_120000.csv
                  snapshot_20251224_130000.csv
              nfl/
                line_movement/
                  snapshot_20251224_120000.csv
          04_output/
            line_movement/
              nba_movements_20251224_120000.csv
              nfl_movements_20251224_120000.csv
      
      To create manually (optional): Click bucket → Create folder → enter names above
   
   c) Set up lifecycle rule (auto-delete old snapshots):
      - Go to bucket → "Management" tab → "Create lifecycle rule"
      - Rule name: 'delete-old-snapshots'
      - Rule scope: Apply to all objects in bucket
          - Check box for 'I acknowledge that this rule will apply to all objects in the bucket.'
      - Lifecycle rule actions: Check "Expire current versions of objects"
      - Days after object creation: 365 (keep full year of history)
      - Click "Create rule"
      
      Note: S3 is Lambda's working storage. You get movements via email (SNS),
      so you rarely need to look at these files. 365 days gives you full year
      of historical data for analysis if needed.
      
      Cost: ~$0.50-1.00/month for year of data.
      
      Alternative: Skip lifecycle rule entirely to keep forever (minimal cost increase).
   
   d) Note the Bucket ARN (for Lambda permissions):
      Example: arn:aws:s3:::betting-line-movement-snapshots/*

3. Create SNS Topic for Email Alerts
   Go to: https://console.aws.amazon.com/sns/
   
   a) Create topic:
      - Topic -> Create topic:
      - Region: us-east-2 (top right, make sure you are in same as your other resources)
      - Type: Standard
      - Name: line-movement-alerts
      - Display name: Line Movement Alerts
      - Click "Create topic"
   
   b) Copy the Topic ARN:
      - After creation, you'll see: arn:aws:sns:us-east-2:123456789012:line-movement-alerts
      - Save this ARN for Lambda environment variables (Step 5)
   
   c) Create email subscription:
      - Click "Create subscription"
      - Protocol: Email
      - Endpoint: your-email@example.com
      - Click "Create subscription"
   
   d) Confirm subscription:
      - Check your email inbox
      - Click "Confirm subscription" link in AWS email
      - Status should change to "Confirmed" in SNS console
   
   Lambda will send alerts when:
   - Significant movements detected (> threshold)
   - Lines crossed zero
   - Steam moves (3+ books same direction)

4. Create Lambda Function
   a) Go to AWS Lambda: https://console.aws.amazon.com/lambda/
   b) Click "Create function"
   c) Configure function:
      - Name: 'track-line-movement-hourly'
      - Runtime: Python 3.12
      - Architecture: x86_64
      - Click "Create function"
   d) Further configure the function:
      - Configuration -> General configuration -> Edit:
      - Memory: 512 MB
      - Ephemeral storage: 512 MB (default is fine)
      - Timeout: 5 minutes
      - Click "Save"

5. Lambda Environment Variables
   - ODDS_API_KEY: (from Secrets Manager)
   - S3_BUCKET: betting-line-movement-snapshots
   - SNS_TOPIC_ARN: (from Step 3, ARN like: arn:aws:sns:us-east-2:123456789012:line-movement-alerts)
   - MOVEMENT_THRESHOLD: 0.5
   - AWS_REGION_NAME: us-east-2
   - Click "Save"

6. Lambda IAM Role Permissions
   a) Lambda -> Configuration -> Permissions tab
   b) Under "Execution role", click the blue "Role name" link (e.g., track-line-movement-hourly-role-d681af1m)
      - This opens the IAM role in a new tab
   c) In the IAM role page, click "Permissions policies" -> "Add permissions" -> "Attach policies"
   d) Search and attach these three policies:
      - SecretsManagerReadWrite (to read API key from Secrets Manager)
      - AmazonS3FullAccess (to store/retrieve snapshots)
      - AmazonSNSFullAccess (to send email alerts)
   e) Click "Add permissions" button to confirm
   
   Note: The role already has CloudWatch Logs permissions (created automatically)

7. Lambda Layers (Python Dependencies)
   This script now requires matplotlib for chart generation, which makes the layer larger.
   
   Required packages for Lambda layer:
   - pandas, numpy, matplotlib, requests (not in Lambda runtime by default)
   
   NOT needed (already available):
   - boto3: Already included in Lambda runtime
   - python-dotenv: Lambda uses environment variables directly
   
   a) Create layer locally:
   ```bash
   mkdir -p lambda_layer/python
   pip install --platform manylinux2014_x86_64 \
     --target=lambda_layer/python \
     --python-version 3.12 \
     --only-binary=:all: \
     pandas numpy matplotlib requests
   cd lambda_layer && zip -r layer.zip python
   ```
   
   b) Upload to S3 (layer will be ~74MB, too large for direct upload):
   ```bash
   aws s3 cp layer.zip s3://betting-line-movement-snapshots/lambda-layers/line-movement-dependencies-layer.zip --region us-east-2
   ```
   
   c) Publish layer from S3:
   ```bash
   aws lambda publish-layer-version \
     --layer-name line-movement-dependencies \
     --description "pandas, numpy, matplotlib, requests for line movement tracking" \
     --content S3Bucket=betting-line-movement-snapshots,S3Key=lambda-layers/line-movement-dependencies-layer.zip \
     --compatible-runtimes python3.12 \
     --compatible-architectures x86_64 \
     --region us-east-2
   ```
   
   Copy the LayerVersionArn from the output (looks like: arn:aws:lambda:us-east-2:ACCOUNT_ID:layer:line-movement-dependencies:1)
   
   d) Attach layer to Lambda function:
   ```bash
   aws lambda update-function-configuration \
     --function-name track-line-movement-hourly \
     --layers arn:aws:lambda:us-east-2:ACCOUNT_ID:layer:line-movement-dependencies:1 \
     --region us-east-2
   ```
   
   Replace ACCOUNT_ID with your AWS account ID from the LayerVersionArn.
   
   Alternative (via Console):
   - Lambda -> track-line-movement-hourly -> Code tab -> Layers section (scroll down)
   - Click "Add a layer"
   - Layer source: Custom layers
   - Choose: line-movement-dependencies
   - Version: 1 (or latest)
   - Click "Add"


8. Lambda Handler Code (Deploy this script)
   
   This script IS the Lambda function code. Deploy it with proper naming:
   
   a) Create deployment package:
   ```bash
   cd /path/to/betting
   cp scripts/track_game_line_movements.py lambda_function.py
   zip lambda_function.zip lambda_function.py
   rm lambda_function.py  # cleanup temp file
   ```
   
   b) Deploy to Lambda:
   ```bash
   aws lambda update-function-code \
     --function-name track-line-movement-hourly \
     --zip-file fileb://lambda_function.zip \
     --region us-east-2
   ```
   
   c) Verify handler is set correctly:
   - Lambda Console → Configuration → General configuration
   - Runtime settings → Handler: lambda_function.lambda_handler
   - (This should already be set, but verify if you get import errors)
   
   Note: Lambda expects the file to be named 'lambda_function.py' with a 
   'lambda_handler' function, but this script auto-detects Lambda environment
   and runs main() automatically.

9. EventBridge Schedule (after testing Lambda manually)
   Go to: https://console.aws.amazon.com/events/
   
   a) Create rule:
      - Dashboard -> Rules -> Create rule -> Create scheduled rule
      - Name: 'line-movement-hourly-tracker'
      - Description: 'Triggers line movement tracking every hour'
      - Event bus: default
      - Click "Next"
   
   b) Set schedule:
      - Schedule pattern: Click "A schedule that runs at a regular rate, such as every 10 minutes"
      - Rate expression: rate(1 hour)
      - Or just enter: 1 hour
      - Click "Next"
      
      Alternative (for specific time, like on the hour every hour):
      - Schedule pattern: Click "A fine-grained schedule..."
      - Cron expression: cron(0 * * * ? *)  (runs at :00 every hour)
      - This runs at 1:00, 2:00, 3:00, etc.
   
   c) Select target:
      - Target type: AWS service
      - Select a target: Lambda function
      - Target location: Target in this account
      - Function: track-line-movement-hourly
      - Click "Next" -> "Next"
   
   d) Review and create:
      - Review settings
      - Click "Create rule"
   
   Your Lambda will now run automatically every hour!

10. Testing
    - Test Lambda manually first
    - Check CloudWatch Logs for errors
    - Verify snapshots saved to S3
    - Verify email alerts received
    - Let run hourly for 24h to test full cycle

COST ESTIMATE:
- Lambda: $0 (free tier: 1M requests/month, we use ~720/month)
- S3: ~$0.10/month (storage for snapshots)
- Secrets Manager: $0.40/month
- SNS: $0.00 (free tier: 1000 emails/month)
- EventBridge: $0 (free tier)
- TOTAL: ~$0.50/month

MONITORING:
- CloudWatch Logs: Check for errors daily
- S3 Bucket: Verify snapshots being created
- Email Alerts: Should receive when movements occur
- API Usage: Monitor at https://the-odds-api.com/account/

RELATED FILES:
- scripts/fetch_historical_nfl_season_lines.py (historical fetching)
- scripts/find_nfl_arb_opportunities.py (real-time arb detection)
- implementation/find_nfl_luck_regression_plays_both_teams.py (uses lines)
- src/nfl_luck_utils.py (helper: load_nfl_betting_lines)
- docs/lambda_function_line_movement.py (Lambda handler - to be created)

------------------------------------------------------------------------------------------------

EMAIL ALERTS WITH LINE MOVEMENT VISUALIZATIONS (SES):

Updated to use AWS SES (Simple Email Service) instead of SNS for HTML emails with inline charts.
Charts show historical line movement for each game with significant movements.

CHART FEATURES:
- Plots both away and home team spreads (mirrored lines)
- Inverted y-axis: favorites (negative) at top, underdogs (positive) at bottom
- Dynamic favorite/underdog detection with color-coded zones (green=favorite, red=underdog)
- Major sportsbooks only (DraftKings, FanDuel, BetMGM, Caesars, BetRivers)
- Color-coded lines per sportsbook with distinct markers
- Shows time range and current favorite in subtitle
- Fetches all historical snapshots from S3 (up to 1 week back)

REQUIRED SETUP FOR SES:

1. Verify Email in AWS SES:
   a) Go to AWS SES Console: https://console.aws.amazon.com/ses/
   b) Make sure region is us-east-2 (Ohio) - top right
   c) Click "Identities" in left sidebar (under Configuration section)
   d) Click "Create identity" button (top right)
   e) Select:
      - Identity type: Email address
      - Email address: mylescgthomas@gmail.com
      - Click "Create identity"
   f) Check inbox and click verification link from AWS
   g) Return to SES → Identities to confirm status shows "Verified"
   
   Check verification status via CLI (requires SES permissions on IAM user):
   aws sesv2 list-email-identities --region us-east-2
   
   Or check specific email:
   aws sesv2 get-email-identity --email-identity mylescgthomas@gmail.com --region us-east-2
   
   Note: Easiest to just verify via console (SES → Identities → check for "Verified" status)

2. Add Lambda Environment Variables:
   (In Lambda Console → Configuration → Environment variables → Edit)

   Note: Lambda Function is 'track-line-movement-hourly'
   
   NEW (required for SES):
   - SES_FROM_EMAIL=mylescgthomas@gmail.com
   - SES_TO_EMAIL=mylescgthomas@gmail.com
   
   EXISTING (keep these):
   - ODDS_API_KEY=<from Secrets Manager>
   - S3_BUCKET=betting-line-movement-snapshots
   - MOVEMENT_THRESHOLD_NBA=0.5
   - MOVEMENT_THRESHOLD_NFL=0.5
   - AWS_REGION_NAME=us-east-2
   
   DEPRECATED (can remove after testing):
   - SNS_TOPIC_ARN=<old SNS topic>

3. Add SES Permissions to Lambda IAM Role:
   a) Lambda → Configuration → Permissions → Execution role (click role name)
   b) Add permissions → Attach policies
   c) Search: AmazonSESFullAccess
   d) Select and click "Add permissions"
   
   Or use custom policy (more restrictive):
   {
     "Version": "2012-10-17",
     "Statement": [
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

4. Increase Lambda Timeout (if needed):
   - Chart generation takes ~1-2 seconds per game
   - Recommended: 120 seconds (2 minutes)
   - Update: Lambda → Configuration → General configuration → Edit → Timeout

5. Test Deployment:
   a) Deploy updated code to Lambda
   b) Test manually: Lambda → Test → Invoke
   c) Check CloudWatch Logs for:
      - "📊 Generating charts for X games..."
      - "✅ Email sent via SES: ..."
      - "   Message ID: XXXXXXXX"
   d) Check inbox for email with inline charts

DEPENDENCIES (already in Lambda layer):
- pandas, numpy, requests (custom layer)
- matplotlib (for chart generation)
- boto3 (included in Lambda runtime)
- Standard library: io.BytesIO, base64, datetime, etc.

NO ADDITIONAL PACKAGES NEEDED - all dependencies already available!

SES NOTES:
- If in SES Sandbox mode: can only send to verified emails
- For production: request SES production access in AWS Support
- Email size limit: 10 MB (should handle 10-20 games comfortably)

CHART DELIVERY (Gmail Compatible):
Gmail blocks base64-encoded images for security. Solution: Host charts on S3.

S3 Path Structure:
  s3://betting-line-movement-snapshots/email-charts/{timestamp}/{game_id}.png
  
  Example:
  email-charts/20251226_090200/abc123_LAL_BOS.png
  email-charts/20251226_090200/def456_GB_BAL.png

Process:
1. Generate chart as PNG bytes (matplotlib)
2. Upload to S3 with public-read ACL
3. Get public S3 URL: https://betting-line-movement-snapshots.s3.us-east-2.amazonaws.com/email-charts/{timestamp}/{game_id}.png
4. Embed in email HTML: <img src="https://...">
5. Gmail trusts S3 URLs and displays images

S3 Bucket Policy (Required):
Make email-charts/* publicly accessible so Gmail can display images.

===============================================================================
OPTION 1: Enable ACLs (RECOMMENDED - Simpler, code already handles it)
===============================================================================

Step 1: Go to S3 Console
  https://s3.console.aws.amazon.com/s3/buckets/betting-line-movement-snapshots

Step 2: Click on bucket name "betting-line-movement-snapshots" (if not already there)

Step 3: Click "Permissions" tab at the top

Step 4: Find "Object Ownership" section (should be 2nd section after Block Public Access)
  
Step 5: Click "Edit" button

Step 6: Select "ACLs enabled" radio button

Step 7: Check the acknowledgment box:
  ☑️ "I acknowledge that ACLs will be restored"

Step 8: Click "Save changes"

Done! The Lambda code already uploads with public-read ACL, so charts will be accessible.

===============================================================================
OPTION 2: Bucket Policy (More complex, requires disabling Block Public Access) [I DID THIS ONE]
===============================================================================

Step 1: Go to S3 Console
  https://s3.console.aws.amazon.com/s3/buckets/betting-line-movement-snapshots

Step 2: Click bucket name "betting-line-movement-snapshots"

Step 3: Click "Permissions" tab

Step 4: Find "Block public access (bucket settings)" section (FIRST section at top)

Step 5: Click "Edit" button on that section

Step 6: Uncheck ALL 4 boxes (or at minimum, uncheck the 3rd one):
  ☐ Block public access to buckets and objects granted through new access control lists (ACLs)
  ☐ Block public access to buckets and objects granted through any access control lists (ACLs)
  ☐ Block public access to buckets and objects granted through new public bucket or access point policies
  ☐ Block public access to buckets and objects granted through any public bucket or access point policies

Step 7: Click "Save changes"

Step 8: Type "confirm" in the text box when prompted

Step 9: Click "Confirm"

Step 10: Scroll down to "Bucket policy" section

Step 11: Click "Edit"

Step 12: Paste this complete policy:
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": [
          "arn:aws:s3:::betting-line-movement-snapshots/email-charts/*",
          "arn:aws:s3:::betting-line-movement-snapshots/tmp/*"
        ]
      }
    ]
  }

Note: The tmp/* path allows testing with manually uploaded files before deployment.

Step 13: Click "Save changes"

Done! Charts in email-charts/* and tmp/* will be publicly accessible.

===============================================================================
Verify It Worked:
===============================================================================
After deploying Lambda and running once, check if chart URL is publicly accessible:
1. Find a chart URL in CloudWatch logs (will be printed during upload)
2. Open URL in browser
3. Should see the PNG image without authentication

Test with existing file:
You can test public access settings using the test PNG you uploaded:
https://betting-line-movement-snapshots.s3.us-east-2.amazonaws.com/tmp/tqs-white-background-zoomed-out.png

If you see "Access Denied" → ACLs not enabled or Block Public Access is on
If image displays → Settings are correct! ✅

Example chart URL format after deployment:
https://betting-line-movement-snapshots.s3.us-east-2.amazonaws.com/email-charts/20251226_090200/abc123_Lakers_Celtics.png

Note: I uploaded a .png here for testing: 

Cleanup Strategy:
- Charts only needed for ~24 hours (until next email)
- Can add S3 lifecycle rule: Delete objects in email-charts/ after 7 days
- Or run monthly cleanup script

TROUBLESHOOTING:
- Email not received: Check CloudWatch logs for errors, verify email in SES
- Charts not displaying: Check logs for "Chart generated" messages
- Permission errors: Verify SES policy attached to Lambda role
- Timeout errors: Increase Lambda timeout to 120-180 seconds

AUTHOR: Thomas Myles
CREATED: 2024-12-24
LAST UPDATED: 2024-12-26 (Added SES + chart visualization)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import glob
import os
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import ssl
import urllib3
from typing import Optional, Tuple, Dict, List
import math
import boto3
from io import StringIO, BytesIO
import json
import base64

# Visualization imports (for chart generation)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Lambda
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# API Configuration
ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'
ODDS_API_REGIONS = 'us'
ODDS_API_MARKETS = 'spreads'
ODDS_API_FORMAT = 'american'

# Sports
SPORT_NBA = 'basketball_nba'
SPORT_NFL = 'americanfootball_nfl'
SUPPORTED_SPORTS = [SPORT_NBA, SPORT_NFL]

# Sport display names (for emails/logging)
SPORT_DISPLAY_NAMES = {
    SPORT_NBA: 'NBA',
    SPORT_NFL: 'NFL'
}

# Reverse mapping: Display name -> API sport key
DISPLAY_NAME_TO_SPORT_KEY = {
    'NBA': SPORT_NBA,
    'NFL': SPORT_NFL
}

# Time windows
LOOKBACK_WINDOW_1H = timedelta(hours=1)
LOOKBACK_WINDOW_24H = timedelta(hours=24)
SNAPSHOT_TIME_TOLERANCE = timedelta(minutes=5)  # Allow 5min variance when finding snapshots
GAME_WINDOW_DAYS = 14  # Fetch games within next 14 days

# Display timezone
DISPLAY_TIMEZONE = 'America/New_York'  # Eastern Time for logging

# AWS Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'betting-line-movement-snapshots') # Set in Lambda environment
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN', '')  # Set in Lambda environment (deprecated - using SES now)
SES_FROM_EMAIL = os.getenv('SES_FROM_EMAIL', 'mylescgthomas@gmail.com')  # Verified sender in SES
SES_TO_EMAIL = os.getenv('SES_TO_EMAIL', 'mylescgthomas@gmail.com')  # Verified recipient
IS_LAMBDA = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ  # AWS automatically sets this env var in Lambda; False when running locally

# Initialize boto3 clients
s3_client = boto3.client('s3')
sns_client = boto3.client('sns') if IS_LAMBDA else None  # Kept for backward compat
ses_client = boto3.client('ses', region_name='us-east-2') if IS_LAMBDA else None  # SES for HTML emails (must match verified email region)

# Vig adjustment formula constants
VIG_BASELINE_PRICE = -110  # Standard vig baseline
VIG_THRESHOLD_CENTS = 3.0  # Beyond this, use steeper curve
VIG_LINEAR_FACTOR = 0.015  # Normal range adjustment per dime
VIG_EXTREME_FACTOR = 0.025  # Extreme range adjustment per dime

# Movement detection - sport-specific thresholds
MOVEMENT_THRESHOLD_NBA = float(os.getenv('MOVEMENT_THRESHOLD_NBA', '2.0'))  # NBA moves more
MOVEMENT_THRESHOLD_NFL = float(os.getenv('MOVEMENT_THRESHOLD_NFL', '1.0'))  # NFL more stable
STEAM_MOVE_MIN_BOOKS = 3  # Min books moving same direction to flag steam move

# Map sports to thresholds
MOVEMENT_THRESHOLDS = {
    SPORT_NBA: MOVEMENT_THRESHOLD_NBA,
    SPORT_NFL: MOVEMENT_THRESHOLD_NFL
}

# Timestamp format for filenames
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# =============================================================================
# S3 HELPER FUNCTIONS
# =============================================================================

def get_s3_key(sport: str, filename: str, file_type: str = 'snapshot', is_test: bool = False) -> str:
    """
    Generate S3 key (path) for a file.
    
    Args:
        sport: 'basketball_nba' or 'americanfootball_nfl'
        filename: e.g., 'snapshot_20251224_120000.csv'
        file_type: 'snapshot' or 'movement'
        is_test: If True, save to tmp/ folder for testing
    
    Returns:
        S3 key path
    """
    sport_short = sport.split('_')[1]  # 'nba' or 'nfl'
    
    if is_test:
        # Save to tmp/ for testing
        if file_type == 'snapshot': # Input files
            return f"tmp/snapshots/{sport_short}/{filename}"
        else: # Output files
            return f"tmp/movements/{sport_short}/{filename}"
    else:
        # Production paths
        if file_type == 'snapshot':
            return f"data/01_input/the-odds-api/{sport_short}/line_movement/{filename}"
        else:  # movement
            return f"data/04_output/line_movement/{filename}"


def list_s3_snapshots(sport: str) -> List[str]:
    """
    List all snapshot files in S3 for a given sport.
    
    Returns:
        List of S3 keys
    """
    sport_short = sport.split('_')[1]
    prefix = f"data/01_input/the-odds-api/{sport_short}/line_movement/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
    except Exception as e:
        print(f"Warning: Failed to list S3 snapshots: {e}")
        return []


def save_dataframe_to_s3(df: pd.DataFrame, sport: str, filename: str, file_type: str = 'snapshot', is_test: bool = False) -> str:
    """
    Save DataFrame to S3 as CSV.
    
    Args:
        is_test: If True, save to tmp/ folder for testing
    
    Returns:
        S3 key (path) where file was saved
    """
    s3_key = get_s3_key(sport, filename, file_type, is_test)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
        return s3_key
    except Exception as e:
        print(f"Error: Failed to save to S3: {s3_key}")
        print(f"   {e}")
        raise


def load_dataframe_from_s3(s3_key: str) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from S3 CSV.
    
    Returns:
        DataFrame or None if not found/error
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return df
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Warning: Failed to load from S3: {s3_key}")
        print(f"   {e}")
        return None


def find_snapshot_near_time_s3(sport: str, target_time: datetime) -> Optional[str]:
    """
    Find snapshot S3 key closest to target_time within SNAPSHOT_TIME_TOLERANCE.
    
    Returns:
        S3 key (path) or None if not found
    """
    all_snapshots = list_s3_snapshots(sport)
    
    if not all_snapshots:
        return None
    
    best_key = None
    best_diff = None
    
    for s3_key in all_snapshots:
        # Extract filename from key: data/.../snapshot_20241224_120000.csv
        filename = s3_key.split('/')[-1]
        timestamp_str = filename.replace('snapshot_', '').replace('.csv', '')
        
        try:
            file_time = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
            # Make timezone-aware if target_time is aware
            if target_time.tzinfo:
                file_time = file_time.replace(tzinfo=timezone.utc)
            
            time_diff = abs(file_time - target_time)
            
            if time_diff <= SNAPSHOT_TIME_TOLERANCE:
                if best_diff is None or time_diff < best_diff:
                    best_diff = time_diff
                    best_key = s3_key
        except ValueError:
            # Skip files with invalid timestamp format
            continue
    
    return best_key


# =============================================================================
# VIG ADJUSTMENT FUNCTIONS
# =============================================================================

def calculate_vig_adjustment(price: int) -> float:
    """
    Calculate vig adjustment for a given American odds price.
    
    Uses tiered approach:
    - Normal range (-140 to +120): Linear at 0.015 per dime
    - Extreme range: Steeper at 0.025 per dime after base
    
    Args:
        price: American odds (e.g., -110, -130, +105)
    
    Returns:
        Adjustment in points to add to raw spread
    """
    vig_cents = (price - VIG_BASELINE_PRICE) / 10.0
    
    if vig_cents <= -VIG_THRESHOLD_CENTS:
        # Extreme high vig (worse odds)
        base_adjustment = -VIG_THRESHOLD_CENTS * VIG_LINEAR_FACTOR
        excess = vig_cents - (-VIG_THRESHOLD_CENTS)
        adjustment = base_adjustment + (excess * VIG_EXTREME_FACTOR)
    elif vig_cents >= VIG_THRESHOLD_CENTS:
        # Extreme low vig (better odds)
        base_adjustment = VIG_THRESHOLD_CENTS * VIG_LINEAR_FACTOR
        excess = vig_cents - VIG_THRESHOLD_CENTS
        adjustment = base_adjustment + (excess * VIG_EXTREME_FACTOR)
    else:
        # Normal range
        adjustment = vig_cents * VIG_LINEAR_FACTOR
    
    return adjustment


def calculate_adjusted_spread(raw_spread: float, price: int) -> float:
    """
    Calculate vig-adjusted spread.
    
    Args:
        raw_spread: Raw spread value (e.g., -2.5)
        price: American odds (e.g., -110)
    
    Returns:
        Adjusted spread accounting for vig
    """
    adjustment = calculate_vig_adjustment(price)
    return raw_spread + adjustment


# =============================================================================
# CHART GENERATION FUNCTIONS
# =============================================================================

def upload_chart_to_s3(chart_bytes: bytes, timestamp: str, game_id: str, away_team: str, home_team: str) -> str:
    """
    Upload chart PNG to S3 and return public URL.
    
    Args:
        chart_bytes: PNG image as bytes
        timestamp: Run timestamp (e.g., '20251226_090200')
        game_id: Unique game identifier
        away_team: Away team name
        home_team: Home team name
    
    Returns:
        Public S3 URL for the chart
    """
    # Create S3 key: email-charts/{timestamp}/{game_id}_{away}_{home}.png
    safe_away = away_team.replace(' ', '_')
    safe_home = home_team.replace(' ', '_')
    s3_key = f"email-charts/{timestamp}/{game_id}_{safe_away}_{safe_home}.png"
    
    try:
        # Upload with public-read ACL
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=chart_bytes,
            ContentType='image/png',
            ACL='public-read'
        )
        
        # Return public URL
        url = f"https://{S3_BUCKET}.s3.us-east-2.amazonaws.com/{s3_key}"
        print(f"      ✅ Chart uploaded to S3: {s3_key}")
        return url
    
    except Exception as e:
        print(f"      ❌ Failed to upload chart to S3: {e}")
        return None


def create_line_movement_chart_for_email(df: pd.DataFrame, title: str = None) -> bytes:
    """
    Create line movement chart for email embedding (matplotlib version for Lambda).
    
    Args:
        df: DataFrame with S3 structure (fetched_at, bookmaker, away_spread, home_spread, away_team, home_team)
        title: Chart title (defaults to game matchup)
    
    Returns:
        PNG image as bytes for base64 encoding
    """
    if df.empty:
        return None
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    
    # Get game info
    away_team = df['away_team'].iloc[0]
    home_team = df['home_team'].iloc[0]
    
    # Get time range for subtitle
    first_snapshot = df['timestamp'].min()
    last_snapshot = df['timestamp'].max()
    time_range_hours = (last_snapshot - first_snapshot).total_seconds() / 3600
    
    # Format times for subtitle
    first_time_str = first_snapshot.strftime('%b %d %I:%M %p')
    last_time_str = last_snapshot.strftime('%b %d %I:%M %p ET')
    
    # Determine current favorite
    latest_spread = df['away_spread'].iloc[-1]
    if latest_spread < 0:
        favorite = away_team
        underdog = home_team
    elif latest_spread > 0:
        favorite = home_team
        underdog = away_team
    else:
        favorite = "Pick'em"
        underdog = "Pick'em"
    
    if not title:
        title = f"{away_team} @ {home_team}"
    
    subtitle = f"{time_range_hours:.0f}h movement ({first_time_str} → {last_time_str})"
    if favorite != "Pick'em":
        subtitle += f" | Current Favorite: {favorite}"
    
    # Focus on major books
    major_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers']
    df_major = df[df['bookmaker'].isin(major_books)].copy()
    
    # Create figure
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color map for bookmakers
    book_colors = {
        'draftkings': '#53D337',
        'fanduel': '#0E8FEF',
        'betmgm': '#BA9000',
        'caesars': '#0033A0',
        'betrivers': '#00A4E4'
    }
    
    # Plot each bookmaker's line - both teams (mirror)
    for bookmaker in major_books:
        book_df = df_major[df_major['bookmaker'] == bookmaker].copy()
        if book_df.empty:
            continue
        
        color = book_colors.get(bookmaker, '#333333')
        
        # Plot away team spread (solid line)
        ax.plot(
            book_df['timestamp'],
            book_df['away_spread'],
            label=f"{bookmaker.upper()} ({away_team})",
            color=color,
            marker='o',
            markersize=5,
            linewidth=3,
            alpha=0.9,
            linestyle='-'
        )
        
        # Plot home team spread (dashed line - mirror)
        ax.plot(
            book_df['timestamp'],
            book_df['home_spread'],
            label=f"{bookmaker.upper()} ({home_team})",
            color=color,
            marker='s',
            markersize=5,
            linewidth=3,
            alpha=0.9,
            linestyle='--'
        )
    
    # Add horizontal line at 0 (pick'em)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6, zorder=1)
    
    # Add shaded regions (green for favorites at top after inversion, red for underdogs at bottom)
    y_min, y_max = ax.get_ylim()
    ax.axhspan(0, y_max, alpha=0.05, color='red', zorder=0)    # Underdog zone (positive values)
    ax.axhspan(y_min, 0, alpha=0.05, color='green', zorder=0)  # Favorite zone (negative values)
    
    # INVERT Y-AXIS: Favorites (negative) at top after inversion
    ax.invert_yaxis()
    
    # Formatting
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spread (points)', fontsize=13, fontweight='bold')
    
    # Title and subtitle
    ax.set_title(title, fontsize=16, fontweight='bold', pad=35)
    ax.text(0.5, 1.03, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=11, style='italic', color='#555')
    
    # Add favorite/underdog zone labels (dynamically determined)
    if favorite != "Pick'em":
        ax.text(0.02, 0.98, f'← {favorite.upper()} - FAVORITES', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.02, f'← {underdog.upper()} - UNDERDOGS', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=10, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend - show all books with both teams
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8, ncol=2, 
              title='Sportsbooks', title_fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Convert to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)
    
    return img_bytes


def fetch_all_snapshots_for_game(sport: str, game_id: str, time_window_hours: int = 168) -> pd.DataFrame:
    """
    Fetch all snapshots for a specific game from S3 within a time window.
    
    Args:
        sport: 'basketball_nba' or 'americanfootball_nfl'
        game_id: Game ID from The Odds API
        time_window_hours: How far back to look (default 1 week)
    
    Returns:
        DataFrame with all snapshots for this game
    """
    all_snapshots = list_s3_snapshots(sport)
    
    if not all_snapshots:
        return pd.DataFrame()
    
    # Calculate time cutoff
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
    
    # Collect data from all relevant snapshots
    game_data = []
    
    for s3_key in all_snapshots:
        # Extract timestamp from filename
        filename = s3_key.split('/')[-1]
        timestamp_str = filename.replace('snapshot_', '').replace('.csv', '')
        
        try:
            file_time = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
            file_time = file_time.replace(tzinfo=timezone.utc)
            
            # Skip if outside time window
            if file_time < cutoff_time:
                continue
            
            # Load snapshot
            df = load_dataframe_from_s3(s3_key)
            if df is None or df.empty:
                continue
            
            # Filter to just this game
            game_df = df[df['game_id'] == game_id]
            if not game_df.empty:
                game_data.append(game_df)
                
        except ValueError:
            continue
    
    if not game_data:
        return pd.DataFrame()
    
    # Combine all snapshots
    combined_df = pd.concat(game_data, ignore_index=True)
    combined_df = combined_df.sort_values('fetched_at')
    
    return combined_df


# =============================================================================
# EMAIL FORMATTING FUNCTIONS
# =============================================================================

def format_movement_email_html(sport_summaries: Dict, all_movements: Dict[str, pd.DataFrame], 
                               sport_thresholds: Dict[str, float], current_time: datetime,
                               sport_to_api_key: Dict[str, str], 
                               current_snapshots: Dict[str, pd.DataFrame] = None) -> str:
    """
    Format movements into HTML email with inline charts.
    
    Args:
        sport_summaries: Dict with summary stats per sport
        all_movements: Dict mapping sport name to movements DataFrame
        sport_thresholds: Dict mapping sport name to threshold  
        current_time: Current timestamp
        sport_to_api_key: Dict mapping sport display name to API sport key (e.g., 'NBA' -> 'basketball_nba')
        current_snapshots: Dict mapping sport name to current snapshot DataFrame
    
    Returns:
        HTML string for email body with embedded charts
    """
    time_et = current_time.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    time_str = time_et.strftime('%b %d, %Y %I:%M %p ET')
    
    # Collect unique game IDs that have movements
    games_with_movements = {}
    for sport_name, df in all_movements.items():
        if df is not None and not df.empty:
            game_ids = df['game_id'].unique()
            for game_id in game_ids:
                if game_id not in games_with_movements:
                    games_with_movements[game_id] = {
                        'sport_name': sport_name,
                        'sport_key': sport_to_api_key.get(sport_name, ''),
                        'away_team': df[df['game_id'] == game_id]['away_team'].iloc[0],
                        'home_team': df[df['game_id'] == game_id]['home_team'].iloc[0]
                    }
    
    # Generate charts for games with movements
    print(f"\n📊 Generating charts for {len(games_with_movements)} games...")
    run_timestamp = current_time.strftime(TIMESTAMP_FORMAT)  # e.g., '20251226_090200'
    game_charts = {}  # Will store game_id -> S3 URL
    
    for game_id, game_info in games_with_movements.items():
        print(f"   Fetching snapshots for {game_info['away_team']} @ {game_info['home_team']}...")
        
        # Fetch all snapshots for this game
        snapshots_df = fetch_all_snapshots_for_game(game_info['sport_key'], game_id, time_window_hours=168)
        
        if not snapshots_df.empty:
            print(f"      Found {len(snapshots_df)} snapshot rows, generating chart...")
            chart_bytes = create_line_movement_chart_for_email(
                snapshots_df,
                title=f"{game_info['away_team']} @ {game_info['home_team']}"
            )
            if chart_bytes:
                # Upload to S3 and get public URL
                chart_url = upload_chart_to_s3(
                    chart_bytes,
                    run_timestamp,
                    game_id,
                    game_info['away_team'],
                    game_info['home_team']
                )
                if chart_url:
                    game_charts[game_id] = chart_url
        else:
            print(f"      ⚠️  No snapshots found for this game")
    
    # Build HTML email
    html_parts = []
    html_parts.append(f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #0066cc;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .subtitle {{
                margin-top: 5px;
                opacity: 0.9;
            }}
            .summary {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #0066cc;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section h2 {{
                color: #0066cc;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .game {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .game h3 {{
                margin-top: 0;
                color: #333;
            }}
            .chart {{
                margin: 15px 0;
                text-align: center;
            }}
            .chart img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .movement-details {{
                font-family: 'Courier New', monospace;
                font-size: 13px;
                background-color: #fff;
                padding: 10px;
                border-radius: 3px;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚨 LINE MOVEMENT ALERT</h1>
            <div class="subtitle">Time: {time_str}</div>
            <div class="subtitle">Thresholds: {', '.join([f"{sport}: ≥{thresh}pts" for sport, thresh in sport_thresholds.items()])}</div>
        </div>
    """)
    
    # Summary section
    html_parts.append('<div class="summary"><h2>Summary</h2>')
    for sport_name, summary in sport_summaries.items():
        df = all_movements.get(sport_name, pd.DataFrame())
        unique_games_with_moves = df['game_id'].nunique() if not df.empty else 0
        unique_games_crossed_zero = df[df['crossed_zero_1h'] | df['crossed_zero_24h']]['game_id'].nunique() if not df.empty else 0
        html_parts.append(f"<p><strong>{sport_name}:</strong> {summary['num_games']} games tracked | {unique_games_with_moves} games with moves | {unique_games_crossed_zero} games crossed zero</p>")
    html_parts.append('</div>')
    
    # Process movements and add charts
    for sport_name in all_movements.keys():
        df = all_movements[sport_name]
        if df is None or df.empty:
            continue
        
        # Group by game
        for game_id in df['game_id'].unique():
            game_df = df[df['game_id'] == game_id]
            first_row = game_df.iloc[0]
            
            html_parts.append(f"""
            <div class="section">
                <h2>{first_row['away_team']} @ {first_row['home_team']}</h2>
            """)
            
            # Add chart if available
            if game_id in game_charts:
                html_parts.append(f"""
                <div class="chart">
                    <img src="{game_charts[game_id]}" alt="Line Movement Chart" style="max-width: 100%; height: auto;">
                </div>
                """)
            
            # Add movement details as text
            movement_text = format_movements_text_for_game(game_df)
            html_parts.append(f"""
                <div class="movement-details">{movement_text}</div>
            </div>
            """)
    
    # Current games tracked section
    if current_snapshots:
        html_parts.append("""
        <div class="section">
            <h2>📊 CURRENT GAMES TRACKED</h2>
        """)
        
        for sport_name in current_snapshots.keys():
            df = current_snapshots[sport_name]
            if df is None or df.empty:
                continue
            
            num_games = df['game_id'].nunique()
            html_parts.append(f"""
            <div class="game">
                <h3>{sport_name} - {num_games} games</h3>
            """)
            
            # Group by game
            for game_id in df['game_id'].unique():
                game_df = df[df['game_id'] == game_id]
                first_row = game_df.iloc[0]
                
                html_parts.append(f"""
                <div style="margin: 15px 0; padding: 10px; background-color: white; border-radius: 3px;">
                    <strong>{first_row['away_team']} @ {first_row['home_team']}</strong><br>
                    Game Time: {first_row['game_time_et']}<br>
                    Books tracking: {len(game_df)}
                    <ul style="margin: 10px 0; padding-left: 20px;">
                """)
                
                # Show top 3 bookmaker lines
                for idx, (_, row) in enumerate(game_df.head(3).iterrows()):
                    html_parts.append(f"""
                        <li>{row['bookmaker']}: Away {row['away_spread']}/{row['away_price']} | Home {row['home_spread']}/{row['home_price']}</li>
                    """)
                
                if len(game_df) > 3:
                    html_parts.append(f"""
                        <li><em>... and {len(game_df) - 3} more books</em></li>
                    """)
                
                html_parts.append("""
                    </ul>
                </div>
                """)
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    
    return ''.join(html_parts)


def format_movements_text_for_game(df: pd.DataFrame) -> str:
    """Format movements for a single game as text."""
    lines = []
    
    # Group by bookmaker
    for bookmaker in df['bookmaker'].unique():
        book_df = df[df['bookmaker'] == bookmaker]
        for _, row in book_df.iterrows():
            side = row['side']
            
            # Format with proper column names
            spread_24h = row['prev_24h_raw_spread'] if pd.notna(row.get('prev_24h_raw_spread')) else '—'
            price_24h = row['prev_24h_price'] if pd.notna(row.get('prev_24h_price')) else '—'
            spread_1h = row['prev_1h_raw_spread'] if pd.notna(row.get('prev_1h_raw_spread')) else '—'
            price_1h = row['prev_1h_price'] if pd.notna(row.get('prev_1h_price')) else '—'
            spread_now = row['current_raw_spread']
            price_now = row['current_price']
            
            lines.append(f"\nBook: {bookmaker}")
            lines.append(f"24h ago: {side} {spread_24h}/{price_24h}")
            lines.append(f"1h ago:  {side} {spread_1h}/{price_1h}")
            lines.append(f"Now:     {side} {spread_now}/{price_now}")
    
    return '\n'.join(lines)


def format_movement_email(sport_summaries: Dict, all_movements: Dict[str, pd.DataFrame], 
                         sport_thresholds: Dict[str, float], current_time: datetime, 
                         current_snapshots: Dict[str, pd.DataFrame] = None) -> str:
    """
    Format movements into plain text email (SNS doesn't support HTML).
    
    Args:
        sport_summaries: Dict with summary stats per sport
        all_movements: Dict mapping sport name to movements DataFrame
        sport_thresholds: Dict mapping sport name to threshold
        current_time: Current timestamp
        current_snapshots: Dict mapping sport name to current snapshot DataFrame
    
    Returns:
        Plain text string for email body
    """
    time_et = current_time.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    time_str = time_et.strftime('%b %d, %Y %I:%M %p ET')
    
    lines = []
    lines.append("=" * 80)
    lines.append("🚨 LINE MOVEMENT ALERT")
    lines.append("=" * 80)
    lines.append(f"Time: {time_str}")
    # Show thresholds per sport
    threshold_strs = [f"{sport}: ≥{thresh}pts" for sport, thresh in sport_thresholds.items()]
    lines.append(f"Thresholds: {', '.join(threshold_strs)}")
    lines.append("")
    
    # Summary section
    lines.append("SUMMARY:")
    lines.append("-" * 80)
    for sport_name, summary in sport_summaries.items():
        df = all_movements.get(sport_name, pd.DataFrame())
        unique_games_with_moves = df['game_id'].nunique() if not df.empty else 0
        unique_games_crossed_zero = df[df['crossed_zero_1h'] | df['crossed_zero_24h']]['game_id'].nunique() if not df.empty else 0
        
        lines.append(f"{sport_name}: {summary['num_games']} games tracked | {unique_games_with_moves} games with moves | {unique_games_crossed_zero} games crossed zero")
    lines.append("")
    
    # Significant movements section - SPLIT BY TIMEFRAME
    # Priority: 1h movements at TOP (most urgent), 24h movements at BOTTOM (context)
    has_any_significant = False
    
    # =========================================================================
    # SECTION 1: 1-HOUR MOVEMENTS (URGENT - AT TOP)
    # =========================================================================
    
    # First: Crossed Zero in Last Hour
    has_crossed_zero_1h = False
    for sport_name in all_movements.keys():
        df = all_movements[sport_name]
        if df is None or df.empty:
            continue
        
        crossed_zero_1h = df[df['crossed_zero_1h'] == True]
        
        if not crossed_zero_1h.empty:
            unique_games = crossed_zero_1h['game_id'].nunique()
            
            if not has_crossed_zero_1h:
                lines.append("=" * 80)
                lines.append("🚨 CROSSED ZERO (Last Hour) - Favorite/Underdog Flip")
                lines.append("=" * 80)
                has_crossed_zero_1h = True
                has_any_significant = True
            
            lines.append(f"\n{sport_name} ({unique_games} {'game' if unique_games == 1 else 'games'}):")
            lines.append("-" * 80)
            lines.extend(format_movements_text(crossed_zero_1h))
    
    # Second: Large moves in Last Hour
    has_large_moves_1h = False
    for sport_name in all_movements.keys():
        df = all_movements[sport_name]
        if df is None or df.empty:
            continue
        
        # Large 1h moves that didn't cross zero in 1h
        large_moves_1h = df[
            (df['significant_hourly'] == True) &
            (df['crossed_zero_1h'] == False)
        ]
        
        if not large_moves_1h.empty:
            unique_games = large_moves_1h['game_id'].nunique()
            
            if not has_large_moves_1h:
                if has_crossed_zero_1h:
                    lines.append("")
                lines.append("=" * 80)
                lines.append(f"📊 LARGE MOVES (Last Hour) - NBA ≥{sport_thresholds.get('NBA', 2.0)}pts, NFL ≥{sport_thresholds.get('NFL', 1.0)}pts")
                lines.append("=" * 80)
                has_large_moves_1h = True
                has_any_significant = True
            
            lines.append(f"\n{sport_name} ({unique_games} {'game' if unique_games == 1 else 'games'}):")
            lines.append("-" * 80)
            lines.extend(format_movements_text(large_moves_1h))
    
    # =========================================================================
    # SECTION 2: 24-HOUR MOVEMENTS (CONTEXT - AT BOTTOM)
    # =========================================================================
    
    # Third: Crossed Zero in 24h (but NOT in last hour)
    has_crossed_zero_24h = False
    for sport_name in all_movements.keys():
        df = all_movements[sport_name]
        if df is None or df.empty:
            continue
        
        # Crossed zero in 24h window but NOT in the last hour
        crossed_zero_24h_only = df[
            (df['crossed_zero_24h'] == True) &
            (df['crossed_zero_1h'] == False)
        ]
        
        if not crossed_zero_24h_only.empty:
            unique_games = crossed_zero_24h_only['game_id'].nunique()
            
            if not has_crossed_zero_24h:
                if has_crossed_zero_1h or has_large_moves_1h:
                    lines.append("")
                lines.append("=" * 80)
                lines.append("⏰ CROSSED ZERO (24h Window) - Longer-Term Trend")
                lines.append("=" * 80)
                has_crossed_zero_24h = True
                has_any_significant = True
            
            lines.append(f"\n{sport_name} ({unique_games} {'game' if unique_games == 1 else 'games'}):")
            lines.append("-" * 80)
            lines.extend(format_movements_text(crossed_zero_24h_only))
    
    # Fourth: Large moves in 24h (but NOT in last hour and didn't cross zero)
    has_large_moves_24h = False
    for sport_name in all_movements.keys():
        df = all_movements[sport_name]
        if df is None or df.empty:
            continue
        
        # Large 24h moves that:
        # - Are significant in 24h window
        # - Did NOT cross zero in 1h or 24h
        # - Are NOT already flagged as significant in 1h
        large_moves_24h_only = df[
            (df['significant_daily'] == True) &
            (df['significant_hourly'] == False) &
            (df['crossed_zero_1h'] == False) &
            (df['crossed_zero_24h'] == False)
        ]
        
        if not large_moves_24h_only.empty:
            unique_games = large_moves_24h_only['game_id'].nunique()
            
            if not has_large_moves_24h:
                if has_crossed_zero_1h or has_large_moves_1h or has_crossed_zero_24h:
                    lines.append("")
                lines.append("=" * 80)
                lines.append(f"📈 LARGE MOVES (24h Window) - NBA ≥{sport_thresholds.get('NBA', 2.0)}pts, NFL ≥{sport_thresholds.get('NFL', 1.0)}pts")
                lines.append("=" * 80)
                has_large_moves_24h = True
                has_any_significant = True
            
            lines.append(f"\n{sport_name} ({unique_games} {'game' if unique_games == 1 else 'games'}):")
            lines.append("-" * 80)
            lines.extend(format_movements_text(large_moves_24h_only))
    
    if not has_any_significant:
        lines.append("=" * 80)
        lines.append("✅ NO SIGNIFICANT MOVEMENTS DETECTED")
        lines.append("=" * 80)
    
    # All other movements section
    lines.append("")
    lines.append("=" * 80)
    lines.append("📋 ALL OTHER MOVEMENTS (Below threshold)")
    lines.append("=" * 80)
    
    has_other = False
    for sport_name in all_movements.keys():
        df = all_movements[sport_name]
        if df is None or df.empty:
            continue
        
        df = all_movements[sport_name]
        if df.empty:
            continue
        
        other = df[
            (df['significant_hourly'] == False) & 
            (df['significant_daily'] == False) &
            (df['crossed_zero_1h'] == False) &
            (df['crossed_zero_24h'] == False)
        ]
        
        if not other.empty:
            # Count unique games
            unique_games = other['game_id'].nunique()
            
            has_other = True
            lines.append(f"\n{sport_name} ({unique_games} {'game' if unique_games == 1 else 'games'}):")
            lines.append("-" * 80)
            lines.extend(format_movements_text(other))
    
    if not has_other:
        lines.append("\nNo other movements detected.")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Current games tracked section (MOVED TO BOTTOM)
    if current_snapshots:
        lines.append("📊 CURRENT GAMES TRACKED")
        lines.append("=" * 80)
        
        for sport_name in current_snapshots.keys():
            df = current_snapshots[sport_name]
            if df is None or df.empty:
                continue
            
            df = current_snapshots[sport_name]
            if df.empty:
                continue
            
            lines.append(f"\n{sport_name} - {df['game_id'].nunique()} games:")
            lines.append("-" * 80)
            
            # Group by game
            for game_id in df['game_id'].unique():
                game_df = df[df['game_id'] == game_id]
                first_row = game_df.iloc[0]
                
                lines.append(f"\n  {first_row['away_team']} @ {first_row['home_team']}")
                lines.append(f"  Game Time: {first_row['game_time_et']}")
                lines.append(f"  Books tracking: {len(game_df)}")
                
                # Show a few bookmaker lines (top 3)
                for idx, (_, row) in enumerate(game_df.head(3).iterrows()):
                    lines.append(f"    • {row['bookmaker']}: Away {row['away_spread']}/{row['away_price']} | Home {row['home_spread']}/{row['home_price']}")
                
                if len(game_df) > 3:
                    lines.append(f"    ... and {len(game_df) - 3} more books")
                lines.append("")
        
        lines.append("")
        lines.append("=" * 80)
    
    return "\n".join(lines)


def format_movements_text(df: pd.DataFrame) -> List[str]:
    """Format movements DataFrame as plain text lines, grouped by game."""
    lines = []
    
    # Group by game (we'll get both sides and merge them)
    game_groups = df.groupby(['game_id', 'away_team', 'home_team', 'bookmaker'])
    
    # Reorganize to group by game first, then bookmaker
    game_bookmaker_data = {}
    for (game_id, away_team, home_team, bookmaker), group in game_groups:
        game_key = (game_id, away_team, home_team)
        if game_key not in game_bookmaker_data:
            game_bookmaker_data[game_key] = {}
        
        # Store away and home rows separately
        for _, row in group.iterrows():
            if row['side'] == 'away':
                if bookmaker not in game_bookmaker_data[game_key]:
                    game_bookmaker_data[game_key][bookmaker] = {}
                game_bookmaker_data[game_key][bookmaker]['away'] = row
            elif row['side'] == 'home':
                if bookmaker not in game_bookmaker_data[game_key]:
                    game_bookmaker_data[game_key][bookmaker] = {}
                game_bookmaker_data[game_key][bookmaker]['home'] = row
    
    # Now format output
    for (game_id, away_team, home_team), bookmakers in game_bookmaker_data.items():
        # Check if any movement in this game crossed zero
        has_crossed_zero = False
        for bookmaker_data in bookmakers.values():
            for side_data in bookmaker_data.values():
                if side_data.get('crossed_zero_1h', False) or side_data.get('crossed_zero_24h', False):
                    has_crossed_zero = True
                    break
        
        crossed_flag = " 🚨" if has_crossed_zero else ""
        
        # Game header
        lines.append(f"  {away_team} @ {home_team}{crossed_flag}")
        lines.append("")
        
        # List all bookmakers for this game
        for bookmaker, side_data in bookmakers.items():
            away_row = side_data.get('away')
            home_row = side_data.get('home')
            
            # Build formatted strings for away/home at each time point
            # 24h ago
            away_24h = f"{away_row['prev_24h_raw_spread']}/{away_row['prev_24h_price']}" if away_row is not None and pd.notna(away_row.get('prev_24h_raw_spread')) else "—"
            home_24h = f"{home_row['prev_24h_raw_spread']}/{home_row['prev_24h_price']}" if home_row is not None and pd.notna(home_row.get('prev_24h_raw_spread')) else "—"
            
            # 1h ago
            away_1h = f"{away_row['prev_1h_raw_spread']}/{away_row['prev_1h_price']}" if away_row is not None and pd.notna(away_row.get('prev_1h_raw_spread')) else "—"
            home_1h = f"{home_row['prev_1h_raw_spread']}/{home_row['prev_1h_price']}" if home_row is not None and pd.notna(home_row.get('prev_1h_raw_spread')) else "—"
            
            # Now
            away_now = f"{away_row['current_raw_spread']}/{away_row['current_price']}" if away_row is not None else "—"
            home_now = f"{home_row['current_raw_spread']}/{home_row['current_price']}" if home_row is not None else "—"
            
            lines.append(f"    Book: {bookmaker}")
            lines.append(f"    24h ago: Away {away_24h} | Home {home_24h}")
            lines.append(f"    1h ago:  Away {away_1h} | Home {home_1h}")
            lines.append(f"    Now:     Away {away_now} | Home {home_now}")
            lines.append("")
        
        # Extra spacing between games
        lines.append("")
    
    return lines


def send_email_via_sns(subject: str, body: str):
    """Send plain text email via AWS SNS."""
    if not SNS_TOPIC_ARN:
        print("Warning: SNS_TOPIC_ARN not set, skipping email")
        return
    
    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=body
        )
        print(f"✅ Email sent via SNS: {subject}")
    except Exception as e:
        print(f"Error: Failed to send email via SNS: {e}")


def send_email_via_ses(subject: str, html_body: str, text_body: str):
    """
    Send HTML email with inline images via AWS SES.
    
    Args:
        subject: Email subject
        html_body: HTML content with base64-encoded images
        text_body: Plain text fallback
    """
    if not ses_client:
        print("Warning: SES client not initialized, skipping email")
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
# HELPER FUNCTIONS
# =============================================================================

def generate_snapshot_filename(timestamp: datetime) -> str:
    """Generate snapshot filename from timestamp."""
    timestamp_str = timestamp.strftime(TIMESTAMP_FORMAT)
    return f"snapshot_{timestamp_str}.csv"


# =============================================================================
# MOVEMENT DETECTION FUNCTIONS
# =============================================================================

def detect_crossed_zero(prev_spread: float, current_spread: float) -> bool:
    """
    Detect if spread crossed zero (favorite/underdog flip).
    
    Only counts as crossing if line goes from positive to negative or vice versa.
    Does NOT count if starting from or ending at exactly 0.0 (pick'em).
    
    Args:
        prev_spread: Previous spread value
        current_spread: Current spread value
    
    Returns:
        True if crossed zero (changed from pos to neg or neg to pos)
    """
    # No movement = no crossing
    if prev_spread == current_spread:
        return False
    
    # Must go from one side to the other (not from/to zero)
    # Examples:
    #   1.5 → -1.5: True ✓ (underdog to favorite)
    #   -2.0 → 1.0: True ✓ (favorite to underdog)
    #   1.5 → 0.0: False ✗ (underdog to pick'em, not a flip)
    #   0.0 → -1.5: False ✗ (pick'em to favorite, not a flip)
    #   0.0 → 0.5: False ✗ (pick'em to underdog, not a flip)
    #   1.5 → 2.0: False ✗ (still underdog)
    return (prev_spread > 0 and current_spread < 0) or (prev_spread < 0 and current_spread > 0)


def calculate_movement_type(prev_raw: float, prev_price: int, 
                           current_raw: float, current_price: int) -> str:
    """
    Determine if movement is line-only, vig-only, or both.
    
    Returns:
        'line_only', 'vig_only', 'both', or 'none'
    """
    line_moved = prev_raw != current_raw
    vig_moved = prev_price != current_price
    
    if line_moved and vig_moved:
        return 'both'
    elif line_moved:
        return 'line_only'
    elif vig_moved:
        return 'vig_only'
    else:
        return 'none'


def determine_movement_direction(prev_adjusted: float, current_adjusted: float) -> str:
    """
    Determine if movement is toward favorite or underdog.
    
    Returns:
        'toward_favorite', 'toward_underdog', or 'none'
    """
    movement = current_adjusted - prev_adjusted
    
    if abs(movement) < 0.01:  # Essentially no movement
        return 'none'
    
    # Negative spread = favorite, positive = underdog
    # Movement toward more negative = toward favorite
    if movement < 0:
        return 'toward_favorite'
    else:
        return 'toward_underdog'


# =============================================================================
# API FUNCTIONS
# =============================================================================

def fetch_odds_from_api(sport: str, api_key: str) -> dict:
    """
    Fetch current odds from The Odds API.
    
    Args:
        sport: Sport key (e.g., 'basketball_nba')
        api_key: The Odds API key
    
    Returns:
        API response as dict
    """
    url = f"{ODDS_API_BASE_URL}/sports/{sport}/odds/"
    
    now = datetime.now(timezone.utc)
    commence_from = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    commence_to = (now + timedelta(days=GAME_WINDOW_DAYS)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    params = {
        'apiKey': api_key,
        'regions': ODDS_API_REGIONS,
        'markets': ODDS_API_MARKETS,
        'oddsFormat': ODDS_API_FORMAT,
        'commenceTimeFrom': commence_from,
        'commenceTimeTo': commence_to,
    }
    
    print(f"\n📡 Fetching {sport} spreads...")
    print(f"   Time window: {commence_from} to {commence_to}")
    
    response = requests.get(url, params=params, verify=False)
    
    if response.status_code != 200:
        print(f"   ❌ API Error: {response.status_code}")
        print(f"   {response.text}")
        return None
    
    # Check rate limit headers
    if 'x-requests-remaining' in response.headers:
        remaining = response.headers['x-requests-remaining']
        print(f"   API requests remaining: {remaining}")
    
    return response.json()


def parse_api_response_to_dataframe(api_data: list, fetched_at: str) -> pd.DataFrame:
    """
    Parse API response into flat DataFrame with one row per game/bookmaker.
    
    Args:
        api_data: List of games from API
        fetched_at: ISO timestamp when data was fetched
    
    Returns:
        DataFrame with columns matching snapshot schema
    """
    rows = []
    
    for game in api_data:
        game_id = game['id']
        game_time = game['commence_time']
        away_team = game['away_team']
        home_team = game['home_team']
        
        # Debug: Log first game to verify home/away assignment
        if len(rows) == 0:
            print(f"\n   📋 Example game from API:")
            print(f"      Away: {away_team}")
            print(f"      Home: {home_team}")
        
        for bookmaker in game.get('bookmakers', []):
            bookmaker_key = bookmaker['key']
            last_bookmaker_update = bookmaker.get('last_update') # API uses last_update, I prefer last_bookmaker_update
            
            for market in bookmaker.get('markets', []):
                if market['key'] == 'spreads':
                    # Find away and home outcomes
                    away_outcome = None
                    home_outcome = None
                    
                    for outcome in market['outcomes']:
                        if outcome['name'] == away_team:
                            away_outcome = outcome
                        elif outcome['name'] == home_team:
                            home_outcome = outcome
                    
                    if away_outcome and home_outcome:
                        away_spread = away_outcome['point']
                        away_price = away_outcome['price']
                        home_spread = home_outcome['point']
                        home_price = home_outcome['price']
                        
                        # Calculate adjusted spreads
                        away_adjusted = calculate_adjusted_spread(away_spread, away_price)
                        home_adjusted = calculate_adjusted_spread(home_spread, home_price)
                        
                        # Convert timestamps to ET for display
                        game_time_et = pd.to_datetime(game_time).tz_convert(DISPLAY_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')
                        fetched_at_et = pd.to_datetime(fetched_at).tz_convert(DISPLAY_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')
                        last_bookmaker_update_et = pd.to_datetime(last_bookmaker_update).tz_convert(DISPLAY_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET') if last_bookmaker_update else None
                        
                        rows.append({
                            'game_id': game_id,
                            'game_time': game_time,
                            'away_team': away_team,
                            'home_team': home_team,
                            'bookmaker': bookmaker_key,
                            'away_spread': away_spread,
                            'away_price': away_price,
                            'away_adjusted_spread': away_adjusted,
                            'home_spread': home_spread,
                            'home_price': home_price,
                            'home_adjusted_spread': home_adjusted,
                            'fetched_at': fetched_at,
                            'last_bookmaker_update': last_bookmaker_update,
                            'game_time_et': game_time_et,
                            'fetched_at_et': fetched_at_et,
                            'last_bookmaker_update_et': last_bookmaker_update_et,
                        })
    
    return pd.DataFrame(rows)


# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Track betting line movement for NBA/NFL spreads'
    )
    parser.add_argument('--sport', type=str, choices=['nba', 'nfl', 'both'],
                       default='both', help='Sport to track (default: both)')
    parser.add_argument('--prod-run', action='store_true',
                       help='Production mode (no prompts)')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report from existing snapshots (no new fetch)')
    parser.add_argument('--check-api-usage', action='store_true',
                       help='Check what API would return and save to S3 tmp/ folder (uses 1 API call)')
    # Deprecated: kept for backwards compatibility but not used
    parser.add_argument('--movement-threshold', type=float, 
                       default=None,
                       help='(Deprecated) Use MOVEMENT_THRESHOLD_NBA and MOVEMENT_THRESHOLD_NFL env vars')
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key and not args.report_only:
        print("\n❌ ERROR: ODDS_API_KEY environment variable not set")
        print("   Set it with: export ODDS_API_KEY='your_key_here'")
        sys.exit(1)
    
    # Determine which sports to process
    if args.sport == 'both':
        sports = [SPORT_NBA, SPORT_NFL]
    elif args.sport == 'nba':
        sports = [SPORT_NBA]
    elif args.sport == 'nfl':
        sports = [SPORT_NFL]
    
    current_time = datetime.now(timezone.utc)
    fetched_at = current_time.isoformat()
    
    # Track what we save and process
    saved_snapshots = []
    saved_movements = []
    sport_summaries = {}  # Track summary info per sport
    current_snapshots = {}  # Track current snapshot DataFrames for email
    
    print(f"\n{'='*80}")
    print(f"Track Line Movement - {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*80}")
    
    for sport in sports:
        print(f"\n{'─'*80}")
        print(f"Processing {sport}")
        print(f"{'─'*80}")
        
        # Fetch current data from API
        if args.report_only:
            print("\n⚠️  REPORT ONLY: Skipping API call")
            continue
        
        api_data = fetch_odds_from_api(sport, api_key)
        
        if not api_data:
            print(f"   ❌ No data returned for {sport}")
            continue
        
        # Parse to DataFrame
        current_df = parse_api_response_to_dataframe(api_data, fetched_at)
        
        if current_df.empty:
            print(f"   ⚠️  No games found for {sport}")
            continue
        
        print(f"   ✅ Fetched {len(current_df)} bookmaker-game combinations")
        print(f"      Games: {current_df['game_id'].nunique()}")
        print(f"      Bookmakers: {current_df['bookmaker'].nunique()}")
        
        # Track summary info
        num_games = current_df['game_id'].nunique()
        # Calculate unique days in display timezone (CT) to match what user sees
        game_times_ct = pd.to_datetime(current_df['game_time']).dt.tz_convert(DISPLAY_TIMEZONE)
        unique_dates = game_times_ct.dt.date.nunique()
        sport_name = SPORT_DISPLAY_NAMES[sport]
        sport_summaries[sport_name] = {
            'num_games': num_games,
            'unique_days': unique_dates,
            'noteworthy_plays': 0  # Will update after movement detection
        }
        
        # Store current snapshot for email
        current_snapshots[sport_name] = current_df
        
        # Show game schedule organized by date
        print(f"\n   📅 Games Schedule:")
        unique_games = current_df[['game_id', 'game_time', 'away_team', 'home_team']].drop_duplicates()
        unique_games['game_time_local'] = pd.to_datetime(unique_games['game_time']).dt.tz_convert(DISPLAY_TIMEZONE)
        unique_games = unique_games.sort_values('game_time')
        
        # Group by date
        unique_games['game_date'] = unique_games['game_time_local'].dt.date
        
        # Get date range from first game to last game (or up to 14 days if fewer games)
        first_game_date = unique_games['game_date'].min()
        last_game_date = unique_games['game_date'].max()
        
        current_date = first_game_date
        while current_date <= last_game_date:
            # Convert date to datetime for formatting
            date_dt = pd.to_datetime(current_date)
            date_header = date_dt.strftime('%a, %b %d')
            
            # Get games for this date
            games_on_date = unique_games[unique_games['game_date'] == current_date]
            
            if len(games_on_date) > 0:
                print(f"\n      {date_header}:")
                for idx, game in games_on_date.iterrows():
                    game_time_str = game['game_time_local'].strftime('%I:%M %p ET')
                    print(f"         {game['away_team']} @ {game['home_team']} - {game_time_str}")
            else:
                print(f"\n      {date_header}:")
                print(f"         (no games)")
            
            # Move to next day
            current_date = current_date + pd.Timedelta(days=1)
        
        # If check-api-usage, save to tmp/ folder for testing
        if args.check_api_usage:
            snapshot_filename = generate_snapshot_filename(current_time)
            s3_key = save_dataframe_to_s3(current_df, sport, snapshot_filename, 'snapshot', is_test=True)
            print(f"\n   ℹ️  TEST mode - saved to S3 tmp/ folder: {s3_key}")
            print(f"      View at: https://s3.console.aws.amazon.com/s3/object/{S3_BUCKET}?prefix={s3_key}")
            continue
        
        # Save current snapshot to S3
        snapshot_filename = generate_snapshot_filename(current_time)
        s3_key = save_dataframe_to_s3(current_df, sport, snapshot_filename, 'snapshot', is_test=False)
        saved_snapshots.append(s3_key)
        print(f"\n   💾 Saved snapshot to S3: {s3_key}")
        
        # Find previous snapshots (1h and 24h ago) in S3
        time_1h_ago = current_time - LOOKBACK_WINDOW_1H
        time_24h_ago = current_time - LOOKBACK_WINDOW_24H
        
        snapshot_1h_key = find_snapshot_near_time_s3(sport, time_1h_ago)
        snapshot_24h_key = find_snapshot_near_time_s3(sport, time_24h_ago)
        
        # Convert times to ET for display
        time_1h_ago_et = time_1h_ago.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        time_24h_ago_et = time_24h_ago.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        
        # Calculate tolerance window for display
        time_1h_start = (time_1h_ago - SNAPSHOT_TIME_TOLERANCE).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        time_1h_end = (time_1h_ago + SNAPSHOT_TIME_TOLERANCE).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        time_24h_start = (time_24h_ago - SNAPSHOT_TIME_TOLERANCE).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        time_24h_end = (time_24h_ago + SNAPSHOT_TIME_TOLERANCE).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        
        print(f"\n   🔍 Looking for historical snapshots in S3:")
        print(f"      1h ago ({time_1h_start.strftime('%H:%M')}-{time_1h_end.strftime('%H:%M ET')}): {'✅ Found' if snapshot_1h_key else '❌ Not found'}")
        if snapshot_1h_key:
            print(f"         {snapshot_1h_key}")
        print(f"      24h ago ({time_24h_start.strftime('%H:%M')}-{time_24h_end.strftime('%H:%M ET')}): {'✅ Found' if snapshot_24h_key else '❌ Not found'}")
        if snapshot_24h_key:
            print(f"         {snapshot_24h_key}")
        
        # Load previous snapshots from S3
        df_1h = load_dataframe_from_s3(snapshot_1h_key) if snapshot_1h_key else None
        df_24h = load_dataframe_from_s3(snapshot_24h_key) if snapshot_24h_key else None
        
        if df_1h is None and df_24h is None:
            print(f"\n   ℹ️  First run for {sport} - no historical data for comparison")
            print(f"      Current snapshot saved to S3")
            print(f"      ⏰ Next run at top of hour will have data for comparison")
            
            # Calculate next top of hour
            current_time_et = current_time.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
            next_hour_et = (current_time_et + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            minutes_until_next = int((next_hour_et - current_time_et).total_seconds() / 60)
            print(f"      📅 Next scheduled run: {next_hour_et.strftime('%I:%M %p ET')} ({minutes_until_next} minutes)")
            
            continue
        
        # Compare and detect movements
        print(f"\n   🔍 Detecting line movements...")
        movements = []
        
        # Get threshold for this sport
        sport_threshold = MOVEMENT_THRESHOLDS[sport]
        
        for _, current_row in current_df.iterrows():
            game_id = current_row['game_id']
            bookmaker = current_row['bookmaker']
            
            # Find matching rows in previous snapshots
            prev_1h_row = None
            prev_24h_row = None
            
            if df_1h is not None:
                mask = (df_1h['game_id'] == game_id) & (df_1h['bookmaker'] == bookmaker)
                if mask.any():
                    prev_1h_row = df_1h[mask].iloc[0]
            
            if df_24h is not None:
                mask = (df_24h['game_id'] == game_id) & (df_24h['bookmaker'] == bookmaker)
                if mask.any():
                    prev_24h_row = df_24h[mask].iloc[0]
            
            # Skip if no comparison data for this game/bookmaker
            if prev_1h_row is None and prev_24h_row is None:
                continue
            
            # Process away side
            movement_away = process_side_movement(
                current_row, prev_1h_row, prev_24h_row, 'away', sport_threshold
            )
            if movement_away:
                movements.append(movement_away)
            
            # Process home side
            movement_home = process_side_movement(
                current_row, prev_1h_row, prev_24h_row, 'home', sport_threshold
            )
            if movement_home:
                movements.append(movement_home)
        
        # Save movements to S3
        if movements:
            movements_df = pd.DataFrame(movements)
            output_filename = f"{sport.split('_')[1]}_movements_{current_time.strftime(TIMESTAMP_FORMAT)}.csv"
            s3_key = save_dataframe_to_s3(movements_df, sport, output_filename, 'movement')
            saved_movements.append((SPORT_DISPLAY_NAMES[sport], movements_df))  # Store for email
            
            # Update noteworthy plays count for this sport
            sport_name = SPORT_DISPLAY_NAMES[sport]
            if sport_name in sport_summaries:
                sport_summaries[sport_name]['noteworthy_plays'] = len(movements_df)
            
            print(f"\n   📊 Movement Summary:")
            print(f"      Total movements detected: {len(movements_df)}")
            print(f"      Significant hourly: {movements_df['significant_hourly'].sum()}")
            print(f"      Significant daily: {movements_df['significant_daily'].sum()}")
            print(f"      Crossed zero (1h): {movements_df['crossed_zero_1h'].sum()}")
            print(f"      Crossed zero (24h): {movements_df['crossed_zero_24h'].sum()}")
            print(f"   💾 Saved movements to S3: {s3_key}")
        else:
            print(f"\n   ℹ️  No significant movements detected")
    
    print(f"\n{'='*80}")
    print("✅ Complete")
    print(f"{'='*80}")
    
    # Summary by sport
    if sport_summaries:
        print(f"\n📊 Summary:")
        for sport_name, summary in sport_summaries.items():
            noteworthy_str = f", {summary['noteworthy_plays']} noteworthy plays" if summary['noteworthy_plays'] > 0 else ""
            print(f"   {sport_name}: grabbed {summary['num_games']} future games on {summary['unique_days']} unique days{noteworthy_str}")
    
    # Summary of saved files in S3
    if saved_snapshots or saved_movements:
        print(f"\n📁 Files Saved to S3:")
        
        if saved_snapshots:
            print(f"\n   Snapshots (s3://{S3_BUCKET}/):")
            for s3_key in saved_snapshots:
                print(f"      {s3_key}")
        
        if saved_movements:
            print(f"\n   Movements:")
            for sport_name, df in saved_movements:
                print(f"      {sport_name}: {len(df)} movements")
    
    # Send email if running in Lambda (always send, even if no movements)
    if IS_LAMBDA:
        print(f"\n📧 Sending email alert...")
        
        # Convert saved_movements list to dict
        movements_dict = {}
        for sport_name, df in saved_movements:
            movements_dict[sport_name] = df
        
        # Build sport thresholds dict for email
        sport_thresholds = {
            'NBA': MOVEMENT_THRESHOLD_NBA,
            'NFL': MOVEMENT_THRESHOLD_NFL
        }
        
        # Generate email with ET timestamp
        current_time_et = current_time.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
        time_str_et = current_time_et.strftime('%b %d, %Y %I:%M %p ET')
        
        if saved_movements:
            subject = f"🚨 Line Movement Alert - {time_str_et}"
        else:
            subject = f"✅ Line Movement Check - No Significant Changes - {time_str_et}"
        
        # Generate HTML email with charts
        html_body = format_movement_email_html(
            sport_summaries, 
            movements_dict, 
            sport_thresholds, 
            current_time,
            DISPLAY_NAME_TO_SPORT_KEY,
            current_snapshots
        )
        
        # Generate plain text fallback
        text_body = format_movement_email(
            sport_summaries, 
            movements_dict, 
            sport_thresholds, 
            current_time, 
            current_snapshots
        )
        
        # Send via SES with HTML and inline images
        send_email_via_ses(subject, html_body, text_body)
    else:
        print(f"\n💻 Local run complete - email only sent when running in Lambda")
    
    print()  # Extra newline


def process_side_movement(current_row, prev_1h_row, prev_24h_row, side: str, threshold: float) -> Optional[dict]:
    """
    Process movement for one side (away or home) of a game/bookmaker.
    
    Returns:
        Dict with movement data, or None if no significant movement
    """
    # Get current values
    current_spread = current_row[f'{side}_spread']
    current_price = current_row[f'{side}_price']
    current_adjusted = current_row[f'{side}_adjusted_spread']
    
    # Initialize movement dict
    movement = {
        'game_id': current_row['game_id'],
        'away_team': current_row['away_team'],
        'home_team': current_row['home_team'],
        'game_time': current_row['game_time'],
        'bookmaker': current_row['bookmaker'],
        'side': side,
        'current_raw_spread': current_spread,
        'current_price': current_price,
        'current_adjusted_spread': current_adjusted,
        'timestamp': current_row['fetched_at'],
    }
    
    # Process 1h comparison
    if prev_1h_row is not None:
        prev_1h_spread = prev_1h_row[f'{side}_spread']
        prev_1h_price = prev_1h_row[f'{side}_price']
        prev_1h_adjusted = prev_1h_row[f'{side}_adjusted_spread']
        
        movement['prev_1h_raw_spread'] = prev_1h_spread
        movement['prev_1h_price'] = prev_1h_price
        movement['prev_1h_adjusted_spread'] = prev_1h_adjusted
        movement['hourly_movement_raw'] = current_spread - prev_1h_spread
        movement['hourly_movement_adjusted'] = current_adjusted - prev_1h_adjusted
        movement['significant_hourly'] = abs(movement['hourly_movement_adjusted']) >= threshold
        movement['crossed_zero_1h'] = detect_crossed_zero(prev_1h_spread, current_spread)
    else:
        movement['prev_1h_raw_spread'] = None
        movement['prev_1h_price'] = None
        movement['prev_1h_adjusted_spread'] = None
        movement['hourly_movement_raw'] = None
        movement['hourly_movement_adjusted'] = None
        movement['significant_hourly'] = False
        movement['crossed_zero_1h'] = False
    
    # Process 24h comparison
    if prev_24h_row is not None:
        prev_24h_spread = prev_24h_row[f'{side}_spread']
        prev_24h_price = prev_24h_row[f'{side}_price']
        prev_24h_adjusted = prev_24h_row[f'{side}_adjusted_spread']
        
        movement['prev_24h_raw_spread'] = prev_24h_spread
        movement['prev_24h_price'] = prev_24h_price
        movement['prev_24h_adjusted_spread'] = prev_24h_adjusted
        movement['daily_movement_raw'] = current_spread - prev_24h_spread
        movement['daily_movement_adjusted'] = current_adjusted - prev_24h_adjusted
        movement['significant_daily'] = abs(movement['daily_movement_adjusted']) >= threshold
        movement['crossed_zero_24h'] = detect_crossed_zero(prev_24h_spread, current_spread)
    else:
        movement['prev_24h_raw_spread'] = None
        movement['prev_24h_price'] = None
        movement['prev_24h_adjusted_spread'] = None
        movement['daily_movement_raw'] = None
        movement['daily_movement_adjusted'] = None
        movement['significant_daily'] = False
        movement['crossed_zero_24h'] = False
    
    # Determine movement type and direction (use 1h if available, else 24h)
    if prev_1h_row is not None:
        movement['movement_type'] = calculate_movement_type(
            prev_1h_row[f'{side}_spread'], prev_1h_row[f'{side}_price'],
            current_spread, current_price
        )
        movement['movement_direction'] = determine_movement_direction(
            prev_1h_row[f'{side}_adjusted_spread'], current_adjusted
        )
    elif prev_24h_row is not None:
        movement['movement_type'] = calculate_movement_type(
            prev_24h_row[f'{side}_spread'], prev_24h_row[f'{side}_price'],
            current_spread, current_price
        )
        movement['movement_direction'] = determine_movement_direction(
            prev_24h_row[f'{side}_adjusted_spread'], current_adjusted
        )
    else:
        movement['movement_type'] = 'none'
        movement['movement_direction'] = 'none'
    
    # Only return if there's significant movement or crossed zero
    if (movement['significant_hourly'] or movement['significant_daily'] or 
        movement['crossed_zero_1h'] or movement['crossed_zero_24h']):
        return movement
    
    return None


if __name__ == '__main__':
    main()


# =============================================================================
# AWS LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    This is the entry point when running in Lambda.
    Runs the main tracking logic and sends email alerts.
    """
    try:
        print("Lambda function started")
        print(f"Event: {json.dumps(event)}")
        
        # Run main logic (will auto-detect Lambda environment)
        main()
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Line movement tracking complete'})
        }
    except Exception as e:
        print(f"Error in Lambda handler: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
