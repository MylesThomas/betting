"""
AWS Lambda Function - All Sports Line Steam Alerts Tracking

Python script: scripts/lambda_function_all_sports_line_steam_alerts_tracking.py
Lambda function name: line-steam-alerts
Handler: lambda_function_all_sports_line_steam_alerts_tracking.lambda_handler

================================================================================
OVERVIEW
================================================================================

Single Lambda that checks multiple sports for line steam (controlled by ACTIVE_SPORTS env var).

Currently Active: NBA + NCAAB (set via ACTIVE_SPORTS=nba,ncaab)
Easily Enable: NFL/NCAAF by changing env var (ACTIVE_SPORTS=nba,ncaab,nfl,ncaaf)

Runs hourly, loops through active sports, sends separate email per sport if steam detected.

================================================================================
WHAT IT DOES
================================================================================

MODE 1 - STEAM DETECTION (Hourly, 10:05am-6:05pm ET):
1. Git clone repo to /tmp/betting
2. Loop through ACTIVE_SPORTS (e.g., nba, ncaab)
3. For each sport:
   - Load hourly line snapshots from S3 (last 7 days)
   - Calculate consensus movements (opening → current)
   - Check for threshold+ point steam (both directions)
   - If steam detected:
     * Save plays to S3: s3://{sport}-betting-mt/data/04_output/plays/line-steam/{date}.csv
     * Send SNS email: "🚨 {SPORT} Line Steam Alert - {date}"
   - If no steam: Silent success (no email)

MODE 2 - DAILY REPORT (1:05pm ET):
1. Loop through ACTIVE_SPORTS
2. For each sport:
   - Calculate yesterday's results (match plays to game outcomes)
   - Save results to S3: s3://{sport}-betting-mt/data/04_output/results/line-steam/{date}.csv
   - Generate YTD report (W-L-T, ROI, breakdowns)
   - Send email: "📊 {SPORT} Line Steam Daily Report - {date}"
   - Email includes: results summary, S3 path, YTD stats

================================================================================
S3 DATA STRUCTURE
================================================================================

Input (Line Snapshots - Shared Bucket):
  s3://betting-line-movement-snapshots/data/01_input/the-odds-api/{sport}/line_movement/
  - snapshot_YYYY-MM-DD_HH-MM-SS.csv (hourly snapshots from line tracking Lambda)

Output (Plays - Sport-Specific Buckets):
  s3://nba-betting-mt/data/04_output/plays/line-steam/{date}.csv
  s3://ncaab-betting-mt/data/04_output/plays/line-steam/{date}.csv
  s3://nfl-betting-mt/data/04_output/plays/line-steam/{date}.csv
  s3://ncaaf-betting-mt/data/04_output/plays/line-steam/{date}.csv
  
  Format: Appends ALL detections (no dedup) - tracks steam evolution
  Columns: detected_at, game_id, game_date, game_time, season, opening_favorite,
           opening_underdog, opening_favorite_spread, current_favorite_spread,
           steam_magnitude, steam_direction, steamed_team, threshold, play_team,
           play_spread, status, actual_margin, cover_margin

Output (Results - Sport-Specific Buckets):
  s3://{sport}-betting-mt/data/04_output/results/line-steam/{date}.csv
  
  Format: Deduped at (game_id, steam_direction) - keeps LARGEST steam magnitude
  Same columns as plays, but status updated to 'won'/'lost'/'push'

================================================================================
EMAIL ALERTS
================================================================================

STEAM DETECTION EMAILS (Hourly when steam detected):

Subject: "🚨 NBA Line Steam Alert - 2026-01-23"
Subject: "🚨 NCAAB Line Steam Alert - 2026-01-23"
Subject: "🚨 NFL Line Steam Alert - 2026-01-26"
Subject: "🚨 NCAAF Line Steam Alert - 2026-08-30"

Body format:
  🏀 NCAAB LINE STEAM ALERT - 2026-01-23
  
  Significant line movement detected (1.0+ points):
  
  ================================================================================
  TOWARD OPENING UNDERDOG (10 games)
  ================================================================================
  
  Game 1: Longwood Lancers vs Charleston Southern Buccaneers
    Opening: Longwood Lancers -1.0 | Charleston Southern Buccaneers +1.0
    Current: Longwood Lancers +1.5 | Charleston Southern Buccaneers -1.5
    Steam: 2.5 points toward opening underdog Charleston Southern Buccaneers
    Game time: 07:00 PM ET
    Snapshots tracked: 8.0 hours
  
  [... more games ...]

No email sent if no steam detected (silent success, check CloudWatch logs).

DAILY REPORT EMAILS (1:05pm ET daily):

Subject: "📊 NBA Line Steam Daily Report - 2026-01-24"
Subject: "📊 NCAAB Line Steam Daily Report - 2026-01-24"

Body format:
  🏀 NCAAB LINE STEAM DAILY REPORT - 2026-01-24
  
  Results calculated for 2026-01-24
  
  [Script output: W-L-T record, ROI, game-by-game results]
  
  Results saved to: s3://ncaab-betting-mt/data/04_output/results/line-steam/2026-01-24.csv

Email sent even if no plays (shows "No plays found" message).

================================================================================
AWS LAMBDA SETUP
================================================================================

1. CREATE FUNCTION
   - Name: line-steam-alerts
   - Runtime: Python 3.12
   - Architecture: x86_64
   - Execution role: betting-dashboard-daily-update-role-ille2llh
   - Handler: lambda_function_all_sports_line_steam_alerts_tracking.lambda_handler

2. UPLOAD CODE
   - Option A: Upload just this file (RECOMMENDED)
     * Code tab → Upload from → .zip file
     * Only need: lambda_function_all_sports_line_steam_alerts_tracking.py
     * Everything else cloned from GitHub at runtime
     
     Create minimal zip:
       cd /Users/thomasmyles/dev/betting/scripts
       zip lambda_minimal.zip lambda_function_all_sports_line_steam_alerts_tracking.py
     
   - Option B: Copy-paste in Console (fastest for updates)
     * Code tab → Edit code inline
     * Copy entire lambda_function_all_sports_line_steam_alerts_tracking.py
     * Paste into lambda_function.py in Console
     * Save (Ctrl+S)
   
   Why this works:
   - Lambda clones repo to /tmp/betting
   - All scripts/configs loaded from /tmp/betting
   - Changes to scripts/configs auto-picked up (no redeployment)
   - Only need to redeploy Lambda if THIS file changes

3. ENVIRONMENT VARIABLES
   GITHUB_REPO_URL = https://github.com/MylesThomas/betting.git
   GITHUB_USERNAME = MylesThomas
   GITHUB_EMAIL = mylescgthomas@gmail.com
   SECRET_NAME = betting-dashboard-secrets
   AWS_REGION_NAME = us-east-2
   SNS_TOPIC_ARN = arn:aws:sns:us-east-2:232692785472:nba-props-alerts
   ACTIVE_SPORTS = nba,ncaab

   To enable NFL/NCAAF: Change ACTIVE_SPORTS = nba,ncaab,nfl,ncaaf
   
   IMPORTANT: Changes to ACTIVE_SPORTS take effect immediately (no redeployment)

4. GENERAL CONFIGURATION
   Memory: 2048 MB
   Ephemeral storage: 2048 MB
   Timeout: 5 min 0 sec

5. LAMBDA LAYERS
   Layer 1: arn:aws:lambda:us-east-2:553035198032:layer:git-lambda2:8
   Layer 2: nba-props-fetcher-dependencies (latest version)

6. IAM PERMISSIONS (Role: betting-dashboard-daily-update-role-ille2llh)
   - AmazonS3FullAccess (read snapshots, write plays/results)
   - AWSLambdaBasicExecutionRole (CloudWatch logs)
   - SecretsManagerReadWrite (read GITHUB_TOKEN, ODDS_API_KEY)
   - SNS Publish (send email alerts)

7. EVENTBRIDGE SCHEDULE
   Name: line-steam-alerts-schedule
   Description: Check line steam hourly for active sports - 24/7
   Schedule: cron(5 * * * ? *)
   Translation: Run at :05 past EVERY hour, 24/7
   Target: Lambda function → line-steam-alerts
   State: Enabled

   Why 24/7?
   - Line tracking Lambda already runs 24/7 collecting snapshots
   - Sharp bettors operate globally (Asian/European markets)
   - Breaking news (injuries, lineups) can drop at midnight
   - Lambda only sends email if steam detected (self-regulating)
   - Cost: ~$0.05/month more than 7am-6pm (negligible)
   - Better to capture everything, adjust thresholds if too noisy
   - Runs ~5 minutes AFTER line tracking Lambda collects snapshots

================================================================================
CONFIGURATION MANAGEMENT (NO REDEPLOYMENT NEEDED!)
================================================================================

The beauty of this Lambda: it clones the repo every run, so changes to scripts
and configs are automatically picked up without redeploying the Lambda.

Change Sport Thresholds:
  1. Edit config/line_steam_config.yaml on GitHub:
     sports:
       ncaab:
         threshold: 2.0  # Was 1.0
  2. Commit + push to main branch
  3. Next Lambda run auto-uses new threshold
  4. NO Lambda redeployment needed!

Change Detection Logic:
  1. Edit scripts/check_line_steam.py on GitHub
  2. Commit + push to main branch  
  3. Next Lambda run uses new code
  4. NO Lambda redeployment needed!

Enable/Disable Sports:
  Lambda → Configuration → Environment variables → Edit
  ACTIVE_SPORTS = nba,ncaab,nfl  (no redeployment, immediate effect)

When to Redeploy Lambda:
  - ONLY if you change THIS FILE (lambda_function_all_sports_line_steam_alerts_tracking.py)
  - Everything else (scripts, configs, utilities) → just push to GitHub

Manual Test via Console:
  Test tab → Create new event:
  {
    "test": true
  }
  
  Expected output:
  - ✅ Status: Succeeded
  - Logs show processing of NBA and NCAAB
  - CloudWatch logs show steam detection for each sport
  - Emails sent for sports with steam detected

Manual Test via CLI:
  aws lambda invoke \
    --function-name line-steam-alerts \
    --payload '{}' \
    --region us-east-2 \
    response.json && cat response.json

View Logs:
  aws logs tail /aws/lambda/line-steam-alerts --follow --region us-east-2

================================================================================
MONITORING
================================================================================

CloudWatch Logs:
  /aws/lambda/line-steam-alerts
  
  Look for:
  - "🎯 Active sports: NBA, NCAAB"
  - "🚨 NBA STEAM DETECTED - Email sent!"
  - "✅ NCAAB - No steam detected"
  - "FINAL SUMMARY" showing results for each sport

Email Check:
  - Check inbox for "🚨 {SPORT} Line Steam Alert" emails
  - Should receive separate email per sport with steam
  - No email = no steam detected (this is normal!)

S3 Data Check:
  aws s3 ls s3://nba-betting-mt/data/04_output/plays/line-steam/
  aws s3 ls s3://ncaab-betting-mt/data/04_output/plays/line-steam/

================================================================================
TROUBLESHOOTING
================================================================================

Lambda timeout:
  - Increase memory to 3008 MB (faster processing)
  - Reduce days_back in config (currently 7 days)
  - Disable one sport temporarily (change ACTIVE_SPORTS)

No snapshots found:
  - Check line tracking Lambda is running hourly
  - Verify S3 bucket: s3://betting-line-movement-snapshots
  - Check prefix: data/01_input/the-odds-api/{sport}/line_movement/

No email sent but steam should be detected:
  - Check CloudWatch logs for "STEAM_DETECTED: YES"
  - Verify SNS_TOPIC_ARN env var
  - Confirm SNS subscription (email confirmed)

Team name mismatch errors:
  - Add mapping to config/line_steam_config.yaml:
    team_name_mappings:
      ncaab:
        "Odds API Name": "ESPN API Name"

================================================================================
DEPENDENCIES
================================================================================

Python Scripts:
  - scripts/check_line_steam.py (sport-agnostic steam detector)
  - scripts/calculate_line_steam_results.py (sport-agnostic results calculator)

Python Modules:
  - src/line_steam_utils.py (SportConfig, snapshot loading, steam detection)
  - src/season_utils.py (get_current_{nba|nfl|ncaab|ncaaf}_season)
  - src/config_loader.py (load config files)

Configuration:
  - config/line_steam_config.yaml (sport configs, thresholds, S3 buckets)
  - config/config.yaml (general config)
  - config/season_dates.yaml (season start/end dates)

Lambda Layers:
  - git-lambda2:8 (git for cloning repo)
  - nba-props-fetcher-dependencies (pandas, numpy, requests, pyyaml, boto3)

AWS Secrets:
  - betting-dashboard-secrets (GITHUB_TOKEN, ODDS_API_KEY)

S3 Buckets:
  - betting-line-movement-snapshots (line snapshots - shared)
  - nba-betting-mt (NBA plays/results)
  - ncaab-betting-mt (NCAAB plays/results)
  - nfl-betting-mt (NFL plays/results)
  - ncaaf-betting-mt (NCAAF plays/results)

================================================================================
EXAMPLE OUTPUT
================================================================================

CloudWatch Logs (when both sports have steam):
  
  🏀🏈 Multi-Sport Line Steam Detector - abc123-def456
  📅 Today: 2026-01-23
  
  🎯 Active sports: NBA, NCAAB
  
  ================================================================================
  🏀 Processing NBA
  ================================================================================
  
  🏀 Season: 2025-26
  🔍 Threshold: 1.0
  📅 Days back: 7
  
  📊 Analyzing NBA line movements for 2026-01-23...
  🚨 NBA STEAM DETECTED - Email sent!
  
  ================================================================================
  🏀 Processing NCAAB
  ================================================================================
  
  🏀 Season: 2025-26
  🔍 Threshold: 1.0
  📅 Days back: 7
  
  📊 Analyzing NCAAB line movements for 2026-01-23...
  🚨 NCAAB STEAM DETECTED - Email sent!
  
  ================================================================================
  FINAL SUMMARY
  ================================================================================
  ✅ NBA: success (steam: True)
  ✅ NCAAB: success (steam: True)
  ================================================================================

Author: Thomas Myles
Date: 2026-01-23 (initial implementation)
Date: 2026-01-24 (refactored to single multi-sport Lambda)
"""

import json
import os
import sys
import subprocess
import boto3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import wraps
from io import BytesIO


# =============================================================================
# TIMING TRACKER
# =============================================================================

# Global timing storage
TIMING_DATA = {}

def timed(func):
    """Decorator to track execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        print(f"⏱️  [{func_name}] Starting...")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            TIMING_DATA[func_name] = elapsed
            print(f"✅ [{func_name}] Completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            TIMING_DATA[func_name] = elapsed
            print(f"❌ [{func_name}] Failed after {elapsed:.2f}s")
            raise
    
    return wrapper


def format_timing_summary():
    """Format timing data as a readable string."""
    if not TIMING_DATA:
        return "No timing data available"
    
    lines = ["=" * 60, "EXECUTION TIMING BREAKDOWN", "=" * 60]
    total_time = sum(TIMING_DATA.values())
    
    for func_name, duration in TIMING_DATA.items():
        pct = (duration / total_time * 100) if total_time > 0 else 0
        lines.append(f"{func_name:.<40} {duration:>6.2f}s ({pct:>5.1f}%)")
    
    lines.append("-" * 60)
    lines.append(f"{'TOTAL':.<40} {total_time:>6.2f}s")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS (execution flow order per rule #12)
# =============================================================================

@timed
def get_today_et():
    """Get today's date in Eastern Time."""
    et_tz = ZoneInfo('America/New_York')
    today = datetime.now(et_tz).strftime('%Y-%m-%d')
    return today


@timed
def run_cmd(cmd, cwd=None, extra_env=None, stream_output=False):
    """
    Run shell command and return (stdout, stderr, returncode).
    
    Args:
        cmd: Command to run
        cwd: Working directory
        extra_env: Additional environment variables
        stream_output: If True, stream output in real-time (for long-running commands)
    """
    env = {
        **os.environ,
        'AWS_DEFAULT_REGION': os.environ['AWS_REGION_NAME'],
        'PYTHONPATH': '/opt/python'
    }
    if extra_env:
        env.update(extra_env)
    
    if stream_output:
        # Real-time streaming for long commands (analysis script)
        print(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        stdout_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            stdout_lines.append(line)
        
        process.wait()
        stdout = ''.join(stdout_lines)
        return stdout, '', process.returncode
    else:
        # Buffered output for quick commands (git clone)
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.stdout, result.stderr, result.returncode


@timed
def clone_repo(token):
    """Clone betting repo to /tmp/betting (shallow clone for speed)."""
    target = '/tmp/betting'
    if os.path.exists(target):
        subprocess.run(['rm', '-rf', target])
    
    repo_url = os.environ['GITHUB_REPO_URL']
    auth_url = repo_url.replace('https://', f'https://{token}@')
    
    _, stderr, code = run_cmd(['git', 'clone', '--depth', '1', auth_url, target])
    if code != 0:
        raise Exception(f"Git clone failed: {stderr}")
    
    return target


@timed
def send_sns(subject, message):
    """Send SNS notification."""
    sns = boto3.client('sns', region_name=os.environ['AWS_REGION_NAME'])
    sns.publish(
        TopicArn=os.environ['SNS_TOPIC_ARN'],
        Subject=subject,
        Message=message
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# Sport icons for emails
SPORT_ICONS = {
    'nba': '🏀',
    'nfl': '🏈',
    'ncaab': '🏀',
    'ncaaf': '🏈'
}

# Default active sports (can be overridden by ACTIVE_SPORTS env var)
DEFAULT_ACTIVE_SPORTS = ['nba', 'ncaab']


def get_active_sports(event=None):
    """
    Get list of active sports from event payload, environment variable, or default.
    
    Priority:
    1. event['active_sports'] (payload override)
    2. ACTIVE_SPORTS env var
    3. DEFAULT_ACTIVE_SPORTS constant
    
    Args:
        event: Lambda event dict (optional)
    
    Returns:
        list: List of sport strings (e.g., ['nba', 'ncaab'])
    
    Examples:
        get_active_sports({'active_sports': 'nba'}) → ['nba']
        get_active_sports({'active_sports': ['nba', 'nfl']}) → ['nba', 'nfl']
        get_active_sports() → from env var or default
    """
    # Priority 1: Event payload override
    if event and 'active_sports' in event:
        sports = event['active_sports']
        # Handle both string ("nba,ncaab") and list (["nba", "ncaab"]) formats
        if isinstance(sports, str):
            return [sport.strip() for sport in sports.split(',')]
        elif isinstance(sports, list):
            return [sport.strip() for sport in sports]
    
    # Priority 2: Environment variable
    active_sports_str = os.getenv('ACTIVE_SPORTS', ','.join(DEFAULT_ACTIVE_SPORTS))
    return [sport.strip() for sport in active_sports_str.split(',')]


# =============================================================================
# MAIN LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    Main Lambda handler - checks all active sports for steam.
    
    Loops through ACTIVE_SPORTS env var (default: nba,ncaab).
    Each sport sends separate email if steam detected.
    
    Environment Variables:
        ACTIVE_SPORTS: Comma-separated list of sports (e.g., "nba,ncaab,nfl")
                      Default: "nba,ncaab"
    
    Example:
        ACTIVE_SPORTS="nba,ncaab" → checks both
        ACTIVE_SPORTS="nba" → checks NBA only
        ACTIVE_SPORTS="nba,nfl,ncaab,ncaaf" → checks all 4
    """
    lambda_start = time.time()
    print(f"🏀🏈 Multi-Sport Line Steam Detector - {context.aws_request_id}")
    
    today = get_today_et()
    print(f"📅 Today: {today}\n")
    
    # Get active sports (from event payload, env var, or default)
    active_sports = get_active_sports(event)
    print(f"🎯 Active sports: {', '.join([s.upper() for s in active_sports])}\n")
    
    # Check if we should run daily report
    et_tz = ZoneInfo('America/New_York')
    current_time_et = datetime.now(et_tz)
    hour = current_time_et.hour
    minute = current_time_et.minute
    
    run_daily_report = (
        (hour == 13 and minute <= 10) or  # 1:00-1:10pm ET
        event.get('daily_report', False)
    )
    
    # Track results for all sports
    all_results = []
    
    try:
        # Get secrets (GITHUB_TOKEN + ODDS_API_KEY)
        print("⏱️  [get_secrets] Starting...")
        secrets_start = time.time()
        sm = boto3.client('secretsmanager', region_name=os.environ['AWS_REGION_NAME'])
        secret = sm.get_secret_value(SecretId=os.environ['SECRET_NAME'])
        secrets = json.loads(secret['SecretString'])
        github_token = secrets['GITHUB_TOKEN']
        odds_api_key = secrets['ODDS_API_KEY']
        secrets_elapsed = time.time() - secrets_start
        TIMING_DATA['get_secrets'] = secrets_elapsed
        print(f"✅ [get_secrets] Completed in {secrets_elapsed:.2f}s\n")
        
        # Clone repo
        print("📦 Cloning...")
        repo_dir = clone_repo(github_token)
        
        # Add src/ to path NOW that repo exists
        print("⏱️  [setup_python_path] Starting...")
        setup_start = time.time()
        src_path = os.path.join(repo_dir, 'src')
        sys.path.insert(0, src_path)
        print(f"   Added to sys.path: {src_path}")
        
        # Debug: Check if season_utils.py exists
        season_utils_path = os.path.join(src_path, 'season_utils.py')
        print(f"   season_utils.py exists: {os.path.exists(season_utils_path)}")
        
        # Import utilities
        try:
            from season_utils import (
                get_current_nba_season,
                get_current_nfl_season,
                get_current_ncaab_season,
                get_current_ncaaf_season
            )
            from line_steam_utils import SportConfig
            print("   ✅ Successfully imported season functions and SportConfig")
        except ImportError as e:
            print(f"   ❌ Import error: {e}")
            # List what's in season_utils
            import season_utils
            print(f"   Available in season_utils: {dir(season_utils)}")
            raise
        
        season_funcs = {
            'nba': get_current_nba_season,
            'nfl': get_current_nfl_season,
            'ncaab': get_current_ncaab_season,
            'ncaaf': get_current_ncaaf_season
        }
        
        setup_elapsed = time.time() - setup_start
        TIMING_DATA['setup_python_path'] = setup_elapsed
        print(f"✅ [setup_python_path] Completed in {setup_elapsed:.2f}s\n")
        
        # =============================================================================
        # LOOP THROUGH EACH ACTIVE SPORT
        # =============================================================================
        for sport in active_sports:
            sport_icon = SPORT_ICONS.get(sport, '🏀')
            sport_upper = sport.upper()
            
            print(f"\n{'='*80}")
            print(f"{sport_icon} Processing {sport_upper}")
            print(f"{'='*80}\n")
            
            try:
                # Get season and config for this sport
                season = season_funcs[sport]()
                sport_config = SportConfig(sport)
                threshold = event.get('threshold', sport_config.threshold)
                days_back = event.get('days_back', sport_config.days_back)
                
                print(f"{sport_icon} Season: {season}")
                print(f"🔍 Threshold: {threshold}")
                print(f"📅 Days back: {days_back}\n")
                
                # =============================================================================
                # DAILY REPORT MODE - Calculate yesterday's results + send YTD report
                # =============================================================================
                if run_daily_report:
                    print(f"\n📊 Running DAILY REPORT for {sport_upper}")
                    
                    yesterday = (datetime.now(et_tz) - timedelta(days=1)).strftime('%Y-%m-%d')
                    print(f"📅 Yesterday: {yesterday}")
                    
                    calc_cmd = [
                        'python3', 'scripts/calculate_line_steam_results.py',
                        '--sport', sport,
                        '--date', yesterday,
                        '--season', season
                    ]
                    
                    calc_start = time.time()
                    stdout, stderr, code = run_cmd(calc_cmd, cwd=repo_dir, stream_output=True)
                    calc_elapsed = time.time() - calc_start
                    TIMING_DATA[f'calculate_results_{sport}'] = calc_elapsed
                    
                    if code != 0:
                        print(f"⚠️  Results calculation had issues: {stderr}")
                        all_results.append({
                            'sport': sport,
                            'mode': 'daily_report',
                            'status': 'error',
                            'error': stderr
                        })
                        continue
                    
                    # Send daily report email
                    report_msg = (
                        f"{sport_icon} {sport_upper} LINE STEAM DAILY REPORT - {yesterday}\n\n"
                        f"Results calculated for {yesterday}\n\n"
                        f"{stdout}\n\n"
                        f"Results saved to: s3://{sport}-betting-mt/data/04_output/results/line-steam/{yesterday}.csv"
                    )
                    send_sns(f"📊 {sport_upper} Line Steam Daily Report - {yesterday}", report_msg)
                    print(f"📧 {sport_upper} daily report email sent!\n")
                    
                    all_results.append({
                        'sport': sport,
                        'mode': 'daily_report',
                        'status': 'success'
                    })
                    
                    continue
                
                # =============================================================================
                # STEAM DETECTION MODE - Check for steam + send alerts
                # =============================================================================
                
                print(f"📊 Analyzing {sport_upper} line movements for {today}...")
                cmd = [
                    'python3', 'scripts/check_line_steam.py',
                    '--sport', sport,
                    '--date', today,
                    '--threshold', str(threshold),
                    '--days-back', str(days_back),
                    '--save-plays',
                    '--season', season
                ]
                
                analysis_start = time.time()
                stdout, stderr, code = run_cmd(
                    cmd, 
                    cwd=repo_dir, 
                    extra_env={'ODDS_API_KEY': odds_api_key}, 
                    stream_output=False  # Suppress individual game logs
                )
                analysis_elapsed = time.time() - analysis_start
                TIMING_DATA[f'analyze_{sport}'] = analysis_elapsed
                
                if code != 0:
                    print(f"❌ {sport_upper} analysis failed: {stderr}")
                    all_results.append({
                        'sport': sport,
                        'status': 'error',
                        'error': stderr
                    })
                    continue
                
                # Check if steam detected
                steam_detected = "STEAM_DETECTED: YES" in stdout
                
                if steam_detected:
                    # Send alert
                    alert_msg = (
                        f"{sport_icon} {sport_upper} LINE STEAM ALERT - {today}\n\n"
                        f"Significant line movement detected ({threshold}+ points):\n\n"
                        f"{stdout}"
                    )
                    send_sns(f"🚨 {sport_upper} Line Steam Alert - {today}", alert_msg)
                    print(f"🚨 {sport_upper} STEAM DETECTED - Email sent!")
                else:
                    print(f"✅ {sport_upper} - No steam detected")
                
                all_results.append({
                    'sport': sport,
                    'status': 'success',
                    'steam_detected': steam_detected
                })
                
            except Exception as e:
                print(f"❌ Error processing {sport_upper}: {str(e)}")
                all_results.append({
                    'sport': sport,
                    'status': 'error',
                    'error': str(e)
                })
        
        # =============================================================================
        # FINAL SUMMARY
        # =============================================================================
        total_elapsed = time.time() - lambda_start
        TIMING_DATA['total_lambda_execution'] = total_elapsed
        
        timing_summary = format_timing_summary()
        print(f"\n{timing_summary}")
        
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        for result in all_results:
            status_icon = '✅' if result['status'] == 'success' else '❌'
            steam_text = f" (steam: {result.get('steam_detected', False)})" if result['status'] == 'success' else ''
            print(f"{status_icon} {result['sport'].upper()}: {result['status']}{steam_text}")
        print(f"{'='*80}\n")
        
        # Round timing data to 2 decimal places for readability
        timing_rounded = {k: round(v, 2) for k, v in TIMING_DATA.items()}
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'date': today,
                'results': all_results,
                'timing': timing_rounded
            })
        }
        
    except Exception as e:
        # Print timing summary on exception
        total_elapsed = time.time() - lambda_start
        TIMING_DATA['total_lambda_execution'] = total_elapsed
        timing_summary = format_timing_summary()
        print(f"\n{timing_summary}")
        
        error_msg = f"Date: {today}\n\n{timing_summary}\n\nError: {str(e)}"
        send_sns(f"❌ Critical Error - Line Steam Check", error_msg)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
