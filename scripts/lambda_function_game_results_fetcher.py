"""
AWS Lambda Function - Daily Multi-Sport Game Results Fetcher

Lambda function name: multi-sport-game-results-fetcher

What it does:
1. Git clone repo
2. Fetch yesterday's game results for active sports:
   - NBA: Player props (Odds API) + game results (NBA API) + game lines (Odds API historical)
   - NCAAB: Game results (ESPN API) + game lines (Odds API historical)
   - NFL/NCAAF: Game results (ESPN API)
3. Upload to S3 (sport-specific buckets)
4. Send SNS notification per sport (success or failure)

Controlled by ACTIVE_SPORTS env var (e.g., "nba,ncaab,nfl,ncaaf")

Runs at 9am ET daily to fetch yesterday's game results.
Isolates data fetching into its own retriable function.

SPORTS SUPPORT:
- NBA: Uses fetch_nba_player_props.py (props + game results) + fetch_historical_nba_season_lines.py (game lines)
  - Props: s3://the-odds-api-mt/nba/historical_player_props/{season}/{date}.csv
  - Games: s3://nba-api-mt/player_game_logs/{season}/{date}.csv
  - Game lines: s3://the-odds-api-mt/nba/historical_game_lines/{season}/nba_game_lines_{date}.csv
  - Note: NBA API player game logs take 12+ hours to become available

- NCAAB: Uses fetch_historical_game_results_espn_api.py (results) + fetch_historical_ncaab_season_lines.py (game lines)
  - Results: s3://ncaab-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
  - Game lines: s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/{date}.csv
- NFL/NCAAF: Uses fetch_historical_game_results_espn_api.py
  - NFL: s3://nfl-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
  - NCAAF: s3://ncaaf-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
  - Note: ESPN API data available within 1-2 hours after games finish

ENVIRONMENT VARIABLES:
- ACTIVE_SPORTS: Comma-separated list (e.g., "nba,ncaab,nfl,ncaaf")
  Default: "nba,ncaab"
  
Enable/Disable Sports:
  Lambda → Configuration → Environment variables → Edit
  ACTIVE_SPORTS = nba,ncaab (disable NFL/NCAAF)
  ACTIVE_SPORTS = nba,ncaab,nfl,ncaaf (enable all)

Performance Notes:
- Uses shallow git clone (--depth 1) to reduce clone time
- Timeout: 5 minutes to handle multiple sports and slow APIs
- Memory: 2048 MB for faster CPU and git operations
- @timed decorator tracks execution time per sport
- Timing breakdown included in CloudWatch logs and SNS notifications
- Real-time output streaming for fetch scripts (stream_output=True)

================================================================================
AWS SETUP INSTRUCTIONS
================================================================================

Creating Lambda Function:
- Name: multi-sport-game-results-fetcher
- Runtime: Python 3.12
- Architecture: x86_64
- Execution role: Use existing -> betting-dashboard-daily-update-role-ille2llh
- Click 'Create function'

Configuration:
- General configuration:
    - Memory: 2048 MB (for faster git clone and multiple sport processing)
    - Ephemeral storage: 2048 MB
    - Timeout: 5 minutes (300 seconds) - handles multiple sports
    - Execution role: betting-dashboard-daily-update-role-ille2llh

- Environment variables -> Edit -> Add:
    - GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
    - GITHUB_USERNAME: MylesThomas
    - GITHUB_EMAIL: mylescgthomas@gmail.com
    - SECRET_NAME: betting-dashboard-secrets
    - AWS_REGION_NAME: us-east-2
    - SNS_TOPIC_ARN: arn:aws:sns:us-east-2:232692785472:nba-props-alerts
    - ACTIVE_SPORTS: nba,ncaab,nfl,ncaaf (or any subset)
    
Note: Seasons are automatically determined from src/season_utils.py (no hardcoding!)

- Layers:
    - Add layer -> Specify ARN -> arn:aws:lambda:us-east-2:553035198032:layer:git-lambda2:8
    - Add layer -> Custom layer -> nba-props-fetcher-dependencies (version 1)
      (Contains: pandas, numpy, requests, pyyaml, python-dotenv, nba-api)

- Permissions (IAM Role: betting-dashboard-daily-update-role-ille2llh):
    - AmazonS3FullAccess (for s3://nba-api-mt bucket)
    - AWSLambdaBasicExecutionRole (for CloudWatch logs)
    - SecretsManagerReadWrite (for betting-dashboard-secrets)
    - SNS Publish permissions (for nba-props-alerts topic)

Verify Secrets in AWS Secrets Manager:
```bash
aws secretsmanager get-secret-value \
  --secret-id betting-dashboard-secrets \
  --region us-east-2 \
  --query 'SecretString' \
  --output text | python3 -c "import sys, json; data=json.load(sys.stdin); print('Keys:', ', '.join(data.keys()))"
```

Expected: Keys: ODDS_API_KEY, GITHUB_TOKEN

EventBridge Schedule (Run at 9:00 AM ET daily):
- Navigate to: AWS Console → Amazon EventBridge → Rules → Create rule
- Define rule detail:
    - Name: multi-sport-game-results-fetcher-scheduler
    - Description: Fetch yesterday's game results (NBA + NCAAB) at 9am ET daily
- Define schedule:
    - Schedule expression: cron(0 14 * * ? *)
    - (14:00 UTC = 9:00 AM ET)
- Select target:
    - Target type: AWS Lambda function
    - Function: multi-sport-game-results-fetcher
    - IMPORTANT - Execution role:
      * Select: "Create a new role for this specific resource"
      * DO NOT reuse existing EventBridge roles
      * This ensures the rule has proper permissions to invoke THIS Lambda
      * Prevents "FailedInvocation" errors from permission issues
- Review + Create

NOTE: NBA API player game logs may not be available if games finished late the previous night.
If issues occur, consider moving schedule to 2pm ET (cron(0 19 * * ? *)) for 12+ hour delay.

Testing:
```bash
# Default test (uses ACTIVE_SPORTS env var - default: nba,ncaab)
echo '{}' > ~/Downloads/tmp/payload.json
aws lambda invoke \
  --function-name multi-sport-game-results-fetcher \
  --cli-binary-format raw-in-base64-out \
  --payload file:///$HOME/Downloads/tmp/payload.json \
  --region us-east-2 \
  ~/Downloads/tmp/lambda_response.json && cat ~/Downloads/tmp/lambda_response.json

# Override active sports - NBA only
echo '{"active_sports": "nba"}' > ~/Downloads/tmp/payload.json
cat ~/Downloads/tmp/payload.json
aws lambda invoke \
  --function-name multi-sport-game-results-fetcher \
  --cli-binary-format raw-in-base64-out \
  --payload file:///$HOME/Downloads/tmp/payload.json \
  --region us-east-2 \
  ~/Downloads/tmp/lambda_response.json && cat ~/Downloads/tmp/lambda_response.json

# Override active sports - NCAAB only
echo '{"active_sports": "ncaab"}' > ~/Downloads/tmp/payload.json
aws lambda invoke \
  --function-name multi-sport-game-results-fetcher \
  --cli-binary-format raw-in-base64-out \
  --payload file:///$HOME/Downloads/tmp/payload.json \
  --region us-east-2 \
  ~/Downloads/tmp/lambda_response.json && cat ~/Downloads/tmp/lambda_response.json

# Multiple sports override
echo '{"active_sports": ["nba", "ncaab", "nfl"]}' > ~/Downloads/tmp/payload.json
aws lambda invoke \
  --function-name multi-sport-game-results-fetcher \
  --cli-binary-format raw-in-base64-out \
  --payload file:///$HOME/Downloads/tmp/payload.json \
  --region us-east-2 \
  ~/Downloads/tmp/lambda_response.json && cat ~/Downloads/tmp/lambda_response.json

# View logs in real-time
aws logs tail /aws/lambda/multi-sport-game-results-fetcher --follow --region us-east-2

# Or from Lambda console: Monitor → View CloudWatch logs
```

Example Timing Output (in logs and email):
```
============================================================
EXECUTION TIMING BREAKDOWN
============================================================
get_yesterday_et.......................... 0.12s (  0.1%)
get_secrets............................... 0.85s (  0.5%)
clone_repo................................ 22.45s ( 13.8%)
setup_python_path......................... 1.23s (  0.8%)
fetch_nba................................. 95.67s ( 58.8%)
fetch_ncaab............................... 38.45s ( 23.6%)
send_sns.................................. 0.34s (  0.2%)
run_cmd................................... 3.67s (  2.3%)
------------------------------------------------------------
TOTAL..................................... 162.78s
============================================================
```

Example Real-Time Streaming Output (NBA):
```
⏱️  [fetch_nba] Starting...
Running: python3 scripts/fetch_nba_player_props.py --date 2026-01-24 --fetch-games --season 2025-26 --s3
================================================================================
CONFIGURATION
================================================================================
API_KEY: ********xxxx
SEASON: 2025-26
...
================================================================================
FETCHING PROPS FOR 2026-01-24 (Friday)
================================================================================
Fetching events for 2026-01-24 (timestamp: 2026-01-24T12:00:00Z)
API call successful - Cost: 1 credits, Remaining: 499,999
Found 12 events for 2026-01-24
Processing game 1/12: Brooklyn Nets @ Cleveland Cavaliers
  ✅ Found 450 player props
[... continues ...]
✅ [fetch_nba] Completed in 95.67s
```

Example Real-Time Streaming Output (NCAAB):
```
⏱️  [fetch_ncaab] Starting...
Running: python3 scripts/fetch_historical_game_results_espn_api.py --sport ncaab --date 2026-01-24 --s3
================================================================================
FETCHING NCAAB GAME RESULTS FOR 2026-01-24
================================================================================
📅 Fetching games for 2026-01-24...
✅ Found 145 completed games
📊 Processing games...
✅ [fetch_ncaab] Completed in 38.45s
```

S3 Output:
```
# NBA
s3://the-odds-api-mt/nba/historical_player_props/2025-26/{date}.csv
s3://nba-api-mt/player_game_logs/2025-26/{date}.csv

# NCAAB
s3://ncaab-betting-mt/data/01_input/historical_game_results/{timestamp}.csv

# NFL (not implemented yet)
s3://nfl-betting-mt/data/01_input/historical_game_results/{timestamp}.csv

# NCAAF (not implemented yet)
s3://ncaaf-betting-mt/data/01_input/historical_game_results/{timestamp}.csv
```

Author: Myles Thomas
Date: 2026-01-12
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
# HELPER FUNCTIONS
# =============================================================================

@timed
def get_yesterday_et():
    """Get yesterday's date in Eastern Time."""
    et_tz = ZoneInfo('America/New_York')
    yesterday = (datetime.now(et_tz) - timedelta(days=1)).strftime('%Y-%m-%d')
    return yesterday


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
        'PYTHONPATH': '/opt/python'  # Lambda layer packages location
    }
    if extra_env:
        env.update(extra_env)
    
    if stream_output:
        # Real-time streaming for long commands (fetch script)
        print(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        stdout_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)  # Print immediately
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
    
    # Use shallow clone (--depth 1) to speed up cloning in Lambda
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

# Sport icons
SPORT_ICONS = {
    'nba': '🏀',
    'nfl': '🏈',
    'ncaab': '🏀',
    'ncaaf': '🏈'
}

# Default active sports
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
    """
    # Priority 1: Event payload override
    if event and 'active_sports' in event:
        sports = event['active_sports']
        if isinstance(sports, str):
            return [sport.strip() for sport in sports.split(',')]
        elif isinstance(sports, list):
            return [sport.strip() for sport in sports]
    
    # Priority 2: Environment variable
    active_sports_str = os.getenv('ACTIVE_SPORTS', ','.join(DEFAULT_ACTIVE_SPORTS))
    return [sport.strip() for sport in active_sports_str.split(',')]


def lambda_handler(event, context):
    """
    Main handler - runs daily at 9am ET to fetch yesterday's game results for active sports.
    
    Sports Supported:
    - NBA: fetch_nba_player_props.py (props + game results)
    - NCAAB: fetch_historical_game_results_espn_api.py
    - NFL: Not implemented yet
    - NCAAF: Not implemented yet
    
    Environment Variables:
        ACTIVE_SPORTS: Comma-separated list (default: "nba,ncaab")
    """
    lambda_start = time.time()
    print(f"🏀🏈 Multi-Sport Game Results Fetcher - {context.aws_request_id}")
    
    yesterday = get_yesterday_et()
    print(f"📅 Yesterday: {yesterday}\n")
    
    # Get active sports
    active_sports = get_active_sports(event)
    print(f"🎯 Active sports: {', '.join([s.upper() for s in active_sports])}\n")
    
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
        sys.path.insert(0, os.path.join(repo_dir, 'src'))
        from season_utils import (
            get_current_nba_season,
            get_current_nfl_season,
            get_current_ncaab_season,
            get_current_ncaaf_season
        )
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
                # Get season for this sport
                season = season_funcs[sport]()
                print(f"{sport_icon} Season: {season}\n")
                
                # =============================================================================
                # NBA - Uses fetch_nba_player_props.py (props + game results)
                # =============================================================================
                if sport == 'nba':
                    print(f"📥 Fetching props + games for {yesterday}...")
                    cmd = [
                        'python3', 'scripts/fetch_nba_player_props.py',
                        '--date', yesterday,
                        '--fetch-games',
                        '--season', season,
                        '--s3'
                    ]
                    
                    fetch_start = time.time()
                    print(f"⏱️  [fetch_{sport}] Starting...")
                    _, stderr, code = run_cmd(
                        cmd, 
                        cwd=repo_dir, 
                        extra_env={'ODDS_API_KEY': odds_api_key}, 
                        stream_output=True
                    )
                    fetch_elapsed = time.time() - fetch_start
                    TIMING_DATA[f'fetch_{sport}'] = fetch_elapsed
                    print(f"✅ [fetch_{sport}] Completed in {fetch_elapsed:.2f}s\n")
                    
                    if code != 0:
                        print(f"❌ {sport_upper} fetch failed: {stderr}")
                        all_results.append({
                            'sport': sport,
                            'status': 'error',
                            'error': stderr
                        })
                        continue
                    
                    print(f"✅ {sport_upper} - Props + games fetched successfully")
                    nba_result = {
                        'sport': sport,
                        'status': 'success',
                        's3_props': f"s3://the-odds-api-mt/nba/historical_player_props/{season}/{yesterday}.csv",
                        's3_games': f"s3://nba-api-mt/player_game_logs/{season}/{yesterday}.csv"
                    }
                    # NBA game lines (spread + moneyline) for same date
                    print(f"📥 Fetching NBA game lines for {yesterday}...")
                    lines_start = time.time()
                    _, lines_stderr, lines_code = run_cmd(
                        [
                            'python3', 'scripts/fetch_historical_nba_season_lines.py',
                            '--date', yesterday,
                            '--no-local-backup'
                        ],
                        cwd=repo_dir,
                        extra_env={'ODDS_API_KEY': odds_api_key},
                        stream_output=True
                    )
                    TIMING_DATA['fetch_nba_game_lines'] = time.time() - lines_start
                    if lines_code == 0:
                        nba_result['s3_game_lines'] = (
                            f"s3://the-odds-api-mt/nba/historical_game_lines/{season}/nba_game_lines_{yesterday}.csv"
                        )
                        print(f"✅ NBA game lines completed\n")
                    else:
                        nba_result['game_lines_error'] = (lines_stderr or '')[:150]
                        print(f"⚠️ NBA game lines failed (results still saved)\n")
                    all_results.append(nba_result)
                
                # =============================================================================
                # NCAAB - Uses fetch_historical_game_results_espn_api.py
                # =============================================================================
                elif sport == 'ncaab':
                    print(f"📥 Fetching game results for {yesterday}...")
                    cmd = [
                        'python3', 'scripts/fetch_historical_game_results_espn_api.py',
                        '--sport', 'ncaab',
                        '--date', yesterday,
                        '--s3'
                    ]
                    
                    fetch_start = time.time()
                    print(f"⏱️  [fetch_{sport}] Starting...")
                    stdout, stderr, code = run_cmd(cmd, cwd=repo_dir, stream_output=True)
                    fetch_elapsed = time.time() - fetch_start
                    TIMING_DATA[f'fetch_{sport}'] = fetch_elapsed
                    print(f"✅ [fetch_{sport}] Completed in {fetch_elapsed:.2f}s\n")
                    
                    if code != 0:
                        print(f"❌ {sport_upper} fetch failed: {stderr}")
                        all_results.append({
                            'sport': sport,
                            'status': 'error',
                            'error': stderr
                        })
                        continue
                    
                    # Extract S3 path from stdout
                    s3_path = None
                    for line in stdout.split('\n'):
                        if 's3://ncaab-betting-mt' in line and '.csv' in line:
                            s3_path = line.strip()
                            break
                    
                    print(f"✅ {sport_upper} - Game results fetched successfully")
                    ncaab_result = {
                        'sport': sport,
                        'status': 'success',
                        's3_games': s3_path or f"s3://ncaab-betting-mt/data/01_input/historical_game_results/"
                    }
                    # NCAAB game lines (spread + totals) for same date
                    print(f"📥 Fetching NCAAB game lines for {yesterday}...")
                    lines_start = time.time()
                    _, lines_stderr, lines_code = run_cmd(
                        [
                            'python3', 'scripts/fetch_historical_ncaab_season_lines.py',
                            '--date', yesterday,
                            '--s3',
                            '--skip-existing'
                        ],
                        cwd=repo_dir,
                        extra_env={'ODDS_API_KEY': odds_api_key},
                        stream_output=True
                    )
                    TIMING_DATA['fetch_ncaab_game_lines'] = time.time() - lines_start
                    if lines_code == 0:
                        ncaab_result['s3_game_lines'] = (
                            f"s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/{yesterday}.csv"
                        )
                        print(f"✅ NCAAB game lines completed\n")
                    else:
                        ncaab_result['game_lines_error'] = (lines_stderr or '')[:150]
                        print(f"⚠️ NCAAB game lines failed (results still saved)\n")
                    all_results.append(ncaab_result)
                
                # =============================================================================
                # NFL / NCAAF - Not Implemented Yet
                # =============================================================================
                elif sport in ['nfl', 'ncaaf']:
                    print(f"⚠️  {sport_upper} - Not implemented yet")
                    all_results.append({
                        'sport': sport,
                        'status': 'not_implemented'
                    })
                
                else:
                    print(f"❌ Unknown sport: {sport_upper}")
                    all_results.append({
                        'sport': sport,
                        'status': 'error',
                        'error': f'Unknown sport: {sport}'
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
            status_icon = '✅' if result['status'] == 'success' else '⚠️' if result['status'] == 'not_implemented' else '❌'
            print(f"{status_icon} {result['sport'].upper()}: {result['status']}")
        print(f"{'='*80}\n")
        
        # Determine overall status
        success_count = sum(1 for r in all_results if r['status'] == 'success')
        error_count = sum(1 for r in all_results if r['status'] == 'error')
        total_count = len(all_results)
        
        # Choose appropriate subject and emoji
        if success_count == total_count:
            subject = f"✅ Game Results Fetched - {yesterday}"
        elif success_count > 0:
            subject = f"⚠️ Partial Success - Game Results - {yesterday}"
        else:
            subject = f"❌ Game Results Fetch Failed - {yesterday}"
        
        # Build notification message
        msg = [
            f"Date: {yesterday}\n",
            timing_summary,
            "\n\nResults:"
        ]
        for result in all_results:
            msg.append(f"\n{result['sport'].upper()}: {result['status']}")
            if result['status'] == 'success':
                if 's3_props' in result:
                    msg.append(f"  Props: ✅ Uploaded to {result['s3_props']}")
                if 's3_games' in result:
                    msg.append(f"  Games: ✅ Uploaded to {result['s3_games']}")
                if 's3_game_lines' in result:
                    msg.append(f"  Game lines: ✅ Uploaded to {result['s3_game_lines']}")
                if 'game_lines_error' in result:
                    msg.append(f"  Game lines: ⚠️ Failed ({result['game_lines_error']})")
            elif result['status'] == 'error':
                if 'error' in result:
                    # Truncate long errors for email readability
                    error_preview = result['error'][:200] + '...' if len(result['error']) > 200 else result['error']
                    msg.append(f"  Error: {error_preview}")
        
        send_sns(subject, '\n'.join(msg))
        
        # Round timing data for JSON response
        timing_rounded = {k: round(v, 2) for k, v in TIMING_DATA.items()}
        
        return {
            'statusCode': 200, 
            'body': json.dumps({
                'date': yesterday,
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
        
        error_msg = f"Date: {yesterday}\n\n{timing_summary}\n\nError: {str(e)}"
        send_sns("❌ Critical Error - Game Results Fetch", error_msg)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

