"""
AWS Lambda Function - Daily NBA Props + Game Results Fetcher

Lambda function name: nba-historical-game-and-props-results-fetcher

What it does:
1. Git clone repo
2. Fetch yesterday's NBA player props (The Odds API) + game results (NBA API)
3. Upload to S3:
   - Props: s3://the-odds-api-mt/nba/historical_player_props/{season}/{date}.csv
   - Game results: s3://nba-api-mt/player_game_logs/{season}/{date}.csv
4. Send SNS notification (success or failure)

Note: The --fetch-games flag fetches BOTH props AND game results.
Runs at 9am ET to fetch data BEFORE the main workflow runs.
Isolates the most common failure point (NBA API) into its own retriable function.

CRITICAL TIMING NOTE - Why Lambda May Timeout:
The NBA API has a significant delay between game completion and data availability:
  ✅ Props (The Odds API): Available immediately for historical dates
  ✅ Scoreboard data (NBA.com): Available within 1-2 hours after games finish
  ❌ Player game logs (NBA API): Takes 12+ HOURS to process and publish
  
If this Lambda runs too early (e.g., 9am ET for games that ended at 1am ET),
the NBA API player game logs endpoint will return empty/invalid JSON and timeout
on retries. This is EXPECTED behavior - not a bug!

Recommended Schedule:
- For games ending at midnight ET → Run Lambda at 2pm ET next day (14+ hours later)
- Current schedule (9am ET) works for games from 2+ days ago
- Consider splitting into two Lambdas if you need fresher data:
    1. Props-only Lambda (runs at 9am - always works)
    2. Games-only Lambda (runs at 2pm - waits for NBA API)

Performance Notes:
- Uses shallow git clone (--depth 1) to reduce clone time
- Increased timeout to 5 minutes to handle NBA API slowness and multiple API calls
- Increased memory to 2048 MB for faster CPU and git operations
- @timed decorator tracks execution time of each function
- Timing breakdown included in CloudWatch logs and SNS notifications
- Real-time output streaming for fetch script to identify bottlenecks (stream_output=True)

================================================================================
AWS SETUP INSTRUCTIONS
================================================================================

Creating Lambda Function:
- Name: nba-historical-game-and-props-results-fetcher
- Runtime: Python 3.12
- Architecture: x86_64
- Execution role: Use existing -> betting-dashboard-daily-update-role-ille2llh
- Click 'Create function'

Configuration:
- General configuration:
    - Memory: 2048 MB (increased for faster git clone and API processing)
    - Ephemeral storage: 2048 MB
    - Timeout: 5 minutes (300 seconds) - NBA API can be slow
    - Execution role: betting-dashboard-daily-update-role-ille2llh

- Environment variables -> Edit -> Add:
    - GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
    - GITHUB_USERNAME: MylesThomas
    - GITHUB_EMAIL: mylescgthomas@gmail.com
    - SECRET_NAME: betting-dashboard-secrets
    - AWS_REGION_NAME: us-east-2
    - SNS_TOPIC_ARN: arn:aws:sns:us-east-2:232692785472:nba-props-alerts
    
Note: Season is automatically determined from src/season_utils.py (no hardcoding!)

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

EventBridge Schedule (Run at 2:00 PM ET daily - AFTER NBA API updates):
- Navigate to: AWS Console → Amazon EventBridge → Rules → Create rule
- Define rule detail:
    - Name: nba-historical-game-and-props-results-fetcher
    - Description: Fetch yesterday's NBA game + props results at 2pm ET (after 12+ hour NBA API delay)
- Define schedule:
    - Schedule expression: cron(0 19 * * ? *)
    - (19:00 UTC = 2:00 PM ET - gives 12+ hours for NBA API to update)
- Select target:
    - Target type: AWS Lambda function
    - Function: nba-historical-game-and-props-results-fetcher
- Review + Create

IMPORTANT: Running at 9am ET will FAIL because NBA API player game logs take 12+ hours to publish!
Props data is available immediately, but game logs need ~14 hours after games finish.

Testing:
```bash
# Invoke from terminal
aws lambda invoke \
  --function-name nba-historical-game-and-props-results-fetcher \
  --payload '{}' \
  --region us-east-2 \
  response.json && cat response.json

# View logs in real-time
aws logs tail /aws/lambda/nba-historical-game-and-props-results-fetcher --follow --region us-east-2

# Or from Lambda console: Monitor → View CloudWatch logs
```

Example Timing Output (in logs and email):
```
============================================================
EXECUTION TIMING BREAKDOWN
============================================================
get_yesterday_et.......................... 0.12s (  0.1%)
get_secrets............................... 0.85s (  0.7%)
clone_repo................................ 22.45s ( 18.2%)
setup_python_path......................... 1.23s (  1.0%)
fetch_props_and_games..................... 95.67s ( 77.5%)
send_sns.................................. 0.34s (  0.3%)
run_cmd................................... 2.67s (  2.2%)
------------------------------------------------------------
TOTAL..................................... 123.33s
============================================================
```

Example Real-Time Streaming Output (shows where script hangs):
```
⏱️  [fetch_props_and_games] Starting...
Running: python3 scripts/fetch_nba_player_props.py --date 2026-01-12 --fetch-games --season 2025-26 --s3
================================================================================
CONFIGURATION
================================================================================
API_KEY: ********xxxx
SEASON: 2025-26
...
================================================================================
FETCHING PROPS FOR 2026-01-12 (Sunday)
================================================================================
Fetching events for 2026-01-12 (timestamp: 2026-01-12T12:00:00Z)
API call successful - Cost: 1 credits, Remaining: 499,999
Found 12 events for 2026-01-12
Processing game 1/12: Brooklyn Nets @ Cleveland Cavaliers
  ✅ Found 450 player props
[... shows exactly where it hangs if there's an issue ...]
```

S3 Output (Both Props + Game Results):
```
s3://the-odds-api-mt/
└── nba/historical_player_props/
    └── 2025-26/
        ├── 2026-01-10.csv
        ├── 2026-01-11.csv
        └── 2026-01-12.csv

s3://nba-api-mt/
└── player_game_logs/
    └── 2025-26/
        ├── 2026-01-10.csv
        ├── 2026-01-11.csv
        └── 2026-01-12.csv

Note: The --fetch-games flag fetches BOTH props (Odds API) AND game results (NBA API).
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


def lambda_handler(event, context):
    """Main handler - runs daily at 9am ET to fetch yesterday's NBA props + game results."""
    lambda_start = time.time()
    print(f"🏀 NBA Props + Game Results Fetcher - {context.aws_request_id}")
    
    yesterday = get_yesterday_et()
    print(f"📅 Yesterday: {yesterday}\n")
    
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
        from season_utils import get_current_nba_season
        season = get_current_nba_season()
        setup_elapsed = time.time() - setup_start
        TIMING_DATA['setup_python_path'] = setup_elapsed
        print(f"✅ [setup_python_path] Completed in {setup_elapsed:.2f}s")
        print(f"🏀 Season: {season}\n")
        
        # Run: python3 scripts/fetch_nba_player_props.py --date YESTERDAY --fetch-games --s3 --season SEASON
        # Note: --fetch-games fetches BOTH props (Odds API) AND game results (NBA API)
        print(f"📥 Fetching props + games for {yesterday}...")
        cmd = ['python3', 'scripts/fetch_nba_player_props.py', '--date', yesterday, '--fetch-games', '--season', season, '--s3']
        
        # This run_cmd is the BIG one - it contains the entire fetch script
        # Use stream_output=True to see real-time progress and identify where it hangs
        fetch_start = time.time()
        print(f"⏱️  [fetch_props_and_games] Starting...")
        _, stderr, code = run_cmd(cmd, cwd=repo_dir, extra_env={'ODDS_API_KEY': odds_api_key}, stream_output=True)
        fetch_elapsed = time.time() - fetch_start
        TIMING_DATA['fetch_props_and_games'] = fetch_elapsed
        print(f"✅ [fetch_props_and_games] Completed in {fetch_elapsed:.2f}s\n")
        
        if code != 0:
            # Print timing summary before failure
            timing_summary = format_timing_summary()
            print(f"\n{timing_summary}")
            
            error_msg = f"Date: {yesterday}\nSeason: {season}\n\n{timing_summary}\n\nError:\n{stderr}"
            send_sns(f"❌ Failed - {yesterday}", error_msg)
            return {'statusCode': 500, 'body': json.dumps({'error': stderr, 'date': yesterday})}
        
        # Success - print timing summary
        total_elapsed = time.time() - lambda_start
        TIMING_DATA['total_lambda_execution'] = total_elapsed
        
        timing_summary = format_timing_summary()
        print(f"\n{timing_summary}")
        
        # Send success notification with timing
        success_msg = (
            f"Date: {yesterday}\n"
            f"Season: {season}\n\n"
            f"{timing_summary}\n\n"
            f"Props: s3://the-odds-api-mt/nba/historical_player_props/{season}/{yesterday}.csv\n"
            f"Games: s3://nba-api-mt/player_game_logs/{season}/{yesterday}.csv"
        )
        send_sns(f"✅ Fetched - {yesterday}", success_msg)
        
        return {
            'statusCode': 200, 
            'body': json.dumps({
                'date': yesterday, 
                'season': season,
                'timing': TIMING_DATA
            })
        }
        
    except Exception as e:
        # Print timing summary on exception
        total_elapsed = time.time() - lambda_start
        TIMING_DATA['total_lambda_execution'] = total_elapsed
        timing_summary = format_timing_summary()
        print(f"\n{timing_summary}")
        
        error_msg = f"Date: {yesterday}\n\n{timing_summary}\n\nError: {str(e)}"
        send_sns("❌ Critical Error", error_msg)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

