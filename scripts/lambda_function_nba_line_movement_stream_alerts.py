"""
AWS Lambda Function - NBA Line Movement Steam Detector

Lambda function name: nba-line-movement-steam-alerts

What it does:
1. Git clone repo
2. Run line movement analysis for today's NBA games
3. Check if any game's consensus line has steamed 2+ points toward the opening underdog
4. Send SNS email alert ONLY if steam detected (otherwise just logs "no email")

Timing:
- Runs ~5 minutes AFTER the line movement tracking script completes
- Line movement script runs hourly to collect/store current lines
- This Lambda analyzes those stored lines and sends alerts when needed

Example Alert Scenario:
- Opening line: LAL -3.5 vs DEN
- Current line: LAL -1.5 vs DEN
- Steam: 2.0 points toward opening underdog DEN
- Alert: "🚨 Line Steam Alert: DEN +3.5 → +1.5 (2.0 pt steam toward underdog)"

Why This Matters:
Sharp money often causes lines to steam toward underdogs. When the consensus line
moves 2+ points away from favorites toward underdogs, it suggests sharp bettors
are loading up on the dog. This is a historically profitable betting signal.

Note: Only sends email when steam is detected. No email = no actionable moves found.

================================================================================
AWS SETUP INSTRUCTIONS
================================================================================

Creating Lambda Function:
- Name: nba-line-movement-steam-alerts
- Runtime: Python 3.12
- Architecture: x86_64
- Execution role: Use existing -> betting-dashboard-daily-update-role-ille2llh
- Click 'Create function'

Configuration:
- General configuration:
    - Memory: 2048 MB (faster git clone and pandas operations)
    - Ephemeral storage: 2048 MB
    - Timeout: 5 minutes (300 seconds)
    - Execution role: betting-dashboard-daily-update-role-ille2llh

- Environment variables -> Edit -> Add:
    - GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
    - GITHUB_USERNAME: MylesThomas
    - GITHUB_EMAIL: mylescgthomas@gmail.com
    - SECRET_NAME: betting-dashboard-secrets
    - AWS_REGION_NAME: us-east-2
    - SNS_TOPIC_ARN: arn:aws:sns:us-east-2:232692785472:nba-props-alerts
    
- Layers:
    - Add layer -> Specify ARN -> arn:aws:lambda:us-east-2:553035198032:layer:git-lambda2:8
    - Add layer -> Custom layer -> nba-props-fetcher-dependencies (version 1)
      (Contains: pandas, numpy, requests, pyyaml, python-dotenv, nba-api)

- Permissions (IAM Role: betting-dashboard-daily-update-role-ille2llh):
    - AmazonS3FullAccess (for s3://the-odds-api-mt bucket)
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

EventBridge Schedule (Run ~5 minutes AFTER line movement tracking script):
- Navigate to: AWS Console → Amazon EventBridge → Rules → Create rule
- Define rule detail:
    - Name: nba-line-movement-steam-alerts
    - Description: Check for 2+ point opening underdog line steam ~5 min after line tracking script runs
- Define schedule:
    - IMPORTANT: This Lambda runs 5 minutes AFTER your line movement tracking Lambda
    - If line tracking runs at :00 (top of hour), set this to :05 past the hour
    - Example (Hourly, 5 min offset): cron(5 12-23 * * ? *)
      (12:05 PM - 11:05 PM UTC = 7:05 AM - 6:05 PM ET)
    - Example (Every 2 hours): cron(5 12-23/2 * * ? *)
      (Runs at 12:05, 14:05, 16:05, 18:05, 20:05, 22:05 UTC)
- Select target:
    - Target type: AWS Lambda function
    - Function: nba-line-movement-steam-alerts
- Review + Create

CRITICAL TIMING NOTES:
- This Lambda must run AFTER the line movement tracking script finishes
- Line tracking script collects and stores current lines in S3
- This Lambda reads those stored lines and checks for steam
- 5-minute delay ensures line data is fully written to S3
- Lines typically move most between 10am-6pm ET (before games start at 7pm)
- Opening lines published ~24 hours (or more) before game (morning of game day)
- Sharps bet early (morning) and late (1-2 hours before tip)
- No email sent = no significant steam detected (this is normal!)

Testing:
```bash
# Invoke from terminal
aws lambda invoke \
  --function-name nba-line-movement-steam-alerts \
  --payload '{}' \
  --region us-east-2 \
  response.json && cat response.json

# View logs in real-time
aws logs tail /aws/lambda/nba-line-movement-steam-alerts --follow --region us-east-2

# Or from Lambda console: Monitor → View CloudWatch logs
```

Example Timing Output (in logs and email):
```
============================================================
EXECUTION TIMING BREAKDOWN
============================================================
get_today_et................................. 0.08s (  0.1%)
get_secrets.................................. 0.92s (  0.8%)
clone_repo................................... 19.34s ( 18.5%)
setup_python_path............................ 0.67s (  0.6%)
run_line_movement_analysis................... 82.15s ( 78.4%)
send_sns..................................... 0.28s (  0.3%)
run_cmd...................................... 1.98s (  1.9%)
------------------------------------------------------------
TOTAL........................................ 104.42s
============================================================
```

Example Alert Output (when steam detected):
```
🚨 NBA LINE STEAM ALERT - 2026-01-13

Significant line movement detected (2+ points toward opening underdog):

Game 1: Denver Nuggets @ Los Angeles Lakers
Opening: DEN +3.5 @ LAL -3.5
Current: DEN +1.5 @ LAL -1.5
Steam: 2.0 points toward opening underdog DEN
Time: 2026-01-13 14:23:15 ET
Books moving: FanDuel, DraftKings, BetMGM

Game 2: Miami Heat @ Boston Celtics
Opening: MIA +7.0 @ BOS -7.0
Current: MIA +4.5 @ BOS -4.5
Steam: 2.5 points toward opening underdog MIA
Time: 2026-01-13 14:23:15 ET
Books moving: Caesars, DraftKings

Recommendation: Consider betting the steamed underdogs (sharp money indicator)

Timing:
- Total execution: 104.42s
- Analysis time: 82.15s
```

Example Silent Output (no steam detected - NO EMAIL SENT):
```
✅ NBA Line Movement Check - 2026-01-13

No significant line steam detected.
Checked 8 games - largest movement was 1.5 points.

Result: No email sent (no actionable steam)
Timing: 98.67s total
```

Analysis Script Output Location:
The Lambda will run a script (likely analyze_line_movement.py or similar) that:
1. Fetches current lines from S3 or Odds API
2. Compares to opening lines
3. Calculates movement toward underdogs
4. Returns DataFrame with games meeting 2+ point threshold

IMPORTANT - Handler Configuration:
After uploading this file to Lambda, you MUST update the handler configuration:
- Make sure in the Lambda console UI, your code is in `lambda_function.py` (does not need to match this file name)
- Go to: Configuration → Runtime settings → Edit
- Set Handler to: lambda_function_nba_line_movement_stream_alerts.lambda_handler
    - The handler format is: {filename_without_py}.{function_name}
- Without this, Lambda will look for 'index.lambda_handler' and fail with ImportModuleError

Author: Myles Thomas
Date: 2026-01-13
"""

import json
import os
import sys
import subprocess
import boto3
import time
from datetime import datetime
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
        'PYTHONPATH': '/opt/python'  # Lambda layer packages location
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
    """Main handler - checks for 2+ point line steam toward opening underdogs."""
    lambda_start = time.time()
    print(f"🏀 NBA Line Movement Steam Detector - {context.aws_request_id}")
    
    today = get_today_et()
    print(f"📅 Today: {today}\n")
    
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
        
        # Run line movement analysis script
        # This checks for 2+ point steam toward opening underdog on TODAY'S games (before they play)
        # Only loads last 3 days of snapshots for performance (no need for old data)
        # --log-individual-games shows detailed breakdown of ALL games today (sorted by start time)
        print(f"📊 Analyzing line movements for {today}...")
        cmd = ['python3', 'scripts/check_nba_line_steam.py', '--date', today, '--threshold', '1.0', '--days-back', '3', '--log-individual-games']
        
        # This run_cmd contains the entire analysis
        # Use stream_output=True to see real-time progress
        analysis_start = time.time()
        print(f"⏱️  [run_line_movement_analysis] Starting...")
        stdout, stderr, code = run_cmd(cmd, cwd=repo_dir, extra_env={'ODDS_API_KEY': odds_api_key}, stream_output=True)
        analysis_elapsed = time.time() - analysis_start
        TIMING_DATA['run_line_movement_analysis'] = analysis_elapsed
        print(f"✅ [run_line_movement_analysis] Completed in {analysis_elapsed:.2f}s\n")
        
        if code != 0:
            # Print timing summary before failure
            timing_summary = format_timing_summary()
            print(f"\n{timing_summary}")
            
            error_msg = f"Date: {today}\nSeason: {season}\n\n{timing_summary}\n\nError:\n{stderr}"
            send_sns(f"❌ Line Steam Check Failed - {today}", error_msg)
            return {'statusCode': 500, 'body': json.dumps({'error': stderr, 'date': today})}
        
        # Parse output to check if steam was detected
        # The analysis script should output "STEAM_DETECTED: YES" or "STEAM_DETECTED: NO" in stdout
        steam_detected = "STEAM_DETECTED: YES" in stdout
        
        # Success - print timing summary
        total_elapsed = time.time() - lambda_start
        TIMING_DATA['total_lambda_execution'] = total_elapsed
        
        timing_summary = format_timing_summary()
        print(f"\n{timing_summary}")
        
        if steam_detected:
            # Send alert with details
            alert_msg = (
                f"🚨 NBA LINE STEAM ALERT - {today}\n\n"
                f"Significant line movement detected (2+ points toward opening underdog):\n\n"
                f"{stdout}\n\n"
                f"{timing_summary}"
            )
            send_sns(f"🚨 Line Steam Alert - {today}", alert_msg)
            print("🚨 STEAM DETECTED - Email alert sent!")
        else:
            # Silent success - no email sent
            print("✅ No significant steam detected - No email sent")
            print(f"Result: Silent run (no actionable steam found)")
        
        return {
            'statusCode': 200, 
            'body': json.dumps({
                'date': today, 
                'season': season,
                'steam_detected': steam_detected,
                'timing': TIMING_DATA
            })
        }
        
    except Exception as e:
        # Print timing summary on exception
        total_elapsed = time.time() - lambda_start
        TIMING_DATA['total_lambda_execution'] = total_elapsed
        timing_summary = format_timing_summary()
        print(f"\n{timing_summary}")
        
        error_msg = f"Date: {today}\n\n{timing_summary}\n\nError: {str(e)}"
        send_sns("❌ Critical Error - Line Steam Check", error_msg)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

