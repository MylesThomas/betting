"""
AWS Lambda Function - NBA Line Movement Steam Detector + Daily YTD Reporter

Lambda function name: nba-line-movement-steam-alerts

What it does:
MODE 1 - STEAM DETECTION (Hourly, 7am-6pm ET):
1. Git clone repo
2. Run line movement analysis for today's NBA games
3. Check if any game's consensus line has steamed threshold+ points (both directions)
4. Save detected plays to S3: s3://nba-betting-mt/data/04_output/plays/line-steam/{date_ET}.csv
5. Send SNS email alert ONLY if steam detected (otherwise just logs "no email")

MODE 2 - DAILY REPORT (1:05pm ET):
1. Wait for game results fetcher (runs at 12:30pm ET) to finish
2. Calculate yesterday's results (match plays to outcomes)
3. Save results to S3: s3://nba-betting-mt/data/04_output/results/line-steam/{date_ET}.csv
4. Generate YTD report (W-L-T, ROI, breakdown by steam size)
5. Send YTD email with performance stats

Timing:
- Steam detection: Runs ~5 minutes AFTER hourly line movement tracking script (e.g., hourly at :05)
- Line movement script runs hourly to collect/store current lines to S3
- Daily report: Runs at 1:05pm ET (~35 min after game results fetcher completes at 12:30pm ET)
- This ensures game results are available before calculating yesterday's outcomes

S3 Data Flow:
1. Plays saved: s3://nba-betting-mt/data/04_output/plays/line-steam/{date_ET}.csv
   - Appends EVERY detection throughout the day (no deduplication)
   - Tracks steam evolution: same game can have multiple detections at different times
   - Example: DAL toward underdog at 9am (2.5pts), 11am (2.0pts), 1pm (3.0pts) = 3 rows
   - Threshold stored in CSV column (not in path)
   - status='pending' until daily report calculates results
   
2. Results saved: s3://nba-betting-mt/data/04_output/results/line-steam/{date_ET}.csv
   - Deduped at (game_id, steam_direction) level - keeps LARGEST steam magnitude
   - Only the strongest signal per game/direction used for YTD tracking
   - Includes: actual_margin, cover_margin, status ('won'/'lost'/'push')

Example Alert Scenario (Steam Detection):
- Underdog steam: LAL -3.5 → LAL -1.5 (2.0 pts toward opening underdog DEN)
- Favorite steam: BOS -7.5 → BOS -9.5 (2.0 pts toward opening favorite BOS)
- Both plays saved to S3 with status='pending'
- Email shows both sections (toward underdog / toward favorite)

Example Daily Report (1:05pm ET):
- Calculates yesterday's results
- Updates plays CSV with outcomes
- Sends YTD email: "12-8-1 (60.0%) | +$318 | +15.2% ROI"

Why This Matters:
Sharp money often causes significant line steam in both directions:
- Toward underdogs: Sharp bettors loading up on dogs (historically profitable)
- Toward favorites: Public money overreaction or sharp fade opportunities
Both signals tracked to compare performance and find optimal betting strategies.

Note: Only sends email when steam is detected (Mode 1) or when generating YTD report (Mode 2).

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
    - Description: Check for 1.0+ point line steam (both directions) ~5 min after line tracking script runs
- Define schedule:
    - IMPORTANT: Two modes - steam detection (hourly) + daily report (2:05pm ET)
    
    MODE 1 - Steam Detection (Hourly):
    - Runs 5 minutes AFTER your line movement tracking Lambda
    - If line tracking runs at :00 (top of hour), set this to :05 past the hour
    - Example (Hourly, 5 min offset): cron(5 12-23 * * ? *)
      (12:05 PM - 11:05 PM UTC = 7:05 AM - 6:05 PM ET)
    
    MODE 2 - Daily Report (1:05pm ET):
    - Automatically triggered by checking current time (13:00-13:10 ET)
    - No separate EventBridge rule needed - same schedule catches it
    - Runs at 1:05pm ET (35 min after game results fetcher at 12:30pm ET)
    - Ensures game results are available before calculating outcomes
    
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

Significant line movement detected (1.0+ points):

================================================================================
TOWARD OPENING UNDERDOG (2 games)
================================================================================

Game 1: Denver Nuggets vs Los Angeles Lakers
  Opening: DEN +3.5 | LAL -3.5
  Current: DEN +1.5 | LAL -1.5
  Steam: 2.0 points toward opening underdog DEN
  Game time: 07:30 PM ET
  Snapshots tracked: 8.5 hours

Game 2: Miami Heat vs Boston Celtics
  Opening: MIA +7.0 | BOS -7.0
  Current: MIA +4.5 | BOS -4.5
  Steam: 2.5 points toward opening underdog MIA
  Game time: 07:00 PM ET
  Snapshots tracked: 9.2 hours

================================================================================
TOWARD OPENING FAVORITE (1 game)
================================================================================

Game 1: Phoenix Suns vs Sacramento Kings
  Opening: PHX -4.5 | SAC +4.5
  Current: PHX -6.5 | SAC +6.5
  Steam: 2.0 points toward opening favorite PHX
  Game time: 10:00 PM ET
  Snapshots tracked: 7.1 hours

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
from io import BytesIO


# =============================================================================
# CONFIGURATION
# =============================================================================

# Steam threshold for detection (can be overridden via event payload)
STEAM_THRESHOLD = 1.0

# Days back to load snapshots (can be overridden via event payload)
DAYS_BACK = 7

# Note: All dates in S3 paths are in ET timezone (America/New_York)
# Ensures consistency with NBA game schedules

# Daily report time (1:05pm ET - runs 35 min after game results fetcher at 12:30pm ET)
DAILY_REPORT_HOUR_ET = 13  # 1pm ET
DAILY_REPORT_MINUTE_WINDOW_ET = 10  # Run if within first 10 minutes of hour

# S3 bucket for storing plays and results
S3_BUCKET_PLAYS = 'nba-betting-mt'

# S3 paths for plays and results (threshold stored in CSV, not in path)
# Plays (pending status, appends each detection): data/04_output/plays/line-steam/{date_ET}.csv
# Results (calculated outcomes): data/04_output/results/line-steam/{date_ET}.csv
S3_PLAYS_PREFIX = 'data/04_output/plays/line-steam'
S3_RESULTS_PREFIX = 'data/04_output/results/line-steam'


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


def save_plays_to_s3(steam_games_df, today, season, threshold):
    """
    Save detected steam plays to S3.
    Appends all detections (no deduplication) to track steam evolution over time.
    
    Strategy:
    - PLAYS: Append every detection (tracks all signals throughout day)
    - RESULTS: Dedupe at (game_id, steam_direction) keeping largest steam magnitude
    
    Same game can have multiple rows in plays file:
    - Different steam directions (toward dog in AM, toward fav in PM)
    - Same direction but different magnitudes (2.5pts at 9am, 2.0pts at 11am, 3.0pts at 1pm)
    
    Results calculation will dedupe and use strongest signal for YTD tracking.
    
    Args:
        steam_games_df: DataFrame with games that have steam
        today: Today's date string in ET timezone (YYYY-MM-DD)
        season: NBA season (e.g., '2025-26')
        threshold: Steam threshold used (stored in CSV, not in path)
    
    S3 Location: s3://nba-betting-mt/data/04_output/plays/line-steam/{date_ET}.csv
    """
    if steam_games_df is None or len(steam_games_df) == 0:
        print("No plays to save")
        return
    
    # S3 path: threshold stored in CSV column, not in folder structure
    s3_key = f"data/04_output/plays/line-steam/{today}.csv"
    s3 = boto3.client('s3')
    
    # Try to load existing plays for today
    existing_plays = None
    try:
        from io import BytesIO
        import pandas as pd
        
        response = s3.get_object(Bucket=S3_BUCKET_PLAYS, Key=s3_key)
        existing_plays = pd.read_csv(BytesIO(response['Body'].read()))
        print(f"📥 Loaded {len(existing_plays)} existing plays from S3")
    except s3.exceptions.NoSuchKey:
        print(f"📝 No existing plays file - creating new one")
    except Exception as e:
        print(f"⚠️  Error loading existing plays: {e}")
    
    # Format current detections as play records
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    et_tz = ZoneInfo('America/New_York')
    detected_at = datetime.now(et_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    plays = []
    for _, game in steam_games_df.iterrows():
        play = {
            'detected_at': detected_at,
            'game_id': game['game_id'],
            'game_date': game['game_time'].tz_convert(et_tz).strftime('%Y-%m-%d'),
            'game_time': game['game_time'].tz_convert(et_tz).strftime('%Y-%m-%d %H:%M:%S'),
            'season': season,
            'opening_favorite': game['opening_favorite'],
            'opening_underdog': game['opening_underdog'],
            'opening_favorite_spread': game['opening_favorite_spread_open'],
            'current_favorite_spread': game['opening_favorite_spread_current'],
            'steam_magnitude': game['steam_magnitude'],
            'steam_direction': 'opening_underdog' if game['steam_toward_opening_underdog'] else 'opening_favorite',
            'steamed_team': game['opening_underdog'] if game['steam_toward_opening_underdog'] else game['opening_favorite'],
            'threshold': threshold,
            'play_team': game['opening_underdog'] if game['steam_toward_opening_underdog'] else game['opening_favorite'],
            'play_spread': -game['opening_favorite_spread_current'] if game['steam_toward_opening_underdog'] else game['opening_favorite_spread_current'],
            'status': 'pending',
            'actual_margin': None,
            'cover_margin': None,
        }
        plays.append(play)
    
    import pandas as pd
    new_plays_df = pd.DataFrame(plays)
    
    # Append all detections (no deduplication - keep steam evolution over time)
    # Same game can have multiple detections as steam magnitude changes throughout day
    # Can dedupe later during analysis if needed
    if existing_plays is not None:
        combined_plays = pd.concat([existing_plays, new_plays_df], ignore_index=True)
        print(f"📊 Appended {len(new_plays_df)} new detections (total: {len(combined_plays)})")
    else:
        combined_plays = new_plays_df
        print(f"📊 Created {len(combined_plays)} new detections")
    
    # Save to S3
    from io import StringIO
    csv_buffer = StringIO()
    combined_plays.to_csv(csv_buffer, index=False)
    
    s3.put_object(
        Bucket=S3_BUCKET_PLAYS,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    
    print(f"✅ Saved plays to s3://{S3_BUCKET_PLAYS}/{s3_key}")


def generate_ytd_report(today, season, threshold):
    """
    Generate YTD report by reading all results from S3.
    Dedupes at (game_id, steam_direction) level, keeping largest steam magnitude.
    
    Args:
        today: Today's date string (ET timezone)
        season: NBA season
        threshold: Steam threshold to filter by (reads from 'threshold' column in CSV)
    
    Returns:
        Formatted report string with W-L-T, ROI, breakdown
        Shows separate stats for underdog steam, favorite steam, and combined
    """
    print(f"\n📊 Generating YTD report for threshold {threshold}...")
    
    s3 = boto3.client('s3')
    results_prefix = "data/04_output/results/line-steam/"
    
    # List all results files
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_PLAYS, Prefix=results_prefix)
    except Exception as e:
        return f"❌ Error loading results from S3: {e}"
    
    if 'Contents' not in response:
        return f"📊 No results found yet"
    
    # Load all results files
    all_results = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if not key.endswith('.csv'):
            continue
        
        try:
            result_obj = s3.get_object(Bucket=S3_BUCKET_PLAYS, Key=key)
            df = pd.read_csv(BytesIO(result_obj['Body'].read()))
            all_results.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_results:
        return f"📊 No results files found"
    
    # Combine all results
    import pandas as pd
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Filter to specified threshold
    results_df = results_df[results_df['threshold'] == threshold]
    
    if len(results_df) == 0:
        return f"📊 No results found for threshold {threshold}"
    
    # Dedupe: keep detection with largest steam magnitude per (game_id, steam_direction)
    # This ensures YTD stats are based on strongest signals only
    original_count = len(results_df)
    results_df = results_df.sort_values('steam_magnitude', ascending=False).drop_duplicates(
        subset=['game_id', 'steam_direction'],
        keep='first'  # Keep row with largest steam_magnitude
    )
    deduped_count = len(results_df)
    print(f"   Deduped results: {original_count} detections → {deduped_count} unique plays")
    
    # Filter to completed games only
    completed = results_df[results_df['status'] != 'pending']
    
    if len(completed) == 0:
        return f"📊 No completed games yet for threshold {threshold}"
    
    # Split by steam direction
    underdog_plays = completed[completed['steam_direction'] == 'opening_underdog']
    favorite_plays = completed[completed['steam_direction'] == 'opening_favorite']
    
    def calc_stats(df, label):
        """Calculate stats for a subset of plays."""
        if len(df) == 0:
            return None
        
        wins = (df['status'] == 'won').sum()
        losses = (df['status'] == 'lost').sum()
        pushes = (df['status'] == 'push').sum()
        total = len(df)
        win_pct = wins / total * 100 if total > 0 else 0
        
        profit_from_wins = wins * 100
        loss_from_losses = losses * 110
        net_profit = profit_from_wins - loss_from_losses
        total_risked = total * 110
        roi_pct = (net_profit / total_risked * 100) if total_risked > 0 else 0
        
        avg_cover_margin = df['cover_margin'].mean()
        avg_steam = df['steam_magnitude'].mean()
        
        return {
            'label': label,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'total': total,
            'win_pct': win_pct,
            'net_profit': net_profit,
            'roi_pct': roi_pct,
            'avg_cover_margin': avg_cover_margin,
            'avg_steam': avg_steam,
            'df': df
        }
    
    # Calculate stats for each subset
    underdog_stats = calc_stats(underdog_plays, 'UNDERDOG STEAM')
    favorite_stats = calc_stats(favorite_plays, 'FAVORITE STEAM')
    combined_stats = calc_stats(completed, 'COMBINED')
    
    # Best/worst plays (overall)
    best_play = completed.loc[completed['cover_margin'].idxmax()]
    worst_play = completed.loc[completed['cover_margin'].idxmin()]
    
    # Format report
    report = f"""
📊 NBA LINE STEAM YTD REPORT - {today}
{'='*80}

SEASON: {season}
THRESHOLD: {threshold}+ points

"""
    
    # Underdog section
    if underdog_stats:
        s = underdog_stats
        report += f"""{'─'*80}
{s['label']} (Sharp Money Indicator):
  Record: {s['wins']}-{s['losses']}-{s['pushes']} ({s['win_pct']:.1f}%)
  ROI: {s['roi_pct']:+.1f}%
  Net Profit: ${s['net_profit']:+,.0f}
  Avg Steam: {s['avg_steam']:.2f} pts
  Avg Cover Margin: {s['avg_cover_margin']:+.2f} pts
  Total Plays: {s['total']}

"""
    
    # Favorite section
    if favorite_stats:
        s = favorite_stats
        report += f"""{'─'*80}
{s['label']} (Fade the Public):
  Record: {s['wins']}-{s['losses']}-{s['pushes']} ({s['win_pct']:.1f}%)
  ROI: {s['roi_pct']:+.1f}%
  Net Profit: ${s['net_profit']:+,.0f}
  Avg Steam: {s['avg_steam']:.2f} pts
  Avg Cover Margin: {s['avg_cover_margin']:+.2f} pts
  Total Plays: {s['total']}

"""
    
    # Combined section
    if combined_stats:
        s = combined_stats
        report += f"""{'─'*80}
{s['label']} (All Steam Plays):
  Record: {s['wins']}-{s['losses']}-{s['pushes']} ({s['win_pct']:.1f}%)
  ROI: {s['roi_pct']:+.1f}%
  Net Profit: ${s['net_profit']:+,.0f}
  Units: {s['net_profit']/110:+.2f} units (flat $110 bets)
  Avg Steam: {s['avg_steam']:.2f} pts
  Avg Cover Margin: {s['avg_cover_margin']:+.2f} pts
  Total Plays: {s['total']}

"""
    
    # Best/worst plays
    report += f"""{'─'*80}
BEST PLAY:
  {best_play['play_team']} {best_play['play_spread']:+.1f} on {best_play['game_date']}
  Result: {best_play['status'].upper()} by {best_play['cover_margin']:+.1f} pts
  Steam: {best_play['steam_magnitude']:.1f} pts toward {best_play['steamed_team']}

WORST PLAY:
  {worst_play['play_team']} {worst_play['play_spread']:+.1f} on {worst_play['game_date']}
  Result: {worst_play['status'].upper()} by {worst_play['cover_margin']:+.1f} pts
  Steam: {worst_play['steam_magnitude']:.1f} pts toward {worst_play['steamed_team']}

{'─'*80}
BREAKDOWN BY STEAM SIZE (Combined):
"""
    
    # Breakdown by steam size (combined)
    for min_steam, max_steam in [(1.0, 1.9), (2.0, 2.9), (3.0, 3.9), (4.0, 10.0)]:
        subset = completed[(completed['steam_magnitude'] >= min_steam) & (completed['steam_magnitude'] < max_steam)]
        if len(subset) > 0:
            sub_wins = (subset['status'] == 'won').sum()
            sub_losses = (subset['status'] == 'lost').sum()
            sub_pushes = (subset['status'] == 'push').sum()
            sub_pct = sub_wins / len(subset) * 100
            report += f"  {min_steam:.1f}-{max_steam:.1f} pts: {sub_wins}-{sub_losses}-{sub_pushes} ({sub_pct:.1f}%) | N={len(subset)}\n"
    
    report += f"\n{'='*80}"
    
    return report


def lambda_handler(event, context):
    """
    Main handler - checks for line steam OR generates YTD report.
    
    Modes:
    1. Steam detection (default): Runs hourly to detect steam and send alerts
    2. Daily report (1:00-1:10pm ET or daily_report=True): Calculate results + send YTD report
    """
    lambda_start = time.time()
    print(f"🏀 NBA Line Movement Steam Detector - {context.aws_request_id}")
    
    today = get_today_et()
    print(f"📅 Today: {today}\n")
    
    # Check if we should run daily report
    et_tz = ZoneInfo('America/New_York')
    current_time_et = datetime.now(et_tz)
    hour = current_time_et.hour
    minute = current_time_et.minute
    
    run_daily_report = (
        (hour == DAILY_REPORT_HOUR_ET and minute <= DAILY_REPORT_MINUTE_WINDOW_ET) or  # 1:00-1:10pm ET
        event.get('daily_report', False)   # Manual trigger
    )
    
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
        
        # Get configuration from event or use defaults
        threshold = event.get('threshold', STEAM_THRESHOLD)
        days_back = event.get('days_back', DAYS_BACK)
        print(f"🔍 Threshold: {threshold}")
        print(f"📅 Days back: {days_back}")
        
        # =============================================================================
        # DAILY REPORT MODE - Calculate yesterday's results + send YTD report
        # =============================================================================
        if run_daily_report:
            print(f"\n📊 Running in DAILY REPORT mode (triggered at {current_time_et.strftime('%I:%M %p ET')})")
            
            # Calculate yesterday's date
            from datetime import timedelta
            yesterday = (datetime.now(et_tz) - timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"📅 Yesterday: {yesterday}")
            
            # Run results calculation for yesterday (processes all thresholds)
            print(f"\n📥 Calculating results for {yesterday}...")
            calc_cmd = [
                'python3', 'scripts/calculate_line_steam_results.py',
                '--date', yesterday,
                '--season', season
                # Note: No --threshold flag - processes all thresholds in file
                # YTD report will filter by threshold when generating stats
            ]
            
            calc_start = time.time()
            stdout, stderr, code = run_cmd(calc_cmd, cwd=repo_dir, stream_output=True)
            calc_elapsed = time.time() - calc_start
            TIMING_DATA['calculate_results'] = calc_elapsed
            
            if code != 0:
                print(f"⚠️  Results calculation had issues: {stderr}")
            
            # Generate YTD report
            import pandas as pd
            ytd_report = generate_ytd_report(today, season, threshold)
            
            # Send YTD report email
            timing_summary = format_timing_summary()
            report_msg = f"{ytd_report}\n\n{timing_summary}"
            send_sns(f"📊 NBA Line Steam YTD Report - {today}", report_msg)
            
            print("✅ Daily report sent!")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'mode': 'daily_report',
                    'date': today,
                    'yesterday': yesterday,
                    'season': season,
                    'timing': TIMING_DATA
                })
            }
        
        # =============================================================================
        # STEAM DETECTION MODE (default) - Check for steam + send alerts
        # =============================================================================
        
        # Run line movement analysis script
        # This checks for threshold+ point steam (both directions) on TODAY'S games (before they play)
        # Only loads last X days of snapshots for performance (no need for old data)
        # --log-individual-games shows detailed breakdown of ALL games today (sorted by start time)
        # --save-plays saves detected steam to S3 (s3://nba-betting-mt/data/04_output/plays/line-steam/)
        print(f"📊 Analyzing line movements for {today}...")
        cmd = [
            'python3', 'scripts/check_nba_line_steam.py', 
            '--date', today, 
            '--threshold', str(threshold), 
            '--days-back', str(days_back), 
            '--log-individual-games',
            '--save-plays',  # Save plays to S3 when steam detected
            '--season', season
        ]
        
        print(f"\n🔧 Command to run:")
        print(f"   {' '.join(cmd)}\n")
        
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
                f"Significant line movement detected ({threshold}+ points):\n\n"
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

