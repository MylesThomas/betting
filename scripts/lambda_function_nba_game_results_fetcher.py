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
    - Memory: 1024 MB
    - Ephemeral storage: 2048 MB
    - Timeout: 1 minute 30 seconds (90 seconds)
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

EventBridge Schedule (Run at 9:00 AM ET daily):
- Navigate to: AWS Console → Amazon EventBridge → Rules → Create rule
- Define rule detail:
    - Name: nba-historical-game-and-props-results-fetcher
    - Description: Fetch yesterday's NBA game + props results at 9am ET daily
- Define schedule:
    - Schedule expression: cron(0 14 * * ? *)
    - (14:00 UTC = 9:00 AM ET)
- Select target:
    - Target type: AWS Lambda function
    - Function: nba-historical-game-and-props-results-fetcher
- Review + Create

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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def get_yesterday_et():
    """Get yesterday's date in Eastern Time."""
    et_tz = ZoneInfo('America/New_York')
    yesterday = (datetime.now(et_tz) - timedelta(days=1)).strftime('%Y-%m-%d')
    return yesterday


def run_cmd(cmd, cwd=None, extra_env=None):
    """Run shell command and return (stdout, stderr, returncode)."""
    env = {
        **os.environ,
        'AWS_DEFAULT_REGION': os.environ['AWS_REGION_NAME'],
        'PYTHONPATH': '/opt/python'  # Lambda layer packages location
    }
    if extra_env:
        env.update(extra_env)
    
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


def clone_repo(token):
    """Clone betting repo to /tmp/betting."""
    target = '/tmp/betting'
    if os.path.exists(target):
        subprocess.run(['rm', '-rf', target])
    
    repo_url = os.environ['GITHUB_REPO_URL']
    auth_url = repo_url.replace('https://', f'https://{token}@')
    
    _, stderr, code = run_cmd(['git', 'clone', auth_url, target])
    if code != 0:
        raise Exception(f"Git clone failed: {stderr}")
    
    return target


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
    print(f"🏀 NBA Props + Game Results Fetcher - {context.aws_request_id}")
    yesterday = get_yesterday_et()
    print(f"📅 Yesterday: {yesterday}\n")
    
    try:
        # Get secrets (GITHUB_TOKEN + ODDS_API_KEY)
        sm = boto3.client('secretsmanager', region_name=os.environ['AWS_REGION_NAME'])
        secret = sm.get_secret_value(SecretId=os.environ['SECRET_NAME'])
        secrets = json.loads(secret['SecretString'])
        github_token = secrets['GITHUB_TOKEN']
        odds_api_key = secrets['ODDS_API_KEY']
        
        # Clone repo
        print("📦 Cloning...")
        repo_dir = clone_repo(github_token)
        
        # Add src/ to path NOW that repo exists
        sys.path.insert(0, os.path.join(repo_dir, 'src'))
        
        # Get current season from src/
        from season_utils import get_current_nba_season
        season = get_current_nba_season()
        print(f"🏀 Season: {season}\n")
        
        # Run: python3 scripts/fetch_nba_player_props.py --date YESTERDAY --fetch-games --s3 --season SEASON
        # Note: --fetch-games fetches BOTH props (Odds API) AND game results (NBA API)
        print(f"📥 Fetching props + games for {yesterday}...")
        cmd = ['python3', 'scripts/fetch_nba_player_props.py', '--date', yesterday, '--fetch-games', '--season', season, '--s3']
        _, stderr, code = run_cmd(cmd, cwd=repo_dir, extra_env={'ODDS_API_KEY': odds_api_key})
        
        if code != 0:
            send_sns(f"❌ Failed - {yesterday}", f"Date: {yesterday}\nSeason: {season}\n\nError:\n{stderr}")
            return {'statusCode': 500, 'body': json.dumps({'error': stderr, 'date': yesterday})}
        
        # Success
        send_sns(
            f"✅ Fetched - {yesterday}", 
            f"Date: {yesterday}\nSeason: {season}\n\nProps: s3://the-odds-api-mt/nba/historical_player_props/{season}/{yesterday}.csv\nGames: s3://nba-api-mt/player_game_logs/{season}/{yesterday}.csv"
        )
        return {'statusCode': 200, 'body': json.dumps({'date': yesterday, 'season': season})}
        
    except Exception as e:
        send_sns("❌ Critical Error", f"Date: {yesterday}\nError: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

