"""
AWS Lambda Function - Daily NBA Props Workflow

Lambda function name: nba-player-scoring-props-daily-workflow

This Lambda function:
1. Fetches secrets from AWS Secrets Manager
2. Clones the GitHub repository
3. Runs the daily NBA props workflow:
   - Finds today's 2D plays (tier × spread)
   - Finds today's 3D plays (tier × spread × scorer_type)
   - Tracks yesterday's performance (BOTH 2D and 3D)
   - Generates daily email (BOTH 2D and 3D)
4. Sends email via SNS

IMPORTANT: Python dependencies are provided via Lambda Layer.

Environment Variables Required:
- GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
- GITHUB_USERNAME: MylesThomas
- GITHUB_EMAIL: mylescgthomas@gmail.com
- SECRET_NAME: betting-dashboard-secrets
- AWS_REGION_NAME: us-east-2
- SNS_TOPIC_ARN: arn:aws:sns:us-east-2:232692785472:nba-props-alerts

Secrets Required (in AWS Secrets Manager - betting-dashboard-secrets in us-east-2):
- ODDS_API_KEY: Your Odds API key
- GITHUB_TOKEN: Your GitHub Personal Access Token

IAM Permissions Required:
- secretsmanager:GetSecretValue
- s3:GetObject, s3:PutObject (for nba-betting-mt bucket)
- sns:Publish

Lambda Configuration:
- Runtime: Python 3.12
- Memory: 1024 MB (need more for data processing)
- Timeout: 15 minutes (900 seconds)
- Ephemeral storage: 2048 MB (for git clone)

Lambda Layers Required:
- use existing layer: 

Author: Myles Thomas
Date: 2026-01-06
"""

import json
import os
import subprocess
import boto3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from botocore.exceptions import ClientError


def send_email_notification(subject, message, topic_arn=None):
    """
    Send email notification via AWS SNS.
    
    Args:
        subject: Email subject
        message: Email body
        topic_arn: SNS topic ARN (optional, defaults to env var)
    """
    if topic_arn is None:
        topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    if not topic_arn:
        print("⚠️  No SNS_TOPIC_ARN configured - skipping email notification")
        return
    
    try:
        sns_client = boto3.client('sns', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-2'))
        response = sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )
        print(f"✅ Email notification sent (MessageId: {response['MessageId']})")
        return True
    except Exception as e:
        print(f"⚠️  Failed to send email notification: {e}")
        return False


def get_secrets():
    """
    Fetch secrets from AWS Secrets Manager.
    
    Returns:
        dict: Contains ODDS_API_KEY and GITHUB_TOKEN
    """
    secret_name = os.environ['SECRET_NAME']
    region_name = os.environ['AWS_REGION_NAME']
    
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise Exception(f"Error retrieving secret: {e}")
    
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret


def run_command(command, cwd=None, env=None):
    """
    Run a shell command and capture output.
    
    Args:
        command: Command string or list
        cwd: Working directory
        env: Environment variables dict
    
    Returns:
        tuple: (stdout, stderr, returncode)
    """
    print(f"🔧 Running: {command}")
    
    # Merge current env with custom env
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    result = subprocess.run(
        command,
        shell=isinstance(command, str),
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    
    return result.stdout, result.stderr, result.returncode


def clone_repository(repo_url, github_token, target_dir='/tmp/betting'):
    """
    Clone the GitHub repository to /tmp.
    
    Args:
        repo_url: GitHub repo URL
        github_token: GitHub personal access token
        target_dir: Where to clone (Lambda /tmp has 512MB-10GB space)
    
    Returns:
        str: Path to cloned repo
    """
    print(f"\n📦 Cloning repository...")
    
    # Clean up if exists
    if os.path.exists(target_dir):
        print(f"   Removing existing directory: {target_dir}")
        subprocess.run(['rm', '-rf', target_dir])
    
    # Insert token into URL for authentication
    if github_token:
        # Convert https://github.com/user/repo.git to https://token@github.com/user/repo.git
        auth_url = repo_url.replace('https://', f'https://{github_token}@')
    else:
        auth_url = repo_url
    
    # Clone
    stdout, stderr, returncode = run_command(
        ['git', 'clone', auth_url, target_dir]
    )
    
    if returncode != 0:
        raise Exception(f"Failed to clone repository: {stderr}")
    
    print(f"✅ Repository cloned to {target_dir}")
    return target_dir


def run_daily_workflow(repo_dir, odds_api_key, season='2025-26'):
    """
    Run the daily NBA props workflow.
    
    Steps:
    1. Find today's 2D plays (tier × spread)
    2. Find today's 3D plays (tier × spread × scorer_type)
    3. Track yesterday's performance (BOTH 2D + 3D)
    4. Generate daily email (BOTH 2D + 3D)
    
    Args:
        repo_dir: Path to cloned repository
        odds_api_key: The Odds API key
        season: NBA season
    
    Returns:
        dict: Workflow results
    """
    print(f"\n{'='*80}")
    print("🏀 Starting Daily NBA Props Workflow")
    print(f"{'='*80}")
    
    # Get dates
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    today = now_et.strftime('%Y-%m-%d')
    yesterday = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"📅 Today (ET): {today}")
    print(f"📅 Yesterday (ET): {yesterday}")
    print(f"🕐 Current time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    
    results = {
        'today': today,
        'yesterday': yesterday,
        'steps': {}
    }
    
    # Environment variables for scripts
    env = {
        'ODDS_API_KEY': odds_api_key,
        'AWS_DEFAULT_REGION': os.environ.get('AWS_REGION_NAME', 'us-east-2')
    }
    
    # Step 1: Find today's 2D plays
    print(f"\n{'='*80}")
    print("Step 1: Finding Today's 2D Plays (tier × spread)")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'scripts/find_role_spread_points_model_plays.py',
        '--season', season,
        '--min-roi', '5.0',
        '--granularity', 'detailed',
        '--save-s3'
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['steps']['2d_plays'] = {
        'success': returncode == 0,
        'output': stdout
    }
    
    if returncode != 0:
        print(f"⚠️  2D play finder failed (non-fatal, continuing...)")
    
    # Step 2: Find today's 3D plays
    print(f"\n{'='*80}")
    print("Step 2: Finding Today's 3D Plays (tier × spread × scorer_type)")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'scripts/find_role_spread_scorer_points_model_plays.py',
        '--season', season,
        '--min-roi', '5.0',
        '--granularity', 'detailed',
        '--rim-scorer-pct', '40',
        '--save-s3'
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['steps']['3d_plays'] = {
        'success': returncode == 0,
        'output': stdout
    }
    
    if returncode != 0:
        print(f"⚠️  3D play finder failed (non-fatal, continuing...)")
    
    # Step 3: Track yesterday's performance (BOTH 2D and 3D)
    print(f"\n{'='*80}")
    print(f"Step 3: Tracking Yesterday's Performance ({yesterday}) - BOTH 2D + 3D")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'scripts/track_daily_plays_performance.py',
        '--date', yesterday,
        '--season', season,
        '--strategy', 'both'  # Track both 2D and 3D
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['steps']['tracking'] = {
        'success': returncode == 0,
        'output': stdout
    }
    
    if returncode != 0:
        print(f"⚠️  Performance tracking failed (non-fatal, continuing...)")
    
    # Step 4: Generate daily email (BOTH 2D and 3D) and send via SNS
    print(f"\n{'='*80}")
    print("Step 4: Generating Daily Email - BOTH 2D + 3D")
    print(f"{'='*80}\n")
    
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    cmd = [
        'python', 'scripts/generate_role_spread_points_model_daily_email.py',
        '--season', season,
        '--plays-date', today,
        '--results-date', yesterday,
        '--strategy', 'both',  # Include both strategies in email
        '--sns-topic', sns_topic_arn  # Send via SNS directly
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['steps']['email'] = {
        'success': returncode == 0,
        'output': stdout,
        'email_sent': returncode == 0  # Email sent if script succeeded
    }
    
    if returncode != 0:
        print(f"⚠️  Email generation/sending failed")
    
    return results


def lambda_handler(event, context):
    """
    Main Lambda handler function.
    
    Args:
        event: Lambda event (can be from EventBridge, API Gateway, etc.)
        context: Lambda context
    
    Returns:
        dict: Response with status and details
    """
    print(f"\n{'='*80}")
    print("🚀 NBA Props Daily Workflow - Lambda Execution Started")
    print(f"{'='*80}")
    print(f"Event: {json.dumps(event, indent=2)}")
    print(f"Request ID: {context.aws_request_id}")
    print(f"Log Group: {context.log_group_name}")
    print(f"Log Stream: {context.log_stream_name}")
    print(f"Remaining time: {context.get_remaining_time_in_millis()}ms\n")
    
    try:
        # Get configuration from environment
        repo_url = os.environ['GITHUB_REPO_URL']
        season = os.environ.get('SEASON', '2025-26')
        
        # Get secrets
        print("🔐 Fetching secrets...")
        secrets = get_secrets()
        github_token = secrets.get('GITHUB_TOKEN')
        odds_api_key = secrets.get('ODDS_API_KEY')
        
        if not odds_api_key:
            raise Exception("ODDS_API_KEY not found in secrets")
        
        print("✅ Secrets retrieved successfully\n")
        
        # Clone repository
        repo_dir = clone_repository(repo_url, github_token)
        
        # Run workflow
        workflow_results = run_daily_workflow(repo_dir, odds_api_key, season)
        
        # Email was sent directly by the workflow script via --sns-topic
        email_sent = workflow_results['steps']['email'].get('email_sent', False)
        
        # Summary
        print(f"\n{'='*80}")
        print("✅ Lambda Execution Complete")
        print(f"{'='*80}")
        print(f"Today: {workflow_results['today']}")
        print(f"Yesterday: {workflow_results['yesterday']}")
        print(f"2D Plays: {'✅' if workflow_results['steps']['2d_plays']['success'] else '❌'}")
        print(f"3D Plays: {'✅' if workflow_results['steps']['3d_plays']['success'] else '❌'}")
        print(f"Tracking: {'✅' if workflow_results['steps']['tracking']['success'] else '❌'}")
        print(f"Email+SNS: {'✅' if email_sent else '❌'}")
        print(f"{'='*80}\n")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Daily NBA props workflow completed successfully',
                'results': {
                    'today': workflow_results['today'],
                    'yesterday': workflow_results['yesterday'],
                    '2d_plays': workflow_results['steps']['2d_plays']['success'],
                    '3d_plays': workflow_results['steps']['3d_plays']['success'],
                    'tracking': workflow_results['steps']['tracking']['success'],
                    'email_sent_via_sns': email_sent
                }
            })
        }
        
    except Exception as e:
        error_msg = f"❌ Lambda execution failed: {str(e)}"
        print(error_msg)
        
        # Send error notification
        send_email_notification(
            subject="❌ NBA Props Workflow Failed",
            message=f"Error: {str(e)}\n\nCheck CloudWatch logs for details."
        )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Daily NBA props workflow failed',
                'error': str(e)
            })
        }

