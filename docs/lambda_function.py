"""
AWS Lambda Function - Daily Dashboard Update

This Lambda function:
1. Fetches secrets from AWS Secrets Manager
2. Clones the GitHub repository
3. Runs the daily update script (find_arb_opportunities.py)
4. Pushes changes back to GitHub
5. Streamlit Cloud auto-deploys the changes

IMPORTANT: Python dependencies (pandas, requests, nba_api, etc.) are provided via
a Lambda Layer, NOT installed at runtime. The layer contains all required packages
pre-built for Linux x86_64. Do NOT add a "pip install" step here - it's unnecessary
and will fail since there's no requirements.txt in the repo root.

Environment Variables Required:
- GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
- GITHUB_USERNAME: MylesThomas
- GITHUB_EMAIL: mylescgthomas@gmail.com
- SECRET_NAME: betting-dashboard-secrets
- AWS_REGION_NAME: us-east-2
- SNS_TOPIC_ARN: arn:aws:sns:us-east-2:ACCOUNT_ID:betting-dashboard-alerts (optional)

Secrets Required (in AWS Secrets Manager):
- ODDS_API_KEY: Your Odds API key
- GITHUB_TOKEN: Your GitHub Personal Access Token

IAM Permissions Required:
- secretsmanager:GetSecretValue
- sns:Publish (if SNS_TOPIC_ARN is configured)

Lambda Configuration:
- Runtime: Python 3.12
- Memory: 512 MB (minimum)
- Timeout: 15 minutes (900 seconds)
- Ephemeral storage: 2048 MB (need space to clone repo)

Lambda Layers Required:
- git-lambda2:8 (provides git binaries)
- betting-dashboard-dependencies:2 (provides Python packages for Linux x86_64)
"""

import json
import os
import subprocess
import boto3
from datetime import datetime
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
    except Exception as e:
        print(f"⚠️  Failed to send email notification: {e}")


def get_secrets():
    """
    Fetch secrets from AWS Secrets Manager.
    
    Returns:
        dict: Contains ODDS_API_KEY and GITHUB_TOKEN
    """
    secret_name = os.environ['SECRET_NAME']
    region_name = os.environ['AWS_REGION_NAME']
    
    # Create a Secrets Manager client
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
        raise Exception(f"Failed to retrieve secret: {e}")
    
    # Parse the secret
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret


def run_command(cmd, cwd=None, env=None):
    """
    Run a shell command and return output.
    
    Args:
        cmd: Command string or list
        cwd: Working directory
        env: Environment variables dict
        
    Returns:
        tuple: (stdout, stderr, return_code)
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    # Merge environment variables
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=command_env,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    return result.stdout, result.stderr, result.returncode


def lambda_handler(event, context):
    """
    Main Lambda handler function.
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        dict: Response with statusCode and body
    """
    print("=" * 80)
    print("🏀 TQS NBA Props Dashboard - Daily Update (AWS Lambda)")
    print("=" * 80)
    print(f"Execution time: {datetime.now().isoformat()}")
    print("")
    
    # Define work_dir at the top so cleanup can access it
    work_dir = '/tmp/betting'
    
    try:
        # Step 1: Get secrets
        print("📊 Step 1: Fetching secrets from AWS Secrets Manager...")
        secrets = get_secrets()
        odds_api_key = secrets['ODDS_API_KEY']
        github_token = secrets['GITHUB_TOKEN']
        print("✅ Secrets retrieved successfully")
        print("")
        
        # Step 2: Set up environment
        github_repo_url = os.environ['GITHUB_REPO_URL']
        github_username = os.environ['GITHUB_USERNAME']
        github_email = os.environ['GITHUB_EMAIL']
        
        # Create authenticated repo URL
        repo_url_with_token = github_repo_url.replace(
            'https://',
            f'https://{github_username}:{github_token}@'
        )
        
        # Step 3: Clone repository
        print("📦 Step 2: Cloning GitHub repository...")
        
        # Clean up any existing directory
        run_command(['rm', '-rf', work_dir])
        
        # Clone
        stdout, stderr, code = run_command([
            'git', 'clone', repo_url_with_token, work_dir
        ])
        
        if code != 0:
            raise Exception(f"Git clone failed: {stderr}")
        
        print("✅ Repository cloned successfully")
        print("")
        
        # Step 4: Configure git
        print("🔧 Step 3: Configuring git...")
        run_command(['git', 'config', 'user.name', github_username], cwd=work_dir)
        run_command(['git', 'config', 'user.email', github_email], cwd=work_dir)
        print("✅ Git configured")
        print("")
        
        # Step 4: Run the arbitrage finder
        print("🔍 Step 4: Finding arbitrage opportunities...")
        print("   (Using packages from Lambda layer at /opt/python)")
        
        # Create .env file with API key
        env_content = f"ODDS_API_KEY={odds_api_key}\n"
        with open(f"{work_dir}/.env", 'w') as f:
            f.write(env_content)
        
        # Set up environment with PYTHONPATH to include Lambda layer
        # (os is already imported at top of file)
        script_env = os.environ.copy()
        script_env['ODDS_API_KEY'] = odds_api_key
        script_env['PYTHONPATH'] = '/opt/python'  # Allows script to access packages in Lambda layer
        
        # Run the script
        stdout, stderr, code = run_command([
            'python', 'scripts/find_arb_opportunities.py',
            '--markets', 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists'
        ], cwd=work_dir, env=script_env)
        
        if code != 0:
            raise Exception(f"Arbitrage finder failed: {stderr}")
        
        print("✅ Arbitrage opportunities found")
        print("")
        
        # Step 5: Commit and push changes
        print("📤 Step 5: Committing and pushing to GitHub...")
        
        # Check if there are changes
        stdout, stderr, code = run_command(['git', 'status', '--porcelain'], cwd=work_dir)
        
        if not stdout.strip():
            print("ℹ️  No changes to commit")
        else:
            # Add all arb files (both arb_*.csv and raw_*.csv)
            run_command(['git', 'add', 'data/04_output/arbs/*.csv'], cwd=work_dir)
            
            # Commit
            today = datetime.now().strftime('%Y-%m-%d')
            commit_msg = f"Daily update: arbs for {today} (automated via AWS Lambda)"
            run_command(['git', 'commit', '-m', commit_msg], cwd=work_dir)
            
            # Push
            stdout, stderr, code = run_command(['git', 'push'], cwd=work_dir)
            
            if code != 0:
                raise Exception(f"Git push failed: {stderr}")
            
            print("✅ Successfully pushed to GitHub")
            print("   Streamlit Cloud will auto-deploy in 1-2 minutes")
        
        print("")
        print("=" * 80)
        print("✅ Dashboard update complete!")
        print("=" * 80)
        print("")
        print("View dashboard at: https://tqs-nba-props-dashboard.streamlit.app")
        
        # Send success notification
        success_message = f"""✅ TQS NBA Props Dashboard - Daily Update Successful

Execution time: {datetime.now().isoformat()}

The arbitrage finder has completed successfully and new data has been pushed to GitHub.

Dashboard: https://tqs-nba-props-dashboard.streamlit.app

Check CloudWatch Logs for details:
https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups/log-group/$252Faws$252Flambda$252Fbetting-dashboard-daily-update
"""
        send_email_notification("✅ Dashboard Update Successful", success_message)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Dashboard updated successfully',
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        print("")
        
        # Send failure notification
        failure_message = f"""❌ TQS NBA Props Dashboard - Daily Update FAILED

Execution time: {datetime.now().isoformat()}

Error: {str(e)}

Check CloudWatch Logs for full details:
https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups/log-group/$252Faws$252Flambda$252Fbetting-dashboard-daily-update

Common issues:
- API key missing or invalid
- Git authentication failed
- Package import errors (check Lambda layer)
- Insufficient permissions
"""
        send_email_notification("❌ Dashboard Update FAILED", failure_message)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Dashboard update failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }
    
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        run_command(['rm', '-rf', work_dir])
        print("✅ Cleanup complete")

