"""
NBA Player Props Ingest Lambda Function

Purpose:
Scheduled Lambda function (EventBridge) that fetches NBA player props data
for yesterday and today (ET timezone) and uploads to S3.

Process:
1. Download betting repo from GitHub (as ZIP)
2. Calculate yesterday/today dates in ET timezone
3. Get current NBA season from season_utils
4. Run fetch_nba_player_props.py for both dates
5. Upload results to S3 (done by script)
6. Send SNS notification on success/failure

Lambda Configuration:
- Runtime: Python 3.11
- Architecture: x86_64
- Timeout: 5 minutes (300 seconds)
- Memory: 1024 MB
- Ephemeral Storage: 2048 MB (need space for repo download)

Lambda Layers:
- pyyaml-python311 (arn:aws:lambda:us-east-2:232609278547:layer:pyyaml-python311:2)
  Built with: pip install --platform manylinux2014_x86_64 --only-binary=:all: PyYAML
  Permissions: aws lambda add-layer-version-permission --layer-name pyyaml-python311 --version-number 2 --statement-id allow-all-accounts --action lambda:GetLayerVersion --principal '*' --region us-east-2
- requests-nba-api-python311 (arn:aws:lambda:us-east-2:232609278547:layer:requests-nba-api-python311:1)
  Built with: pip install --platform manylinux2014_x86_64 --only-binary=:all: requests==2.31.0 nba_api==1.4.1 python-dotenv==1.0.0
  Permissions: aws lambda add-layer-version-permission --layer-name requests-nba-api-python311 --version-number 1 --statement-id allow-all-accounts --action lambda:GetLayerVersion --principal '*' --region us-east-2
- pandas-numpy-python311 (arn:aws:lambda:us-east-2:232609278547:layer:pandas-numpy-python311:1)
  Built with: pip install --platform manylinux2014_x86_64 --only-binary=:all: pandas==2.1.4 numpy==1.26.2
  Permissions: aws lambda add-layer-version-permission --layer-name pandas-numpy-python311 --version-number 1 --statement-id allow-all-accounts --action lambda:GetLayerVersion --principal '*' --region us-east-2

Note: Custom layers require permissions to be accessible by Lambda functions.
Use the aws lambda add-layer-version-permission command shown above for each custom layer.

Environment Variables Required:
- GITHUB_REPO_URL: URL of betting repo
- GITHUB_USERNAME: GitHub username for auth
- GITHUB_EMAIL: GitHub email for git config
- SECRET_NAME: AWS Secrets Manager secret name (contains GITHUB_TOKEN)
- SNS_TOPIC_ARN: SNS topic for notifications
- AWS_REGION_NAME: AWS region (us-east-2)
- ODDS_API_KEY: The Odds API key for fetching props

IAM Permissions Required:
- SecretsManager: GetSecretValue (for GitHub token)
- S3: PutObject (for uploading props data)
- SNS: Publish (for notifications)
- CloudWatch Logs: CreateLogGroup, CreateLogStream, PutLogEvents

EventBridge Schedule:
Run daily at 8:00 AM ET (after games finish and odds are finalized)
- Cron: cron(0 13 * * ? *) UTC = 8:00 AM ET
"""

import os
import sys
import json
import subprocess
import shutil
import zipfile
import requests
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import boto3
from botocore.exceptions import ClientError


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variables
GITHUB_REPO_URL = os.environ['GITHUB_REPO_URL']
GITHUB_USERNAME = os.environ['GITHUB_USERNAME']
GITHUB_EMAIL = os.environ['GITHUB_EMAIL']
SECRET_NAME = os.environ['SECRET_NAME']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
AWS_REGION = os.environ.get('AWS_REGION_NAME', 'us-east-2')

# Lambda temp directory
WORK_DIR = Path('/tmp/betting')

# Boto3 clients
secrets_client = boto3.client('secretsmanager', region_name=AWS_REGION)
sns_client = boto3.client('sns', region_name=AWS_REGION)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_secret(secret_name: str) -> dict:
    """
    Retrieve secret from AWS Secrets Manager.
    
    Args:
        secret_name: Name of secret in Secrets Manager
    
    Returns:
        Dict with secret key-value pairs
    """
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        print(f"❌ Error retrieving secret {secret_name}: {e}")
        raise


def run_cmd(cmd: list, cwd=None, check=True, env=None) -> subprocess.CompletedProcess:
    """
    Run shell command and return result.
    
    Args:
        cmd: Command as list (e.g., ['git', 'clone', url])
        cwd: Working directory (default: None)
        check: Raise exception on non-zero exit (default: True)
        env: Environment variables dict (default: None, inherits from parent)
    
    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    print(f"   Running: {' '.join(cmd)}")
    
    # Merge environment variables
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise yet, we want to print output first
            env=full_env
        )
        
        if result.stdout:
            print(f"   stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"   stderr: {result.stderr.strip()}")
        
        # Now raise if check=True and command failed
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, 
                cmd, 
                output=result.stdout, 
                stderr=result.stderr
            )
        
        return result
        
    except subprocess.CalledProcessError as e:
        # Re-raise with output already printed
        raise


def get_et_date(days_ago: int = 0) -> str:
    """
    Get date in ET timezone (YYYY-MM-DD format).
    
    Args:
        days_ago: Number of days in the past (0 = today, 1 = yesterday)
    
    Returns:
        Date string in YYYY-MM-DD format
    """
    et_tz = ZoneInfo('America/New_York')
    et_now = datetime.now(et_tz)
    target_date = et_now - timedelta(days=days_ago)
    return target_date.strftime('%Y-%m-%d')


def download_repo(github_token: str) -> Path:
    """
    Download betting repo as ZIP (no git required).
    
    Args:
        github_token: GitHub personal access token
    
    Returns:
        Path to extracted repo
    """
    import zipfile
    import requests
    
    print("📦 Downloading repository...")
    
    # Clean up old download if exists
    if WORK_DIR.exists():
        print(f"   Cleaning up old download: {WORK_DIR}")
        shutil.rmtree(WORK_DIR)
    
    # Download main branch as ZIP
    # Format: https://api.github.com/repos/{owner}/{repo}/zipball/main
    zip_url = f"https://api.github.com/repos/MylesThomas/betting/zipball/main"
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    print(f"   Downloading from: {zip_url}")
    response = requests.get(zip_url, headers=headers, timeout=60)
    response.raise_for_status()
    
    # Save ZIP to temp file
    zip_path = Path('/tmp/repo.zip')
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    print(f"   ✅ Downloaded {len(response.content) / 1024 / 1024:.1f} MB")
    
    # Extract ZIP
    print(f"   Extracting to {WORK_DIR.parent}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(WORK_DIR.parent)
    
    # GitHub ZIP extracts to: MylesThomas-betting-{commit_hash}/
    # We need to rename it to /tmp/betting
    extracted_dirs = list(WORK_DIR.parent.glob('MylesThomas-betting-*'))
    if not extracted_dirs:
        raise Exception("Could not find extracted repo directory")
    
    extracted_dir = extracted_dirs[0]
    extracted_dir.rename(WORK_DIR)
    
    # Clean up ZIP
    zip_path.unlink()
    
    print(f"   ✅ Repository downloaded to {WORK_DIR}")
    return WORK_DIR


def send_sns(subject: str, message: str, success: bool = True):
    """
    Send SNS notification.
    
    Args:
        subject: Email subject line
        message: Email body
        success: If True, prepend ✅, else ❌
    """
    emoji = "✅" if success else "❌"
    full_subject = f"{emoji} {subject}"
    
    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=full_subject[:100],  # SNS subject max 100 chars
            Message=message
        )
        print(f"   📧 SNS sent: {full_subject}")
    except Exception as e:
        print(f"   ⚠️  Failed to send SNS: {e}")


# =============================================================================
# MAIN HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    Lambda handler for NBA player props ingestion.
    
    Args:
        event: Lambda event (from EventBridge scheduler)
        context: Lambda context
    
    Returns:
        Dict with statusCode and result
    """
    print("="*80)
    print("NBA PLAYER PROPS INGEST - Lambda Function")
    print("="*80)
    print()
    
    start_time = datetime.now()
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    try:
        # =====================================================================
        # STEP 1: Get GitHub token from Secrets Manager
        # =====================================================================
        print("="*80)
        print("STEP 1: Retrieving GitHub token from Secrets Manager...")
        print("="*80)
        
        secrets = get_secret(SECRET_NAME)
        github_token = secrets.get('GITHUB_TOKEN')
        
        if not github_token:
            raise ValueError("GITHUB_TOKEN not found in Secrets Manager")
        
        print("   ✅ GitHub token retrieved")
        print()
        
        # =====================================================================
        # STEP 2: Download repository
        # =====================================================================
        print("="*80)
        print("STEP 2: Downloading betting repository...")
        print("="*80)
        
        repo_path = download_repo(github_token)
        print()
        
        # =====================================================================
        # STEP 3: Calculate dates (ET timezone)
        # =====================================================================
        print("="*80)
        print("STEP 3: Calculating dates (ET timezone)...")
        print("="*80)
        
        yesterday = get_et_date(days_ago=1)
        today = get_et_date(days_ago=0)
        
        print(f"   Yesterday (ET): {yesterday}")
        print(f"   Today (ET):     {today}")
        print()
        
        # =====================================================================
        # STEP 4: Get current NBA season
        # =====================================================================
        print("="*80)
        print("STEP 4: Getting current NBA season...")
        print("="*80)
        
        # Add src to path
        sys.path.insert(0, str(repo_path / 'src'))
        
        # Import season_utils
        from season_utils import get_current_nba_season
        
        season = get_current_nba_season()
        print(f"   📅 Current NBA Season: {season}")
        print()
        
        # =====================================================================
        # STEP 5: Fetch props for YESTERDAY
        # =====================================================================
        print("="*80)
        print(f"STEP 5: Fetching player props for YESTERDAY ({yesterday})...")
        print("="*80)
        
        # CRITICAL: Lambda layers are mounted at /opt/python, but subprocess.run()
        # spawns a NEW Python interpreter that doesn't inherit the Lambda runtime's
        # sys.path. We MUST set PYTHONPATH to the FULL path where packages are:
        # /opt/python/lib/python3.11/site-packages (NOT just /opt/python)
        #
        # Layer structure: /opt/python/lib/python3.11/site-packages/requests/
        # Without this: ModuleNotFoundError: No module named 'requests'
        # With this: subprocess finds packages in layers
        python_env = {
            'PYTHONPATH': '/opt/python/lib/python3.11/site-packages'
        }
        
        # Debug: Verify ALL required packages are accessible
        print("   🔍 Verifying layer packages...")
        print()
        
        required_packages = [
            ('yaml', 'PyYAML', 'pyyaml-python311:2'),
            ('requests', 'requests', 'requests-nba-api-python311:1'),
            ('nba_api', 'nba_api', 'requests-nba-api-python311:1'),
            ('dotenv', 'python-dotenv', 'requests-nba-api-python311:1'),
            ('pandas', 'pandas', 'pandas-numpy-python311:1'),
            ('numpy', 'numpy', 'pandas-numpy-python311:1'),
        ]
        
        all_packages_found = True
        for module_name, package_name, layer_name in required_packages:
            result = run_cmd(
                ['python3', '-c', f'import {module_name}; print("{module_name}:", {module_name}.__version__ if hasattr({module_name}, "__version__") else "OK")'],
                cwd=repo_path,
                check=False,
                env=python_env
            )
            if result.returncode != 0:
                print(f"      ❌ {package_name} NOT FOUND (from {layer_name})")
                all_packages_found = False
            else:
                print(f"      ✅ {package_name} found (from {layer_name})")
        
        print()
        
        if not all_packages_found:
            raise Exception("Missing required packages! Check Lambda layers are attached and have permissions.")
        
        print("   ✅ All packages verified! Running script...")
        print()
        
        run_cmd([
            'python3',
            'scripts/fetch_nba_player_props.py',
            '--date', yesterday,
            '--s3',
            '--season', season,
            '--force',
            '--fetch-games'
        ], cwd=repo_path, env=python_env)
        
        print(f"   ✅ Props fetched for {yesterday}")
        print()
        
        # =====================================================================
        # STEP 6: Fetch props for TODAY
        # =====================================================================
        print("="*80)
        print(f"STEP 6: Fetching player props for TODAY ({today})...")
        print("="*80)
        
        # CRITICAL: Same PYTHONPATH setup as STEP 5 (see comment above)
        # Lambda layers: /opt/python/lib/python3.11/site-packages
        python_env = {
            'PYTHONPATH': '/opt/python/lib/python3.11/site-packages'
        }
        
        run_cmd([
            'python3',
            'scripts/fetch_nba_player_props.py',
            '--date', today,
            '--s3',
            '--season', season,
            '--force',
            '--fetch-games'
        ], cwd=repo_path, env=python_env)
        
        print(f"   ✅ Props fetched for {today}")
        print()
        
        # =====================================================================
        # SUCCESS - Send notification
        # =====================================================================
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print("="*80)
        print("✅ SUCCESS")
        print("="*80)
        print(f"⏱️  Total time: {elapsed:.1f}s")
        print()
        
        # Send success SNS
        message = f"""NBA Player Props Ingest - Success

Dates Processed:
- Yesterday: {yesterday}
- Today: {today}

NBA Season: {season}

Execution Time: {elapsed:.1f}s
Function: {context.function_name}
Request ID: {context.aws_request_id}
"""
        
        send_sns(
            subject=f"Props Ingest Success - {today}",
            message=message,
            success=True
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'yesterday': yesterday,
                'today': today,
                'season': season,
                'elapsed_seconds': elapsed
            })
        }
    
    except Exception as e:
        # =====================================================================
        # FAILURE - Send notification
        # =====================================================================
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print("="*80)
        print("❌ FAILURE")
        print("="*80)
        print(f"Error: {e}")
        print(f"⏱️  Time before failure: {elapsed:.1f}s")
        print()
        
        # Send failure SNS
        message = f"""NBA Player Props Ingest - FAILURE

Error: {str(e)}

Execution Time: {elapsed:.1f}s
Function: {context.function_name}
Request ID: {context.aws_request_id}

Check CloudWatch Logs:
https://console.aws.amazon.com/cloudwatch/home?region={AWS_REGION}#logsV2:log-groups/log-group/$252Faws$252Flambda$252F{context.function_name}
"""
        
        send_sns(
            subject=f"Props Ingest FAILED - {get_et_date()}",
            message=message,
            success=False
        )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'elapsed_seconds': elapsed
            })
        }
