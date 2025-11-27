"""
AWS Lambda Function - Weekly NFL Regression Plays Finder

This Lambda function:
1. Fetches secrets from AWS Secrets Manager
2. Clones the GitHub repository
3. Downloads Unexpected Points data (Kevin Cole's Google Sheet - PENDING IMPLEMENTATION)
4. Fetches latest NFL betting lines from The Odds API
5. Runs the NFL regression plays finder (find_nfl_regression_plays.py)
6. Commits and pushes generated data files to GitHub
7. Sends email with play recommendations via AWS SNS

IMPORTANT: Python dependencies (pandas, requests, etc.) are provided via
a Lambda Layer, NOT installed at runtime. The layer contains all required packages
pre-built for Linux x86_64.

Trigger Schedule:
- Every Monday at 12:00 PM ET (17:00 UTC)
- Note: May need re-run after Monday Night Football completes

WORKFLOW PREREQUISITES & MISSING COMPONENTS:
========================================

STEP 0A: Download Unexpected Points Data (🚧 NOT YET IMPLEMENTED)
------------------------------------------------------------------
The regression analysis requires box score data with 'adj_score' (expected score) from:
- Source: Kevin Cole's Google Sheet
- URL: https://docs.google.com/spreadsheets/d/1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558/
- Target: data/01_input/unexpected_points/Unexpected Points Subscriber Data.xlsx

BLOCKER: Authentication required for private Google Sheet
- Current script (scripts/download_unexpected_points_data.py) uses interactive OAuth
- Interactive browser-based auth incompatible with Lambda execution
- SSL/certificate errors encountered during testing

SOLUTIONS (pick one):
a) Service Account Auth (RECOMMENDED): Non-interactive, Lambda-compatible
   - Create Google Cloud service account
   - Grant sheet access to service account email
   - Store credentials in AWS Secrets Manager
   - Update script to use service account instead of OAuth

b) Manual S3 Upload: Simplest but manual weekly task
   - Download .xlsx manually each Monday before Lambda runs
   - Upload to S3 bucket
   - Lambda downloads from S3 instead of Google Sheets

c) Public URL: Only if sheet can be made public
   - Share sheet as "Anyone with link can view"
   - Lambda downloads via direct URL

STEP 0B: Fetch Latest NFL Lines (✅ IMPLEMENTED)
------------------------------------------------
Script: scripts/fetch_nfl_season_lines.py --prod-run
- Fetches season-long betting lines from The Odds API
- Skips existing files (no redundant API calls)
- Non-interactive mode for automation
- Saves to: data/01_input/the-odds-api/nfl/game_lines/historical/

STEP 0C: Generate Tracking File (❌ NOT NEEDED IN LAMBDA)
----------------------------------------------------------
Script: backtesting/20251126_nfl_spread_covering_vs_score_differential.py
- This is a ONE-TIME backtesting script
- Output (nfl_game_by_game_tracking_threshold_7.csv) is already in repository
- The play finder consumes this file; it does NOT regenerate it weekly
- DO NOT include this script in the Lambda pipeline

STEP 1: Find Regression Plays (✅ IMPLEMENTED)
----------------------------------------------
Script: implementation/find_nfl_regression_plays.py --current-week --verbose-mode --no-safe-mode
- Identifies teams due for regression based on "luck" metrics
- Requires the tracking file from Step 0C (already exists in repo)
- Outputs play recommendations to data/04_output/

Environment Variables Required:
- GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
- GITHUB_USERNAME: MylesThomas
- GITHUB_EMAIL: mylescgthomas@gmail.com
- SECRET_NAME: betting-dashboard-secrets
- AWS_REGION_NAME: us-east-2
- SNS_TOPIC_ARN: arn:aws:sns:us-east-2:ACCOUNT_ID:nfl-plays-alerts

Secrets Required (in AWS Secrets Manager):
- ODDS_API_KEY: Your Odds API key
- GITHUB_TOKEN: Your GitHub Personal Access Token
- (Future) GOOGLE_SHEETS_SERVICE_ACCOUNT: JSON credentials for Google Sheets API

IAM Permissions Required:
- secretsmanager:GetSecretValue
- sns:Publish
- (Future) s3:GetObject if using S3 for Unexpected Points data

Lambda Configuration:
- Runtime: Python 3.12
- Memory: 1024 MB (minimum)
- Timeout: 15 minutes (900 seconds)
- Ephemeral storage: 2048 MB (need space to clone repo)

Lambda Layers Required:
- git-lambda2:8 (provides git binaries)
- betting-dashboard-dependencies:X (provides Python packages for Linux x86_64)

Git Automation:
- Commits include dynamically generated NFL week numbers
- Stages all files in data/01_input/, data/03_intermediate/, data/04_output/
- Pushes to main branch using GitHub Personal Access Token
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


def parse_plays_from_output(output):
    """
    Parse play recommendations from script output.
    
    Args:
        output: Script stdout string
        
    Returns:
        dict: {
            'week': str,
            'primary_plays': list,
            'secondary_plays': list,
            'total_plays': int,
            'summary_section': str
        }
    """
    lines = output.split('\n')
    
    plays_info = {
        'week': None,
        'primary_plays': [],
        'secondary_plays': [],
        'total_plays': 0,
        'summary_section': ''
    }
    
    # Find the betting opportunities section
    in_opportunities = False
    in_primary = False
    in_secondary = False
    summary_started = False
    
    for line in lines:
        # Capture week number
        if 'WEEK' in line.upper() and not plays_info['week']:
            for word in line.split():
                if word.isdigit():
                    plays_info['week'] = word
                    break
        
        # Start of opportunities section
        if 'BETTING OPPORTUNITIES FOR WEEK' in line:
            in_opportunities = True
            summary_started = True
            plays_info['summary_section'] += line + '\n'
            continue
        
        if in_opportunities:
            plays_info['summary_section'] += line + '\n'
            
            # Parse total plays count
            if 'Found' in line and 'plays' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i + 1 < len(parts) and parts[i + 1] == 'plays':
                        plays_info['total_plays'] = int(part)
                        break
            
            # Track which section we're in
            if 'PRIMARY STRATEGY' in line:
                in_primary = True
                in_secondary = False
            elif 'SECONDARY STRATEGY' in line:
                in_primary = False
                in_secondary = True
            
            # Capture plays
            if '✅ BET:' in line:
                bet_info = line.strip()
                if in_primary:
                    plays_info['primary_plays'].append(bet_info)
                elif in_secondary:
                    plays_info['secondary_plays'].append(bet_info)
            
            # End of summary section
            if '✅ COMPLETE' in line or 'Summary:' in line:
                break
    
    return plays_info


def format_email_body(plays_info, full_output, execution_time):
    """
    Format email body with TL;DR, warning, logs, and summary.
    
    Args:
        plays_info: Dict with parsed play information
        full_output: Complete terminal output
        execution_time: ISO format timestamp
        
    Returns:
        str: Formatted email body
    """
    week = plays_info['week'] or 'UNKNOWN'
    total = plays_info['total_plays']
    primary_count = len(plays_info['primary_plays'])
    secondary_count = len(plays_info['secondary_plays'])
    
    # Build email
    email = []
    
    # TL;DR Section
    email.append("=" * 80)
    email.append(f"🏈 NFL WEEK {week} BETTING PLAYS - TL;DR")
    email.append("=" * 80)
    email.append("")
    
    if total == 0:
        email.append("❌ NO PLAYS FOUND THIS WEEK")
        email.append("")
        email.append("No teams met the regression criteria for this week.")
    else:
        email.append(f"✅ FOUND {total} PLAY{'S' if total != 1 else ''} THIS WEEK")
        email.append(f"   • Primary Strategy (Unlucky Favorites): {primary_count}")
        email.append(f"   • Secondary Strategy (Lucky Big Underdogs): {secondary_count}")
        email.append("")
        
        if plays_info['primary_plays']:
            email.append("🔥 PRIMARY PLAYS (67-73% ATS, +27-39% ROI):")
            for play in plays_info['primary_plays']:
                email.append(f"   {play}")
            email.append("")
        
        if plays_info['secondary_plays']:
            email.append("💰 SECONDARY PLAYS (71% ATS, +36% ROI):")
            for play in plays_info['secondary_plays']:
                email.append(f"   {play}")
            email.append("")
    
    # Warning Section
    email.append("=" * 80)
    email.append("⚠️  IMPORTANT NOTES")
    email.append("=" * 80)
    email.append("")
    email.append("📅 This analysis ran on Monday at 12:00 PM ET")
    email.append("🏈 Monday Night Football may not be included in the analysis")
    email.append("")
    email.append("If MNF results affect Week {week} teams with upcoming games:".format(week=week))
    email.append("   → Consider re-running after MNF completes")
    email.append("   → Manually trigger Lambda function, or")
    email.append("   → Run locally: python implementation/find_nfl_regression_plays.py --current-week --verbose-mode --no-safe-mode")
    email.append("")
    email.append("Strategy Background:")
    email.append("   • Primary: Back teams that underperformed by 7+ points last week (unlucky)")
    email.append("   • Secondary: Back teams that overperformed by 7+ points last week AND are 7+ point underdogs")
    email.append("   • Based on regression-to-mean analysis through Week 12 of 2025 season")
    email.append("")
    
    # Full Logs Section
    email.append("=" * 80)
    email.append("📋 FULL EXECUTION LOGS")
    email.append("=" * 80)
    email.append("")
    email.append(f"Execution Time: {execution_time}")
    email.append("")
    email.append(full_output)
    email.append("")
    
    # Summary Repeated
    email.append("=" * 80)
    email.append(f"📊 SUMMARY (Week {week})")
    email.append("=" * 80)
    email.append("")
    email.append(plays_info['summary_section'])
    
    # Footer
    email.append("=" * 80)
    email.append("📍 Data Files")
    email.append("=" * 80)
    email.append("")
    email.append("Results saved to:")
    email.append(f"   • data/04_output/nfl/todays_plays/nfl_luck_regression_plays_week_{week}_*.csv")
    email.append(f"   • data/04_output/nfl/todays_plays/nfl_luck_regression_all_teams_week_{week}_*.csv")
    email.append("")
    email.append("GitHub: https://github.com/MylesThomas/betting")
    email.append("")
    email.append("CloudWatch Logs:")
    email.append("https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups/log-group/$252Faws$252Flambda$252Fnfl-regression-plays-weekly")
    email.append("")
    
    return '\n'.join(email)


def lambda_handler(event, context):
    """
    Main Lambda handler function.
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        dict: Response with statusCode and body
    """
    execution_time = datetime.now().isoformat()
    
    print("=" * 80)
    print("🏈 NFL Regression Plays Finder - Weekly Update (AWS Lambda)")
    print("=" * 80)
    print(f"Execution time: {execution_time}")
    print("")
    
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
        
        repo_url_with_token = github_repo_url.replace(
            'https://',
            f'https://{github_username}:{github_token}@'
        )
        
        # Step 3: Clone repository
        print("📦 Step 2: Cloning GitHub repository...")
        run_command(['rm', '-rf', work_dir])
        
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
        
        # Create .env file with API key
        env_content = f"ODDS_API_KEY={odds_api_key}\n"
        with open(f"{work_dir}/.env", 'w') as f:
            f.write(env_content)
        
        # Set up environment
        script_env = os.environ.copy()
        script_env['ODDS_API_KEY'] = odds_api_key
        script_env['PYTHONPATH'] = '/opt/python'
        
        # Step 5: Download latest Unexpected Points data
        print("📊 Step 4A: Downloading Unexpected Points data...")
        print("   TODO: User is creating download script")
        print("   For now, assuming file exists from previous run")
        # TODO: Add call to download script once ready
        # stdout, stderr, code = run_command([
        #     'python', 'scripts/download_unexpected_points.py'
        # ], cwd=work_dir, env=script_env)
        print("✅ Unexpected Points data ready")
        print("")
        
        # Step 6: Fetch latest NFL betting lines
        print("📊 Step 4B: Fetching latest NFL betting lines...")
        stdout, stderr, code = run_command([
            'python', 'scripts/fetch_nfl_season_lines.py',
            '--prod-run'
        ], cwd=work_dir, env=script_env)
        
        if code != 0:
            raise Exception(f"NFL lines fetch failed: {stderr}")
        
        print("✅ NFL betting lines updated")
        print("")
        
        # Step 7: Find NFL regression plays for this week
        print("🏈 Step 4C: Finding NFL regression plays for this week...")
        print("   (Using packages from Lambda layer at /opt/python)")
        print("")
        
        # Run the script with verbose output and no prompts
        stdout, stderr, code = run_command([
            'python', 'implementation/find_nfl_regression_plays.py',
            '--current-week',
            '--verbose-mode',
            '--no-safe-mode'
        ], cwd=work_dir, env=script_env)
        
        if code != 0:
            raise Exception(f"NFL regression finder failed: {stderr}")
        
        print("✅ NFL regression analysis complete")
        print("")
        
        # Step 8: Parse results
        print("📊 Step 5: Parsing results...")
        plays_info = parse_plays_from_output(stdout)
        
        week = plays_info['week'] or 'UNKNOWN'
        total_plays = plays_info['total_plays']
        
        print(f"   Week: {week}")
        print(f"   Total plays: {total_plays}")
        print(f"   Primary: {len(plays_info['primary_plays'])}")
        print(f"   Secondary: {len(plays_info['secondary_plays'])}")
        print("")
        
        # Step 9: Format and send email
        print("📧 Step 6: Sending email notification...")
        
        # Combine stdout and stderr for full output
        full_output = stdout
        if stderr:
            full_output += "\n\n--- STDERR ---\n" + stderr
        
        email_body = format_email_body(plays_info, full_output, execution_time)
        
        # Subject line
        if total_plays == 0:
            subject = f"🏈 NFL Week {week}: No Plays Found"
        elif total_plays == 1:
            subject = f"🏈 NFL Week {week}: 1 Play Found"
        else:
            subject = f"🏈 NFL Week {week}: {total_plays} Plays Found"
        
        send_email_notification(subject, email_body)
        
        print("✅ Email notification sent")
        print("")
        
        # Step 10: Optionally commit and push results to GitHub
        print("📤 Step 7: Checking for files to commit...")
        
        stdout_status, stderr, code = run_command(['git', 'status', '--porcelain'], cwd=work_dir)
        
        if not stdout_status.strip():
            print("ℹ️  No changes to commit")
        else:
            # Add all files from data directories (input, intermediate, output)
            run_command(['git', 'add', 'data/01_input/'], cwd=work_dir)
            run_command(['git', 'add', 'data/03_intermediate/'], cwd=work_dir)
            run_command(['git', 'add', 'data/04_output/'], cwd=work_dir)
            
            # Determine week numbers from analysis
            # Week format: current week is the upcoming week to bet on
            # Last week is the week that just finished (used for luck calculation)
            current_week = int(week) if week and week.isdigit() else 0
            last_week = current_week - 1 if current_week > 1 else 0
            
            # Create dynamic commit message
            if current_week > 0 and last_week > 0:
                commit_msg = f"nfl: add week {current_week} plays (post week {last_week} analysis)"
            elif current_week > 0:
                commit_msg = f"nfl: add week {current_week} regression plays via Lambda"
            else:
                commit_msg = "nfl: add regression plays via Lambda"
            
            run_command(['git', 'commit', '-m', commit_msg], cwd=work_dir)
            
            # Push
            stdout_push, stderr_push, code = run_command(['git', 'push'], cwd=work_dir)
            
            if code != 0:
                raise Exception(f"Git push failed: {stderr_push}")
            
            print("✅ Successfully pushed results to GitHub")
        
        print("")
        print("=" * 80)
        print("✅ NFL regression plays finder complete!")
        print("=" * 80)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'NFL plays finder completed successfully',
                'week': week,
                'total_plays': total_plays,
                'primary_plays': len(plays_info['primary_plays']),
                'secondary_plays': len(plays_info['secondary_plays']),
                'timestamp': execution_time
            })
        }
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        print("")
        
        # Send failure notification
        failure_message = f"""❌ NFL Regression Plays Finder - FAILED

Execution time: {execution_time}

Error: {str(e)}

Check CloudWatch Logs for full details:
https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups/log-group/$252Faws$252Flambda$252Fnfl-regression-plays-weekly

Common issues:
- API key missing or invalid
- Git authentication failed
- Package import errors (check Lambda layer)
- Insufficient permissions
- Unexpected Points data not updated
- Odds API rate limit hit

Manual run command:
cd /Users/thomasmyles/dev/betting
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode --no-safe-mode
"""
        send_email_notification("❌ NFL Plays Finder FAILED", failure_message)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'NFL plays finder failed',
                'error': str(e),
                'timestamp': execution_time
            })
        }
    
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        run_command(['rm', '-rf', work_dir])
        print("✅ Cleanup complete")

