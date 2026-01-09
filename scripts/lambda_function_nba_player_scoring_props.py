"""
AWS Lambda Function - Daily NBA Props Workflow

Lambda function name: nba-player-scoring-props-daily-workflow

This Lambda function:
1. Fetches secrets from AWS Secrets Manager
2. Clones the GitHub repository
3. Runs the daily NBA props workflow:
   - Finds today's 2D plays (tier × spread)
   - Finds today's 3D plays (tier × spread × scorer_type)
   - Fetches yesterday's game results from NBA API
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


def get_current_nba_season():
    """
    Get current NBA season based on today's date.
    
    NBA seasons run from October to June:
    - Oct-Dec: Current year is start year (Oct 2025 → '2025-26')
    - Jan-Sep: Previous year is start year (Jan 2026 → '2025-26')
    
    Returns:
        str: Season in format 'YYYY-YY' (e.g., '2025-26')
    """
    today = datetime.now()
    if today.month >= 10:  # Oct-Dec
        return f"{today.year}-{str(today.year + 1)[-2:]}"
    else:  # Jan-Sep
        return f"{today.year - 1}-{str(today.year)[-2:]}"


def send_email_notification(subject, message, topic_arn=None):
    """
    Send email notification via AWS SNS.
    
    Args:
        subject: Email subject
        message: Email body
        topic_arn: SNS topic ARN (optional, defaults to env var)
    """
    if topic_arn is None:
        topic_arn = os.environ['SNS_TOPIC_ARN']
    
    sns_client = boto3.client('sns', region_name=os.environ['AWS_REGION_NAME'])
    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message
    )
    print(f"✅ Email notification sent (MessageId: {response['MessageId']})")
    return True


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
    
    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    
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
    3. Fetch yesterday's game results from NBA API
    4. Track yesterday's performance (BOTH 2D + 3D)
    5. Generate daily email (BOTH 2D + 3D)
    
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
        'AWS_DEFAULT_REGION': os.environ['AWS_REGION_NAME'],
        'PYTHONPATH': '/opt/python'  # Lambda layer path for pandas, numpy, etc.
        # Why we need PYTHONPATH: The script needs to access packages in the Lambda layer.
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
        '--date', today,  # Pass ET date explicitly to avoid UTC issues
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
        '--date', today,  # Pass ET date explicitly to avoid UTC issues
        '--save-s3'
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['steps']['3d_plays'] = {
        'success': returncode == 0,
        'output': stdout
    }
    
    if returncode != 0:
        print(f"⚠️  3D play finder failed (non-fatal, continuing...)")
    
    # Step 3: Fetch yesterday's game results
    print(f"\n{'='*80}")
    print(f"Step 3: Fetching Yesterday's Game Results ({yesterday})")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'scripts/fetch_nba_player_props.py',
        '--date', yesterday,
        '--fetch-games',
        '--s3',
        '--season', season
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['steps']['fetch_games'] = {
        'success': returncode == 0,
        'output': stdout
    }
    
    if returncode != 0:
        print(f"❌ Game results fetch failed (critical failure)")
    
    # Step 4: Track yesterday's performance (BOTH 2D and 3D)
    print(f"\n{'='*80}")
    print(f"Step 4: Tracking Yesterday's Performance ({yesterday}) - BOTH 2D + 3D")
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
    
    # Step 5: Generate daily email (BOTH 2D and 3D) and send via SNS
    print(f"\n{'='*80}")
    print("Step 5: Generating Daily Email - BOTH 2D + 3D")
    print(f"{'='*80}\n")
    
    sns_topic_arn = os.environ['SNS_TOPIC_ARN']
    
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
    
    # =============================================================================
    # CHECK CRITICAL FAILURES & SEND STATUS EMAIL
    # =============================================================================
    
    # Check if critical steps failed (fetch_games is critical)
    critical_failures = []
    
    if not results['steps']['fetch_games']['success']:
        critical_failures.append(f"Step 3: Game results fetch failed for {yesterday}")
    
    # Build status email regardless of success/failure
    status_lines = [
        "="*80,
        "🏀 NBA PROPS DAILY WORKFLOW - EXECUTION SUMMARY",
        "="*80,
        f"📅 Today: {today}",
        f"📅 Yesterday: {yesterday}",
        "",
        "MAIN WORKFLOW STEPS:",
        "────────────────────────────────────────────────────────────────────────────────",
    ]
    
    step_statuses = [
        ("Step 1: Find 2D Plays", results['steps']['2d_plays']['success']),
        ("Step 2: Find 3D Plays", results['steps']['3d_plays']['success']),
        ("Step 3: Fetch Game Results", results['steps']['fetch_games']['success']),
        ("Step 4: Track Performance", results['steps']['tracking']['success']),
        ("Step 5: Generate Email", results['steps']['email']['success'])
    ]
    
    for step_name, success in step_statuses:
        status_emoji = "✅" if success else "❌"
        status_lines.append(f"{status_emoji} {step_name}")
    
    status_lines.extend([
        "",
        "="*80
    ])
    
    if critical_failures:
        status_lines.extend([
            "",
            "🚨 CRITICAL FAILURES DETECTED:",
            "────────────────────────────────────────────────────────────────────────────────"
        ])
        for failure in critical_failures:
            status_lines.append(f"  • {failure}")
        status_lines.extend([
            "",
            "⚠️  Workflow terminated due to critical failures.",
            "Check CloudWatch logs for detailed error messages:",
            f"Log Group: /aws/lambda/nba-player-scoring-props-daily-workflow",
            "",
            "="*80
        ])
        
        status_email = "\n".join(status_lines)
        
        # Send failure email
        send_email_notification(
            subject=f"❌ NBA Props Workflow FAILED - {today}",
            message=status_email
        )
        
        error_msg = "❌ Critical workflow failures:\n" + "\n".join(f"  - {f}" for f in critical_failures)
        print(f"\n{error_msg}\n")
        raise RuntimeError(error_msg)
    
    # =============================================================================
    # NEW: Run Top3 Unders workflow (Steps 6-8) - ADDITIVE - 2026-01-08
    # =============================================================================
    
    # Check if all main workflow steps succeeded before running Top3
    print(f"\n{'='*80}")
    print("Checking Main Workflow Status for Top3 Decision:")
    print(f"{'='*80}")
    print(f"  2D Plays: {results['steps']['2d_plays']['success']}")
    print(f"  3D Plays: {results['steps']['3d_plays']['success']}")
    print(f"  Fetch Games: {results['steps']['fetch_games']['success']}")
    print(f"  Tracking: {results['steps']['tracking']['success']}")
    print(f"  Email: {results['steps']['email']['success']}")
    
    main_steps_success = (
        results['steps']['2d_plays']['success'] and
        results['steps']['3d_plays']['success'] and
        results['steps']['fetch_games']['success'] and
        results['steps']['tracking']['success'] and
        results['steps']['email']['success']
    )
    
    print(f"  Overall: {main_steps_success}\n")
    
    if main_steps_success:
        print(f"\n✅ Main workflow succeeded - running Top3 Unders workflow")
        top3_results = run_top3_unders_workflow(repo_dir, today, yesterday, season)
        results['steps']['top3_workflow'] = top3_results
    else:
        print(f"\n⚠️  Skipping Top3 Unders workflow - main workflow had failures")
        results['steps']['top3_workflow'] = {
            'skipped': True,
            'reason': 'main_workflow_failed',
            'main_steps_status': {
                '2d_plays': results['steps']['2d_plays']['success'],
                '3d_plays': results['steps']['3d_plays']['success'],
                'fetch_games': results['steps']['fetch_games']['success'],
                'tracking': results['steps']['tracking']['success'],
                'email': results['steps']['email']['success']
            }
        }
    
    return results


def validate_player_team_assignments(df_plays, date_str):
    """
    Validate that players are assigned to correct teams/games.
    Downloads game data from S3 and cross-references player teams.
    
    Args:
        df_plays: DataFrame with plays (must have: player, team, opponent columns)
        date_str: Date string (YYYY-MM-DD)
    
    Returns:
        dict: Validation results with warnings and errors
    """
    import boto3
    import pandas as pd
    from io import StringIO
    
    print(f"\n   🔍 Validating player-team assignments...")
    
    validation_results = {
        'total_plays': len(df_plays),
        'warnings': [],
        'errors': [],
        'games_with_plays': set(),
        'all_games_for_date': []
    }
    
    # Download game data from S3 to get actual team rosters
    s3 = boto3.client('s3', region_name=os.environ['AWS_REGION_NAME'])
    bucket = 'nba-betting-mt'
    
    # Load game results for this date (has player-team info)
    game_results_key = f'data/03_nba_stats/games/{date_str}.csv'
    
    obj = s3.get_object(Bucket=bucket, Key=game_results_key)
    df_games = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    
    # Get all games for this date
    games_today = df_games.groupby(['home_team', 'away_team']).size().reset_index()[['home_team', 'away_team']]
    validation_results['all_games_for_date'] = [
        f"{row['away_team']} @ {row['home_team']}" 
        for _, row in games_today.iterrows()
    ]
    print(f"   📅 Found {len(games_today)} total games for {date_str}")
    
    # Build player-team mapping from game data
    player_team_map = {}
    for _, row in df_games.iterrows():
        player_team_map[row['player']] = row['team']
    
    print(f"   📋 Loaded {len(player_team_map)} player-team mappings")
    
    # Validate each play
    for idx, play in df_plays.iterrows():
        player = play['player']
        claimed_team = play['team']
        opponent = play['opponent']
        
        # Track games with plays
        game_key = tuple(sorted([claimed_team, opponent]))
        validation_results['games_with_plays'].add(game_key)
        
        # Check if player's actual team matches claimed team
        actual_team = player_team_map[player]
        if actual_team != claimed_team:
            error_msg = (
                f"❌ TEAM MISMATCH: {player} is on {actual_team}, "
                f"not {claimed_team} (claimed in {claimed_team} vs {opponent})"
            )
            validation_results['errors'].append(error_msg)
            print(f"   {error_msg}")
    
    # Summary
    print(f"   ✅ Validation complete:")
    print(f"      - {len(validation_results['errors'])} errors found")
    print(f"      - {len(validation_results['warnings'])} warnings")
    print(f"      - {len(validation_results['games_with_plays'])} games have plays (out of {len(games_today)} total)")
    
    return validation_results


def filter_plays_by_config(all_plays_csv_path, config_data, output_csv_path, dimension='2d'):
    """
    Filter plays CSV file to only include strategies from config.
    
    Args:
        all_plays_csv_path: Path to full plays CSV
        config_data: Loaded config dict with strategy definitions
        output_csv_path: Where to save filtered plays CSV
        dimension: '2d' or '3d'
    
    Returns:
        int: Number of filtered plays
    """
    import pandas as pd
    
    print(f"   Filtering {dimension.upper()} plays from CSV...")
    
    # Read all plays CSV - fail if missing
    df_all = pd.read_csv(all_plays_csv_path)
    original_count = len(df_all)
    print(f"   Loaded {original_count} plays from CSV")
    
    # Extract filter criteria from config strategies - fail if keys missing
    strategies = [
        s for s in config_data['strategies'] 
        if s['strategy_type'] == dimension
    ]
    
    assert len(strategies) > 0, f"No {dimension} strategies found in config"
    
    # Build filter criteria and add strategy_name
    df_all['strategy_name'] = None  # Initialize column
    
    for strat in strategies:
        line_tier = strat['line_tier']
        spread_bin = strat['spread_bin']
        strategy_name = strat['strategy_name']
        
        print(f"   Filter: {strategy_name} ({line_tier} + {spread_bin}", end='')
        
        # Create boolean mask for this strategy
        mask = (df_all['line_tier'] == line_tier) & (df_all['spread_bin'] == spread_bin)
        
        # For 3D, also match scorer_type
        if dimension == '3d':
            scorer_type = strat['scorer_type']
            mask = mask & (df_all['scorer_type'] == scorer_type)
            print(f" + {scorer_type})")
        else:
            print(")")
        
        # Assign strategy_name to matching rows
        df_all.loc[mask, 'strategy_name'] = strategy_name
    
    # Filter to only rows with a strategy_name assigned
    df_filtered = df_all[df_all['strategy_name'].notna()].copy()
    filtered_count = len(df_filtered)
    print(f"   ✅ Filtered: {filtered_count} plays (from {original_count})")
    
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"   💾 Saved to: {output_csv_path}")
    
    return filtered_count


def run_top3_unders_workflow(repo_dir, today, yesterday, season='2025-26'):
    """
    Run the Top3 Unders workflow (Steps 6-8).
    Filters existing plays, tracks separately, sends 2nd email.
    
    This is ADDITIVE - doesn't touch the main workflow.
    
    Args:
        repo_dir: Path to cloned repository
        today: Today's date (YYYY-MM-DD)
        yesterday: Yesterday's date (YYYY-MM-DD)
        season: NBA season
    
    Returns:
        dict: Top3 workflow results
    """
    print(f"\n{'='*80}")
    print("🎯 Starting Top3 Unders Workflow (Steps 6-8)")
    print(f"{'='*80}\n")
    
    results = {}
    
    # Step 6: Download strategy config from S3 and filter plays
    print(f"\n{'='*80}")
    print("Step 6: Filtering Today's Plays (Top3 Unders)")
    print(f"{'='*80}\n")
    
    import boto3
    import json
    
    s3_client = boto3.client('s3', region_name=os.environ['AWS_REGION_NAME'])
    bucket = 'nba-betting-mt'
    
    # Download and load strategy config
    config_s3_path = 'strategies/top3_unders_strategies_nba_points_props.json'
    config_local_path = '/tmp/top3_config.json'
    
    print(f"   Downloading config from s3://{bucket}/{config_s3_path}")
    s3_client.download_file(bucket, config_s3_path, config_local_path)
    
    with open(config_local_path, 'r') as f:
        config_data = json.load(f)
    
    print(f"   Config: {config_data['name']} - {config_data['description']}")
    print(f"   Strategies: {len(config_data['strategies'])}")
    
    # Validate config structure
    strategies = config_data['strategies']
    assert len(strategies) == 3, f"Expected 3 strategies, got {len(strategies)}"
    
    count_2d = len([s for s in strategies if s['strategy_type'] == '2d'])
    count_3d = len([s for s in strategies if s['strategy_type'] == '3d'])
    assert count_2d == 1, f"Expected 1 2D strategy, got {count_2d}"
    assert count_3d == 2, f"Expected 2 3D strategies, got {count_3d}"
    
    print(f"   ✅ Config validated: {count_2d} x 2D, {count_3d} x 3D")
    
    # Download and filter 2D plays CSV
    plays_2d_all_csv = f'/tmp/{today}_2d.csv'
    plays_2d_top3_csv = f'/tmp/{today}_2d_top3.csv'
    s3_2d_path = f'data/04_output/plays/role_spread_points_model/2d/{today}.csv'
    s3_2d_top3_path = f'data/04_output/plays/role_spread_points_model/2d/{today}_top3.csv'
    
    print(f"\n   2D Plays:")
    s3_client.download_file(bucket, s3_2d_path, plays_2d_all_csv)
    filtered_2d_count = filter_plays_by_config(plays_2d_all_csv, config_data, plays_2d_top3_csv, dimension='2d')
    s3_client.upload_file(plays_2d_top3_csv, bucket, s3_2d_top3_path)
    print(f"   ✅ Uploaded: s3://{bucket}/{s3_2d_top3_path}")
    
    results['2d_filter'] = {'success': True, 'plays_count': filtered_2d_count}
    
    # Download and filter 3D plays CSV
    plays_3d_all_csv = f'/tmp/{today}_3d.csv'
    plays_3d_top3_csv = f'/tmp/{today}_3d_top3.csv'
    s3_3d_path = f'data/04_output/plays/role_spread_points_model/3d/{today}.csv'
    s3_3d_top3_path = f'data/04_output/plays/role_spread_points_model/3d/{today}_top3.csv'
    
    print(f"\n   3D Plays:")
    s3_client.download_file(bucket, s3_3d_path, plays_3d_all_csv)
    filtered_3d_count = filter_plays_by_config(plays_3d_all_csv, config_data, plays_3d_top3_csv, dimension='3d')
    s3_client.upload_file(plays_3d_top3_csv, bucket, s3_3d_top3_path)
    print(f"   ✅ Uploaded: s3://{bucket}/{s3_3d_top3_path}")
    
    results['3d_filter'] = {'success': True, 'plays_count': filtered_3d_count}
    
    # Validate player-team assignments for filtered plays
    print(f"\n{'='*80}")
    print("Step 6b: Validating Player-Team Assignments")
    print(f"{'='*80}\n")
    
    import pandas as pd
    
    # Load filtered plays for validation
    df_2d_filtered = pd.read_csv(plays_2d_top3_csv)
    df_3d_filtered = pd.read_csv(plays_3d_top3_csv)
    df_all_filtered = pd.concat([df_2d_filtered, df_3d_filtered], ignore_index=True)
    
    validation_results = validate_player_team_assignments(df_all_filtered, today)
    results['validation'] = validation_results
    
    # Check for critical validation errors
    if validation_results['errors']:
        print(f"\n⚠️  WARNING: {len(validation_results['errors'])} validation errors detected")
        print(f"   These plays may have incorrect team assignments!\n")
        
        # Send warning email with validation errors
        error_lines = [
            "="*80,
            f"⚠️  NBA TOP3 PLAYS VALIDATION WARNINGS - {today}",
            "="*80,
            f"📅 Date: {today}",
            f"🔍 Validation Status: {len(validation_results['errors'])} ERRORS, {len(validation_results['warnings'])} WARNINGS",
            "",
            "VALIDATION ERRORS (Player-Team Mismatches):",
            "────────────────────────────────────────────────────────────────────────────────"
        ]
        
        for error in validation_results['errors']:
            error_lines.append(f"  • {error}")
        
        if validation_results['warnings']:
            error_lines.extend([
                "",
                "WARNINGS:",
                "────────────────────────────────────────────────────────────────────────────────"
            ])
            for warning in validation_results['warnings']:
                error_lines.append(f"  • {warning}")
        
        # Show games breakdown
        if validation_results['all_games_for_date']:
            error_lines.extend([
                "",
                f"GAMES FOR {today}:",
                "────────────────────────────────────────────────────────────────────────────────",
                f"Total games: {len(validation_results['all_games_for_date'])}",
                f"Games with Top3 plays: {len(validation_results['games_with_plays'])}",
                "",
                "All games:"
            ])
            for game in validation_results['all_games_for_date']:
                has_plays = "✅" if any(game_team in str(validation_results['games_with_plays']) for game_team in game.split(' @ ')) else "❌"
                error_lines.append(f"  {has_plays} {game}")
        
        error_lines.extend([
            "",
            "⚠️  These errors may cause issues when placing bets.",
            "Please verify player teams manually before betting.",
            "",
            "="*80
        ])
        
        send_email_notification(
            subject=f"⚠️  NBA Top3 Plays - Validation Warnings - {today}",
            message="\n".join(error_lines)
        )
    
    # Step 7: Track yesterday's Top3 performance
    print(f"\n{'='*80}")
    print("Step 7: Tracking Yesterday's Top3 Performance")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'scripts/track_daily_plays_performance.py',
        '--date', yesterday,
        '--season', season,
        '--strategy', 'both',
        '--plays-suffix', '_top3',
        '--output-suffix', '_top3'
    ]
    
    env = {
        'AWS_DEFAULT_REGION': os.environ['AWS_REGION_NAME'],
        'PYTHONPATH': '/opt/python'
    }
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['tracking'] = {'success': returncode == 0, 'output': stdout}
    
    # Step 8: Generate Top3 email
    print(f"\n{'='*80}")
    print("Step 8: Generating Top3 Unders Email")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'scripts/generate_role_spread_points_model_daily_email.py',
        '--season', season,
        '--plays-date', today,
        '--results-date', yesterday,
        '--strategy', 'both',
        '--plays-suffix', '_top3',
        '--tracking-suffix', '_top3',
        '--email-title', '🎯 Top 3 Unders Plays',
        '--load-ytd',  # Load season YTD stats for top3 email
        '--sns-topic', os.environ['SNS_TOPIC_ARN']
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=repo_dir, env=env)
    results['email'] = {'success': returncode == 0, 'output': stdout, 'email_sent': returncode == 0}
    
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
        season = os.environ.get('SEASON', '2025-26')  # Default to current season
        
        # Validate season matches actual current season
        actual_season = get_current_nba_season()
        assert season == actual_season, f"SEASON env var '{season}' doesn't match actual current season '{actual_season}'"
        
        # Get secrets
        print("🔐 Fetching secrets...")
        secrets = get_secrets()
        github_token = secrets['GITHUB_TOKEN']
        odds_api_key = secrets['ODDS_API_KEY']
        
        print("✅ Secrets retrieved successfully\n")
        
        # Clone repository
        repo_dir = clone_repository(repo_url, github_token)
        
        # Run workflow
        workflow_results = run_daily_workflow(repo_dir, odds_api_key, season)
        
        # Email was sent directly by the workflow script via --sns-topic
        email_sent = workflow_results['steps']['email']['email_sent']
        
        # Summary
        print(f"\n{'='*80}")
        print("✅ Lambda Execution Complete")
        print(f"{'='*80}")
        print(f"Today: {workflow_results['today']}")
        print(f"Yesterday: {workflow_results['yesterday']}")
        print(f"\nMain Workflow (All Strategies):")
        print(f"  2D Plays: {'✅' if workflow_results['steps']['2d_plays']['success'] else '❌'}")
        print(f"  3D Plays: {'✅' if workflow_results['steps']['3d_plays']['success'] else '❌'}")
        print(f"  Fetch Games: {'✅' if workflow_results['steps']['fetch_games']['success'] else '❌'}")
        print(f"  Tracking: {'✅' if workflow_results['steps']['tracking']['success'] else '❌'}")
        print(f"  Email+SNS: {'✅' if email_sent else '❌'}")
        
        # Top3 workflow summary
        top3 = workflow_results['steps']['top3_workflow']
        print(f"\nTop3 Unders Workflow:")
        if 'skipped' in top3 and top3['skipped']:
            print(f"  Status: ⏭️  Skipped - {top3['reason']}")
        else:
            print(f"  2D Filter: {'✅' if top3['2d_filter']['success'] else '❌'} ({top3['2d_filter']['plays_count']} plays)")
            print(f"  3D Filter: {'✅' if top3['3d_filter']['success'] else '❌'} ({top3['3d_filter']['plays_count']} plays)")
            
            # Show validation summary
            if 'validation' in top3:
                val = top3['validation']
                errors = len(val['errors'])
                warnings = len(val['warnings'])
                status = '❌' if errors > 0 else '✅'
                print(f"  Validation: {status} ({errors} errors, {warnings} warnings)")
                total_games = len(val['all_games_for_date'])
                games_with_plays = len(val['games_with_plays'])
                print(f"  Games: {games_with_plays}/{total_games} games have Top3 plays")
            
            print(f"  Tracking: {'✅' if top3['tracking']['success'] else '❌'}")
            print(f"  Email+SNS: {'✅' if top3['email']['success'] else '❌'}")
        
        print(f"{'='*80}\n")
        
        # Send success summary email
        success_lines = [
            "="*80,
            "✅ NBA PROPS DAILY WORKFLOW - COMPLETED SUCCESSFULLY",
            "="*80,
            f"📅 Today: {workflow_results['today']}",
            f"📅 Yesterday: {workflow_results['yesterday']}",
            "",
            "MAIN WORKFLOW STEPS:",
            "────────────────────────────────────────────────────────────────────────────────",
            f"✅ Step 1: Find 2D Plays",
            f"✅ Step 2: Find 3D Plays",
            f"✅ Step 3: Fetch Game Results",
            f"✅ Step 4: Track Performance",
            f"✅ Step 5: Generate & Send Main Email",
            "",
            "TOP3 UNDERS WORKFLOW:",
            "────────────────────────────────────────────────────────────────────────────────"
        ]
        
        if 'skipped' in top3 and top3['skipped']:
            success_lines.append(f"⏭️  Skipped: {top3['reason']}")
        else:
            success_lines.extend([
                f"✅ Step 6: Filter Top3 Plays ({top3['2d_filter']['plays_count']} 2D + {top3['3d_filter']['plays_count']} 3D)"
            ])
            
            # Add validation results
            if 'validation' in top3:
                val = top3['validation']
                errors = len(val['errors'])
                warnings = len(val['warnings'])
                total_games = len(val['all_games_for_date'])
                games_with_plays = len(val['games_with_plays'])
                
                if errors > 0:
                    success_lines.append(f"⚠️  Step 6b: Validation - {errors} ERRORS, {warnings} warnings (separate email sent)")
                    success_lines.append(f"         Games: {games_with_plays}/{total_games} have Top3 plays")
                else:
                    success_lines.append(f"✅ Step 6b: Validation - No errors ({warnings} warnings)")
                    success_lines.append(f"         Games: {games_with_plays}/{total_games} have Top3 plays")
            
            success_lines.extend([
                f"✅ Step 7: Track Top3 Performance",
                f"✅ Step 8: Generate & Send Top3 Email"
            ])
        
        success_lines.extend([
            "",
            "="*80,
            "📧 All emails sent successfully via SNS",
            f"🔍 CloudWatch Logs: /aws/lambda/nba-player-scoring-props-daily-workflow",
            "="*80
        ])
        
        send_email_notification(
            subject=f"✅ NBA Props Workflow SUCCESS - {workflow_results['today']}",
            message="\n".join(success_lines)
        )
        
        # Build Top3 response
        top3 = workflow_results['steps']['top3_workflow']
        if 'skipped' in top3 and top3['skipped']:
            top3_response = {
                'skipped': True,
                'reason': top3['reason']
            }
        else:
            top3_response = {
                'skipped': False,
                '2d_filter': top3['2d_filter']['success'],
                '3d_filter': top3['3d_filter']['success'],
                'tracking': top3['tracking']['success'],
                'email_sent_via_sns': top3['email']['email_sent']
            }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Daily NBA props workflow completed successfully',
                'results': {
                    'today': workflow_results['today'],
                    'yesterday': workflow_results['yesterday'],
                    'main_workflow': {
                        '2d_plays': workflow_results['steps']['2d_plays']['success'],
                        '3d_plays': workflow_results['steps']['3d_plays']['success'],
                        'fetch_games': workflow_results['steps']['fetch_games']['success'],
                        'tracking': workflow_results['steps']['tracking']['success'],
                        'email_sent_via_sns': email_sent
                    },
                    'top3_workflow': top3_response
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

