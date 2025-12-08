"""
AWS Lambda Function - NBA Arb Alert (15-minute interval)

This Lambda function runs every 15 minutes and:
1. Clones the GitHub repo
2. Fetches live NBA props from The Odds API (today's games only, ET timezone)
3. Finds ALL arbitrage opportunities (any profit > 0)
4. Saves output to repo with timestamp (arb_output_YYYYMMDD_HHMMSS.csv)
5. Commits and pushes to GitHub
6. Sends email alert via SNS if any arb has 5%+ edge

OUTPUT FILES:
    - Saved every 15 minutes to data/04_output/nba/arbs/arb_output_YYYYMMDD_HHMMSS.csv
    - Timestamp is in ET timezone (matches game dates)
    - Each file contains best arb per player/market/line at that snapshot
    - Dashboard reads all files for the day and dedupes by player/market/line,
      keeping the BEST expected_profit seen across all snapshots

DEDUPLICATION STRATEGY:
    - Lambda: Saves snapshot with best odds at that moment
    - Dashboard: Reads all day's files, groups by (player, market, line),
      keeps row with highest expected_profit_pct
    - This captures the best opportunity even if lines move

LIVE PROP MARKET BEHAVIOR:
    - Lines change frequently throughout the day
    - Same player/market/line may have different odds each 15-min snapshot
    - By saving all snapshots, we capture when arbs appear/disappear
    - Dashboard shows the BEST historical opportunity for each player/market/line
    - Late in day (after games start/end), fewer arbs expected

Environment Variables Required:
- GITHUB_REPO_URL: https://github.com/MylesThomas/betting.git
- GITHUB_USERNAME: MylesThomas
- GITHUB_EMAIL: mylescgthomas@gmail.com
- SECRET_NAME: betting-dashboard-secrets
- AWS_REGION_NAME: us-east-2
- SNS_TOPIC_ARN: arn:aws:sns:us-east-2:ACCOUNT_ID:betting-arb-alerts (optional)
- MIN_PROFIT_PCT: 5.0 (optional, default 5.0)

Secrets Required (in AWS Secrets Manager):
- ODDS_API_KEY: Your Odds API key
- GITHUB_TOKEN: Your GitHub Personal Access Token

Lambda Configuration:
- Runtime: Python 3.12
- Memory: 512 MB (needs space for git clone)
- Timeout: 120 seconds (git operations take time)
- Ephemeral storage: 1024 MB

Lambda Layers Required:
- git-lambda2 (provides git binaries)
- betting-dashboard-dependencies (provides pandas, requests)

Schedule (EventBridge):
- Rate: rate(15 minutes)
- Or cron for game hours only: cron(0/15 15-3 * * ? *)  # 10am-10pm ET

Author: Myles Thomas
Date: 2025-12-06
"""

import json
import os
import subprocess
import boto3
from datetime import datetime
from zoneinfo import ZoneInfo
from botocore.exceptions import ClientError

# These come from Lambda layer
import requests
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'basketball_nba'
REGIONS = 'us'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'
TIMEZONE = 'America/New_York'

# Markets to check
MARKETS = 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists'

# Market display names
MARKET_DISPLAY_NAMES = {
    'player_threes': 'Threes',
    'player_points': 'Points',
    'player_rebounds': 'Rebounds',
    'player_assists': 'Assists',
    'player_blocks': 'Blocks',
    'player_steals': 'Steals',
    'player_double_double': 'Double-Double',
    'player_triple_double': 'Triple-Double',
    'player_points_rebounds_assists': 'Pts+Reb+Ast'
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_secrets():
    """
    Fetch secrets from AWS Secrets Manager.
    
    Returns:
        dict: Contains ODDS_API_KEY and GITHUB_TOKEN
    """
    # For local testing, check environment variables first
    odds_key = os.environ.get('ODDS_API_KEY')
    github_token = os.environ.get('GITHUB_TOKEN')
    
    if odds_key:
        return {'ODDS_API_KEY': odds_key, 'GITHUB_TOKEN': github_token}
    
    # Fetch from Secrets Manager
    secret_name = os.environ.get('SECRET_NAME', 'betting-dashboard-secrets')
    region_name = os.environ.get('AWS_REGION_NAME', 'us-east-2')
    
    client = boto3.client('secretsmanager', region_name=region_name)
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise Exception(f"Failed to retrieve secret: {e}")
    
    return json.loads(response['SecretString'])


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
        print(f"STDOUT: {result.stdout[:500]}")  # Truncate long output
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")
    
    return result.stdout, result.stderr, result.returncode


def send_email(subject, message):
    """Send email via SNS."""
    topic_arn = os.environ.get('SNS_TOPIC_ARN')
    if not topic_arn:
        print("⚠️  No SNS_TOPIC_ARN - skipping email")
        return
    
    region = os.environ.get('AWS_REGION_NAME', 'us-east-2')
    sns = boto3.client('sns', region_name=region)
    
    response = sns.publish(
        TopicArn=topic_arn,
        Subject=subject[:100],  # SNS subject limit
        Message=message
    )
    print(f"✅ Email sent (MessageId: {response['MessageId']})")


def american_to_probability(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_arb(over_odds, under_odds):
    """Calculate arb metrics."""
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    total_prob = over_prob + under_prob
    
    is_arb = total_prob < 1.0
    profit_pct = ((1 / total_prob) - 1) * 100 if total_prob > 0 else 0
    
    return {
        'is_arb': is_arb,
        'profit_pct': profit_pct,
        'total_prob': total_prob
    }


def calculate_stakes(over_odds, under_odds, total=100):
    """Calculate optimal stake allocation."""
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    
    over_stake = (over_prob / (over_prob + under_prob)) * total
    under_stake = (under_prob / (over_prob + under_prob)) * total
    
    return round(over_stake, 2), round(under_stake, 2)


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_todays_events(api_key):
    """Get today's NBA events."""
    url = f"{API_BASE_URL}/sports/{SPORT}/events"
    response = requests.get(url, params={'apiKey': api_key}, verify=False)
    response.raise_for_status()
    
    events = response.json()
    
    # Filter for today
    tz = ZoneInfo(TIMEZONE)
    today = datetime.now(tz).date()
    
    todays = []
    for event in events:
        event_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        if event_time.astimezone(tz).date() == today:
            todays.append(event)
    
    return todays


def get_event_props(api_key, event_id):
    """Get props for a single event."""
    url = f"{API_BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT
    }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    return response.json()


def parse_props(event_data):
    """Parse event props into list of dicts."""
    props = []
    
    game = f"{event_data['away_team']} @ {event_data['home_team']}"
    game_time = event_data.get('commence_time')
    
    for bookmaker in event_data.get('bookmakers', []):
        book = bookmaker['key']
        
        for market in bookmaker.get('markets', []):
            market_key = market['key']
            
            player_lines = {}
            for outcome in market.get('outcomes', []):
                player = outcome.get('description', 'Unknown')
                line = outcome.get('point')
                odds = outcome.get('price')
                bet_type = outcome.get('name')
                
                key = (player, line)
                if key not in player_lines:
                    player_lines[key] = {
                        'player': player,
                        'market': market_key,
                        'line': line,
                        'bookmaker': book,
                        'game': game,
                        'game_time': game_time
                    }
                
                if bet_type == 'Over':
                    player_lines[key]['over_odds'] = odds
                elif bet_type == 'Under':
                    player_lines[key]['under_odds'] = odds
            
            props.extend(player_lines.values())
    
    return props


def find_arbs(all_props, min_profit_pct=0.0, total_stake=100.0):
    """
    Find all arbs with profit >= min_profit_pct (default 0 = all arbs).
    
    Returns list of dicts with full arb info compatible with dashboard schema.
    """
    if not all_props:
        return []
    
    df = pd.DataFrame(all_props)
    
    arbs = []
    
    for (market, player, line), group in df.groupby(['market', 'player', 'line']):
        overs = group[group['over_odds'].notna()]
        unders = group[group['under_odds'].notna()]
        
        if overs.empty or unders.empty:
            continue
        
        best_over_idx = overs['over_odds'].idxmax()
        best_under_idx = unders['under_odds'].idxmax()
        
        best_over = overs.loc[best_over_idx]
        best_under = unders.loc[best_under_idx]
        
        arb = calculate_arb(best_over['over_odds'], best_under['under_odds'])
        
        if arb['is_arb'] and arb['profit_pct'] >= min_profit_pct:
            over_stake, under_stake = calculate_stakes(
                best_over['over_odds'], 
                best_under['under_odds'],
                total=total_stake
            )
            
            # Calculate returns
            over_odds = best_over['over_odds']
            under_odds = best_under['under_odds']
            
            if over_odds > 0:
                over_return = over_stake * (1 + over_odds / 100)
            else:
                over_return = over_stake * (1 + 100 / abs(over_odds))
            
            if under_odds > 0:
                under_return = under_stake * (1 + under_odds / 100)
            else:
                under_return = under_stake * (1 + 100 / abs(under_odds))
            
            guaranteed_profit = min(over_return, under_return) - total_stake
            
            # Build recommendation string
            recommendation = f"Bet ${over_stake:.2f} Over @ {best_over['bookmaker']}, ${under_stake:.2f} Under @ {best_under['bookmaker']}"
            
            arbs.append({
                'player': player,
                'market': market,
                'line': line,
                'best_over_odds': int(over_odds),
                'best_over_book': best_over['bookmaker'],
                'best_over_implied': arb['total_prob'] - american_to_probability(under_odds),  # over prob
                'best_under_odds': int(under_odds),
                'best_under_book': best_under['bookmaker'],
                'best_under_implied': american_to_probability(under_odds),
                'total_prob': arb['total_prob'],
                'expected_profit_pct': arb['profit_pct'],
                'is_arb': arb['is_arb'],
                'over_stake': over_stake,
                'under_stake': under_stake,
                'over_return': round(over_return, 2),
                'under_return': round(under_return, 2),
                'guaranteed_profit': round(guaranteed_profit, 2),
                'total_wager': total_stake,
                'recommendation': recommendation,
                'game': group['game'].iloc[0],
                'game_time': group['game_time'].iloc[0],
                'num_bookmakers': len(group['bookmaker'].unique())
            })
    
    return sorted(arbs, key=lambda x: x['expected_profit_pct'], reverse=True)


def save_arb_output(arbs_df, work_dir, timestamp=None):
    """
    Save arb output to the cloned repo directory.
    
    Args:
        arbs_df: DataFrame with arb data
        work_dir: Path to cloned repo (e.g., /tmp/betting)
        timestamp: Optional datetime for filename (defaults to now in ET)
    
    Returns:
        str: Path where file was saved
    """
    if timestamp is None:
        timestamp = datetime.now(ZoneInfo(TIMEZONE))
    
    filename = f"arb_output_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    
    output_dir = os.path.join(work_dir, 'data/04_output/nba/arbs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    arbs_df.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")
    return output_path


# ============================================================================
# EMAIL FORMATTING
# ============================================================================

def format_arb_email(high_value_arbs, other_arbs):
    """Format arbs into email body with high-value at top, others below."""
    now = datetime.now(ZoneInfo(TIMEZONE))
    total_arbs = len(high_value_arbs) + len(other_arbs)
    
    # Header depends on whether we have high-value arbs
    if high_value_arbs:
        header = "🚨 high-value nba arbs found! 🚨"
        arb_summary = f"arbs found: {total_arbs} ({len(high_value_arbs)} high-value)"
    else:
        header = "📊 nba arb scan complete"
        arb_summary = f"arbs found: {total_arbs} (none above threshold)"
    
    lines = [
        header,
        "",
        f"time: {now.strftime('%Y-%m-%d %I:%M %p ET')}",
        arb_summary,
        "",
        "=" * 50,
        ""
    ]
    
    # High-value arbs (detailed format)
    if high_value_arbs:
        for i, arb in enumerate(high_value_arbs, 1):
            market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
            
            lines.extend([
                f"#{i} - {arb['expected_profit_pct']:.2f}% PROFIT",
                f"   Player: {arb['player']}",
                f"   Market: {market_display} {arb['line']}",
                f"   Game: {arb['game']}",
                "",
                f"   📈 OVER {arb['line']}: {arb['best_over_odds']:+d} @ {arb['best_over_book']}",
                f"   📉 UNDER {arb['line']}: {arb['best_under_odds']:+d} @ {arb['best_under_book']}",
                "",
                f"   💰 Stake $100 total:",
                f"      → ${arb['over_stake']:.2f} on OVER @ {arb['best_over_book']}",
                f"      → ${arb['under_stake']:.2f} on UNDER @ {arb['best_under_book']}",
                f"      → Guaranteed profit: ${arb['guaranteed_profit']:.2f}",
                "",
                "-" * 50,
                ""
            ])
    
    # Other arbs (compact format)
    if other_arbs:
        if high_value_arbs:
            lines.extend([
                "",
                "=" * 50,
                "📋 other arbs (below threshold):",
                "=" * 50,
                ""
            ])
        
        for i, arb in enumerate(other_arbs, len(high_value_arbs) + 1):
            market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
            lines.extend([
                f"#{i} - {arb['expected_profit_pct']:.2f}% | {arb['player']} | {market_display} {arb['line']}",
                f"     Game: {arb['game']}",
                f"     Over {arb['best_over_odds']:+d} @ {arb['best_over_book']} | Under {arb['best_under_odds']:+d} @ {arb['best_under_book']}",
                ""
            ])
    
    lines.extend([
        "",
        "⚡ act fast - lines move quickly!",
        "",
        "Dashboard: https://tqs-props-dashboard.streamlit.app"
    ])
    
    return "\n".join(lines)


# ============================================================================
# LAMBDA HANDLER
# ============================================================================

def lambda_handler(event, context):
    """Main Lambda handler - fetches arbs, saves to git, sends alerts."""
    now = datetime.now(ZoneInfo(TIMEZONE))
    work_dir = '/tmp/betting'
    
    print("=" * 60)
    print("🏀 NBA Arb Alert Check (15-min)")
    print(f"Time: {now.strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)
    
    min_profit = float(os.environ.get('MIN_PROFIT_PCT', '5.0'))
    print(f"Looking for arbs with {min_profit}%+ edge...")
    
    try:
        # Step 1: Get secrets
        print("\n📊 Step 1: Fetching secrets...")
        secrets = get_secrets()
        odds_api_key = secrets['ODDS_API_KEY']
        github_token = secrets.get('GITHUB_TOKEN')
        print("✅ Secrets retrieved")
        
        # Step 2: Clone repo (only in Lambda, skip locally)
        is_lambda = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None
        
        if is_lambda and github_token:
            print("\n📦 Step 2: Cloning GitHub repository...")
            github_repo_url = os.environ.get('GITHUB_REPO_URL', 'https://github.com/MylesThomas/betting.git')
            github_username = os.environ.get('GITHUB_USERNAME', 'MylesThomas')
            github_email = os.environ.get('GITHUB_EMAIL', 'mylescgthomas@gmail.com')
            
            repo_url_with_token = github_repo_url.replace(
                'https://',
                f'https://{github_username}:{github_token}@'
            )
            
            run_command(['rm', '-rf', work_dir])
            stdout, stderr, code = run_command(['git', 'clone', '--depth', '1', repo_url_with_token, work_dir])
            
            if code != 0:
                raise Exception(f"Git clone failed: {stderr}")
            
            run_command(['git', 'config', 'user.name', github_username], cwd=work_dir)
            run_command(['git', 'config', 'user.email', github_email], cwd=work_dir)
            print("✅ Repository cloned")
        else:
            # Local testing - use project root
            work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"📂 Using local directory: {work_dir}")
        
        # Step 3: Fetch today's events (ET timezone)
        print("\n🔍 Step 3: Fetching today's NBA events...")
        events = get_todays_events(odds_api_key)
        print(f"Found {len(events)} games today (ET)")
        
        if not events:
            print("No games today - saving empty file")
            empty_df = pd.DataFrame(columns=[
                'player', 'market', 'line', 'best_over_odds', 'best_over_book',
                'best_over_implied', 'best_under_odds', 'best_under_book',
                'best_under_implied', 'total_prob', 'expected_profit_pct', 'is_arb',
                'over_stake', 'under_stake', 'over_return', 'under_return',
                'guaranteed_profit', 'total_wager', 'recommendation', 'game',
                'game_time', 'num_bookmakers'
            ])
            save_arb_output(empty_df, work_dir, timestamp=now)
            return {'statusCode': 200, 'body': 'No games today'}
        
        # Step 4: Fetch props for all games
        print("\n📥 Step 4: Fetching props for each game...")
        all_props = []
        for event in events:
            try:
                props_data = get_event_props(odds_api_key, event['id'])
                props = parse_props(props_data)
                all_props.extend(props)
                print(f"  ✓ {event['away_team']} @ {event['home_team']}: {len(props)} props")
            except Exception as e:
                print(f"  ✗ Error fetching {event['id']}: {e}")
        
        print(f"\nTotal props: {len(all_props)}")
        
        # Step 5: Find arbs
        print("\n🔍 Step 5: Finding arbitrage opportunities...")
        all_arbs = find_arbs(all_props, min_profit_pct=0.0)
        
        print(f"\n{'='*60}")
        print(f"📊 ALL ARBS FOUND: {len(all_arbs)}")
        print(f"{'='*60}")
        
        if all_arbs:
            for i, arb in enumerate(all_arbs, 1):
                market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
                alert_flag = "🚨" if arb['expected_profit_pct'] >= min_profit else "  "
                print(f"{alert_flag} {i:2d}. {arb['expected_profit_pct']:5.2f}% | {arb['player']:<25s} | {market_display} {arb['line']}")
                print(f"        Over {arb['best_over_odds']:+4d} @ {arb['best_over_book']:<12s} | Under {arb['best_under_odds']:+4d} @ {arb['best_under_book']}")
        else:
            print("   No arbs found at all.")
        
        print(f"{'='*60}\n")
        
        # Step 6: Save output file (sorted by expected_profit_pct descending)
        print("💾 Step 6: Saving output file...")
        if all_arbs:
            arbs_df = pd.DataFrame(all_arbs)
            arbs_df = arbs_df.sort_values('expected_profit_pct', ascending=False)
        else:
            arbs_df = pd.DataFrame(columns=[
                'player', 'market', 'line', 'best_over_odds', 'best_over_book',
                'best_over_implied', 'best_under_odds', 'best_under_book',
                'best_under_implied', 'total_prob', 'expected_profit_pct', 'is_arb',
                'over_stake', 'under_stake', 'over_return', 'under_return',
                'guaranteed_profit', 'total_wager', 'recommendation', 'game',
                'game_time', 'num_bookmakers'
            ])
        
        output_path = save_arb_output(arbs_df, work_dir, timestamp=now)
        
        # Step 7: Commit and push (only in Lambda)
        if is_lambda and github_token:
            print("\n📤 Step 7: Committing and pushing to GitHub...")
            
            run_command(['git', 'add', 'data/04_output/nba/arbs/*.csv'], cwd=work_dir)
            
            stdout, stderr, code = run_command(['git', 'status', '--porcelain'], cwd=work_dir)
            
            if stdout.strip():
                commit_msg = f"arb-alert: {now.strftime('%Y-%m-%d %H:%M')} ET - {len(all_arbs)} arbs"
                run_command(['git', 'commit', '-m', commit_msg], cwd=work_dir)
                
                stdout, stderr, code = run_command(['git', 'push'], cwd=work_dir)
                
                if code != 0:
                    print(f"⚠️  Git push failed (non-fatal): {stderr}")
                else:
                    print("✅ Pushed to GitHub")
            else:
                print("ℹ️  No changes to commit")
        
        # Step 8: Send email alert if any arbs found
        high_value_arbs = [a for a in all_arbs if a['expected_profit_pct'] >= min_profit]
        other_arbs = [a for a in all_arbs if a['expected_profit_pct'] < min_profit]
        
        print(f"\nFound {len(high_value_arbs)} arbs with {min_profit}%+ edge (email threshold)")
        
        if all_arbs:
            # Subject depends on whether we have high-value arbs
            if high_value_arbs:
                best_profit = high_value_arbs[0]['expected_profit_pct']
                subject = f"🚨 {len(high_value_arbs)} NBA ARB(S) FOUND! BEST: {best_profit:.1f}%"
            else:
                best_profit = all_arbs[0]['expected_profit_pct']
                subject = f"📊 nba arb scan: {len(all_arbs)} arbs found (best: {best_profit:.1f}%)"
            
            message = format_arb_email(high_value_arbs, other_arbs)
            
            print("\n" + "=" * 60)
            print("📧 SENDING ALERT EMAIL")
            print("=" * 60)
            
            send_email(subject, message)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'total_arbs': len(all_arbs),
                    'high_value_arbs': len(high_value_arbs),
                    'best_profit': best_profit,
                    'alert_sent': True,
                    'output_file': output_path
                })
            }
        else:
            print("No arbs found - no alert sent")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'total_arbs': 0,
                    'high_value_arbs': 0,
                    'alert_sent': False,
                    'output_file': output_path
                })
            }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"""❌ NBA Arb Alert Check FAILED

Time: {now.isoformat()}
Error: {str(e)}

Check CloudWatch logs for details.
"""
        send_email("❌ NBA Arb Alert Failed", error_msg)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    
    finally:
        # Clean up in Lambda
        if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
            print("\n🧹 Cleaning up...")
            run_command(['rm', '-rf', '/tmp/betting'])


# ============================================================================
# LOCAL TESTING
# ============================================================================

if __name__ == "__main__":
    # For local testing
    import ssl
    import urllib3
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Mock event/context
    result = lambda_handler({}, None)
    print(f"\nResult: {result}")

