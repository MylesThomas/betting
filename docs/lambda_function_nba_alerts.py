"""
AWS Lambda Function - NBA Arb Alert (15-minute interval)

This lightweight Lambda function:
1. Fetches live NBA props from The Odds API
2. Finds arbitrage opportunities
3. Sends email alert via SNS if any arb has 5%+ edge

NO git cloning - just fetch, analyze, and alert.

Environment Variables Required:
- ODDS_API_KEY: Your Odds API key (can also use Secrets Manager)
- SECRET_NAME: betting-dashboard-secrets (optional - for secrets manager)
- AWS_REGION_NAME: us-east-2
- SNS_TOPIC_ARN: arn:aws:sns:us-east-2:ACCOUNT_ID:betting-arb-alerts
- MIN_PROFIT_PCT: 5.0 (optional, default 5.0)

Lambda Configuration:
- Runtime: Python 3.12
- Memory: 256 MB (lightweight)
- Timeout: 60 seconds (arb finder is fast ~15s)

Lambda Layers Required:
- betting-dashboard-dependencies (provides pandas, requests)

Schedule (EventBridge):
- Rate: rate(15 minutes)
- Or cron for specific hours: cron(0/15 10-23 * * ? *)  # Every 15 min, 10am-11pm UTC

Author: Myles Thomas
Date: 2025-12-06
"""

import json
import os
import boto3
from datetime import datetime
from zoneinfo import ZoneInfo

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

def get_api_key():
    """Get ODDS_API_KEY from environment or Secrets Manager."""
    # First check environment variable
    api_key = os.environ.get('ODDS_API_KEY')
    if api_key:
        return api_key
    
    # Fall back to Secrets Manager
    secret_name = os.environ.get('SECRET_NAME')
    if secret_name:
        region_name = os.environ.get('AWS_REGION_NAME', 'us-east-2')
        client = boto3.client('secretsmanager', region_name=region_name)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response['SecretString'])
        return secrets.get('ODDS_API_KEY')
    
    raise ValueError("No ODDS_API_KEY found in environment or Secrets Manager")


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


def find_arbs(all_props, min_profit_pct=0.0):
    """Find all arbs with profit >= min_profit_pct (default 0 = all arbs)."""
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
                best_under['under_odds']
            )
            
            arbs.append({
                'player': player,
                'market': market,
                'line': line,
                'game': group['game'].iloc[0],
                'game_time': group['game_time'].iloc[0],
                'profit_pct': arb['profit_pct'],
                'over_odds': int(best_over['over_odds']),
                'over_book': best_over['bookmaker'],
                'under_odds': int(best_under['under_odds']),
                'under_book': best_under['bookmaker'],
                'over_stake': over_stake,
                'under_stake': under_stake
            })
    
    return sorted(arbs, key=lambda x: x['profit_pct'], reverse=True)


# ============================================================================
# EMAIL FORMATTING
# ============================================================================

def format_arb_email(arbs):
    """Format arbs into email body."""
    now = datetime.now(ZoneInfo(TIMEZONE))
    
    lines = [
        "🚨 HIGH-VALUE NBA ARBS FOUND! 🚨",
        "",
        f"Time: {now.strftime('%Y-%m-%d %I:%M %p ET')}",
        f"Arbs found: {len(arbs)}",
        "",
        "=" * 50,
        ""
    ]
    
    for i, arb in enumerate(arbs, 1):
        market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
        
        lines.extend([
            f"#{i} - {arb['profit_pct']:.2f}% PROFIT",
            f"   Player: {arb['player']}",
            f"   Market: {market_display} {arb['line']}",
            f"   Game: {arb['game']}",
            "",
            f"   📈 OVER {arb['line']}: {arb['over_odds']:+d} @ {arb['over_book']}",
            f"   📉 UNDER {arb['line']}: {arb['under_odds']:+d} @ {arb['under_book']}",
            "",
            f"   💰 Stake $100 total:",
            f"      → ${arb['over_stake']:.2f} on OVER @ {arb['over_book']}",
            f"      → ${arb['under_stake']:.2f} on UNDER @ {arb['under_book']}",
            f"      → Guaranteed profit: ${arb['profit_pct']:.2f}",
            "",
            "-" * 50,
            ""
        ])
    
    lines.extend([
        "",
        "⚡ ACT FAST - Lines move quickly!",
        "",
        "Dashboard: https://tqs-props-dashboard.streamlit.app"
    ])
    
    return "\n".join(lines)


# ============================================================================
# LAMBDA HANDLER
# ============================================================================

def lambda_handler(event, context):
    """Main Lambda handler."""
    print("=" * 60)
    print("🏀 NBA Arb Alert Check")
    print(f"Time: {datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)
    
    min_profit = float(os.environ.get('MIN_PROFIT_PCT', '5.0'))
    print(f"Looking for arbs with {min_profit}%+ edge...")
    
    try:
        # Get API key
        api_key = get_api_key()
        
        # Get today's events
        events = get_todays_events(api_key)
        print(f"Found {len(events)} games today")
        
        if not events:
            print("No games today - exiting")
            return {'statusCode': 200, 'body': 'No games today'}
        
        # Fetch props for all games
        all_props = []
        for event in events:
            try:
                props_data = get_event_props(api_key, event['id'])
                props = parse_props(props_data)
                all_props.extend(props)
                print(f"  ✓ {event['away_team']} @ {event['home_team']}: {len(props)} props")
            except Exception as e:
                print(f"  ✗ Error fetching {event['id']}: {e}")
        
        print(f"\nTotal props: {len(all_props)}")
        
        # Find ALL arbs (any profit > 0)
        all_arbs = find_arbs(all_props, min_profit_pct=0.0)
        
        print(f"\n{'='*60}")
        print(f"📊 ALL ARBS FOUND: {len(all_arbs)}")
        print(f"{'='*60}")
        
        if all_arbs:
            for i, arb in enumerate(all_arbs, 1):
                market_display = MARKET_DISPLAY_NAMES.get(arb['market'], arb['market'])
                alert_flag = "🚨" if arb['profit_pct'] >= min_profit else "  "
                print(f"{alert_flag} {i:2d}. {arb['profit_pct']:5.2f}% | {arb['player']:<25s} | {market_display} {arb['line']}")
                print(f"        Over {arb['over_odds']:+4d} @ {arb['over_book']:<12s} | Under {arb['under_odds']:+4d} @ {arb['under_book']}")
        else:
            print("   No arbs found at all.")
        
        print(f"{'='*60}\n")
        
        # Filter for high-value arbs (email threshold)
        high_value_arbs = [a for a in all_arbs if a['profit_pct'] >= min_profit]
        
        print(f"Found {len(high_value_arbs)} arbs with {min_profit}%+ edge (email threshold)")
        
        if high_value_arbs:
            # Send alert!
            subject = f"🚨 {len(high_value_arbs)} NBA Arb(s) Found! Best: {high_value_arbs[0]['profit_pct']:.1f}%"
            message = format_arb_email(high_value_arbs)
            
            print("\n" + "=" * 60)
            print("📧 SENDING ALERT EMAIL")
            print("=" * 60)
            
            send_email(subject, message)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'total_arbs': len(all_arbs),
                    'high_value_arbs': len(high_value_arbs),
                    'best_profit': high_value_arbs[0]['profit_pct'],
                    'alert_sent': True
                })
            }
        else:
            print(f"No arbs with {min_profit}%+ edge - no alert sent")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'total_arbs': len(all_arbs),
                    'high_value_arbs': 0,
                    'alert_sent': False
                })
            }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Send error notification
        error_msg = f"""❌ NBA Arb Alert Check FAILED

Time: {datetime.now(ZoneInfo(TIMEZONE)).isoformat()}
Error: {str(e)}

Check CloudWatch logs for details.
"""
        send_email("❌ NBA Arb Alert Failed", error_msg)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


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

