"""
Track Live Betting Lines During NBA Games (In-Play Odds)

Lambda function name: track-live-odds-per-minute

PURPOSE:
Capture live spreads and moneylines every minute during games to build dataset 
of real-time win probability expectations. This complements the pre-game hourly 
line movement tracker by focusing on IN-GAME odds.

WHY THIS MATTERS:
Live lines reveal:
- Real-time win probability as game unfolds
- Market's reaction to momentum shifts, runs, injuries
- Sharp money adjusting to game flow
- Closing live line value (compare final live line to actual result)
- In-game betting opportunities

CORE DIFFERENCES FROM PRE-GAME TRACKER:
Pre-game tracker (lambda_function_track_game_line_movements.py):
- Runs every 1 HOUR
- Tracks games BEFORE they start
- Spreads only
- ~24 API calls/day per sport

Live tracker (this script):
- Runs every 1 MINUTE
- Tracks games IN PROGRESS
- Spreads + Moneylines
- Only calls Odds API when games are live (ESPN-first optimization)

FUNCTIONALITY:
1. Check ESPN API (free) for live games
2. If no live games: Skip Odds API call (saves API credits ~73%)
3. If live games exist: Fetch odds from Odds API
4. Parse odds data (one record per game-bookmaker)
5. Parse ESPN data (one record per game with scores/status)
6. Write 2 separate Parquet files to S3 (immutable, one file per invocation)

OPTIMIZATION - ESPN-First Check + Live-Only Filtering:
- ESPN API is FREE and returns game status
- Check ESPN first to count live games (game_status == 'in')
- Only call Odds API if live games exist
- Filter Odds API results to ONLY include live games (default behavior)
- Typical savings: ~73% of API costs
  * Without: 1,440 calls/day (every minute) = ~$630/month
  * With: ~180 calls/day (only when live) = ~$162/month
  * Monthly savings: ~$468
- Note: Can track upcoming games by setting TRACK_UPCOMING_GAMES="true" (not recommended)

DATA STRUCTURE:
Stored as PARQUET files (separate for odds and ESPN data).

File naming: {YYYYMMDD}_{HHMMSS}.parquet (e.g., 20260201_170400.parquet)
Timestamp in Eastern Time (America/New_York)

Storage structure:
s3://nba-betting-mt/data/01_input/live_odds/
├── the-odds-api/
│   ├── 20260201_170400.parquet
│   ├── 20260201_170500.parquet
│   └── ...
└── espn/
    ├── 20260201_170400.parquet
    ├── 20260201_170500.parquet
    └── ...

Why separate files?
- Clean separation of concerns (odds vs game state)
- No merge complexity (join on-the-fly during analysis with DuckDB)
- Easier to query specific data (e.g., "show me all live scores")

Why one file per invocation?
- No append complexity (Parquet doesn't natively support append)
- Immutable files (no risk of corruption)
- Fast writes (no read-before-write)
- DuckDB handles many files efficiently with glob patterns

CRITICAL: Each odds row is ONE bookmaker's line for ONE game at ONE point in time.
We iterate: game → bookmaker, storing each book's line separately.

Odds data columns:
- query_time: When we started the API call (UTC ISO format)
- collection_timestamp: Same as query_time (for consistency)
- game_id: Unique game identifier from Odds API
- away_team: Away team name
- home_team: Home team name
- commence_time: Scheduled start time (from Odds API)
- bookmaker: Bookmaker key (e.g., 'fanduel', 'draftkings')
- bookmaker_last_update: When bookmaker last updated this line
- away_spread: Raw spread for away team (e.g., -3.5)
- away_spread_price: American odds for away spread (e.g., -110)
- home_spread: Raw spread for home team (e.g., +3.5)
- home_spread_price: American odds for home spread (e.g., -110)
- away_ml: Moneyline for away team (e.g., -150, +200)
- home_ml: Moneyline for home team (e.g., +130, -250)

ESPN data columns:
- query_time: When we started the API call (UTC ISO format)
- collection_timestamp: Same as query_time (for consistency)
- espn_game_id: Unique game identifier from ESPN
- away_team_espn: Away team name from ESPN
- home_team_espn: Home team name from ESPN
- away_score: Current score for away team
- home_score: Current score for home team
- game_status: 'pre', 'in', or 'post' (from ESPN API)
- game_status_description: Human-readable status
- period: Quarter number
- display_clock: Time remaining display (e.g., "8:45")
- time_remaining_minutes: Calculated minutes remaining (e.g., 8.75)

Note on game_status values:
- 'pre': game hasn't started yet
- 'in': game is currently live/in progress
- 'post': game has finished

QUERYING DATA WITH DUCKDB:
No consolidation needed - DuckDB queries S3 directly with glob patterns:

```python
import duckdb
con = duckdb.connect()

# All odds for Feb 1, live games only
df = con.execute(\"\"\"
    SELECT * 
    FROM 's3://nba-betting-mt/data/01_input/live_odds/the-odds-api/20260201_*.parquet'
\"\"\").df()

# Join odds + ESPN to see line movement with scores
df = con.execute(\"\"\"
    SELECT 
        o.game_id,
        o.away_team,
        o.bookmaker,
        o.away_spread,
        o.away_ml,
        e.away_score,
        e.home_score,
        e.period,
        e.time_remaining_minutes,
        o.collection_timestamp
    FROM 's3://nba-betting-mt/data/01_input/live_odds/the-odds-api/20260201_*.parquet' o
    LEFT JOIN 's3://nba-betting-mt/data/01_input/live_odds/espn/20260201_*.parquet' e
      ON o.collection_timestamp = e.collection_timestamp
      AND o.away_team = e.away_team_espn
      AND o.home_team = e.home_team_espn
    WHERE o.game_id = 'abc123'
    ORDER BY o.collection_timestamp
\"\"\").df()
```

API ENDPOINTS:

Odds API (live odds):
  GET https://api.the-odds-api.com/v4/sports/{sport}/odds
  Parameters:
    - regions=us
    - markets=spreads,h2h
    - oddsFormat=american
    - bookmakers=fanduel,draftkings,betmgm,etc. (comma-separated)

ESPN API (game scores/status):
  GET http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  
  Returns:
    - Live scores
    - Period (quarter)
    - Time remaining
    - Game status (pre, in, post)
  
  Note: Free, no API key required (but rate-limited)

COST ANALYSIS:

Odds API (WITH ESPN-first optimization):
  Typical NBA day: 10 games × 3 hours × 60 min = 180 requests/day
  Monthly: 180 × 30 = 5,400 requests
  Cost: ~$162/month (at $0.03/request)

AWS Lambda:
  - 1 invocation/min when EventBridge triggers
  - Free tier: 1M requests/month (more than enough)
  - Execution time: ~3-5 seconds
  - Free tier: 400,000 GB-seconds/month (more than enough)

S3:
  - ~5-10MB/day × 30 days = 150-300MB/month
  - Cost: ~$0.01/month (negligible)

Total: ~$162/month (mostly Odds API)

Environment Variables Required:
- ODDS_API_KEY: Your Odds API key
- TRACK_UPCOMING_GAMES: (Optional) Set to "true" to track upcoming games in addition to live games.
  Default: "false" (only track live games)

IAM Permissions Required:
- s3:PutObject (for nba-betting-mt bucket)

Lambda Configuration:
- Runtime: Python 3.11
- Memory: 512 MB (pandas needs this)
- Timeout: 30 seconds
- Ephemeral storage: 512 MB (default is fine)

Lambda Layers Required:
Option 1 (Recommended): AWS SDK for pandas (AWS-managed, optimized for Lambda)
- Includes: pandas, numpy, pyarrow, boto3, s3fs, and more
- Python 3.11 (us-east-2): arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:25
- Full list of ARNs: https://aws-sdk-pandas.readthedocs.io/en/stable/layers.html
- Note: Still need to add 'requests' separately (small layer or include in deployment package)

EventBridge Schedule Setup (for automated per-minute execution):
- Navigate to: AWS Console → Amazon EventBridge → Rules → Create rule
- Define rule detail:
    - Name: track-live-odds-per-minute-scheduler
    - Description: Track live NBA odds every minute
- Define schedule:
    - Schedule expression: rate(1 minute)
    - Runs continuously (Lambda will skip Odds API if no live games detected)
- Select target:
    - Target type: AWS Lambda function
    - Function: track-live-odds-per-minute
    - IMPORTANT - Execution role:
      * Select: "Create a new role for this specific resource"
      * DO NOT reuse existing EventBridge roles
      * This ensures the rule has proper permissions to invoke THIS Lambda
      * Prevents "FailedInvocation" errors from permission issues
- Review + Create

TESTING:

Local testing:
  python scripts/lambda_function_track_live_odds.py --sport nba --prod-run

Lambda testing:
  1. Manual invoke via AWS Console (use empty test event {})
  2. Check CloudWatch logs for output
  3. Verify S3 files created:
     aws s3 ls s3://nba-betting-mt/data/01_input/live_odds/the-odds-api/
     aws s3 ls s3://nba-betting-mt/data/01_input/live_odds/espn/

EXPECTED BEHAVIOR:

When live games exist (e.g., 7 PM ET):
```
✅ ESPN API: Retrieved 10 games (2 live)
✅ Odds API: Retrieved 10 games, 220 records
💾 Wrote 220 records to s3://.../the-odds-api/20260201_190000.parquet
💾 Wrote 10 records to s3://.../espn/20260201_190000.parquet
```
API calls used: 1

When no live games (e.g., 3 AM ET):
```
✅ ESPN API: Retrieved 10 games (0 live)
ℹ️ No live games - skipping Odds API call (saved 1 API credit)
ℹ️ No live games - skipping file write (no data to save)
```
API calls used: 0 ✨
Files written: 0 ✨

ANALYSIS USE CASES:
After collecting data, analyze:
1. Live line accuracy: How well does live spread predict final margin?
2. Win probability calibration: Do 60% ML favorites win 60% of time?
3. In-game momentum: When do lines move most (runs, quarters, halftime)?
4. Closing live line value: Compare final live line to actual result
5. Steam detection: Rapid moves across multiple books (>0.5 pts in 1 min)
6. Arbitrage: Cross-book differences in live lines
7. ML vs Spread correlation: Do they agree on win probability?

MODULAR DESIGN:
Currently implemented:
  ✅ NBA (basketball_nba)
     - Team normalization: "Los Angeles Clippers" (Odds API) → "LA Clippers" (ESPN)
  ✅ NCAAB (basketball_ncaab)
     - Team normalization: "St" → "State", "Univ." → "University", "Miss" → "Mississippi"

Not yet implemented (but ready for extension):
  ⚠️ NFL (americanfootball_nfl)
  ⚠️ NCAAF (americanfootball_ncaaf)

To add new sport:
  1. Add sport key to SUPPORTED_SPORTS dict
  2. Add ESPN endpoint for that sport
  3. Create team name normalization mapping if needed (see src/nba_team_name_mapping.py)
  4. Test during live games

AUTHOR: Thomas Myles
CREATED: 2026-01-31
UPDATED: 2026-02-16 (Added NBA team name normalization for Clippers)
"""

import os
import sys
import argparse
import warnings
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Suppress SSL warnings for Lambda (may have cert issues)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Find project root (look for .git or .gitignore)
current_file = Path(__file__).resolve()
project_root = current_file.parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

# Import team name normalization
try:
    from src.ncaab_team_name_mapping import normalize_ncaab_team_name
except ImportError:
    print("⚠️  WARNING: ncaab_team_name_mapping.py not found - using fallback normalization")
    # Fallback if import fails (Lambda environment)
    def normalize_ncaab_team_name(name: str) -> str:
        """Fallback normalization if mapping module not available."""
        # Basic normalization rules
        normalized = name
        if " St " in normalized:
            normalized = normalized.replace(" St ", " State ")
        if "Univ." in normalized:
            normalized = normalized.replace("Univ.", "University")
        if normalized.startswith("Miss "):
            normalized = normalized.replace("Miss ", "Mississippi ", 1)
        return normalized

try:
    from src.nba_team_name_mapping import normalize_nba_team_name
except ImportError:
    print("⚠️  WARNING: nba_team_name_mapping.py not found - using fallback normalization")
    # Fallback if import fails (Lambda environment)
    def normalize_nba_team_name(name: str) -> str:
        """Fallback normalization if mapping module not available."""
        # Handle LA Clippers difference between Odds API and ESPN
        if name == "Los Angeles Clippers":
            return "LA Clippers"
        return name


# =============================================================================
# CONFIGURATION
# =============================================================================

# API Keys and endpoints
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
ESPN_NBA_SCOREBOARD = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
ESPN_NCAAB_SCOREBOARD = 'http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'

# Data paths
DATA_ROOT = project_root / 'data'

# S3 Configuration
# Bucket varies by sport
S3_BUCKETS = {
    'nba': 'nba-betting-mt',
    'ncaab': 'ncaab-betting-mt',
}
S3_BASE_PATH = 'data/01_input/live_odds'
IS_LAMBDA = bool(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))

# Tracking Configuration
# By default, only track games that are currently live (game_status == 'in')
# Set TRACK_UPCOMING_GAMES="true" to include upcoming games (not recommended - wastes API calls)
TRACK_UPCOMING_GAMES = os.getenv('TRACK_UPCOMING_GAMES', 'false').lower() == 'true'

# Sports
SPORT_NBA = 'basketball_nba'
SPORT_NCAAB = 'basketball_ncaab'
# SPORT_NFL = 'americanfootball_nfl'
# SPORT_NCAAF = 'americanfootball_ncaaf'

SUPPORTED_SPORTS = {
    'nba': SPORT_NBA,
    'ncaab': SPORT_NCAAB,
    # 'nfl': SPORT_NFL,      # NOT IMPLEMENTED YET
    # 'ncaaf': SPORT_NCAAF,  # NOT IMPLEMENTED YET
}

# Display timezone
DISPLAY_TIMEZONE = 'America/New_York' # Eastern Timezone


# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    # Status
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'refresh': '🔄',
    'save': '💾',
    
    # Analysis
    'chart': '📊',
    'fire': '🔥',
    'calendar': '📅',
    'money': '💰',
    'clock': '⏰',
    
    # Sports
    'nba': '🏀',
    'nfl': '🏈',
}


# =============================================================================
# HELPER FUNCTIONS (in execution order)
# =============================================================================

def get_current_time_et():
    """Get current time in Eastern Time for display."""
    return datetime.now(ZoneInfo(DISPLAY_TIMEZONE))


def ml_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -150, +200)
    
    Returns:
        Implied probability as decimal (e.g., 0.60 for 60%)
    
    Examples:
        -150 → 0.60 (60% - favorite)
        +130 → 0.435 (43.5% - underdog)
    """
    if odds < 0:
        # Favorite: |odds| / (|odds| + 100)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog: 100 / (odds + 100)
        return 100 / (odds + 100)


def remove_vig(away_prob: float, home_prob: float) -> tuple:
    """
    Remove vig to get true probabilities (normalized to sum to 100%).
    
    Args:
        away_prob: Raw implied probability for away team (with vig)
        home_prob: Raw implied probability for home team (with vig)
    
    Returns:
        Tuple of (away_true_prob, home_true_prob) that sum to 1.0
    
    Example:
        Raw: away 0.60, home 0.435 → total 1.035 (3.5% vig)
        True: away 0.580, home 0.420 (sum to 1.0)
    """
    total = away_prob + home_prob
    return (away_prob / total, home_prob / total)


def calculate_vig_adjusted_spread(spread: float, price: int) -> float:
    """
    Calculate vig-adjusted spread score.
    
    Reuses methodology from pre-game tracker (lambda_function_track_game_line_movements.py).
    Adjusts spread based on price to reveal bookmaker's "true" line.
    
    Args:
        spread: Raw spread (e.g., -2.5)
        price: American odds (e.g., -110, -130, +100)
    
    Returns:
        Adjusted spread incorporating vig
    
    Examples:
        -2.5 @ -110 → -2.50 (baseline)
        -2.5 @ -130 → -2.53 (high vig = line shaded up)
        -2.5 @ +100 → -2.485 (low vig = line shaded down)
    """
    # Calculate vig distance from baseline (-110)
    vig_cents = (price - (-110)) / 10  # in dimes
    
    # Apply tiered adjustment (handles extremes)
    if vig_cents <= -3.0:
        # Extreme high vig (price >= -140)
        base_adjustment = -3.0 * 0.015
        excess = vig_cents - (-3.0)
        adjustment = base_adjustment + (excess * 0.025)
    elif vig_cents >= 3.0:
        # Extreme low vig (price <= +80)
        base_adjustment = 3.0 * 0.015
        excess = vig_cents - 3.0
        adjustment = base_adjustment + (excess * 0.025)
    else:
        # Normal range: linear
        adjustment = vig_cents * 0.015
    
    return spread + adjustment


def fetch_live_odds(sport: str) -> list:
    """
    Fetch live odds from The Odds API.
    
    Gets spreads + moneylines for ALL available bookmakers (no filtering).
    
    Args:
        sport: Sport key (e.g., 'basketball_nba')
    
    Returns:
        List of game dicts from API
    
    Raises:
        requests.HTTPError if API call fails
    """
    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'spreads,h2h',  # h2h = moneylines
        'oddsFormat': 'american',
        # NO bookmakers filter - get ALL books
    }
    
    print(f"{EMOJI['refresh']} Fetching live odds from Odds API...")
    print(f"   Endpoint: {url}")
    print(f"   Markets: spreads, h2h (moneylines)")
    print(f"   Bookmakers: ALL (not filtered)")
    
    response = requests.get(url, params=params, timeout=15, verify=False)
    response.raise_for_status()
    
    games = response.json()
    
    # Log API usage
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')
    print(f"{EMOJI['money']} API Usage: {used} used, {remaining} remaining")
    
    return games


def fetch_game_scores(sport: str = 'nba') -> dict:
    """
    Fetch live game scores from ESPN API.
    
    Gets scores, period, time remaining, game status.
    
    Args:
        sport: 'nba' or 'ncaab'
    
    Returns:
        ESPN scoreboard dict
    
    Raises:
        NotImplementedError if sport not supported
        requests.HTTPError if API call fails
    """
    if sport == 'nba':
        url = ESPN_NBA_SCOREBOARD
    elif sport == 'ncaab':
        url = ESPN_NCAAB_SCOREBOARD
    else:
        raise NotImplementedError(f"ESPN endpoint for {sport} not implemented yet")
    
    print(f"{EMOJI['refresh']} Fetching scores from ESPN API...")
    print(f"   Endpoint: {url}")
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    return response.json()


def match_game_to_score(game: dict, espn_data: dict, sport: str = 'basketball_nba') -> dict:
    """
    Match Odds API game to ESPN score data by team names.
    
    Uses team name normalization to handle format differences between APIs.
    - NCAAB: The Odds API uses abbreviations (St, Univ., Miss) that ESPN expands.
    - NBA: The Odds API uses "Los Angeles Clippers" while ESPN uses "LA Clippers"
    
    Args:
        game: Game dict from Odds API (has away_team, home_team)
        espn_data: ESPN scoreboard dict
        sport: Sport key (e.g., 'basketball_nba', 'basketball_ncaab')
    
    Returns:
        Dict with score info, or empty dict if no match found
        {
            'away_score': int,
            'home_score': int,
            'period': str (e.g., '2Q', '3Q'),
            'time_remaining': float (minutes),
            'game_status': str (e.g., 'in', 'halftime')
        }
    """
    # Get team names from Odds API and normalize for ESPN matching
    odds_away_team = game['away_team']
    odds_home_team = game['home_team']
    
    # Normalize Odds API names to match ESPN format (sport-specific)
    if sport == 'basketball_ncaab':
        away_team_normalized = normalize_ncaab_team_name(odds_away_team)
        home_team_normalized = normalize_ncaab_team_name(odds_home_team)
    else:  # basketball_nba
        away_team_normalized = normalize_nba_team_name(odds_away_team)
        home_team_normalized = normalize_nba_team_name(odds_home_team)
    
    events = espn_data['events']
    
    for event in events:
        competition = event['competitions'][0]
        competitors = competition['competitors']
        
        espn_away = next((c for c in competitors if c['homeAway'] == 'away'), None)
        espn_home = next((c for c in competitors if c['homeAway'] == 'home'), None)
        
        # Get ESPN team names
        espn_away_name = espn_away['team']['displayName']
        espn_home_name = espn_home['team']['displayName']
        
        # Try exact match first (most common case)
        if espn_away_name == odds_away_team and espn_home_name == odds_home_team:
            # Direct match - no normalization needed
            pass
        # Try normalized match
        elif espn_away_name == away_team_normalized and espn_home_name == home_team_normalized:
            # Matched after normalization
            pass
        else:
            # No match for this game
            continue
        
        # Match found - extract score info
        status = event['status']
        
        # Parse time remaining
        time_remaining = None
        if 'displayClock' in status:
            clock_str = status['displayClock']
            try:
                parts = clock_str.split(':')
                if len(parts) == 2:
                    mins = int(parts[0])
                    secs = int(parts[1])
                    time_remaining = mins + secs / 60
            except:
                time_remaining = None
        
        return {
            'away_score': int(espn_away['score']),
            'home_score': int(espn_home['score']),
            'period': str(status.get('period', '')),
            'time_remaining': time_remaining,
            'game_status': status['type']['state'],
        }
    
    # No match found
    return {}
    """
    Match Odds API game to ESPN score data by team names.
    
    Args:
        game: Game dict from Odds API (has away_team, home_team)
        espn_data: ESPN scoreboard dict
    
    Returns:
        Dict with score info, or empty dict if no match found
        {
            'away_score': int,
            'home_score': int,
            'period': str (e.g., '2Q', '3Q'),
            'time_remaining': float (minutes),
            'game_status': str (e.g., 'in', 'halftime')
        }
    """
    away_team = game['away_team']
    home_team = game['home_team']
    
    events = espn_data['events']
    
    for event in events:
        competition = event['competitions'][0]
        competitors = competition['competitors']
        
        espn_away = next((c for c in competitors if c['homeAway'] == 'away'), None)
        espn_home = next((c for c in competitors if c['homeAway'] == 'home'), None)
        
        # Match by team display name
        if (espn_away['team']['displayName'] == away_team and 
            espn_home['team']['displayName'] == home_team):
            
            status = event['status']
            
            # Parse time remaining
            time_remaining = None
            if 'displayClock' in status:
                clock_str = status['displayClock']
                try:
                    parts = clock_str.split(':')
                    if len(parts) == 2:
                        mins = int(parts[0])
                        secs = int(parts[1])
                        time_remaining = mins + secs / 60
                except:
                    time_remaining = None
            
            return {
                'away_score': int(espn_away['score']),
                'home_score': int(espn_home['score']),
                'period': str(status.get('period', '')),
                'time_remaining': time_remaining,
                'game_status': status['type']['state'],
            }
    
    # No match found
    return {}


def create_snapshot_row(game: dict, bookmaker: dict, score_info: dict, 
                       fetched_at: datetime) -> dict:
    """
    Create a single Parquet row for one bookmaker's line.
    
    Args:
        game: Game dict from Odds API
        bookmaker: Bookmaker dict with markets
        score_info: Score dict from match_game_to_score()
        fetched_at: Timestamp when we fetched this data (UTC)
    
    Returns:
        Dict with all columns for Parquet row
    """
    # Extract markets
    spreads_market = next((m for m in bookmaker['markets'] if m['key'] == 'spreads'), None)
    h2h_market = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
    
    # Find away/home outcomes in spreads
    away_spread_outcome = next((o for o in spreads_market['outcomes'] if o['name'] == game['away_team']), None)
    home_spread_outcome = next((o for o in spreads_market['outcomes'] if o['name'] == game['home_team']), None)
    
    away_spread = away_spread_outcome['point']
    away_spread_price = away_spread_outcome['price']
    home_spread = home_spread_outcome['point']
    home_spread_price = home_spread_outcome['price']
    
    # Calculate vig-adjusted spreads
    away_adjusted_spread = calculate_vig_adjusted_spread(away_spread, away_spread_price)
    home_adjusted_spread = calculate_vig_adjusted_spread(home_spread, home_spread_price)
    
    # Find away/home outcomes in moneylines
    away_ml_outcome = next((o for o in h2h_market['outcomes'] if o['name'] == game['away_team']), None)
    home_ml_outcome = next((o for o in h2h_market['outcomes'] if o['name'] == game['home_team']), None)
    
    away_moneyline = away_ml_outcome['price']
    home_moneyline = home_ml_outcome['price']
    
    # Calculate moneyline probabilities
    away_ml_implied = ml_to_implied_prob(away_moneyline)
    home_ml_implied = ml_to_implied_prob(home_moneyline)
    away_ml_true, home_ml_true = remove_vig(away_ml_implied, home_ml_implied)
    
    # Calculate staleness
    last_update_str = bookmaker['last_update']
    last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
    seconds_since_update = (fetched_at - last_update).total_seconds()
    
    # Build row
    row = {
        # Core identifiers
        'game_id': game['id'],
        'sport_key': game['sport_key'],
        'game_time': game['commence_time'],
        'away_team': game['away_team'],
        'home_team': game['home_team'],
        'bookmaker': bookmaker['key'],
        'fetched_at': fetched_at.isoformat(),
        'last_bookmaker_update': last_update.isoformat(),
        'seconds_since_update': seconds_since_update,
        
        # Spreads
        'away_spread': away_spread,
        'away_spread_price': away_spread_price,
        'home_spread': home_spread,
        'home_spread_price': home_spread_price,
        'away_adjusted_spread': away_adjusted_spread,
        'home_adjusted_spread': home_adjusted_spread,
        
        # Moneylines
        'away_moneyline': away_moneyline,
        'home_moneyline': home_moneyline,
        'away_ml_implied_prob': away_ml_implied,
        'home_ml_implied_prob': home_ml_implied,
        'away_ml_true_prob': away_ml_true,
        'home_ml_true_prob': home_ml_true,
        
        # Game state (from ESPN)
        'away_score': score_info.get('away_score'),
        'home_score': score_info.get('home_score'),
        'period': score_info.get('period'),
        'time_remaining': score_info.get('time_remaining'),
        'game_status': score_info.get('game_status'),
    }
    
    # Calculate score differential if scores available
    if row['away_score'] is not None and row['home_score'] is not None:
        row['score_differential'] = row['away_score'] - row['home_score']
    else:
        row['score_differential'] = None
    
    return row


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_odds_data(odds_games: list, collection_timestamp: datetime) -> list:
    """
    Parse Odds API response into flat records for Parquet.
    
    One record per game-bookmaker combination.
    
    Args:
        odds_games: List of games from Odds API
        collection_timestamp: When we collected this data (UTC datetime)
        
    Returns:
        List of dicts (one per bookmaker per game)
    """
    records = []
    
    for game in odds_games:
        game_id = game['id']
        sport_key = game['sport_key']
        away_team = game['away_team']
        home_team = game['home_team']
        commence_time = game['commence_time']
        
        bookmakers = game.get('bookmakers', [])
        
        # Get ESPN score data for this game (if available)
        # Note: In Lambda, we'll fetch ESPN data once for all games, then match here
        # For now, set to empty dict - will be filled in by caller
        score_info = {}
        
        for book in bookmakers:
            book_key = book['key']
            book_last_update = book['last_update']
            
            # Find spreads and h2h markets
            spreads_market = next((m for m in book['markets'] if m['key'] == 'spreads'), None)
            h2h_market = next((m for m in book['markets'] if m['key'] == 'h2h'), None)
            
            # Skip if no spreads (edge case)
            if not spreads_market:
                continue
            
            # Extract spread data
            away_spread_outcome = None
            home_spread_outcome = None
            
            for outcome in spreads_market['outcomes']:
                if outcome['name'] == away_team:
                    away_spread_outcome = outcome
                elif outcome['name'] == home_team:
                    home_spread_outcome = outcome
            
            if not away_spread_outcome or not home_spread_outcome:
                continue
            
            away_spread = away_spread_outcome.get('point')
            away_spread_price = away_spread_outcome.get('price')
            home_spread = home_spread_outcome.get('point')
            home_spread_price = home_spread_outcome.get('price')
            
            # Extract moneyline data (may be None if not offered)
            away_ml = None
            home_ml = None
            away_ml_implied = None
            home_ml_implied = None
            away_ml_true = None
            home_ml_true = None
            
            if h2h_market:
                for outcome in h2h_market['outcomes']:
                    if outcome['name'] == away_team:
                        away_ml = outcome.get('price')
                    elif outcome['name'] == home_team:
                        home_ml = outcome.get('price')
                
                # Calculate ML probabilities if both available
                if away_ml is not None and home_ml is not None:
                    away_ml_implied = ml_to_implied_prob(away_ml)
                    home_ml_implied = ml_to_implied_prob(home_ml)
                    away_ml_true, home_ml_true = remove_vig(away_ml_implied, home_ml_implied)
            
            # Calculate vig-adjusted spreads
            away_adjusted_spread = calculate_vig_adjusted_spread(away_spread, away_spread_price) if away_spread is not None else None
            home_adjusted_spread = calculate_vig_adjusted_spread(home_spread, home_spread_price) if home_spread is not None else None
            
            # Calculate staleness
            last_update = datetime.fromisoformat(book_last_update.replace('Z', '+00:00'))
            seconds_since_update = (collection_timestamp - last_update).total_seconds()
            
            # Build row
            row = {
                # Core identifiers
                'game_id': game_id,
                'sport_key': sport_key,
                'game_time': commence_time,
                'away_team': away_team,
                'home_team': home_team,
                'bookmaker': book_key,
                'fetched_at': collection_timestamp.isoformat(),
                'last_bookmaker_update': book_last_update,
                'seconds_since_update': seconds_since_update,
                
                # Spreads
                'away_spread': away_spread,
                'away_spread_price': away_spread_price,
                'home_spread': home_spread,
                'home_spread_price': home_spread_price,
                'away_adjusted_spread': away_adjusted_spread,
                'home_adjusted_spread': home_adjusted_spread,
                
                # Moneylines
                'away_ml': away_ml,
                'home_ml': home_ml,
                'away_ml_implied_prob': away_ml_implied,
                'home_ml_implied_prob': home_ml_implied,
                'away_ml_true_prob': away_ml_true,
                'home_ml_true_prob': home_ml_true,
                
                # Game state (to be filled by caller with ESPN data)
                'away_score': None,
                'home_score': None,
                'period': None,
                'time_remaining': None,
                'game_status': None,
                'score_differential': None,
            }
            
            records.append(row)
    
    return records


def parse_espn_data(espn_data: dict, collection_timestamp: str) -> list:
    """
    Parse ESPN scoreboard response into flat records for Parquet.
    
    One record per game.
    
    Args:
        espn_data: ESPN scoreboard response
        collection_timestamp: When we collected this data (ISO format string)
        
    Returns:
        List of dicts (one per game)
        
    Note:
        ESPN API game_status values (from status['type']['state']):
        - 'pre': game hasn't started yet
        - 'in': game is currently live/in progress
        - 'post': game has finished
    """
    records = []
    
    events = espn_data.get('events', [])
    
    for event in events:
        competition = event['competitions'][0]
        competitors = competition['competitors']
        
        away_team_obj = next((c for c in competitors if c['homeAway'] == 'away'), None)
        home_team_obj = next((c for c in competitors if c['homeAway'] == 'home'), None)
        
        if not away_team_obj or not home_team_obj:
            continue
        
        away_team = away_team_obj['team']['displayName']
        home_team = home_team_obj['team']['displayName']
        
        status = event['status']
        
        # Parse clock if live
        period = status.get('period')
        display_clock = status.get('displayClock')
        time_remaining_minutes = None
        
        if display_clock and status['type']['state'] == 'in':
            try:
                parts = display_clock.split(':')
                if len(parts) == 2:
                    mins = int(parts[0])
                    secs = int(parts[1])
                    time_remaining_minutes = mins + secs / 60
            except:
                pass
        
        records.append({
            'query_time': collection_timestamp,
            'collection_timestamp': collection_timestamp,
            'espn_game_id': event['id'],
            'away_team_espn': away_team,
            'home_team_espn': home_team,
            'away_score': int(away_team_obj['score']) if away_team_obj.get('score') else None,
            'home_score': int(home_team_obj['score']) if home_team_obj.get('score') else None,
            'game_status': status['type']['state'],
            'game_status_description': status['type']['description'],
            'period': period,
            'display_clock': display_clock,
            'time_remaining_minutes': time_remaining_minutes,
        })
    
    # Filter out pre-game records - only track live ('in') and finished ('post') games
    records = [r for r in records if r['game_status'] in ['in', 'post']]
    
    return records


def write_parquet_to_s3(records: list, s3_key: str, bucket: str):
    """
    Write records to S3 as Parquet file.
    
    Args:
        records: List of dicts to write
        s3_key: S3 key (path) for the file
        bucket: S3 bucket name
        
    Raises:
        Exception if write fails
    """
    import pandas as pd
    import io
    
    if not records:
        print(f"{EMOJI['warning']} No records to write")
        return
    
    df = pd.DataFrame(records)
    
    # Convert to parquet in memory
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    parquet_buffer.seek(0)
    
    # Write to S3
    if IS_LAMBDA:
        import boto3
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        print(f"{EMOJI['save']} Wrote {len(records)} records to s3://{bucket}/{s3_key}")
    else:
        # Local testing - write to local filesystem
        local_path = DATA_ROOT / s3_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(parquet_buffer.getvalue())
        print(f"{EMOJI['save']} Wrote {len(records)} records to {local_path}")


def merge_scores_into_odds(odds_records: list, game_scores: dict) -> list:
    """
    Merge ESPN score data into odds records.
    
    Args:
        odds_records: List of odds records from parse_odds_data()
        game_scores: Dict from parse_espn_data()
        
    Returns:
        Updated odds_records with score fields filled in
    """
    for record in odds_records:
        key = (record['away_team'], record['home_team'])
        score_info = game_scores.get(key, {})
        
        record['away_score'] = score_info.get('away_score')
        record['home_score'] = score_info.get('home_score')
        record['period'] = score_info.get('period')
        record['time_remaining'] = score_info.get('time_remaining')
        record['game_status'] = score_info.get('game_status')
        record['score_differential'] = score_info.get('score_differential')
    
    return odds_records


def main(sport: str = 'nba', prod_run: bool = False):
    """
    Main execution function for Lambda.
    
    Collects one snapshot of live odds and writes to S3 (or local for testing).
    
    OPTIMIZATION: Checks ESPN API first (free) to see if any games are live.
    Only calls Odds API if live games exist (saves API credits).
    
    Args:
        sport: Sport to track ('nba', 'nfl', etc.)
        prod_run: If True, suppress prompts and print less
        
    Returns:
        Dict with execution results for Lambda response
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['nba']} LIVE ODDS TRACKER - {sport.upper()}")
    print(f"{'='*80}\n")
    
    # Validate sport
    if sport not in SUPPORTED_SPORTS:
        error_msg = f"Sport '{sport}' not supported. Supported: {list(SUPPORTED_SPORTS.keys())}"
        print(f"{EMOJI['error']} {error_msg}\n")
        return {
            'statusCode': 400,
            'body': error_msg
        }
    
    sport_key = SUPPORTED_SPORTS[sport]
    
    # Get S3 bucket for this sport
    s3_bucket = S3_BUCKETS.get(sport, 'nba-betting-mt')  # Default to NBA bucket
    
    # Check API key
    if not ODDS_API_KEY:
        error_msg = "ODDS_API_KEY not found in environment"
        print(f"{EMOJI['error']} {error_msg}\n")
        return {
            'statusCode': 500,
            'body': error_msg
        }
    
    # Capture collection timestamp
    query_time = datetime.now(timezone.utc)
    collection_timestamp = query_time.isoformat()
    now_et = query_time.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    
    # File naming: {date}_{time}.parquet (e.g., 20260201_170400.parquet)
    date_str = now_et.strftime('%Y%m%d')
    time_str = now_et.strftime('%H%M%S')
    filename = f"{date_str}_{time_str}.parquet"
    
    # S3 keys
    odds_s3_key = f"{S3_BASE_PATH}/the-odds-api/{filename}"
    espn_s3_key = f"{S3_BASE_PATH}/espn/{filename}"
    
    print(f"{EMOJI['refresh']} Collection timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"{EMOJI['info']} Filename: {filename}\n")
    
    try:
        # Step 1: Check ESPN API first (free) to see if any games are live
        print(f"{EMOJI['refresh']} Step 1: Checking ESPN API for live games...")
        espn_data = fetch_game_scores(sport)
        espn_records = parse_espn_data(espn_data, collection_timestamp)
        
        # Count live and finished games
        num_live = sum(1 for r in espn_records if r['game_status'] == 'in')
        num_post = sum(1 for r in espn_records if r['game_status'] == 'post')
        num_trackable = num_live + num_post
        
        print(f"{EMOJI['success']} ESPN API: Retrieved {len(espn_records)} games ({num_live} live, {num_post} finished)")
        
        # Step 2: Only call Odds API if there are live games
        if num_live == 0:
            print(f"\n{EMOJI['info']} No live games - skipping Odds API call (saved 1 API credit)")
            
            # Check if we have any finished games to save
            if num_post == 0:
                print(f"{EMOJI['info']} No live or finished games - skipping file write (no data to save)\n")
                return {
                    'statusCode': 200,
                    'body': {
                        'message': 'No live or finished games - no files written',
                        'num_games': len(espn_records),
                        'num_live_games': 0,
                        'num_post_games': 0,
                        'odds_api_calls': 0,
                        'api_calls_saved': 1,
                    }
                }
            else:
                # We have finished games - save ESPN data only (no odds)
                print(f"{EMOJI['save']} Saving ESPN data for {num_post} finished game(s) (final scores)...")
                write_parquet_to_s3(espn_records, espn_s3_key, s3_bucket)
                print(f"{EMOJI['success']} Wrote {len(espn_records)} records to {espn_s3_key}\n")
                
                return {
                    'statusCode': 200,
                    'body': {
                        'message': f'No live games - saved {num_post} finished game(s) for final scores',
                        'num_games': len(espn_records),
                        'num_live_games': 0,
                        'num_post_games': num_post,
                        'odds_api_calls': 0,
                        'api_calls_saved': 1,
                    }
                }
        
        # Step 3: Fetch odds (only if live games exist)
        print(f"\n{EMOJI['fire']} Step 2: Fetching odds for {num_live} live games...")
        odds_games = fetch_live_odds(sport_key)
        odds_records = parse_odds_data(odds_games, query_time)
        
        total_games_from_api = len(odds_games)
        total_records_before_filter = len(odds_records)
        
        print(f"{EMOJI['success']} Odds API: Retrieved {total_games_from_api} games, {total_records_before_filter} records")
        
        # Step 3.5: Filter to only live games (unless TRACK_UPCOMING_GAMES is enabled)
        if not TRACK_UPCOMING_GAMES:
            # Get list of live game IDs from ESPN
            live_game_teams = {
                (r['away_team_espn'], r['home_team_espn'])
                for r in espn_records
                if r['game_status'] == 'in'
            }
            
            # Normalize Odds API team names for comparison with ESPN
            # NBA: "Los Angeles Clippers" -> "LA Clippers"
            # NCAAB: "Alabama St Hornets" -> "Alabama State Hornets"
            normalize_fn = normalize_nba_team_name if sport_key == 'basketball_nba' else normalize_ncaab_team_name
            
            # Filter odds records to only live games (with normalization)
            odds_records_filtered = [
                r for r in odds_records
                if (normalize_fn(r['away_team']), normalize_fn(r['home_team'])) in live_game_teams
            ]
            
            # Also filter ESPN records to only live games
            espn_records_filtered = [
                r for r in espn_records
                if r['game_status'] == 'in'
            ]
            
            games_filtered_out = total_games_from_api - len({(r['away_team'], r['home_team']) for r in odds_records_filtered})
            records_filtered_out = total_records_before_filter - len(odds_records_filtered)
            
            print(f"{EMOJI['info']} Filtered to LIVE GAMES ONLY:")
            print(f"   Kept: {len({(r['away_team'], r['home_team']) for r in odds_records_filtered})} live games")
            print(f"   Filtered out: {games_filtered_out} upcoming/finished games ({records_filtered_out} records)")
            
            odds_records = odds_records_filtered
            espn_records = espn_records_filtered
        else:
            print(f"{EMOJI['warning']} TRACK_UPCOMING_GAMES=true: Keeping all games (live + upcoming)")
        
        # Step 4: Write both files to S3
        print(f"\n{EMOJI['save']} Step 3: Writing to S3...")
        write_parquet_to_s3(odds_records, odds_s3_key, s3_bucket)
        write_parquet_to_s3(espn_records, espn_s3_key, s3_bucket)
        
        # Skip detailed display if no records to show
        if not odds_records:
            print(f"\n{EMOJI['info']} No records to display (all games filtered out)")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'No live games after filtering',
                    'num_games': 0,
                    'num_live_games': 0,
                    'odds_api_calls': 1,
                }
            }
        
        # Step 5: Display detailed game info (like test script)
        print(f"\n{EMOJI['chart']} GAME DETAILS:")
        
        # Convert to DataFrames for easy display
        import pandas as pd
        odds_df = pd.DataFrame(odds_records)
        espn_df = pd.DataFrame(espn_records)
        
        # Calculate consensus for each game (matches test script exactly)
        consensus = odds_df.groupby(['away_team', 'home_team']).agg({
            'away_spread': 'median',
            'home_spread': 'median',
            'away_ml': 'median',
            'home_ml': 'median',
        }).reset_index()
        
        # Normalize odds team names for joining with ESPN data
        normalize_fn = normalize_nba_team_name if sport_key == 'basketball_nba' else normalize_ncaab_team_name
        consensus['away_team_normalized'] = consensus['away_team'].apply(normalize_fn)
        consensus['home_team_normalized'] = consensus['home_team'].apply(normalize_fn)
        
        # Join with ESPN to get game status (using normalized team names)
        consensus = consensus.merge(
            espn_df[['away_team_espn', 'home_team_espn', 'game_status', 'away_score', 'home_score']],
            left_on=['away_team_normalized', 'home_team_normalized'],
            right_on=['away_team_espn', 'home_team_espn'],
            how='left'
        )
        
        # Split into live vs upcoming
        live_games = consensus[consensus['game_status'] == 'in']
        upcoming_games = consensus[consensus['game_status'] != 'in']
        
        # Display live games
        if len(live_games) > 0:
            print(f"\n{EMOJI['fire']} LIVE GAMES ({len(live_games)}):")
            for _, game in live_games.iterrows():
                matchup = f"{game['away_team']} @ {game['home_team']}"
                away_spread = f"{game['away_spread']:+.1f}" if pd.notna(game['away_spread']) else "N/A"
                home_spread = f"{game['home_spread']:+.1f}" if pd.notna(game['home_spread']) else "N/A"
                away_ml = f"{int(game['away_ml']):+d}" if pd.notna(game['away_ml']) else "N/A"
                home_ml = f"{int(game['home_ml']):+d}" if pd.notna(game['home_ml']) else "N/A"
                
                print(f"  {matchup}")
                print(f"    Spread: {away_spread} | {home_spread}  |  ML: {away_ml} | {home_ml}")
                
                # Show score on separate line if available
                if pd.notna(game['away_score']) and pd.notna(game['home_score']):
                    print(f"    {EMOJI['nba']} Score: {int(game['away_score'])}-{int(game['home_score'])}")
        
        # Display upcoming games (only if TRACK_UPCOMING_GAMES is enabled)
        if TRACK_UPCOMING_GAMES and len(upcoming_games) > 0:
            print(f"\n{EMOJI['chart']} UPCOMING GAMES ({len(upcoming_games)}):")
            for _, game in upcoming_games.iterrows():
                matchup = f"{game['away_team']} @ {game['home_team']}"
                away_spread = f"{game['away_spread']:+.1f}" if pd.notna(game['away_spread']) else "N/A"
                home_spread = f"{game['home_spread']:+.1f}" if pd.notna(game['home_spread']) else "N/A"
                away_ml = f"{int(game['away_ml']):+d}" if pd.notna(game['away_ml']) else "N/A"
                home_ml = f"{int(game['home_ml']):+d}" if pd.notna(game['home_ml']) else "N/A"
                
                print(f"  {matchup}")
                print(f"    Spread: {away_spread} | {home_spread}  |  ML: {away_ml} | {home_ml}")
        
        print(f"\n{EMOJI['success']} Snapshot complete!")
        print(f"   Games tracked: {len({(r['away_team'], r['home_team']) for r in odds_records})} live")
        print(f"   Odds records: {len(odds_records)}")
        print(f"   ESPN records: {len(espn_records)}")
        print(f"   API calls used: 1")
        print()
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Snapshot collected successfully',
                'num_games': len({(r['away_team'], r['home_team']) for r in odds_records}),
                'num_live_games': len({(r['away_team'], r['home_team']) for r in odds_records}),
                'num_post_games': num_post,
                'odds_records': len(odds_records),
                'espn_records': len(espn_records),
                'odds_api_calls': 1,
                'odds_s3_key': odds_s3_key,
                'espn_s3_key': espn_s3_key,
            }
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"\n{EMOJI['error']} {error_msg}\n")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': error_msg
        }


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Triggered by EventBridge rule every 1 minute.
    
    Args:
        event: Lambda event dict (from EventBridge)
               Can include: {"sport": "nba"} or {"sport": "ncaab"}
        context: Lambda context object
        
    Returns:
        Dict with statusCode and body
    """
    # Get sport from event, fall back to environment variable, default to NBA
    sport = event.get('sport') if event else None
    if not sport:
        sport = os.getenv('DEFAULT_SPORT', 'nba')
    
    print(f"Lambda invoked for sport: {sport}")
    
    # Lambda runs in production mode (no prompts)
    return main(sport=sport, prod_run=True)


if __name__ == '__main__':
    """Local testing with CLI args."""
    parser = argparse.ArgumentParser(description='Track live betting lines')
    parser.add_argument(
        '--sport',
        type=str,
        default='nba',
        choices=list(SUPPORTED_SPORTS.keys()),
        help='Sport to track (default: nba)'
    )
    parser.add_argument(
        '--prod-run',
        action='store_true',
        help='Production mode (no prompts)'
    )
    args = parser.parse_args()
    
    result = main(sport=args.sport, prod_run=args.prod_run)
    
    # Print result
    import json
    print(f"\n{'='*80}")
    print("EXECUTION RESULT")
    print(f"{'='*80}")
    print(json.dumps(result, indent=2))
    print()
