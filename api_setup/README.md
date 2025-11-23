# NBA Prop Betting API Setup

This folder contains scripts to set up and test data sources for NBA prop betting trend analysis.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd api_setup
pip install -r requirements.txt
```

Or using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up The Odds API (Optional but Recommended)

1. Go to [https://the-odds-api.com/](https://the-odds-api.com/)
2. Sign up for a free account (500 requests/month)
3. Copy your API key
4. Create a `.env` file in this directory:

```bash
cp .env.example .env
```

5. Edit `.env` and add your API key:
```
ODDS_API_KEY=your_actual_key_here
```

### 3. Test the APIs

**Test NBA API (no key required):**
```bash
python nba_api_setup.py
```

**Test The Odds API:**
```bash
python odds_api_setup.py
```

## 📊 What Each API Provides

### NBA API (`nba_api_setup.py`)
- ✅ **FREE** - No API key required
- Player game-by-game stats (all seasons)
- Points, rebounds, assists, 3PT made, blocks, steals, etc.
- Historical data for backtesting trends
- Perfect for analyzing "under streaks"

**Example usage:**
```python
from nba_api_setup import get_player_game_log, analyze_trend

# Get Draymond Green's game log
df = get_player_game_log("Draymond Green", "2024-25")

# Analyze 3PT under trend
trend = analyze_trend(df, 'FG3M', threshold=0.5, direction='under')
print(f"Current streak: {trend['current_streak']} games")
```

### The Odds API (`odds_api_setup.py`)
- Free tier: 500 requests/month
- NBA game odds (moneyline, spreads, totals)
- Player props (may require paid plan)
- Multiple sportsbooks

**Note:** The free tier focuses on game odds. For comprehensive player props, consider:
- Scraping PrizePicks or Underdog Fantasy
- DraftKings/FanDuel APIs (require account)
- Upgrading to paid Odds API plan

## 🎯 Strategy: Breaking Bad Trends

The core strategy is identifying players on long "under" streaks who may be due for regression to the mean.

**Example:**
- Draymond Green goes under 0.5 3PT made for 10 consecutive games
- Historical average: 0.8 3PT per game
- Hypothesis: He's due for a game with 1-2 threes

**Key Stats to Track:**
- `FG3M` - Three-pointers made
- `PTS` - Points
- `REB` - Rebounds
- `AST` - Assists
- `BLK` - Blocks
- `STL` - Steals

## 📁 Files Overview

- `requirements.txt` - Python dependencies
- `.env.example` - Template for environment variables
- `nba_api_setup.py` - NBA stats API setup and testing
- `odds_api_setup.py` - The Odds API setup and testing
- `README.md` - This file

## 🔄 Next Steps

After setting up the APIs, you can:

1. **Build a trend tracker** - Automatically scan for long under/over streaks
2. **Create alerts** - Get notified when a player hits X-game streak
3. **Backtest strategies** - Test historical performance of trend betting
4. **Integrate prop lines** - Combine NBA stats with real-time betting lines

## 🐛 Troubleshooting

**NBA API issues:**
- Rate limiting: Add `time.sleep(0.6)` between requests
- Player not found: Check spelling/use exact name
- No data: Season may not have started or player inactive

**Odds API issues:**
- 401 error: Invalid API key
- 422 error: Feature not available on free plan
- No games: Off-season or no games scheduled

## ⚠️ Known Limitations

### Data Source Mismatches

**Props vs Actual Game Results:**
- **Props data** comes from The Odds API (betting lines for scheduled games)
- **Game results** come from NBA API (actual games played)
- These sources can mismatch in several ways:

**1. Special tournament games:**
- **NBA Cup Championship** (Bucks vs. Thunder on Dec 17, 2024)
- The Odds API has betting lines for these games
- NBA calendar (`nba_calendar/`) excludes them (only fetches `season_type='Regular Season'`)
- Result: Props data exists, but no game results in calendar

**2. Postponed/Cancelled games:**
- **Rockets @ Hawks** (originally scheduled Jan 11, 2025) - postponed
- **Bucks @ Pelicans** (originally scheduled Jan 22, 2025) - postponed
- The Odds API captured props for the originally scheduled date
- The game was never played on that date
- NBA API won't have game results for the scheduled date, only for the rescheduled date
- Result: Props data exists for the scheduled date, but NULLs when joining with actual game stats

**Recommendation:** When joining props with game results, filter out or flag rows where game stats are NULL - these represent postponed/cancelled games or games not in the Regular Season calendar.

## 📚 Resources

- [nba_api documentation](https://github.com/swar/nba-api)
- [The Odds API docs](https://the-odds-api.com/liveapi/guides/v4/)
- [Basketball Reference](https://www.basketball-reference.com/) - Manual research

