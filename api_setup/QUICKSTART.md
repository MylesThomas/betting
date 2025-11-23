# 🏀 NBA Prop Betting - Quick Start Guide

## ✅ What's Working Now

### 1. NBA API (FREE - No Key Needed) ✅
The NBA API is **fully functional** and ready to use!

**Test it:**
```bash
cd /Users/thomasmyles/dev/betting/api_setup
python nba_api_setup.py
```

**What you can do:**
- Get game-by-game stats for ANY NBA player
- Analyze trends (consecutive overs/unders)
- Scan multiple players for regression opportunities
- Access full historical data

**Example Output:**
```
🚨 Rudy Gobert: 72 game streak under 0.5 3PM (avg: 0.00)
🚨 Clint Capela: 55 game streak under 0.5 3PM (avg: 0.00)
🚨 Steven Adams: 58 game streak under 0.5 3PM (avg: 0.00)
```

### 2. The Odds API (Requires Free API Key) ⚠️
Ready to use once you add your API key.

**Setup Steps:**
1. Go to https://the-odds-api.com/
2. Sign up (it's free - 500 requests/month)
3. Copy your API key
4. Edit the `.env` file and replace `your_api_key_here` with your actual key

**Test it after setup:**
```bash
python odds_api_setup.py
```

## 🎯 Your Strategy: Breaking Bad Trends

The NBA API just found these real opportunities:
- **Rudy Gobert**: 72 consecutive games under 0.5 3-pointers
- **Clint Capela**: 55 consecutive games under 0.5 3-pointers  
- **Steven Adams**: 58 consecutive games under 0.5 3-pointers

*Note: These centers almost never shoot 3s, so this isn't a "due for regression" play. But the tool works! You can customize it for real opportunities.*

## 🚀 How to Use the NBA API

### Get a Player's Game Log
```python
from nba_api_setup import get_player_game_log, analyze_trend

# Fetch recent games
df = get_player_game_log("Stephen Curry", "2024-25")

# Analyze a trend
trend = analyze_trend(df, 'PTS', threshold=27.5, direction='under')
print(f"Current streak: {trend['current_streak']} games")
```

### Available Stats to Track
- `PTS` - Points
- `REB` - Rebounds
- `AST` - Assists
- `FG3M` - Three-pointers made
- `BLK` - Blocks
- `STL` - Steals
- `FGM` - Field goals made
- `FTM` - Free throws made

### Scan Multiple Players
```python
from nba_api_setup import find_long_bad_trends

players = [
    "LeBron James",
    "Stephen Curry", 
    "Kevin Durant",
    "Luka Doncic"
]

opportunities = find_long_bad_trends(
    players,
    stat_column='PTS',
    threshold=25.5,
    min_streak=7
)
```

## 📊 Next Steps

1. **Start with NBA API**: It's working now, no setup needed
2. **Get Odds API Key**: Takes 2 minutes, free 500 requests/month
3. **Build Your Scanner**: Create custom scripts to find opportunities
4. **Track Props Daily**: Monitor player trends and prop lines

## 🔧 Troubleshooting

**NBA API Issues:**
- Already fixed SSL certificate issues (working on macOS)
- If you get rate limited, add `time.sleep(0.6)` between requests
- Season dates: Use "2024-25" format for current season

**Odds API Issues:**
- 401 error = Invalid API key
- 422 error = Feature not available (player props may need paid plan)
- Free tier focuses on game odds; player props may be limited

## 📁 File Overview

```
api_setup/
├── nba_api_setup.py     # ✅ Working - NBA player stats
├── odds_api_setup.py    # ⚠️  Needs API key - Betting lines
├── requirements.txt     # ✅ Installed
├── .env                 # ⚠️  Add your Odds API key here
├── README.md            # Full documentation
└── QUICKSTART.md        # This file
```

## 💡 Pro Tips

1. **Rate Limiting**: NBA API is free but rate-limited. Space out requests.
2. **Data Quality**: Some stats may be missing for injured/inactive players.
3. **Historical Data**: You can fetch data from previous seasons (e.g., "2023-24").
4. **Prop Lines**: Consider scraping PrizePicks/DraftKings for player prop totals.

---

**Ready to start? Run this:**
```bash
cd /Users/thomasmyles/dev/betting/api_setup
python nba_api_setup.py
```

It will show you real trend data from current NBA players!

