# Daily 3PT Under Betting System

## 📊 Strategy Summary

Based on backtesting the 2024-25 NBA season (11,038 props):

| Line | Win Rate | ROI | Strategy |
|------|----------|-----|----------|
| 0.5 | 45.2% | -6.91% | ❌ **AVOID** |
| 1.5 | 51.1% | -4.36% | ❌ **AVOID** |
| **2.5** | **53.8%** | **+1.01%** | ✅ **BET** |
| **3.5** | **56.4%** | **+3.95%** | ✅✅ **BET** (BEST!) |
| **4.5** | **57.0%** | **+4.31%** | ✅✅ **BET** |

**Key Finding:** Lines 2.5+ are profitable. The market underprices unders on high 3PT lines.

---

## 🚀 Quick Start

### 1. Set Your API Key

```bash
# Add to ~/.zshrc or ~/.bashrc
export ODDS_API_KEY='your_key_here'

# Apply changes
source ~/.zshrc  # or source ~/.bashrc
```

### 2. Run Manually (Test)

```bash
cd /Users/thomasmyles/dev/betting

# Find today's opportunities
python3 scripts/find_profitable_3pt_unders.py

# Specific date
python3 scripts/find_profitable_3pt_unders.py --date 2024-11-24

# Only lines 3.5+ (highest ROI)
python3 scripts/find_profitable_3pt_unders.py --min-line 3.5

# Adjust odds range
python3 scripts/find_profitable_3pt_unders.py --min-odds -200
```

### 3. Set Up Daily Automation

#### Option A: Cron Job (Mac/Linux)

```bash
# Edit crontab
crontab -e

# Add this line to run at 10 AM daily:
0 10 * * * /Users/thomasmyles/dev/betting/scripts/run_daily_betting_finder.sh

# Or run at 2 PM daily (closer to game time):
0 14 * * * /Users/thomasmyles/dev/betting/scripts/run_daily_betting_finder.sh
```

#### Option B: Manual Daily Run

```bash
# Just run the shell script
./scripts/run_daily_betting_finder.sh
```

---

## 📂 Output & Logs

Results are saved to:
```
logs/daily_bets/betting_opportunities_YYYYMMDD.log
```

Example:
```
logs/daily_bets/betting_opportunities_20241124.log
```

---

## 🎯 What the Script Does

1. **Fetches today's NBA games** from the-odds-api
2. **Gets 3PT player props** for each game
3. **Filters for profitable opportunities**:
   - Lines >= 2.5 (configurable)
   - Excludes bookmaker traps (odds < -300)
4. **Calculates expected value** based on backtest data
5. **Ranks opportunities** by highest EV

---

## 📋 Example Output

```
🎯 PROFITABLE 3PT UNDER OPPORTUNITIES
================================================================================

📅 Lakers vs Warriors
   Start: 2024-11-24 07:30 PM

#    Player                   Line   Odds     Book                 Bet        Impl%    Exp%     Edge     EV        
--------------------------------------------------------------------------------------------------------------------
1    Stephen Curry            3.5    -120     fanduel             $120.00    54.5%    56.4%    +1.9%    $+3.20    
2    LeBron James             2.5    -110     draftkings          $110.00    52.4%    53.8%    +1.4%    $+2.40    
3    Anthony Davis            2.5    +105     betmgm              $95.24     48.8%    53.8%    +5.0%    $+8.50    

📊 SUMMARY
================================================================================

Total Opportunities: 3
Positive EV Bets: 3
Total Stake: $325.24
Total Expected Value: $+14.10
Average Edge: +2.8%
Expected ROI: +4.3%
```

---

## ⚙️ Configuration

Edit `scripts/find_profitable_3pt_unders.py` to adjust:

```python
# Betting Configuration
TARGET_WIN = 100  # Bet to win $100
MAX_ODDS_THRESHOLD = -300  # Skip odds worse than this

# Strategy Configuration
PROFITABLE_LINE_THRESHOLD = 2.5  # Lines >= 2.5 showed positive ROI
```

---

## 🔧 Command Line Options

```bash
# Find opportunities for today
python3 scripts/find_profitable_3pt_unders.py

# Specific date
python3 scripts/find_profitable_3pt_unders.py --date 2024-11-24

# Only highest lines (3.5+)
python3 scripts/find_profitable_3pt_unders.py --min-line 3.5

# Adjust odds filters
python3 scripts/find_profitable_3pt_unders.py --min-odds -200 --max-odds +150

# Combination
python3 scripts/find_profitable_3pt_unders.py --min-line 3.5 --min-odds -150
```

---

## 📈 Tracking Results

### Manual Tracking

Create a spreadsheet with:
- Date
- Player
- Line
- Odds
- Result (Win/Loss)
- Profit

### Automated (Future Enhancement)

TODO: Add result tracking to validate ongoing performance

---

## 🎲 Bankroll Management

**Recommended:**
- Start small (e.g., $10-25 per bet)
- Use fixed stake sizing
- Keep 20-30 units in bankroll
- Track results for 100+ bets before scaling

**Example:**
- Bankroll: $500
- Unit size: $25
- 20 units available

---

## ⚠️ Important Notes

1. **This is a data-driven edge, not a guarantee**
   - Expected ROI: +1% to +4%
   - Requires large sample size
   - Short-term variance is normal

2. **Odds shopping is critical**
   - Script finds BEST odds across books
   - 10-20 point difference is common
   - Sign up for multiple sportsbooks

3. **Line movement matters**
   - Check closer to game time
   - Lines shift based on news/injuries
   - Be flexible

4. **Responsible gambling**
   - Only bet what you can afford to lose
   - Set daily/weekly limits
   - Take breaks

---

## 🔄 Updating Backtest Data

As the season progresses, update the win rates:

```bash
# Re-run backtest with latest data
cd backtesting
python3 20251121_nba_3pt_prop_miss_streaks.py --blind-under-by-line

# Update EXPECTED_WIN_RATES in find_profitable_3pt_unders.py
```

---

## 📞 Support

Questions? Check:
1. `api_setup/QUICKSTART.md` - Odds API setup
2. `docs/PLAYER_NAME_MATCHING.md` - Player name issues
3. Backtest analysis: `backtesting/20251121_nba_3pt_prop_miss_streaks.py`

---

## 🎯 Next Steps

1. ✅ Run manually today to test
2. ✅ Set up API key if not done
3. ✅ Configure cron job for daily automation
4. ✅ Start tracking results
5. ⏳ Refine filters based on ongoing performance

Good luck! 🍀

