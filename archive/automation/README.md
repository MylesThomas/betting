# Fast Automated Betting - FanDuel & DraftKings

Production-ready betting bots designed for **SPEED**. Can place bets in 2-5 seconds when triggered.

## ⚠️ CRITICAL WARNINGS

1. **Violates sportsbook Terms of Service**
2. **Account suspension and fund loss are likely**
3. **For educational purposes only**
4. **Use at your own risk - you are responsible for all consequences**

---

## 🎯 What You Have

Two production-ready betting bots:
- **FanDuel** - `fanduel_bet_placer.py` (656 lines)
- **DraftKings** - `draftkings_bet_placer.py` (672 lines)

### Key Features
- **Session persistence** - Stay logged in 24/7
- **Fast execution** - 2-5 seconds from trigger to bet
- **Screenshot debugging** - Visual confirmation at each step
- **Error handling** - Graceful failures with detailed logging
- **Dry-run mode** - Test safely before going live
- **Anti-detection** - Stealth browser configuration

---

## 📦 Installation

### 1. Install Dependencies

```bash
cd automation

# Use Python 3.12.7 (3.13+ also works with latest Playwright)
pyenv shell 3.12.7

pip install -r requirements.txt
playwright install chromium
```

### 2. Configure Credentials

Create/edit `.env` file:

```bash
# FanDuel
FANDUEL_USERNAME=your_email@example.com
FANDUEL_PASSWORD=your_password

# DraftKings
DRAFTKINGS_USERNAME=your_email@example.com
DRAFTKINGS_PASSWORD=your_password
```

### 3. Test Login

```bash
# Test FanDuel
python fanduel_bet_placer.py --login-only

# Test DraftKings
python draftkings_bet_placer.py --login-only
```

---

## 🚀 Quick Start - Fast Betting

### The Speed Problem

**Manual betting is too slow:**
- Open phone/browser: 5-10s
- Navigate to bet: 5-10s
- Enter stake & place: 5-10s
- **Total: 15-30 seconds** ❌

By then, the +EV line has moved.

### The Solution: Keep-Session Mode

Keep browsers logged in 24/7, ready for instant betting.

#### Step 1: Start Session Keepers

```bash
# Terminal 1 - FanDuel always logged in
python fanduel_bet_placer.py --keep-session

# Terminal 2 - DraftKings always logged in
python draftkings_bet_placer.py --keep-session
```

These will:
- Stay logged in continuously
- Refresh every 5 minutes
- Be ready for instant betting

#### Step 2: When +EV Opportunity Appears

```bash
# Test with dry run first
python fanduel_bet_placer.py --bet "NFL:Moneyline:Chiefs:-150:10" --dry-run

# Then place live bet (2-5 second execution!)
python fanduel_bet_placer.py --bet "NFL:Moneyline:Chiefs:-150:10" --live
```

**Speed with keep-session: 2-5 seconds** ✅

---

## 📖 Usage Modes

### Mode 1: Login Test (Safe)
```bash
python fanduel_bet_placer.py --login-only
python draftkings_bet_placer.py --login-only
```
- Tests login only
- Saves session for reuse
- No bets placed
- Takes screenshots for debugging

### Mode 2: Keep Session Alive (Recommended for Speed)
```bash
python fanduel_bet_placer.py --keep-session
python draftkings_bet_placer.py --keep-session
```
- Stays logged in forever
- Ready for instant betting
- Auto-refreshes every 5 minutes
- Run in separate terminals

### Mode 3: Single Bet - Dry Run (Safe Testing)
```bash
python fanduel_bet_placer.py --bet "NFL:Moneyline:Chiefs:-150:10" --dry-run
```
- Goes through full flow
- **Stops before placing bet**
- Good for testing selectors
- Saves screenshots

### Mode 4: Single Bet - Live (Actually Places Bet) ⚠️
```bash
python fanduel_bet_placer.py --bet "NFL:Moneyline:Chiefs:-150:10" --live
```
- **Actually places the bet**
- 3 second warning to cancel
- Use at your own risk

---

## 🎲 Bet Format

All bets use this format:
```
Sport:BetType:Selection:Odds:Amount
```

### Examples

```bash
# NFL moneyline
"NFL:Moneyline:Kansas City Chiefs:-150:25"

# NBA spread
"NBA:Spread:Lakers +5.5:-110:50"

# Player prop
"NBA:Player Prop:LeBron James o24.5 points:-115:20"

# NFL total
"NFL:Total:Over 47.5:-110:30"
```

---

## 🔧 How It Works

### Technical Flow

1. **Stealth Browser Setup**
   - Launches Chromium with anti-detection scripts
   - Removes `navigator.webdriver` flag
   - Real user agent and browser fingerprint
   - Human-like delays (500-1500ms random)

2. **Session Management**
   - Saves cookies after successful login
   - Restores session on subsequent runs
   - Auto-refresh to keep session alive

3. **Login**
   - Navigates to sportsbook
   - Handles cookie consent
   - Fills credentials
   - Verifies success with multiple indicators

4. **Bet Placement**
   - Navigates to sport section
   - Finds selection (team/player/prop)
   - Adds to bet slip
   - Enters stake amount
   - Places bet (or stops in dry-run)

5. **Verification**
   - Checks for confirmation message
   - Takes screenshot
   - Logs all actions

### Anti-Detection Techniques

- Non-headless mode (visible browser)
- Randomized human-like delays
- Proper viewport and timezone
- Real browser fingerprints
- Session warming

---

## 🔗 Integration with Odds Monitor

### Option A: Subprocess Call

```python
import subprocess

def place_bet_fanduel(sport, bet_type, selection, odds, amount):
    bet_str = f"{sport}:{bet_type}:{selection}:{odds}:{amount}"
    cmd = [
        "python", 
        "automation/fanduel_bet_placer.py",
        "--bet", bet_str,
        "--live"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
```

### Option B: Direct Import (Advanced)

```python
from automation.fanduel_bet_placer import FanDuelBot, BetRequest, FanDuelCredentials, FanDuelConfig

async def place_bet_async():
    credentials = FanDuelCredentials.from_env()
    config = FanDuelConfig()
    bot = FanDuelBot(credentials, config, dry_run=False)
    
    await bot.init_browser(use_saved_session=True)
    
    bet = BetRequest(
        sport="NFL",
        bet_type="Moneyline",
        selection="Chiefs",
        odds="-150",
        amount=25.0
    )
    
    success = await bot.place_bet(bet)
    await bot.close_browser()
    return success
```

---

## 🐛 Troubleshooting

### "Login failed"

**Causes:**
- Wrong credentials in `.env`
- Captcha/2FA required
- Account already logged in elsewhere
- Bot detected

**Debug:**
1. Check screenshots: `fd_*_*.png` or `dk_*_*.png`
2. Check logs: `fanduel_bot_logs.txt` or `draftkings_bot_logs.txt`
3. Run with visible browser (default) to watch
4. Try different browser profile

### "Could not find selection"

**Causes:**
- Team/player name doesn't match exactly
- Game not available yet
- Different format on sportsbook
- UI selectors changed

**Fix:**
1. Check screenshot to see what's on page
2. Update selectors in bot code
3. Try alternate team name format
4. Check if game is live vs upcoming

### "Session expired"

**Fix:**
```bash
# Delete old session and re-login
rm fanduel_session.json
python fanduel_bet_placer.py --login-only
```

### Bot Detected / Account Suspended

**This is expected risk:**
- Violates Terms of Service
- Sportsbooks have sophisticated detection
- Consider using separate test accounts
- May need to bet manually after detection

---

## 📝 Updating Selectors

Sportsbooks update their UI regularly. When selectors break:

### 1. Find New Selectors

```bash
# Run bot and let it fail
python fanduel_bet_placer.py --login-only

# Check screenshot to see current page
open fd_error_*.png
```

### 2. Inspect Element in Browser

1. Open FanDuel/DK in Chrome
2. Right-click → Inspect Element
3. Find the button/input you need
4. Copy selector

### 3. Update Bot Code

Find the selector list in the bot code:

```python
# Example: Login button selectors
login_selectors = [
    'button:has-text("Log In")',
    'a:has-text("Log In")',
    'button:has-text("Login")',
    '[data-test-id="login-button"]',
    'text=Log In',
    # Add your new selector here
    'button.new-login-class'
]
```

The bot tries multiple selectors, so add new ones to the list.

---

## ⚡ Speed Optimization Tips

### Current Performance

| Approach | Time | Notes |
|----------|------|-------|
| Manual | 15-30s | Too slow for sharp lines |
| Bot without session | 20-30s | Login takes time |
| **Bot with keep-session** | **2-5s** | ✅ Recommended |

### Further Optimizations

1. **Pre-navigate to sport**
   - Keep NFL/NBA page open
   - Save 1-2 seconds

2. **Reduce timeouts**
   - In config: `element_timeout = 5000` → `3000`
   - Risky if network is slow

3. **Multiple browsers**
   - Run FD + DK simultaneously
   - Hedge opportunities

4. **Headless mode**
   - Set `headless = True` in config
   - Slightly faster, but harder to debug

---

## 📂 Files

```
automation/
├── fanduel_bet_placer.py      # FanDuel bot (656 lines)
├── draftkings_bet_placer.py   # DraftKings bot (672 lines)
├── test_playwright_setup.py   # Test Playwright installation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Generated Files (gitignored)

```
.env                           # Your credentials
fanduel_session.json          # Saved FD session
draftkings_session.json       # Saved DK session
fanduel_bot_logs.txt          # FD logs
draftkings_bot_logs.txt       # DK logs
fd_*.png                      # FD screenshots
dk_*.png                      # DK screenshots
```

---

## 🎓 Technical Details

### Dependencies

- `playwright>=1.48.0` - Browser automation (works with Python 3.13)
- `python-dotenv==1.0.0` - Environment variable management

### Browser Requirements

- Chromium (auto-installed by Playwright)
- ~200MB download
- macOS/Linux/Windows supported

### Session Management

Sessions are saved as JSON files containing:
- Cookies
- Local storage
- Session storage

These allow instant login on subsequent runs.

---

## ⚠️ Legal & Ethical Considerations

### Terms of Service Violations

Both FanDuel and DraftKings explicitly prohibit:
- Automated betting
- Bot usage
- Scripting

**Using this tool violates their ToS.**

### Risks

1. **Account suspension** - Immediate and permanent
2. **Fund forfeiture** - Money in account may be lost
3. **IP ban** - May affect future accounts
4. **Legal issues** - Depending on jurisdiction

### Why This Exists

Educational demonstration of:
- Browser automation techniques
- Anti-detection methods
- High-performance scraping
- Session management

**Not recommended for production use.**

---

## 🔮 Future Improvements

### Potential Enhancements

1. **API reverse-engineering**
   - Use internal endpoints directly
   - Sub-1-second execution
   - More reliable than UI scraping

2. **Machine learning for detection**
   - Identify pattern changes in UI
   - Auto-update selectors
   - Predict detection attempts

3. **Multi-book orchestration**
   - Coordinate bets across books
   - Arbitrage execution
   - Hedge management

4. **Cloud deployment**
   - AWS Lambda + headless Chrome
   - Faster network to sportsbooks
   - Auto-scaling

---

## 🆘 Support

### Debug Information

When asking for help, provide:
1. Log file (`fanduel_bot_logs.txt` or `draftkings_bot_logs.txt`)
2. Latest screenshot (`fd_*.png` or `dk_*.png`)
3. Command you ran
4. Python version (`python --version`)
5. Playwright version (`pip show playwright`)

### Common Issues Database

See troubleshooting section above for most common issues.

---

## 📄 License

Educational purposes only. No warranty provided. Use at your own risk.

**You are responsible for all consequences of using this tool.**
