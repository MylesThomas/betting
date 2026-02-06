# Betting

Data-driven betting strategies for NBA and NFL using historical analysis, API data, and statistical modeling.

## Setup

This project uses `uv` for fast package management with Python 3.13. See [SETUP.md](SETUP.md) for full installation instructions.

**Quick start:**
```bash
# 1. Install/update dependencies (creates/updates .venv)
uv sync
uv sync --native-tls

# 2. Activate the virtual environment
source .venv/bin/activate
```

### Troubleshooting

**TLS Certificate Error with `uv sync`:**

If you encounter an "invalid peer certificate: UnknownIssuer" error when running `uv sync`, use the `--native-tls` flag:

```bash
uv sync --native-tls
```

This issue typically occurs on corporate networks or systems with custom certificate authorities. The flag enables use of system TLS certificates instead of the bundled certificates.

## Project Structure

### Core Directories

- **`data/`** - Data pipeline organized by processing stage (most historical data stored on Amazon S3):
  - `01_input/` - Raw data from APIs (The Odds API, NBA API, Unexpected Points)
  - `02_cache/` - Cached rosters and player-team mappings
  - `03_intermediate/` - Processed props, consensus lines, game logs
  - `04_output/` - Final betting opportunities and analysis results

- **`scripts/`** - Data fetching and processing utilities
  - API data fetchers (props, odds, game logs)
  - Cache builders (rosters, player mappings)
  - Data transformation and aggregation scripts

- **`backtesting/`** - Historical strategy analysis and validation
  - NBA 3PT props modeling
  - NFL luck regression analysis
  - Performance metrics and ROI calculations

- **`implementation/`** - Production betting finders
  - `find_3pt_underdog_unders_today.py` - NBA 3PT under opportunities
  - `find_nfl_regression_plays.py` - NFL luck regression plays
  - `find_todays_plays.py` - Combined daily recommendations

- **`src/`** - Shared utility modules
  - Team/player name normalization
  - Odds calculations and conversions
  - Config loading and data helpers

- **`streamlit_app/`** - Web dashboard for viewing opportunities
  - Real-time arbitrage monitoring
  - Historical performance tracking
  - Interactive filters and exports

- **`analysis/`** - Ad-hoc research and explorations
  - Market efficiency studies
  - Vig analysis
  - Shot quality modeling

### Supporting Directories

- **`docs/`** - Setup guides and AWS automation configs
- **`content/`** - Generated reports, visualizations, and writing
- **`config/`** - Configuration files (API keys, thresholds, mappings)
- **`automation/`** - Bet placement scripts (Playwright/Selenium)

## Current Strategies

### NBA
- **Points Props**: Role-based scoring models considering usage rate and matchup

### NFL
- **Luck Regression**: Backing unlucky favorites and lucky big underdogs based on expected points differential

## Automation

- **NBA Dashboard**: Fully automated via AWS Lambda + EventBridge (runs daily at 7 AM ET)
- **NFL Alerts**: Manual (planned: weekly email alerts via Lambda)

See `docs/AWS_AUTOMATION_CHECKLIST.md` for setup details.

## Resources

- [The Odds API](https://the-odds-api.com/liveapi/guides/v4/)
- [NBA API (nba_api)](https://github.com/swar/nba_api)
- [Unexpected Points NFL Data](https://docs.google.com/spreadsheets/d/1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558/edit?usp=sharing)

## Arbitrage Dashboard

```bash
# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run the dashboard
streamlit run streamlit_app/app.py
```

Dashboard opens at `http://localhost:8501` and displays real-time arbitrage opportunities.

---

**Note**: This is a personal analytics project. Strategies are based on historical analysis and are not guaranteed to be profitable. Always gamble responsibly.
