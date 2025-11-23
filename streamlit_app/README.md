# NBA Arbitrage Dashboard

Local Streamlit dashboard for viewing NBA arbitrage opportunities.

## Quick Start

```bash
# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run the dashboard
streamlit run streamlit_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

- 📊 View today's arbitrage opportunities
- 🔄 Manual "Run Now" button to find new arbs
- 📈 Key metrics (total arbs, avg profit, best opportunity)
- 📅 Historical performance tracking
- 💾 Download opportunities as CSV
- 🎚️ Filter by minimum profit percentage

## Future Deployment: AWS EC2 (Option A)

**Architecture:**
```
EC2 instance (t2.small, ~$15/month)
├── Cron job → runs arb finder at 7am ET daily
├── Streamlit app → always running on port 8501
└── Data → stored locally at /data/arbs/
```

**Deployment steps are documented in `app.py` docstring.**

## Screenshots

(Coming soon after we test it locally)

## Development

The dashboard reads from `data/arbs/` directory. Make sure you have arb data files:
```
data/arbs/
├── arb_threes_20251121.csv
├── arb_points_20251121.csv
└── ...
```

Run the arb finder to generate data:
```bash
python scripts/find_arb_opportunities.py --markets player_threes
```

## Troubleshooting

### Player Team Cache Issues

If you see missing teams or players (should have 525 players across 30 teams):

**Quick Fix: Rebuild the cache**
```bash
# Step 1: Fetch latest rosters from NBA API
python scripts/build_full_roster_cache.py

# Step 2: Update player_team_cache.csv
python3 << 'EOF'
import pandas as pd
from datetime import datetime
full_roster = pd.read_csv('data/nba_full_roster_cache.csv')
player_team_cache = pd.DataFrame({
    'player_normalized': full_roster['player_normalized'],
    'team': full_roster['team'],
    'timestamp': datetime.now().isoformat()
})
player_team_cache = player_team_cache.drop_duplicates(subset=['player_normalized'], keep='first')
player_team_cache = player_team_cache.sort_values('player_normalized')
player_team_cache.to_csv('data/player_team_cache.csv', index=False)
print(f"✅ Updated player_team_cache.csv with {len(player_team_cache)} players")
EOF
```

**When to rebuild:**
- Missing teams (should have all 30 NBA teams)
- After major trades
- At the start of a new season
- If players are showing up as "Unknown Team"

**Alternative:** Click the "Invalidate Cache" button in the dashboard sidebar to rebuild from the web interface.

