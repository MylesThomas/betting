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

