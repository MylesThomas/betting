# NBA Historical Game Lines Fetching

## Summary

✅ **Playoff games ARE included!** Tested and confirmed.  
✅ **S3 Storage**: Files automatically saved to `s3://the-odds-api-mt/nba/historical/YYYY-YY/`  
✅ **Local Backup**: Optional local backup to `data/01_input/the-odds-api/nba/game_lines/historical/`

## S3 Directory Structure

```
s3://the-odds-api-mt/nba/historical/
  2020-21/
    nba_game_lines_2020-12-22.csv
    nba_game_lines_2020-12-23.csv
    ...
    nba_game_lines_2021-07-20.csv  (Finals)
  2021-22/
    nba_game_lines_2021-10-19.csv
    ...
    nba_game_lines_2022-06-16.csv  (Finals)
  2022-23/
    nba_game_lines_2022-10-18.csv
    ...
  2023-24/
    nba_game_lines_2023-10-24.csv
    ...
    nba_game_lines_2024-06-17.csv  (Finals)
  2024-25/
    nba_game_lines_2024-10-22.csv
    ...
  2025-26/
    nba_game_lines_2025-10-21.csv
    ...
```

Files are organized by **season** (e.g., `2024-25`) not by individual dates, making it easier to:
- Download entire seasons at once
- Query specific seasons in analytics
- Separate regular season from playoffs within the same directory

## Available Data

The Odds API has NBA game lines (moneyline + spreads) available starting from:

**Earliest Season: 2020-21** (December 2020)

### Seasons Available:
- ✅ 2020-21 (COVID delayed season: Dec 2020 - July 2021)
- ✅ 2021-22 (Oct 2021 - June 2022)
- ✅ 2022-23 (Oct 2022 - June 2023)
- ✅ 2023-24 (Oct 2023 - June 2024)
- ✅ 2024-25 (Oct 2024 - June 2025)
- ✅ 2025-26 (Oct 2025 - June 2026) - Current season

**Total: 6 seasons of historical NBA game lines + playoffs**

## Scripts

### 1. Test Coverage Scripts

#### Check which seasons have data
```bash
python scripts/test_nba_historical_game_lines_coverage.py
```
- Tests opening night for each season 2012-2025
- Shows earliest available season
- Cost: ~13 credits (1 per test)

#### Check if playoffs are included
```bash
python scripts/test_nba_playoff_coverage.py
```
- Tests Finals, Conference Finals, and Round 1 games
- Confirms playoffs ARE available
- Cost: ~4 credits (1 per test)

### 2. Fetch Historical Game Lines

```bash
# Test single date first (saves to S3 + local backup)
python scripts/fetch_historical_nba_season_lines.py --test-date 2024-10-22

# Test with S3 only (no local backup)
python scripts/fetch_historical_nba_season_lines.py --test-date 2024-10-22 --no-local-backup

# Dry run (shows cost estimate)
python scripts/fetch_historical_nba_season_lines.py --season 2024

# Production run (actually fetches data and saves to S3)
python scripts/fetch_historical_nba_season_lines.py --season 2024 --prod-run
```

#### Available Seasons:
- `--season 2020` (2020-21)
- `--season 2021` (2021-22)
- `--season 2022` (2022-23)
- `--season 2023` (2023-24)
- `--season 2024` (2024-25)
- `--season 2025` (2025-26) - Current

## Data Format

### S3 Storage (Primary)
```
s3://the-odds-api-mt/nba/historical/
  2024-25/
    nba_game_lines_2024-10-22.csv
    nba_game_lines_2024-10-23.csv
    ...
  2025-26/
    nba_game_lines_2025-10-21.csv
    ...
```

### Local Backup (Optional)
```
data/01_input/the-odds-api/nba/game_lines/historical/
  nba_game_lines_2024-10-22.csv
  nba_game_lines_2024-10-23.csv
  ...
```

### CSV Columns
- `game_id`: Unique game identifier
- `game_time`: Game start time (UTC)
- `away_team`: Away team name
- `home_team`: Home team name
- `bookmaker`: Bookmaker name (DraftKings, FanDuel, etc.)
- `bookmaker_key`: Bookmaker key
- `last_update`: When line was last updated
- `market`: 'moneyline' or 'spread'
- `away_line`: Away spread (null for moneyline)
- `away_odds`: Away odds (American format)
- `home_line`: Home spread (null for moneyline)
- `home_odds`: Home odds (American format)

### Example Data
```csv
game_id,game_time,away_team,home_team,bookmaker,market,away_line,away_odds,home_line,home_odds
92cf...,2024-10-22 23:40:00,New York Knicks,Boston Celtics,DraftKings,spread,6.0,-108,-6.0,-112
92cf...,2024-10-22 23:40:00,New York Knicks,Boston Celtics,DraftKings,moneyline,,,200,,-245
```

## API Costs

### Per Date:
- 1 credit to check for games
- 20 credits per game (10 for moneyline + 10 for spreads)

### Full Season Estimates:
- **Regular season**: ~1,230 games × 20 = **24,600 credits**
- **Playoffs**: ~90 games × 20 = **1,800 credits**
- **Total per season**: ~**26,400 credits**

### All 6 Seasons:
- 6 seasons × 26,400 = **~158,400 credits**
- At $0.001/credit = **~$158**

## Usage Examples

### Fetch Current Season (2025-26)
```bash
# Dry run first to see estimate
python scripts/fetch_historical_nba_season_lines.py --season 2025

# Then fetch for real
python scripts/fetch_historical_nba_season_lines.py --season 2025 --prod-run
```

### Fetch All Historical Seasons
```bash
# 2020-21 season
python scripts/fetch_historical_nba_season_lines.py --season 2020 --prod-run

# 2021-22 season
python scripts/fetch_historical_nba_season_lines.py --season 2021 --prod-run

# 2022-23 season
python scripts/fetch_historical_nba_season_lines.py --season 2022 --prod-run

# 2023-24 season
python scripts/fetch_historical_nba_season_lines.py --season 2023 --prod-run

# 2024-25 season
python scripts/fetch_historical_nba_season_lines.py --season 2024 --prod-run

# 2025-26 season (current)
python scripts/fetch_historical_nba_season_lines.py --season 2025 --prod-run
```

## Features

- ✅ **S3 Storage**: Automatically saves to S3 bucket `the-odds-api-mt`
- ✅ **Season Organization**: Files organized by season (e.g., `nba/historical/2024-25/`)
- ✅ **Local Backup**: Optional local backup (use `--no-local-backup` to skip)
- ✅ **Smart Skipping**: Checks S3 first, skips dates that already exist (won't re-fetch)
- ✅ **Timezone Handling**: Filters games to correct date in ET timezone
- ✅ **Rate Limiting**: Built-in delays to avoid API throttling
- ✅ **Progress Tracking**: Real-time credit usage and progress updates
- ✅ **Both Markets**: Moneyline and spreads in one file
- ✅ **Multiple Bookmakers**: Capture all available bookmakers for line shopping
- ✅ **Playoffs Included**: Automatically includes playoff games

## Access S3 Data

### AWS CLI
```bash
# List all NBA seasons
aws s3 ls s3://the-odds-api-mt/nba/historical/

# List all files in a season
aws s3 ls s3://the-odds-api-mt/nba/historical/2024-25/ --human-readable

# Download entire season
aws s3 sync s3://the-odds-api-mt/nba/historical/2024-25/ ./data/nba_lines_2024-25/

# Download specific file
aws s3 cp s3://the-odds-api-mt/nba/historical/2024-25/nba_game_lines_2024-10-22.csv .

# Count files in a season
aws s3 ls s3://the-odds-api-mt/nba/historical/2024-25/ | wc -l
```

### Python (boto3)
```python
import boto3
import pandas as pd

s3_client = boto3.client('s3')

# Read a specific date from S3
bucket = 'the-odds-api-mt'
key = 'nba/historical/2024-25/nba_game_lines_2024-10-22.csv'

obj = s3_client.get_object(Bucket=bucket, Key=key)
df = pd.read_csv(obj['Body'])

print(f"Loaded {len(df)} lines for {df['game_id'].nunique()} games")

# List all files in a season
prefix = 'nba/historical/2024-25/'
response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
files = [obj['Key'] for obj in response.get('Contents', [])]
print(f"Found {len(files)} files in 2024-25 season")

# Load entire season into one DataFrame
dfs = []
for key in files:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    dfs.append(df)

season_df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(season_df)} lines for {season_df['game_id'].nunique()} games")
```

### AWS Console
View files in browser:  
https://s3.console.aws.amazon.com/s3/buckets/the-odds-api-mt?prefix=nba/historical/

---



Once you have the data, you can:

1. **Merge with game results** from `nba_api` data
2. **Analyze line movement** (if fetching multiple times per day)
3. **Find value bets** by comparing closing lines to results
4. **Calculate ROI** for different betting strategies
5. **Identify profitable systems** (home dogs, rest advantages, etc.)

## Related Files

- `fetch_historical_nfl_season_lines.py` - Similar script for NFL
- `data/01_input/nba_api/historical/` - NBA game results
- `src/nba_gamelog_utils.py` - Utilities for NBA data

