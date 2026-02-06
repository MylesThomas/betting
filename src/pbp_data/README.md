# PBP Data Collection Module

Collect NBA play-by-play data for Monte Carlo simulation in incremental stages with caching.

## Architecture

Similar to `src/player_team_history`, this module:
- ✅ Caches intermediate results
- ✅ Resumes from where it left off
- ✅ Can be run in timeout loops for robustness
- ✅ Separates data collection (API calls) from processing (no API)

## Cache Location

```
~/Downloads/tmp/player_points_monte_carlo/
├── game_ids/
│   ├── 20251022.csv
│   ├── 20251023.csv
│   └── ...
├── pbp_data/
│   ├── 401810584_20260204.json
│   ├── 401810585_20260204.json
│   └── ...
└── progress/
    ├── game_ids_progress.json
    └── pbp_progress.json
```

## Stages

### Stage 1: Get Game IDs

Collects all game IDs for the season by date.

**Single run:**
```bash
python src/pbp_data/01_get_game_ids.py --verbose
```

**Loop (recommended for robustness):**
```bash
while true; do
  echo ""
  echo "========================================"
  echo "Starting run (will run for 30 seconds)..."
  
  timeout 30 python src/pbp_data/01_get_game_ids.py --verbose
  
  echo "Restarting in 0.1 seconds..."
  sleep 0.1
done
```

**Output:**
- Caches: `~/Downloads/tmp/player_points_monte_carlo/game_ids/{date}.csv`
- Progress: `~/Downloads/tmp/player_points_monte_carlo/progress/game_ids_progress.json`

### Stage 2: Get Play-by-Play Data

Downloads ESPN play-by-play data (includes box scores) for each game.

**Single run:**
```bash
python src/pbp_data/02_get_pbp_data.py --verbose
```

**Loop (recommended for robustness):**
```bash
while true; do
  timeout 30 python src/pbp_data/02_get_pbp_data.py --verbose || break
  sleep 0.1
done
```

**Output:**
- Caches: `~/Downloads/tmp/player_points_monte_carlo/pbp_data/{game_id}_{date}.json`
- Progress: `~/Downloads/tmp/player_points_monte_carlo/progress/pbp_progress.json`

### Stage 3: Process Data

Converts cached JSON into parquet files. No API calls needed.

**Single run:**
```bash
python src/pbp_data/03_process_data.py --verbose
```

**Output:**
- `data/minute_by_minute.parquet` - Minute-by-minute cumulative points for every player in every game

### Stage 4: Build Player Profiles

Builds historical profiles for each player from minute-by-minute data.

**Single run:**
```bash
python src/pbp_data/04_build_profiles.py --verbose
```

**Output:**
- `data/player_profiles.parquet` - Historical stats and distributions for each player
  - Aggregate stats: avg points, std, percentiles
  - Points per minute distributions
  - Quarter splits (Q1-Q4 averages)
  - Complete game history (all games, not limited to 100)

### Stage 5: Validate (Optional)

Note: Validation requires the source JSON files from Stage 2. If those files have been cleaned up, validation can be skipped as the data has already been processed.

Validates minutes played and points against box scores.

**Single run:**
```bash
python src/pbp_data/05_validate.py --verbose
```

**Output:**
- `data/metadata/validation_results.parquet`

## Features

### Incremental Progress

Each stage tracks what's been completed:
- Skips already-processed items
- Can be interrupted and resumed
- Progress saved after each successful item

### Timeout Loops

Designed to run in timeout loops for robustness:
- ESPN API sometimes times out
- Loop automatically restarts after timeout
- Progress preserved across restarts
- Eventually completes all items

### Rate Limiting

Built-in delays to avoid overwhelming ESPN API:
- 0.2s between date requests (Stage 1)
- 0.5s between game requests (Stage 2)

## Status

- [x] Stage 1: Get game IDs
- [x] Stage 2: Get PBP data (includes box scores)  
- [x] Stage 3: Process data (minute-by-minute)
- [x] Stage 4: Build player profiles
- [x] Stage 5: Validate (source JSON files required)
