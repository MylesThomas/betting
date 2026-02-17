#!/bin/bash
# Query Live Odds Snapshot - DuckDB Analysis Script
#
# USAGE: ./tmp/query_live_odds_snapshot.sh [TIMESTAMP] [SPORT]
# EXAMPLES:
#   ./tmp/query_live_odds_snapshot.sh 20260216_225518 ncaab
#   ./tmp/query_live_odds_snapshot.sh 20260216_230000 nba
#
# Flattens odds data from 6 rows (one per bookmaker) to 1 row per game
# Joins with ESPN live data (score, time remaining, game status)

set -euo pipefail

# Parameters
TIMESTAMP=${1:-"20260216_225518"}
SPORT=${2:-"ncaab"}
SPORT_UPPER=$(echo "$SPORT" | tr '[:lower:]' '[:upper:]')

echo "📊 Querying ${SPORT_UPPER} live odds snapshot: ${TIMESTAMP}"

# Check if odds file exists (only present for live games)
if aws s3 ls "s3://${SPORT}-betting-mt/data/01_input/live_odds/the-odds-api/${TIMESTAMP}.parquet" 2>/dev/null; then
    echo "✅ Odds file exists - this is a live game snapshot"
else
    echo "⚠️  No odds file found - this is a post-game snapshot (final scores only)"
    echo "📊 Querying ESPN data only..."
    
    duckdb -line -c "
INSTALL httpfs;
LOAD httpfs;
SET s3_region='us-east-2';
SET s3_access_key_id='$(aws configure get aws_access_key_id)';
SET s3_secret_access_key='$(aws configure get aws_secret_access_key)';

-- Post-game snapshot: ESPN data only (no odds)
SELECT 
    '🏀 POST-GAME SNAPSHOT (Final Score)' as status,
    away_team_espn || ' @ ' || home_team_espn as matchup,
    away_score,
    home_score,
    CASE 
        WHEN away_score > home_score THEN away_team_espn || ' WINS'
        WHEN home_score > away_score THEN home_team_espn || ' WINS'
        ELSE 'TIE'
    END as winner,
    (away_score - home_score) as margin,
    game_status,
    game_status_description,
    period,
    display_clock,
    espn_game_id,
    collection_timestamp
FROM 's3://${SPORT}-betting-mt/data/01_input/live_odds/espn/${TIMESTAMP}.parquet'
;
"
    exit 0
fi

duckdb -line -c "
INSTALL httpfs;
LOAD httpfs;
SET s3_region='us-east-2';
SET s3_access_key_id='$(aws configure get aws_access_key_id)';
SET s3_secret_access_key='$(aws configure get aws_secret_access_key)';

WITH odds_raw AS (
    SELECT *
    FROM 's3://${SPORT}-betting-mt/data/01_input/live_odds/the-odds-api/${TIMESTAMP}.parquet'
),

odds_flattened AS (
    SELECT
        game_id,
        sport_key,
        game_time,
        away_team,
        home_team,
        fetched_at,
        
        -- DraftKings
        MAX(CASE WHEN bookmaker = 'draftkings' THEN away_spread END) as dk_away_spread,
        MAX(CASE WHEN bookmaker = 'draftkings' THEN home_spread END) as dk_home_spread,
        MAX(CASE WHEN bookmaker = 'draftkings' THEN away_ml END) as dk_away_ml,
        MAX(CASE WHEN bookmaker = 'draftkings' THEN home_ml END) as dk_home_ml,
        MAX(CASE WHEN bookmaker = 'draftkings' THEN away_adjusted_spread END) as dk_away_adj_spread,
        MAX(CASE WHEN bookmaker = 'draftkings' THEN home_adjusted_spread END) as dk_home_adj_spread,
        
        -- FanDuel
        MAX(CASE WHEN bookmaker = 'fanduel' THEN away_spread END) as fd_away_spread,
        MAX(CASE WHEN bookmaker = 'fanduel' THEN home_spread END) as fd_home_spread,
        MAX(CASE WHEN bookmaker = 'fanduel' THEN away_ml END) as fd_away_ml,
        MAX(CASE WHEN bookmaker = 'fanduel' THEN home_ml END) as fd_home_ml,
        MAX(CASE WHEN bookmaker = 'fanduel' THEN away_adjusted_spread END) as fd_away_adj_spread,
        MAX(CASE WHEN bookmaker = 'fanduel' THEN home_adjusted_spread END) as fd_home_adj_spread,
        
        -- BetMGM
        MAX(CASE WHEN bookmaker = 'betmgm' THEN away_spread END) as mgm_away_spread,
        MAX(CASE WHEN bookmaker = 'betmgm' THEN home_spread END) as mgm_home_spread,
        MAX(CASE WHEN bookmaker = 'betmgm' THEN away_ml END) as mgm_away_ml,
        MAX(CASE WHEN bookmaker = 'betmgm' THEN home_ml END) as mgm_home_ml,
        MAX(CASE WHEN bookmaker = 'betmgm' THEN away_adjusted_spread END) as mgm_away_adj_spread,
        MAX(CASE WHEN bookmaker = 'betmgm' THEN home_adjusted_spread END) as mgm_home_adj_spread,
        
        -- BetRivers
        MAX(CASE WHEN bookmaker = 'betrivers' THEN away_spread END) as br_away_spread,
        MAX(CASE WHEN bookmaker = 'betrivers' THEN home_spread END) as br_home_spread,
        MAX(CASE WHEN bookmaker = 'betrivers' THEN away_ml END) as br_away_ml,
        MAX(CASE WHEN bookmaker = 'betrivers' THEN home_ml END) as br_home_ml,
        MAX(CASE WHEN bookmaker = 'betrivers' THEN away_adjusted_spread END) as br_away_adj_spread,
        MAX(CASE WHEN bookmaker = 'betrivers' THEN home_adjusted_spread END) as br_home_adj_spread,
        
        -- Caesars/William Hill
        MAX(CASE WHEN bookmaker = 'williamhill_us' THEN away_spread END) as wh_away_spread,
        MAX(CASE WHEN bookmaker = 'williamhill_us' THEN home_spread END) as wh_home_spread,
        MAX(CASE WHEN bookmaker = 'williamhill_us' THEN away_ml END) as wh_away_ml,
        MAX(CASE WHEN bookmaker = 'williamhill_us' THEN home_ml END) as wh_home_ml,
        MAX(CASE WHEN bookmaker = 'williamhill_us' THEN away_adjusted_spread END) as wh_away_adj_spread,
        MAX(CASE WHEN bookmaker = 'williamhill_us' THEN home_adjusted_spread END) as wh_home_adj_spread,
        
        -- Fanatics
        MAX(CASE WHEN bookmaker = 'fanatics' THEN away_spread END) as fan_away_spread,
        MAX(CASE WHEN bookmaker = 'fanatics' THEN home_spread END) as fan_home_spread,
        MAX(CASE WHEN bookmaker = 'fanatics' THEN away_ml END) as fan_away_ml,
        MAX(CASE WHEN bookmaker = 'fanatics' THEN home_ml END) as fan_home_ml,
        MAX(CASE WHEN bookmaker = 'fanatics' THEN away_adjusted_spread END) as fan_away_adj_spread,
        MAX(CASE WHEN bookmaker = 'fanatics' THEN home_adjusted_spread END) as fan_home_adj_spread,
        
        -- Market averages
        AVG(away_spread) as avg_away_spread,
        AVG(home_spread) as avg_home_spread,
        AVG(away_ml) as avg_away_ml,
        AVG(home_ml) as avg_home_ml,
        AVG(away_ml_true_prob) as avg_away_win_prob,
        AVG(home_ml_true_prob) as avg_home_win_prob,
        
        -- Best lines for bettors
        MAX(away_spread) as best_away_spread,
        MIN(home_spread) as best_home_spread,
        MAX(away_ml) as best_away_ml,
        MAX(home_ml) as best_home_ml
        
    FROM odds_raw
    GROUP BY 
        game_id, sport_key, game_time, away_team, home_team, fetched_at
),

espn_data AS (
    SELECT *
    FROM 's3://${SPORT}-betting-mt/data/01_input/live_odds/espn/${TIMESTAMP}.parquet'
)

SELECT 
    -- Game Info
    o.away_team,
    o.home_team,
    e.away_score,
    e.home_score,
    (e.away_score - e.home_score) as away_lead,
    e.game_status,
    e.period,
    e.display_clock,
    e.time_remaining_minutes,
    
    -- DraftKings Lines
    o.dk_away_spread,
    o.dk_home_spread,
    o.dk_away_ml,
    o.dk_home_ml,
    o.dk_away_adj_spread,
    o.dk_home_adj_spread,
    
    -- FanDuel Lines
    o.fd_away_spread,
    o.fd_home_spread,
    o.fd_away_ml,
    o.fd_home_ml,
    o.fd_away_adj_spread,
    o.fd_home_adj_spread,
    
    -- BetMGM Lines
    o.mgm_away_spread,
    o.mgm_home_spread,
    o.mgm_away_ml,
    o.mgm_home_ml,
    o.mgm_away_adj_spread,
    o.mgm_home_adj_spread,
    
    -- BetRivers Lines
    o.br_away_spread,
    o.br_home_spread,
    o.br_away_ml,
    o.br_home_ml,
    o.br_away_adj_spread,
    o.br_home_adj_spread,
    
    -- Caesars Lines
    o.wh_away_spread,
    o.wh_home_spread,
    o.wh_away_ml,
    o.wh_home_ml,
    o.wh_away_adj_spread,
    o.wh_home_adj_spread,
    
    -- Fanatics Lines
    o.fan_away_spread,
    o.fan_home_spread,
    o.fan_away_ml,
    o.fan_home_ml,
    o.fan_away_adj_spread,
    o.fan_home_adj_spread,
    
    -- Market Stats
    o.avg_away_spread,
    o.avg_home_spread,
    o.avg_away_ml,
    o.avg_home_ml,
    o.avg_away_win_prob,
    o.avg_home_win_prob,
    
    -- Best Lines
    o.best_away_spread,
    o.best_home_spread,
    o.best_away_ml,
    o.best_home_ml,
    
    -- Metadata
    o.game_id,
    e.espn_game_id,
    o.fetched_at,
    e.collection_timestamp
    
FROM odds_flattened o
LEFT JOIN espn_data e 
    ON o.fetched_at = e.collection_timestamp
    AND o.away_team = e.away_team_espn
    AND o.home_team = e.home_team_espn
;
"
