#!/bin/bash
"""
Monitor NBA Season Fetch Progress

Checks S3 to see how many files have been uploaded for each season.
Also checks if the fetch process is still running.

Usage:
    bash scripts/monitor_nba_fetch.sh
"""

echo "================================================================================"
echo "🏀 NBA HISTORICAL FETCH - PROGRESS MONITOR"
echo "================================================================================"
echo ""

# Check if fetch process is running
if ps aux | grep "fetch_historical_nba_season_lines" | grep -v grep > /dev/null; then
    echo "✅ Fetch process is RUNNING"
    echo ""
    ps aux | grep "fetch_historical_nba_season_lines" | grep -v grep | awk '{print "   PID: " $2 " | Season: " $15}'
    echo ""
else
    echo "⏹️  No fetch process currently running"
    echo ""
fi

# Check log file
if [ -f "logs/nba_fetch_2020.log" ]; then
    echo "📝 Latest log activity:"
    tail -5 logs/nba_fetch_2020.log | grep -E "(Processing|Saved|FETCHING)" | tail -3
    echo ""
fi

echo "================================================================================"
echo "📊 FILES IN S3 BY SEASON"
echo "================================================================================"
echo ""

seasons=("2020-21" "2021-22" "2022-23" "2023-24" "2024-25" "2025-26")

for season in "${seasons[@]}"; do
    count=$(aws s3 ls s3://the-odds-api-mt/nba/historical/$season/ 2>/dev/null | wc -l | tr -d ' ')
    
    if [ "$count" -gt 0 ]; then
        echo "  $season: $count files ✅"
    else
        echo "  $season: 0 files ⏳"
    fi
done

echo ""
echo "================================================================================"
echo "💰 API CREDITS REMAINING"
echo "================================================================================"
echo ""

# Try to get from log file
if [ -f "logs/nba_fetch_2020.log" ]; then
    credits=$(tail -50 logs/nba_fetch_2020.log | grep -o "Remaining: [0-9,]*" | tail -1 | cut -d' ' -f2)
    if [ ! -z "$credits" ]; then
        echo "  Credits: $credits"
    else
        echo "  (Check log file for details)"
    fi
else
    echo "  (No log file found)"
fi

echo ""
echo "================================================================================"
echo ""
echo "Commands:"
echo "  Watch progress: watch -n 10 bash scripts/monitor_nba_fetch.sh"
echo "  View log: tail -f logs/nba_fetch_2020.log"
echo "  View S3: aws s3 ls s3://the-odds-api-mt/nba/historical/ --recursive | tail -20"
echo ""

