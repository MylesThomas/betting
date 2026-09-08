#!/bin/bash
REPO="/Users/thomasmyles/dev/betting"
UV="/Users/thomasmyles/.local/bin/uv"
SCRIPT="src/levels_fyi_scraper/scrape_overview.py"
LOG="/tmp/levels_scrape.log"

cd "$REPO" && $UV run python $SCRIPT >> "$LOG" 2>&1
EXIT=$?

if [ $EXIT -ne 0 ]; then
    osascript -e "display notification \"Check $LOG for details\" with title \"Levels.fyi scraper FAILED (exit $EXIT)\""
fi
