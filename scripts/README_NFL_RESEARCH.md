# NFL Game Research Tool

## Overview

This tool helps you research NFL games systematically to gather context for your betting posts. Instead of fabricating narratives, you can use these scripts to organize your research from real sources.

## What We Built

### 1. `research_nfl_games.py` - Main Research Tool ✅

**What it does:**
- Generates structured research queries for each game
- Creates templates for taking notes
- Outputs in multiple formats (JSON, CSV, Markdown)

**How to use:**
```bash
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games.py
```

**Output files:**
- `nfl_week16_research_checklist_TIMESTAMP.md` - Markdown checklist with all search queries
- `nfl_week16_search_queries_TIMESTAMP.csv` - CSV with all queries for easy reference
- `nfl_week16_research_guide_TIMESTAMP.json` - JSON template to fill in your research notes

### 2. `scrape_nfl_game_context.py` - Direct Scraper (Has Issues) ⚠️

**Status:** Built but has SSL certificate issues when trying to scrape NFL.com directly.

**Why it doesn't work well:**
- SSL verification errors
- NFL.com may have anti-scraping measures
- Website structure makes it hard to extract clean data

**Recommendation:** Don't use this for now. Use the manual research workflow instead.

### 3. `fetch_nfl_game_context_web.py` - Template Generator

**Status:** Creates templates but doesn't fetch live data.

## Recommended Workflow

### For This Week (Quick Fix):

1. **Run the research tool:**
   ```bash
   cd /Users/thomasmyles/dev/betting/scripts
   python research_nfl_games.py
   ```

2. **Open the markdown checklist:**
   ```bash
   open nfl_week16_research_checklist_20251217_003506.md
   ```

3. **Manually research each game:**
   - Go through each search query in the checklist
   - Search on Google, ESPN, RotoWire, NFL.com
   - Take notes in the markdown file or JSON template

4. **Key things to look for:**
   - **Injuries:** Who's out? Who's questionable? Impact players?
   - **Recent form:** How did each team perform last 3 games?
   - **Storylines:** Playoff implications? Coaching changes? Weather?
   - **Betting context:** Has the line moved? Where's public money going?

5. **Write your post:**
   - Use your research notes to write informed narratives
   - Combine with your luck/efficiency data
   - No more fabricating QB names or game details!

### Example Research Process:

For **Patriots @ Ravens**:

1. Search: "Patriots Week 16 2025 injury report"
   - Note: Drake Maye status, any key OL injuries

2. Search: "Ravens Week 16 2025 injury report"
   - Note: Lamar Jackson status, defensive injuries

3. Search: "Patriots recent performance last 3 games"
   - Note: Offensive efficiency, points scored, close games?

4. Search: "Ravens recent performance last 3 games"
   - Note: That 24-0 shutout you mentioned - who did they face?

5. Search: "Patriots Ravens betting line Week 16"
   - Note: Opening line, current line, movement

6. **Fill in your post:**
   ```
   Baltimore's 24-0 shutout of [ACTUAL OPPONENT] was inflated—
   [ACTUAL CONTEXT ABOUT INJURIES/BACKUP QB]. New England 
   [ACTUAL RECENT PERFORMANCE]. At +3, we're fading an 
   overvalued shutout and backing a Patriots team that's 
   [ACTUAL CONTEXT].
   ```

## Future Improvements

### Option A: API Integration
- Sign up for SportsDataIO or similar NFL API
- Programmatically fetch injury reports, stats, news
- More reliable than web scraping

### Option B: Improved Scraping
- Fix SSL issues
- Use Selenium for JavaScript-heavy sites
- Parse multiple sources (ESPN, RotoWire, etc.)

### Option C: Hybrid Approach
- Use API for structured data (injuries, stats)
- Manual research for narratives and storylines
- Combine both in your posts

## Files Generated This Session

```
/Users/thomasmyles/dev/betting/scripts/
├── research_nfl_games.py                          # Main tool ✅
├── scrape_nfl_game_context.py                     # Has SSL issues ⚠️
├── fetch_nfl_game_context_web.py                  # Template only
├── nfl_week16_research_checklist_20251217_003506.md    # Use this! ✅
├── nfl_week16_search_queries_20251217_003506.csv       # Reference
└── nfl_week16_research_guide_20251217_003506.json      # Fill this in
```

## Quick Commands

```bash
# Generate research templates for a new week
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games.py

# View the checklist
open nfl_week16_research_checklist_20251217_003506.md

# View search queries in CSV
open nfl_week16_search_queries_20251217_003506.csv
```

## Next Steps

1. **For this week:** Use the manual research workflow with the checklist
2. **For future weeks:** Decide if you want to invest in an API or improve scraping
3. **Update the script:** Edit `research_nfl_games.py` to change week number or games

## Questions?

The tool is designed to be simple and flexible. You can:
- Modify the search queries in `research_nfl_games.py`
- Add more data points to track
- Change output formats
- Integrate with your existing betting data

The key insight: **Don't fabricate narratives. Research first, then write.**

