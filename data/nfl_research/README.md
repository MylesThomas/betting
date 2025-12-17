# NFL Game Research - Week 16

## What's in This Folder

This folder contains research templates and tools for writing informed NFL betting posts.

### Files

1. **`nfl_week16_research_checklist_20251217_003506.md`** ⭐ **START HERE**
   - Markdown checklist with all search queries
   - Space to take notes for each game
   - Organized by game and category

2. **`nfl_week16_search_queries_20251217_003506.csv`**
   - All search queries in CSV format
   - Easy to import into spreadsheets
   - 60+ queries across 5 games

3. **`nfl_week16_research_guide_20251217_003506.json`**
   - JSON template with structured note-taking fields
   - Can be filled in programmatically or manually
   - Includes all query categories

4. **`EXAMPLE_RESEARCH_TO_POST.md`** ⭐ **READ THIS SECOND**
   - Shows how to transform research into post content
   - Before/after examples
   - Template for writing each play

## Your Week 16 Games

Research needed for these 7 games:

1. **CIN @ MIA** - Bengals -1.5 vs Dolphins
2. **NE @ BAL** - Patriots +3.0 vs Ravens
3. **LV @ HOU** - Raiders +14.5 vs Texans
4. **ATL @ ARI** - Cardinals +2.5 vs Falcons
5. **PIT @ DET** - Steelers +7.0 vs Lions
6. **MIN @ NYG** - Giants +3.0 vs Vikings
7. **TB @ CAR** - Buccaneers -3.0 vs Panthers

## Quick Start Guide

### Step 1: Generate Updated Checklist
```bash
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games_cli.py --week 16
```

Then open the latest checklist file.

### Step 2: Research Each Game

For each game, search for:
- **Injury reports** - Who's out? Who's questionable?
- **Recent performance** - Last 3 games for each team
- **Game context** - What actually happened in Week 15?
- **Betting lines** - Opening line, current line, movement

### Step 3: Take Notes

Fill in the notes sections in the markdown file:
- Game storylines
- Key injuries (both teams)
- Recent form (both teams)
- Betting context
- Other relevant info

### Step 4: Write Your Post

Use the template in `EXAMPLE_RESEARCH_TO_POST.md` to combine:
- Your research notes (real context)
- Your luck/efficiency data (your edge)
- The betting line (the opportunity)

## Key Insight

**The Problem:** Your original post had fabricated details (wrong QB, vague context)

**The Solution:** Research first, then write with specific, verifiable details

**The Result:** Posts that sound informed and credible, not generic

## Example Transformation

### Before (Generic):
> "Baltimore's shutout was inflated. New England was unlucky. Bet Patriots +3."

### After (Informed):
> "Baltimore's 24-0 shutout of [TEAM] came after [QB] left in Q1 with [INJURY]. New England lost 31-24 to Buffalo but [SPECIFIC CONTEXT]. The efficiency numbers show BAL +18.6 luck (season high) vs NE -5.7. At +3, we're fading an unsustainable performance."

## Tools Available

Located in `/Users/thomasmyles/dev/betting/scripts/`:

1. **`research_nfl_games.py`** - Generates these research templates
2. **`README_NFL_RESEARCH.md`** - Full documentation
3. **`scrape_nfl_game_context.py`** - Attempted scraper (has issues)

## For Future Weeks

To generate new research templates:

```bash
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games.py
```

This will create new files in this directory with updated timestamps.

## Research Sources to Use

- **NFL.com** - Official injury reports, team news
- **ESPN** - Game recaps, analysis
- **RotoWire** - Injury details, lineup news
- **Action Network** - Betting line movements
- **TeamRankings** - Stats and trends

## Time Estimate

- **Per game:** 10-15 minutes of research
- **7 games:** ~90 minutes total
- **Writing post:** 30-45 minutes with notes

**Total:** ~2 hours for a well-researched post (vs. 30 minutes of fabrication that gets called out)

## Questions?

See `/Users/thomasmyles/dev/betting/scripts/README_NFL_RESEARCH.md` for:
- Full tool documentation
- Future improvement ideas
- API integration options
- Troubleshooting

---

**Remember:** The goal isn't perfection, it's credibility. Real details > vague narratives.

