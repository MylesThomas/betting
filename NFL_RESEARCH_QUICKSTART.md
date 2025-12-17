# NFL Game Research - Quick Start Guide

## TL;DR

You need to research NFL games before writing betting posts. This tool makes it systematic.

## The Fastest Way to Start

```bash
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games_cli.py --week 16
```

This generates a checklist with 60+ search queries for your Week 16 games.

## What You Get

After running the command, you'll have:

1. **Markdown Checklist** - All search queries organized by game
2. **CSV File** - Queries in spreadsheet format
3. **JSON Template** - Structured note-taking template

All saved to: `/Users/thomasmyles/dev/betting/data/nfl_research/`

## Your Workflow

### 1. Generate Research Template (2 minutes)

```bash
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games_cli.py --week 16  # or whatever week number
```

### 2. Open the Checklist (1 minute)

```bash
open /Users/thomasmyles/dev/betting/data/nfl_research/nfl_week16_research_checklist_*.md
```

### 3. Research Each Game (10-15 min per game, 7 games total)

Work through the search queries:
- ✅ Injury reports (who's out?)
- ✅ Recent performance (last 3 games)
- ✅ Game storylines (playoff implications?)
- ✅ Betting context (line movement)

Take notes in the markdown file as you go.

### 4. Write Your Post (30-45 minutes)

Use your notes + your luck data to write informed narratives.

**Template:**
```
[TEAM]'s [SCORE] [win/loss] against [ACTUAL OPPONENT] [REAL CONTEXT 
FROM RESEARCH]. [OTHER TEAM] [ACTUAL RECENT PERFORMANCE]. The efficiency 
numbers show [TEAM] posted a [LUCK VALUE] luck rating [CONTEXT], while 
[OTHER TEAM]'s [LUCK VALUE] suggests [CONTEXT]. At [SPREAD], we're 
[BETTING ANGLE WITH REAL CONTEXT].
```

## Example: Before vs After

### ❌ Before (Fabricated)
> "Baltimore's shutout was inflated. New England was unlucky. Bet Patriots +3."

### ✅ After (Researched)
> "Baltimore's 24-0 shutout of the Giants came after Daniel Jones left in Q1 with a concussion, forcing third-stringer Tommy DeVito to play three quarters. New England lost 31-24 to Buffalo but covered the 10-point spread and outgained them 385-340. The efficiency numbers show BAL +18.6 luck (season high) vs NE -5.7. At +3, we're fading an unsustainable performance."

## Commands Cheat Sheet

```bash
# Generate research for Week 16 (default games)
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games_cli.py --week 16

# Generate for Week 17 with custom games
python research_nfl_games_cli.py --week 17 --games "KC,TEN" "PHI,WAS" "BUF,CLE"

# See all options
python research_nfl_games_cli.py --help

# Open the checklist
open ../data/nfl_research/nfl_week*_research_checklist_*.md

# View all research files
ls -lh ../data/nfl_research/
```

## Files & Documentation

### Start Here
- **This file** - Quick start guide
- `/data/nfl_research/README.md` - Workflow guide
- `/data/nfl_research/EXAMPLE_RESEARCH_TO_POST.md` - Writing examples

### Tools
- `/scripts/research_nfl_games_cli.py` - Main research tool ⭐
- `/scripts/research_nfl_games.py` - Alternative version
- `/scripts/README_NFL_RESEARCH.md` - Technical docs

### Your Research Files
- `/data/nfl_research/nfl_week16_research_checklist_*.md` - Your checklist ⭐
- `/data/nfl_research/nfl_week16_search_queries_*.csv` - Queries CSV
- `/data/nfl_research/nfl_week16_research_guide_*.json` - Notes template

## Time Breakdown

- Generate template: **2 minutes**
- Research 7 games: **70-105 minutes** (10-15 min each)
- Write post: **30-45 minutes**
- **Total: ~2 hours** for a well-researched post

## What This Solves

✅ No more fabricating QB names  
✅ No more vague narratives  
✅ No more getting called out for wrong details  
✅ Posts sound informed and credible  
✅ Systematic research process  

## What This Doesn't Do

❌ Doesn't automatically fetch data (you still need to search)  
❌ Doesn't write the post for you  
❌ Doesn't replace your analysis (just adds context)  

## For Week 16 Specifically

Your 7 games to research:
1. **CIN @ MIA** - Bengals -1.5 vs Dolphins
2. **NE @ BAL** - Patriots +3.0 vs Ravens
3. **LV @ HOU** - Raiders +14.5 vs Texans
4. **ATL @ ARI** - Cardinals +2.5 vs Falcons
5. **PIT @ DET** - Steelers +7.0 vs Lions
6. **MIN @ NYG** - Giants +3.0 vs Vikings
7. **TB @ CAR** - Buccaneers -3.0 vs Panthers

Generate the updated checklist with all 7 games:

```bash
cd /Users/thomasmyles/dev/betting/scripts
python research_nfl_games_cli.py --week 16
```

Then open the latest checklist and start researching.

## Need More Help?

1. **Workflow questions:** Read `/data/nfl_research/README.md`
2. **Writing questions:** Read `/data/nfl_research/EXAMPLE_RESEARCH_TO_POST.md`
3. **Technical questions:** Read `/scripts/README_NFL_RESEARCH.md`
4. **Full summary:** Read `/SUMMARY_NFL_RESEARCH_TOOL.md`

## The Bottom Line

**Old way:** Write post in 30 minutes with fabricated details → get called out

**New way:** Research for 90 minutes with real details → build credibility

**This tool:** Makes research systematic instead of ad-hoc

---

**Remember:** Real details > fabricated narratives. Your luck data is the edge, but context is the credibility.

