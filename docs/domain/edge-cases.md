# Edge Cases in Sports Betting

**Status:** 🚧 DRAFT - Needs your review  
**Last updated:** 2026-02-13

**⚠️ REVIEW INSTRUCTIONS:**
- ➕ Add edge cases YOU'VE actually encountered
- 📝 Add how YOUR code handles them
- ✅ Mark what's accurate
- ❌ Flag what's wrong or doesn't apply to NBA props

---

## Postponed Games

### What Happens

**Game postponed before start (COVID, weather, etc.):**
- Most books: **Props void**, bets refunded
- Some books: Props **rollover** to rescheduled date (rare)
- Check book's specific rules

**Timing matters:**
- Postponed > 24 hours before → Usually void immediately
- Postponed < 1 hour before → May take time to process voids

### How to Handle in Code

**Data implications:**
- Props with game_date in past but no result → Check if postponed
- Don't count as losses if voided
- Historical data: Flag postponed games (don't use in models)

**YOUR ACTUAL APPROACH:**
- **Not explicitly handled** in current code
- Postponed games naturally **fail to join** between data sources
- Date + team combo doesn't match up (game scheduled but didn't happen)
- Result: Postponed games are **automatically excluded** from analysis
- No special flag needed - just no matching game data

**Implication for agents:**
- If props data exists but no game result → Likely postponed
- Data pipeline handles this naturally via failed joins
- Don't need complex postponed game detection logic

---

## Late Scratches (Player Ruled Out)

### What Happens

**Player ruled out before game:**
- **Props lines stay posted** (books don't always pull them immediately)
- You can often **still bet them** up until day of game
- Sometimes even **into the evening** for really late scratches
- Books may not remove the market until minutes before tip-off

**If player doesn't enter the game:**
- **Bets are voided** (almost all books)
- You get your money back
- Bet doesn't count as win or loss

**Key principle:** If player plays **any minutes**, bet stands. If **DNP** (did not play), bet voids.

### Real-World Impact

**Example:**
- You bet Curry Over 29.5 at 6:00 PM
- Curry ruled out (injury) at 7:15 PM (15 min before tip-off)
- Curry DNP (doesn't enter game)
- Your bet: **Voided** (money back)

**Risk is low:**
- Unlike what most think, you're protected if player doesn't play
- Worst case: Bet voids, you break even
- Exception: If player plays 1 minute then exits → Bet stands (you likely lose)

### Strategy Implications

**The real risk is:**
- Player plays **limited minutes** (not DNP)
- Example: Plays 8 minutes, exits with injury, scores 2 points
- Bet stands, you likely lose
- This is why late injury news matters (reduced minutes, not DNP)

**Lines stay up:**
- Books slow to pull props for late scratches
- Can accidentally bet on a player who won't play
- But you're protected (void if DNP)
- Sharp bettors may know injury info before public

**⚠️ YOUR INPUT:**
- Do you check injury reports before betting?
- Have you accidentally bet players who were scratched?
- How often do you see voided bets in your data?

---

## Line Freezes

### What Happens

**Book stops taking action on a market:**
- Props become unavailable (can't place bets)
- Odds disappear or show "not available"
- Can happen to entire game or specific player

### Why It Happens

**Common reasons:**
1. **Breaking injury news** (star player questionable)
2. **Suspicious betting patterns** (sharp money flooding one side)
3. **Technical issues** (pricing error detected)
4. **Limit reached** (book hit max liability on one side)

**How long it lasts:**
- Minutes to hours
- Sometimes market never reopens (pulled entirely)

### How to Detect

**In your data:**
- Props present in earlier fetch, missing in later fetch
- All books drop same player's markets simultaneously
- Odds change from -110 to "N/A"

**⚠️ YOUR INPUT:**
- Do you track line freezes in your data pipeline?
- How do you detect a market being pulled?
- Do you have alerts for when key props disappear?

---

## Stat Corrections

### What Happens

**Official scorer changes ruling after game:**
- Assist becomes no assist
- Rebound assigned to different player
- Steal credited to different player

**Betting implications:**
- **Props usually stand as graded** (based on initial box score)
- Books very rarely regrade based on stat corrections
- Exception: Major errors caught within hours

### Why It Matters

**For analysis:**
- Historical data may not match final official stats
- If using official NBA stats, they might differ from betting results
- Important for backtesting (which stat source did books use?)

**⚠️ YOUR INPUT:**
- Have you encountered stat correction issues?
- Which stat source do you use? (NBA API, ESPN, etc.)
- Do you ever see discrepancies between sources?

---

## Overtime Handling

### Standard Rule

**Player props INCLUDE overtime** unless explicitly stated "regulation only"

**Example:**
- Curry Over 29.5 points
- Regulation: 28 points
- Overtime: 4 points
- Total: 32 points → **Over wins**

### Rare Exception

**Some books offer "regulation only" props:**
- Explicitly labeled (e.g., "Points - Regulation Only")
- OT points don't count
- Less common, higher vig

### Data Consistency

**For historical analysis:**
- Ensure OT is included/excluded consistently
- Box scores from NBA API include OT by default
- If filtering, document clearly

**⚠️ YOUR INPUT:**
- Do you filter out OT games in your analysis?
- Or include them since props count OT?
- How do you handle this in backtesting?

---

## Injuries Mid-Game

### What Happens

**Player injured and exits early:**
- **Props usually stand** (not voided)
- If he played any minutes, bet counts
- Rare exception: Some books void if player plays < 5 minutes

**Example:**
- Curry Over 29.5 points
- Curry scores 8 points in Q1, then injured (out for game)
- Your bet: **Likely loses** (Under wins at 8 points)

### Risk Factor

**This is normal variance:**
- Part of betting props
- Can't be predicted
- Included in long-term edge calculations

**⚠️ YOUR INPUT:**
- Do you track "games where player exited early" in your data?
- Is this a significant risk factor in your models?

---

## Trade Deadline

### What Happens

**Player traded mid-season:**
- Player-team mappings change immediately
- Props may be unavailable during transition (24-48 hours)
- Historical data needs careful handling

### Data Implications

**Player-team cache:**
- Must update within 24 hours of trade
- Historical mappings preserved (for backtesting)
- Current team vs historical team distinction

**Props behavior:**
- New team, new role, new usage (affects projections)
- May take 5-10 games for props to stabilize
- Opportunity: Books slow to adjust projections

**⚠️ YOUR INPUT:**
- How do you handle player trades in your data pipeline?
- Do you have alerts for trade deadline activity?
- Have you found edges on newly-traded players?

---

## Load Management / DNP-Rest

### What Happens

**Star player sits out for rest (no injury):**
- Announcement usually comes 1-3 hours before game
- Props may still be available until announcement
- Late scratches (covered above)

**Common in NBA:**
- Back-to-backs (second night)
- End of season (playoff-bound teams)
- Veteran stars (load management)

### Strategy Implications

**Risk factors:**
- Older stars on back-to-backs (LeBron, etc.)
- Teams locked into playoff seed (late season)
- Coach history (Popovich notorious for rest)

**⚠️ YOUR INPUT:**
- Do you factor load management risk into models?
- Avoid betting on back-to-backs?
- Track coaches who rest players frequently?

---

## Back-to-Back Games

### What Happens

**Team plays consecutive nights:**
- More fatigue (affects performance)
- Stars may get reduced minutes
- Higher injury risk
- Sometimes stars sit entirely (see load management)

### Impact on Props

**General patterns (not guaranteed):**
- Slightly lower scoring on back-to-back
- But books adjust lines for this
- Not automatic value without mispricing

**⚠️ YOUR INPUT:**
- Do you track back-to-backs in your models?
- Have you found statistical edge on B2B games?
- Or is it priced in?

---

## Blowouts & Garbage Time

### What Happens

**Game becomes non-competitive (20+ point lead):**
- Starters sit in 4th quarter
- Bench players get extended minutes
- Stat accumulation stops for starters

### Impact on Props

**For starters:**
- May not reach over (stopped at 25 min played instead of 35)
- Significant risk for overs

**For bench:**
- Opportunity to hit overs (garbage time minutes)

### Predictability

**Hard to predict:**
- Blowouts happen, but when?
- Books don't price this in perfectly (too variable)
- More risk for high-usage stars (need full game to hit overs)

**⚠️ YOUR INPUT:**
- Do you avoid props on games with large spreads?
- Track blowout frequency by team?
- Is this a significant factor in your models?

---

## Game Script Changes

### What Happens

**Game flow affects player usage:**
- Blowout → Starters sit (covered above)
- Close game → Stars play full minutes
- Foul trouble → Reduced minutes
- Early foul trouble → Compensate later (more aggressive)

### Foul Trouble Specifically

**Player with 2 fouls in Q1:**
- Often sits rest of 1st half
- Reduced minutes overall
- May be more aggressive in 2nd half (make up stats)

**Data signal:**
- Props may move if foul trouble observed
- Live betting markets adjust immediately

**⚠️ YOUR INPUT:**
- Do you use live/in-game data?
- Or only pre-game props?

---

## Multiple Games Same Day (Rare)

### What Happens

**NBA occasionally has doubleheaders (team plays 2 games same day):**
- Very rare
- Usually makeup games
- Affects props significantly

**Data implications:**
- Which game does the prop refer to?
- Must specify game_id, not just game_date
- Player may not play both games

**⚠️ YOUR INPUT:**
- Have you encountered this?
- How does your data structure handle multiple games per day?

---

## Summary: Edge Cases Checklist

**For agents working with props data:**

```
✅ Check for postponed games (void bets, exclude from analysis)
✅ Handle late scratches (player DNP but bet stands)
✅ Detect line freezes (props disappear from market)
✅ Include OT in stats (unless "regulation only")
✅ Update player-team mappings after trades
✅ Flag back-to-back games (fatigue factor)
✅ Be aware of blowout risk (starters sit)
✅ Handle foul trouble (reduced minutes)
```

**⚠️ YOUR PRIORITIES:**
Which of these edge cases matter most for YOUR betting strategies?

---

## Related Documents

- `docs/domain/betting-fundamentals.md` - Basic prop mechanics
- `docs/domain/market-mechanics.md` - How news affects lines
- `docs/domain/data-quality-standards.md` - How to detect data issues
- `docs/domain/nba-vs-nfl.md` - NBA-specific patterns

---

## For Agents

**When encountering unusual data:**
1. Check if game was postponed
2. Check if player was scratched late
3. Check if prop line was pulled/frozen
4. Verify player is on correct team (trades)
5. Check if game went to OT (affects stats)
6. Flag outliers for human review

**Remember:** Edge cases are normal. Don't try to handle every scenario perfectly—focus on the 95% case and flag the rest.
