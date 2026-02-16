# Market Mechanics

**Status:** 🚧 DRAFT - Needs your review  
**Last updated:** 2026-02-13  
**Source:** General betting knowledge + YOUR corrections needed

**⚠️ REVIEW INSTRUCTIONS:**
- ✅ Mark what's accurate
- ❌ Flag what's wrong or misleading  
- ➕ Add YOUR specific thresholds and practices
- 📝 Add real examples from your work

---

## How Betting Lines Work

### Opening Lines

**Who sets them:**
- Sportsbook oddsmakers create the initial line
- Use statistical models, power ratings, historical data
- For major markets, books also copy/adjust from market-makers (Pinnacle, Circa, etc.)

**For player props specifically:**
- Created by algorithms/models
- Reviewed by traders
- Adjusted quickly based on early action
- Lower limits than game lines (books more willing to move aggressively)

**⚠️ YOUR INPUT NEEDED:**
- Do you see consistent differences in opening lines across books?
- Which books tend to have "softer" opening lines?

---

## Line Movement: Why Lines Change

### Primary Drivers

**1. Betting action (most common)**
- Heavy money on one side forces book to adjust
- Goal: Balance risk or respect sharp information
- Not always about balancing action 50/50

**2. Sharp money signals**
- Large bets from known sharp bettors
- Coordinated syndicate action
- Books respect this more than public money

**3. Breaking news**
- Injury reports (player ruled out/questionable)
- Weather changes (outdoor sports)
- Lineup changes, coaching decisions
- Trade deadline moves

**4. Market correction**
- Other books move first (sharper books)
- Followers adjust to stay competitive
- Prevents arbitrage opportunities

### Sharp vs Public Money

**Sharp money:**
- Professional bettors with proven track records
- Larger, strategic wagers
- Early betting (right when lines open or on news)
- Sportsbooks respect and move lines
- Often triggers **reverse line movement** (RLM)

**Public money:**
- Casual fans betting on emotion/hype
- Smaller bets, higher volume
- Late betting (close to game time)
- Favors favorites, popular teams, overs
- Books may not move lines despite public action

**Reverse Line Movement (RLM):**
- Line moves opposite to public betting percentages
- Example: 75% of bets on Chiefs -6, but line moves to Chiefs -4.5
- Signals sharp money on underdog despite public preference
- **Not a guarantee** (sharp bettors lose too)

**⚠️ YOUR INPUT NEEDED:**
- Do you track sharp vs public money in your analysis?
- Have you observed reliable RLM patterns in NBA props?
- How quickly do you see prop lines move after opening?

---

## Steam Moves

### What is Steam?

**Steam = Coordinated sharp action across multiple books simultaneously**

**Characteristics:**
- Multiple sportsbooks move the same direction quickly (within minutes)
- Large, synchronized line shifts
- Triggered by sharp syndicates or breaking news
- Often happens before public knows why

**Example:**
- 10:00 AM: Curry Over 29.5 at -110 across all books
- 10:03 AM: DraftKings moves to 30.5 at -110
- 10:04 AM: FanDuel moves to 30.5 at -110
- 10:05 AM: BetMGM moves to 30.5 at -110
- **Steam detected** → Sharp money hit all books at once

**Not steam:**
- One book moves, others don't follow
- Gradual drift over hours
- Book-specific limit or error

### How to Detect Steam

**Signals:**
1. **Speed:** Line moves 0.5+ points in < 5 minutes
2. **Consensus:** 3+ major books move together
3. **Direction:** All move same way (not books splitting)
4. **Magnitude:** Significant move (not just vig adjustment)

**⚠️ YOUR INPUT NEEDED:**
- What are YOUR specific thresholds for detecting steam?
- How many books do you require to move together?
- What time window do you use? (5 min? 10 min?)
- Do you differentiate steam vs injury news?

---

## Line Movement Significance: When to Care

### General Thresholds (Industry Standard)

**Player props:**
- **< 0.25 points:** Noise / vig adjustment (ignore)
- **0.5 - 1.0 points:** Potentially significant (investigate)
- **1.0+ points:** Very significant (sharp action or news)

**Odds changes:**
- **< 5 cents:** Noise (e.g., -110 to -115)
- **5-10 cents:** Moderate significance
- **10+ cents:** High significance (e.g., -110 to -120)

**⚠️ YOUR INPUT NEEDED:**
- What are YOUR actual thresholds from your work?
- You mentioned >10 cents for totals, >0.5 for spreads earlier
- What about for player props specifically?
- Do you have different thresholds for different markets?

---

## Arbitrage (Arb) Opportunities

### What is Arbitrage?

**Arbitrage = Betting both sides across different books to guarantee profit**

**How it works:**
- Book A: Curry Over 29.5 at -105
- Book B: Curry Under 29.5 at -105
- Combined implied probability < 100% → Guaranteed profit

**Example calculation:**
- Over at -105: 51.22% implied
- Under at -105: 51.22% implied
- Total: 102.44% → **No arb** (would lose 2.44%)

- Over at +100: 50% implied
- Under at +100: 50% implied
- Total: 100% → **Break-even** (rare)

- Over at +105: 48.78% implied
- Under at +100: 50% implied
- Total: 98.78% → **1.22% arb** (guaranteed profit)

### Why Arbs Exist

**Temporary mispricing:**
- Different books use different models
- One book slow to adjust to news
- Liquidity differences across books
- Promotional lines (new user offers)

**Timing differences:**
- Sharp books (Pinnacle) move first
- Slower books lag behind
- Creates brief arbitrage window

### Why They Close Quickly

**Speed:**
- Arbs typically last **seconds to minutes**
- Sharp bettors correct inefficiencies fast
- Automated bots scan for arbs constantly

**Book reactions:**
- Limit or ban arbitrage bettors
- Void bets if detected
- Reduce max bet limits

**Profit margins:**
- Typical arbs: 1-3% of total stake
- Need large capital to make meaningful money
- After fees/limits, often not worth it

**⚠️ YOUR INPUT NEEDED:**
- Do you actively look for arbitrage opportunities?
- What arb % makes it worth the effort for you?
- Have you been limited by books for arbing?
- Do you use any arb detection tools?

---

## Market Efficiency

### What is Market Efficiency?

**Efficient market = Prices accurately reflect all available information**

**Characteristics:**
- Hard to find edges (mispriced lines)
- Sharp bettors active
- Quick price corrections
- Low vig / tight spreads

**Inefficient market = Mispricing exists that can be exploited**

**Characteristics:**
- Slower price corrections
- Fewer sharp bettors
- Higher vig
- Opportunities for data-driven bettors

### Efficiency by Market Type

**Most efficient (hardest to beat):**
- **NFL game lines** (spreads/totals) - <1% can beat long-term
- NFL player props (when available)
- High-volume markets (primetime games)
- Moneylines on heavy favorites

**Moderately efficient (still beatable with edges):**
- **NBA game lines** (spreads/totals) - Sharp but more opportunity
- NBA player props (main markets: points, rebounds, assists)
- High-volume NBA props

**Less efficient (potential edges):**
- Low-volume NBA props (blocks, steals, double-doubles)
- Alternate lines
- Niche markets (first basket, etc.)
- Less popular teams/games

**⚠️ YOUR INPUT NEEDED:**
- Where have you found the most edge in your work?
- NBA player points props efficient or still beatable?
- What markets do you avoid because they're too efficient?
- Confirm: NFL too efficient to bother with (<1% can beat)?

---

## Sportsbook Differences: Sharp vs Soft Books

### What is a "Sharp" Book?

**Sharp books (market-making books)** are sportsbooks that:
- Take **larger limits** ($10k-$100k+ on major markets)
- Move lines **aggressively** based on betting action
- **Welcome sharp bettors** (don't limit winners quickly)
- Use betting action itself as a **core pricing signal**
- Operate on **lower margins** (reduced vig/juice)

**Examples of sharp books:**
- **Pinnacle** (international, lowest vig in industry)
- **Circa** (Las Vegas-based)
- **Bookmaker** (offshore)
- Betting exchanges (Betfair, etc.)

**Philosophy:**
- Price discovery through high volume
- Efficient markets
- Accept that sharps will win sometimes
- Make money on volume, not limiting winners

### What is a "Soft" Book?

**Soft books (retail books)** are sportsbooks that:
- **Copy or shade lines** from sharp books
- Offer more **promos, SGPs, boosts** (marketing-heavy)
- Use **higher hold percentages** (more vig)
- **Limit winning players** quickly (within days/weeks)
- Move **slower on props** (especially early)

**Examples of soft books:**
- **DraftKings**
- **FanDuel**
- **BetMGM**
- **Caesars**

**Philosophy:**
- Manage risk by limiting winners
- Target recreational bettors
- Higher margins per bet
- Customer segmentation over price discovery

---

## How Sharp vs Soft Books Affect Markets

### Price Discovery Process

**Sharp books lead, soft books follow:**

1. **Sharp books** (Pinnacle, Circa) post opening lines
2. **Sharp bettors** hit the lines (minutes after posting)
3. **Sharp books move** aggressively based on this action
4. **Soft books observe** and adjust (copy the move)
5. Market reaches equilibrium

**For major NFL sides/totals:**
- Pinnacle moves first (sharp money)
- Retail books follow within minutes
- Market becomes efficient quickly

**For NBA player props:**
- Sharp books move fast
- Soft books slower to adjust (lower limits, more cautious)
- **Inefficiencies persist longer** (opportunity window)

### Line Movement Behavior Differences

**Sharp book line movement:**
- Large respected bet ($50k) → **Move immediately**
- May move **before other books** (leading indicator)
- Will move off "bad" numbers even without public action
- Trust their models + respect sharp action

**Example:**
- Sharp bettor hits NFL total $50K at 44.5
- Pinnacle moves to 45.0 instantly
- Retail books follow within 5-10 minutes

**Soft book line movement:**
- Observe sharp book moves → **Copy half the move**
- Wait to see if others follow
- More **sensitive on props** (lower limits)
- Will **limit players** rather than keep moving number

**Example:**
- Sharp bettor bets NBA prop at 24.5 points
- Soft book may:
  - Move to 25.5 quickly
  - Slash user's limits immediately
  - Flag account for review

---

## Player Props: Where the Gap Widens

### Sharp Books on Props

- **Lower limits** than game lines (but still $5k-$10k)
- Still **tolerate winners**
- Efficient but **beatable early** (first 5-10 minutes)
- Will move 1-2 points on moderate action

### Soft Books on Props

- **Much softer opening lines** (wider spreads from fair value)
- **Slower adjustment** early (first 30 minutes)
- Heavy **account monitoring** (flag winners fast)
- Move on **smaller bet sizes** ($500-$1k can move line)
- **Opportunity:** Soft books misprice more, but limit you faster

**Why this matters:**
- **NBA props attacked heavily at open** (sharps hit soft books first)
- **NFL props more efficient** earlier (fewer games, more focus)
- Books react **much faster in NBA markets** (daily volume)

---

## Opening Line Differences

### Major Markets (NFL spreads/totals)

**Differences usually small:**
- 0.5 to 1 point spread between books
- Sharp books: Tighter juice (-107/-103 splits)
- Soft books: Standard -110/-110 (higher hold)

### Player Props (especially NBA)

**Differences can be significant:**
- **1-3 point gaps** common on opening
- **Juice discrepancies** wide (-105/-115 vs -110/-110)

**Soft books tend to:**
- Copy median market line
- Shade toward **public bias** (overs, favorites, star players)
- Example: Public loves Curry overs → DK shades line up 0.5-1 point

**Sharp books:**
- Shade toward **respected action** (sharp money)
- Price closer to **true probability**
- Less influenced by public sentiment

**⚠️ YOUR INPUT:**
- Do you line shop across books for best prices?
- Which books consistently have better NBA prop prices?
- Have you been limited by soft books?

---

## Why This Matters for Your Strategy

### 1. Efficiency Implications

**Sharp books = closer to true price**
- Pinnacle closing line is "the truth"
- Use as benchmark for CLV measurement
- Harder to beat, but if you do, you have real edge

**Soft books = more opportunity (but limits)**
- Opening lines softer (more mispricing)
- Window of opportunity early (first 30 min)
- But will limit you if you win

### 2. Arbitrage Opportunities

**Sharp vs soft discrepancies create arb edges:**
- Pinnacle: Curry Under 29.5 at -105
- DraftKings: Curry Under 29.5 at +100
- Arbitrage: Bet under on both sides (if opposite side works)

### 3. Line Movement Signals

**Sharp book moves often indicate:**
- Injury information (not public yet)
- Model-driven action (algo-based betting)
- Syndicate play (coordinated sharp action)

**Soft book moves often indicate:**
- Copying sharp books (follow the leader)
- Liability management (too much one-sided action)
- Public money (copying other soft books)

**⚠️ YOUR INPUT:**
- Do you track which book moves first?
- Use sharp book moves as signal for where to bet soft books?
- How do you use line movement in your strategy?

---

## Risk Management Philosophy

| Aspect | Sharp Book | Soft Book |
|--------|------------|-----------|
| **Response to winners** | Move price | Limit player |
| **Margins** | Lower (1-2%) | Higher (4-5%+) |
| **Strategy** | Price discovery | Customer segmentation |
| **Sharp bettors** | Accept them | Restrict them |
| **Limits** | High ($10k-$100k) | Low-Medium ($500-$5k) |
| **Props limits** | Medium ($5k-$10k) | Low ($100-$1k) |
| **Line movement** | Aggressive, fast | Conservative, slower |
| **Account longevity** | Years (if you win) | Weeks-Months (if you win) |

---

## DraftKings, FanDuel, BetMGM: Specifics

**⚠️ YOUR INPUT NEEDED - This is all you:**
- Which books have best prices for NBA props?
- Which books move lines fastest?
- Which books have highest limits?
- Which books have best promos?
- Data quality differences (API reliability)?
- Any books you prefer/avoid and why?
- Have you been limited? By which books?

---

## Your Actual Approach: Small Stakes on Soft Books

**YOUR STRATEGY:**

**Unit size:** ~$20 per bet

**Why this works:**
- Small enough to **blend with recreational flow**
- Don't trigger scrutiny or limits
- Can operate on soft books indefinitely
- Capture pricing inefficiencies without moving markets

**Edge sources (your approach):**

1. **Target early lines** before market correction
   - Soft books slower to adjust in first 10-30 minutes
   - Get down before sharp money forces correction

2. **Exploit sharp vs retail discrepancies**
   - Compare Pinnacle (sharp) vs DK/FD (soft)
   - When soft book is off by 1-2 points, bet it
   - Market will correct, you got better number

3. **Play slower-moving prop markets**
   - NBA props (especially secondary markets)
   - Soft books don't adjust as fast as game lines
   - More time to find and exploit edges

4. **Take advantage of public bias shading**
   - Soft books shade toward public preferences:
     - Overs (public loves overs)
     - Star players (Curry, LeBron overbet)
     - Favorites (public avoids underdogs)
   - When bias creates mispricing, bet opposite

**Key insight:** At small stake levels, soft books are an **opportunity** not a **constraint**
- Not large enough to move markets
- Can consistently capture small pricing edges
- Edge compounds over many bets (volume strategy)
- Don't need huge wins per bet, need many +EV bets

**Why you don't use sharp books (Pinnacle):**
- **Requested API access** (emailed Pinnacle) but don't currently have it
- Currently use soft books via The Odds API (DK, FD, BetMGM, etc.)
- If/when Pinnacle access granted:
  - Use as **reference** for true line (CLV measurement)
  - Compare soft book prices to Pinnacle
  - Identify where soft books are mispriced
- For now: Soft books' inefficiencies are your advantage

---

**General differences (industry knowledge):**

**DraftKings:**
- High liquidity, mainstream
- Competitive odds (copies Pinnacle closely)
- Good API data
- Limits: Low-moderate on props ($500-$2k typical)
- Will limit winners relatively quickly

**FanDuel:**
- Similar to DK
- Sometimes slower to adjust lines (opportunity)
- Good promos
- Limits: Similar to DK

**BetMGM:**
- Part of larger MGM network
- Can be slower to move (more opportunity)
- Lower limits on props
- Less sharp clientele

**Pinnacle (reference, not available in most US states):**
- Industry standard for "true" line
- Lowest vig
- Highest limits
- Used by other books as reference

---

## Key Takeaways for Agents

**When you see in data/code:**

- **Line movement > 0.5 points in < 10 min** → Investigate (likely steam or news)
- **All books moving together** → Steam detected
- **One book out of line** → Potential arb or book error
- **Heavy RLM** → Sharp money opposite to public

**Remember:**
- Line movement is a signal, not a guarantee
- Always verify moves against news (injuries, lineups)
- Opening lines ≠ closing lines
- Closing line is the "sharpest" number
- Market efficiency varies by sport/market

**⚠️ YOUR FINAL REVIEW:**
- What did I get wrong?
- What's missing that's critical for your work?
- What general knowledge doesn't apply to NBA props?
- Any specific book behaviors or patterns you've observed?

---

## Related Documents

- `docs/domain/betting-fundamentals.md` - Odds, vig, prop basics
- `docs/domain/data-quality-standards.md` - What makes line movement data "good"
- `docs/domain/edge-cases.md` - How news events affect lines
- `docs/domain/nba-vs-nfl.md` - Market efficiency differences

---

## Implementation Notes

**Relevant code:**
- `src/line_steam_utils.py` - Line movement detection logic
- `config/line_steam_config.yaml` - Your actual thresholds
- `scripts/check_line_steam.py` - Steam detection script
- `scripts/build_arb_cache.py` - Arbitrage detection

**⚠️ TODO:** Review those files to extract your ACTUAL thresholds and add them here!
