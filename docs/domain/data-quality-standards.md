# Data Quality Standards

**Status:** 📝 Planned - YOUR EXPERTISE NEEDED  
**Last updated:** 2026-02-13

This document will define what makes betting data "good" or "bad".

## Planned Sections

### Props Data Quality

**Good props data has:**
- Timestamp < 5 minutes old (for live lines)
- All required fields present (player_id, player_name, market, line, odds)
- Odds in valid range (-10000 to +10000)
- Player IDs match known roster
- Team names normalized

**Red flags:**
- Timestamp > 1 hour old
- Missing player_id or odds
- Odds stuck at opening across multiple fetches
- Player listed but marked inactive
- Mismatched player-team pairs

### Line Movement Data Quality

**Significant movement:**
- Totals: >10 cents (e.g., -110 → -120)
- Spreads: >0.5 points
- Moneyline: >15 cents of implied probability
- Multiple books moving together (steam)

**Insignificant (noise):**
- < 5 cent moves on totals
- < 0.25 point spreads
- One book moves, others don't

### Cache Quality

**Roster cache:**
- Updated within 7 days
- All active players present
- Inactive players flagged
- Correct team assignments

**Player-team mappings:**
- Current season only
- Trade updates within 24 hours
- Historical mapping preserved

### S3 Data Quality

**File structure:**
- Consistent naming (YYYY-MM-DD timestamps)
- No gaps in daily data
- No duplicate files for same timestamp
- Reasonable file sizes (not 0 bytes, not absurdly large)

---

**To be written in:** Phase 2.3 (Domain Expertise Encoding)  
**Requires:** Human expertise - what have you learned makes data reliable?
