# Live Odds Plot Improvements Checklist

## Session: 2026-02-16

### Quick Wins (Visual Clarity)
- [ ] 1. Add consensus line (median across all books) - thick reference line
- [ ] 2. Add opening line reference - dotted horizontal at game start
- [ ] 3. Add quarter markers - vertical lines at Q1/Q2/Q3/Q4 transitions

### Context & Annotations
- [ ] 4. Total movement annotation - "Spread moved 8.5 pts" in subtitle
- [ ] 5. Steam markers - highlight when line moves >0.5 pts/min
- [ ] 6. Closing line vs result - show if favorites covered

### Analysis
- [ ] 7. Disagreement zones - highlight when books differ >2 pts
- [ ] 8. Implied win % - convert ML to probability
- [ ] 9. Data quality indicators - fade opacity when stale

### Polish
- [ ] 10. Interactive Plotly version - hover tooltips, zoom/pan
- [ ] 11. Multi-game dashboard - 6 games on one page
- [ ] 12. Config file - YAML to control features

---

## Implementation Notes

### Current Status
Starting with improvements 1-3 (consensus line, opening line, quarter markers)

### Test Game
Using: Milwaukee Bucks games for testing
