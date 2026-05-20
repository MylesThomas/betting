# ROI and P&L

**Summary**: Unit staking convention, how P&L is calculated, and what ROI actually means in this codebase.

**Last updated**: 2026-05-20

---

## Unit staking convention

Every bet risks exactly **1 unit**. Wins return the profit (not including stake). Losses cost 1 unit.

```python
def american_profit(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

# P&L per bet:
if outcome == "win":
    pnl += american_profit(odds)   # e.g. +0.909 at -110
else:
    pnl -= 1.0                      # always -1 on a loss
```

## ROI definition

```
ROI = total_pnl / n_bets
```

ROI is **profit per unit bet**, not a percentage. ROI = 0.05 means you made 0.05 units per bet (5 cents per $1 wagered). This is sometimes called "yield."

| ROI | Meaning |
|-----|---------|
| -0.052 | Lost 5.2 cents per $1 — roughly breakeven at -110 with 50% hit rate |
| 0.0 | Exactly breakeven |
| +0.05 | Very good — 5% yield is exceptional in sports betting |
| +9.63 | **Impossible with normal odds — check for extreme long-shots in the bet pool** |

## Break-even hit rates

| Odds | Break-even hit rate |
|------|-------------------|
| -110 | 52.38% |
| -115 | 53.49% |
| -120 | 54.55% |
| -105 | 51.22% |
| +100 | 50.00% |

Formula: `break_even = implied_prob = abs(odds) / (abs(odds) + 100)` for negative odds.

## Odds filter for ROI analysis

Always filter to "main market" odds before computing ROI:
```python
p_mkt = df['under_odds'].apply(american_to_implied_prob)
odds_ok = (p_mkt >= 0.35) & (p_mkt <= 0.65)   # ~-190 to +190 range
```

Without this filter, a single win at +800 odds (8-unit profit) inflates aggregate ROI massively and makes results look far better than they are. These extreme cases are not realistic betting opportunities.

## Common mistakes

- **Reporting ROI as a percentage**: ROI = 0.05 is already a ratio. Don't multiply by 100 and report "5000%."
- **Not filtering extreme odds**: ROI of 9.6x on n=311 bets almost always means long-shot contamination.
- **Confusing ROI with total profit**: ROI = 0.05 on n=100 bets = 5 units total profit. Scale matters.

## Related

- [[american-odds]]
- [[edge-calibration]]
