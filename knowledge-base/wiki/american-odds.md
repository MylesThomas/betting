# American Odds

**Summary**: American odds are not on a linear scale — never average, sum, or compare them arithmetically.

**Last updated**: 2026-05-20

---

## How they work

American odds express how much you win or lose on a $100 stake.

| Odds | Meaning | Implied probability |
|------|---------|-------------------|
| -110 | Bet $110 to win $100 | 110 / (110+100) = 52.4% |
| -115 | Bet $115 to win $100 | 115 / (115+100) = 53.5% |
| +130 | Bet $100 to win $130 | 100 / (130+100) = 43.5% |

## Implied probability formula

```python
def american_to_implied_prob(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)
```

This is the **raw (vigged)** probability. Over + under implied probs sum to > 1.0 (the vig).

## De-vig (fair probability)

To get the true market probability, normalize each side:

```python
p_over_raw  = american_to_implied_prob(over_odds)
p_under_raw = american_to_implied_prob(under_odds)
total       = p_over_raw + p_under_raw          # e.g. 1.069 at 6.9% vig

p_over_fair  = p_over_raw  / total
p_under_fair = p_under_raw / total              # now sum to 1.0
```

Use de-vigged probs for calibration plots and Brier scores.
Use raw (vigged) probs for edge calculation (edge = model prob − what you're actually paying).

## Profit on a winning bet

```python
def american_profit(odds: float) -> float:
    """Return profit in units on a 1-unit winning bet."""
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)
```

## Rules

1. **Never take the arithmetic mean of American odds.** The mean of -120 and +100 is NOT -10. Convert to implied probability first, average in probability space, then convert back if needed.
2. **Never sum American odds** — they're not additive.
3. **Use the median** when you need a single representative odds value (e.g. median under odds = -115).
4. **Always de-vig before calibration plots** — raw implied probs are biased upward by the vig and will make models look poorly calibrated.

## Common mistakes

- `df['under_odds'].mean()` → returns -92 for a book that's mostly -115. Meaningless. Use `df['under_odds'].apply(american_to_implied_prob).mean()` instead.
- Comparing model P(under) to raw market implied P(under) in a calibration diagram — the market line will appear systematically high. Always de-vig first.

## Typical NBA props vig

- Standard: ~6–7% total (both sides sum to ~1.067)
- Median under odds on points props: **-115** (53.5% implied)
- Break-even hit rate at -115: **53.5%**
- Break-even hit rate at -110: **52.4%**

## Related

- [[roi-and-pnl]]
- [[edge-calibration]]
