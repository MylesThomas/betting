# Spread cover rule (ATS)

**Single source of truth:** `src/odds_utils.did_cover_spread()`

Do **not** reimplement spread-cover logic anywhere else. Use this function so the rule is defined once and tests protect it.

---

## Convention

- **home_spread** = the home team's spread (negative when home is favored).
- Example: home_spread = -10.5 means home is favored by 10.5 points.

## Rule

- **Home covers** when: (home_score - away_score) **>** -home_spread.  
  So with home_spread = -10.5, home must win by more than 10.5 (margin > 10.5).
- **Away covers** when: (away_score - home_score) **>** home_spread.  
  With home_spread = -10.5, away's line is +10.5; away covers when away margin > -10.5 (e.g. lose by 7 → -7 > -10.5 ✓).

## Examples (home_spread = -10.5 → home favored by 10.5, away gets +10.5)

| Scenario              | Final (H–A) | Bet side | Margin vs line                    | Result   |
|------------------------|-------------|----------|-----------------------------------|----------|
| Home fav **covers**    | 85–68       | Home     | Home margin 17 > 10.5 ✓           | **True** |
| Home fav **doesn’t**   | 75–68       | Home     | Home margin 7 < 10.5 ✗           | **False**|
| Away dog **covers**    | 75–68       | Away     | Away lost by 7; 7 < 10.5 → +10.5 covers ✓ | **True** |
| Away dog **doesn’t**   | 85–68       | Away     | Away lost by 17; 17 > 10.5 → +10.5 doesn’t cover ✗ | **False**|

In code (same -10.5 line):

```python
did_cover_spread(85, 68, -10.5, bet_home=True)   # True  (home covers)
did_cover_spread(75, 68, -10.5, bet_home=True)   # False (home doesn’t)
did_cover_spread(75, 68, -10.5, bet_home=False)  # True  (away covers)
did_cover_spread(85, 68, -10.5, bet_home=False)  # False (away doesn’t)
```

## Usage

```python
from src.odds_utils import did_cover_spread

did_cover_spread(home_score, away_score, home_spread, bet_home=True)   # did home cover?
did_cover_spread(home_score, away_score, home_spread, bet_home=False)  # did away cover?
```

## Tests

See `tests/unit/test_odds_utils.py` for locked-in examples. Any change to the rule must pass those tests.
