# Model Evaluation

**Summary**: How to correctly interpret Brier scores, log-loss, and ROI when evaluating betting models.

**Last updated**: 2026-05-20

---

## Brier score

Measures mean squared error of probability predictions. Lower is better.

```python
brier = mean((p_pred - y_actual)^2)
```

- **Perfect model**: 0.0
- **Coin flip (p=0.5 always)**: 0.25
- **NBA points props market**: ~0.250
- **Typical model improvement**: 0.001–0.003 over market baseline

A Brier improvement of 0.001 is real but small. It does NOT guarantee profitable betting — the edge may not be large enough to overcome vig.

## Calibration plot

Shows predicted probability vs actual hit rate. A perfectly calibrated model lies on the diagonal.

**Always de-vig market probabilities before plotting calibration.** Raw implied probs include vig (sum > 1), which makes the market baseline look systematically miscalibrated even when it isn't.

```python
# Wrong — market will look off-diagonal due to vig:
frac_pos, mean_pred = calibration_curve(y, market_implied_prob_raw)

# Correct — de-vig first:
p_over_dv  = p_over_raw  / (p_over_raw + p_under_raw)
p_under_dv = p_under_raw / (p_over_raw + p_under_raw)
frac_pos, mean_pred = calibration_curve(y, p_under_dv)
```

Also restrict x-axis to the realistic range (0.40–0.65 for NBA props) — calibration at extremes is unreliable.

## Train/test discipline

- **Always use time-based splits** — random splits leak future information
- **Walk-forward is the gold standard**: train on season N, test on season N+1
- **Never evaluate ROI on the same season used for training** — even for hyper-parameter selection

## Interpreting ROI results

| ROI | n_bets | Verdict |
|-----|--------|---------|
| Any | < 100 | Noise. Ignore. |
| < 0 | > 1000 | Model has no edge at this threshold |
| ~0 | > 1000 | Breaking even against vig — not profitable |
| +0.02 to +0.05 | > 500 | Potentially real edge — check in second fold |
| > 0.10 | Any | Suspect — check for odds filter contamination or data leak |

## XGBoost on small samples

XGBoost consistently overfits on single-season data for NBA props (~15k rows). It beats market on train but underperforms on test. Prefer logistic regression unless you have 3+ seasons of training data.

## Common mistakes

- Reporting Brier improvement without checking if it survives in a second test fold
- Trusting ROI from a test set that is only playoffs (structurally different from regular season training data)
- Confusing Brier improvement with profitability — they measure different things

## Related

- [[edge-calibration]]
- [[roi-and-pnl]]
- [[american-odds]]
