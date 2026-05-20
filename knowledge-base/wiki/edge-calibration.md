# Edge Calibration

**Summary**: How model edge is computed, what shrinkage does, and the parametric normal approach used in this codebase.

**Last updated**: 2026-05-20

---

## What is edge?

Edge = model's probability estimate − market's implied probability (raw, vigged).

```python
edge_under = p_under_model - p_under_market   # positive = bet the under
edge_over  = p_over_model  - p_over_market    # positive = bet the over
```

Use the **raw (vigged)** market probability here, not the de-vigged version. Edge must beat the vig to be profitable.

## Shrinkage

Raw model predictions are often overconfident. Shrinkage anchors the prediction back toward the market consensus:

```python
# shrink=0.0 → use model fully; shrink=1.0 → use market fully
p_adj = p_market + (1 - shrink) * (p_model - p_market)
edge  = p_adj - p_market
     = (1 - shrink) * (p_model - p_market)
```

Standard sweep: shrink ∈ {0.0, 0.25, 0.50}.

Higher shrinkage → fewer and more conservative bets → typically better ROI stability.

## Parametric normal approach (regression models)

When using a regression model that outputs a point prediction `yhat`:

```python
from scipy.stats import norm

sigma = residuals.std()                        # global residual std
p_over = norm.sf(line, loc=yhat, scale=sigma)  # P(actual > line)
p_under = norm.cdf(line, loc=yhat, scale=sigma)

# With shrinkage:
mean_adj = consensus_line + (1 - shrink) * (yhat - consensus_line)
p_over   = norm.sf(line, loc=mean_adj, scale=sigma)
```

**Sigma options:**
- Global: one sigma for all players (simpler, used in points model)
- Per-player rolling: player-specific variance (used in rebounds model — better)

## Direct classification approach

For logistic regression / XGBoost:

```python
p_under_model = model.predict_proba(X)[:, 1]
edge_under    = p_under_model - p_under_market
```

No Normal distribution assumption. The model directly outputs P(under).

## Min-edge threshold

Only bet when `edge > min_edge`. Standard sweep: {0.01, 0.02, 0.05, 0.08}.

- **0.01–0.02**: High volume, captures real edge but also noise
- **0.05+**: Low volume, only the model's most confident calls
- At 0.08+ with n < 100: results are noise, not signal

## Common mistakes

- Using de-vigged market prob in edge formula — edge will be understated (you're actually fighting raw vigged odds)
- Not applying an odds filter before computing ROI from edge bins — extreme long-shots inflate apparent edge
- Trusting ROI at min_edge thresholds with n < 200 bets

## Related

- [[american-odds]]
- [[roi-and-pnl]]
- [[model-evaluation]]
