"""
VIX Mean Reversion — Decision v2 (Step 5b)
Incorporates MAE analysis, expanded FV window sweep, and compounded returns.
"""
import pandas as pd
import numpy as np
from pathlib import Path

HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
TRADES_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_trades.parquet")

trades = pd.read_parquet(TRADES_PATH)
trades["exit_date"] = pd.to_datetime(trades["exit_date"])

# From Step 2b MAE analysis
mae_over_100 = 9   # short trades with MAE > 100%
mae_over_200 = 3   # short trades with MAE > 200%
total_shorts = 136
short_pnl_total = trades[trades["direction"]=="short"]["pnl_units"].sum()
long_pnl_total  = trades[trades["direction"]=="long"]["pnl_units"].sum()

# Blown-out short P&L at 100% threshold (from MAE script output)
# 9 trades blown out — pnl if removed
blown_100_entries = [
    pd.Timestamp("2020-02-21"), pd.Timestamp("2008-09-09"), pd.Timestamp("2018-01-18"),
    pd.Timestamp("2025-02-21"), pd.Timestamp("2024-07-18"), pd.Timestamp("2018-10-08"),
    pd.Timestamp("2007-06-06"), pd.Timestamp("2015-08-20"), pd.Timestamp("2011-07-27"),
]
blown_pnl = trades[(trades["direction"]=="short") &
                   (trades["entry_date"].isin(blown_100_entries))]["pnl_units"].sum()
short_pnl_survivable_100 = short_pnl_total - blown_pnl
combined_survivable_100  = long_pnl_total + short_pnl_survivable_100

# From Step 4b: best FV window is 126d
best_fv    = 126
best_fv_u  = 54.84

step5b_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 5b — DECISION v2                                               -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 5b</span>
    <h2 style="margin:0; border:none; padding:0;">Decision v2 — Updated with MAE, Compounding &amp; Expanded Sweep</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_decision_v2.py</div>

  <h3>What Changed Since v1 Decision</h3>
  <table>
    <thead><tr><th>Item</th><th>Finding</th><th>Impact on v2</th></tr></thead>
    <tbody>
      <tr>
        <td>MAE — short trades</td>
        <td>{mae_over_100}/{total_shorts} shorts ({mae_over_100/total_shorts:.0%}) had MAE &gt;100% — would be margin-called in a real account. Worst: 2020 (+384%), 2008 (+217%), 2018 (+205%).</td>
        <td class="fail">Short direction is not practically executable as-is. Requires either a hard stop-loss or long-only.</td>
      </tr>
      <tr>
        <td>Survivable short P&L (100% margin)</td>
        <td>Removing the {mae_over_100} blow-out shorts leaves short P&L at {short_pnl_survivable_100:+.2f}u (was {short_pnl_total:+.2f}u). Combined drops from {(long_pnl_total+short_pnl_total):+.2f}u to {combined_survivable_100:+.2f}u.</td>
        <td>Short still adds value even after margin blow-outs — but only if you can survive the path.</td>
      </tr>
      <tr>
        <td>Fair value window (expanded)</td>
        <td>126d SMA best ({best_fv_u:.1f}u). 200d SMA second (50.3u). 50d and 63d have lower win rates and calmar due to noisier signal.</td>
        <td class="pass">Upgrade from 252d → 126d SMA in v2.</td>
      </tr>
      <tr>
        <td>Compounded CAGR</td>
        <td>Long only: 94.2% CAGR, −31% max DD. Short only: 62.3% CAGR, −61% max DD. Combined: 214.6% CAGR, −55% max DD.</td>
        <td>Long-only drawdown is half of combined/short — cleaner risk profile.</td>
      </tr>
    </tbody>
  </table>

  <h3>Direction Decision</h3>
  <div class="flag-box">
    <strong>Short trades fail the margin survivability test.</strong> 9 of 136 short trades (7%) had open losses exceeding 100% before reverting. In 2020, a short entered at VIX=17 experienced a 384% adverse move before closing. No real account survives this without either a stop-loss (which changes the trade entirely) or enormous capital reserves. The backtest P&L for shorts is real but the path is not executable.
  </div>
  <div class="ok-box">
    <strong>Recommendation: Long only for v2.</strong> Long-only CAGR is 94.2% with −31% max compounded drawdown. Most long MAEs are small (VIX falling further below SMA is bounded by ~9 VIX floor). Short adds 62.3% CAGR but at −61% drawdown and margin-call risk in crisis regimes — not worth it until a stop-loss rule is designed and tested.
  </div>

  <h3>Recommended v2 Configuration</h3>
  <div class="config-block">
    <div class="config-row"><span class="config-key">fair_value_method</span><span class="config-val pass">126d SMA (upgraded from 252d — +9.3u total units gain)</span></div>
    <div class="config-row"><span class="config-key">entry_threshold</span><span class="config-val">10% (unchanged — better per-trade efficiency than 5%)</span></div>
    <div class="config-row"><span class="config-key">exit_threshold</span><span class="config-val">2% (take profit when within 2% of fair value — +3.7u gain, shorter avg hold)</span></div>
    <div class="config-row"><span class="config-key">direction</span><span class="config-val pass">Long only (removes margin-call risk; long dominates risk-adjusted)</span></div>
    <div class="config-row"><span class="config-key">sizing</span><span class="config-val">100% of bankroll per trade (for research; real sizing TBD when vehicle is chosen)</span></div>
  </div>

  <h3>v3 Research Agenda</h3>
  <ol style="padding-left:20px; line-height:2.2;">
    <li><strong>Short stop-loss rule:</strong> Test adding a hard stop at −30–50% adverse move on short trades. Does it improve risk-adjusted returns enough to justify keeping shorts?</li>
    <li><strong>Long-only v2 full backtest:</strong> Re-run Steps 2–3 with 126d SMA + 10% entry + 2% exit + long only. Confirm metrics improve as predicted by sweep data.</li>
    <li><strong>Regime filter:</strong> Test restricting entries to VIX absolute level in [12, 35]. Avoids deep sub-floor longs and extreme spike shorts.</li>
    <li><strong>Real vehicle analysis:</strong> VIX is not tradeable. Evaluate VIX futures (roll cost), VXX (path decay), or UVXY (2× leveraged). Each changes the P&L profile materially — this backtest is a research signal only.</li>
    <li><strong>Fixed % sizing:</strong> Test 10–25% of bankroll per trade instead of 100%. Reduces CAGR but makes the drawdown curve survivable at real account sizes.</li>
  </ol>

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step5b_html + "\n</body>")
HTML_PATH.write_text(updated)
print("Step 5b written to HTML.")
print(f"HTML size: {len(updated):,} bytes")
print(f"\n=== ALL FOLLOW-UP ITEMS COMPLETE ===")
print(f"  2b MAE analysis        : done")
print(f"  4b FV window sweep     : done (50d, 200d added)")
print(f"  3b Compounded returns  : done ($1k start, 100% sizing)")
print(f"  5b Decision v2         : done")
