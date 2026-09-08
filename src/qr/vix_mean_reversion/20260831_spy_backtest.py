"""
Track A — SPY as vehicle for VIX mean-reversion signal.

Signal: same v2 config (126d SMA, 10% entry, 2% exit, long only).
Vehicle: go long SPY on entry, exit SPY on exit signal.
P&L: SPY % return over the hold window.

Key question: does the edge survive on SPY, and how does 2008/2020 look?
"""
import yfinance as yf
import pandas as pd
import numpy as np
import io, base64, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
V2_TRADES   = Path.home() / "Downloads/tmp/vix-mean-reversion_v2_trades.parquet"
OUT_TRADES  = Path.home() / "Downloads/tmp/vix-mean-reversion_spy_trades.parquet"

SMA_WINDOW      = 126
ENTRY_THRESHOLD = 0.90   # VIX <= SMA * 0.90
EXIT_THRESHOLD  = 0.98   # VIX >= SMA * 0.98

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── 1. Data ────────────────────────────────────────────────────────────────────
print("Pulling VIX and SPY...")
vix_raw = yf.download("^VIX", start="1990-01-02", auto_adjust=True, progress=False)
spy_raw = yf.download("SPY",  start="1990-01-02", auto_adjust=True, progress=False)

vix = vix_raw["Close"].squeeze().rename("vix")
spy = spy_raw["Close"].squeeze().rename("spy")

df = pd.DataFrame({"vix": vix, "spy": spy}).dropna()
df.index = pd.to_datetime(df.index)
df = df.sort_index()

print(f"  VIX: {vix.index[0].date()} – {vix.index[-1].date()}  ({len(vix)} rows)")
print(f"  SPY: {spy.index[0].date()} – {spy.index[-1].date()}  ({len(spy)} rows)")
print(f"  Overlap: {df.index[0].date()} – {df.index[-1].date()}  ({len(df)} rows)")

# Fair value — no lookahead (shift 1)
df["sma"] = df["vix"].shift(1).rolling(SMA_WINDOW).mean()
df = df.dropna(subset=["sma"])

# ── 2. Backtest engine ────────────────────────────────────────────────────────
print("Running SPY backtest...")
in_trade     = False
entry_date   = None
entry_vix    = None
entry_spy    = None
entry_fv     = None
trades       = []

for date, row in df.iterrows():
    vix_val = row["vix"]
    spy_val = row["spy"]
    sma_val = row["sma"]

    if not in_trade:
        if vix_val <= sma_val * ENTRY_THRESHOLD:
            in_trade   = True
            entry_date = date
            entry_vix  = vix_val
            entry_spy  = spy_val
            entry_fv   = sma_val
    else:
        if vix_val >= sma_val * EXIT_THRESHOLD:
            spy_pnl_pct = (spy_val - entry_spy) / entry_spy * 100
            vix_pnl_pct = (vix_val - entry_vix) / entry_vix * 100

            trades.append({
                "entry_date":        entry_date,
                "exit_date":         date,
                "entry_vix":         entry_vix,
                "exit_vix":          vix_val,
                "entry_spy":         entry_spy,
                "exit_spy":          spy_val,
                "fair_value":        entry_fv,
                "deviation_pct":     (entry_vix - entry_fv) / entry_fv * 100,
                "hold_days":         (date - entry_date).days,
                "spy_pnl_pct":       spy_pnl_pct,
                "spy_pnl_units":     spy_pnl_pct / 100,
                "vix_pnl_pct":       vix_pnl_pct,
                "vix_pnl_units":     vix_pnl_pct / 100,
            })
            in_trade = False

res = pd.DataFrame(trades)
res["spy_cum_units"] = res["spy_pnl_units"].cumsum()
res["vix_cum_units"] = res["vix_pnl_units"].cumsum()

# Compounded balance
balance = 1_000.0
balances = []
for pct in res["spy_pnl_pct"]:
    balance *= (1 + pct / 100)
    balances.append(balance)
res["balance"] = balances

res.to_parquet(OUT_TRADES)
print(f"  Trades: {len(res)}  ({res['entry_date'].dt.year.min()}–{res['exit_date'].dt.year.max()})")

# ── 3. Stats ───────────────────────────────────────────────────────────────────
wins   = res[res["spy_pnl_pct"] > 0]
losses = res[res["spy_pnl_pct"] <= 0]

n_trades  = len(res)
win_rate  = len(wins) / n_trades
total_u   = res["spy_pnl_units"].sum()
avg_win   = wins["spy_pnl_pct"].mean()
avg_loss  = losses["spy_pnl_pct"].mean()

peak   = res["spy_cum_units"].cummax()
max_dd = (res["spy_cum_units"] - peak).min()
calmar = total_u / abs(max_dd) if max_dd != 0 else np.inf

years_span   = res["exit_date"].dt.year.max() - res["entry_date"].dt.year.min() + 1
ann_return   = total_u / years_span * 100
final_bal    = res["balance"].iloc[-1]
cagr         = (final_bal / 1_000) ** (1 / years_span) - 1

print(f"\n=== Track A — SPY Summary ({n_trades} trades) ===")
print(f"  Win rate:          {win_rate:.1%}")
print(f"  Total units:       {total_u:+.2f}u")
print(f"  Avg win:           {avg_win:+.1f}%  (n={len(wins)})")
print(f"  Avg loss:          {avg_loss:+.1f}%  (n={len(losses)})")
print(f"  Max drawdown:      {max_dd:.2f}u")
print(f"  Calmar:            {calmar:.2f}")
print(f"  Annualized return: {ann_return:+.2f}u/yr")
print(f"  Final balance:     ${final_bal:,.0f}")
print(f"  CAGR:              {cagr:.1%}")

# Compare on same trade window vs VIX close
print(f"\n  VIX close on same trades:")
print(f"    Win rate: {(res['vix_pnl_pct'] > 0).mean():.1%}")
print(f"    Total u:  {res['vix_pnl_units'].sum():+.2f}u")
print(f"    Avg win:  {res.loc[res['vix_pnl_pct']>0,'vix_pnl_pct'].mean():+.1f}%")

# Year by year
res["year"] = res["entry_date"].dt.year
yearly = res.groupby("year").agg(
    n=("spy_pnl_units","count"),
    spy_win_rate=("spy_pnl_pct", lambda x: (x>0).mean()),
    spy_units=("spy_pnl_units","sum"),
    spy_avg_pct=("spy_pnl_pct","mean"),
    vix_units=("vix_pnl_units","sum"),
    avg_hold=("hold_days","mean"),
).reset_index()
eoy = res.groupby("year")["balance"].last()
yearly["eoy_balance"] = yearly["year"].map(eoy)
print(f"\nYear-by-year:")
print(yearly.to_string(index=False))

# Stress test: 2008, 2020, 2022
stress_years = [2008, 2020, 2022]
print(f"\nStress test years:")
for yr in stress_years:
    sub = res[res["year"] == yr]
    if len(sub) == 0:
        print(f"  {yr}: no trades")
        continue
    print(f"  {yr}: {len(sub)} trades | win rate {(sub['spy_pnl_pct']>0).mean():.0%} | "
          f"units {sub['spy_pnl_units'].sum():+.2f} | worst trade {sub['spy_pnl_pct'].min():+.1f}%")

# ── 4. Charts ──────────────────────────────────────────────────────────────────

# Chart 1: Equity curve — SPY units vs VIX units
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

ax = axes[0]
ax.plot(res["exit_date"], res["spy_cum_units"], color="#4caf83", lw=1.8, label="SPY vehicle")
ax.plot(res["exit_date"], res["vix_cum_units"], color="#5b9bd5", lw=1.4, ls="--", alpha=0.7, label="VIX close (baseline)")
ax.axhline(0, color="gray", lw=0.6)
# Shade drawdown on SPY
peak_vals = res["spy_cum_units"].cummax()
ax.fill_between(res["exit_date"], res["spy_cum_units"], peak_vals,
                where=res["spy_cum_units"] < peak_vals, alpha=0.25, color="#e05c5c", label="SPY drawdown")
ax.legend(fontsize=9)
ax.set_title("Track A — Equity Curve: SPY Vehicle vs VIX Close Baseline", fontsize=12)
ax.set_ylabel("Cumulative units")
ax.yaxis.grid(True, color="gray", alpha=0.2, lw=0.5)
ax.set_axisbelow(True)

# Chart 2: SPY pnl scatter vs VIX pnl
ax2 = axes[1]
colors = ["#4caf83" if v > 0 else "#e05c5c" for v in res["spy_pnl_pct"]]
ax2.scatter(res["vix_pnl_pct"], res["spy_pnl_pct"], c=colors, s=35, alpha=0.7)
ax2.axhline(0, color="black", lw=0.8)
ax2.axvline(0, color="black", lw=0.8)
# Annotate stress years
for yr in stress_years:
    sub = res[res["year"] == yr]
    for _, row in sub.iterrows():
        if row["spy_pnl_pct"] < -3:
            ax2.annotate(str(yr), (row["vix_pnl_pct"], row["spy_pnl_pct"]),
                        fontsize=7, color="#e05c5c", xytext=(3, 3), textcoords="offset points")
ax2.set_xlabel("VIX % change (entry → exit)")
ax2.set_ylabel("SPY % change (entry → exit)")
ax2.set_title("SPY Return vs VIX Return per Trade (red = SPY loss)", fontsize=11)
ax2.yaxis.grid(True, alpha=0.2)

plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# Chart 3: Year-by-year bar comparison (SPY units vs VIX units)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
colors = ["#4caf83" if v >= 0 else "#e05c5c" for v in yearly["spy_units"]]
bars = ax.bar(yearly["year"], yearly["spy_units"], color=colors, alpha=0.85, width=0.7)
for bar, val in zip(bars, yearly["spy_units"]):
    va = "bottom" if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val>=0 else -0.05),
            f"{val:+.1f}", ha="center", va=va, fontsize=7.5, fontweight="bold", color="#2d3148")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("SPY Vehicle — Annual Units", fontsize=11)
ax.set_ylabel("Units")
ax.set_xticks(yearly["year"])
ax.set_xticklabels(yearly["year"], rotation=45, ha="right", fontsize=8)
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)

ax = axes[1]
colors = ["#4caf83" if v >= 0 else "#e05c5c" for v in yearly["vix_units"]]
bars = ax.bar(yearly["year"], yearly["vix_units"], color=colors, alpha=0.85, width=0.7)
for bar, val in zip(bars, yearly["vix_units"]):
    va = "bottom" if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val>=0 else -0.05),
            f"{val:+.1f}", ha="center", va=va, fontsize=7.5, fontweight="bold", color="#2d3148")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("VIX Close Baseline — Annual Units (same trades)", fontsize=11)
ax.set_ylabel("Units")
ax.set_xticks(yearly["year"])
ax.set_xticklabels(yearly["year"], rotation=45, ha="right", fontsize=8)
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)

plt.tight_layout()
yearly_b64 = fig_to_b64(fig)

# ── 5. Passing bar verdict ────────────────────────────────────────────────────
years_profitable = (yearly["spy_units"] > 0).sum()
years_total      = len(yearly)
breadth_pass = years_profitable / years_total >= 2/3

checks = [
    ("n_trades ≥ 50",           n_trades >= 50,         f"{n_trades}"),
    ("win_rate ≥ 0.55",         win_rate >= 0.55,        f"{win_rate:.1%}"),
    ("total_units > 0",         total_u > 0,             f"{total_u:+.2f}u"),
    ("calmar > 1",              calmar > 1,              f"{calmar:.2f}"),
    ("yearly breadth ≥ 2/3",    breadth_pass,            f"{years_profitable}/{years_total} years"),
]
all_pass = all(p for _, p, _ in checks)
verdict  = "✅ PASSES" if all_pass else "❌ FAILS"

print(f"\nPassing bar verdict: {verdict}")
for label, passed, actual in checks:
    print(f"  {'✅' if passed else '❌'}  {label}: {actual}")

# ── 6. HTML section ───────────────────────────────────────────────────────────
verdict_rows = ""
for label, passed, actual in checks:
    color = "#4caf83" if passed else "#e05c5c"
    icon  = "✅" if passed else "❌"
    verdict_rows += f"<tr><td>{label}</td><td style='color:{color}'>{icon} {actual}</td></tr>"

yearly_rows = ""
for _, row in yearly.iterrows():
    spy_color = "#4caf83" if row["spy_units"] >= 0 else "#e05c5c"
    vix_color = "#4caf83" if row["vix_units"] >= 0 else "#e05c5c"
    yearly_rows += f"""<tr>
      <td>{int(row['year'])}</td>
      <td>{int(row['n'])}</td>
      <td>{row['spy_win_rate']:.0%}</td>
      <td style='color:{spy_color}'>{row['spy_units']:+.2f}</td>
      <td>{row['spy_avg_pct']:+.1f}%</td>
      <td style='color:{vix_color}'>{row['vix_units']:+.2f}</td>
      <td>{row['avg_hold']:.0f}d</td>
      <td>${row['eoy_balance']:,.0f}</td>
    </tr>"""

stress_rows = ""
for yr in [2008, 2020, 2022]:
    sub = res[res["year"] == yr]
    if len(sub) == 0:
        stress_rows += f"<tr><td>{yr}</td><td colspan='5'>No trades</td></tr>"
        continue
    wr = (sub["spy_pnl_pct"] > 0).mean()
    units = sub["spy_pnl_units"].sum()
    worst = sub["spy_pnl_pct"].min()
    avg_h = sub["hold_days"].mean()
    color = "#4caf83" if units >= 0 else "#e05c5c"
    stress_rows += f"""<tr>
      <td><strong>{yr}</strong></td>
      <td>{len(sub)}</td>
      <td>{wr:.0%}</td>
      <td style='color:{color}'>{units:+.2f}u</td>
      <td>{worst:+.1f}%</td>
      <td>{avg_h:.0f}d</td>
    </tr>"""

win_color  = "#4caf83" if win_rate >= 0.55 else "#e05c5c"
unit_color = "#4caf83" if total_u >= 0 else "#e05c5c"

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TRACK A — SPY Vehicle Backtest                                    -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">TRACK A</span>
    <h2 style="margin:0; border:none; padding:0;">SPY Vehicle Backtest — VIX Signal, SPY Execution</h2>
  </div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">Signal</span><span class="config-val">v2 (126d SMA, 10% entry, 2% exit, long only)</span></div>
    <div class="config-row"><span class="config-key">Vehicle</span><span class="config-val">SPY — long on entry signal, exit on exit signal</span></div>
    <div class="config-row"><span class="config-key">P&amp;L</span><span class="config-val">SPY % return over the hold window (entry close → exit close)</span></div>
    <div class="config-row"><span class="config-key">Data</span><span class="config-val">yfinance ^VIX + SPY (adjusted close)</span></div>
    <div class="config-row"><span class="config-key">Date range</span><span class="config-val">{res['entry_date'].dt.year.min()}–{res['exit_date'].dt.year.max()} ({n_trades} trades)</span></div>
    <div class="config-row"><span class="config-key">Sizing</span><span class="config-val">100% of $1k bankroll per trade (research baseline)</span></div>
  </div>

  <h3>Summary vs VIX Close Baseline</h3>
  <table>
    <thead><tr><th>Metric</th><th>SPY Vehicle (Track A)</th><th>VIX Close Baseline (v2)</th></tr></thead>
    <tbody>
      <tr><td>Trades</td><td>{n_trades}</td><td>229</td></tr>
      <tr><td>Win rate</td><td style="color:{win_color}">{win_rate:.1%}</td><td>96.5%</td></tr>
      <tr><td>Total units</td><td style="color:{unit_color}">{total_u:+.2f}u</td><td>+35.06u</td></tr>
      <tr><td>Avg win %</td><td>{avg_win:+.1f}%</td><td>+16.3%</td></tr>
      <tr><td>Avg loss %</td><td>{avg_loss:+.1f}%</td><td>−12.3%</td></tr>
      <tr><td>Max drawdown</td><td>{max_dd:.2f}u</td><td>−0.35u</td></tr>
      <tr><td>Calmar</td><td>{calmar:.2f}</td><td>99.14</td></tr>
      <tr><td>CAGR ({years_span}y)</td><td>{cagr:.1%}</td><td>139.7%</td></tr>
      <tr><td>Final balance ($1k start)</td><td>${final_bal:,.0f}</td><td>$16T+</td></tr>
    </tbody>
  </table>

  <h3>Equity Curve and Trade Scatter</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Track A equity">

  <h3>Annual Units Comparison</h3>
  <img src="data:image/png;base64,{yearly_b64}" alt="Track A yearly">

  <h3>Stress Test Years (2008, 2020, 2022)</h3>
  <table>
    <thead><tr><th>Year</th><th>N</th><th>Win rate</th><th>Units</th><th>Worst trade</th><th>Avg hold</th></tr></thead>
    <tbody>{stress_rows}</tbody>
  </table>

  <h3>Passing Bar Verdict: <span style="color:{'#4caf83' if all_pass else '#e05c5c'}">{verdict}</span></h3>
  <table>
    <thead><tr><th>Check</th><th>Result</th></tr></thead>
    <tbody>{verdict_rows}</tbody>
  </table>

  <h3>Year-by-Year Detail</h3>
  <table>
    <thead><tr><th>Year</th><th>N</th><th>SPY win%</th><th>SPY units</th><th>SPY avg%</th><th>VIX units</th><th>Avg hold</th><th>EOY balance</th></tr></thead>
    <tbody>{yearly_rows}</tbody>
  </table>
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nTrack A section written. HTML size: {len(html):,} bytes")
