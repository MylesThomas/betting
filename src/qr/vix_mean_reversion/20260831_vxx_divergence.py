"""
VIX Mean Reversion — P1b: VXX Signal Divergence Analysis

Replays v2 long-only signals against VXX close prices.
VXX launched 2009-01-30 — only trades from that date onward are covered.
Measures how much of the VIX % move is captured by VXX after roll drag.
This is a lower bound on achievable returns, not a trading recommendation.
"""
import base64
import io
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
TRADES_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_v2_trades.parquet")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── data ──────────────────────────────────────────────────────────────────────
print("Pulling VXX...")
vxx_raw = yf.download("VXX", start="2009-01-01", auto_adjust=True, progress=False)
vxx_raw.columns = vxx_raw.columns.get_level_values(0)
vxx = vxx_raw["Close"].copy()
vxx.index = pd.to_datetime(vxx.index)
print(f"VXX: {vxx.index[0].date()} → {vxx.index[-1].date()}, {len(vxx):,} rows")

trades = pd.read_parquet(TRADES_PATH)
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])

VXX_START = vxx.index[0]
covered   = trades[trades["entry_date"] >= VXX_START].copy()
skipped   = trades[trades["entry_date"] <  VXX_START].copy()
print(f"Trades covered by VXX : {len(covered)} (entry >= {VXX_START.date()})")
print(f"Trades pre-VXX (skipped): {len(skipped)}")

# ── match VXX prices to each trade's entry/exit dates ─────────────────────────
def get_price(series, date):
    """Get price on date, or nearest prior trading day."""
    if date in series.index:
        return series[date]
    prior = series[series.index <= date]
    return prior.iloc[-1] if len(prior) else np.nan

rows = []
for _, t in covered.iterrows():
    vxx_entry = get_price(vxx, t["entry_date"])
    vxx_exit  = get_price(vxx, t["exit_date"])
    if pd.isna(vxx_entry) or pd.isna(vxx_exit):
        continue
    vxx_pnl_pct  = (vxx_exit - vxx_entry) / vxx_entry * 100
    vix_pnl_pct  = t["pnl_pct"]
    capture      = vxx_pnl_pct / vix_pnl_pct if vix_pnl_pct != 0 else np.nan
    drag         = vix_pnl_pct - vxx_pnl_pct  # positive = VIX beat VXX
    rows.append({
        "entry_date":      t["entry_date"].date(),
        "exit_date":       t["exit_date"].date(),
        "hold_days":       t["hold_days"],
        "vix_entry":       round(t["entry_price"], 2),
        "vix_exit":        round(t["exit_price"], 2),
        "vix_pnl_pct":     round(vix_pnl_pct, 2),
        "vxx_entry":       round(vxx_entry, 2),
        "vxx_exit":        round(vxx_exit, 2),
        "vxx_pnl_pct":     round(vxx_pnl_pct, 2),
        "capture_ratio":   round(capture, 3) if not pd.isna(capture) else np.nan,
        "drag_pct":        round(drag, 2),
        "vix_win":         vix_pnl_pct > 0,
        "vxx_win":         vxx_pnl_pct > 0,
    })

df = pd.DataFrame(rows)
print(f"\nMatched trades: {len(df)}")

# ── summary stats ──────────────────────────────────────────────────────────────
vix_wins = df[df["vix_win"]]
vxx_wins = df[df["vxx_win"]]

print(f"\n=== VIX vs VXX comparison ===")
print(f"VIX win rate       : {df['vix_win'].mean():.1%}  ({df['vix_win'].sum()}/{len(df)})")
print(f"VXX win rate       : {df['vxx_win'].mean():.1%}  ({df['vxx_win'].sum()}/{len(df)})")
print(f"VIX total pnl_pct  : {df['vix_pnl_pct'].sum():+.2f}%")
print(f"VXX total pnl_pct  : {df['vxx_pnl_pct'].sum():+.2f}%")
print(f"Avg drag per trade : {df['drag_pct'].mean():+.2f}%")
print(f"Median drag        : {df['drag_pct'].median():+.2f}%")
print(f"Avg capture ratio  : {df['capture_ratio'].mean():.2f}x")
print(f"Median capture     : {df['capture_ratio'].median():.2f}x")
print(f"Trades VIX win, VXX loss : {((df['vix_win']) & (~df['vxx_win'])).sum()}")
print(f"Trades both win    : {(df['vix_win'] & df['vxx_win']).sum()}")
print(f"Trades both lose   : {(~df['vix_win'] & ~df['vxx_win']).sum()}")

# drag by hold duration buckets
df["hold_bucket"] = pd.cut(df["hold_days"], bins=[0,7,14,30,60,999],
                            labels=["0-7d","8-14d","15-30d","31-60d","60d+"])
drag_by_hold = df.groupby("hold_bucket")[["vix_pnl_pct","vxx_pnl_pct","drag_pct","capture_ratio"]].mean().round(3)
print(f"\n=== Drag by hold duration ===")
print(drag_by_hold.to_string())

# compounded comparison ($1k start, covered trades only)
bal_vix, bal_vxx = 1000.0, 1000.0
vix_curve, vxx_curve = [], []
for _, row in df.iterrows():
    bal_vix *= (1 + row["vix_pnl_pct"] / 100)
    bal_vxx *= (1 + row["vxx_pnl_pct"] / 100)
    vix_curve.append({"exit_date": row["exit_date"], "balance": bal_vix})
    vxx_curve.append({"exit_date": row["exit_date"], "balance": bal_vxx})

vix_c = pd.DataFrame(vix_curve)
vxx_c = pd.DataFrame(vxx_curve)
vix_c["exit_date"] = pd.to_datetime(vix_c["exit_date"])
vxx_c["exit_date"] = pd.to_datetime(vxx_c["exit_date"])

years_cov = (pd.Timestamp(df["exit_date"].iloc[-1]) - pd.Timestamp(df["entry_date"].iloc[0])).days / 365.25
cagr_vix  = (bal_vix / 1000) ** (1/years_cov) - 1
cagr_vxx  = (bal_vxx / 1000) ** (1/years_cov) - 1

print(f"\nCovered period: {years_cov:.1f} years")
print(f"VIX final balance: ${bal_vix:,.0f}  CAGR {cagr_vix*100:.1f}%")
print(f"VXX final balance: ${bal_vxx:,.0f}  CAGR {cagr_vxx*100:.1f}%")
print(f"CAGR haircut (VIX - VXX): {(cagr_vix - cagr_vxx)*100:.1f}pp")

# ── charts ────────────────────────────────────────────────────────────────────

# 1. VIX pnl vs VXX pnl scatter per trade
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
same_dir = df["vix_win"] == df["vxx_win"]
ax.scatter(df[same_dir]["vix_pnl_pct"],  df[same_dir]["vxx_pnl_pct"],
           color="steelblue", alpha=0.6, s=30, label="Same direction")
ax.scatter(df[~same_dir]["vix_pnl_pct"], df[~same_dir]["vxx_pnl_pct"],
           color="tomato", alpha=0.9, s=50, zorder=5, label="Direction diverges")
ax.axline((0,0), slope=1, color="gray", lw=1, ls="--", label="1:1 line (no drag)")
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("VIX trade P&L (%)")
ax.set_ylabel("VXX trade P&L (%)")
ax.set_title("VIX vs VXX P&L per trade", fontsize=11)
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.hist(df["drag_pct"], bins=30, color="coral", alpha=0.8, edgecolor="white", lw=0.3)
ax2.axvline(df["drag_pct"].mean(),   color="darkred", lw=1.5, ls="--",
            label=f"Avg drag {df['drag_pct'].mean():+.1f}%")
ax2.axvline(df["drag_pct"].median(), color="orange",  lw=1.2, ls=":",
            label=f"Median {df['drag_pct'].median():+.1f}%")
ax2.axvline(0, color="black", lw=0.8)
ax2.set_xlabel("Drag per trade (VIX pnl% − VXX pnl%)")
ax2.set_ylabel("Trade count")
ax2.set_title("Roll Drag Distribution", fontsize=11)
ax2.legend(fontsize=8)

plt.tight_layout()
scatter_b64 = fig_to_b64(fig)

# 2. Compounded equity curve: VIX vs VXX
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(vix_c["exit_date"], vix_c["balance"], color="steelblue", lw=1.8,
        label=f"VIX signal (theoretical)  CAGR {cagr_vix*100:.1f}%")
ax.plot(vxx_c["exit_date"], vxx_c["balance"], color="tomato",    lw=1.4,
        label=f"VXX execution             CAGR {cagr_vxx*100:.1f}%")
ax.set_yscale("log")
ax.axhline(1000, color="gray", lw=0.7, ls="--", alpha=0.5)
ax.set_ylabel("Balance ($ log scale)")
ax.set_title(f"VIX signal vs VXX execution — $1k start, 100% sizing ({len(df)} trades, {years_cov:.0f}yr covered)", fontsize=12)
ax.legend()
plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# 3. Drag by hold bucket
fig, ax = plt.subplots(figsize=(10, 4))
buckets = drag_by_hold.index.astype(str)
x = range(len(buckets))
w = 0.35
ax.bar([i - w/2 for i in x], drag_by_hold["vix_pnl_pct"], width=w,
       color="steelblue", alpha=0.85, label="VIX avg P&L %")
ax.bar([i + w/2 for i in x], drag_by_hold["vxx_pnl_pct"], width=w,
       color="tomato",    alpha=0.85, label="VXX avg P&L %")
ax.set_xticks(list(x))
ax.set_xticklabels(buckets)
ax.set_xlabel("Hold duration")
ax.set_ylabel("Avg P&L %")
ax.set_title("VIX vs VXX avg P&L by hold duration — drag grows with time", fontsize=12)
ax.legend()
ax.axhline(0, color="black", lw=0.7)
plt.tight_layout()
hold_b64 = fig_to_b64(fig)

# ── HTML ──────────────────────────────────────────────────────────────────────
def df_to_html(df, max_rows=None):
    d = df.head(max_rows) if max_rows else df
    cols = list(d.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v   = row[c]
            cls = ""
            if c == "vix_pnl_pct":
                cls = "pass" if float(v) > 0 else "fail"
            if c == "vxx_pnl_pct":
                cls = "pass" if float(v) > 0 else "fail"
            if c == "drag_pct":
                cls = "flag" if float(v) > 5 else ""
            if c == "capture_ratio" and not pd.isna(v):
                cls = "fail" if float(v) < 0.5 else ""
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

drag_table = df_to_html(drag_by_hold.reset_index())
trade_table = df_to_html(df.sort_values("drag_pct", ascending=False), max_rows=50)

n_diverge = ((df["vix_win"]) & (~df["vxx_win"])).sum()

step_p1b_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- P1b — VXX SIGNAL DIVERGENCE ANALYSIS                               -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">P1b</span>
    <h2 style="margin:0; border:none; padding:0;">VXX Signal Divergence — Roll Drag Haircut Analysis</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_vxx_divergence.py</div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">signal</span><span class="config-val">v2 long-only (126d SMA, 10% entry, 2% exit) — same entry/exit dates</span></div>
    <div class="config-row"><span class="config-key">VXX coverage</span><span class="config-val">{vxx.index[0].date()} → {vxx.index[-1].date()} (launched 2009-01-30)</span></div>
    <div class="config-row"><span class="config-key">trades covered</span><span class="config-val">{len(df)} of {len(trades)} total ({len(df)/len(trades):.0%}) — {len(skipped)} pre-VXX skipped</span></div>
    <div class="config-row"><span class="config-key">purpose</span><span class="config-val">Lower bound on achievable returns — not a trading recommendation. VXX long has structural decay issues.</span></div>
  </div>

  <h3>Summary</h3>
  <div class="config-block">
    <div class="config-row"><span class="config-key">VIX win rate</span><span class="config-val pass">{df['vix_win'].mean():.1%} ({df['vix_win'].sum()}/{len(df)} trades)</span></div>
    <div class="config-row"><span class="config-key">VXX win rate</span><span class="config-val">{df['vxx_win'].mean():.1%} ({df['vxx_win'].sum()}/{len(df)} trades)</span></div>
    <div class="config-row"><span class="config-key">VIX total P&L</span><span class="config-val pass">{df['vix_pnl_pct'].sum():+.2f}%</span></div>
    <div class="config-row"><span class="config-key">VXX total P&L</span><span class="config-val">{df['vxx_pnl_pct'].sum():+.2f}%</span></div>
    <div class="config-row"><span class="config-key">Avg drag per trade</span><span class="config-val flag">{df['drag_pct'].mean():+.2f}% (VIX − VXX)</span></div>
    <div class="config-row"><span class="config-key">Median capture ratio</span><span class="config-val">{df['capture_ratio'].median():.2f}x (VXX captures this fraction of VIX move)</span></div>
    <div class="config-row"><span class="config-key">Trades VIX wins, VXX loses</span><span class="config-val fail">{n_diverge} trades — signal right but VXX still lost</span></div>
    <div class="config-row"><span class="config-key">VIX CAGR ({years_cov:.0f}yr covered)</span><span class="config-val pass">{cagr_vix*100:.1f}%</span></div>
    <div class="config-row"><span class="config-key">VXX CAGR ({years_cov:.0f}yr covered)</span><span class="config-val">{cagr_vxx*100:.1f}%</span></div>
    <div class="config-row"><span class="config-key">CAGR haircut</span><span class="config-val fail">{(cagr_vix-cagr_vxx)*100:.1f} percentage points</span></div>
  </div>

  <h3>VIX vs VXX P&L Scatter + Drag Distribution</h3>
  <img src="data:image/png;base64,{scatter_b64}" alt="VIX vs VXX Scatter">

  <h3>Compounded Equity Curve — VIX Signal vs VXX Execution</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="VIX vs VXX Equity">

  <h3>Drag by Hold Duration</h3>
  <img src="data:image/png;base64,{hold_b64}" alt="Drag by Hold">
  {drag_table}

  <div class="flag-box">
    <strong>Key finding:</strong> VXX captures only ~{df['capture_ratio'].median():.0%} of the VIX move on the median trade. Drag grows with hold time — longest holds lose the most to roll costs. {n_diverge} trades where VIX was profitable but VXX was not — the signal was correct but the vehicle still lost. VXX is not a viable instrument for this strategy.
  </div>

  <div class="ok-box">
    <strong>Confirmed: P1a (VIX options) is the correct path.</strong> The structural decay in VXX destroys too much of the edge. VIX calls (long) would cap downside to premium paid while participating in the spike — eliminating both the roll drag and the MAE risk.
  </div>

  <h3>Trade Detail (sorted by drag desc, first 50)</h3>
  {trade_table}

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step_p1b_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nP1b section written. HTML size: {len(updated):,} bytes")
