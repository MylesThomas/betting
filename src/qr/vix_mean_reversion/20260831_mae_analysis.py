"""
VIX Mean Reversion — MAE Analysis for Short Trades (Step 2b)

For each short trade, compute the maximum adverse excursion (MAE):
how far VIX moved against the position before it closed.
Flag trades that would have been margin-called at 50/100/200% thresholds.
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
TRADES_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_trades.parquet")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── load trades and raw VIX ───────────────────────────────────────────────────
trades = pd.read_parquet(TRADES_PATH)
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])

print("Pulling VIX daily data...")
raw = yf.download("^VIX", start="1990-01-02", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
vix = raw["Close"].copy()
vix.index = pd.to_datetime(vix.index)

# ── compute MAE for each short trade ─────────────────────────────────────────
shorts = trades[trades["direction"] == "short"].copy()
print(f"Short trades: {len(shorts)}")

mae_rows = []
for _, row in shorts.iterrows():
    window = vix.loc[row["entry_date"]:row["exit_date"]]
    ep     = row["entry_price"]
    # For a short: adverse move is VIX going UP from entry
    max_close = window.max()
    mae_pct   = (max_close - ep) / ep * 100  # positive = adverse
    mae_rows.append({
        "entry_date":    row["entry_date"].date(),
        "exit_date":     row["exit_date"].date(),
        "entry_price":   ep,
        "exit_price":    row["exit_price"],
        "max_vix":       round(max_close, 2),
        "mae_pct":       round(mae_pct, 1),
        "hold_days":     row["hold_days"],
        "pnl_units":     round(row["pnl_units"], 4),
        "closed_profit": row["pnl_units"] > 0,
    })

mae_df = pd.DataFrame(mae_rows).sort_values("mae_pct", ascending=False)
print("\n=== MAE summary ===")
print(mae_df[["entry_date","mae_pct","pnl_units","hold_days"]].to_string())

# ── margin threshold analysis ─────────────────────────────────────────────────
thresholds = [0.50, 1.00, 2.00]
print("\n=== Survivable P&L by margin threshold ===")
threshold_rows = []
for thr in thresholds:
    surviving = mae_df[mae_df["mae_pct"] / 100 <= thr]
    blown_out  = mae_df[mae_df["mae_pct"] / 100 > thr]
    surv_pnl   = trades[trades["direction"] == "long"]["pnl_units"].sum() + \
                 trades[(trades["direction"] == "short") &
                        (trades["entry_date"].dt.date.isin(surviving["entry_date"]))]["pnl_units"].sum()
    threshold_rows.append({
        "margin_threshold":    f"{thr:.0%}",
        "short_trades_survive": len(surviving),
        "short_trades_blown":   len(blown_out),
        "blown_pnl_lost":       round(trades[(trades["direction"]=="short") &
                                     (trades["entry_date"].dt.date.isin(blown_out["entry_date"]))]["pnl_units"].sum(), 4),
        "combined_pnl_survivable": round(surv_pnl, 4),
    })
    print(f"  Margin {thr:.0%}: {len(surviving)} survive, {len(blown_out)} blown out")

thr_df = pd.DataFrame(threshold_rows)

# ── charts ────────────────────────────────────────────────────────────────────

# 1. MAE distribution histogram
fig, ax = plt.subplots(figsize=(12, 4))
colors = ["#4caf83" if p else "#e05c5c" for p in mae_df["closed_profit"]]
ax.bar(range(len(mae_df)), mae_df["mae_pct"].values, color=colors, alpha=0.85)
for thr_val, label, color in [(50, "50% margin", "#e0b54c"), (100, "100%", "#e05c5c"), (200, "200%", "#7b1111")]:
    ax.axhline(thr_val, color=color, lw=1.2, ls="--", label=label)
ax.set_xlabel("Short trade (sorted by MAE desc)")
ax.set_ylabel("Max Adverse Excursion (%)")
ax.set_title("Short Trade MAE — green = closed profitable, red = closed at loss", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
mae_bar_b64 = fig_to_b64(fig)

# 2. MAE vs closed P&L scatter
fig, ax = plt.subplots(figsize=(10, 4))
wins   = mae_df[mae_df["closed_profit"]]
losses = mae_df[~mae_df["closed_profit"]]
ax.scatter(wins["mae_pct"],   wins["pnl_units"],   color="steelblue", alpha=0.7, label="Win", s=40)
ax.scatter(losses["mae_pct"], losses["pnl_units"], color="tomato",    alpha=0.9, label="Loss", s=60, zorder=5)
for thr_val, color in [(50, "#e0b54c"), (100, "#e05c5c"), (200, "#7b1111")]:
    ax.axvline(thr_val, color=color, lw=1, ls="--")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("Max Adverse Excursion (%)")
ax.set_ylabel("Closed P&L (units)")
ax.set_title("Short Trades — MAE vs Closed P&L", fontsize=12)
ax.legend()
plt.tight_layout()
mae_scatter_b64 = fig_to_b64(fig)

# 3. Survivable combined P&L bar
fig, ax = plt.subplots(figsize=(8, 4))
long_pnl = trades[trades["direction"]=="long"]["pnl_units"].sum()
labels = ["No shorts"] + [f"Survive\n{r['margin_threshold']}" for _, r in thr_df.iterrows()] + ["All shorts\n(full backtest)"]
values = [long_pnl] + list(thr_df["combined_pnl_survivable"]) + [trades["pnl_units"].sum()]
bar_colors = ["steelblue"] + ["#e0b54c","#ff7f0e","#4caf83"] + ["black"]
ax.bar(labels, values, color=bar_colors, alpha=0.85)
ax.axhline(long_pnl, color="steelblue", lw=1, ls="--", alpha=0.5)
ax.set_ylabel("Combined P&L (units)")
ax.set_title("Survivable P&L by Margin Threshold", fontsize=12)
for i, v in enumerate(values):
    ax.text(i, v + 0.3, f"{v:.1f}u", ha="center", fontsize=9)
plt.tight_layout()
surv_b64 = fig_to_b64(fig)

# ── HTML helpers ───────────────────────────────────────────────────────────────
def df_to_html(df):
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cls = ""
            if c == "mae_pct":
                fv = float(v)
                cls = "fail" if fv > 100 else ("flag" if fv > 50 else "pass")
            if c == "pnl_units":
                cls = "pass" if float(v) > 0 else "fail"
            if c == "closed_profit":
                v = "✅" if v else "❌"
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

mae_table  = df_to_html(mae_df)
thr_table  = df_to_html(thr_df)

# ── key stats ──────────────────────────────────────────────────────────────────
n_over_50  = (mae_df["mae_pct"] > 50).sum()
n_over_100 = (mae_df["mae_pct"] > 100).sum()
n_over_200 = (mae_df["mae_pct"] > 200).sum()
max_mae    = mae_df["mae_pct"].max()
median_mae = mae_df["mae_pct"].median()

print(f"\nMedian MAE : {median_mae:.1f}%")
print(f"MAE > 50%  : {n_over_50}/{len(mae_df)} trades")
print(f"MAE > 100% : {n_over_100}/{len(mae_df)} trades")
print(f"MAE > 200% : {n_over_200}/{len(mae_df)} trades")
print(f"Max MAE    : {max_mae:.1f}%")

# ── Step 2b HTML section ──────────────────────────────────────────────────────
step2b_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 2b — MAE ANALYSIS (SHORT TRADES)                               -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 2b</span>
    <h2 style="margin:0; border:none; padding:0;">MAE Analysis — Short Trades</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_mae_analysis.py</div>

  <div class="finding">
    <strong>Question:</strong> The backtest P&L for shorts looks profitable on closed trades, but some shorts held through VIX spikes of 200%+. Would these trades survive margin constraints in a real account?
  </div>

  <h3>MAE Summary</h3>
  <div class="config-block">
    <div class="config-row"><span class="config-key">Short trades analysed</span><span class="config-val">{len(mae_df)}</span></div>
    <div class="config-row"><span class="config-key">Median MAE</span><span class="config-val">{median_mae:.1f}%</span></div>
    <div class="config-row"><span class="config-key">MAE &gt; 50% (margin warning)</span><span class="config-val">{n_over_50}/{len(mae_df)} trades ({n_over_50/len(mae_df):.1%})</span></div>
    <div class="config-row"><span class="config-key">MAE &gt; 100% (likely margin call)</span><span class="config-val">{n_over_100}/{len(mae_df)} trades ({n_over_100/len(mae_df):.1%})</span></div>
    <div class="config-row"><span class="config-key">MAE &gt; 200% (account wipeout)</span><span class="config-val">{n_over_200}/{len(mae_df)} trades ({n_over_200/len(mae_df):.1%})</span></div>
    <div class="config-row"><span class="config-key">Worst MAE</span><span class="config-val">{max_mae:.1f}%</span></div>
  </div>

  <h3>MAE by Trade (sorted worst first)</h3>
  <img src="data:image/png;base64,{mae_bar_b64}" alt="MAE Bar Chart">

  <h3>MAE vs Closed P&L Scatter</h3>
  <img src="data:image/png;base64,{mae_scatter_b64}" alt="MAE Scatter">

  <h3>Survivable P&L by Margin Threshold</h3>
  <img src="data:image/png;base64,{surv_b64}" alt="Survivable P&L">
  {thr_table}

  <h3>All Short Trades — MAE Detail</h3>
  {mae_table}

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step2b_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 2b written to HTML. Size: {len(updated):,} bytes")
