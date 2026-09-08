"""
VIX Mean Reversion — Performance Analysis (Step 3)
"""
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
TRADES_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_trades.parquet")

trades = pd.read_parquet(TRADES_PATH)
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
trades["year"]       = trades["exit_date"].dt.year

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── max drawdown ───────────────────────────────────────────────────────────────
def max_drawdown(series):
    cum = series.cumsum()
    peak = cum.cummax()
    dd   = (cum - peak)
    return dd.min()

def drawdown_dates(series):
    cum = series.sort_index().cumsum()
    peak = cum.cummax()
    dd   = cum - peak
    trough_idx = dd.idxmin()
    peak_idx   = cum[:trough_idx].idxmax()
    return peak_idx, trough_idx

# ── overall summary ────────────────────────────────────────────────────────────
summary_rows = []
years_in_sample = (trades["exit_date"].max() - trades["exit_date"].min()).days / 365.25

for grp_label, grp_df in [("long", trades[trades["direction"]=="long"]),
                            ("short", trades[trades["direction"]=="short"]),
                            ("both", trades)]:
    if len(grp_df) == 0:
        continue
    wins      = grp_df[grp_df["pnl_units"] > 0]
    losses    = grp_df[grp_df["pnl_units"] <= 0]
    total_u   = grp_df["pnl_units"].sum()
    mdd       = max_drawdown(grp_df.sort_values("exit_date")["pnl_units"])
    calmar    = total_u / abs(mdd) if mdd != 0 else np.nan
    win_pf    = wins["pnl_units"].sum()
    loss_pf   = losses["pnl_units"].sum()
    pf        = win_pf / abs(loss_pf) if loss_pf != 0 else np.nan
    ann_ret   = total_u / years_in_sample * 100

    # approximate daily pnl series for Sharpe
    daily = grp_df.set_index("exit_date")["pnl_units"].resample("D").sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan

    summary_rows.append({
        "direction":            grp_label,
        "n_trades":             len(grp_df),
        "win_rate":             f"{len(wins)/len(grp_df):.1%}",
        "avg_win_units":        f"{wins['pnl_units'].mean():.4f}" if len(wins) else "—",
        "avg_loss_units":       f"{losses['pnl_units'].mean():.4f}" if len(losses) else "—",
        "profit_factor":        f"{pf:.2f}",
        "total_units":          f"{total_u:+.4f}",
        "roi_pct":              f"{total_u*100:+.2f}%",
        "avg_hold_days":        f"{grp_df['hold_days'].mean():.1f}",
        "max_drawdown_units":   f"{mdd:.4f}",
        "calmar":               f"{calmar:.2f}",
        "annualized_return_pct":f"{ann_ret:+.2f}%",
        "sharpe_approx":        f"{sharpe:.2f}",
    })

summary_df = pd.DataFrame(summary_rows)
print("=== Overall Summary ===")
print(summary_df.to_string(index=False))

# ── year-by-year ───────────────────────────────────────────────────────────────
yearly_rows = []
for year in sorted(trades["year"].unique()):
    sub = trades[trades["year"] == year]
    wins = sub[sub["pnl_units"] > 0]
    tu   = sub["pnl_units"].sum()
    mdd  = max_drawdown(sub.sort_values("exit_date")["pnl_units"])
    yearly_rows.append({
        "year":                year,
        "n_trades":            len(sub),
        "n_long":              (sub["direction"]=="long").sum(),
        "n_short":             (sub["direction"]=="short").sum(),
        "win_rate":            f"{len(wins)/len(sub):.1%}",
        "total_units":         tu,
        "roi_pct":             f"{tu*100:+.2f}%",
        "max_drawdown_units":  f"{mdd:.4f}",
        "profitable":          tu > 0,
    })

yearly_df = pd.DataFrame(yearly_rows).sort_values("year")
n_profitable_years = yearly_df["profitable"].sum()
n_total_years      = len(yearly_df)
print(f"\n=== Year-by-Year: {n_profitable_years}/{n_total_years} profitable ===")
print(yearly_df.drop("profitable", axis=1).to_string(index=False))

# ── charts ────────────────────────────────────────────────────────────────────

# 1. Equity curve with MDD marked
fig, ax = plt.subplots(figsize=(14, 4))
colors = {"long": "steelblue", "short": "tomato"}
for d in ["long", "short"]:
    sub = trades[trades["direction"]==d].sort_values("exit_date")
    ax.plot(sub["exit_date"], sub["pnl_units"].cumsum(), color=colors[d], lw=1.0, alpha=0.7, label=f"{d.capitalize()} only")

all_sorted = trades.sort_values("exit_date")
cum_all    = all_sorted["pnl_units"].cumsum()
ax.plot(all_sorted["exit_date"], cum_all, color="black", lw=1.8, label="Combined", zorder=5)

# Mark MDD period for combined
pk, tr = drawdown_dates(all_sorted.set_index("exit_date")["pnl_units"])
ax.axvspan(pk, tr, color="red", alpha=0.12, label=f"Max DD period ({pk.date()}→{tr.date()})")
ax.axhline(0, color="gray", lw=0.7, ls="--")
ax.set_title("Equity Curve — VIX Mean Reversion (252d SMA, ±10%, combined)", fontsize=12)
ax.set_ylabel("Cumulative P&L (units)")
ax.legend(fontsize=9)
plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# 2. Deviation at entry — winners vs losers
fig, ax = plt.subplots(figsize=(12, 4))
wins_dev   = trades[trades["pnl_units"] > 0]["deviation_pct_at_entry"]
losses_dev = trades[trades["pnl_units"] <= 0]["deviation_pct_at_entry"]
bins = np.linspace(trades["deviation_pct_at_entry"].min(), trades["deviation_pct_at_entry"].max(), 40)
ax.hist(wins_dev,   bins=bins, color="steelblue", alpha=0.6, label=f"Winners (n={len(wins_dev)})")
ax.hist(losses_dev, bins=bins, color="tomato",    alpha=0.6, label=f"Losers  (n={len(losses_dev)})")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Deviation from fair value at entry (%)")
ax.set_ylabel("Trade count")
ax.set_title("Deviation at Entry — Winners vs Losers", fontsize=12)
ax.legend()
plt.tight_layout()
dev_dist_b64 = fig_to_b64(fig)

# 3. Hold time distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, d, color in [(axes[0], "long", "steelblue"), (axes[1], "short", "tomato")]:
    sub = trades[trades["direction"]==d]
    w   = sub[sub["pnl_units"] > 0]["hold_days"]
    l   = sub[sub["pnl_units"] <= 0]["hold_days"]
    ax.hist(w, bins=20, color=color,  alpha=0.6, label=f"Win  (n={len(w)}, med={w.median():.0f}d)")
    ax.hist(l, bins=20, color="gray", alpha=0.6, label=f"Loss (n={len(l)}, med={l.median():.0f}d)")
    ax.axvline(60, color="red", lw=1, ls="--", label="60d threshold")
    ax.set_title(f"Hold Days — {d.capitalize()}", fontsize=11)
    ax.set_xlabel("Hold days")
    ax.legend(fontsize=8)
plt.tight_layout()
hold_b64 = fig_to_b64(fig)

# 4. Year-by-year bar chart
fig, ax = plt.subplots(figsize=(14, 4))
colors_bar = ["#4caf83" if p else "#e05c5c" for p in yearly_df["profitable"]]
ax.bar(yearly_df["year"], yearly_df["total_units"], color=colors_bar, edgecolor="none", alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Annual P&L (units) — green = profitable year", fontsize=12)
ax.set_ylabel("Units")
ax.set_xlabel("Year")
plt.tight_layout()
yearly_bar_b64 = fig_to_b64(fig)

# ── HTML helpers ───────────────────────────────────────────────────────────────
def df_to_html(df, highlight_col=None, neg_red=False):
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cls = ""
            if c == highlight_col:
                try:
                    fv = float(str(v).replace("%","").replace("+",""))
                    cls = "pass" if fv > 0 else "fail"
                except: pass
            if c == "direction":
                cls = "pass" if str(v) == "long" else ("fail" if str(v) == "short" else "")
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

summary_table  = df_to_html(summary_df, highlight_col="total_units")
yearly_display = yearly_df.drop("profitable", axis=1).copy()
yearly_display["total_units"] = yearly_display["total_units"].apply(lambda x: f"{x:+.4f}")
yearly_table   = df_to_html(yearly_display, highlight_col="total_units")

# ── Step 3 HTML section ────────────────────────────────────────────────────────
step3_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 3 — PERFORMANCE ANALYSIS                                        -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 3</span>
    <h2 style="margin:0; border:none; padding:0;">Performance Analysis</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_performance.py</div>

  <h3>Overall Summary (by direction)</h3>
  {summary_table}
  <p style="color:var(--muted);font-size:11px;">* Sharpe is approximate: mean(daily_pnl) / std(daily_pnl) × √252. Not a rigorous Sharpe ratio.</p>

  <h3>Equity Curve (with Max Drawdown period)</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Equity Curve">

  <h3>Year-by-Year P&L</h3>
  <img src="data:image/png;base64,{yearly_bar_b64}" alt="Annual P&L">
  {yearly_table}
  <div class="finding">
    <strong>Yearly breadth:</strong> {n_profitable_years}/{n_total_years} years profitable ({n_profitable_years/n_total_years:.1%}). Threshold for passing bar is ≥ 2/3 ({2/3:.1%}).
  </div>

  <h3>Deviation at Entry — Winners vs Losers</h3>
  <img src="data:image/png;base64,{dev_dist_b64}" alt="Deviation Distribution">

  <h3>Hold Time Distribution</h3>
  <img src="data:image/png;base64,{hold_b64}" alt="Hold Time">

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step3_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 3 section written to HTML. Size: {len(updated):,} bytes")
