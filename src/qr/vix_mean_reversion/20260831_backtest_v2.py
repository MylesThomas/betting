"""
VIX Mean Reversion — v2 Full Backtest (Step 6)

Config:
  fair_value : 126d SMA (shifted 1 — no lookahead)
  entry      : 10% below fair value
  exit       : 2% from fair value (take profit early)
  direction  : long only
  sizing     : 100% bankroll, $1,000 start (compounded)
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
TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)

FV_WINDOW   = 126
ENTRY_THR   = 0.10
EXIT_THR    = 0.02
DIRECTION   = "long"
START_BAL   = 1_000.0

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── data ──────────────────────────────────────────────────────────────────────
print("Pulling VIX...")
raw = yf.download("^VIX", start="1990-01-02", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
df = raw[["Close"]].copy()
df.columns = ["close"]
df.index = pd.to_datetime(df.index)
df["fv"]      = df["close"].rolling(FV_WINDOW, min_periods=FV_WINDOW).mean().shift(1)
df["dev_pct"] = (df["close"] - df["fv"]) / df["fv"] * 100
df_bt = df.dropna(subset=["fv"]).copy()
print(f"Backtestable window: {df_bt.index[0].date()} → {df_bt.index[-1].date()} ({len(df_bt):,} rows)")

# ── backtest ──────────────────────────────────────────────────────────────────
trades, position = [], None
for date, row in df_bt.iterrows():
    close, fv, dev = row["close"], row["fv"], row["dev_pct"]
    if position is not None:
        if dev >= -(EXIT_THR * 100):
            ep       = position["entry_price"]
            pnl_pct  = (close - ep) / ep * 100
            trades.append({
                "entry_date":             position["entry_date"],
                "exit_date":              date,
                "direction":              "long",
                "entry_price":            ep,
                "exit_price":             close,
                "fair_value_at_entry":    position["fv_at_entry"],
                "deviation_pct_at_entry": (ep - position["fv_at_entry"]) / position["fv_at_entry"] * 100,
                "hold_days":              (date - position["entry_date"]).days,
                "pnl_pct":                pnl_pct,
                "pnl_units":              pnl_pct / 100,
            })
            position = None
    if position is None and dev <= -(ENTRY_THR * 100):
        position = {"entry_date": date, "entry_price": close, "fv_at_entry": fv}

trades_df = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
trades_df["pnl_units"].cumsum()
trades_df["cumulative_units"] = trades_df["pnl_units"].cumsum()

# compounded balance
bal = START_BAL
balances = []
for pct in trades_df["pnl_pct"]:
    bal *= (1 + pct / 100)
    balances.append(bal)
trades_df["balance"] = balances

trades_df.to_parquet(TRADES_PATH)
print(f"Trades: {len(trades_df)}")

# ── validation ────────────────────────────────────────────────────────────────
print("\n--- Validation ---")
print(f"Null fv at entry   : {trades_df['fair_value_at_entry'].isna().sum()}")
print(f"Long above fv      : {(trades_df['entry_price'] >= trades_df['fair_value_at_entry']).sum()}")
print(f"Negative hold days : {(trades_df['hold_days'] <= 0).sum()}")
t_s = trades_df.sort_values("entry_date")
overlaps = sum(t_s.iloc[i]["entry_date"] < t_s.iloc[i-1]["exit_date"] for i in range(1, len(t_s)))
print(f"Overlapping trades : {overlaps}")

# ── performance summary ───────────────────────────────────────────────────────
wins   = trades_df[trades_df["pnl_units"] > 0]
losses = trades_df[trades_df["pnl_units"] <= 0]
total_u = trades_df["pnl_units"].sum()
cu      = trades_df["pnl_units"].cumsum()
mdd_u   = (cu - cu.cummax()).min()
calmar  = total_u / abs(mdd_u) if mdd_u != 0 else np.nan
years   = (trades_df["exit_date"].max() - trades_df["exit_date"].min()).days / 365.25
ann_ret = total_u / years * 100
pf      = wins["pnl_units"].sum() / abs(losses["pnl_units"].sum()) if len(losses) else np.nan
final_bal = trades_df["balance"].iloc[-1]
cagr      = (final_bal / START_BAL) ** (1 / years) - 1
peak_b    = trades_df["balance"].cummax()
mdd_pct   = ((trades_df["balance"] - peak_b) / peak_b * 100).min()

daily = trades_df.set_index("exit_date")["pnl_units"].resample("D").sum()
sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan

summary = {
    "n_trades":             len(trades_df),
    "win_rate":             f"{len(wins)/len(trades_df):.1%}",
    "avg_win_pct":          f"+{wins['pnl_pct'].mean():.2f}%",
    "avg_loss_pct":         f"{losses['pnl_pct'].mean():.2f}%" if len(losses) else "—",
    "avg_win_units":        f"+{wins['pnl_units'].mean():.4f}",
    "avg_loss_units":       f"{losses['pnl_units'].mean():.4f}" if len(losses) else "—",
    "profit_factor":        f"{pf:.2f}",
    "total_units":          f"{total_u:+.4f}",
    "avg_hold_days":        f"{trades_df['hold_days'].mean():.1f}",
    "max_drawdown_units":   f"{mdd_u:.4f}",
    "calmar":               f"{calmar:.2f}",
    "annualized_return_pct":f"{ann_ret:+.2f}%",
    "sharpe_approx":        f"{sharpe:.2f}",
    "starting_balance":     f"${START_BAL:,.0f}",
    "final_balance":        f"${final_bal:,.0f}",
    "cagr":                 f"{cagr*100:.1f}%",
    "max_drawdown_pct":     f"{mdd_pct:.1f}%",
}
for k, v in summary.items():
    print(f"  {k:30s}: {v}")

# ── year-by-year ──────────────────────────────────────────────────────────────
trades_df["year"] = trades_df["exit_date"].dt.year
yearly_rows = []
for yr in sorted(trades_df["year"].unique()):
    sub  = trades_df[trades_df["year"] == yr]
    wins_yr = sub[sub["pnl_units"] > 0]
    tu   = sub["pnl_units"].sum()
    cu_y = sub["pnl_units"].cumsum()
    mdd_y = (cu_y - cu_y.cummax()).min()
    eoy_bal = sub["balance"].iloc[-1]
    yearly_rows.append({
        "year":          yr,
        "n_trades":      len(sub),
        "win_rate":      f"{len(wins_yr)/len(sub):.0%}",
        "avg_win_pct":   f"+{wins_yr['pnl_pct'].mean():.1f}%" if len(wins_yr) else "—",
        "avg_loss_pct":  f"{sub[sub['pnl_units']<=0]['pnl_pct'].mean():.1f}%" if len(sub[sub['pnl_units']<=0]) else "—",
        "total_units":   tu,
        "roi_pct":       f"{tu*100:+.1f}%",
        "eoy_balance":   f"${eoy_bal:,.0f}",
        "profitable":    tu > 0,
    })

yearly_df = pd.DataFrame(yearly_rows)
n_prof = yearly_df["profitable"].sum()
n_tot  = len(yearly_df)
print(f"\nYearly breadth: {n_prof}/{n_tot}")

# ── charts ────────────────────────────────────────────────────────────────────

# 1. Dual panel: flat units + compounded balance
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.plot(trades_df["exit_date"], trades_df["cumulative_units"], color="steelblue", lw=1.5)
ax.fill_between(trades_df["exit_date"], trades_df["cumulative_units"], 0,
                where=trades_df["cumulative_units"] >= 0, alpha=0.15, color="steelblue")
ax.axhline(0, color="gray", lw=0.7, ls="--")
ax.set_ylabel("Cumulative P&L (units)")
ax.set_title("v2 — Long Only, 126d SMA, 10% entry, 2% exit", fontsize=12)

# mark MDD on units
peak_u = trades_df["cumulative_units"].cummax()
dd_u   = trades_df["cumulative_units"] - peak_u
trough = dd_u.idxmin()
peak_i = trades_df["cumulative_units"][:trough].idxmax()
ax.axvspan(trades_df.loc[peak_i, "exit_date"], trades_df.loc[trough, "exit_date"],
           color="tomato", alpha=0.15, label=f"Max DD ({mdd_u:.3f}u)")
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.plot(trades_df["exit_date"], trades_df["balance"], color="black", lw=1.5)
ax2.set_yscale("log")
ax2.set_ylabel("Balance ($ log scale)")
ax2.set_title(f"Compounded Balance — ${START_BAL:,.0f} start, 100% sizing | CAGR {cagr*100:.1f}%", fontsize=11)
ax2.axhline(START_BAL, color="gray", lw=0.7, ls="--", alpha=0.5)

plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# 2. Year-by-year bar (units)
fig, ax = plt.subplots(figsize=(14, 4))
colors = ["#4caf83" if p else "#e05c5c" for p in yearly_df["profitable"]]
ax.bar(yearly_df["year"], yearly_df["total_units"], color=colors, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("P&L (units)")
ax.set_title("v2 Annual P&L — green = profitable year", fontsize=12)
plt.tight_layout()
yearly_b64 = fig_to_b64(fig)

# 3. Win/loss pct distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(wins["pnl_pct"],   bins=30, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
axes[0].axvline(wins["pnl_pct"].mean(), color="navy", lw=1.5, ls="--",
                label=f"Avg +{wins['pnl_pct'].mean():.1f}%")
axes[0].set_title("Win trade % account change")
axes[0].set_xlabel("% account value gain")
axes[0].legend()

if len(losses):
    axes[1].hist(losses["pnl_pct"], bins=15, color="tomato", alpha=0.8, edgecolor="white", lw=0.3)
    axes[1].axvline(losses["pnl_pct"].mean(), color="darkred", lw=1.5, ls="--",
                    label=f"Avg {losses['pnl_pct'].mean():.1f}%")
    axes[1].set_title("Loss trade % account change")
    axes[1].set_xlabel("% account value loss")
    axes[1].legend()
plt.suptitle("v2 — % Account Value Change per Trade", fontsize=12)
plt.tight_layout()
dist_b64 = fig_to_b64(fig)

# ── HTML helpers ───────────────────────────────────────────────────────────────
def summary_table_html(d):
    rows = "".join(
        f"<tr><td>{k}</td><td class=\"{'pass' if '+' in str(v) or (isinstance(v,str) and not v.startswith('-')) else 'fail'}\">{v}</td></tr>"
        for k, v in d.items()
    )
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"

def yearly_table_html(df):
    cols = [c for c in df.columns if c != "profitable"]
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v   = row[c]
            cls = ""
            if c == "total_units":
                cls = "pass" if float(v) > 0 else "fail"
                v   = f"{float(v):+.4f}"
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

def val_row(label, req, actual, passed):
    icon = "✅" if passed else "❌"
    cls  = "pass" if passed else "fail"
    return f'<tr><td>{label}</td><td>{req}</td><td class="{cls}">{actual}</td><td class="{cls}">{icon}</td></tr>'

verdict = f"""
<table>
  <thead><tr><th>Check</th><th>Threshold</th><th>Actual</th><th>Pass/Fail</th></tr></thead>
  <tbody>
    {val_row("n_trades",       "≥ 50",   len(trades_df),                    len(trades_df) >= 50)}
    {val_row("total_units",    "> 0",    f"{total_u:+.4f}u",                total_u > 0)}
    {val_row("calmar",         "> 1",    f"{calmar:.2f}",                   calmar > 1)}
    {val_row("win_rate",       "≥ 45%",  f"{len(wins)/len(trades_df):.1%}", len(wins)/len(trades_df) >= 0.45)}
    {val_row("yearly breadth", "≥ 2/3",  f"{n_prof}/{n_tot} ({n_prof/n_tot:.1%})", n_prof/n_tot >= 2/3)}
  </tbody>
</table>"""

step6_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 6 — v2 BACKTEST (126d SMA · 10% entry · 2% exit · long only) -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 6</span>
    <h2 style="margin:0; border:none; padding:0;">v2 Backtest — 126d SMA · 10% entry · 2% exit · Long Only</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_backtest_v2.py</div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">fair_value_method</span><span class="config-val">126d SMA (shifted 1 day — no lookahead)</span></div>
    <div class="config-row"><span class="config-key">entry_threshold</span><span class="config-val">10% below fair value</span></div>
    <div class="config-row"><span class="config-key">exit_threshold</span><span class="config-val">2% from fair value (exit when within 2% — take profit early)</span></div>
    <div class="config-row"><span class="config-key">direction</span><span class="config-val">Long only</span></div>
    <div class="config-row"><span class="config-key">backtestable window</span><span class="config-val">{df_bt.index[0].date()} → {df_bt.index[-1].date()}</span></div>
  </div>

  <h3>Passing Bar</h3>
  {verdict}

  <h3>Performance Summary</h3>
  {summary_table_html(summary)}
  <p style="color:var(--muted);font-size:11px;">avg_win_pct / avg_loss_pct = mean % account value change per trade (= pnl_pct at 100% sizing). Sharpe is approximate.</p>

  <h3>Equity Curve (flat units) + Compounded Balance</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Equity Curve v2">

  <h3>% Account Value Change per Trade — Distribution</h3>
  <img src="data:image/png;base64,{dist_b64}" alt="Win/Loss Distribution">

  <h3>Year-by-Year</h3>
  <img src="data:image/png;base64,{yearly_b64}" alt="Annual P&L v2">
  {yearly_table_html(yearly_df)}
  <div class="finding">Yearly breadth: {n_prof}/{n_tot} years profitable ({n_prof/n_tot:.1%}).</div>

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step6_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 6 written. HTML size: {len(updated):,} bytes")
