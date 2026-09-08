"""
VIX Mean Reversion — Backtest Engine (Step 2)

Parameters (v1 defaults):
  fair_value_method : 252d SMA (shifted 1 day — no lookahead)
  entry_threshold   : 0.10 (10%)
  exit_threshold    : 0.00 (exit when |deviation| <= 0% — full centerline cross)
  direction         : both (long + short, one position at a time)
"""
import json
import base64
import io
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
FAIR_VALUE_WINDOW = 252
ENTRY_THRESHOLD   = 0.10
EXIT_THRESHOLD    = 0.00
DIRECTION         = "both"   # "long", "short", "both"
HTML_PATH = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
TRADES_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_trades.parquet")
TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── data ──────────────────────────────────────────────────────────────────────
print("Pulling VIX data...")
raw = yf.download("^VIX", start="1990-01-02", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
df = raw[["Close"]].copy()
df.columns = ["close"]
df.index = pd.to_datetime(df.index)

# Fair value: 252d SMA, shifted 1 day so day-T uses only days 1..T-1
df["fv"] = df["close"].rolling(FAIR_VALUE_WINDOW, min_periods=FAIR_VALUE_WINDOW).mean().shift(1)

# Deviation from fair value (%)
df["dev_pct"] = (df["close"] - df["fv"]) / df["fv"] * 100

# Drop warm-up period (no valid fv)
df_bt = df.dropna(subset=["fv"]).copy()
print(f"Backtestable window: {df_bt.index[0].date()} → {df_bt.index[-1].date()} ({len(df_bt):,} rows)")

# ── backtest loop ─────────────────────────────────────────────────────────────
print(f"Running backtest — direction={DIRECTION}, entry={ENTRY_THRESHOLD:.0%}, exit={EXIT_THRESHOLD:.0%} ...")

trades = []
position = None  # None | {"direction": str, "entry_date": ts, "entry_price": float, "fv_at_entry": float}

for date, row in df_bt.iterrows():
    close = row["close"]
    fv    = row["fv"]
    dev   = row["dev_pct"]

    # ── if in a position: check exit ─────────────────────────────────────────
    if position is not None:
        dir_ = position["direction"]
        # Exit when deviation crosses back to within exit_threshold of fair value
        # Long: entered below fv, exit when close >= fv * (1 + exit_threshold) — i.e. dev >= -exit_threshold*100
        # Short: entered above fv, exit when close <= fv * (1 - exit_threshold) — i.e. dev <=  exit_threshold*100
        should_exit = False
        if dir_ == "long"  and dev >= -(EXIT_THRESHOLD * 100):
            should_exit = True
        if dir_ == "short" and dev <=  (EXIT_THRESHOLD * 100):
            should_exit = True

        if should_exit:
            ep   = position["entry_price"]
            xp   = close
            fve  = position["fv_at_entry"]
            dev_e = (ep - fve) / fve * 100
            if dir_ == "long":
                pnl_pct = (xp - ep) / ep * 100
            else:
                pnl_pct = (ep - xp) / ep * 100
            pnl_units = pnl_pct / 100
            hold = (date - position["entry_date"]).days
            trades.append({
                "entry_date":            position["entry_date"],
                "exit_date":             date,
                "direction":             dir_,
                "entry_price":           ep,
                "exit_price":            xp,
                "fair_value_at_entry":   fve,
                "deviation_pct_at_entry": dev_e,
                "hold_days":             hold,
                "pnl_pct":               pnl_pct,
                "pnl_units":             pnl_units,
            })
            position = None

    # ── if flat: check entry ──────────────────────────────────────────────────
    if position is None:
        enter_long  = (DIRECTION in ("long",  "both")) and (dev <= -(ENTRY_THRESHOLD * 100))
        enter_short = (DIRECTION in ("short", "both")) and (dev >=  (ENTRY_THRESHOLD * 100))
        if enter_long:
            position = {"direction": "long",  "entry_date": date, "entry_price": close, "fv_at_entry": fv}
        elif enter_short:
            position = {"direction": "short", "entry_date": date, "entry_price": close, "fv_at_entry": fv}

trades_df = pd.DataFrame(trades)
trades_df["cumulative_units"] = trades_df["pnl_units"].cumsum()

print(f"Total closed trades: {len(trades_df):,}")
print(trades_df.groupby("direction")[["pnl_units"]].sum())

# ── save parquet ──────────────────────────────────────────────────────────────
trades_df.to_parquet(TRADES_PATH)
print(f"Trade log saved to {TRADES_PATH}")

# ── validation checks ─────────────────────────────────────────────────────────
print("\n--- Validation ---")
nulls_in_fv = trades_df["fair_value_at_entry"].isna().sum()
print(f"Null fv at entry         : {nulls_in_fv}")

warmup_end = df_bt.index[0]
early_trades = (trades_df["entry_date"] < warmup_end).sum()
print(f"Trades before warmup end : {early_trades}")

# Overlap check
trades_df_sorted = trades_df.sort_values("entry_date")
overlaps = 0
for i in range(1, len(trades_df_sorted)):
    if trades_df_sorted.iloc[i]["entry_date"] < trades_df_sorted.iloc[i-1]["exit_date"]:
        overlaps += 1
print(f"Overlapping trades       : {overlaps}")

long_violations  = ((trades_df["direction"] == "long")  & (trades_df["entry_price"] >= trades_df["fair_value_at_entry"])).sum()
short_violations = ((trades_df["direction"] == "short") & (trades_df["entry_price"] <= trades_df["fair_value_at_entry"])).sum()
print(f"Long entry above fv      : {long_violations}")
print(f"Short entry below fv     : {short_violations}")

neg_hold = (trades_df["hold_days"] <= 0).sum()
print(f"Zero/negative hold days  : {neg_hold}")

# ── plot: equity curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
for d, color, label in [("long", "steelblue", "Long"), ("short", "tomato", "Short")]:
    sub = trades_df[trades_df["direction"] == d].copy()
    if len(sub):
        sub = sub.sort_values("exit_date")
        ax.plot(sub["exit_date"], sub["pnl_units"].cumsum(), color=color, lw=1.2, label=f"{label} ({len(sub)} trades)")

all_sorted = trades_df.sort_values("exit_date")
ax.plot(all_sorted["exit_date"], all_sorted["cumulative_units"], color="black", lw=1.8, label="Combined", zorder=5)
ax.axhline(0, color="gray", lw=0.7, ls="--")
ax.set_title("Equity Curve — VIX Mean Reversion (v1: 252d SMA, ±10%, both directions)", fontsize=12)
ax.set_ylabel("Cumulative P&L (units)")
ax.legend()
plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# ── plot: trade log sample ─────────────────────────────────────────────────────
# First 50 rows for HTML table
sample = trades_df.head(50).copy()
for col in ["entry_price", "exit_price", "fair_value_at_entry", "deviation_pct_at_entry", "pnl_pct", "pnl_units", "cumulative_units"]:
    sample[col] = sample[col].round(4)

def df_to_html_table(df, max_rows=50):
    rows = []
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows.append(f"<thead><tr>{header}</tr></thead><tbody>")
    for _, row in df.iterrows():
        cells = []
        for c, v in zip(df.columns, row):
            if c == "direction":
                cls = "pass" if v == "long" else "fail"
                cells.append(f'<td class="{cls}">{v}</td>')
            elif c == "pnl_units":
                cls = "pass" if float(v) > 0 else "fail"
                cells.append(f'<td class="{cls}">{v}</td>')
            else:
                cells.append(f"<td>{v}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    rows.append("</tbody>")
    return f"<table>{''.join(rows)}</table>"

trade_table_html = df_to_html_table(sample)

# ── build validation table ────────────────────────────────────────────────────
def check_row(label, threshold, actual, passed):
    icon = "✅" if passed else "❌"
    cls  = "pass" if passed else "fail"
    return f'<tr><td>{label}</td><td>{threshold}</td><td class="{cls}">{actual}</td><td class="{cls}">{icon}</td></tr>'

validation_html = f"""
<table>
  <thead><tr><th>Check</th><th>Requirement</th><th>Actual</th><th>Result</th></tr></thead>
  <tbody>
    {check_row("Null fv at entry", "0", nulls_in_fv, nulls_in_fv == 0)}
    {check_row("Trades before warmup", "0", early_trades, early_trades == 0)}
    {check_row("Overlapping trades", "0", overlaps, overlaps == 0)}
    {check_row("Long entries above fv", "0", long_violations, long_violations == 0)}
    {check_row("Short entries below fv", "0", short_violations, short_violations == 0)}
    {check_row("Zero/negative hold days", "0", neg_hold, neg_hold == 0)}
  </tbody>
</table>
"""

# ── write Step 2 HTML section ─────────────────────────────────────────────────
n_total  = len(trades_df)
n_long   = (trades_df["direction"] == "long").sum()
n_short  = (trades_df["direction"] == "short").sum()
total_u  = trades_df["pnl_units"].sum()

step2_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 2 — BACKTEST ENGINE                                            -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 2</span>
    <h2 style="margin:0; border:none; padding:0;">Backtest Engine</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_backtest.py</div>

  <h3>Config</h3>
  <div class="config-block">
    <div class="config-row"><span class="config-key">fair_value_method</span><span class="config-val">252d SMA, shifted 1 day (no lookahead)</span></div>
    <div class="config-row"><span class="config-key">entry_threshold</span><span class="config-val">±10%</span></div>
    <div class="config-row"><span class="config-key">exit_threshold</span><span class="config-val">0% (full centerline cross)</span></div>
    <div class="config-row"><span class="config-key">direction</span><span class="config-val">both (one position at a time)</span></div>
    <div class="config-row"><span class="config-key">backtestable window</span><span class="config-val">{df_bt.index[0].date()} → {df_bt.index[-1].date()} ({len(df_bt):,} rows)</span></div>
  </div>

  <h3>Trade Summary</h3>
  <div class="config-block">
    <div class="config-row"><span class="config-key">total trades</span><span class="config-val">{n_total}</span></div>
    <div class="config-row"><span class="config-key">long trades</span><span class="config-val">{n_long}</span></div>
    <div class="config-row"><span class="config-key">short trades</span><span class="config-val">{n_short}</span></div>
    <div class="config-row"><span class="config-key">net P&amp;L (all trades)</span><span class="config-val">{total_u:+.4f} units</span></div>
  </div>

  <h3>Equity Curve</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Equity Curve">

  <h3>Trade Log (first {min(50, n_total)} of {n_total} trades)</h3>
  {trade_table_html}

  <h3>Validation Checks</h3>
  {validation_html}

</section>
"""

html = HTML_PATH.read_text()
updated = html.replace("</body>", step2_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 2 section written to HTML. Size: {len(updated):,} bytes")
