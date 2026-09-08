"""
P1a — VIX Call Options Backtest
Using OptionsDX EOD data (2010–2023) against v2 long entry signals.

Strategy:
  - On each v2 long entry date: buy ATM VIX calls with DTE >= 40
  - Strike: closest to VIX close (UNDERLYING_LAST) at entry
  - Exit: on v2 exit date, value same contract at mid price
    If contract expired before exit, record as full premium loss
  - P&L: (exit_mid - entry_mid) / entry_mid * 100 (% of premium paid)

Assumptions:
  - Execute at mid price (bid+ask)/2 — no slippage model
  - 100% of $1k bankroll into premium each trade (same sizing as VIX close backtest)
  - Trades outside 2010–2023 window: excluded with note
"""
import pandas as pd
import numpy as np
import glob, io, base64, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RAW_DIR    = Path.home() / "Downloads/tmp/vix_options_raw"
TRADES_PATH = Path.home() / "Downloads/tmp/vix-mean-reversion_v2_trades.parquet"
HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
OUT_TRADES  = Path.home() / "Downloads/tmp/vix-mean-reversion_p1a_trades.parquet"

# ── 1. Load options data ───────────────────────────────────────────────────────
print("Loading options data...")
cols_needed = [
    "[QUOTE_DATE]", "[UNDERLYING_LAST]", "[EXPIRE_DATE]", "[DTE]",
    "[C_BID]", "[C_ASK]", "[STRIKE]", "[C_DELTA]", "[C_IV]", "[C_THETA]"
]

frames = []
for f in sorted(glob.glob(str(RAW_DIR / "*.txt"))):
    try:
        df = pd.read_csv(f, usecols=lambda c: c.strip() in [x.strip("[]") for x in [
            "QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE",
            "C_BID", "C_ASK", "STRIKE", "C_DELTA", "C_IV", "C_THETA"
        ]] or c.strip(" []") in [
            "QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE",
            "C_BID", "C_ASK", "STRIKE", "C_DELTA", "C_IV", "C_THETA"
        ], low_memory=False)
        frames.append(df)
    except Exception as e:
        print(f"  skip {f}: {e}")

# Columns have bracket notation — normalize
all_opts = pd.concat(frames, ignore_index=True)
all_opts.columns = [c.strip(" []") for c in all_opts.columns]
print(f"  Loaded {len(all_opts):,} rows | columns: {list(all_opts.columns)}")

# Parse dates
all_opts["QUOTE_DATE"]  = pd.to_datetime(all_opts["QUOTE_DATE"].str.strip())
all_opts["EXPIRE_DATE"] = pd.to_datetime(all_opts["EXPIRE_DATE"].str.strip())

# Numeric
for col in ["UNDERLYING_LAST", "DTE", "C_BID", "C_ASK", "STRIKE", "C_DELTA", "C_IV", "C_THETA"]:
    all_opts[col] = pd.to_numeric(all_opts[col], errors="coerce")

all_opts["c_mid"] = (all_opts["C_BID"] + all_opts["C_ASK"]) / 2

print(f"  Date range: {all_opts['QUOTE_DATE'].min().date()} – {all_opts['QUOTE_DATE'].max().date()}")
print(f"  Unique quote dates: {all_opts['QUOTE_DATE'].nunique()}")

# ── 2. Load v2 long trades ─────────────────────────────────────────────────────
v2 = pd.read_parquet(TRADES_PATH)
v2 = v2[v2["direction"] == "long"].copy()
v2["entry_date"] = pd.to_datetime(v2["entry_date"])
v2["exit_date"]  = pd.to_datetime(v2["exit_date"])
print(f"\nv2 long trades: {len(v2)}")

# Index options by quote date for fast lookup
opts_by_date = {d: grp for d, grp in all_opts.groupby("QUOTE_DATE")}

# ── 3. Run backtest ────────────────────────────────────────────────────────────
print("\nRunning P1a backtest...")
results = []
skipped = {"no_options_date": 0, "no_valid_contract": 0, "no_exit_date": 0}

MIN_DTE = 40  # minimum DTE at entry

for _, trade in v2.iterrows():
    entry_dt = trade["entry_date"]
    exit_dt  = trade["exit_date"]
    vix_at_entry = trade["entry_price"]

    # Skip if outside options data window
    if entry_dt not in opts_by_date:
        skipped["no_options_date"] += 1
        continue

    day_opts = opts_by_date[entry_dt]

    # Filter: DTE >= MIN_DTE, valid mid price
    candidates = day_opts[
        (day_opts["DTE"] >= MIN_DTE) &
        (day_opts["c_mid"] > 0) &
        (day_opts["C_BID"] > 0)
    ].copy()

    if candidates.empty:
        skipped["no_valid_contract"] += 1
        continue

    # Pick nearest expiry that has DTE >= MIN_DTE
    min_expiry = candidates["EXPIRE_DATE"].min()
    candidates = candidates[candidates["EXPIRE_DATE"] == min_expiry]

    # ATM strike: closest to VIX close
    candidates["strike_dist"] = (candidates["STRIKE"] - vix_at_entry).abs()
    best = candidates.loc[candidates["strike_dist"].idxmin()]

    entry_mid    = best["c_mid"]
    entry_strike = best["STRIKE"]
    entry_expiry = best["EXPIRE_DATE"]
    entry_dte    = best["DTE"]
    entry_delta  = best["C_DELTA"]
    entry_iv     = best["C_IV"]

    # Find exit value — same contract on exit date
    outcome = "expired_worthless"
    exit_mid = 0.0

    if exit_dt in opts_by_date:
        exit_day = opts_by_date[exit_dt]
        same_contract = exit_day[
            (exit_day["EXPIRE_DATE"] == entry_expiry) &
            (exit_day["STRIKE"] == entry_strike)
        ]
        if not same_contract.empty:
            row = same_contract.iloc[0]
            exit_mid = (row["C_BID"] + row["C_ASK"]) / 2
            if pd.notna(exit_mid) and exit_mid > 0:
                outcome = "closed"
            else:
                # Use C_LAST fallback
                exit_mid = 0.0
                outcome = "expired_worthless"
        else:
            # Contract expired or not found → check if expiry < exit
            if entry_expiry < exit_dt:
                outcome = "expired_worthless"
                exit_mid = 0.0
            else:
                skipped["no_exit_date"] += 1
                continue
    elif exit_dt > entry_expiry:
        outcome = "expired_worthless"
        exit_mid = 0.0
    else:
        skipped["no_exit_date"] += 1
        continue

    pnl_pct   = (exit_mid - entry_mid) / entry_mid * 100
    pnl_units = pnl_pct / 100

    results.append({
        "entry_date":     entry_dt,
        "exit_date":      exit_dt,
        "vix_entry":      vix_at_entry,
        "vix_exit":       trade["exit_price"],
        "vix_pnl_pct":    trade["pnl_pct"],
        "strike":         entry_strike,
        "expiry":         entry_expiry,
        "entry_dte":      entry_dte,
        "entry_delta":    entry_delta,
        "entry_iv":       entry_iv,
        "entry_mid":      entry_mid,
        "exit_mid":       exit_mid,
        "hold_days":      (exit_dt - entry_dt).days,
        "outcome":        outcome,
        "pnl_pct":        pnl_pct,
        "pnl_units":      pnl_units,
    })

res = pd.DataFrame(results)
res["cumulative_units"] = res["pnl_units"].cumsum()

# Compounded balance ($1k, 100% of bankroll per trade into premium)
balance = 1_000.0
balances = []
for _, row in res.iterrows():
    balance = balance * (1 + row["pnl_pct"] / 100)
    balances.append(balance)
res["balance"] = balances

res.to_parquet(OUT_TRADES)
print(f"\nTrades processed: {len(res)}")
print(f"Skipped — no options data for date: {skipped['no_options_date']}")
print(f"Skipped — no valid contract:        {skipped['no_valid_contract']}")
print(f"Skipped — no exit date data:        {skipped['no_exit_date']}")
print(f"Outcomes: {res['outcome'].value_counts().to_dict()}")

# ── 4. Summary stats ───────────────────────────────────────────────────────────
wins = res[res["pnl_pct"] > 0]
losses = res[res["pnl_pct"] <= 0]

n_trades  = len(res)
win_rate  = len(wins) / n_trades
total_u   = res["pnl_units"].sum()
avg_win   = wins["pnl_pct"].mean()
avg_loss  = losses["pnl_pct"].mean()

# Drawdown
peak = res["cumulative_units"].cummax()
dd   = (res["cumulative_units"] - peak)
max_dd = dd.min()
calmar = total_u / abs(max_dd) if max_dd != 0 else np.inf

entry_yr = res["entry_date"].dt.year
years_span = entry_yr.max() - entry_yr.min() + 1
ann_return = total_u / years_span * 100

final_balance = res["balance"].iloc[-1]
cagr = (final_balance / 1_000) ** (1 / years_span) - 1

print(f"\n=== P1a Summary ({len(res)} trades, {entry_yr.min()}–{entry_yr.max()}) ===")
print(f"  Win rate:          {win_rate:.1%}")
print(f"  Total units:       {total_u:+.2f}u")
print(f"  Avg win:           {avg_win:+.1f}%  (n={len(wins)})")
print(f"  Avg loss:          {avg_loss:+.1f}%  (n={len(losses)})")
print(f"  Max drawdown:      {max_dd:.2f}u")
print(f"  Calmar:            {calmar:.2f}")
print(f"  Annualized return: {ann_return:+.1f}u/yr")
print(f"  Final balance:     ${final_balance:,.0f}")
print(f"  CAGR:              {cagr:.1%}")

# Compare vs VIX close for same trades
print(f"\n  VIX close P&L on same {n_trades} trades:")
print(f"    Avg VIX pnl_pct: {res['vix_pnl_pct'].mean():+.1f}%")
print(f"    VIX win rate:    {(res['vix_pnl_pct'] > 0).mean():.1%}")

# Year by year
res["year"] = res["entry_date"].dt.year
yearly = res.groupby("year").agg(
    n_trades=("pnl_units", "count"),
    win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    total_units=("pnl_units", "sum"),
    avg_pnl_pct=("pnl_pct", "mean"),
).reset_index()
print(f"\nYear-by-year:")
print(yearly.to_string(index=False))

# ── 5. Charts ──────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# Chart 1: P&L scatter — VIX pnl vs options pnl
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4caf83" if v > 0 else "#e05c5c" for v in res["pnl_pct"]]
ax.scatter(res["vix_pnl_pct"], res["pnl_pct"], c=colors, alpha=0.65, s=35)
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("VIX close P&L %", fontsize=11)
ax.set_ylabel("Options P&L % (of premium)", fontsize=11)
ax.set_title("P1a — VIX Call Return vs VIX Close Return (per trade)", fontsize=12)
ax.yaxis.grid(True, color="gray", alpha=0.25, lw=0.5)
plt.tight_layout()
scatter_b64 = fig_to_b64(fig)

# Chart 2: Equity curves comparison
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(res["entry_date"], res["cumulative_units"], color="#4caf83", lw=1.8, label="VIX calls (P1a)")
# VIX close cumulative on same trades
res["vix_cumulative"] = (res["vix_pnl_pct"] / 100).cumsum()
ax.plot(res["entry_date"], res["vix_cumulative"], color="#5b9bd5", lw=1.4, ls="--", label="VIX close (same trades)")
ax.axhline(0, color="gray", lw=0.6)
ax.legend()
ax.set_title("P1a — Equity Curve: VIX Calls vs VIX Close (2010–2023 overlap)", fontsize=12)
ax.set_ylabel("Cumulative units")
ax.yaxis.grid(True, color="gray", alpha=0.2, lw=0.5)
plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# Chart 3: Win rate by hold bucket
res["hold_bucket"] = pd.cut(res["hold_days"], bins=[0, 14, 30, 60, 999],
                             labels=["0-14d", "15-30d", "31-60d", "60d+"])
bucket = res.groupby("hold_bucket", observed=True).agg(
    n=("pnl_pct", "count"),
    win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    avg_pnl_pct=("pnl_pct", "mean"),
    expired_pct=("outcome", lambda x: (x == "expired_worthless").mean()),
).reset_index()
print(f"\nBy hold bucket:\n{bucket.to_string(index=False)}")

# ── 6. Build HTML section ──────────────────────────────────────────────────────
yearly_rows = ""
for _, row in yearly.iterrows():
    wr_color = "#4caf83" if row["win_rate"] >= 0.5 else "#e05c5c"
    u_color  = "#4caf83" if row["total_units"] >= 0 else "#e05c5c"
    yearly_rows += f"""<tr>
      <td>{int(row['year'])}</td>
      <td>{int(row['n_trades'])}</td>
      <td style="color:{wr_color}">{row['win_rate']:.0%}</td>
      <td style="color:{u_color}">{row['total_units']:+.2f}</td>
      <td>{row['avg_pnl_pct']:+.1f}%</td>
    </tr>"""

bucket_rows = ""
for _, row in bucket.iterrows():
    exp_color = "#e05c5c" if row["expired_pct"] > 0.1 else "#4caf83"
    bucket_rows += f"""<tr>
      <td>{row['hold_bucket']}</td>
      <td>{int(row['n'])}</td>
      <td>{row['win_rate']:.0%}</td>
      <td>{row['avg_pnl_pct']:+.1f}%</td>
      <td style="color:{exp_color}">{row['expired_pct']:.0%}</td>
    </tr>"""

outcome_counts = res["outcome"].value_counts().to_dict()
closed_n       = outcome_counts.get("closed", 0)
expired_n      = outcome_counts.get("expired_worthless", 0)

win_color  = "#4caf83" if win_rate >= 0.5 else "#e05c5c"
unit_color = "#4caf83" if total_u >= 0 else "#e05c5c"

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP P1a — VIX Calls Backtest (OptionsDX 2010–2023)              -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">P1a</span>
    <h2 style="margin:0; border:none; padding:0;">VIX Call Options Backtest — OptionsDX EOD Data (2010–2023)</h2>
  </div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">Data source</span><span class="config-val">OptionsDX free EOD VIX option chains (2010–2023)</span></div>
    <div class="config-row"><span class="config-key">Signals</span><span class="config-val">v2 long entry/exit dates (126d SMA, 10% entry, 2% exit)</span></div>
    <div class="config-row"><span class="config-key">Strike</span><span class="config-val">ATM — closest to VIX close at entry</span></div>
    <div class="config-row"><span class="config-key">Min DTE at entry</span><span class="config-val">40 days (buffer above 27d avg hold)</span></div>
    <div class="config-row"><span class="config-key">Expiry selection</span><span class="config-val">Nearest expiry with DTE ≥ 40</span></div>
    <div class="config-row"><span class="config-key">Execution price</span><span class="config-val">Mid (bid+ask)/2 — no slippage model</span></div>
    <div class="config-row"><span class="config-key">Sizing</span><span class="config-val">100% of $1k bankroll into premium each trade</span></div>
    <div class="config-row"><span class="config-key">If contract expired before exit</span><span class="config-val">Record as full premium loss (worst case)</span></div>
    <div class="config-row"><span class="config-key">Trades excluded</span><span class="config-val">v2 signals where entry_date outside 2010–2023 data window</span></div>
  </div>

  <h3>Coverage</h3>
  <div class="finding">
    v2 produced <strong>229 long trades (1990–2026)</strong>. OptionsDX data covers <strong>{n_trades} trades ({entry_yr.min()}–{entry_yr.max()})</strong>.
    Excluded: {skipped['no_options_date']} (no data for date), {skipped['no_valid_contract']} (no valid contract), {skipped['no_exit_date']} (exit date not found).
  </div>
  <div class="finding">
    Outcomes: <strong>{closed_n} closed at exit signal</strong> ({closed_n/n_trades:.0%}),
    <strong>{expired_n} expired worthless before exit</strong> ({expired_n/n_trades:.0%}).
  </div>

  <h3>P1a Summary vs VIX Close (same {n_trades} trades)</h3>
  <table>
    <thead><tr><th>Metric</th><th>VIX Calls (P1a)</th><th>VIX Close (baseline)</th></tr></thead>
    <tbody>
      <tr><td>Trades</td><td>{n_trades}</td><td>{n_trades}</td></tr>
      <tr><td>Win rate</td><td style="color:{win_color}">{win_rate:.1%}</td><td>{(res['vix_pnl_pct']>0).mean():.1%}</td></tr>
      <tr><td>Total units</td><td style="color:{unit_color}">{total_u:+.2f}</td><td>{res['vix_pnl_pct'].sum()/100:+.2f}</td></tr>
      <tr><td>Avg win %</td><td>{avg_win:+.1f}%</td><td>{res.loc[res['vix_pnl_pct']>0,'vix_pnl_pct'].mean():+.1f}%</td></tr>
      <tr><td>Avg loss %</td><td>{avg_loss:+.1f}%</td><td>{res.loc[res['vix_pnl_pct']<=0,'vix_pnl_pct'].mean():+.1f}%</td></tr>
      <tr><td>Max drawdown</td><td>{max_dd:.2f}u</td><td>—</td></tr>
      <tr><td>Calmar</td><td>{calmar:.2f}</td><td>—</td></tr>
      <tr><td>CAGR ({years_span}y)</td><td>{cagr:.1%}</td><td>—</td></tr>
      <tr><td>Final balance ($1k start)</td><td>${final_balance:,.0f}</td><td>—</td></tr>
    </tbody>
  </table>

  <h3>P&L Scatter: Options vs VIX Close</h3>
  <img src="data:image/png;base64,{scatter_b64}" alt="P1a scatter">

  <h3>Equity Curve Comparison</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="P1a equity">

  <h3>By Hold Duration</h3>
  <table>
    <thead><tr><th>Hold bucket</th><th>N</th><th>Win rate</th><th>Avg P&L %</th><th>Expired worthless</th></tr></thead>
    <tbody>{bucket_rows}</tbody>
  </table>

  <h3>Year by Year</h3>
  <table>
    <thead><tr><th>Year</th><th>N</th><th>Win rate</th><th>Units</th><th>Avg P&L %</th></tr></thead>
    <tbody>{yearly_rows}</tbody>
  </table>
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nP1a section written. HTML size: {len(html):,} bytes")
