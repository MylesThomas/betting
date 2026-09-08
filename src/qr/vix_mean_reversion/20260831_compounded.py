"""
VIX Mean Reversion — Compounded Returns (Step 3b)
Starting balance: $1,000. 100% of bankroll per trade.
"""
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
TRADES_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_trades.parquet")

STARTING_BALANCE = 1_000.0

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

trades = pd.read_parquet(TRADES_PATH)
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
trades = trades.sort_values("exit_date").reset_index(drop=True)

# ── compounded balance per direction ─────────────────────────────────────────
def compound_series(subset, start=1000.0):
    bal = start
    rows = []
    for _, row in subset.iterrows():
        bal *= (1 + row["pnl_pct"] / 100)
        rows.append({"exit_date": row["exit_date"], "balance": bal,
                     "pnl_pct": row["pnl_pct"], "direction": row["direction"]})
    return pd.DataFrame(rows)

long_c  = compound_series(trades[trades["direction"]=="long"].sort_values("exit_date"))
short_c = compound_series(trades[trades["direction"]=="short"].sort_values("exit_date"))
both_c  = compound_series(trades)

def summarise(series_df, label):
    bal     = series_df["balance"].iloc[-1]
    years   = (series_df["exit_date"].max() - series_df["exit_date"].min()).days / 365.25
    cagr    = (bal / STARTING_BALANCE) ** (1 / years) - 1
    peak    = series_df["balance"].cummax()
    dd_pct  = ((series_df["balance"] - peak) / peak * 100)
    mdd_pct = dd_pct.min()
    n       = len(series_df)
    wins    = (series_df["pnl_pct"] > 0).sum()
    return {
        "direction":        label,
        "n_trades":         n,
        "win_rate":         f"{wins/n:.1%}",
        "starting_balance": f"${STARTING_BALANCE:,.0f}",
        "final_balance":    f"${bal:,.0f}",
        "total_return_pct": f"{(bal/STARTING_BALANCE - 1)*100:,.0f}%",
        "cagr":             f"{cagr*100:.1f}%",
        "max_drawdown_pct": f"{mdd_pct:.1f}%",
    }

summary_df = pd.DataFrame([
    summarise(long_c,  "long"),
    summarise(short_c, "short"),
    summarise(both_c,  "both"),
])
print(summary_df.to_string(index=False))

# year-by-year balance at end of each year
trades["year"] = trades["exit_date"].dt.year
yearly_rows = []
for label, series_df in [("long", long_c), ("short", short_c), ("both", both_c)]:
    series_df["year"] = pd.to_datetime(series_df["exit_date"]).dt.year
    for yr in sorted(trades["year"].unique()):
        yr_trades = series_df[series_df["year"] == yr]
        if len(yr_trades) == 0:
            continue
        bal_end   = yr_trades["balance"].iloc[-1]
        n_yr      = len(yr_trades)
        yearly_rows.append({"year": yr, "direction": label,
                             "n_trades": n_yr, "balance_eoy": round(bal_end, 2)})

yearly_df = pd.DataFrame(yearly_rows)

# ── charts ────────────────────────────────────────────────────────────────────

# 1. Compounded equity curve — log scale
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.plot(long_c["exit_date"],  long_c["balance"],  color="steelblue", lw=1.2, label="Long only")
ax.plot(short_c["exit_date"], short_c["balance"], color="tomato",    lw=1.0, alpha=0.7, label="Short only")
ax.plot(both_c["exit_date"],  both_c["balance"],  color="black",     lw=1.8, label="Combined")
ax.set_yscale("log")
ax.set_ylabel("Balance ($, log scale)")
ax.set_title("Compounded Returns — $1,000 starting balance, 100% sizing, log scale", fontsize=12)
ax.legend()
ax.axhline(STARTING_BALANCE, color="gray", lw=0.7, ls="--", alpha=0.5)

# drawdown panel
ax2 = axes[1]
peak_both = both_c["balance"].cummax()
dd_both   = (both_c["balance"] - peak_both) / peak_both * 100
ax2.fill_between(both_c["exit_date"], dd_both, 0, color="tomato", alpha=0.4)
ax2.plot(both_c["exit_date"], dd_both, color="tomato", lw=0.8)
ax2.set_ylabel("Drawdown (%)")
ax2.set_title("Combined — % Drawdown from Peak", fontsize=11)
ax2.axhline(0, color="black", lw=0.7)

plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# 2. Year-by-year EOY balance (combined)
both_yr = yearly_df[yearly_df["direction"]=="both"].sort_values("year")
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(both_yr["year"], np.log10(both_yr["balance_eoy"]), color="#5b8cf0", alpha=0.85)
ax.set_ylabel("log₁₀(Balance $)")
ax.set_title("Combined EOY Balance by Year (log scale, $1k start)", fontsize=12)
for i, (_, row) in enumerate(both_yr.iterrows()):
    ax.text(row["year"], np.log10(row["balance_eoy"]) + 0.05,
            f"${row['balance_eoy']:,.0f}" if row["balance_eoy"] < 1e7 else f"${row['balance_eoy']:.1e}",
            ha="center", fontsize=6, rotation=90)
plt.tight_layout()
yearly_b64 = fig_to_b64(fig)

# ── HTML ──────────────────────────────────────────────────────────────────────
def df_to_html(df):
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v   = row[c]
            cls = "pass" if c == "direction" and str(v) == "long" else ""
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

both_final   = both_c["balance"].iloc[-1]
both_years   = (both_c["exit_date"].max() - both_c["exit_date"].min()).days / 365.25
both_cagr    = ((both_final / STARTING_BALANCE) ** (1 / both_years) - 1) * 100
peak_b       = both_c["balance"].cummax()
mdd_b        = ((both_c["balance"] - peak_b) / peak_b * 100).min()

step3b_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 3b — COMPOUNDED RETURNS ($1,000 START)                         -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 3b</span>
    <h2 style="margin:0; border:none; padding:0;">Compounded Returns — $1,000 Starting Balance</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_compounded.py</div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">starting_balance</span><span class="config-val">$1,000</span></div>
    <div class="config-row"><span class="config-key">sizing</span><span class="config-val">100% of bankroll per trade (entire balance deployed each trade)</span></div>
    <div class="config-row"><span class="config-key">note</span><span class="config-val">Final balances are theoretical — VIX is not directly tradeable and 100% sizing is not practical at scale. CAGR is the interpretable metric.</span></div>
  </div>

  <h3>Summary</h3>
  {df_to_html(summary_df)}

  <h3>Compounded Equity Curve (log scale) + Drawdown</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Compounded Equity Curve">

  <h3>Combined EOY Balance by Year</h3>
  <img src="data:image/png;base64,{yearly_b64}" alt="EOY Balance">

  <div class="flag-box">
    <strong>Drawdown is worse in % terms on a compounded basis:</strong> The 2008–2009 short crisis trades produced the largest % peak-to-trough drawdown on the combined curve. On flat units it was −0.59u; on the compounded curve the same period is a larger % of the then-current balance. This is expected — compounding amplifies both gains and losses.
  </div>
  <div class="ok-box">
    <strong>Combined CAGR: {both_cagr:.1f}%</strong> — $1,000 grows to ${both_final:,.0f} over {both_years:.0f} years. Max drawdown {mdd_b:.1f}% from peak on the compounded curve.
  </div>

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step3b_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 3b written. HTML size: {len(updated):,} bytes")
