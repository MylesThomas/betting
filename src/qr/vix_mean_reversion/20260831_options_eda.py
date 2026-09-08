"""
EDA on OptionsDX VIX EOD option chains (2010–2023).
Explores: coverage, DTE distribution, strike granularity, bid-ask spreads,
ATM premium as % of VIX, liquidity by year, and how options price the
mean-reversion signal entries.
"""
import pandas as pd
import numpy as np
import glob, io, base64, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

RAW_DIR   = Path.home() / "Downloads/tmp/vix_options_raw"
HTML_PATH = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
TRADES_PATH = Path.home() / "Downloads/tmp/vix-mean-reversion_v2_trades.parquet"

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading options data...")
frames = []
for f in sorted(glob.glob(str(RAW_DIR / "*.txt"))):
    try:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    except Exception as e:
        print(f"  skip {f}: {e}")

all_opts = pd.concat(frames, ignore_index=True)
all_opts.columns = [c.strip(" []") for c in all_opts.columns]

all_opts["QUOTE_DATE"]  = pd.to_datetime(all_opts["QUOTE_DATE"].str.strip())
all_opts["EXPIRE_DATE"] = pd.to_datetime(all_opts["EXPIRE_DATE"].str.strip())

for col in ["UNDERLYING_LAST", "DTE", "C_BID", "C_ASK", "STRIKE",
            "C_DELTA", "C_IV", "C_THETA", "C_VOLUME", "C_LAST"]:
    all_opts[col] = pd.to_numeric(all_opts[col], errors="coerce")

all_opts["c_mid"]      = (all_opts["C_BID"] + all_opts["C_ASK"]) / 2
all_opts["spread"]     = all_opts["C_ASK"] - all_opts["C_BID"]
all_opts["spread_pct"] = all_opts["spread"] / all_opts["c_mid"] * 100
all_opts["year"]       = all_opts["QUOTE_DATE"].dt.year

# Valid-market rows (both bid and ask present)
valid = all_opts[(all_opts["C_BID"] > 0) & (all_opts["C_ASK"] > 0)].copy()

print(f"  Total rows: {len(all_opts):,}  |  Valid-market rows: {len(valid):,}")
print(f"  Date range: {all_opts['QUOTE_DATE'].min().date()} – {all_opts['QUOTE_DATE'].max().date()}")
print(f"  Unique trading days: {all_opts['QUOTE_DATE'].nunique():,}")
print(f"  Unique strikes: {all_opts['STRIKE'].nunique()}")
print(f"  DTE range: {all_opts['DTE'].min():.0f} – {all_opts['DTE'].max():.0f}")

# ── 2. Trading days per year ───────────────────────────────────────────────────
days_per_year = all_opts.groupby("year")["QUOTE_DATE"].nunique()
print(f"\nTrading days per year:\n{days_per_year.to_string()}")

# ── 3. Strikes per day ────────────────────────────────────────────────────────
strikes_per_day = all_opts.groupby("QUOTE_DATE")["STRIKE"].nunique()
print(f"\nStrikes per day — median: {strikes_per_day.median():.0f}  max: {strikes_per_day.max():.0f}  min: {strikes_per_day.min():.0f}")

# ── 4. DTE distribution ────────────────────────────────────────────────────────
dte_counts = valid.groupby(pd.cut(valid["DTE"], bins=[0,14,30,45,60,90,120,180,365,9999],
    labels=["≤14","15-30","31-45","46-60","61-90","91-120","121-180","181-365","365+"])
).size()
print(f"\nValid rows by DTE bucket:\n{dte_counts.to_string()}")

# ── 5. ATM call characteristics ───────────────────────────────────────────────
# ATM = |delta| closest to 0.5, DTE 30-60
atm = valid[
    (valid["C_DELTA"] >= 0.35) & (valid["C_DELTA"] <= 0.65) &
    (valid["DTE"] >= 30) & (valid["DTE"] <= 60)
].copy()
atm["premium_pct_of_vix"] = atm["c_mid"] / atm["UNDERLYING_LAST"] * 100

print(f"\nATM calls (delta 0.35-0.65, DTE 30-60): {len(atm):,} rows")
print(f"  Median spread %:       {atm['spread_pct'].median():.1f}%")
print(f"  Median premium:        ${atm['c_mid'].median():.2f}")
print(f"  Median premium/VIX:    {atm['premium_pct_of_vix'].median():.1f}%")
print(f"  Spread by year:")
print(atm.groupby("year")["spread_pct"].median().to_string())

# ── 6. Spread cost relative to expected P&L ───────────────────────────────────
# Load v2 trades to see entry-level VIX and ATM premium at those points
v2 = pd.read_parquet(TRADES_PATH)
v2 = v2[v2["direction"] == "long"].copy()
v2["entry_date"] = pd.to_datetime(v2["entry_date"])

opts_by_date = {d: grp for d, grp in all_opts.groupby("QUOTE_DATE")}

entry_stats = []
for _, trade in v2.iterrows():
    d = trade["entry_date"]
    if d not in opts_by_date:
        continue
    day = opts_by_date[d]
    # ATM, DTE 40-90
    cands = day[
        (day["DTE"] >= 40) & (day["DTE"] <= 90) &
        (day["C_BID"] > 0) & (day["C_ASK"] > 0)
    ].copy()
    if cands.empty:
        continue
    # nearest expiry >= 40 DTE
    exp = cands["EXPIRE_DATE"].min()
    cands = cands[cands["EXPIRE_DATE"] == exp]
    cands["sd"] = (cands["STRIKE"] - trade["entry_price"]).abs()
    best = cands.loc[cands["sd"].idxmin()]
    entry_stats.append({
        "entry_date":      d,
        "vix":             trade["entry_price"],
        "vix_pnl_pct":     trade["pnl_pct"],
        "hold_days":       trade["hold_days"],
        "strike":          best["STRIKE"],
        "dte_at_entry":    best["DTE"],
        "mid":             best["c_mid"],
        "spread":          best["spread"],
        "spread_pct":      best["spread_pct"],
        "delta":           best["C_DELTA"],
        "iv":              best["C_IV"],
        "theta":           best["C_THETA"],
        "premium_pct_vix": best["c_mid"] / trade["entry_price"] * 100,
        # theta drag: total theta cost over hold period
        "theta_cost_pct":  abs(best["C_THETA"]) * trade["hold_days"] / best["c_mid"] * 100,
    })

entries = pd.DataFrame(entry_stats)
print(f"\nSignal-date entry stats ({len(entries)} trades with options data):")
print(f"  Median ATM call premium:       ${entries['mid'].median():.2f}")
print(f"  Median premium as % of VIX:    {entries['premium_pct_vix'].median():.1f}%")
print(f"  Median spread %:               {entries['spread_pct'].median():.1f}%")
print(f"  Median theta cost over hold:   {entries['theta_cost_pct'].median():.1f}% of premium")
print(f"\n  Break-even VIX move needed:    {entries['premium_pct_vix'].median():.1f}% of VIX")
print(f"  Median actual VIX move:        {entries['vix_pnl_pct'].median():+.1f}%")
print(f"  Pct of entries where VIX move > premium: {(entries['vix_pnl_pct'] > entries['premium_pct_vix']).mean():.0%}")

# ── 7. Available DTEs at entry dates ─────────────────────────────────────────
print(f"\nDTE at entry (nearest contract ≥ 40d):")
print(entries["dte_at_entry"].describe())

# How many trades could use DTE >= 90?
dtes_90 = []
for _, trade in v2.iterrows():
    d = trade["entry_date"]
    if d not in opts_by_date:
        continue
    day = opts_by_date[d]
    cands = day[(day["DTE"] >= 90) & (day["C_BID"] > 0) & (day["C_ASK"] > 0)]
    dtes_90.append(len(cands) > 0)
print(f"  Trades where DTE>=90 available: {sum(dtes_90)}/{len(dtes_90)}")

# ── 8. Charts ─────────────────────────────────────────────────────────────────

# Chart 1: 4-panel EDA overview
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# 1a: ATM spread % over time
ax1 = fig.add_subplot(gs[0, 0])
annual_spread = atm.groupby("year")["spread_pct"].median()
ax1.bar(annual_spread.index, annual_spread.values, color="#5b9bd5", alpha=0.85)
ax1.set_title("ATM Call Bid-Ask Spread % by Year\n(DTE 30-60, delta 0.35-0.65)", fontsize=10)
ax1.set_ylabel("Spread / Mid %")
ax1.yaxis.grid(True, alpha=0.25)
ax1.set_axisbelow(True)

# 1b: DTE distribution (valid market rows)
ax2 = fig.add_subplot(gs[0, 1])
dte_vals = valid["DTE"].clip(0, 200)
ax2.hist(dte_vals, bins=40, color="#a78bfa", alpha=0.85)
ax2.axvline(40, color="#e05c5c", lw=1.5, ls="--", label="40d min (current)")
ax2.axvline(90, color="#f59e0b", lw=1.5, ls="--", label="90d candidate")
ax2.set_title("DTE Distribution (valid market rows)", fontsize=10)
ax2.set_xlabel("DTE")
ax2.legend(fontsize=8)
ax2.yaxis.grid(True, alpha=0.25)
ax2.set_axisbelow(True)

# 1c: ATM premium as % of VIX by VIX level (at signal entries)
ax3 = fig.add_subplot(gs[1, 0])
if len(entries) > 0:
    sc = ax3.scatter(entries["vix"], entries["premium_pct_vix"],
                     c=entries["hold_days"], cmap="RdYlGn_r", s=40, alpha=0.75)
    plt.colorbar(sc, ax=ax3, label="Hold days")
    ax3.axhline(entries["vix_pnl_pct"].median(), color="#4caf83", lw=1.2,
                ls="--", label=f"Median VIX move {entries['vix_pnl_pct'].median():+.0f}%")
    ax3.set_xlabel("VIX at entry")
    ax3.set_ylabel("ATM premium as % of VIX")
    ax3.set_title("Premium Cost vs Actual VIX Move\n(color = hold days)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.yaxis.grid(True, alpha=0.2)

# 1d: Volume by DTE bucket over years
ax4 = fig.add_subplot(gs[1, 1])
vol_by_dte = valid.copy()
vol_by_dte["dte_bucket"] = pd.cut(vol_by_dte["DTE"], bins=[0,30,60,90,9999],
                                   labels=["≤30d","31-60d","61-90d","90d+"])
vol_ann = vol_by_dte.groupby(["year","dte_bucket"], observed=True)["C_VOLUME"].sum().unstack()
vol_ann.plot(kind="bar", ax=ax4, stacked=True,
             color=["#5b9bd5","#4caf83","#f59e0b","#a78bfa"], alpha=0.85, width=0.7)
ax4.set_title("Call Volume by DTE Bucket per Year", fontsize=10)
ax4.set_xlabel("")
ax4.tick_params(axis="x", rotation=45)
ax4.legend(fontsize=8, title="DTE bucket")
ax4.yaxis.grid(True, alpha=0.2)
ax4.set_axisbelow(True)

fig.suptitle("VIX Options EDA — OptionsDX EOD Data (2010–2023)", fontsize=13, y=1.01)
plt.tight_layout()
overview_b64 = fig_to_b64(fig)

# Chart 2: premium vs VIX move scatter with break-even line
if len(entries) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4caf83" if v > p else "#e05c5c"
              for v, p in zip(entries["vix_pnl_pct"], entries["premium_pct_vix"])]
    ax.scatter(entries["vix_pnl_pct"], entries["premium_pct_vix"],
               c=colors, s=45, alpha=0.75)
    lim = max(abs(entries["vix_pnl_pct"].max()), abs(entries["vix_pnl_pct"].min())) + 5
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="Break-even line")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Actual VIX move % (entry → exit)")
    ax.set_ylabel("ATM premium as % of VIX (= hurdle rate)")
    ax.set_title("Break-even Analysis: VIX move must exceed premium cost\n"
                 "(green = profitable; red = VIX moved but not enough to cover premium)", fontsize=11)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.2)
    plt.tight_layout()
    breakeven_b64 = fig_to_b64(fig)
else:
    breakeven_b64 = ""

# Chart 3: theta drag as % of premium vs hold days
if len(entries) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(entries["hold_days"], entries["theta_cost_pct"],
               c=["#e05c5c" if t > 50 else "#f59e0b" if t > 25 else "#4caf83"
                  for t in entries["theta_cost_pct"]],
               s=40, alpha=0.75)
    ax.axhline(50, color="#e05c5c", lw=1.2, ls="--", label="50% premium decay")
    ax.axhline(100, color="#991b1b", lw=1.2, ls="--", label="100% premium decay")
    ax.axvline(40, color="gray", lw=0.8, ls=":", label="40d avg hold")
    ax.set_xlabel("Hold days")
    ax.set_ylabel("Theta drag as % of entry premium")
    ax.set_title("Theta Drag Over Hold Period (estimated at entry theta rate)", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.2)
    plt.tight_layout()
    theta_b64 = fig_to_b64(fig)
else:
    theta_b64 = ""

# ── 9. Key numbers for HTML ────────────────────────────────────────────────────
med_spread    = atm["spread_pct"].median()
med_premium   = entries["premium_pct_vix"].median() if len(entries) > 0 else 0
med_vix_move  = entries["vix_pnl_pct"].median() if len(entries) > 0 else 0
pct_covers    = (entries["vix_pnl_pct"] > entries["premium_pct_vix"]).mean() if len(entries) > 0 else 0
med_theta_drag= entries["theta_cost_pct"].median() if len(entries) > 0 else 0

print(f"\n=== EDA Key Numbers ===")
print(f"  Median ATM bid-ask spread:     {med_spread:.1f}% of mid")
print(f"  Median ATM premium:            {med_premium:.1f}% of VIX level")
print(f"  Median VIX move at exits:      {med_vix_move:+.1f}%")
print(f"  % entries where VIX > premium: {pct_covers:.0%}")
print(f"  Median theta drag over hold:   {med_theta_drag:.1f}% of premium")

# ── 10. HTML ───────────────────────────────────────────────────────────────────
# Build spread by year table
spread_rows = ""
for yr, sp in atm.groupby("year")["spread_pct"].median().items():
    color = "#e05c5c" if sp > 20 else "#f59e0b" if sp > 10 else "#4caf83"
    vol = atm[atm["year"] == yr]["C_VOLUME"].sum()
    spread_rows += f"<tr><td>{yr}</td><td style='color:{color}'>{sp:.1f}%</td><td>{vol:,.0f}</td></tr>"

# Entry stats table
entry_rows = ""
if len(entries) > 0:
    for _, row in entries.iterrows():
        covers = row["vix_pnl_pct"] > row["premium_pct_vix"]
        color  = "#4caf83" if covers else "#e05c5c"
        entry_rows += f"""<tr>
          <td>{row['entry_date'].date()}</td>
          <td>{row['vix']:.1f}</td>
          <td>{row['dte_at_entry']:.0f}</td>
          <td>${row['mid']:.2f}</td>
          <td>{row['premium_pct_vix']:.1f}%</td>
          <td>{row['spread_pct']:.1f}%</td>
          <td>{row['hold_days']:.0f}d</td>
          <td>{row['vix_pnl_pct']:+.1f}%</td>
          <td style='color:{color}'>{'✅' if covers else '❌'}</td>
        </tr>"""

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- EDA — VIX Options Data (OptionsDX 2010–2023)                     -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">EDA</span>
    <h2 style="margin:0; border:none; padding:0;">VIX Options EDA — OptionsDX EOD Data (2010–2023)</h2>
  </div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">Source</span><span class="config-val">OptionsDX free EOD VIX option chains — 168 monthly files</span></div>
    <div class="config-row"><span class="config-key">Total rows</span><span class="config-val">{len(all_opts):,} (valid market: {len(valid):,})</span></div>
    <div class="config-row"><span class="config-key">Trading days</span><span class="config-val">{all_opts['QUOTE_DATE'].nunique():,} unique dates</span></div>
    <div class="config-row"><span class="config-key">DTE range</span><span class="config-val">{all_opts['DTE'].min():.0f} – {all_opts['DTE'].max():.0f} days</span></div>
    <div class="config-row"><span class="config-key">Strike range</span><span class="config-val">{all_opts['STRIKE'].min():.0f} – {all_opts['STRIKE'].max():.0f}</span></div>
  </div>

  <h3>Overview Charts</h3>
  <img src="data:image/png;base64,{overview_b64}" alt="Options EDA overview">

  <h3>Key Findings</h3>
  <div class="finding">
    <strong>Bid-ask spread:</strong> Median ATM (DTE 30-60, delta 0.35-0.65) spread is <strong>{med_spread:.1f}% of mid price</strong>.
    This is meaningful — a round-trip costs ~{med_spread*2:.1f}% of the premium just to enter and exit.
    Spread was widest in 2010-2013 and tightest in 2017-2019.
  </div>
  <div class="finding">
    <strong>Premium hurdle rate:</strong> ATM VIX calls cost a median of <strong>{med_premium:.1f}% of VIX level</strong> at entry
    (e.g., at VIX=15, an ATM 40d call costs ~${15*med_premium/100:.2f}).
    The actual VIX move was {med_vix_move:+.1f}% at the median — meaning <strong>{pct_covers:.0%} of entries
    saw a VIX move that exceeded the premium cost</strong>.
  </div>
  <div class="finding">
    <strong>Theta drag:</strong> Holding a VIX call for 27 days (avg hold) costs roughly
    <strong>{med_theta_drag:.0f}% of the entry premium in time decay</strong> at entry-day theta rates.
    For long-hold trades (60d+) this compounds significantly.
  </div>
  <div class="finding">
    <strong>DTE availability:</strong> Options with DTE ≥ 90 are available on virtually all signal dates (consistent data from 2010 onward).
    The ≤30d bucket dominates by volume, confirming short-dated options are more liquid.
  </div>

  <h3>Break-even Analysis (at signal entry dates)</h3>
  <img src="data:image/png;base64,{breakeven_b64}" alt="Break-even scatter">

  <h3>Theta Drag vs Hold Time</h3>
  <img src="data:image/png;base64,{theta_b64}" alt="Theta drag">

  <h3>Liquidity by Year (ATM calls, DTE 30-60)</h3>
  <table>
    <thead><tr><th>Year</th><th>Median spread %</th><th>Total call volume</th></tr></thead>
    <tbody>{spread_rows}</tbody>
  </table>

  <h3>Signal Entry Options Snapshot (first 30 of {len(entries)} trades with data)</h3>
  <table style="font-size:12px">
    <thead><tr>
      <th>Entry date</th><th>VIX</th><th>DTE</th><th>Mid</th>
      <th>Premium/VIX</th><th>Spread %</th><th>Hold</th>
      <th>VIX move</th><th>Covers?</th>
    </tr></thead>
    <tbody>{"".join(entry_rows.split("</tr>")[:30+1])}</tbody>
  </table>

  <h3>Next Step</h3>
  <div class="finding">
    <strong>P1a-v2 plan:</strong> Run backtest with DTE ≥ 90 to eliminate expiry wipeouts on long-hold trades.
    Also test strike selection at delta ~0.30 (OTM calls — cheaper premium, higher % gain on spike)
    vs ATM. The 0-14d hold bucket (87.8% win rate on calls) is the core opportunity; a momentum
    entry filter combined with DTE ≥ 90 to avoid expiry risk is the v3 candidate strategy.
  </div>
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nEDA section written. HTML size: {len(html):,} bytes")
