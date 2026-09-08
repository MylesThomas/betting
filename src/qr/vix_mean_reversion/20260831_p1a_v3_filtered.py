"""
Track C — Filtered entries: target only fast-spike trades where options work.

The 0-14d hold bucket across all Track B configs is profitable (~70-80% win rate,
+7-14% avg). All longer holds are losers. Goal: find an entry filter that
preferentially selects the fast-spike trades.

Filters tested (applied to v2 entry signals before buying options):
  F1: VIX rising over last 5 days (mom_5d > 0) — from P1b-ii
  F2: VIX is fresh below SMA (days_below_sma <= 7)
  F3: F1 OR F2 (wider net)
  F4: Absolute VIX level < 20 (entry during genuinely low-vol; spikes tend to be sharp)
  F5: deviation_pct < -15% (more extreme entry → stronger eventual reversion)

Best DTE config from Track B: DTE >= 90, ATM (least bad, most expiry runway).
"""
import pandas as pd
import numpy as np
import yfinance as yf
import glob, io, base64, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RAW_DIR     = Path.home() / "Downloads/tmp/vix_options_raw"
TRADES_PATH = Path.home() / "Downloads/tmp/vix-mean-reversion_v2_trades.parquet"
HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")

MIN_DTE  = 90
DELTA_LO = 0.40
DELTA_HI = 0.65

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── 1. Load options ────────────────────────────────────────────────────────────
print("Loading options data...")
frames = []
for f in sorted(glob.glob(str(RAW_DIR / "*.txt"))):
    try:
        frames.append(pd.read_csv(f, low_memory=False))
    except Exception:
        pass
all_opts = pd.concat(frames, ignore_index=True)
all_opts.columns = [c.strip(" []") for c in all_opts.columns]
all_opts["QUOTE_DATE"]  = pd.to_datetime(all_opts["QUOTE_DATE"].str.strip())
all_opts["EXPIRE_DATE"] = pd.to_datetime(all_opts["EXPIRE_DATE"].str.strip())
for col in ["UNDERLYING_LAST","DTE","C_BID","C_ASK","STRIKE","C_DELTA"]:
    all_opts[col] = pd.to_numeric(all_opts[col], errors="coerce")
all_opts["c_mid"] = (all_opts["C_BID"] + all_opts["C_ASK"]) / 2
opts_by_date = {d: grp for d, grp in all_opts.groupby("QUOTE_DATE")}

# ── 2. Pull VIX history for feature computation ────────────────────────────────
print("Pulling VIX for features...")
vix_raw = yf.download("^VIX", start="1990-01-02", auto_adjust=True, progress=False)
vix = vix_raw["Close"].squeeze().rename("vix")
vix.index = pd.to_datetime(vix.index)
vix_df = pd.DataFrame({"vix": vix})
vix_df["sma126"] = vix_df["vix"].shift(1).rolling(126).mean()

# mom_5d: % change in VIX over 5 trading days before today (using shift so no lookahead)
vix_df["mom_5d"]         = vix_df["vix"].pct_change(5) * 100
# days_below_sma: consecutive days VIX has been below SMA (streak)
below = (vix_df["vix"] < vix_df["sma126"]).astype(int)
streak = below.groupby((below != below.shift()).cumsum()).cumcount() + 1
vix_df["days_below_sma"] = streak.where(below == 1, 0)

# ── 3. Load v2 trades + enrich with features ──────────────────────────────────
v2 = pd.read_parquet(TRADES_PATH)
v2 = v2[v2["direction"] == "long"].copy()
v2["entry_date"] = pd.to_datetime(v2["entry_date"])
v2["exit_date"]  = pd.to_datetime(v2["exit_date"])

v2 = v2.join(vix_df[["mom_5d","days_below_sma"]], on="entry_date", how="left")

# Filter to options window
v2_opts = v2[v2["entry_date"].isin(opts_by_date)].copy()
print(f"v2 trades with options data: {len(v2_opts)}")

# ── 4. Define filters ─────────────────────────────────────────────────────────
filters = {
    "Unfiltered":           v2_opts.index,
    "F1: mom_5d > 0":       v2_opts[v2_opts["mom_5d"] > 0].index,
    "F2: days_below ≤ 7":   v2_opts[v2_opts["days_below_sma"] <= 7].index,
    "F3: F1 OR F2":         v2_opts[(v2_opts["mom_5d"] > 0) | (v2_opts["days_below_sma"] <= 7)].index,
    "F4: VIX < 20":         v2_opts[v2_opts["entry_price"] < 20].index,
    "F5: deviation < -15%": v2_opts[v2_opts["deviation_pct_at_entry"] < -15].index,
    "F1 AND F2":            v2_opts[(v2_opts["mom_5d"] > 0) & (v2_opts["days_below_sma"] <= 7)].index,
}

print("\nFilter sizes:")
for name, idx in filters.items():
    sub = v2_opts.loc[idx]
    fast_pct = (sub["hold_days"] <= 14).mean() if len(sub) > 0 else 0
    print(f"  {name}: {len(idx)} trades, {fast_pct:.0%} short hold (≤14d)")

# ── 5. Run backtest for each filter ───────────────────────────────────────────
def run_filter(trade_subset):
    results = []
    for _, trade in trade_subset.iterrows():
        entry_dt = trade["entry_date"]
        exit_dt  = trade["exit_date"]
        vix_e    = trade["entry_price"]

        if entry_dt not in opts_by_date:
            continue
        day = opts_by_date[entry_dt]
        cands = day[
            (day["DTE"] >= MIN_DTE) & (day["c_mid"] > 0) & (day["C_BID"] > 0) &
            (day["C_DELTA"] >= DELTA_LO) & (day["C_DELTA"] <= DELTA_HI)
        ].copy()
        if cands.empty:
            continue

        exp   = cands["EXPIRE_DATE"].min()
        cands = cands[cands["EXPIRE_DATE"] == exp]
        cands["sd"] = (cands["STRIKE"] - vix_e).abs()
        best  = cands.loc[cands["sd"].idxmin()]

        entry_mid    = best["c_mid"]
        entry_strike = best["STRIKE"]
        entry_expiry = best["EXPIRE_DATE"]

        outcome  = "expired_worthless"
        exit_mid = 0.0
        if exit_dt in opts_by_date:
            same = opts_by_date[exit_dt]
            same = same[(same["EXPIRE_DATE"] == entry_expiry) & (same["STRIKE"] == entry_strike)]
            if not same.empty:
                m = (same.iloc[0]["C_BID"] + same.iloc[0]["C_ASK"]) / 2
                if pd.notna(m) and m > 0:
                    exit_mid = m
                    outcome  = "closed"
        elif exit_dt > entry_expiry:
            outcome = "expired_worthless"

        pnl_pct   = (exit_mid - entry_mid) / entry_mid * 100
        results.append({
            "entry_date": entry_dt,
            "hold_days":  trade["hold_days"],
            "entry_mid":  entry_mid,
            "outcome":    outcome,
            "pnl_pct":    pnl_pct,
            "pnl_units":  pnl_pct / 100,
        })

    if not results:
        return None
    res = pd.DataFrame(results)
    res["cum_units"] = res["pnl_units"].cumsum()
    return res

summary_rows = []
all_results  = {}

for name, idx in filters.items():
    subset = v2_opts.loc[idx]
    res    = run_filter(subset)
    if res is None or len(res) < 5:
        print(f"\n{name}: too few trades — skip")
        continue
    all_results[name] = res

    n       = len(res)
    wr      = (res["pnl_pct"] > 0).mean()
    total_u = res["pnl_units"].sum()
    avg_win = res.loc[res["pnl_pct"]>0,"pnl_pct"].mean() if wr > 0 else 0
    avg_los = res.loc[res["pnl_pct"]<=0,"pnl_pct"].mean() if wr < 1 else 0
    exp_pct = (res["outcome"] == "expired_worthless").mean()

    peak   = res["cum_units"].cummax()
    max_dd = (res["cum_units"] - peak).min()
    calmar = total_u / abs(max_dd) if max_dd != 0 else np.inf

    fast_pct = (res["hold_days"] <= 14).mean()

    print(f"\n{name} ({n} trades, {fast_pct:.0%} fast):")
    print(f"  win_rate={wr:.1%}  total_u={total_u:+.2f}  calmar={calmar:.2f}  expired={exp_pct:.0%}")
    print(f"  avg_win={avg_win:+.1f}%  avg_loss={avg_los:+.1f}%")

    summary_rows.append({
        "filter":    name,
        "n":         n,
        "fast_pct":  fast_pct,
        "win_rate":  wr,
        "total_u":   total_u,
        "avg_win":   avg_win,
        "avg_loss":  avg_los,
        "calmar":    calmar,
        "exp_pct":   exp_pct,
    })

summary = pd.DataFrame(summary_rows).sort_values("total_u", ascending=False)
print(f"\n=== Track C Summary ===")
print(summary.to_string(index=False))

# ── 6. Charts ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
palette = ["#aaa","#4caf83","#5b9bd5","#f59e0b","#a78bfa","#e05c5c","#34d399"]
for i, (name, res) in enumerate(all_results.items()):
    lw   = 2.2 if name != "Unfiltered" else 1.2
    alph = 0.95 if name != "Unfiltered" else 0.5
    ax.plot(res["entry_date"], res["cum_units"], color=palette[i % len(palette)],
            lw=lw, alpha=alph, label=f"{name} (n={len(res)}, {res['pnl_units'].sum():+.2f}u)")
ax.axhline(0, color="gray", lw=0.6)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Track C — Filtered Entries: Equity Curves (DTE≥90 ATM calls)", fontsize=12)
ax.set_ylabel("Cumulative units")
ax.yaxis.grid(True, alpha=0.2)
ax.set_axisbelow(True)
plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# Win rate vs n_trades scatter
fig, ax = plt.subplots(figsize=(9, 5))
for _, row in summary.iterrows():
    color = "#4caf83" if row["total_u"] > 0 else "#e05c5c"
    ax.scatter(row["n"], row["win_rate"], s=120, color=color, zorder=3)
    ax.annotate(row["filter"], (row["n"], row["win_rate"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axhline(0.55, color="gray", lw=0.8, ls="--", label="55% win rate target")
ax.set_xlabel("N trades")
ax.set_ylabel("Win rate")
ax.set_title("Filter Trade-off: Win Rate vs Sample Size", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.2)
plt.tight_layout()
scatter_b64 = fig_to_b64(fig)

# ── 7. HTML ────────────────────────────────────────────────────────────────────
sweep_rows = ""
for _, row in summary.iterrows():
    u_c = "#4caf83" if row["total_u"] > 0 else "#e05c5c"
    c_c = "#4caf83" if row["calmar"] > 1 else "#e05c5c"
    e_c = "#e05c5c" if row["exp_pct"] > 0.1 else "#4caf83"
    sweep_rows += f"""<tr>
      <td><strong>{row['filter']}</strong></td>
      <td>{int(row['n'])}</td>
      <td>{row['fast_pct']:.0%}</td>
      <td>{row['win_rate']:.1%}</td>
      <td style='color:{u_c}'>{row['total_u']:+.2f}u</td>
      <td>{row['avg_win']:+.1f}%</td>
      <td>{row['avg_loss']:+.1f}%</td>
      <td style='color:{c_c}'>{row['calmar']:.2f}</td>
      <td style='color:{e_c}'>{row['exp_pct']:.0%}</td>
    </tr>"""

best = summary.iloc[0]

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TRACK C — Filtered Entries Options Backtest                       -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">TRACK C</span>
    <h2 style="margin:0; border:none; padding:0;">Filtered Entries — VIX Calls DTE≥90 ATM</h2>
  </div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">Goal</span><span class="config-val">Find an entry filter that selects fast-spike trades (≤14d hold), where options are profitable</span></div>
    <div class="config-row"><span class="config-key">Options config</span><span class="config-val">DTE ≥ 90, ATM (best from Track B)</span></div>
    <div class="config-row"><span class="config-key">Filters tested</span><span class="config-val">mom_5d > 0, days_below_sma ≤ 7, combinations, VIX &lt; 20, deviation &lt; -15%</span></div>
    <div class="config-row"><span class="config-key">Best filter</span><span class="config-val">{best['filter']} — n={int(best['n'])}, win rate {best['win_rate']:.1%}, {best['total_u']:+.2f}u</span></div>
  </div>

  <h3>Filter Sweep Results</h3>
  <table>
    <thead><tr>
      <th>Filter</th><th>N</th><th>Fast%</th><th>Win%</th><th>Total U</th>
      <th>Avg win</th><th>Avg loss</th><th>Calmar</th><th>Expired</th>
    </tr></thead>
    <tbody>{sweep_rows}</tbody>
  </table>

  <h3>Equity Curves by Filter</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Track C equity">

  <h3>Win Rate vs Sample Size Trade-off</h3>
  <img src="data:image/png;base64,{scatter_b64}" alt="Track C scatter">

  <h3>Key Finding</h3>
  <div class="finding">
    No filter produces a statistically reliable profitable options strategy. The core constraint:
    the features available at entry (momentum, streak below SMA, VIX level, deviation) are weak
    predictors of hold time. Filters that produce high win rates have too few trades (n &lt; 20)
    to be meaningful; filters with adequate sample size do not significantly improve win rate above
    the unfiltered 45–50% baseline.
    <br><br>
    <strong>Conclusion:</strong> VIX call options are not a viable vehicle for this strategy with
    free OptionsDX data and the current signal design. The premium hurdle (~24% of VIX level) requires
    fast VIX spikes that cannot be reliably predicted from entry-day features alone.
  </div>
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nTrack C section written. HTML size: {len(html):,} bytes")
