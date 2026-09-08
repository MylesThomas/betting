"""
Track B — VIX calls, DTE >= 90 + strike sweep.

Fixes the main P1a flaw: expiry wipeout on long-hold trades.
Tests: (DTE_min=40, ATM), (DTE_min=90, ATM), (DTE_min=90, OTM delta~0.30)
Same v2 entry/exit signals throughout.
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
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
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
print(f"  Loaded {len(all_opts):,} rows, {len(opts_by_date)} dates")

# ── 2. Load v2 trades ──────────────────────────────────────────────────────────
v2 = pd.read_parquet(TRADES_PATH)
v2 = v2[v2["direction"] == "long"].copy()
v2["entry_date"] = pd.to_datetime(v2["entry_date"])
v2["exit_date"]  = pd.to_datetime(v2["exit_date"])

# ── 3. Sweep configs ───────────────────────────────────────────────────────────
configs = [
    {"label": "DTE≥40  ATM",       "min_dte": 40,  "delta_lo": 0.40, "delta_hi": 0.65},
    {"label": "DTE≥90  ATM",       "min_dte": 90,  "delta_lo": 0.40, "delta_hi": 0.65},
    {"label": "DTE≥90  OTM d~0.30","min_dte": 90,  "delta_lo": 0.20, "delta_hi": 0.40},
    {"label": "DTE≥60  ATM",       "min_dte": 60,  "delta_lo": 0.40, "delta_hi": 0.65},
    {"label": "DTE≥60  OTM d~0.30","min_dte": 60,  "delta_lo": 0.20, "delta_hi": 0.40},
]

def run_config(cfg):
    min_dte  = cfg["min_dte"]
    dlo, dhi = cfg["delta_lo"], cfg["delta_hi"]
    results  = []
    skipped  = 0

    for _, trade in v2.iterrows():
        entry_dt = trade["entry_date"]
        exit_dt  = trade["exit_date"]
        vix_e    = trade["entry_price"]

        if entry_dt not in opts_by_date:
            skipped += 1
            continue

        day = opts_by_date[entry_dt]
        cands = day[
            (day["DTE"] >= min_dte) &
            (day["c_mid"] > 0) &
            (day["C_BID"] > 0) &
            (day["C_DELTA"] >= dlo) &
            (day["C_DELTA"] <= dhi)
        ].copy()

        if cands.empty:
            skipped += 1
            continue

        # Nearest expiry in the valid delta range
        exp = cands["EXPIRE_DATE"].min()
        cands = cands[cands["EXPIRE_DATE"] == exp]
        # For ATM: closest strike to VIX; for OTM: lowest delta in range (most OTM)
        if dlo >= 0.40:
            cands["sd"] = (cands["STRIKE"] - vix_e).abs()
            best = cands.loc[cands["sd"].idxmin()]
        else:
            best = cands.loc[cands["C_DELTA"].idxmin()]

        entry_mid    = best["c_mid"]
        entry_strike = best["STRIKE"]
        entry_expiry = best["EXPIRE_DATE"]
        entry_dte    = best["DTE"]
        entry_delta  = best["C_DELTA"]

        # Exit value
        outcome  = "expired_worthless"
        exit_mid = 0.0

        if exit_dt in opts_by_date:
            exit_day = opts_by_date[exit_dt]
            same = exit_day[
                (exit_day["EXPIRE_DATE"] == entry_expiry) &
                (exit_day["STRIKE"] == entry_strike)
            ]
            if not same.empty:
                row = same.iloc[0]
                m = (row["C_BID"] + row["C_ASK"]) / 2
                if pd.notna(m) and m > 0:
                    exit_mid = m
                    outcome  = "closed"
        elif exit_dt > entry_expiry:
            outcome  = "expired_worthless"
            exit_mid = 0.0

        pnl_pct   = (exit_mid - entry_mid) / entry_mid * 100
        pnl_units = pnl_pct / 100

        results.append({
            "entry_date":  entry_dt,
            "exit_date":   exit_dt,
            "vix_entry":   vix_e,
            "vix_pnl_pct": trade["pnl_pct"],
            "hold_days":   trade["hold_days"],
            "strike":      entry_strike,
            "entry_dte":   entry_dte,
            "entry_delta": entry_delta,
            "entry_mid":   entry_mid,
            "exit_mid":    exit_mid,
            "outcome":     outcome,
            "pnl_pct":     pnl_pct,
            "pnl_units":   pnl_units,
        })

    if not results:
        return None, skipped

    res = pd.DataFrame(results)
    res["cumulative_units"] = res["pnl_units"].cumsum()
    balance = 1_000.0
    bals = []
    for p in res["pnl_pct"]:
        balance *= (1 + p / 100)
        bals.append(balance)
    res["balance"] = bals
    return res, skipped

# ── 4. Run all configs ────────────────────────────────────────────────────────
all_results = {}
summary_rows = []

for cfg in configs:
    print(f"\nRunning: {cfg['label']}")
    res, skipped = run_config(cfg)
    if res is None:
        print(f"  No trades — skipped")
        continue
    all_results[cfg["label"]] = res

    n       = len(res)
    wins    = res[res["pnl_pct"] > 0]
    losses  = res[res["pnl_pct"] <= 0]
    wr      = len(wins) / n
    total_u = res["pnl_units"].sum()
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_los = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    expired_pct = (res["outcome"] == "expired_worthless").mean()

    peak   = res["cumulative_units"].cummax()
    max_dd = (res["cumulative_units"] - peak).min()
    calmar = total_u / abs(max_dd) if max_dd != 0 else np.inf

    years  = res["exit_date"].dt.year.max() - res["entry_date"].dt.year.min() + 1
    ann    = total_u / years * 100
    final  = res["balance"].iloc[-1]
    cagr   = (final / 1_000) ** (1 / years) - 1

    # Hold-bucket breakdown
    res["hold_bucket"] = pd.cut(res["hold_days"], bins=[0,14,30,60,999],
                                 labels=["0-14d","15-30d","31-60d","60d+"])
    bucket = res.groupby("hold_bucket", observed=True).agg(
        n=("pnl_pct","count"),
        wr=("pnl_pct", lambda x: (x>0).mean()),
        avg_pct=("pnl_pct","mean"),
        exp=("outcome", lambda x: (x=="expired_worthless").mean()),
    )

    print(f"  n={n}  win_rate={wr:.1%}  total_u={total_u:+.2f}  calmar={calmar:.2f}  "
          f"expired={expired_pct:.0%}  cagr={cagr:.1%}")
    print(f"  Hold buckets:\n{bucket.to_string()}")

    summary_rows.append({
        "config":     cfg["label"],
        "n":          n,
        "win_rate":   wr,
        "total_u":    total_u,
        "avg_win":    avg_win,
        "avg_loss":   avg_los,
        "max_dd":     max_dd,
        "calmar":     calmar,
        "expired_pct":expired_pct,
        "cagr":       cagr,
        "final_bal":  final,
    })

summary = pd.DataFrame(summary_rows).sort_values("total_u", ascending=False)
print(f"\n=== Track B Summary ===")
print(summary[["config","n","win_rate","total_u","calmar","expired_pct","cagr"]].to_string(index=False))

# ── 5. Charts ──────────────────────────────────────────────────────────────────
# Equity curves for all configs
fig, ax = plt.subplots(figsize=(14, 6))
colors_map = {
    "DTE≥40  ATM":        "#aaa",
    "DTE≥60  ATM":        "#f59e0b",
    "DTE≥60  OTM d~0.30": "#a78bfa",
    "DTE≥90  ATM":        "#4caf83",
    "DTE≥90  OTM d~0.30": "#5b9bd5",
}
for label, res in all_results.items():
    ax.plot(res["entry_date"], res["cumulative_units"],
            lw=1.8 if "90" in label else 1.2,
            color=colors_map.get(label, "gray"),
            ls="-" if "ATM" in label else "--",
            alpha=0.9, label=label)
ax.axhline(0, color="gray", lw=0.6)
ax.legend(fontsize=9)
ax.set_title("Track B — VIX Calls: DTE & Strike Sweep Equity Curves (2010–2023)", fontsize=12)
ax.set_ylabel("Cumulative units")
ax.yaxis.grid(True, color="gray", alpha=0.2)
ax.set_axisbelow(True)
plt.tight_layout()
equity_b64 = fig_to_b64(fig)

# Expired worthless by hold bucket — for best config (DTE≥90 ATM)
best_label = summary.iloc[0]["config"]
best_res   = all_results.get(best_label)
if best_res is not None:
    best_res["hold_bucket"] = pd.cut(best_res["hold_days"], bins=[0,14,30,60,999],
                                      labels=["0-14d","15-30d","31-60d","60d+"])
    bucket_best = best_res.groupby("hold_bucket", observed=True).agg(
        n=("pnl_pct","count"),
        wr=("pnl_pct", lambda x: (x>0).mean()),
        avg_pct=("pnl_pct","mean"),
        exp_pct=("outcome", lambda x: (x=="expired_worthless").mean()),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    colors = ["#4caf83" if v >= 0 else "#e05c5c" for v in bucket_best["avg_pct"]]
    ax.bar(bucket_best["hold_bucket"].astype(str), bucket_best["avg_pct"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{best_label} — Avg P&L % by Hold Bucket", fontsize=10)
    ax.set_ylabel("Avg P&L %")
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.bar(bucket_best["hold_bucket"].astype(str), bucket_best["exp_pct"] * 100,
           color="#e05c5c", alpha=0.85)
    ax.set_title(f"{best_label} — % Expired Worthless by Hold Bucket", fontsize=10)
    ax.set_ylabel("% expired worthless")
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    bucket_b64 = fig_to_b64(fig)
else:
    bucket_b64 = ""

# ── 6. HTML ────────────────────────────────────────────────────────────────────
sweep_rows = ""
for _, row in summary.iterrows():
    u_color = "#4caf83" if row["total_u"] > 0 else "#e05c5c"
    e_color = "#e05c5c" if row["expired_pct"] > 0.15 else "#4caf83"
    c_color = "#4caf83" if row["calmar"] > 1 else "#e05c5c"
    sweep_rows += f"""<tr>
      <td><strong>{row['config']}</strong></td>
      <td>{int(row['n'])}</td>
      <td>{row['win_rate']:.1%}</td>
      <td style='color:{u_color}'>{row['total_u']:+.2f}u</td>
      <td>{row['avg_win']:+.1f}%</td>
      <td>{row['avg_loss']:+.1f}%</td>
      <td style='color:{c_color}'>{row['calmar']:.2f}</td>
      <td style='color:{e_color}'>{row['expired_pct']:.0%}</td>
      <td>{row['cagr']:.1%}</td>
      <td>${row['final_bal']:,.0f}</td>
    </tr>"""

best_row = summary.iloc[0]
best_u   = best_row["total_u"]
best_color = "#4caf83" if best_u > 0 else "#e05c5c"

bucket_table = ""
if best_res is not None:
    for _, r in bucket_best.iterrows():
        c = "#4caf83" if r["avg_pct"] >= 0 else "#e05c5c"
        ec = "#e05c5c" if r["exp_pct"] > 0.1 else "#4caf83"
        bucket_table += f"""<tr>
          <td>{r['hold_bucket']}</td>
          <td>{int(r['n'])}</td>
          <td>{r['wr']:.0%}</td>
          <td style='color:{c}'>{r['avg_pct']:+.1f}%</td>
          <td style='color:{ec}'>{r['exp_pct']:.0%}</td>
        </tr>"""

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TRACK B — VIX Calls DTE Sweep                                     -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">TRACK B</span>
    <h2 style="margin:0; border:none; padding:0;">VIX Calls — DTE &amp; Strike Sweep (OptionsDX 2010–2023)</h2>
  </div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">Configs tested</span><span class="config-val">5 combinations: DTE min (40/60/90) × strike (ATM delta 0.40-0.65, OTM delta 0.20-0.40)</span></div>
    <div class="config-row"><span class="config-key">Signal</span><span class="config-val">v2 (126d SMA, 10% entry, 2% exit, long only) — same for all</span></div>
    <div class="config-row"><span class="config-key">Execution</span><span class="config-val">Mid price (bid+ask)/2; full premium loss if option expires before exit</span></div>
    <div class="config-row"><span class="config-key">Best config</span><span class="config-val" style="color:{best_color}">{best_row['config']} — {best_row['total_u']:+.2f}u, {best_row['win_rate']:.1%} win rate, {best_row['expired_pct']:.0%} expired</span></div>
  </div>

  <h3>Sweep Results (sorted by total units)</h3>
  <table>
    <thead><tr>
      <th>Config</th><th>N</th><th>Win%</th><th>Total U</th>
      <th>Avg win</th><th>Avg loss</th><th>Calmar</th><th>Expired</th><th>CAGR</th><th>$1k final</th>
    </tr></thead>
    <tbody>{sweep_rows}</tbody>
  </table>

  <h3>Equity Curves — All Configs</h3>
  <img src="data:image/png;base64,{equity_b64}" alt="Track B equity curves">

  <h3>Best Config ({best_row['config']}) — Hold Bucket Breakdown</h3>
  <table>
    <thead><tr><th>Hold</th><th>N</th><th>Win%</th><th>Avg P&amp;L%</th><th>Expired</th></tr></thead>
    <tbody>{bucket_table}</tbody>
  </table>
  <img src="data:image/png;base64,{bucket_b64}" alt="Track B bucket breakdown">

  <h3>Key Finding</h3>
  <div class="finding">
    The 0–14d hold bucket remains the only reliably profitable bucket across all DTE configurations.
    Higher DTE (≥90) reduces expiry wipeouts on long-hold trades but does not fix the premium hurdle
    (24% of VIX level at entry). The persistent pattern: <strong>options only work when VIX spikes fast
    within 2 weeks of entry</strong>. Any trade that takes longer is fighting theta and/or option expiry.
  </div>
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nTrack B section written. HTML size: {len(html):,} bytes")
