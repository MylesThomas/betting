"""
SPX Puts Backtest — v4 candidate.

Signal: same v2 (126d SMA, 10% entry, 2% exit, long only on VIX).
Vehicle: buy ATM SPX puts when VIX entry fires; exit when VIX exit fires.

Why this might work where VIX calls didn't:
  - When VIX is low, IV is compressed → SPX puts are cheap (lower premium hurdle)
  - When VIX mean-reverts upward, SPX drops AND IV expands → puts gain from
    both delta (SPX down) and vega (vol expansion). VIX calls only had delta.
  - SPX options are the most liquid options market; tighter spreads than VIX options.

Run: extract SPX 7z files first, then run this script.
Expected SPX file location: ~/Downloads/tmp/spx_options_raw/
"""
import pandas as pd
import numpy as np
import glob, io, base64, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SPX_DIR    = Path.home() / "Downloads/tmp/spx_options_raw"
TRADES_PATH = Path.home() / "Downloads/tmp/vix-mean-reversion_v2_trades.parquet"
HTML_PATH   = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
OUT_TRADES  = Path.home() / "Downloads/tmp/vix-mean-reversion_spx_puts_trades.parquet"

MIN_DTE  = 40   # minimum DTE at entry; try 40 and 90
DELTA_LO = 0.35 # put delta range (puts have negative delta; OptionsDX may store abs)
DELTA_HI = 0.65

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── 1. Extract archives if not already done ────────────────────────────────────
import subprocess, os
spx_archives = sorted(glob.glob(str(Path.home() / "Downloads/spx_eod*.7z")))
if spx_archives and not any(SPX_DIR.glob("*.txt")):
    print(f"Extracting {len(spx_archives)} SPX archives...")
    SPX_DIR.mkdir(parents=True, exist_ok=True)
    for archive in spx_archives:
        subprocess.run(["7z", "x", archive, f"-o{SPX_DIR}", "-y"],
                       capture_output=True)
    print(f"  Done — {len(list(SPX_DIR.glob('*.txt')))} files extracted")
else:
    print(f"SPX raw dir: {len(list(SPX_DIR.glob('*.txt')))} files")

txt_files = sorted(glob.glob(str(SPX_DIR / "*.txt")))
if not txt_files:
    raise FileNotFoundError(
        f"No SPX data files found in {SPX_DIR}.\n"
        "Download SPX option chains from optionsdx.com (free, same as VIX),\n"
        "then re-run this script."
    )

# ── 2. Load SPX options ────────────────────────────────────────────────────────
print(f"Loading {len(txt_files)} SPX files...")
frames = []
for f in txt_files:
    try:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    except Exception as e:
        print(f"  skip {f}: {e}")

all_opts = pd.concat(frames, ignore_index=True)
all_opts.columns = [c.strip(" []") for c in all_opts.columns]

all_opts["QUOTE_DATE"]  = pd.to_datetime(all_opts["QUOTE_DATE"].str.strip())
all_opts["EXPIRE_DATE"] = pd.to_datetime(all_opts["EXPIRE_DATE"].str.strip())

# SPX puts: use P_ columns
for col in ["UNDERLYING_LAST", "DTE", "P_BID", "P_ASK", "STRIKE", "P_DELTA", "P_IV", "P_THETA"]:
    if col in all_opts.columns:
        all_opts[col] = pd.to_numeric(all_opts[col], errors="coerce")

# Put mid price
all_opts["p_mid"] = (all_opts["P_BID"] + all_opts["P_ASK"]) / 2

# P_DELTA for puts is negative in standard convention; OptionsDX may store absolute value
# Check which convention is used:
sample_delta = all_opts["P_DELTA"].dropna().head(100)
print(f"  P_DELTA sample (first 5 non-null): {sample_delta.values[:5]}")
# If mostly negative → standard; if mostly positive → absolute value stored
delta_positive = (sample_delta > 0).mean()
print(f"  P_DELTA positive fraction: {delta_positive:.0%}")
# Use absolute value for filtering regardless
all_opts["p_delta_abs"] = all_opts["P_DELTA"].abs()

print(f"  Loaded {len(all_opts):,} rows | dates: {all_opts['QUOTE_DATE'].nunique():,}")
print(f"  Date range: {all_opts['QUOTE_DATE'].min().date()} – {all_opts['QUOTE_DATE'].max().date()}")
print(f"  Sample UNDERLYING_LAST (SPX): {all_opts['UNDERLYING_LAST'].dropna().iloc[:3].values}")
print(f"  Strike range: {all_opts['STRIKE'].min():.0f} – {all_opts['STRIKE'].max():.0f}")
print(f"  DTE range: {all_opts['DTE'].min():.0f} – {all_opts['DTE'].max():.0f}")

opts_by_date = {d: grp for d, grp in all_opts.groupby("QUOTE_DATE")}

# ── 3. Load v2 trades ─────────────────────────────────────────────────────────
v2 = pd.read_parquet(TRADES_PATH)
v2 = v2[v2["direction"] == "long"].copy()
v2["entry_date"] = pd.to_datetime(v2["entry_date"])
v2["exit_date"]  = pd.to_datetime(v2["exit_date"])
print(f"\nv2 long trades: {len(v2)} | in SPX window: {v2['entry_date'].isin(opts_by_date).sum()}")

# ── 4. Run sweep: DTE 40 and DTE 90 ──────────────────────────────────────────
configs = [
    {"label": "DTE≥40 ATM put",       "min_dte": 40, "dlo": 0.40, "dhi": 0.65},
    {"label": "DTE≥90 ATM put",       "min_dte": 90, "dlo": 0.40, "dhi": 0.65},
    {"label": "DTE≥40 OTM put d~0.25","min_dte": 40, "dlo": 0.15, "dhi": 0.35},
    {"label": "DTE≥90 OTM put d~0.25","min_dte": 90, "dlo": 0.15, "dhi": 0.35},
]

def run_config(cfg):
    min_dte, dlo, dhi = cfg["min_dte"], cfg["dlo"], cfg["dhi"]
    results = []
    skipped = {"no_date": 0, "no_contract": 0, "no_exit": 0}

    for _, trade in v2.iterrows():
        entry_dt = trade["entry_date"]
        exit_dt  = trade["exit_date"]
        spx_ref  = None  # will get from options data

        if entry_dt not in opts_by_date:
            skipped["no_date"] += 1
            continue

        day = opts_by_date[entry_dt]
        cands = day[
            (day["DTE"] >= min_dte) &
            (day["p_mid"] > 0) &
            (day["P_BID"] > 0) &
            (day["p_delta_abs"] >= dlo) &
            (day["p_delta_abs"] <= dhi)
        ].copy()

        if cands.empty:
            skipped["no_contract"] += 1
            continue

        # Nearest qualifying expiry
        exp   = cands["EXPIRE_DATE"].min()
        cands = cands[cands["EXPIRE_DATE"] == exp]

        # ATM: closest to underlying; OTM (dlo < 0.35): pick lowest abs delta (most OTM)
        if dlo >= 0.35:
            spx_price = cands["UNDERLYING_LAST"].iloc[0]
            cands["sd"] = (cands["STRIKE"] - spx_price).abs()
            best = cands.loc[cands["sd"].idxmin()]
        else:
            best = cands.loc[cands["p_delta_abs"].idxmin()]

        entry_mid    = best["p_mid"]
        entry_strike = best["STRIKE"]
        entry_expiry = best["EXPIRE_DATE"]
        entry_dte    = best["DTE"]
        entry_delta  = best["p_delta_abs"]
        entry_spx    = best["UNDERLYING_LAST"]
        premium_pct  = entry_mid / entry_spx * 100  # premium as % of SPX

        # Exit value
        outcome  = "expired_worthless"
        exit_mid = 0.0
        exit_spx = None

        if exit_dt in opts_by_date:
            exit_day = opts_by_date[exit_dt]
            same = exit_day[
                (exit_day["EXPIRE_DATE"] == entry_expiry) &
                (exit_day["STRIKE"] == entry_strike)
            ]
            if not same.empty:
                m = (same.iloc[0]["P_BID"] + same.iloc[0]["P_ASK"]) / 2
                if pd.notna(m) and m > 0:
                    exit_mid = m
                    exit_spx = same.iloc[0]["UNDERLYING_LAST"]
                    outcome  = "closed"
            if outcome != "closed" and entry_expiry >= exit_dt:
                skipped["no_exit"] += 1
                continue
        elif exit_dt > entry_expiry:
            outcome = "expired_worthless"
        else:
            skipped["no_exit"] += 1
            continue

        pnl_pct   = (exit_mid - entry_mid) / entry_mid * 100
        pnl_units = pnl_pct / 100
        spx_move  = (exit_spx - entry_spx) / entry_spx * 100 if exit_spx else None

        results.append({
            "entry_date":    entry_dt,
            "exit_date":     exit_dt,
            "vix_entry":     trade["entry_price"],
            "vix_pnl_pct":   trade["pnl_pct"],
            "entry_spx":     entry_spx,
            "exit_spx":      exit_spx,
            "spx_move_pct":  spx_move,
            "strike":        entry_strike,
            "entry_expiry":  entry_expiry,
            "entry_dte":     entry_dte,
            "entry_delta":   entry_delta,
            "premium_pct":   premium_pct,
            "entry_mid":     entry_mid,
            "exit_mid":      exit_mid,
            "hold_days":     trade["hold_days"],
            "outcome":       outcome,
            "pnl_pct":       pnl_pct,
            "pnl_units":     pnl_units,
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

all_results  = {}
summary_rows = []

for cfg in configs:
    print(f"\nRunning: {cfg['label']}")
    res, skipped = run_config(cfg)
    if res is None:
        print(f"  No trades")
        continue

    all_results[cfg["label"]] = res
    n       = len(res)
    wr      = (res["pnl_pct"] > 0).mean()
    total_u = res["pnl_units"].sum()
    avg_w   = res.loc[res["pnl_pct"] > 0, "pnl_pct"].mean() if wr > 0 else 0
    avg_l   = res.loc[res["pnl_pct"] <= 0, "pnl_pct"].mean() if wr < 1 else 0
    exp_pct = (res["outcome"] == "expired_worthless").mean()
    med_prem= res["premium_pct"].median()

    peak   = res["cumulative_units"].cummax()
    max_dd = (res["cumulative_units"] - peak).min()
    calmar = total_u / abs(max_dd) if max_dd != 0 else np.inf
    years  = res["exit_date"].dt.year.max() - res["entry_date"].dt.year.min() + 1
    final  = res["balance"].iloc[-1]
    cagr   = (final / 1_000) ** (1 / years) - 1

    # Hold bucket
    res["hold_bucket"] = pd.cut(res["hold_days"], bins=[0,14,30,60,999],
                                 labels=["0-14d","15-30d","31-60d","60d+"])
    bucket = res.groupby("hold_bucket", observed=True).agg(
        n=("pnl_pct","count"),
        wr=("pnl_pct", lambda x: (x>0).mean()),
        avg=("pnl_pct","mean"),
        exp=("outcome", lambda x: (x=="expired_worthless").mean()),
    )

    print(f"  n={n} wr={wr:.1%} total_u={total_u:+.2f} calmar={calmar:.2f} "
          f"expired={exp_pct:.0%} med_premium={med_prem:.2f}% cagr={cagr:.1%}")
    print(f"  skipped: {skipped}")
    print(f"  hold buckets:\n{bucket.to_string()}")

    summary_rows.append({
        "config":    cfg["label"],
        "n":         n,
        "win_rate":  wr,
        "total_u":   total_u,
        "avg_win":   avg_w,
        "avg_loss":  avg_l,
        "max_dd":    max_dd,
        "calmar":    calmar,
        "exp_pct":   exp_pct,
        "med_prem":  med_prem,
        "cagr":      cagr,
        "final_bal": final,
    })

summary = pd.DataFrame(summary_rows).sort_values("total_u", ascending=False)
print(f"\n=== SPX Puts Summary ===")
print(summary[["config","n","win_rate","total_u","calmar","exp_pct","med_prem","cagr"]].to_string(index=False))

# ── 5. Charts ──────────────────────────────────────────────────────────────────
if all_results:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Equity curves
    ax = axes[0]
    palette = ["#4caf83","#5b9bd5","#f59e0b","#a78bfa"]
    for i, (label, res) in enumerate(all_results.items()):
        ax.plot(res["entry_date"], res["cumulative_units"],
                color=palette[i % len(palette)], lw=1.8, alpha=0.9, label=label)
    ax.axhline(0, color="gray", lw=0.6)
    ax.legend(fontsize=8)
    ax.set_title("SPX Puts — Equity Curves", fontsize=11)
    ax.set_ylabel("Cumulative units")
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    # SPX move vs put P&L for best config
    best_label = summary.iloc[0]["config"]
    best_res   = all_results.get(best_label)
    ax = axes[1]
    if best_res is not None and "spx_move_pct" in best_res.columns:
        closed = best_res[best_res["outcome"] == "closed"].dropna(subset=["spx_move_pct"])
        colors = ["#4caf83" if v > 0 else "#e05c5c" for v in closed["pnl_pct"]]
        ax.scatter(closed["spx_move_pct"], closed["pnl_pct"],
                   c=colors, s=40, alpha=0.75)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SPX % move (entry → exit)")
        ax.set_ylabel("Put P&L % (of premium)")
        ax.set_title(f"SPX Put P&L vs SPX Move\n({best_label})", fontsize=10)
        ax.yaxis.grid(True, alpha=0.2)

    plt.tight_layout()
    chart_b64 = fig_to_b64(fig)
else:
    chart_b64 = ""

# ── 6. HTML ────────────────────────────────────────────────────────────────────
sweep_rows = ""
for _, row in summary.iterrows():
    u_c = "#4caf83" if row["total_u"] > 0 else "#e05c5c"
    c_c = "#4caf83" if row["calmar"] > 1 else "#e05c5c"
    sweep_rows += f"""<tr>
      <td><strong>{row['config']}</strong></td>
      <td>{int(row['n'])}</td>
      <td>{row['win_rate']:.1%}</td>
      <td style='color:{u_c}'>{row['total_u']:+.2f}u</td>
      <td>{row['avg_win']:+.1f}%</td>
      <td>{row['avg_loss']:+.1f}%</td>
      <td style='color:{c_c}'>{row['calmar']:.2f}</td>
      <td>{row['exp_pct']:.0%}</td>
      <td>{row['med_prem']:.2f}%</td>
      <td>{row['cagr']:.1%}</td>
      <td>${row['final_bal']:,.0f}</td>
    </tr>"""

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- SPX PUTS — v4 candidate                                           -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">SPX PUTS</span>
    <h2 style="margin:0; border:none; padding:0;">SPX Puts Backtest — VIX Signal, SPX Put Execution</h2>
  </div>

  <div class="config-block">
    <div class="config-row"><span class="config-key">Signal</span><span class="config-val">v2 (126d SMA, 10% entry, 2% exit, long only on VIX)</span></div>
    <div class="config-row"><span class="config-key">Vehicle</span><span class="config-val">Buy ATM or OTM SPX put on VIX entry signal; sell on VIX exit signal</span></div>
    <div class="config-row"><span class="config-key">Rationale</span><span class="config-val">VIX UP = SPX DOWN; puts profit from both delta (SPX drop) and vega (IV expansion). Low VIX at entry = cheap puts.</span></div>
    <div class="config-row"><span class="config-key">Data</span><span class="config-val">OptionsDX free EOD SPX option chains (2010–2023)</span></div>
    <div class="config-row"><span class="config-key">Configs</span><span class="config-val">DTE ≥ 40 and ≥ 90; ATM (delta 0.40-0.65) and OTM (delta 0.15-0.35)</span></div>
  </div>

  <h3>Sweep Results (sorted by total units)</h3>
  <table>
    <thead><tr>
      <th>Config</th><th>N</th><th>Win%</th><th>Total U</th>
      <th>Avg win</th><th>Avg loss</th><th>Calmar</th><th>Expired</th>
      <th>Med premium %</th><th>CAGR</th><th>$1k final</th>
    </tr></thead>
    <tbody>{sweep_rows}</tbody>
  </table>

  {"<h3>Equity Curves & Trade Scatter</h3><img src='data:image/png;base64," + chart_b64 + "' alt='SPX puts'>" if chart_b64 else ""}
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nSPX puts section written. HTML size: {len(html):,} bytes")
