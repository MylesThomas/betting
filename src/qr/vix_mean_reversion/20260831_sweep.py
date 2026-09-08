"""
VIX Mean Reversion — Sensitivity Sweep (Step 4)

Sweep 1: entry_threshold × direction (24 rows)
Sweep 2: exit_threshold at best entry threshold (per direction)
Sweep 3: fair_value_window at defaults
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

HTML_PATH = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")
SWEEP_PATH = Path("/Users/thomasmyles/Downloads/tmp/vix-mean-reversion_sweep.parquet")
SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── data ──────────────────────────────────────────────────────────────────────
print("Pulling VIX data...")
raw = yf.download("^VIX", start="1990-01-02", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
df_full = raw[["Close"]].copy()
df_full.columns = ["close"]
df_full.index = pd.to_datetime(df_full.index)

# ── backtest function ─────────────────────────────────────────────────────────
def run_backtest(df_full, fv_window=252, entry_thr=0.10, exit_thr=0.00, direction="both"):
    df = df_full.copy()
    df["fv"]      = df["close"].rolling(fv_window, min_periods=fv_window).mean().shift(1)
    df["dev_pct"] = (df["close"] - df["fv"]) / df["fv"] * 100
    df_bt = df.dropna(subset=["fv"]).copy()

    trades = []
    position = None
    for date, row in df_bt.iterrows():
        close, fv, dev = row["close"], row["fv"], row["dev_pct"]
        if position is not None:
            d = position["direction"]
            should_exit = (d == "long" and dev >= -(exit_thr * 100)) or \
                          (d == "short" and dev <= (exit_thr * 100))
            if should_exit:
                ep  = position["entry_price"]
                pnl = ((close - ep) / ep * 100) if d == "long" else ((ep - close) / ep * 100)
                trades.append({
                    "exit_date":  date,
                    "direction":  d,
                    "pnl_units":  pnl / 100,
                    "hold_days":  (date - position["entry_date"]).days,
                    "dev_at_entry": (ep - position["fv_at_entry"]) / position["fv_at_entry"] * 100,
                })
                position = None
        if position is None:
            el = (direction in ("long",  "both")) and dev <= -(entry_thr * 100)
            es = (direction in ("short", "both")) and dev >=  (entry_thr * 100)
            if el:
                position = {"direction": "long",  "entry_date": date, "entry_price": close, "fv_at_entry": fv}
            elif es:
                position = {"direction": "short", "entry_date": date, "entry_price": close, "fv_at_entry": fv}

    if not trades:
        return None
    t = pd.DataFrame(trades)
    wins   = t[t["pnl_units"] > 0]
    losses = t[t["pnl_units"] <= 0]
    cu     = t["pnl_units"].cumsum()
    mdd    = (cu - cu.cummax()).min()
    total  = t["pnl_units"].sum()
    return {
        "n_trades":          len(t),
        "win_rate":          len(wins) / len(t),
        "total_units":       total,
        "roi_pct":           total * 100,
        "max_drawdown_units":mdd,
        "calmar":            total / abs(mdd) if mdd != 0 else np.nan,
        "avg_hold_days":     t["hold_days"].mean(),
        "profit_factor":     wins["pnl_units"].sum() / abs(losses["pnl_units"].sum()) if len(losses) else np.nan,
    }

# ── Sweep 1: entry_threshold × direction ─────────────────────────────────────
print("Sweep 1: entry threshold × direction...")
entry_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
directions       = ["long", "short", "both"]
sweep1_rows = []
for et in entry_thresholds:
    for d in directions:
        r = run_backtest(df_full, fv_window=252, entry_thr=et, exit_thr=0.00, direction=d)
        if r:
            sweep1_rows.append({"entry_threshold": et, "direction": d, **r})

sweep1 = pd.DataFrame(sweep1_rows).sort_values("total_units", ascending=False)
print(sweep1[["entry_threshold","direction","n_trades","win_rate","total_units","calmar","avg_hold_days"]].to_string(index=False))

# ── Sweep 2: exit_threshold (at 10% entry, per direction) ─────────────────────
print("\nSweep 2: exit threshold...")
exit_thresholds = [0.00, 0.02, 0.05, 0.08]
sweep2_rows = []
for xt in exit_thresholds:
    for d in directions:
        r = run_backtest(df_full, fv_window=252, entry_thr=0.10, exit_thr=xt, direction=d)
        if r:
            sweep2_rows.append({"exit_threshold": xt, "direction": d, **r})

sweep2 = pd.DataFrame(sweep2_rows).sort_values("total_units", ascending=False)
print(sweep2[["exit_threshold","direction","n_trades","win_rate","total_units","calmar","avg_hold_days"]].to_string(index=False))

# ── Sweep 3: fair value window ─────────────────────────────────────────────────
print("\nSweep 3: fair value window...")
fv_windows = [63, 126, 252, 504]
sweep3_rows = []
for w in fv_windows:
    r = run_backtest(df_full, fv_window=w, entry_thr=0.10, exit_thr=0.00, direction="both")
    if r:
        sweep3_rows.append({"fv_window": w, **r})

sweep3 = pd.DataFrame(sweep3_rows).sort_values("total_units", ascending=False)
print(sweep3[["fv_window","n_trades","win_rate","total_units","calmar","avg_hold_days"]].to_string(index=False))

# ── save ────────────────────────────────────────────────────────────────────────
sweep1.to_parquet(SWEEP_PATH)

# ── charts ────────────────────────────────────────────────────────────────────

# Sweep 1 heatmap: total_units by (threshold, direction)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, d, color in zip(axes, ["long","short","both"], ["steelblue","tomato","black"]):
    sub = sweep1[sweep1["direction"]==d].sort_values("entry_threshold")
    ax.bar(sub["entry_threshold"].astype(str), sub["total_units"], color=color, alpha=0.8)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title(f"Entry Threshold vs Total Units — {d.capitalize()}", fontsize=10)
    ax.set_xlabel("Entry threshold")
    ax.set_ylabel("Total units")
    for i, (_, row) in enumerate(sub.iterrows()):
        flag = " ⚠" if row["n_trades"] < 20 else ""
        ax.text(i, row["total_units"] + 0.2, f"{row['n_trades']}t{flag}", ha="center", fontsize=7)
plt.tight_layout()
sweep1_b64 = fig_to_b64(fig)

# Sweep 2 line: exit_threshold vs total_units per direction
fig, ax = plt.subplots(figsize=(10, 4))
for d, color in [("long","steelblue"),("short","tomato"),("both","black")]:
    sub = sweep2[sweep2["direction"]==d].sort_values("exit_threshold")
    ax.plot(sub["exit_threshold"]*100, sub["total_units"], marker="o", color=color, label=d.capitalize(), lw=1.5)
ax.set_xlabel("Exit threshold (%)")
ax.set_ylabel("Total units")
ax.set_title("Exit Threshold Sweep (entry=10%, 252d SMA)", fontsize=12)
ax.legend()
ax.axvline(0, color="gray", lw=0.7, ls="--", label="Default (0%)")
plt.tight_layout()
sweep2_b64 = fig_to_b64(fig)

# Sweep 3 bar: fv_window vs total_units
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(sweep3["fv_window"].astype(str), sweep3["total_units"], color="purple", alpha=0.8)
ax.axhline(0, color="gray", lw=0.7)
ax.set_xlabel("FV window (days)")
ax.set_ylabel("Total units")
ax.set_title("Fair Value Window Sweep (entry=10%, exit=0%, both)", fontsize=12)
for i, (_, row) in enumerate(sweep3.iterrows()):
    ax.text(i, row["total_units"] + 0.2, f"{row['n_trades']}t", ha="center", fontsize=8)
plt.tight_layout()
sweep3_b64 = fig_to_b64(fig)

# ── HTML helpers ───────────────────────────────────────────────────────────────
def df_to_html(df, fmt_cols=None):
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cls = ""
            if c in ("total_units", "calmar"):
                try:
                    fv = float(v)
                    cls = "pass" if fv > 0 else "fail"
                except: pass
            if c == "direction":
                cls = {"long":"pass","short":"fail","both":""}.get(str(v),"")
            if c == "n_trades" and int(v) < 20:
                cls = "flag"
            if c in ("win_rate",):
                v = f"{float(v):.1%}"
            elif c in ("total_units","calmar","avg_hold_days","max_drawdown_units","profit_factor","roi_pct"):
                try: v = f"{float(v):.4f}"
                except: pass
            elif c in ("entry_threshold","exit_threshold"):
                try: v = f"{float(v):.0%}"
                except: pass
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

sweep1_table = df_to_html(sweep1)
sweep2_table = df_to_html(sweep2)
sweep3_table = df_to_html(sweep3)

# Best from sweep 1 (long only)
best_long = sweep1[sweep1["direction"]=="long"].iloc[0]
best_short = sweep1[sweep1["direction"]=="short"].iloc[0]
best_both  = sweep1[sweep1["direction"]=="both"].iloc[0]

step4_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 4 — SENSITIVITY SWEEP                                           -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 4</span>
    <h2 style="margin:0; border:none; padding:0;">Sensitivity Sweep</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_sweep.py</div>

  <h3>Sweep 1 — Entry Threshold × Direction (sorted by total_units desc)</h3>
  <img src="data:image/png;base64,{sweep1_b64}" alt="Sweep 1 Chart">
  {sweep1_table}
  <div class="finding">
    <strong>Best long:</strong> entry={best_long['entry_threshold']:.0%}, {best_long['n_trades']:.0f} trades, {best_long['total_units']:+.4f}u, calmar {best_long['calmar']:.2f} &nbsp;|&nbsp;
    <strong>Best short:</strong> entry={best_short['entry_threshold']:.0%}, {best_short['n_trades']:.0f} trades, {best_short['total_units']:+.4f}u &nbsp;|&nbsp;
    <strong>Best both:</strong> entry={best_both['entry_threshold']:.0%}, {best_both['n_trades']:.0f} trades, {best_both['total_units']:+.4f}u
  </div>

  <h3>Sweep 2 — Exit Threshold (entry=10%, 252d SMA)</h3>
  <img src="data:image/png;base64,{sweep2_b64}" alt="Sweep 2 Chart">
  {sweep2_table}

  <h3>Sweep 3 — Fair Value Window (entry=10%, exit=0%, both)</h3>
  <img src="data:image/png;base64,{sweep3_b64}" alt="Sweep 3 Chart">
  {sweep3_table}

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step4_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 4 section written to HTML. Size: {len(updated):,} bytes")
