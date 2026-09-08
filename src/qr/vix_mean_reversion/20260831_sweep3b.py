"""
VIX Mean Reversion — Sweep 3b: FV Window expanded (Step 4b)
Adds 50d and 200d to original [63, 126, 252, 504] sweep.
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

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

print("Pulling VIX data...")
raw = yf.download("^VIX", start="1990-01-02", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
df_full = raw[["Close"]].copy()
df_full.columns = ["close"]
df_full.index = pd.to_datetime(df_full.index)

def run_backtest(df_full, fv_window=252, entry_thr=0.10, exit_thr=0.00, direction="both"):
    df = df_full.copy()
    df["fv"]      = df["close"].rolling(fv_window, min_periods=fv_window).mean().shift(1)
    df["dev_pct"] = (df["close"] - df["fv"]) / df["fv"] * 100
    df_bt = df.dropna(subset=["fv"]).copy()

    trades, position = [], None
    for date, row in df_bt.iterrows():
        close, fv, dev = row["close"], row["fv"], row["dev_pct"]
        if position is not None:
            d = position["direction"]
            should_exit = (d == "long" and dev >= -(exit_thr * 100)) or \
                          (d == "short" and dev <= (exit_thr * 100))
            if should_exit:
                ep  = position["entry_price"]
                pnl_pct = ((close - ep) / ep * 100) if d == "long" else ((ep - close) / ep * 100)
                trades.append({"exit_date": date, "direction": d,
                                "pnl_pct": pnl_pct, "pnl_units": pnl_pct / 100,
                                "hold_days": (date - position["entry_date"]).days})
                position = None
        if position is None:
            el = (direction in ("long", "both")) and dev <= -(entry_thr * 100)
            es = (direction in ("short", "both")) and dev >=  (entry_thr * 100)
            if el:
                position = {"direction": "long",  "entry_date": date, "entry_price": close, "fv_at_entry": fv}
            elif es:
                position = {"direction": "short", "entry_date": date, "entry_price": close, "fv_at_entry": fv}

    if not trades:
        return None
    t   = pd.DataFrame(trades)
    w   = t[t["pnl_units"] > 0]
    l   = t[t["pnl_units"] <= 0]
    cu  = t["pnl_units"].cumsum()
    mdd = (cu - cu.cummax()).min()
    tot = t["pnl_units"].sum()
    # compounded balance
    bal = 1000.0
    for pct in t["pnl_pct"]:
        bal *= (1 + pct / 100)
    years = (t["exit_date"].max() - t["exit_date"].min()).days / 365.25
    cagr  = (bal / 1000) ** (1 / years) - 1 if years > 0 else 0
    return {
        "n_trades":           len(t),
        "win_rate":           len(w) / len(t),
        "total_units":        tot,
        "max_drawdown_units": mdd,
        "calmar":             tot / abs(mdd) if mdd != 0 else np.nan,
        "avg_hold_days":      t["hold_days"].mean(),
        "final_balance":      round(bal, 2),
        "cagr_pct":           round(cagr * 100, 2),
    }

fv_windows = [50, 63, 126, 200, 252, 504]
rows = []
for w in fv_windows:
    r = run_backtest(df_full, fv_window=w, entry_thr=0.10, exit_thr=0.00, direction="both")
    if r:
        rows.append({"fv_window": w, **r})

sweep_df = pd.DataFrame(rows).sort_values("total_units", ascending=False)
print(sweep_df[["fv_window","n_trades","win_rate","total_units","calmar","final_balance","cagr_pct","avg_hold_days"]].to_string(index=False))

# chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = [("total_units", "Total Units"), ("final_balance", "Final Balance ($1k start)"), ("calmar", "Calmar")]
for ax, (col, label) in zip(axes, metrics):
    vals = sweep_df[col].values
    colors = ["#4caf83" if v == max(vals) else "#5b8cf0" for v in vals]
    ax.bar(sweep_df["fv_window"].astype(str), vals, color=colors, alpha=0.85)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("FV window (days)")
    for i, (_, row) in enumerate(sweep_df.iterrows()):
        ax.text(i, vals[i] + max(vals)*0.01, f"{row['n_trades']}t", ha="center", fontsize=8)
plt.suptitle("Fair Value Window Sweep (entry=10%, exit=0%, both)", fontsize=12, y=1.02)
plt.tight_layout()
chart_b64 = fig_to_b64(fig)

def df_to_html(df):
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cls = ""
            if c in ("total_units","calmar","final_balance","cagr_pct"):
                try:
                    fv2 = float(v)
                    cls = "pass" if fv2 == sweep_df[c].max() else ""
                except: pass
            if c == "win_rate":
                v = f"{float(v):.1%}"
            elif c in ("total_units","max_drawdown_units","avg_hold_days"):
                try: v = f"{float(v):.4f}"
                except: pass
            elif c == "calmar":
                try: v = f"{float(v):.2f}"
                except: pass
            elif c == "cagr_pct":
                try: v = f"{float(v):.2f}%"
                except: pass
            cells.append(f'<td class="{cls}">{v}</td>' if cls else f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

best = sweep_df.iloc[0]

step4b_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- STEP 4b — SENSITIVITY SWEEP v2 (FV WINDOW incl. 50d/200d)          -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">STEP 4b</span>
    <h2 style="margin:0; border:none; padding:0;">Sensitivity Sweep v2 — Fair Value Window (incl. 50d &amp; 200d)</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_sweep3b.py</div>

  <div class="finding">
    Added 50d and 200d SMAs to the original sweep. Includes compounded final balance and CAGR alongside flat unit metrics. Best window highlighted in green.
  </div>

  <img src="data:image/png;base64,{chart_b64}" alt="FV Window Sweep">
  {df_to_html(sweep_df)}

  <div class="ok-box">
    <strong>Best window: {best['fv_window']:.0f}d SMA</strong> — {best['n_trades']:.0f} trades, {best['total_units']:+.2f}u, ${best['final_balance']:,.0f} final balance, {best['cagr_pct']:.1f}% CAGR, calmar {best['calmar']:.1f}.
  </div>
</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step4b_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nStep 4b written. HTML size: {len(updated):,} bytes")
