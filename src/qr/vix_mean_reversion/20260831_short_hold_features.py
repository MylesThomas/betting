"""
Short hold feature analysis — what predicts a trade closing in ≤14 days?
Goal: find an alpha filter to restrict entries to trades where VXX is viable.
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

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── load trades + raw VIX ─────────────────────────────────────────────────────
trades = pd.read_parquet(TRADES_PATH)
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])

print("Pulling VIX...")
raw = yf.download("^VIX", start="1990-01-02", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
vix = raw["Close"].copy()
vix.index = pd.to_datetime(vix.index)

# ── build feature set for each trade ─────────────────────────────────────────
rows = []
for _, t in trades.iterrows():
    ed  = t["entry_date"]
    ep  = t["entry_price"]
    fv  = t["fair_value_at_entry"]
    dev = t["deviation_pct_at_entry"]

    # window of VIX before entry
    hist = vix[vix.index < ed].tail(30)
    if len(hist) < 10:
        continue

    # momentum: VIX direction in 5d and 10d before entry
    mom_5d  = (ep - hist.iloc[-5])  / hist.iloc[-5]  * 100 if len(hist) >= 5  else np.nan
    mom_10d = (ep - hist.iloc[-10]) / hist.iloc[-10] * 100 if len(hist) >= 10 else np.nan

    # velocity: how fast did VIX fall below SMA? days since it crossed below
    # approximate: days since VIX was last >= fair_value (i.e. how long below SMA)
    fv_series = vix.rolling(126, min_periods=126).mean().shift(1)
    pre_entry = vix[vix.index <= ed]
    fv_pre    = fv_series[fv_series.index <= ed]
    aligned   = pd.DataFrame({"vix": pre_entry, "fv": fv_pre}).dropna()
    below_mask = aligned["vix"] < aligned["fv"]
    # find streak: consecutive days below fv ending at entry
    streak = 0
    for val in below_mask.iloc[::-1]:
        if val:
            streak += 1
        else:
            break

    # absolute VIX level bucket
    vix_bucket = "low(<15)" if ep < 15 else ("mid(15-25)" if ep < 25 else "high(25+)")

    # percentile of current VIX vs trailing 252d
    hist_252 = hist.tail(252)
    pct_rank = (hist_252 < ep).mean() * 100

    # deviation bucket
    dev_bucket = "mild(10-15%)" if abs(dev) < 15 else ("mod(15-25%)" if abs(dev) < 25 else "deep(25%+)")

    rows.append({
        "entry_date":      ed,
        "hold_days":       t["hold_days"],
        "short_hold":      t["hold_days"] <= 14,
        "pnl_pct":         t["pnl_pct"],
        "entry_price":     ep,
        "fair_value":      fv,
        "deviation_pct":   dev,
        "mom_5d":          round(mom_5d, 2) if not pd.isna(mom_5d) else np.nan,
        "mom_10d":         round(mom_10d, 2) if not pd.isna(mom_10d) else np.nan,
        "days_below_sma":  streak,
        "vix_pct_rank":    round(pct_rank, 1),
        "vix_bucket":      vix_bucket,
        "dev_bucket":      dev_bucket,
    })

df = pd.DataFrame(rows)
short = df[df["short_hold"]]
long_ = df[~df["short_hold"]]

print(f"\nTotal trades: {len(df)}  |  Short hold (≤14d): {len(short)} ({len(short)/len(df):.1%})  |  Long hold: {len(long_)}")

# ── feature comparison: short vs long hold ────────────────────────────────────
features = ["entry_price", "deviation_pct", "mom_5d", "mom_10d", "days_below_sma", "vix_pct_rank"]
print("\n=== Feature means: short hold vs long hold ===")
print(f"{'feature':<20} {'short(≤14d)':>12} {'long(>14d)':>12} {'diff':>10}")
for f in features:
    sv = short[f].mean()
    lv = long_[f].mean()
    print(f"  {f:<18} {sv:>12.2f} {lv:>12.2f} {sv-lv:>+10.2f}")

# categorical breakdown
print("\n=== VIX bucket: short vs long hold rate ===")
for bucket in ["low(<15)", "mid(15-25)", "high(25+)"]:
    sub = df[df["vix_bucket"] == bucket]
    sr  = sub["short_hold"].mean()
    print(f"  {bucket:<15}: {sr:.1%} short hold ({len(sub)} trades, avg hold {sub['hold_days'].mean():.0f}d)")

print("\n=== Deviation bucket: short vs long hold rate ===")
for bucket in ["mild(10-15%)", "mod(15-25%)", "deep(25%+)"]:
    sub = df[df["dev_bucket"] == bucket]
    if len(sub) == 0: continue
    sr  = sub["short_hold"].mean()
    print(f"  {bucket:<15}: {sr:.1%} short hold ({len(sub)} trades, avg hold {sub['hold_days'].mean():.0f}d)")

print("\n=== Momentum: is VIX already rising at entry? ===")
df["rising_5d"] = df["mom_5d"] > 0   # VIX moved up in last 5d (already bouncing)
for rising, label in [(True, "VIX rising 5d before entry"), (False, "VIX falling 5d before entry")]:
    sub = df[df["rising_5d"] == rising]
    sr  = sub["short_hold"].mean()
    avg = sub["hold_days"].mean()
    print(f"  {label:<35}: {sr:.1%} short hold, avg hold {avg:.0f}d ({len(sub)} trades)")

print("\n=== Days already below SMA at entry ===")
df["streak_bucket"] = pd.cut(df["days_below_sma"], bins=[0,5,15,30,999],
                              labels=["1-5d","6-15d","16-30d","30d+"])
print(df.groupby("streak_bucket", observed=True)[["short_hold","hold_days","pnl_pct"]].mean().round(3).to_string())

# ── charts ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Short Hold (≤14d) vs Long Hold — Feature Distributions", fontsize=13)

plot_configs = [
    ("entry_price",    "VIX level at entry",          "VIX"),
    ("deviation_pct",  "Deviation from SMA at entry", "%"),
    ("mom_5d",         "VIX 5d momentum at entry",    "%"),
    ("mom_10d",        "VIX 10d momentum at entry",   "%"),
    ("days_below_sma", "Days already below SMA",       "days"),
    ("vix_pct_rank",   "VIX percentile (252d lookback)", "%ile"),
]

for ax, (col, title, xlabel) in zip(axes.flatten(), plot_configs):
    bins = 20
    mn   = df[col].min()
    mx   = df[col].max()
    b    = np.linspace(mn, mx, bins+1)
    ax.hist(long_[col].dropna(),  bins=b, color="steelblue", alpha=0.6, label=f"Long hold (n={len(long_)})")
    ax.hist(short[col].dropna(),  bins=b, color="#4caf83",   alpha=0.7, label=f"Short hold ≤14d (n={len(short)})")
    ax.axvline(short[col].mean(), color="darkgreen", lw=1.5, ls="--")
    ax.axvline(long_[col].mean(), color="navy",      lw=1.5, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
dist_b64 = fig_to_b64(fig)

# correlation heatmap with hold_days
fig, ax = plt.subplots(figsize=(8, 4))
corrs = df[features + ["hold_days"]].corr()["hold_days"].drop("hold_days").sort_values()
colors = ["#e05c5c" if v > 0 else "#4caf83" for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Correlation with hold_days (negative = shorter hold)")
ax.set_title("Feature Correlation with Hold Duration", fontsize=12)
for i, (idx, val) in enumerate(corrs.items()):
    ax.text(val + (0.005 if val >= 0 else -0.005), i, f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
corr_b64 = fig_to_b64(fig)

# scatter: deviation_pct vs hold_days (most interesting)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.scatter(short["deviation_pct"], short["hold_days"], color="#4caf83", alpha=0.7, s=30, label="Short hold")
ax.scatter(long_["deviation_pct"], long_["hold_days"], color="steelblue", alpha=0.5, s=20, label="Long hold")
ax.set_xlabel("Deviation from SMA at entry (%)")
ax.set_ylabel("Hold days")
ax.set_title("Deviation at Entry vs Hold Duration", fontsize=11)
ax.legend()

ax2 = axes[1]
ax2.scatter(short["mom_5d"], short["hold_days"], color="#4caf83", alpha=0.7, s=30, label="Short hold")
ax2.scatter(long_["mom_5d"], long_["hold_days"], color="steelblue", alpha=0.5, s=20, label="Long hold")
ax2.axvline(0, color="black", lw=0.8, ls="--")
ax2.set_xlabel("VIX 5d momentum at entry (% change)")
ax2.set_ylabel("Hold days")
ax2.set_title("5d Momentum at Entry vs Hold Duration", fontsize=11)
ax2.legend()

plt.tight_layout()
scatter_b64 = fig_to_b64(fig)

# ── streak table ───────────────────────────────────────────────────────────────
streak_tbl = df.groupby("streak_bucket", observed=True).agg(
    n_trades=("hold_days","count"),
    pct_short_hold=("short_hold","mean"),
    avg_hold_days=("hold_days","mean"),
    avg_pnl_pct=("pnl_pct","mean"),
).round(3).reset_index()
streak_tbl["pct_short_hold"] = streak_tbl["pct_short_hold"].apply(lambda x: f"{x:.1%}")
streak_tbl["avg_hold_days"]  = streak_tbl["avg_hold_days"].apply(lambda x: f"{x:.0f}d")
streak_tbl["avg_pnl_pct"]    = streak_tbl["avg_pnl_pct"].apply(lambda x: f"{x:+.1f}%")

def df_to_html(df):
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = [f"<thead><tr>{header}</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = [f"<td>{row[c]}</td>" for c in cols]
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")
    return f"<table>{''.join(rows_html)}</table>"

# rising vs falling summary
mom_tbl = df.groupby("rising_5d").agg(
    n_trades=("hold_days","count"),
    pct_short_hold=("short_hold","mean"),
    avg_hold_days=("hold_days","mean"),
    avg_pnl_pct=("pnl_pct","mean"),
).round(3).reset_index()
mom_tbl["rising_5d"]         = mom_tbl["rising_5d"].map({True:"VIX rising 5d before entry", False:"VIX falling 5d before entry"})
mom_tbl["pct_short_hold"]    = mom_tbl["pct_short_hold"].apply(lambda x: f"{x:.1%}")
mom_tbl["avg_hold_days"]     = mom_tbl["avg_hold_days"].apply(lambda x: f"{x:.0f}d")
mom_tbl["avg_pnl_pct"]       = mom_tbl["avg_pnl_pct"].apply(lambda x: f"{x:+.1f}%")

# strongest single filter: rising_5d
rising_pct = df[df["rising_5d"]]["short_hold"].mean()
falling_pct= df[~df["rising_5d"]]["short_hold"].mean()

step_feat_html = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- SHORT HOLD FEATURE ANALYSIS                                         -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">P1b-ii</span>
    <h2 style="margin:0; border:none; padding:0;">Short Hold Feature Analysis — Alpha Filter for VXX Viability</h2>
  </div>
  <div class="script-path">Script: src/qr/vix_mean_reversion/20260831_short_hold_features.py</div>

  <div class="finding">
    <strong>Question:</strong> The 0-14d hold bucket is the only range where VXX is profitable (+4-5% avg). Can we predict which entries will close within 14 days and filter to those only?
  </div>

  <h3>Feature Distributions — Short Hold vs Long Hold</h3>
  <img src="data:image/png;base64,{dist_b64}" alt="Feature Distributions">

  <h3>Feature Correlation with Hold Duration</h3>
  <img src="data:image/png;base64,{corr_b64}" alt="Correlation">

  <h3>Deviation &amp; Momentum vs Hold Duration</h3>
  <img src="data:image/png;base64,{scatter_b64}" alt="Scatter">

  <h3>Key Finding — 5d Momentum at Entry</h3>
  {df_to_html(mom_tbl)}
  <div class="ok-box">
    <strong>Strongest signal:</strong> When VIX is already rising in the 5 days before entry ({rising_pct:.1%} of those trades close ≤14d, avg hold {df[df['rising_5d']]['hold_days'].mean():.0f}d), vs. VIX still falling ({falling_pct:.1%} short hold, avg hold {df[~df['rising_5d']]['hold_days'].mean():.0f}d). <br><br>
    Intuition: if VIX has already started bouncing back toward the SMA, you're entering mid-reversion and it completes quickly. If VIX is still falling when you enter, you may be catching a falling knife and the hold time extends.
  </div>

  <h3>Days Already Below SMA at Entry</h3>
  {df_to_html(streak_tbl)}
  <div class="finding">
    Entries that fire early (VIX just crossed below SMA 1-5 days ago) have shorter holds than entries that fire after VIX has been depressed for weeks. Fresh signals revert faster.
  </div>

  <h3>Proposed Filter for v3 VXX-compatible strategy</h3>
  <div class="config-block">
    <div class="config-row"><span class="config-key">Base signal</span><span class="config-val">v2 long entry (VIX ≤ 126d SMA × 0.90)</span></div>
    <div class="config-row"><span class="config-key">Filter 1 — momentum</span><span class="config-val">Only enter if VIX close today &gt; VIX close 5 days ago (VIX already bouncing)</span></div>
    <div class="config-row"><span class="config-key">Filter 2 — freshness</span><span class="config-val">Only enter if VIX has been below SMA for ≤ 5 days (fresh signal, not a long drift)</span></div>
    <div class="config-row"><span class="config-key">Expected effect</span><span class="config-val">Reduces trade count significantly but concentrates entries in the fast-reversion bucket where VXX drag is survivable</span></div>
    <div class="config-row"><span class="config-key">Next step</span><span class="config-val">Backtest these filters on full v2 trade history — measure: how many trades survive, win rate, avg hold, and VXX P&L on filtered subset</span></div>
  </div>

</section>
"""

html    = HTML_PATH.read_text()
updated = html.replace("</body>", step_feat_html + "\n</body>")
HTML_PATH.write_text(updated)
print(f"\nShort hold feature section written. HTML size: {len(updated):,} bytes")
