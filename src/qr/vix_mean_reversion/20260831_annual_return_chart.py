"""
Annual % account balance change — v2 backtest. Appends chart to Step 6 section.
"""
import base64
import io
import pandas as pd
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

trades = pd.read_parquet(TRADES_PATH)
trades["exit_date"] = pd.to_datetime(trades["exit_date"])
trades["year"]      = trades["exit_date"].dt.year

# EOY balance = last trade balance in that year
eoy = trades.groupby("year")["balance"].last().sort_index()

# SOY balance = EOY of prior year, or $1000 for first year
soy = eoy.shift(1)
soy.iloc[0] = 1_000.0

annual_pct = (eoy - soy) / soy * 100

print("Annual % account change:")
for yr, pct in annual_pct.items():
    print(f"  {yr}: {pct:+.1f}%  (EOY ${eoy[yr]:,.0f})")

# ── chart ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))

colors = ["#4caf83" if v >= 0 else "#e05c5c" for v in annual_pct.values]
bars = ax.bar(annual_pct.index, annual_pct.values, color=colors, alpha=0.88, width=0.7)

# label each bar
for bar, pct in zip(bars, annual_pct.values):
    va  = "bottom" if pct >= 0 else "top"
    off = 1.5 if pct >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width() / 2, pct + off,
            f"{pct:+.0f}%", ha="center", va=va, fontsize=7.5, fontweight="bold",
            color="#2d3148")

ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Annual % change in account value", fontsize=11)
ax.set_title("v2 — Annual % Account Return (long only · 126d SMA · 10% entry · 2% exit · $1k start, 100% sizing)",
             fontsize=12)
ax.set_xticks(annual_pct.index)
ax.set_xticklabels(annual_pct.index, rotation=45, ha="right", fontsize=9)

# light horizontal grid
ax.yaxis.grid(True, color="gray", alpha=0.25, lw=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
chart_b64 = fig_to_b64(fig)

# ── inject chart after yearly table in Step 6 section ────────────────────────
new_block = f"""
  <h3>Annual % Account Return</h3>
  <img src="data:image/png;base64,{chart_b64}" alt="Annual % Account Return">
"""

# Insert just before the closing </section> tag of Step 6
marker = "<!-- ═══════════════════════════════════════════════════════════════════ -->\n<!-- STEP 6"
# Find last </section> before end of file and insert there — safer: use unique string
target = '<div class="finding">Yearly breadth:'
html   = HTML_PATH.read_text()
html   = html.replace(target, new_block + "\n  " + target, 1)
HTML_PATH.write_text(html)
print(f"\nChart appended to Step 6. HTML size: {len(html):,} bytes")
