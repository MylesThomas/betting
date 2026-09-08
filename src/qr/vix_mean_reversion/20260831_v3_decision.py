"""
v3 Decision — summarize all three tracks and write final verdict to HTML.
"""
import io, base64
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HTML_PATH = Path("/Users/thomasmyles/dev/betting/knowledge-base/raw/20260830-vix-mean-reversion.html")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# Summary table data (from runs)
tracks = [
    {
        "track":    "v2 Baseline (VIX close)",
        "vehicle":  "VIX index — not investable",
        "n":        229,
        "win_rate": "96.5%",
        "total_u":  "+35.06u",
        "calmar":   "99.14",
        "verdict":  "✅ Strong edge — but VIX not tradeable",
        "pass":     True,
        "blocker":  "VIX not an investable instrument",
    },
    {
        "track":    "Track A — SPY",
        "vehicle":  "SPY long on VIX signal",
        "n":        208,
        "win_rate": "36.5%",
        "total_u":  "+0.68u",
        "calmar":   "0.83",
        "verdict":  "❌ Wrong direction",
        "pass":     False,
        "blocker":  "VIX mean-reversion = VIX goes UP = SPY goes DOWN. Directional mismatch.",
    },
    {
        "track":    "Track B — VIX calls (DTE≥90 ATM)",
        "vehicle":  "VIX ATM calls, DTE ≥ 90d",
        "n":        72,
        "win_rate": "45.8%",
        "total_u":  "−7.28u",
        "calmar":   "−1.03",
        "verdict":  "❌ Premium hurdle too high",
        "pass":     False,
        "blocker":  "ATM calls cost 24% of VIX level. Median VIX move is only 14.5%. 31-60d and 60d+ holds are structural losses.",
    },
    {
        "track":    "Track C — Filtered calls (F1 AND F2)",
        "vehicle":  "VIX calls, entry filter only",
        "n":        9,
        "win_rate": "77.8%",
        "total_u":  "+0.77u",
        "calmar":   "12.93",
        "verdict":  "⚠️ Edge exists — n=9 too thin",
        "pass":     False,
        "blocker":  "Only 9 qualifying trades over 14 years (0.6/yr). Not implementable.",
    },
]

# Vehicles ruled out summary
ruled_out = [
    ("VXX / VXXB",   "Contango destroys returns for long VIX; −71% vs +1,032% VIX on same signals"),
    ("SVXY",         "90%+ single-day drawdown in 2018 (Volmageddon); MAE data confirms 205% adverse"),
    ("SPY long",     "VIX reversion is VIX UP = SPY DOWN; 36.5% win rate, no edge"),
    ("VIX calls",    "24% premium hurdle; only 19% of entries see VIX move that covers premium"),
    ("VIX futures",  "Best structural fit; blocked by data cost (CBOE historical not free)"),
]

# What remains
remaining = [
    {
        "path":   "VIX futures (paid data)",
        "why":    "Direct expression, no premium hurdle, no contango for long. CBOE historical futures back to 2004.",
        "cost":   "CBOE DataShop: ~$50–200 one-time for historical. Real-time: exchange fees (~$100/mo).",
        "status": "Deferred — awaiting data purchase decision",
    },
    {
        "path":   "SPX puts when VIX is low",
        "why":    "Buy SPX puts when VIX ≤ SMA × 0.90 (IV cheap). Profit when VIX spikes (SPX drops). Same signal, different expression. Avoids VIX-futures complexity.",
        "cost":   "OptionsDX SPX data — free for same years; SPX options are deep and liquid (tight spreads).",
        "status": "Ready to test — free data available",
    },
    {
        "path":   "VIX futures ETF (UVIX)",
        "why":    "2× long VIX futures ETF, launched 2022. Limited history but directly tradeable retail instrument.",
        "cost":   "Free via yfinance; only 2022–present data",
        "status": "Too short for backtest; worth tracking live",
    },
]

print("v3 Decision summary:")
for t in tracks:
    print(f"  {t['track']}: {t['verdict']}")

print("\nRemaining paths:")
for r in remaining:
    print(f"  {r['path']}: {r['status']}")

# ── Chart: vehicle comparison ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
labels  = [t["track"].split("—")[0].strip() for t in tracks]
total_u = [35.06, 0.68, -7.28, 0.77]
colors  = ["#5b9bd5", "#e05c5c", "#e05c5c", "#f59e0b"]
bars    = ax.bar(labels, total_u, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, total_u):
    va = "bottom" if val >= 0 else "top"
    off = 0.3 if val >= 0 else -0.3
    ax.text(bar.get_x() + bar.get_width()/2, val + off,
            f"{val:+.1f}u", ha="center", va=va, fontsize=10, fontweight="bold", color="#2d3148")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("v3 Track Comparison — Total Units (v2 baseline not investable)", fontsize=12)
ax.set_ylabel("Total units")
patches = [
    mpatches.Patch(color="#5b9bd5", label="Not investable (VIX close)"),
    mpatches.Patch(color="#e05c5c", label="Investable — fails"),
    mpatches.Patch(color="#f59e0b", label="Investable — passes but n=9"),
]
ax.legend(handles=patches, fontsize=9)
ax.yaxis.grid(True, alpha=0.2)
ax.set_axisbelow(True)
plt.tight_layout()
chart_b64 = fig_to_b64(fig)

# ── HTML ───────────────────────────────────────────────────────────────────────
track_rows = ""
for t in tracks:
    color = "#4caf83" if t["pass"] else ("#f59e0b" if "⚠️" in t["verdict"] else "#e05c5c")
    track_rows += f"""<tr>
      <td><strong>{t['track']}</strong></td>
      <td>{t['vehicle']}</td>
      <td>{t['n']}</td>
      <td>{t['win_rate']}</td>
      <td>{t['total_u']}</td>
      <td>{t['calmar']}</td>
      <td style='color:{color}'>{t['verdict']}</td>
    </tr>
    <tr><td colspan='7' style='font-size:11px;color:#8888aa;padding:2px 8px 8px'>
      {t['blocker']}
    </td></tr>"""

ruled_rows = ""
for veh, reason in ruled_out:
    ruled_rows += f"<tr><td><strong>{veh}</strong></td><td>{reason}</td></tr>"

next_rows = ""
for r in remaining:
    status_color = "#4caf83" if "Ready" in r["status"] else "#f59e0b"
    next_rows += f"""<tr>
      <td><strong>{r['path']}</strong></td>
      <td>{r['why']}</td>
      <td>{r['cost']}</td>
      <td style='color:{status_color}'>{r['status']}</td>
    </tr>"""

html_section = f"""
<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- V3 DECISION                                                        -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<section>
  <div class="step-header">
    <span class="step-badge">V3 DECISION</span>
    <h2 style="margin:0; border:none; padding:0;">v3 — Investable Vehicle Decision</h2>
  </div>

  <h3>All Three Tracks</h3>
  <img src="data:image/png;base64,{chart_b64}" alt="v3 track comparison">
  <table>
    <thead><tr><th>Track</th><th>Vehicle</th><th>N</th><th>Win%</th><th>Total U</th><th>Calmar</th><th>Verdict</th></tr></thead>
    <tbody>{track_rows}</tbody>
  </table>

  <h3>Vehicles Ruled Out</h3>
  <table>
    <thead><tr><th>Instrument</th><th>Reason ruled out</th></tr></thead>
    <tbody>{ruled_rows}</tbody>
  </table>

  <h3>Core Problem</h3>
  <div class="finding">
    The v2 signal has a genuine statistical edge (96.5% win rate, calmar 99.14 on VIX close).
    The problem is not the signal — it is the absence of a cheap, clean instrument that pays off when
    VIX rises from a low level. Every investable instrument introduces a structural cost
    (contango, premium decay, directional mismatch) that erases the edge on the majority of trades.
    The only bucket where instruments work is the 0–14d fast-spike subset — but fast spikes cannot be
    reliably predicted from entry-day features.
  </div>

  <h3>Remaining Paths</h3>
  <table>
    <thead><tr><th>Path</th><th>Why</th><th>Data cost</th><th>Status</th></tr></thead>
    <tbody>{next_rows}</tbody>
  </table>

  <h3>Recommended Next Step</h3>
  <div class="finding">
    <strong>SPX puts when VIX is low</strong> — free OptionsDX data already downloaded covers SPX as well.
    When VIX ≤ SMA × 0.90, buy ATM or slightly OTM SPX puts; exit when VIX ≥ SMA × 0.98.
    This is the same entry/exit signal but expressed through an instrument that profits when SPX falls
    (= when VIX spikes). SPX options are the most liquid options market in the world; spreads are tight.
    Unlike VIX calls (which cost 24% of VIX level), SPX put premiums scale with SPX level and IV —
    and when VIX is low, IV is compressed, making SPX puts cheaper. This is the structural mirror image.
    <br><br>
    If SPX puts fail too: the conclusion is that VIX futures data is required and the decision becomes
    whether to purchase CBOE historical data (~$50–200) to unlock the P1a futures path.
  </div>
</section>
"""

html = HTML_PATH.read_text()
html += html_section
HTML_PATH.write_text(html)
print(f"\nv3 Decision section written. HTML size: {len(html):,} bytes")
