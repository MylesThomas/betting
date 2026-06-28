"""
EDA: NFL player sacks props 2025 — calibration tables (Over and Under).

All analysis on 0.5 line only (99.4% of the dataset).
For each implied-prob bin, shows W-L-T record, win/loss/push rates, and units +/-.

Outcome definitions (0.5 line):
  Over bet  → W: sacks >= 1.0  |  L: sacks == 0.0  |  T: sacks == 0.5
  Under bet → W: sacks == 0.0  |  L: sacks >= 1.0  |  T: sacks == 0.5

Units per bet (no de-vig, raw American odds):
  Win  → payout = 100 / |odds| if odds < 0, else odds / 100
  Loss → -1 unit
  Push → 0 units

Output: ~/Downloads/tmp/nfl_sacks_eda_2025.html

Run:
  python nfl_sacks_modeling/scripts/eda_sacks_2025.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUT_HTML = Path.home() / "Downloads" / "tmp" / "nfl_sacks_eda_2025.html"
JOINED   = Path.home() / "Downloads" / "tmp" / "nfl_sacks_joined_2025.parquet"

BIN_WIDTH = 0.10


# ── Helpers ────────────────────────────────────────────────────────────────────

def units_on_win(american_odds: float) -> float:
    if american_odds < 0:
        return 100.0 / abs(american_odds)
    return american_odds / 100.0


def fmt_american(odds: float) -> str:
    return f"+{int(odds)}" if odds >= 0 else str(int(odds))


# ── Load ───────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_parquet(JOINED)
    df = df[
        df["prop_median_price_over"].notna() &
        (df["defense_snaps"] > 0)
    ].copy()

    # Outcome flags
    df["is_over"]  = (df["sacks"] >= 1.0).astype(int)
    df["is_under"] = (df["sacks"] == 0.0).astype(int)
    df["is_push"]  = (df["sacks"] == 0.5).astype(int)

    # Win units per bet (using stored median American odds)
    df["over_win_units"]  = df["prop_median_price_over"].apply(units_on_win)
    df["under_win_units"] = df["prop_median_price_under"].apply(units_on_win)

    WARN_THRESHOLD = 5.0
    over_bad  = df[df["over_win_units"]  > WARN_THRESHOLD]
    under_bad = df[df["under_win_units"] > WARN_THRESHOLD]
    if len(over_bad):
        print(f"\n  WARNING: {len(over_bad)} rows with over_win_units > {WARN_THRESHOLD} (likely bad price):")
        print(over_bad[["week","player","prop_median_price_over","over_win_units"]].to_string(index=False))
    if len(under_bad):
        print(f"\n  WARNING: {len(under_bad)} rows with under_win_units > {WARN_THRESHOLD} (likely bad price):")
        print(under_bad[["week","player","prop_median_price_under","under_win_units"]].to_string(index=False))

    print(f"\nRows (0.5 line, played): {len(df)}")
    print(f"  Over  hits: {df['is_over'].sum()}  ({df['is_over'].mean():.1%})")
    print(f"  Under hits: {df['is_under'].sum()}  ({df['is_under'].mean():.1%})")
    print(f"  Pushes    : {df['is_push'].sum()}  ({df['is_push'].mean():.1%})")
    return df


# ── Build table ────────────────────────────────────────────────────────────────

def build_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    side: 'over' or 'under'
    Bins by the implied prob for that side, computes W-L-T and units +/-.
    """
    prob_col      = f"prop_median_impl_{side}"
    price_col     = f"prop_median_price_{side}"
    win_units_col = f"{side}_win_units"
    win_col       = "is_over"  if side == "over"  else "is_under"
    loss_col      = "is_under" if side == "over"  else "is_over"

    sub = df[df[prob_col].notna()].copy()

    lo   = np.floor(sub[prob_col].min() / BIN_WIDTH) * BIN_WIDTH
    hi   = np.ceil(sub[prob_col].max()  / BIN_WIDTH) * BIN_WIDTH
    bins = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)

    sub["_bin"] = pd.cut(sub[prob_col], bins=bins, include_lowest=True)

    rows = []
    for bin_label, grp in sub.groupby("_bin", observed=True):
        if grp.empty:
            continue
        n     = len(grp)
        w     = grp[win_col].sum()
        l     = grp[loss_col].sum()
        t     = grp["is_push"].sum()
        # units: wins pay win_units, losses cost -1, pushes cost 0
        units = (grp[win_col] * grp[win_units_col]).sum() - grp[loss_col].sum()

        impl_lo = bin_label.left
        impl_hi = bin_label.right
        # representative American odds for the bin midpoint
        mid_impl  = (impl_lo + impl_hi) / 2
        mid_price = grp[price_col].median()

        # Max drawdown (chronological order within bin)
        grp_sorted  = grp.sort_values(["week", "game_id"])
        bet_units   = (grp_sorted[win_col] * grp_sorted[win_units_col]
                       - grp_sorted[loss_col])
        cumul       = bet_units.cumsum()
        peak        = cumul.cummax()
        max_dd      = (peak - cumul).max()

        avg_impl = grp[prob_col].mean()

        rows.append({
            "Implied prob": f"{impl_lo:.0%}–{impl_hi:.0%}",
            "Avg impl%": avg_impl,
            "Odds (median)": fmt_american(mid_price),
            "n":   n,
            "W":   int(w),
            "L":   int(l),
            "T":   int(t),
            "Win%":  w / n,
            "Loss%": l / n,
            "Push%": t / n,
            "Units +/-": units,
            "Max DD": -max_dd,
            "_n":    n,
            "_units": units,
            "_win_rate": w / n,
            "_impl_mid": mid_impl,
        })

    return pd.DataFrame(rows)


# ── HTML table ─────────────────────────────────────────────────────────────────

def to_html_table(df: pd.DataFrame, title: str, subtitle: str) -> str:
    display_cols = ["Implied prob", "Avg impl%", "Odds (median)", "n", "W", "L", "T",
                    "Win%", "Loss%", "Push%", "Units +/-", "Max DD"]

    # Totals row
    n_tot     = df["n"].sum()
    w_tot     = df["W"].sum()
    l_tot     = df["L"].sum()
    t_tot     = df["T"].sum()
    units_tot = df["_units"].sum()

    def fmt_cell(col, val):
        if col in ("Win%", "Loss%", "Push%"):
            return f"{val:.1%}"
        if col == "Avg impl%":
            return "—" if isinstance(val, str) else f"{val:.1%}"
        if col == "Units +/-":
            color = "green" if val > 0 else "red"
            return f'<span style="color:{color};font-weight:bold">{val:+.2f}</span>'
        if col == "Max DD":
            return "—" if isinstance(val, str) else f'<span style="color:red">{val:.2f}</span>'
        if col in ("W", "L", "T", "n"):
            return str(int(val))
        return str(val)

    rows_html = ""
    for _, row in df.iterrows():
        n   = row["_n"]
        u   = row["_units"]
        dim = 'opacity:0.45;' if n < 15 else ''
        bg  = f'background:rgba(44,160,44,{min(abs(u/n)*0.8,0.18):.2f})' if u > 0 \
              else f'background:rgba(214,39,40,{min(abs(u/n)*0.8,0.18):.2f})'
        cells = "".join(
            f"<td>{fmt_cell(c, row[c])}</td>" for c in display_cols
        )
        rows_html += f'<tr style="{bg};{dim}">{cells}</tr>\n'

    # Totals row
    totals = {
        "Implied prob": "TOTAL", "Avg impl%": "—", "Odds (median)": "—",
        "n": n_tot, "W": w_tot, "L": l_tot, "T": t_tot,
        "Win%": w_tot / n_tot, "Loss%": l_tot / n_tot, "Push%": t_tot / n_tot,
        "Units +/-": units_tot, "Max DD": "—",
    }
    tc = "".join(f"<td><b>{fmt_cell(c, totals[c])}</b></td>" for c in display_cols)
    rows_html += f'<tr style="background:#eee;border-top:2px solid #333">{tc}</tr>\n'

    header = "".join(f"<th>{c}</th>" for c in display_cols)

    return f"""
<div style="margin-bottom:56px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">{title}</h2>
  <p style="font-family:monospace;color:#555;margin-top:0;font-size:13px;">{subtitle}</p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">{header}</tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p style="font-family:monospace;font-size:11px;color:#999;margin-top:6px;">
    Rows with n &lt; 15 dimmed. Row shading = units/n magnitude.
  </p>
</div>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = load()

    over_df  = build_table(df, "over")
    under_df = build_table(df, "under")

    over_html = to_html_table(
        over_df,
        title    = "Over calibration — betting the Over (0.5 sacks line)",
        subtitle = "W = sacks ≥ 1.0  |  L = sacks = 0  |  T = sacks = 0.5 (push, money back)",
    )
    under_html = to_html_table(
        under_df,
        title    = "Under calibration — betting the Under (0.5 sacks line)",
        subtitle = "W = sacks = 0  |  L = sacks ≥ 1.0  |  T = sacks = 0.5 (push, money back)",
    )

    n        = len(df)
    over_rt  = df["is_over"].mean()
    under_rt = df["is_under"].mean()
    push_rt  = df["is_push"].mean()

    new_section = f"""
<hr style="margin:60px 0;border:none;border-top:3px solid #333;">
<h1 style="font-family:sans-serif;">After Filtering Out Alt Lines — Prices from 0.5 Line Only</h1>
<div style="font-family:monospace;background:#e8f5e9;padding:14px;border-radius:6px;margin-bottom:32px;font-size:13px;border-left:4px solid #2ca02c;">
  <b>Filter applied:</b> <code>prop_median_price_over/under</code> and <code>prop_median_impl_over/under</code>
  now computed only from bookmaker rows where <code>point == 0.5</code> (standard sacks line).
  Alt lines (0.25, 0.75, 1.5, 2.5, etc.) are excluded from price &amp; probability aggregation —
  they no longer pollute the median with near-zero prices that inflated units to impossible values.<br><br>
  Line = 0.5 only &nbsp;|&nbsp;
  n = {n:,} player-game rows &nbsp;|&nbsp;
  Overall Over hit: {over_rt:.1%} &nbsp;|&nbsp;
  Overall Under hit: {under_rt:.1%} &nbsp;|&nbsp;
  Push (0.5 sacks): {push_rt:.1%}
</div>
{over_html}
{under_html}"""

    # Append new section to the existing HTML (preserves original buggy tables above)
    existing = OUT_HTML.read_text(encoding="utf-8")
    updated  = existing.replace("</body>", new_section + "\n</body>")
    OUT_HTML.write_text(updated, encoding="utf-8")
    print(f"Appended 'after' section → {OUT_HTML}")


if __name__ == "__main__":
    main()
