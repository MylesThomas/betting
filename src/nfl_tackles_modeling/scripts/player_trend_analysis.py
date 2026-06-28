"""
Player trend analysis — deep dive into top N players by bet count.

Steps covered:
  1. Identify top N players by bet count under the production config
  2. Build per-player week-over-week timeline (line, edge, streak, running units)
  3. Q1: Line drift — after a win/loss, how does the line move next game?

Output: ~/Downloads/nfl_player_trend_analysis.html

Run:
  python src/nfl_tackles_modeling/scripts/player_trend_analysis.py
  python src/nfl_tackles_modeling/scripts/player_trend_analysis.py --n-players 5
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_labeled.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_tackles_artifacts"
OUT_HTML     = Path.home() / "Downloads" / "nfl_player_trend_analysis.html"

TARGET = "tackles_combined"

# ── Production config (mirrors infer.py) ──────────────────────────────────────
DIRECTION      = "UNDER"
EDGE_THRESHOLD = 0.05
LINE_MIN       = 4.5
LINE_MAX       = 9.5
MIN_BOOKS      = 1
N_PLAYERS      = 10

# ── Model constants ───────────────────────────────────────────────────────────
HYBRID_NEGBIN_THRESHOLD = 4.5
N_BOOT = 10_000
JUICE  = 110
WIN_PAYOUT = 100 / JUICE
RNG    = np.random.default_rng(42)

POS_GROUP_MAP = {
    "LB": "LB", "CB": "CB", "DB": "CB",
    "S": "S",  "FS": "S",  "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}
BEST_FEATS = [
    "offered_line", "game_total", "proj_opp_score", "tackle_rate_L16",
    "pos_LB", "pos_CB", "pos_S", "pos_DL", "consensus_over_prob",
]
DROP_POSITIONS = ["WR", "FB"]


# ── Inference (self-contained, same logic as infer.py) ────────────────────────

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position_group"] = df["position"].map(POS_GROUP_MAP)
    for g in ["LB", "CB", "S", "DL"]:
        df[f"pos_{g}"] = (df["position_group"] == g).astype(int)
    over_cols  = [c for c in df.columns if c.endswith("_over_price")]
    under_cols = [c for c in df.columns if c.endswith("_under_price")]
    if over_cols and under_cols:
        def to_imp(col):
            s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            return np.where(np.isnan(s), np.nan,
                            np.where(s < 0, -s / (-s + 100), 100 / (s + 100)))
        om  = np.column_stack([to_imp(c) for c in over_cols])
        um  = np.column_stack([to_imp(c) for c in under_cols])
        tot = om + um
        df["consensus_over_prob"] = np.nanmean(np.where(tot > 0, om / tot, np.nan), axis=1)
        over_books  = {c.replace("_over_price",  "") for c in over_cols}
        under_books = {c.replace("_under_price", "") for c in under_cols}
        paired = over_books & under_books
        df["n_books"] = sum(
            (df[f"{b}_over_price"].notna() & df[f"{b}_under_price"].notna()).astype(int)
            for b in paired
        )
        # Consensus under price: avg implied prob across books → back to American odds
        under_imp = np.column_stack([to_imp(f"{b}_under_price") for b in paired])
        avg_imp   = np.nanmean(under_imp, axis=1)
        # implied prob → American odds: p≥0.5 → negative, p<0.5 → positive
        american  = np.where(
            np.isnan(avg_imp), np.nan,
            np.where(avg_imp >= 0.5,
                     -avg_imp / (1 - avg_imp) * 100,
                     (1 - avg_imp) / avg_imp * 100)
        )
        df["consensus_under_price"] = pd.array(np.round(american), dtype="Int64")
    return df


def run_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    ols, residuals = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]
    result = df.copy()
    mask = result[BEST_FEATS + ["consensus_over_prob"]].notna().all(axis=1)
    idx  = result.index[mask]
    X    = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)

    mu_c  = np.clip(nb_mu, 1e-3, None)
    n_nb  = 1.0 / nb_alpha
    p_nb  = nbinom.sf(np.floor(line).astype(int), n=n_nb, p=n_nb / (n_nb + mu_c))
    samp  = RNG.choice(residuals, size=(len(ols_pred), N_BOOT))
    p_bt  = ((ols_pred[:, None] + samp) > line[:, None]).mean(axis=1)
    p_hyb = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)

    p_mkt = result.loc[idx, "consensus_over_prob"].to_numpy(dtype=float)
    edge  = p_hyb - p_mkt
    rec   = np.select([edge > EDGE_THRESHOLD, edge < -EDGE_THRESHOLD],
                      ["OVER", "UNDER"], default="PASS")

    result.loc[idx, "ols_pred"]       = np.round(ols_pred, 3)
    result.loc[idx, "p_hybrid"]       = np.round(p_hyb, 4)
    result.loc[idx, "p_market"]       = np.round(p_mkt, 4)
    result.loc[idx, "edge"]           = np.round(edge, 4)
    result.loc[idx, "recommendation"] = rec

    result["actual_over"] = (result[TARGET] > result["offered_line"]).astype(float)
    result["bet_correct"] = np.where(
        result["recommendation"] == "OVER",  result["actual_over"],
        np.where(result["recommendation"] == "UNDER", 1 - result["actual_over"], np.nan),
    )
    return result


def filter_bets(results: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (results["recommendation"] == DIRECTION) &
        (results["offered_line"] >= LINE_MIN) &
        (results["offered_line"] <= LINE_MAX) &
        (results["edge"].abs() >= EDGE_THRESHOLD) &
        results["ols_pred"].notna()
    )
    if "n_books" in results.columns:
        mask &= results["n_books"] >= MIN_BOOKS
    return results[mask].copy()


# ── Timeline builder ──────────────────────────────────────────────────────────

def build_timeline(player_results: pd.DataFrame, bet_keys: set) -> pd.DataFrame:
    """
    Build the week-over-week timeline for one player.
    bet_keys: set of (season, week) tuples where we placed a bet.
    """
    df = player_results.sort_values(["season", "week"]).reset_index(drop=True)
    df["is_bet"] = df.apply(lambda r: (r["season"], r["week"]) in bet_keys, axis=1)

    # Line delta vs previous game appearance
    df["delta_line"] = df["offered_line"].diff().round(2)

    # Streak: consecutive bet results; resets on no-bet week
    streak = 0
    streaks = []
    for _, row in df.iterrows():
        if not row["is_bet"]:
            streak = 0
        else:
            correct = row.get("bet_correct", np.nan)
            if pd.isna(correct):
                streak = 0
            elif correct == 1:
                streak = max(streak, 0) + 1
            else:
                streak = min(streak, 0) - 1
        streaks.append(streak if row["is_bet"] else np.nan)
    df["streak"] = streaks

    # Running units (cumulative P&L, bet rows only)
    units = 0
    running = []
    for _, row in df.iterrows():
        if row["is_bet"] and not pd.isna(row.get("bet_correct", np.nan)):
            units += WIN_PAYOUT if row["bet_correct"] == 1 else -1.0
        running.append(round(units, 2) if row["is_bet"] else np.nan)
    df["running_units"] = running

    return df


# ── Q1: Line drift analysis ───────────────────────────────────────────────────

def line_drift_analysis(timelines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each bet row, find the player's next game and measure line change.
    Returns a DataFrame of bet→next-game pairs with delta_next_line.
    """
    records = []
    for player, df in timelines.items():
        for i, row in df[df["is_bet"]].iterrows():
            later = df[(df["season"] > row["season"]) |
                       ((df["season"] == row["season"]) & (df["week"] > row["week"]))]
            if later.empty:
                continue
            nxt = later.iloc[0]
            p_under_cur  = 1 - float(row["p_market"])   if pd.notna(row.get("p_market"))   else np.nan
            p_under_nxt  = 1 - float(nxt["p_market"])   if pd.notna(nxt.get("p_market"))   else np.nan
            delta_p_under = round(p_under_nxt - p_under_cur, 4) if not (np.isnan(p_under_cur) or np.isnan(p_under_nxt)) else np.nan
            records.append({
                "player":           player,
                "season":           int(row["season"]),
                "week":             int(row["week"]),
                "line":             row["offered_line"],
                "odds":             row.get("consensus_under_price", pd.NA),
                "p_under":          round(p_under_cur, 4) if not np.isnan(p_under_cur) else np.nan,
                "result":           "Under" if row["bet_correct"] == 1 else "Over",
                "actual":           row[TARGET],
                "next_season":      int(nxt["season"]),
                "next_week":        int(nxt["week"]),
                "next_line":        nxt["offered_line"],
                "next_odds":        nxt.get("consensus_under_price", pd.NA),
                "next_p_under":     round(p_under_nxt, 4) if not np.isnan(p_under_nxt) else np.nan,
                "delta_next_line":  round(nxt["offered_line"] - row["offered_line"], 2),
                "delta_p_under":    delta_p_under,
                "next_is_bet":      nxt["is_bet"],
            })
    return pd.DataFrame(records)


# ── Q2: Juice drift analysis ──────────────────────────────────────────────────

def juice_drift_analysis(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose market response (Δ P(under)) by whether the line actually moved.
    When the line is flat, any Δ P(under) is pure juice shift.
    Returns drift_df enriched with line_move bucket and implied-prob columns.
    """
    df = drift_df.copy()

    df["line_move"] = "Flat"
    df.loc[df["delta_next_line"] >  0.24, "line_move"] = "Line ↑"
    df.loc[df["delta_next_line"] < -0.24, "line_move"] = "Line ↓"

    def amer_to_imp(col):
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isnan(s), np.nan,
                        np.where(s < 0, -s / (-s + 100), 100 / (s + 100)))

    df["imp_cur"] = amer_to_imp("odds")
    df["imp_nxt"] = amer_to_imp("next_odds")
    return df


def render_juice_drift_section(drift_df: pd.DataFrame) -> str:
    df = juice_drift_analysis(drift_df)

    LINE_MOVE_ORDER = ["Line ↓", "Flat", "Line ↑"]

    THRESHOLD = 0.005  # 0.5pp — treat smaller moves as flat juice

    summary = (
        df.groupby(["result", "line_move"])
        .agg(
            n            =("delta_p_under", "count"),
            avg_dp       =("delta_p_under", "mean"),
            pct_toward   =("delta_p_under", lambda x: (x >  THRESHOLD).mean() * 100),
            pct_flat_j   =("delta_p_under", lambda x: (x.abs() <= THRESHOLD).mean() * 100),
            pct_away     =("delta_p_under", lambda x: (x < -THRESHOLD).mean() * 100),
            avg_imp_cur  =("imp_cur", "mean"),
            avg_imp_nxt  =("imp_nxt", "mean"),
        )
        .round(4)
        .reset_index()
    )

    def imp_to_amer(p):
        if np.isnan(p):
            return "—"
        if p >= 0.5:
            return f"{-p / (1 - p) * 100:.0f}"
        return f"+{(1 - p) / p * 100:.0f}"

    header_style = ("padding:10px 14px;background:#21262d;color:#8b949e;"
                    "font-size:12px;font-weight:600;text-align:center;"
                    "border-bottom:2px solid #30363d;white-space:nowrap")
    headers = ["Result", "Line move", "N",
               "Avg Δ P(under)", "% juice → Under", "% flat", "% juice → Over",
               "Avg under price (cur → next)"]
    ths = "".join(f"<th style='{header_style}'>{h}</th>" for h in headers)

    rows = []
    for result_val in ["Under", "Over"]:
        res_color = "#3fb950" if result_val == "Under" else "#f85149"
        sub = summary[summary["result"] == result_val]
        # Sort by defined order
        sub = sub.set_index("line_move").reindex(LINE_MOVE_ORDER).dropna(how="all").reset_index()
        first = True
        for _, r in sub.iterrows():
            dp      = r["avg_dp"]
            dp_bg, dp_fg = drift_punder_color(dp)
            dp_str  = f"{dp * 100:+.1f}pp" if pd.notna(dp) else "—"

            cur_str = f"{r['avg_imp_cur']*100:.1f}%" if pd.notna(r["avg_imp_cur"]) else "—"
            nxt_str = f"{r['avg_imp_nxt']*100:.1f}%" if pd.notna(r["avg_imp_nxt"]) else "—"
            price_str = f"{cur_str} → {nxt_str}"

            result_cell = (f'<td style="text-align:center;padding:10px 14px;font-weight:700;'
                           f'color:{res_color};background:#161b22">{result_val}</td>'
                           if first else
                           f'<td style="background:#161b22"></td>')
            first = False

            rows.append(f"""<tr style="border-bottom:1px solid #21262d">
              {result_cell}
              <td style="text-align:center;padding:10px 14px;background:#161b22;color:#e6edf3">{r['line_move']}</td>
              <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{int(r['n'])}</td>
              <td style="text-align:center;padding:10px 14px;background:{dp_bg};color:{dp_fg};font-weight:700">{dp_str}</td>
              <td style="text-align:center;padding:10px 14px;background:#161b22;color:#3fb950">{r['pct_toward']:.1f}%</td>
              <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{r['pct_flat_j']:.1f}%</td>
              <td style="text-align:center;padding:10px 14px;background:#161b22;color:#f85149">{r['pct_away']:.1f}%</td>
              <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e;font-family:monospace;font-size:11px">{price_str}</td>
            </tr>""")
        # separator row between result groups
        rows.append(f'<tr><td colspan="{len(headers)}" style="height:6px;background:#0d1117"></td></tr>')

    # Flat-line callout: pure juice drift numbers
    flat = df[df["line_move"] == "Flat"]
    flat_under = flat[flat["result"] == "Under"]
    flat_over  = flat[flat["result"] == "Over"]

    def flat_summary(grp, label, color):
        if grp.empty:
            return ""
        pct_j = (grp["delta_p_under"].abs() > THRESHOLD).mean() * 100
        avg   = grp["delta_p_under"].mean() * 100
        return (f'<span style="color:{color};font-weight:700">{label}</span> '
                f'({len(grp)} flat-line pairs): '
                f'{pct_j:.0f}% had meaningful juice shift · avg Δ P(under) = '
                f'<span style="color:{"#3fb950" if avg >= 0 else "#f85149"};font-weight:700">{avg:+.1f}pp</span>')

    callout = " &nbsp;·&nbsp; ".join(filter(None, [
        flat_summary(flat_under, "After Under", "#3fb950"),
        flat_summary(flat_over,  "After Over",  "#f85149"),
    ]))

    return f"""
    <div style="margin-bottom:20px">
      <div style="font-size:12px;color:#8b949e;margin-bottom:12px">
        When the line holds flat, any Δ P(under) is pure juice movement.
        <span style="color:#3fb950">Green</span> = market moved toward UNDER ·
        <span style="color:#f85149">Red</span> = market moved away from UNDER.
        Threshold: ±0.5pp.
      </div>
      <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:860px;margin-bottom:18px">
        <thead><tr>{ths}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      <div style="font-size:12px;color:#8b949e;padding:10px 14px;background:#161b22;
                  border-left:3px solid #30363d;max-width:860px">
        <strong style="color:#e6edf3">Flat-line pairs (pure juice drift):</strong><br>
        {callout}
      </div>
    </div>"""


# ── Q3: Edge persistence ──────────────────────────────────────────────────────

def edge_persistence_analysis(all_timelines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build consecutive bet pairs per player.
    Each row: (prev_edge_abs, curr_edge_abs, curr_correct, weeks_apart).
    """
    records = []
    for player, df in all_timelines.items():
        bets = df[df["is_bet"] & df["edge"].notna() & df["bet_correct"].notna()].copy()
        bets = bets.sort_values(["season", "week"]).reset_index(drop=True)
        for i in range(len(bets) - 1):
            prev = bets.iloc[i]
            curr = bets.iloc[i + 1]
            weeks_apart = (
                (curr["season"] - prev["season"]) * 18 +
                (curr["week"]   - prev["week"])
            )
            records.append({
                "player":         player,
                "prev_season":    int(prev["season"]),
                "prev_week":      int(prev["week"]),
                "prev_edge_abs":  abs(float(prev["edge"])),
                "prev_correct":   int(prev["bet_correct"]),
                "curr_season":    int(curr["season"]),
                "curr_week":      int(curr["week"]),
                "curr_edge_abs":  abs(float(curr["edge"])),
                "curr_correct":   int(curr["bet_correct"]),
                "weeks_apart":    int(weeks_apart),
            })
    return pd.DataFrame(records)


def render_edge_persistence_section(pairs: pd.DataFrame) -> str:
    from scipy.stats import spearmanr

    if pairs.empty:
        return "<p>No consecutive bet pairs found.</p>"

    # Spearman lag-1 autocorrelation
    r, pval = spearmanr(pairs["prev_edge_abs"], pairs["curr_edge_abs"])

    # Quartile bins on prev_edge_abs
    pairs = pairs.copy()
    pairs["bucket"] = pd.qcut(pairs["prev_edge_abs"], q=4,
                               labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    summary = pairs.groupby("bucket", observed=True).agg(
        n              =("curr_edge_abs", "count"),
        avg_prev_edge  =("prev_edge_abs", "mean"),
        avg_curr_edge  =("curr_edge_abs", "mean"),
        hit_rate       =("curr_correct",  "mean"),
    ).round(4).reset_index()

    # Color: is avg_curr_edge higher or lower than avg_prev_edge?
    def edge_delta_color(prev, curr):
        diff = curr - prev
        if abs(diff) < 0.002:
            return "#8b949e"
        return "#3fb950" if diff > 0 else "#f85149"

    def hit_rate_color(hr):
        if hr >= 0.58:
            return "#3fb950"
        if hr >= 0.52:
            return "#e3b341"
        return "#f85149"

    hs = ("padding:10px 14px;background:#21262d;color:#8b949e;"
          "font-size:12px;font-weight:600;text-align:center;"
          "border-bottom:2px solid #30363d;white-space:nowrap")
    headers = ["Prior edge bucket", "N pairs",
               "Avg prior edge", "Avg next edge", "Δ edge", "Hit rate (next bet)"]
    ths = "".join(f"<th style='{hs}'>{h}</th>" for h in headers)

    rows = []
    for _, r_row in summary.iterrows():
        prev_e = r_row["avg_prev_edge"] * 100
        curr_e = r_row["avg_curr_edge"] * 100
        diff   = curr_e - prev_e
        d_col  = edge_delta_color(r_row["avg_prev_edge"], r_row["avg_curr_edge"])
        hr_col = hit_rate_color(r_row["hit_rate"])
        rows.append(f"""<tr style="border-bottom:1px solid #21262d">
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#e6edf3;font-weight:600">{r_row['bucket']}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{int(r_row['n'])}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#79c0ff;font-weight:700">{prev_e:.1f}pp</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#79c0ff;font-weight:700">{curr_e:.1f}pp</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:{d_col};font-weight:700">{diff:+.1f}pp</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:{hr_col};font-weight:700">{r_row['hit_rate']*100:.1f}%</td>
        </tr>""")

    # Spearman callout color
    r_color = "#3fb950" if r > 0.2 else ("#e3b341" if r > 0.05 else "#f85149")
    sig_str  = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
    r_interp = (
        "strong persistence — same players keep showing up with edge"  if r > 0.4 else
        "moderate persistence — some structural player-level edge"      if r > 0.2 else
        "weak persistence — edge is mostly game-specific"               if r > 0.05 else
        "near-zero — treat each game independently"
    )

    n_total  = len(pairs)
    n_players = pairs["player"].nunique()

    return f"""
    <div style="margin-bottom:20px">
      <div style="font-size:12px;color:#8b949e;margin-bottom:16px">
        {n_total:,} consecutive bet pairs across {n_players} players.
        Each row is one (bet N → bet N+1) pair for the same player, sorted chronologically.
      </div>

      <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:720px;margin-bottom:20px">
        <thead><tr>{ths}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>

      <div style="font-size:12px;color:#8b949e;padding:12px 16px;background:#161b22;
                  border-left:3px solid #30363d;max-width:720px">
        <strong style="color:#e6edf3">Lag-1 Spearman r = </strong>
        <span style="color:{r_color};font-weight:700">{r:+.3f}</span>
        <span style="color:#8b949e"> ({sig_str}) — {r_interp}</span><br>
        <span style="color:#8b949e;font-size:11px">
          High r → player-structural edge (same players chronically mispriced).<br>
          Near-zero r → edge is game-specific; don't chase a player just because last week was high edge.
        </span>
      </div>
    </div>"""


# ── Q4: Back-to-back hit rate ─────────────────────────────────────────────────

def back_to_back_analysis(all_timelines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each bet row, compute consecutive bet depth:
      1 = isolated (prior appearance was not a bet, or no prior appearance)
      2 = 2nd consecutive bet with no no-bet game in between
      3+ = third or more consecutive bets
    """
    records = []
    for player, df in all_timelines.items():
        df_s = df.sort_values(["season", "week"]).reset_index(drop=True)
        consec = 0
        for _, row in df_s.iterrows():
            if row["is_bet"]:
                consec += 1
            else:
                consec = 0
            if row["is_bet"] and pd.notna(row.get("bet_correct")):
                records.append({
                    "player":     player,
                    "season":     int(row["season"]),
                    "week":       int(row["week"]),
                    "depth":      consec,
                    "hit":        int(row["bet_correct"]),
                    "edge_abs":   abs(float(row["edge"])) if pd.notna(row.get("edge")) else np.nan,
                    "line":       row["offered_line"],
                })
    return pd.DataFrame(records)


def render_back_to_back_section(bb: pd.DataFrame) -> str:
    if bb.empty:
        return "<p>No data.</p>"

    bb = bb.copy()
    bb["depth_bucket"] = bb["depth"].apply(
        lambda d: "1 — Isolated" if d == 1 else ("2 — 2nd straight" if d == 2 else "3+ — 3rd+ straight")
    )
    BUCKET_ORDER = ["1 — Isolated", "2 — 2nd straight", "3+ — 3rd+ straight"]

    summary = (
        bb.groupby("depth_bucket")
        .agg(
            n        =("hit",      "count"),
            hit_rate =("hit",      "mean"),
            avg_edge =("edge_abs", "mean"),
            avg_line =("line",     "mean"),
        )
        .round(4)
        .reindex(BUCKET_ORDER)
        .reset_index()
    )

    # Baseline = isolated hit rate
    baseline_hr = summary.loc[summary["depth_bucket"] == "1 — Isolated", "hit_rate"].values
    baseline_hr = float(baseline_hr[0]) if len(baseline_hr) else np.nan

    def hr_color(hr):
        if np.isnan(hr): return "#8b949e"
        if hr >= 0.60:   return "#3fb950"
        if hr >= 0.55:   return "#e3b341"
        return "#f85149"

    def delta_color(d):
        if abs(d) < 0.005: return "#8b949e"
        return "#3fb950" if d > 0 else "#f85149"

    hs = ("padding:10px 14px;background:#21262d;color:#8b949e;"
          "font-size:12px;font-weight:600;text-align:center;"
          "border-bottom:2px solid #30363d;white-space:nowrap")
    headers = ["Consecutive depth", "N bets", "Hit rate", "vs Isolated", "Avg edge", "Avg line"]
    ths = "".join(f"<th style='{hs}'>{h}</th>" for h in headers)

    rows = []
    for _, r in summary.iterrows():
        hr     = r["hit_rate"]
        delta  = hr - baseline_hr if not np.isnan(baseline_hr) and r["depth_bucket"] != "1 — Isolated" else np.nan
        d_str  = f"{delta*100:+.1f}pp" if pd.notna(delta) else "—"
        d_col  = delta_color(delta) if pd.notna(delta) else "#8b949e"
        rows.append(f"""<tr style="border-bottom:1px solid #21262d">
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#e6edf3;font-weight:600">{r['depth_bucket']}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{int(r['n'])}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:{hr_color(hr)};font-weight:700">{hr*100:.1f}%</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:{d_col};font-weight:700">{d_str}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#79c0ff">{r['avg_edge']*100:.1f}pp</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{r['avg_line']:.2f}</td>
        </tr>""")

    # Depth distribution
    depth_counts = bb["depth"].value_counts().sort_index()
    dist_parts = []
    for d, cnt in depth_counts.items():
        pct = cnt / len(bb) * 100
        dist_parts.append(f"depth {d}: {cnt:,} ({pct:.1f}%)")
    dist_str = " &nbsp;·&nbsp; ".join(dist_parts[:6])
    if len(depth_counts) > 6:
        dist_str += f" &nbsp;·&nbsp; depth 7+: {depth_counts[depth_counts.index >= 7].sum():,}"

    return f"""
    <div style="margin-bottom:20px">
      <div style="font-size:12px;color:#8b949e;margin-bottom:16px">
        {len(bb):,} bet rows across {bb['player'].nunique()} players.
        Consecutive depth resets on any no-bet game appearance.
      </div>

      <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:700px;margin-bottom:20px">
        <thead><tr>{ths}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>

      <div style="font-size:12px;color:#8b949e;padding:12px 16px;background:#161b22;
                  border-left:3px solid #30363d;max-width:700px">
        <strong style="color:#e6edf3">Depth distribution:</strong>
        {dist_str}
      </div>
    </div>"""


# ── Q5: Streak-conditioned hit rate ───────────────────────────────────────────

def streak_conditioned_analysis(all_timelines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each bet, record the streak entering that bet (prior bet's streak)
    and whether this bet won. Prior streak = 0 means first bet or post-reset.
    """
    records = []
    for player, df in all_timelines.items():
        bets = (df[df["is_bet"] & df["bet_correct"].notna()]
                .sort_values(["season", "week"])
                .reset_index(drop=True))
        for i, row in bets.iterrows():
            prior_streak = int(bets.loc[i - 1, "streak"]) if i > 0 and pd.notna(bets.loc[i - 1, "streak"]) else 0
            records.append({
                "player":        player,
                "season":        int(row["season"]),
                "week":          int(row["week"]),
                "prior_streak":  prior_streak,
                "hit":           int(row["bet_correct"]),
                "edge_abs":      abs(float(row["edge"])) if pd.notna(row.get("edge")) else np.nan,
                "line":          row["offered_line"],
            })
    return pd.DataFrame(records)


def render_streak_conditioned_section(sc: pd.DataFrame) -> str:
    if sc.empty:
        return "<p>No data.</p>"

    sc = sc.copy()

    def bucket(s):
        if s <= -3:  return "Losing 3+  (≤ −3)"
        if s <= -1:  return "Losing 1–2  (−1 / −2)"
        if s == 0:   return "No prior streak  (0)"
        if s <= 2:   return "Winning 1–2  (+1 / +2)"
        return           "Winning 3+  (≥ +3)"

    BUCKET_ORDER = [
        "Losing 3+  (≤ −3)",
        "Losing 1–2  (−1 / −2)",
        "No prior streak  (0)",
        "Winning 1–2  (+1 / +2)",
        "Winning 3+  (≥ +3)",
    ]
    BUCKET_COLORS = {
        "Losing 3+  (≤ −3)":       "#f85149",
        "Losing 1–2  (−1 / −2)":   "#e3a37a",
        "No prior streak  (0)":     "#8b949e",
        "Winning 1–2  (+1 / +2)":  "#7ecf8e",
        "Winning 3+  (≥ +3)":      "#3fb950",
    }

    sc["bucket"] = sc["prior_streak"].apply(bucket)

    summary = (
        sc.groupby("bucket")
        .agg(
            n        =("hit",      "count"),
            hit_rate =("hit",      "mean"),
            avg_edge =("edge_abs", "mean"),
            avg_line =("line",     "mean"),
        )
        .round(4)
        .reindex(BUCKET_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    baseline_hr = summary.loc[summary["bucket"] == "No prior streak  (0)", "hit_rate"].values
    baseline_hr = float(baseline_hr[0]) if len(baseline_hr) else np.nan

    def hr_color(hr):
        if np.isnan(hr): return "#8b949e"
        if hr >= 0.60:   return "#3fb950"
        if hr >= 0.55:   return "#e3b341"
        return "#f85149"

    def delta_color(d):
        if pd.isna(d) or abs(d) < 0.005: return "#8b949e"
        return "#3fb950" if d > 0 else "#f85149"

    hs = ("padding:10px 14px;background:#21262d;color:#8b949e;"
          "font-size:12px;font-weight:600;text-align:center;"
          "border-bottom:2px solid #30363d;white-space:nowrap")
    headers = ["Prior streak", "N bets", "Hit rate", "vs No-streak", "Avg edge", "Avg line"]
    ths = "".join(f"<th style='{hs}'>{h}</th>" for h in headers)

    rows = []
    for _, r in summary.iterrows():
        hr     = r["hit_rate"]
        is_base = r["bucket"] == "No prior streak  (0)"
        delta  = (hr - baseline_hr) if (not is_base and not np.isnan(baseline_hr)) else np.nan
        d_str  = f"{delta*100:+.1f}pp" if pd.notna(delta) else "—"
        b_col  = BUCKET_COLORS.get(r["bucket"], "#8b949e")
        rows.append(f"""<tr style="border-bottom:1px solid #21262d">
          <td style="text-align:center;padding:10px 14px;background:#161b22;
                     color:{b_col};font-weight:700">{r['bucket']}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{int(r['n'])}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;
                     color:{hr_color(hr)};font-weight:700">{hr*100:.1f}%</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;
                     color:{delta_color(delta)};font-weight:700">{d_str}</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#79c0ff">{r['avg_edge']*100:.1f}pp</td>
          <td style="text-align:center;padding:10px 14px;background:#161b22;color:#8b949e">{r['avg_line']:.2f}</td>
        </tr>""")

    n_total   = len(sc)
    n_players = sc["player"].nunique()
    streak_dist = sc["prior_streak"].value_counts().sort_index()
    hot  = int((sc["prior_streak"] >= 3).sum())
    cold = int((sc["prior_streak"] <= -3).sum())

    return f"""
    <div style="margin-bottom:20px">
      <div style="font-size:12px;color:#8b949e;margin-bottom:16px">
        {n_total:,} bet rows across {n_players} players.
        Prior streak = streak value after the previous bet on this player (0 = first bet or reset).
        {hot:,} bets entered on a hot streak (≥+3) · {cold:,} entered on a cold streak (≤−3).
      </div>

      <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:720px;margin-bottom:20px">
        <thead><tr>{ths}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>

      <div style="font-size:12px;color:#8b949e;padding:12px 16px;background:#161b22;
                  border-left:3px solid #30363d;max-width:720px">
        If hit rate drops at <span style="color:#3fb950">Winning 3+</span>: market overcorrected, adjusted lines are tougher.<br>
        If hit rate rises at <span style="color:#f85149">Losing 3+</span>: mean reversion — market over-adjusted in our favor.<br>
        Flat across all buckets = streaks are noise, edge is stable regardless of recent run.
      </div>
    </div>"""


# ── Color helpers ─────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return int(a + (b - a) * max(0.0, min(1.0, t)))


def _rgb(r, g, b):
    return f"rgb({r},{g},{b})"


def result_color(val):
    if pd.isna(val):
        return "#1c2333", "#8b949e"
    return ("#1a3a1a", "#3fb950") if val == 1 else ("#3a1a1a", "#f85149")


def streak_color(v):
    if pd.isna(v) or v == 0:
        return "#161b22", "#8b949e"
    if v > 0:
        t = min(1.0, v / 5)
        r = _lerp(40, 30, t); g = _lerp(100, 210, t); b = _lerp(40, 80, t)
        return _rgb(r, g, b), "#e6edf3"
    else:
        t = min(1.0, abs(v) / 5)
        r = _lerp(100, 200, t); g = _lerp(40, 30, t); b = _lerp(40, 30, t)
        return _rgb(r, g, b), "#e6edf3"


def units_color(v):
    if pd.isna(v):
        return "#161b22", "#8b949e"
    if v >= 0:
        t = min(1.0, v / 30)
        r = _lerp(40, 30, t); g = _lerp(100, 200, t); b = _lerp(40, 80, t)
        return _rgb(r, g, b), "#e6edf3"
    else:
        t = min(1.0, abs(v) / 20)
        r = _lerp(100, 200, t); g = _lerp(40, 30, t); b = _lerp(40, 30, t)
        return _rgb(r, g, b), "#e6edf3"


def delta_actual_color(v):
    """For (actual - line): negative = under (green), positive = over (red)."""
    if pd.isna(v):
        return "#161b22", "#8b949e"
    if v == 0:
        return "#1c2333", "#8b949e"
    if v < 0:
        t = min(1.0, abs(v) / 5)
        return _rgb(40, _lerp(100, 210, t), _lerp(60, 80, t)), "#e6edf3"
    else:
        t = min(1.0, v / 5)
        return _rgb(_lerp(80, 200, t), _lerp(40, 30, t), 40), "#e6edf3"


def delta_line_color(v):
    if pd.isna(v):
        return "#161b22", "#8b949e"
    if abs(v) < 0.1:
        return "#1c2333", "#8b949e"
    if v > 0:   # line went up — harder to go under
        t = min(1.0, v / 2)
        return _rgb(_lerp(60, 180, t), _lerp(40, 30, t), 40), "#e6edf3"
    else:       # line went down — easier to go under
        t = min(1.0, abs(v) / 2)
        return _rgb(40, _lerp(80, 200, t), _lerp(60, 80, t)), "#e6edf3"


def drift_summary_color(v):
    """For Δ Line: positive = line up = bad (red), negative = line down = good (green)."""
    if pd.isna(v):
        return "#1c2333", "#8b949e"
    if abs(v) < 0.05:
        return "#1c2333", "#8b949e"
    if v > 0:
        t = min(1.0, v / 1.5)
        return _rgb(_lerp(80, 200, t), _lerp(40, 30, t), 40), "#e6edf3"
    else:
        t = min(1.0, abs(v) / 1.5)
        return _rgb(40, _lerp(100, 210, t), _lerp(60, 80, t)), "#e6edf3"


def drift_punder_color(v):
    """For Δ P(under): positive = market moved toward under = good (green), negative = bad (red)."""
    if pd.isna(v):
        return "#1c2333", "#8b949e"
    if abs(v * 100) < 0.1:
        return "#1c2333", "#8b949e"
    if v > 0:
        t = min(1.0, v / 0.02)
        return _rgb(40, _lerp(100, 210, t), _lerp(60, 80, t)), "#e6edf3"
    else:
        t = min(1.0, abs(v) / 0.02)
        return _rgb(_lerp(80, 200, t), _lerp(40, 30, t), 40), "#e6edf3"


# ── HTML rendering ────────────────────────────────────────────────────────────

def _td(content, bg="#161b22", fg="#e6edf3", bold=False, align="center"):
    bw = "font-weight:600;" if bold else ""
    return (f'<td style="background:{bg};color:{fg};{bw}text-align:{align};'
            f'padding:6px 10px;border-bottom:1px solid #21262d">{content}</td>')


def render_player_table(df: pd.DataFrame, player: str) -> str:
    rows = []
    for _, row in df.iterrows():
        is_bet = row["is_bet"]
        row_bg = "#0d1117" if not is_bet else "#161b22"

        res_bg, res_fg = result_color(row.get("bet_correct", np.nan) if is_bet else np.nan)
        sk_bg,  sk_fg  = streak_color(row.get("streak", np.nan))
        ru_bg,  ru_fg  = units_color(row.get("running_units", np.nan))
        dl_bg,  dl_fg  = delta_line_color(row.get("delta_line", np.nan))

        result_str = ("✓ Under" if row["bet_correct"] == 1 else "✗ Over") if is_bet and not pd.isna(row.get("bet_correct", np.nan)) else ("— no bet" if not is_bet else "—")
        streak_str = f"{'+' if row['streak'] > 0 else ''}{row['streak']:.0f}" if is_bet and not pd.isna(row.get("streak")) else "—"
        ru_str     = f"{row['running_units']:+.2f}" if is_bet and not pd.isna(row.get("running_units")) else "—"
        dl_str     = f"{row['delta_line']:+.1f}" if not pd.isna(row.get("delta_line", np.nan)) else "—"
        # For UNDER bets: flip sign so edge shows advantage in our direction (always positive)
        edge_str   = f"+{abs(row['edge'])*100:.1f}pp" if is_bet and not pd.isna(row.get("edge", np.nan)) else "—"
        p_mkt_str  = f"{row['p_market']*100:.1f}%" if not pd.isna(row.get("p_market", np.nan)) else "—"
        p_mod_str  = f"{row['p_hybrid']*100:.1f}%" if not pd.isna(row.get("p_hybrid", np.nan)) else "—"
        actual_str = f"{row[TARGET]:.0f}" if not pd.isna(row.get(TARGET, np.nan)) else "—"
        _actual = row.get(TARGET, np.nan)
        delta_val = float(_actual - row["offered_line"]) if pd.notna(_actual) else np.nan
        da_bg, da_fg = delta_actual_color(delta_val)
        delta_str = f"{delta_val:+.1f}" if pd.notna(delta_val) else "—"

        bet_marker = "●" if is_bet else "○"
        bet_fg     = "#79c0ff" if is_bet else "#484f58"

        cells = (
            _td(f"{row['season']} W{int(row['week'])}", bg=row_bg, fg="#8b949e") +
            _td(f"{row['offered_line']:.1f}", bg=row_bg) +
            _td(dl_str, dl_bg, dl_fg) +
            _td(p_mkt_str, bg=row_bg, fg="#8b949e") +
            _td(p_mod_str, bg=row_bg, fg="#8b949e") +
            _td(edge_str, bg=row_bg, fg="#79c0ff" if is_bet else "#484f58") +
            _td(bet_marker, bg=row_bg, fg=bet_fg, bold=True) +
            _td(actual_str, bg=row_bg, fg="#8b949e") +
            _td(delta_str, da_bg, da_fg, bold=True) +
            _td(result_str, res_bg, res_fg, bold=is_bet) +
            _td(streak_str, sk_bg, sk_fg, bold=True) +
            _td(ru_str, ru_bg, ru_fg, bold=True)
        )
        rows.append(f"<tr>{cells}</tr>")

    header_style = "padding:8px 10px;background:#21262d;color:#8b949e;font-size:11px;font-weight:600;text-align:center;border-bottom:2px solid #30363d;white-space:nowrap"
    headers = ["Season/Wk", "Line", "Δ Line", "P(mkt)", "P(model)", "Edge (UNDER)", "Bet?", "Actual", "Δ", "Result", "Streak", "Running Units"]
    ths = "".join(f"<th style='{header_style}'>{h}</th>" for h in headers)

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead><tr>{ths}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def render_drift_section(drift_df: pd.DataFrame) -> str:
    if drift_df.empty:
        return "<p>No data.</p>"

    summary = drift_df.groupby("result").agg(
        n=("delta_next_line", "count"),
        avg_delta=("delta_next_line", "mean"),
        median_delta=("delta_next_line", "median"),
        pct_up=("delta_next_line", lambda x: (x > 0).mean() * 100),
        pct_flat=("delta_next_line", lambda x: (x == 0).mean() * 100),
        pct_down=("delta_next_line", lambda x: (x < 0).mean() * 100),
        avg_delta_p_under=("delta_p_under", "mean"),
    ).round(3).reset_index()

    def fmt_example(row):
        odds_str      = f" ({int(row['odds']):+d})"      if pd.notna(row.get("odds"))      else ""
        next_odds_str = f" ({int(row['next_odds']):+d})" if pd.notna(row.get("next_odds")) else ""
        return (f"{row['player'].split()[-1]} "
                f"{int(row['season'])} W{int(row['week'])} "
                f"u{row['line']:.1f}{odds_str} → "
                f"{int(row['next_season'])} W{int(row['next_week'])} "
                f"u{row['next_line']:.1f}{next_odds_str}")

    # Min/max examples per result:
    #   Under — min = barely moves/goes up (market ignores), max = biggest drop (market reacts)
    #   Over  — min = barely moves/drops (market ignores), max = biggest rise (market overreacts)
    minmax = {}
    for res, grp in drift_df.groupby("result"):
        grp_s = grp.sort_values("delta_next_line")
        if res == "Under":
            minmax[res] = {"min": fmt_example(grp_s.iloc[-1]),   # least negative / highest = mkt ignores
                           "max": fmt_example(grp_s.iloc[0])}    # most negative = mkt drops line
        else:  # Over
            minmax[res] = {"min": fmt_example(grp_s.iloc[0]),    # most negative / lowest = mkt ignores
                           "max": fmt_example(grp_s.iloc[-1])}   # most positive = mkt raises line

    header_style = "padding:10px 16px;background:#21262d;color:#8b949e;font-size:12px;font-weight:600;text-align:center;border-bottom:2px solid #30363d"
    headers = ["Result", "N", "Avg Δ Line", "Avg Δ P(under) →next game", "Median Δ Line", "% Line ↑", "% Flat", "% Line ↓",
               "Min (least reaction)", "Max (most reaction)"]
    ths = "".join(f"<th style='{header_style};text-align:left' >{h}</th>"
                  if h.startswith("Min") or h.startswith("Max")
                  else f"<th style='{header_style}'>{h}</th>"
                  for h in headers)

    rows = []
    for _, r in summary.iterrows():
        res_color = "#3fb950" if r["result"] == "Under" else "#f85149"
        avg_bg, avg_fg = drift_summary_color(r["avg_delta"])
        med_bg, med_fg = drift_summary_color(r["median_delta"])
        ex_min = minmax.get(r["result"], {}).get("min", "—")
        ex_max = minmax.get(r["result"], {}).get("max", "—")
        td_ex  = "text-align:left;padding:10px 16px;background:#161b22;color:#8b949e;font-family:monospace;font-size:11px"
        dp_val = r["avg_delta_p_under"]
        dp_bg, dp_fg = drift_punder_color(dp_val)
        dp_str = f"{dp_val*100:+.1f}pp" if not pd.isna(dp_val) else "—"
        cells = (
            f'<td style="text-align:center;padding:10px 16px;font-weight:700;color:{res_color};background:#161b22">{r["result"]}</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:#161b22;color:#8b949e">{r["n"]:.0f}</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:{avg_bg};color:{avg_fg};font-weight:700">{r["avg_delta"]:+.3f}</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:{dp_bg};color:{dp_fg};font-weight:700">{dp_str}</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:{med_bg};color:{med_fg};font-weight:700">{r["median_delta"]:+.3f}</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:#161b22;color:#e6edf3">{r["pct_up"]:.1f}%</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:#161b22;color:#8b949e">{r["pct_flat"]:.1f}%</td>' +
            f'<td style="text-align:center;padding:10px 16px;background:#161b22;color:#e6edf3">{r["pct_down"]:.1f}%</td>' +
            f'<td style="{td_ex}">{ex_min}</td>' +
            f'<td style="{td_ex}">{ex_max}</td>'
        )
        rows.append(f"<tr style='border-bottom:1px solid #21262d'>{cells}</tr>")

    # Per-player breakdown
    per_player = drift_df.groupby(["player", "result"]).agg(
        n=("delta_next_line", "count"),
        avg_delta=("delta_next_line", "mean"),
    ).round(2).reset_index()

    # Min/max examples per player+result
    player_examples = {}
    for (player, res), grp in drift_df.groupby(["player", "result"]):
        grp_s = grp.sort_values("delta_next_line")
        if res == "Under":
            player_examples[(player, res, "min")] = fmt_example(grp_s.iloc[-1])
            player_examples[(player, res, "max")] = fmt_example(grp_s.iloc[0])
        else:
            player_examples[(player, res, "min")] = fmt_example(grp_s.iloc[0])
            player_examples[(player, res, "max")] = fmt_example(grp_s.iloc[-1])

    player_rows = []
    for player in per_player["player"].unique():
        sub = per_player[per_player["player"] == player]
        win_row  = sub[sub["result"] == "Under"]
        loss_row = sub[sub["result"] == "Over"]
        w_avg = win_row["avg_delta"].values[0] if not win_row.empty else np.nan
        l_avg = loss_row["avg_delta"].values[0] if not loss_row.empty else np.nan
        w_n   = int(win_row["n"].values[0]) if not win_row.empty else 0
        l_n   = int(loss_row["n"].values[0]) if not loss_row.empty else 0
        w_bg, w_fg = drift_summary_color(w_avg)
        l_bg, l_fg = drift_summary_color(l_avg)
        td_p = "padding:7px 12px;background:#161b22;color:#8b949e;text-align:left;font-family:monospace;font-size:11px"
        w_min = player_examples.get((player, "Under", "min"), "—")
        w_max = player_examples.get((player, "Under", "max"), "—")
        l_min = player_examples.get((player, "Over",  "min"), "—")
        l_max = player_examples.get((player, "Over",  "max"), "—")
        player_rows.append(f"""
        <tr style="border-bottom:1px solid #21262d">
          <td style="padding:7px 14px;background:#161b22;color:#e6edf3;text-align:left;font-weight:600">{player}</td>
          <td style="padding:7px 14px;background:{w_bg};color:{w_fg};text-align:center;font-weight:600">{f'{w_avg:+.2f} (n={w_n})' if not np.isnan(w_avg) else '—'}</td>
          <td style="{td_p}">min: {w_min}<br><span style="color:#484f58">max: {w_max}</span></td>
          <td style="padding:7px 14px;background:{l_bg};color:{l_fg};text-align:center;font-weight:600">{f'{l_avg:+.2f} (n={l_n})' if not np.isnan(l_avg) else '—'}</td>
          <td style="{td_p}">min: {l_min}<br><span style="color:#484f58">max: {l_max}</span></td>
        </tr>""")

    return f"""
    <div style="margin-bottom:24px">
      <div style="font-size:13px;color:#8b949e;margin-bottom:12px">
        Δ Line = next game line − current game line &nbsp;·&nbsp;
        <span style="color:#f85149">Red = line went up (harder to go UNDER)</span> &nbsp;·&nbsp;
        <span style="color:#3fb950">Green = line went down (easier to go UNDER)</span>
      </div>
      <div style="font-size:12px;color:#8b949e;margin-bottom:16px;padding:10px 14px;background:#161b22;border-left:3px solid #30363d;max-width:820px">
        <strong style="color:#e6edf3">Avg Δ P(under) →next game</strong> = market's implied P(under) next game minus current game.<br>
        <span style="color:#f85149">Negative (red)</span> = market moved <em>against</em> the UNDER next game (e.g. line dropped or juice shifted toward Over).<br>
        <span style="color:#3fb950">Positive (green)</span> = market moved <em>toward</em> the UNDER next game.<br>
        Both rows showing negative is expected: after an Under hit, the market lowers the line (harder to go under again); after an Over, the market raises the line but also juices the Over side — net effect still slightly negative.
      </div>
      <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:700px;margin-bottom:28px">
        <thead><tr>{ths}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>

      <div style="font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:10px">Per-Player Breakdown</div>
      <table style="border-collapse:collapse;font-size:12px;width:100%;max-width:560px">
        <thead>
          <tr style="background:#21262d">
            <th style="padding:8px 14px;color:#8b949e;font-size:11px;text-align:left">Player</th>
            <th style="padding:8px 14px;color:#3fb950;font-size:11px;text-align:center">Avg Δ Line after Under</th>
            <th style="padding:8px 14px;color:#3fb950;font-size:11px;text-align:left">Min / Max Example (Under)</th>
            <th style="padding:8px 14px;color:#f85149;font-size:11px;text-align:center">Avg Δ Line after Over</th>
            <th style="padding:8px 14px;color:#f85149;font-size:11px;text-align:left">Min / Max Example (Over)</th>
          </tr>
        </thead>
        <tbody>{"".join(player_rows)}</tbody>
      </table>
    </div>"""


def render_error_analysis(bets: pd.DataFrame) -> str:
    """
    Group bet rows by actual outcome (Over/Under the line).
    Shows mean error, MAE, and std — captures asymmetry in miss magnitude.
    """
    df = bets.copy()
    df["error"] = df[TARGET] - df["offered_line"]
    df["outcome"] = df["actual_over"].map({1.0: "Over", 0.0: "Under"})

    summary = df.groupby("outcome").agg(
        n=("error", "count"),
        mean_error=("error", "mean"),
        mae=("error", lambda x: x.abs().mean()),
        std=("error", "std"),
        pct_miss_lt1=("error", lambda x: (x.abs() < 1).mean() * 100),
    ).round(2).reset_index()

    header_style = "padding:10px 16px;background:#21262d;color:#8b949e;font-size:12px;font-weight:600;text-align:center;border-bottom:2px solid #30363d"
    headers = ["Outcome", "N bets", "Mean Error", "MAE", "Std Dev", "% within 1 tackle"]
    ths = "".join(f"<th style='{header_style}'>{h}</th>" for h in headers)

    rows = []
    for _, r in summary.iterrows():
        is_under = r["outcome"] == "Under"
        outcome_color = "#3fb950" if is_under else "#f85149"
        me_color = "#3fb950" if r["mean_error"] < 0 else "#f85149"
        rows.append(f"""
        <tr style="border-bottom:1px solid #21262d">
          <td style="text-align:center;padding:10px 16px;font-weight:700;color:{outcome_color};background:#161b22">{r['outcome']}</td>
          <td style="text-align:center;padding:10px 16px;background:#161b22;color:#8b949e">{r['n']:.0f}</td>
          <td style="text-align:center;padding:10px 16px;background:#161b22;color:{me_color};font-weight:700">{r['mean_error']:+.2f}</td>
          <td style="text-align:center;padding:10px 16px;background:#161b22;color:#e6edf3;font-weight:700">{r['mae']:.2f}</td>
          <td style="text-align:center;padding:10px 16px;background:#161b22;color:#8b949e">{r['std']:.2f}</td>
          <td style="text-align:center;padding:10px 16px;background:#161b22;color:#8b949e">{r['pct_miss_lt1']:.1f}%</td>
        </tr>""")

    return f"""
    <div style="margin-bottom:20px">
      <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:620px">
        <thead><tr>{ths}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      <div style="font-size:12px;color:#8b949e;margin-top:10px">
        Mean Error = avg(actual − line). MAE = mean absolute error (magnitude regardless of direction).
        Negative mean error on Under rows = player came in well below the line on average.
      </div>
    </div>"""


def render_player_summary_bar(df: pd.DataFrame) -> str:
    bets = df[df["is_bet"] & df["bet_correct"].notna()]
    n = len(bets)
    if n == 0:
        return ""
    hr = bets["bet_correct"].mean()
    total_units = bets.apply(lambda r: WIN_PAYOUT if r["bet_correct"] == 1 else -1.0, axis=1).sum()
    avg_edge = bets["edge"].abs().mean() * 100
    max_streak = bets["streak"].max() if bets["streak"].notna().any() else 0
    return (f'<span style="color:#8b949e;font-size:12px">'
            f'Bets: <b style="color:#e6edf3">{n}</b> &nbsp;·&nbsp; '
            f'Hit rate: <b style="color:#3fb950">{hr*100:.1f}%</b> &nbsp;·&nbsp; '
            f'Units: <b style="color:#{"3fb950" if total_units>=0 else "f85149"}">{total_units:+.2f}</b> &nbsp;·&nbsp; '
            f'Avg edge: <b style="color:#79c0ff">{avg_edge:.1f}pp</b> &nbsp;·&nbsp; '
            f'Max win streak: <b style="color:#e6edf3">{max_streak:.0f}</b>'
            f'</span>')


def generate_html(top_players: list[str], timelines: dict, drift_df: pd.DataFrame,
                  results: pd.DataFrame, meta: dict, bets: pd.DataFrame,
                  edge_pairs: pd.DataFrame, bb_df: pd.DataFrame,
                  sc_df: pd.DataFrame) -> str:

    player_sections = []
    for player in top_players:
        df   = timelines[player]
        pos  = results[results["player_name"] == player]["position"].iloc[0]
        team = results[results["player_name"] == player]["team"].iloc[0]
        summary_bar = render_player_summary_bar(df)
        table       = render_player_table(df, player)
        player_sections.append(f"""
        <div class="player-section">
          <div class="player-header">
            <div>
              <span class="player-name">{player}</span>
              <span class="player-meta">{pos} · {team}</span>
            </div>
            {summary_bar}
          </div>
          <div class="table-wrapper">{table}</div>
        </div>""")

    drift_html     = render_drift_section(drift_df)
    juice_html     = render_juice_drift_section(drift_df)
    error_html     = render_error_analysis(bets)
    edge_pers_html = render_edge_persistence_section(edge_pairs)
    bb_html        = render_back_to_back_section(bb_df)
    sc_html        = render_streak_conditioned_section(sc_df)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NFL Tackles · Player Trend Analysis</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px; line-height: 1.5; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 28px 24px 60px; }}
  header {{ margin-bottom: 32px; }}
  header h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 8px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px;
    font-weight:600; margin-right:8px; }}
  .badge.warning {{ background:#5a2500; color:#f0883e; border:1px solid #7d4220; }}
  .badge.config  {{ background:#1f3a5f; color:#79c0ff; border:1px solid #1f6feb; }}
  .meta {{ color:#8b949e; font-size:12px; margin-top:10px; }}
  .meta span {{ margin-right:20px; }}
  .section {{ margin-bottom:48px; }}
  .section-title {{ font-size:16px; font-weight:600; color:#e6edf3; margin-bottom:6px; }}
  .section-sub {{ font-size:12px; color:#8b949e; margin-bottom:16px; }}
  .player-section {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
    margin-bottom:24px; overflow:hidden; }}
  .player-header {{ display:flex; justify-content:space-between; align-items:center;
    padding:14px 18px; background:#21262d; border-bottom:1px solid #30363d; flex-wrap:wrap; gap:8px; }}
  .player-name {{ font-size:15px; font-weight:700; color:#e6edf3; margin-right:10px; }}
  .player-meta {{ font-size:12px; color:#8b949e; }}
  .table-wrapper {{ overflow-x:auto; }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>🏈 NFL Tackles · Player Trend Analysis</h1>
    <div style="margin-top:8px">
      <span class="badge warning">⚠ In-sample</span>
      <span class="badge config">UNDER · edge≥{EDGE_THRESHOLD} · lines {LINE_MIN}–{LINE_MAX} · min_books={MIN_BOOKS}</span>
    </div>
    <div class="meta">
      <span>Seasons: {meta['train_seasons']}</span>
      <span>Top {N_PLAYERS} player timelines · Q1 drift uses all players</span>
      <span>Model: OLS market_L16_game_ctx_pos_overprob</span>
    </div>
  </header>

  <!-- Error asymmetry -->
  <div class="section">
    <div class="section-title">Result Error Analysis</div>
    <div class="section-sub">
      When our UNDER bet wins, how far under was the player? When it loses, how far over?
      Asymmetry here (big under-misses vs small over-misses) is a signal the prop line sits above true expectation.
    </div>
    {error_html}
  </div>

  <!-- Q3: Edge Persistence -->
  <div class="section">
    <div class="section-title">Q3 · Edge Persistence — Does high edge predict high edge next game?</div>
    <div class="section-sub">
      Lag-1 autocorrelation of model edge across consecutive bets per player.
      High persistence = player-structural mispricing. Near-zero = treat each game independently.
    </div>
    {edge_pers_html}
  </div>

  <!-- Q4: Back-to-back hit rate -->
  <div class="section">
    <div class="section-title">Q4 · Back-to-Back Hit Rate — Isolated vs consecutive bets</div>
    <div class="section-sub">
      Does hit rate change when we've been on the same player multiple games running?
      Consecutive depth resets on any no-bet appearance.
    </div>
    {bb_html}
  </div>

  <!-- Q5: Streak-conditioned hit rate -->
  <div class="section">
    <div class="section-title">Q5 · Streak-Conditioned Hit Rate — Does a hot/cold streak predict the next outcome?</div>
    <div class="section-sub">
      Hit rate of each bet conditioned on the streak entering that bet.
      Tests whether the market overcorrects after win streaks or mean-reverts after loss streaks.
    </div>
    {sc_html}
  </div>

  <!-- Q2: Juice Drift -->
  <div class="section">
    <div class="section-title">Q2 · Juice Drift — Does the market adjust without moving the line?</div>
    <div class="section-sub">
      Δ P(under) broken out by whether the line moved at all.
      Flat-line rows isolate pure juice shift from line movement.
    </div>
    {juice_html}
  </div>

  <!-- Q1: Line Drift -->
  <div class="section">
    <div class="section-title">Q1 · Line Drift After Outcome</div>
    <div class="section-sub">
      After the player goes UNDER (our bet hits), does the book raise the line next week?
      After the player goes OVER (our bet misses), does the line drop?
    </div>
    {drift_html}
  </div>

  <!-- Per-player timelines -->
  <div class="section">
    <div class="section-title">Per-Player Timelines</div>
    <div class="section-sub">
      ● = bet placed &nbsp;·&nbsp; ○ = no bet (outside filter or PASS) &nbsp;·&nbsp;
      Δ Line = vs previous game appearance &nbsp;·&nbsp;
      Streak resets on no-bet weeks
    </div>
    {"".join(player_sections)}
  </div>

</div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-players", type=int, default=N_PLAYERS)
    args = parser.parse_args()

    print(f"\n  Loading artifacts from {ARTIFACT_DIR}...")
    artifacts = {
        "ols":       joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib"),
        "residuals": np.load(ARTIFACT_DIR / "residuals.npy"),
        "nb_coefs":  np.load(ARTIFACT_DIR / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0]),
    }
    meta = json.loads((ARTIFACT_DIR / "meta.json").read_text())

    print(f"  Loading labeled dataset...")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].notna() & ~df["position"].isin(DROP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"    {len(df):,} rows")

    print(f"  Running inference...")
    results = run_inference(df, artifacts)
    bets    = filter_bets(results)
    print(f"    {len(bets):,} bets under production config")

    # Step 1 — top N players by bet count
    top_players = (
        bets.groupby("player_name").size()
        .sort_values(ascending=False)
        .head(args.n_players)
        .index.tolist()
    )
    print(f"\n  Top {args.n_players} players by bet count:")
    for p in top_players:
        n = (bets["player_name"] == p).sum()
        hr = bets[bets["player_name"] == p]["bet_correct"].mean()
        print(f"    {p:<30} {n:>3} bets  {hr*100:.1f}% hit rate")

    # Step 2 — build timelines for top-N (display) AND all players (Q1 drift)
    print(f"\n  Building timelines...")
    all_players = bets["player_name"].unique().tolist()
    all_timelines = {}
    for player in all_players:
        player_results = results[results["player_name"] == player].copy()
        bet_keys = set(zip(
            bets[bets["player_name"] == player]["season"],
            bets[bets["player_name"] == player]["week"],
        ))
        all_timelines[player] = build_timeline(player_results, bet_keys)

    timelines = {p: all_timelines[p] for p in top_players}
    print(f"    {len(all_players)} players total, showing top {args.n_players} in timelines")

    # Step 3 Q1 — line drift across ALL players
    print(f"  Computing line drift (all {len(all_players)} players)...")
    drift_df = line_drift_analysis(all_timelines)

    # Step 3 Q3 — edge persistence across ALL players
    print(f"  Computing edge persistence (all {len(all_players)} players)...")
    edge_pairs = edge_persistence_analysis(all_timelines)
    print(f"    {len(edge_pairs):,} consecutive bet pairs")

    # Step 3 Q4 — back-to-back hit rate
    print(f"  Computing back-to-back hit rate...")
    bb_df = back_to_back_analysis(all_timelines)
    print(f"    {len(bb_df):,} bet rows, depth distribution: "
          + ", ".join(f"d{d}={n}" for d, n in bb_df["depth"].value_counts().sort_index().items()))

    # Step 3 Q5 — streak-conditioned hit rate
    print(f"  Computing streak-conditioned hit rate...")
    sc_df = streak_conditioned_analysis(all_timelines)
    hot  = (sc_df["prior_streak"] >= 3).sum()
    cold = (sc_df["prior_streak"] <= -3).sum()
    print(f"    {len(sc_df):,} bet rows · hot streak entries: {hot} · cold streak entries: {cold}")

    print(f"\n  Line drift summary:")
    for result_val, grp in drift_df.groupby("result"):
        print(f"    After {result_val}: avg Δ line = {grp['delta_next_line'].mean():+.3f}  "
              f"(n={len(grp)}, median={grp['delta_next_line'].median():+.3f})")

    print(f"\n  Generating HTML...")
    html = generate_html(top_players, timelines, drift_df, results, meta, bets, edge_pairs, bb_df, sc_df)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Saved → {OUT_HTML}")
    print(f"  Open with: open '{OUT_HTML}'\n")


if __name__ == "__main__":
    main()
