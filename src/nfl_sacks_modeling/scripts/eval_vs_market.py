"""
Evaluate model predictions vs de-vigged market for NFL sacks props.

Filters to rows where BOTH Over and Under prices are present — removes FanDuel
(Overs only) and any other one-sided books from the de-vig calculation.

De-vig method: proportional
    implied_over  = |odds| / (|odds| + 100)  if odds < 0
                  = 100   / (odds  + 100)    if odds > 0
    true_P_over   = implied_over / (implied_over + implied_under)

Metrics reported for model and market side-by-side:
    Brier score, log-loss, AUC-ROC

P&L simulation:
    Bet Over  when model_prob > market_prob  (at posted Over price)
    Bet Under when model_prob < market_prob  (at posted Under price)
    Flat 1-unit wager per row, push = 0 units.

Input:  ~/Downloads/tmp/nfl_sacks_features_2025.parquet
        ~/Downloads/tmp/nfl_sacks_model_2025.pkl
Output: ~/Downloads/tmp/nfl_sacks_eval_vs_market.html

Run:
    python src/nfl_sacks_modeling/scripts/eval_vs_market.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
MODEL_PKL   = Path.home() / "Downloads" / "tmp" / "nfl_sacks_model_2025.pkl"
OUT_HTML    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_eval_vs_market.html"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


def american_to_implied(price: float) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100)
    return 100 / (price + 100)


def units_on_win(price: float) -> float:
    if price < 0:
        return 100 / abs(price)
    return price / 100


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    windows = cfg["rolling_windows"]
    rolling = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    return rolling + ["game_total", "team_spread", "games_played_ytd"], ["pos_group", "pos_side"]


# ── Data prep ──────────────────────────────────────────────────────────────────

def load_eval_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES)

    n_total = len(df)
    n_no_over  = df["prop_median_price_over"].isna().sum()
    n_no_under = df["prop_median_price_under"].isna().sum()
    n_one_sided = (
        df["prop_median_price_over"].notna() & df["prop_median_price_under"].isna()
    ).sum()

    print(f"Raw rows               : {n_total}")
    print(f"  Missing Over price   : {n_no_over}")
    print(f"  Missing Under price  : {n_no_under}")
    print(f"  Over only (no Under) : {n_one_sided}  ← one-sided market (FanDuel etc.), dropped")

    # Require both sides + resolved outcome (no pushes)
    mask = (
        df["prop_median_price_over"].notna() &
        df["prop_median_price_under"].notna() &
        df["target"].notna()
    )
    df = df[mask].copy()
    print(f"After filter           : {len(df)} rows  ({int((df['target']==1).sum())} Over hits, "
          f"{int((df['target']==0).sum())} Under hits)")

    # De-vigged market probability
    df["_impl_over"]  = df["prop_median_price_over"].apply(american_to_implied)
    df["_impl_under"] = df["prop_median_price_under"].apply(american_to_implied)
    df["market_prob"] = df["_impl_over"] / (df["_impl_over"] + df["_impl_under"])
    df.drop(columns=["_impl_over", "_impl_under"], inplace=True)

    return df


# ── Model OOS predictions ──────────────────────────────────────────────────────

def get_model_proba(pipe, df: pd.DataFrame, numeric_cols: list, cat_cols: list) -> np.ndarray:
    X = df[numeric_cols + cat_cols]
    y = df["target"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold CV for OOS model predictions...")
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return proba


# ── P&L simulation ─────────────────────────────────────────────────────────────

def simulate_pnl(df: pd.DataFrame, model_prob: np.ndarray) -> pd.DataFrame:
    results = []
    for i, row in enumerate(df.itertuples(index=False)):
        edge = model_prob[i] - row.market_prob
        if edge > 0:  # bet Over
            price = row.prop_median_price_over
            units = units_on_win(price) if row.target == 1 else -1.0
            side  = "Over"
        else:  # bet Under
            price = row.prop_median_price_under
            units = units_on_win(price) if row.target == 0 else -1.0
            side  = "Under"
        results.append({
            "week":         row.week,
            "player":       row.player,
            "model_prob":   model_prob[i],
            "market_prob":  row.market_prob,
            "edge":         edge,
            "bet_side":     side,
            "price":        price,
            "target":       row.target,
            "units":        units,
        })
    return pd.DataFrame(results).sort_values(["week", "player"]).reset_index(drop=True)


# ── HTML ───────────────────────────────────────────────────────────────────────

def build_html(
    df: pd.DataFrame,
    model_prob: np.ndarray,
    pnl_df: pd.DataFrame,
    n_dropped: int,
) -> str:
    y = df["target"].astype(int)
    mkt = df["market_prob"].values

    # Metrics
    model_brier  = brier_score_loss(y, model_prob)
    market_brier = brier_score_loss(y, mkt)
    model_ll     = log_loss(y, model_prob)
    market_ll    = log_loss(y, mkt)
    model_auc    = roc_auc_score(y, model_prob)
    market_auc   = roc_auc_score(y, mkt)

    def better(model_val, mkt_val, lower_is_better=True):
        if lower_is_better:
            return "model" if model_val < mkt_val else "market"
        return "model" if model_val > mkt_val else "market"

    # Summary box
    total_units = pnl_df["units"].sum()
    n_bets      = len(pnl_df)
    roi         = total_units / n_bets
    n_over_bets = (pnl_df["bet_side"] == "Over").sum()
    n_under_bets = (pnl_df["bet_side"] == "Under").sum()
    pnl_color   = "green" if total_units > 0 else "red"

    summary_box = f"""
<div style="font-family:monospace;background:#e8f5e9;padding:14px;border-radius:6px;
            margin-bottom:28px;font-size:13px;border-left:4px solid #2ca02c;">
  <b>Filter:</b> both Over &amp; Under prices required &nbsp;|&nbsp;
  {n_dropped} one-sided rows dropped (FanDuel etc.) &nbsp;|&nbsp;
  <b>{len(df):,} rows evaluated</b><br>
  De-vig: proportional — true_P_over = implied_over / (implied_over + implied_under)
</div>"""

    # Metrics table
    def fmt_winner(val, mkt_val, lower_is_better=True):
        is_better = (val < mkt_val) if lower_is_better else (val > mkt_val)
        style = "color:green;font-weight:bold" if is_better else ""
        return f'<span style="{style}">{val:.5f}</span>'

    metrics_html = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">Model vs Market — Accuracy Metrics</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Model probabilities from 5-fold CV (OOS). Market = de-vigged implied probability.
    Lower Brier + log-loss is better. Higher AUC is better. <b>Green = winner.</b>
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Metric</th><th>Model</th><th>Market</th><th>Winner</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Brier score ↓</td>
          <td>{fmt_winner(model_brier, market_brier)}</td>
          <td>{fmt_winner(market_brier, model_brier)}</td>
          <td>{better(model_brier, market_brier)}</td></tr>
      <tr><td>Log-loss ↓</td>
          <td>{fmt_winner(model_ll, market_ll)}</td>
          <td>{fmt_winner(market_ll, model_ll)}</td>
          <td>{better(model_ll, market_ll)}</td></tr>
      <tr><td>AUC-ROC ↑</td>
          <td>{fmt_winner(model_auc, market_auc, lower_is_better=False)}</td>
          <td>{fmt_winner(market_auc, model_auc, lower_is_better=False)}</td>
          <td>{better(model_auc, market_auc, lower_is_better=False)}</td></tr>
    </tbody>
  </table>
</div>"""

    # P&L by week table
    week_pnl = (pnl_df.groupby("week")["units"]
                .agg(bets="count", units="sum")
                .reset_index())
    week_pnl["cumulative"] = week_pnl["units"].cumsum()

    pnl_rows = ""
    for _, row in week_pnl.iterrows():
        u_color = "green" if row["units"] >= 0 else "red"
        c_color = "green" if row["cumulative"] >= 0 else "red"
        pnl_rows += (
            f"<tr><td>Wk {int(row['week'])}</td>"
            f"<td>{int(row['bets'])}</td>"
            f"<td><span style='color:{u_color};font-weight:bold'>{row['units']:+.2f}</span></td>"
            f"<td><span style='color:{c_color};font-weight:bold'>{row['cumulative']:+.2f}</span></td></tr>\n"
        )
    # Totals row
    pnl_rows += (
        f"<tr style='background:#eee;border-top:2px solid #333'>"
        f"<td><b>TOTAL</b></td><td><b>{n_bets}</b></td>"
        f"<td><b><span style='color:{pnl_color}'>{total_units:+.2f}u</span></b></td>"
        f"<td><b><span style='color:{pnl_color}'>{roi:+.3f} u/bet</span></b></td></tr>\n"
    )

    pnl_html = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">Simulated P&amp;L — Bet Every Row, 1 Unit Flat</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Bet Over when model_prob &gt; market_prob, else bet Under. No threshold — every row gets a bet.<br>
    Over bets: {n_over_bets} &nbsp;|&nbsp; Under bets: {n_under_bets} &nbsp;|&nbsp;
    Total: <b><span style="color:{pnl_color}">{total_units:+.2f} units</span></b>
    &nbsp;|&nbsp; ROI: <b><span style="color:{pnl_color}">{roi:+.3f} u/bet</span></b>
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Week</th><th>Bets</th><th>Units</th><th>Cumulative</th>
      </tr>
    </thead>
    <tbody>{pnl_rows}</tbody>
  </table>
</div>"""

    # Edge distribution table (deciles of |edge|)
    pnl_df["abs_edge"] = pnl_df["edge"].abs()
    pnl_df["edge_decile"] = pd.qcut(pnl_df["abs_edge"], q=5, labels=False, duplicates="drop")
    edge_grps = pnl_df.groupby("edge_decile").agg(
        n=("units", "count"),
        avg_abs_edge=("abs_edge", "mean"),
        units=("units", "sum"),
    ).reset_index()

    edge_rows = ""
    for _, row in edge_grps.iterrows():
        u_color = "green" if row["units"] >= 0 else "red"
        edge_rows += (
            f"<tr><td>{int(row['edge_decile'])+1} (smallest → largest)</td>"
            f"<td>{row['avg_abs_edge']:.1%}</td>"
            f"<td>{int(row['n'])}</td>"
            f"<td><span style='color:{u_color};font-weight:bold'>{row['units']:+.2f}</span></td></tr>\n"
        )

    edge_html = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">P&amp;L by Edge Size (quintiles of |model − market|)</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    If the model has real edge, rows with larger disagreement vs market should produce better P&amp;L.
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Quintile</th><th>Avg |edge|</th><th>n bets</th><th>Units</th>
      </tr>
    </thead>
    <tbody>{edge_rows}</tbody>
  </table>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks — Model vs Market</title>
  <style>
    body {{ font-family: monospace; margin: 40px; background: #fafafa; }}
    table td, table th {{ padding: 6px 14px; border-bottom: 1px solid #ddd; text-align: left; }}
    h1 {{ font-family: sans-serif; }}
  </style>
</head>
<body>
  <h1>NFL Sacks Props — Model vs De-vigged Market</h1>
  {summary_box}
  {metrics_html}
  {pnl_html}
  {edge_html}
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    print("Loading data...")
    raw = pd.read_parquet(FEATURES)
    n_one_sided = (
        raw["prop_median_price_over"].notna() & raw["prop_median_price_under"].isna()
    ).sum()
    df = load_eval_data(cfg)

    print("Loading model...")
    with open(MODEL_PKL, "rb") as f:
        pipe = pickle.load(f)

    numeric_cols, cat_cols = feature_lists(cfg)
    model_prob = get_model_proba(pipe, df, numeric_cols, cat_cols)

    print("\nMetrics:")
    y   = df["target"].astype(int)
    mkt = df["market_prob"].values
    print(f"  Brier  — model: {brier_score_loss(y, model_prob):.5f}  "
          f"market: {brier_score_loss(y, mkt):.5f}")
    print(f"  LogLoss— model: {log_loss(y, model_prob):.5f}  "
          f"market: {log_loss(y, mkt):.5f}")
    print(f"  AUC    — model: {roc_auc_score(y, model_prob):.5f}  "
          f"market: {roc_auc_score(y, mkt):.5f}")

    print("\nSimulating P&L...")
    pnl_df = simulate_pnl(df, model_prob)
    total  = pnl_df["units"].sum()
    roi    = total / len(pnl_df)
    print(f"  Total units: {total:+.2f}  |  ROI: {roi:+.4f} u/bet  ({len(pnl_df)} bets)")

    html = build_html(df, model_prob, pnl_df, n_dropped=n_one_sided)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUT_HTML}")


if __name__ == "__main__":
    main()
