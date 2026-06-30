"""
Grid search over probability threshold × direction × min edge.
Train: 2024 (M7 LR)  |  Holdout: 2025  |  Flat $1 unit stakes

Scoring granularity: player-game-book (per-book rows from raw props).
Model scores once per player-game; edge is computed vs each book's specific
implied odds. Myles Garrett at BetOnline and FanDuel are two separate rows.

Grid:
  threshold : model P(over) cutoff — under bets trigger when p_over < threshold,
              over bets trigger when p_over > (1 - threshold)
  direction : under | over | both
  min_edge  : minimum (model_prob - book_implied) required to place bet

P&L per bet (flat $1):
  win  → (1 / book_implied - 1)
  lose → -1

Outputs:
  ~/Downloads/tmp/sacks_threshold_search.csv
  Printed top-40 ranked by Units Won (min 20 bets), full table saved

Run:
  python src/nfl_sacks_modeling/scripts/threshold_search.py            # OOS (default)
  python src/nfl_sacks_modeling/scripts/threshold_search.py --in-sample
"""

import argparse
import glob
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TMP       = Path.home() / "Downloads" / "tmp"
F24       = TMP / "nfl_sacks_features_2024.parquet"
F25       = TMP / "nfl_sacks_features_2025.parquet"
PROPS_DIR = TMP / "nfl_defensive_props"
OUT       = TMP / "sacks_threshold_search.csv"

# ── Name normalisation (config-driven) ───────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
_DOTS_RE     = re.compile(r"(?<=[A-Za-z])\.")  # strips any dot after a letter (handles A.J. → AJ)


def _load_name_norm_config() -> tuple[re.Pattern, dict[str, str]]:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f).get("player_name_normalization", {})
    suffixes  = cfg.get("strip_suffixes", [])
    pattern   = r"\s+(" + "|".join(re.escape(s) for s in suffixes) + r")$"
    suffix_re = re.compile(pattern, re.IGNORECASE)
    aliases   = {k.lower(): v.lower() for k, v in cfg.get("aliases", {}).items()}
    return suffix_re, aliases


_SUFFIX_RE, _NAME_ALIASES = _load_name_norm_config()


def _normalize(name: str) -> str:
    name = _SUFFIX_RE.sub("", name.strip())
    name = _DOTS_RE.sub("", name)
    name = name.lower()
    return _NAME_ALIASES.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────

FEATURES   = ["prop_median_impl_over", "qbhit_rate_L16", "sack_rate_Lcareer"]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
DIRECTIONS = ["under", "over", "both"]
MIN_EDGES  = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15]
MIN_BETS   = 20

SEP = "=" * 110


def implied_prob(price: float) -> float:
    return abs(price) / (abs(price) + 100.0) if price < 0 else 100.0 / (price + 100.0)


def fit_and_predict(in_sample: bool = False):
    f24 = pd.read_parquet(TMP / "nfl_sacks_features_2024.parquet")
    f25 = pd.read_parquet(TMP / "nfl_sacks_features_2025.parquet")

    if in_sample:
        train = pd.concat([f24, f25], ignore_index=True)
        score = train.copy()
    else:
        train = f24
        score = f25

    train = train[train["target"].notna()].copy()

    feats = [f for f in FEATURES if f in train.columns]
    fill  = train[feats].mean()

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc",  StandardScaler()),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(train[feats], train["target"].astype(int))

    score = score.copy()
    score["p_over"] = pipe.predict_proba(score[feats].fillna(fill))[:, 1]
    return score[["game_id", "player", "week", "target", "p_over"]]


def build_scoring_dataset(predictions: pd.DataFrame, in_sample: bool = False) -> pd.DataFrame:
    """One row per player-game-book. Over and under implied probs as columns."""
    seasons = ["2024", "2025"] if in_sample else ["2025"]
    files = []
    for s in seasons:
        files += glob.glob(str(PROPS_DIR / s / "*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    sacks = raw[
        (raw["outcome_name"].isin(["Over", "Under"])) &
        (raw["point"] == 0.5)
    ].copy()
    sacks["implied"] = sacks["price"].apply(implied_prob)

    # latest snapshot per player-game-book-side, then pivot to wide
    sacks = (sacks.sort_values("last_update")
                  .drop_duplicates(["nfl_game_id", "bookmaker", "outcome_desc", "outcome_name"], keep="last"))

    wide = (sacks.pivot_table(
                index=["nfl_game_id", "bookmaker", "outcome_desc"],
                columns="outcome_name",
                values="implied",
                aggfunc="first",
            )
            .rename(columns={"Over": "over_implied", "Under": "under_implied"})
            .reset_index()
            .rename(columns={"nfl_game_id": "game_id", "outcome_desc": "player"}))

    wide["_norm"]        = wide["player"].map(_normalize)
    predictions["_norm"] = predictions["player"].map(_normalize)
    scored = wide.merge(predictions, on=["game_id", "_norm"], how="inner")
    scored = scored.drop(columns=["player_x", "_norm"], errors="ignore")
    scored = scored.rename(columns={"player_y": "player"})
    col_order = ["game_id", "bookmaker", "player", "over_implied", "under_implied",
                 "week", "target", "p_over", "edge_under", "edge_over", "would_bet", "bet_won"]
    scored = scored[[c for c in col_order if c in scored.columns]]

    n_raw    = wide["player"].nunique()
    n_joined = scored["player"].nunique()
    n_lost   = n_raw - n_joined
    print(f"  Raw prop players: {n_raw}  |  Joined to predictions: {n_joined}  "
          f"({n_lost} lost to name mismatch)")
    print(f"  Scoring rows: {len(scored):,}  (player-game-book combos)")

    if n_lost > 0:
        props_norm     = set(wide["player"].map(_normalize))
        pred_norm      = set(predictions["player"].map(_normalize))
        unmatched_norm = props_norm - pred_norm
        unmatched      = sorted(
            wide.loc[wide["player"].map(_normalize).isin(unmatched_norm), "player"].unique()
        )
        print(f"\n  === {n_lost} props players with no prediction match ===")
        for name in unmatched:
            n_rows = wide[wide["player"] == name].shape[0]
            print(f"    [{n_rows:>3} rows]  '{name}'")

    return scored


def eval_combo(df, direction, threshold, min_edge):
    bets = []

    if direction in ("under", "both"):
        cands = df[df["under_implied"].notna() & df["target"].notna()].copy()
        cands = cands[cands["p_over"] < threshold]
        cands["edge"] = (1 - cands["p_over"]) - cands["under_implied"]
        cands = cands[cands["edge"] >= min_edge]
        if len(cands):
            wins = (cands["target"] == 0).astype(float)
            cands["pnl"]         = wins * (1 / cands["under_implied"] - 1) - (1 - wins)
            cands["bet_implied"] = cands["under_implied"]
            total_impl = cands["over_implied"] + cands["under_implied"]
            cands["bet_impl_devig"] = cands["under_implied"] / total_impl
            bets.append(cands)

    if direction in ("over", "both"):
        cands = df[df["over_implied"].notna() & df["target"].notna()].copy()
        cands = cands[cands["p_over"] > (1 - threshold)]
        cands["edge"] = cands["p_over"] - cands["over_implied"]
        cands = cands[cands["edge"] >= min_edge]
        if len(cands):
            wins = (cands["target"] == 1).astype(float)
            cands["pnl"]         = wins * (1 / cands["over_implied"] - 1) - (1 - wins)
            cands["bet_implied"] = cands["over_implied"]
            total_impl = cands["over_implied"] + cands["under_implied"]
            cands["bet_impl_devig"] = cands["over_implied"] / total_impl
            bets.append(cands)

    if not bets:
        return None

    pool = pd.concat(bets, ignore_index=True).sort_values("week")
    n     = len(pool)
    wins  = (pool["pnl"] > 0).sum()
    total = pool["pnl"].sum()

    cum   = pool["pnl"].cumsum()
    max_dd = round((cum.cummax() - cum).max(), 2)

    pct_over = round(pool["target"].mean(), 4)

    return {
        "n_bets":       n,
        "hit_rate":     round(wins / n, 4),
        "ev_unit":      round(total / n, 4),
        "pct_over":     pct_over,
        "mean_edge":    round(pool["edge"].mean(), 4),
        "avg_impl":     round(pool["bet_implied"].mean(), 4),
        "avg_impl_dev": round(pool["bet_impl_devig"].mean(), 4),
        "units_won": round(total, 2),
        "max_dd":    max_dd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-sample", action="store_true", help="Train+score on 2024+2025 combined")
    args      = parser.parse_args()
    in_sample = args.in_sample

    mode_label = "IN-SAMPLE (2024+2025 train+score)" if in_sample else "OOS (train 2024 · holdout 2025)"

    print(f"\n{SEP}")
    print(f"  Threshold grid search — {mode_label} | M7 LR | Flat $1 stakes")
    print("  Scoring at player-game-book level (raw per-book props)")
    print(SEP)

    predictions = fit_and_predict(in_sample=in_sample)
    df = build_scoring_dataset(predictions, in_sample=in_sample)

    n_with_target = df["target"].notna().sum()
    print(f"  Bettable rows with outcome: {n_with_target:,}  "
          f"(pos={int((df['target']==1).sum())}, neg={int((df['target']==0).sum())})")
    print(f"  Grid: {len(THRESHOLDS)} thresholds × {len(DIRECTIONS)} directions × {len(MIN_EDGES)} min_edges = "
          f"{len(THRESHOLDS)*len(DIRECTIONS)*len(MIN_EDGES)} combos\n")

    rows = []
    for threshold, direction, min_edge in product(THRESHOLDS, DIRECTIONS, MIN_EDGES):
        result = eval_combo(df, direction, threshold, min_edge)
        if result:
            rows.append({
                "threshold": threshold,
                "direction": direction,
                "min_edge":  min_edge,
                **result,
            })

    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT, index=False)

    # save scored player-game-book dataset for SQL sense checks
    scored_out = TMP / ("nfl_sacks_scored_insample.parquet" if in_sample else "nfl_sacks_scored_2025.parquet")
    df_save = df.copy()
    df_save["edge_under"] = (1 - df_save["p_over"]) - df_save["under_implied"]
    df_save["edge_over"]  = df_save["p_over"] - df_save["over_implied"]
    df_save["would_bet"]  = (df_save["p_over"] < 0.40) & (df_save["edge_under"] >= 0.05) & df_save["under_implied"].notna()
    df_save["bet_won"]    = df_save["would_bet"] & (df_save["target"] == 0)
    df_save.to_parquet(scored_out, index=False)
    print(f"  Scored dataset → {scored_out}")

    display = (all_df[all_df["n_bets"] >= MIN_BETS]
               .sort_values("units_won", ascending=False)
               .reset_index(drop=True))
    display.insert(0, "rank", display.index + 1)

    print(f"  Ranked by Units Won (min {MIN_BETS} bets) — {len(display)} combos shown\n")
    print(f"{'Rk':<4} {'Thresh':>7} {'Dir':<7} {'Edge':>6} {'N':>6} "
          f"{'HitRate':>8} {'EV/Unit':>8} {'%Over':>6} {'MeanEdge':>9} {'Units':>8} {'MaxDD':>7} {'Impl(vig)':>10} {'Impl(dev)':>10}")
    print("-" * 112)
    for _, r in display.head(40).iterrows():
        print(f"{int(r['rank']):<4} {r['threshold']:>7.2f} {r['direction']:<7} {r['min_edge']:>6.2f} "
              f"{int(r['n_bets']):>6} {r['hit_rate']:>8.1%} {r['ev_unit']:>+8.4f} "
              f"{r['pct_over']:>6.1%} {r['mean_edge']:>9.4f} {r['units_won']:>+8.2f} {r['max_dd']:>7.2f} "
              f"{r['avg_impl']:>10.1%} {r['avg_impl_dev']:>10.1%}")

    print(f"\n  Full table ({len(all_df)} combos) saved → {OUT}")
    generate_html(all_df, scoring_rows=len(df), bettable_rows=int(df["target"].notna().sum()), in_sample=in_sample)
    print(f"{SEP}\n")

    # ── direction summary at threshold=0.30 ──────────────────────────────────
    print("  Direction breakdown (threshold=0.30, varying min_edge):\n")
    subset = all_df[
        (all_df["threshold"] == 0.30) &
        (all_df["min_edge"].isin([0.00, 0.01, 0.03, 0.05, 0.10]))
    ].sort_values(["direction", "min_edge"])
    print(f"  {'Dir':<7} {'Edge':>6} {'N':>6} {'HitRate':>8} {'EV/Unit':>8} {'%Over':>6} {'Units':>8} {'MaxDD':>7}")
    print("  " + "-" * 65)
    for _, r in subset.iterrows():
        print(f"  {r['direction']:<7} {r['min_edge']:>6.2f} {int(r['n_bets']):>6} "
              f"{r['hit_rate']:>8.1%} {r['ev_unit']:>+8.4f} {r['pct_over']:>6.1%} "
              f"{r['units_won']:>+8.2f} {r['max_dd']:>7.2f}")


def _out_html(in_sample: bool) -> Path:
    return TMP / ("sacks_threshold_search_insample.html" if in_sample else "sacks_threshold_search.html")


def _dir_color(d: str) -> str:
    return {"under": "#1f4e8c", "over": "#7d3a00", "both": "#1a4a2e"}[d]


def _dir_text(d: str) -> str:
    return {"under": "#79c0ff", "over": "#f0883e", "both": "#56d364"}[d]


def _pnl_color(v: float) -> str:
    if v > 10:  return "rgb(30,200,80)"
    if v > 0:   return "rgb(60,160,80)"
    if v > -5:  return "rgb(200,80,60)"
    return "rgb(180,40,40)"


def _hit_color(v: float) -> str:
    if v >= 0.85: return "rgb(30,200,80)"
    if v >= 0.75: return "rgb(80,180,80)"
    if v >= 0.65: return "rgb(160,160,60)"
    return "rgb(180,80,60)"


def _ev_color(v: float) -> str:
    if v > 0.15:  return "rgb(30,200,80)"
    if v > 0.05:  return "rgb(80,180,80)"
    if v > 0:     return "rgb(130,160,80)"
    return "rgb(180,80,60)"


def generate_html(all_df: pd.DataFrame, scoring_rows: int, bettable_rows: int, in_sample: bool = False) -> None:
    MIN_BETS_CARD = 20
    eligible = all_df[all_df["n_bets"] >= MIN_BETS_CARD].copy()

    # ── picks ─────────────────────────────────────────────────────────────────
    rec    = eligible.sort_values("units_won",   ascending=False).iloc[0]
    analyst = eligible.sort_values("ev_unit",    ascending=False).iloc[0]

    def best_for_dir(d):
        sub = eligible[eligible["direction"] == d]
        return sub.sort_values("units_won", ascending=False).iloc[0] if len(sub) else None

    best_under = best_for_dir("under")
    best_over  = best_for_dir("over")
    best_both  = best_for_dir("both")

    # ── table rows ────────────────────────────────────────────────────────────
    table_rows_html = ""
    sorted_df = eligible.sort_values("units_won", ascending=False).reset_index(drop=True)
    for i, r in sorted_df.iterrows():
        is_rec     = (r["direction"] == rec["direction"]     and r["min_edge"] == rec["min_edge"]     and r["threshold"] == rec["threshold"])
        is_analyst = (r["direction"] == analyst["direction"] and r["min_edge"] == analyst["min_edge"] and r["threshold"] == analyst["threshold"])
        star = " ★" if is_rec else (" ♦" if is_analyst else "")

        dr_cls = f"dr-{r['direction']}"
        bg     = "#1c2333" if i % 2 == 0 else "#161b22"

        def cell(val, color=None, bold=False):
            style = f"padding:8px 10px;text-align:center;border-bottom:1px solid #21262d;"
            inner_style = ""
            if color:
                inner_style = f"background:{color};padding:2px 8px;border-radius:4px;font-weight:700;"
            content = f'<span style="{inner_style}">{val}</span>' if color else val
            if bold:
                content = f"<b>{content}</b>"
            return f'<td style="{style}">{content}</td>'

        dir_badge = (f'<span style="background:{_dir_color(r["direction"])};color:{_dir_text(r["direction"])};'
                     f'padding:2px 10px;border-radius:4px;font-weight:700;font-size:12px">'
                     f'{r["direction"].upper()}{star}</span>')

        row_html = (
            f'<tr class="{dr_cls}" style="background:{bg}">'
            + f'<td style="padding:8px 10px;text-align:center;border-bottom:1px solid #21262d;">{dir_badge}</td>'
            + cell(f'{r["min_edge"]:.2f}')
            + cell(f'{r["threshold"]:.2f}')
            + cell(f'{int(r["n_bets"]):,}')
            + cell(f'{r["hit_rate"]:.1%}', color=_hit_color(r["hit_rate"]))
            + cell(f'{r["ev_unit"]:+.4f}', color=_ev_color(r["ev_unit"]))
            + cell(f'{r["pct_over"]:.1%}')
            + cell(f'{r["mean_edge"]:.4f}')
            + cell(f'{r["units_won"]:+.2f}', color=_pnl_color(r["units_won"]))
            + cell(f'{r["max_dd"]:.2f}')
            + cell(f'{r["avg_impl"]:.1%}')
            + cell(f'{r["avg_impl_dev"]:.1%}')
            + '</tr>'
        )
        table_rows_html += row_html

    # ── summary card ──────────────────────────────────────────────────────────
    def card_html(label, row):
        if row is None:
            return f'<div class="card"><div class="card-header" style="background:#21262d"><span class="card-dir">{label}</span></div><div class="card-body"><p style="color:#8b949e;font-size:12px">No combos with ≥{MIN_BETS_CARD} bets</p></div></div>'
        bg = _dir_color(label.lower())
        tc = _dir_text(label.lower())
        return f'''<div class="card">
      <div class="card-header" style="background:{bg};color:{tc}">
        <span class="card-dir">{label}</span><span class="card-rank">Best Config</span>
      </div>
      <div class="card-body">
        <div class="stat-row"><span class="stat-label">Units won</span>
          <span class="stat-val" style="background:{_pnl_color(row["units_won"])};color:#e6edf3">{row["units_won"]:+.2f}</span></div>
        <div class="stat-row"><span class="stat-label">EV / unit</span>
          <span class="stat-val" style="background:{_ev_color(row["ev_unit"])};color:#e6edf3">{row["ev_unit"]:+.4f}</span></div>
        <div class="stat-row"><span class="stat-label">Hit rate</span>
          <span class="stat-val" style="background:{_hit_color(row["hit_rate"])};color:#e6edf3">{row["hit_rate"]:.1%}</span></div>
        <div class="stat-row"><span class="stat-label">Bets</span>
          <span class="stat-val">{int(row["n_bets"]):,}</span></div>
        <div class="config-line">edge ≥ {row["min_edge"]:.2f} &nbsp;·&nbsp; thresh {row["threshold"]:.2f} &nbsp;·&nbsp; max DD {row["max_dd"]:.2f}</div>
      </div></div>'''

    def rec_line(row, color="#79c0ff"):
        return (f'direction=<span style="color:{color}">{row["direction"].upper()}</span> &nbsp;·&nbsp; '
                f'edge≥<span style="color:{color}">{row["min_edge"]:.2f}</span> &nbsp;·&nbsp; '
                f'thresh≤<span style="color:{color}">{row["threshold"]:.2f}</span>'
                f'&nbsp;&nbsp;→&nbsp;&nbsp;'
                f'{int(row["n_bets"]):,} bets &nbsp;·&nbsp; '
                f'{row["hit_rate"]:.1%} hit rate &nbsp;·&nbsp; '
                f'EV {row["ev_unit"]:+.4f}/unit &nbsp;·&nbsp; '
                f'<b>{row["units_won"]:+.2f} units</b> &nbsp;·&nbsp; '
                f'max DD {row["max_dd"]:.2f} ({"in-sample" if in_sample else "OOS holdout"})')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Sacks — Param Sweep</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.5; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 28px 24px 60px; }}
  header {{ margin-bottom: 28px; }}
  header h1 {{ font-size: 24px; font-weight: 700; color: #e6edf3; margin-bottom: 8px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 8px; }}
  .badge.oos {{ background: #1f3a5f; color: #79c0ff; border: 1px solid #1f6feb; }}
  .meta-row {{ margin-top: 10px; color: #8b949e; font-size: 12px; }}
  .meta-row span {{ margin-right: 20px; }}
  .cards-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .card-header {{ padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }}
  .card-dir {{ font-size: 15px; font-weight: 700; letter-spacing: 0.5px; }}
  .card-rank {{ font-size: 11px; opacity: 0.75; }}
  .card-body {{ padding: 16px; }}
  .stat-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .stat-label {{ color: #8b949e; font-size: 12px; }}
  .stat-val {{ font-size: 16px; font-weight: 700; padding: 3px 10px; border-radius: 6px; background: #21262d; }}
  .config-line {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid #21262d; font-size: 11px; color: #8b949e; font-family: monospace; }}
  .section-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
  .section-title {{ font-size: 16px; font-weight: 600; color: #e6edf3; }}
  .filter-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
  .filter-label {{ font-size: 12px; color: #8b949e; margin-right: 4px; }}
  .filter-btn {{ padding: 5px 14px; border-radius: 20px; border: 1px solid #30363d; background: #21262d; color: #8b949e; font-size: 12px; font-weight: 600; cursor: pointer; transition: all .15s; }}
  .filter-btn:hover {{ border-color: #8b949e; color: #e6edf3; }}
  .filter-btn.active {{ background: #388bfd; color: #fff; border-color: #388bfd; }}
  .table-wrapper {{ overflow-x: auto; border: 1px solid #21262d; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ user-select: none; }}
  th:hover {{ color: #e6edf3 !important; background: #30363d !important; }}
  tr.hidden {{ display: none; }}
  .rec-box {{ background: #1c2333; border: 1px solid #f0883e; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px; }}
  .rec-box h3 {{ font-size: 13px; color: #f0883e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
  .rec-params {{ font-family: monospace; font-size: 14px; color: #e6edf3; }}
  .rec-params span {{ color: #79c0ff; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🏈 NFL Sacks · Inference Parameter Sweep</h1>
    <div style="margin-top:8px">
      {'<span class="badge" style="background:#5a2500;color:#f0883e;border:1px solid #7d4220">⚠ In-sample — model trained on this data</span>' if in_sample else '<span class="badge oos">OOS — train 2024 · holdout 2025</span>'}
      <span class="badge oos">Line: 0.5 only · Under = 0 sacks · Over = 1+ sacks</span>
    </div>
    <div class="meta-row" style="margin-top:12px">
      <span>Model: M7 LR (prop_median_impl_over + qbhit_rate_L16 + sack_rate_Lcareer)</span>
      <span>Scoring rows: {scoring_rows:,} (player-game-book)</span>
      <span>With outcome: {bettable_rows:,}</span>
      <span>Combos: {len(all_df)}</span>
    </div>
  </header>

  <div class="rec-box">
    <h3>★ Recommended Production Config</h3>
    <div class="rec-params">{rec_line(rec)}</div>
  </div>
  <div class="rec-box" style="border-color:#58a6ff">
    <h3 style="color:#58a6ff">♦ Analyst Pick (highest EV/unit)</h3>
    <div class="rec-params">{rec_line(analyst, color="#79c0ff")}</div>
  </div>

  <div class="cards-grid" style="margin-top:24px">
    {card_html("UNDER", best_under)}
    {card_html("OVER",  best_over)}
    {card_html("BOTH",  best_both)}
  </div>

  <div class="table-section">
    <div class="section-header">
      <span class="section-title">All Parameter Combinations</span>
      <span style="font-size:12px;color:#8b949e">Sorted by Units Won ↓ &nbsp;·&nbsp; Click headers to re-sort &nbsp;·&nbsp; ★ = data pick &nbsp;·&nbsp; ♦ = analyst pick</span>
    </div>
    <div class="filter-row">
      <span class="filter-label">Direction:</span>
      <button class="filter-btn active" onclick="filterDir('all', this)">All</button>
      <button class="filter-btn" onclick="filterDir('under', this)">UNDER</button>
      <button class="filter-btn" onclick="filterDir('over', this)">OVER</button>
      <button class="filter-btn" onclick="filterDir('both', this)">BOTH</button>
    </div>
    <div class="table-wrapper">
    <table id="sweep-table" style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr>
        {"".join(f'<th onclick="sortTable({i})" style="cursor:pointer;padding:10px 12px;background:#21262d;color:#8b949e;font-weight:600;font-size:12px;text-align:center;border-bottom:2px solid #30363d;white-space:nowrap">{col} ↕</th>' for i, col in enumerate(["Direction","Edge","Threshold","# Bets","Hit Rate %","EV/Unit","% Over","Mean Edge pp","Units Won","Max DD","Impl (vig)","Impl (devig)"]))}
      </tr></thead>
      <tbody>
        {table_rows_html}
      </tbody>
    </table>
    </div>
  </div>
</div>
<script>
const sortDir = {{}};
function sortTable(col) {{
  const tbl = document.getElementById('sweep-table');
  const rows = Array.from(tbl.tBodies[0].rows);
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let va = a.cells[col].innerText.replace('%','').replace('+','');
    let vb = b.cells[col].innerText.replace('%','').replace('+','');
    va = va === '—' ? (sortDir[col] ? Infinity : -Infinity) : parseFloat(va) || va;
    vb = vb === '—' ? (sortDir[col] ? Infinity : -Infinity) : parseFloat(vb) || vb;
    return sortDir[col] ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }});
  rows.forEach(r => tbl.tBodies[0].appendChild(r));
}}
function filterDir(dir, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#sweep-table tbody tr').forEach(r => {{
    if (dir === 'all') {{ r.classList.remove('hidden'); return; }}
    r.classList.toggle('hidden', !r.classList.contains('dr-' + dir));
  }});
}}
</script>
</body>
</html>"""

    out_html = _out_html(in_sample)
    with open(out_html, "w") as f:
        f.write(html)
    print(f"  HTML report → {out_html}")


if __name__ == "__main__":
    main()
