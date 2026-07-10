"""
MLB Pitcher Strikeouts — Step 3d: Team Spread / Game Context Features
=====================================================================
Tests whether game-level moneyline / run-line features reduce OOF RMSE
on top of the v5 7-feature baseline.

Requires: 20260709_fetch_game_lines.py to have been run first.

New candidate features (derived from h2h + spreads):
  team_ml_prob  — no-vig win probability for pitcher's team (from h2h)
  opp_ml_prob   — 1 − team_ml_prob
  team_rl_prob  — no-vig prob pitcher's team covers −1.5 (from spreads)
  is_favorite   — binary: team_ml_prob > 0.5
  ml_delta      — team_ml_prob − 0.5 (signed distance from even; negative = underdog)

OOF design (same walk-forward folds as 20260705_model.py):
  Fold 1: train 2024,       test 2025
  Fold 2: train 2024+2025,  test 2026

Outputs:
  Appends a new <h2> section to:
    knowledge-base/raw/20260703-mlb-pitcher-strikeouts-v2.html
  Local CSVs:
    ~/Downloads/tmp/mlb_strikeouts/step3d_individual.csv
    ~/Downloads/tmp/mlb_strikeouts/step3d_combos.csv

Usage:
  python src/mlb_strikeouts_modeling/scripts/20260709_spread_features.py
  python src/mlb_strikeouts_modeling/scripts/20260709_spread_features.py --no-append
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
LABELED_KEY = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"
LINES_KEY   = "mlb/game_lines/mlb_game_lines.parquet"
HTML_PATH   = REPO_ROOT / "knowledge-base/raw/20260703-mlb-pitcher-strikeouts-v2.html"
OUT_DIR     = Path.home() / "Downloads/tmp/mlb_strikeouts"

TARGET   = "strikeouts"
OOF_FOLDS = [
    (2025, [2024]),
    (2026, [2024, 2025]),
]

V5_FEATURES = [
    "k_roll_career", "k_roll_c5",
    "opp_k_against_season", "is_home",
    "consensus_line",
    "over_price_bucket_fine",
    "under_price_bucket_fine",
]

NEW_FEATURES = [
    "team_ml_prob",
    "opp_ml_prob",
    "team_rl_prob",
    "is_favorite",
    "ml_delta",
]

# Combos to test: v5 baseline + one or two new features
COMBOS = [
    ("v5 baseline",                        V5_FEATURES),
    ("v5 + team_ml_prob",                  V5_FEATURES + ["team_ml_prob"]),
    ("v5 + opp_ml_prob",                   V5_FEATURES + ["opp_ml_prob"]),
    ("v5 + team_rl_prob",                  V5_FEATURES + ["team_rl_prob"]),
    ("v5 + is_favorite",                   V5_FEATURES + ["is_favorite"]),
    ("v5 + ml_delta",                      V5_FEATURES + ["ml_delta"]),
    ("v5 + team_ml_prob + team_rl_prob",   V5_FEATURES + ["team_ml_prob", "team_rl_prob"]),
    ("v5 + team_ml_prob + ml_delta",       V5_FEATURES + ["team_ml_prob", "ml_delta"]),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_labeled() -> pd.DataFrame:
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=LABELED_KEY)["Body"].read()
    df   = pd.read_parquet(BytesIO(body))
    if "season" not in df.columns:
        df["season"] = df.get("season_y", df.get("season_x"))
    return df.drop_duplicates(subset=["player_key", "game_date"], keep="first").copy()


def load_game_lines() -> pd.DataFrame:
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=LINES_KEY)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def build_dataset(labeled: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Join game lines to labeled dataset and engineer new features."""
    df = labeled.merge(
        lines[["event_id", "home_ml_prob", "away_ml_prob", "home_rl_prob", "away_rl_prob"]],
        on="event_id",
        how="left",
    )

    # Pitcher's team probabilities — depends on whether they're home or away
    df["team_ml_prob"] = np.where(df["is_home"] == 1, df["home_ml_prob"], df["away_ml_prob"])
    df["opp_ml_prob"]  = np.where(df["is_home"] == 1, df["away_ml_prob"], df["home_ml_prob"])
    df["team_rl_prob"] = np.where(df["is_home"] == 1, df["home_rl_prob"], df["away_rl_prob"])
    df["is_favorite"]  = (df["team_ml_prob"] > 0.5).astype(float)
    df["ml_delta"]     = df["team_ml_prob"] - 0.5

    match_rate = df["team_ml_prob"].notna().mean()
    print(f"  Game lines match rate: {match_rate:.1%}  ({df['team_ml_prob'].notna().sum()} / {len(df)} player-game rows)")

    return df


# ── OLS OOF ────────────────────────────────────────────────────────────────────

def ols_oof(df: pd.DataFrame, features: list[str]) -> dict:
    """Walk-forward OOF OLS. Returns averaged metrics across folds."""
    fold_metrics = []
    for test_season, train_seasons in OOF_FOLDS:
        cols = features + [TARGET]
        train = df[df["season"].isin(train_seasons)].dropna(subset=cols)
        test  = df[df["season"] == test_season].dropna(subset=cols)
        if len(train) < 100 or len(test) < 50:
            continue
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(train[features], train[TARGET])
        yhat = pipe.predict(test[features])
        rmse = float(np.sqrt(mean_squared_error(test[TARGET].values, yhat)))
        r2   = float(r2_score(test[TARGET].values, yhat))
        fold_metrics.append({"n": len(test), "rmse": rmse, "r2": r2, "fold": test_season})

    if not fold_metrics:
        return {"n": 0, "rmse": float("nan"), "r2": float("nan")}
    return {
        "n":    sum(m["n"]   for m in fold_metrics),
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "r2":   float(np.mean([m["r2"]   for m in fold_metrics])),
    }


# ── HTML generation ────────────────────────────────────────────────────────────

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"

_TH = "padding:7px 10px;background:#2c3e50;color:#fff;font-size:12px;text-align:left;white-space:nowrap"
_TD = "padding:6px 10px;border-bottom:1px solid #e0e0e0;font-size:12px"


def _delta_badge(delta: float) -> str:
    if pd.isna(delta):
        return "<span style='color:#888'>—</span>"
    color = "#276221" if delta < -0.001 else "#c0392b" if delta > 0.001 else "#888"
    sign  = "↓" if delta < -0.001 else "↑" if delta > 0.001 else "–"
    return f"<span style='color:{color};font-weight:bold'>{sign}{abs(delta):.4f}</span>"


def _rmse_cell(rmse: float, baseline: float) -> str:
    color = "#276221" if rmse < baseline - 0.0005 else "#c0392b" if rmse > baseline + 0.0005 else "#555"
    return f"<td style='{_TD};color:{color};font-weight:bold'>{rmse:.4f}</td>"


def build_html_section(
    individual_results: list[dict],
    combo_results:      list[dict],
    baseline_rmse:      float,
    match_rate:         float,
    n_events_fetched:   int,
    n_events_total:     int,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d ~%H:%M ET")

    # Individual rankings table
    ind_rows = ""
    for rec in individual_results:
        delta = rec["rmse"] - baseline_rmse
        ind_rows += (
            f"<tr>"
            f"<td style='{_TD};font-family:{_MONO}'>{rec['feature']}</td>"
            f"<td style='{_TD};text-align:center'>{rec['n']:,}</td>"
            f"<td style='{_TD};text-align:center;font-weight:bold'>{rec['rmse']:.4f}</td>"
            f"<td style='{_TD};text-align:center'>{rec['r2']:+.4f}</td>"
            f"<td style='{_TD};text-align:center'>{_delta_badge(delta)}</td>"
            f"<td style='{_TD}'>{rec['note']}</td>"
            f"</tr>\n"
        )

    # Combo results table
    combo_rows = ""
    for rec in combo_results:
        delta = rec["rmse"] - baseline_rmse
        is_base = rec["label"] == "v5 baseline"
        row_style = "background:#f0f7ff" if is_base else ""
        fw = "bold" if is_base else "normal"
        rmse_td = (
            f"<td style='{_TD};text-align:center;font-weight:bold'>{rec['rmse']:.4f}</td>"
            if is_base else _rmse_cell(rec["rmse"], baseline_rmse)
        )
        combo_rows += (
            f"<tr style='{row_style}'>"
            f"<td style='{_TD};font-weight:{fw}'>{rec['label']}</td>"
            f"<td style='{_TD};text-align:center'>{len(rec['features'])}</td>"
            f"<td style='{_TD};text-align:center'>{rec['n']:,}</td>"
            + rmse_td +
            f"<td style='{_TD};text-align:center'>{rec['r2']:+.4f}</td>"
            f"<td style='{_TD};text-align:center'>{_delta_badge(delta)}</td>"
            f"</tr>\n"
        )

    best_combo = min(combo_results, key=lambda x: x["rmse"])
    improvement = baseline_rmse - best_combo["rmse"]
    if improvement > 0.005:
        conclusion_class = "pass"
        conclusion_text  = (
            f"<strong>ACCEPT</strong> — best combo (<code>{best_combo['label']}</code>) "
            f"reduces OOF RMSE by <strong>{improvement:.4f}</strong> vs v5 baseline. "
            f"Add to feature set and retrain."
        )
    elif improvement > 0.001:
        conclusion_class = "note"
        conclusion_text  = (
            f"<strong>MARGINAL</strong> — best combo improves RMSE by only {improvement:.4f}. "
            f"Not worth adding; noise risk outweighs gain."
        )
    else:
        conclusion_class = "fail"
        conclusion_text  = (
            f"<strong>REJECT</strong> — no meaningful RMSE improvement (best delta: {improvement:.4f}). "
            f"Team spread does not add signal on top of v5 features. "
            f"<code>consensus_line</code> already captures expected-game-quality signal."
        )

    return f"""
<h2>Step 3d — Team Spread / Game Context Features &nbsp;<span style="font-size:13px;color:#6b7280;">{now_str}</span></h2>

<div class="note">
<strong>Hypothesis:</strong> The pitcher's team moneyline (win probability) encodes expected game script —
big favorites get more run support, stay in longer, and strikeout totals should be higher.
This tests whether h2h / run-line features add signal on top of v5's 7 existing features.
</div>

<h3>Data</h3>
<ul style="font-size:13px">
  <li>Source: Odds API historical endpoint (<code>h2h,spreads</code>) — 20 credits/event</li>
  <li>Snapshot time: 11am, 1pm, or 3pm ET on game day (first that returns ≥1 book)</li>
  <li>Events fetched: <strong>{n_events_fetched:,}</strong> of {n_events_total:,} unique events in labeled dataset</li>
  <li>Player-game match rate: <strong>{match_rate:.1%}</strong> (rows with game lines joined)</li>
  <li>No-vig method: proportional de-vig across both sides of h2h / spreads market</li>
  <li>Consensus: median no-vig prob across all US books returning that market</li>
</ul>

<h3>New features</h3>
<table style="width:auto;margin-bottom:16px">
  <tr>
    <th style="{_TH}">Feature</th>
    <th style="{_TH}">Description</th>
    <th style="{_TH}">Source</th>
    <th style="{_TH}">Book-invariant?</th>
  </tr>
  <tr><td style="{_TD};font-family:{_MONO}">team_ml_prob</td><td style="{_TD}">No-vig win prob for pitcher's team</td><td style="{_TD}">h2h</td><td style="{_TD}">Yes (consensus median)</td></tr>
  <tr style="background:#f9f9f9"><td style="{_TD};font-family:{_MONO}">opp_ml_prob</td><td style="{_TD}">1 − team_ml_prob (opponent win prob)</td><td style="{_TD}">h2h</td><td style="{_TD}">Yes</td></tr>
  <tr><td style="{_TD};font-family:{_MONO}">team_rl_prob</td><td style="{_TD}">No-vig prob pitcher's team covers −1.5 run line</td><td style="{_TD}">spreads</td><td style="{_TD}">Yes (consensus median)</td></tr>
  <tr style="background:#f9f9f9"><td style="{_TD};font-family:{_MONO}">is_favorite</td><td style="{_TD}">Binary: team_ml_prob &gt; 0.5</td><td style="{_TD}">derived</td><td style="{_TD}">Yes</td></tr>
  <tr><td style="{_TD};font-family:{_MONO}">ml_delta</td><td style="{_TD}">team_ml_prob − 0.5 (signed distance from even)</td><td style="{_TD}">derived</td><td style="{_TD}">Yes</td></tr>
</table>

<h3>Step 3a — Individual feature rankings (OLS, walk-forward OOF)</h3>
<p style="font-size:12px;color:#555">Baseline: <code>consensus_line</code> alone. Lower RMSE = better. ↓ delta = improvement.</p>
<table>
  <tr>
    <th style="{_TH}">Feature</th>
    <th style="{_TH}">n (OOF rows)</th>
    <th style="{_TH}">RMSE</th>
    <th style="{_TH}">R²</th>
    <th style="{_TH}">vs baseline</th>
    <th style="{_TH}">Notes</th>
  </tr>
  {ind_rows}
</table>

<h3>Step 3b — Combo tests (v5 baseline + spread features)</h3>
<p style="font-size:12px;color:#555">v5 baseline RMSE = <strong>{baseline_rmse:.4f}</strong>. Accept threshold: RMSE improvement ≥ 0.005.</p>
<table>
  <tr>
    <th style="{_TH}">Model</th>
    <th style="{_TH}">N features</th>
    <th style="{_TH}">n (OOF rows)</th>
    <th style="{_TH}">OOF RMSE</th>
    <th style="{_TH}">R²</th>
    <th style="{_TH}">vs v5 baseline</th>
  </tr>
  {combo_rows}
</table>

<div class="{conclusion_class}" style="margin-top:16px">
  <strong>Conclusion:</strong> {conclusion_text}
</div>
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-append", action="store_true", help="Print HTML section, do not append to file")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading labeled dataset...")
    labeled = load_labeled()
    print(f"  {len(labeled):,} player-game rows · seasons: {sorted(labeled['season'].unique())}")

    print("Loading game lines...")
    lines = load_game_lines()
    print(f"  {len(lines):,} events with game lines")

    print("Building dataset with spread features...")
    df = build_dataset(labeled, lines)
    match_rate = float(df["team_ml_prob"].notna().mean())

    print("\nStep 3a — Individual feature rankings...")
    # Baseline: consensus_line alone
    baseline_ind = ols_oof(df, ["consensus_line"])
    baseline_rmse = baseline_ind["rmse"]
    print(f"  consensus_line alone: RMSE={baseline_rmse:.4f}")

    individual_results = []
    for feat in NEW_FEATURES:
        sub = df.dropna(subset=[feat])
        metrics = ols_oof(sub, [feat])
        delta   = metrics["rmse"] - baseline_rmse
        note    = "better than baseline" if delta < -0.001 else "no improvement alone" if delta > 0.001 else "≈ baseline"
        rec = {"feature": feat, **metrics, "note": note}
        individual_results.append(rec)
        print(f"  {feat:<20} RMSE={metrics['rmse']:.4f}  Δ={delta:+.4f}  R²={metrics['r2']:+.4f}")

    ind_df = pd.DataFrame(individual_results)
    ind_df.to_csv(OUT_DIR / "step3d_individual.csv", index=False)

    print("\nStep 3b — Combo tests...")
    combo_results = []
    for label, features in COMBOS:
        sub     = df.dropna(subset=features)
        metrics = ols_oof(sub, features)
        delta   = metrics["rmse"] - baseline_rmse
        rec = {"label": label, "features": features, **metrics}
        combo_results.append(rec)
        marker = "✓" if delta < -0.005 else ("~" if abs(delta) <= 0.005 else "✗")
        print(f"  {marker} {label:<42} RMSE={metrics['rmse']:.4f}  Δ={delta:+.4f}")

    combo_df = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in combo_results])
    combo_df.to_csv(OUT_DIR / "step3d_combos.csv", index=False)

    # Use v5 baseline RMSE (not consensus_line-alone) as the acceptance threshold
    v5_baseline_rmse = next(r["rmse"] for r in combo_results if r["label"] == "v5 baseline")

    html_section = build_html_section(
        individual_results = individual_results,
        combo_results      = combo_results,
        baseline_rmse      = v5_baseline_rmse,
        match_rate         = match_rate,
        n_events_fetched   = len(lines),
        n_events_total     = labeled["event_id"].nunique() if "event_id" in labeled.columns else 0,
    )

    if args.no_append:
        print("\n--- HTML section (not appended) ---")
        print(html_section[:2000], "...")
    else:
        with open(HTML_PATH, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(html_section)
        print(f"\nAppended HTML section → {HTML_PATH}")

    print(f"CSVs saved → {OUT_DIR}/step3d_*.csv")


if __name__ == "__main__":
    main()
