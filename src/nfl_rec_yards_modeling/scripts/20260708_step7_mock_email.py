"""
Step 7 — Full 3-section mock email for NFL WR/TE receiving yards.

  Section 1 (top): Today's plays       — per-book rows, all scored players
  Section 2:       Yesterday's results  — settled bets with actual outcomes
  Section 3:       All-time results     — OOF backtested through 2024 season

All data sourced from the labeled parquet + model artifacts in ~/Downloads/tmp/.
Synthetic per-book rows: 5 books per player-game, slight per-book market variation.
Odds computed from market probs using a standard -110/-110 vig assumption.

Output: ~/Downloads/tmp/nfl_rec_yards_mock_email.html
"""

import os
import sys
import unittest.mock as mock
import warnings

# Patch external deps before importing run_pipeline
for mod in ["boto3", "botocore", "botocore.exceptions", "requests"]:
    sys.modules.setdefault(mod, mock.MagicMock())

sys.path.insert(0, os.path.dirname(__file__))

import html as html_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom

import run_pipeline as rp

warnings.filterwarnings("ignore")

LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_artifacts"
OUT_HTML     = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_mock_email.html"

ASSUMED_VIG = 0.0476   # standard -110/-110 vig
N_BOOT      = 5_000
RNG         = np.random.default_rng(42)

MOCK_BOOKS = [
    "draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet",
]

# Per-book jitter: slight offset to fair probs so each book looks distinct
BOOK_JITTER = [-0.006, -0.003, 0.0, 0.003, 0.006]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _imp_to_amer(imp: float) -> int:
    imp = min(max(float(imp), 0.01), 0.99)
    if imp >= 0.5:
        return round(-imp / (1 - imp) * 100)
    else:
        return round((1 - imp) / imp * 100)


def _load_artifacts() -> dict:
    return {
        "ols":       joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib"),
        "residuals": np.load(ARTIFACT_DIR / "residuals.npy"),
        "nb_coefs":  np.load(ARTIFACT_DIR / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0]),
    }


def _score(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    ols, residuals = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]

    result = df.copy()
    mask   = result[rp.BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]

    if idx.empty:
        result["ols_pred"]       = np.nan
        result["p_hybrid"]       = np.nan
        result["p_market"]       = result.get("market_under_prob", np.nan)
        result["edge"]           = np.nan
        result["recommendation"] = "PASS"
        return result

    X    = result.loc[idx, rp.BEST_FEATS].to_numpy(dtype=float)
    line = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.clip(np.exp(X_const @ nb_coefs), 1e-3, None)
    n_nb     = 1.0 / nb_alpha
    p_nb     = nbinom.cdf(np.floor(line).astype(int), n=n_nb, p=n_nb / (n_nb + nb_mu))
    samp     = RNG.choice(residuals, size=(len(ols_pred), N_BOOT))
    p_bt     = ((ols_pred[:, None] + samp) <= line[:, None]).mean(axis=1)
    p_hyb    = np.clip(np.where(line < rp.HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb), 0.01, 0.99)

    p_mkt = result.loc[idx, "market_under_prob"].to_numpy(dtype=float)
    edge  = p_hyb - p_mkt
    rec   = np.select(
        [edge > rp.EDGE_THRESHOLD, edge < -rp.EDGE_THRESHOLD],
        ["UNDER", "OVER"], default="PASS",
    )

    result.loc[idx, "ols_pred"]       = np.round(ols_pred, 3)
    result.loc[idx, "p_hybrid"]       = np.round(p_hyb, 4)
    result.loc[idx, "p_market"]       = np.round(p_mkt, 4)
    result.loc[idx, "edge"]           = np.round(edge, 4)
    result.loc[idx, "recommendation"] = rec
    return result


def _expand_per_book(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each player-game row to one row per mock book with synthetic odds."""
    rows = []
    for _, r in df.iterrows():
        fair_under = float(r["market_under_prob"])
        fair_over  = 1.0 - fair_under
        for book, jitter in zip(MOCK_BOOKS, BOOK_JITTER):
            bu = min(max(fair_under + jitter, 0.02), 0.98)
            bo = 1.0 - bu
            raw_under = bu * (1 + ASSUMED_VIG)
            raw_over  = bo * (1 + ASSUMED_VIG)
            raw_total = raw_under + raw_over
            rows.append({
                **r.to_dict(),
                "book":                  book,
                "market_under_prob":     bu,
                "market_over_prob":      bo,
                "raw_under_prob":        raw_under,
                "raw_over_prob":         raw_over,
                "raw_total":             raw_total,
                "over_price":            _imp_to_amer(raw_over),
                "under_price":           _imp_to_amer(raw_under),
                "consensus_over_price":  _imp_to_amer(raw_over),
                "consensus_under_price": _imp_to_amer(raw_under),
                "n_books":               len(MOCK_BOOKS),
            })
    return pd.DataFrame(rows)


def _prep_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns expected by scoring + HTML builders."""
    df = df.copy()
    df["pos_TE"]      = (df["position"] == "TE").astype(int)
    df["team"]        = df["recent_team"]
    df["opponent"]    = df["player_opponent"]
    df["player_name"] = df["player_name_norm"].str.title()
    df["player_norm"] = df["player_name_norm"]
    df["event_id"]    = df["game_id"].astype(str)
    df["home_team"]   = df.apply(
        lambda r: r["recent_team"] if r["is_home"] else r["player_opponent"], axis=1
    )
    df["away_team"]   = df.apply(
        lambda r: r["player_opponent"] if r["is_home"] else r["recent_team"], axis=1
    )
    # Assign game times: randomize between 1pm/4pm/8pm slots for realism
    game_slots = {0: "1:00 PM ET", 1: "4:25 PM ET", 2: "8:20 PM ET"}
    game_time_map = {
        eid: game_slots[i % 3]
        for i, eid in enumerate(df["event_id"].unique())
    }
    df["game_time_et"]  = df["event_id"].map(game_time_map)
    df["game_sort_key"] = df["event_id"]
    return df


def _filter_bets(scored: pd.DataFrame) -> pd.DataFrame:
    bets = scored[
        (scored["recommendation"] == rp.DIRECTION) &
        (scored["edge"].abs() >= rp.EDGE_THRESHOLD) &
        (scored["offered_line"] >= rp.LINE_MIN) &
        (scored["offered_line"] <= rp.LINE_MAX) &
        scored["ols_pred"].notna()
    ].copy()
    if not bets.empty:
        bets = (
            bets.sort_values("consensus_over_price", ascending=False)
            .drop_duplicates(subset=["player_norm", "offered_line"])
        )
    bets["streak"]             = 0
    bets["cold_streak_warning"] = False
    return bets


def _build_history(base: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Score the full base (one book per row) and settle vs actual outcomes."""
    df = base.copy()
    df["book"]                  = "draftkings"
    df["market_over_prob"]      = 1.0 - df["market_under_prob"]
    df["raw_under_prob"]        = df["market_under_prob"] * (1 + ASSUMED_VIG)
    df["raw_over_prob"]         = df["market_over_prob"]  * (1 + ASSUMED_VIG)
    df["raw_total"]             = df["raw_under_prob"] + df["raw_over_prob"]
    df["consensus_over_price"]  = df["raw_over_prob"].apply(_imp_to_amer)
    df["consensus_under_price"] = df["raw_under_prob"].apply(_imp_to_amer)
    df["n_books"]               = 1

    scored = _score(df, artifacts)
    bets = scored[
        (scored["recommendation"] == rp.DIRECTION) &
        (scored["edge"].abs() >= rp.EDGE_THRESHOLD) &
        (scored["offered_line"] >= rp.LINE_MIN) &
        scored["ols_pred"].notna()
    ].copy()
    if bets.empty:
        return pd.DataFrame()

    bets["outcome"] = (bets["receiving_yards"] > bets["offered_line"]).map(
        {True: "win", False: "loss"}
    )
    return bets[["player_norm", "team", "opponent", "season", "gameday",
                 "offered_line", "outcome", "consensus_over_price",
                 "p_hybrid", "p_market", "edge", "book"]].copy()


def _build_yesterday_settled(
    yest_df: pd.DataFrame, artifacts: dict, yesterday: str
) -> pd.DataFrame:
    """Score yesterday's players (single book) and settle vs actual outcomes."""
    if yest_df.empty:
        return pd.DataFrame()
    df = yest_df.copy()
    df["book"]                  = "draftkings"
    df["market_over_prob"]      = 1.0 - df["market_under_prob"]
    df["raw_under_prob"]        = df["market_under_prob"] * (1 + ASSUMED_VIG)
    df["raw_over_prob"]         = df["market_over_prob"]  * (1 + ASSUMED_VIG)
    df["raw_total"]             = df["raw_under_prob"] + df["raw_over_prob"]
    df["consensus_over_price"]  = df["raw_over_prob"].apply(_imp_to_amer)
    df["consensus_under_price"] = df["raw_under_prob"].apply(_imp_to_amer)
    df["n_books"]               = 1

    scored = _score(df, artifacts)
    bets = scored[
        (scored["recommendation"] == rp.DIRECTION) &
        (scored["edge"].abs() >= rp.EDGE_THRESHOLD) &
        scored["ols_pred"].notna()
    ].copy()
    if bets.empty:
        return pd.DataFrame()

    bets["outcome"]      = (bets["receiving_yards"] > bets["offered_line"]).map(
        {True: "win", False: "loss"}
    )
    bets["actual_yards"] = bets["receiving_yards"].fillna(0)
    bets["gameday"]      = yesterday
    bets["recommendation"] = rp.DIRECTION
    return bets[["player_name", "player_norm", "team", "opponent", "season", "gameday",
                 "book", "offered_line", "actual_yards", "outcome", "recommendation",
                 "consensus_over_price", "consensus_under_price",
                 "p_hybrid", "p_market", "edge"]].copy()


def _pick_game_days(base: pd.DataFrame) -> tuple[str, str]:
    """Find a 'today' with ≥15 scoreable rows and a valid 'yesterday' before it."""
    days_2024 = sorted(
        base[(base["season"] == 2024)]["gameday"].dropna().unique()
    )
    today = None
    for day in reversed(days_2024):
        n_scoreable = (
            base[(base["gameday"] == day) & base[rp.BEST_FEATS].notna().all(axis=1)]
            .shape[0]
        )
        if n_scoreable >= 15:
            today = day
            break

    if today is None:
        raise RuntimeError("Could not find a suitable 'today' in 2024 data.")

    today_idx = days_2024.index(today)
    yesterday = None
    for day in reversed(days_2024[:today_idx]):
        if base[base["gameday"] == day].shape[0] >= 5:
            yesterday = day
            break

    return today, yesterday


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nNFL Rec Yards — Step 7 Mock Email")
    print("=" * 45)

    print("\n  Loading labeled data...")
    base = pd.read_parquet(LABELED_PATH)
    base = base[base["position"].isin(["WR", "TE"])].copy()
    base = _prep_base_cols(base)
    print(f"    {len(base):,} rows  |  seasons: {sorted(base['season'].unique())}")

    print("\n  Loading artifacts...")
    artifacts = _load_artifacts()

    today, yesterday = _pick_game_days(base)
    print(f"  Today:     {today}")
    print(f"  Yesterday: {yesterday}")

    # ── Today (Section 1) ────────────────────────────────────────────────────
    print("\n  Scoring today's rows (per-book expansion)...")
    today_base   = base[base["gameday"] == today].copy()
    today_pb     = _expand_per_book(today_base)
    today_scored = _score(today_pb, artifacts)
    bets         = _filter_bets(today_scored)
    print(f"    {today_scored['player_norm'].nunique()} players scored  |  {len(bets)} qualifying plays")

    # ── Yesterday (Section 2) ────────────────────────────────────────────────
    print("\n  Building yesterday's settled bets...")
    yest_base     = base[base["gameday"] == yesterday].copy() if yesterday else pd.DataFrame()
    yest_settled  = _build_yesterday_settled(yest_base, artifacts, yesterday or "")
    if not yest_settled.empty:
        n_w = (yest_settled["outcome"] == "win").sum()
        n_l = (yest_settled["outcome"] == "loss").sum()
        print(f"    {len(yest_settled)} bets settled  |  {n_w}W {n_l}L")
    else:
        print("    No qualifying bets on yesterday.")

    # ── All-time (Section 3) ─────────────────────────────────────────────────
    print("\n  Building all-time history (OOF, single book)...")
    history = _build_history(base, artifacts)
    if not history.empty:
        n_w = (history["outcome"] == "win").sum()
        n_l = (history["outcome"] == "loss").sum()
        print(f"    {len(history):,} settled bets  |  {n_w}W {n_l}L")
    else:
        print("    No history found.")

    # ── Build email ───────────────────────────────────────────────────────────
    print("\n  Building HTML...")
    html_body = rp.build_recommendations_html(
        all_scored=today_scored,
        bets=bets,
        gameday=today,
        history=history,
        yesterday_settled=yest_settled,
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html_body, encoding="utf-8")

    print(f"\n  Mock email written → {OUT_HTML}")
    print(f"\n  To view:  open {OUT_HTML}")
    print()


if __name__ == "__main__":
    main()
