"""
Over Mirror — MLB Total Bases UNDER 1.5 (2024-2026)

Sanity check: for every per-book row qualifying for the UNDER strategy
(edge_under >= 5%, line 1.5), compute P&L as if the OVER side were taken.

Single coherent source built on-the-fly:
  1. S3 Statcast actuals  → rolling features
  2. S3 market_raw (per-book) → joined with features, scored with model
  No settled history stitching.

Expected: OVER ROI clearly negative. Positive OVER ROI = model bets against itself profitably = alarming.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

S3_BUCKET  = "the-odds-api-mt"
ACT_KEY    = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
MKT_KEY    = "mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet"
MODEL_KEY  = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"

MIN_BET_EDGE   = 0.05
TARGET_LINE    = 1.5
ROLLING_WINDOWS = [1, 3, 5, 10, 20]


# ── Name normalisation ────────────────────────────────────────────────────────

MANUAL_MAP = {
    "daniel vogelbach":   "Dan Vogelbach",
    "michael a taylor":   "Michael Taylor",
    "max muncy (2002)":   "Max Muncy",
    "diego a castillo":   "Diego Castillo",
    "james jarvis":       "Jim Jarvis",
    "donnie walton":      "Donovan Walton",
    "josh kuroda-grauer": "Joshua Kuroda-Grauer",
}

_MANUAL_NORM: dict[str, str] = {}


def _normalize_raw(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


def _init_manual_norm() -> None:
    global _MANUAL_NORM
    _MANUAL_NORM = {_normalize_raw(k): _normalize_raw(v) for k, v in MANUAL_MAP.items()}


def normalize_name(name: str) -> str:
    n = _normalize_raw(name)
    return _MANUAL_NORM.get(n, n)


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3", region_name="us-east-2")


def load_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def load_bundle(key: str) -> dict:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return joblib.load(BytesIO(body))


# ── Rolling features (mirrors build_spine.py) ─────────────────────────────────

def build_rolling_features(actuals: pd.DataFrame) -> pd.DataFrame:
    actuals = actuals.copy()
    actuals["game_date"] = pd.to_datetime(actuals["game_date"])
    actuals = actuals.sort_values(["name_norm", "game_date"]).reset_index(drop=True)

    frames = []
    for _, grp in actuals.groupby("name_norm", sort=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        grp["games_played_career"] = range(len(grp))
        for w in ROLLING_WINDOWS:
            grp[f"tb_L{w}"] = grp["total_bases"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"hr_L{w}"] = grp["home_runs"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"ab_L{w}"] = grp["ab"].shift(1).rolling(w, min_periods=1).mean()
        grp["tb_Lcareer"] = grp["total_bases"].shift(1).expanding().mean()
        grp["hr_Lcareer"] = grp["home_runs"].shift(1).expanding().mean()
        grp["ab_Lcareer"] = grp["ab"].shift(1).expanding().mean()
        grp["tb_Lseason"] = (
            grp.groupby("season")["total_bases"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )
        frames.append(grp)

    return pd.concat(frames, ignore_index=True)


# ── Scoring (mirrors 20260710_grid_search_raw_edge.py) ───────────────────────

def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    features     = bundle["features_numeric"]
    calib_models = bundle.get("calib_models", {})

    unique_pg = (
        df[["name_norm", "game_date", "line"] + features]
        .drop_duplicates(subset=["name_norm", "game_date", "line"])
        .dropna(subset=features)
        .copy()
    )
    X = unique_pg[features].values.astype(float)
    unique_pg["y_hat"] = model.predict(scaler.transform(X)).astype(float)

    pieces = []
    for line, calib in calib_models.items():
        sub = unique_pg[unique_pg["line"] == line].copy()
        if sub.empty:
            continue
        proba = calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1]
        sub["p_model_over"]  = np.clip(proba, 0.01, 0.99)
        sub["p_model_under"] = 1.0 - sub["p_model_over"]
        pieces.append(sub)

    p_df = pd.concat(pieces, ignore_index=True)
    df   = df.merge(
        p_df[["name_norm", "game_date", "line", "p_model_under"]],
        on=["name_norm", "game_date", "line"],
        how="inner",
    )
    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["edge_under"]     = df["p_model_under"] - df["raw_prob_under"]
    return df


# ── P&L ───────────────────────────────────────────────────────────────────────

def _over_pnl(actual_tb, over_price) -> float:
    return float(over_price) - 1.0 if actual_tb >= 2 else -1.0


def _under_pnl(actual_tb, under_price) -> float:
    return float(under_price) - 1.0 if actual_tb <= 1 else -1.0


# ── Reporting ─────────────────────────────────────────────────────────────────

W = 84
_HDR = (
    f"  {'Group':<22} {'n':>7}  {'under%':>7}  {'under_net':>11}  {'under_roi':>9}  "
    f"{'over%':>7}  {'over_net':>11}  {'over_roi':>9}"
)


def _row(label: str, gdf: pd.DataFrame) -> str:
    n = len(gdf)
    u_wins = int((gdf["actual_tb"] <= 1).sum())
    o_wins = int((gdf["actual_tb"] >= 2).sum())
    u_net  = gdf["under_pnl"].sum()
    o_net  = gdf["over_pnl"].sum()
    return (
        f"  {label:<22} {n:>7}  {u_wins/n:>7.1%}  {u_net:>+11.2f}  {u_net/n:>+8.1%}  "
        f"{o_wins/n:>7.1%}  {o_net:>+11.2f}  {o_net/n:>+8.1%}"
    )


def print_table(title: str, df: pd.DataFrame, group_col: str, chronological: bool = False) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")
    print(_HDR)
    print(f"  {'-' * (W - 2)}")
    groups = list(df.groupby(group_col, sort=chronological, observed=True))
    if not chronological:
        groups.sort(key=lambda x: x[1]["under_pnl"].sum(), reverse=True)
    for label, gdf in groups:
        print(_row(str(label), gdf))
    print(f"  {'-' * (W - 2)}")
    print(_row("TOTAL", df))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_manual_norm()

    # ── Actuals + rolling features ────────────────────────────────────────────
    print("Loading Statcast actuals from S3…")
    act = load_parquet(ACT_KEY)
    act = act[act["ab"] >= 1].copy()
    act["name_norm"] = act["player_name"].map(normalize_name)
    act["game_date"] = pd.to_datetime(act["game_date"])
    act["season"]    = act["game_date"].dt.year

    # Aggregate doubleheaders (market prices single-game props; actuals must match)
    sum_cols = ["total_bases", "home_runs", "ab", "hits"]
    act = (
        act.groupby(["name_norm", "game_date"], sort=False)
        .agg(
            **{c: (c, "sum") for c in sum_cols if c in act.columns},
            player_name = ("player_name", "first"),
            season      = ("season",      "first"),
            n_games     = ("game_date",   "count"),
        )
        .reset_index()
    )
    act["is_doubleheader"] = (act["n_games"] > 1).astype(int)

    print(f"  {len(act):,} player-dates | DH rows: {act['is_doubleheader'].sum():,}")
    print("Building rolling features…")
    feat = build_rolling_features(act)
    feat["game_date"] = feat["game_date"].dt.strftime("%Y-%m-%d")

    # ── Market raw (per-book) ─────────────────────────────────────────────────
    print("Loading market_raw from S3…")
    mkt = load_parquet(MKT_KEY)
    mkt = mkt[
        (mkt["market_key"] == "batter_total_bases") &
        (mkt["line"].astype(float) == TARGET_LINE) &
        (mkt["over_price"].notna())  & (mkt["over_price"]  > 1) &
        (mkt["under_price"].notna()) & (mkt["under_price"] > 1)
    ].copy()
    mkt["name_norm"] = mkt["player_name"].map(normalize_name)
    mkt["game_date"] = pd.to_datetime(mkt["game_date"]).dt.strftime("%Y-%m-%d")

    # One row per (name_norm, game_date, bookmaker, line): last snapshot
    mkt = (
        mkt.sort_values("snapshot")
        .groupby(["name_norm", "game_date", "bookmaker", "line"], sort=False)
        .last()
        .reset_index()
    )
    print(f"  {len(mkt):,} per-book rows after dedup")

    # ── Join features → market ────────────────────────────────────────────────
    feat_cols = (
        ["name_norm", "game_date", "total_bases", "is_doubleheader",
         "tb_Lcareer", "hr_Lcareer", "ab_Lcareer", "tb_Lseason"]
        + [f"tb_L{w}" for w in ROLLING_WINDOWS]
        + [f"hr_L{w}" for w in ROLLING_WINDOWS]
        + [f"ab_L{w}" for w in ROLLING_WINDOWS]
    )
    df = mkt.merge(feat[feat_cols], on=["name_norm", "game_date"], how="inner")

    # Drop doubleheader rows (market posts single-game props; DH actuals are summed — mismatch)
    dh = df["is_doubleheader"].sum()
    df = df[df["is_doubleheader"] == 0].copy()
    print(f"  After feature join + DH drop: {len(df):,} rows ({dh:,} DH rows removed)")

    # min_line/max_line per player-game (model features)
    line_range = (
        df.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    df = df.merge(line_range, on=["name_norm", "game_date"], how="left")

    # ── Score ─────────────────────────────────────────────────────────────────
    print("Loading model + scoring…")
    bundle = load_bundle(MODEL_KEY)
    df = score(df, bundle)

    # ── Filter ────────────────────────────────────────────────────────────────
    df = df[df["edge_under"] >= MIN_BET_EDGE].copy()
    print(f"  Qualifying rows (edge_under >= {MIN_BET_EDGE:.0%}): {len(df):,}")

    # ── P&L ───────────────────────────────────────────────────────────────────
    df["actual_tb"] = df["total_bases"].astype(float)
    df["over_pnl"]  = df.apply(lambda r: _over_pnl(r["actual_tb"],  r["over_price"]),  axis=1)
    df["under_pnl"] = df.apply(lambda r: _under_pnl(r["actual_tb"], r["under_price"]), axis=1)

    df["season"]    = pd.to_datetime(df["game_date"]).dt.year
    df["month"]     = df["game_date"].str[:7]

    print(f"\n{'=' * W}")
    print(f"  DATASET: {len(df):,} rows | {df['game_date'].min()} → {df['game_date'].max()}")
    print(f"  Seasons: {sorted(df['season'].unique().tolist())}")
    print(f"{'=' * W}")

    print_table("Table 1: Monthly",         df, "month",     chronological=True)
    print_table("Table 2: By season",        df, "season",    chronological=True)

    df["edge_bucket"] = pd.cut(
        df["edge_under"],
        bins=[0.05, 0.10, 0.15, 1.0],
        labels=["[5-10%)", "[10-15%)", "[15%+]"],
        right=False,
    )
    print_table("Table 3: Edge bucket",       df.dropna(subset=["edge_bucket"]), "edge_bucket", chronological=True)
    print_table("Table 4: By bookmaker",      df, "bookmaker")

    print(f"\n{'=' * W}")
    print("  Done.")


if __name__ == "__main__":
    main()
