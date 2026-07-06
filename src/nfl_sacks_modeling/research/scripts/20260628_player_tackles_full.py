"""
v2_player_tackles_full.py — full player_tackles_assists analysis pipeline

Usage:
  python v2_player_tackles_full.py --fake-data              # test with hardcoded data
  python v2_player_tackles_full.py --season 2025            # real run from S3
  python v2_player_tackles_full.py --season 2025 --player "Roquan Smith"

Requires:
  - AWS credentials configured (for S3)
  - ODDS_API_KEY in .env (only used if rebuilding backfill, not for this script)
  - The 2025 NFL backfill already in S3 under:
      s3://the-odds-api-mt/nfl/all_markets/{season}/{nfl_game_id}.parquet
    with BOOKMAKERS = ["draftkings", "fanduel"]

Fake data flag:
  --fake-data uses hardcoded game/player/line data that mimics real API output,
  allowing full pipeline testing without S3 or API credits.
"""

import argparse
import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import Pipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

import nfl_data_py as nfl

S3_BUCKET  = "the-odds-api-mt"
SEASON     = 2025
BOOKMAKERS = ["draftkings", "fanduel"]
MARKET     = "player_tackles_assists"

# ── Fake data ─────────────────────────────────────────────────────────────────
# Compact definition: player → [(week, line, dk_over, dk_under, fd_over, fd_under)]
# DNP weeks are simply omitted — the walkthrough detects them from the schedule.
# build_fake_odds() expands this into raw API rows using the real schedule for game IDs.

FAKE_PLAYER_TEAMS = {
    "Myles Garrett": "CLE",
    "Roquan Smith":  "BAL",
    "Jordyn Brooks": "MIA",
    "Maxx Crosby":   "LV",
}

FAKE_PLAYER_LINES = {
    # (week, line, dk_over, dk_under, fd_over, fd_under)
    "Myles Garrett": [
        ( 1, 2.5, -120, -110, -115, -115),
        ( 2, 3.5,  105, -135,  110, -140),
        ( 3, 3.5, -115, -115, -112, -118),
        ( 4, 3.5, -130,  100, -125,  105),
        ( 5, 3.0, -110, -120, -108, -122),
        ( 6, 2.5,  110, -140,  108, -138),
        ( 7, 3.5, -115, -115, -110, -120),
        ( 8, 3.5, -120, -110, -118, -112),
        (10, 3.0,  100, -130,  102, -132),
        (11, 3.5, -115, -115, -112, -118),
        (12, 3.5, -125, -105, -122, -108),
        (13, 3.5, -130,  100, -128,  102),
        (14, 3.5, -140,  110, -135,  106),
        (15, 3.5, -120, -110, -118, -112),
        (16, 3.0,  105, -135,  108, -138),
        (17, 2.5,  110, -140,  112, -142),
        (18, 2.5,  115, -145,  112, -142),
    ],
    "Roquan Smith": [
        # DNP wks 5, 6 — omitted
        ( 1,  9.5, -115, -115, -112, -118),
        ( 2,  8.5, -125, -105, -122, -108),
        ( 3,  9.5, -110, -120, -108, -122),
        ( 4,  8.5,  100, -130,  102, -132),
        ( 8,  8.5, -130,  100, -128,  102),
        ( 9,  8.5,  100, -130,  102, -132),
        (10,  9.5,  110, -140,  108, -138),
        (11,  8.5, -145,  115, -148,  118),
        (12,  8.5, -105, -125, -108, -122),
        (13,  8.5, -120, -110, -118, -112),
        (14,  9.5,  105, -135,  108, -138),
        (15,  9.5,  105, -135,  108, -138),
        (16,  8.5, -135,  105, -132,  102),
        (17,  9.5, -135,  105, -132,  102),
        (18,  9.5, -110, -120, -108, -122),
    ],
    "Jordyn Brooks": [
        ( 1, 10.5, -115, -115, -112, -118),
        ( 2,  9.5, -120, -110, -118, -112),
        ( 3, 10.5, -110, -120, -108, -122),
        ( 4, 11.5, -115, -115, -112, -118),
        ( 5, 10.5, -120, -110, -118, -112),
        ( 6,  9.5, -110, -120, -108, -122),
        ( 7,  9.5, -105, -125, -108, -122),
        ( 8, 10.5, -115, -115, -112, -118),
        ( 9,  9.5, -110, -120, -108, -122),
        (10, 10.5, -115, -115, -112, -118),
        (11, 11.5, -120, -110, -118, -112),
        (13, 10.5, -115, -115, -112, -118),
        (14,  9.5, -110, -120, -108, -122),
        (15, 10.5, -120, -110, -118, -112),
        (16, 10.5, -115, -115, -112, -118),
        (17,  9.5, -110, -120, -108, -122),
        (18, 10.5, -115, -115, -112, -118),
    ],
    "Maxx Crosby": [
        # DNP wks 17, 18 — omitted
        ( 1, 3.5, -125, -105, -122, -108),
        ( 2, 3.5, -115, -115, -112, -118),
        ( 3, 4.5, -110, -120, -108, -122),
        ( 4, 4.5, -120, -110, -118, -112),
        ( 5, 3.5, -115, -115, -112, -118),
        ( 6, 3.5, -110, -120, -108, -122),
        ( 7, 3.5, -165,  135, -160,  130),
        ( 9, 3.5, -115, -115, -112, -118),
        (10, 4.5, -120, -110, -118, -112),
        (11, 4.5, -115, -115, -112, -118),
        (12, 5.5, -110, -120, -108, -122),
        (13, 5.5, -115, -115, -112, -118),
        (14, 4.5,  115, -145,  112, -142),
        (15, 4.5,  100, -130,  102, -132),
        (16, 4.5,  115, -145,  112, -142),
    ],
}


def build_fake_odds(sched: pd.DataFrame) -> pd.DataFrame:
    """Expand FAKE_PLAYER_LINES into raw odds rows using the real schedule for game IDs."""
    # Build week → game_id lookup per team
    home = sched[["game_id","week","home_team"]].rename(columns={"home_team":"team"})
    away = sched[["game_id","week","away_team"]].rename(columns={"away_team":"team"})
    team_week_to_gid = (pd.concat([home, away])
                        .set_index(["team","week"])["game_id"].to_dict())

    rows = []
    for player, lines in FAKE_PLAYER_LINES.items():
        team = FAKE_PLAYER_TEAMS[player]
        for (week, line, dk_o, dk_u, fd_o, fd_u) in lines:
            gid = team_week_to_gid.get((team, week))
            if gid is None:
                continue
            for book, price_o, price_u in [("draftkings", dk_o, dk_u), ("fanduel", fd_o, fd_u)]:
                rows.append({"nfl_game_id": gid, "bookmaker": book, "market": MARKET,
                             "outcome_name": "Over",  "outcome_desc": player,
                             "point": line, "price": price_o})
                rows.append({"nfl_game_id": gid, "bookmaker": book, "market": MARKET,
                             "outcome_name": "Under", "outcome_desc": player,
                             "point": line, "price": price_u})
    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def american_to_implied(price: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    return np.where(price < 0, -price / (-price + 100), 100 / (price + 100))


def consensus_lines(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw odds rows (multi-book), compute per player-game consensus line.

    Strategy:
      1. For each bookmaker, compute vig-free fair prob Over.
      2. Average fair probs across books → consensus_fair_prob_over.
      3. Take the modal line (point) across books as the consensus line.
         If books disagree on the point, flag it.

    Returns one row per (nfl_game_id, player_name) with:
      line, consensus_fair_prob_over, n_books, books_agree_on_line
    """
    over  = raw[raw["outcome_name"] == "Over" ].rename(columns={"price": "price_over",  "point": "line"})
    under = raw[raw["outcome_name"] == "Under"].rename(columns={"price": "price_under"})

    paired = over.merge(
        under[["nfl_game_id", "outcome_desc", "bookmaker", "price_under"]],
        on=["nfl_game_id", "outcome_desc", "bookmaker"],
        how="inner"
    ).rename(columns={"outcome_desc": "player_name"})

    paired["imp_over"]  = american_to_implied(paired["price_over"])
    paired["imp_under"] = american_to_implied(paired["price_under"])
    paired["vig"]       = paired["imp_over"] + paired["imp_under"] - 1
    paired["fair_prob_over"] = paired["imp_over"] / (paired["imp_over"] + paired["imp_under"])

    # Consensus: average fair prob across books, modal line
    def agg_consensus(g):
        modal_line = g["line"].mode().iloc[0]
        return pd.Series({
            "line":                    modal_line,
            "consensus_fair_prob_over": g["fair_prob_over"].mean(),
            "avg_vig":                 g["vig"].mean(),
            "n_books":                 g["bookmaker"].nunique(),
            "books_agree_on_line":     (g["line"] == modal_line).all(),
            "books":                   ",".join(sorted(g["bookmaker"].unique())),
            "juice_over_dk":           g.loc[g["bookmaker"] == "draftkings", "price_over"].iloc[0]
                                       if "draftkings" in g["bookmaker"].values else np.nan,
            "juice_under_dk":          g.loc[g["bookmaker"] == "draftkings", "price_under"].iloc[0]
                                       if "draftkings" in g["bookmaker"].values else np.nan,
            "juice_over_fd":           g.loc[g["bookmaker"] == "fanduel", "price_over"].iloc[0]
                                       if "fanduel" in g["bookmaker"].values else np.nan,
            "juice_under_fd":          g.loc[g["bookmaker"] == "fanduel", "price_under"].iloc[0]
                                       if "fanduel" in g["bookmaker"].values else np.nan,
        })

    return (paired
            .groupby(["nfl_game_id", "player_name"])
            .apply(agg_consensus, include_groups=False)
            .reset_index())


def load_odds_from_s3(season: int) -> pd.DataFrame:
    """Load all player_tackles_assists rows from S3 for the given season."""
    s3        = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    prefix    = f"nfl/all_markets/{season}"

    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]
    print(f"  {len(keys)} game files in S3")

    rows = []
    for key in keys:
        df  = pd.read_parquet(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))
        sub = df[(df["market"] == MARKET) & (df["bookmaker"].isin(BOOKMAKERS))]
        if not sub.empty:
            rows.append(sub)

    if not rows:
        sys.exit(f"No {MARKET} rows found in S3 for season {season} with bookmakers {BOOKMAKERS}. "
                 f"Re-run the backfill with BOOKMAKERS={BOOKMAKERS}.")

    return pd.concat(rows, ignore_index=True)


def load_pbp_features(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (player_games, team_opponent_map).
    player_games has one row per player-game with tackle actuals + rolling features.
    """
    print("Loading player lookup...")
    players    = nfl.import_players()[["gsis_id", "display_name"]].dropna(subset=["gsis_id"])
    id_to_name = players.set_index("gsis_id")["display_name"].to_dict()

    print("Loading PBP...")
    pbp = nfl.import_pbp_data([season])

    # Tackles
    solo_frames, assist_frames = [], []
    for i in range(1, 3):
        sub = pbp[pbp[f"solo_tackle_{i}_player_id"].notna()][[
            "game_id", "week", f"solo_tackle_{i}_player_id", f"solo_tackle_{i}_team"
        ]].copy()
        sub["player_name"] = sub[f"solo_tackle_{i}_player_id"].map(id_to_name)
        sub = sub.rename(columns={f"solo_tackle_{i}_team": "team"})[
            ["game_id", "week", "team", "player_name"]]
        sub["solo_tackles"] = 1
        solo_frames.append(sub)

    for i in range(1, 5):
        sub = pbp[pbp[f"assist_tackle_{i}_player_id"].notna()][[
            "game_id", "week", f"assist_tackle_{i}_player_id", f"assist_tackle_{i}_team"
        ]].copy()
        sub["player_name"] = sub[f"assist_tackle_{i}_player_id"].map(id_to_name)
        sub = sub.rename(columns={f"assist_tackle_{i}_team": "team"})[
            ["game_id", "week", "team", "player_name"]]
        sub["assist_tackles"] = 1
        assist_frames.append(sub)

    solos   = pd.concat(solo_frames).groupby(["game_id","week","team","player_name"])["solo_tackles"].sum().reset_index()
    assists = pd.concat(assist_frames).groupby(["game_id","week","team","player_name"])["assist_tackles"].sum().reset_index()
    pg = (solos.merge(assists, on=["game_id","week","team","player_name"], how="outer").fillna(0))
    pg["tackles_assists"] = pg["solo_tackles"] + pg["assist_tackles"]

    # Opponent map
    sched = nfl.import_schedules([season])[["game_id","week","home_team","away_team"]].copy()
    home  = sched.rename(columns={"home_team":"team","away_team":"opponent"})[["game_id","week","team","opponent"]]
    away  = sched.rename(columns={"away_team":"team","home_team":"opponent"})[["game_id","week","team","opponent"]]
    team_opp = pd.concat([home, away])
    pg = pg.merge(team_opp, on=["game_id","week","team"], how="left")

    # Opponent run rate (rolling L3)
    print("Computing opponent run rates...")
    run_rates = (
        pbp[pbp["play_type"].isin(["run","pass"])]
        .groupby(["game_id","week","posteam"])
        .agg(runs=("play_type", lambda x: (x=="run").sum()), plays=("play_type","count"))
        .assign(run_rate=lambda d: d["runs"]/d["plays"])
        .reset_index().rename(columns={"posteam":"offense_team"})
        .sort_values(["offense_team","week"])
    )
    run_rates["opp_run_rate_l3"] = (
        run_rates.groupby("offense_team")["run_rate"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
    )
    pg = pg.merge(
        run_rates[["game_id","offense_team","opp_run_rate_l3"]],
        left_on=["game_id","opponent"], right_on=["game_id","offense_team"], how="left"
    ).drop(columns=["offense_team"])

    # Snap counts
    print("Loading snap counts...")
    snaps = nfl.import_snap_counts([season])[["game_id","week","player","position","team","defense_pct"]].rename(columns={"player":"player_name"})
    snaps = snaps.sort_values(["player_name","week"])
    snaps["defense_pct_l3"] = (
        snaps.groupby("player_name")["defense_pct"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
    )
    pg = pg.merge(snaps[["game_id","player_name","position","defense_pct","defense_pct_l3"]], on=["game_id","player_name"], how="left")

    # Rolling tackle features
    pg = pg.sort_values(["player_name","week"]).reset_index(drop=True)
    for col in ["tackles_assists","solo_tackles","assist_tackles"]:
        grp = pg.groupby("player_name")[col]
        pg[f"{col}_l3"]     = grp.transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        pg[f"{col}_l18"]    = grp.transform(lambda s: s.rolling(18, min_periods=1).mean().shift(1))
        pg[f"{col}_career"] = grp.transform(lambda s: s.expanding().mean().shift(1))
        # n = how many games went into the rolling mean (so display shows "4.5(n=2)")
        pg[f"{col}_l3_n"]   = grp.transform(lambda s: s.rolling(3, min_periods=1).count().shift(1))
        pg[f"{col}_l18_n"]  = grp.transform(lambda s: s.rolling(18, min_periods=1).count().shift(1))

    print(f"  Player-game rows: {len(pg)}  |  unique players: {pg['player_name'].nunique()}")
    return pg, sched


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_calibration(merged: pd.DataFrame):
    print("\n── Calibration: actual Over rate vs consensus implied (vig-free) ───────────")
    merged["prob_bucket"] = pd.cut(
        merged["consensus_fair_prob_over"],
        bins=[0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0],
        labels=["<35%","35-40%","40-45%","45-50%","50-55%","55-60%","60-65%",">65%"]
    )
    calib = (merged.groupby("prob_bucket", observed=True)
             .agg(n=("went_over","count"), implied=("consensus_fair_prob_over","mean"), actual=("went_over","mean"))
             .assign(edge=lambda d: d["actual"] - d["implied"])
             .round(3))

    print(f"{'bucket':<10} {'n':>5}  {'implied':>8}  {'actual':>8}  {'edge':>8}  signal")
    print("-" * 55)
    for bucket, row in calib.iterrows():
        signal = "✅ Over" if row["edge"] > 0.03 else ("✅ Under" if row["edge"] < -0.03 else "—")
        print(f"{str(bucket):<10} {int(row['n']):>5}  {row['implied']:>8.3f}  {row['actual']:>8.3f}  {row['edge']:>+8.3f}  {signal}")

    brier          = ((merged["went_over"] - merged["consensus_fair_prob_over"])**2).mean()
    baseline_brier = ((merged["went_over"] - merged["went_over"].mean())**2).mean()
    print(f"\nBrier score : {brier:.4f}  (naive baseline: {baseline_brier:.4f})")
    print(f"Skill score : {1 - brier/baseline_brier:+.3f}")

    print("\n── By position ──────────────────────────────────────────────────────────────")
    pos = (merged[merged["position"].notna()]
           .groupby("position")
           .agg(n=("went_over","count"), avg_line=("line","mean"),
                implied=("consensus_fair_prob_over","mean"), actual=("went_over","mean"))
           .assign(edge=lambda d: d["actual"] - d["implied"])
           .round(3).sort_values("n", ascending=False))

    print(f"{'pos':<6} {'n':>5}  {'avg_line':>9}  {'implied':>8}  {'actual':>8}  {'edge':>8}")
    print("-" * 55)
    for p, row in pos.iterrows():
        flag = " ✅" if abs(row["edge"]) > 0.03 else ""
        print(f"{p:<6} {int(row['n']):>5}  {row['avg_line']:>9.2f}  {row['implied']:>8.3f}  {row['actual']:>8.3f}  {row['edge']:>+8.3f}{flag}")

    print(f"\nN player-games: {len(merged)}  |  overall actual over rate: {merged['went_over'].mean():.3f}")

    # Book agreement check
    disagreed = merged[~merged["books_agree_on_line"]]
    print(f"\nGames where DK/FD posted different lines: {len(disagreed)} ({len(disagreed)/len(merged):.1%})")
    if not disagreed.empty:
        print(disagreed[["player_name","week","line","books","consensus_fair_prob_over"]].head(5).to_string(index=False))


def player_walkthrough(name: str, merged_df: pd.DataFrame,
                       player_games_df: pd.DataFrame, sched_df: pd.DataFrame):
    W = 148
    print(f"\n{'='*W}")
    print(f"  PLAYER WALKTHROUGH: {name}")
    print(f"{'='*W}")

    with_line = merged_df[merged_df["player_name"] == name].sort_values("week")
    played    = player_games_df[player_games_df["player_name"] == name][
        ["game_id","week","team","tackles_assists","solo_tackles","assist_tackles",
         "tackles_assists_l3","tackles_assists_l3_n",
         "tackles_assists_l18","tackles_assists_l18_n",
         "defense_pct_l3","opponent"]
    ].sort_values("week")

    if played.empty and with_line.empty:
        print(f"  '{name}' not found in PBP or lines.")
        return

    team = played["team"].iloc[0] if not played.empty else with_line["team"].iloc[0]
    pos  = (with_line["position"].dropna().iloc[0]
            if not with_line.empty and with_line["position"].notna().any() else "?")

    team_sched = sched_df[
        (sched_df["home_team"] == team) | (sched_df["away_team"] == team)
    ][["game_id","week","home_team","away_team"]].copy()
    team_sched["opponent"] = team_sched.apply(
        lambda r: r["away_team"] if r["home_team"] == team else r["home_team"], axis=1)
    team_sched = team_sched.sort_values("week")

    played_gids = set(played["game_id"])
    line_weeks  = set(with_line["week"])
    n_line = len(with_line)
    n_play = len(played)
    n_dnp  = len(team_sched) - n_play

    print(f"  Pos: {pos}  Team: {team}  Played: {n_play}  DNP: {n_dnp}  "
          f"Games with line: {n_line}/{n_play}  Books: {', '.join(BOOKMAKERS)}")
    print()

    hdr = (f"{'Wk':>3}  {'Opp':<5}  {'Status':<11}  {'Line':>5}  {'FairP%':>7}  "
           f"{'DK O/U':>11}  {'FD O/U':>11}  {'L3(n)':>8}  {'L18(n)':>9}  {'Snap%':>6}  "
           f"{'MdlProb':>8}  {'MdlEdge':>8}  {'Actual':>7}  Result")
    print(hdr)
    print("-" * W)

    results = []
    for _, g in team_sched.iterrows():
        wk  = int(g["week"])
        opp = g["opponent"]
        gid = g["game_id"]

        if gid not in played_gids:
            print(f"{wk:>3}  {opp:<5}  {'DNP':.<11}")
            continue

        p        = played[played["game_id"] == gid].iloc[0]
        actual   = int(p["tackles_assists"])
        l3_n     = int(p["tackles_assists_l3_n"])  if pd.notna(p.get("tackles_assists_l3_n"))  else 0
        l18_n    = int(p["tackles_assists_l18_n"]) if pd.notna(p.get("tackles_assists_l18_n")) else 0
        l3_str   = f"{p['tackles_assists_l3']:.1f}({l3_n})"   if pd.notna(p["tackles_assists_l3"])  else "first"
        l18_str  = f"{p['tackles_assists_l18']:.1f}({l18_n})" if pd.notna(p["tackles_assists_l18"]) else "first"
        snap_str = f"{p['defense_pct_l3']:.0%}"                if pd.notna(p["defense_pct_l3"])      else "?"

        if wk in line_weeks:
            r        = with_line[with_line["week"] == wk].iloc[0]
            line     = r["line"]
            fair_p   = r["consensus_fair_prob_over"]
            residual = r["residual"]
            went     = int(r["went_over"])
            result   = f"OVER  (+{residual:.1f})" if went else f"UNDER ({residual:.1f})"
            dk_odds  = (f"{int(r['juice_over_dk']):+d}/{int(r['juice_under_dk']):+d}"
                        if pd.notna(r.get("juice_over_dk")) else "—")
            fd_odds  = (f"{int(r['juice_over_fd']):+d}/{int(r['juice_under_fd']):+d}"
                        if pd.notna(r.get("juice_over_fd")) else "—")
            mdl_prob = r.get("model_prob")
            mdl_edge = r.get("model_edge")
            mdl_prob_str = f"{mdl_prob:.1%}" if pd.notna(mdl_prob) else "—"
            mdl_edge_str = f"{mdl_edge:+.1%}" if pd.notna(mdl_edge) else "—"
            results.append({"week": wk, "went_over": went, "fair_prob": fair_p, "residual": residual,
                             "juice_under_dk": r["juice_under_dk"]})
            print(f"{wk:>3}  {opp:<5}  {'LINE POSTED':<11}  {line:>5.1f}  {fair_p:>7.1%}  "
                  f"{dk_odds:>11}  {fd_odds:>11}  {l3_str:>8}  {l18_str:>9}  {snap_str:>6}  "
                  f"{mdl_prob_str:>8}  {mdl_edge_str:>8}  {actual:>7}  {result}")
        else:
            print(f"{wk:>3}  {opp:<5}  {'NO LINE':<11}  {'—':>5}  {'—':>7}  "
                  f"{'—':>11}  {'—':>11}  {l3_str:>8}  {l18_str:>9}  {snap_str:>6}  "
                  f"{'—':>8}  {'—':>8}  {actual:>7}")

    if not results:
        print("-" * W)
        print("  No lines posted for this player this season.")
        return

    res_df  = pd.DataFrame(results)
    over_n  = int(res_df["went_over"].sum())
    under_n = len(res_df) - over_n
    units   = 0.0
    for _, r in res_df.iterrows():
        j = r["juice_under_dk"]
        if pd.isna(j):
            continue
        units += (100 / abs(j) if j < 0 else j / 100) if r["went_over"] == 0 else -1.0

    print("-" * W)
    print(f"  Over: {over_n}/{len(res_df)} ({over_n/len(res_df):.0%})  "
          f"Under: {under_n}/{len(res_df)} ({under_n/len(res_df):.0%})  "
          f"Avg consensus fair prob Over: {res_df['fair_prob'].mean():.1%}  "
          f"Avg residual: {res_df['residual'].mean():+.2f}  "
          f"P&L (1u Under): {units:+.2f}u")


# ── Modeling ──────────────────────────────────────────────────────────────────

def run_model(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic regression to predict went_over using PBP-derived features.
    Key question: do our features add predictive power on top of what the
    market's consensus line already knows?

    Features are expressed as deltas from the posted line so the model is
    comparing player trajectory to the market expectation, not raw counts.

    Uses out-of-fold (OOF) cross-validation so reported probabilities are
    never trained on the rows they predict — no data leakage.

    Modifies merged in-place to add model_prob and model_edge columns.
    """
    if not _SKLEARN:
        print("\n[modeling skipped — run: pip install scikit-learn]")
        return merged

    print("\n── Model: Logistic Regression (out-of-fold CV) ──────────────────────────────────")

    df = merged.copy()

    # Derived features: player recent trajectory minus posted line.
    # Positive line_vs_l3 means player has been averaging above the current line.
    df["line_vs_l3"]     = df["tackles_assists_l3"]     - df["line"]
    df["line_vs_l18"]    = df["tackles_assists_l18"]    - df["line"]
    df["line_vs_career"] = df["tackles_assists_career"] - df["line"]

    FEATURES = [
        "line_vs_l3",      # hot/cold vs the line in last 3 games
        "line_vs_l18",     # sustained vs the line over L18
        "line_vs_career",  # career expectation vs the line
        "defense_pct_l3",  # recent snap share (opportunity proxy)
        "opp_run_rate_l3", # opponent run rate (more runs = more tackle opps)
        "line",            # absolute line level (DE ~3.5, LB ~9.5 — model needs to know scale)
    ]

    df_m = df.dropna(subset=FEATURES + ["went_over"]).copy()
    n = len(df_m)

    if n < 20:
        print(f"  Only {n} complete rows after dropping NaN — need more data to model.")
        return merged

    X = df_m[FEATURES].values
    y = df_m["went_over"].values

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    n_folds = min(5, n // 8)
    n_folds = max(n_folds, 2)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof_probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    df_m = df_m.copy()
    df_m["model_prob"] = oof_probs
    df_m["model_edge"] = df_m["model_prob"] - df_m["consensus_fair_prob_over"]

    # Full-data fit just for interpretable coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    coefs = pipe.named_steps["lr"].coef_[0]

    print(f"  N rows for model: {n}  |  CV folds: {n_folds}  |  "
          f"Over rate: {y.mean():.1%}  (baseline Brier denominator)")

    print(f"\n  Feature coefficients (full-data fit, for interpretation only):")
    print(f"  {'Feature':<20}  {'Coef':>8}  interpretation")
    print("  " + "-" * 62)
    for feat, coef in sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True):
        direction = "higher → bet Over " if coef > 0 else "higher → bet Under"
        print(f"  {feat:<20}  {coef:>+8.3f}  {direction}")

    # Brier score comparison: model vs consensus vs naive baseline
    brier_m = ((df_m["went_over"] - df_m["model_prob"])**2).mean()
    brier_c = ((df_m["went_over"] - df_m["consensus_fair_prob_over"])**2).mean()
    brier_b = ((df_m["went_over"] - df_m["went_over"].mean())**2).mean()
    print(f"\n  Brier scores (lower = better predictions):")
    print(f"  Naive baseline (always predict mean): {brier_b:.4f}")
    print(f"  Consensus implied prob:               {brier_c:.4f}  skill={1-brier_c/brier_b:+.3f}")
    print(f"  Model (OOF):                          {brier_m:.4f}  skill={1-brier_m/brier_b:+.3f}")
    verdict = "✅ model adds value beyond the market" if brier_m < brier_c else "❌ model does NOT beat the market (expected with small N)"
    print(f"  {verdict}")

    # ── Backtest ──────────────────────────────────────────────────────────────
    # Three separate strategies so we can see if the edge is symmetric or one-sided.
    # model_edge > +t  → bet Over  (model thinks Over is more likely than market does)
    # model_edge < −t  → bet Under (model thinks Under is more likely than market does)

    def _bet_pnl(bets: pd.DataFrame, side: str):
        """Return (n_hits, total_pnl) for a set of bets on 'over' or 'under' at DK prices."""
        juice_col = "juice_over_dk" if side == "over" else "juice_under_dk"
        hits, pnl = 0, 0.0
        for _, row in bets.iterrows():
            j   = row.get(juice_col, np.nan)
            j   = j if pd.notna(j) else -110
            hit = int(row["went_over"]) == (1 if side == "over" else 0)
            pnl += (100 / abs(j) if j < 0 else j / 100) if hit else -1.0
            hits += int(hit)
        return hits, pnl

    THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    # Columns: Thresh | N bets | Hit% | Total units (absolute profit) | u/bet (efficiency)
    HDR = f"  {'Thresh':>7}  {'N bets':>6}  {'Hit%':>6}  {'Total u':>9}  {'u/bet':>7}"
    SEP = "  " + "-" * 43

    print(f"\n  ── Backtest 1: OVER only  (bet Over when model_edge > +t) ──────────────────────")
    print(HDR); print(SEP)
    for t in THRESHOLDS:
        bets = df_m[df_m["model_edge"] > t]
        if bets.empty:
            continue
        h, p = _bet_pnl(bets, "over")
        print(f"  {t:>7.0%}  {len(bets):>6}  {h/len(bets):>6.1%}  {p:>+9.2f}u  {p/len(bets):>+7.3f}u")

    print(f"\n  ── Backtest 2: UNDER only  (bet Under when model_edge < −t) ────────────────────")
    print(HDR); print(SEP)
    for t in THRESHOLDS:
        bets = df_m[df_m["model_edge"] < -t]
        if bets.empty:
            continue
        h, p = _bet_pnl(bets, "under")
        print(f"  {t:>7.0%}  {len(bets):>6}  {h/len(bets):>6.1%}  {p:>+9.2f}u  {p/len(bets):>+7.3f}u")

    print(f"\n  ── Backtest 3: BOTH ways  (Over when edge > +t, Under when edge < −t) ──────────")
    print(f"  {'Thresh':>7}  {'N bets':>6}  {'N_over':>7}  {'N_under':>8}  {'Hit%':>6}  {'Total u':>9}  {'u/bet':>7}")
    print("  " + "-" * 59)
    for t in THRESHOLDS:
        o_bets = df_m[df_m["model_edge"] >  t]
        u_bets = df_m[df_m["model_edge"] < -t]
        nb = len(o_bets) + len(u_bets)
        if nb == 0:
            continue
        oh, op = _bet_pnl(o_bets, "over")
        uh, up = _bet_pnl(u_bets, "under")
        hits, pnl = oh + uh, op + up
        print(f"  {t:>7.0%}  {nb:>6}  {len(o_bets):>7}  {len(u_bets):>8}  "
              f"{hits/nb:>6.1%}  {pnl:>+9.2f}u  {pnl/nb:>+7.3f}u")

    # Model edge distribution — where are the confident calls?
    print(f"\n  Model edge distribution (model_prob − consensus_fair_prob):")
    edge_buckets = pd.cut(df_m["model_edge"],
                          bins=[-1, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 1],
                          labels=["< -15%","-15 to -10%","-10 to -5%","-5 to 0%",
                                  "0 to +5%","+5 to +10%","+10 to +15%","> +15%"])
    edge_dist = (df_m.groupby(edge_buckets, observed=True)
                 .agg(n=("went_over","count"), actual=("went_over","mean"),
                      model=("model_prob","mean"), consensus=("consensus_fair_prob_over","mean"))
                 .round(3))
    print(f"  {'edge bucket':<15}  {'n':>4}  {'model%':>7}  {'consensus%':>11}  {'actual%':>8}")
    print("  " + "-" * 52)
    for bucket, row in edge_dist.iterrows():
        if row["n"] == 0:
            continue
        print(f"  {str(bucket):<15}  {int(row['n']):>4}  {row['model']:>7.1%}  "
              f"{row['consensus']:>11.1%}  {row['actual']:>8.1%}")

    # Write model predictions back into merged (only rows that were in df_m)
    merged = merged.copy()
    merged["model_prob"] = np.nan
    merged["model_edge"] = np.nan
    merged.loc[df_m.index, "model_prob"] = df_m["model_prob"].values
    merged.loc[df_m.index, "model_edge"] = df_m["model_edge"].values
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="player_tackles_assists full analysis")
    parser.add_argument("--fake-data", action="store_true",
                        help="Use hardcoded fake odds data instead of S3 (for pipeline testing)")
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument("--player", type=str, default=None,
                        help="Run walkthrough for a specific player only")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  player_tackles_assists analysis — season {args.season}")
    print(f"  Bookmakers : {BOOKMAKERS}")
    print(f"  Mode       : {'FAKE DATA' if args.fake_data else 'S3'}")
    print(f"{'='*60}\n")

    # ── Step 1: load schedule (shared by fake-data expand + consensus week attach) ──
    print("Loading schedule...")
    sched_raw = nfl.import_schedules([args.season])[["game_id","week","home_team","away_team"]]
    sched     = sched_raw.rename(columns={"game_id": "nfl_game_id"})

    # ── Step 2: load odds ─────────────────────────────────────────────────────
    if args.fake_data:
        print("Loading fake odds data...")
        raw_odds = build_fake_odds(sched_raw)
        print(f"  {len(raw_odds)} rows  |  {raw_odds['nfl_game_id'].nunique()} games  "
              f"|  {raw_odds['outcome_desc'].nunique()} unique players")
    else:
        print("Loading odds from S3...")
        raw_odds = load_odds_from_s3(args.season)
        print(f"  {len(raw_odds)} rows  |  {raw_odds['nfl_game_id'].nunique()} games  "
              f"|  {raw_odds['outcome_desc'].nunique()} unique players")

    # ── Step 3: consensus lines ───────────────────────────────────────────────
    print("\nComputing consensus lines...")
    consensus = consensus_lines(raw_odds)
    n_single_book = (consensus["n_books"] == 1).sum()
    n_both_books  = (consensus["n_books"] == 2).sum()
    n_disagree    = (~consensus["books_agree_on_line"]).sum()
    print(f"  {len(consensus)} player-game lines  |  "
          f"both books: {n_both_books}  single book: {n_single_book}  "
          f"line disagreement: {n_disagree}")
    consensus = consensus.merge(sched[["nfl_game_id","week"]], on="nfl_game_id", how="left")

    # Game totals from S3 (skip in fake mode — not critical)
    if not args.fake_data:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        total_rows = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"nfl/all_markets/{args.season}"):
            for obj in page.get("Contents", []):
                df  = pd.read_parquet(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()))
                sub = df[df["market"] == "totals"]
                if not sub.empty:
                    total_rows.append(sub)
        if total_rows:
            totals_df   = pd.concat(total_rows, ignore_index=True)
            game_totals = (totals_df[totals_df["outcome_name"] == "Over"]
                           .groupby("nfl_game_id")["point"].first().reset_index()
                           .rename(columns={"point": "game_total"}))
            consensus = consensus.merge(game_totals, on="nfl_game_id", how="left")

    # ── Step 4: PBP features ──────────────────────────────────────────────────
    player_games, _ = load_pbp_features(args.season)

    # ── Step 5: join ──────────────────────────────────────────────────────────
    merged = player_games.merge(consensus, on=["week","player_name"], how="inner")
    merged["residual"]  = merged["tackles_assists"] - merged["line"]
    merged["went_over"] = (merged["tackles_assists"] > merged["line"]).astype(int)

    unmatched = consensus[~consensus["player_name"].isin(player_games["player_name"])]["player_name"].nunique()
    print(f"\nJoined: {len(merged)} player-games  |  {merged['player_name'].nunique()} players  "
          f"|  {unmatched} consensus names unmatched in PBP")

    if len(merged) == 0:
        print("\nNo rows in merged — check name matching between Bovada/DK/FD and nfl_data_py.")
        if args.fake_data:
            print("Fake data player names must exactly match nfl_data_py display_name values.")
        sys.exit(1)

    # ── Step 6: calibration + position analysis ───────────────────────────────
    run_calibration(merged)

    # ── Step 7: modeling ──────────────────────────────────────────────────────
    merged = run_model(merged)

    # ── Step 8: player walkthroughs ───────────────────────────────────────────
    sched_for_walk = sched_raw  # reuse — already loaded

    if args.player:
        players_to_show = [args.player]
    else:
        # Top players by number of lines posted
        players_to_show = (merged.groupby("player_name")["week"].count()
                           .sort_values(ascending=False).head(4).index.tolist())

    for player in players_to_show:
        player_walkthrough(player, merged, player_games, sched_for_walk)

    print("\nDone.")


if __name__ == "__main__":
    main()
