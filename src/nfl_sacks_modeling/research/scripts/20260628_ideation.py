"""
v1_ideation.py — player_tackles_assists feature set exploration

Features tested:
  Rolling (L3, L18, career): tackles_assists, solo_tackles, assist_tackles
  Snap%: defense_pct (rolling L3 prior)
  Position: LB / DB / DL (categorical)
  Opponent run rate: rolling L3 of opp run play %
  Game total (over/under line): proxy for total plays / pace
  Spread: game script — big favorite = opponent may run clock late
"""

import sys
from pathlib import Path
from io import BytesIO

import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

import nfl_data_py as nfl

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/all_markets/2025"
SEASON    = 2025

# ── 0. Player name lookup: gsis_id → display_name ─────────────────────────────

print("Loading player lookup...")
players = nfl.import_players()[["gsis_id", "display_name", "position"]].dropna(subset=["gsis_id"])
id_to_name = players.set_index("gsis_id")["display_name"].to_dict()

# ── 1. PBP → per-player-game tackles + assists ────────────────────────────────

print("Loading PBP...")
pbp = nfl.import_pbp_data([SEASON])

solo_frames = []
for i in range(1, 3):
    sub = pbp[pbp[f"solo_tackle_{i}_player_id"].notna()][[
        "game_id", "week", f"solo_tackle_{i}_player_id", f"solo_tackle_{i}_team"
    ]].copy()
    sub["player_name"] = sub[f"solo_tackle_{i}_player_id"].map(id_to_name)
    sub = sub.rename(columns={f"solo_tackle_{i}_team": "team"})[
        ["game_id", "week", "team", "player_name"]]
    sub["solo_tackles"] = 1
    solo_frames.append(sub)

assist_frames = []
for i in range(1, 5):
    sub = pbp[pbp[f"assist_tackle_{i}_player_id"].notna()][[
        "game_id", "week", f"assist_tackle_{i}_player_id", f"assist_tackle_{i}_team"
    ]].copy()
    sub["player_name"] = sub[f"assist_tackle_{i}_player_id"].map(id_to_name)
    sub = sub.rename(columns={f"assist_tackle_{i}_team": "team"})[
        ["game_id", "week", "team", "player_name"]]
    sub["assist_tackles"] = 1
    assist_frames.append(sub)

solos = (pd.concat(solo_frames)
         .groupby(["game_id", "week", "team", "player_name"])["solo_tackles"]
         .sum().reset_index())
assists = (pd.concat(assist_frames)
           .groupby(["game_id", "week", "team", "player_name"])["assist_tackles"]
           .sum().reset_index())

player_games = (solos
                .merge(assists, on=["game_id", "week", "team", "player_name"], how="outer")
                .fillna(0))
player_games["tackles_assists"] = player_games["solo_tackles"] + player_games["assist_tackles"]

# ── 2. Opponent run rate per game (rolling L3 prior) ─────────────────────────
# For each game, what % of the opposing offense's plays were runs?
# We roll this so we only use info available before the game.

print("Computing opponent run rates...")
game_run_rates = (
    pbp[pbp["play_type"].isin(["run", "pass"])]
    .groupby(["game_id", "week", "posteam"])
    .agg(runs=("play_type", lambda x: (x == "run").sum()),
         plays=("play_type", "count"))
    .assign(run_rate=lambda d: d["runs"] / d["plays"])
    .reset_index()
    .rename(columns={"posteam": "offense_team"})
    .sort_values(["offense_team", "week"])
)
# Rolling L3 prior run rate per offensive team
game_run_rates["opp_run_rate_l3"] = (
    game_run_rates.groupby("offense_team")["run_rate"]
    .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
)

# Attach to player_games via opponent (team that tackled = defense, offense_team = opponent)
# We need to know who the tackling team's opponent is
sched = nfl.import_schedules([SEASON])[["game_id", "week", "home_team", "away_team"]].copy()
# Expand to one row per team: (game_id, team, opponent)
home = sched[["game_id", "week", "home_team", "away_team"]].rename(
    columns={"home_team": "team", "away_team": "opponent"})
away = sched[["game_id", "week", "away_team", "home_team"]].rename(
    columns={"away_team": "team", "home_team": "opponent"})
team_opponent = pd.concat([home, away])

player_games = player_games.merge(team_opponent, on=["game_id", "week", "team"], how="left")
player_games = player_games.merge(
    game_run_rates[["game_id", "offense_team", "opp_run_rate_l3"]],
    left_on=["game_id", "opponent"],
    right_on=["game_id", "offense_team"],
    how="left"
).drop(columns=["offense_team"])

# ── 3. Snap counts → defense_pct + position (rolling L3 prior) ───────────────

print("Loading snap counts...")
snaps = nfl.import_snap_counts([SEASON])[[
    "game_id", "week", "player", "position", "team",
    "defense_snaps", "defense_pct"
]].rename(columns={"player": "player_name"})

# Rolling L3 prior snap% per player
snaps = snaps.sort_values(["player_name", "week"])
snaps["defense_pct_l3"] = (
    snaps.groupby("player_name")["defense_pct"]
    .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
)

player_games = player_games.merge(
    snaps[["game_id", "player_name", "position", "defense_pct", "defense_pct_l3"]],
    on=["game_id", "player_name"],
    how="left"
)

# ── 4. Rolling tackle features (L3, L18, career) ─────────────────────────────

player_games = player_games.sort_values(["player_name", "week"]).reset_index(drop=True)

def rolling_prior(series, n):
    return series.rolling(n, min_periods=1).mean().shift(1)

def expanding_prior(series):
    return series.expanding().mean().shift(1)

for col in ["tackles_assists", "solo_tackles", "assist_tackles"]:
    grp = player_games.groupby("player_name")[col]
    player_games[f"{col}_l3"]     = grp.transform(lambda s: rolling_prior(s, 3))
    player_games[f"{col}_l18"]    = grp.transform(lambda s: rolling_prior(s, 18))
    player_games[f"{col}_career"] = grp.transform(expanding_prior)

print(f"Player-game rows: {len(player_games)}  |  unique players: {player_games['player_name'].nunique()}")

# ── 5. Load Bovada lines + game totals + spreads from S3 ──────────────────────

print("Loading S3 data (lines + totals + spreads)...")
s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")
keys = [obj["Key"]
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        for obj in page.get("Contents", [])]

tackle_rows, total_rows, spread_rows = [], [], []
for key in keys:
    df = pd.read_parquet(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))
    for mkt, bucket in [
        ("player_tackles_assists", tackle_rows),
        ("totals",                 total_rows),
        ("spreads",                spread_rows),
    ]:
        sub = df[df["market"] == mkt]
        if not sub.empty:
            bucket.append(sub)

lines   = pd.concat(tackle_rows, ignore_index=True)
totals  = pd.concat(total_rows,  ignore_index=True)
spreads = pd.concat(spread_rows, ignore_index=True)

# Closing lines (Over only)
lines_over = (lines[lines["outcome_name"] == "Over"]
              [["nfl_game_id", "outcome_desc", "point", "price"]]
              .rename(columns={"outcome_desc": "player_name", "point": "line"}))

# Game total per game (Over line point = total)
game_totals = (totals[totals["outcome_name"] == "Over"]
               .groupby("nfl_game_id")["point"].first().reset_index()
               .rename(columns={"point": "game_total"}))

# Spread per team per game — each team has a spread outcome
# outcome_name = team name, point = spread from that team's perspective
game_spreads = (spreads[["nfl_game_id", "outcome_name", "point"]]
                .rename(columns={"outcome_name": "team_name_bovada", "point": "spread"}))

# Attach week to lines
sched_w = nfl.import_schedules([SEASON])[["game_id", "week"]].rename(columns={"game_id": "nfl_game_id"})
lines_over = lines_over.merge(sched_w, on="nfl_game_id", how="left")
lines_over = lines_over.merge(game_totals, on="nfl_game_id", how="left")

print(f"  Lines: {len(lines_over)} over rows  |  {lines_over['player_name'].nunique()} players  |  {lines_over['nfl_game_id'].nunique()} games")

# ── 6. Join + implied probability ────────────────────────────────────────────

# Need both Over and Under juice to compute vig-free prob — fetch Under lines too
lines_all = (lines[["nfl_game_id", "outcome_desc", "outcome_name", "point", "price"]]
             .rename(columns={"outcome_desc": "player_name", "point": "line"}))
lines_all = lines_all.merge(sched_w, on="nfl_game_id", how="left")
lines_all = lines_all.merge(game_totals, on="nfl_game_id", how="left")

lines_over_j  = lines_all[lines_all["outcome_name"] == "Over" ][["nfl_game_id","player_name","week","line","game_total","price"]].rename(columns={"price": "juice_over"})
lines_under_j = lines_all[lines_all["outcome_name"] == "Under"][["nfl_game_id","player_name","price"]].rename(columns={"price": "juice_under"})
lines_juice = lines_over_j.merge(lines_under_j, on=["nfl_game_id","player_name"], how="inner")

def american_to_implied(price):
    """Raw implied probability (includes vig)."""
    price = pd.to_numeric(price, errors="coerce")
    return np.where(price < 0, -price / (-price + 100), 100 / (price + 100))

lines_juice["imp_over"]  = american_to_implied(lines_juice["juice_over"])
lines_juice["imp_under"] = american_to_implied(lines_juice["juice_under"])
lines_juice["vig"]       = lines_juice["imp_over"] + lines_juice["imp_under"] - 1

# Vig-free (fair) probability of Over
lines_juice["fair_prob_over"] = lines_juice["imp_over"] / (lines_juice["imp_over"] + lines_juice["imp_under"])

unmatched_n = lines_juice[~lines_juice["player_name"].isin(player_games["player_name"])]["player_name"].nunique()

merged = player_games.merge(lines_juice, on=["week", "player_name"], how="inner")
merged["residual"]  = merged["tackles_assists"] - merged["line"]
merged["went_over"] = (merged["tackles_assists"] > merged["line"]).astype(int)

print(f"\nMerged: {len(merged)} player-games  |  {merged['player_name'].nunique()} players  |  {unmatched_n} names unmatched")
print(f"Avg vig: {merged['vig'].mean():.3f}  |  fair_prob_over range: {merged['fair_prob_over'].min():.2f} – {merged['fair_prob_over'].max():.2f}")

# ── 7. Calibration analysis (Brier-style) ────────────────────────────────────
# Bucket by fair_prob_over, compare actual hit rate vs implied.
# Edge = actual_hit_rate - fair_prob_over (positive → Over has value, negative → Under has value)

print("\n── Calibration: actual Over rate vs Bovada implied (vig-free) ──────────────")
merged["prob_bucket"] = pd.cut(
    merged["fair_prob_over"],
    bins=[0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0],
    labels=["<35%","35-40%","40-45%","45-50%","50-55%","55-60%","60-65%",">65%"]
)

calib = (merged.groupby("prob_bucket", observed=True)
         .agg(n=("went_over","count"),
              implied_over=("fair_prob_over","mean"),
              actual_over=("went_over","mean"))
         .assign(edge=lambda d: d["actual_over"] - d["implied_over"])
         .round(3))

print(f"{'bucket':<10} {'n':>5}  {'implied':>8}  {'actual':>8}  {'edge':>8}  signal")
print("-" * 55)
for bucket, row in calib.iterrows():
    signal = "✅ Over" if row["edge"] > 0.03 else ("✅ Under" if row["edge"] < -0.03 else "—")
    print(f"{str(bucket):<10} {int(row['n']):>5}  {row['implied_over']:>8.3f}  {row['actual_over']:>8.3f}  {row['edge']:>+8.3f}  {signal}")

# Brier score — lower = better calibrated (0 = perfect, 0.25 = random)
merged["brier_contrib"] = (merged["went_over"] - merged["fair_prob_over"]) ** 2
brier = merged["brier_contrib"].mean()
# Baseline: always predict the mean over rate
baseline_brier = ((merged["went_over"] - merged["went_over"].mean()) ** 2).mean()
print(f"\nBrier score : {brier:.4f}  (baseline naive: {baseline_brier:.4f})")
print(f"Skill score : {1 - brier/baseline_brier:+.3f}  (positive = Bovada better than naive; negative = worse)")

# ── 8. Calibration by position ───────────────────────────────────────────────

print("\n── By position ──────────────────────────────────────────────────────────────")
pos = (merged[merged["position"].notna()]
       .groupby("position")
       .agg(n=("went_over","count"),
            avg_line=("line","mean"),
            implied_over=("fair_prob_over","mean"),
            actual_over=("went_over","mean"))
       .assign(edge=lambda d: d["actual_over"] - d["implied_over"])
       .round(3)
       .sort_values("n", ascending=False))

print(f"{'pos':<6} {'n':>5}  {'avg_line':>9}  {'implied':>8}  {'actual':>8}  {'edge':>8}")
print("-" * 55)
for p, row in pos.iterrows():
    flag = " ✅" if abs(row["edge"]) > 0.03 else ""
    print(f"{p:<6} {int(row['n']):>5}  {row['avg_line']:>9.2f}  {row['implied_over']:>8.3f}  {row['actual_over']:>8.3f}  {row['edge']:>+8.3f}{flag}")

print(f"\nN player-games: {len(merged)}  |  overall actual over rate: {merged['went_over'].mean():.3f}")

# ── 9. Player game-log walkthrough ───────────────────────────────────────────

def player_walkthrough(name: str, merged_df: pd.DataFrame,
                       player_games_df: pd.DataFrame, sched_df: pd.DataFrame):
    """
    Game-by-game breakdown for one player across all games their team played.

    Three row types:
      LINE POSTED  — player played + Bovada posted a line → full stats
      NO LINE      — player played (appeared in PBP) but no Bovada line posted
      DNP          — player's team played but player had zero tackles (injured/inactive)
    """
    W = 103
    print(f"\n{'='*W}")
    print(f"  PLAYER WALKTHROUGH: {name}")
    print(f"{'='*W}")

    with_line = merged_df[merged_df["player_name"] == name].sort_values("week")
    played    = player_games_df[player_games_df["player_name"] == name][
        ["game_id", "week", "team", "tackles_assists", "solo_tackles", "assist_tackles",
         "tackles_assists_l3", "tackles_assists_l18", "defense_pct_l3", "opponent"]
    ].sort_values("week")

    if played.empty and with_line.empty:
        print(f"  '{name}' not found in PBP or lines.")
        return

    team = played["team"].iloc[0] if not played.empty else with_line["team"].iloc[0]
    pos  = with_line["position"].dropna().iloc[0] if not with_line.empty and with_line["position"].notna().any() else "?"

    # All games this team played
    team_sched = sched_df[
        (sched_df["home_team"] == team) | (sched_df["away_team"] == team)
    ][["game_id","week","home_team","away_team"]].copy()
    team_sched["opponent"] = team_sched.apply(
        lambda r: r["away_team"] if r["home_team"] == team else r["home_team"], axis=1)
    team_sched = team_sched.sort_values("week")

    played_game_ids  = set(played["game_id"])
    line_weeks       = set(with_line["week"])

    n_line = len(with_line)
    n_play = len(played)
    n_dnp  = len(team_sched) - n_play

    print(f"  Position: {pos}   Team: {team}   "
          f"Played: {n_play}   DNP: {n_dnp}   Games with Bovada line: {n_line}/{n_play} played")
    print()

    hdr = (f"{'Wk':>3}  {'Opp':<5}  {'Status':<11}  {'Line':>5}  {'Juice':>6}  {'Impl%':>6}  "
           f"{'L3':>5}  {'L18':>6}  {'Snap%':>6}  {'Actual':>7}  {'Result'}")
    print(hdr)
    print("-" * W)

    results = []
    for _, g in team_sched.iterrows():
        wk  = int(g["week"])
        opp = g["opponent"]
        gid = g["game_id"]

        if gid not in played_game_ids:
            print(f"{wk:>3}  {opp:<5}  {'DNP':.<11}  {'—':>5}  {'—':>6}  {'—':>6}  "
                  f"{'—':>5}  {'—':>6}  {'—':>6}  {'—':>7}")
            continue

        p = played[played["game_id"] == gid].iloc[0]
        actual = int(p["tackles_assists"])
        l3_str   = f"{p['tackles_assists_l3']:.1f}"  if pd.notna(p["tackles_assists_l3"])  else "first"
        l18_str  = f"{p['tackles_assists_l18']:.1f}" if pd.notna(p["tackles_assists_l18"]) else "first"
        snap_str = f"{p['defense_pct_l3']:.0%}"      if pd.notna(p["defense_pct_l3"])      else "?"

        if wk in line_weeks:
            r = with_line[with_line["week"] == wk].iloc[0]
            line     = r["line"]
            juice    = r["juice_over"]
            implied  = r["fair_prob_over"]
            residual = r["residual"]
            went     = int(r["went_over"])
            juice_str  = f"{int(juice):+d}" if pd.notna(juice) else "?"
            result_str = f"OVER  (+{residual:.1f})" if went else f"UNDER ({residual:.1f})"
            status     = "LINE POSTED"
            results.append({"week": wk, "went_over": went, "implied": implied,
                             "residual": residual, "juice_under": r["juice_under"]})
            print(f"{wk:>3}  {opp:<5}  {status:<11}  {line:>5.1f}  {juice_str:>6}  "
                  f"{implied:>6.1%}  {l3_str:>5}  {l18_str:>6}  {snap_str:>6}  "
                  f"{actual:>7}  {result_str}")
        else:
            print(f"{wk:>3}  {opp:<5}  {'NO LINE':<11}  {'—':>5}  {'—':>6}  {'—':>6}  "
                  f"{l3_str:>5}  {l18_str:>6}  {snap_str:>6}  {actual:>7}")

    if not results:
        print("-" * W)
        print("  No lines posted for this player this season.")
        return

    res_df  = pd.DataFrame(results)
    over_n  = int(res_df["went_over"].sum())
    under_n = len(res_df) - over_n

    print("-" * W)
    print(f"  Over: {over_n}/{len(res_df)} ({over_n/len(res_df):.0%})   "
          f"Under: {under_n}/{len(res_df)} ({under_n/len(res_df):.0%})   "
          f"Avg implied Over: {res_df['implied'].mean():.1%}   "
          f"Avg residual: {res_df['residual'].mean():+.2f}")

    units = 0.0
    for _, r in res_df.iterrows():
        j = r["juice_under"]
        if pd.isna(j):
            continue
        units += (100 / abs(j) if j < 0 else j / 100) if r["went_over"] == 0 else -1.0
    print(f"  P&L (1u Under every game with line): {units:+.2f}u  over {len(res_df)} bets")


# Run walkthrough for a few interesting players
sched_full = nfl.import_schedules([SEASON])[["game_id","week","home_team","away_team"]]

for player in ["Myles Garrett", "Micah Parsons", "Roquan Smith", "Maxx Crosby"]:
    player_walkthrough(player, merged, player_games, sched_full)

print("\nDone v1.")
