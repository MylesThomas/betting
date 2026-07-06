"""
NBA Player Points — Step 2: Feature Engineering / Spine
=========================================================
Builds a rolling feature matrix at the player-game level, joined to
market consensus data. All features are strictly no-lookahead (shift(1)
before rolling). One row per player-game (not per book).

Features:
  pts_L1, pts_L3, pts_L5, pts_L10, pts_L20  — season-scoped rolling PTS avg
  pts_career                                  — career rolling PTS avg (across all seasons)
  min_L5, min_L20                             — season-scoped rolling MIN avg
  fga_L5                                      — season-scoped rolling FGA avg
  is_home                                     — 1=home, 0=away
  days_rest                                   — days since last game (capped at 14)
  games_into_season                           — 0-indexed games played in season before this one
  opp_pts_allowed_L10                         — opponent's rolling 10-game avg pts allowed (team total)

Market features (one row per player-game consensus):
  offered_line                                — median line across all two-sided books
  novig_prob_over                             — consensus no-vig P(over): avg across books
  n_books                                     — number of two-sided books posting this player-game
  min_line, max_line                          — line spread across books

Target:
  pts_actual                                  — actual PTS scored
  is_over                                     — 1 if pts_actual > offered_line, 0 otherwise

Outputs:
  ~/Downloads/tmp/points_eda/points_spine.parquet  — full spine for local inspection
  s3://the-odds-api-mt/nba/points_model/spine/nba_points_spine.parquet
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR     = Path.home() / "Downloads/tmp/points_eda"
SPINE_KEY   = "nba/points_model/spine/nba_points_spine.parquet"
SPINE_BUCKET = "the-odds-api-mt"

ROLL_WINDOWS = [1, 3, 5, 10, 20]   # season-scoped
SPOT_CHECK   = "stephen curry"

# Bovada team-total rows to exclude (player name contains these substrings)
BOVADA_TEAM_TOTAL_PATTERNS = ["alternate total", "total"]


def normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def parse_matchup(matchup: str):
    """Return (team_abbr, opp_abbr, is_home) from MATCHUP string."""
    m = str(matchup)
    if " vs. " in m:
        parts = m.split(" vs. ")
        return parts[0].strip(), parts[1].strip(), 1
    elif " @ " in m:
        parts = m.split(" @ ")
        return parts[0].strip(), parts[1].strip(), 0
    return None, None, None


def build_game_spine(logs: pd.DataFrame) -> pd.DataFrame:
    """Build rolling features from game logs. One row per player-game."""
    df = logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df = df.sort_values(["player_key", "season", "game_date"]).reset_index(drop=True)

    # ── Parse MATCHUP ─────────────────────────────────────────────────────────
    parsed = df["MATCHUP"].apply(lambda m: pd.Series(parse_matchup(m),
                                                      index=["team_abbr", "opp_abbr", "is_home"]))
    df = pd.concat([df, parsed], axis=1)

    # ── Opponent pts allowed (team total) ─────────────────────────────────────
    # For each game: sum all player PTS for each team → that's how many pts the
    # opposing team allowed on that date.
    team_pts = (
        df.groupby(["game_date", "team_abbr"])["PTS"].sum()
        .reset_index()
        .rename(columns={"team_abbr": "opp_abbr", "PTS": "opp_pts_on_date"})
    )
    # Rolling 10-game avg of pts allowed (strictly before current game)
    team_pts = team_pts.sort_values(["opp_abbr", "game_date"])
    team_pts["opp_pts_allowed_L10"] = (
        team_pts.groupby("opp_abbr")["opp_pts_on_date"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )
    df = df.merge(
        team_pts[["game_date", "opp_abbr", "opp_pts_allowed_L10"]],
        on=["game_date", "opp_abbr"],
        how="left",
    )

    # ── Days rest ─────────────────────────────────────────────────────────────
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df["days_rest"] = (
        df.groupby("player_key")["game_date_dt"]
        .transform(lambda s: s.diff().dt.days)
        .clip(upper=14)
    )

    # ── Games into season (0-indexed, strictly prior games) ───────────────────
    df["games_into_season"] = df.groupby(["player_key", "season"]).cumcount()

    # ── Rolling PTS features (season-scoped, no-lookahead) ────────────────────
    grp_season = df.groupby(["player_key", "season"], sort=False)
    grp_career = df.groupby("player_key", sort=False)

    def roll_shift(series, w, min_p=None):
        if min_p is None:
            min_p = min(3, w)
        return series.shift(1).rolling(w, min_periods=min_p).mean()

    for w in ROLL_WINDOWS:
        df[f"pts_L{w}"] = grp_season["PTS"].transform(lambda s: roll_shift(s, w))
        if w == 5:
            df["min_L5"]  = grp_season["MIN"].transform(lambda s: roll_shift(s, w))
            df["fga_L5"]  = grp_season["FGA"].transform(lambda s: roll_shift(s, w))
        if w == 20:
            df["min_L20"] = grp_season["MIN"].transform(lambda s: roll_shift(s, w))

    df["pts_career"] = grp_career["PTS"].transform(lambda s: roll_shift(s, 9999, min_p=3))

    return df


def build_market_consensus(props: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player-game: median line, avg novig P(over), n_books.
    Excludes one-sided rows (under_odds IS NULL) and bovada team totals.
    """
    p = props.copy()

    # Drop one-sided rows (mybookieag alternate lines)
    p = p[p["under_odds"].notna()]

    # Drop bovada team-total rows
    p = p[~p["player"].str.lower().str.contains("total|alternate total", na=False)]

    # Compute novig_prob_over per row
    def raw_prob(odds):
        odds = pd.to_numeric(odds, errors="coerce")
        profit = np.where(odds >= 0, odds / 100.0, 100.0 / odds.abs())
        return 1.0 / (1.0 + profit)

    p["raw_p_over"]  = raw_prob(p["over_odds"])
    p["raw_p_under"] = raw_prob(p["under_odds"])
    p["novig_p_over"] = p["raw_p_over"] / (p["raw_p_over"] + p["raw_p_under"])

    # Aggregate per player-game
    mkt = (
        p.groupby(["player_key", "game_date"], as_index=False)
        .agg(
            player=("player", "first"),
            offered_line=("prop_line", "median"),
            min_line=("prop_line", "min"),
            max_line=("prop_line", "max"),
            novig_prob_over=("novig_p_over", "mean"),
            n_books=("bookmaker", "nunique"),
        )
    )
    return mkt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    # ── Load local parquets from Step 1 ───────────────────────────────────────
    print("Loading Step 1 parquets...", flush=True)
    logs  = pd.read_parquet(OUT_DIR / "points_game_logs.parquet")
    props = pd.read_parquet(OUT_DIR / "points_props_raw.parquet")
    print(f"  logs:  {len(logs):,} rows")
    print(f"  props: {len(props):,} rows")

    # ── Build game feature spine ──────────────────────────────────────────────
    print("\nBuilding game feature spine...", flush=True)
    game_spine = build_game_spine(logs)
    print(f"  game_spine rows: {len(game_spine):,}")
    print(f"  opp_pts_allowed_L10 null rate: {game_spine['opp_pts_allowed_L10'].isna().mean():.1%}")
    print(f"  pts_L5 null rate: {game_spine['pts_L5'].isna().mean():.1%}")
    print(f"  pts_career null rate: {game_spine['pts_career'].isna().mean():.1%}")

    # ── Build market consensus ─────────────────────────────────────────────────
    print("\nBuilding market consensus...", flush=True)
    mkt = build_market_consensus(props)
    print(f"  market consensus rows: {len(mkt):,}")
    print(f"  Books: median n_books={mkt['n_books'].median():.0f}, mean={mkt['n_books'].mean():.1f}")

    # ── Join: props rows that have a game log entry ───────────────────────────
    print("\nJoining market to game spine...", flush=True)
    spine = mkt.merge(
        game_spine[[
            "player_key", "game_date", "season", "PTS", "MIN",
            "is_home", "days_rest", "games_into_season",
            "opp_pts_allowed_L10",
            "pts_L1", "pts_L3", "pts_L5", "pts_L10", "pts_L20", "pts_career",
            "min_L5", "min_L20", "fga_L5",
        ]],
        on=["player_key", "game_date"],
        how="left",
    )

    # Target
    spine["pts_actual"] = spine["PTS"]
    spine["is_over"] = (spine["pts_actual"] > spine["offered_line"]).astype(float)
    spine.loc[spine["pts_actual"].isna(), "is_over"] = np.nan

    # DNP rows (no game log match)
    spine["dnp"] = spine["pts_actual"].isna().astype(int)

    # Settled rows only (has actual PTS)
    settled = spine[spine["pts_actual"].notna()].copy()

    print(f"  Total prop rows: {len(spine):,}")
    print(f"  Settled (has actual PTS): {len(settled):,}  ({len(settled)/len(spine):.1%})")
    print(f"  DNP rate: {spine['dnp'].mean():.1%}")

    # ── Join quality ──────────────────────────────────────────────────────────
    n_matched = spine["PTS"].notna().sum()
    print(f"\nJoin quality: {n_matched:,} / {len(spine):,} = {n_matched/len(spine):.1%} matched to game log")

    # ── Null rates across features ─────────────────────────────────────────────
    print("\nNull rates (settled rows only):")
    feat_cols = ["pts_L1","pts_L3","pts_L5","pts_L10","pts_L20","pts_career",
                 "min_L5","min_L20","fga_L5","is_home","days_rest",
                 "games_into_season","opp_pts_allowed_L10","offered_line","novig_prob_over"]
    for col in feat_cols:
        if col in settled.columns:
            rate = settled[col].isna().mean()
            flag = " ⚠ " if rate > 0.10 else ""
            print(f"  {col:<25}: {rate:.1%}{flag}")

    # ── Spot-check: Stephen Curry ─────────────────────────────────────────────
    print(f"\n── Spot-check: {SPOT_CHECK} ──")
    curry = settled[settled["player_key"] == SPOT_CHECK].sort_values("game_date")
    print(f"  Total games: {len(curry)}")
    if len(curry) > 0:
        print(f"  Date range: {curry['game_date'].min()} → {curry['game_date'].max()}")
        display_cols = ["game_date","season","pts_actual","offered_line","pts_L1","pts_L5",
                        "pts_career","is_home","days_rest","opp_pts_allowed_L10","novig_prob_over","is_over"]
        display_cols = [c for c in display_cols if c in curry.columns]
        print(curry[display_cols].tail(12).to_string(index=False))

    # Season-start check: first 3 games of 2024-25 for Curry
    print(f"\n  Season-start check (first 3 games of 2024-25):")
    curry_s2 = curry[curry["season"] == "2024-25"].sort_values("game_date").head(3)
    if len(curry_s2) > 0:
        print(curry_s2[["game_date","pts_actual","pts_L1","pts_L3","pts_L5","pts_career"]].to_string(index=False))
        print("  → pts_L1/L3/L5 should be NaN or based only on 2024-25 games so far")
        print("  → pts_career should reflect prior career avg (from 2023-24 and earlier)")

    # ── Save ──────────────────────────────────────────────────────────────────
    spine.to_parquet(OUT_DIR / "points_spine.parquet", index=False)
    print(f"\nSaved locally → {OUT_DIR}/points_spine.parquet")

    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=SPINE_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    print(f"Uploaded → s3://{SPINE_BUCKET}/{SPINE_KEY}")

    print("\nDone.")


if __name__ == "__main__":
    main()
