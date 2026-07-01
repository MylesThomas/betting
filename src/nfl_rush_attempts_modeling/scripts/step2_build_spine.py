"""
Step 2 — Build historical spine for NFL rush attempts modeling.

Actuals source: nfl_data_py weekly data (carries column), REG + POST seasons.
  - Includes 0-carry active players (unlike PFR rush data which only records ≥1 carry)
  - season_type 'POST' = playoff weeks 19–22 (is_playoff = 1)

Market data: S3 props_backfill/{2023,2024,2025}/ (player_rush_attempts)

Design:
  - Spine is built at player-game level (one row per player-game)
  - Rolling features use only data strictly prior to game G (no lookahead)
  - per-book rows are created in build_labeled_dataset.py (Step 2b)

Output:
  ~/Downloads/tmp/rush_attempts/spine.parquet
"""

from __future__ import annotations

import re
import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import nfl_data_py as nfl
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR   = Path.home() / "Downloads" / "tmp" / "rush_attempts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S3_BUCKET  = "the-odds-api-mt"
BACKFILL_S = [2023, 2024, 2025]
MARKET     = "player_rush_attempts"

WINDOWS = [1, 3, 5, 8, 16]  # career added separately

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_TEAM_PAREN = re.compile(r"\s*\([A-Z]{2,4}\)\s*$")   # strips " (BAL)", " (LAR)" etc.
_SPECIAL_RE = re.compile(r"['\.\-,]")
_INIT_RE    = re.compile(r"(?<!\w)([a-z])\s([a-z])(?=\s|\b)")  # "c j" → "cj"

# Manual nickname/alias overrides (market name post-norm → box score canonical name post-norm)
_ALIASES: dict[str, str] = {
    "eli mitchell":     "elijah mitchell",   # market nickname vs full name
    "josh dobbs":       "joshua dobbs",      # nickname vs full name
    "josh palmers":     "joshua palmers",    # similar pattern
}

def _norm(name: str) -> str:
    s = str(name).strip()
    s = _TEAM_PAREN.sub("", s)       # strip " (BAL)" team tags
    s = _SUFFIX_RE.sub("", s)        # strip Jr./III/etc.
    s = _SPECIAL_RE.sub(" ", s)      # strip punctuation (. - ' ,)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # Collapse single-letter initials: "c j stroud" → "cj stroud"
    s = _INIT_RE.sub(lambda m: m.group(1) + m.group(2), s)
    s = _ALIASES.get(s, s)
    return s


# ── 1. Load actuals ───────────────────────────────────────────────────────────
#
# import_weekly_data works for 2022–2024 and captures 0-carry active players.
# 2025 is unavailable there (404), so we use import_weekly_pfr(s_type='rush')
# for 2025 and supplement with snap_counts to catch 0-carry active players.
#
# Player position comes from import_players() keyed on player_id / pfr_player_id.

def _load_player_positions() -> pd.DataFrame:
    players = nfl.import_players()[
        ["gsis_id", "pfr_id", "position", "display_name"]
    ].dropna(subset=["gsis_id"])
    players = players.rename(columns={"gsis_id": "player_id"})
    return players[["player_id", "pfr_id", "position", "display_name"]]


def _load_weekly_data_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load nfl_data_py weekly stats for seasons where it's available (2023–2024)."""
    frames = []
    for s in seasons:
        try:
            df = nfl.import_weekly_data(years=[s])
            frames.append(df)
        except Exception as e:
            print(f"    weekly_data({s}): unavailable ({e!r}), will use PFR fallback")
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    keep = [
        "player_id", "player_display_name", "position",
        "recent_team", "season", "week", "season_type", "opponent_team",
        "carries", "rushing_yards", "rushing_epa", "receptions", "targets",
    ]
    df = raw[[c for c in keep if c in raw.columns]].copy()
    df["data_source"] = "weekly_data"
    return df


def _load_pfr_rush_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load PFR weekly rush data for given seasons (includes ≥1-carry players only)."""
    frames = []
    for s in seasons:
        df = nfl.import_weekly_pfr(s_type="rush", years=[s])
        frames.append(df)
    pfr = pd.concat(frames, ignore_index=True)

    # Map PFR player_id → gsis_id + position
    pos_map = _load_player_positions()
    pfr = pfr.merge(
        pos_map[["pfr_id", "player_id", "position", "display_name"]].dropna(subset=["pfr_id"]),
        left_on="pfr_player_id", right_on="pfr_id", how="left"
    )
    # Use PFR name as fallback for display_name
    pfr["player_display_name"] = pfr["display_name"].fillna(pfr["pfr_player_name"])

    # Map season_type from game_type (REG / WC / DIV / CON / SB → POST)
    pfr["season_type"] = pfr["game_type"].apply(
        lambda x: "REG" if x == "REG" else "POST"
    )
    pfr = pfr.rename(columns={
        "team": "recent_team",
        "opponent": "opponent_team",
    })

    keep = [
        "player_id", "player_display_name", "position",
        "recent_team", "season", "week", "season_type", "opponent_team",
        "carries", "rushing_yards",
    ]
    keep = [c for c in keep if c in pfr.columns]
    df = pfr[keep].copy()
    df["data_source"] = "pfr_rush"
    return df


def _add_zero_carry_2025(pfr_2025: pd.DataFrame, team_sched_2025: pd.DataFrame) -> pd.DataFrame:
    """
    For 2025: find players in snap counts who were active but not in PFR rush data
    (0 carries) — supplement so we can correctly label market rows as unders.
    """
    print("    Supplementing 2025 with snap count data (0-carry active players)...")
    snaps = nfl.import_snap_counts([2025])
    # Offensive snaps only
    off_snaps = snaps[snaps["offense_snaps"] > 0][
        ["game_id", "week", "season", "pfr_player_id", "player", "position", "team",
         "offense_snaps"]
    ].copy()

    # Map pfr_player_id → player_display_name
    pos_map = _load_player_positions()
    off_snaps = off_snaps.merge(
        pos_map[["pfr_id", "player_id", "display_name"]].dropna(subset=["pfr_id"]),
        left_on="pfr_player_id", right_on="pfr_id", how="left"
    )
    off_snaps["player_display_name"] = off_snaps["display_name"].fillna(off_snaps["player"])

    # Find game opponent from schedule
    sched = team_sched_2025[["season", "week", "team", "opponent"]].copy()
    off_snaps = off_snaps.merge(sched, on=["season", "week", "team"], how="left")
    off_snaps = off_snaps.rename(columns={"team": "recent_team", "opponent": "opponent_team"})

    # Players already in PFR rush (had ≥1 carry)
    pfr_keys = set(zip(pfr_2025["recent_team"], pfr_2025["season"], pfr_2025["week"],
                       pfr_2025["player_display_name"]))

    # Filter to positions that actually get rush attempts prop lines
    skill_positions = {"RB", "QB", "WR", "TE", "FB", "HB"}
    off_snaps = off_snaps[off_snaps["position"].isin(skill_positions)].copy()

    # Keep only those NOT in PFR rush (0-carry players)
    off_snaps["_key"] = list(zip(off_snaps["recent_team"], off_snaps["season"],
                                  off_snaps["week"], off_snaps["player_display_name"]))
    zero_carry = off_snaps[~off_snaps["_key"].isin(pfr_keys)].copy()
    zero_carry["carries"] = 0
    zero_carry["rushing_yards"] = 0.0
    zero_carry["season_type"] = "REG"  # snap counts only cover REG
    zero_carry["data_source"] = "snap_counts_zero_carry"

    keep = ["player_id", "player_display_name", "position",
            "recent_team", "season", "week", "season_type", "opponent_team",
            "carries", "rushing_yards", "data_source"]
    zero_carry = zero_carry[[c for c in keep if c in zero_carry.columns]].copy()
    zero_carry = zero_carry.dropna(subset=["player_display_name"])

    print(f"    Added {len(zero_carry):,} 0-carry active rows for 2025")
    return zero_carry


def load_actuals(team_sched_2025: pd.DataFrame) -> pd.DataFrame:
    print("Loading weekly actuals (carries + context)...")

    # 2023 + 2024: use weekly_data (includes 0-carry active players)
    wd = _load_weekly_data_seasons([2023, 2024])
    if len(wd):
        print(f"  weekly_data (2023–2024): {len(wd):,} rows")

    # 2025: PFR rush data
    pfr_2025 = _load_pfr_rush_seasons([2025])
    print(f"  PFR rush (2025): {len(pfr_2025):,} rows")

    # 2025 zero-carry supplement
    zero_2025 = _add_zero_carry_2025(pfr_2025, team_sched_2025)

    # Combine
    frames = []
    if len(wd):
        frames.append(wd)
    frames.extend([pfr_2025, zero_2025])
    df = pd.concat(frames, ignore_index=True)

    df["player_name_norm"] = df["player_display_name"].apply(_norm)
    df["is_playoff"] = (df["season_type"] == "POST").astype(int)
    df["carries"] = pd.to_numeric(df["carries"], errors="coerce").fillna(0).astype(int)
    df["rushing_yards"] = pd.to_numeric(df["rushing_yards"], errors="coerce").fillna(0)
    df["position"] = df["position"].fillna("UNK")

    # Any stat check (to detect true DNPs in weekly_data rows)
    activity_cols = [c for c in ["carries", "receptions", "targets"] if c in df.columns]
    df["any_stat"] = df[activity_cols].sum(axis=1)

    print(f"\n  Combined actuals: {len(df):,} rows | seasons {sorted(df['season'].unique())}")
    print(f"  0-carry rows: {(df['carries'] == 0).sum():,} ({(df['carries'] == 0).mean():.1%})")
    print(f"  Positions: {df['position'].value_counts().head(8).to_dict()}")
    return df


# ── 2. Load schedule for home/away, game total, dates ─────────────────────────

def load_schedule() -> pd.DataFrame:
    print("Loading schedule (home/away, game totals, dates)...")
    sched_frames = []
    for season in BACKFILL_S:
        s = nfl.import_schedules([season])
        sched_frames.append(s)
    sched = pd.concat(sched_frames, ignore_index=True)

    keep = ["game_id", "season", "week", "gameday", "gametime",
            "home_team", "away_team", "total_line", "spread_line", "game_type"]
    sched = sched[[c for c in keep if c in sched.columns]].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")

    # One row per team per game (home and away)
    home = sched.assign(team=sched["home_team"], is_home=1,
                        opponent=sched["away_team"],
                        game_total=sched["total_line"])[
        ["game_id", "season", "week", "team", "is_home", "opponent",
         "gameday", "game_total", "game_type"]
    ]
    away = sched.assign(team=sched["away_team"], is_home=0,
                        opponent=sched["home_team"],
                        game_total=sched["total_line"])[
        ["game_id", "season", "week", "team", "is_home", "opponent",
         "gameday", "game_total", "game_type"]
    ]
    team_sched = pd.concat([home, away], ignore_index=True)
    print(f"  Schedule: {len(team_sched):,} team-game rows | "
          f"{team_sched['game_type'].value_counts().to_dict()}")
    return team_sched


# ── 3. Compute opponent rush defense (carries allowed per game) ────────────────

def build_opp_rush_defense(actuals: pd.DataFrame, team_sched: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game: rolling avg carries ALLOWED by that team's defense
    in prior N games (using the opponent's total carries against them).
    """
    print("Computing opponent rush defense features...")

    # Total carries per team per game (offensive carries by that team's players)
    team_carries = (
        actuals.groupby(["season", "week", "recent_team"])["carries"]
        .sum()
        .reset_index()
        .rename(columns={"carries": "off_carries", "recent_team": "team"})
    )

    # Join with schedule to get: for each game, how many carries did each team allow?
    # "team X allowed Y carries" = opponents of X had Y carries in that game
    # Map: off_carries of away team = carries_allowed by home team (and vice versa)
    merged = team_sched.merge(
        team_carries, on=["season", "week", "team"], how="left"
    )

    # carries_allowed by team T = off_carries of T's opponents
    # team_sched has (team, opponent) pairs; join team_carries on opponent side
    opp_carries = (
        team_sched[["season", "week", "team", "opponent"]]
        .merge(
            team_carries.rename(columns={"team": "opponent_team_carries"}),
            left_on=["season", "week", "opponent"],
            right_on=["season", "week", "opponent_team_carries"],
            how="left"
        )
        .rename(columns={"off_carries": "carries_allowed"})
        [["season", "week", "team", "carries_allowed"]]
    )

    # Rolling carries_allowed for each team (as the defensive team)
    opp_sorted = (
        opp_carries
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

    for w in [8, 16]:
        opp_sorted[f"opp_carry_allowed_L{w}"] = (
            opp_sorted.groupby("team")["carries_allowed"]
            .transform(lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean())
        )
    opp_sorted["opp_carry_allowed_Lcareer"] = (
        opp_sorted.groupby("team")["carries_allowed"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    # Rename: the "team" column here is the defensive team; we'll join on opponent_team
    return opp_sorted.rename(columns={"team": "def_team"})[
        ["season", "week", "def_team",
         "opp_carry_allowed_L8", "opp_carry_allowed_L16", "opp_carry_allowed_Lcareer"]
    ]


# ── 4. Load market consensus per player-game ──────────────────────────────────

def load_market_consensus() -> pd.DataFrame:
    """
    Returns one row per (nfl_game_id, player_name_norm) with:
      consensus_point, consensus_over_prob, n_books
    """
    print("Loading market consensus lines...")
    s3 = boto3.client("s3")
    frames = []
    for season in BACKFILL_S:
        prefix = f"nfl/props_backfill/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in resp.get("Contents", []):
            buf = BytesIO()
            s3.download_fileobj(S3_BUCKET, obj["Key"], buf)
            buf.seek(0)
            df = pd.read_parquet(buf)
            if MARKET in df["market"].values:
                frames.append(df[df["market"] == MARKET])

    raw = pd.concat(frames, ignore_index=True)
    raw["player_name_norm"] = raw["outcome_desc"].apply(_norm)
    raw["direction"] = raw["outcome_name"].str.strip().str.lower()
    raw["point"] = pd.to_numeric(raw["point"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["game_season"] = raw["nfl_game_id"].str.split("_").str[0].astype(int)
    raw["game_week"] = raw["nfl_game_id"].str.split("_").str[1].astype(int)

    # Dedup to one Over row per player-game-book (latest snapshot)
    over = (
        raw[raw["direction"] == "over"]
        .sort_values("snapshot")
        .drop_duplicates(subset=["nfl_game_id", "player_name_norm", "bookmaker"], keep="last")
    )
    under = (
        raw[raw["direction"] == "under"]
        .sort_values("snapshot")
        .drop_duplicates(subset=["nfl_game_id", "player_name_norm", "bookmaker"], keep="last")
    )

    # Consensus = median line across books for this player-game
    consensus = (
        over.groupby(["nfl_game_id", "player_name_norm", "game_season", "game_week"])
        .agg(
            consensus_point=("point", "median"),
            n_books=("bookmaker", "nunique"),
        )
        .reset_index()
    )

    # No-vig over probability: for each book, no-vig P(over) = over_impl / (over_impl + under_impl)
    # where over_impl = 1/over_decimal, under_impl = 1/under_decimal
    # We compute per-book no-vig then average across books
    def american_to_decimal(p):
        return np.where(p > 0, p / 100 + 1, 100 / (-p) + 1)

    over_m = over.rename(columns={"price": "over_price", "point": "over_point"})
    under_m = under.rename(columns={"price": "under_price"})

    vig_df = over_m.merge(
        under_m[["nfl_game_id", "player_name_norm", "bookmaker", "under_price"]],
        on=["nfl_game_id", "player_name_norm", "bookmaker"], how="inner"
    )
    vig_df["over_dec"] = american_to_decimal(vig_df["over_price"])
    vig_df["under_dec"] = american_to_decimal(vig_df["under_price"])
    vig_df["over_impl"] = 1 / vig_df["over_dec"]
    vig_df["under_impl"] = 1 / vig_df["under_dec"]
    vig_df["over_novvig"] = vig_df["over_impl"] / (vig_df["over_impl"] + vig_df["under_impl"])

    book_over_prob = (
        vig_df.groupby(["nfl_game_id", "player_name_norm"])["over_novvig"]
        .mean()
        .reset_index()
        .rename(columns={"over_novvig": "consensus_over_prob"})
    )

    consensus = consensus.merge(book_over_prob, on=["nfl_game_id", "player_name_norm"], how="left")

    # Per-book rows for training (kept for labeled dataset step)
    per_book = vig_df[["nfl_game_id", "player_name_norm", "bookmaker",
                        "over_point", "over_price", "under_price", "over_novvig"]].copy()
    per_book = per_book.rename(columns={
        "over_point": "book_line",
        "over_price": "book_over_price",
        "under_price": "book_under_price",
        "over_novvig": "book_over_prob",
    })

    print(f"  Market: {len(consensus):,} consensus player-game rows | "
          f"{consensus['nfl_game_id'].nunique():,} games | "
          f"avg {consensus['n_books'].mean():.1f} books/row")

    return consensus, per_book


# ── 5. Build spine ────────────────────────────────────────────────────────────

def build_spine(actuals: pd.DataFrame, team_sched: pd.DataFrame,
                opp_def: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    print("Building spine...")

    # Join actuals with schedule context
    df = actuals.merge(
        team_sched[["season", "week", "team", "is_home", "game_total",
                    "game_id", "gameday", "game_type"]],
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"],
        how="left"
    )

    # nfl_game_id key: for market join
    # nfl_data_py uses game_id format e.g. "2023_01_ARI_WAS"
    df = df.rename(columns={"game_id": "nfl_game_id"})
    df["is_playoff"] = (df["season_type"] == "POST").astype(int)

    # Position dummies
    df["position"] = df["position"].fillna("UNK")
    df["pos_RB"] = (df["position"] == "RB").astype(int)
    df["pos_QB"] = (df["position"] == "QB").astype(int)

    # ── Rolling carry features (per player) ───────────────────────────────────
    # Sort by player + chronological order
    df["_roll_key"] = df["player_id"].fillna(df["player_name_norm"])
    df = df.sort_values(["_roll_key", "season", "week"]).reset_index(drop=True)

    for w in WINDOWS:
        df[f"carry_rate_L{w}"] = (
            df.groupby("_roll_key")["carries"]
            .transform(lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean())
        )
        df[f"rush_yards_L{w}"] = (
            df.groupby("_roll_key")["rushing_yards"]
            .transform(lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean())
        )
    df["carry_rate_Lcareer"] = (
        df.groupby("_roll_key")["carries"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["rush_yards_Lcareer"] = (
        df.groupby("_roll_key")["rushing_yards"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["games_played"] = (
        df.groupby("_roll_key")["carries"]
        .transform(lambda s: s.shift(1).expanding().count())
    )
    df = df.drop(columns=["_roll_key"])

    # ── Join opponent defense features ────────────────────────────────────────
    df = df.merge(
        opp_def, left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "def_team"], how="left"
    ).drop(columns=["def_team"], errors="ignore")

    # ── Join market consensus ──────────────────────────────────────────────────
    df = df.merge(
        consensus, on=["nfl_game_id", "player_name_norm"], how="left"
    )

    # ── Over rate vs consensus line ────────────────────────────────────────────
    # Only compute for rows with a consensus line (player had a market)
    # After joining consensus, compute rolling over/under vs that game's line
    # We use is_over_consensus = (carries > consensus_point) for each row
    df["is_over_consensus"] = (df["carries"] > df["consensus_point"]).astype(float)
    df.loc[df["consensus_point"].isna(), "is_over_consensus"] = float("nan")

    df = df.sort_values(["player_name_norm", "season", "week"]).reset_index(drop=True)
    for w in [3, 5, 8, 16]:
        df[f"over_rate_L{w}"] = (
            df.groupby("player_name_norm")["is_over_consensus"]
            .transform(lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean())
        )
    df["over_rate_Lcareer"] = (
        df.groupby("player_name_norm")["is_over_consensus"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    print(f"  Spine: {len(df):,} rows total")
    print(f"  Rows with consensus_point: {df['consensus_point'].notna().sum():,} "
          f"({df['consensus_point'].notna().mean():.1%})")
    return df


# ── 6. Create training dataset (filter to rows with market + actuals) ─────────

def build_training_set(spine: pd.DataFrame, per_book: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game-book for training (following tackles pattern)."""
    print("Building per-book training dataset...")

    # Training rows: spine rows where we have a market AND a valid carry actual
    train_spine = spine[spine["consensus_point"].notna()].copy()

    # Join per-book rows
    labeled = train_spine.merge(
        per_book, on=["nfl_game_id", "player_name_norm"], how="inner"
    )

    # Label
    labeled["is_over"] = (labeled["carries"] > labeled["book_line"]).astype(int)

    # For variable-line market: compute edge-related columns
    labeled["edge_raw"] = labeled["book_over_prob"] - 0.5  # vs 50/50 baseline

    keep_cols = [
        # identifiers
        "nfl_game_id", "player_name_norm", "player_display_name", "bookmaker",
        "season", "week", "is_playoff",
        # actuals
        "carries", "is_over",
        # market
        "book_line", "book_over_price", "book_under_price", "book_over_prob",
        "consensus_point", "consensus_over_prob", "n_books",
        # rolling features
        "carry_rate_L1", "carry_rate_L3", "carry_rate_L5", "carry_rate_L8",
        "carry_rate_L16", "carry_rate_Lcareer",
        "rush_yards_L1", "rush_yards_L3", "rush_yards_L5", "rush_yards_L8",
        "rush_yards_L16", "rush_yards_Lcareer",
        "over_rate_L3", "over_rate_L5", "over_rate_L8", "over_rate_L16",
        "over_rate_Lcareer",
        # opponent defense
        "opp_carry_allowed_L8", "opp_carry_allowed_L16", "opp_carry_allowed_Lcareer",
        # game context
        "is_home", "game_total", "position", "pos_RB", "pos_QB",
        "games_played",
    ]
    keep_cols = [c for c in keep_cols if c in labeled.columns]
    labeled = labeled[keep_cols].copy()

    print(f"  Training rows: {len(labeled):,} | "
          f"unique player-games: {labeled[['nfl_game_id','player_name_norm']].drop_duplicates().shape[0]:,}")
    print(f"  Is_over rate: {labeled['is_over'].mean():.3f}")
    print(f"  Missing consensus_point: {labeled['consensus_point'].isna().sum()}")
    return labeled


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Schedule must load first (needed for 0-carry supplement in 2025)
    team_sched = load_schedule()
    team_sched_2025 = team_sched[team_sched["season"] == 2025].copy()

    actuals   = load_actuals(team_sched_2025)
    actuals.to_parquet(OUT_DIR / "actuals.parquet", index=False)

    opp_def    = build_opp_rush_defense(actuals, team_sched)
    consensus, per_book = load_market_consensus()

    spine = build_spine(actuals, team_sched, opp_def, consensus)
    spine.to_parquet(OUT_DIR / "spine.parquet", index=False)
    print(f"Saved spine to: {OUT_DIR / 'spine.parquet'}")

    training = build_training_set(spine, per_book)
    training.to_parquet(OUT_DIR / "training.parquet", index=False)
    print(f"Saved training to: {OUT_DIR / 'training.parquet'}")

    print("\n=== Step 2 complete ===")
