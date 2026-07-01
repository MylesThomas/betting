"""
Step 1 — Data Pull + EDA for NFL RB Rush Attempts

Pulls:
  1. Market data (player_rush_attempts) from props_backfill S3 + all_markets/2025
  2. Box score rushing stats from nfl_data_py (PFR weekly offense)

Saves outputs to:
  ~/Downloads/tmp/rush_attempts/
"""

from __future__ import annotations

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

OUT_DIR = Path.home() / "Downloads" / "tmp" / "rush_attempts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S3_BUCKET = "the-odds-api-mt"
BACKFILL_SEASONS = [2023, 2024, 2025]
MARKET = "player_rush_attempts"

s3 = boto3.client("s3")


# ── 1. Load market data ────────────────────────────────────────────────────────

def load_market_data() -> pd.DataFrame:
    frames = []

    for season in BACKFILL_SEASONS:
        prefix = f"nfl/props_backfill/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".parquet")]
        print(f"  Season {season}: {len(keys)} game files in props_backfill")

        for key in keys:
            buf = BytesIO()
            s3.download_fileobj(S3_BUCKET, key, buf)
            buf.seek(0)
            df = pd.read_parquet(buf)
            if MARKET in df["market"].values:
                frames.append(df[df["market"] == MARKET].copy())

    # Also load all_markets/2025 as supplementary (different schema — has bookmaker column)
    prefix2 = "nfl/all_markets/2025/"
    resp2 = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix2)
    keys2 = [obj["Key"] for obj in resp2.get("Contents", []) if obj["Key"].endswith(".parquet")]
    print(f"  all_markets/2025: {len(keys2)} game files")

    am_frames = []
    for key in keys2:
        buf = BytesIO()
        s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        df = pd.read_parquet(buf)
        if MARKET in df["market"].values:
            am_frames.append(df[df["market"] == MARKET].copy())

    if am_frames:
        am = pd.concat(am_frames, ignore_index=True)
        # Normalize all_markets schema to match props_backfill schema
        # props_backfill: market, bookmaker, last_update, outcome_name, outcome_desc, point, price, nfl_game_id, season, snapshot
        # all_markets:    odds_api_event_id, home_team, away_team, commence_time, snapshot_time, market, bookmaker, last_update, outcome_name, outcome_desc, price, point, nfl_game_id, season
        am = am.rename(columns={"snapshot_time": "snapshot"})
        am_cols = ["market", "bookmaker", "last_update", "outcome_name", "outcome_desc", "point", "price", "nfl_game_id", "season", "snapshot"]
        am = am[[c for c in am_cols if c in am.columns]]
        frames.append(am)
        print(f"  all_markets/2025 rush_attempts rows: {len(am)}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal raw market rows: {len(combined):,}")
    return combined


def normalize_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Parse game season and week from nfl_game_id (format: 2023_01_ARI_WAS)
    df["game_season"] = df["nfl_game_id"].str.split("_").str[0].astype(int)
    df["game_week"] = df["nfl_game_id"].str.split("_").str[1].astype(int)

    # Parse snapshot to date
    df["snapshot"] = pd.to_datetime(df["snapshot"], utc=True, errors="coerce")
    df["snapshot_date"] = df["snapshot"].dt.date

    # outcome_name = 'Over'/'Under', outcome_desc = player name
    df["direction"] = df["outcome_name"].str.strip().str.lower()  # 'over' or 'under'
    df["player_name"] = df["outcome_desc"].str.strip()

    return df


# ── 2. EDA on market data ─────────────────────────────────────────────────────

def eda_market(df: pd.DataFrame):
    print("\n" + "="*60)
    print("MARKET EDA")
    print("="*60)

    # Dedup to one Over row per player-game-book (latest snapshot)
    over = df[df["direction"] == "over"].copy()
    print(f"\nOver rows (raw): {len(over):,}")
    print(f"Under rows (raw): {len(df[df['direction']=='under']):,}")

    # Coverage by season + week
    print("\n--- Coverage by season ---")
    season_cov = over.groupby("game_season")["nfl_game_id"].nunique().reset_index()
    season_cov.columns = ["season", "games_with_lines"]
    print(season_cov.to_string(index=False))

    print("\n--- Books represented ---")
    book_counts = over.groupby("bookmaker")["nfl_game_id"].nunique().sort_values(ascending=False)
    print(book_counts.head(20).to_string())

    # Deduplicate: one row per player-game-book (latest snapshot)
    over_dedup = (
        over.sort_values("snapshot")
        .drop_duplicates(subset=["nfl_game_id", "player_name", "bookmaker"], keep="last")
        .reset_index(drop=True)
    )
    print(f"\nDeduped over rows (latest snapshot per player-game-book): {len(over_dedup):,}")
    print(f"Unique player-games (any book): {over_dedup.groupby(['nfl_game_id','player_name']).ngroups:,}")
    print(f"Unique games: {over_dedup['nfl_game_id'].nunique()}")
    print(f"Unique players: {over_dedup['player_name'].nunique()}")

    # Line distribution
    print("\n--- Line distribution (point values) ---")
    line_dist = over_dedup["point"].value_counts().sort_index()
    print(line_dist.head(30).to_string())

    print(f"\nLine stats:")
    print(f"  Min: {over_dedup['point'].min()}")
    print(f"  Max: {over_dedup['point'].max()}")
    print(f"  Median: {over_dedup['point'].median()}")
    print(f"  Mean: {over_dedup['point'].mean():.2f}")
    print(f"  Std: {over_dedup['point'].std():.2f}")

    # Null check
    print("\n--- Null rates ---")
    for col in ["player_name", "point", "price", "nfl_game_id", "bookmaker"]:
        null_rate = df[col].isnull().mean()
        print(f"  {col}: {null_rate:.1%}")

    return over_dedup


# ── 3. Load box score / rushing actuals ───────────────────────────────────────

def load_rushing_actuals() -> pd.DataFrame:
    print("\n" + "="*60)
    print("BOX SCORE / RUSHING ACTUALS")
    print("="*60)

    seasons = [2023, 2024, 2025]
    frames = []
    for season in seasons:
        print(f"  Loading PFR weekly offense for {season}...")
        df = nfl.import_weekly_pfr(s_type="rush", years=[season])
        frames.append(df)

    pfr = pd.concat(frames, ignore_index=True)
    print(f"\nRaw PFR rows: {len(pfr):,}")
    print(f"Columns: {pfr.columns.tolist()}")

    # Check rushing-relevant columns
    rush_cols = [c for c in pfr.columns if "rush" in c.lower() or "carry" in c.lower() or "att" in c.lower()]
    print(f"\nRush-related columns: {rush_cols}")

    return pfr


def normalize_rushing(pfr: pd.DataFrame) -> pd.DataFrame:
    import re

    SUFFIX_RE = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
    SPECIAL_RE = re.compile(r"['\.\-,]")

    def norm_name(name: str) -> str:
        s = str(name).strip()
        s = SUFFIX_RE.sub("", s)
        s = SPECIAL_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    pfr = pfr.copy()

    # PFR rush data uses pfr_player_name (not player_display_name)
    pfr["player_name_norm"] = pfr["pfr_player_name"].apply(norm_name)

    # 'carries' is the rush attempts column in import_weekly_pfr(s_type='rush')
    pfr = pfr.rename(columns={"carries": "rush_attempts_actual"})
    print(f"  Rush attempts column: 'carries' → 'rush_attempts_actual'")

    # Join position from players table via pfr_player_id
    players = nfl.import_players()[["pfr_id", "position", "display_name"]].dropna(subset=["pfr_id"])
    players = players.rename(columns={"pfr_id": "pfr_player_id"})
    pfr = pfr.merge(players[["pfr_player_id", "position"]].drop_duplicates("pfr_player_id"),
                    on="pfr_player_id", how="left")

    # Filter to regular season (game_type REG)
    if "game_type" in pfr.columns:
        pfr = pfr[pfr["game_type"] == "REG"].copy()
        print(f"  Filtered to REG season type")

    keep = ["pfr_player_id", "pfr_player_name", "player_name_norm", "position",
            "team", "opponent", "season", "week", "rush_attempts_actual",
            "rushing_yards_before_contact", "rushing_yards_after_contact"]
    keep = [c for c in keep if c in pfr.columns]
    pfr = pfr[keep].copy()

    print(f"\nPFR after normalize: {len(pfr):,} rows")
    print(f"Positions: {pfr['position'].value_counts().head(10).to_dict()}")
    print(f"\nRush attempts stats (all positions):")
    print(pfr["rush_attempts_actual"].describe())

    # RB focus
    rb_pfr = pfr[pfr["position"] == "RB"].copy()
    print(f"\nRB-only rows: {len(rb_pfr):,}")
    if len(rb_pfr) > 0:
        print(f"RB rush attempts:")
        print(rb_pfr["rush_attempts_actual"].describe())
        print(f"\nRB games with 0 rush attempts: {(rb_pfr['rush_attempts_actual'] == 0).sum():,} ({(rb_pfr['rush_attempts_actual'] == 0).mean():.1%})")

    return pfr


# ── 4. Compute over/under/push hit rates ─────────────────────────────────────

def compute_hit_rates(market_dedup: pd.DataFrame, pfr: pd.DataFrame) -> pd.DataFrame:
    import re
    SUFFIX_RE = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
    SPECIAL_RE = re.compile(r"['\.\-,]")

    def norm_name(name: str) -> str:
        s = str(name).strip()
        s = SUFFIX_RE.sub("", s)
        s = SPECIAL_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    market_dedup = market_dedup.copy()
    market_dedup["player_name_norm"] = market_dedup["player_name"].apply(norm_name)

    # PFR key: player_name_norm + season + week
    pfr_key = pfr[["player_name_norm", "season", "week", "rush_attempts_actual"]].copy()
    pfr_key = pfr_key.dropna(subset=["rush_attempts_actual"])

    joined = market_dedup.merge(
        pfr_key,
        left_on=["player_name_norm", "game_season", "game_week"],
        right_on=["player_name_norm", "season", "week"],
        how="left"
    )

    n_total = len(joined)
    n_matched = joined["rush_attempts_actual"].notna().sum()
    print(f"\nJoin quality: {n_matched:,}/{n_total:,} = {n_matched/n_total:.1%} matched")

    # Compute outcome
    joined["is_over"] = joined["rush_attempts_actual"] > joined["point"]
    joined["is_push"] = joined["rush_attempts_actual"] == joined["point"]
    joined["is_under"] = joined["rush_attempts_actual"] < joined["point"]

    # Hit rates on matched rows only
    matched = joined[joined["rush_attempts_actual"].notna()].copy()
    n = len(matched)
    over_rate = matched["is_over"].mean()
    push_rate = matched["is_push"].mean()
    under_rate = matched["is_under"].mean()

    print(f"\nHit rates (on {n:,} matched player-game-book rows):")
    print(f"  Over:  {over_rate:.3f} ({over_rate:.1%})")
    print(f"  Push:  {push_rate:.3f} ({push_rate:.1%})")
    print(f"  Under: {under_rate:.3f} ({under_rate:.1%})")
    print(f"  Sum:   {over_rate + push_rate + under_rate:.4f}")

    # DNP rate: line posted but player has 0 attempts (DNP/inactive)
    # Typically DNP means rush_attempts_actual == 0 but also could be null if player truly didn't play
    dnp_proxy = matched[matched["rush_attempts_actual"] == 0]
    print(f"\nDNP proxy (0 rush attempts despite line being posted): {len(dnp_proxy):,} ({len(dnp_proxy)/n:.1%})")

    return joined


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load market data
    raw = load_market_data()
    market_norm = normalize_market(raw)

    # Save raw market
    raw_out = OUT_DIR / "market_raw.parquet"
    market_norm.to_parquet(raw_out, index=False)
    print(f"\nSaved raw market to: {raw_out}")

    # 2. EDA
    market_dedup = eda_market(market_norm)
    dedup_out = OUT_DIR / "market_dedup.parquet"
    market_dedup.to_parquet(dedup_out, index=False)
    print(f"Saved deduped market to: {dedup_out}")

    # 3. Box score / actuals
    pfr_raw = load_rushing_actuals()
    pfr_norm = normalize_rushing(pfr_raw)
    pfr_out = OUT_DIR / "pfr_rushing.parquet"
    pfr_norm.to_parquet(pfr_out, index=False)
    print(f"Saved PFR rushing to: {pfr_out}")

    # 4. Hit rates
    joined = compute_hit_rates(market_dedup, pfr_norm)
    joined_out = OUT_DIR / "market_joined.parquet"
    joined.to_parquet(joined_out, index=False)
    print(f"Saved joined to: {joined_out}")

    print("\n=== Step 1 EDA complete ===")
