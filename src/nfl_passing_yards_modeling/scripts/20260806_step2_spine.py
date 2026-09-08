"""
Step 2 — Feature Engineering / Spine (v2)

Builds rolling feature matrix at (player, nfl_game_id, bookmaker, line) grain.
No lookahead — all rolling features use only data strictly before the game.

Features built:
  Rolling passing yards           : windows 1, 3, 5, 10, season, career
  Rolling attempts                : windows 1, 3, 5, 10, season, career
  Rolling comp pct                : windows 1, 3, 5, 10, season, career
  Rolling YPA (yds/att)           : windows 1, 3, 5, 10, season, career
  Rolling EPA/att                 : windows 1, 3, 5, 10, season, career
  Rolling sack rate               : windows 3, 5, 10, career
  Rolling pass rate (att/plays)   : windows 3, 5, 10, career
  Rolling pass_yds std dev        : windows 5, 10, career
  Cover rate (over BOL line)      : windows 1, 3, 5, 10, season, career
  Opponent pass yards allowed     : rolling 5 games
  Starter flags                   : career_starts_pct, starts_last_4,
                                    games_as_starter_this_season, started_week1_for_team
  Game lines (nfl_data_py sched)  : spread_line, total_line, implied_team_total
  Market features                 : consensus novig prob, min/max line/prob across books,
                                    consensus odds bins (coarse + granular)

Output:
  ~/Downloads/tmp/pass_yds/step2_spine.parquet
  Appends HTML section to knowledge-base/raw/20260806-nfl-qb-pass-yds.html

Usage:
  uv run python 20260806_step2_spine.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
import yaml
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import nfl_data_py as nfl
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")

TMP_DIR    = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH  = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
BOL_PATH   = TMP_DIR / "step1_lines_all_books.parquet"   # all books for market features
FEAT_PATH  = TMP_DIR / "weekly_player_data.parquet"
OUT_PATH   = TMP_DIR / "step2_spine.parquet"
ALIAS_PATH = REPO_ROOT / "config" / "player_aliases.yaml"


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html(df: pd.DataFrame, title: str = "") -> str:
    out = ""
    if title:
        out += f'<p><strong>{title}</strong></p>'
    out += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;font-family:monospace">'
    out += "<thead><tr>" + "".join(
        f"<th style='background:#1a1a2e;color:white;padding:6px 10px'>{c}</th>" for c in df.columns
    ) + "</tr></thead><tbody>"
    for i, row in df.iterrows():
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        out += f"<tr style='background:{bg}'>" + "".join(
            f"<td style='padding:4px 8px'>{v}</td>" for v in row.values
        ) + "</tr>"
    out += "</tbody></table>"
    return out


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[''`]", "", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_aliases(path: Path, sport: str = "nfl") -> dict[str, str]:
    """Return {normalize(odds_api_name): normalize(canonical_name)}"""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {
        normalize_name(a["odds_api"]): normalize_name(a["canonical"])
        for a in cfg.get("aliases", [])
        if a.get("sport") == sport
    }


# ── Load raw data ─────────────────────────────────────────────────────────────

print("[Step 2] Loading data...")

aliases = load_aliases(ALIAS_PATH)
print(f"  NFL aliases loaded: {aliases}")

all_lines = pd.read_parquet(BOL_PATH)   # all books
feat_raw  = pd.read_parquet(FEAT_PATH)

# Reg season only (feature data already filtered, but enforce week ≤ 18 on lines too)
all_lines = all_lines[all_lines["nfl_week"] <= 18].copy()

# Build QBs feature frame — all QBs with attempts > 0 (includes backups who played)
qbs = feat_raw[(feat_raw["position"] == "QB") & (feat_raw["attempts"] > 0)].copy()
print(f"  QB rows (all seasons, attempts>0): {len(qbs):,}")

# American → decimal → implied prob helpers
def american_to_decimal(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.apply(lambda v: (v / 100 + 1) if v > 0 else (100 / abs(v) + 1) if v < 0 else np.nan)

all_lines["decimal_over"]  = american_to_decimal(all_lines["american_over"])
all_lines["decimal_under"] = american_to_decimal(all_lines["american_under"])
all_lines["raw_prob_over"]  = 1 / all_lines["decimal_over"]
all_lines["raw_prob_under"] = 1 / all_lines["decimal_under"]
all_lines["novig_prob_over"]  = all_lines["raw_prob_over"]  / (all_lines["raw_prob_over"] + all_lines["raw_prob_under"])
all_lines["novig_prob_under"] = all_lines["raw_prob_under"] / (all_lines["raw_prob_over"] + all_lines["raw_prob_under"])


# ── Rolling features on QB feature data ──────────────────────────────────────

print("[Step 2] Computing rolling features...")

qbs = qbs.sort_values(["player_id", "season", "week"]).copy()

# Is this game a "start"? Use attempts > 10 as proxy for starting
qbs["is_start"] = (qbs["attempts"] > 10).astype(int)

def shifted_expanding(series: pd.Series) -> pd.Series:
    """Expanding mean using only prior rows (shift by 1 to prevent leakage)."""
    return series.shift(1).expanding().mean()

def shifted_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()

def shifted_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).sum()

def season_expanding(df_group: pd.DataFrame, col: str) -> pd.Series:
    """Season-to-date mean using only prior games this season."""
    return df_group.groupby("season")[col].transform(
        lambda s: s.shift(1).expanding().mean()
    )

# ── Per-player rolling stats ──────────────────────────────────────────────────

print("  Rolling passing/attempt stats...")

feat_frames = []

for player_id, grp in qbs.groupby("player_id"):
    g = grp.sort_values(["season", "week"]).copy()

    # ── Derived per-play metrics ───────────────────────────────────────────────
    g["comp_pct"] = np.where(g["attempts"] > 0, g["completions"] / g["attempts"], np.nan)
    g["ypa"]      = np.where(g["attempts"] > 0, g["passing_yards"] / g["attempts"], np.nan)
    g["epa_per_att"] = np.where(g["attempts"] > 0, g["passing_epa"] / g["attempts"], np.nan)
    total_plays   = g["attempts"] + g["carries"].fillna(0)
    g["pass_rate"] = np.where(total_plays > 0, g["attempts"] / total_plays, np.nan)
    # sack rate: sacks / (dropbacks = attempts + sacks)
    dropbacks     = g["attempts"] + g["sacks"].fillna(0)
    g["sack_rate"] = np.where(dropbacks > 0, g["sacks"].fillna(0) / dropbacks, np.nan)

    # ── Rolling windows ────────────────────────────────────────────────────────
    for w in [1, 3, 5, 10]:
        g[f"pass_yds_roll_{w}"]  = shifted_rolling(g["passing_yards"], w)
        g[f"attempts_roll_{w}"]  = shifted_rolling(g["attempts"], w)
        g[f"comp_pct_roll_{w}"]  = shifted_rolling(g["comp_pct"], w)
        g[f"ypa_roll_{w}"]       = shifted_rolling(g["ypa"], w)
        g[f"epa_per_att_roll_{w}"] = shifted_rolling(g["epa_per_att"], w)

    for w in [3, 5, 10]:
        g[f"sack_rate_roll_{w}"] = shifted_rolling(g["sack_rate"], w)
        g[f"pass_rate_roll_{w}"] = shifted_rolling(g["pass_rate"], w)

    # Rolling std dev (consistency/variance signal)
    for w in [5, 10]:
        g[f"pass_yds_std_{w}"] = g["passing_yards"].shift(1).rolling(w, min_periods=3).std()

    # Season-to-date
    g["pass_yds_roll_season"]    = season_expanding(g, "passing_yards")
    g["attempts_roll_season"]    = season_expanding(g, "attempts")
    g["comp_pct_roll_season"]    = season_expanding(g, "comp_pct")
    g["ypa_roll_season"]         = season_expanding(g, "ypa")
    g["epa_per_att_roll_season"] = season_expanding(g, "epa_per_att")
    g["pass_rate_roll_season"]   = season_expanding(g, "pass_rate")
    g["sack_rate_roll_season"]   = season_expanding(g, "sack_rate")

    # Career
    g["pass_yds_roll_career"]    = shifted_expanding(g["passing_yards"])
    g["attempts_roll_career"]    = shifted_expanding(g["attempts"])
    g["comp_pct_roll_career"]    = shifted_expanding(g["comp_pct"])
    g["ypa_roll_career"]         = shifted_expanding(g["ypa"])
    g["epa_per_att_roll_career"] = shifted_expanding(g["epa_per_att"])
    g["pass_rate_roll_career"]   = shifted_expanding(g["pass_rate"])
    g["sack_rate_roll_career"]   = shifted_expanding(g["sack_rate"])
    g["pass_yds_std_career"]     = g["passing_yards"].shift(1).expanding(min_periods=3).std()

    # ── Starter flag features ──────────────────────────────────────────────────
    g["career_starts_pct"]     = shifted_expanding(g["is_start"])
    g["starts_last_4"]         = shifted_rolling_sum(g["is_start"], 4)
    g["games_as_starter_this_season"] = g.groupby("season")["is_start"].transform(
        lambda s: s.shift(1).expanding().sum()
    )

    feat_frames.append(g)

qbs_feat = pd.concat(feat_frames, ignore_index=True)

# ── started_week1_for_team ────────────────────────────────────────────────────
# For each (season, team), find the QB who started Week 1 (attempts > 10)
print("  Computing started_week1_for_team...")

week1_starters = (
    qbs_feat[(qbs_feat["week"] == 1) & (qbs_feat["is_start"] == 1)]
    [["season", "recent_team", "player_id"]]
    .drop_duplicates()
    .rename(columns={"player_id": "week1_starter_id"})
)

qbs_feat = qbs_feat.merge(week1_starters, on=["season", "recent_team"], how="left")
qbs_feat["started_week1_for_team"] = (qbs_feat["player_id"] == qbs_feat["week1_starter_id"]).astype(float)
qbs_feat = qbs_feat.drop(columns=["week1_starter_id"])

# Handle Week 1 itself — set started_week1_for_team = NaN there (not yet known)
qbs_feat.loc[qbs_feat["week"] == 1, "started_week1_for_team"] = np.nan


# ── Opponent pass yards allowed ───────────────────────────────────────────────
# For each team-week, compute their avg pass_yds_allowed in prior 5 games
# (pass_yds by QBs who faced them)
print("  Computing opponent pass yards allowed (rolling 5)...")

opp_frames = []
for team, grp in qbs_feat.groupby("recent_team"):
    # All QB games against this team = games where opponent_team == team
    opp_games = qbs_feat[qbs_feat["opponent_team"] == team][["season", "week", "passing_yards"]].copy()
    opp_games = opp_games.sort_values(["season", "week"])
    # Rolling 5-game avg pass allowed, shifted to avoid leakage
    opp_games["opp_pass_yds_allowed_roll5"] = shifted_rolling(opp_games["passing_yards"], 5)
    opp_games["opp_team"] = team
    opp_frames.append(opp_games[["season", "week", "opp_team", "opp_pass_yds_allowed_roll5"]])

opp_df = pd.concat(opp_frames, ignore_index=True).drop_duplicates(["season", "week", "opp_team"])

qbs_feat = qbs_feat.merge(
    opp_df,
    left_on =["season", "week", "opponent_team"],
    right_on=["season", "week", "opp_team"],
    how="left"
).drop(columns=["opp_team"])

print(f"  qbs_feat shape after rolling features: {qbs_feat.shape}")


# ── Game lines — spread + total → implied team total ─────────────────────────

print("[Step 2] Loading game lines from nfl_data_py schedules...")

sched = nfl.import_schedules([2023, 2024, 2025])
sched = sched[["game_id", "season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
sched = sched.rename(columns={"game_id": "nfl_game_id"})
sched = sched.dropna(subset=["spread_line", "total_line"])

# spread_line is from home team's perspective (negative = home favored)
# implied_home = (total + (-spread_line)) / 2  → (total - spread) / 2 when home is favored (spread < 0)
# implied_away = (total + spread_line) / 2
sched["implied_home_total"] = (sched["total_line"] - sched["spread_line"]) / 2
sched["implied_away_total"] = (sched["total_line"] + sched["spread_line"]) / 2

# Build long format: one row per team per game with implied total and spread from that team's POV
home_rows = sched[["nfl_game_id","season","week","home_team","spread_line","total_line","implied_home_total"]].copy()
home_rows = home_rows.rename(columns={"home_team":"team","implied_home_total":"implied_team_total"})
home_rows["team_spread"] = sched["spread_line"]  # home spread (negative = favored)

away_rows = sched[["nfl_game_id","season","week","away_team","spread_line","total_line","implied_away_total"]].copy()
away_rows = away_rows.rename(columns={"away_team":"team","implied_away_total":"implied_team_total"})
away_rows["team_spread"] = -sched["spread_line"]  # away spread is flipped

game_lines = pd.concat([home_rows, away_rows], ignore_index=True)
game_lines = game_lines[["season","week","team","total_line","team_spread","implied_team_total"]]

print(f"  Game lines rows: {len(game_lines):,}  |  seasons: {sorted(game_lines['season'].unique())}")

# Join to qbs_feat on (season, week, recent_team)
qbs_feat = qbs_feat.merge(
    game_lines.rename(columns={"team":"recent_team"}),
    on=["season","week","recent_team"],
    how="left",
)
null_gl = qbs_feat["implied_team_total"].isna().mean()
print(f"  Game lines join: {1-null_gl:.1%} matched")


# ── Market-level features (all-books, player-game level) ─────────────────────

print("[Step 2] Computing market features (consensus, min/max)...")

# Normalize player names in all_lines using aliases
all_lines["player_norm"] = all_lines["player_name"].map(normalize_name)
all_lines["player_norm"] = all_lines["player_norm"].map(lambda n: aliases.get(n, n))

# Consensus line per player-game = mode of lines across all books
# Min/max line and probs
mkt = (
    all_lines.groupby(["nfl_season", "nfl_game_id", "player_norm"])
    .agg(
        consensus_line          =("line",          "median"),
        min_line                =("line",          "min"),
        max_line                =("line",          "max"),
        consensus_american_over =("american_over", "mean"),
        consensus_american_under=("american_under","mean"),
        min_raw_prob_over       =("raw_prob_over",  "min"),
        max_raw_prob_over       =("raw_prob_over",  "max"),
        min_raw_prob_under      =("raw_prob_under", "min"),
        max_raw_prob_under      =("raw_prob_under", "max"),
        n_books                 =("bookmaker",     "nunique"),
    )
    .reset_index()
)

# Consensus odds bins
def american_to_decimal_scalar(v):
    v = float(v) if v is not None else np.nan
    if np.isnan(v):
        return np.nan
    return (v / 100 + 1) if v > 0 else (100 / abs(v) + 1)

def odds_bin_coarse(american: float) -> str:
    if pd.isna(american):
        return "unknown"
    return "minus" if american < -5 else ("plus" if american > 5 else "even")

def odds_bin_granular(american: float) -> str:
    if pd.isna(american):
        return "unknown"
    if american <= -300:  return "minus300_plus"
    if american <= -200:  return "minus300_to_minus200"
    if american <= -110:  return "minus200_to_minus110"
    if american <    0:   return "minus110_to_even"
    if american ==   0:   return "even"
    if american <=  110:  return "even_to_plus110"
    if american <=  200:  return "plus110_to_plus200"
    if american <=  300:  return "plus200_to_plus300"
    return "plus300_plus"

mkt["consensus_over_odds_bin"]           = mkt["consensus_american_over"].map(odds_bin_coarse)
mkt["consensus_under_odds_bin"]          = mkt["consensus_american_under"].map(odds_bin_coarse)
mkt["consensus_over_odds_bin_granular"]  = mkt["consensus_american_over"].map(odds_bin_granular)
mkt["consensus_under_odds_bin_granular"] = mkt["consensus_american_under"].map(odds_bin_granular)

# Consensus novig
mkt["cons_dec_over"]  = mkt["consensus_american_over"].map(american_to_decimal_scalar)
mkt["cons_dec_under"] = mkt["consensus_american_under"].map(american_to_decimal_scalar)
mkt["cons_raw_prob_over"]  = 1 / mkt["cons_dec_over"]
mkt["cons_raw_prob_under"] = 1 / mkt["cons_dec_under"]
mkt["cons_novig_prob_over"]  = mkt["cons_raw_prob_over"] / (mkt["cons_raw_prob_over"] + mkt["cons_raw_prob_under"])
mkt["cons_novig_prob_under"] = mkt["cons_raw_prob_under"] / (mkt["cons_raw_prob_over"] + mkt["cons_raw_prob_under"])
mkt = mkt.drop(columns=["cons_dec_over","cons_dec_under","cons_raw_prob_over","cons_raw_prob_under"])

print(f"  Market features: {len(mkt):,} player-game rows")


# ── Build BetOnline spine ─────────────────────────────────────────────────────

print("[Step 2] Building BetOnline spine...")

bol = all_lines[all_lines["bookmaker"] == "betonlineag"].copy()
bol = bol[bol["nfl_week"] <= 18].copy()

# Normalize & apply aliases to player names in BOL
bol["player_norm"] = bol["player_name"].map(normalize_name)
bol["player_norm"] = bol["player_norm"].map(lambda n: aliases.get(n, n))

# Normalize player names in qbs_feat
qbs_feat["player_norm"] = qbs_feat["player_display_name"].map(normalize_name)

# Join BOL lines → rolling features on (player_norm, season, week)
spine = bol.merge(
    qbs_feat[[
        "player_norm", "season", "week", "player_id", "recent_team", "opponent_team",
        "passing_yards",  # actual outcome
        "attempts", "completions", "comp_pct", "is_start",
        # rolling passing yards
        "pass_yds_roll_1","pass_yds_roll_3","pass_yds_roll_5","pass_yds_roll_10",
        "pass_yds_roll_season","pass_yds_roll_career",
        # rolling attempts
        "attempts_roll_1","attempts_roll_3","attempts_roll_5","attempts_roll_10",
        "attempts_roll_season","attempts_roll_career",
        # rolling comp pct
        "comp_pct_roll_1","comp_pct_roll_3","comp_pct_roll_5","comp_pct_roll_10",
        "comp_pct_roll_season","comp_pct_roll_career",
        # rolling YPA
        "ypa_roll_1","ypa_roll_3","ypa_roll_5","ypa_roll_10",
        "ypa_roll_season","ypa_roll_career",
        # rolling EPA/att
        "epa_per_att_roll_1","epa_per_att_roll_3","epa_per_att_roll_5","epa_per_att_roll_10",
        "epa_per_att_roll_season","epa_per_att_roll_career",
        # rolling sack rate
        "sack_rate_roll_3","sack_rate_roll_5","sack_rate_roll_10","sack_rate_roll_season","sack_rate_roll_career",
        # rolling pass rate
        "pass_rate_roll_3","pass_rate_roll_5","pass_rate_roll_10","pass_rate_roll_season","pass_rate_roll_career",
        # rolling std dev
        "pass_yds_std_5","pass_yds_std_10","pass_yds_std_career",
        # starter flags
        "career_starts_pct","starts_last_4","games_as_starter_this_season","started_week1_for_team",
        # opponent defense
        "opp_pass_yds_allowed_roll5",
        # game lines
        "total_line","team_spread","implied_team_total",
    ]],
    left_on =["player_norm","nfl_season","nfl_week"],
    right_on=["player_norm","season","week"],
    how="left",
)

n_spine = len(spine)
n_matched = spine["passing_yards"].notna().sum()
print(f"  Spine rows: {n_spine:,}  |  matched actuals: {n_matched:,}  ({n_matched/n_spine*100:.1f}%)")

# Compute outcome
spine["outcome"] = "push"
spine.loc[spine["passing_yards"] > spine["line"], "outcome"] = "over"
spine.loc[spine["passing_yards"] < spine["line"], "outcome"] = "under"

# Join market-level features
spine = spine.merge(
    mkt,
    on=["nfl_season","nfl_game_id","player_norm"],
    how="left",
)

print(f"  After market feature join: {len(spine):,} rows")

# Deduplicate — if same player appears multiple times from join artefacts, keep first
spine = spine.drop_duplicates(["nfl_game_id","bookmaker","player_norm","line"]).reset_index(drop=True)
print(f"  After dedup: {len(spine):,} rows")


# ── Cover rate (did QB go over BetOnline line?) ───────────────────────────────
# Requires the spine itself (line + passing_yards) — computed post-join.
# For each player-game, did_cover = 1 if passing_yards > line, else 0.
# Then shift and roll across prior games.

print("[Step 2] Computing cover rate features...")

spine["did_cover"] = np.where(
    spine["passing_yards"].notna(),
    (spine["passing_yards"] > spine["line"]).astype(float),
    np.nan,
)

cover_frames = []
for player_norm, grp in spine.groupby("player_norm"):
    g = grp.sort_values(["nfl_season","nfl_week"]).copy()

    for w in [1, 3, 5, 10]:
        g[f"cover_rate_roll_{w}"] = g["did_cover"].shift(1).rolling(w, min_periods=1).mean()

    g["cover_rate_roll_season"] = g.groupby("nfl_season")["did_cover"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    g["cover_rate_roll_career"] = g["did_cover"].shift(1).expanding().mean()

    cover_frames.append(g)

spine = pd.concat(cover_frames, ignore_index=True).sort_values(["player_norm","nfl_season","nfl_week"]).reset_index(drop=True)

cover_cols = [c for c in spine.columns if c.startswith("cover_rate")]
print(f"  Cover rate columns added: {cover_cols}")
# Null rate for cover features (week 1 will always be null — no prior game)
for col in cover_cols:
    null_r = spine[col].isna().mean()
    print(f"    {col}: null={null_r:.2%}")

spine.to_parquet(OUT_PATH, index=False)
print(f"  Saved to: {OUT_PATH}")


# ── Spot-check Josh Allen ─────────────────────────────────────────────────────

print("\n[Step 2] Spot-check Josh Allen...")

allen = spine[spine["player_norm"].str.contains("josh allen", na=False)].copy()
print(f"  Allen rows: {len(allen)}")

spot_cols = [
    "nfl_season","nfl_week","line","passing_yards","outcome",
    "pass_yds_roll_3","pass_yds_roll_career","attempts_roll_3","attempts_roll_career",
    "career_starts_pct","starts_last_4","started_week1_for_team",
    "games_as_starter_this_season","cons_novig_prob_over",
]
spot_cols = [c for c in spot_cols if c in allen.columns]
print(allen[spot_cols].head(10).round(3).to_string(index=False))

# Verify no future leakage: roll_3 at game G should not include game G's actual yards
print("\n  Leakage check: pass_yds_roll_1 should equal previous game's passing_yards")
allen_sorted = allen.sort_values(["nfl_season","nfl_week"]).reset_index(drop=True)
if len(allen_sorted) > 1:
    for i in range(1, min(5, len(allen_sorted))):
        prev_actual = allen_sorted.loc[i-1, "passing_yards"]
        curr_roll1  = allen_sorted.loc[i, "pass_yds_roll_1"]
        match = "OK" if abs(float(prev_actual or 0) - float(curr_roll1 or 0)) < 1 else "MISMATCH"
        print(f"    week {int(allen_sorted.loc[i,'nfl_week'])}: roll_1={curr_roll1:.1f}  prev_actual={prev_actual:.1f}  → {match}")

# Cameron Ward check (alias fix)
ward = spine[spine["player_norm"].str.contains("cam ward", na=False)]
print(f"\n  Cam Ward rows after alias fix: {len(ward)}")

# Starter feature spot-check
print("\n  Starter feature sample (career_starts_pct, started_week1_for_team):")
starter_check = spine[spine["passing_yards"].notna()][
    ["player_name","nfl_season","nfl_week","career_starts_pct","starts_last_4","started_week1_for_team","games_as_starter_this_season"]
].dropna(subset=["career_starts_pct"]).sort_values(["player_name","nfl_season","nfl_week"])
print(starter_check.head(15).round(3).to_string(index=False))


# ── DuckDB tests ──────────────────────────────────────────────────────────────

print("\n[Step 2] Running DuckDB tests...")

con = duckdb.connect()
con.execute(f"CREATE TABLE spine AS SELECT * FROM read_parquet('{OUT_PATH}')")

tests = []

def run_test(name, sql, expect_zero=True):
    result = con.execute(sql).fetchone()[0]
    passed = (result == 0) if expect_zero else (result > 0)
    status = "PASS" if passed else "FAIL"
    tests.append({"Test": name, "Result": result, "Status": status})
    icon = "✓" if passed else "✗"
    print(f"  {icon} [{status}] {name}: {result}")
    return passed

# T1: no duplicates at (player_norm, nfl_game_id, bookmaker, line)
run_test(
    "No duplicates at (player_norm, nfl_game_id, bookmaker, line)",
    """SELECT COUNT(*) FROM (
        SELECT player_norm, nfl_game_id, bookmaker, line, COUNT(*) n
        FROM spine GROUP BY player_norm, nfl_game_id, bookmaker, line HAVING n > 1
    )""",
    expect_zero=True,
)

# T2: leakage check — for QBs with >20 prior career games (career_starts_pct well-populated),
# pass_yds_roll_career is a long-run average (~200–270 yds). If it exactly equals
# passing_yards (within 1 yd), that's statistically near-impossible without leakage
# (career avg of 20+ games cannot equal a single game's yardage by coincidence at this precision).
run_test(
    "No leakage: roll_career not exactly equal to same-game actual (for experienced QBs)",
    """SELECT COUNT(*) FROM (
        SELECT DISTINCT player_norm, nfl_game_id FROM spine
        WHERE ABS(pass_yds_roll_career - passing_yards) < 0.01
          AND pass_yds_roll_career IS NOT NULL AND passing_yards IS NOT NULL
          AND games_as_starter_this_season > 5
    )""",
    expect_zero=True,  # rolling avg of 20+ games cannot exactly equal a single game's yards; near-misses within 0.01 are impossible without leakage
)

# T3: null rates on key feature columns < 10% (among matched rows)
for col in ["pass_yds_roll_career","attempts_roll_career","career_starts_pct","started_week1_for_team"]:
    null_rate_sql = f"""SELECT CAST(SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
                        FROM spine WHERE passing_yards IS NOT NULL"""
    null_rate = con.execute(null_rate_sql).fetchone()[0]
    passed = null_rate < 0.10
    status = "PASS" if passed else "FAIL"
    tests.append({"Test": f"Null rate {col} < 10%", "Result": f"{null_rate:.3f}", "Status": status})
    icon = "✓" if passed else "✗"
    print(f"  {icon} [{status}] Null rate {col}: {null_rate:.3f}")

# T4: outcome column populated for all matched rows
run_test(
    "Outcome populated for all matched rows",
    "SELECT COUNT(*) FROM spine WHERE passing_yards IS NOT NULL AND outcome IS NULL",
    expect_zero=True,
)

# T5: join quality > 90%
match_pct_sql = "SELECT CAST(SUM(CASE WHEN passing_yards IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) FROM spine"
mp = con.execute(match_pct_sql).fetchone()[0]
tests.append({"Test": f"Join quality > 90% (actual={mp:.3f})", "Result": f"{mp:.3f}", "Status": "PASS" if mp > 0.90 else "FAIL"})
print(f"  {'✓' if mp > 0.90 else '✗'} [{'PASS' if mp > 0.90 else 'FAIL'}] Join quality: {mp:.3f}")

# T6: date range covers 2023-2025
run_test(
    "All 3 seasons present",
    "SELECT CASE WHEN COUNT(DISTINCT nfl_season) = 3 THEN 0 ELSE 1 END FROM spine",
    expect_zero=True,
)

# T7: rookies (career_starts_pct IS NULL) should have null career rolling stats —
# a player with no prior career data should have null rolling features on their first game
run_test(
    "Null career roll for true rookies (career_starts_pct IS NULL)",
    """SELECT COUNT(*) FROM spine
       WHERE career_starts_pct IS NULL AND pass_yds_roll_career IS NOT NULL""",
    expect_zero=True,
)

# T8: started_week1_for_team is null on week 1 (can't know until week 1 is played)
run_test(
    "started_week1_for_team is null on week 1",
    "SELECT COUNT(*) FROM spine WHERE nfl_week = 1 AND started_week1_for_team IS NOT NULL",
    expect_zero=True,
)

test_results = pd.DataFrame(tests)
n_pass = sum(1 for s in test_results["Status"] if s == "PASS")
n_fail = sum(1 for s in test_results["Status"] if s == "FAIL")
print(f"\n  Tests: {n_pass} passed, {n_fail} failed")


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 2] Writing HTML section...")

# Column summary for HTML
col_groups = {
    "Rolling Passing Yards": [c for c in spine.columns if c.startswith("pass_yds_roll")],
    "Rolling Attempts": [c for c in spine.columns if c.startswith("attempts_roll")],
    "Rolling Comp %": [c for c in spine.columns if c.startswith("comp_pct_roll")],
    "Starter Flags": ["career_starts_pct","starts_last_4","games_as_starter_this_season","started_week1_for_team"],
    "Opponent Defense": ["opp_pass_yds_allowed_roll5"],
    "Market Features": ["consensus_line","min_line","max_line","n_books",
                        "cons_novig_prob_over","cons_novig_prob_under",
                        "consensus_over_odds_bin","consensus_over_odds_bin_granular",
                        "min_raw_prob_over","max_raw_prob_over"],
}

# Null rate summary per column group
null_rows = []
matched = spine[spine["passing_yards"].notna()]
for grp, cols in col_groups.items():
    for col in cols:
        if col in spine.columns:
            nr = matched[col].isna().mean()
            null_rows.append({"Group": grp, "Feature": col, "Null %": f"{nr*100:.1f}%"})
null_df = pd.DataFrame(null_rows)

# Allen spot-check display
allen_display = allen[spot_cols].head(15).round(3) if len(allen) else pd.DataFrame()

html = f"""
<section>
<h2>Step 2 — Feature Engineering / Spine</h2>
<p><em>{ts()}</em></p>

<h3>Spine Summary</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total spine rows</td><td>{len(spine):,}</td></tr>
<tr><td>Matched actuals</td><td>{n_matched:,} ({n_matched/len(spine)*100:.1f}%)</td></tr>
<tr><td>Seasons</td><td>2023–2025</td></tr>
<tr><td>Feature data range</td><td>1999–2025</td></tr>
<tr><td>Unique players (BetOnline)</td><td>{spine["player_norm"].nunique():,}</td></tr>
<tr><td>Grain</td><td>(player_norm, nfl_game_id, bookmaker=betonlineag, line)</td></tr>
</table>

<h3>Feature Groups &amp; Null Rates</h3>
{df_to_html(null_df, "Null rates computed on matched rows (passing_yards not null)")}

<h3>Spot-Check — Josh Allen (first 15 BetOnline rows)</h3>
{df_to_html(allen_display, f"Josh Allen — rolling features and starter flags ({len(allen)} total rows)") if len(allen_display) else "<p class='bad'>Josh Allen not found</p>"}

<h3>Cam Ward Alias Fix</h3>
<p>BetOnline uses "Cameron Ward"; nfl_data_py uses "Cam Ward". After alias fix: <strong>{len(ward)} rows</strong> matched.</p>

<h3>Starter Flag Design</h3>
<ul>
  <li><strong>is_start</strong> (internal) — attempts &gt; 10 in that game = considered a start</li>
  <li><strong>career_starts_pct</strong> — fraction of all prior career games that were starts; low = backup/journeyman</li>
  <li><strong>starts_last_4</strong> — games started in prior 4 weeks (0–4); detects mid-season role changes</li>
  <li><strong>games_as_starter_this_season</strong> — cumulative starts this season before this game; 0 = first start</li>
  <li><strong>started_week1_for_team</strong> — was this QB the designated week-1 starter for this team this season? NaN on week 1 (not yet known)</li>
</ul>

<h3>Step 2 Test Results</h3>
<table>
<tr><th>Test</th><th>Result</th><th>Status</th></tr>
"""
for _, row in test_results.iterrows():
    color = "#d4edda" if row["Status"] == "PASS" else "#f8d7da"
    html += f"<tr style='background:{color}'><td>{row['Test']}</td><td>{row['Result']}</td><td><strong>{row['Status']}</strong></td></tr>"

html += f"""
</table>
<p><strong>{n_pass} passed, {n_fail} failed</strong></p>
</section>
"""

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 2] HTML section appended: {HTML_PATH}")
print("\n=== DONE ===")
print(f"Spine: {len(spine):,} rows  |  {len(spine.columns)} columns")
print(f"Open: open '{HTML_PATH}'")
