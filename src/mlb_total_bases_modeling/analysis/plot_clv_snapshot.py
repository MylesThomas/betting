"""
Plot odds movement + CLV markers for a single player on a given game date.

Loads all snapshots for the game_date (including those captured on prior days),
plots pre-game odds movement per bookmaker, with day-boundary and first-pitch markers.

By default saves to /tmp/ and opens in Chrome. Pass --show to display interactively instead.

Usage:
  uv run python src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py --game-date 2026-09-07 --player "Blaze Alexander"
  uv run python src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py --game-date 2026-09-07 --player "Blaze Alexander" --show
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SNAPSHOT_ROOT = Path.home() / "Downloads/tmp/total_bases/prop_snapshots"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
CONSENSUS_COLOR = "black"
GAME_START_COLOR = "#cc3333"
DAY_BOUNDARY_COLOR = "#aaaaaa"


def _to_et(ts: str) -> datetime:
    return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=UTC).astimezone(ET)


def _to_epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=UTC).timestamp()


def load_player(game_date: str, player: str) -> pd.DataFrame:
    year = game_date[:4]
    folder = SNAPSHOT_ROOT / year / game_date
    files = list(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets found in {folder} — run aws s3 sync first")

    frames = []
    for f in sorted(files):
        frame = pd.read_parquet(f)
        # parse UTC timestamp from filename → ET display
        ts_part = f.stem.replace("snapshot_", "")  # e.g. 20260907_113147
        ts_utc  = datetime.strptime(ts_part, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        ts_et   = ts_utc.astimezone(ET).strftime("%-I:%M %p ET")
        print(f"  {f.name}  ({ts_et})  →  {len(frame):,} rows")
        frames.append(frame)
    print(f"  total files: {len(frames)}, total rows: {sum(len(f) for f in frames):,}")
    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["player_name"] == player)
        & (df["market_key"] == "batter_total_bases")
        & (df["over_line"] == 1.5)
        & df["under_american_odds"].notna()
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for player='{player}' on {game_date}")

    df["snapshot_et"]    = df["snapshot_ts_utc"].apply(_to_et)
    df["snapshot_epoch"] = df["snapshot_ts_utc"].apply(_to_epoch)
    df = df.sort_values("snapshot_epoch")

    # Drop post-game-start snapshots — in-play odds are garbage for CLV
    commence_epoch = _to_epoch(df["commence_time"].iloc[0])
    df = df[df["snapshot_epoch"] <= commence_epoch].copy()

    return df


def _midnight_et_boundaries(df: pd.DataFrame) -> list[datetime]:
    """Return midnight ET for each calendar-day boundary in the data range."""
    if df.empty:
        return []
    start = df["snapshot_et"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    end   = df["snapshot_et"].max()
    boundaries = []
    cur = start + timedelta(days=1)
    while cur <= end:
        boundaries.append(cur)
        cur += timedelta(days=1)
    return boundaries


def plot(game_date: str, player: str, show: bool = False) -> None:
    df = load_player(game_date, player)

    commence_time  = df["commence_time"].iloc[0]
    commence_epoch = _to_epoch(commence_time)
    commence_et    = _to_et(commence_time)

    books = sorted(df["bookmaker"].unique())
    cmap  = plt.colormaps["tab10"]
    book_colors = {b: cmap(i / max(len(books), 1)) for i, b in enumerate(books)}

    fig, ax = plt.subplots(figsize=(14, 6))

    # ── Day-boundary vertical lines ───────────────────────────────────────────
    for boundary in _midnight_et_boundaries(df):
        ax.axvline(boundary, color=DAY_BOUNDARY_COLOR, linestyle="--", linewidth=1, zorder=1)
        ax.text(boundary, ax.get_ylim()[1], f" {boundary.strftime('%b %-d')}",
                color=DAY_BOUNDARY_COLOR, fontsize=7, va="top", rotation=0)

    # ── Per-book lines ────────────────────────────────────────────────────────
    first_seen_odds: dict[str, int] = {}
    closing_odds:    dict[str, int] = {}

    for book in books:
        bdf   = df[df["bookmaker"] == book].sort_values("snapshot_epoch")
        color = book_colors[book]
        ax.plot(bdf["snapshot_et"], bdf["under_american_odds"],
                color=color, linewidth=1.5, label=book)

        # First-seen marker (★)
        fs = bdf[bdf["binary_player_game_first_seen"].astype(bool)]
        if not fs.empty:
            fs_odds = int(fs["under_american_odds"].iloc[0])
            first_seen_odds[book] = fs_odds
            ax.scatter(fs["snapshot_et"].iloc[0], fs_odds,
                       color=color, marker="*", s=200, zorder=5)

        # Closing marker (◆) — last snapshot before first pitch
        pre = bdf[bdf["snapshot_epoch"] < commence_epoch]
        if not pre.empty:
            cl_odds = int(pre["under_american_odds"].iloc[-1])
            closing_odds[book] = cl_odds
            ax.scatter(pre["snapshot_et"].iloc[-1], cl_odds,
                       color=color, marker="D", s=60, zorder=5)

    # ── Consensus median line ─────────────────────────────────────────────────
    consensus = (
        df.groupby("snapshot_et")["under_american_odds"]
        .median()
        .reset_index()
        .sort_values("snapshot_et")
    )
    ax.plot(consensus["snapshot_et"], consensus["under_american_odds"],
            color=CONSENSUS_COLOR, linewidth=2.5, linestyle="--",
            label="consensus (median)", zorder=4)

    # ── First pitch vertical line ─────────────────────────────────────────────
    ax.axvline(commence_et, color=GAME_START_COLOR, linestyle="-", linewidth=2,
               label=f"first pitch ({commence_et.strftime('%H:%M ET')})", zorder=3)

    # ── CLV annotation box ────────────────────────────────────────────────────
    shared = [b for b in books if b in first_seen_odds and b in closing_odds]
    if shared:
        lines = ["CLV (first-seen vs close)"]
        for book in shared:
            clv  = first_seen_odds[book] - closing_odds[book]
            sign = "+" if clv >= 0 else ""
            lines.append(f"{book}: {sign}{clv}¢")
        ax.annotate(
            "\n".join(lines),
            xy=(0.99, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
        )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"{player}  |  game date {game_date}  |  UNDER 1.5 total bases", fontsize=13)
    ax.set_xlabel("Time (ET)")
    ax.set_ylabel("Under American odds")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H:%M", tz=ET))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=ET))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.25)
    fig.text(0.5, 0.01, "★ first-seen price   ◆ closing price   --- consensus median",
             ha="center", fontsize=8, color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if show:
        plt.show()
    else:
        slug = player.lower().replace(" ", "_")
        out  = Path(f"/tmp/mlb_tb_{slug}_{game_date}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")
        subprocess.run(["open", "-a", "Google Chrome", str(out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-date", required=True, dest="game_date", help="Game date YYYY-MM-DD")
    parser.add_argument("--player",    required=True, help="Player name (case-sensitive)")
    parser.add_argument("--show",      action="store_true", help="Display interactively instead of saving")
    args = parser.parse_args()
    plot(args.game_date, args.player, show=args.show)


if __name__ == "__main__":
    main()
