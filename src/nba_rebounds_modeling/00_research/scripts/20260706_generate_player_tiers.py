"""
Classify NBA players into recognition tiers using Claude Haiku.

Plan:
  1. Load unique player_normalized names from rebounds_props.parquet (~499 players)
  2. If output parquet already exists (and no --overwrite), load it as the cache
     so a partial run can be resumed without re-calling the API
  3. For each player not yet classified, call Claude Haiku in batches of 50
  4. After each batch, persist only the classified-so-far rows to the output parquet
     (so remaining players are absent, not "unknown", preserving resume correctness)
  5. After all batches, write the final parquet with all 499 players

Tiers: superstar / known_starter / fringe / unknown

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/20260706_generate_player_tiers.py
    python src/nba_rebounds_modeling/00_research/scripts/20260706_generate_player_tiers.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TIERS = ("superstar", "known_starter", "fringe", "unknown")
BATCH_SIZE = 50

SYSTEM_PROMPT = """\
You are an NBA expert. For each player name I give you, classify their general public recognition tier \
as of the 2025-26 NBA season using exactly one of these labels:

  superstar     — household name; perennial All-Star or All-NBA; casual fans know them instantly
  known_starter — solid starting-caliber player; knowledgeable fans know them well
  fringe        — rotation / bench player; only dedicated fans recognize them
  unknown       — two-way contract, G-League call-up, or so rarely played that even die-hards may not know them

Rules:
- Use the player's peak reputation through the 2025-26 season.
- A player who was a superstar but has declined is still "superstar" if the general public still recognizes them.
- Return ONLY valid JSON: an object mapping each player name to one of the four tier strings.
- Do not add commentary or markdown fences — raw JSON only.

Example input: ["LeBron James", "Kyle Anderson", "Isaiah Todd"]
Example output: {"LeBron James": "superstar", "Kyle Anderson": "fringe", "Isaiah Todd": "unknown"}
"""


def classify_batch(client: anthropic.Anthropic, players: list[str]) -> dict[str, str]:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(players)}],
    )
    text = response.content[0].text.strip()
    # Strip markdown fences if the model wraps output despite instructions.
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def save_progress(cache: dict[str, str], out_path: Path) -> None:
    """Write only classified players — absent rows stay absent so resume works."""
    rows = [(p, t) for p, t in cache.items()]
    df = pd.DataFrame(rows, columns=["player_normalized", "recognition_tier"])
    df["recognition_tier"] = pd.Categorical(df["recognition_tier"], categories=list(TIERS), ordered=True)
    df.to_parquet(out_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--props", default="~/Downloads/tmp/rebounds_props.parquet")
    p.add_argument("--out", default="~/Downloads/tmp/rebounds_player_tiers.parquet")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--overwrite", action="store_true", help="Reclassify all players, ignoring existing output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    props_path = Path(args.props).expanduser()
    out_path = Path(args.out).expanduser()

    all_players = sorted(
        pd.read_parquet(props_path, columns=["player_normalized"])["player_normalized"]
        .dropna()
        .unique()
        .tolist()
    )
    print(f"Unique players: {len(all_players)}")

    cache: dict[str, str] = {}
    if not args.overwrite and out_path.exists():
        existing = pd.read_parquet(out_path)
        cache = dict(zip(existing["player_normalized"], existing["recognition_tier"].astype(str)))
        print(f"Resuming: {len(cache)} already classified")

    remaining = [p for p in all_players if p not in cache]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
    else:
        client = anthropic.Anthropic()
        batches = [remaining[i: i + args.batch_size] for i in range(0, len(remaining), args.batch_size)]

        for idx, batch in enumerate(batches, 1):
            print(f"Batch {idx}/{len(batches)} ({len(batch)} players)...", end=" ", flush=True)
            try:
                result = classify_batch(client, batch)
                for player, tier in result.items():
                    cache[player] = tier if tier in TIERS else "unknown"
                    if tier not in TIERS:
                        print(f"\n[WARN] bad tier '{tier}' for '{player}' → unknown")
                for player in batch:
                    if player not in cache:
                        print(f"\n[WARN] missing '{player}' in response → unknown")
                        cache[player] = "unknown"
                print("done")
            except anthropic.AuthenticationError:
                raise
            except Exception as e:
                print(f"FAILED: {e}")
                for player in batch:
                    cache.setdefault(player, "unknown")

            # Save only classified rows so remaining players stay absent (not "unknown").
            save_progress(cache, out_path)

            if idx < len(batches):
                time.sleep(0.3)

    # Final write: all players, unclassified ones get "unknown".
    final_rows = [(p, cache.get(p, "unknown")) for p in all_players]
    final_df = pd.DataFrame(final_rows, columns=["player_normalized", "recognition_tier"])
    final_df["recognition_tier"] = pd.Categorical(final_df["recognition_tier"], categories=list(TIERS), ordered=True)
    final_df.to_parquet(out_path, index=False)
    print(f"\nDone. {len(final_df)} rows → {out_path}")
    print(final_df["recognition_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
