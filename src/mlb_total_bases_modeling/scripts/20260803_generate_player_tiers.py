"""
Classify MLB batters into recognition tiers using Claude Haiku.

  1. Load unique (name_norm, player_name) pairs from regression spine on S3
  2. If output parquet already exists (no --overwrite), resume from cache
  3. Classify via Claude Haiku in batches of 50 (display name → tier, keyed by name_norm)
  4. Save progress after each batch
  5. Final write: all players

Tiers: superstar / known_starter / fringe / unknown

  superstar     — household name; perennial All-Star; casual baseball fans know them instantly
  known_starter — solid MLB regular; knowledgeable fans know them well
  fringe        — bench / platoon / minor league call-up; only dedicated fans recognize them
  unknown       — rarely appeared; even die-hards may not know them by name

Usage:
  python src/mlb_total_bases_modeling/scripts/20260803_generate_player_tiers.py
  python src/mlb_total_bases_modeling/scripts/20260803_generate_player_tiers.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path

import anthropic
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = "the-odds-api-mt"
SPINE_KEY = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"

TIERS      = ("superstar", "known_starter", "fringe", "unknown")
BATCH_SIZE = 50
OUT_PATH   = Path.home() / "Downloads/tmp/mlb_total_bases/mlb_tb_player_tiers.parquet"

SYSTEM_PROMPT = """\
You are an MLB expert. For each player name I give you, classify their general public recognition tier \
as of the 2025-26 MLB season using exactly one of these labels:

  superstar     — household name; perennial All-Star or Silver Slugger; casual baseball fans know them instantly
                  (e.g. Shohei Ohtani, Aaron Judge, Mookie Betts, Freddie Freeman, Bryce Harper)
  known_starter — solid MLB regular; starts most games; knowledgeable fans know them well
  fringe        — bench / platoon / call-up player; only dedicated fans or fantasy players recognize them
  unknown       — rarely appeared, minor league call-up, or so new that even die-hards may not know them

Rules:
- Use the player's peak reputation through the 2025-26 MLB season.
- A player who was a superstar but has declined is still "superstar" if casual fans still recognize them by name.
- Return ONLY valid JSON: an object mapping each player name to one of the four tier strings.
- Do not add commentary or markdown fences — raw JSON only.

Example input: ["Shohei Ohtani", "Tommy Edman", "Bligh Madris"]
Example output: {"Shohei Ohtani": "superstar", "Tommy Edman": "known_starter", "Bligh Madris": "fringe"}
"""


def classify_batch(client: anthropic.Anthropic, players: list[str]) -> dict[str, str]:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(players)}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def save_progress(cache: dict[str, str], norm_to_display: dict[str, str], out_path: Path) -> None:
    rows = [(norm, norm_to_display.get(norm, norm), cache[norm]) for norm in cache]
    df = pd.DataFrame(rows, columns=["name_norm", "player_name", "recognition_tier"])
    df["recognition_tier"] = pd.Categorical(df["recognition_tier"], categories=list(TIERS), ordered=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("Loading spine from S3...")
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)
    spine = pd.read_parquet(BytesIO(obj["Body"].read()), columns=["name_norm"])
    spine = spine.dropna(subset=["name_norm"])

    # Display name = title-cased norm (e.g. "shohei ohtani" → "Shohei Ohtani")
    all_norms = sorted(spine["name_norm"].unique().tolist())
    norm_to_display = {n: n.title() for n in all_norms}
    print(f"Unique players: {len(all_norms)}")

    cache: dict[str, str] = {}
    if not args.overwrite and OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        cache = dict(zip(existing["name_norm"], existing["recognition_tier"].astype(str)))
        print(f"Resuming: {len(cache)} already classified")

    remaining = [n for n in all_norms if n not in cache]
    print(f"Remaining: {len(remaining)}")

    if remaining:
        client  = anthropic.Anthropic()
        batches = [remaining[i: i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
        for idx, batch in enumerate(batches, 1):
            display_names = [norm_to_display.get(n, n) for n in batch]
            print(f"Batch {idx}/{len(batches)} ({len(batch)} players)...", end=" ", flush=True)
            try:
                result = classify_batch(client, display_names)
                # result keys are display names; map back to norm
                display_to_norm = {norm_to_display.get(n, n): n for n in batch}
                for display, tier in result.items():
                    norm = display_to_norm.get(display, display)
                    cache[norm] = tier if tier in TIERS else "unknown"
                    if tier not in TIERS:
                        print(f"\n[WARN] bad tier '{tier}' for '{display}' → unknown")
                for norm in batch:
                    if norm not in cache:
                        print(f"\n[WARN] missing '{norm_to_display.get(norm, norm)}' → unknown")
                        cache[norm] = "unknown"
                print("done")
            except anthropic.AuthenticationError:
                raise
            except Exception as e:
                print(f"FAILED: {e}")
                for norm in batch:
                    cache.setdefault(norm, "unknown")

            save_progress(cache, norm_to_display, OUT_PATH)
            if idx < len(batches):
                time.sleep(0.3)

    # Final write
    final_rows = [(n, norm_to_display.get(n, n), cache.get(n, "unknown")) for n in all_norms]
    final_df = pd.DataFrame(final_rows, columns=["name_norm", "player_name", "recognition_tier"])
    final_df["recognition_tier"] = pd.Categorical(final_df["recognition_tier"], categories=list(TIERS), ordered=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_PATH, index=False)
    print(f"\nDone. {len(final_df)} players → {OUT_PATH}")
    print(final_df["recognition_tier"].value_counts().to_string())

    # Spot-check a few names
    for name in ["shohei ohtani", "aaron judge", "freddie freeman", "tommy edman"]:
        row = final_df[final_df["name_norm"] == name]
        if not row.empty:
            print(f"  {name}: {row.iloc[0]['recognition_tier']}")


if __name__ == "__main__":
    main()
