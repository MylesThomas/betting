"""
Compare v1 / v2 / v3 historical spines to validate sack source changes.

v1: PBP extraction, full_to_pbp() has suffix bug (broken)
v2: PBP extraction, suffix bug fixed
v3: PFR box score (def_sacks), joined on pfr_player_id — no name matching

All comparisons are restricted to 2018–2025 (v3 availability window).

Run:
    python src/nfl_sacks_modeling/scripts/compare_spine_v1_v2.py
"""

from pathlib import Path
import duckdb
import pandas as pd

V1 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine.parquet"
V2 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine_v2.parquet"
V3 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine_v3.parquet"

SUFFIX_PLAYERS = [
    "George Karlaftis", "Patrick Jones", "Dorance Armstrong",
    "Deatrich Wise", "Broderick Washington", "Willie Gay",
    "Antoine Winfield", "Devin Bush", "Jessie Bates",
    "Maurice Hurst", "Jeremiah Trotter", "Andre Carter",
]

# Players who had unexpected losses in v2 vs v1
LOSS_PLAYERS = ["Robert Quinn", "Khalil Mack", "Joey Bosa", "Aaron Donald",
                "Chris Long", "Melvin Ingram", "Bruce Irvin"]


def header(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    missing = [(p, lbl) for p, lbl in [(V1,"v1"),(V2,"v2"),(V3,"v3")] if not p.exists()]
    if missing:
        for p, lbl in missing:
            script = f"build_historical_spine{'_'+lbl if lbl!='v1' else ''}.py"
            print(f"Missing {lbl}: run {script} first")
        return

    con = duckdb.connect()
    con.execute(f"CREATE VIEW v1 AS SELECT * FROM read_parquet('{V1}') WHERE season >= 2018")
    con.execute(f"CREATE VIEW v2 AS SELECT * FROM read_parquet('{V2}') WHERE season >= 2018")
    con.execute(f"CREATE VIEW v3 AS SELECT * FROM read_parquet('{V3}')")

    # ── 1. Row and player counts ───────────────────────────────────────────────
    header("1. Row and player counts (2018–2025 only)")
    print(con.execute("""
        SELECT version, COUNT(*) AS rows, COUNT(DISTINCT pfr_player_id) AS players,
               ROUND(SUM(sacks),1) AS total_sacks
        FROM (
            SELECT 'v1' AS version, pfr_player_id, sacks FROM v1
            UNION ALL
            SELECT 'v2', pfr_player_id, sacks FROM v2
            UNION ALL
            SELECT 'v3', pfr_player_id, sacks FROM v3
        )
        GROUP BY version ORDER BY version
    """).df().to_string(index=False))

    # ── 2. Total sacks per season, all three versions ──────────────────────────
    header("2. Total sacks by season — v1 vs v2 vs v3")
    print(con.execute("""
        SELECT season,
            ROUND(SUM(CASE WHEN src='v1' THEN sacks ELSE 0 END),1) AS v1,
            ROUND(SUM(CASE WHEN src='v2' THEN sacks ELSE 0 END),1) AS v2,
            ROUND(SUM(CASE WHEN src='v3' THEN sacks ELSE 0 END),1) AS v3,
            ROUND(SUM(CASE WHEN src='v2' THEN sacks ELSE 0 END)
                - SUM(CASE WHEN src='v1' THEN sacks ELSE 0 END),1) AS v2_v1_delta,
            ROUND(SUM(CASE WHEN src='v3' THEN sacks ELSE 0 END)
                - SUM(CASE WHEN src='v1' THEN sacks ELSE 0 END),1) AS v3_v1_delta
        FROM (
            SELECT season, sacks, 'v1' AS src FROM v1
            UNION ALL SELECT season, sacks, 'v2' FROM v2
            UNION ALL SELECT season, sacks, 'v3' FROM v3
        )
        GROUP BY season ORDER BY season
    """).df().to_string(index=False))

    # ── 3. Suffix players — all 3 versions ────────────────────────────────────
    header("3. Known suffix players — career sacks across all 3 versions")
    name_filter = " OR ".join(f"player ILIKE '%{n}%'" for n in SUFFIX_PLAYERS)
    print(con.execute(f"""
        WITH totals AS (
            SELECT src, player, pfr_player_id, ROUND(SUM(sacks),1) AS career_sacks
            FROM (
                SELECT 'v1' AS src, player, pfr_player_id, sacks FROM v1
                UNION ALL SELECT 'v2', player, pfr_player_id, sacks FROM v2
                UNION ALL SELECT 'v3', player, pfr_player_id, sacks FROM v3
            )
            WHERE {name_filter}
            GROUP BY src, player, pfr_player_id
        )
        SELECT player,
            MAX(CASE WHEN src='v1' THEN career_sacks END) AS v1,
            MAX(CASE WHEN src='v2' THEN career_sacks END) AS v2,
            MAX(CASE WHEN src='v3' THEN career_sacks END) AS v3
        FROM totals
        GROUP BY player, pfr_player_id
        ORDER BY player
    """).df().to_string(index=False))

    # ── 4. Unexpected loss players — all 3 versions ───────────────────────────
    header("4. Players with unexpected losses in v2 — v3 as tiebreaker")
    loss_filter = " OR ".join(f"player ILIKE '%{n}%'" for n in LOSS_PLAYERS)
    print(con.execute(f"""
        WITH totals AS (
            SELECT src, player, pfr_player_id, ROUND(SUM(sacks),1) AS career_sacks
            FROM (
                SELECT 'v1' AS src, player, pfr_player_id, sacks FROM v1
                UNION ALL SELECT 'v2', player, pfr_player_id, sacks FROM v2
                UNION ALL SELECT 'v3', player, pfr_player_id, sacks FROM v3
            )
            WHERE {loss_filter}
            GROUP BY src, player, pfr_player_id
        )
        SELECT player,
            MAX(CASE WHEN src='v1' THEN career_sacks END) AS v1,
            MAX(CASE WHEN src='v2' THEN career_sacks END) AS v2,
            MAX(CASE WHEN src='v3' THEN career_sacks END) AS v3,
            MAX(CASE WHEN src='v3' THEN career_sacks END)
              - MAX(CASE WHEN src='v1' THEN career_sacks END) AS v3_v1_delta
        FROM totals
        GROUP BY player, pfr_player_id
        ORDER BY v3_v1_delta DESC
    """).df().to_string(index=False))

    # ── 5. All players changed — v3 as ground truth ───────────────────────────
    header("5. Players where v1 and v3 disagree most (v3 = ground truth)")
    print(con.execute("""
        WITH v1t AS (SELECT pfr_player_id, player, SUM(sacks) AS v1 FROM v1 GROUP BY pfr_player_id, player),
             v3t AS (SELECT pfr_player_id,         SUM(sacks) AS v3 FROM v3 GROUP BY pfr_player_id)
        SELECT a.player, ROUND(a.v1,1) AS v1, ROUND(b.v3,1) AS v3,
               ROUND(b.v3 - a.v1, 1) AS delta
        FROM v1t a JOIN v3t b USING (pfr_player_id)
        WHERE ABS(b.v3 - a.v1) > 2
        ORDER BY ABS(b.v3 - a.v1) DESC
        LIMIT 40
    """).df().to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
