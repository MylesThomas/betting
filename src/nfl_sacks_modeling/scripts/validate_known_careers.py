"""
Validate known career/season sack totals against v1, v2, v3 spines.

v3 only covers 2018+, so career checks compare over the v3 window only.

Run:
    python src/nfl_sacks_modeling/scripts/validate_known_careers.py
"""

from pathlib import Path
import duckdb

V1 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine.parquet"
V2 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine_v2.parquet"
V3 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine_v3.parquet"

V3_MIN_SEASON = 2018

KNOWN = {
    "Rashan Gary": {
        2019: 2.0, 2020: 5.0, 2021: 9.5, 2022: 6.0,
        2023: 9.0, 2024: 7.5, 2025: 7.5,
        "career": 46.5,
    },
    "Aaron Donald": {
        2014: 9.0, 2015: 11.0, 2016: 8.0, 2017: 11.0, 2018: 20.5,
        2019: 12.5, 2020: 13.5, 2021: 12.5, 2022: 5.0, 2023: 8.0,
        "career": 111.0,
    },
    "Myles Garrett": {
        2017: 7.0, 2018: 13.5, 2019: 10.0, 2020: 12.0, 2021: 16.0,
        2022: 16.0, 2023: 14.0, 2024: 14.0, 2025: 23.0,
        "career": 125.5,
    },
    "Micah Parsons": {
        2021: 13.0, 2022: 13.5, 2023: 14.0, 2024: 12.0, 2025: 12.5,
        "career": 65.0,
    },
}

SEP = "=" * 70


def run():
    missing = [(p, l) for p, l in [(V1, "v1"), (V2, "v2"), (V3, "v3")] if not p.exists()]
    if missing:
        for p, l in missing:
            print(f"Missing {l}: {p}")
        return

    con = duckdb.connect()
    con.execute(f"CREATE VIEW v1 AS SELECT * FROM read_parquet('{V1}')")
    con.execute(f"CREATE VIEW v2 AS SELECT * FROM read_parquet('{V2}')")
    con.execute(f"CREATE VIEW v3 AS SELECT * FROM read_parquet('{V3}')")

    for player, expected in KNOWN.items():
        print(f"\n{SEP}\n  {player}\n{SEP}")

        career_expected = expected["career"]
        season_rows = {k: v for k, v in expected.items() if k != "career"}

        rows = con.execute(f"""
            WITH combined AS (
                SELECT 'v1' AS src, season, SUM(sacks) AS sacks
                FROM v1 WHERE player ILIKE '%{player}%' GROUP BY season
                UNION ALL
                SELECT 'v2', season, SUM(sacks)
                FROM v2 WHERE player ILIKE '%{player}%' GROUP BY season
                UNION ALL
                SELECT 'v3', season, SUM(sacks)
                FROM v3 WHERE player ILIKE '%{player}%' GROUP BY season
            )
            SELECT season,
                MAX(CASE WHEN src='v1' THEN ROUND(sacks,1) END) AS v1,
                MAX(CASE WHEN src='v2' THEN ROUND(sacks,1) END) AS v2,
                MAX(CASE WHEN src='v3' THEN ROUND(sacks,1) END) AS v3
            FROM combined
            GROUP BY season ORDER BY season
        """).df()

        if rows.empty:
            print("  NOT FOUND in any spine")
            continue

        rows["expected"] = rows["season"].map(season_rows)

        def v3_status(r):
            if r["expected"] is None:
                return "n/a"
            if r["v3"] is None or (r["v3"] != r["v3"]):  # NaN check
                return "n/a (pre-2018)"
            return "✓" if abs(r["v3"] - r["expected"]) < 0.1 else "✗"

        rows["v3_ok"] = rows.apply(v3_status, axis=1)
        print(rows[["season", "expected", "v1", "v2", "v3", "v3_ok"]].to_string(index=False))

        # Career — compare over v3 window only
        v3_seasons = {s: v for s, v in season_rows.items() if s >= V3_MIN_SEASON}
        career_v3_expected = sum(v3_seasons.values())

        career_v1 = con.execute(f"SELECT ROUND(SUM(sacks),1) FROM v1 WHERE player ILIKE '%{player}%'").fetchone()[0]
        career_v2 = con.execute(f"SELECT ROUND(SUM(sacks),1) FROM v2 WHERE player ILIKE '%{player}%'").fetchone()[0]
        career_v3 = con.execute(f"SELECT ROUND(SUM(sacks),1) FROM v3 WHERE player ILIKE '%{player}%'").fetchone()[0]

        v3_career_ok = "✓" if career_v3 is not None and abs(career_v3 - career_v3_expected) < 0.1 else "✗"
        print(f"\n  Career (all yrs) expected={career_expected}  v1={career_v1}  v2={career_v2}")
        print(f"  Career (2018+)   expected={career_v3_expected}  v3={career_v3}  v3_ok={v3_career_ok}")

    print()


if __name__ == "__main__":
    run()
