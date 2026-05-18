"""
Diagnose season type needed per date.
Tests Regular Season, PlayIn, and Playoffs for key dates.
"""
import time
import requests
import urllib3
urllib3.disable_warnings()

from nba_api.stats.endpoints import playergamelogs

SEASON = '2025-26'
TEST_DATES = [
    ('2026-04-14', 'play-in day 1'),
    ('2026-04-15', 'play-in day 2'),
    ('2026-04-17', 'play-in day 3'),
    ('2026-04-19', 'playoffs round 1'),
]
SEASON_TYPES = ['Regular Season', 'PlayIn', 'Playoffs']

for date_str, label in TEST_DATES:
    print(f"\n=== {date_str} ({label}) ===")
    for stype in SEASON_TYPES:
        t0 = time.time()
        try:
            gl = playergamelogs.PlayerGameLogs(
                season_nullable=SEASON,
                season_type_nullable=stype,
                date_from_nullable=date_str,
                date_to_nullable=date_str
            )
            df = gl.get_data_frames()[0]
            elapsed = time.time() - t0
            if not df.empty:
                sample_min = df['MIN'].head(3).tolist()
                print(f"  {stype:<18} {elapsed:.1f}s  → {len(df)} rows  MIN sample: {sample_min}")
            else:
                print(f"  {stype:<18} {elapsed:.1f}s  → 0 rows")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {stype:<18} {elapsed:.1f}s  → EXCEPTION: {type(e).__name__}: {e}")
