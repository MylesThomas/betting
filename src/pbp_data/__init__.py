"""
PBP Data Collection Module

Collect NBA play-by-play data in stages:
1. Get game IDs by date
2. Download PBP data for each game
3. Process into parquet files
4. Validate with box scores
"""

__all__ = ['config', 'utils']
