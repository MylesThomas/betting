# Data Pipeline Architecture

**Status:** 📝 Planned  
**Last updated:** 2026-02-13

This document will explain:

## Why 01-04 Stage Model

**01_input/** - Raw data from APIs
- Reasoning: Keep original responses for debugging
- Never modified after writing
- Timestamped for historical analysis

**02_cache/** - Lookup tables
- Reasoning: Expensive-to-fetch, rarely-changing data
- Rosters, player-team mappings, team metadata
- Refreshed weekly or on-demand

**03_intermediate/** - Processed data
- Reasoning: Aggregated/cleaned data ready for analysis
- Consensus lines across books
- Normalized player names
- Derived features

**04_output/** - Final results
- Reasoning: Betting opportunities, analysis outputs
- What humans/dashboards consume
- Never used as input to other processing

## Data Flow Patterns

- Sequential: 01 → 02 → 03 → 04
- Idempotent: Re-running should produce same output
- Timestamped: All files include date for versioning

## S3 Strategy

- Why mirror local structure
- Retention policies
- Cost optimization

---

**To be written in:** Phase 3 (Architectural Boundaries)
