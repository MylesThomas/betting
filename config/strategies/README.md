# Strategy Registry

This directory is a **tracking record** of what is in production and why — not a config file the pipelines read from.

Each `.yaml` file documents one sport's live strategies: the model artifact, inference thresholds, IS/OOS performance, S3 paths, and deployment status. The pipelines have their own config files under `src/{pipeline}/config.yaml` which is what they actually read at runtime.

Update these files when:
- A strategy goes live or gets disabled (`status: live / disabled`)
- Inference thresholds change
- A model is retrained (new AUC, new artifact path)
- IS/OOS performance is updated after a season

## Current status

| Sport | Strategy | Status |
|-------|----------|--------|
| MLB   | total_bases (UNDER 1.5) | live |
| NFL   | sacks, rush_attempts, rec_yards | not_deployed — enable before 2026-09-09 |
| NBA   | points, assists | not_deployed — enable before 2026-10-28 |
