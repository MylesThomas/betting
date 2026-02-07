# Strategy Upgrade: v4 → v5 (2026-02-07)

## Overview

Upgraded from Top 3 Unders (v4) to Enhanced Unders (v5) with **15 total strategies**.

## Changes

### What Changed
- **v4**: 3 strategies (1x2D + 2x3D)
- **v5**: 15 strategies (13x2D + 2x3D)
- **Added**: 13 new 2D strategies across Elite/Superstar/High Star/Star/Supplemental tiers
- **Kept**: 2 existing 3D strategies
- **Merged**: `star_narrow_under` with enhanced stats (881 games vs 832)

### Files Updated
1. ✅ Created `enhanced_unders_v5.json` config
2. ✅ Uploaded to S3: `s3://nba-betting-mt/strategies/enhanced_unders_v5.json`
3. ✅ Updated `lambda_function_nba_player_scoring_props.py` to use v5
4. ✅ Committed changes to git

## Strategy Breakdown

### Tier 1: Elite (1 strategy)
| # | Strategy | Line Tier | Spread | W-L-T | Hit Rate | ROI | Edge | Sample |
|---|----------|-----------|--------|-------|----------|-----|------|--------|
| 1 | elite_pickem_under | 35-40 | Pick'em (-2 to +2) | 9-1-0 | 90.0% | **+79.0%** | +40.0% | 10 |

### Tier 2: Superstar (2 strategies)
| # | Strategy | Line Tier | Spread | W-L-T | Hit Rate | ROI | Edge | Sample |
|---|----------|-----------|--------|-------|----------|-----|------|--------|
| 2 | superstar_2-6dog_under | 30-35 | 2-6 Dog | 54-30-0 | 64.3% | **+25.0%** | +14.3% | 84 |
| 3 | superstar_10-15fav_under | 30-35 | 10-15 Fav | 63-37-0 | 63.0% | **+22.3%** | +13.0% | 100 |

### Tier 3: High Star (3 strategies)
| # | Strategy | Line Tier | Spread | W-L-T | Hit Rate | ROI | Edge | Sample |
|---|----------|-----------|--------|-------|----------|-----|------|--------|
| 4 | highstar_10-15dog_under | 25-30 | 10-15 Dog | 44-28-0 | 61.1% | **+18.3%** | +11.1% | 72 |
| 5 | highstar_2-6dog_under | 25-30 | 2-6 Dog | 244-171-0 | 58.8% | **+13.5%** | +8.8% | **415** ⭐ |
| 6 | highstar_6-10dog_under | 25-30 | 6-10 Dog | 122-91-0 | 57.3% | **+10.3%** | +7.3% | 213 |

### Tier 4: Star (4 strategies)
| # | Strategy | Line Tier | Spread | W-L-T | Hit Rate | ROI | Edge | Sample |
|---|----------|-----------|--------|-------|----------|-----|------|--------|
| 7 | star_10-15dog_under | 20-25 | 10-15 Dog | 193-145-2 | 57.1% | **+9.9%** | +7.1% | 340 |
| 8 | superstar_2-6fav_under | 30-35 | 2-6 Fav | 101-77-0 | 56.7% | **+9.2%** | +6.7% | 178 |
| 9 | star_6-10dog_under | 20-25 | 6-10 Dog | 386-298-2 | 56.4% | **+8.5%** | +6.4% | 686 |
| 10 | star_narrow_under | 20-25 | 2-6 Fav | 495-386-0 | 56.2% | **+8.0%** | +6.2% | **881** 🏆 |

### Tier 5: Supplemental (3 strategies)
| # | Strategy | Line Tier | Spread | W-L-T | Hit Rate | ROI | Edge | Sample |
|---|----------|-----------|--------|-------|----------|-----|------|--------|
| 11 | highstar_10-15fav_under | 25-30 | 10-15 Fav | 201-163-1 | 55.2% | **+5.9%** | +5.2% | 365 |
| 12 | superstar_6-10fav_under | 30-35 | 6-10 Fav | 75-62-1 | 54.7% | **+4.9%** | +4.7% | 138 |
| 13 | highstar_pickem_under | 25-30 | Pick'em (-2 to +2) | 185-154-0 | 54.6% | **+4.6%** | +4.6% | 339 |

### 3D Legacy Strategies (2 strategies from v4)
| # | Strategy | Line Tier | Spread | Scorer Type | Profitable Seasons |
|---|----------|-----------|--------|-------------|-------------------|
| 14 | role_pickem_rim_under | 10-15 | Pick'em | Rim Attacker | **3/3** ✅ |
| 15 | bench_pickem_rim_under | 5-10 | Pick'em | Rim Attacker | 2/3 |

## Performance Highlights

- **Highest ROI**: elite_pickem_under (79.0%)
- **Highest Hit Rate**: elite_pickem_under (90.0%)
- **Largest Sample**: star_narrow_under (881 games)
- **Most Reliable**: star_narrow_under (3/3 profitable seasons)
- **Best Volume + Reliability**: highstar_2-6dog_under (415 games, 58.8% hit, 13.5% ROI)

## Deployment Status

### ✅ Completed
1. Created v5 config with 15 strategies
2. Uploaded to S3: `enhanced_unders_v5.json`
3. Updated lambda function code
4. Updated validation logic (expect 15 strategies, not 3)
5. Committed changes to git

### ⏭️ Next Steps
1. **Deploy Lambda**: Push updated lambda code to AWS
2. **Monitor**: Track performance for 1-2 weeks
3. **Validate**: Compare live results to backtest expectations
4. **Adjust**: Fine-tune Kelly bankroll sizing if needed

## Lambda Deployment

To deploy the updated lambda function:

```bash
# Option 1: Via AWS Console
# - Navigate to Lambda → nba-player-scoring-props-daily-workflow
# - Upload new code or update via deployment package

# Option 2: Via AWS CLI (if you have deployment setup)
# aws lambda update-function-code \
#   --function-name nba-player-scoring-props-daily-workflow \
#   --zip-file fileb://deployment.zip
```

## Expected Impact

### Volume
- **Before (v4)**: ~3-5 plays per day
- **After (v5)**: ~10-20 plays per day (estimated 3-4x increase)

### Diversification
- More spread across player tiers (Elite → Star)
- More spread scenarios covered (Dog, Fav, Pick'em)
- Reduced concentration risk

### Risk
- Lower per-strategy ROI on some plays (minimum 4.6% vs previous 8.0%+)
- Offset by higher volume and diversification
- Kelly sizing will automatically adjust for lower-edge plays

## Monitoring Plan

Track daily for next 2 weeks:
1. Hit rate vs. expected (overall and per tier)
2. ROI vs. backtest
3. Volume of plays per day
4. Kelly bankroll trajectory
5. Any anomalies or unexpected patterns

---

**Date**: 2026-02-07  
**Author**: Myles Thomas  
**Config Version**: v5  
**Lambda Function**: `nba-player-scoring-props-daily-workflow`
