#!/usr/bin/env python3
"""
Update Kelly Criterion settings in S3 config.

Usage:
    # Set to Quarter Kelly (25%)
    python scripts/update_kelly_settings.py --fractional-kelly 0.25
    
    # Set to Half Kelly (50%)
    python scripts/update_kelly_settings.py --fractional-kelly 0.5
    
    # Set to Full Kelly (100%)
    python scripts/update_kelly_settings.py --fractional-kelly 1.0
    
    # Change max Kelly cap
    python scripts/update_kelly_settings.py --max-kelly 0.15
    
    # Change both at once
    python scripts/update_kelly_settings.py --fractional-kelly 0.25 --max-kelly 0.10
    
    # View current settings
    python scripts/update_kelly_settings.py --show

Context:
Created to easily adjust Kelly betting parameters without manually editing JSON.
Fractional Kelly controls risk (0.25 = quarter Kelly = conservative, 1.0 = full Kelly = aggressive).
Max Kelly caps any single bet (default 0.10 = 10% max).
"""

import boto3
import json
import argparse
import sys

S3_BUCKET = 'nba-betting-mt'
S3_KEY = 'config/kelly_bankroll_tracker.json'


def load_config():
    """Load current Kelly config from S3"""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    return json.loads(response['Body'].read().decode('utf-8'))


def save_config(config):
    """Save Kelly config to S3"""
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_KEY,
        Body=json.dumps(config, indent=2),
        ContentType='application/json'
    )


def kelly_label(fractional_kelly):
    """Return human-readable label for fractional Kelly value"""
    if fractional_kelly == 1.0:
        return "Full Kelly (aggressive)"
    elif fractional_kelly == 0.5:
        return "Half Kelly (moderate)"
    elif fractional_kelly == 0.25:
        return "Quarter Kelly (conservative)"
    else:
        return f"{fractional_kelly:.2f}x Kelly"


def show_config(config):
    """Display current Kelly settings"""
    print("="*70)
    print("CURRENT KELLY SETTINGS")
    print("="*70)
    print(f"Bankroll:        ${config['current_bankroll']:,.2f}")
    print(f"Fractional Kelly: {config['fractional_kelly']*100:.0f}% - {kelly_label(config['fractional_kelly'])}")
    print(f"Max Kelly Cap:    {config['max_kelly']*100:.0f}%")
    print("="*70)
    print()
    print("Examples:")
    print("  5% edge @ -110 odds:")
    print(f"    Full Kelly (1.0):     ~10.0% of bankroll = ${config['current_bankroll']*0.10:,.0f}")
    print(f"    Half Kelly (0.5):     ~5.0% of bankroll  = ${config['current_bankroll']*0.05:,.0f}")
    print(f"    Quarter Kelly (0.25): ~2.5% of bankroll  = ${config['current_bankroll']*0.025:,.0f}")
    print(f"    Current ({config['fractional_kelly']:.2f}):      ~{10*config['fractional_kelly']:.1f}% of bankroll  = ${config['current_bankroll']*0.10*config['fractional_kelly']:,.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Update Kelly Criterion settings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Conservative (Quarter Kelly)
  python scripts/update_kelly_settings.py --fractional-kelly 0.25
  
  # Moderate (Half Kelly)
  python scripts/update_kelly_settings.py --fractional-kelly 0.5
  
  # Aggressive (Full Kelly)
  python scripts/update_kelly_settings.py --fractional-kelly 1.0
  
  # View current settings
  python scripts/update_kelly_settings.py --show
        """
    )
    parser.add_argument('--fractional-kelly', type=float,
                       help='Fractional Kelly (0.25=quarter, 0.5=half, 1.0=full)')
    parser.add_argument('--max-kelly', type=float,
                       help='Max Kelly cap as decimal (e.g., 0.10 = 10%%)')
    parser.add_argument('--show', action='store_true',
                       help='Show current settings')
    
    args = parser.parse_args()
    
    # Load current config
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Error loading config from S3: {e}")
        sys.exit(1)
    
    # If --show, just display and exit
    if args.show:
        show_config(config)
        return
    
    # Check if any updates requested
    if args.fractional_kelly is None and args.max_kelly is None:
        print("❌ No updates specified. Use --fractional-kelly or --max-kelly (or --show to view)")
        print()
        show_config(config)
        sys.exit(1)
    
    # Show current settings
    print("Current settings:")
    print(f"  Fractional Kelly: {config['fractional_kelly']*100:.0f}% - {kelly_label(config['fractional_kelly'])}")
    print(f"  Max Kelly Cap:    {config['max_kelly']*100:.0f}%")
    print()
    
    # Apply updates
    changed = False
    if args.fractional_kelly is not None:
        if args.fractional_kelly < 0 or args.fractional_kelly > 1:
            print("❌ Error: fractional_kelly must be between 0 and 1")
            sys.exit(1)
        config['fractional_kelly'] = args.fractional_kelly
        changed = True
    
    if args.max_kelly is not None:
        if args.max_kelly < 0 or args.max_kelly > 1:
            print("❌ Error: max_kelly must be between 0 and 1")
            sys.exit(1)
        config['max_kelly'] = args.max_kelly
        changed = True
    
    if changed:
        # Save to S3
        try:
            save_config(config)
            print("✅ Updated Kelly settings:")
            print(f"  Fractional Kelly: {config['fractional_kelly']*100:.0f}% - {kelly_label(config['fractional_kelly'])}")
            print(f"  Max Kelly Cap:    {config['max_kelly']*100:.0f}%")
            print()
            print(f"Saved to: s3://{S3_BUCKET}/{S3_KEY}")
        except Exception as e:
            print(f"❌ Error saving config to S3: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
