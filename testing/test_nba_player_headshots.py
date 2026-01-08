"""
Test NBA player headshots - figure out how to get and join player images.

Purpose:
- Test NBA API player headshot URLs
- Validate that headshot images are accessible
- Create a mapping of PLAYER_ID to headshot URL
- Test joining headshots with defensive disruptors data

Context:
NBA player headshots are available from several sources:
1. NBA.com: https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{PLAYER_ID}.png
2. stats.nba.com: https://cdn.nba.com/headshots/nba/latest/1040x760/{PLAYER_ID}.png
3. ESPN: https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{PLAYER_ID}.png

We'll test which source works best and is most reliable.

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 testing/test_nba_player_headshots.py
"""

import pandas as pd
import requests
from pathlib import Path
import sys
import ssl
import urllib3

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Test players from our defensive disruptors analysis
TEST_PLAYERS = {
    'Zach Edey': 1641744,
    'Cam Christie': 1642353,
    'Rudy Gobert': 203497,
    'Draymond Green': 203110,
    'Trae Young': 1629027,
    'Kevin Love': 201567,
    'LeBron James': 2544,  # Test with a well-known player
}

# Different headshot URL patterns to test
HEADSHOT_URL_PATTERNS = {
    'nba_cms_260x190': 'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png',
    'nba_cdn_1040x760': 'https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png',
    'espn_full': 'https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{player_id}.png',
    'espn_500': 'https://a.espncdn.com/i/headshots/nba/players/full/{player_id}.png',
}


def test_headshot_url(player_name, player_id, url_pattern_name, url_pattern):
    """
    Test if a headshot URL is accessible.
    
    Args:
        player_name: Player name for display
        player_id: NBA player ID
        url_pattern_name: Name of the URL pattern being tested
        url_pattern: URL pattern with {player_id} placeholder
        
    Returns:
        tuple: (success: bool, url: str, status_code: int, content_length: int)
    """
    url = url_pattern.format(player_id=player_id)
    
    try:
        # Use GET instead of HEAD (some servers block HEAD requests)
        # Only fetch first 1KB to check if image exists
        response = requests.get(url, timeout=10, allow_redirects=True, 
                               verify=False, stream=True)
        status_code = response.status_code
        
        # Read first chunk to get content length
        content_length = 0
        if status_code == 200:
            chunk = next(response.iter_content(1024), None)
            if chunk:
                content_length = len(chunk)
                # Close the stream
                response.close()
        
        if status_code == 200 and content_length > 0:
            return (True, url, status_code, content_length)
        else:
            return (False, url, status_code, content_length)
            
    except Exception as e:
        return (False, url, 0, 0)


def test_all_patterns():
    """Test all URL patterns for all test players"""
    print("="*80)
    print("NBA PLAYER HEADSHOT URL TEST")
    print("="*80 + "\n")
    
    results = []
    
    for player_name, player_id in TEST_PLAYERS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {player_name} (ID: {player_id})")
        print(f"{'='*80}")
        
        for pattern_name, pattern in HEADSHOT_URL_PATTERNS.items():
            success, url, status_code, content_length = test_headshot_url(
                player_name, player_id, pattern_name, pattern
            )
            
            status_icon = "✅" if success else "❌"
            print(f"\n{status_icon} {pattern_name}:")
            print(f"   URL: {url}")
            print(f"   Status: {status_code}")
            print(f"   Size: {content_length:,} bytes")
            
            results.append({
                'player_name': player_name,
                'player_id': player_id,
                'pattern_name': pattern_name,
                'success': success,
                'url': url,
                'status_code': status_code,
                'content_length': content_length
            })
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze which URL pattern works best"""
    print("\n" + "="*80)
    print("ANALYSIS: Which URL Pattern Works Best?")
    print("="*80 + "\n")
    
    # Success rate by pattern
    print("📊 Success Rate by URL Pattern:")
    success_by_pattern = results_df.groupby('pattern_name').agg({
        'success': ['sum', 'count', 'mean']
    }).round(3)
    success_by_pattern.columns = ['Successful', 'Total', 'Success Rate']
    print(success_by_pattern.to_string())
    
    # Best pattern
    if results_df['success'].sum() > 0:
        best_pattern = results_df[results_df['success']].groupby('pattern_name').size().idxmax()
        best_count = results_df[results_df['success']].groupby('pattern_name').size().max()
        print(f"\n🏆 Best Pattern: {best_pattern} ({best_count}/{len(TEST_PLAYERS)} players)")
    else:
        print(f"\n❌ No successful patterns found!")
        # Default to nba_cdn_1040x760 as fallback
        best_pattern = 'nba_cdn_1040x760'
        print(f"   Using fallback: {best_pattern}")
    
    # Average file size for successful headshots
    print("\n📏 Average File Size for Successful Headshots:")
    avg_size_by_pattern = results_df[results_df['success']].groupby('pattern_name')['content_length'].mean()
    for pattern, avg_size in avg_size_by_pattern.items():
        print(f"   {pattern}: {avg_size:,.0f} bytes ({avg_size/1024:.1f} KB)")
    
    return best_pattern


def test_with_defensive_disruptors_data(best_pattern):
    """Test joining headshots with defensive disruptors data"""
    print("\n" + "="*80)
    print("TEST: Join Headshots with Defensive Disruptors Data")
    print("="*80 + "\n")
    
    # Read defensive disruptors CSV
    csv_file = repo_root / 'data/04_output/nba/defensive_disruptors_2025_26.csv'
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Run analyze_defensive_disruptors.py first!")
        return None
    
    print(f"📂 Reading: {csv_file.name}")
    df = pd.read_csv(csv_file)
    print(f"   ✅ Loaded {len(df)} players\n")
    
    # Add headshot URL column using best pattern
    url_pattern = HEADSHOT_URL_PATTERNS[best_pattern]
    df['headshot_url'] = df['PLAYER_ID'].apply(lambda x: url_pattern.format(player_id=x))
    
    # Test a few random headshots
    print("🎯 Testing Random Sample of Headshot URLs:")
    sample_players = df.sample(min(10, len(df)))
    
    success_count = 0
    for _, player in sample_players.iterrows():
        success, url, status_code, content_length = test_headshot_url(
            player['PLAYER_NAME'], 
            player['PLAYER_ID'],
            best_pattern,
            url_pattern
        )
        
        status_icon = "✅" if success else "❌"
        print(f"   {status_icon} {player['PLAYER_NAME']}: {status_code} ({content_length:,} bytes)")
        
        if success:
            success_count += 1
    
    success_rate = success_count / len(sample_players) * 100
    print(f"\n📊 Sample Success Rate: {success_count}/{len(sample_players)} ({success_rate:.1f}%)")
    
    # Show sample of data with headshots
    print("\n📋 Sample Data with Headshots:")
    display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'DEF_IMPACT', 'headshot_url']
    print(df[display_cols].head(5).to_string(index=False))
    
    return df


def generate_headshot_url_function(best_pattern):
    """Generate a Python function to use in the main visualization script"""
    print("\n" + "="*80)
    print("RECOMMENDED CODE FOR MAIN SCRIPT")
    print("="*80 + "\n")
    
    url_pattern = HEADSHOT_URL_PATTERNS[best_pattern]
    
    code = f'''
def add_player_headshots(df):
    """
    Add player headshot URLs to dataframe.
    
    Uses {best_pattern} pattern which had the highest success rate.
    
    Args:
        df: DataFrame with PLAYER_ID column
        
    Returns:
        DataFrame with headshot_url column added
    """
    df['headshot_url'] = df['PLAYER_ID'].apply(
        lambda x: '{url_pattern}'.format(player_id=x)
    )
    return df
'''
    
    print("💡 Add this function to viz_nba_defensive_disruptors_gt.py:")
    print(code)
    
    print("\n💡 Usage in prepare_data_for_visualization():")
    print("   df_display = add_player_headshots(df_display)")


def main():
    """Run all headshot tests"""
    
    # Test all URL patterns
    results_df = test_all_patterns()
    
    # Analyze results
    best_pattern = analyze_results(results_df)
    
    # Test with actual data
    df_with_headshots = test_with_defensive_disruptors_data(best_pattern)
    
    # Generate code for main script
    generate_headshot_url_function(best_pattern)
    
    print("\n" + "="*80)
    print("✅ HEADSHOT TEST COMPLETE")
    print("="*80)
    print(f"\n🎯 Recommended URL Pattern: {best_pattern}")
    print(f"   URL: {HEADSHOT_URL_PATTERNS[best_pattern]}\n")


if __name__ == "__main__":
    main()

