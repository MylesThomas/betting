"""
Test MVP Player Headshot Sources - Find the Highest Quality Images

Purpose:
- Test multiple headshot sources for MVP candidates
- Compare image quality, resolution, and file size
- Download actual images to inspect visually
- Identify the best source for sharp, high-quality headshots

Context:
Headshots appear blurry in the MVP viz table at 25px height.
Need to find the highest resolution source to minimize pixelation when scaled down.

Available Sources:
1. NBA CDN 1040x760: https://cdn.nba.com/headshots/nba/latest/1040x760/{PLAYER_ID}.png
2. NBA CDN 260x190: https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{PLAYER_ID}.png
3. ESPN Full: https://a.espncdn.com/i/headshots/nba/players/full/{PLAYER_ID}.png
4. ESPN Combiner: https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{PLAYER_ID}.png
5. ESPN 500x500: https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{PLAYER_ID}.png&w=500&h=500
6. ESPN 350x254: https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{PLAYER_ID}.png&w=350&h=254

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 testing/test_mvp_headshot_sources.py
"""

import requests
import sys
from pathlib import Path
from PIL import Image
from io import BytesIO
import ssl
import urllib3

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

repo_root = Path(__file__).parent.parent

# MVP candidates with their PLAYER_IDs
MVP_CANDIDATES = {
    'Shai Gilgeous-Alexander': 1628983,
    'Luka Doncic': 1629029,
    'Cade Cunningham': 1630595,
    'Jaylen Brown': 1627759,
    'Jalen Brunson': 1628973,
    'Anthony Edwards': 1630162,
}

# Different headshot URL patterns to test
HEADSHOT_SOURCES = {
    'nba_cdn_1040x760': 'https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png',
    'nba_cdn_260x190': 'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png',
    'espn_full': 'https://a.espncdn.com/i/headshots/nba/players/full/{player_id}.png',
    'espn_combiner': 'https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{player_id}.png',
    'espn_500x500': 'https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{player_id}.png&w=500&h=500',
    'espn_350x254': 'https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{player_id}.png&w=350&h=254',
}


def test_headshot_source(player_name, player_id, source_name, url_pattern):
    """
    Test a headshot source and return detailed info.
    
    Returns:
        dict with: success, url, status_code, file_size, width, height, format
    """
    url = url_pattern.format(player_id=player_id)
    
    try:
        response = requests.get(url, timeout=10, verify=False)
        
        if response.status_code != 200:
            return {
                'player_name': player_name,
                'player_id': player_id,
                'source_name': source_name,
                'success': False,
                'url': url,
                'status_code': response.status_code,
                'file_size': 0,
                'width': 0,
                'height': 0,
                'format': None,
                'resolution': 0
            }
        
        # Load image to get dimensions
        img = Image.open(BytesIO(response.content))
        width, height = img.size
        format_type = img.format
        file_size = len(response.content)
        resolution = width * height  # Total pixels
        
        return {
            'player_name': player_name,
            'player_id': player_id,
            'source_name': source_name,
            'success': True,
            'url': url,
            'status_code': response.status_code,
            'file_size': file_size,
            'width': width,
            'height': height,
            'format': format_type,
            'resolution': resolution
        }
        
    except Exception as e:
        return {
            'player_name': player_name,
            'player_id': player_id,
            'source_name': source_name,
            'success': False,
            'url': url,
            'status_code': 0,
            'file_size': 0,
            'width': 0,
            'height': 0,
            'format': None,
            'resolution': 0,
            'error': str(e)
        }


def download_sample_images(results, output_dir):
    """
    Download sample images from each successful source for visual comparison.
    """
    output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading sample images to: {output_dir}")
    
    # Get Shai's headshots from all sources
    shai_results = [r for r in results if r['player_name'] == 'Shai Gilgeous-Alexander' and r['success']]
    
    for result in shai_results:
        try:
            response = requests.get(result['url'], timeout=10, verify=False)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # Save original
                filename = f"shai_{result['source_name']}.png"
                img.save(output_dir / filename)
                
                # Save scaled down to 25px (what we use in viz)
                img_small = img.copy()
                img_small.thumbnail((25, 25), Image.Resampling.LANCZOS)
                filename_small = f"shai_{result['source_name']}_25px.png"
                img_small.save(output_dir / filename_small)
                
                print(f"   ✅ {result['source_name']}: {result['width']}x{result['height']} → saved original + 25px version")
        except Exception as e:
            print(f"   ❌ {result['source_name']}: {e}")


def main():
    """Run all headshot source tests"""
    
    print("="*80)
    print("MVP HEADSHOT SOURCE QUALITY TEST")
    print("="*80)
    print(f"\nTesting {len(MVP_CANDIDATES)} MVP candidates")
    print(f"Testing {len(HEADSHOT_SOURCES)} different sources\n")
    
    results = []
    
    # Test each player with each source
    for player_name, player_id in MVP_CANDIDATES.items():
        print(f"\n{'─'*80}")
        print(f"Testing: {player_name} (ID: {player_id})")
        print(f"{'─'*80}")
        
        for source_name, url_pattern in HEADSHOT_SOURCES.items():
            result = test_headshot_source(player_name, player_id, source_name, url_pattern)
            results.append(result)
            
            if result['success']:
                print(f"✅ {source_name:20s} | {result['width']:4d}x{result['height']:4d} | "
                      f"{result['file_size']/1024:6.1f}KB | {result['format']}")
            else:
                print(f"❌ {source_name:20s} | Status: {result['status_code']}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: Best Sources for High-Quality Headshots")
    print("="*80)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("\n❌ No successful sources found!")
        return
    
    # Group by source
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in successful_results:
        by_source[r['source_name']].append(r)
    
    print("\n📊 Success Rate by Source:")
    print(f"{'Source':<25} {'Success':<10} {'Avg Resolution':<20} {'Avg Size':<15}")
    print("─"*80)
    
    source_stats = []
    for source_name, source_results in by_source.items():
        success_count = len(source_results)
        total_count = len(MVP_CANDIDATES)
        avg_resolution = sum(r['resolution'] for r in source_results) / len(source_results)
        avg_width = sum(r['width'] for r in source_results) / len(source_results)
        avg_height = sum(r['height'] for r in source_results) / len(source_results)
        avg_size = sum(r['file_size'] for r in source_results) / len(source_results)
        
        source_stats.append({
            'source_name': source_name,
            'success_count': success_count,
            'total_count': total_count,
            'success_rate': success_count / total_count,
            'avg_resolution': avg_resolution,
            'avg_width': avg_width,
            'avg_height': avg_height,
            'avg_size': avg_size
        })
        
        print(f"{source_name:<25} {success_count}/{total_count:<8} "
              f"{avg_width:.0f}x{avg_height:.0f} ({avg_resolution/1000:.0f}K pixels)  "
              f"{avg_size/1024:.1f}KB")
    
    # Sort by resolution (highest quality)
    source_stats.sort(key=lambda x: x['avg_resolution'], reverse=True)
    
    print("\n🏆 RECOMMENDED SOURCE (Highest Resolution):")
    best = source_stats[0]
    print(f"   Source: {best['source_name']}")
    print(f"   Resolution: {best['avg_width']:.0f}x{best['avg_height']:.0f} ({best['avg_resolution']/1000:.0f}K pixels)")
    print(f"   File Size: {best['avg_size']/1024:.1f}KB")
    print(f"   Success Rate: {best['success_count']}/{best['total_count']} ({best['success_rate']*100:.0f}%)")
    print(f"   URL Pattern: {HEADSHOT_SOURCES[best['source_name']]}")
    
    # Download sample images
    download_sample_images(results, 'data/99_tmp/headshot_samples')
    
    print("\n" + "="*80)
    print("📸 VISUAL COMPARISON")
    print("="*80)
    print("\nSample images saved for Shai Gilgeous-Alexander:")
    print("   - Original size from each source")
    print("   - Scaled to 25px (actual viz size)")
    print("\nCompare the 25px versions to see which source looks sharpest!")
    print(f"\nLocation: {repo_root / 'data/99_tmp/headshot_samples'}")
    
    print("\n" + "="*80)
    print("💡 IMPLEMENTATION RECOMMENDATION")
    print("="*80)
    print(f"\nUpdate viz_nba_mvp_gt.py to use: {best['source_name']}")
    print(f"\nURL: {HEADSHOT_SOURCES[best['source_name']]}")
    print("\nKey changes:")
    print("1. Use highest resolution source")
    print("2. Download at full size (don't thumbnail before base64)")
    print("3. Let R/gtExtras handle the scaling for best quality")


if __name__ == "__main__":
    main()

