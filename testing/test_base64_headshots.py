"""
Test converting NBA player headshots to base64 data URIs.

Purpose:
- Download NBA player headshot images
- Convert to base64 data URIs
- Test if this approach works better with R/gt rendering

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 testing/test_base64_headshots.py
"""

import pandas as pd
import requests
import base64
from io import BytesIO
from PIL import Image
import ssl
import urllib3

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_and_convert_to_base64(url, max_size=(100, 100)):
    """
    Download image and convert to base64 data URI.
    
    Args:
        url: Image URL
        max_size: Tuple of (width, height) to resize to
        
    Returns:
        base64 data URI string or None if failed
    """
    try:
        # Download image
        response = requests.get(url, verify=False, timeout=10)
        if response.status_code != 200:
            return None
        
        # Open image
        img = Image.open(BytesIO(response.content))
        
        # Resize to reduce file size
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to PNG bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create data URI
        data_uri = f"data:image/png;base64,{img_base64}"
        
        return data_uri
        
    except Exception as e:
        print(f"Error converting {url}: {e}")
        return None


def test_conversion():
    """Test converting a few player headshots"""
    print("="*80)
    print("TEST: Converting NBA Headshots to Base64 Data URIs")
    print("="*80 + "\n")
    
    test_players = [
        ("LeBron James", 2544),
        ("Stephen Curry", 201939),
        ("Giannis Antetokounmpo", 203507),
    ]
    
    results = []
    
    for player_name, player_id in test_players:
        url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        print(f"Converting {player_name}...")
        
        data_uri = download_and_convert_to_base64(url, max_size=(60, 60))
        
        if data_uri:
            print(f"  ✅ Success! Data URI length: {len(data_uri):,} characters")
            results.append({
                'player_name': player_name,
                'player_id': player_id,
                'original_url': url,
                'data_uri': data_uri[:100] + '...',  # Truncate for display
                'data_uri_full': data_uri
            })
        else:
            print(f"  ❌ Failed")
    
    print(f"\n✅ Converted {len(results)}/{len(test_players)} images")
    
    return pd.DataFrame(results)


def main():
    df = test_conversion()
    
    if len(df) > 0:
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        print("\n💡 Base64 data URIs work and can be embedded directly in HTML")
        print("   This avoids HTTPS loading issues in webshot2")
        print("\n📝 To integrate:")
        print("   1. Add base64 conversion function to viz script")
        print("   2. Convert headshots before passing to R")
        print("   3. Use data URIs instead of external URLs")
        print("\n⚠️  Trade-off:")
        print("   - Pros: Images always load, no HTTPS issues")
        print("   - Cons: Larger file size, slower initial conversion")
        print()


if __name__ == "__main__":
    main()

