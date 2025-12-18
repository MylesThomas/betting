"""
Quick test to verify Playwright is installed correctly.
Run this before trying the Bovada bot.

Usage:
    python test_playwright_setup.py
"""

import asyncio
from playwright.async_api import async_playwright


async def test_playwright():
    """Test that Playwright is working"""
    print("🧪 Testing Playwright setup...")
    
    try:
        async with async_playwright() as p:
            print("✅ Playwright imported successfully")
            
            # Launch browser
            print("🌐 Launching browser...")
            browser = await p.chromium.launch(headless=False)
            print("✅ Browser launched")
            
            # Create page
            page = await browser.new_page()
            print("✅ Page created")
            
            # Navigate to a test site
            print("🔗 Navigating to example.com...")
            await page.goto("https://example.com")
            print("✅ Navigation successful")
            
            # Get title
            title = await page.title()
            print(f"✅ Page title: {title}")
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Close
            await browser.close()
            print("✅ Browser closed")
            
            print("\n🎉 All tests passed! Playwright is ready to use.")
            return True
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you ran: pip install playwright")
        print("2. Make sure you ran: playwright install chromium")
        return False


if __name__ == "__main__":
    asyncio.run(test_playwright())

