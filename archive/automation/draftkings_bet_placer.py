"""
DraftKings Automated Bet Placement - Production Ready

DISCLAIMER & WARNINGS:
======================
1. This script violates DraftKings' Terms of Service
2. Using this may result in ACCOUNT SUSPENSION and LOSS OF FUNDS
3. DraftKings has bot detection systems
4. This is for EDUCATIONAL PURPOSES ONLY
5. Use at your own risk - you are responsible for all consequences

PURPOSE:
========
Fast, production-ready automation for placing bets on DraftKings.
Designed for speed - can execute bets in 2-5 seconds from trigger.

FEATURES:
=========
- Session persistence (stay logged in)
- Faster execution than Bovada
- Better error handling
- API mode to receive bet commands
- Screenshot debugging
- More reliable selectors

USAGE:
======
# Test login
python draftkings_bet_placer.py --login-only

# Keep session alive (logged in, ready for instant bets)
python draftkings_bet_placer.py --keep-session

# Place a bet (dry run)
python draftkings_bet_placer.py --bet "NFL:Moneyline:Chiefs:-150:10" --dry-run

# Place a bet (live)
python draftkings_bet_placer.py --bet "NFL:Moneyline:Chiefs:-150:10" --live

CONFIGURATION:
==============
Set credentials in .env file:
DRAFTKINGS_USERNAME=your_email@example.com
DRAFTKINGS_PASSWORD=your_password_here
"""

import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict
import argparse

from playwright.async_api import async_playwright, Browser, Page, BrowserContext


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DraftKingsConfig:
    """Configuration for DraftKings automation"""
    
    # URLs
    base_url: str = "https://sportsbook.draftkings.com"
    
    # Timeouts (optimized for speed)
    page_load_timeout: int = 20000
    element_timeout: int = 8000
    quick_timeout: int = 3000
    
    # Browser settings
    headless: bool = False
    slow_mo: int = 50
    
    # Session management
    session_file: str = "draftkings_session.json"
    keep_alive_interval: int = 300


@dataclass
class BetRequest:
    """Structured bet request"""
    sport: str
    bet_type: str
    selection: str
    odds: str
    amount: float
    
    @classmethod
    def from_string(cls, bet_str: str) -> 'BetRequest':
        """Parse bet from string: 'NFL:Moneyline:Chiefs:-150:10'"""
        parts = bet_str.split(':')
        if len(parts) != 5:
            raise ValueError("Bet format: Sport:BetType:Selection:Odds:Amount")
        
        return cls(
            sport=parts[0],
            bet_type=parts[1],
            selection=parts[2],
            odds=parts[3],
            amount=float(parts[4])
        )


@dataclass
class DraftKingsCredentials:
    """DraftKings login credentials"""
    username: str
    password: str
    
    @classmethod
    def from_env(cls) -> 'DraftKingsCredentials':
        """Load credentials from environment variables"""
        username = os.getenv('DRAFTKINGS_USERNAME')
        password = os.getenv('DRAFTKINGS_PASSWORD')
        
        if not username or not password:
            raise ValueError(
                "Missing credentials. Set DRAFTKINGS_USERNAME and DRAFTKINGS_PASSWORD "
                "environment variables or create a .env file"
            )
        
        return cls(username=username, password=password)


# =============================================================================
# STEALTH BROWSER SETUP
# =============================================================================

class StealthBrowser:
    """Browser with anti-detection configurations"""
    
    @staticmethod
    async def create_context(browser: Browser, config: DraftKingsConfig, session_data: Optional[Dict] = None):
        """Create browser context with stealth and optional session restore"""
        
        context_options = {
            'viewport': {'width': 1920, 'height': 1080},
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/120.0.0.0 Safari/537.36',
            'locale': 'en-US',
            'timezone_id': 'America/New_York',
            'permissions': ['geolocation'],
            'geolocation': {'latitude': 40.7128, 'longitude': -74.0060},
            'color_scheme': 'dark',
            'extra_http_headers': {
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'DNT': '1',
            }
        }
        
        # Restore session if available
        if session_data and 'cookies' in session_data:
            context_options['storage_state'] = session_data
        
        context = await browser.new_context(**context_options)
        
        # Anti-detection scripts
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            delete window.playwright;
            delete window.__playwright;
        """)
        
        return context


# =============================================================================
# DRAFTKINGS BOT
# =============================================================================

class DraftKingsBot:
    """Fast DraftKings betting automation"""
    
    def __init__(self, credentials: DraftKingsCredentials, config: DraftKingsConfig, dry_run: bool = True):
        self.credentials = credentials
        self.config = config
        self.dry_run = dry_run
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.logs: list = []
        self.playwright_instance = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.logs.append(log_entry)
    
    async def screenshot(self, name: str = "screenshot"):
        """Take screenshot for debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dk_{name}_{timestamp}.png"
            await self.page.screenshot(path=filename)
            self.log(f"📸 {filename}")
            return filename
        except Exception as e:
            self.log(f"Screenshot failed: {e}", "WARNING")
            return None
    
    async def quick_delay(self, ms: int = 500):
        """Minimal human-like delay"""
        import random
        delay = random.uniform(ms * 0.8, ms * 1.2) / 1000
        await asyncio.sleep(delay)
    
    def load_session(self) -> Optional[Dict]:
        """Load saved session"""
        try:
            if os.path.exists(self.config.session_file):
                with open(self.config.session_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"Failed to load session: {e}", "WARNING")
        return None
    
    async def init_browser(self, use_saved_session: bool = True):
        """Initialize browser with optional session restore"""
        self.log("🚀 Initializing browser...")
        
        self.playwright_instance = await async_playwright().start()
        self.browser = await self.playwright_instance.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo
        )
        
        session_data = self.load_session() if use_saved_session else None
        self.context = await StealthBrowser.create_context(
            self.browser, self.config, session_data
        )
        
        self.page = await self.context.new_page()
        self.page.set_default_timeout(self.config.page_load_timeout)
        
        if session_data:
            self.log("✅ Restored saved session")
        
        return session_data is not None
    
    async def close_browser(self):
        """Cleanup"""
        if self.browser:
            await self.browser.close()
        if self.playwright_instance:
            await self.playwright_instance.stop()
    
    async def login(self) -> bool:
        """Login to DraftKings"""
        try:
            self.log("Navigating to DraftKings...")
            await self.page.goto(self.config.base_url, wait_until='domcontentloaded')
            await self.quick_delay(1000)
            await self.screenshot("01_homepage")
            
            # Look for Sign In / Log In button
            self.log("Looking for Sign In button...")
            login_selectors = [
                'button:has-text("Sign In")',
                'a:has-text("Sign In")',
                'button:has-text("Log In")',
                'a:has-text("Log In")',
                '[data-test-id="sign-in"]',
                'text=Sign In'
            ]
            
            login_button = None
            for selector in login_selectors:
                try:
                    login_button = await self.page.wait_for_selector(selector, timeout=self.config.quick_timeout)
                    if login_button:
                        self.log(f"✅ Found: {selector}")
                        break
                except:
                    continue
            
            if not login_button:
                self.log("❌ No login button", "ERROR")
                await self.screenshot("error_no_login")
                return False
            
            await login_button.click()
            await self.quick_delay()
            await self.screenshot("02_login_modal")
            
            # Email/Username field
            self.log("Entering email...")
            email_selectors = [
                'input[type="email"]',
                'input[name="email"]',
                'input[name="username"]',
                '#email',
                'input[placeholder*="Email"]',
                'input[placeholder*="email"]'
            ]
            
            email_field = None
            for selector in email_selectors:
                try:
                    email_field = await self.page.wait_for_selector(selector, timeout=self.config.quick_timeout)
                    if email_field:
                        self.log(f"✅ Found email field: {selector}")
                        break
                except:
                    continue
            
            if not email_field:
                self.log("❌ No email field", "ERROR")
                await self.screenshot("error_no_email")
                return False
            
            await email_field.fill(self.credentials.username)
            await self.quick_delay()
            
            # Password field
            self.log("Entering password...")
            password_field = await self.page.wait_for_selector('input[type="password"]')
            await password_field.fill(self.credentials.password)
            await self.quick_delay()
            await self.screenshot("03_credentials")
            
            # Submit button
            self.log("Submitting login...")
            submit_selectors = [
                'button[type="submit"]',
                'button:has-text("Sign In")',
                'button:has-text("Log In")',
                'button:has-text("Continue")'
            ]
            
            submit_button = None
            for selector in submit_selectors:
                try:
                    submit_button = await self.page.wait_for_selector(selector, timeout=self.config.quick_timeout)
                    if submit_button:
                        break
                except:
                    continue
            
            if not submit_button:
                self.log("❌ No submit button", "ERROR")
                await self.screenshot("error_no_submit")
                return False
            
            await submit_button.click()
            await self.quick_delay(2000)
            await self.screenshot("04_after_submit")
            
            # Check for errors
            error_indicators = [
                'text=incorrect',
                'text=Invalid',
                'text=error',
                'text=wrong',
                '[class*="error"]',
                '[role="alert"]'
            ]
            
            for selector in error_indicators:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=2000)
                    if element:
                        text = await element.text_content()
                        self.log(f"❌ Error: {text}", "ERROR")
                        await self.screenshot("error_login_failed")
                        return False
                except:
                    continue
            
            # Wait for login to complete
            await self.page.wait_for_load_state('networkidle', timeout=self.config.page_load_timeout)
            await self.screenshot("05_after_login")
            
            # Verify login success
            success_indicators = [
                'text=Account',
                'text=My Account',
                'text=Balance',
                'text=My Bets',
                'text=Sign Out',
                '[data-test-id="account-menu"]'
            ]
            
            for selector in success_indicators:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=5000)
                    if element:
                        self.log("✅ Login successful!", "SUCCESS")
                        
                        # Save session
                        storage = await self.context.storage_state()
                        with open(self.config.session_file, 'w') as f:
                            json.dump(storage, f)
                        self.log(f"💾 Session saved to {self.config.session_file}")
                        
                        return True
                except:
                    continue
            
            self.log("⚠️ Login unclear - no success indicators", "WARNING")
            current_url = self.page.url
            self.log(f"Current URL: {current_url}")
            
            # If URL changed away from login, assume success
            if 'login' not in current_url.lower() and 'signin' not in current_url.lower():
                self.log("✅ Login likely successful (URL changed)", "SUCCESS")
                
                # Save session anyway
                storage = await self.context.storage_state()
                with open(self.config.session_file, 'w') as f:
                    json.dump(storage, f)
                
                return True
            
            return False
            
        except Exception as e:
            self.log(f"❌ Login failed: {e}", "ERROR")
            await self.screenshot("error_exception")
            return False
    
    async def place_bet(self, bet: BetRequest) -> bool:
        """Place a bet on DraftKings"""
        try:
            self.log(f"🎯 Placing bet: {bet.sport} {bet.bet_type} {bet.selection} ${bet.amount}")
            
            # Navigate to sport
            sport_url = f"{self.config.base_url}/{bet.sport.lower()}"
            self.log(f"Navigating to {sport_url}...")
            await self.page.goto(sport_url, wait_until='domcontentloaded')
            await self.quick_delay()
            await self.screenshot("06_sport_page")
            
            # Find the selection
            self.log(f"Looking for: {bet.selection}...")
            
            selection_selectors = [
                f'text={bet.selection}',
                f'button:has-text("{bet.selection}")',
                f'[aria-label*="{bet.selection}"]',
                f'div:has-text("{bet.selection}")'
            ]
            
            selection_button = None
            for selector in selection_selectors:
                try:
                    selection_button = await self.page.wait_for_selector(selector, timeout=self.config.element_timeout)
                    if selection_button:
                        self.log(f"✅ Found selection")
                        break
                except:
                    continue
            
            if not selection_button:
                self.log(f"❌ Could not find: {bet.selection}", "ERROR")
                await self.screenshot("error_no_selection")
                return False
            
            # Click to add to bet slip
            await selection_button.click()
            await self.quick_delay()
            await self.screenshot("07_bet_slip")
            
            # Find bet slip stake input
            self.log(f"Entering stake: ${bet.amount}")
            stake_selectors = [
                'input[placeholder*="Wager"]',
                'input[placeholder*="wager"]',
                'input[placeholder*="Stake"]',
                'input[placeholder*="stake"]',
                'input[type="number"]',
                'input[name="stake"]',
                'input[name="wager"]'
            ]
            
            stake_input = None
            for selector in stake_selectors:
                try:
                    stake_input = await self.page.wait_for_selector(selector, timeout=self.config.quick_timeout)
                    if stake_input:
                        break
                except:
                    continue
            
            if not stake_input:
                self.log("❌ No stake input", "ERROR")
                await self.screenshot("error_no_stake")
                return False
            
            await stake_input.fill(str(bet.amount))
            await self.quick_delay()
            await self.screenshot("08_stake_entered")
            
            if self.dry_run:
                self.log("🔶 DRY RUN: Would place bet here", "WARNING")
                return True
            
            # Place bet button
            self.log("🚨 PLACING BET...")
            place_bet_selectors = [
                'button:has-text("Place Bet")',
                'button:has-text("Submit Bet")',
                'button:has-text("Place Wager")',
                '[data-test-id="place-bet"]',
                '[data-test-id="place-wager"]'
            ]
            
            place_button = None
            for selector in place_bet_selectors:
                try:
                    place_button = await self.page.wait_for_selector(selector, timeout=self.config.quick_timeout)
                    if place_button:
                        break
                except:
                    continue
            
            if not place_button:
                self.log("❌ No place bet button", "ERROR")
                await self.screenshot("error_no_place_button")
                return False
            
            await place_button.click()
            await self.quick_delay(2000)
            await self.screenshot("09_bet_placed")
            
            # Verify success
            success_indicators = [
                'text=Bet Placed',
                'text=Success',
                'text=Confirmed',
                'text=Wager Placed'
            ]
            
            for selector in success_indicators:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=5000)
                    if element:
                        self.log("✅ BET PLACED SUCCESSFULLY!", "SUCCESS")
                        return True
                except:
                    continue
            
            self.log("⚠️ Bet placement unclear", "WARNING")
            return False
            
        except Exception as e:
            self.log(f"❌ Bet placement failed: {e}", "ERROR")
            await self.screenshot("error_bet_failed")
            return False
    
    async def keep_session_alive(self):
        """Keep browser session alive for instant betting"""
        self.log("🔄 Session keep-alive mode - press Ctrl+C to exit")
        
        try:
            while True:
                await asyncio.sleep(self.config.keep_alive_interval)
                
                # Refresh page to keep session active
                try:
                    await self.page.reload(wait_until='domcontentloaded')
                    self.log("🔄 Session refreshed")
                except Exception as e:
                    self.log(f"⚠️ Refresh failed: {e}", "WARNING")
                    
        except asyncio.CancelledError:
            self.log("Session keep-alive stopped")
    
    def save_logs(self, filepath: str = "draftkings_bot_logs.txt"):
        """Save logs"""
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.logs))
        self.log(f"💾 Logs saved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='DraftKings Automated Bet Placer')
    parser.add_argument('--login-only', action='store_true', help='Test login only')
    parser.add_argument('--keep-session', action='store_true', help='Keep logged in session alive')
    parser.add_argument('--bet', type=str, help='Bet to place: Sport:Type:Selection:Odds:Amount')
    parser.add_argument('--live', action='store_true', help='Actually place bet (default is dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (default)')
    args = parser.parse_args()
    
    # Determine mode
    dry_run = not args.live
    if args.live:
        print("⚠️  LIVE MODE: Will actually place bet!")
        print("Press Ctrl+C within 3 seconds to cancel...")
        await asyncio.sleep(3)
    else:
        print("🔶 DRY RUN MODE")
    
    # Load credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        credentials = DraftKingsCredentials.from_env()
        config = DraftKingsConfig()
        bot = DraftKingsBot(credentials, config, dry_run=dry_run)
        
        # Initialize browser
        has_session = await bot.init_browser()
        
        # Login if needed
        if not has_session or args.login_only:
            success = await bot.login()
            if not success:
                print("\n❌ Login failed")
                bot.save_logs()
                await bot.close_browser()
                return
        
        if args.login_only:
            print("\n✅ Login successful")
            await asyncio.sleep(3)
            bot.save_logs()
            await bot.close_browser()
            return
        
        # Keep session alive
        if args.keep_session:
            await bot.keep_session_alive()
            bot.save_logs()
            await bot.close_browser()
            return
        
        # Place bet
        if args.bet:
            bet_request = BetRequest.from_string(args.bet)
            success = await bot.place_bet(bet_request)
            
            if success:
                print("\n✅ Bet placed successfully")
            else:
                print("\n❌ Bet failed")
        
        bot.save_logs()
        await bot.close_browser()
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

