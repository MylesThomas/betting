"""
Automated Levels.fyi login via Playwright — captures the Cognito JWT so
scraper scripts can call the API without manual DevTools token extraction.

Token expires every 1 hour; Cognito refresh token lasts 30 days.
This module handles both: initial login and silent token refresh.

SETUP:
  uv add playwright
  uv run playwright install chromium

USAGE:
  export LEVELS_EMAIL="you@email.com"
  export LEVELS_PASSWORD="yourpassword"  # pragma: allowlist secret
  uv run python analysis/sentiment_snap/levels_auth.py
  # Prints the Bearer token and saves to data/.levels_token
"""

import os
import json
import time
from pathlib import Path

TOKEN_FILE = Path(__file__).parent / "data" / ".levels_token"
LOGIN_URL  = "https://www.levels.fyi/login"


def get_cached_token() -> str | None:
    """Return cached token if it exists and isn't expired."""
    if not TOKEN_FILE.exists():
        return None
    data = json.loads(TOKEN_FILE.read_text())
    expires_at = data.get("expires_at", 0)
    if time.time() < expires_at - 60:  # 60s buffer
        return data["token"]
    return None


def fetch_token_via_playwright() -> str:
    """Log in with Playwright, intercept the Bearer JWT, cache it."""
    from playwright.sync_api import sync_playwright

    email    = os.environ.get("LEVELS_EMAIL")
    password = os.environ.get("LEVELS_PASSWORD")
    if not email or not password:
        raise EnvironmentError(
            "Set LEVELS_EMAIL and LEVELS_PASSWORD env vars."
        )

    captured_token: list[str] = []

    def on_request(request):
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and not captured_token:
            captured_token.append(auth.removeprefix("Bearer "))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("request", on_request)

        print("Navigating to login page...")
        page.goto(LOGIN_URL, wait_until="networkidle", timeout=30_000)

        # Wait for React to render the form (field is type=text name=username)
        page.wait_for_selector('input[name="username"]', timeout=15_000)

        # Fill credentials
        page.fill('input[name="username"]', email)
        page.fill('input[type="password"]', password)

        # Click Sign In button (don't rely on Enter — MUI forms can be tricky)
        page.click('button:has-text("Sign In")')

        # Wait for a post-login API call to fire (which will carry the token)
        page.wait_for_load_state("networkidle", timeout=20_000)

        # If no token captured yet, navigate to a salary page to trigger API call
        if not captured_token:
            page.goto(
                "https://www.levels.fyi/companies/snap/salaries/",
                wait_until="networkidle",
                timeout=30_000,
            )

        browser.close()

    if not captured_token:
        raise RuntimeError(
            "Could not capture token. Check credentials or try headless=False to debug."
        )

    token = captured_token[0]

    # Cache with 55-minute expiry (Cognito tokens last 1 hour)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({
        "token": token,
        "expires_at": time.time() + 55 * 60,
    }))

    return token


def get_token() -> str:
    """Return a valid Bearer token, refreshing via Playwright if needed."""
    cached = get_cached_token()
    if cached:
        print("Using cached token.")
        return cached
    print("Token expired or missing — logging in via Playwright...")
    return fetch_token_via_playwright()


if __name__ == "__main__":
    token = get_token()
    print(f"\nToken (first 40 chars): {token[:40]}...")
    print(f"Cached to: {TOKEN_FILE}")
    print("\nTo use manually:")
    print(f"  export LEVELS_BEARER='{token[:40]}...'")
