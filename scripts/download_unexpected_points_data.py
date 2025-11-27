#!/usr/bin/env python3
"""
Download unexpected points data from Google Sheets
This is box score data with 'adj_score' calculations

Requirements:
    pip install gspread google-auth google-auth-oauthlib google-auth-httplib2 openpyxl pandas

Setup - OAuth (Recommended for personal use):
    1. Go to https://console.cloud.google.com/
    
    2. Create a project or use existing
       - Example: 95003446653 (Default Gemini Project)
    
    3. Enable Google Sheets API
       - Search bar -> 'Google Sheets API' -> Click 'Enable'
    
    4. Create OAuth 2.0 credentials:
       a) Click 'Create Credentials' button (top right)
          - Select 'OAuth client ID'
          - If prompted, configure OAuth consent screen:
            * Choose 'External'
            * App name: 'Betting Data Downloader'
            * User support email: select your email from dropdown
            * Scroll down to Developer contact: your email
            * Click 'Save and Continue'
            * Scopes page: Just click 'Save and Continue' (don't add anything)
            * Test users page: STOP HERE - continue to step 5 below
       b) Configure OAuth Client:
          - Application type: 'Desktop app' (NOT Web application!)
          - Name: 'Betting Desktop Client'
          - Click 'Create'
       c) Download the credentials:
          - In the popup, click 'Download JSON'
          - Click 'Done'
    
    5. Add yourself as a test user (IMPORTANT):
       - If you're still on OAuth consent screen setup from step 4a:
         * On Test users page, click "+ ADD USERS"
         * Enter your email (mylescgthomas@gmail.com)
         * Click "Add"
         * Click 'Save and Continue'
         * Complete the summary page
         
       - If you already finished the consent screen:
         * Go to "OAuth consent screen" in left sidebar (or Search Bar -> 'OAuth consent screen')
         * Scroll down to "Test users" section
         * Click "+ ADD USERS"
         * Enter your email (mylescgthomas@gmail.com)
         * Click "Save"
       - This allows you to use the app while it's in testing mode
    
    6. Move the downloaded file:
       mv ~/Downloads/client_secret_*.json ~/dev/betting/credentials.json
    
    7. Run the script:
       cd /Users/thomasmyles/dev/betting && pip install gspread google-auth google-auth-oauthlib google-auth-httplib2 openpyxl pandas
       python scripts/download_unexpected_points_data.py
       (First run will open browser to authorize - copy the URL to Safari if needed)

NOTE: THIS WILL GET SCRIPT TO RUN ON LOCAL, BUT IDK ABOUT ON SERVER (LAMBDA)

STUCK HERE ANYWAYS:
Error 403: access_denied
Request details: access_type=offline scope=https://www.googleapis.com/auth/spreadsheets.readonly response_type=code redirect_uri=http://localhost:58542/ state=bbuglNzzNTp6WxJM1KKhd193elMOmW flowName=GeneralOAuthFlow client_id=95003446653-mpjh1cin7all96u6f6fi33j6b7voik2f.apps.googleusercontent.com

Alternative - Service Account (SIMPLER - Recommended):
    1. Go to https://console.cloud.google.com/
       - Select your project (Default Gemini Project)
    
    2. Create a service account:
       a) Go to "APIs & Services" > "Credentials" (or search "Credentials")
       b) Click "Create Credentials" > "Service Account"
       c) Service account details:
          - Name: 'betting-sheets-reader'
          - Service account ID: (auto-fills)
          - Description: 'Read betting data from Google Sheets'
          - Click "Create and Continue"
       d) Grant access (optional):
          - Skip this - just click "Continue"
       e) Grant users access (optional):
          - Skip this - just click "Done"
    
    3. Download the service account key:
       a) You'll be back at the Credentials page
       b) Under "Service Accounts" section, find your new service account
       c) Click on the service account email (betting-sheets-reader@...)
       d) Go to "Keys" tab
       e) Click "Add Key" > "Create new key"
       f) Choose "JSON"
       g) Click "Create" - the key file will download
    
    4. Move the downloaded file:
       mv ~/Downloads/*-*.json ~/dev/betting/service_account.json
    
    5. Share your Google Sheet with the service account:
       a) Open the service_account.json file and copy the "client_email" value
          - It looks like: betting-sheets-reader@your-project.iam.gserviceaccount.com
       b) Go to your Google Sheet: 
          https://docs.google.com/spreadsheets/d/1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558/edit
       c) Click the "Share" button (top right)
       d) Paste the service account email
       e) Make sure it has "Viewer" access
       f) Uncheck "Notify people" (it's a service account, not a person)
       g) Click "Share"

       Note: I don't own this file, so have to ask Kevin Cole for access...
    
    6. Run the script:
       python scripts/download_unexpected_points_data.py
       (No browser authentication needed!)

Security:
    - credentials.json and service_account.json are in .gitignore
    - Never commit these files to git
"""

import os
from datetime import datetime
from pathlib import Path
import io

# Fix SSL certificate issues on macOS (must be done BEFORE importing/using requests)
import ssl
import urllib3

# Disable SSL verification warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import gspread
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as OAuthCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle


SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']


def get_credentials_oauth():
    """Get credentials using OAuth2 (user login flow)"""
    creds = None
    token_path = Path.home() / '.betting' / 'token.pickle'
    credentials_path = Path(__file__).parent.parent / 'credentials.json'
    
    # Token file stores user's access and refresh tokens
    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                print(f"✗ Error: credentials.json not found at {credentials_path}")
                print("Please download OAuth credentials from Google Cloud Console")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES)
            
            # Print the auth URL so user can copy it to Safari if needed
            print("\n" + "="*70)
            print("AUTHENTICATION REQUIRED")
            print("="*70)
            print("A browser window will open for Google authentication.")
            print("If it opens in the wrong browser, the URL will also be printed below.")
            print("Copy the URL and paste it into Safari to authenticate.")
            print("="*70 + "\n")
            
            # Run the local server - this will print the URL and open browser
            creds = flow.run_local_server(port=0, open_browser=True)
        
        # Save the credentials for the next run
        token_path.parent.mkdir(exist_ok=True)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds


def get_credentials_service_account():
    """Get credentials using service account"""
    credentials_path = Path(__file__).parent.parent / 'service_account.json'
    
    if not credentials_path.exists():
        return None
    
    creds = Credentials.from_service_account_file(
        str(credentials_path), 
        scopes=SCOPES
    )
    return creds


def download_unexpected_points_data():
    """
    Download the unexpected points Google Sheet as XLSX
    Saves to ~/Downloads directory
    """
    # Google Sheets ID from the URL
    sheet_id = "1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558"
    
    # Set up the download path
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    print(f"Downloading unexpected points data from Google Sheets...")
    print(f"Sheet ID: {sheet_id}")
    
    try:
        # Try service account first, then OAuth
        creds = get_credentials_service_account()
        if creds:
            print("Using service account credentials...")
        else:
            print("Using OAuth credentials (browser login)...")
            creds = get_credentials_oauth()
            
        if not creds:
            print("✗ No valid credentials found")
            print("\nPlease set up credentials:")
            print("  Option 1: Place service_account.json in project root")
            print("  Option 2: Place credentials.json in project root for OAuth")
            return None
        
        # Authorize gspread
        client = gspread.authorize(creds)
        
        # Open the spreadsheet
        spreadsheet = client.open_by_key(sheet_id)
        
        print(f"✓ Opened spreadsheet: {spreadsheet.title}")
        print(f"  Sheets: {[ws.title for ws in spreadsheet.worksheets()]}")
        
        # Get all worksheets as Excel file
        # Note: gspread doesn't directly export to xlsx, so we'll use pandas
        import pandas as pd
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unexpected_points_data_{timestamp}.xlsx"
        filepath = downloads_dir / filename
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for worksheet in spreadsheet.worksheets():
                print(f"  Downloading sheet: {worksheet.title}")
                df = pd.DataFrame(worksheet.get_all_records())
                df.to_excel(writer, sheet_name=worksheet.title, index=False)
        
        print(f"✓ Successfully downloaded to: {filepath}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")
        
        # Also save a copy without timestamp for easy reference
        latest_filepath = downloads_dir / "unexpected_points_data_latest.xlsx"
        import shutil
        shutil.copy(filepath, latest_filepath)
        print(f"✓ Also saved as: {latest_filepath}")
        
        return filepath
        
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    download_unexpected_points_data()

