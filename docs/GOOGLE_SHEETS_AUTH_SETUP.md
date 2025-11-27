# Google Sheets Authentication Setup

This guide explains how to set up authentication to download private Google Sheets.

## Quick Setup (Recommended: Service Account)

### Step 1: Create a Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select or create a project
3. Enable the Google Sheets API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

4. Create a service account:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Name it something like "betting-sheets-reader"
   - Click "Create and Continue"
   - Skip optional steps, click "Done"

5. Create and download the key:
   - Click on your new service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON"
   - Download the file

6. Save the file:
   ```bash
   mv ~/Downloads/your-project-*.json /Users/thomasmyles/dev/betting/service_account.json
   ```

### Step 2: Share the Google Sheet

1. Open your Google Sheet
2. Click "Share" button
3. Copy the service account email from the JSON file (looks like: `betting-sheets-reader@your-project.iam.gserviceaccount.com`)
4. Paste it into the share dialog
5. Give it "Viewer" access
6. Click "Share"

### Step 3: Install Required Packages

```bash
pip install gspread google-auth google-auth-oauthlib google-auth-httplib2 openpyxl pandas
```

### Step 4: Run the Script

```bash
python scripts/download_unexpected_points_data.py
```

---

## Alternative Setup (OAuth - Browser Login)

If you prefer to authenticate with your own Google account:

### Step 1: Create OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Sheets API (same as above)
3. Go to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "OAuth client ID"
5. If prompted, configure the OAuth consent screen:
   - Choose "External"
   - Fill in app name: "Betting Data Downloader"
   - Add your email
   - Save
6. Choose "Desktop app" as application type
7. Name it "Betting Desktop Client"
8. Click "Create"
9. Download the credentials JSON file

### Step 2: Save Credentials

```bash
mv ~/Downloads/client_secret_*.json /Users/thomasmyles/dev/betting/credentials.json
```

### Step 3: Run the Script

The first time you run it, a browser window will open asking you to log in and authorize the app:

```bash
python scripts/download_unexpected_points_data.py
```

After the first run, your credentials will be saved in `~/.betting/token.pickle` and you won't need to log in again.

---

## Troubleshooting

### "No valid credentials found"
- Make sure you have either `service_account.json` or `credentials.json` in the project root
- Check file permissions

### "Permission denied" when accessing the sheet
- For service account: Make sure you shared the sheet with the service account email
- For OAuth: Make sure you're logged in with an account that has access to the sheet

### SSL Certificate Errors
- The script already includes SSL fixes for macOS
- If issues persist, check `api_setup/fixing_ssl.md`

### Import errors
- Make sure all packages are installed:
  ```bash
  pip install gspread google-auth google-auth-oauthlib google-auth-httplib2 openpyxl pandas
  ```

---

## Files Created

After running the script successfully:

- `~/Downloads/unexpected_points_data_YYYYMMDD_HHMMSS.xlsx` - Timestamped version
- `~/Downloads/unexpected_points_data_latest.xlsx` - Latest version (overwritten each time)
- `~/.betting/token.pickle` - Cached OAuth credentials (if using OAuth)

---

## Security Notes

⚠️ **Important:**
- Never commit `service_account.json` or `credentials.json` to git
- Never share your service account key or credentials
- Add these files to `.gitignore`

To add to `.gitignore`:
```bash
echo "service_account.json" >> .gitignore
echo "credentials.json" >> .gitignore
```

