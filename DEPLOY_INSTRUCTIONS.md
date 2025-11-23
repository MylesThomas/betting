# Deployment Instructions

## After creating GitHub repo, run:

```bash
cd /Users/thomasmyles/dev/betting

# Stage all files
git add .

# Commit
git commit -m "Initial commit: NBA arbitrage dashboard"

# Add remote (REPLACE WITH YOUR REPO URL)
git remote add origin https://github.com/YOUR_USERNAME/nba-arb-finder.git

# Push
git branch -M main
git push -u origin main
```

## Then deploy to Streamlit Cloud:
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repo: nba-arb-finder
5. Main file path: streamlit_app/app.py
6. Click "Deploy"!

Done! 🚀
