# Fixing SSL Certificate Issues with NBA API on macOS

## TL;DR - What Fixed It

**Problem:** `SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed`

**Solution:** Monkey-patch the `requests` library to disable SSL verification before importing `nba_api`.

```python
import ssl
import urllib3
import requests

# Disable SSL verification warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests Session to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Now import nba_api (it will use the patched requests)
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
```

---

## Full Sequence: Trial and Error

### Issue Context

When trying to fetch NBA data using `nba_api`, we encountered SSL certificate verification errors on macOS. The NBA stats API at `stats.nba.com` couldn't be reached due to missing/invalid SSL certificates in the Python environment.

**Error Message:**
```
HTTPSConnectionPool(host='stats.nba.com', port=443): Max retries exceeded with url: 
/stats/playergamelog?... (Caused by SSLError(SSLCertVerificationError(1, 
'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local 
issuer certificate (_ssl.c:1000)')))
```

---

### Attempt 1: Run macOS Certificate Install Script ❌

**What we tried:**
```bash
cd /Applications/Python\ 3.12/ && ./Install\ Certificates.command
```

**Rationale:** macOS Python installations include a script to install SSL certificates from `certifi`.

**Result:** 
- Script ran but with warnings
- SSL error persisted
- The certificates were installed to system Python, but we're using pyenv Python

**Why it failed:** We're using a pyenv-managed Python (`/Users/thomasmyles/.pyenv/versions/3.12.7`), not the system Python, so the system certificate install didn't help.

---

### Attempt 2: Basic SSL Context Override ❌

**What we tried:**
```python
import ssl
import certifi

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
```

**Rationale:** Override Python's default SSL context to not verify certificates.

**Result:** 
- Still failed with same error
- `nba_api` uses `requests` library internally which has its own SSL handling

**Why it failed:** The `requests` library doesn't use Python's default SSL context; it has its own verification mechanism.

---

### Attempt 3: Add urllib3 Warning Suppression ❌

**What we tried:**
```python
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

**Rationale:** Suppress SSL warnings in urllib3 (which requests uses under the hood).

**Result:** 
- Warnings suppressed, but SSL verification still happening
- Same SSL error

**Why it failed:** Suppressing warnings doesn't disable verification; `requests` was still trying to verify certificates.

---

### Attempt 4: Upgrade certifi Package ❌

**What we tried:**
```bash
pip install --upgrade certifi
```

**Rationale:** Get the latest SSL certificates bundle.

**Result:** 
- Upgraded certifi from `2023.11.17` to `2025.11.12`
- Created dependency conflict (nba_api requires `certifi<2024.0.0`)
- Still same SSL error

**Why it failed:** Even with latest certificates, the root issue was certificate validation, not missing certificates. Also broke compatibility.

---

### Attempt 5: Downgrade certifi ❌

**What we tried:**
```bash
pip install 'certifi>=2023.7.22,<2024.0.0'
```

**Rationale:** Match nba_api's exact certifi requirements.

**Result:** 
- Restored compatible certifi version
- SSL error persisted

**Why it failed:** The certificate bundle wasn't the issue; verification itself was the problem.

---

### Attempt 6: Environment Variables ❌

**What we tried:**
```bash
PYTHONHTTPSVERIFY=0 CURL_CA_BUNDLE="" python nba_api_setup.py
```

**Rationale:** Use environment variables to disable SSL verification.

**Result:** 
- Variables set correctly
- SSL error still occurred
- `requests` library ignores these environment variables

**Why it failed:** The `requests` library doesn't respect `PYTHONHTTPSVERIFY` or `CURL_CA_BUNDLE` variables.

---

### Attempt 7: Monkey-Patch requests.Session ✅ SUCCESS!

**What we tried:**
```python
import ssl
import urllib3
import requests

# Disable SSL verification warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Import nba_api AFTER patching
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
```

**Rationale:** 
- The `nba_api` uses `requests.Session` internally
- By patching the `Session.request` method BEFORE importing `nba_api`, we force all requests to use `verify=False`
- This disables SSL certificate verification at the requests level

**Result:** 
- ✅ **SUCCESS!** 
- API calls work perfectly
- Fetched Draymond Green's game log successfully
- Found 3 players with long under-streaks

**Why it worked:** 
- We intercepted requests at the exact point where `nba_api` makes them
- The patch was applied before `nba_api` was imported, so all its internal Session objects use the patched method
- `verify=False` directly tells `requests` to skip SSL verification

---

## Key Learnings

1. **Order matters:** The monkey-patch must be applied BEFORE importing the library that uses requests.

2. **requests is independent:** The `requests` library doesn't respect Python's default SSL context or standard environment variables.

3. **pyenv complications:** Using pyenv Python means system-level certificate installations don't help.

4. **nba_api internals:** The library uses `requests.Session` objects, which we can patch at the class level.

5. **Security trade-off:** Disabling SSL verification is NOT recommended for production. This is acceptable for:
   - Local development
   - Accessing trusted APIs (like official NBA stats)
   - When certificate issues are environment-specific, not security-related

---

## Alternative Solutions (Not Tested)

If you want to keep SSL verification enabled, try:

### Option 1: Install Certificates to pyenv Python
```bash
# Find your pyenv Python's certificate script
~/.pyenv/versions/3.12.7/bin/python -m pip install --upgrade certifi
~/.pyenv/versions/3.12.7/bin/python -c "import certifi; print(certifi.where())"
```

### Option 2: Use requests-specific environment variable
```bash
export REQUESTS_CA_BUNDLE=/path/to/certifi/cacert.pem
```

### Option 3: Set up proper CA certificates
```bash
# On macOS with Homebrew
brew install ca-certificates
export SSL_CERT_FILE=/usr/local/etc/ca-certificates/cert.pem
```

### Option 4: Use system certificates
```python
import ssl
import certifi
import requests

session = requests.Session()
session.verify = certifi.where()
```

---

## When to Use This Fix

**Use the monkey-patch when:**
- ✅ Local development only
- ✅ Trusted API (like NBA stats)
- ✅ SSL errors block functionality
- ✅ You've exhausted proper certificate solutions

**Don't use this when:**
- ❌ Production environment
- ❌ Handling sensitive data
- ❌ User authentication/passwords
- ❌ Financial transactions
- ❌ Any security-critical application

---

## Final Code Structure

```python
# 1. Import standard libraries
import pandas as pd
import time

# 2. Import and configure SSL/requests BEFORE nba_api
import ssl
import urllib3
import requests

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# 3. NOW import nba_api (uses patched requests)
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# 4. Rest of your code works normally
player_id = players.find_players_by_full_name("Draymond Green")[0]['id']
gamelog = playergamelog.PlayerGameLog(player_id=player_id, season="2024-25")
df = gamelog.get_data_frames()[0]
print(df.head())
```

---

## References

- [Python SSL Documentation](https://docs.python.org/3/library/ssl.html)
- [Requests SSL Verification](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification)
- [nba_api GitHub Issues](https://github.com/swar/nba-api/issues)
- [macOS Python SSL Issues](https://bugs.python.org/issue28150)

