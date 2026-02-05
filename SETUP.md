# Development Environment Setup

## Overview

This project uses `uv` for fast Python package management with Python 3.13.

## Quick Start

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 2. Verify installation

```bash
python --version  # Should show Python 3.13.5
which python      # Should point to .venv/bin/python
```

### 3. Test core imports

```bash
python -c "import pandas; import numpy; import pyarrow; import nba_api; print('✅ All packages working')"
```

## Installing Additional Packages

### Using uv (recommended - much faster)

```bash
# Install a new package
uv pip install package-name

# Install with specific version
uv pip install package-name==1.2.3

# Update pyproject.toml dependencies and sync
# 1. Add package to pyproject.toml dependencies list
# 2. Run: uv pip install -e .
```

### Using pip (fallback)

```bash
pip install package-name
```

## Key Dependencies

### Data Processing
- pandas (>=2.1.4)
- numpy (>=1.26.2)
- pyarrow (>=14.0.0)
- duckdb (>=0.9.0)

### Sports APIs
- nba_api (>=1.4.1)

### AWS
- boto3 (>=1.26.0)

### Web Automation
- playwright (>=1.48.0)

### Visualization
- matplotlib (>=3.8.0)
- streamlit (>=1.28.0)

## Project Structure

```
betting/
├── .venv/              # Virtual environment (gitignored)
├── pyproject.toml      # Project config & dependencies
├── src/                # Source utilities
├── scripts/            # Lambda functions & automation
├── analysis/           # Analysis scripts
├── config/             # YAML configs
├── notebooks/          # Jupyter notebooks
└── tests/              # Test files
```

## Common Commands

### Deactivate environment
```bash
deactivate
```

### Recreate environment from scratch
```bash
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
playwright install
```

### List installed packages
```bash
uv pip list
```

### Show package info
```bash
uv pip show package-name
```

### Freeze current environment
```bash
uv pip freeze > requirements-frozen.txt
```

## Running Lambda Functions

Lambda functions in `scripts/` can be tested locally:

```bash
# Activate environment first
source .venv/bin/activate

# Run a lambda function
python scripts/lambda_function_nba_player_scoring_props.py
```

## Running Analysis Scripts

```bash
source .venv/bin/activate
python analysis/detect_nba_dnp_scenarios.py
```

## Streamlit Dashboard

```bash
source .venv/bin/activate
cd streamlit_app
streamlit run app.py
```

## Browser Automation (Playwright)

Browsers are already installed. To reinstall or update:

```bash
playwright install
```

## Troubleshooting

### ImportError: No module named 'xxx'

Make sure the virtual environment is activated:
```bash
source .venv/bin/activate
```

### Python version mismatch

Verify you're using the venv Python:
```bash
which python  # Should be .venv/bin/python
python --version  # Should be 3.13.x
```

### uv not found

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Why uv?

`uv` is 10-100x faster than pip because it's written in Rust and uses:
- Parallel dependency resolution
- Efficient caching
- Optimized network requests

Example: Installing all dependencies took ~6 seconds with uv vs ~60+ seconds with pip.
