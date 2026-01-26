"""
Transcribe screenshots from Monte Carlo or Bust book chapter.

This script reads all screenshots in order and uses OCR to extract text.
We'll manually transcribe since the Read tool should handle images.
"""
import os
from pathlib import Path

# Get all screenshots in order
screenshots_dir = Path("/Users/thomasmyles/dev/betting/books/screenshots")
screenshots = sorted(screenshots_dir.glob("Screenshot*.png"))

print(f"Found {len(screenshots)} screenshots")
for i, screenshot in enumerate(screenshots, 1):
    print(f"{i}. {screenshot.name}")
