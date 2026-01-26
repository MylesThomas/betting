"""
Extract text from Monte Carlo book screenshots using OCR.

Context: User owns the book "Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors"
and took 50 screenshots of the "Prediction Models" chapter. This script transcribes the screenshots
into one continuous text file for implementation reference.

Usage:
    cd /Users/thomasmyles/dev/betting
    python books/transcribe_with_ocr.py --test  # Process only first page for testing
    python books/transcribe_with_ocr.py --full-run \
        --input-dir "/Users/thomasmyles/Downloads/tmp/monte carlo book/prediction models" \
        --output-file ~/Downloads/tmp/soccer_script.md
"""
import os
import argparse
from pathlib import Path

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("Installing required packages...")
    os.system("pip install pillow pytesseract")
    from PIL import Image
    import pytesseract

def transcribe_screenshots(input_dir, output_file, test_mode=False, full_run=False):
    """Read all screenshots in order and extract text."""
    screenshots_dir = Path(input_dir)
    screenshots = sorted(screenshots_dir.glob("Screenshot*.png"))
    
    output_path = Path(output_file).expanduser()
    
    all_text = []
    all_text.append("# PREDICTION MODELS")
    all_text.append("\n**Source:** Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors\n")
    all_text.append("**Chapter:** Prediction Models\n")
    all_text.append("=" * 80 + "\n")
    
    # Sort screenshots by name / time taken (in the filename)
    screenshots = sorted(screenshots, key=lambda x: x.name.split('_')[-1].split('.')[0])
    
    # Determine mode
    if test_mode:
        screenshots = screenshots[:1]
        print(f"TEST MODE: Processing only first screenshot...")
    elif full_run:
        print(f"FULL RUN MODE: Processing all {len(screenshots)} screenshots...")
    else:
        print(f"Processing {len(screenshots)} screenshots...")
    
    for i, screenshot_path in enumerate(screenshots, 1):
        print(f"Processing {i}/{len(screenshots)}: {screenshot_path.name}")
        
        try:
            # Open image
            img = Image.open(screenshot_path)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(img)
            
            # Log extracted text
            print(f"\n{'='*80}")
            print(f"Text from {screenshot_path.name}:")
            print(f"{'='*80}")
            print(text)
            print(f"{'='*80}\n")
            
            # Add to output
            all_text.append(f"\n--- Page {i} ---\n")
            all_text.append(text)
            all_text.append("\n")
            
        except Exception as e:
            print(f"Error processing {screenshot_path.name}: {e}")
            all_text.append(f"\n--- Page {i} (ERROR) ---\n")
            all_text.append(f"Error: {e}\n")
    
    # Write to file
    output_text = "\n".join(all_text)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text)
    
    print(f"\nComplete! Transcribed text saved to: {output_path}")
    print(f"Total characters: {len(output_text)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe Monte Carlo book screenshots using OCR"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only the first screenshot"
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Full run mode: process all screenshots (explicit confirmation)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/Users/thomasmyles/Downloads/tmp/monte carlo book/prediction models",
        help="Directory containing the screenshots to process"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/Users/thomasmyles/dev/betting/books/monte_carlo_prediction_models_full_chapter.md",
        help="Output file path for the transcribed text"
    )
    args = parser.parse_args()
    
    transcribe_screenshots(
        input_dir=args.input_dir,
        output_file=args.output_file,
        test_mode=args.test,
        full_run=args.full_run
    )
