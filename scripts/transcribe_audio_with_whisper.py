"""
Audio Transcription Using Whisper Large-v2 Model

Context:
Thomas wants to transcribe audio files locally using OpenAI's Whisper large-v2 model.
This script reads an audio file and generates a text transcript.

Usage:
    python3 scripts/transcribe_audio_with_whisper.py

Input:
    ~/Downloads/GLT9188591642.mp3 (or specify path via command line)

Output:
    Same directory as input file with .txt extension
    Example: ~/Downloads/GLT9188591642.txt

Requirements:
    pip install openai-whisper
    
Note:
    First run will download the large-v2 model (~3GB).
    Subsequent runs will use cached model.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Error: whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def transcribe_audio(audio_path, model_name="large-v2", output_path=None):
    """
    Transcribe audio file using Whisper model.
    
    Args:
        audio_path: Path to audio file (mp3, wav, m4a, etc.)
        model_name: Whisper model to use (tiny, base, small, medium, large, large-v2)
        output_path: Optional output path for transcript. If None, uses same dir as audio.
    
    Returns:
        Path to output transcript file
    """
    audio_path = Path(audio_path).expanduser()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"🎤 Loading Whisper model: {model_name}")
    print("   (First run will download ~3GB model)")
    model = whisper.load_model(model_name)
    
    print(f"\n📝 Transcribing: {audio_path.name}")
    print(f"   File size: {audio_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"   Progress will be shown below...\n")
    
    # Transcribe with verbose progress
    result = model.transcribe(str(audio_path), verbose=True)
    
    # Determine output path
    if output_path is None:
        output_path = audio_path.with_suffix('.txt')
    else:
        output_path = Path(output_path).expanduser()
    
    # Write transcript
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    print(f"\n✅ Transcript saved to: {output_path}")
    print(f"   Word count: {len(result['text'].split())}")
    print(f"   Character count: {len(result['text'])}")
    
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio file using Whisper large-v2 model"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="~/Downloads/GLT9188591642.mp3",
        help="Path to audio file (default: ~/Downloads/GLT9188591642.mp3)"
    )
    parser.add_argument(
        "--model",
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model to use (default: large-v2)"
    )
    parser.add_argument(
        "--output",
        help="Output path for transcript (default: same dir as audio with .txt extension)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = transcribe_audio(
            audio_path=args.audio_file,
            model_name=args.model,
            output_path=args.output
        )
        
        # Print preview
        print("\n📄 Transcript preview:")
        print("-" * 80)
        with open(output_path, 'r', encoding='utf-8') as f:
            preview = f.read()[:500]
            print(preview)
            if len(f.read()) > 500:
                print("\n... (truncated)")
        print("-" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
