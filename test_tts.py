import os
import subprocess
from kokoro import KModel, KPipeline
import soundfile as sf

# 1. Configuration - Using Absolute Paths for Toolforge
FFMPEG_BIN = os.path.expanduser("~/bin/ffmpeg")
OUTPUT_WAV = "test_output.wav"
OUTPUT_MP3 = "test_output.mp3"

def main():
    print("--- Initializing Kokoro Pipeline ---")
    # 'a' for American English, 'b' for British
    pipeline = KPipeline(lang_code='a') 

    text = "Hello! This is a test of the wiki TTS system running on Wikimedia Toolforge."

    print("--- Generating Audio ---")
    # generator yields (graphemes, phonemes, audio_tensor)
    generator = pipeline(text, voice='af_bella', speed=1, split_pattern=r'\n+')
    
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(OUTPUT_WAV, audio, 24000) # Kokoro uses 24kHz
        print(f"Generated segment {i}")

    print("--- Converting to MP3 via Static FFmpeg ---")
    try:
        subprocess.run([
            FFMPEG_BIN, "-y", 
            "-i", OUTPUT_WAV, 
            "-codec:a", "libmp3lame", 
            "-q:a", "2", 
            OUTPUT_MP3
        ], check=True)
        print(f"Success! Created {OUTPUT_MP3}")
    except Exception as e:
        print(f"FFmpeg Error: {e}")

if __name__ == "__main__":
    main()