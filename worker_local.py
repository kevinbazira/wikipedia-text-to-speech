import os
import subprocess
import soundfile as sf
import numpy as np
from celery import Celery
import onnxruntime as ort
import asyncio

# Initialize Celery to use local Redis as the message broker
local_redis_url = "redis://localhost:6379/0"
toolforge_redis_url = "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379/0"
app = Celery("wiki_tts", broker=local_redis_url) # Toolforge Redis

# Global variable to hold the ONNX model in memory
kokoro_model = None

# IMPORTANT: Fix ONNX thread-thrashing (Monkeypatch)
# Intercept ONNX Runtime initialization to strictly force 1 thread
original_init = ort.InferenceSession.__init__

def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    
    # Force 1 thread for mathematical operations
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    
    # Call the original initialization with our strict options
    original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)

# Apply the patch!
ort.InferenceSession.__init__ = patched_init


def get_pipeline():
    """
    Lazy-loads the Kokoro ONNX model only once per worker process.
    """
    global kokoro_model
    if kokoro_model is None:
        print("Loading highly optimized Kokoro-ONNX model into memory...")
        from kokoro_onnx import Kokoro
        
        # Initialize with the files we downloaded via wget
        # kokoro_model = Kokoro("kokoro-v0_19.onnx", "voices.json")
        kokoro_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    return kokoro_model

@app.task
def generate_section_audio(article: str, section: str, text: str):
    print(f"Processing: {article} -> {section}")
    
    kokoro = get_pipeline()
    
    # 1. Prepare File Paths
    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")
    
    output_dir = f"./audio_output/{safe_article}"
    os.makedirs(output_dir, exist_ok=True)
    
    wav_path = f"{output_dir}/{safe_section}.wav"
    mp3_path = f"{output_dir}/{safe_section}.mp3"

    # 2. Generate and Save Audio
    # Because ONNX is so fast, we can safely generate the whole section synchronously
    # without needing complex asyncio streaming loops!
    audio_array, sample_rate = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
    
    # Save the complete array to disk at once
    sf.write(wav_path, audio_array, samplerate=sample_rate)

    # 3. Convert to highly compressed MP3 using FFmpeg
    # Parameters: -ac 1 (Mono), -b:a 64k (64kbps bitrate, perfect for speech)
    # local run uses "ffmpeg" assuming it's in PATH, toolforge uses absolute path "/data/project/wiki-tts/bin/ffmpeg"
    ffmpeg_cmd =[
        "ffmpeg", "-y", "-i", wav_path,
        "-vn", "-ar", str(sample_rate), "-ac", "1", "-b:a", "64k",
        mp3_path
    ]
    # Run FFmpeg quietly
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 4. Clean up the massive WAV file
    os.remove(wav_path)
    
    print(f"Finished: {mp3_path}")
    return mp3_path