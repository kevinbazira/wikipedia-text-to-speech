import os
import subprocess
import soundfile as sf
import numpy as np
from celery import Celery

# Initialize Celery to use local Redis as the message broker
app = Celery('tts_worker', broker='redis://localhost:6379/0')

# Global variable to hold the model in memory
pipeline = None

def get_pipeline():
    """
    Lazy-loads the Kokoro model only once per worker process.
    This ensures we don't reload the model for every task, which would be very inefficient.
    """
    global pipeline
    if pipeline is None:
        print("Loading Kokoro model into memory...")
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code='a', device='cpu')
    return pipeline

@app.task
def generate_section_audio(article: str, section: str, text: str):
    print(f"Processing: {article} -> {section}")
    
    pipe = get_pipeline()
    
    # 1. Generate Audio Chunks
    # Kokoro yields tuples of (graphemes, phonemes, audio_array)
    generator = pipe(text, voice='af_heart', speed=1.0)
    
    audio_chunks =[]
    sample_rate = 24000
    for _, _, audio in generator:
        if audio is not None:
            audio_chunks.append(audio)
            
    if not audio_chunks:
        print(f"No audio generated for {section}")
        return None
        
    # Stitch all sentences together into one long audio array
    full_audio = np.concatenate(audio_chunks)
    
    # 2. Prepare File Paths
    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")
    
    output_dir = f"./audio_output/{safe_article}"
    os.makedirs(output_dir, exist_ok=True)
    
    wav_path = f"{output_dir}/{safe_section}.wav"
    mp3_path = f"{output_dir}/{safe_section}.mp3"
    
    # 3. Save as temporary WAV
    sf.write(wav_path, full_audio, samplerate=sample_rate)
    
    # 4. Convert to highly compressed MP3 using FFmpeg
    # Parameters: -ac 1 (Mono), -b:a 64k (64kbps bitrate, perfect for speech)
    ffmpeg_cmd =[
        "ffmpeg", "-y", "-i", wav_path,
        "-vn", "-ar", str(sample_rate), "-ac", "1", "-b:a", "64k",
        mp3_path
    ]
    # Run FFmpeg quietly
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 5. Clean up the massive WAV file
    os.remove(wav_path)
    
    print(f"Finished: {mp3_path}")
    return mp3_path