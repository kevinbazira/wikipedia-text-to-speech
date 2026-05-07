import os
import re
import subprocess
import numpy as np
from celery import Celery
from celery.signals import worker_process_init
import onnxruntime as ort

# Initialize Celery to use Redis as the message broker
local_redis_url = "redis://localhost:6379/0"
toolforge_redis_url = "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379/0"
app = Celery("wiki_tts", broker=local_redis_url)

# Global variable to hold the ONNX model in memory
kokoro_model = None

# ---------------------------------------------------------------------------
# ONNX Runtime threading
#
# The original code forced intra_op_num_threads=1 unconditionally.  That's
# overly conservative — it disables ONNX's thread-to-core affinity (which
# requires the default 0 / unset value) AND leaves parallelism on the table.
#
# Set ORT_NUM_THREADS per environment:
#   Local (96-core EPYC):      4   (good balance, no oversubscription)
#   Toolforge --cpu 1:         1
#   Toolforge --cpu 2:         2
# ---------------------------------------------------------------------------
NUM_THREADS = int(os.environ.get("ORT_NUM_THREADS", "4"))

original_init = ort.InferenceSession.__init__


def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()

    sess_options.intra_op_num_threads = NUM_THREADS
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)


ort.InferenceSession.__init__ = patched_init

# ---------------------------------------------------------------------------
# Text chunking
#
# The benchmark showed ~0.1 s/char on the Celery worker (Docker) — the
# bottleneck is linear (vocoder), not quadratic (attention).  So chunking
# doesn't help with O(n²) but splitting very long passages still improves
# memory usage and allows finer-grained progress tracking.
# ---------------------------------------------------------------------------

def _split_text(text: str, max_chars: int = 800) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            buf = ""
            for word in words:
                if len(buf) + len(word) + 1 > max_chars:
                    if buf:
                        chunks.append(buf)
                    buf = word
                else:
                    buf = buf + " " + word if buf else word
            current = buf
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = current + " " + sentence if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def _crossfade(a: np.ndarray, b: np.ndarray, fade_len: int = 120) -> np.ndarray:
    if fade_len <= 0 or len(a) < fade_len or len(b) < fade_len:
        return np.concatenate([a, b])

    fade_out = np.linspace(1, 0, fade_len)
    fade_in = np.linspace(0, 1, fade_len)

    amp = a.copy()
    bmp = b.copy()
    amp[-fade_len:] *= fade_out
    bmp[:fade_len] *= fade_in

    return np.concatenate([amp[:-fade_len], amp[-fade_len:] + bmp[:fade_len], bmp[fade_len:]])


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

@worker_process_init.connect
def init_worker(**kwargs):
    global kokoro_model
    print("Pre-loading Kokoro-ONNX FP32 model into memory...")
    from kokoro_onnx import Kokoro
    # FP32 is faster than INT8 on CPUs without VNNI (confirmed by benchmark)
    kokoro_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@app.task
def generate_section_audio(article: str, section: str, text: str):
    print(f"Processing: {article} -> {section}")

    global kokoro_model

    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")

    output_dir = f"./audio_output/{safe_article}"
    os.makedirs(output_dir, exist_ok=True)

    mp3_path = f"{output_dir}/{safe_section}.mp3"

    # Generate audio, chunking long passages
    chunks = _split_text(text, max_chars=800)

    if len(chunks) == 1:
        audio_array, sample_rate = kokoro_model.create(text, voice="af_heart", speed=1.0, lang="en-us")
    else:
        print(f"  Split into {len(chunks)} chunks (max 800 chars each)")
        sample_rate = None
        audio_parts = []
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i + 1}/{len(chunks)} (len={len(chunk)} chars)")
            chunk_audio, sr = kokoro_model.create(chunk, voice="af_heart", speed=1.0, lang="en-us")
            if sample_rate is None:
                sample_rate = sr
            audio_parts.append(chunk_audio)

        audio_array = audio_parts[0]
        for part in audio_parts[1:]:
            audio_array = _crossfade(audio_array, part, fade_len=120)

    # Pipe audio directly to FFmpeg (no intermediate WAV file)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-vn", "-b:a", "64k", mp3_path
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    process.communicate(input=audio_array.tobytes())

    print(f"Finished: {mp3_path}")
    return mp3_path
