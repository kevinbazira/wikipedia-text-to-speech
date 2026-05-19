import os
import re
import subprocess

import numpy as np
import onnxruntime as ort
from celery import Celery
from celery.signals import worker_process_init

from wiki_tts.config import (
    AUDIO_OUTPUT_DIR,
    CELERY_BROKER_URL,
    CELERY_KEY_PREFIX,
    CELERY_RESULT_BACKEND,
    CELERY_TASK_DEFAULT_QUEUE,
    FFMPEG_PATH,
    MODEL_FILE,
    ORT_NUM_THREADS,
    VOICES_FILE,
)
from wiki_tts.text import clean_spoken_text, init_nemo

# ── Celery app ───────────────────────────────────────────────────────────────
app = Celery("wiki_tts", broker=CELERY_BROKER_URL)
app.conf.update(
    task_default_queue=CELERY_TASK_DEFAULT_QUEUE,
    result_backend=CELERY_RESULT_BACKEND,
    key_prefix=CELERY_KEY_PREFIX,
)

# ── ONNX Runtime threading patch ─────────────────────────────────────────────
original_init = ort.InferenceSession.__init__


def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ORT_NUM_THREADS
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)


ort.InferenceSession.__init__ = patched_init

# ─── Global model reference ──────────────────────────────────────────────────
kokoro_model = None


# ── Text chunking ─────────────────────────────────────────────────────────────


def _split_text(text: str, max_chars: int = 800) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
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


# ── Model initialisation ─────────────────────────────────────────────────────


@worker_process_init.connect
def init_worker(**kwargs):
    global kokoro_model

    print("Pre-loading Kokoro-ONNX FP32 model into memory...")
    from kokoro_onnx import Kokoro

    kokoro_model = Kokoro(MODEL_FILE, VOICES_FILE)

    init_nemo()


# ── Task ──────────────────────────────────────────────────────────────────────


@app.task
def generate_section_audio(article: str, section: str, text: str):
    print(f"Processing: {article} -> {section}")

    global kokoro_model

    # Normalize raw Wikipedia text with NeMo (warmed up during worker init).
    # NeMo handles numbers, units, dates, currency, abbreviations, etc.
    text = clean_spoken_text(text)

    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")

    output_dir = f"{AUDIO_OUTPUT_DIR}/{safe_article}"
    os.makedirs(output_dir, exist_ok=True)

    mp3_path = f"{output_dir}/{safe_section}.mp3"

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

    ffmpeg_cmd = [
        FFMPEG_PATH,
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-vn",
        "-b:a",
        "64k",
        mp3_path,
    ]

    process = subprocess.Popen(  # noqa: S603
        ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    process.communicate(input=audio_array.tobytes())

    print(f"Finished: {mp3_path}")
    return mp3_path
