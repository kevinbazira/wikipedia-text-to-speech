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
from wiki_tts.timestamps import align_words, init_aligner, timestamps_to_vtt

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
    sess_options.enable_cpu_mem_arena = False
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


# ── Model initialisation ─────────────────────────────────────────────────────


@worker_process_init.connect
def init_worker(**kwargs):
    global kokoro_model

    print("Pre-loading Kokoro-ONNX FP32 model into memory...")
    from kokoro_onnx import Kokoro

    kokoro_model = Kokoro(MODEL_FILE, VOICES_FILE)

    init_nemo()
    init_aligner()


# ── Task ──────────────────────────────────────────────────────────────────────


@app.task
def generate_section_audio(article: str, section: str, text: str):
    print(f"Processing: {article} -> {section}")

    global kokoro_model

    text = clean_spoken_text(text)

    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")

    output_dir = f"{AUDIO_OUTPUT_DIR}/{safe_article}"
    os.makedirs(output_dir, exist_ok=True)

    mp3_path = f"{output_dir}/{safe_section}.mp3"
    vtt_path = f"{output_dir}/{safe_section}.vtt"

    # ── Section heading announcement ──
    heading_text = f"{article}." if section == "Lead" else f"{section}."
    heading_audio, sample_rate = kokoro_model.create(heading_text, voice="af_heart", speed=1.0, lang="en-us")
    pause = np.zeros(int(sample_rate * 1.0), dtype=heading_audio.dtype)
    print(f"  Prepended heading: '{heading_text}'")

    # ── Start FFmpeg early (streaming writer) ──
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

    all_word_timestamps: list[dict] = []
    current_time_ms = 0.0

    try:
        # ── Stream heading + pause ──
        process.stdin.write(heading_audio.tobytes())
        process.stdin.write(pause.tobytes())

        heading_ts = align_words(heading_audio, sample_rate, heading_text)
        for t in heading_ts:
            t["start_ms"] += current_time_ms
            t["end_ms"] += current_time_ms
            all_word_timestamps.append(t)

        current_time_ms += (len(heading_audio) / sample_rate) * 1000 + 1000.0

        # ── Streaming content chunks ──
        chunks = _split_text(text, max_chars=400)
        print(f"  Split into {len(chunks)} chunks (max 400 chars each)")

        fade_len = 120
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        prev_tail: np.ndarray | None = None

        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i + 1}/{len(chunks)} (len={len(chunk)} chars)")
            chunk_audio, _ = kokoro_model.create(chunk, voice="af_heart", speed=1.0, lang="en-us")

            # Align immediately (O(1) per chunk)
            chunk_ts = align_words(chunk_audio, sample_rate, chunk)
            for t in chunk_ts:
                t["start_ms"] += current_time_ms
                t["end_ms"] += current_time_ms
                all_word_timestamps.append(t)

            current_time_ms += (len(chunk_audio) / sample_rate) * 1000

            # Stream to FFmpeg with live crossfade
            if len(chunks) == 1:
                process.stdin.write(chunk_audio.tobytes())
            elif i == 0:
                chunk_audio[-fade_len:] *= fade_out
                prev_tail = chunk_audio[-fade_len:].copy()
                process.stdin.write(chunk_audio[:-fade_len].tobytes())
            elif i < len(chunks) - 1:
                chunk_audio[:fade_len] *= fade_in
                chunk_audio[:fade_len] += prev_tail
                chunk_audio[-fade_len:] *= fade_out
                prev_tail = chunk_audio[-fade_len:].copy()
                process.stdin.write(chunk_audio[:-fade_len].tobytes())
            else:
                chunk_audio[:fade_len] *= fade_in
                chunk_audio[:fade_len] += prev_tail
                process.stdin.write(chunk_audio.tobytes())

        # ── Finalize ──
        process.stdin.close()
        process.wait()

        if all_word_timestamps:
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(timestamps_to_vtt(all_word_timestamps))
            print(f"  Saved {len(all_word_timestamps)} word timestamps")

        print(f"Finished: {mp3_path}")
        return mp3_path

    except Exception:
        process.stdin.close()
        process.wait()
        raise
