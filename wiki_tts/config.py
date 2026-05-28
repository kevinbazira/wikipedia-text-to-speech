import os
import tempfile
from pathlib import Path

# ── Environment detection ────────────────────────────────────────────────────
IS_TOOLFORGE = os.path.exists("/data/project/wiki-tts")

# ── Redis ────────────────────────────────────────────────────────────────────
TOOLFORGE_REDIS_URL = "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379/0"
LOCAL_REDIS_URL = "redis://localhost:6379/0"
REDIS_URL = TOOLFORGE_REDIS_URL if IS_TOOLFORGE else LOCAL_REDIS_URL

LOCK_TTL = 1800  # Redis lock TTL (30 minutes)

# ── Celery ───────────────────────────────────────────────────────────────────
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_DEFAULT_QUEUE = "wiki-tts-queue"
CELERY_KEY_PREFIX = "wiki-tts:"

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
ORT_NUM_THREADS = int(os.environ.get("ORT_NUM_THREADS", "1"))

# ── Paths ────────────────────────────────────────────────────────────────────
AUDIO_OUTPUT_DIR = "./audio_output"
MIN_TEXT_LENGTH = 50

MODEL_DIR = str(Path(__file__).resolve().parent.parent / "models")

# Kokoro
KOKORO_MODEL_DIR = os.path.join(MODEL_DIR, "kokoro")
KOKORO_MODEL = os.path.join(KOKORO_MODEL_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(KOKORO_MODEL_DIR, "voices-v1.0.bin")

# Wav2Vec2
WAV2VEC2_MODEL_DIR = os.path.join(MODEL_DIR, "wav2vec2")
WAV2VEC2_MODEL = os.path.join(WAV2VEC2_MODEL_DIR, "model.onnx")
WAV2VEC2_PROCESSOR_DIR = os.path.join(WAV2VEC2_MODEL_DIR, "processor")

# ffmpeg
TOOLFORGE_FFMPEG = "/data/project/wiki-tts/bin/ffmpeg"
LOCAL_FFMPEG = "ffmpeg"
FFMPEG_PATH = TOOLFORGE_FFMPEG if IS_TOOLFORGE else LOCAL_FFMPEG

# ── NeMo Text Processing ────────────────────────────────────────────────────
NEMO_WHITELIST = str(Path(__file__).resolve().parent / "nemo_whitelist.tsv")
TOOLFORGE_NEMO_CACHE = "/data/project/wiki-tts/nemo_cache"
LOCAL_NEMO_CACHE = os.path.join(tempfile.gettempdir(), "wiki-tts-nemo-grammars")
NEMO_GRAMMAR_CACHE = TOOLFORGE_NEMO_CACHE if IS_TOOLFORGE else LOCAL_NEMO_CACHE
