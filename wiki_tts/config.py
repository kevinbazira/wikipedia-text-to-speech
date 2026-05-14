import os

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

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

TOOLFORGE_FFMPEG = "/data/project/wiki-tts/bin/ffmpeg"
SYSTEM_FFMPEG = "ffmpeg"
FFMPEG_PATH = TOOLFORGE_FFMPEG if IS_TOOLFORGE else SYSTEM_FFMPEG
