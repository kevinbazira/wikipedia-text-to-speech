# wiki-tts KServe model-server

Custom [KServe](https://kserve.github.io/website/) model-server for Wikipedia TTS inference. Wraps the Kokoro ONNX text-to-speech model and Wav2Vec2-CTC forced aligner behind a single `predict()` call that takes pre-normalized text segments and returns concatenated float32 PCM audio with word-level timestamps.

## Directory structure

```
kserve/
├── server.py           # KServe Model subclass + entrypoint
├── inference.py        # TTS + alignment + crossfade pipeline
├── alignment.py        # Wav2Vec2-CTC forced alignment
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build
├── deploy.yaml         # KServe InferenceService manifest
└── README.md
```

## Prerequisites

Download the ONNX models into `models/` at the project root before building or running:

```bash
# From the project root
python3 scripts/download_models.py
```

This populates:

```
models/
├── kokoro/
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
└── wav2vec2/
    ├── model.onnx
    └── processor/
        ├── config.json
        ├── preprocessor_config.json
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── vocab.json
```

## Build the container

```bash
docker build -t wiki-tts-kserve kserve/
```

## Run locally (Docker)

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/mnt/models:ro" \
  wiki-tts-kserve
```

Wait for the startup log line `Model loaded successfully!`, then test:

```bash
# Health check
curl -s http://localhost:8080/health | python3 -m json.tool

# Model metadata (also the KServe readiness probe)
curl -s http://localhost:8080/v1/models/wiki-tts | python3 -m json.tool

# TTS + alignment inference
curl -s -X POST http://localhost:8080/v1/models/wiki-tts:predict \
  -H 'Content-Type: application/json' \
  -d '{
    "segments": [
      {"text": "Hello world.", "voice": "af_heart"}
    ]
  }' | python3 -m json.tool
```

## Run locally (without Docker)

```bash
cd kserve
pip install -r requirements.txt
MODEL_DIR="$(pwd)/../models" python server.py
```

## Run the test suite

The unit tests in `tests/` cover the text normalization, timestamp formatting, and worker chunking logic which are shared between v0 and v1. Alignment-specific tests can be run by importing the KServe aligner directly:

```bash
# From the project root, run all existing tests
python3 -m pytest tests/

# Test the KServe alignment module in isolation (requires models downloaded)
cd kserve
MODEL_DIR="$(pwd)/../models" python3 -c "
from alignment import Aligner
import numpy as np
a = Aligner('$MODEL_DIR/wav2vec2')
t = a.align(np.zeros(16000, dtype=np.float32), 16000, 'test')
print('Aligner loaded OK, alignment result:', t)
"
```

## API contract

### `POST /v1/models/wiki-tts:predict`

**Request:**

```json
{
  "segments": [
    {
      "text": "Earth is the third planet from the Sun.",
      "voice": "af_heart",
      "speed": 1.0,
      "lang": "en-us"
    },
    {
      "text": "It orbits at a distance of 150 million km."
    }
  ],
  "default_voice": "af_heart",
  "default_speed": 1.0,
  "default_lang": "en-us"
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `segments` | Yes | — | List of text segments to synthesize (ordered) |
| `segments[].text` | Yes | — | Pre-normalized text for TTS |
| `segments[].voice` | No | `"af_heart"` | Kokoro voice profile |
| `segments[].speed` | No | `1.0` | Speaking rate |
| `segments[].lang` | No | `"en-us"` | Language code |
| `default_voice` | No | `"af_heart"` | Fallback voice for segments that don't specify one |
| `default_speed` | No | `1.0` | Fallback speaking rate |
| `default_lang` | No | `"en-us"` | Fallback language code |

**Segment size:** Each segment's `text` should be pre-chunked (e.g. via the orchestrator's `_split_text`) to stay under ~800 characters — Kokoro's practical input-length limit. Longer text may be silently truncated by the model. The model-server logs a warning for segments exceeding this threshold but does not reject them.

**Response size:** The response contains base64-encoded float32 PCM audio (24 kHz mono). A typical 10-second section is ~2.5 MB encoded. Callers should send one section per request (heading segment + a few content chunks) rather than batching an entire article. Sending dozens of sections in a single request will produce a multi-hundred-MB response.

**Response:**

```json
{
  "audio_b64": "<base64-encoded float32 PCM, 24 kHz mono>",
  "sample_rate": 24000,
  "duration_ms": 8250.3,
  "timestamps": [
    {"word": "Earth", "start_ms": 0.0, "end_ms": 380.0},
    {"word": "is", "start_ms": 380.0, "end_ms": 490.0}
  ]
}
```

| Field | Description |
|---|---|
| `audio_b64` | Base64-encoded float32 PCM audio (24 kHz, mono, little-endian) |
| `sample_rate` | Audio sample rate in Hz (always 24000) |
| `duration_ms` | Total audio duration in milliseconds |
| `timestamps` | Word-level start/end times in milliseconds, accumulated across segments |

### `GET /v1/models/wiki-tts`

Returns model metadata. Used by KServe as the readiness probe — returns 200 once `load()` completes.

### `GET /health`

Liveness probe — returns 200 if the process is alive.

## Deploy to LiftWing

1. Push the container to your registry:

   ```bash
   docker tag wiki-tts-kserve <registry>/wiki-tts-kserve:v1.0
   docker push <registry>/wiki-tts-kserve:v1.0
   ```

2. Edit `deploy.yaml` — replace `<registry>/wiki-tts-kserve:latest` with your image.

3. Ensure models are available at `/mnt/models/` in the KServe model storage (PVC or S3-compatible bucket).

4. Apply:

   ```bash
   kubectl apply -f kserve/deploy.yaml
   ```

5. Verify:

   ```bash
   kubectl get inferenceservices wiki-tts
   kubectl get pods -l app=wiki-tts
   ```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `wiki-tts` | Model name registered with KServe |
| `MODEL_DIR` | `/mnt/models` | Root directory for model files |
| `KOKORO_MODEL` | `$MODEL_DIR/kokoro/kokoro-v1.0.onnx` | Path to Kokoro ONNX model |
| `KOKORO_VOICES` | `$MODEL_DIR/kokoro/voices-v1.0.bin` | Path to Kokoro voice pack |
| `WAV2VEC2_MODEL_DIR` | `$MODEL_DIR/wav2vec2` | Directory containing Wav2Vec2 ONNX model + processor |
| `ORT_NUM_THREADS` | `1` | ONNX Runtime intra-op thread count |
| `OMP_NUM_THREADS` | `1` | OpenMP thread count |
