#!/usr/bin/env python3
"""Download all model files required by the TTS prototype.

Downloads Kokoro TTS and Wav2Vec2-CTC alignment models to the
directory structure expected by ``wiki_tts/config.py``::

    models/
    ├── kokoro/
    │   ├── kokoro-v1.0.onnx
    │   └── voices-v1.0.bin
    └── wav2vec2/
        ├── model.onnx
        └── processor/
"""

import os
import sys
import urllib.request
from pathlib import Path

# Allow project imports regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import Wav2Vec2Processor

from wiki_tts.config import (
    KOKORO_MODEL,
    KOKORO_MODEL_DIR,
    KOKORO_VOICES,
    WAV2VEC2_MODEL,
    WAV2VEC2_MODEL_DIR,
    WAV2VEC2_PROCESSOR_DIR,
)

_KOKORO_URLS = [
    (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        KOKORO_MODEL,
    ),
    (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        KOKORO_VOICES,
    ),
]

_WAV2VEC2_MODEL_URL = (
    "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/onnx/model_quantized.onnx"
)


def _download(url: str, dest: str) -> None:
    """Download *url* to *dest*, skipping if the file already exists."""
    if os.path.exists(dest):
        print(f"  SKIP  {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  FETCH {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    print(f"  SAVE  {dest}")


def download_kokoro() -> None:
    """Download Kokoro ONNX model and voice pack."""
    print(f"[kokoro] -> {KOKORO_MODEL_DIR}")
    for url, dest in _KOKORO_URLS:
        _download(url, dest)


def download_wav2vec2() -> None:
    """Download Wav2Vec2-CTC ONNX model and processor files."""
    print(f"[wav2vec2] -> {WAV2VEC2_MODEL_DIR}")
    _download(_WAV2VEC2_MODEL_URL, WAV2VEC2_MODEL)

    if os.path.exists(WAV2VEC2_PROCESSOR_DIR):
        print(f"  SKIP  {WAV2VEC2_PROCESSOR_DIR}/")
    else:
        print("  FETCH Wav2Vec2 processor from HuggingFace hub")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        processor.save_pretrained(WAV2VEC2_PROCESSOR_DIR)
        print(f"  SAVE  {WAV2VEC2_PROCESSOR_DIR}/")


if __name__ == "__main__":
    download_kokoro()
    print()
    download_wav2vec2()
    print()
    print("All models downloaded.")
