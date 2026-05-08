"""
TTS backend benchmark: Kokoro-ONNX (INT8 & FP32) vs Kokoro KPipeline (PyTorch).

Measures time-per-character and real-time factor (RTF) on identical input,
so we can identify which backend is fastest on this hardware.

Usage (single-request latency):
    OMP_NUM_THREADS=1 ORT_NUM_THREADS=<N> python3 benchmark_tts.py

Usage (concurrent — simulates N users submitting at once):
    OMP_NUM_THREADS=1 ORT_NUM_THREADS=<N> python3 benchmark_tts.py --concurrent <N>
"""

import os
import time
import subprocess
import numpy as np

# Apply the same ONNX thread monkeypatch as the workers so the benchmark
# reflects real deployment performance, not thread-thrashing artefacts.
# Set ORT_NUM_THREADS to control (default 4 is a safe balance).
import onnxruntime as ort

_NUM_THREADS = int(os.environ.get("ORT_NUM_THREADS", "4"))
_original_init = ort.InferenceSession.__init__


def _patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = _NUM_THREADS
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)


ort.InferenceSession.__init__ = _patched_init

BENCHMARK_TEXT = (
    "Earth is the third planet from the Sun and the only astronomical object "
    "known to harbor life. This is enabled by Earth being an ocean world, the "
    "only one in the Solar System sustaining liquid surface water. Almost all "
    "of Earth's water is contained in its global ocean, covering 70.8% of "
    "Earth's crust. The remaining 29.2% of Earth's crust is land, most of which "
    "is located in the form of continental landmasses within Earth's land hemisphere. "
    "Most of Earth's land is somewhat humid and covered by vegetation, while large "
    "sheets of ice at Earth's polar deserts retain more water than Earth's groundwater, "
    "lakes, rivers and atmospheric water combined. Earth's crust consists of slowly "
    "moving tectonic plates, which interact to produce mountain ranges, volcanoes, "
    "and earthquakes. Earth's liquid outer core generates the magnetic field that "
    "shapes the magnetosphere of Earth, deflecting destructive solar winds."
)

SHORT_TEXT = "Wikipedia is a free online encyclopedia, created and edited by volunteers."


def bench_onnx(model_path: str, label: str):
    """Benchmark kokoro-onnx with the given model file."""
    print(f"\n{'='*60}")
    print(f"  kokoro-onnx ({label})")
    print(f"{'='*60}")

    try:
        from kokoro_onnx import Kokoro
        try:
            from kokoro_onnx import __version__ as kov
        except ImportError:
            kov = "unknown"
    except ImportError:
        print("  SKIP — kokoro-onnx not installed")
        return None

    print(f"  Library version: {kov}")
    print(f"  Model: {model_path}")

    # Load model
    t0 = time.time()
    kokoro = Kokoro(model_path, "voices-v1.0.bin")
    load_time = time.time() - t0
    print(f"  Model load: {load_time:.1f}s")

    # Warmup
    _ = kokoro.create("Warmup.", voice="af_heart", lang="en-us")

    # Benchmark short text
    print(f"\n  --- Short text ({len(SHORT_TEXT)} chars) ---")
    t0 = time.time()
    audio, sr = kokoro.create(SHORT_TEXT, voice="af_heart", lang="en-us")
    elapsed = time.time() - t0
    audio_duration = len(audio) / sr
    print(f"  Time: {elapsed:.2f}s, Audio: {audio_duration:.1f}s, RTF: {elapsed/audio_duration:.2f}x")
    print(f"  s/char: {elapsed/len(SHORT_TEXT):.4f}")

    # Benchmark long text
    print(f"\n  --- Long text ({len(BENCHMARK_TEXT)} chars) ---")
    t0 = time.time()
    audio, sr = kokoro.create(BENCHMARK_TEXT, voice="af_heart", lang="en-us")
    elapsed = time.time() - t0
    audio_duration = len(audio) / sr
    print(f"  Time: {elapsed:.2f}s, Audio: {audio_duration:.1f}s, RTF: {elapsed/audio_duration:.2f}x")
    print(f"  s/char: {elapsed/len(BENCHMARK_TEXT):.4f}")

    return {
        "backend": f"onnx-{label}",
        "long_s_per_char": elapsed / len(BENCHMARK_TEXT),
        "long_rtf": elapsed / audio_duration,
    }


def bench_kpipeline():
    """Benchmark original Kokoro KPipeline (PyTorch)."""
    print(f"\n{'='*60}")
    print(f"  Kokoro KPipeline (PyTorch)")
    print(f"{'='*60}")

    try:
        from kokoro import KPipeline
    except ImportError:
        print("  SKIP — kokoro (PyTorch) not installed")
        return None

    import torch
    print(f"  PyTorch version: {torch.__version__}")

    # Load pipeline
    t0 = time.time()
    pipeline = KPipeline(lang_code='a')
    load_time = time.time() - t0
    print(f"  Model load: {load_time:.1f}s")

    # Warmup
    for _ in pipeline("Warmup.", voice='af_heart', speed=1.0):
        pass

    # Benchmark short text
    print(f"\n  --- Short text ({len(SHORT_TEXT)} chars) ---")
    t0 = time.time()
    audio_parts = []
    sr = 24000
    for _, _, audio in pipeline(SHORT_TEXT, voice='af_heart', speed=1.0):
        audio_parts.append(audio.numpy() if hasattr(audio, 'numpy') else audio)
    elapsed = time.time() - t0
    full_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
    audio_duration = len(full_audio) / sr
    print(f"  Time: {elapsed:.2f}s, Audio: {audio_duration:.1f}s, RTF: {elapsed/audio_duration:.2f}x")
    print(f"  s/char: {elapsed/len(SHORT_TEXT):.4f}")

    # Benchmark long text
    print(f"\n  --- Long text ({len(BENCHMARK_TEXT)} chars) ---")
    t0 = time.time()
    audio_parts = []
    for _, _, audio in pipeline(BENCHMARK_TEXT, voice='af_heart', speed=1.0):
        audio_parts.append(audio.numpy() if hasattr(audio, 'numpy') else audio)
    elapsed = time.time() - t0
    full_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
    audio_duration = len(full_audio) / sr
    print(f"  Time: {elapsed:.2f}s, Audio: {audio_duration:.1f}s, RTF: {elapsed/audio_duration:.2f}x")
    print(f"  s/char: {elapsed/len(BENCHMARK_TEXT):.4f}")

    return {
        "backend": "kpipeline",
        "long_s_per_char": elapsed / len(BENCHMARK_TEXT),
        "long_rtf": elapsed / audio_duration,
    }


def check_cpu():
    """Print CPU features relevant to ONNX performance."""
    print(f"\n{'='*60}")
    print(f"  System info")
    print(f"{'='*60}")

    import os
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        print(f"  CPU: {info.get('brand_raw', 'unknown')}")
        print(f"  Arch: {info.get('arch', 'unknown')}")
        flags = info.get('flags', [])
        if 'avx512_vnni' in flags:
            print(f"  INT8 acceleration: AVX-512 VNNI ✓")
        elif 'avx2' in flags:
            print(f"  INT8 acceleration: AVX2 (no VNNI) — INT8 model may be slower")
        else:
            print(f"  INT8 acceleration: None — INT8 model likely slower")
        print(f"  Cores: {os.cpu_count()}")
    except ImportError:
        print(f"  Install py-cpuinfo for CPU details: pip install py-cpuinfo")
        print(f"  Cores: {os.cpu_count()}")

    import onnxruntime as ort
    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")


if __name__ == "__main__":
    check_cpu()

    results = []
    if (r := bench_onnx("kokoro-v1.0.onnx", "FP32")):
        results.append(r)
    if (r := bench_onnx("kokoro-v1.0.int8.onnx", "INT8")):
        results.append(r)
    if (r := bench_kpipeline()):
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (long text)")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['backend']:20s}  {r['long_s_per_char']:.4f} s/char  RTF={r['long_rtf']:.2f}x")
