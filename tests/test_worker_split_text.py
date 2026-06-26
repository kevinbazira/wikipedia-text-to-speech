"""Unit tests for worker text chunking without loading model dependencies."""

import sys
import types


def _install_worker_dependency_stubs():
    numpy = types.SimpleNamespace(
        ndarray=object,
        zeros=lambda *args, **kwargs: None,
        linspace=lambda *args, **kwargs: None,
        float32=object,
    )
    sys.modules["numpy"] = numpy

    onnxruntime = types.SimpleNamespace(
        InferenceSession=type("InferenceSession", (), {"__init__": lambda self, *args, **kwargs: None}),
        SessionOptions=lambda: types.SimpleNamespace(),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=object()),
    )
    sys.modules["onnxruntime"] = onnxruntime

    celery_module = types.ModuleType("celery")

    class DummyCelery:
        def __init__(self, *args, **kwargs):
            self.conf = types.SimpleNamespace(update=lambda **kwargs: None)

        def task(self, func):
            return func

    celery_module.Celery = DummyCelery
    sys.modules["celery"] = celery_module

    celery_signals = types.ModuleType("celery.signals")
    celery_signals.worker_process_init = types.SimpleNamespace(connect=lambda func: func)
    sys.modules["celery.signals"] = celery_signals

    sys.modules.setdefault("redis", types.SimpleNamespace(from_url=lambda url: None))

    transformers = types.ModuleType("transformers")
    transformers.Wav2Vec2Processor = object
    sys.modules.setdefault("transformers", transformers)
    scipy = types.SimpleNamespace(signal=types.SimpleNamespace(resample=lambda audio, n: audio))
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", sys.modules["scipy"].signal)


_install_worker_dependency_stubs()

from wiki_tts.worker import _split_text  # noqa: E402


def test_split_text_returns_short_text_as_single_chunk():
    assert _split_text("Short text.", max_chars=50) == ["Short text."]


def test_split_text_prefers_sentence_boundaries():
    chunks = _split_text("First sentence. Second sentence. Third sentence.", max_chars=32)

    assert chunks == ["First sentence. Second sentence.", "Third sentence."]
    assert all(len(chunk) <= 32 for chunk in chunks)


def test_split_text_splits_long_sentence_on_word_boundaries():
    chunks = _split_text("alpha beta gamma delta", max_chars=10)

    assert chunks == ["alpha beta", "gamma", "delta"]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_text_ignores_empty_sentence_fragments():
    chunks = _split_text("One.   Two?\n\nThree!", max_chars=10)

    assert chunks == ["One. Two?", "Three!"]
