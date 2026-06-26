"""Unit tests for pure timestamp formatting and fallback allocation."""

import sys
import types


def _install_timestamp_dependency_stubs():
    sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))
    sys.modules.setdefault("onnxruntime", types.SimpleNamespace())
    scipy = types.SimpleNamespace(signal=types.SimpleNamespace(resample=lambda audio, n: audio))
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", sys.modules["scipy"].signal)

    transformers = types.ModuleType("transformers")
    transformers.Wav2Vec2Processor = object
    sys.modules.setdefault("transformers", transformers)


_install_timestamp_dependency_stubs()

from wiki_tts.timestamps import (  # noqa: E402
    _ms_to_vtt,
    _proportional_timestamps,
    timestamps_to_json,
    timestamps_to_vtt,
)


def test_ms_to_vtt_formats_hours_minutes_seconds_and_milliseconds():
    assert _ms_to_vtt(3_723_456) == "01:02:03.456"


def test_proportional_timestamps_distributes_duration_by_word_length():
    timestamps = _proportional_timestamps("a bbb", 800)

    assert timestamps == [
        {"word": "a", "start_ms": 0.0, "end_ms": 200.0},
        {"word": "bbb", "start_ms": 200.0, "end_ms": 800.0},
    ]


def test_proportional_timestamps_handles_blank_input():
    assert _proportional_timestamps("", 1000) == []
    assert _proportional_timestamps("   ", 1000) == []


def test_timestamps_to_vtt_emits_webvtt_cues():
    result = timestamps_to_vtt([{"word": "Hello", "start_ms": 0, "end_ms": 1250}])

    assert result == "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello\n"


def test_timestamps_to_json_pretty_prints_timestamp_data():
    result = timestamps_to_json([{"word": "Hi", "start_ms": 0, "end_ms": 20}])

    assert '"word": "Hi"' in result
    assert result.startswith("[\n")
