"""Kokoro TTS + Wav2Vec2 forced alignment inference pipeline.

This is the core inference engine extracted from ``wiki_tts/worker.py``.
It takes pre-normalized text segments, runs TTS and alignment on each, and
returns concatenated float32 PCM audio with accumulated word-level timestamps.
"""

import logging

import numpy as np
from alignment import Aligner

logger = logging.getLogger(__name__)

FADE_LEN = 120  # samples for crossfade envelope


class TTSInferencePipeline:
    """Thin wrapper around Kokoro ONNX + Wav2Vec2 CTC aligner.

    Models are loaded once at construction time (triggered by the KServe
    startup hook).  ``.predict()`` is called per inference request.
    """

    def __init__(self, kokoro_model: str, kokoro_voices: str, wav2vec2_model_dir: str):
        logger.info("Loading Kokoro ONNX model from %s...", kokoro_model)
        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(kokoro_model, kokoro_voices)
        self.sample_rate = 24000

        logger.info("Loading Wav2Vec2 aligner from %s...", wav2vec2_model_dir)
        self.aligner = Aligner(wav2vec2_model_dir)

    def predict(
        self,
        segments: list[dict],
        default_voice: str = "af_heart",
        default_speed: float = 1.0,
        default_lang: str = "en-us",
    ) -> dict:
        """Generate concatenated audio + word timestamps for a sequence of text segments.

        Segments are crossfaded together for natural-sounding transitions.
        Timestamps are accumulated so they refer to positions in the final
        concatenated audio.

        Args:
            segments: List of ``{"text": str, "voice"?: str, "speed"?: float, "lang"?: str}``.
            default_voice: Voice to use for segments that don't specify one.
            default_speed: Speaking rate (1.0 = normal).
            default_lang: Language code passed to Kokoro.

        Returns:
            ``{"audio": np.ndarray (float32), "sample_rate": int, "timestamps": list[dict]}``
        """
        audio_chunks: list[np.ndarray] = []
        all_timestamps: list[dict] = []
        current_time_ms = 0.0

        fade_out = np.linspace(1, 0, FADE_LEN, dtype=np.float32)
        fade_in = np.linspace(0, 1, FADE_LEN, dtype=np.float32)
        prev_tail: np.ndarray | None = None

        for i, seg in enumerate(segments):
            text = seg["text"]
            voice = seg.get("voice", default_voice)
            speed = seg.get("speed", default_speed)
            lang = seg.get("lang", default_lang)

            chunk_audio, _ = self.kokoro.create(text, voice=voice, speed=speed, lang=lang)

            # Word-level alignment
            chunk_ts = self.aligner.align(chunk_audio, self.sample_rate, text)
            for t in chunk_ts:
                t["start_ms"] += current_time_ms
                t["end_ms"] += current_time_ms
                all_timestamps.append(t)

            chunk_duration_ms = (len(chunk_audio) / self.sample_rate) * 1000
            current_time_ms += chunk_duration_ms

            # Crossfade between consecutive chunks
            if len(segments) == 1:
                audio_chunks.append(chunk_audio)
            elif i == 0:
                chunk_audio[-FADE_LEN:] *= fade_out
                prev_tail = chunk_audio[-FADE_LEN:].copy()
                audio_chunks.append(chunk_audio[:-FADE_LEN])
            elif i < len(segments) - 1:
                chunk_audio[:FADE_LEN] *= fade_in
                chunk_audio[:FADE_LEN] += prev_tail
                chunk_audio[-FADE_LEN:] *= fade_out
                prev_tail = chunk_audio[-FADE_LEN:].copy()
                audio_chunks.append(chunk_audio[:-FADE_LEN])
            else:
                chunk_audio[:FADE_LEN] *= fade_in
                chunk_audio[:FADE_LEN] += prev_tail
                audio_chunks.append(chunk_audio)

        audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)

        return {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "timestamps": all_timestamps,
        }
