"""Word-level timestamp generation via CTC forced alignment.

Uses Wav2Vec2-CTC (ONNX) to align generated audio with the source text,
producing per-word start/end millisecond boundaries suitable for
word-level "follow along" highlighting (WebVTT or JSON).

The aligner is loaded lazily (same pattern as NeMo in text.py) and is
wrapped in try/except at the callsite so that alignment failures never
block audio generation.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.signal
from transformers import Wav2Vec2Processor

logger = logging.getLogger(__name__)

ALIGNER_SR = 16000  # Wav2Vec2 expects 16 kHz input
FRAME_DURATION_MS = 20  # Wav2Vec2 frame stride at 16 kHz (20 ms per frame)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "wav2vec2"

_aligner_session = None
_aligner_processor = None


# ── Initialisation (lazy, called once per worker) ──────────────────────────


def init_aligner() -> None:
    """Lazy-load the Wav2Vec2-CTC ONNX model and processor from local paths.

    Both the ONNX model and processor files are pre-downloaded to
    ``models/wav2vec2/`` (see setup instructions in README).
    Safe to call multiple times — only loads on the first invocation.
    """
    global _aligner_session, _aligner_processor
    if _aligner_session is not None:
        return

    model_path = MODEL_DIR / "model.onnx"
    processor_path = MODEL_DIR / "processor"
    if not model_path.exists() or not processor_path.exists():
        logger.warning(
            "Wav2Vec2 model or processor not found at %s; word timestamps unavailable",
            MODEL_DIR,
        )
        return

    logger.info("Initialising Wav2Vec2-CTC aligner...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = False
    _aligner_session = ort.InferenceSession(str(model_path), sess_options)
    _aligner_processor = Wav2Vec2Processor.from_pretrained(str(processor_path))
    logger.info("Wav2Vec2-CTC aligner ready.")


# ── Resampling (24 kHz → 16 kHz) ────────────────────────────────────────────


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """FFT-based resample to avoid aliasing artifacts."""
    num_samples = int(round(len(audio) * target_sr / orig_sr))
    return scipy.signal.resample(audio, num_samples)


# ── CTC forced alignment (pure numpy) ────────────────────────────────────────


def _ctc_word_alignment(
    logits: np.ndarray,
    text: str,
    processor: Wav2Vec2Processor,
) -> list[dict]:
    """Run CTC forced alignment against the known text.

    The algorithm:
      1. Argmax per frame → most likely character.
      2. Collapse consecutive duplicates, remove blank tokens.
      3. Group consecutive character segments into words using character
         counts from the known text.

    Returns a list of ``{"word": str, "start_ms": float, "end_ms": float}``
    ordered left to right, or an empty list if alignment fails.
    """
    blank_id = processor.tokenizer.pad_token_id
    vocab = processor.tokenizer.get_vocab()
    id2char = {v: k for k, v in vocab.items()}

    # ── 1. Argmax per frame ────────────────────────────────────────────
    ids = np.argmax(logits[0], axis=-1)

    # ── 2. Collapse duplicates, remove blanks, track frame ranges ──────
    # Each segment: (character_id, start_frame, end_frame)
    segments: list[tuple[int, int, int]] = []
    prev = blank_id
    seg_start: int | None = None

    for t, cid in enumerate(ids):
        if cid == blank_id:
            if prev != blank_id and seg_start is not None:
                segments.append((prev, seg_start, t))
                seg_start = None
        else:
            if cid != prev:
                if prev != blank_id and seg_start is not None:
                    segments.append((prev, seg_start, t))
                seg_start = t
        prev = cid

    if prev != blank_id and seg_start is not None:
        segments.append((prev, seg_start, len(ids)))

    if not segments:
        return []

    # ── 3. Match recognised segments to known words ────────────────────
    recognised = "".join(id2char.get(s[0], "") for s in segments).upper()

    # Build clean reference (uppercase alphanumeric only)
    words = text.split()
    clean_words: list[str] = []
    for w in words:
        cleaned = "".join(c for c in w if c.isalnum())
        if cleaned:
            clean_words.append(cleaned.upper())

    clean_ref = "".join(clean_words).upper()

    # Quick sanity check: if the recognised text is far from the reference,
    # fall back to proportional time distribution per word.
    if len(recognised) < len(clean_ref) * 0.5:
        logger.warning(
            "CTC alignment too short (%d chars vs %d expected); falling back to proportional timing",
            len(recognised),
            len(clean_ref),
        )
        return _proportional_timestamps(text, len(ids) * FRAME_DURATION_MS)

    # Align recognised characters to reference characters using
    # edit-distance-based alignment (character-level).
    alignment = _character_align(recognised, clean_ref)

    # Walk the alignment and assign each segment to a word.
    word_timestamps: list[dict] = []
    word_idx = 0  # index into clean_words
    char_in_word = 0  # character position within current word
    word_start_global: int | None = None

    for seg_idx, ref_char_idx in enumerate(segments_aligned(segments, alignment)):
        if ref_char_idx is None:
            continue  # deletion in recognised text — skip

        if word_idx >= len(clean_words):
            break

        word_len = len(clean_words[word_idx])

        if char_in_word == 0:
            word_start_global = segments[seg_idx][1]

        char_in_word += 1

        if char_in_word >= word_len:
            # End of current word
            word_end_frame = segments[seg_idx][2]
            word_timestamps.append(
                {
                    "word": words[word_idx],  # original casing + punctuation
                    "start_ms": word_start_global * FRAME_DURATION_MS,
                    "end_ms": word_end_frame * FRAME_DURATION_MS,
                }
            )
            word_idx += 1
            char_in_word = 0
            word_start_global = None

    return word_timestamps


def _character_align(recognised: str, reference: str) -> list[int | None]:
    """Align recognised characters to reference characters.

    Uses a simple edit-distance-based (Needleman-Wunsch) alignment and
    returns a list where each entry is:
      - an index into ``reference`` if the recognised character aligns,
      - ``None`` for insertions (extra recognised character).

    The reference index is non-decreasing (monotonic).
    """
    m, n = len(recognised), len(reference)

    if m == 0 or n == 0:
        return []

    # DP: score[i][j] = edit distance between recognised[:i] and reference[:j]
    score = np.zeros((m + 1, n + 1), dtype=np.int32)
    score[0, :] = np.arange(n + 1)
    score[:, 0] = np.arange(m + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if recognised[i - 1] == reference[j - 1] else 1
            score[i, j] = min(
                score[i - 1, j] + 1,  # deletion in reference
                score[i, j - 1] + 1,  # insertion in reference
                score[i - 1, j - 1] + cost,  # match / substitution
            )

    # Backtrack
    alignment: list[int | None] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i, j] == score[i - 1, j - 1] + (0 if recognised[i - 1] == reference[j - 1] else 1):
            alignment.append(j - 1 if recognised[i - 1] == reference[j - 1] else None)
            i -= 1
            j -= 1
        elif i > 0 and score[i, j] == score[i - 1, j] + 1:
            alignment.append(None)  # insertion in recognised
            i -= 1
        else:
            # This case handles j > 0 (insertion in reference) — skip
            j -= 1

    alignment.reverse()
    return alignment


def segments_aligned(
    segments: list[tuple[int, int, int]],
    alignment: list[int | None],
) -> list[int | None]:
    """Map each segment to its reference character index via the alignment.

    This is a simple 1:1 mapping — the i-th segment corresponds to the
    i-th recognised character, so we just return the i-th alignment entry.
    """
    result: list[int | None] = []
    for i in range(min(len(segments), len(alignment))):
        result.append(alignment[i])
    # If there are more segments than alignment entries, tag the rest as None
    for _ in range(len(segments) - len(alignment)):
        result.append(None)
    return result


def _proportional_timestamps(text: str, total_duration_ms: float) -> list[dict]:
    """Fallback: distribute time proportionally across words.

    Used when CTC alignment quality is too low for reliable word boundaries.
    """
    words = text.split()
    if not words:
        return []

    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    timestamps: list[dict] = []
    current_ms = 0.0
    for word in words:
        word_duration = (len(word) / total_chars) * total_duration_ms
        timestamps.append(
            {
                "word": word,
                "start_ms": current_ms,
                "end_ms": current_ms + word_duration,
            }
        )
        current_ms += word_duration
    return timestamps


# ── Public API ───────────────────────────────────────────────────────────────


def align_words(audio: np.ndarray, sample_rate: int, text: str) -> list[dict]:
    """Full pipeline: resample → Wav2Vec2 inference → CTC decode → word timestamps.

    Returns a list of ``{"word": str, "start_ms": float, "end_ms": float}``,
    or an empty list if the aligner was not initialised.
    """
    if _aligner_session is None or _aligner_processor is None:
        return []

    # Resample from model sample rate to 16 kHz
    audio_16k = resample(audio, sample_rate, ALIGNER_SR)

    inputs = _aligner_processor(
        audio_16k,
        sampling_rate=ALIGNER_SR,
        return_tensors="np",
        padding=True,
    )

    logits = _aligner_session.run(None, {"input_values": inputs.input_values})[0]

    return _ctc_word_alignment(logits, text, _aligner_processor)


# ── Output formatting ────────────────────────────────────────────────────────


def timestamps_to_vtt(timestamps: list[dict]) -> str:
    """Format word timestamps as WebVTT."""
    lines = ["WEBVTT\n"]
    for t in timestamps:
        start = _ms_to_vtt(t["start_ms"])
        end = _ms_to_vtt(t["end_ms"])
        lines.append(f"{start} --> {end}\n{t['word']}\n")
    return "\n".join(lines)


def timestamps_to_json(timestamps: list[dict]) -> str:
    """Format word timestamps as a JSON array."""
    import json

    return json.dumps(timestamps, indent=2)


def _ms_to_vtt(ms: float) -> str:
    """Convert milliseconds to WebVTT timestamp (``HH:MM:SS.mmm``)."""
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
