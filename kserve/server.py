import base64
import logging
import os

import numpy as np
from inference import TTSInferencePipeline

import kserve
from kserve.errors import InferenceError, InvalidInput, ModelMissingError

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)
logger = logging.getLogger(__name__)


class WikipediaTTSModel(kserve.Model):
    def __init__(
        self,
        name: str,
        kokoro_model: str,
        kokoro_voices: str,
        wav2vec2_model_dir: str,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.kokoro_model = kokoro_model
        self.kokoro_voices = kokoro_voices
        self.wav2vec2_model_dir = wav2vec2_model_dir
        self.pipeline: TTSInferencePipeline | None = None
        self.ready = False

    def load(self) -> None:
        """Load Kokoro ONNX and Wav2Vec2 ONNX models into memory."""
        try:
            logger.info("Loading inference pipeline (Kokoro + Wav2Vec2)...")
            self.pipeline = TTSInferencePipeline(
                self.kokoro_model,
                self.kokoro_voices,
                self.wav2vec2_model_dir,
            )
            self.ready = True
            logger.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load TTS models. Reason: {e}"
            logger.critical(error_message)
            raise ModelMissingError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        """Validate input and extract segment list with defaults."""
        segments = payload.get("segments")
        if not segments:
            error_message = "`segments` is required and must be a non-empty list."
            logger.error(error_message)
            raise InvalidInput(error_message)

        if not isinstance(segments, list):
            raise InvalidInput("`segments` must be a list.")

        for i, seg in enumerate(segments):
            if "text" not in seg:
                raise InvalidInput(f"Segment {i} is missing `text`.")
            if not isinstance(seg["text"], str) or not seg["text"].strip():
                raise InvalidInput(f"Segment {i} has empty or non-string `text`.")
            # Kokoro has a practical input-length limit (~800 chars).  The
            # orchestrator is expected to pre-chunk via _split_text().  This
            # is a non-fatal warning — the model may silently truncate.
            if len(seg["text"]) > 800:
                logger.warning(
                    "Segment %d is %d chars (max recommended: 800); "
                    "text may be truncated by Kokoro.  Pre-chunk via _split_text().",
                    i,
                    len(seg["text"]),
                )
            # MIN_TEXT_LENGTH filtering (v0's 50-char gate) is the
            # orchestrator's responsibility — it owns text cleaning.

        return {
            "segments": segments,
            "default_voice": payload.get("default_voice", "af_heart"),
            "default_speed": float(payload.get("default_speed", 1.0)),
            "default_lang": payload.get("default_lang", "en-us"),
        }

    def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """Run TTS inference + forced alignment on the preprocessed segments."""
        try:
            logger.info(
                "Running inference on %d segments (voice=%s)...",
                len(inputs["segments"]),
                inputs["default_voice"],
            )
            result = self.pipeline.predict(
                segments=inputs["segments"],
                default_voice=inputs["default_voice"],
                default_speed=inputs["default_speed"],
                default_lang=inputs["default_lang"],
            )
            return result

        except Exception as e:
            error_message = f"Error during TTS inference: {e}"
            logger.error(error_message)
            raise InferenceError(error_message)

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """Encode float32 PCM audio as base64 and attach metadata."""
        try:
            audio: np.ndarray = inputs["audio"]
            sample_rate: int = inputs["sample_rate"]
            timestamps: list[dict] = inputs["timestamps"]

            audio_bytes = audio.tobytes()
            duration_ms = (len(audio) / sample_rate) * 1000

            return {
                "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                "sample_rate": sample_rate,
                "duration_ms": round(duration_ms, 1),
                "timestamps": timestamps,
            }

        except Exception as e:
            error_message = f"Error during post-processing: {e}"
            logger.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "wiki-tts")
    model_dir = os.environ.get("MODEL_DIR", "/mnt/models")

    kokoro_model = os.environ.get("KOKORO_MODEL", os.path.join(model_dir, "kokoro", "kokoro-v1.0.onnx"))
    kokoro_voices = os.environ.get("KOKORO_VOICES", os.path.join(model_dir, "kokoro", "voices-v1.0.bin"))
    wav2vec2_model_dir = os.environ.get("WAV2VEC2_MODEL_DIR", os.path.join(model_dir, "wav2vec2"))

    model = WikipediaTTSModel(
        name=model_name,
        kokoro_model=kokoro_model,
        kokoro_voices=kokoro_voices,
        wav2vec2_model_dir=wav2vec2_model_dir,
    )

    model.load()
    kserve.ModelServer().start([model])
