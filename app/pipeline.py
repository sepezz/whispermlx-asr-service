"""
Shared ASR pipeline stage functions.

Extracts the 3-stage pipeline (transcribe -> align -> diarize) into
reusable functions consumed by the FastAPI endpoints.
Powered by whispermlx (MLX backend on Apple Silicon).
"""

import gc
import logging
import math
import os
import threading
import time
import warnings
from typing import Any

# Suppress pyannote's torchcodec warning -- we decode audio via whispermlx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np
import whispermlx
from whispermlx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "mps")
# COMPUTE_TYPE and BATCH_SIZE are accepted for API compatibility with the
# original CUDA-based service but are INERT under the MLX backend.  Setting
# them will not error, but they have no effect on inference behaviour.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", os.path.expanduser("~/.cache/whisperx-asr"))
DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")

# Idle model eviction. Set MODEL_KEEP_ALIVE_SECONDS > 0 to unload Whisper
# models that have not been used in that many seconds. Floor of 30s on the
# sweep interval to avoid pegging a thread on tight loops.
MODEL_KEEP_ALIVE_SECONDS = int(os.getenv("MODEL_KEEP_ALIVE_SECONDS", "0"))
MODEL_EVICTION_INTERVAL_SECONDS = max(30, int(os.getenv("MODEL_EVICTION_INTERVAL_SECONDS", "60")))

# MLX model map: short names → HuggingFace repo IDs for the MLX backend.
# Sourced from whispermlx.asr.MLX_MODEL_MAP. Duplicated here so that
# get_canonical_models() and resolve_model_name() work without a
# faster-whisper dependency. Keep in sync with the upstream whispermlx package.
MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v1": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def get_canonical_models() -> list:
    """
    Canonical model names accepted by the whispermlx MLX backend.

    Sourced from the MLX model map keys.
    """
    return list(MLX_MODEL_MAP.keys())


# OpenAI-style aliases → canonical MLX model names. These are kept for
# backwards compatibility on the request path; new clients should use the
# canonical names returned by /v1/models.
_MODEL_ALIASES = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}


def resolve_model_name(model: str) -> str:
    """
    Resolve a user-supplied model identifier to a canonical MLX model name.

    Accepts canonical names (tiny, large-v3, ...) as-is and maps OpenAI-style
    aliases (whisper-tiny, whisper-large-v3, ...) to their canonical equivalents.
    Unknown values are returned unchanged so the engine can produce its own
    validation error.
    """
    if not model:
        return DEFAULT_MODEL
    canonical = set(get_canonical_models())
    if model in canonical:
        return model
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if model.startswith("whisper-"):
        stripped = model[len("whisper-") :]
        if stripped in canonical:
            return stripped
    return model


_model_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: dict[str, Any] = {}
_whisper_models_last_used: dict[str, float] = {}
_align_models: dict[str, tuple[Any, Any]] = {}
_diarize_pipeline: DiarizationPipeline | None = None

_eviction_thread_lock = threading.Lock()
_eviction_thread_started = False


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup.

    Release unused buffers from both MLX Whisper and PyTorch's supporting
    stages. Neither allocator can clear the other allocator's cache.
    """
    gc.collect()
    try:
        import mlx.core

        if hasattr(mlx.core, "clear_cache"):
            mlx.core.clear_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    logger.debug("GPU memory cache cleared")


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def load_whisper_model(model_name: str):
    """Load whispermlx model with caching (thread-safe)."""
    if model_name not in _whisper_models:
        with _model_load_lock:
            if model_name not in _whisper_models:
                logger.info(f"Loading whispermlx model: {model_name}")
                model = whispermlx.load_model(
                    model_name,
                    device=DEVICE,
                )
                _whisper_models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
                # Pre-register the eviction counter time series for this model
                # so the row appears in /metrics with value 0 from the moment
                # the model is loaded, instead of only after the first eviction.
                try:
                    from app import metrics as prom_metrics

                    prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=model_name)
                except Exception:
                    pass
    _whisper_models_last_used[model_name] = time.time()
    _ensure_eviction_thread()
    return _whisper_models[model_name]


def _ensure_eviction_thread():
    """Lazily start the idle-model eviction daemon (no-op if disabled)."""
    global _eviction_thread_started
    if MODEL_KEEP_ALIVE_SECONDS <= 0 or _eviction_thread_started:
        return
    with _eviction_thread_lock:
        if _eviction_thread_started:
            return
        t = threading.Thread(target=_eviction_loop, daemon=True, name="model-evictor")
        t.start()
        _eviction_thread_started = True
        logger.info(
            f"Idle model eviction enabled: unload after "
            f"{MODEL_KEEP_ALIVE_SECONDS}s idle, sweep every "
            f"{MODEL_EVICTION_INTERVAL_SECONDS}s"
        )


def _run_eviction_sweep() -> bool:
    """Run a single eviction sweep over the cached Whisper models.

    Evicts any model whose last-used timestamp is older than
    ``MODEL_KEEP_ALIVE_SECONDS``.  Returns ``True`` if at least one model
    was evicted (so the caller can decide whether to clear GPU memory).

    This is extracted from ``_eviction_loop`` so unit tests can exercise
    the real eviction code path without duplicating the sweep logic.
    """
    if MODEL_KEEP_ALIVE_SECONDS <= 0:
        return False
    now = time.time()
    candidates = [
        name
        for name, last in list(_whisper_models_last_used.items())
        if now - last > MODEL_KEEP_ALIVE_SECONDS and name in _whisper_models
    ]
    evicted_any = False
    for name in candidates:
        with _model_load_lock:
            last = _whisper_models_last_used.get(name, 0)
            if name in _whisper_models and now - last > MODEL_KEEP_ALIVE_SECONDS:
                logger.info(f"Evicting idle model {name}")
                del _whisper_models[name]
                _whisper_models_last_used.pop(name, None)
                evicted_any = True
                try:
                    from app import metrics as prom_metrics

                    prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()
                except Exception:
                    pass
    if evicted_any:
        clear_gpu_memory()
    return evicted_any


def _eviction_loop():
    while True:
        time.sleep(MODEL_EVICTION_INTERVAL_SECONDS)
        _run_eviction_sweep()


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whispermlx.load_align_model(
                    language_code=language_code,
                    device=DEVICE,
                    model_dir=CACHE_DIR,
                )
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    return _align_models[language_code]


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-community-1")
                _diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    token=HF_TOKEN,
                    device=DEVICE,
                )
                logger.info("Diarization pipeline loaded")
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    progress_callback=None,
) -> dict:
    """Run whispermlx transcription and return raw result dict.

    hotwords: accepted for API compatibility but IGNORED by the MLX backend.
              A warning is logged; no error is raised.
    initial_prompt: set per-request on the shared cached model (reset in finally).
    """
    clear_gpu_memory()
    whisper_model = load_whisper_model(model_name)

    # Hotwords is a no-op: the MLX backend has no hotwords mechanism.
    if hotwords is not None:
        logger.warning(
            "The MLX backend ignores hotwords; the parameter is accepted for "
            "API compatibility but has no effect on transcription."
        )

    # Set per-request initial_prompt on the shared cached model.
    # Must reset in finally to avoid leaking into subsequent requests.
    if initial_prompt is not None:
        whisper_model.initial_prompt = initial_prompt

    logger.info("Starting transcription...")
    try:
        kwargs = {"language": language, "task": task}
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        result = whisper_model.transcribe(audio, **kwargs)
    finally:
        # Always reset initial_prompt to avoid leaking to next request
        if initial_prompt is not None:
            whisper_model.initial_prompt = None
        clear_gpu_memory()

    detected_language = result.get("language", language or "en")
    logger.info(f"Transcription complete. Detected language: {detected_language}")

    return result


# ---------------------------------------------------------------------------
# Stage 2 -- Alignment
# ---------------------------------------------------------------------------
def align(audio: np.ndarray, result: dict, progress_callback=None, *, strict: bool = False) -> dict:
    """Run Wav2Vec2 alignment to get word-level timestamps."""
    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    clear_gpu_memory()
    try:
        model_a, metadata = load_align_model(detected_language)
        kwargs = {"return_char_alignments": False}
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        result = whispermlx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            **kwargs,
        )
        logger.info("Timestamp alignment complete")
    except Exception as e:
        if strict:
            raise
        logger.warning(f"Timestamp alignment failed: {e}, continuing without word-level timestamps")
    finally:
        clear_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# Stage 3 -- Diarization
# ---------------------------------------------------------------------------
def diarize(
    audio: np.ndarray,
    result: dict,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    return_speaker_embeddings: bool = False,
) -> tuple[dict, dict | None]:
    """
    Run pyannote speaker diarization and assign speakers to segments.

    Returns (result_with_speakers, speaker_embeddings_or_None).
    """
    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

    logger.info("Starting speaker diarization...")
    speaker_embeddings = None
    try:
        diarize_model = load_diarize_pipeline()

        diarize_params: dict[str, Any] = {}
        if num_speakers is not None:
            diarize_params["num_speakers"] = num_speakers
            logger.info(f"Diarization with exact speaker count: {num_speakers}")
        else:
            if min_speakers is not None:
                diarize_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_params["max_speakers"] = max_speakers
            logger.info(f"Diarization with speaker range: {min_speakers}-{max_speakers}")

        if return_speaker_embeddings:
            diarize_params["return_embeddings"] = True
            logger.info("Speaker embeddings will be returned")

        diarize_output = diarize_model(audio, **diarize_params)

        if return_speaker_embeddings and isinstance(diarize_output, tuple):
            diarize_segments, speaker_embeddings = diarize_output
            logger.info(f"Received speaker embeddings for {len(speaker_embeddings)} speakers")
        else:
            diarize_segments = diarize_output

        if hasattr(diarize_segments, "exclusive_speaker_diarization"):
            diarize_segments = diarize_segments.exclusive_speaker_diarization
            logger.info("Using exclusive speaker diarization for better timestamp reconciliation")

        result = whispermlx.assign_word_speakers(diarize_segments, result)

        # Re-split coarse segments along diarization turn boundaries when
        # no segment has word-level data (word_timestamps=false path).
        # This fixes the case where assign_word_speakers collapses multi-speaker
        # audio to a single dominant speaker on a coarse segment.
        result = _resplit_segments_on_diarization_turns(result, diarize_segments)

        logger.info("Speaker diarization complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}, continuing without diarization")

    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Segment re-split helper (for diarize=true + word_timestamps=false)
# ---------------------------------------------------------------------------


def _resplit_segments_on_diarization_turns(result: dict, diarize_segments) -> dict:
    """
    Re-split coarse transcript segments along diarization turn boundaries
    when no segment has word-level data (word_timestamps=false path).

    ROOT CAUSE: whispermlx.transcribe returns ONE coarse merged segment for
    audio shorter than the 30s VAD chunk, and whispermlx.assign_word_speakers
    only assigns the single dominant speaker to that one segment, collapsing
    multi-speaker audio to 1 speaker.

    This helper re-splits segments along the diarization turn boundaries so
    that each speaker run becomes its own sub-segment with the correct label.

    Rules:
    1. Guard: if any segment already has word-level data, return immediately
       (the word_timestamps=true / aligned path is completely untouched).
    2. For each coarse segment, clip diarization turns to the segment span,
       merge consecutive same-speaker runs, and emit one sub-segment per
       speaker run with start/end set to the clipped turn bounds.
    3. Apportion segment text across sub-segments by duration (no per-word
       timing available) so each sub-segment text is non-empty where possible.
    4. NEVER add a 'words' key to any segment.
    5. If a segment has zero overlapping turns, leave it as-is (retains the
       dominant-speaker label from assign_word_speakers) so behavior never
       regresses.
    """
    segments = result.get("segments", [])

    # Guard: if any segment already has word-level data, do nothing
    if any(seg.get("words") for seg in segments):
        return result

    # Extract diarization turns from the DataFrame
    try:
        turns = [(row["start"], row["end"], row["speaker"]) for _, row in diarize_segments.iterrows()]
    except (AttributeError, KeyError, TypeError):
        # diarize_segments is not a usable DataFrame; return unchanged
        return result

    if not turns:
        return result

    # Sort turns by start time
    turns.sort(key=lambda t: (t[0], t[1]))

    new_segments = []
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        seg_text = seg.get("text", "")

        # Clip turns to the segment span
        clipped = []
        for t_start, t_end, t_speaker in turns:
            c_start = max(t_start, seg_start)
            c_end = min(t_end, seg_end)
            if c_start < c_end:  # Has positive overlap
                clipped.append((c_start, c_end, t_speaker))

        if not clipped:
            # No overlapping turns: leave segment as-is (retains dominant speaker)
            new_segments.append(seg)
            continue

        # Merge consecutive same-speaker runs
        merged = [clipped[0]]
        for t_start, t_end, t_speaker in clipped[1:]:
            prev_start, prev_end, prev_speaker = merged[-1]
            if t_speaker == prev_speaker and t_start <= prev_end:
                # Extend the previous run
                merged[-1] = (prev_start, max(prev_end, t_end), prev_speaker)
            else:
                merged.append((t_start, t_end, t_speaker))

        # Apportion text across sub-segments by duration
        total_duration = sum(end - start for start, end, _ in merged)
        if total_duration <= 0:
            new_segments.append(seg)
            continue

        text_words = seg_text.strip().split()
        n_words = len(text_words)

        if n_words == 0:
            # No words to apportion; each sub-segment gets the full text
            for sub_start, sub_end, sub_speaker in merged:
                sub_seg = {
                    "start": sub_start,
                    "end": sub_end,
                    "text": seg_text,
                    "speaker": sub_speaker,
                }
                for key in seg:
                    if key not in sub_seg and key != "words":
                        sub_seg[key] = seg[key]
                new_segments.append(sub_seg)
            continue

        # Distribute words proportionally based on duration
        durations = [end - start for start, end, _ in merged]
        remaining_subsegments = len(merged)
        word_cursor = 0

        for i, (sub_start, sub_end, sub_speaker) in enumerate(merged):
            remaining_subsegments = len(merged) - i
            if i < len(merged) - 1:
                n = max(1, round(n_words * durations[i] / total_duration))
                # Ensure at least 1 word per remaining sub-segment
                max_n = n_words - word_cursor - remaining_subsegments + 1
                n = min(n, max(max_n, 1))
            else:
                n = n_words - word_cursor  # remainder

            sub_words = text_words[word_cursor : word_cursor + n]
            sub_text = " ".join(sub_words) if sub_words else seg_text
            word_cursor += n

            sub_seg = {
                "start": sub_start,
                "end": sub_end,
                "text": sub_text,
                "speaker": sub_speaker,
            }
            # Copy other keys from original segment (except 'words')
            for key in seg:
                if key not in sub_seg and key != "words":
                    sub_seg[key] = seg[key]

            new_segments.append(sub_seg)

    result["segments"] = new_segments
    return result


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
def sanitize_float_values(obj):
    """Recursively sanitize float values for JSON compliance (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    word_timestamps: bool = True,
    should_diarize: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    return_speaker_embeddings: bool = False,
) -> tuple[dict, dict | None]:
    """
    Run the full 3-stage pipeline: transcribe -> align -> diarize.

    Returns (result, speaker_embeddings_or_None).
    """
    result = transcribe(
        audio,
        model_name=model_name,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )

    if word_timestamps:
        result = align(audio, result)

    speaker_embeddings = None
    if should_diarize:
        result, speaker_embeddings = diarize(
            audio,
            result,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

    return result, speaker_embeddings
