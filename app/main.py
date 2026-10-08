"""
Whispermlx ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints
"""

import logging
import os
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import whispermlx
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from app import metrics as prom_metrics
from app.pipeline import BATCH_SIZE
from app.pipeline import COMPUTE_TYPE
from app.pipeline import DEFAULT_MODEL
from app.pipeline import DEVICE
from app.pipeline import _whisper_models as loaded_models
from app.pipeline import format_timestamp
from app.pipeline import get_canonical_models
from app.pipeline import load_whisper_model
from app.pipeline import resolve_model_name
from app.pipeline import run_pipeline
from app.pipeline import sanitize_float_values
from app.queue import get_queue_metrics
from app.queue import run_in_queue
from app.version import __version__

# Suppress pyannote pooling warnings about degrees of freedom
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))
SERVE_MODE = "simple"  # Ray Serve removed; always simple mode
VALID_OUTPUT_FORMATS = {"json", "text", "srt", "vtt", "tsv"}

logger.info(f"Whispermlx ASR Service v{__version__} initialized on device: {DEVICE}")
logger.info(f"Compute type: {COMPUTE_TYPE} (inert under MLX), Batch size: {BATCH_SIZE} (inert under MLX)")
logger.info(f"Default model: {DEFAULT_MODEL}, Serve mode: {SERVE_MODE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload models on startup (modern API).

    Uses the FastAPI lifespan context-manager instead of the deprecated
    startup event handler. PRELOAD_MODEL still warms the model at startup.
    """
    prom_metrics.SERVICE_INFO.info(
        {
            "version": __version__,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "serve_mode": SERVE_MODE,
        }
    )
    preload_model = os.getenv("PRELOAD_MODEL", None)
    if preload_model:
        logger.info(f"Preloading model on startup: {preload_model}")
        try:
            load_whisper_model(preload_model)
            logger.info(f"Successfully preloaded model: {preload_model}")
        except Exception as e:
            logger.error(f"Failed to preload model {preload_model}: {str(e)}")
    # Reflect preloaded model(s) in the loaded-models gauge so that
    # /metrics shows whisperx_loaded_models >= 1 before any request
    # (VAL-CROSS-016).
    prom_metrics.LOADED_MODELS.set(len(loaded_models))
    from app.transcription_jobs import recover_jobs

    await recover_jobs()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Whispermlx ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using whispermlx",
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def openai_validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Convert FastAPI RequestValidationError to the OpenAI error envelope
    ONLY for /v1/ paths.  Non-/v1/ paths (e.g. /asr) keep the bare
    FastAPI 422 {"detail": [...]} shape so existing /asr clients are
    unaffected (VAL-ASR-011, VAL-OPS-014).
    """
    if request.url.path.startswith("/v1/"):
        errors = exc.errors()
        # Build a descriptive message and extract the first failing field as param
        messages = []
        param = None
        for err in errors:
            loc = err.get("loc", [])
            field = loc[-1] if loc else None
            msg = err.get("msg", "Validation error")
            if field:
                messages.append(f"{field}: {msg}")
                if param is None:
                    param = str(field)
            else:
                messages.append(msg)
        message = "; ".join(messages) if messages else "Validation error"

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": None,
                }
            },
        )
    # Non-/v1/ paths: return the default FastAPI 422 detail shape
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Whispermlx ASR API",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "serve_mode": SERVE_MODE,
    }


@app.post("/asr")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: str | None = Query(None),
    initial_prompt: str | None = Query(None),
    hotwords: str | None = Query(None),
    word_timestamps: bool = Query(True),
    output_format: str = Query("json"),
    output: str | None = Query(None),
    model: str = Query(DEFAULT_MODEL),
    num_speakers: int | None = Query(None),
    min_speakers: int | None = Query(None),
    max_speakers: int | None = Query(None),
    diarize: bool | None = Query(None),
    enable_diarization: bool | None = Query(None),
    return_speaker_embeddings: bool | None = Query(None),
):
    """
    Main ASR endpoint compatible with openai-whisper-asr-webservice

    Args:
        audio_file: Audio file to transcribe
        task: transcribe or translate
        language: Language code (e.g., 'en', 'es', 'fr')
        initial_prompt: Optional prompt to guide the model
        word_timestamps: Return word-level timestamps
        output_format: json, text, srt, vtt, or tsv
        model: whispermlx model name (tiny, base, small, medium, large-v2, large-v3)
        num_speakers: Exact number of speakers (if known, overrides min/max)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        diarize: Enable speaker diarization (compatible with whisper-asr-webservice)
        enable_diarization: Alias for diarize (deprecated, use diarize instead)
        return_speaker_embeddings: Return speaker embeddings (256-dimensional vectors)
    """
    temp_audio_path = None
    request_started = time.time()
    metric_status = "error"
    prom_metrics.ACTIVE_TRANSCRIPTIONS.inc()

    try:
        # Handle legacy parameter names
        if output is not None:
            output_format = output

        # Validate output_format early so invalid formats are rejected
        # before any pipeline execution (avoids wasted compute).
        if output_format not in VALID_OUTPUT_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format: {output_format}",
            )

        # Map OpenAI-style aliases (whisper-tiny, whisper-large-v3, whisper-1, ...)
        # to canonical MLX model names so /asr accepts the same identifiers
        # advertised by /v1/models.
        model = resolve_model_name(model)

        # Validate resolved model against the MLX canonical list.
        # Unknown names would cause whispermlx.load_model to fail with an
        # opaque error; surface a clean 400 instead.
        canonical_models = set(get_canonical_models())
        if model not in canonical_models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model}. Supported models: {', '.join(sorted(canonical_models))}",
            )

        # Resolve diarization toggle
        if diarize is not None or enable_diarization is not None:
            should_diarize = (diarize is True) or (enable_diarization is True)
        else:
            should_diarize = True
        if return_speaker_embeddings is None:
            return_speaker_embeddings = False

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as temp_file:
            temp_audio_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        prom_metrics.AUDIO_SIZE_MB.observe(file_size_mb)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB. "
                f"Large files may cause out-of-memory errors.",
            )

        if file_size_mb > 100:
            logger.warning(f"Processing large file ({file_size_mb:.1f}MB) - may consume significant VRAM")

        logger.info(
            f"Processing audio file: {audio_file.filename} ({file_size_mb:.1f}MB), model: {model}, language: {language}"
        )

        # Load audio
        audio = whispermlx.load_audio(temp_audio_path)
        prom_metrics.AUDIO_DURATION.observe(len(audio) / 16000.0)

        # Run pipeline through the async queue (GPU semaphore)
        result, speaker_embeddings = await run_in_queue(
            run_pipeline,
            audio,
            model_name=model,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            should_diarize=should_diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

        detected_language = result.get("language", language or "en")

        # Format output based on requested format
        if output_format == "json":
            # Legacy shape: text is a JSON ARRAY mirroring segments
            # (same structure as the segments array), NOT a joined string.
            # Preserves drop-in compatibility with the original whisper-asr-webservice.
            response_data = {
                "text": result.get("segments", []),
                "language": detected_language,
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", []),
            }

            if return_speaker_embeddings and speaker_embeddings:
                response_data["speaker_embeddings"] = sanitize_float_values(speaker_embeddings)
                logger.info(f"Including speaker embeddings in response: {list(speaker_embeddings.keys())}")

            metric_status = "ok"
            return JSONResponse(content=response_data)

        elif output_format == "text":
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            metric_status = "ok"
            return {"text": text}

        elif output_format == "srt":
            srt_content = []
            for i, segment in enumerate(result.get("segments", []), 1):
                start_time = format_timestamp(segment.get("start", 0))
                end_time = format_timestamp(segment.get("end", 0))
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

            metric_status = "ok"
            return {"srt": "\n".join(srt_content)}

        elif output_format == "vtt":
            vtt_content = ["WEBVTT\n"]
            for segment in result.get("segments", []):
                start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
                end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")

            metric_status = "ok"
            return {"vtt": "\n".join(vtt_content)}

        elif output_format == "tsv":
            tsv_content = ["start\tend\ttext\tspeaker"]
            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")

            metric_status = "ok"
            return {"tsv": "\n".join(tsv_content)}

    except HTTPException as e:
        metric_status = f"http_{e.status_code}"
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        prom_metrics.ACTIVE_TRANSCRIPTIONS.dec()
        prom_metrics.REQUEST_DURATION.labels(endpoint="/asr").observe(time.time() - request_started)
        prom_metrics.REQUESTS_TOTAL.labels(endpoint="/asr", status=metric_status).inc()
        prom_metrics.refresh_vram()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
        "serve_mode": SERVE_MODE,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics in OpenMetrics text format."""
    prom_metrics.LOADED_MODELS.set(len(loaded_models))
    prom_metrics.refresh_vram()
    body, content_type = prom_metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/queue-metrics")
async def queue_metrics():
    """Queue and pipeline state (JSON; the old /metrics shape)."""
    data = {
        "serve_mode": SERVE_MODE,
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
        "queue": get_queue_metrics(),
    }
    return data


# Register OpenAI-compatible API routers
# Import here to avoid circular imports (openai_compat imports from this module)
from app.openai_compat import models_router
from app.openai_compat import router as openai_router

app.include_router(openai_router)
app.include_router(models_router)
from app.transcription_jobs import router as jobs_router

app.include_router(jobs_router)


def main():
    """Entry point for uvx / console_scripts."""
    import uvicorn

    port = int(os.getenv("PORT", "9001"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
