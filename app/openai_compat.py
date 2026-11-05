"""
OpenAI-compatible Whisper API endpoints
POST /v1/audio/transcriptions
POST /v1/audio/translations
GET /v1/models
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import whispermlx
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse

from app import metrics as prom_metrics
from app.pipeline import DEFAULT_MODEL
from app.pipeline import align as pipeline_align
from app.pipeline import format_timestamp
from app.pipeline import get_canonical_models
from app.pipeline import resolve_model_name
from app.pipeline import transcribe as pipeline_transcribe
from app.queue import run_in_queue
from app.schemas import OpenAIErrorDetail
from app.schemas import OpenAIErrorResponse
from app.schemas import ResponseFormat
from app.schemas import TranscriptionSegment
from app.schemas import TranscriptionVerboseJsonResponse
from app.schemas import TranscriptionWord

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))

router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])
models_router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

# Model mapping: OpenAI model names to MLX model names
MODEL_MAPPING = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}


def create_openai_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Create OpenAI-compatible error response"""
    error_response = OpenAIErrorResponse(
        error=OpenAIErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=error_response.model_dump())


def format_verbose_json_response(
    result: dict, task: str, language: str, duration: float, include_words: bool, include_segments: bool
) -> TranscriptionVerboseJsonResponse:
    """Format whispermlx result as OpenAI verbose_json response"""

    full_text = " ".join([seg.get("text", "").strip() for seg in result.get("segments", [])]).strip()

    segments = []
    if include_segments:
        for idx, seg in enumerate(result.get("segments", [])):
            segments.append(
                TranscriptionSegment(
                    id=idx,
                    seek=int(seg.get("start", 0) * 100),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    tokens=[],
                    temperature=0.0,
                    avg_logprob=0.0,
                    compression_ratio=0.0,
                    no_speech_prob=0.0,
                )
            )

    words = None
    if include_words:
        words = []
        word_segments = result.get("word_segments", [])
        if not word_segments:
            for seg in result.get("segments", []):
                word_segments.extend(seg.get("words", []))

        for word_data in word_segments:
            if "word" in word_data and "start" in word_data and "end" in word_data:
                words.append(
                    TranscriptionWord(
                        word=word_data["word"].strip(), start=word_data.get("start", 0.0), end=word_data.get("end", 0.0)
                    )
                )

    return TranscriptionVerboseJsonResponse(
        task=task,
        language=language,
        duration=duration,
        text=full_text,
        segments=segments,
        words=words,
        warnings=result.get("warnings") or None,
        untranscribed_regions=result.get("untranscribed_regions") or None,
    )


def format_srt_response(result: dict) -> str:
    """Format whispermlx result as SRT subtitle format"""
    srt_content = []
    for i, segment in enumerate(result.get("segments", []), 1):
        start_time = format_timestamp(segment.get("start", 0))
        end_time = format_timestamp(segment.get("end", 0))
        text = segment.get("text", "").strip()
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)


def format_vtt_response(result: dict) -> str:
    """Format whispermlx result as WebVTT subtitle format"""
    vtt_content = ["WEBVTT\n"]
    for segment in result.get("segments", []):
        start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
        end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
        text = segment.get("text", "").strip()
        vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
    return "\n".join(vtt_content)


def _run_transcribe_and_align(
    audio,
    whisperx_model: str,
    language: str | None,
    task: str,
    need_word_timestamps: bool,
    hotwords: str | None = None,
    initial_prompt: str | None = None,
):
    """Blocking helper that runs transcription + optional alignment."""
    result = pipeline_transcribe(
        audio,
        model_name=whisperx_model,
        language=language,
        task=task,
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )
    if need_word_timestamps:
        result = pipeline_align(audio, result)
    return result


async def process_audio(
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: ResponseFormat,
    temperature: float,
    timestamp_granularities: list[str],
    task: str = "transcribe",
    hotwords: str | None = None,
    endpoint: str = "/v1/audio/transcriptions",
) -> JSONResponse | PlainTextResponse:
    """
    Core audio processing function shared by transcriptions and translations endpoints
    """
    temp_audio_path = None
    request_started = time.time()
    metric_status = "ok"
    prom_metrics.ACTIVE_TRANSCRIPTIONS.inc()

    try:
        # Validate model using resolve_model_name (handles aliases + MLX canonical names)
        resolved_model = resolve_model_name(model)
        canonical_models = set(get_canonical_models())
        if resolved_model not in canonical_models and model not in MODEL_MAPPING:
            metric_status = "http_400"
            return create_openai_error(
                400,
                f"Invalid model: {model}. Supported: whisper-1, or MLX models ({', '.join(sorted(canonical_models))})",
                param="model",
            )
        whisperx_model = resolved_model

        # Validate timestamp_granularities requires verbose_json
        if timestamp_granularities and response_format != ResponseFormat.VERBOSE_JSON:
            metric_status = "http_400"
            return create_openai_error(
                400, "timestamp_granularities requires response_format='verbose_json'", param="timestamp_granularities"
            )

        # Validate temperature range
        if temperature < 0 or temperature > 1:
            metric_status = "http_400"
            return create_openai_error(400, "temperature must be between 0 and 1", param="temperature")

        # Save uploaded file to temporary location
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_audio_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        prom_metrics.AUDIO_SIZE_MB.observe(file_size_mb)
        if file_size_mb > MAX_FILE_SIZE_MB:
            metric_status = "http_413"
            return create_openai_error(
                413, f"File too large ({file_size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB", code="file_too_large"
            )

        logger.info(
            f"OpenAI-compat: Processing {file.filename} ({file_size_mb:.1f}MB), model: {whisperx_model}, task: {task}"
        )

        # Load audio
        audio = whispermlx.load_audio(temp_audio_path)
        duration = len(audio) / 16000  # whispermlx loads at 16kHz
        prom_metrics.AUDIO_DURATION.observe(duration)

        need_word_timestamps = response_format == ResponseFormat.VERBOSE_JSON and "word" in timestamp_granularities

        # Run through the async queue
        result = await run_in_queue(
            _run_transcribe_and_align,
            audio,
            whisperx_model,
            language,
            task,
            need_word_timestamps,
            hotwords=hotwords,
            initial_prompt=prompt,
        )

        detected_language = result.get("language", language or "en")

        # Format response based on requested format
        if response_format == ResponseFormat.JSON:
            full_text = " ".join([seg.get("text", "").strip() for seg in result.get("segments", [])]).strip()
            return JSONResponse(content={"text": full_text})

        elif response_format == ResponseFormat.TEXT:
            full_text = " ".join([seg.get("text", "").strip() for seg in result.get("segments", [])]).strip()
            return PlainTextResponse(content=full_text)

        elif response_format == ResponseFormat.SRT:
            srt_content = format_srt_response(result)
            return PlainTextResponse(content=srt_content, media_type="text/plain")

        elif response_format == ResponseFormat.VTT:
            vtt_content = format_vtt_response(result)
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")

        elif response_format == ResponseFormat.VERBOSE_JSON:
            include_words = "word" in timestamp_granularities
            include_segments = "segment" in timestamp_granularities or not timestamp_granularities

            response = format_verbose_json_response(
                result=result,
                task=task,
                language=detected_language,
                duration=duration,
                include_words=include_words,
                include_segments=include_segments,
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        metric_status = "http_400"
        return create_openai_error(400, f"Unsupported response format: {response_format}")

    except HTTPException as e:
        metric_status = f"http_{e.status_code}"
        return create_openai_error(e.status_code, e.detail)

    except Exception as e:
        metric_status = "error"
        logger.error(f"OpenAI-compat error: {str(e)}", exc_info=True)
        return create_openai_error(500, f"Internal server error: {str(e)}", error_type="server_error")

    finally:
        prom_metrics.ACTIVE_TRANSCRIPTIONS.dec()
        prom_metrics.REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - request_started)
        prom_metrics.REQUESTS_TOTAL.labels(endpoint=endpoint, status=metric_status).inc()
        prom_metrics.refresh_vram()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@router.post("/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(..., description="Model ID (whisper-1, large-v3, etc.)"),
    language: str | None = Form(None, description="ISO-639-1 language code"),
    prompt: str | None = Form(None, description="Guidance text/context"),
    hotwords: str | None = Form(None, description="Hotwords/phrases to bias transcription"),
    response_format: ResponseFormat = Form(
        ResponseFormat.JSON, description="Output format: json, text, srt, verbose_json, vtt"
    ),
    temperature: float = Form(0.0, description="Sampling temperature (0-1)"),
):
    """
    Transcribes audio into the input language.

    OpenAI-compatible endpoint: POST /v1/audio/transcriptions
    """
    form_data = await request.form()
    timestamp_granularities = form_data.getlist("timestamp_granularities[]")

    if not timestamp_granularities:
        timestamp_granularities = []
    if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
        timestamp_granularities = ["segment"]

    return await process_audio(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        task="transcribe",
        hotwords=hotwords,
        endpoint="/v1/audio/transcriptions",
    )


@router.post("/translations")
async def create_translation(
    request: Request,
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(..., description="Model ID (whisper-1, large-v3, etc.)"),
    prompt: str | None = Form(None, description="Guidance text/context"),
    hotwords: str | None = Form(None, description="Hotwords/phrases to bias transcription"),
    response_format: ResponseFormat = Form(
        ResponseFormat.JSON, description="Output format: json, text, srt, verbose_json, vtt"
    ),
    temperature: float = Form(0.0, description="Sampling temperature (0-1)"),
):
    """
    Translates audio into English.

    OpenAI-compatible endpoint: POST /v1/audio/translations
    """
    form_data = await request.form()
    timestamp_granularities = form_data.getlist("timestamp_granularities[]")

    if not timestamp_granularities:
        timestamp_granularities = []
    if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
        timestamp_granularities = ["segment"]

    return await process_audio(
        file=file,
        model=model,
        language=None,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        task="translate",
        hotwords=hotwords,
        endpoint="/v1/audio/translations",
    )


# Available whisper models. Built from the MLX model map so this list
# stays in sync with whatever the whispermlx backend supports.
def _build_available_models():
    models = [{"id": "whisper-1", "object": "model", "owned_by": "openai"}]
    for name in get_canonical_models():
        models.append({"id": name, "object": "model", "owned_by": "whispermlx"})
    return models


AVAILABLE_MODELS = _build_available_models()


@models_router.get("/models")
async def list_models():
    """
    List available models.

    OpenAI-compatible endpoint: GET /v1/models
    """
    return {"object": "list", "data": AVAILABLE_MODELS}


@models_router.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    Get details about a specific model.

    OpenAI-compatible endpoint: GET /v1/models/{model_id}
    """
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model

    return create_openai_error(
        404, f"Model '{model_id}' not found", error_type="invalid_request_error", code="model_not_found"
    )
