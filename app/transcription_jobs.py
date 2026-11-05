"""Durable, idempotent transcription jobs with per-stage progress."""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from uuid import UUID

import whispermlx
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile

from app.job_locks import AlreadyOwned
from app.job_locks import exclusive_file
from app.openai_compat import format_verbose_json_response
from app.pipeline import align
from app.pipeline import get_canonical_models
from app.pipeline import resolve_model_name
from app.pipeline import transcribe
from app.queue import run_in_queue

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/audio/jobs", tags=["Transcription jobs"])
STORE = Path(os.getenv("TRANSCRIPTION_JOBS_DIR", "jobs"))
_lock = threading.RLock()
_tasks: dict[str, asyncio.Task] = {}


def _path(job_id: str) -> Path:
    try:
        UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID")
    return STORE / f"{job_id}.json"


def read_job(job_id: str) -> dict:
    with _lock:
        try:
            return json.loads(_path(job_id).read_text())
        except FileNotFoundError:
            raise HTTPException(404, "Transcription job not found")
        except (ValueError, OSError):
            raise HTTPException(410, "Saved transcription job cannot be read")


def update_job(job_id: str, **changes) -> dict:
    with _lock:
        job = read_job(job_id)
        job.update(changes, updated_at=time.time())
        path = _path(job_id)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job))
        temp.replace(path)
        return job


def public_job(job: dict) -> dict:
    return {
        key: job.get(key)
        for key in (
            "id",
            "status",
            "stage",
            "progress",
            "created_at",
            "updated_at",
            "error",
            "warnings",
            "result_ready",
            "failure_code",
            "retryable",
        )
    }


def _process(job_id: str) -> dict:
    job = read_job(job_id)
    last_write = 0.0
    last_stage = None

    def report(stage: str, percent: float | None = None):
        nonlocal last_write, last_stage
        now = time.monotonic()
        if stage != last_stage or percent is None or percent >= 100 or now - last_write >= 1:
            update_job(job_id, stage=stage, progress=percent)
            last_write = now
            last_stage = stage

    report("loading_audio")
    audio_path = STORE / f"{job_id}.audio"
    if not audio_path.is_file():
        raise FileNotFoundError("Uploaded audio is missing. The client can safely re-upload the retained source.")
    audio = whispermlx.load_audio(str(audio_path))
    duration = len(audio) / 16000
    report("transcribing", 0)
    checkpoint = STORE / f"{job_id}.transcribed.json"
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        logger.info("Resuming saved transcription before alignment: %s", job_id)
    else:
        result = transcribe(
            audio,
            model_name=job["model"],
            language=job["language"],
            initial_prompt=job["prompt"],
            progress_callback=lambda percent: report("transcribing", percent),
        )
        if result.get("segments"):
            temp = checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps(result))
            temp.replace(checkpoint)
    if not result.get("segments"):
        raise RuntimeError("Transcription returned no speech segments")
    if job["word_timestamps"]:
        report("aligning", 0)
        result = align(audio, result, progress_callback=lambda percent: report("aligning", percent), strict=True)
    report("saving_result")
    update_job(job_id, warnings=result.get("warnings", []))
    return format_verbose_json_response(
        result,
        task="transcribe",
        language=result.get("language", "en"),
        duration=duration,
        include_words=job["word_timestamps"],
        include_segments=True,
    ).model_dump(exclude_none=True)


async def _run(job_id: str):
    try:
        # Keep queued status until the executor starts this particular job.
        def execute():
            with exclusive_file(STORE / f".{job_id}.lock"):
                saved = read_job(job_id)
                if saved["status"] == "completed" and (STORE / f"{job_id}.result.json").exists():
                    return
                if saved["status"] not in ("queued", "running"):
                    return
                update_job(job_id, status="running", stage="loading_audio", progress=None)
                result = _process(job_id)
                result_path = STORE / f"{job_id}.result.json"
                temp = result_path.with_suffix(".tmp")
                temp.write_text(json.dumps(result))
                temp.replace(result_path)
                update_job(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100,
                    result_ready=True,
                    failure_code=None,
                    retryable=False,
                )
                (STORE / f"{job_id}.audio").unlink(missing_ok=True)

        await run_in_queue(execute)
    except AlreadyOwned:
        logger.info("Another process owns transcription job %s. Leaving its input and state untouched.", job_id)
    except asyncio.CancelledError:
        # Startup recovery will resume jobs not marked completed or failed.
        raise
    except Exception as exc:
        logger.exception("Transcription job %s failed", job_id)
        code = (
            "input_missing"
            if isinstance(exc, FileNotFoundError)
            else ("audio_decode_failed" if "Failed to load audio" in str(exc) else "processing_failed")
        )
        detail = "\n".join(str(exc).splitlines()[-6:]) if code == "audio_decode_failed" else str(exc)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {detail}"[:1800],
            failure_code=code,
            retryable=code == "input_missing" and read_job(job_id).get("input_reuploads", 0) < 2,
            traceback=traceback.format_exc(),
        )
        # Keep failed inputs and completed checkpoints for recovery and audit.
    finally:
        _tasks.pop(job_id, None)


def schedule(job_id: str):
    if job_id not in _tasks:
        _tasks[job_id] = asyncio.create_task(_run(job_id))


async def recover_jobs():
    STORE.mkdir(parents=True, exist_ok=True)
    for path in STORE.glob("*.json"):
        if path.name.endswith((".result.json", ".transcribed.json")):
            continue
        try:
            job = json.loads(path.read_text())
        except (ValueError, OSError):
            logger.exception("Cannot recover saved transcription record %s", path)
            continue
        try:
            UUID(job.get("id", ""))
            if job.get("status") not in ("queued", "running", "completed", "failed"):
                raise ValueError('Unsupported job status')
        except (ValueError, TypeError, AttributeError):
            logger.error("Ignoring invalid job record %s", path.name)
            continue
        if (
            job.get("status") == "failed"
            and not job.get("failure_code")
            and "No such file or directory" in str(job.get("error", ""))
            and ".audio" in str(job.get("error", ""))
        ):
            update_job(job["id"], failure_code="input_missing", retryable=job.get("input_reuploads", 0) < 2)
        if job["status"] in ("queued", "running"):
            job_id = job["id"]
            if (STORE / f"{job_id}.audio").exists():
                update_job(job_id, status="queued", stage="queued", progress=None)
                schedule(job_id)
            else:
                update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    error="Service restarted but saved audio is missing",
                    failure_code="input_missing",
                    retryable=job.get("input_reuploads", 0) < 2,
                )


@router.get("/capabilities")
async def capabilities():
    return {"durable_jobs": True, "stage_progress": True, "version": 1}


@router.post("", status_code=202)
async def submit(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    model: str = Form("small"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    word_timestamps: bool = Form(True),
):
    path = _path(job_id)
    model = resolve_model_name(model)
    if model not in get_canonical_models():
        raise HTTPException(400, "Unsupported transcription model")
    STORE.mkdir(parents=True, exist_ok=True)
    # Stream uploads to disk rather than keeping a second copy in memory.
    temp = STORE / f"{job_id}.{time.time_ns()}.upload"
    digest = hashlib.sha256()
    size = 0
    try:
        with temp.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > int(os.getenv("MAX_FILE_SIZE_MB", "1000")) * 1024 * 1024:
                    raise HTTPException(413, "Audio file too large")
                digest.update(chunk)
                output.write(chunk)
        if not size:
            raise HTTPException(400, "Empty audio file")
        options = {
            "digest": digest.hexdigest(),
            "model": model,
            "language": language,
            "prompt": prompt,
            "word_timestamps": word_timestamps,
        }
        with _lock:
            if path.exists():
                job = read_job(job_id)
                if any(job.get(key) != value for key, value in options.items()):
                    raise HTTPException(409, "Job ID already belongs to a different request")
                if (
                    job["status"] == "failed"
                    and job.get("failure_code") == "input_missing"
                    and job.get("input_reuploads", 0) < 2
                ):
                    temp.replace(STORE / f"{job_id}.audio")
                    job = update_job(
                        job_id,
                        status="queued",
                        stage="queued",
                        progress=None,
                        error=None,
                        failure_code=None,
                        retryable=False,
                        input_reuploads=job.get("input_reuploads", 0) + 1,
                    )
                    schedule(job_id)
                return public_job(job)
            temp.replace(STORE / f"{job_id}.audio")
            now = time.time()
            job = dict(
                options,
                id=job_id,
                status="queued",
                stage="queued",
                progress=None,
                created_at=now,
                updated_at=now,
                error=None,
                result_ready=False,
            )
            pending = path.with_suffix(".tmp")
            pending.write_text(json.dumps(job))
            pending.replace(path)
        schedule(job_id)
        return public_job(job)
    finally:
        temp.unlink(missing_ok=True)
        await file.close()


@router.get("/{job_id}")
async def status(job_id: str):
    return public_job(read_job(job_id))


@router.get("/{job_id}/result")
async def result(job_id: str):
    job = read_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Transcription result is not ready")
    return json.loads((STORE / f"{job_id}.result.json").read_text())
