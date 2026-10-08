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
        )
    }


def _process(job_id: str) -> dict:
    job = read_job(job_id)
    last_write = 0.0

    def report(stage: str, percent: float | None = None):
        nonlocal last_write
        now = time.monotonic()
        if percent is None or percent >= 100 or now - last_write >= 1:
            update_job(job_id, stage=stage, progress=percent)
            last_write = now

    report("loading_audio")
    audio = whispermlx.load_audio(str(STORE / f"{job_id}.audio"))
    duration = len(audio) / 16000
    report("detecting_speech")
    result = transcribe(
        audio,
        model_name=job["model"],
        language=job["language"],
        initial_prompt=job["prompt"],
        progress_callback=lambda percent: report("transcribing", percent),
    )
    if not result.get("segments"):
        raise RuntimeError("Transcription returned no speech segments")
    if job["word_timestamps"]:
        report("aligning", 0)
        result = align(audio, result, progress_callback=lambda percent: report("aligning", percent), strict=True)
    report("saving_result")
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
            update_job(job_id, status="running", stage="loading_audio", progress=None)
            return _process(job_id)

        result = await run_in_queue(execute)
        result_path = STORE / f"{job_id}.result.json"
        temp = result_path.with_suffix(".tmp")
        temp.write_text(json.dumps(result))
        temp.replace(result_path)
        update_job(job_id, status="completed", stage="completed", progress=100, result_ready=True)
        (STORE / f"{job_id}.audio").unlink(missing_ok=True)
    except asyncio.CancelledError:
        # Startup recovery will resume jobs not marked completed or failed.
        raise
    except Exception as exc:
        logger.exception("Transcription job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        (STORE / f"{job_id}.audio").unlink(missing_ok=True)
    finally:
        _tasks.pop(job_id, None)


def schedule(job_id: str):
    if job_id not in _tasks:
        _tasks[job_id] = asyncio.create_task(_run(job_id))


async def recover_jobs():
    STORE.mkdir(parents=True, exist_ok=True)
    for path in STORE.glob("*.json"):
        if path.name.endswith(".result.json"):
            continue
        try:
            job = json.loads(path.read_text())
        except (ValueError, OSError):
            logger.exception("Cannot recover saved transcription record %s", path)
            continue
        if job["status"] in ("queued", "running"):
            job_id = job["id"]
            if (STORE / f"{job_id}.audio").exists():
                update_job(job_id, status="queued", stage="queued", progress=None)
                schedule(job_id)
            else:
                update_job(
                    job_id, status="failed", stage="failed", error="Service restarted but saved audio is missing"
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
