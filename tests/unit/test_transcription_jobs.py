"""Saved jobs survive disconnects and restarts without duplicate submission."""

import asyncio
import io
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi import UploadFile

from app import transcription_jobs as jobs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "STORE", tmp_path)
    scheduled = []
    monkeypatch.setattr(jobs, "schedule", scheduled.append)
    return scheduled


def submit(job_id, content=b"audio"):
    return asyncio.run(jobs.submit(UploadFile(file=io.BytesIO(content), filename="test.wav"), job_id, "small", None, None, True))


def test_submission_is_idempotent_and_conflicts_are_rejected(store):
    job_id = str(uuid4())
    first = submit(job_id)
    assert first["status"] == "queued"
    assert submit(job_id)["id"] == first["id"]
    assert store == [job_id]
    with pytest.raises(HTTPException) as error:
        submit(job_id, b"different audio")
    assert error.value.status_code == 409
    assert (jobs.STORE / f"{job_id}.audio").read_bytes() == b"audio"


def test_restart_resumes_saved_audio_and_marks_missing_audio_failed(store):
    resumable, missing = str(uuid4()), str(uuid4())
    submit(resumable)
    submit(missing)
    jobs.update_job(resumable, status="running", stage="aligning", progress=45)
    (jobs.STORE / f"{missing}.audio").unlink()
    store.clear()
    asyncio.run(jobs.recover_jobs())
    assert store == [resumable]
    assert jobs.read_job(resumable)["status"] == "queued"
    assert jobs.read_job(missing)["status"] == "failed"


def test_processing_error_is_saved_with_traceback(store, monkeypatch):
    job_id = str(uuid4())
    submit(job_id)

    async def fail(fn):
        raise RuntimeError("alignment failed")

    monkeypatch.setattr(jobs, "run_in_queue", fail)
    asyncio.run(jobs._run(job_id))
    record = jobs.read_job(job_id)
    assert record["status"] == "failed"
    assert "alignment failed" in record["error"]
    assert "RuntimeError" in record["traceback"]
    assert not (jobs.STORE / f"{job_id}.audio").exists()


def test_finished_result_remains_available_after_audio_cleanup(store, monkeypatch):
    job_id = str(uuid4())
    submit(job_id)

    async def execute(fn):
        return fn()

    monkeypatch.setattr(jobs, "run_in_queue", execute)
    monkeypatch.setattr(jobs, "_process", lambda _: {"text": "hello", "segments": []})
    asyncio.run(jobs._run(job_id))
    assert jobs.read_job(job_id)["status"] == "completed"
    assert asyncio.run(jobs.result(job_id))["text"] == "hello"
    assert not (jobs.STORE / f"{job_id}.audio").exists()


def test_stage_callbacks_persist_percentages(store, monkeypatch):
    import numpy as np

    job_id = str(uuid4())
    submit(job_id)
    stages = []
    original = jobs.update_job

    def update(*args, **changes):
        if "stage" in changes:
            stages.append((changes["stage"], changes.get("progress")))
        return original(*args, **changes)

    def transcribe(audio, **kwargs):
        kwargs["progress_callback"](100)
        return {"segments": [{"start": 0, "end": 1, "text": "hello"}], "language": "en"}

    def align(audio, result, **kwargs):
        assert kwargs["strict"] is True
        kwargs["progress_callback"](100)
        return result

    monkeypatch.setattr(jobs, "update_job", update)
    monkeypatch.setattr(jobs.whispermlx, "load_audio", lambda _: np.zeros(16000))
    monkeypatch.setattr(jobs, "transcribe", transcribe)
    monkeypatch.setattr(jobs, "align", align)
    assert jobs._process(job_id)["text"] == "hello"
    assert ("detecting_speech", None) in stages
    assert ("transcribing", 100) in stages
    assert ("aligning", 100) in stages
