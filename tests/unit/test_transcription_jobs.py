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
    return asyncio.run(
        jobs.submit(UploadFile(file=io.BytesIO(content), filename="test.wav"), job_id, "small", None, None, True)
    )


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


def test_restart_ignores_transcription_checkpoint_as_a_job_record(store):
    job_id = str(uuid4())
    submit(job_id)
    (jobs.STORE / f"{job_id}.transcribed.json").write_text('{"segments": []}')
    store.clear()
    asyncio.run(jobs.recover_jobs())
    assert store == [job_id]


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
    assert (jobs.STORE / f"{job_id}.audio").exists()


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
    # A stale scheduled execution must not reopen the completed job after its
    # successful cleanup and report a spurious missing-input failure.
    monkeypatch.setattr(jobs, "_process", lambda _: (_ for _ in ()).throw(AssertionError("Duplicate execution")))
    asyncio.run(jobs._run(job_id))
    assert jobs.read_job(job_id)["status"] == "completed"


def test_invalid_record_does_not_stop_recovery_of_valid_jobs(store):
    job_id = str(uuid4())
    submit(job_id)
    (jobs.STORE / "invalid.json").write_text('{"unexpected":"data"}')
    store.clear()
    asyncio.run(jobs.recover_jobs())
    assert store == [job_id]


def test_missing_upload_can_be_restored_with_same_job_id(store):
    job_id = str(uuid4())
    submit(job_id)
    (jobs.STORE / f"{job_id}.audio").unlink()
    jobs.update_job(job_id, status="failed", failure_code="input_missing", retryable=True)
    restored = submit(job_id)
    assert restored["status"] == "queued"
    assert (jobs.STORE / f"{job_id}.audio").read_bytes() == b"audio"
    assert jobs.read_job(job_id)["input_reuploads"] == 1
    assert store == [job_id, job_id]


def test_legacy_missing_file_failure_is_made_recoverable_without_running_without_audio(store):
    job_id = str(uuid4())
    submit(job_id)
    (jobs.STORE / f"{job_id}.audio").unlink()
    jobs.update_job(job_id, status="failed", error=f"Failed to load audio: No such file or directory {job_id}.audio")
    store.clear()
    asyncio.run(jobs.recover_jobs())
    record = jobs.read_job(job_id)
    assert record["failure_code"] == "input_missing" and record["retryable"] is True
    assert store == []


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
    assert ("transcribing", 0) in stages
    assert not any(stage == "detecting_speech" for stage, _ in stages)
    assert ("transcribing", 100) in stages
    assert ("aligning", 100) in stages


def test_completed_transcription_is_checkpointed_before_alignment(store, monkeypatch):
    import json

    import numpy as np

    job_id = str(uuid4())
    submit(job_id)
    native = {"segments": [{"start": 0, "end": 1, "text": "hello"}], "language": "en"}
    monkeypatch.setattr(jobs.whispermlx, "load_audio", lambda _: np.zeros(16000))
    monkeypatch.setattr(jobs, "transcribe", lambda *args, **kwargs: native)

    def failed_alignment(*args, **kwargs):
        assert json.loads((jobs.STORE / f"{job_id}.transcribed.json").read_text()) == native
        raise RuntimeError("aligner failed")

    monkeypatch.setattr(jobs, "align", failed_alignment)
    with pytest.raises(RuntimeError, match="aligner failed"):
        jobs._process(job_id)

    def no_second_transcription(*args, **kwargs):
        raise AssertionError("Completed transcription must be reused")

    monkeypatch.setattr(jobs, "transcribe", no_second_transcription)
    monkeypatch.setattr(jobs, "align", lambda audio, result, **kwargs: result)
    assert jobs._process(job_id)["text"] == "hello"
