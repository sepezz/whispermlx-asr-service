"""Regression coverage for live/test storage isolation and duplicate ownership."""

import asyncio
import io
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import UploadFile

from app import transcription_jobs as jobs
from app.job_locks import AlreadyOwned
from app.job_locks import exclusive_file


def test_every_unit_test_uses_storage_outside_the_live_checkout():
    live_default = Path(__file__).resolve().parents[2] / "jobs"
    assert jobs.STORE.resolve() != live_default.resolve()
    assert not jobs.STORE.resolve().is_relative_to(live_default.resolve())


def test_two_workers_cannot_own_the_same_storage(tmp_path):
    lock = tmp_path / ".service.lock"
    with exclusive_file(lock), pytest.raises(AlreadyOwned), exclusive_file(lock):
        raise AssertionError("Second owner entered")


def test_duplicate_job_execution_cannot_delete_the_owned_input(monkeypatch):
    job_id = str(uuid4())
    monkeypatch.setattr(jobs, "schedule", lambda _: None)
    asyncio.run(
        jobs.submit(UploadFile(file=io.BytesIO(b"audio"), filename="test.wav"), job_id, "small", None, None, True)
    )

    async def execute(fn):
        return fn()

    monkeypatch.setattr(jobs, "run_in_queue", execute)
    with exclusive_file(jobs.STORE / f".{job_id}.lock"):
        asyncio.run(jobs._run(job_id))
    assert jobs.read_job(job_id)["status"] == "queued"
    assert (jobs.STORE / f"{job_id}.audio").read_bytes() == b"audio"
