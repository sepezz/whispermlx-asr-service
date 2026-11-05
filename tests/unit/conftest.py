"""Shared fixtures for unit tests (whispermlx mocked, fast, no model downloads)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Set this during collection, before importing any application module. A
# TestClient lifespan must never recover the running service's real jobs.
_test_job_root = tempfile.TemporaryDirectory(prefix="whispermlx-unit-jobs-")
os.environ["TRANSCRIPTION_JOBS_DIR"] = _test_job_root.name


@pytest.fixture(autouse=True)
def _isolate_job_store(tmp_path, monkeypatch):
    from app import transcription_jobs as jobs

    isolated = tmp_path / "jobs"
    monkeypatch.setenv("TRANSCRIPTION_JOBS_DIR", str(isolated))
    monkeypatch.setattr(jobs, "STORE", isolated)
    monkeypatch.setattr(jobs, "_tasks", {})
    yield
    for task in list(jobs._tasks.values()):
        task.cancel()


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test collected under tests/unit with the ``unit`` marker.

    This lets callers run ``pytest -m unit`` (or exclude with ``-m "not unit"``)
    without requiring each test file to add decorators individually.

    Only marks tests whose file path is under tests/unit so that integration
    tests collected in the same session are not accidentally marked unit.
    """
    unit_marker = pytest.mark.unit
    unit_dir = str(Path(__file__).resolve().parent)
    for item in items:
        if unit_dir in str(item.fspath):
            item.add_marker(unit_marker)


@pytest.fixture(autouse=True)
def _reset_pipeline_state(_isolate_job_store):
    """Reset pipeline module-level caches between tests to avoid cross-contamination.

    Clears the whisper model cache, last-used timestamps, and the eviction
    thread flag so each test starts from a clean state regardless of what
    prior tests loaded.

    CRITICAL: ``test_pipeline_transcribe`` calls ``importlib.reload(app.pipeline)``
    which replaces ``_whisper_models`` with a new dict.  ``app.main`` still holds
    a reference to the *old* dict via ``from app.pipeline import _whisper_models
    as loaded_models``.  We re-link ``app.main.loaded_models`` to the current
    ``app.pipeline._whisper_models`` so TestClient-based tests see the same dict
    the pipeline code mutates.
    """
    try:
        import app.pipeline as pipe_mod

        pipe_mod._whisper_models.clear()
        pipe_mod._whisper_models_last_used.clear()
        pipe_mod._eviction_thread_started = False
    except ImportError:
        pass

    # Re-link app.main.loaded_models to the (possibly reloaded) pipeline dict
    try:
        import app.main as main_mod
        import app.pipeline as pipe_mod

        main_mod.loaded_models = pipe_mod._whisper_models
    except (ImportError, AttributeError):
        pass

    yield

    # Clean up after the test as well
    try:
        import app.pipeline as pipe_mod

        pipe_mod._whisper_models.clear()
        pipe_mod._whisper_models_last_used.clear()
        pipe_mod._eviction_thread_started = False
    except ImportError:
        pass


@pytest.fixture()
def mock_whispermlx():
    """
    Patch the whispermlx import used by app.pipeline so no real model is loaded.

    Returns the mock module dict so tests can configure return values.
    """
    mock_module = MagicMock()
    mock_module.load_model.return_value = MagicMock()
    mock_module.load_align_model.return_value = (
        MagicMock(),
        {"language": "en", "dictionary": {}, "type": "huggingface"},
    )
    mock_module.align.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}
        ],
        "word_segments": [{"word": "hello", "start": 0.0, "end": 1.0}],
    }
    mock_module.load_audio.return_value = MagicMock()
    with patch.dict("sys.modules", {"whispermlx": mock_module}):
        yield mock_module
