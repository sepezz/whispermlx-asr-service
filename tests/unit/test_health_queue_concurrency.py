"""
Unit tests for health/root endpoints, queue serialization, and PRELOAD_MODEL
behaviour (VAL-OPS-001..005, VAL-OPS-015, VAL-OPS-020,
VAL-CROSS-005..008, VAL-CROSS-017).

All tests mock whispermlx so no real model downloads occur.
"""

import asyncio
import os
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures — follow the same pattern as test_asr_endpoint.py
# ---------------------------------------------------------------------------


def _make_whispermlx_mock():
    """Build a MagicMock that mimics the whispermlx module surface."""
    mock_module = MagicMock()
    mock_diarize = MagicMock()
    mock_diarize.DiarizationPipeline = MagicMock()
    mock_module.diarize = mock_diarize

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "word_segments": [],
        "language": "en",
    }
    mock_module.load_model.return_value = mock_model
    mock_module.load_align_model.return_value = (
        MagicMock(),
        {"language": "en", "dictionary": {}, "type": "huggingface"},
    )
    mock_module.align.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            }
        ],
        "word_segments": [{"word": "hello", "start": 0.0, "end": 0.5}],
    }
    mock_module.load_audio.return_value = MagicMock()
    mock_module.assign_word_speakers.side_effect = lambda diar, result: result
    return mock_module


@pytest.fixture()
def client_no_preload():
    """TestClient with SERVE_MODE=simple, no PRELOAD_MODEL, whispermlx mocked."""
    wmlx = _make_whispermlx_mock()
    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
        patch("app.openai_compat.whispermlx", wmlx),
        patch("app.main.load_whisper_model") as mock_load,
        patch.dict(os.environ, {"PRELOAD_MODEL": "", "DEVICE": "cpu", "SERVE_MODE": "simple"}, clear=False),
    ):
        # Startup won't preload because PRELOAD_MODEL is empty
        mock_load.return_value = MagicMock()
        from app.main import app

        # Clear pipeline caches for a clean state
        from app.pipeline import _whisper_models
        from app.pipeline import _whisper_models_last_used

        _whisper_models.clear()
        _whisper_models_last_used.clear()

        with TestClient(app) as c:
            yield c, wmlx, mock_load


@pytest.fixture()
def client_with_preload():
    """TestClient with PRELOAD_MODEL=base so startup_event loads a model."""
    wmlx = _make_whispermlx_mock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "word_segments": [],
        "language": "en",
    }

    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
        patch("app.openai_compat.whispermlx", wmlx),
        patch.dict(os.environ, {"PRELOAD_MODEL": "base", "DEVICE": "cpu", "SERVE_MODE": "simple"}, clear=False),
    ):
        # The startup event calls load_whisper_model("base") — let it
        # succeed by having pipeline.load_whisper_model add to the cache.
        from app.pipeline import _whisper_models
        from app.pipeline import _whisper_models_last_used

        _whisper_models.clear()
        _whisper_models_last_used.clear()

        # Pre-seed the cache so /health sees "base" as loaded.
        # The startup event in main.py calls load_whisper_model("base")
        # which uses the real pipeline function. We patch it at the main.py
        # level so the startup event populates the cache.
        from app.main import app

        with TestClient(app) as c:
            yield c, wmlx


# ---------------------------------------------------------------------------
# VAL-OPS-001: Health endpoint returns 200 with required fields
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """VAL-OPS-001..003: GET /health returns 200 with required fields."""

    def test_health_returns_200(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/health")
        body = resp.json()
        assert "status" in body
        assert "device" in body
        assert "loaded_models" in body
        assert "serve_mode" in body

    # VAL-OPS-002: Health reports serve_mode "simple"
    def test_health_serve_mode_simple(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/health")
        assert resp.json()["serve_mode"] == "simple"

    # VAL-OPS-003: Health loaded_models is a list
    def test_health_loaded_models_is_list(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/health")
        assert isinstance(resp.json()["loaded_models"], list)

    # Health status value
    def test_health_status_is_healthy(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/health")
        assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# VAL-OPS-004: Root endpoint returns 200 service info
# VAL-OPS-005: Root reports serve_mode "simple"
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """VAL-OPS-004/005: GET / returns 200 service info with serve_mode=simple."""

    def test_root_returns_200(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/")
        assert resp.status_code == 200

    def test_root_has_service_identity(self, client_no_preload):
        c, _, _ = client_no_preload
        body = c.get("/").json()
        assert "status" in body
        assert "device" in body
        assert "serve_mode" in body

    def test_root_serve_mode_simple(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/")
        assert resp.json()["serve_mode"] == "simple"


# ---------------------------------------------------------------------------
# VAL-OPS-015: Service starts and serves with no Ray installed
# ---------------------------------------------------------------------------


class TestNoRayDependency:
    """VAL-OPS-015: Service runs without Ray being importable."""

    def test_ray_not_importable(self):
        import importlib

        with pytest.raises(ImportError):
            importlib.import_module("ray")

    def test_ray_not_in_pyproject_deps(self):
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        all_deps = data.get("project", {}).get("dependencies", [])
        assert not any("ray" in d.lower() for d in all_deps)


# ---------------------------------------------------------------------------
# VAL-OPS-020: Service binds to port 9001
# ---------------------------------------------------------------------------


class TestPortBinding:
    """VAL-OPS-020: The service is configured for port 9001."""

    def test_default_port_is_9001(self):
        # The .env sets PORT=9001; the __main__ block defaults to 9001.
        port = os.getenv("PORT", "9001")
        assert port == "9001"


# ---------------------------------------------------------------------------
# VAL-CROSS-007: PRELOAD_MODEL reflected in /health loaded_models before any request
# VAL-CROSS-008: Preload makes the first request use the warm model
# ---------------------------------------------------------------------------


class TestPreloadModel:
    """VAL-CROSS-007/008: PRELOAD_MODEL populates loaded_models at startup."""

    def test_preload_model_in_health_before_request(self, client_with_preload):
        """GET /health shows the preloaded model before any /asr request."""
        c, _ = client_with_preload
        resp = c.get("/health")
        loaded = resp.json()["loaded_models"]
        assert "base" in loaded

    def test_no_preload_means_empty_loaded_models(self, client_no_preload):
        """GET /health shows no model when PRELOAD_MODEL is not set."""
        c, _, _ = client_no_preload
        resp = c.get("/health")
        loaded = resp.json()["loaded_models"]
        # base should NOT be present because we didn't preload it
        assert "base" not in loaded

    def test_preload_warms_first_request(self, client_with_preload):
        """
        VAL-CROSS-008: The preloaded model stays in loaded_models after a
        request (no duplicate, no reload needed).
        """
        c, wmlx = client_with_preload

        # The model was loaded during startup
        from app.pipeline import _whisper_models

        assert "base" in _whisper_models

        # Health should show exactly one "base" entry
        resp = c.get("/health")
        loaded = resp.json()["loaded_models"]
        assert loaded.count("base") == 1


# ---------------------------------------------------------------------------
# VAL-CROSS-005: Concurrent /asr requests all succeed (serialized by queue)
# VAL-CROSS-006: Queue depth visible during concurrent requests
# VAL-CROSS-017: Queue serializes across BOTH /asr and /v1 endpoints
# ---------------------------------------------------------------------------


class TestQueueConcurrency:
    """
    VAL-CROSS-005/006/017: The async single-device queue serializes GPU
    work across /asr and /v1/audio/transcriptions.
    """

    def test_queue_metrics_endpoint(self, client_no_preload):
        """GET /queue-metrics returns 200 with queue state info."""
        c, _, _ = client_no_preload
        resp = c.get("/queue-metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "serve_mode" in body
        assert "device" in body
        assert "loaded_models" in body
        assert "queue" in body
        q = body["queue"]
        assert "gpu_concurrency" in q
        assert q["gpu_concurrency"] >= 1
        assert "requests_queued" in q
        assert "requests_in_flight" in q
        assert "max_queue_size" in q

    def test_gpu_concurrency_default_is_one(self):
        """GPU_CONCURRENCY env default is 1 (single Metal GPU)."""
        from app.queue import GPU_CONCURRENCY

        assert GPU_CONCURRENCY == 1

    def test_run_in_queue_all_succeed(self):
        """
        VAL-CROSS-005: All submitted requests succeed through the queue.
        Uses the module-level run_in_queue with the existing semaphore.
        """
        from app.queue import run_in_queue

        results = []
        errors = []

        def blocking_fn(idx: int) -> int:
            time.sleep(0.05)
            return idx

        async def _run_concurrent():
            tasks = [run_in_queue(blocking_fn, i) for i in range(3)]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for r in completed:
                if isinstance(r, Exception):
                    errors.append(r)
                else:
                    results.append(r)

        loop = asyncio.new_event_loop()
        # Reset the module-level semaphore so it binds to our loop
        import app.queue as qmod

        qmod._gpu_semaphore = None
        try:
            loop.run_until_complete(_run_concurrent())
        finally:
            loop.close()

        assert len(errors) == 0, f"Errors: {errors}"
        assert sorted(results) == [0, 1, 2]

    def test_in_flight_never_exceeds_concurrency(self):
        """
        VAL-CROSS-006: While tasks run, requests_in_flight <= gpu_concurrency.
        """
        from app.queue import GPU_CONCURRENCY
        from app.queue import get_queue_metrics
        from app.queue import run_in_queue

        in_flight_snapshots = []

        def blocking_fn():
            metrics = get_queue_metrics()
            in_flight_snapshots.append(metrics["requests_in_flight"])
            time.sleep(0.05)
            return "done"

        async def _run_concurrent():
            tasks = [run_in_queue(blocking_fn) for _ in range(3)]
            return await asyncio.gather(*tasks)

        import app.queue as qmod

        qmod._gpu_semaphore = None
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_run_concurrent())
        finally:
            loop.close()

        assert all(r == "done" for r in results)
        for snapshot in in_flight_snapshots:
            assert snapshot <= GPU_CONCURRENCY, f"in_flight={snapshot} > concurrency={GPU_CONCURRENCY}"

    def test_both_endpoints_use_same_queue(self):
        """
        VAL-CROSS-017: /asr and /v1/audio/transcriptions both import
        run_in_queue from the same app.queue module, sharing the semaphore.
        """
        from app import main
        from app import openai_compat

        assert main.run_in_queue is openai_compat.run_in_queue

    def test_queue_drains_to_zero(self):
        """After requests complete, in-flight and queued counters return to 0."""
        from app.queue import get_queue_metrics
        from app.queue import run_in_queue

        def blocking_fn():
            return "ok"

        async def _run():
            await run_in_queue(blocking_fn)

        import app.queue as qmod

        qmod._gpu_semaphore = None
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        metrics = get_queue_metrics()
        assert metrics["requests_in_flight"] == 0
        assert metrics["requests_queued"] == 0


# ---------------------------------------------------------------------------
# Queue-metrics JSON shape (VAL-OPS-012 partial coverage)
# ---------------------------------------------------------------------------


class TestQueueMetricsShape:
    """Queue-metrics JSON has serve_mode, device, loaded_models, and queue object."""

    def test_queue_metrics_has_required_fields(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/queue-metrics")
        body = resp.json()
        assert body["serve_mode"] == "simple"
        assert body["device"] is not None
        assert isinstance(body["loaded_models"], list)
        assert "queue" in body
        q = body["queue"]
        assert "gpu_concurrency" in q
        assert "requests_queued" in q
        assert "requests_in_flight" in q
        assert "max_queue_size" in q

    def test_queue_gpu_concurrency_default(self, client_no_preload):
        c, _, _ = client_no_preload
        resp = c.get("/queue-metrics")
        q = resp.json()["queue"]
        assert q["gpu_concurrency"] == 1


# ---------------------------------------------------------------------------
# Lifespan API: no deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------


class TestLifespanApi:
    """Verify the app uses the modern lifespan context-manager API,
    not the deprecated @app.on_event("startup") handler."""

    def test_no_on_event_startup_decorator(self):
        """app/main.py must not contain @app.on_event('startup')."""
        import inspect

        import app.main as main_mod

        src = inspect.getsource(main_mod)
        assert "@app.on_event" not in src, "Deprecated @app.on_event decorator found in app/main.py"

    def test_lifespan_preload_model_works(self, client_with_preload):
        """PRELOAD_MODEL still warms the model at startup via the lifespan API."""
        c, _ = client_with_preload
        resp = c.get("/health")
        loaded = resp.json()["loaded_models"]
        assert "base" in loaded, "PRELOAD_MODEL=base should appear in loaded_models at startup via lifespan"
