"""
Unit tests for idle model eviction and loaded_models tracking covering
VAL-OPS-029, VAL-OPS-030, VAL-CROSS-015.

All tests mock whispermlx so no real model downloads occur.

CRITICAL: these tests exercise the REAL eviction code path by calling
``pipeline._run_eviction_sweep()`` directly.  They do NOT duplicate the
eviction-sweep logic inline, so if the eviction function changes the tests
will fail rather than false-pass.
"""

import io
import os
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
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
    mock_module.load_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_module.assign_word_speakers.side_effect = lambda diar, result: result
    return mock_module


FAKE_AUDIO = io.BytesIO(b"\x00" * 1024)


def _post_asr(client, model="tiny", params=None, file_content=None):
    """Helper to POST to /asr with optional query params."""
    if file_content is None:
        file_content = FAKE_AUDIO
    url = f"/asr?model={model}"
    if params:
        url += "&" + "&".join(f"{k}={v}" for k, v in params.items())
    return client.post(
        url,
        files={"audio_file": ("test.wav", file_content, "audio/wav")},
    )


def _extract_counter_value(body: str, metric_name: str, labels: dict[str, str]) -> float:
    """Extract a Prometheus counter sample value from exposition text."""
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for name_variant in (f"{metric_name}_total", metric_name):
            prefix = f"{name_variant}{{{label_str}}}"
            if stripped.startswith(prefix):
                parts = stripped.split()
                if len(parts) >= 2:
                    return float(parts[-1])
    return 0.0


def _reset_pipeline(pipe_mod, keep_alive_seconds=1):
    """Clear pipeline caches and set eviction config for testing."""
    pipe_mod.MODEL_KEEP_ALIVE_SECONDS = keep_alive_seconds
    pipe_mod._whisper_models.clear()
    pipe_mod._whisper_models_last_used.clear()
    pipe_mod._eviction_thread_started = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_preload():
    """
    TestClient with whispermlx mocked, no PRELOAD_MODEL, and run_in_queue
    NOT mocked so the real pipeline code path adds models to the cache.
    """
    wmlx = _make_whispermlx_mock()
    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
        patch("app.openai_compat.whispermlx", wmlx),
        patch.dict(
            os.environ,
            {
                "PRELOAD_MODEL": "",
                "DEVICE": "cpu",
                "SERVE_MODE": "simple",
                "MODEL_KEEP_ALIVE_SECONDS": "0",
            },
            clear=False,
        ),
    ):
        import app.pipeline as pipe_mod

        _reset_pipeline(pipe_mod, keep_alive_seconds=0)

        from app.main import app

        with TestClient(app) as c:
            yield c, wmlx


@pytest.fixture()
def client_with_eviction():
    """
    TestClient with MODEL_KEEP_ALIVE_SECONDS=1 and a short sweep interval
    so eviction happens quickly.  The eviction thread is not started by
    default (it's lazy); tests trigger it manually via _run_eviction_sweep().
    """
    wmlx = _make_whispermlx_mock()
    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
        patch("app.openai_compat.whispermlx", wmlx),
        patch.dict(
            os.environ,
            {
                "PRELOAD_MODEL": "",
                "DEVICE": "cpu",
                "SERVE_MODE": "simple",
                "MODEL_KEEP_ALIVE_SECONDS": "1",
                "MODEL_EVICTION_INTERVAL_SECONDS": "30",
            },
            clear=False,
        ),
    ):
        import app.pipeline as pipe_mod

        _reset_pipeline(pipe_mod, keep_alive_seconds=1)
        pipe_mod.MODEL_EVICTION_INTERVAL_SECONDS = max(30, int(os.getenv("MODEL_EVICTION_INTERVAL_SECONDS", "30")))

        from app.main import app

        with TestClient(app) as c:
            yield c, wmlx


# ---------------------------------------------------------------------------
# VAL-OPS-029: loaded_models reflects a model cached by a request
# ---------------------------------------------------------------------------


class TestLoadedModelsReflectsRequest:
    """
    VAL-OPS-029: After a request loads a model, /health loaded_models lists it.

    Before any transcription the model is absent; after a successful POST /asr
    it appears in loaded_models.
    """

    def test_tiny_absent_before_request(self, client_no_preload):
        """GET /health before any request does NOT list 'tiny' in loaded_models."""
        c, _ = client_no_preload
        resp = c.get("/health")
        assert resp.status_code == 200
        loaded = resp.json()["loaded_models"]
        assert "tiny" not in loaded

    def test_tiny_present_after_asr_request(self, client_no_preload):
        """
        After a successful POST /asr with model=tiny, GET /health lists 'tiny'
        in loaded_models (the in-process MLX cache is populated by the request).
        """
        c, _ = client_no_preload

        # Make a request that loads the 'tiny' model
        resp = _post_asr(c, model="tiny")
        assert resp.status_code == 200

        # Now /health should list 'tiny'
        health = c.get("/health")
        loaded = health.json()["loaded_models"]
        assert "tiny" in loaded

    def test_different_models_tracked(self, client_no_preload):
        """
        Two requests with different models both appear in loaded_models.
        """
        c, _ = client_no_preload

        resp1 = _post_asr(c, model="tiny")
        assert resp1.status_code == 200

        resp2 = _post_asr(c, model="base")
        assert resp2.status_code == 200

        health = c.get("/health")
        loaded = health.json()["loaded_models"]
        assert "tiny" in loaded
        assert "base" in loaded


# ---------------------------------------------------------------------------
# VAL-OPS-030: Idle model eviction unloads cached models end-to-end
# ---------------------------------------------------------------------------


class TestIdleModelEviction:
    """
    VAL-OPS-030: With MODEL_KEEP_ALIVE_SECONDS set, an idle model is evicted,
    whisperx_model_evictions_total increments, and the model disappears from
    loaded_models.

    These tests call the REAL ``_run_eviction_sweep()`` function rather than
    duplicating the eviction logic inline, so they exercise the actual
    pipeline eviction code path and cannot false-pass if that function changes.
    """

    def test_eviction_removes_model_from_cache(self):
        """
        A model idle past the keep-alive window is removed from the cache
        when ``_run_eviction_sweep()`` is called.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch.dict(os.environ, {"MODEL_KEEP_ALIVE_SECONDS": "1"}, clear=False),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            # Load a model
            pipe_mod.load_whisper_model("tiny")
            assert "tiny" in pipe_mod._whisper_models

            # Simulate the model being idle: set last_used far in the past
            pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

            # Call the REAL eviction sweep function
            evicted = pipe_mod._run_eviction_sweep()

            # Model should be evicted
            assert evicted is True
            assert "tiny" not in pipe_mod._whisper_models
            assert "tiny" not in pipe_mod._whisper_models_last_used

    def test_eviction_counter_increments(self):
        """
        After ``_run_eviction_sweep()`` evicts a model, the
        whisperx_model_evictions_total counter increments.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch.dict(os.environ, {"MODEL_KEEP_ALIVE_SECONDS": "1"}, clear=False),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app import metrics as prom_metrics

            # Load the model (pre-registers counter)
            pipe_mod.load_whisper_model("tiny")
            assert "tiny" in pipe_mod._whisper_models

            # Simulate idle by setting last_used far in the past
            pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

            before = prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model="tiny")._value.get()

            # Call the REAL eviction sweep function
            pipe_mod._run_eviction_sweep()

            after = prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model="tiny")._value.get()
            assert after > before
            assert after >= 1

    def test_health_no_longer_lists_evicted_model(self):
        """
        After ``_run_eviction_sweep()`` evicts a model, /health loaded_models
        no longer lists it.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app.main import app

            with TestClient(app) as c:
                # Load tiny via a request
                resp = _post_asr(c, model="tiny")
                assert resp.status_code == 200

                # Verify it's in loaded_models
                health = c.get("/health")
                assert "tiny" in health.json()["loaded_models"]

                # Simulate idle: set last_used far in the past
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

                # Call the REAL eviction sweep function
                pipe_mod._run_eviction_sweep()

                # Now /health should NOT list tiny
                health2 = c.get("/health")
                assert "tiny" not in health2.json()["loaded_models"]

    def test_eviction_counter_in_metrics_output(self):
        """
        After ``_run_eviction_sweep()`` evicts a model, /metrics exposes
        whisperx_model_evictions_total{model="tiny"} >= 1.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app.main import app

            with TestClient(app) as c:
                # Load tiny via a request
                _post_asr(c, model="tiny")

                # Simulate idle and run the REAL eviction sweep
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                pipe_mod._run_eviction_sweep()

                # Check /metrics output
                metrics_resp = c.get("/metrics")
                body = metrics_resp.text
                value = _extract_counter_value(
                    body,
                    "whisperx_model_evictions_total",
                    {"model": "tiny"},
                )
                assert value >= 1


# ---------------------------------------------------------------------------
# VAL-CROSS-015: After eviction, next request reloads transparently
# ---------------------------------------------------------------------------


class TestTransparentReloadAfterEviction:
    """
    VAL-CROSS-015: After eviction, a subsequent request with the same model
    still returns HTTP 200 (the evicted model reloads transparently) and the
    model reappears in /health loaded_models.

    These tests call the REAL ``_run_eviction_sweep()`` function to evict
    models, exercising the actual pipeline eviction code path.
    """

    def test_request_succeeds_after_eviction(self):
        """
        A POST /asr with model=tiny succeeds after the model was evicted
        by ``_run_eviction_sweep()``.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app.main import app

            with TestClient(app) as c:
                # First request loads tiny
                resp1 = _post_asr(c, model="tiny")
                assert resp1.status_code == 200
                assert "tiny" in c.get("/health").json()["loaded_models"]

                # Simulate idle and evict using the REAL sweep function
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                pipe_mod._run_eviction_sweep()

                # Verify eviction happened
                assert "tiny" not in c.get("/health").json()["loaded_models"]

                # Second request with same model should reload transparently
                resp2 = _post_asr(c, model="tiny")
                assert resp2.status_code == 200

    def test_model_reappears_in_loaded_models_after_reload(self):
        """
        After eviction and a successful reload request, /health loaded_models
        lists the model again.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app.main import app

            with TestClient(app) as c:
                # Load, evict, then reload
                _post_asr(c, model="tiny")
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100

                # Evict using the REAL sweep function
                pipe_mod._run_eviction_sweep()

                # tiny is evicted
                assert "tiny" not in c.get("/health").json()["loaded_models"]

                # Re-request
                resp = _post_asr(c, model="tiny")
                assert resp.status_code == 200

                # tiny reappears
                loaded = c.get("/health").json()["loaded_models"]
                assert "tiny" in loaded

    def test_eviction_counter_cumulative_after_multiple_evictions(self):
        """
        Evicting the same model twice via ``_run_eviction_sweep()`` increments
        the counter to >= 2.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app.main import app

            with TestClient(app) as c:
                # First load + evict cycle
                _post_asr(c, model="tiny")
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                pipe_mod._run_eviction_sweep()

                # Second load + evict cycle
                _post_asr(c, model="tiny")
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                pipe_mod._run_eviction_sweep()

                # Counter should be >= 2
                metrics_body = c.get("/metrics").text
                value = _extract_counter_value(
                    metrics_body,
                    "whisperx_model_evictions_total",
                    {"model": "tiny"},
                )
                assert value >= 2

    def test_non_evicted_model_stays_loaded(self):
        """
        Only idle models are evicted; recently-used models stay in cache.
        ``_run_eviction_sweep()`` should only evict models past the keep-alive.
        """
        wmlx = _make_whispermlx_mock()
        with (
            patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
            patch("app.main.whispermlx", wmlx),
            patch("app.openai_compat.whispermlx", wmlx),
            patch.dict(
                os.environ,
                {
                    "PRELOAD_MODEL": "",
                    "DEVICE": "cpu",
                    "SERVE_MODE": "simple",
                    "MODEL_KEEP_ALIVE_SECONDS": "1",
                },
                clear=False,
            ),
        ):
            import app.pipeline as pipe_mod

            _reset_pipeline(pipe_mod, keep_alive_seconds=1)

            from app.main import app

            with TestClient(app) as c:
                # Load both models
                _post_asr(c, model="tiny")
                _post_asr(c, model="base")

                # Make tiny idle (set last_used far in the past)
                pipe_mod._whisper_models_last_used["tiny"] = time.time() - 100
                # base is recent (default from load_whisper_model, should be current)

                # Run the REAL eviction sweep
                pipe_mod._run_eviction_sweep()

                # tiny evicted, base stays
                loaded = c.get("/health").json()["loaded_models"]
                assert "tiny" not in loaded
                assert "base" in loaded
