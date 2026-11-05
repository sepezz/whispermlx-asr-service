"""
Unit tests for the /metrics endpoint, VRAM gauge, service_info labels,
and request-counter behaviour covering VAL-OPS-006..012, VAL-OPS-019,
VAL-OPS-032, VAL-OPS-033, VAL-CROSS-016.

All tests mock whispermlx so no real model downloads occur.
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
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
    # Return a real numpy array so diarization code paths don't crash
    mock_module.load_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_module.assign_word_speakers.side_effect = lambda diar, result: result
    return mock_module


FAKE_AUDIO = io.BytesIO(b"\x00" * 1024)


@pytest.fixture()
def client():
    """
    TestClient with SERVE_MODE=simple, whispermlx mocked, and
    run_in_queue mocked to return a controlled pipeline result.
    """
    wmlx = _make_whispermlx_mock()
    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
        patch("app.openai_compat.whispermlx", wmlx),
        patch("app.main.run_in_queue") as mock_queue,
        patch("app.openai_compat.run_in_queue") as mock_oai_queue,
        patch.dict(os.environ, {"PRELOAD_MODEL": "", "DEVICE": "cpu", "SERVE_MODE": "simple"}, clear=False),
    ):
        from app.pipeline import _whisper_models
        from app.pipeline import _whisper_models_last_used

        _whisper_models.clear()
        _whisper_models_last_used.clear()

        from app.main import app

        # Default pipeline result for /asr
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "word_segments": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_queue.side_effect = _return_result
        mock_oai_queue.side_effect = _return_result

        with TestClient(app) as c:
            yield c, wmlx, mock_queue, mock_oai_queue


@pytest.fixture()
def client_with_preload():
    """TestClient with PRELOAD_MODEL=base so startup_event loads a model."""
    wmlx = _make_whispermlx_mock()

    with (
        patch("app.main.whispermlx", wmlx),
        patch("app.pipeline.whispermlx", wmlx),
        patch("app.pipeline.FullAudioWhisper", wmlx.load_model),
        patch("app.openai_compat.whispermlx", wmlx),
        patch.dict(os.environ, {"PRELOAD_MODEL": "base", "DEVICE": "cpu", "SERVE_MODE": "simple"}, clear=False),
    ):
        from app.pipeline import _whisper_models
        from app.pipeline import _whisper_models_last_used

        _whisper_models.clear()
        _whisper_models_last_used.clear()

        from app.main import app

        with TestClient(app) as c:
            yield c, wmlx


def _post_asr(client, params=None, file_content=None):
    """Helper to POST to /asr with optional query params."""
    if file_content is None:
        file_content = FAKE_AUDIO
    url = "/asr"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return client.post(
        url,
        files={"audio_file": ("test.wav", file_content, "audio/wav")},
    )


# ---------------------------------------------------------------------------
# VAL-OPS-006: /metrics returns 200 OpenMetrics text
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """VAL-OPS-006: GET /metrics returns 200 OpenMetrics text."""

    def test_metrics_returns_200(self, client):
        c, _, _, _ = client
        resp = c.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_text(self, client):
        c, _, _, _ = client
        resp = c.get("/metrics")
        ct = resp.headers.get("content-type", "")
        assert ct.startswith("text/plain")

    def test_metrics_body_nonempty(self, client):
        c, _, _, _ = client
        resp = c.get("/metrics")
        assert len(resp.text) > 0


# ---------------------------------------------------------------------------
# VAL-OPS-007: /metrics exposes whisperx_requests_total
# ---------------------------------------------------------------------------


class TestMetricsRequestsTotal:
    """VAL-OPS-007: whisperx_requests_total present in metrics output."""

    def test_requests_total_type_line(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        assert "# TYPE whisperx_requests_total counter" in body

    def test_requests_total_has_labels_after_request(self, client):
        """After a request, counter samples carry endpoint and status labels."""
        c, _, _, _ = client
        # Make a request first so the counter has been observed
        _post_asr(c, params={"output_format": "text"})
        body = c.get("/metrics").text
        # prometheus_client appends _total to counter names in exposition
        assert "whisperx_requests_total_total{endpoint=" in body or "whisperx_requests_total{endpoint=" in body


# ---------------------------------------------------------------------------
# VAL-OPS-008: VRAM gauge reports MLX active memory (or 0)
# ---------------------------------------------------------------------------


class TestMetricsVRAMGauge:
    """VAL-OPS-008: whisperx_vram_allocated_bytes gauge present, >= 0."""

    def test_vram_gauge_type_line(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        assert "# TYPE whisperx_vram_allocated_bytes gauge" in body

    def test_vram_gauge_has_numeric_value(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        for line in body.splitlines():
            if line.startswith("whisperx_vram_allocated_bytes ") and not line.startswith("#"):
                value = float(line.split()[-1])
                assert value >= 0
                return
        pytest.fail("whisperx_vram_allocated_bytes sample line not found")


# ---------------------------------------------------------------------------
# VAL-OPS-009 / VAL-OPS-019: No torch.cuda in metrics path
# ---------------------------------------------------------------------------


class TestNoTorchCudaInMetrics:
    """VAL-OPS-009/019: Metrics endpoint works without torch.cuda; source has no torch.cuda."""

    def test_metrics_endpoint_no_cuda_error(self, client):
        """Hitting /metrics does not produce a 500 from missing torch.cuda."""
        c, _, _, _ = client
        resp = c.get("/metrics")
        assert resp.status_code == 200

    def test_no_torch_cuda_in_metrics_source(self):
        """Source file app/metrics.py contains no torch.cuda reference."""
        src = Path(__file__).resolve().parents[2] / "app" / "metrics.py"
        content = src.read_text()
        assert "torch.cuda" not in content

    def test_no_torch_cuda_in_app_source(self):
        """No torch.cuda reference exists in active app/ code (VAL-OPS-023).
        Excludes serve_app.py / serve_deployments.py which are dead Ray
        modules removed by a separate feature."""
        app_dir = Path(__file__).resolve().parents[2] / "app"
        excluded = {"serve_app.py", "serve_deployments.py"}
        for py_file in app_dir.glob("*.py"):
            if py_file.name in excluded:
                continue
            content = py_file.read_text()
            assert "torch.cuda" not in content, f"torch.cuda found in {py_file}"

    def test_refresh_vram_uses_mlx_gracefully(self):
        """refresh_vram() does not crash when mlx.core.get_active_memory is missing."""
        from app.metrics import refresh_vram

        # mlx.core may or may not have get_active_memory; either way,
        # refresh_vram should not raise.
        refresh_vram()  # should not raise

    def test_refresh_vram_sets_gauge_on_success(self):
        """When mlx.core.get_active_memory exists, VRAM gauge is set to its return value.

        This test patches ``mlx.core.get_active_memory`` to return a known value
        and then asserts that the ``whisperx_vram_allocated_bytes`` gauge actually
        reflects that value after calling ``refresh_vram()``.  Previously this test
        was a vacuous pass: it patched the function but never checked the gauge.
        """
        import mlx.core
        from prometheus_client import generate_latest

        from app.metrics import refresh_vram

        original_fn = getattr(mlx.core, "get_active_memory", None)
        patched_value = 12345678
        try:
            # Patch get_active_memory to return a known value
            mlx.core.get_active_memory = MagicMock(return_value=patched_value)
            refresh_vram()
        finally:
            if original_fn is not None:
                mlx.core.get_active_memory = original_fn
            elif hasattr(mlx.core, "get_active_memory"):
                delattr(mlx.core, "get_active_memory")

        # Assert the gauge was actually set to the patched value.
        # We read the Prometheus exposition output and parse the gauge sample.
        body = generate_latest().decode("utf-8")
        found = False
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("whisperx_vram_allocated_bytes ") or stripped.startswith(
                "whisperx_vram_allocated_bytes{"
            ):
                value = float(stripped.split()[-1])
                assert value == patched_value, (
                    f"VRAM gauge was {value}, expected {patched_value} "
                    f"(the patched mlx.core.get_active_memory return value)"
                )
                found = True
                break
        assert found, "whisperx_vram_allocated_bytes sample line not found in metrics output"


# ---------------------------------------------------------------------------
# VAL-OPS-010: service_info reports serve_mode=simple and non-CUDA device
# ---------------------------------------------------------------------------


class TestServiceInfoLabels:
    """VAL-OPS-010: service_info has serve_mode=simple and non-CUDA device."""

    def test_service_info_present(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        assert "whisperx_service_info" in body

    def test_service_info_serve_mode_simple(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        assert 'serve_mode="simple"' in body

    def test_service_info_device_not_cuda(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        for line in body.splitlines():
            if "whisperx_service_info" in line and not line.startswith("#"):
                assert 'device="cuda"' not in line
                assert "device=" in line
                return
        pytest.fail("whisperx_service_info sample line not found")


# ---------------------------------------------------------------------------
# VAL-OPS-011: A served request increments whisperx_requests_total
# ---------------------------------------------------------------------------


class TestRequestsTotalIncrements:
    """VAL-OPS-011: A served /asr request increments the counter."""

    def test_asr_request_increments_requests_total(self, client):
        c, _, _, _ = client

        # Make a successful /asr request
        resp = _post_asr(c, params={"output_format": "text"})
        assert resp.status_code == 200

        # Scrape metrics
        body = c.get("/metrics").text
        ok_value = _extract_counter_value(body, "whisperx_requests_total", {"endpoint": "/asr", "status": "ok"})
        assert ok_value >= 1

    def test_counter_increments_per_request(self, client):
        c, _, _, _ = client

        _post_asr(c, params={"output_format": "text"})
        body1 = c.get("/metrics").text
        ok1 = _extract_counter_value(body1, "whisperx_requests_total", {"endpoint": "/asr", "status": "ok"})

        _post_asr(c, params={"output_format": "text"})
        body2 = c.get("/metrics").text
        ok2 = _extract_counter_value(body2, "whisperx_requests_total", {"endpoint": "/asr", "status": "ok"})

        assert ok2 > ok1


# ---------------------------------------------------------------------------
# VAL-OPS-012: /queue-metrics returns queue-state JSON
# ---------------------------------------------------------------------------


class TestQueueMetricsEndpoint:
    """VAL-OPS-012: /queue-metrics returns JSON with queue state."""

    def test_queue_metrics_returns_200(self, client):
        c, _, _, _ = client
        resp = c.get("/queue-metrics")
        assert resp.status_code == 200

    def test_queue_metrics_serve_mode_simple(self, client):
        c, _, _, _ = client
        body = c.get("/queue-metrics").json()
        assert body["serve_mode"] == "simple"

    def test_queue_metrics_has_queue_object(self, client):
        c, _, _, _ = client
        body = c.get("/queue-metrics").json()
        assert "queue" in body
        q = body["queue"]
        assert "requests_queued" in q
        assert "requests_in_flight" in q
        assert "gpu_concurrency" in q
        assert "max_queue_size" in q


# ---------------------------------------------------------------------------
# VAL-OPS-032: Failed requests recorded under distinct http_4xx label
# ---------------------------------------------------------------------------


class TestFailedRequestsTracking:
    """VAL-OPS-032: Failed requests tracked under http_4xx status label."""

    def test_bad_output_format_increments_http_400(self, client):
        c, _, _, _ = client

        # Baseline
        body_before = c.get("/metrics").text
        http_400_before = _extract_counter_value(
            body_before, "whisperx_requests_total", {"endpoint": "/asr", "status": "http_400"}
        )

        # Send a request with invalid output_format → 400
        resp = _post_asr(c, params={"output_format": "docx"})
        assert resp.status_code == 400

        # Post-request scrape
        body_after = c.get("/metrics").text
        http_400_after = _extract_counter_value(
            body_after, "whisperx_requests_total", {"endpoint": "/asr", "status": "http_400"}
        )

        assert http_400_after > http_400_before

    def test_ok_counter_unchanged_on_failure(self, client):
        c, _, _, _ = client

        _post_asr(c, params={"output_format": "text"})
        body_before = c.get("/metrics").text
        ok_before = _extract_counter_value(body_before, "whisperx_requests_total", {"endpoint": "/asr", "status": "ok"})

        # Bad request
        _post_asr(c, params={"output_format": "docx"})

        body_after = c.get("/metrics").text
        ok_after = _extract_counter_value(body_after, "whisperx_requests_total", {"endpoint": "/asr", "status": "ok"})

        assert ok_after == ok_before


# ---------------------------------------------------------------------------
# VAL-OPS-033: Active transcriptions gauge returns to 0 at idle
# ---------------------------------------------------------------------------


class TestActiveTranscriptionsGauge:
    """VAL-OPS-033: whisperx_active_transcriptions returns to 0 at idle."""

    def test_gauge_is_zero_at_idle(self, client):
        c, _, _, _ = client
        body = c.get("/metrics").text
        value = _extract_gauge_value(body, "whisperx_active_transcriptions")
        assert value == 0.0

    def test_gauge_returns_to_zero_after_request(self, client):
        c, _, _, _ = client

        # Make a request
        _post_asr(c, params={"output_format": "text"})

        # After the request completes, gauge should be back to 0
        body = c.get("/metrics").text
        value = _extract_gauge_value(body, "whisperx_active_transcriptions")
        assert value == 0.0


# ---------------------------------------------------------------------------
# VAL-CROSS-016: PRELOAD_MODEL reflected in loaded_models gauge before any request
# ---------------------------------------------------------------------------


class TestPreloadModelGauge:
    """VAL-CROSS-016: whisperx_loaded_models gauge >= 1 at startup with PRELOAD_MODEL."""

    def test_gauge_reflects_preload_before_request(self, client_with_preload):
        c, _ = client_with_preload
        body = c.get("/metrics").text
        value = _extract_gauge_value(body, "whisperx_loaded_models")
        # With PRELOAD_MODEL=base, at least 1 model should be loaded
        assert value >= 1

    def test_gauge_zero_without_preload(self, client):
        c, _, _, _ = client
        # Before any request and without PRELOAD_MODEL, gauge should be 0
        body = c.get("/metrics").text
        value = _extract_gauge_value(body, "whisperx_loaded_models")
        assert value == 0.0


# ---------------------------------------------------------------------------
# OpenAI endpoints should also track metrics
# ---------------------------------------------------------------------------


class TestOpenAIEndpointMetrics:
    """Verify /v1/audio/transcriptions and /v1/audio/translations track metrics."""

    def test_transcriptions_increments_requests_total(self, client):
        c, _, _, mock_oai_queue = client

        # Need a pipeline result with segments
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "word_segments": [],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_oai_queue.side_effect = _return_result

        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
            data={"model": "whisper-1"},
        )

        body = c.get("/metrics").text
        ok_value = _extract_counter_value(
            body, "whisperx_requests_total", {"endpoint": "/v1/audio/transcriptions", "status": "ok"}
        )
        if resp.status_code == 200:
            assert ok_value >= 1

    def test_transcriptions_bad_model_increments_http_400(self, client):
        c, _, _, _ = client

        body_before = c.get("/metrics").text
        http_400_before = _extract_counter_value(
            body_before, "whisperx_requests_total", {"endpoint": "/v1/audio/transcriptions", "status": "http_400"}
        )

        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
            data={"model": "gpt-4o"},
        )
        assert resp.status_code == 400

        body_after = c.get("/metrics").text
        http_400_after = _extract_counter_value(
            body_after, "whisperx_requests_total", {"endpoint": "/v1/audio/transcriptions", "status": "http_400"}
        )

        assert http_400_after > http_400_before

    def test_translations_increments_requests_total(self, client):
        c, _, _, mock_oai_queue = client

        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "word_segments": [],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_oai_queue.side_effect = _return_result

        resp = c.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
            data={"model": "whisper-1"},
        )

        body = c.get("/metrics").text
        ok_value = _extract_counter_value(
            body, "whisperx_requests_total", {"endpoint": "/v1/audio/translations", "status": "ok"}
        )
        if resp.status_code == 200:
            assert ok_value >= 1

    def test_transcriptions_active_gauge_returns_to_zero(self, client):
        c, _, _, mock_oai_queue = client

        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "word_segments": [],
            "language": "en",
        }

        async def _return_result(*args, **kwargs):
            return result, None

        mock_oai_queue.side_effect = _return_result

        c.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", FAKE_AUDIO, "audio/wav")},
            data={"model": "whisper-1"},
        )

        body = c.get("/metrics").text
        value = _extract_gauge_value(body, "whisperx_active_transcriptions")
        assert value == 0.0


# ---------------------------------------------------------------------------
# Helper: parse Prometheus exposition text
# ---------------------------------------------------------------------------


def _extract_counter_value(body: str, metric_name: str, labels: dict[str, str]) -> float:
    """
    Extract the numeric value of a Prometheus counter sample from exposition text.

    Matches lines like:
      whisperx_requests_total{endpoint="/asr",status="ok"} 5.0
    or (with _total suffix for counters):
      whisperx_requests_total_total{endpoint="/asr",status="ok"} 5.0
    """
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # prometheus_client appends _total to counter names in exposition
        for name_variant in (f"{metric_name}_total", metric_name):
            prefix = f"{name_variant}{{{label_str}}}"
            if stripped.startswith(prefix):
                parts = stripped.split()
                if len(parts) >= 2:
                    return float(parts[-1])
    # Not found — counter hasn't been observed yet, return 0
    return 0.0


def _extract_gauge_value(body: str, metric_name: str) -> float:
    """
    Extract the numeric value of a Prometheus gauge from exposition text.

    Matches lines like:
      whisperx_active_transcriptions 0.0
    or:
      whisperx_active_transcriptions{} 0.0
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Match: metric_name <value> or metric_name{...} <value>
        if stripped.startswith(metric_name) and not stripped.startswith(metric_name + "_"):
            # Ensure we're not matching a different metric that starts with the same prefix
            after_name = stripped[len(metric_name) :]
            if after_name and after_name[0] not in (" ", "{"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                return float(parts[-1])
    pytest.fail(f"Gauge {metric_name} not found in metrics body")
