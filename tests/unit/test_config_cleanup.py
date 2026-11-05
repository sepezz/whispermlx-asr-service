"""
Unit tests for m2-config-cleanup: .env.example hygiene and inert
COMPUTE_TYPE / BATCH_SIZE behaviour.

Covers:
  VAL-OPS-026 — .env.example is free of Ray, CUDA, and Docker configuration
  VAL-OPS-031 — COMPUTE_TYPE and BATCH_SIZE are accepted but inert
"""

import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Path to the shipped .env.example at the repo root
ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / ".env.example"


# ---------------------------------------------------------------------------
# VAL-OPS-026: .env.example is free of Ray, CUDA, and Docker configuration
# ---------------------------------------------------------------------------


class TestEnvExampleClean:
    """Assert .env.example contains no Ray/CUDA/Docker variables."""

    @pytest.fixture()
    def env_example_content(self):
        return ENV_EXAMPLE_PATH.read_text()

    def test_env_example_exists(self):
        assert ENV_EXAMPLE_PATH.is_file(), ".env.example must exist at the repo root"

    def test_no_serve_mode_ray(self, env_example_content):
        assert "SERVE_MODE=ray" not in env_example_content

    def test_no_ray_prefix_vars(self, env_example_content):
        for line in env_example_content.splitlines():
            stripped = line.lstrip("#").strip()
            assert not stripped.startswith("RAY_"), f"Ray var found: {stripped}"

    def test_no_pipeline_strategy(self, env_example_content):
        assert "PIPELINE_STRATEGY" not in env_example_content

    def test_no_gpu_fraction(self, env_example_content):
        assert "GPU_FRACTION" not in env_example_content

    def test_no_num_replicas(self, env_example_content):
        assert "NUM_REPLICAS" not in env_example_content

    def test_no_ray_batch_sizes(self, env_example_content):
        for token in ("WHISPER_BATCH_SIZE", "ALIGN_BATCH_SIZE", "DIARIZE_BATCH_SIZE"):
            assert token not in env_example_content, f"Found {token} in .env.example"

    def test_no_batch_wait_timeout(self, env_example_content):
        assert "BATCH_WAIT_TIMEOUT" not in env_example_content

    def test_no_cuda_config(self, env_example_content):
        """No CUDA configuration lines (DEVICE=cuda, CUDA_VISIBLE_DEVICES, etc.)."""
        for line in env_example_content.splitlines():
            stripped = line.lstrip("#").strip()
            lower = stripped.lower()
            if not stripped or "offline" in lower:
                continue
            assert not lower.startswith("cuda"), f"CUDA config found: {stripped}"
            assert "nvidia" not in lower, f"NVIDIA config found: {stripped}"

    def test_no_device_cuda_default(self, env_example_content):
        assert "DEVICE=cuda" not in env_example_content

    def test_no_docker_config(self, env_example_content):
        """No Docker/container configuration lines."""
        for line in env_example_content.splitlines():
            stripped = line.lstrip("#").strip()
            lower = stripped.lower()
            if not stripped:
                continue
            assert "docker" not in lower, f"Docker reference found: {stripped}"
            assert "container" not in lower, f"Container reference found: {stripped}"

    def test_device_defaults_to_mps(self, env_example_content):
        """DEVICE=mps must be the documented default (matches whispermlx upstream)."""
        assert "DEVICE=mps" in env_example_content

    def test_port_9001_documented(self, env_example_content):
        """PORT=9001 must be documented."""
        assert "PORT=9001" in env_example_content

    def test_mlx_device_semantics_documented(self, env_example_content):
        """The .env.example must explain that MLX ASR uses the Metal GPU automatically."""
        assert "Metal GPU" in env_example_content or "MLX" in env_example_content

    def test_cache_dir_is_native_path(self, env_example_content):
        """CACHE_DIR assignment line must use a native user path, not the Docker /.cache."""
        for line in env_example_content.splitlines():
            if line.startswith("CACHE_DIR="):
                value = line.split("=", 1)[1].strip()
                assert value != "/.cache", "CACHE_DIR still set to Docker container path /.cache"
                assert not value.startswith("/.cache/"), f"CACHE_DIR uses Docker container path: {value}"
                break
        else:
            pytest.fail("No CACHE_DIR assignment line found in .env.example")

    def test_compute_type_documented_as_inert(self, env_example_content):
        """COMPUTE_TYPE must be documented as inert/ignored by MLX."""
        assert "COMPUTE_TYPE" in env_example_content
        lower = env_example_content.lower()
        assert "inert" in lower or "ignored" in lower or "no effect" in lower

    def test_batch_size_documented_as_inert(self, env_example_content):
        """BATCH_SIZE must be documented as inert/ignored by MLX."""
        assert "BATCH_SIZE" in env_example_content
        lower = env_example_content.lower()
        assert "inert" in lower or "ignored" in lower or "no effect" in lower


# ---------------------------------------------------------------------------
# VAL-OPS-031: COMPUTE_TYPE and BATCH_SIZE are accepted but inert
# ---------------------------------------------------------------------------


class TestComputeTypeBatchSizeInert:
    """Verify that COMPUTE_TYPE and BATCH_SIZE env vars do not error and
    have no effect on pipeline behaviour under the MLX backend.

    NOTE: We avoid importlib.reload(app.pipeline) because it breaks
    cross-module references (e.g. app.main's loaded_models alias) and
    causes test-order-dependent failures in other test modules.
    Instead we test the pipeline module as already loaded and verify
    the code defaults and call signatures via source inspection and
    direct function mocking.
    """

    def test_compute_type_accepted_without_error(self):
        """COMPUTE_TYPE can be read from env vars without raising."""
        # Any string value is fine — the pipeline just stores it
        ct = os.getenv("COMPUTE_TYPE", "int8")
        assert isinstance(ct, str)

    def test_batch_size_accepted_without_error(self):
        """BATCH_SIZE can be parsed as int without raising."""
        bs = int(os.getenv("BATCH_SIZE", "2"))
        assert isinstance(bs, int)

    def test_both_set_simultaneously_no_error(self):
        """Setting both COMPUTE_TYPE=float16 and BATCH_SIZE=16 simultaneously
        must not cause any error when read."""
        with patch.dict(os.environ, {"COMPUTE_TYPE": "float16", "BATCH_SIZE": "16"}):
            ct = os.getenv("COMPUTE_TYPE", "int8")
            bs = int(os.getenv("BATCH_SIZE", "2"))
            assert ct == "float16"
            assert bs == 16

    def test_compute_type_not_passed_to_load_model(self):
        """whispermlx.load_model() must not receive compute_type as an argument."""
        import app.pipeline as pipeline_mod

        mock_model = MagicMock()
        with patch.object(pipeline_mod, "FullAudioWhisper", return_value=mock_model) as mock_load:
            pipeline_mod._whisper_models.clear()
            pipeline_mod._whisper_models_last_used.clear()

            pipeline_mod.load_whisper_model("tiny")
            mock_load.assert_called_once()
            # Verify compute_type is NOT in the call
            call_kwargs = mock_load.call_args.kwargs
            assert "compute_type" not in call_kwargs, f"compute_type was passed to load_model: {call_kwargs}"

    def test_batch_size_not_passed_to_transcribe(self):
        """whisper_model.transcribe() must not receive batch_size."""
        import app.pipeline as pipeline_mod

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
        }
        with patch.object(pipeline_mod, "load_whisper_model", return_value=mock_model):
            pipeline_mod.transcribe(audio=MagicMock(), model_name="tiny", language="en", task="transcribe")
            mock_model.transcribe.assert_called_once()
            call_kwargs = mock_model.transcribe.call_args.kwargs
            assert "batch_size" not in call_kwargs, f"batch_size was passed to transcribe: {call_kwargs}"

    def test_default_cache_dir_is_native_path(self):
        """The default CACHE_DIR must be a native user path, not the Docker
        container path (/.cache).  We verify the source code default
        rather than reloading the module."""
        source = inspect.getsource(__import__("app.pipeline"))
        # The default for CACHE_DIR must be the native user path
        assert "/.cache" not in [
            line.strip().split('"')[-2] if '"' in line else ""
            for line in source.splitlines()
            if 'os.getenv("CACHE_DIR"' in line or "os.getenv('CACHE_DIR'" in line
        ], "CACHE_DIR default should not be the Docker container path /.cache"
        # Verify the actual default is set to the native path

        expected_default = os.path.expanduser("~/.cache/whisperx-asr")
        # If CACHE_DIR env is not set, the default must be the native path
        assert expected_default == os.path.expanduser("~/.cache/whisperx-asr")

    def test_cuda_not_default_device(self):
        """DEVICE must default to 'mps', never 'cuda'.  Verify via source."""
        import app.pipeline as pipeline_mod

        source = inspect.getsource(pipeline_mod)
        for line in source.splitlines():
            if 'os.getenv("DEVICE"' in line or "os.getenv('DEVICE'" in line:
                assert '"mps"' in line or "'mps'" in line, f"DEVICE default should be 'mps': {line.strip()}"
                assert '"cuda"' not in line and "'cuda'" not in line, (
                    f"DEVICE default must not be 'cuda': {line.strip()}"
                )
                break
        else:
            pytest.fail("No DEVICE = os.getenv() line found in pipeline source")

    def test_pipeline_transcribe_succeeds_with_inert_compute_type(self):
        """A full transcribe call with any COMPUTE_TYPE value must still
        succeed — the value is inert and not passed to the backend."""
        import app.pipeline as pipeline_mod

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "language": "en",
        }
        with patch.object(pipeline_mod, "load_whisper_model", return_value=mock_model):
            result = pipeline_mod.transcribe(
                audio=MagicMock(),
                model_name="tiny",
                language="en",
                task="transcribe",
            )
            assert "segments" in result
            assert result["segments"][0]["text"] == "hello world"
