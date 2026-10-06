"""Regression coverage for retained GPU caches after failed pipeline stages."""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest


def test_failed_transcription_releases_caches_and_next_request_succeeds():
    from app import pipeline

    model = MagicMock()
    model.transcribe.side_effect = [
        RuntimeError("MPS backend out of memory"),
        {"segments": [{"start": 0, "end": 1, "text": "hello"}], "language": "en"},
    ]
    audio = np.zeros(16000, dtype=np.float32)
    with (
        patch.object(pipeline, "load_whisper_model", return_value=model),
        patch.object(pipeline, "clear_gpu_memory") as cleanup,
    ):
        with pytest.raises(RuntimeError, match="out of memory"):
            pipeline.transcribe(audio, initial_prompt="test prompt")
        assert model.initial_prompt is None
        assert cleanup.call_count == 2
        assert pipeline.transcribe(audio)["segments"][0]["text"] == "hello"
        assert cleanup.call_count == 4


def test_failed_alignment_preserves_transcript_and_releases_caches():
    from app import pipeline

    result = {"segments": [{"start": 0, "end": 1, "text": "hello"}], "language": "en"}
    with (
        patch.object(pipeline, "load_align_model", return_value=(MagicMock(), {})),
        patch.object(pipeline.whispermlx, "align", side_effect=RuntimeError("MPS backend out of memory")),
        patch.object(pipeline, "clear_gpu_memory") as cleanup,
    ):
        assert pipeline.align(np.zeros(16000, dtype=np.float32), result) == result
        assert cleanup.call_count == 2


def test_cleanup_releases_both_allocators_even_if_mlx_cleanup_fails():
    from app import pipeline

    torch = MagicMock()
    mlx_core = MagicMock()
    mlx_core.clear_cache.side_effect = RuntimeError("MLX cleanup failed")
    mlx = MagicMock(core=mlx_core)
    with patch.dict("sys.modules", {"torch": torch, "mlx": mlx, "mlx.core": mlx_core}):
        pipeline.clear_gpu_memory()
    mlx_core.clear_cache.assert_called_once()
    torch.mps.empty_cache.assert_called_once()
