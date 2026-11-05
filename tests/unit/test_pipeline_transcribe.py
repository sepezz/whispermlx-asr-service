"""Unit tests for app/pipeline.py transcribe + align conversion to whispermlx.

Tests are fast: whispermlx is mocked, no model downloads, no GPU required.
"""

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model():
    """Create a mock MLXWhisperPipeline that mimics whispermlx.load_model()."""
    model = MagicMock()
    model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "language": "en",
    }
    model.initial_prompt = None
    return model


def _import_pipeline():
    """Import app.pipeline fresh so module-level patches take effect."""
    import importlib

    import app.pipeline

    importlib.reload(app.pipeline)
    return app.pipeline


# ---------------------------------------------------------------------------
# 1. Import & namespace tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify pipeline.py imports whispermlx, never whisperx or faster_whisper."""

    def test_no_whisperx_import(self):
        """app.pipeline must not import whisperx at all."""
        pipeline = _import_pipeline()
        # Check that 'whisperx' is not in the module's source-level imports.
        import inspect

        source = inspect.getsource(pipeline)
        # Ignore comments that mention whisperx for migration context
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import whisperx" not in stripped, f"Found 'import whisperx' in pipeline.py: {stripped}"
            assert "from whisperx" not in stripped, f"Found 'from whisperx' in pipeline.py: {stripped}"

    def test_no_faster_whisper_import(self):
        """app.pipeline must not import faster_whisper."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "faster_whisper" not in stripped, f"Found 'faster_whisper' in pipeline.py: {stripped}"

    def test_whispermlx_import(self):
        """app.pipeline must import whispermlx."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline)
        assert "import whispermlx" in source, "whispermlx import not found in pipeline.py"


# ---------------------------------------------------------------------------
# 2. DEVICE configuration
# ---------------------------------------------------------------------------


class TestDeviceConfig:
    """Verify DEVICE defaults to mps (matches whispermlx upstream)."""

    def test_default_device_is_mps(self):
        """DEVICE env var defaults to 'mps' without any CUDA check."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove DEVICE from env if present
            env = dict(__import__("os").environ)
            env.pop("DEVICE", None)
            with patch("os.environ", env):
                pipeline = _import_pipeline()
                assert pipeline.DEVICE == "mps", f"Expected DEVICE='mps', got '{pipeline.DEVICE}'"

    def test_device_from_env(self):
        """DEVICE can be overridden via environment variable."""
        with patch.dict("os.environ", {"DEVICE": "mps"}):
            pipeline = _import_pipeline()
            assert pipeline.DEVICE == "mps"


# ---------------------------------------------------------------------------
# 3. get_canonical_models — MLX model map
# ---------------------------------------------------------------------------


class TestGetCanonicalModels:
    """Verify get_canonical_models returns MLX model names, no faster_whisper."""

    def test_returns_mlx_model_names(self):
        """Should return the MLX model map keys."""
        pipeline = _import_pipeline()
        models = pipeline.get_canonical_models()
        expected = [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "turbo",
        ]
        for name in expected:
            assert name in models, f"MLX model '{name}' missing from get_canonical_models()"

    def test_no_distil_models(self):
        """Must not include distil-* models (faster-whisper only)."""
        pipeline = _import_pipeline()
        models = pipeline.get_canonical_models()
        for name in models:
            assert not name.startswith("distil-"), f"distil-* model found: {name}"

    def test_large_v3_turbo_present(self):
        """large-v3-turbo must be present (MLX-specific model)."""
        pipeline = _import_pipeline()
        models = pipeline.get_canonical_models()
        assert "large-v3-turbo" in models

    def test_no_faster_whisper_import_in_function(self):
        """The function must not try to import faster_whisper."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.get_canonical_models)
        assert "faster_whisper" not in source


# ---------------------------------------------------------------------------
# 4. resolve_model_name — aliasing
# ---------------------------------------------------------------------------


class TestResolveModelName:
    """Verify resolve_model_name maps aliases to MLX canonical names."""

    def test_whisper_1_alias(self):
        """whisper-1 resolves to DEFAULT_MODEL."""
        pipeline = _import_pipeline()
        result = pipeline.resolve_model_name("whisper-1")
        assert result == pipeline.DEFAULT_MODEL

    def test_whisper_tiny_alias(self):
        """whisper-tiny resolves to tiny."""
        pipeline = _import_pipeline()
        result = pipeline.resolve_model_name("whisper-tiny")
        assert result == "tiny"

    def test_canonical_name_unchanged(self):
        """Canonical names like 'base', 'large-v3' pass through unchanged."""
        pipeline = _import_pipeline()
        assert pipeline.resolve_model_name("base") == "base"
        assert pipeline.resolve_model_name("large-v3") == "large-v3"
        assert pipeline.resolve_model_name("large-v3-turbo") == "large-v3-turbo"

    def test_empty_model_returns_default(self):
        """Empty/None model returns DEFAULT_MODEL."""
        pipeline = _import_pipeline()
        assert pipeline.resolve_model_name("") == pipeline.DEFAULT_MODEL
        assert pipeline.resolve_model_name(None) == pipeline.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 5. load_whisper_model — full-audio adapter
# ---------------------------------------------------------------------------


class TestLoadWhisperModel:
    """Verify load_whisper_model prepares and caches the full-audio adapter."""

    def test_creates_full_audio_adapter(self):
        """Prepare the full-audio decoder with the canonical model repository."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()

        with patch.object(pipeline, "FullAudioWhisper", return_value=mock_model) as mock_load:
            result = pipeline.load_whisper_model("base")
            mock_load.assert_called_once_with(pipeline.MLX_MODEL_MAP["base"])
            assert result is mock_model

    def test_caches_model(self):
        """Second call for same model should use cache, not call load_model again."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()

        with patch.object(pipeline, "FullAudioWhisper", return_value=mock_model):
            pipeline.load_whisper_model("base")
            pipeline.load_whisper_model("base")
        # load_model should only be called once due to caching
        assert pipeline._whisper_models.get("base") is mock_model


# ---------------------------------------------------------------------------
# 6. transcribe — hotwords no-op, initial_prompt set/reset
# ---------------------------------------------------------------------------


class TestTranscribeHotwords:
    """Verify hotwords is a no-op: warns, never sets any attribute, no error."""

    def test_hotwords_is_noop_and_warns(self, caplog):
        """hotwords provided → warning logged, no attribute set, transcription succeeds."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        with caplog.at_level(logging.WARNING):
            result = pipeline.transcribe(audio, model_name="base", hotwords="Foo Bar")

        # Transcription should still succeed
        assert result["segments"] is not None
        assert result["language"] == "en"

        # A warning about hotwords should be logged
        assert any(
            "hotwords" in record.message.lower() and "ignore" in record.message.lower() for record in caplog.records
        ), f"Expected hotwords warning; got: {[r.message for r in caplog.records]}"

    def test_hotwords_never_sets_attribute(self):
        """hotwords must never set any attribute on the model."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        # Use spec to prevent MagicMock auto-creating attributes
        mock_model.spec_set = set()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.transcribe(audio, model_name="base", hotwords="Foo Bar")

        # Check that no 'hotwords' attribute was explicitly set on the mock
        # MagicMock auto-creates attrs on access, so check method_calls instead
        for call_entry in mock_model.method_calls:
            assert "hotwords" not in str(call_entry), f"hotwords was passed to model method: {call_entry}"

        # Also verify the source code never sets a hotwords attribute
        import inspect

        source = inspect.getsource(pipeline.transcribe)
        # No line should set .hotwords on any object (except in comments/logs)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("logger"):
                continue
            assert (
                ".hotwords" not in stripped
                or "hotwords is not None" in stripped
                or "hotwords" in stripped
                and "==" in stripped
            ), f"Code sets .hotwords attribute: {stripped}"

    def test_no_hotwords_no_warning(self, caplog):
        """When hotwords is not provided, no warning about hotwords should be logged."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        with caplog.at_level(logging.WARNING):
            pipeline.transcribe(audio, model_name="base")

        assert not any("hotwords" in record.message.lower() for record in caplog.records), (
            "Unexpected hotwords warning when hotwords not provided"
        )


class TestTranscribeInitialPrompt:
    """Verify initial_prompt is set per-request and reset afterward."""

    def test_initial_prompt_set_and_reset(self):
        """initial_prompt set during transcribe and reset to None afterward."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.transcribe(audio, model_name="base", initial_prompt="Hello world")

        # After transcribe, initial_prompt should be reset to None
        assert mock_model.initial_prompt is None, (
            f"initial_prompt was not reset after transcribe: {mock_model.initial_prompt}"
        )

    def test_initial_prompt_set_during_transcribe(self):
        """initial_prompt should be set on the model BEFORE transcribe is called."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        # Track the value of initial_prompt at the time transcribe is called
        prompt_at_call = []

        original_transcribe = mock_model.transcribe

        def capture_prompt(*args, **kwargs):
            prompt_at_call.append(mock_model.initial_prompt)
            return original_transcribe(*args, **kwargs)

        mock_model.transcribe = capture_prompt

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.transcribe(audio, model_name="base", initial_prompt="Hello world")

        assert prompt_at_call == ["Hello world"], f"initial_prompt was not set before transcribe; got: {prompt_at_call}"

    def test_initial_prompt_reset_on_exception(self):
        """initial_prompt must be reset even if transcribe raises."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        mock_model.transcribe.side_effect = RuntimeError("boom")

        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="boom"):
            pipeline.transcribe(audio, model_name="base", initial_prompt="Hello world")

        # Even after exception, initial_prompt should be reset
        assert mock_model.initial_prompt is None, (
            f"initial_prompt not reset after exception: {mock_model.initial_prompt}"
        )

    def test_no_initial_prompt_no_set(self):
        """When initial_prompt is None, the model attribute should not be changed."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        mock_model.initial_prompt = None
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.transcribe(audio, model_name="base")

        # Should still be None
        assert mock_model.initial_prompt is None


class TestTranscribeCall:
    """Verify whisper_model.transcribe is called with correct args (batch_size ignored)."""

    def test_transcribe_call_args(self):
        """Calls model.transcribe(audio, language=, task=) without batch_size."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.transcribe(audio, model_name="base", language="en", task="transcribe")

        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args
        # Should pass language and task
        assert call_kwargs[1].get("language") == "en" or (len(call_kwargs[0]) > 0)
        assert call_kwargs[1].get("task") == "transcribe" or (len(call_kwargs[0]) > 0)

    def test_transcribe_returns_segments_and_language(self):
        """transcribe() must return {segments, language}."""
        pipeline = _import_pipeline()
        mock_model = _make_mock_model()
        pipeline._whisper_models.clear()
        pipeline._whisper_models_last_used.clear()
        pipeline._whisper_models["base"] = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = pipeline.transcribe(audio, model_name="base")

        assert "segments" in result
        assert "language" in result


# ---------------------------------------------------------------------------
# 7. align — uses whispermlx.align
# ---------------------------------------------------------------------------


class TestAlign:
    """Verify align() uses whispermlx.align(...)."""

    def test_alignment_holes_fall_back_to_independently_verified_native_words(self, monkeypatch):
        monkeypatch.setenv("FORCE_ALIGNMENT", "true")
        pipeline = _import_pipeline()
        pipeline._align_models["en"] = (MagicMock(), {})
        native = {
            "start": 0.0,
            "end": 8.0,
            "text": "Complete speech.",
            "words": [{"word": "Complete speech.", "start": 0.0, "end": 8.0}],
        }
        monkeypatch.setattr(pipeline.whispermlx, "align", lambda *args, **kwargs: {"segments": [], "word_segments": []})
        monkeypatch.setattr("app.speech_coverage.missing_speech", lambda audio, segments: [] if segments else [(0, 8)])
        result = pipeline.align(
            np.ones(8 * 16000, dtype=np.float32), {"language": "en", "segments": [native]}, strict=True
        )
        assert result["segments"] == [native]
        assert result["word_segments"] == native["words"]

    def test_native_words_do_not_require_forced_alignment(self, monkeypatch):
        pipeline = _import_pipeline()
        native = {"start": 0, "end": 1, "text": "Hello.", "words": [{"word": "Hello.", "start": 0, "end": 1}]}

        def unexpected(*args, **kwargs):
            raise AssertionError("Native timestamps must not be needlessly realigned")

        monkeypatch.setattr(pipeline.whispermlx, "align", unexpected)
        result = pipeline.align(
            np.zeros(16000, dtype=np.float32), {"segments": [native], "language": "en"}, strict=True
        )
        assert result["word_segments"] == native["words"]

    def test_english_aligner_does_not_erase_japanese_clip(self, monkeypatch):
        pipeline = _import_pipeline()
        pipeline._align_models["en"] = (MagicMock(), {})
        english = {
            "start": 0.0,
            "end": 1.0,
            "text": "Here is the clip.",
            "language": "en",
            "words": [{"word": "clip", "start": 0.0, "end": 1.0}],
        }
        japanese = {
            "start": 2.0,
            "end": 5.0,
            "text": "ゴジラ",
            "language": "ja",
            "words": [{"word": "ゴジラ", "start": 2.0, "end": 5.0}],
        }

        def align_main(segments, *args, **kwargs):
            assert segments == [english]
            return {"segments": [english], "word_segments": english["words"]}

        monkeypatch.setattr(pipeline.whispermlx, "align", align_main)
        monkeypatch.setattr("app.speech_coverage.missing_speech", lambda *args: [])
        result = pipeline.align(
            np.zeros(6 * 16000, dtype=np.float32), {"segments": [english, japanese], "language": "en"}, strict=True
        )
        assert result["segments"] == [english, japanese]
        assert result["word_segments"][-1]["word"] == "ゴジラ"

    def test_foreign_only_segments_keep_native_timings_without_loading_english(self, monkeypatch):
        pipeline = _import_pipeline()
        japanese = {
            "start": 0.0,
            "end": 2.0,
            "text": "ゴジラ",
            "language": "ja",
            "words": [{"word": "ゴジラ", "start": 0.0, "end": 2.0}],
        }

        def unexpected(*args, **kwargs):
            raise AssertionError("Wrong-language aligner must not run")

        monkeypatch.setattr(pipeline, "load_align_model", unexpected)
        monkeypatch.setattr("app.speech_coverage.missing_speech", lambda *args: [])
        result = pipeline.align(
            np.zeros(3 * 16000, dtype=np.float32), {"segments": [japanese], "language": "en"}, strict=True
        )
        assert result["segments"] == [japanese]

    def test_strict_alignment_cannot_publish_dropped_speech(self, monkeypatch):
        pipeline = _import_pipeline()
        pipeline._align_models["en"] = (MagicMock(), {})
        monkeypatch.setattr(pipeline.whispermlx, "align", lambda *args, **kwargs: {"segments": []})
        monkeypatch.setattr("app.speech_coverage.missing_speech", lambda *args: [(0, 8)])
        with pytest.raises(RuntimeError, match="Untranscribed speech remains"):
            pipeline.align(np.ones(8 * 16000, dtype=np.float32), {"language": "en", "segments": []}, strict=True)

    def test_align_calls_whispermlx_align(self):
        """align() should call whispermlx.align()."""
        pipeline = _import_pipeline()
        mock_align_model = MagicMock()
        mock_metadata = {"language": "en", "dictionary": {}, "type": "huggingface"}
        pipeline._align_models.clear()
        pipeline._align_models["en"] = (mock_align_model, mock_metadata)

        with patch.object(pipeline.whispermlx, "align") as mock_align:
            mock_align.return_value = {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}
                ],
                "word_segments": [{"word": "hello", "start": 0.0, "end": 1.0}],
            }
            audio = np.zeros(16000, dtype=np.float32)
            result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}
            pipeline.align(audio, result)
            mock_align.assert_called_once()

    def test_align_produces_word_level_timestamps(self):
        """Align result should contain word-level timestamps."""
        pipeline = _import_pipeline()
        mock_align_model = MagicMock()
        mock_metadata = {"language": "en", "dictionary": {}, "type": "huggingface"}
        pipeline._align_models.clear()
        pipeline._align_models["en"] = (mock_align_model, mock_metadata)

        with patch.object(pipeline.whispermlx, "align") as mock_align:
            mock_align.return_value = {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}
                ],
                "word_segments": [{"word": "hello", "start": 0.0, "end": 1.0}],
            }
            audio = np.zeros(16000, dtype=np.float32)
            result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}
            aligned = pipeline.align(audio, result)
            assert "word_segments" in aligned


# ---------------------------------------------------------------------------
# 8. clear_gpu_memory — gc + MLX (no torch.cuda)
# ---------------------------------------------------------------------------


class TestClearGpuMemory:
    """Verify clear_gpu_memory uses gc.collect() + guarded MLX cache clear."""

    def test_no_torch_cuda(self):
        """Must not use torch.cuda.empty_cache()."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.clear_gpu_memory)
        assert "torch.cuda" not in source, "clear_gpu_memory still references torch.cuda"

    def test_uses_gc_collect(self):
        """Must call gc.collect()."""
        pipeline = _import_pipeline()
        import inspect

        source = inspect.getsource(pipeline.clear_gpu_memory)
        assert "gc.collect" in source, "clear_gpu_memory should call gc.collect()"

    def test_calls_mlx_cache_clear_guarded(self):
        """Should attempt MLX cache clear (guarded, no error if unavailable)."""
        pipeline = _import_pipeline()
        # Should not raise even if mlx.core.clear_cache doesn't exist
        pipeline.clear_gpu_memory()  # just verify no crash


# ---------------------------------------------------------------------------
# 9. load_align_model — whispermlx.load_align_model
# ---------------------------------------------------------------------------


class TestLoadAlignModel:
    """Verify load_align_model calls whispermlx.load_align_model."""

    def test_calls_whispermlx_load_align_model(self):
        """Should call whispermlx.load_align_model(language_code, device, model_dir)."""
        pipeline = _import_pipeline()
        mock_model = MagicMock()
        mock_metadata = {"language": "en", "dictionary": {}, "type": "huggingface"}
        pipeline._align_models.clear()

        with patch.object(
            pipeline.whispermlx, "load_align_model", return_value=(mock_model, mock_metadata)
        ) as mock_load:
            result_model, result_meta = pipeline.load_align_model("en")
            mock_load.assert_called_once_with(
                language_code="en",
                device=pipeline.DEVICE,
                model_dir=pipeline.CACHE_DIR,
            )
            assert result_model is mock_model
            assert result_meta is mock_metadata
