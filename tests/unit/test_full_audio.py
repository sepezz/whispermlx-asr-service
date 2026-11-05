"""Coverage and stitching regressions for full-timeline decoding."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.full_audio import SAMPLE_RATE
from app.full_audio import FullAudioWhisper


@pytest.fixture(autouse=True)
def no_real_speech_audit(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda audio, segments: [])


def segment(*words):
    return {
        "text": "".join(word[0] for word in words),
        "words": [{"word": text, "start": start, "end": end} for text, start, end in words],
    }


def decoder(monkeypatch, results):
    results = iter(results)

    def next_result(*args, **kwargs):
        result = next(results, {"segments": []})
        if isinstance(result, Exception):
            raise result
        return result

    decode = Mock(side_effect=next_result)
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=decode))
    return decode


def test_all_samples_including_tail_are_submitted(monkeypatch):
    audio = np.arange(61 * SAMPLE_RATE, dtype=np.float32)
    decode = decoder(monkeypatch, [{"language": "en", "segments": []}] * 4)
    progress = []
    FullAudioWhisper("small").transcribe(audio, progress_callback=progress.append)
    covered = np.zeros(len(audio), dtype=bool)
    for call in decode.call_args_list[:4]:
        chunk = call.args[0]
        start = int(chunk[0])
        covered[start : start + len(chunk)] = True
        assert len(chunk) <= 28 * SAMPLE_RATE
        assert call.kwargs["no_speech_threshold"] is None
    assert covered.all()
    assert progress[:4] == pytest.approx([20, 40, 60, 80])
    assert progress[-1] == 100
    assert progress == sorted(progress)


def test_overlap_keeps_each_boundary_word_once(monkeypatch):
    # Second window starts at 16s. The shared phrase must appear once.
    decoder(
        monkeypatch,
        [
            {"language": "en", "segments": [segment((" before", 23, 23.8), (" after", 24, 24.8))]},
            {"language": "en", "segments": [segment((" before", 7, 7.8), (" after", 8, 8.8))]},
        ],
    )
    result = FullAudioWhisper("small").transcribe(np.ones(30 * SAMPLE_RATE))
    words = [w for s in result["segments"] for w in s["words"]]
    assert [w["word"].strip() for w in words] == ["before", "after"]
    assert words[1]["start"] == 24


def test_decoder_failure_aborts_instead_of_losing_window(monkeypatch):
    decoder(monkeypatch, [{"segments": []}, RuntimeError("decoder failed")])
    with pytest.raises(RuntimeError, match="decoder failed"):
        FullAudioWhisper("small").transcribe(np.ones(30 * SAMPLE_RATE))


def test_missing_word_timestamps_fail_instead_of_discarding_text(monkeypatch):
    decoder(monkeypatch, [{"segments": [{"text": "Sign up to Revolut"}]}])
    with pytest.raises(RuntimeError, match="without word timings"):
        FullAudioWhisper("small").transcribe(np.ones(SAMPLE_RATE))


def test_language_and_prompt_are_forwarded(monkeypatch):
    decode = decoder(monkeypatch, [{"language": "en", "segments": []}] * 2)
    model = FullAudioWhisper("repo/model")
    model.initial_prompt = "British history"
    model.transcribe(np.ones(30 * SAMPLE_RATE))
    assert decode.call_args.kwargs["language"] == "en"
    assert decode.call_args.kwargs["initial_prompt"] == "British history"
    assert decode.call_args.kwargs["path_or_hf_repo"] == "repo/model"


def test_invalid_audio_is_rejected(monkeypatch):
    decode = decoder(monkeypatch, [])
    with pytest.raises(ValueError, match="finite mono"):
        FullAudioWhisper("small").transcribe(np.array([np.nan]))
    decode.assert_not_called()


def test_exact_digital_silence_does_not_keep_hallucinated_text(monkeypatch):
    decode = decoder(monkeypatch, [{"language": "en", "segments": [segment((" you", 0, 1))]}])
    result = FullAudioWhisper("small").transcribe(np.zeros(SAMPLE_RATE))
    decode.assert_not_called()
    assert result["segments"] == []


def test_recovers_missing_ad_between_recognized_speech(monkeypatch):
    decode = decoder(
        monkeypatch,
        [
            {"language": "en", "segments": [segment((" Sponsor.", 0, 6.54))]},
            {"segments": [segment((" bubbles.", 8.341, 8.784))]},
            {"segments": [segment((" There's a new sheriff in town.", 1, 3))]},
            {"segments": [segment((" New Aero Milky Bar.", 1, 3))]},
        ],
    )
    result = FullAudioWhisper("small").transcribe(np.ones(26 * SAMPLE_RATE))
    assert [s["text"] for s in result["segments"]] == [
        "Sponsor.",
        "There's a new sheriff in town.",
        "New Aero Milky Bar.",
        "bubbles.",
    ]
    assert result["segments"][1]["start"] == pytest.approx(7.04)
    assert decode.call_count == 4


def test_recovery_does_not_duplicate_context_words(monkeypatch):
    decoder(
        monkeypatch,
        [
            {"segments": [segment((" before", 0, 1), (" after", 8, 9))]},
            {"segments": [segment((" before", 0, 0.4), (" ad", 1, 2), (" after", 7.6, 8.5))]},
        ],
    )
    result = FullAudioWhisper("small").transcribe(np.ones(10 * SAMPLE_RATE))
    assert [s["text"] for s in result["segments"]] == ["before", "ad", "after"]


def test_low_quality_recovery_is_rejected(monkeypatch):
    bad = segment((" repeat repeat repeat", 0, 3))
    bad["compression_ratio"] = 4.2
    decoder(monkeypatch, [{"segments": []}, {"segments": [bad]}])
    assert FullAudioWhisper("small").transcribe(np.ones(6 * SAMPLE_RATE))["segments"] == []


def test_recovery_failure_aborts_instead_of_reporting_complete(monkeypatch):
    decoder(monkeypatch, [{"segments": []}, RuntimeError("recovery failed")])
    with pytest.raises(RuntimeError, match="recovery failed"):
        FullAudioWhisper("small").transcribe(np.ones(6 * SAMPLE_RATE))


def test_stronger_model_recovers_speech_omitted_by_both_small_passes(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda audio, segments: [(0, 10)])
    decode = decoder(
        monkeypatch,
        [
            {"segments": []},
            {"segments": []},
            {"segments": [segment((" Cheaper electricity between seven and two.", 0, 10))]},
        ],
    )
    result = FullAudioWhisper("small").transcribe(np.ones(10 * SAMPLE_RATE))
    assert "Cheaper electricity" in result["segments"][0]["text"]
    assert decode.call_args.kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_unrecovered_speech_fails_instead_of_marking_complete(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda audio, segments: [(0, 10)])
    decoder(monkeypatch, [{"segments": []}] * 3)
    progress = []
    with pytest.raises(RuntimeError, match="Untranscribed speech remains"):
        FullAudioWhisper("small").transcribe(np.ones(10 * SAMPLE_RATE), progress_callback=progress.append)
    assert 100 not in progress


def test_detector_uncertainty_is_retained_as_timed_audio_instead_of_rejecting_episode(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda *args: [(5, 8)])
    decoder(monkeypatch, [{"language": "en", "segments": [segment(("Programme.", 0, 1))]}])
    result = FullAudioWhisper("small").transcribe(np.ones(10 * SAMPLE_RATE), language="en")
    unknown = [s for s in result["segments"] if s.get("untranscribed")]
    assert [(s["start"], s["end"], s["text"]) for s in unknown] == [(5, 8, "[Untranscribed audio]")]
    assert unknown[0]["words"] == []
    assert result["warnings"] and result["untranscribed_regions"] == [{"start": 5, "end": 8}]


def test_turbo_uses_large_v3_for_suspect_speech(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda audio, segments: [(0, 10)])
    decode = decoder(
        monkeypatch,
        [
            {"segments": []},
            {"segments": []},
            {"segments": [segment((" Recovered speech.", 0, 10))]},
        ],
    )
    FullAudioWhisper("mlx-community/whisper-large-v3-turbo").transcribe(np.ones(10 * SAMPLE_RATE))
    assert decode.call_args.kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
    assert decode.call_args.kwargs["language"] is None


def test_recovery_retains_the_detected_language_for_alignment(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda *args: [(0, 8)])
    decoder(
        monkeypatch,
        [
            {"language": "en", "segments": []},
            {"language": "en", "segments": []},
            {"language": "ja", "segments": [segment(("ゴジラ", 0, 8))]},
        ],
    )
    result = FullAudioWhisper("small").transcribe(np.ones(8 * SAMPLE_RATE), language="en")
    assert result["language"] == "en"
    assert result["segments"][0]["language"] == "ja"


def test_short_recovery_failure_tries_wider_context(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda *args: [(20, 26)])

    def read(audio, **kwargs):
        if kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo" and len(audio) == 28 * SAMPLE_RATE:
            start = float(audio[0]) / SAMPLE_RATE
            return {"language": "en", "segments": [segment((" Recovered review.", 20 - start, 26 - start))]}
        return {"language": "en", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=read))
    result = FullAudioWhisper("small").transcribe(np.arange(60 * SAMPLE_RATE, dtype=np.float32), language="en")
    assert any(s["text"] == "Recovered review." for s in result["segments"])


def test_recovery_view_near_episode_end_can_start_after_previous_speaker(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda *args: [(6, 10)])

    def read(audio, **kwargs):
        start = float(audio[0]) / SAMPLE_RATE
        if kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo" and start >= 5:
            return {"language": "en", "segments": [segment((" Recovered.", 6 - start, 10 - start))]}
        return {"language": "en", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=read))
    result = FullAudioWhisper("small").transcribe(np.arange(12 * SAMPLE_RATE, dtype=np.float32), language="en")
    assert any(s["text"] == "Recovered." for s in result["segments"])


def test_short_foreign_speech_island_can_select_language_without_english_context(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda *args: [(20, 22)])

    def read(audio, **kwargs):
        start = float(audio[0]) / SAMPLE_RATE
        if kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo" and len(audio) < 3 * SAMPLE_RATE:
            assert kwargs["language"] is None
            return {"language": "ja", "segments": [segment(("ゴジラ", 20 - start, 22 - start))]}
        return {"language": "en", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=read))
    result = FullAudioWhisper("small").transcribe(np.arange(40 * SAMPLE_RATE, dtype=np.float32), language="en")
    assert any(s["language"] == "ja" and s["text"] == "ゴジラ" for s in result["segments"])


def test_detected_island_language_is_used_for_wider_recovery(monkeypatch):
    monkeypatch.setattr("app.full_audio.missing_speech", lambda *args: [(20, 22)])

    def read(audio, **kwargs):
        start = float(audio[0]) / SAMPLE_RATE
        if kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo":
            if len(audio) < 3 * SAMPLE_RATE:
                return {"language": "ja", "segments": [segment(("ゴ", 20 - start, 20.1 - start))]}
            if kwargs["language"] == "ja":
                return {"language": "ja", "segments": [segment(("ゴジラ", 20 - start, 22 - start))]}
        return {"language": "en", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=read))
    result = FullAudioWhisper("small").transcribe(np.arange(60 * SAMPLE_RATE, dtype=np.float32), language="en")
    assert any(s["language"] == "ja" and s["text"] == "ゴジラ" for s in result["segments"])


def test_audit_error_does_not_bypass_coverage_check(monkeypatch):
    def broken_audit(*args):
        raise RuntimeError("speech audit unavailable")

    monkeypatch.setattr("app.full_audio.missing_speech", broken_audit)
    decoder(monkeypatch, [{"segments": []}] * 2)
    with pytest.raises(RuntimeError, match="speech audit unavailable"):
        FullAudioWhisper("small").transcribe(np.ones(6 * SAMPLE_RATE))


def test_sparse_retry_words_cannot_hide_the_original_omission(monkeypatch):
    calls = []

    def audit(audio, segments):
        calls.append(segments.copy())
        return [(2.5, 12)]

    monkeypatch.setattr("app.full_audio.missing_speech", audit)
    decoder(
        monkeypatch,
        [
            {"segments": [segment((" Savings seekers.", 0, 2))]},
            {"segments": [segment((" We hear you!", 4, 10))]},
            {"segments": [segment((" Cheaper electricity between seven and two.", 2.1, 11.5))]},
        ],
    )
    result = FullAudioWhisper("small").transcribe(np.ones(12 * SAMPLE_RATE))
    text = " ".join(s["text"] for s in result["segments"])
    assert "We hear you" not in text
    assert "electricity" in text
    assert len(calls) == 1
    assert len(calls[0]) == 1
