"""Independent coverage checks, including omissions that survive a retry."""

import numpy as np
import pytest

from app import speech_coverage as coverage


def segments(start, end):
    return [{"words": [{"word": "recognized", "start": start, "end": end}]}]


def test_subtract_handles_overlaps_and_unsorted_coverage():
    assert coverage.subtract([(0, 10)], [(4, 7), (1, 5), (8, 9)]) == [(0, 1), (7, 8), (9, 10)]


def test_finds_leading_and_trailing_untranscribed_speech(monkeypatch):
    monkeypatch.setattr(coverage, "speech_intervals", lambda audio: [(0, len(audio) / 16000)])
    missing = coverage.missing_speech(np.ones(10 * 16000), segments(4, 6))
    assert missing == [(0, 3.75), (6.25, 10)]


def test_silence_and_music_without_speech_do_not_require_recognition(monkeypatch):
    monkeypatch.setattr(coverage, "speech_intervals", lambda audio: [])
    assert coverage.missing_speech(np.ones(10 * 16000), []) == []


def test_short_boundary_uncertainty_is_allowed():
    coverage.assert_recovered([(0, 10)], segments(0.5, 9.5))


def test_recovered_one_phrase_does_not_pass_a_long_speech_region():
    with pytest.raises(RuntimeError, match="Untranscribed speech remains"):
        coverage.assert_recovered([(0, 12)], segments(0, 2.32))


def test_fragmented_omissions_are_counted_together():
    words = [{"words": [{"word": "word", "start": a, "end": a + 0.1} for a in range(10)]}]
    with pytest.raises(RuntimeError, match="Untranscribed speech remains"):
        coverage.assert_recovered([(0, 10)], words)
