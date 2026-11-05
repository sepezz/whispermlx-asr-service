"""Audit recognition coverage. VAD never decides what audio gets transcribed or cut."""

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
WORD_MARGIN_S = 0.25
MIN_GAP_S = 2.0
MAX_UNCOVERED_SPEECH_S = 1.5
UNCLEAR_AUDIO_TEXT = "[Untranscribed audio]"


def is_unclear_audio(segment):
    return bool(segment.get("untranscribed")) or segment.get("text", "").strip() == UNCLEAR_AUDIO_TEXT


def unclear_ranges(segments):
    return [(s["start"], s["end"]) for s in segments if is_unclear_audio(s)]


def subtract(intervals, covered):
    """Subtract a union of covered ranges from candidate intervals."""
    out = []
    for start, end in intervals:
        cursor = start
        for a, b in sorted(covered):
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                out.append((cursor, min(a, end)))
            cursor = max(cursor, b)
        if cursor < end:
            out.append((cursor, end))
    return out


def word_ranges(segments):
    return [
        (w["start"] - WORD_MARGIN_S, w["end"] + WORD_MARGIN_S)
        for s in segments
        for w in s.get("words", [])
        if w.get("word", "").strip() and w["end"] > w["start"]
    ]


@lru_cache(maxsize=1)
def _vad():
    from whispermlx.vads.pyannote import load_vad_model

    return load_vad_model("cpu")


def speech_intervals(audio):
    """Independent speech evidence from the bundled local model."""
    import torch
    from whispermlx.vads.pyannote import Binarize

    if not np.any(audio):
        return []
    scores = _vad()({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
    annotation = Binarize(onset=0.5, offset=0.363, min_duration_on=0.1, min_duration_off=0.1)(scores)
    duration = len(audio) / SAMPLE_RATE
    return [(max(0, s.start), min(duration, s.end)) for s in annotation.get_timeline().support()]


def missing_speech(audio, segments):
    """Return speech inside substantial word-free gaps, including both edges."""
    duration = len(audio) / SAMPLE_RATE
    gaps = subtract([(0, duration)], word_ranges(segments) + unclear_ranges(segments))
    missing = []
    for a, b in gaps:
        if b - a < MIN_GAP_S:
            continue
        # Bounded audit windows keep memory use stable on long recordings.
        cursor = a
        speech = []
        while cursor < b:
            end = min(b, cursor + 24)
            chunk = audio[round(cursor * SAMPLE_RATE) : round(end * SAMPLE_RATE)]
            speech.extend((cursor + lo, cursor + hi) for lo, hi in speech_intervals(chunk))
            cursor = end
        if sum(hi - lo for lo, hi in speech) >= MAX_UNCOVERED_SPEECH_S:
            missing.extend(speech)
    return missing


def unresolved_speech(speech, segments):
    covered = word_ranges(segments)
    # Count all omitted speech in each region, including fragmented omissions.
    return [
        (a, b) for a, b in speech if sum(hi - lo for lo, hi in subtract([(a, b)], covered)) >= MAX_UNCOVERED_SPEECH_S
    ]


def assert_recovered(speech, segments):
    unresolved = unresolved_speech(speech, segments)
    if unresolved:
        preview = ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in unresolved[:5])
        raise RuntimeError(
            f"Untranscribed speech remains after stronger recovery: {preview}. Episode is not ready for ad detection."
        )
