"""Decode the complete timeline without a speech detector discarding regions."""

import logging
import math

import numpy as np

from app.speech_coverage import UNCLEAR_AUDIO_TEXT
from app.speech_coverage import assert_recovered
from app.speech_coverage import missing_speech
from app.speech_coverage import subtract
from app.speech_coverage import unresolved_speech
from app.transcript_stitch import stitch_windows

logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000
WINDOW_SECONDS = 20
CONTEXT_SECONDS = 4
GAP_SECONDS = 5
RECOVERY_WINDOW_SECONDS = 12
RECOVERY_MODEL = "mlx-community/whisper-large-v3-turbo"
LARGE_RECOVERY_MODEL = "mlx-community/whisper-large-v3-mlx"


class FullAudioWhisper:
    """Decode 28-second bites with eight seconds of shared audio.

    Retain both readings of shared audio, then reconcile their words using
    text and timestamps. Every input is below Whisper's 30-second window.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.initial_prompt = None

    def transcribe(self, audio, *, language=None, task="transcribe", progress_callback=None):
        import mlx_whisper

        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Transcription requires finite mono audio at 16 kHz")
        duration = len(audio) / SAMPLE_RATE
        count = math.ceil(duration / WINDOW_SECONDS)
        segments = []
        windows = []

        def decode(start, end, *, model_path=None, auto_language=False, language_hint=None):
            chunk = audio[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)]
            if not np.any(chunk):
                return {"segments": []}
            return mlx_whisper.transcribe(
                chunk,
                path_or_hf_repo=model_path or self.model_path,
                language=language_hint or (None if auto_language else language),
                task=task,
                initial_prompt=self.initial_prompt,
                word_timestamps=True,
                no_speech_threshold=None,
                condition_on_previous_text=False,
                temperature=0.0,
                verbose=None,
            )

        def collect(result, start, owned_start, owned_end, *, clip=False):
            collected = []
            for segment in result.get("segments", []):
                words = segment.get("words", [])
                if segment.get("text", "").strip() and not words:
                    raise RuntimeError(f"Decoder returned text without word timings at {start:.3f}s")
                kept = []
                for word in words:
                    word_start = start + float(word["start"])
                    word_end = start + float(word["end"])
                    if not (math.isfinite(word_start) and math.isfinite(word_end) and word_start <= word_end):
                        raise RuntimeError(f"Decoder returned invalid word timings at {start:.3f}s")
                    midpoint = (word_start + word_end) / 2
                    if owned_start <= midpoint < owned_end:
                        kept.append(
                            {
                                **word,
                                "start": max(owned_start if clip else 0, word_start),
                                "end": min(owned_end if clip else duration, word_end),
                            }
                        )
                runs = []
                for word in kept:
                    if not runs or word["start"] - runs[-1][-1]["end"] >= GAP_SECONDS:
                        runs.append([])
                    runs[-1].append(word)
                for run in runs:
                    collected.append(
                        {
                            "start": run[0]["start"],
                            "end": run[-1]["end"],
                            "text": "".join(word["word"] for word in run).strip(),
                            "words": run,
                            "avg_logprob": segment.get("avg_logprob", 0.0),
                            "language": segment.get("language") or result.get("language") or language,
                        }
                    )
            return collected

        for index in range(count):
            owned_start = index * WINDOW_SECONDS
            owned_end = min(duration, owned_start + WINDOW_SECONDS)
            start = max(0, owned_start - CONTEXT_SECONDS)
            end = min(duration, owned_end + CONTEXT_SECONDS)
            # Do not prefilter with VAD or let the decoder skip an input window
            # because it predicts "no speech". Empty text remains valid silence.
            result = decode(start, end)
            if language is None:
                language = result.get("language")
            windows.append((start, end, collect(result, start, start, end)))
            if progress_callback is not None:
                progress_callback((index + 1) / count * 80)
        segments = stitch_windows(windows)
        # Receiving every sample does not guarantee Whisper returns every word.
        # Decode long word-free regions again in shorter windows with fresh
        # context. Never infer that an untranscribed region is safe to remove.
        words = sorted((w for s in segments for w in s["words"]), key=lambda w: w["start"])
        gaps = []
        cursor = 0.0
        for word in words + [{"start": duration, "end": duration}]:
            if word["start"] - cursor >= GAP_SECONDS:
                gap_end = word["start"]
                gap_start = cursor
                while gap_start < gap_end:
                    recovery_end = min(gap_end, gap_start + RECOVERY_WINDOW_SECONDS)
                    gaps.append((gap_start, recovery_end))
                    gap_start = recovery_end
            cursor = max(cursor, word["end"])
        # Audit the first pass, before another weak decode can disguise an
        # omission with a few incorrectly stretched words.
        speech = missing_speech(audio, segments)
        recovered = []
        for index, (owned_start, owned_end) in enumerate(gaps):
            start = max(0, owned_start - 0.5)
            end = min(duration, owned_end + 0.5)
            result = decode(start, end)
            # Reject the decoder's usual low-quality/hallucination signals.
            result = {
                **result,
                "segments": [
                    s
                    for s in result.get("segments", [])
                    if s.get("avg_logprob", 0) >= -1 and s.get("compression_ratio", 0) <= 2.4
                ],
            }
            recovered.extend(collect(result, start, owned_start, owned_end, clip=True))
            if progress_callback is not None:
                progress_callback(80 + (index + 1) / len(gaps) * 10)
        segments.extend(recovered)
        segments.sort(key=lambda s: (s["start"], s["end"]))
        # Retrying the same small model is not evidence that speech was covered.
        # Audit independently, then use a different model on the whole suspect
        # gap so its context is preserved. Replace only that gap's recognition.
        strong_windows = []
        for a, b in speech:
            if strong_windows and a <= strong_windows[-1][1] + 1:
                strong_windows[-1] = (strong_windows[-1][0], max(b, strong_windows[-1][1]))
            else:
                strong_windows.append((a, b))
        stronger = []
        if strong_windows:
            retained = []
            for a, b in subtract([(0, duration)], strong_windows):
                retained.extend(collect({"segments": segments}, 0, a, b, clip=True))
            segments = retained
        for index, (a, b) in enumerate(strong_windows):
            cursor = a
            while cursor < b:
                owned_end = min(b, cursor + RECOVERY_WINDOW_SECONDS)
                start = max(0, cursor - CONTEXT_SECONDS)
                end = min(duration, owned_end + CONTEXT_SECONDS)
                recovery_model = LARGE_RECOVERY_MODEL if self.model_path == RECOVERY_MODEL else RECOVERY_MODEL
                result = decode(start, end, model_path=recovery_model, auto_language=True)
                result = {
                    **result,
                    "segments": [
                        s
                        for s in result.get("segments", [])
                        if s.get("avg_logprob", 0) >= -1 and s.get("compression_ratio", 0) <= 2.4
                    ],
                }
                stronger.extend(collect(result, start, cursor, owned_end, clip=True))
                cursor = owned_end
            if progress_callback is not None:
                progress_callback(90 + (index + 1) / len(strong_windows) * 9)
        segments.extend(stronger)
        segments.sort(key=lambda s: (s["start"], s["end"]))
        # Film clips can switch language, and short crops can make Whisper
        # report only the speaker before a transition. Retry unresolved regions
        # in views of up to 28 seconds, with different placements of the seam.
        # Keep the existing reading if a new view does not improve coverage.
        language_hints = {}
        for placement in ("speech", None, 14.0, 22.0, 6.0):
            remaining = unresolved_speech(speech, segments)
            if not remaining:
                break
            for a, b in remaining:
                cursor = a
                while cursor < b:
                    owned_end = min(b, cursor + 12)
                    center = (cursor + owned_end) / 2
                    if placement == "speech":
                        start = max(0, cursor - 0.15)
                        end = min(duration, owned_end + 0.15)
                    else:
                        anchor = cursor - 0.5 if placement is None else center - placement
                        start = max(0, min(anchor, duration - 0.01))
                        end = min(duration, start + 28)
                    hint = language_hints.get((a, b))
                    result = decode(start, end, model_path=recovery_model, auto_language=True, language_hint=hint)
                    result = {
                        **result,
                        "segments": [
                            s
                            for s in result.get("segments", [])
                            if s.get("avg_logprob", 0) >= -1 and s.get("compression_ratio", 0) <= 2.4
                        ],
                    }
                    candidate = collect(result, start, cursor, owned_end, clip=True)
                    if placement == "speech" and candidate and result.get("language"):
                        # Detect language on the speech island itself. A long
                        # English lead-in can otherwise hide a brief film clip.
                        language_hints[a, b] = result["language"]

                    def uncovered(readings, lo, hi):
                        words = [(w["start"] - 0.25, w["end"] + 0.25) for s in readings for w in s["words"]]
                        return sum(b - a for a, b in subtract([(lo, hi)], words))

                    if uncovered(candidate, cursor, owned_end) < uncovered(segments, cursor, owned_end):
                        retained = []
                        for lo, hi in subtract([(0, duration)], [(cursor, owned_end)]):
                            retained.extend(collect({"segments": segments}, 0, lo, hi, clip=True))
                        segments = retained + candidate
                        segments.sort(key=lambda s: (s["start"], s["end"]))
                    cursor = owned_end
        remaining = unresolved_speech(speech, segments)
        warnings = []
        if remaining:
            if not any(s.get("words") for s in segments):
                assert_recovered(speech, segments)
            # VAD is evidence for recovery, not proof that every sound must
            # have words. Preserve uncertain audio explicitly for detection.
            for a, b in remaining:
                segments.append({"start": a, "end": b, "text": UNCLEAR_AUDIO_TEXT, "words": [], "untranscribed": True})
                warnings.append(
                    f"Audio at {a:.2f}-{b:.2f}s remains unclear after recovery. Retained as a timed transcript entry."
                )
            segments.sort(key=lambda s: (s["start"], s["end"]))
            logger.warning("Retained %d uncertain audio regions as timed transcript entries", len(remaining))
        if progress_callback is not None:
            progress_callback(100)
        logger.info("Retried %d word-free windows, recovered %d segments", len(gaps), len(recovered))
        logger.info(
            "Speech coverage audit: %d suspect regions, %d stronger recovery segments",
            len(strong_windows),
            len(stronger),
        )
        logger.info("Decoded complete audio timeline: %.3fs in %d overlapping windows", duration, count)
        return {
            "segments": segments,
            "language": language or "en",
            "warnings": warnings,
            "untranscribed_regions": [{"start": a, "end": b} for a, b in remaining],
        }
