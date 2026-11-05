"""Reconcile neighbouring audio windows without discarding either overlap."""

import re
from dataclasses import dataclass


@dataclass
class Reading:
    word: dict
    window_start: float
    window_end: float
    source: tuple[int, int]
    logprob: float
    language: str | None = None

    @property
    def context(self):
        return min(self.word["start"] - self.window_start, self.window_end - self.word["end"])


def normalized(readings):
    return "".join(re.sub(r"\W+", "", r.word["word"].casefold()) for r in readings)


def match_score(left, right):
    a, b = left[0].word["start"], left[-1].word["end"]
    c, d = right[0].word["start"], right[-1].word["end"]
    distance = abs((a + b - c - d) / 2)
    if normalized(left) and normalized(left) == normalized(right) and distance <= 1.0:
        return 3 * max(len(left), len(right)) - distance
    # A different spelling at the same audio instant is an alternative reading.
    # Grouped matches above also handle "Milkybar" versus "Milky bar".
    overlap = min(b, d) - max(a, c)
    if len(left) == len(right) == 1 and distance <= 0.35 and overlap > 0.6 * max(0.01, min(b - a, d - c)):
        return 0.5
    return None


def preferred(left, right):
    def quality(group):
        probabilities = [r.word["probability"] for r in group if "probability" in r.word]
        return sum(probabilities) / len(probabilities) if probabilities else None

    lq, rq = quality(left), quality(right)
    if lq is not None and rq is not None and abs(lq - rq) > 0.15:
        return left if lq > rq else right
    # The centre of a window has audio context on both sides of the phrase.
    return left if min(r.context for r in left) >= min(r.context for r in right) else right


def reconcile(left, right):
    """Time-constrained sequence alignment on just the shared audio region."""
    n, m = len(left), len(right)
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    steps = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i:
                scores[i][j] = scores[i - 1][j]
                steps[i, j] = (1, 0)
            if j and (not i or scores[i][j - 1] > scores[i][j]):
                scores[i][j] = scores[i][j - 1]
                steps[i, j] = (0, 1)
            for a in range(1, min(3, i) + 1):
                for b in range(1, min(3, j) + 1):
                    score = match_score(left[i - a : i], right[j - b : j])
                    if score is not None and scores[i - a][j - b] + score > scores[i][j]:
                        scores[i][j] = scores[i - a][j - b] + score
                        steps[i, j] = (a, b)
    groups = []
    i, j = n, m
    while i or j:
        a, b = steps[i, j]
        if a and b:
            groups.append(preferred(left[i - a : i], right[j - b : j]))
        elif a:
            groups.append(left[i - a : i])
        else:
            groups.append(right[j - b : j])
        i, j = i - a, j - b
    return sorted((r for group in reversed(groups) for r in group), key=lambda r: (r.word["start"], r.word["end"]))


def stitch_windows(windows):
    """Keep unique words from either window, deduplicating only shared time."""
    merged = []
    previous_end = 0
    for index, (start, end, segments) in enumerate(windows):
        incoming = [
            Reading(w, start, end, (index, sid), s.get("avg_logprob", 0), s.get("language"))
            for sid, s in enumerate(segments)
            for w in s["words"]
        ]
        incoming.sort(key=lambda r: (r.word["start"], r.word["end"]))
        # Limit matching to the actual common audio, with timing-drift slack.
        if start < previous_end:
            left = [r for r in merged if r.word["end"] >= start - 1]
            right = [r for r in incoming if r.word["start"] <= previous_end + 1]
            prefix = [r for r in merged if r.word["end"] < start - 1]
            suffix = [r for r in incoming if r.word["start"] > previous_end + 1]
            merged = prefix + reconcile(left, right) + suffix
        else:
            merged.extend(incoming)
        previous_end = end
    segments = []
    source = None
    for r in merged:
        if source != r.source or r.word["start"] - segments[-1]["end"] >= 5:
            segments.append(
                {
                    "start": r.word["start"],
                    "end": r.word["end"],
                    "text": "",
                    "words": [],
                    "avg_logprob": r.logprob,
                    "language": r.language,
                }
            )
        s = segments[-1]
        s["words"].append(dict(r.word))
        s["end"] = max(s["end"], r.word["end"])
        s["text"] = "".join(w["word"] for w in s["words"]).strip()
        source = r.source
    return segments
