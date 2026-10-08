"""Forward Pyannote inference batch progress during a serialized ASR job."""

from contextlib import contextmanager


@contextmanager
def speech_detection_progress(model, callback):
    """WhisperMLX does not forward its VAD inference hook, so scope it here.

    All model calls are serialized by the service queue. Restore the cached
    inference object's method even when detection or transcription fails.
    """
    vad = getattr(model, "vad_model", None)
    pipeline = getattr(vad, "vad_pipeline", None)
    inference = getattr(pipeline, "_segmentation", None)
    if callback is None or inference is None:
        yield
        return
    original = inference.slide
    had_override = "slide" in vars(inference)
    previous = vars(inference).get("slide")

    def slide(waveform, sample_rate, hook=None):
        def report(*, completed, total):
            if hook is not None:
                hook(completed=completed, total=total)
            if total > 0:
                callback(min(100.0, max(0.0, completed / total * 100)))

        return original(waveform, sample_rate, hook=report)

    inference.slide = slide
    try:
        yield
    finally:
        if had_override:
            inference.slide = previous
        else:
            del inference.slide
