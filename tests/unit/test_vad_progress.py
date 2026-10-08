"""Detection hooks report batches without leaking onto the cached model."""

from types import SimpleNamespace

import pytest

from app.vad_progress import speech_detection_progress


class Inference:
    def slide(self, waveform, sample_rate, hook=None):
        hook(completed=0, total=3)
        hook(completed=2, total=3)
        hook(completed=4, total=3)  # Pyannote's last batch may overshoot.
        if waveform == "fail":
            raise RuntimeError("detection failed")
        return "segments"


@pytest.mark.parametrize("fail", [False, True])
def test_batch_progress_forwards_existing_hook_and_restores_method(fail):
    inference = Inference()
    model = SimpleNamespace(vad_model=SimpleNamespace(vad_pipeline=SimpleNamespace(_segmentation=inference)))
    progress, existing = [], []
    with speech_detection_progress(model, progress.append):

        def run():
            return inference.slide("fail" if fail else "audio", 16000, hook=lambda **event: existing.append(event))

        if fail:
            with pytest.raises(RuntimeError, match="detection failed"):
                run()
        else:
            assert run() == "segments"
    assert progress == pytest.approx([0, 200 / 3, 100])
    assert len(existing) == 3
    assert "slide" not in vars(inference)
    assert inference.slide.__func__ is Inference.slide


def test_existing_instance_override_is_restored():
    inference = Inference()
    previous = inference.slide
    inference.slide = previous
    model = SimpleNamespace(vad_model=SimpleNamespace(vad_pipeline=SimpleNamespace(_segmentation=inference)))
    with speech_detection_progress(model, lambda _: None):
        inference.slide("audio", 16000)
    assert inference.slide is previous


def test_other_vad_backends_continue_without_a_hook():
    with speech_detection_progress(SimpleNamespace(), lambda _: None):
        pass
