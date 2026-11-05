"""Shared audio must rescue omissions, without erasing real repetitions."""

from app.transcript_stitch import stitch_windows


def window(start, end, *words):
    return (start, end, [{"words": [{"word": " " + text, "start": a, "end": b} for text, a, b in words]}])


def texts(windows):
    return [w["word"].strip() for s in stitch_windows(windows) for w in s["words"]]


def test_retains_overlap_words_only_recognized_by_right_window():
    assert texts(
        [
            window(0, 24, ("Before.", 15, 15.5)),
            window(16, 44, ("Electricity", 18, 18.5), ("savings.", 19, 19.5)),
        ]
    ) == ["Before.", "Electricity", "savings."]


def test_retains_overlap_words_only_recognized_by_left_window():
    assert texts(
        [
            window(0, 24, ("Milkybar", 21, 21.5), ("ad.", 22, 22.5)),
            window(16, 44, ("Programme.", 25, 26)),
        ]
    ) == ["Milkybar", "ad.", "Programme."]


def test_matches_shared_phrase_with_punctuation_and_timing_drift():
    assert texts(
        [
            window(0, 24, ("Save", 20, 20.5), ("electricity!", 20.6, 21)),
            window(16, 44, ("save", 20.4, 20.9), ("electricity.", 21, 21.4)),
        ]
    ) == ["save", "electricity."]


def test_preserves_a_phrase_repeated_at_different_times():
    assert texts(
        [
            window(0, 24, ("Save", 18, 18.3), ("now.", 18.4, 18.7), ("Save", 21, 21.3), ("now.", 21.4, 21.7)),
            window(16, 44, ("Save", 18, 18.3), ("now.", 18.4, 18.7), ("Save", 21, 21.3), ("now.", 21.4, 21.7)),
        ]
    ) == ["Save", "now.", "Save", "now."]


def test_a_single_word_repeated_across_the_seam_is_not_collapsed():
    assert texts(
        [
            window(0, 24, ("Yes", 19, 19.2), ("yes", 19.6, 19.8)),
            window(16, 44, ("Yes", 19, 19.2), ("yes", 19.6, 19.8)),
        ]
    ) == ["Yes", "yes"]


def test_handles_different_word_splitting_without_duplicating_brand():
    assert texts(
        [
            window(0, 24, ("New", 21, 21.3), ("Milkybar.", 21.4, 22)),
            window(16, 44, ("New", 21, 21.3), ("Milky", 21.4, 21.7), ("bar.", 21.7, 22)),
        ]
    ) == ["New", "Milky", "bar."]


def test_alternative_at_same_time_prefers_more_context():
    assert texts(
        [
            window(0, 24, ("sail", 23, 23.6)),
            window(16, 44, ("sale", 23.1, 23.7)),
        ]
    ) == ["sale"]


def test_high_word_confidence_can_override_window_position():
    left = window(0, 24, ("sale", 23, 23.6))
    right = window(16, 44, ("sail", 23, 23.6))
    left[2][0]["words"][0]["probability"] = 0.99
    right[2][0]["words"][0]["probability"] = 0.3
    assert texts([left, right]) == ["sale"]


def test_empty_neighbour_cannot_erase_existing_reading():
    assert texts([window(0, 24, ("Advert.", 21, 22)), (16, 44, [])]) == ["Advert."]


def test_identical_text_outside_shared_audio_is_preserved():
    assert texts([window(0, 24, ("Again.", 1, 2)), window(16, 44, ("Again.", 30, 31))]) == ["Again.", "Again."]


def test_three_windows_keep_one_copy_of_each_spoken_word():
    assert texts(
        [
            window(0, 24, ("first", 20, 21)),
            window(16, 44, ("first", 20, 21), ("second", 40, 41)),
            window(36, 54, ("second", 40, 41), ("third", 50, 51)),
        ]
    ) == ["first", "second", "third"]
