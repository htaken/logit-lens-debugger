"""Tests for logit_lens_debugger.diff -- pure functions, no model required."""

from __future__ import annotations

import pytest

from logit_lens_debugger.diff import diff_token_sequences, select_positions, token_class
from logit_lens_debugger.types import DiffSpan


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (" 150", "digit"),
        ("abc", "alpha"),
        ("カタカナー", "kana"),
        ("漢字", "kanji"),
        ("、", "punct"),
        (" ", "space"),
        ("a1", "mixed"),
    ],
)
def test_token_class(token, expected):
    assert token_class(token) == expected


def test_diff_replace_whole_word_number():
    spans = diff_token_sequences(["sales", "were", "150", "yen"], ["sales", "were", "1500", "yen"])
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (2, 3)
    assert span.output_range == (2, 3)
    assert span.source_text == "150"
    assert span.output_text == "1500"


def test_digit_expansion_on():
    spans = diff_token_sequences(["1", "5", "0"], ["1", "5", "0", "0"], expand_spans=True)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (0, 3)
    assert span.output_range == (0, 4)
    assert span.source_text == "150"
    assert span.output_text == "1500"


def test_digit_expansion_off():
    spans = diff_token_sequences(["1", "5", "0"], ["1", "5", "0", "0"], expand_spans=False)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "insert"
    assert span.source_range == (3, 3)
    assert span.output_range == (3, 4)


def test_expansion_does_not_cross_class_boundary():
    spans = diff_token_sequences(
        ["x", "1", "5", "0", "yen"], ["x", "1", "5", "0", "0", "yen"], expand_spans=True
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.output_range == (1, 5)
    assert span.source_range == (1, 4)
    assert span.output_text == "1500"
    assert span.source_text == "150"


def test_merging_of_overlapping_expanded_spans():
    # Two separate inserts ("9" at the front, "0" at the back of "150") each expand
    # to swallow the whole surrounding digit run; the expanded ranges overlap and
    # must be merged into a single replace span. "a" and "b" are alpha (not
    # expandable, see the class boundary tests below) so they stay out of the span.
    spans = diff_token_sequences(
        ["a", "1", "5", "0", "b"], ["a", "9", "1", "5", "0", "0", "b"], expand_spans=True
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (1, 4)
    assert span.output_range == (1, 6)
    assert span.source_text == "150"
    assert span.output_text == "91500"


def test_identical_sequences_produce_no_spans():
    assert diff_token_sequences(["a", "b", "c"], ["a", "b", "c"]) == []


def test_alpha_is_not_expandable_across_word_boundaries():
    # With real tokenizers, word tokens carry a leading space and every English
    # word is class "alpha". Since "alpha" is not expandable, a one-word replace
    # must stay a single-token span instead of swallowing the whole sentence.
    source = [" the", " capital", " of", " japan", " is", " tokyo"]
    output = [" the", " capital", " of", " japan", " is", " osaka"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (5, 6)
    assert span.output_range == (5, 6)
    assert span.source_text == " tokyo"
    assert span.output_text == " osaka"


def test_digit_expansion_with_leading_space_tokens():
    # Qwen-style tokenization: word tokens carry a leading space, digit
    # continuations do not. The inserted digit ("0", no leading space) joins the
    # previous word/number, so it must pull in the whole number (including its
    # leading space) but not " were".
    source = [" sales", " were", " 1", "5", "0", " yen"]
    output = [" sales", " were", " 1", "5", "0", "0", " yen"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (2, 5)
    assert span.output_range == (2, 6)
    assert span.source_text == " 150"
    assert span.output_text == " 1500"


def test_insert_of_new_number_after_existing_one_stays_an_insert():
    # A brand-new number (" 200", leading space) inserted right after an existing
    # one (" 150") must not be misreported as replacing the previous number: the
    # inserted content itself starts a new word, so the empty (source) side must
    # not reach across the boundary to absorb "0" from "150".
    source = [" 1", "5", "0", " yen"]
    output = [" 1", "5", "0", " 2", "0", "0", " yen"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "insert"
    assert span.source_range == (3, 3)
    assert span.output_range == (3, 6)
    assert span.output_text == " 200"


def test_delete_of_new_number_before_existing_one_stays_a_delete():
    # Mirror of the insert case above: deleting " 200" that sits right before an
    # existing " yen" must not absorb tokens from "150" on its left.
    source = [" 1", "5", "0", " 2", "0", "0", " yen"]
    output = [" 1", "5", "0", " yen"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "delete"
    assert span.source_range == (3, 6)
    assert span.output_range == (3, 3)
    assert span.source_text == " 200"


def test_expansion_stops_at_adjacent_number_word_boundary():
    # Two adjacent numbers (" 2024" and " 150"/" 1500"); expanding the second
    # number's insert must not cross into the first number even though both are
    # digit runs.
    source = [" 2", "0", "2", "4", " 1", "5", "0"]
    output = [" 2", "0", "2", "4", " 1", "5", "0", "0"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.source_range == (4, 7)
    assert span.output_range == (4, 8)
    assert span.source_text == " 150"
    assert span.output_text == " 1500"


def test_japanese_delete_does_not_swallow_neighbours():
    # Deleting "150" (digit) between "は" (kana) and "万円" (kanji) must not
    # absorb either neighbor even though kana/kanji are themselves expandable.
    source = ["売", "上", "は", "1", "5", "0", "万", "円"]
    output = ["売", "上", "は", "万", "円"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "delete"
    assert span.output_range == (3, 3)
    assert span.source_range == (3, 6)
    assert span.source_text == "150"


def test_japanese_insert_of_same_class_digit_still_expands():
    source = ["売", "上", "は", "1", "5", "0", "万", "円"]
    output = ["売", "上", "は", "1", "5", "0", "0", "万", "円"]
    spans = diff_token_sequences(source, output)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (3, 6)
    assert span.output_range == (3, 7)
    assert span.source_text == "150"
    assert span.output_text == "1500"


def test_kanji_run_expansion():
    # Inserting "語" right after the kanji run "日本" pulls the whole run into
    # the span (source has no more tokens to give it a class of its own).
    spans = diff_token_sequences(["日", "本"], ["日", "本", "語"])
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "replace"
    assert span.source_range == (0, 2)
    assert span.output_range == (0, 3)
    assert span.source_text == "日本"
    assert span.output_text == "日本語"


def _span(kind, source_range, output_range, source_text="", output_text=""):
    return DiffSpan(
        kind=kind,
        source_range=source_range,
        output_range=output_range,
        source_text=source_text,
        output_text=output_text,
    )


def test_select_positions_replace():
    span = _span("replace", (2, 3), (2, 3), "150", "1500")
    assert select_positions([span], prompt_length=0, output_length=5) == [(1, span)]


def test_select_positions_replace_with_context():
    span = _span("replace", (2, 3), (2, 3), "150", "1500")
    pairs = select_positions([span], prompt_length=0, output_length=5, context=1)
    assert pairs == [(0, span), (1, span), (2, span)]


def test_select_positions_insert():
    span = _span("insert", (3, 3), (3, 4), "", "0")
    assert select_positions([span], prompt_length=0, output_length=5) == [(2, span)]


def test_select_positions_delete_with_include_after_delete():
    span = _span("delete", (2, 3), (2, 2), "foo", "")
    assert select_positions([span], prompt_length=0, output_length=5) == [(1, span)]


def test_select_positions_delete_without_include_after_delete():
    span = _span("delete", (2, 3), (2, 2), "foo", "")
    assert (
        select_positions([span], prompt_length=0, output_length=5, include_after_delete=False) == []
    )


def test_select_positions_clips_at_both_ends():
    left_span = _span("replace", (0, 1), (0, 1), "x", "y")
    left_positions = [p for p, _ in select_positions([left_span], 0, 5, context=3)]
    assert left_positions == [0, 1, 2]  # j=-1 skipped (negative position)

    right_span = _span("replace", (4, 5), (4, 5), "x", "y")
    right_positions = [p for p, _ in select_positions([right_span], 0, 5, context=3)]
    assert right_positions == [0, 1, 2, 3]  # j clipped to output_length - 1 = 4


def test_select_positions_skips_negative_position_with_no_prompt():
    span = _span("replace", (0, 1), (0, 1), "x", "y")
    # j=0 with prompt_length=0 maps to position -1, which must be dropped.
    assert select_positions([span], prompt_length=0, output_length=5) == []


def test_select_positions_dedupes_touching_spans():
    span_a = _span("replace", (0, 1), (0, 1), "a", "b")
    span_b = _span("replace", (1, 2), (1, 2), "c", "d")
    pairs = select_positions([span_a, span_b], prompt_length=0, output_length=5, context=1)
    assert pairs == [(0, span_a), (1, span_b)]


def test_select_positions_no_spans_gives_no_positions():
    assert select_positions([], prompt_length=0, output_length=5) == []
