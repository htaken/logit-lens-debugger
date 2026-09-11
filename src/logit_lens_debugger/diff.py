"""Token-level diff between a source text and an LLM output, and position selection.

Pure stdlib -- no torch/transformers import -- so it can be exercised with the
``ToyTokenizer`` used in tests and works for any :class:`~.adapters.base.ModelAdapter`.
The diff is computed over the *tokenizer's own token strings*, not characters, so it
needs no ``offset_mapping`` and works for languages without whitespace.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Sequence

from .types import DiffSpan

#: Character classes whose runs get swallowed whole by a neighboring diff span
#: (e.g. Qwen tokenizes "1500" as "1","5","0","0", so a change touching one digit
#: should expand to cover the whole number). "alpha" is deliberately excluded: with
#: real tokenizers, word tokens carry a leading space (" sales", " were", " yen")
#: and every English word is class "alpha", so expanding it would swallow whole
#: sentences up to the next punctuation. See the word-boundary rule below.
_EXPANDABLE_CLASSES = frozenset({"digit", "kana", "kanji"})


def _char_name(char: str) -> str:
    try:
        return unicodedata.name(char)
    except ValueError:
        return ""


def token_class(token: str) -> str:
    """Classify a token string for diff-span expansion purposes.

    The token is stripped of leading/trailing whitespace first. Returns one of
    ``"space"``, ``"digit"``, ``"alpha"``, ``"kana"``, ``"kanji"``, ``"punct"`` or
    ``"mixed"``. Only ``{"digit", "kana", "kanji"}`` are *expandable* (see
    ``_EXPANDABLE_CLASSES``); ``"alpha"`` is a valid classification but is not
    treated as expandable.
    """
    stripped = token.strip()
    if not stripped:
        return "space"
    if stripped.isdigit():
        return "digit"
    if all(unicodedata.category(c).startswith("L") and "LATIN" in _char_name(c) for c in stripped):
        return "alpha"
    if all(("HIRAGANA" in _char_name(c)) or ("KATAKANA" in _char_name(c)) for c in stripped):
        return "kana"
    if all(_char_name(c).startswith("CJK UNIFIED IDEOGRAPH") for c in stripped):
        return "kanji"
    if all(unicodedata.category(c).startswith(("P", "S")) for c in stripped):
        return "punct"
    return "mixed"


def _starts_new_word(token: str) -> bool:
    """Whether ``token`` opens a fresh word (leading-space convention of HF tokenizers).

    Expansion must never cross this boundary: a token with a leading space marks
    the start of a new word/number, so whatever comes before it belongs to a
    different unit even if it happens to share the same character class.
    """
    return token[:1].isspace()


def _expand_nonempty(tokens: list[str], start: int, end: int) -> tuple[int, int]:
    left_class = token_class(tokens[start])
    if left_class in _EXPANDABLE_CLASSES:
        while (
            start > 0
            and not _starts_new_word(tokens[start])
            and token_class(tokens[start - 1]) == left_class
        ):
            start -= 1
    right_class = token_class(tokens[end - 1])
    if right_class in _EXPANDABLE_CLASSES:
        while (
            end < len(tokens)
            and not _starts_new_word(tokens[end])
            and token_class(tokens[end]) == right_class
        ):
            end += 1
    return start, end


def _expand_empty(
    tokens: list[str],
    boundary: int,
    left_anchor_class: str | None,
    right_anchor_class: str | None,
    left_joins_previous_word: bool,
) -> tuple[int, int]:
    """Expand a zero-length range sitting at ``boundary`` (an insert/delete point).

    There is no token *in* the range to read a class from, so each side's anchor
    class is the class of the corresponding boundary token of the *other*
    (non-empty) side of the same opcode -- computed before that side was itself
    expanded. A neighbor is absorbed only if its class matches that anchor;
    absorbing an unrelated expandable neighbor (e.g. a kana particle sitting next
    to a deleted number) would misrepresent the change.

    The first absorption on each side must also respect word boundaries, or an
    insert/delete right after one word/number can be misreported as a replace of
    the *previous* one:

    - Left: absorbing ``tokens[boundary - 1]`` is only valid if the changed
      content itself continues the word before it -- i.e. the non-empty side's
      first token does not start a new word (``left_joins_previous_word``).
    - Right: absorbing ``tokens[boundary]`` is only valid if that token does not
      itself start a new word (it would otherwise be a separate word/number that
      merely happens to sit right after the change).

    Once the first neighbor on a side is absorbed, expansion continues outward
    with the normal same-class / word-boundary rule.
    """
    start = end = boundary
    if (
        left_joins_previous_word
        and left_anchor_class is not None
        and left_anchor_class in _EXPANDABLE_CLASSES
        and start > 0
        and token_class(tokens[start - 1]) == left_anchor_class
    ):
        start -= 1
        while (
            start > 0
            and not _starts_new_word(tokens[start])
            and token_class(tokens[start - 1]) == left_anchor_class
        ):
            start -= 1
    if (
        right_anchor_class is not None
        and right_anchor_class in _EXPANDABLE_CLASSES
        and end < len(tokens)
        and not _starts_new_word(tokens[end])
        and token_class(tokens[end]) == right_anchor_class
    ):
        end += 1
        while (
            end < len(tokens)
            and not _starts_new_word(tokens[end])
            and token_class(tokens[end]) == right_anchor_class
        ):
            end += 1
    return start, end


def _expand_span(
    source: list[str],
    output: list[str],
    source_range: tuple[int, int],
    output_range: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Expand one opcode's source/output ranges to whole same-class token runs.

    ``replace`` spans (both ranges non-empty) expand each side independently.
    ``insert``/``delete`` spans have one empty side; its anchor classes -- and
    whether it may even reach across the boundary at all (``left_joins_previous_word``,
    see ``_expand_empty``) -- are taken from the *unexpanded* boundary tokens of
    the non-empty side, so e.g. an inserted digit only pulls in neighboring source
    tokens that are themselves digits and only when the inserted content itself
    doesn't start a fresh word/number.
    """
    source_empty = source_range[0] >= source_range[1]
    output_empty = output_range[0] >= output_range[1]

    if not source_empty and not output_empty:
        return (
            _expand_nonempty(source, *source_range),
            _expand_nonempty(output, *output_range),
        )
    if output_empty:
        # delete: source non-empty, output empty.
        first = source[source_range[0]]
        left_anchor = token_class(first)
        right_anchor = token_class(source[source_range[1] - 1])
        expanded_source = _expand_nonempty(source, *source_range)
        expanded_output = _expand_empty(
            output, output_range[0], left_anchor, right_anchor, not _starts_new_word(first)
        )
        return expanded_source, expanded_output
    # insert: source empty, output non-empty.
    first = output[output_range[0]]
    left_anchor = token_class(first)
    right_anchor = token_class(output[output_range[1] - 1])
    expanded_output = _expand_nonempty(output, *output_range)
    expanded_source = _expand_empty(
        source, source_range[0], left_anchor, right_anchor, not _starts_new_word(first)
    )
    return expanded_source, expanded_output


def _span_kind(source_range: tuple[int, int], output_range: tuple[int, int]) -> str:
    source_nonempty = source_range[0] < source_range[1]
    output_nonempty = output_range[0] < output_range[1]
    if source_nonempty and output_nonempty:
        return "replace"
    if output_nonempty:
        return "insert"
    return "delete"


def _make_span(
    source: list[str],
    output: list[str],
    source_range: tuple[int, int],
    output_range: tuple[int, int],
    kind: str,
) -> DiffSpan:
    s0, s1 = source_range
    o0, o1 = output_range
    return DiffSpan(
        kind=kind,
        source_range=(s0, s1),
        output_range=(o0, o1),
        source_text="".join(source[s0:s1]),
        output_text="".join(output[o0:o1]),
    )


def _ranges_overlap(prev_end: int, next_start: int) -> bool:
    return next_start < prev_end


def _merge_spans(spans: list[DiffSpan], source: list[str], output: list[str]) -> list[DiffSpan]:
    if not spans:
        return spans
    current = sorted(spans, key=lambda s: (s.output_range[0], s.source_range[0]))
    changed = True
    while changed:
        changed = False
        merged: list[DiffSpan] = []
        for span in current:
            if merged and (
                _ranges_overlap(merged[-1].output_range[1], span.output_range[0])
                or _ranges_overlap(merged[-1].source_range[1], span.source_range[0])
            ):
                prev = merged[-1]
                source_range = (
                    min(prev.source_range[0], span.source_range[0]),
                    max(prev.source_range[1], span.source_range[1]),
                )
                output_range = (
                    min(prev.output_range[0], span.output_range[0]),
                    max(prev.output_range[1], span.output_range[1]),
                )
                kind = _span_kind(source_range, output_range)
                merged[-1] = _make_span(source, output, source_range, output_range, kind)
                changed = True
            else:
                merged.append(span)
        current = sorted(merged, key=lambda s: (s.output_range[0], s.source_range[0]))
    return current


def diff_token_sequences(
    source_tokens: Sequence[str], output_tokens: Sequence[str], expand_spans: bool = True
) -> list[DiffSpan]:
    """Diff two token-string sequences and return the non-equal spans.

    Uses :class:`difflib.SequenceMatcher` (``autojunk=False``) over the token
    strings themselves. When ``expand_spans`` (the default), each span is grown to
    cover whole runs of the same character class (digits, kana, kanji -- *not*
    Latin letters, see ``_EXPANDABLE_CLASSES``) so e.g. a single-digit-token change
    inside a longer number covers the whole number. Expansion never crosses a
    word boundary: with real tokenizers a word-initial token carries a leading
    space (``" sales"``), and expansion stops rather than continuing into a
    neighboring word/number even if it happens to share the same class. Overlapping
    expanded spans are merged. Spans are sorted by
    ``(output_range.start, source_range.start)``.
    """
    source = list(source_tokens)
    output = list(output_tokens)
    opcodes = difflib.SequenceMatcher(None, source, output, autojunk=False).get_opcodes()

    spans: list[DiffSpan] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        source_range = (i1, i2)
        output_range = (j1, j2)
        kind = tag
        if expand_spans:
            source_range, output_range = _expand_span(source, output, source_range, output_range)
            kind = _span_kind(source_range, output_range)
        spans.append(_make_span(source, output, source_range, output_range, kind))

    spans = _merge_spans(spans, source, output)
    spans.sort(key=lambda s: (s.output_range[0], s.source_range[0]))
    return spans


def select_positions(
    spans: Sequence[DiffSpan],
    prompt_length: int,
    output_length: int,
    context: int = 0,
    include_after_delete: bool = True,
) -> list[tuple[int, DiffSpan | None]]:
    """Pick the residual-stream positions to analyze for a set of diff spans.

    Output token ``j`` is predicted from residual stream position
    ``prompt_length + j - 1``; positions that would be negative (the first output
    token with no prompt) are skipped. For ``replace``/``insert`` spans every output
    token in ``[start - context, end + context)`` is inspected. ``delete`` spans have
    no output tokens of their own; if ``include_after_delete``, the output token right
    after the deletion point (plus ``context``) is inspected instead. Indices are
    clipped to ``[0, output_length)``. Returns ``(position, span)`` pairs, deduplicated
    by position (keeping the first span encountered) and sorted by position.
    """
    if output_length <= 0:
        return []

    chosen: dict[int, DiffSpan | None] = {}
    for span in spans:
        start, end = span.output_range
        if span.kind == "delete":
            if not include_after_delete:
                continue
            candidate_range = range(start - context, start + 1 + context)
        else:
            candidate_range = range(start - context, end + context)

        seen_js: set[int] = set()
        for raw_j in candidate_range:
            j = max(0, min(output_length - 1, raw_j))
            if j in seen_js:
                continue
            seen_js.add(j)
            position = prompt_length + j - 1
            if position < 0:
                continue
            chosen.setdefault(position, span)

    return [(position, chosen[position]) for position in sorted(chosen)]
