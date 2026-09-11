"""High-level API: ``LogitLensAnalyzer.analyze()`` / ``.compare()`` and model comparison."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .adapters.base import ModelAdapter
from .adapters.registry import load_adapter
from .anomaly import detect_anomalies as _score_anomalies
from .comparison import align_layers, compare_results, vocab_identical
from .diff import diff_token_sequences, select_positions
from .engine import LogitLensEngine
from .types import (
    AnomalyConfig,
    AnomalyReport,
    ComparisonResult,
    LensMode,
    LogitLensResult,
    MultiPositionResult,
    PositionResult,
)


def normalize_index(index: int, length: int) -> int:
    """Turn a possibly-negative token index into a valid non-negative one."""
    if length <= 0:
        raise ValueError("Input produced no tokens")
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise IndexError(f"token_index {index} is out of range for {length} tokens")
    return resolved


class LogitLensAnalyzer:
    """Run a logit lens over a Hugging Face causal LM.

    Example::

        analyzer = LogitLensAnalyzer("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
        result = analyzer.analyze("The capital of Japan is", token_index=-1, top_k=10)
        result.layers[10].top_tokens[0].token
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        lens_mode: LensMode | str = LensMode.FINAL_NORM,
        adapter: ModelAdapter | None = None,
        **model_kwargs: Any,
    ):
        if adapter is None:
            if model_name is None:
                raise ValueError("Provide either model_name or an adapter")
            adapter = load_adapter(model_name, device=device, dtype=dtype, **model_kwargs)
        self.adapter = adapter
        self.engine = LogitLensEngine(adapter, lens_mode)

    @property
    def lens_mode(self) -> LensMode:
        return self.engine.lens_mode

    @lens_mode.setter
    def lens_mode(self, value: LensMode | str) -> None:
        self.engine.lens_mode = LensMode(value)

    @property
    def model_name(self) -> str:
        return self.adapter.model_name

    # ------------------------------------------------------------------ analyze

    def analyze(
        self,
        text: str,
        token_index: int = -1,
        top_k: int = 10,
        target_token: str | int | None = None,
    ) -> LogitLensResult:
        """Project the residual stream at ``token_index`` through every layer.

        Predictions are for the token *following* ``token_index``. If
        ``target_token`` is omitted and a following token exists in ``text``,
        that token becomes the target so its rank/probability per layer is
        reported. To ask why the model produced the token at position ``j``,
        analyze ``token_index=j - 1``.
        """
        result, _ = self._analyze_detailed(text, token_index, top_k, target_token)
        return result

    def _analyze_detailed(
        self,
        text: str,
        token_index: int = -1,
        top_k: int = 10,
        target_token: str | int | None = None,
        keep_log_probs: bool = False,
    ) -> tuple[LogitLensResult, torch.Tensor | None]:
        """Shared body of :meth:`analyze` and :meth:`compare`.

        Returns the usual :class:`LogitLensResult` plus, when ``keep_log_probs``,
        the full-vocabulary log-probabilities from :meth:`~.engine.LogitLensEngine.run_detailed`
        (``[num_layers + 1, 1, V]``, one position) -- ``None`` otherwise.
        """
        input_ids = self.adapter.tokenize(text)
        ids: list[int] = input_ids[0].tolist()
        position = normalize_index(token_index, len(ids))

        if target_token is not None:
            target_id = self._resolve_target(target_token)
        elif position + 1 < len(ids):
            target_id = ids[position + 1]
        else:
            target_id = None

        output = self.engine.run_detailed(
            input_ids,
            [position],
            top_k=top_k,
            target_token_id=target_id,
            keep_log_probs=keep_log_probs,
        )
        result = LogitLensResult(
            model_name=self.model_name,
            lens_mode=self.lens_mode,
            text=text,
            input_tokens=[self.adapter.decode_token(i) for i in ids],
            input_token_ids=ids,
            token_index=position,
            top_k=top_k,
            layers=output.layers[0],
            target_token=self.adapter.decode_token(target_id) if target_id is not None else None,
            target_token_id=target_id,
            metadata={"device": str(self.adapter.device), "vocab_size": self.adapter.vocab_size},
        )
        return result, output.log_probs

    def _resolve_target(self, target_token: str | int) -> int:
        if isinstance(target_token, int):
            return target_token
        ids = self.adapter.encode(target_token)
        if not ids:
            raise ValueError(f"target_token {target_token!r} produced no tokens")
        if len(ids) > 1:
            pieces = [self.adapter.decode_token(i) for i in ids]
            warnings.warn(
                f"target_token {target_token!r} tokenizes to {len(ids)} pieces {pieces}; "
                f"tracking only the first piece {pieces[0]!r}",
                stacklevel=3,
            )
        return ids[0]

    def _output_ids(self, output: str | Sequence[int]) -> list[int]:
        """Token ids for ``output``: encoded if a string, taken as-is (a fresh list) otherwise."""
        return self.adapter.encode(output) if isinstance(output, str) else list(output)

    def _decode_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.adapter.decode_token(i) for i in ids]

    # ------------------------------------------------------------- multi-position

    def analyze_positions(
        self,
        text: str | Sequence[int],
        positions: Sequence[int],
        top_k: int = 10,
        targets: Sequence[int | str | None] | None = None,
        prompt_length: int = 0,
        chunk_size: int | None = None,
    ) -> MultiPositionResult:
        """Run the lens over several token positions of one context in one (or few) forwards.

        ``text`` may be a string (tokenized with the adapter) or an already-built id
        sequence, so callers that assembled a context themselves (e.g. prompt +
        generated output) can pass it straight through. ``positions`` are
        normalized (negative indices allowed), deduplicated and sorted ascending.

        Each position's target defaults to the token that actually follows it in
        the context; pass ``targets`` (same length as ``positions``, matched by
        original order) to override per position -- ``None`` entries keep the
        default, ints are used as-is, strings are resolved like ``analyze``'s
        ``target_token``.

        ``chunk_size`` trades memory for extra forward passes: ``None`` runs the
        engine once for all positions; otherwise positions are split into chunks
        of that size, each requiring its own forward. Results are identical either
        way.
        """
        if not positions:
            raise ValueError("positions must not be empty")

        ids: list[int] = (
            self.adapter.tokenize(text)[0].tolist() if isinstance(text, str) else list(text)
        )
        length = len(ids)
        normalized = [normalize_index(p, length) for p in positions]

        if targets is None:
            target_requests: Sequence[int | str | None] = [None] * len(positions)
        else:
            target_requests = targets
            if len(target_requests) != len(positions):
                raise ValueError(
                    f"targets has {len(target_requests)} entries, expected {len(positions)} "
                    "(one per position)"
                )

        by_position: dict[int, int | str | None] = {}
        for pos, treq in zip(normalized, target_requests, strict=True):
            by_position.setdefault(pos, treq)
        sorted_positions = sorted(by_position)

        target_ids: list[int | None] = []
        for pos in sorted_positions:
            treq = by_position[pos]
            if treq is None:
                target_ids.append(ids[pos + 1] if pos + 1 < length else None)
            else:
                target_ids.append(self._resolve_target(treq))

        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.adapter.device)
        if chunk_size is None:
            layers_by_position = self.engine.run(
                input_tensor, sorted_positions, top_k=top_k, target_token_id=target_ids
            )
        else:
            layers_by_position = []
            for start in range(0, len(sorted_positions), chunk_size):
                chunk_positions = sorted_positions[start : start + chunk_size]
                chunk_targets = target_ids[start : start + chunk_size]
                layers_by_position.extend(
                    self.engine.run(
                        input_tensor, chunk_positions, top_k=top_k, target_token_id=chunk_targets
                    )
                )

        position_results = [
            PositionResult(
                token_index=pos,
                layers=layers,
                target_token=self.adapter.decode_token(tid) if tid is not None else None,
                target_token_id=tid,
            )
            for pos, layers, tid in zip(
                sorted_positions, layers_by_position, target_ids, strict=True
            )
        ]

        return MultiPositionResult(
            model_name=self.model_name,
            lens_mode=self.lens_mode,
            text=self.adapter.decode(ids),
            input_tokens=[self.adapter.decode_token(i) for i in ids],
            input_token_ids=ids,
            prompt_length=prompt_length,
            top_k=top_k,
            positions=position_results,
            metadata={"device": str(self.adapter.device), "vocab_size": self.adapter.vocab_size},
        )

    def analyze_generation(
        self,
        prompt: str,
        output: str | Sequence[int],
        top_k: int = 10,
        positions: Sequence[int] | None = None,
        chunk_size: int | None = None,
    ) -> MultiPositionResult:
        """Analyze the residual stream that produced each (or select) token of ``output``.

        ``prompt`` and ``output`` are tokenized *separately* and concatenated so
        the id sequence matches a real generation exactly -- tokenizing the joined
        string could merge tokens across the prompt/output boundary and no longer
        match what was actually generated. Pass ``output`` as an id sequence to
        replay a generation log exactly; a string is encoded without special tokens.
        """
        prompt_ids: list[int] = self.adapter.tokenize(prompt)[0].tolist()
        output_ids = self._output_ids(output)
        context_ids = prompt_ids + output_ids
        prompt_length = len(prompt_ids)

        js = range(len(output_ids)) if positions is None else positions
        resolved_positions = [prompt_length + j - 1 for j in js]
        resolved_positions = [p for p in resolved_positions if p >= 0]

        return self.analyze_positions(
            context_ids,
            resolved_positions,
            top_k=top_k,
            prompt_length=prompt_length,
            chunk_size=chunk_size,
        )

    def analyze_diff(
        self,
        source: str,
        output: str | Sequence[int],
        prompt: str | None = None,
        top_k: int = 10,
        context: int = 0,
        expand_spans: bool = True,
        include_after_delete: bool = True,
        chunk_size: int | None = None,
    ) -> MultiPositionResult:
        """Diff ``source`` against ``output`` and analyze only the flagged positions.

        ``prompt`` is the text actually given to the model; when omitted, ``source``
        stands in for it (an approximation: the model saw the source and produced
        ``output``, so the source itself becomes the context). The diff is computed
        over each token's decoded string (see :mod:`.diff`), not character offsets,
        so it needs no fast-tokenizer support and works for languages without
        whitespace. If the token sequences are identical, no positions are analyzed
        and the engine is never invoked.
        """
        prompt_text = source if prompt is None else prompt
        prompt_ids: list[int] = self.adapter.tokenize(prompt_text)[0].tolist()
        output_ids = self._output_ids(output)
        context_ids = prompt_ids + output_ids
        prompt_length = len(prompt_ids)

        source_tokens = self._decode_tokens(self.adapter.encode(source))
        output_tokens = self._decode_tokens(output_ids)
        spans = diff_token_sequences(source_tokens, output_tokens, expand_spans=expand_spans)
        pairs = select_positions(
            spans,
            prompt_length,
            len(output_ids),
            context=context,
            include_after_delete=include_after_delete,
        )

        if not pairs:
            return MultiPositionResult(
                model_name=self.model_name,
                lens_mode=self.lens_mode,
                text=self.adapter.decode(context_ids),
                input_tokens=[self.adapter.decode_token(i) for i in context_ids],
                input_token_ids=context_ids,
                prompt_length=prompt_length,
                top_k=top_k,
                positions=[],
                diff_spans=spans,
                metadata={
                    "device": str(self.adapter.device),
                    "vocab_size": self.adapter.vocab_size,
                },
            )

        span_by_position = dict(pairs)
        result = self.analyze_positions(
            context_ids,
            list(span_by_position),
            top_k=top_k,
            prompt_length=prompt_length,
            chunk_size=chunk_size,
        )
        positions_with_spans = [
            replace(pos_result, diff_span=span_by_position.get(pos_result.token_index))
            for pos_result in result.positions
        ]
        return replace(result, positions=positions_with_spans, diff_spans=spans)

    # ----------------------------------------------------------------- anomaly

    def detect_anomalies(
        self,
        prompt: str,
        output: str | Sequence[int],
        source: str | None = None,
        top_k: int = 10,
        config: AnomalyConfig | None = None,
        only_diff: bool = False,
        context: int = 0,
        chunk_size: int | None = 64,
        entity_spans: Sequence[tuple[int, int]] | None = None,
    ) -> AnomalyReport:
        """Score every (or, with ``only_diff``, every diff-flagged) output token for anomalies.

        With ``only_diff`` (requires ``source``), only the positions
        :meth:`analyze_diff` flags are scored. Otherwise every output token is
        scored (:meth:`analyze_generation`); if ``source`` is also given, its diff
        against ``output`` is additionally computed so ``in_diff`` still reflects
        the changed span even though every position gets a score. ``chunk_size``
        defaults to 64 here (rather than ``None``) since a full scan can cover many
        more positions than a diff-driven one -- see
        :meth:`analyze_positions` for the memory/forward-pass trade-off.

        See :func:`logit_lens_debugger.anomaly.detect_anomalies` for the signal
        definitions and :class:`~.types.AnomalyConfig` for the weights/thresholds.
        ``entity_spans`` are output-token-index ranges from an external NER pass
        (see the ``entity`` signal); NER itself is out of scope for this package.
        """
        if only_diff and source is None:
            raise ValueError("only_diff=True requires source")

        source_tokens = (
            self._decode_tokens(self.adapter.encode(source)) if source is not None else None
        )

        if only_diff:
            result = self.analyze_diff(
                source, output, prompt=prompt, top_k=top_k, context=context, chunk_size=chunk_size
            )
        else:
            result = self.analyze_generation(prompt, output, top_k=top_k, chunk_size=chunk_size)
            if source is not None:
                output_ids = self._output_ids(output)
                output_tokens = self._decode_tokens(output_ids)
                spans = diff_token_sequences(source_tokens, output_tokens)
                span_by_position = dict(
                    select_positions(spans, result.prompt_length, len(output_ids))
                )
                positions = [
                    replace(pos, diff_span=span_by_position.get(pos.token_index, pos.diff_span))
                    for pos in result.positions
                ]
                result = replace(result, positions=positions, diff_spans=spans)

        return _score_anomalies(result, source_tokens, config, entity_spans)

    # ------------------------------------------------------------------ compare

    def compare(
        self,
        text_a: str,
        text_b: str,
        token_index: int = -1,
        top_k: int = 10,
        target_token: str | int | None = None,
        token_index_b: int | None = None,
        keep_log_probs: bool = True,
    ) -> ComparisonResult:
        """Run the lens on two texts and compare layer-wise: top-1, KL/JS, overlap, target rank.

        ``token_index_b`` (default: same as ``token_index``) lets the two texts be
        probed at different positions -- useful when they have different lengths but
        both should be read from the end (``token_index=token_index_b=-1``, the
        default). ``target_token``, if given, is resolved once and tracked in both
        texts; otherwise each text tracks its own following token, as in
        :meth:`analyze`.

        With ``keep_log_probs`` (default ``True``), the full-vocabulary
        log-probabilities for the compared position are kept just long enough to
        compute ``cmp.layers[i].kl_a_b`` / ``.kl_b_a`` / ``.js`` -- one position costs
        ``(num_layers + 1) * vocab_size`` float32 values per text (~15 MB for
        Qwen2.5-0.5B); pass ``False`` to skip this and get ``None`` for those fields.
        """
        if token_index_b is None:
            token_index_b = token_index

        result_a, log_probs_a = self._analyze_detailed(
            text_a,
            token_index=token_index,
            top_k=top_k,
            target_token=target_token,
            keep_log_probs=keep_log_probs,
        )
        result_b, log_probs_b = self._analyze_detailed(
            text_b,
            token_index=token_index_b,
            top_k=top_k,
            target_token=target_token,
            keep_log_probs=keep_log_probs,
        )
        if log_probs_a is not None:
            log_probs_a = log_probs_a[:, 0, :]
        if log_probs_b is not None:
            log_probs_b = log_probs_b[:, 0, :]

        return compare_results(
            result_a,
            result_b,
            log_probs_a=log_probs_a,
            log_probs_b=log_probs_b,
            kind="prompt",
            metadata={"device": str(self.adapter.device), "keep_log_probs": keep_log_probs},
        )


# ---------------------------------------------------------------------- model comparison


@dataclass
class ModelSide:
    """One analyzed side of a model-vs-model comparison, produced by :func:`_model_side`.

    ``log_probs`` is ``[num_layers + 1, V]`` (already sliced to the single compared
    position) when ``keep_log_probs`` was requested, else ``None``. ``vocab_fingerprint``
    lets :func:`compare_model_names` decide ``same_vocab`` from the two sides alone, once
    the adapter that produced it may already have been released (``sequential=True``).
    """

    result: LogitLensResult
    log_probs: torch.Tensor | None
    vocab_fingerprint: str | None
    model_name: str


def _model_side(
    analyzer: LogitLensAnalyzer,
    text: str,
    token_index: int,
    char_index: int | None,
    top_k: int,
    target_token: str | None,
    keep_log_probs: bool,
) -> ModelSide:
    """Analyze one side of a model comparison.

    ``char_index``, when given, overrides ``token_index`` via
    ``analyzer.adapter.char_to_token`` so the two models can be probed at "the same
    place in the text" even when their tokenizers segment it differently.
    ``target_token`` must be a ``str`` or ``None`` -- unlike :meth:`LogitLensAnalyzer.analyze`,
    an int token id would be specific to one model's vocabulary and cannot be resolved
    in the other's, so it is rejected outright rather than silently misinterpreted.
    """
    if target_token is not None and not isinstance(target_token, str):
        raise TypeError(
            f"target_token must be a str or None for model comparison, got "
            f"{type(target_token).__name__} (int token ids are vocabulary-specific and "
            "cannot be shared across models)"
        )
    if char_index is not None:
        token_index = analyzer.adapter.char_to_token(text, char_index)

    result, log_probs = analyzer._analyze_detailed(
        text,
        token_index=token_index,
        top_k=top_k,
        target_token=target_token,
        keep_log_probs=keep_log_probs,
    )
    if log_probs is not None:
        log_probs = log_probs[:, 0, :]
    return ModelSide(
        result=result,
        log_probs=log_probs,
        vocab_fingerprint=analyzer.adapter.vocab_fingerprint(),
        model_name=analyzer.model_name,
    )


def _combine_model_sides(
    side_a: ModelSide,
    side_b: ModelSide,
    same_vocab: bool,
    keep_log_probs: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Build the ``kind="model"`` :class:`ComparisonResult` from two analyzed sides.

    Layers are aligned by relative depth (:func:`~.comparison.align_layers`); comparison
    is by decoded token text unless the two models share a vocabulary, in which case
    token ids (and, if kept, full-vocabulary log-probabilities for KL/JS) are used too.
    """
    alignment = align_layers(side_a.result.num_layers, side_b.result.num_layers)
    compare_by = "token_id" if same_vocab else "text"
    log_probs_a = side_a.log_probs if same_vocab else None
    log_probs_b = side_b.log_probs if same_vocab else None

    metadata = {
        "model_a": side_a.model_name,
        "model_b": side_b.model_name,
        "same_vocab": same_vocab,
        "token_index_a": side_a.result.token_index,
        "token_index_b": side_b.result.token_index,
        "char_index": None,
        "keep_log_probs": keep_log_probs,
        **(extra_metadata or {}),
    }
    return compare_results(
        side_a.result,
        side_b.result,
        log_probs_a=log_probs_a,
        log_probs_b=log_probs_b,
        alignment=alignment,
        compare_by=compare_by,
        kind="model",
        metadata=metadata,
    )


def compare_models(
    analyzer_a: LogitLensAnalyzer,
    analyzer_b: LogitLensAnalyzer,
    text: str,
    token_index: int = -1,
    char_index: int | None = None,
    top_k: int = 10,
    target_token: str | None = None,
    keep_log_probs: bool = True,
) -> ComparisonResult:
    """Compare the same text through two already-loaded models, layer-wise.

    Absorbs the three ways two models can mismatch (see the module docs / README):
    different layer counts (aligned by relative depth), different vocabularies (falls
    back to comparing decoded token text; KL/JS need identical vocabularies, checked via
    :func:`~.comparison.vocab_identical`) and different tokenization of the same text
    (``token_index=-1``, the default, keeps both at "the last token"; use ``char_index``
    to align by character offset instead). Both analyzers stay loaded for the whole
    call -- use :func:`compare_model_names` for the memory-constrained, load-one-at-a-time
    path.
    """
    side_a = _model_side(
        analyzer_a, text, token_index, char_index, top_k, target_token, keep_log_probs
    )
    side_b = _model_side(
        analyzer_b, text, token_index, char_index, top_k, target_token, keep_log_probs
    )
    same_vocab = vocab_identical(analyzer_a.adapter, analyzer_b.adapter)
    return _combine_model_sides(
        side_a, side_b, same_vocab, keep_log_probs, extra_metadata={"char_index": char_index}
    )


def compare_model_names(
    model_a: str,
    model_b: str,
    text: str,
    *,
    token_index: int = -1,
    char_index: int | None = None,
    top_k: int = 10,
    target_token: str | None = None,
    keep_log_probs: bool = True,
    sequential: bool = True,
    factory: Callable[[str], LogitLensAnalyzer] | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    lens_mode: LensMode | str = LensMode.FINAL_NORM,
    **model_kwargs: Any,
) -> ComparisonResult:
    """Load two models by name and compare them, as :func:`compare_models`.

    ``factory`` builds a :class:`LogitLensAnalyzer` from a model name (default:
    ``LogitLensAnalyzer(name, device=device, dtype=dtype, lens_mode=lens_mode,
    **model_kwargs)``); override it to control loading (e.g. a custom adapter,
    quantization kwargs, or -- in tests -- an in-memory tiny model per name).

    ``sequential=True`` (the default) builds and analyzes model A, records its result
    and vocabulary fingerprint, then releases it (``del`` plus ``gc.collect()`` and the
    accelerator's cache) *before* building model B -- on a memory-constrained machine
    (e.g. an M2 16 GB Mac) two mid-size models may not fit at once. ``same_vocab`` is
    then decided from the two fingerprints alone (both non-``None`` and equal), since
    the two adapters are never alive together. ``sequential=False`` keeps both models
    loaded and delegates to :func:`compare_models` (needed if ``vocab_identical``'s
    same-tokenizer-object fast path should apply, e.g. two checkpoints sharing one
    loaded tokenizer via a custom ``factory``).
    """
    if factory is None:

        def factory(name: str) -> LogitLensAnalyzer:
            return LogitLensAnalyzer(
                name, device=device, dtype=dtype, lens_mode=lens_mode, **model_kwargs
            )

    if not sequential:
        analyzer_a = factory(model_a)
        analyzer_b = factory(model_b)
        return compare_models(
            analyzer_a,
            analyzer_b,
            text,
            token_index=token_index,
            char_index=char_index,
            top_k=top_k,
            target_token=target_token,
            keep_log_probs=keep_log_probs,
        )

    analyzer_a = factory(model_a)
    side_a = _model_side(
        analyzer_a, text, token_index, char_index, top_k, target_token, keep_log_probs
    )
    del analyzer_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()

    analyzer_b = factory(model_b)
    side_b = _model_side(
        analyzer_b, text, token_index, char_index, top_k, target_token, keep_log_probs
    )

    same_vocab = (
        side_a.vocab_fingerprint is not None
        and side_b.vocab_fingerprint is not None
        and side_a.vocab_fingerprint == side_b.vocab_fingerprint
    )
    return _combine_model_sides(
        side_a, side_b, same_vocab, keep_log_probs, extra_metadata={"char_index": char_index}
    )
