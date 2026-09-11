"""Logit Lens Debugger: layer-wise next-token predictions for Hugging Face causal LMs."""

from .adapters import (
    HFCausalLMAdapter,
    LlamaAdapter,
    ModelAdapter,
    QwenAdapter,
    UnsupportedModelError,
    adapter_for_model,
    load_adapter,
    register_adapter,
)
from .analyzer import LogitLensAnalyzer
from .diff import diff_token_sequences, select_positions, token_class
from .engine import LogitLensEngine
from .metrics import compute_instability
from .types import (
    ComparisonResult,
    DiffSpan,
    InstabilityMetrics,
    LayerResult,
    LensMode,
    LogitLensResult,
    MultiPositionResult,
    PositionResult,
    TokenPrediction,
)

__version__ = "0.1.0"

__all__ = [
    "ComparisonResult",
    "DiffSpan",
    "HFCausalLMAdapter",
    "InstabilityMetrics",
    "LayerResult",
    "LensMode",
    "LlamaAdapter",
    "LogitLensAnalyzer",
    "LogitLensEngine",
    "LogitLensResult",
    "ModelAdapter",
    "MultiPositionResult",
    "PositionResult",
    "QwenAdapter",
    "TokenPrediction",
    "UnsupportedModelError",
    "__version__",
    "adapter_for_model",
    "compute_instability",
    "diff_token_sequences",
    "load_adapter",
    "register_adapter",
    "select_positions",
    "token_class",
]
