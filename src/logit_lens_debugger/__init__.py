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
from .analyzer import (
    LogitLensAnalyzer,
    ModelSide,
    compare_model_names,
    compare_models,
)
from .anomaly import detect_anomalies, score_position
from .comparison import align_layers, compare_results, vocab_identical
from .diff import diff_token_sequences, select_positions, token_class
from .engine import LogitLensEngine
from .metrics import compute_instability
from .types import (
    DEFAULT_ANOMALY_WEIGHTS,
    AnomalyConfig,
    AnomalyReport,
    ComparisonResult,
    DiffSpan,
    InstabilityMetrics,
    LayerComparison,
    LayerResult,
    LensMode,
    LogitLensResult,
    MultiPositionResult,
    PositionAnomaly,
    PositionResult,
    TokenPrediction,
)

__version__ = "0.1.0"

__all__ = [
    "AnomalyConfig",
    "AnomalyReport",
    "ComparisonResult",
    "DEFAULT_ANOMALY_WEIGHTS",
    "DiffSpan",
    "HFCausalLMAdapter",
    "InstabilityMetrics",
    "LayerComparison",
    "LayerResult",
    "LensMode",
    "LlamaAdapter",
    "LogitLensAnalyzer",
    "LogitLensEngine",
    "LogitLensResult",
    "ModelAdapter",
    "ModelSide",
    "MultiPositionResult",
    "PositionAnomaly",
    "PositionResult",
    "QwenAdapter",
    "TokenPrediction",
    "UnsupportedModelError",
    "__version__",
    "adapter_for_model",
    "align_layers",
    "compare_model_names",
    "compare_models",
    "compare_results",
    "compute_instability",
    "detect_anomalies",
    "diff_token_sequences",
    "load_adapter",
    "register_adapter",
    "score_position",
    "select_positions",
    "token_class",
    "vocab_identical",
]
