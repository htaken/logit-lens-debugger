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
from .engine import LogitLensEngine
from .types import ComparisonResult, LayerResult, LensMode, LogitLensResult, TokenPrediction

__version__ = "0.1.0"

__all__ = [
    "ComparisonResult",
    "HFCausalLMAdapter",
    "LayerResult",
    "LensMode",
    "LlamaAdapter",
    "LogitLensAnalyzer",
    "LogitLensEngine",
    "LogitLensResult",
    "ModelAdapter",
    "QwenAdapter",
    "TokenPrediction",
    "UnsupportedModelError",
    "__version__",
    "adapter_for_model",
    "load_adapter",
    "register_adapter",
]
