"""
moe — Mixture of Experts para Clasificación Médica Multimodal
"""

from .preprocess  import AdaptivePreprocessor
from .backbone    import SharedBackbone, build_backbone
from .router_knn  import KNNRouter, build_router
from .experts     import load_all_experts, EXPERT_META
from .moe_model   import MoEModel
from .utils       import (
    format_prediction, print_prediction,
    save_result_json, load_results_json,
    get_device, Timer, setup_logger,
)

__version__ = "1.0.0"
__all__ = [
    "AdaptivePreprocessor",
    "SharedBackbone", "build_backbone",
    "KNNRouter", "build_router",
    "load_all_experts", "EXPERT_META",
    "MoEModel",
    "format_prediction", "print_prediction",
    "save_result_json", "load_results_json",
    "get_device", "Timer", "setup_logger",
]
