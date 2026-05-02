"""
lightllm - 轻量化本地LLM运行工具
更轻、更快、更智能的 Ollama 替代方案
"""

__version__ = "1.0.0"
__author__ = "lightllm Team"

from .model_converter import ModelConverter
from .model_manager import (
    DEFAULT_MODEL_DIR,
    ModelCatalog,
    ModelDownloader,
    get_model_info,
    get_system_info,
    list_popular_models,
)

__all__ = [
    "ModelDownloader",
    "ModelCatalog",
    "list_popular_models",
    "get_model_info",
    "get_system_info",
    "DEFAULT_MODEL_DIR",
    "ModelConverter",
    "ModelConverter",
]


