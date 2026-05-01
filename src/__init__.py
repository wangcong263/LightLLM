"""
LightLLM - 轻量化本地LLM运行工具
更轻、更快、更智能的 Ollama 替代方案
"""

__version__ = "1.0.0"
__author__ = "LightLLM Team"

# 只导出核心类，避免导入错误
try:
    from .core.engine import LLMEngine
    __all__ = ["LLMEngine"]
except ImportError as e:
    __all__ = []
    print(f"Warning: Could not import core engine: {e}")