"""
LightLLM - 轻量化本地 LLM 推理引擎

安装依赖:
    pip install llama-cpp-python
    
快速开始:
    from lightllm import LLMEngine
    engine = LLMEngine("model.gguf")
    await engine.load()
    
    async for result in engine.generate("Hello!"):
        print(result.content, end="")
"""
from .src.core.engine import LLMEngine, ModelManager, GenerationConfig, BackendType
from .config import PRESET_MODELS, MODEL_CACHE_DIR

__version__ = "1.0.0"
__all__ = [
    "LLMEngine",
    "ModelManager", 
    "GenerationConfig",
    "BackendType",
    "PRESET_MODELS",
    "MODEL_CACHE_DIR",
]