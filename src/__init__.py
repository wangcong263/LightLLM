"""
LightLLM - 轻量化本地LLM运行工具
更轻、更快、更智能的Ollama替代方案
"""

__version__ = "1.0.0"
__author__ = "LightLLM Team"

from .core.engine import LLMEngine, ModelConfig, ModelManager
from .api.server import LightLLMAPI, ChatCompletionRequest
from .agent.bridge import AgentBridge
from .optimizer.skills_optimizer import SkillsOptimizer, Skill, SkillType, TokenBudget
from .optimizer.context_compressor import ContextCompressor

__all__ = [
    "LLMEngine",
    "ModelConfig", 
    "ModelManager",
    "LightLLMAPI",
    "ChatCompletionRequest",
    "AgentBridge",
    "SkillsOptimizer",
    "Skill",
    "SkillType",
    "TokenBudget",
    "ContextCompressor"
]