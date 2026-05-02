"""
测试套件
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.engine import LLMEngine, ModelConfig, ModelManager
from src.optimizer.skills_optimizer import SkillsOptimizer, Skill, SkillType, TokenBudget

# 可选导入 - API 服务需要额外的依赖
try:
    from src.api.server import LightLLMAPI, ChatCompletionRequest
    HAS_API = True
except ImportError:
    HAS_API = False


class TestLLMEngine:
    """测试LLM引擎"""

    def test_engine_init(self):
        engine = LLMEngine()
        assert engine.backend is None
        assert engine.config is None

    def test_detect_backend(self):
        engine = LLMEngine()
        # 检查 engine 有正确的方法
        assert hasattr(engine, 'set_backend')
        assert hasattr(engine, 'configure')
        assert hasattr(engine, 'load')
        assert hasattr(engine, 'generate')


class TestModelManager:
    """测试模型管理器"""

    @pytest.fixture
    def manager(self):
        return ModelManager()

    @pytest.fixture
    def config(self):
        return ModelConfig(name="test", path="./test.gguf")

    def test_register_model(self, manager, config):
        manager.register(config)
        assert "test" in manager.models
        assert manager.get("test") == config

    def test_unregister_model(self, manager, config):
        manager.register(config)
        result = manager.unregister("test")
        assert result is True
        assert "test" not in manager.models

    def test_list_models(self, manager, config):
        manager.register(config)
        models = manager.list_models()
        assert "test" in models


class TestSkillsOptimizer:
    """测试Skills优化器"""

    @pytest.fixture
    def optimizer(self):
        return SkillsOptimizer()

    def test_register_skill(self, optimizer):
        skill = Skill(
            name="test-skill",
            type=SkillType.TOOL,
            description="Test skill"
        )
        optimizer.register(skill)
        assert "test-skill" in optimizer.skills

    def test_unregister_skill(self, optimizer):
        skill = Skill(name="test", type=SkillType.ACTION, description="")
        optimizer.register(skill)
        result = optimizer.unregister("test")
        assert result is True
        assert "test" not in optimizer.skills

    def test_get_skill(self, optimizer):
        skill = Skill(name="test", type=SkillType.TOOL, description="")
        optimizer.register(skill)
        retrieved = optimizer.get_skill("test")
        assert retrieved == skill

    def test_get_statistics(self, optimizer):
        stats = optimizer.get_statistics()
        assert "total_calls" in stats
        assert "success_rate" in stats


class TestTokenBudget:
    """测试Token预算"""

    def test_allocate(self):
        budget = TokenBudget(max_tokens=1000)
        budget.allocate(500, "test")
        assert budget.total_allocated == 500

    def test_get_available(self):
        budget = TokenBudget(max_tokens=1000)
        budget.allocate(500, "test")
        assert budget.get_available() == 500

    def test_release(self):
        budget = TokenBudget(max_tokens=1000)
        budget.allocate(500, "test")
        budget.release("test")
        assert budget.total_allocated == 0


@pytest.mark.skipif(not HAS_API, reason="需要 fastapi/uvicorn 依赖")
class TestLightLLMAPI:
    """测试API服务"""

    @pytest.fixture
    def api(self):
        return LightLLMAPI(host="localhost", port=8080)

    def test_api_init(self, api):
        assert api.host == "localhost"
        assert api.port == 8080

    def test_health(self, api):
        health = asyncio.run(api.health())
        assert health["status"] == "ok"