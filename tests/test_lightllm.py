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
from src.api.server import LightLLMAPI, ChatCompletionRequest
from src.optimizer.skills_optimizer import SkillsOptimizer, Skill, SkillType, TokenBudget
from src.optimizer.context_compressor import ContextCompressor


class TestLLMEngine:
    """测试LLM引擎"""

    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            name="test-model",
            path="./test.gguf",
            context_length=2048,
            threads=2,
        )

    def test_engine_init(self, model_config):
        engine = LLMEngine(model_config)
        assert engine.config.name == "test-model"
        assert not engine._is_loaded

    def test_detect_backend(self, model_config):
        engine = LLMEngine(model_config)
        # 应该能检测到至少一个后端
        backend = engine._detect_backend()
        assert backend in ["llama_cpp", "vllm", "ctranslate2"]

    @pytest.mark.asyncio
    async def test_load_model(self, model_config):
        engine = LLMEngine(model_config)

        # 模拟加载失败（模型文件不存在）
        result = await engine.load_model()
        # 文件不存在时应该返回False
        assert result is False


class TestModelManager:
    """测试模型管理器"""

    @pytest.fixture
    def manager(self):
        return ModelManager()

    @pytest.fixture
    def config(self):
        return ModelConfig(name="test", path="./test.gguf")

    def test_add_model(self, manager, config):
        engine = manager.add_model("test", config)
        assert "test" in manager.models
        assert engine.config.name == "test"

    def test_get_current_empty(self, manager):
        assert manager.get_current() is None


class TestSkillsOptimizer:
    """测试Skills优化器"""

    @pytest.fixture
    def optimizer(self):
        return SkillsOptimizer()

    def test_register_skill(self, optimizer):
        skill = Skill(
            name="test-skill",
            type=SkillType.FILE_OPERATION,
            description="Test skill"
        )
        optimizer.register_skill(skill)
        assert "test-skill" in optimizer.skills

    def test_unregister_skill(self, optimizer):
        skill = Skill(name="test", type=SkillType.SEARCH, description="")
        optimizer.register_skill(skill)
        optimizer.unregister_skill("test")
        assert "test" not in optimizer.skills

    @pytest.mark.asyncio
    async def test_call_skill_not_found(self, optimizer):
        with pytest.raises(ValueError, match="Skill not found"):
            await optimizer.call_skill("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_skill_with_cache(self, optimizer):
        skill = Skill(name="cached", type=SkillType.SEARCH, description="", cacheable=True)
        skill.handler = AsyncMock(return_value={"result": "success"})
        optimizer.register_skill(skill)

        # 第一次调用
        result1 = await optimizer.call_skill("cached", {"param": "value"})
        assert result1["result"] == "success"

        # 第二次调用应该命中缓存
        result2 = await optimizer.call_skill("cached", {"param": "value"})
        assert optimizer.cache_hits == 1

    def test_stats(self, optimizer):
        stats = optimizer.get_stats()
        assert "total_calls" in stats
        assert "cache_hit_rate" in stats


class TestTokenBudget:
    """测试Token预算"""

    def test_allocate(self):
        budget = TokenBudget(max_tokens=1000)

        assert budget.allocate(500, "test") is True
        assert budget.used_tokens == 500

        # 超出预算
        assert budget.allocate(600, "test2") is False

    def test_release(self):
        budget = TokenBudget(max_tokens=1000)
        budget.allocate(500, "test")
        budget.release(200)
        assert budget.used_tokens == 300

    def test_reset(self):
        budget = TokenBudget(max_tokens=1000)
        budget.allocate(500, "test")
        budget.reset()
        assert budget.used_tokens == 0


class TestContextCompressor:
    """测试上下文压缩"""

    @pytest.fixture
    def compressor(self):
        return ContextCompressor(target_tokens=8192)

    def test_compress_short_messages(self, compressor):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        compressed = compressor.compress(messages)
        assert len(compressed) == 2

    def test_compress_long_messages(self, compressor):
        messages = [
            {"role": "user", "content": "A" * 5000},
            {"role": "assistant", "content": "B" * 5000},
            {"role": "user", "content": "C"},
        ]
        compressed = compressor.compress(messages)
        # 应该被压缩
        assert compressor._estimate_tokens(compressed) <= compressor.target_tokens

    def test_remove_duplicates(self, compressor):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Hello"},  # 重复
            {"role": "assistant", "content": "Hi"},
        ]
        unique = compressor._remove_duplicates(messages)
        assert len(unique) == 2


class TestLightLLMAPI:
    """测试API服务"""

    @pytest.fixture
    def api(self):
        return LightLLMAPI(host="localhost", port=8080)

    def test_api_init(self, api):
        assert api.host == "localhost"
        assert api.port == 8080
        assert api.server is None

    def test_health(self, api):
        health = asyncio.run(api.health())
        assert health["status"] == "ok"

    def test_cache_stats(self, api):
        stats = asyncio.run(api.cache_stats())
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# 集成测试
class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """测试完整流程"""
        # 1. 创建优化器
        optimizer = SkillsOptimizer()
        skill = Skill(name="process", type=SkillType.EXECUTION, description="")
        optimizer.register_skill(skill)

        # 2. 创建压缩器
        compressor = ContextCompressor()

        # 3. 验证集成
        messages = [{"role": "user", "content": "Test"}]
        compressed = compressor.compress(messages)

        stats = optimizer.get_stats()
        assert stats["registered_skills"] == 1
        assert len(compressed) == 1


