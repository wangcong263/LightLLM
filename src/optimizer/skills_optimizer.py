"""
Skills调用优化器
减少Token使用，提高调用效率
"""
import re
import hashlib
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """技能类型"""
    FILE_OPERATION = "file_operation"
    CODE_GENERATION = "code_generation"
    SEARCH = "search"
    EXECUTION = "execution"
    MEMORY = "memory"
    WEB = "web"
    CUSTOM = "custom"


@dataclass
class Skill:
    """技能定义"""
    name: str
    type: SkillType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None
    cacheable: bool = True
    priority: int = 0


@dataclass
class SkillCall:
    """技能调用"""
    skill_name: str
    parameters: Dict[str, Any]
    context: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)
    token_cost: int = 0


class SkillsOptimizer:
    """
    Skills调用优化器

    优化策略：
    1. 智能缓存 - 避免重复调用
    2. 参数压缩 - 减少Token使用
    3. 批量处理 - 减少调用次数
    4. 优先级调度 - 优化执行顺序
    5. 结果复用 - 共享中间结果
    """

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.call_history: List[SkillCall] = []
        self.cache: Dict[str, Any] = {}
        self.batch_queue: List[SkillCall] = []
        self.batch_size = 5
        self.batch_timeout = 0.5  # 秒

        # 统计
        self.total_calls = 0
        self.cache_hits = 0
        self.token_saved = 0

    def register_skill(self, skill: Skill) -> None:
        """注册技能"""
        self.skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def unregister_skill(self, name: str) -> None:
        """取消注册技能"""
        if name in self.skills:
            del self.skills[name]

    async def call_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict] = None,
        force_cache: bool = False,
    ) -> Any:
        """
        调用技能 - 带优化

        优化点：
        1. 缓存检查
        2. 参数归一化
        3. 结果缓存
        """
        self.total_calls += 1

        # 检查技能是否存在
        if skill_name not in self.skills:
            raise ValueError(f"Skill not found: {skill_name}")

        skill = self.skills[skill_name]

        # 生成缓存键
        cache_key = self._generate_cache_key(skill_name, parameters)

        # 缓存检查
        if skill.cacheable and not force_cache:
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.debug(f"Cache hit for {skill_name}")
                return self.cache[cache_key]

        # 参数优化
        optimized_params = self._optimize_parameters(skill, parameters)

        # 调用技能
        start_time = time.time()
        try:
            if skill.handler:
                result = await skill.handler(optimized_params, context)
            else:
                result = await self._default_handler(skill, optimized_params, context)
        except Exception as e:
            logger.error(f"Skill call failed: {skill_name}, {e}")
            raise

        # 计算Token节省
        original_size = self._estimate_size(parameters)
        optimized_size = self._estimate_size(optimized_params)
        self.token_saved += original_size - optimized_size

        # 缓存结果
        if skill.cacheable:
            self.cache[cache_key] = result

        # 记录调用
        self.call_history.append(SkillCall(
            skill_name=skill_name,
            parameters=parameters,
            context=context,
            token_cost=optimized_size,
        ))

        return result

    async def call_skills_batch(
        self,
        calls: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[Any]:
        """
        批量调用技能

        优化策略：
        1. 依赖分析 - 识别可并行的调用
        2. 资源优化 - 复用资源
        3. 结果聚合 - 统一返回
        """
        # 分析依赖
        independent = []
        dependent = []

        for call in calls:
            if self._is_independent(call):
                independent.append(call)
            else:
                dependent.append(call)

        results = []

        # 并行执行独立调用
        if parallel and independent:
            tasks = [
                self.call_skill(c["skill"], c["params"], c.get("context"))
                for c in independent
            ]
            results.extend(await asyncio.gather(*tasks, return_exceptions=True))

        # 顺序执行依赖调用
        for call in dependent:
            result = await self.call_skill(
                call["skill"],
                call["params"],
                call.get("context")
            )
            results.append(result)

        return results

    def _optimize_parameters(self, skill: Skill, params: Dict) -> Dict:
        """
        优化参数字符串

        策略：
        1. 移除冗余参数
        2. 简化长字符串
        3. 使用引用替代重复内容
        """
        optimized = {}

        for key, value in params.items():
            if value is None:
                continue

            if isinstance(value, str) and len(value) > 1000:
                # 长字符串使用哈希引用
                optimized[f"_ref_{key}"] = self._hash_string(value)
                optimized[key] = self._compress_string(value)
            elif isinstance(value, dict):
                # 递归优化
                optimized[key] = self._optimize_parameters(skill, value)
            elif isinstance(value, list):
                # 列表优化
                optimized[key] = [
                    self._compress_string(v) if isinstance(v, str) else v
                    for v in value[:10]  # 限制长度
                ]
            else:
                optimized[key] = value

        return optimized

    def _compress_string(self, text: str) -> str:
        """压缩字符串"""
        if len(text) < 100:
            return text

        # 简单压缩：移除多余空白
        compressed = re.sub(r'\s+', ' ', text)
        compressed = compressed.strip()

        # 如果仍然很长，截断并添加标记
        if len(compressed) > 500:
            return compressed[:250] + "...[truncated]"

        return compressed

    def _generate_cache_key(self, skill_name: str, params: Dict) -> str:
        """生成缓存键"""
        # 归一化参数
        normalized = self._normalize_params(params)
        key_str = f"{skill_name}:{normalized}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _normalize_params(self, params: Dict) -> str:
        """归一化参数"""
        items = []
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, str):
                # 截断长字符串
                v = v[:100] if len(v) > 100 else v
            items.append(f"{k}={v}")
        return "|".join(items)

    def _hash_string(self, text: str) -> str:
        """哈希长字符串"""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _estimate_size(self, obj: Any) -> int:
        """估计Token大小"""
        import sys
        return len(str(obj)) // 4  # 粗略估计

    def _is_independent(self, call: Dict) -> bool:
        """判断是否独立调用"""
        return True  # 简化实现

    async def _default_handler(
        self,
        skill: Skill,
        params: Dict,
        context: Optional[Dict]
    ) -> Any:
        """默认处理器"""
        return {"status": "ok", "skill": skill.name}

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_calls),
            "token_saved": self.token_saved,
            "cache_size": len(self.cache),
            "registered_skills": len(self.skills),
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("Cache cleared")


class SkillsRegistry:
    """
    Skills注册表
    管理GitHub等来源的Skills
    """

    def __init__(self, optimizer: SkillsOptimizer):
        self.optimizer = optimizer
        self.github_skills: Dict[str, Dict] = {}

    def register_github_skill(
        self,
        name: str,
        repo: str,
        skill_type: SkillType,
        description: str,
    ) -> Skill:
        """注册GitHub上的Skill"""
        skill = Skill(
            name=name,
            type=skill_type,
            description=description,
            cacheable=True,
        )

        self.github_skills[name] = {
            "repo": repo,
            "type": skill_type,
        }

        self.optimizer.register_skill(skill)
        return skill

    def load_skill_from_github(self, repo: str, path: str) -> Optional[Dict]:
        """从GitHub加载Skill"""
        # 实现GitHub API调用
        return None  # 简化实现


class TokenBudget:
    """
    Token预算管理
    确保Token使用在限制内
    """

    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.budget_history: List[Dict] = []

    def allocate(self, tokens: int, purpose: str) -> bool:
        """分配Token"""
        if self.used_tokens + tokens > self.max_tokens:
            logger.warning(f"Token budget exceeded: {purpose}")
            return False

        self.used_tokens += tokens
        self.budget_history.append({
            "purpose": purpose,
            "tokens": tokens,
            "timestamp": time.time(),
        })
        return True

    def release(self, tokens: int):
        """释放Token"""
        self.used_tokens = max(0, self.used_tokens - tokens)

    def reset(self):
        """重置预算"""
        self.used_tokens = 0
        self.budget_history.clear()

    def get_remaining(self) -> int:
        """获取剩余Token"""
        return self.max_tokens - self.used_tokens

    def get_usage(self) -> Dict:
        """获取使用情况"""
        return {
            "max": self.max_tokens,
            "used": self.used_tokens,
            "remaining": self.get_remaining(),
            "usage_rate": self.used_tokens / self.max_tokens,
        }



class GitHubSkillsOptimizer:
    """GitHub技能优化器 - 用于从GitHub获取和优化技能"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.cache = {}

    def optimize_skill(self, skill):
        """优化单个技能"""
        return self.optimizer.optimize(skill)

    def optimize_batch(self, skills):
        """批量优化技能"""
        return [self.optimize_skill(s) for s in skills]

    def get_stats(self):
        """获取优化统计"""
        return {
            "optimizer_type": "github",
            "cached_skills": len(self.cache),
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
