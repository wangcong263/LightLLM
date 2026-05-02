#!/usr/bin/env python3
"""Skills Optimizer - Optimizes agent skills and function calling"""
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """技能类型"""
    TOOL = "tool"
    ACTION = "action"
    CONTEXT = "context"
    TEMPLATE = "template"


@dataclass
class Skill:
    """技能定义"""
    name: str
    type: SkillType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable[..., Any]] = None
    enabled: bool = True


@dataclass
class SkillCall:
    """技能调用"""
    skill_name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    success: bool = False


class SkillsOptimizer:
    """技能优化器"""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.call_history: List[SkillCall] = []

    def register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def optimize_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        """优化提示词"""
        # Extract potential skill calls from prompt
        patterns = [
            r"@(\w+)",  # @skill_name
            r"use\s+(\w+)",  # use skill_name
            r"call\s+(\w+)",  # call skill_name
        ]

        found_skills = set()
        for pattern in patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            found_skills.update(matches)

        # Add relevant skills to context
        if context is None:
            context = {}

        if "available_skills" not in context:
            context["available_skills"] = []

        for skill_name in found_skills:
            if skill_name in self.skills and self.skills[skill_name].enabled:
                skill = self.skills[skill_name]
                if skill.name not in context["available_skills"]:
                    context["available_skills"].append(skill.name)

        return prompt

    def call_skill(
        self, skill_name: str, parameters: Optional[Dict] = None, context: Optional[Dict] = None
    ) -> Any:
        """调用技能"""
        skill = self.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        if not skill.enabled:
            raise RuntimeError(f"Skill is disabled: {skill_name}")

        params = parameters or {}
        call = SkillCall(skill_name=skill_name, parameters=params)

        try:
            if skill.handler:
                result = skill.handler(**params)
                call.result = result
                call.success = True
            else:
                result = f"Skill {skill_name} has no handler"
                call.result = result
        except Exception as err:
            logger.error(f"Skill call failed: {err}")
            call.result = str(err)
            call.success = False

        self.call_history.append(call)
        return call

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_calls = len(self.call_history)
        successful_calls = sum(1 for c in self.call_history if c.success)

        skill_usage: Dict[str, int] = {}
        for call in self.call_history:
            skill_usage[call.skill_name] = skill_usage.get(call.skill_name, 0) + 1

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "skill_usage": skill_usage,
        }

    def reset_history(self) -> None:
        """重置调用历史"""
        self.call_history.clear()


class GitHubSkillsOptimizer:
    """GitHub 技能优化器"""

    def __init__(self, optimizer: Optional[SkillsOptimizer] = None):
        self.optimizer = optimizer or SkillsOptimizer()

    def optimize_github_prompt(self, prompt: str, repo_info: Optional[Dict] = None) -> str:
        """优化 GitHub 相关提示词"""
        if repo_info is None:
            repo_info = {}

        # Add repo context
        if "repo" in repo_info:
            prompt = f"Repository: {repo_info['repo']}\n{prompt}"

        if "language" in repo_info:
            prompt = f"Language: {repo_info['language']}\n{prompt}"

        # Use base optimizer
        return self.optimizer.optimize_prompt(prompt)

    def suggest_skills(self, task: str) -> List[str]:
        """建议相关技能"""
        suggestions = []

        task_lower = task.lower()
        if "code" in task_lower or "function" in task_lower:
            suggestions.append("code_generator")
        if "test" in task_lower:
            suggestions.append("test_writer")
        if "debug" in task_lower or "error" in task_lower:
            suggestions.append("debugger")
        if "review" in task_lower:
            suggestions.append("code_reviewer")

        return suggestions


class SkillsRegistry:
    """技能注册表"""

    def __init__(self):
        self._registry: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._registry[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._registry.get(name)

    def list_all(self) -> List[Skill]:
        return list(self._registry.values())

    def list_by_type(self, skill_type: SkillType) -> List[Skill]:
        return [s for s in self._registry.values() if s.type == skill_type]

    def load_from_json(self, path: str) -> None:
        """从 JSON 文件加载技能"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for skill_data in data.get("skills", []):
            skill_type = SkillType(skill_data.get("type", "tool"))
            skill = Skill(
                name=skill_data["name"],
                type=skill_type,
                description=skill_data.get("description", ""),
                parameters=skill_data.get("parameters", {}),
                enabled=skill_data.get("enabled", True),
            )
            self.register(skill)

    def save_to_json(self, path: str) -> None:
        """保存技能到 JSON 文件"""
        data = {
            "skills": [
                {
                    "name": s.name,
                    "type": s.type.value,
                    "description": s.description,
                    "parameters": s.parameters,
                    "enabled": s.enabled,
                }
                for s in self._registry.values()
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class TokenBudget:
    """Token 预算管理器"""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.used_tokens: int = 0

    def allocate(self, tokens: int, context: Optional[Dict] = None) -> bool:
        """分配 token"""
        if self.used_tokens + tokens > self.max_tokens:
            return False
        self.used_tokens += tokens
        return True

    def release(self, tokens: int) -> None:
        """释放 token"""
        self.used_tokens = max(0, self.used_tokens - tokens)

    def reset(self) -> None:
        """重置预算"""
        self.used_tokens = 0

    def get_available(self) -> int:
        """获取可用 token 数"""
        return self.max_tokens - self.used_tokens