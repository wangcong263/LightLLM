"""
上下文压缩器 - 减少Token使用
"""
import re
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .skills_optimizer import SkillsOptimizer

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    上下文压缩器

    策略：
    1. 删除冗余信息
    2. 提取关键实体
    3. 摘要长文本
    4. 合并相似内容
    """

    def __init__(self, target_tokens: int = 8192):
        self.target_tokens = target_tokens
        self.importance_keywords = {
            "function", "class", "def", "return", "import",
            "error", "fix", "bug", "feature", "implement",
            "config", "setup", "init", "run", "execute",
        }

    def compress(self, messages: List[Dict]) -> List[Dict]:
        """压缩消息列表"""
        current_tokens = self._estimate_tokens(messages)

        if current_tokens <= self.target_tokens:
            return messages

        # 逐步压缩
        compressed = messages.copy()

        # 1. 移除重复内容
        compressed = self._remove_duplicates(compressed)

        # 2. 压缩每个消息
        compressed = [self._compress_message(m) for m in compressed]

        # 3. 如果还不够，摘要旧消息
        while self._estimate_tokens(compressed) > self.target_tokens and len(compressed) > 2:
            compressed = self._summarize_oldest(compressed)

        return compressed

    def _remove_duplicates(self, messages: List[Dict]) -> List[Dict]:
        """移除重复消息"""
        seen = set()
        unique = []

        for msg in messages:
            key = f"{msg.get('role', '')}:{msg.get('content', '')[:100]}"
            if key not in seen:
                seen.add(key)
                unique.append(msg)

        return unique

    def _compress_message(self, message: Dict) -> Dict:
        """压缩单条消息"""
        content = message.get("content", "")

        if not content:
            return message

        # 短内容不压缩
        if len(content) < 500:
            return message

        # 压缩长文本
        compressed = self._compress_text(content)

        return {**message, "content": compressed}

    def _compress_text(self, text: str) -> str:
        """压缩文本"""
        # 移除注释
        lines = text.split('\n')
        code_lines = []

        for line in lines:
            stripped = line.strip()
            # 保留代码，移除注释
            if stripped.startswith('#') or stripped.startswith('//'):
                continue
            code_lines.append(line)

        text = '\n'.join(code_lines)

        # 移除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 如果还是太长，截断
        if len(text) > 2000:
            text = text[:1000] + '\n...\n' + text[-500:]

        return text

    def _summarize_oldest(self, messages: List[Dict]) -> List[Dict]:
        """摘要最老的消息"""
        if len(messages) <= 2:
            return messages

        # 保留系统消息和最新消息
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        recent = messages[-2:]

        # 摘要中间消息
        middle = messages[1:-2] if len(messages) > 2 else []

        if middle:
            summary = self._create_summary(middle)
            summarized = [summary]
        else:
            summarized = []

        # 重新构建
        result = []
        if system_msg:
            result.append(system_msg)
        if summarized:
            result.extend(summarized)
        result.extend(recent)

        return result

    def _create_summary(self, messages: List[Dict]) -> Dict:
        """创建摘要"""
        # 提取关键信息
        entities = set()
        actions = []

        for msg in messages:
            content = msg.get("content", "")

            # 提取关键词
            words = re.findall(r'\b\w+\b', content.lower())
            entities.update(w for w in words if len(w) > 5)

            # 提取操作
            if any(kw in content.lower() for kw in ["create", "delete", "update", "modify"]):
                actions.append(content[:100])

        summary = f"[Previous context summarized: {len(messages)} messages, {len(entities)} key concepts]"

        return {
            "role": "system",
            "content": summary,
        }

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """估计Token数量"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(content) // 4  # 粗略估计

        # 加上消息结构开销
        total += len(messages) * 4

        return total


class StreamingOptimizer:
    """
    流式输出优化器
    减少首token延迟
    """

    def __init__(self):
        self.prefetch_enabled = True
        self.predictive_decode = True

    async def stream_with_optimization(self, generator):
        """优化流式输出"""
        buffer = ""
        min_chunk_size = 4

        async for chunk in generator:
            buffer += chunk

            # 小块合并
            if len(buffer) >= min_chunk_size:
                yield buffer
                buffer = ""

        # 输出剩余
        if buffer:
            yield buffer
