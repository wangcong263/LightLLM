#!/usr/bin/env python3
"""上下文压缩器 - 智能压缩对话上下文"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CompressionConfig:
    """压缩配置"""
    max_tokens: int = 4000
    preserve_system: bool = True
    preserve_last_n: int = 2


class ContextCompressor:
    """上下文压缩器"""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()

    def compress(self, messages: List[Dict]) -> List[Dict]:
        """压缩消息列表"""
        if not messages:
            return []

        compressed = []

        # 保留系统消息
        if self.config.preserve_system:
            for msg in messages:
                if msg.get("role") == "system":
                    compressed.append(msg)

        # 保留最近的消息
        non_system = [m for m in messages if m.get("role") != "system"]
        recent = non_system[-self.config.preserve_last_n:] if self.config.preserve_last_n > 0 else []

        # 简化中间消息
        if len(non_system) > self.config.preserve_last_n:
            middle_count = len(non_system) - self.config.preserve_last_n
            # 添加摘要标记
            compressed.append({
                "role": "system",
                "content": f"[已省略 {middle_count} 条消息]"
            })

        compressed.extend(recent)
        return compressed

    def estimate_tokens(self, text: str) -> int:
        """简单估算token数量"""
        return len(text) // 4

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "max_tokens": self.config.max_tokens,
            "preserve_system": self.config.preserve_system,
            "preserve_last_n": self.config.preserve_last_n,
        }