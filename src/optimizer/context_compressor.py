#!/usr/bin/env python3
"""Context Compressor - Reduces context length while preserving meaning"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CompressionConfig:
    """Configuration for context compression"""
    max_length: int = 2048
    preserve_system: bool = True
    strategy: str = "smart"


class ContextCompressor:
    """Context compression engine"""

    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()

    def compress(self, text: str) -> str:
        """Compress text while preserving meaning"""
        if len(text) <= self.config.max_length:
            return text

        # Simple truncation with ellipsis
        return text[: self.config.max_length - 3] + "..."

    def compress_messages(self, messages: list) -> list:
        """Compress a list of messages"""
        if not self.config.preserve_system:
            return messages

        # Keep system message, compress others
        result = []
        for msg in messages:
            if msg.get("role") == "system":
                result.append(msg)
            else:
                compressed_content = self.compress(msg.get("content", ""))
                result.append({**msg, "content": compressed_content})

        return result


# CLI entry point
if __name__ == "__main__":
    import json
    import sys

    compressor = ContextCompressor()

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        compressed = compressor.compress_messages(data)
        print(json.dumps(compressed, ensure_ascii=False, indent=2))
    else:
        # Test with sample data
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great, thank you for asking!"},
        ]
        compressed = compressor.compress_messages(test_messages)
        print(json.dumps(compressed, ensure_ascii=False, indent=2))