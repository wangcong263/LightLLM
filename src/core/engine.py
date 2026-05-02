#!/usr/bin/env python3
"""LightLLM 核心LLM运行引擎 支持 llama.cpp / vLLM / CTranslate2 多后端"""
import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """支持的LLM后端类型"""
    LLAMA_CPP = "llama.cpp"
    VLLM = "vLLM"
    CTRANSFORMERS = "CTransformers"
    OPENAI = "OpenAI"


@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        defaults = cls()
        for key, value in data.items():
            if hasattr(defaults, key):
                setattr(defaults, key, value)
        return defaults


@dataclass
class StreamResult:
    """流式输出结果"""
    token: str
    token_id: Optional[int] = None
    usage: Optional[Dict[str, int]] = None
    finished: bool = False


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    path: str
    backend: str = "llama.cpp"
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    n_threads: Optional[int] = None
    n_batch: int = 512
    verbose: bool = False


class ModelManager:
    """模型管理器"""

    def __init__(self):
        self.models: Dict[str, Any] = {}

    def register(self, name: str, config: ModelConfig) -> None:
        self.models[name] = config

    def get(self, name: str) -> Optional[ModelConfig]:
        return self.models.get(name)


class ModelBackend:
    """模型后端基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    async def load(self) -> None:
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, AsyncIterator[StreamResult]]:
        raise NotImplementedError

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, AsyncIterator[StreamResult]]:
        raise NotImplementedError

    def unload(self) -> None:
        if self.model:
            del self.model
            self.model = None


class LlamaCppBackend(ModelBackend):
    """llama.cpp 后端实现"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._tokenizer = None

    async def load(self) -> None:
        try:
            from llama_cpp import Llama
            from llama_cpp import LlamaTokenizer
        except ImportError as err:
            raise ImportError(
                "llama-cpp-python 未安装。请运行: pip install llama-cpp-python"
            ) from err

        logger.info(f"Loading model: {self.config.name}")
        logger.info(f"Model path: {self.config.path}")

        try:
            self.model = Llama(
                model_path=self.config.path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                n_batch=self.config.n_batch,
                verbose=self.config.verbose,
            )
            self._tokenizer = LlamaTokenizer(self.model)
            logger.info("Model loaded successfully")
        except Exception as err:
            logger.error(f"Failed to load model: {err}")
            raise RuntimeError(f"Model loading failed: {err}") from err

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, AsyncIterator[StreamResult]]:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        gen_config = config or GenerationConfig()

        if gen_config.stream:
            return self._stream_generate(prompt, gen_config)
        else:
            result = self.model(
                prompt,
                max_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repeat_penalty=gen_config.repeat_penalty,
            )
            return result["choices"][0]["text"]

    async def _stream_generate(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        for token in self.model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stream=True,
        ):
            yield StreamResult(
                token=token["choices"][0]["text"],
                finished=token.get("choices", [{}])[0].get("finish_reason") == "stop",
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, AsyncIterator[StreamResult]]:
        prompt = self._format_messages(messages)
        return await self.generate(prompt, config)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


class VLLMBackend(ModelBackend):
    """vLLM 后端实现"""

    async def load(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as err:
            raise ImportError(
                "vllm 未安装。请运行: pip install vllm"
            ) from err

        logger.info(f"Loading vLLM model: {self.config.name}")

        try:
            self.model = LLM(
                model=self.config.path,
                tensor_parallel_size=1,
                trust_remote_code=True,
            )
            logger.info("vLLM model loaded successfully")
        except Exception as err:
            logger.error(f"Failed to load vLLM model: {err}")
            raise RuntimeError(f"vLLM model loading failed: {err}") from err

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if not self.model:
            raise RuntimeError("vLLM model not loaded. Call load() first.")

        from vllm import SamplingParams

        gen_config = config or GenerationConfig()

        try:
            outputs = self.model.generate(
                [prompt],
                SamplingParams(
                    max_tokens=gen_config.max_tokens,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                ),
            )
            return outputs[0].outputs[0].text
        except Exception as err:
            logger.error(f"vLLM generation failed: {err}")
            raise RuntimeError(f"Generation failed: {err}") from err

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        prompt = self._format_messages(messages)
        return await self.generate(prompt, config)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


class CTransformersBackend(ModelBackend):
    """CTransformers 后端实现"""

    async def load(self) -> None:
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError as err:
            raise ImportError(
                "ctransformers 未安装。请运行: pip install ctransformers"
            ) from err

        logger.info(f"Loading CTransformers model: {self.config.name}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.path,
                model_type="llama",
                config={
                    "context_length": self.config.n_ctx,
                    "gpu_layers": self.config.n_gpu_layers,
                },
            )
            logger.info("CTransformers model loaded successfully")
        except Exception as err:
            logger.error(f"Failed to load CTransformers model: {err}")
            raise RuntimeError(f"Model loading failed: {err}") from err

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        gen_config = config or GenerationConfig()

        try:
            output = self.model(
                prompt,
                max_new_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repetition_penalty=gen_config.repeat_penalty,
            )
            return output
        except Exception as err:
            logger.error(f"Generation failed: {err}")
            raise RuntimeError(f"Generation failed: {err}") from err

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        prompt = self._format_messages(messages)
        return await self.generate(prompt, config)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


class LLMEngine:
    """LLM 引擎主类"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config
        self.backend: Optional[ModelBackend] = None

    def load(self, config: Optional[ModelConfig] = None) -> None:
        """加载模型"""
        if config:
            self.config = config

        if not self.config:
            raise ValueError("Model config is required")

        backend_type = BackendType(self.config.backend)
        logger.info(f"Initializing backend: {backend_type.value}")

        if backend_type == BackendType.LLAMA_CPP:
            self.backend = LlamaCppBackend(self.config)
        elif backend_type == BackendType.VLLM:
            self.backend = VLLMBackend(self.config)
        elif backend_type == BackendType.CTRANSFORMERS:
            self.backend = CTransformersBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        asyncio.run(self.backend.load())

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, AsyncIterator[StreamResult]]:
        """生成文本"""
        if not self.backend:
            raise RuntimeError("Model not loaded. Call load() first.")
        return await self.backend.generate(prompt, config)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, AsyncIterator[StreamResult]]:
        """聊天"""
        if not self.backend:
            raise RuntimeError("Model not loaded. Call load() first.")
        return await self.backend.chat(messages, config)

    def unload(self) -> None:
        """卸载模型"""
        if self.backend:
            self.backend.unload()
            self.backend = None