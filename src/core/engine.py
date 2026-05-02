#!/usr/bin/env python3
"""LightLLM core LLM engine - supports llama.cpp / vLLM / CTranslate2 backends"""
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """LLM backend types"""
    LLAMA_CPP = "llama.cpp"
    VLLM = "vLLM"
    CTRANSFORMERS = "CTransformers"


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class StreamResult:
    """Streaming result"""
    token: str
    token_id: Optional[int] = None
    finished: bool = False


@dataclass
class ModelBackend:
    """Model backend base class"""
    backend_type: BackendType
    model_path: str

    async def load(self):
        """Load model"""
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """Generate text"""
        raise NotImplementedError

    async def stream(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """Stream generated text"""
        raise NotImplementedError

    async def unload(self):
        """Unload model"""
        raise NotImplementedError


class LlamaCppBackend(ModelBackend):
    """llama.cpp backend"""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        super().__init__(BackendType.LLAMA_CPP, model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._model = None
        self._tokenizer = None

    async def load(self):
        try:
            from llama_cpp import Llama, LlamaTokenizer
        except ImportError as err:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            ) from err

        logger.info(f"Loading llama.cpp model from {self.model_path}")
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
        )
        self._tokenizer = LlamaTokenizer(self._model)
        logger.info("Model loaded successfully")

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        output = self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or [],
        )
        return output["choices"][0]["text"]

    async def stream(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        async for token in self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ):
            yield StreamResult(
                token=token.get("content", ""),
                finished=token.get("finish_reason") is not None,
            )

    async def unload(self):
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")


class LLMEngine:
    """LLM engine - main entry point"""

    def __init__(self):
        self.backend: Optional[ModelBackend] = None
        self.config: Optional[GenerationConfig] = None

    def set_backend(self, backend: ModelBackend):
        """Set backend"""
        self.backend = backend

    def configure(self, config: GenerationConfig):
        """Configure generation"""
        self.config = config

    async def load(self, model_path: str, backend_type: BackendType = BackendType.LLAMA_CPP, **kwargs):
        """Load model"""
        if backend_type == BackendType.LLAMA_CPP:
            self.backend = LlamaCppBackend(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
        await self.backend.load()

    async def generate(self, prompt: str) -> str:
        """Generate text"""
        if not self.backend:
            raise RuntimeError("No backend configured. Call load() first.")
        config = self.config or GenerationConfig()
        return await self.backend.generate(prompt, config)

    async def stream(self, prompt: str) -> AsyncIterator[StreamResult]:
        """Stream text"""
        if not self.backend:
            raise RuntimeError("No backend configured. Call load() first.")
        config = self.config or GenerationConfig()
        async for result in self.backend.stream(prompt, config):
            yield result

    async def unload(self):
        """Unload model"""
        if self.backend:
            await self.backend.unload()
            self.backend = None


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    path: str
    backend: BackendType = BackendType.LLAMA_CPP
    n_ctx: int = 2048
    n_threads: Optional[int] = None


class ModelManager:
    """Model manager"""

    def __init__(self):
        self.models: dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.engine = LLMEngine()

    def register(self, config: ModelConfig):
        """Register model"""
        self.models[config.name] = config

    def unregister(self, name: str) -> bool:
        """Unregister model"""
        if name in self.models:
            del self.models[name]
            return True
        return False

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model config"""
        return self.models.get(name)

    def list_models(self) -> list[str]:
        """List all models"""
        return list(self.models.keys())
