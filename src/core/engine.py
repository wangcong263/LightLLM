#!/usr/bin/env python3
"""
LightLLM 核心LLM运行引擎
支持 llama.cpp / vLLM / CTranslate2 多后端
"""
import asyncio
import os
import logging
from typing import Optional, List, Dict, Any, AsyncIterator, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """支持的推理后端"""
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    CTRANSFORMERS = "ctransformers"


@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repeat_penalty: float = 1.1
    stream: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        return cls(
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 50),
            repeat_penalty=data.get("repeat_penalty", 1.1),
            stream=data.get("stream", True),
        )


@dataclass
class StreamResult:
    """流式生成结果"""
    content: str
    done: bool
    token_id: Optional[int] = None
    usage: Optional[Dict[str, int]] = None


class ModelBackend:
    """模型后端基类"""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    async def load(self) -> None:
        """加载模型"""
        raise NotImplementedError

    async def unload(self) -> None:
        """卸载模型"""
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """生成文本"""
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


class LlamaCppBackend(ModelBackend):
    """llama.cpp 后端"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.n_ctx = kwargs.get("n_ctx", 4096)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", 0)
        self.verbose = kwargs.get("verbose", False)
        self._llama = None
        self._tokenizer = None

    async def load(self) -> None:
        """加载 llama.cpp 模型"""
        try:
            from llama_cpp import Llama
            from llama_cpp import LlamaTokenizer
        except ImportError as err:
            raise ImportError("llama-cpp-python 未安装。请运行: pip install llama-cpp-python") from err

        logger.info(f"Loading llama.cpp model from {self.model_path}")

        self._llama = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

        try:
            self._tokenizer = LlamaTokenizer(self._llama)
        except Exception:
            self._tokenizer = None

        self.model = self._llama
        self.tokenizer = self._tokenizer
        logger.info("llama.cpp model loaded successfully")

    async def unload(self) -> None:
        """卸载模型"""
        self.model = None
        self.tokenizer = None
        self._llama = None
        self._tokenizer = None
        import gc
        gc.collect()

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """llama.cpp 流式生成"""

        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        if config.stream:
            async for result in self._stream_llama_cpp(prompt, config):
                yield result
        else:
            result = await self._non_stream_llama_cpp(prompt, config)
            yield result

    async def _stream_llama_cpp(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """llama.cpp 流式生成实现"""

        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise ImportError("llama-cpp-python 未安装。请运行: pip install llama-cpp-python") from err

        # 创建异步流式生成器
        def create_stream():
            return self._llama.create_completion(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=[],
                echo=False,
                stream=True,
            )

        # 在线程池中运行同步 llama.cpp 调用
        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(None, create_stream)

        content = ""
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content += delta["content"]
                    yield StreamResult(
                        content=delta["content"],
                        done=False,
                    )

        # 完成
        yield StreamResult(content="", done=True)

    async def _non_stream_llama_cpp(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> StreamResult:
        """llama.cpp 非流式生成"""

        def create_completion():
            return self._llama(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                echo=False,
                stream=False,
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, create_completion)

        content = ""
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("text", "")

        return StreamResult(content=content, done=True)


class VLLMBackend(ModelBackend):
    """vLLM 后端 (需要 NVIDIA GPU)"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.port = kwargs.get("port", 8000)
        self._async_engine = None

    async def load(self) -> None:
        """加载 vLLM 模型"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as err:
            raise ImportError("vllm 未安装。请运行: pip install vllm") from err

        logger.info(f"Loading vLLM model from {self.model_path}")

        loop = asyncio.get_event_loop()
        self._async_engine = await loop.run_in_executor(
            None,
            lambda: LLM(
                model=str(self.model_path),
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
        )

        self.model = self._async_engine
        logger.info("vLLM model loaded successfully")

    async def unload(self) -> None:
        """卸载模型"""
        self.model = None
        self._async_engine = None
        import gc
        gc.collect()

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """vLLM 流式生成"""

        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        if config.stream:
            async for result in self._stream_vllm(prompt, config):
                yield result
        else:
            result = await self._non_stream_vllm(prompt, config)
            yield result

    async def _stream_vllm(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """vLLM 流式生成"""

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        # vLLM 的异步生成
        # 注意: vLLM 0.2.7+ 支持 async output
        loop = asyncio.get_event_loop()

        # 使用同步接口（vLLM 当前版本的推荐方式）
        def generate_sync():
            outputs = self._async_engine.generate(prompt, sampling_params)
            return outputs

        outputs = await loop.run_in_executor(None, generate_sync)

        if outputs:
            output = outputs[0]
            for i, output_token in enumerate(output.outputs[0].token_ids):
                # 获取 token 对应的文本片段（这里简化为逐字输出）
                if i < len(output.outputs[0].output_str):
                    char = output.outputs[0].output_str[i]
                    yield StreamResult(
                        content=char,
                        done=False,
                        token_id=output_token,
                    )

        yield StreamResult(content="", done=True)

    async def _non_stream_vllm(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> StreamResult:
        """vLLM 非流式生成"""

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        loop = asyncio.get_event_loop()

        def generate_sync():
            outputs = self._async_engine.generate(prompt, sampling_params)
            return outputs

        outputs = await loop.run_in_executor(None, generate_sync)

        content = ""
        if outputs:
            content = outputs[0].outputs[0].output_str

        return StreamResult(content=content, done=True)


class CTransformersBackend(ModelBackend):
    """CTransformers (CTranslate2) 后端"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.model_type = kwargs.get("model_type", "llama")
        self.lib_type = kwargs.get("lib_type", "cpu")

    async def load(self) -> None:
        """加载 CTransformers 模型"""
        try:
            from ctransformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ImportError("ctransformers 未安装。请运行: pip install ctransformers") from err

        logger.info(f"Loading CTransformers model from {self.model_path}")

        loop = asyncio.get_event_loop()

        def load_model():
            return AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                model_type=self.model_type,
                lib_type=self.lib_type,
            )

        def load_tokenizer():
            try:
                return AutoTokenizer.from_pretrained(str(self.model_path))
            except Exception:
                return None

        self.model = await loop.run_in_executor(None, load_model)
        self.tokenizer = await loop.run_in_executor(None, load_tokenizer)

        logger.info("CTransformers model loaded successfully")

    async def unload(self) -> None:
        """卸载模型"""
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[StreamResult]:
        """CTransformers 流式生成"""

        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_event_loop()

        if config.stream:
            def stream_generate():
                for token in self.model(
                    prompt,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repeat_penalty,
                    stream=True,
                ):
                    yield token

            async for token in loop.run_in_executor(None, lambda: list(stream_generate())):
                yield StreamResult(content=token, done=False)

            yield StreamResult(content="", done=True)
        else:
            def generate_sync():
                return self.model(
                    prompt,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repeat_penalty,
                )

            content = await loop.run_in_executor(None, generate_sync)
            yield StreamResult(content=content, done=True)


class LLMEngine:
    """
    LightLLM 核心引擎
    统一管理多后端模型加载和推理
    """

    def __init__(
        self,
        model_path: str,
        backend: Union[BackendType, str] = BackendType.LLAMA_CPP,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.model_path = Path(model_path)
        self.config = config or {}
        self.kwargs = kwargs

        if isinstance(backend, str):
            backend = BackendType(backend)
        self.backend_type = backend

        self._backend: Optional[ModelBackend] = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """加载模型"""
        async with self._lock:
            if self._backend and self._backend.is_loaded:
                logger.warning("Model already loaded")
                return

            # 创建后端实例
            backend_map = {
                BackendType.LLAMA_CPP: LlamaCppBackend,
                BackendType.VLLM: VLLMBackend,
                BackendType.CTRANSFORMERS: CTransformersBackend,
            }

            backend_class = backend_map.get(self.backend_type)
            if not backend_class:
                raise ValueError(f"Unsupported backend: {self.backend_type}")

            self._backend = backend_class(str(self.model_path), **self.kwargs)
            await self._backend.load()

    async def unload(self) -> None:
        """卸载模型"""
        async with self._lock:
            if self._backend:
                await self._backend.unload()
                self._backend = None

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamResult]:
        """
        生成文本

        Args:
            prompt: 用户输入
            system: 系统提示
            config: 生成配置

        Yields:
            StreamResult: 流式结果
        """
        if not self._backend or not self._backend.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        # 构建完整 prompt
        full_prompt = prompt
        if system:
            full_prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"

        async for result in self._backend.generate(full_prompt, config):
            yield result

    async def complete(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """一次性生成（非流式）"""
        if config is None:
            config = GenerationConfig(stream=False)
        else:
            config.stream = False

        result = ""
        async for chunk in self.generate(prompt, config=config):
            if chunk.content:
                result += chunk.content

        return result

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._backend is not None and self._backend.is_loaded

    @property
    def backend_name(self) -> str:
        """获取后端名称"""
        return self.backend_type.value


# 便捷工厂函数
async def create_engine(
    model_path: str,
    backend: str = "llama_cpp",
    **kwargs
) -> LLMEngine:
    """创建 LLM 引擎"""
    engine = LLMEngine(model_path, backend=backend, **kwargs)
    await engine.load()
    return engine
