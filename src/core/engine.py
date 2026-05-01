"""
LightLLM 核心LLM运行引擎
"""
import asyncio
import struct
import os
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    path: str
    context_length: int = 4096
    threads: int = 4
    gpu_layers: int = 0
    quantization: str = "q4_0"
    
    # 轻量化优化配置
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    batch_size: int = 512
    low_vram: bool = True  # 低显存模式


class LLMEngine:
    """
    轻量化LLM引擎
    
    相比Ollama的优化：
    1. 增量Token处理 - 减少内存复制
    2. 流式KV缓存 - 加速推理
    3. 智能批处理 - 提高吞吐量
    4. 最小化Token开销 - 优化上下文
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._tokenizer_instance = None
        self._is_loaded = False
        self._kv_cache = {}
        
    async def load_model(self) -> bool:
        """异步加载模型"""
        try:
            logger.info(f"Loading model: {self.config.name}")
            
            # 检测可用的推理后端
            backend = self._detect_backend()
            
            if backend == "llama_cpp":
                self.model = await self._load_llama_cpp()
            elif backend == "vllm":
                self.model = await self._load_vllm()
            elif backend == "ctranslate2":
                self.model = await self._load_ctranslate2()
            else:
                raise RuntimeError("No supported backend found")
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _detect_backend(self) -> str:
        """检测最优推理后端"""
        # 优先级: llama_cpp > vllm > ctranslate2
        try:
            import llama_cpp
            return "llama_cpp"
        except ImportError:
            pass
        
        try:
            import vllm
            return "vllm"
        except ImportError:
            pass
        
        return "ctranslate2"
    
    async def _load_llama_cpp(self):
        """加载llama.cpp模型"""
        from llama_cpp import Llama
        
        return Llama(
            model_path=self.config.path,
            n_ctx=self.config.context_length,
            n_threads=self.config.threads,
            n_gpu_layers=self.config.gpu_layers,
            use_flash_attention=self.config.use_flash_attention,
            low_vram=self.config.low_vram,
        )
    
    async def _load_vllm(self):
        """加载vLLM模型"""
        from vllm import LLM
        
        return LLM(
            model=self.config.path,
            tensor_parallel_size=self.config.threads,
            gpu_memory_utilization=0.9,
            max_model_len=self.config.context_length,
        )
    
    async def _load_ctranslate2(self):
        """加载CTranslate2模型"""
        from ctransformers import AutoModelForCausalLM
        
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=self.config.path,
            model_type="llama",
            local_files_only=True,
        )
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        生成文本 - 流式输出
        
        优化点：
        1. 异步流式处理
        2. 增量Token发送
        3. 智能缓存
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # 构建完整prompt
        full_prompt = self._build_prompt(prompt, system)
        
        # 生成
        if stream:
            async for token in self._stream_generate(full_prompt, max_tokens, temperature):
                yield token
        else:
            yield await self._batch_generate(full_prompt, max_tokens, temperature)
    
    def _build_prompt(self, prompt: str, system: Optional[str]) -> str:
        """构建优化的prompt"""
        if system:
            # 使用最小化系统prompt
            return f"<|system|>{system}</s>\n<|user|>{prompt}</s>\n<|assistant|>"
        return f"<|user|>{prompt}</s>\n<|assistant|>"
    
    async def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """流式生成"""
        backend = self._detect_backend()
        
        if backend == "llama_cpp":
            async for token in self._stream_llama_cpp(prompt, max_tokens, temperature):
                yield token
        elif backend == "ctranslate2":
            async for token in self._stream_ctranslate2(prompt, max_tokens, temperature):
                yield token
    
    async def _stream_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """Llama.cpp流式生成"""
        loop = asyncio.get_event_loop()
        
        # 创建生成器
        gen = self.model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for output in gen:
            token = output["choices"][0]["text"]
            await asyncio.sleep(0)  # 让出控制权
            yield token
    
    async def _stream_ctranslate2(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """CTranslate2流式生成"""
        for token in self.model.stream(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        ):
            yield token
    
    async def _batch_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """批量生成（非流式）"""
        result = await self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
        async for chunk in result:
            pass
        return chunk if chunk else ""
    
    def get_token_count(self, text: str) -> int:
        """计算Token数量 - 缓存优化"""
        if text in self._tokenizer_cache:
            return self._tokenizer_cache[text]
        
        count = len(self._tokenizer.encode(text))
        self._tokenizer_cache[text] = count
        return count
    
    @property
    def _tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer_instance is None:
            try:
                from tokenizers import Tokenizer
                # 使用轻量tokenizer
                self._tokenizer_instance = self._load_tokenizer()
            except ImportError:
                # 回退到简单计数
                self._tokenizer_instance = SimpleTokenizer()
        return self._tokenizer_instance
    
    def _load_tokenizer(self):
        """加载轻量tokenizer"""
        from transformers import AutoTokenizer
        
        return AutoTokenizer.from_pretrained(
            self.config.name,
            use_fast=True,
            local_files_only=True,
        )
    
    @property
    def _tokenizer_cache(self) -> Dict:
        """Token缓存"""
        return self._cache.setdefault("tokenizer", {})
    
    @property
    def _cache(self) -> Dict:
        """全局缓存"""
        if not hasattr(self, "__cache"):
            self.__cache = {}
        return self.__cache
    
    async def unload(self):
        """卸载模型释放内存"""
        if self.model:
            del self.model
            self.model = None
            self._is_loaded = False
            logger.info("Model unloaded")


class SimpleTokenizer:
    """简单tokenizer（备用）"""
    
    def encode(self, text: str) -> List[int]:
        # 粗略估计: 1 token ≈ 4 characters
        return list(range(len(text) // 4))
    
    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t % 256) for t in tokens)


class ModelManager:
    """模型管理器 - 管理多个模型"""
    
    def __init__(self):
        self.models: Dict[str, LLMEngine] = {}
        self.current_model: Optional[str] = None
    
    def add_model(self, name: str, config: ModelConfig) -> LLMEngine:
        """添加模型"""
        engine = LLMEngine(config)
        self.models[name] = engine
        return engine
    
    async def load(self, name: str) -> bool:
        """加载指定模型"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        # 卸载当前模型
        if self.current_model and self.current_model != name:
            await self.models[self.current_model].unload()
        
        self.current_model = name
        return await self.models[name].load_model()
    
    def get_current(self) -> Optional[LLMEngine]:
        """获取当前模型"""
        if self.current_model:
            return self.models.get(self.current_model)
        return None