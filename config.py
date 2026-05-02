"""
LightLLM 配置文件
支持多种推理后端和模型格式
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 模型缓存目录
MODEL_CACHE_DIR = Path(os.environ.get("LIGHTLLM_MODEL_DIR", 
    str(Path.home() / ".cache" / "lightllm" / "models")))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BackendType(Enum):
    """支持的推理后端"""
    LLAMA_CPP = "llama_cpp"      # llama.cpp (CPU/GPU)
    VLLM = "vllm"                 # vLLM (需要NVIDIA GPU)
    CTRANSFORMERS = "ctransformers"  # CTranslate2
    OPENAI = "openai"             # OpenAI API 兼容
    ANTHROPIC = "anthropic"       # Anthropic API


@dataclass
class BackendConfig:
    """后端通用配置"""
    backend: BackendType = BackendType.LLAMA_CPP
    n_ctx: int = 4096             # 上下文长度
    n_threads: int = None         # CPU线程数 (None=自动)
    n_gpu_layers: int = 0         # GPU加速层数
    use_flash_attention: bool = True  # Flash Attention
    low_vram: bool = True         # 低显存模式
    
    # 量化配置 (llama.cpp)
    n_threads_batch: int = None  # 批处理线程
    rope_freq_base: float = 0     # RoPE 频率基数


@dataclass
class LlamaCppConfig(BackendConfig):
    """llama.cpp 专用配置"""
    backend: BackendType = BackendType.LLAMA_CPP
    n_ctx: int = 4096
    n_threads: int = None         # 自动检测
    n_gpu_layers: int = 24       # 默认开启GPU
    rope_freq_base: float = 10000_000  # Qwen 默认
    rope_freq_scale: float = 1.0
    
    # 内存优化
    use_mlock: bool = True        # 锁定内存
    use_mmap: bool = True         # 内存映射
    no_perf: bool = True          # 禁用性能计数器
    
    # 生成参数
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: List[str] = field(default_factory=lambda: [
        "</s>", "<|endoftext|>", "\n\n"
    ])


@dataclass 
class VLLMConfig(BackendConfig):
    """vLLM 专用配置"""
    backend: BackendType = BackendType.VLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    enforce_eager: bool = False   # 强制 eager 模式
    swap_space: int = 4           # Swap空间 GB


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    backend: BackendType = BackendType.LLAMA_CPP
    quantize: str = "Q4_K_M"      # 量化等级
    context_length: int = 4096
    recommended_gpu_layers: int = 32
    
    # 模型格式
    is_gguf: bool = True
    is_safetensors: bool = False
    
    @property
    def size_mb(self) -> float:
        """估算模型大小"""
        import os
        if os.path.exists(self.path):
            return os.path.getsize(self.path) / (1024 * 1024)
        return 0


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or str(PROJECT_ROOT / "config.json")
        self._config: Dict = {}
        self._load()
    
    def _load(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "default_backend": "llama_cpp",
            "model_path": str(MODEL_CACHE_DIR),
            "context_length": 4096,
            "gpu_layers": 32,
            "threads": os.cpu_count() or 4,
            "generation": {
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }
    
    def save(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default=None):
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def set(self, key: str, value):
        """设置配置项"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


# 全局配置实例
config = ConfigManager()


# 预设模型配置
PRESET_MODELS = {
    "tinyllama": {
        "name": "TinyLlama 1.1B",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 650,
        "context_length": 2048,
        "gpu_layers": 16,
        "description": "最小可用模型，CPU可运行"
    },
    "phi-2": {
        "name": "Phi-2",
        "repo": "TheBloke/phi-2-GGUF",
        "file": "phi-2.Q4_K_M.gguf",
        "size_mb": 1800,
        "context_length": 2048,
        "gpu_layers": 24,
        "description": "微软Phi-2 2.7B，平衡性能"
    },
    "qwen2-0.5b": {
        "name": "Qwen2 0.5B",
        "repo": "Qwen",  # 实际需要替换
        "file": "qwen2-0.5b-integer-quants.Q4_K_M.gguf",
        "size_mb": 400,
        "context_length": 8192,
        "gpu_layers": 32,
        "description": "通义千问2 0.5B，支持长上下文"
    },
    "llama2-7b": {
        "name": "Llama-2 7B",
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.Q4_K_M.gguf",
        "size_mb": 3800,
        "context_length": 4096,
        "gpu_layers": 35,
        "description": "Llama-2 7B，需要4GB+显存"
    },
    "mistral-7b": {
        "name": "Mistral 7B",
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_mb": 4100,
        "context_length": 8192,
        "gpu_layers": 35,
        "description": "Mistral 7B，优质对话模型"
    },
    "qwen2.5-0.5b": {
        "name": "Qwen2.5 0.5B",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_mb": 400,
        "context_length": 8192,
        "gpu_layers": 32,
        "description": "Qwen2.5 0.5B，最新中国开源模型"
    }
}


def get_model_path(model_name: str) -> Optional[str]:
    """获取已下载模型的完整路径"""
    model_dir = MODEL_CACHE_DIR / model_name
    if model_dir.exists():
        for ext in ['*.gguf', '*.bin', '*.safetensors']:
            from glob import glob
            matches = glob(str(model_dir / ext))
            if matches:
                return matches[0]
    return None


def create_backend_config(backend: BackendType = BackendType.LLAMA_CPP, **kwargs) -> BackendConfig:
    """创建后端配置"""
    configs = {
        BackendType.LLAMA_CPP: LlamaCppConfig,
        BackendType.VLLM: VLLMConfig,
    }
    
    config_class = configs.get(backend, BackendConfig)
    return config_class(**kwargs)