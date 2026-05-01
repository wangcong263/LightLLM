#!/usr/bin/env python3
"""
LightLLM 模型管理器 v1.1
支持多种来源下载和管理本地大模型
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Callable, Any
from enum import Enum
import argparse
import hashlib
import requests
from urllib.parse import quote

# ============================================
# 配置
# ============================================

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "lightllm" / "models"
DEFAULT_PORT = 8000
DEFAULT_CONTEXT_LEN = 4096

# ============================================
# 枚举定义
# ============================================

class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    MOBAGEN = "mobagen"
    LOCAL = "local"

class QuantizeLevel(Enum):
    FP16 = "FP16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q4_K_M = "Q4_K_M"
    Q4_0 = "Q4_0"
    Q3_K_M = "Q3_K_M"
    Q2_K = "Q2_K"
    GGUF_F16 = "gguf-f16"
    GGUF_Q4_0 = "gguf-q4_0"
    GGUF_Q4_K_M = "gguf-q4_K_M"

# ============================================
# 数据类
# ============================================

@dataclass
class ModelConfig:
    id: str
    name: str
    size_gb: float
    memory_mb: int
    source: ModelSource
    repo_id: str
    filename: str
    description: str = ""
    min_ggml_version: str = "0.10.0"
    default_quantize: str = "Q4_K_M"
    requires_gpu: bool = False
    recommended_scenarios: List[str] = None
    
    def __post_init__(self):
        if self.recommended_scenarios is None:
            self.recommended_scenarios = []
        if isinstance(self.source, str):
            self.source = ModelSource(self.source)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "size_gb": self.size_gb,
            "memory_mb": self.memory_mb,
            "source": self.source.value if isinstance(self.source, ModelSource) else self.source,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "description": self.description,
            "min_ggml_version": self.min_ggml_version,
            "default_quantize": self.default_quantize,
            "requires_gpu": self.requires_gpu,
            "recommended_scenarios": self.recommended_scenarios,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        return cls(**data)

@dataclass
class DownloadProgress:
    model_id: str
    downloaded: int  # bytes
    total: int  # bytes
    speed: str  # "1.2 MB/s"
    status: str  # "downloading" / "completed" / "failed"
    
    def percent(self) -> float:
        if self.total == 0:
            return 0
        return round(self.downloaded / self.total * 100, 1)

# ============================================
# 模型目录
# ============================================

class ModelCatalog:
    """内置模型目录"""
    
    MODELS: Dict[str, ModelConfig] = {
        # Tiny 模型 (< 1B 参数)
        "tinyllama-1.1b": ModelConfig(
            id="tinyllama-1.1b",
            name="TinyLlama 1.1B",
            size_gb=0.65,
            memory_mb=1024,
            source=ModelSource.HUGGINGFACE,
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            description="极轻量级模型，适合学习和测试",
            recommended_scenarios=["学习", "测试", "嵌入式"],
            requires_gpu=False,
        ),
        # 小模型 (1-3B)
        "qwen2.5-0.5b": ModelConfig(
            id="qwen2.5-0.5b",
            name="Qwen2.5 0.5B",
            size_gb=0.35,
            memory_mb=1024,
            source=ModelSource.MODELSCOPE,
            repo_id="qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            description="阿里通义千问轻量版，中文支持好",
            recommended_scenarios=["中文对话", "轻量任务"],
            requires_gpu=False,
        ),
        "qwen2.5-1.5b": ModelConfig(
            id="qwen2.5-1.5b",
            name="Qwen2.5 1.5B",
            size_gb=0.94,
            memory_mb=2048,
            source=ModelSource.MODELSCOPE,
            repo_id="qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            description="阿里通义千问，适合日常对话",
            recommended_scenarios=["中文对话", "问答", "写作"],
            requires_gpu=False,
        ),
        "phi-2": ModelConfig(
            id="phi-2",
            name="Phi-2 2.7B",
            size_gb=1.8,
            memory_mb=4096,
            source=ModelSource.HUGGINGFACE,
            repo_id="TheBloke/phi-2-GGUF",
            filename="phi-2.Q4_K_M.gguf",
            description="微软小模型，推理能力强",
            recommended_scenarios=["推理", "代码", "问答"],
            requires_gpu=False,
        ),
        "qwen2.5-3b": ModelConfig(
            id="qwen2.5-3b",
            name="Qwen2.5 3B",
            size_gb=1.9,
            memory_mb=4096,
            source=ModelSource.MODELSCOPE,
            repo_id="qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            description="阿里通义千问，性价比高",
            recommended_scenarios=["中文对话", "代码", "写作"],
            requires_gpu=False,
        ),
        # 中等模型 (7B)
        "qwen2.5-7b": ModelConfig(
            id="qwen2.5-7b",
            name="Qwen2.5 7B",
            size_gb=4.4,
            memory_mb=8192,
            source=ModelSource.MODELSCOPE,
            repo_id="qwen/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            description="阿里通义千问主力版本，效果好",
            recommended_scenarios=["复杂对话", "代码生成", "长文本"],
            requires_gpu=True,
        ),
        "llama-3.2-3b": ModelConfig(
            id="llama-3.2-3b",
            name="Llama 3.2 3B",
            size_gb=1.9,
            memory_mb=4096,
            source=ModelSource.HUGGINGFACE,
            repo_id="TheBloke/Llama-3.2-3B-Instruct-GGUF",
            filename="llama-3.2-3b-instruct-q4_k_m.gguf",
            description="Meta 最新开源，中英文俱佳",
            recommended_scenarios=["通用对话", "翻译", "创作"],
            requires_gpu=True,
        ),
        "mistral-7b": ModelConfig(
            id="mistral-7b",
            name="Mistral 7B v0.3",
            size_gb=4.1,
            memory_mb=8192,
            source=ModelSource.HUGGINGFACE,
            repo_id="TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
            filename="mistral-7b-instruct-v0.3.q4_k_m.gguf",
            description="欧洲最强开源模型，推理优秀",
            recommended_scenarios=["推理", "代码", "复杂任务"],
            requires_gpu=True,
        ),
        "yi-1.5-6b": ModelConfig(
            id="yi-1.5-6b",
            name="Yi 1.5 6B",
            size_gb=3.8,
            memory_mb=8192,
            source=ModelSource.HUGGINGFACE,
            repo_id="TheBloke/Yi-1.5-6B-Chat-GGUF",
            filename="yi-1.5-6b-chat-q4_k_m.gguf",
            description="零一万物，中文能力强",
            recommended_scenarios=["中文对话", "长文本", "分析"],
            requires_gpu=True,
        ),
        # 大模型 (8B+)
        "llama-3.1-8b": ModelConfig(
            id="llama-3.1-8b",
            name="Llama 3.1 8B",
            size_gb=4.9,
            memory_mb=12288,
            source=ModelSource.HUGGINGFACE,
            repo_id="TheBloke/Llama-3.1-8B-Instruct-GGUF",
            filename="llama-3.1-8b-instruct-q4_k_m.gguf",
            description="Meta 最新旗舰开源，效果出色",
            recommended_scenarios=["高级对话", "复杂推理", "代码"],
            requires_gpu=True,
        ),
        "qwen2.5-14b": ModelConfig(
            id="qwen2.5-14b",
            name="Qwen2.5 14B",
            size_gb=8.2,
            memory_mb=16384,
            source=ModelSource.MODELSCOPE,
            repo_id="qwen/Qwen2.5-14B-Instruct-GGUF",
            filename="qwen2.5-14b-instruct-q4_k_m.gguf",
            description="阿里通义千问大杯，效果更好",
            recommended_scenarios=["专业对话", "复杂分析", "长文本"],
            requires_gpu=True,
        ),
        "qwen2.5-32b": ModelConfig(
            id="qwen2.5-32b",
            name="Qwen2.5 32B",
            size_gb=18.5,
            memory_mb=32768,
            source=ModelSource.MODELSCOPE,
            repo_id="qwen/Qwen2.5-32B-Instruct-GGUF",
            filename="qwen2.5-32b-instruct-q4_k_m.gguf",
            description="阿里通义千问超大杯，效果接近GPT-4",
            recommended_scenarios=["专业领域", "复杂推理", "科研"],
            requires_gpu=True,
        ),
        # embedding 模型
        "nomic-embed-text": ModelConfig(
            id="nomic-embed-text",
            name="Nomic Embed Text",
            size_gb=0.27,
            memory_mb=1024,
            source=ModelSource.HUGGINGFACE,
            repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
            filename="nomic-embed-text-v1.5.q4_k_m.gguf",
            description="高质量文本嵌入模型",
            recommended_scenarios=["向量化", "语义搜索", "RAG"],
            requires_gpu=False,
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelConfig]:
        return cls.MODELS
    
    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelConfig]:
        return cls.MODELS.get(model_id)
    
    @classmethod
    def list_by_size(cls, max_size_gb: float = 10) -> List[ModelConfig]:
        return [m for m in cls.MODELS.values() if m.size_gb <= max_size_gb]
    
    @classmethod
    def recommend_for_memory(cls, available_mb: int) -> List[ModelConfig]:
        available_gb = available_mb / 1024
        recommended = []
        for m in sorted(cls.MODELS.values(), key=lambda x: x.memory_mb):
            if m.memory_mb <= available_mb * 0.8:
                recommended.append(m)
        return recommended

# ============================================
# 模型下载器
# ============================================

class ModelDownloader:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, progress_callback: Optional[Callable] = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self._running_downloads: Dict[str, subprocess.Popen] = {}
    
    def _notify(self, progress: DownloadProgress):
        if self.progress_callback:
            self.progress_callback(progress)
    
    def _install_deps(self):
        """安装必要的依赖"""
        print("正在安装依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "-q"], check=True)
    
    def download(
        self,
        model_config: ModelConfig,
        quantize: str = None,
        force: bool = False
    ) -> Path:
        """下载模型"""
        if quantize is None:
            quantize = model_config.default_quantize
        
        # 构造文件名
        filename = model_config.filename
        if quantize and quantize != model_config.default_quantize:
            filename = filename.replace(model_config.default_quantize, quantize)
        
        model_path = self.model_dir / filename
        
        # 检查是否已存在
        if model_path.exists() and not force:
            print(f"模型已存在: {model_path}")
            return model_path
        
        # 根据来源下载
        if model_config.source == ModelSource.HUGGINGFACE:
            return self._download_huggingface(model_config, filename, quantize)
        elif model_config.source == ModelSource.MODELSCOPE:
            return self._download_modelscope(model_config, filename, quantize)
        else:
            raise ValueError(f"不支持的来源: {model_config.source}")
    
    def _download_huggingface(
        self,
        model_config: ModelConfig,
        filename: str,
        quantize: str
    ) -> Path:
        """从 HuggingFace 下载"""
        print(f"从 HuggingFace 下载: {model_config.repo_id}/{filename}")
        
        base_url = f"https://huggingface.co/{model_config.repo_id}/resolve/main/{filename}"
        
        # 创建下载目录
        temp_dir = self.model_dir / ".download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        
        try:
            response = requests.get(base_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = downloaded / total_size * 100
                            speed = downloaded / 1024 / 1024  # MB
                            self._notify(DownloadProgress(
                                model_id=model_config.id,
                                downloaded=downloaded,
                                total=total_size,
                                speed=f"{speed:.1f} MB/s",
                                status="downloading"
                            ))
            
            # 移动到最终位置
            final_path = self.model_dir / filename
            shutil.move(str(temp_path), str(final_path))
            
            self._notify(DownloadProgress(
                model_id=model_config.id,
                downloaded=total_size,
                total=total_size,
                speed="",
                status="completed"
            ))
            
            return final_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _download_modelscope(
        self,
        model_config: ModelConfig,
        filename: str,
        quantize: str
    ) -> Path:
        """从 ModelScope 下载（国内加速）"""
        print(f"从 ModelScope 下载: {model_config.repo_id}/{filename}")
        
        # ModelScope 下载链接格式
        base_url = f"https://modelscope.cn/models/{model_config.repo_id}/resolve/master/{filename}"
        
        temp_dir = self.model_dir / ".download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        
        try:
            response = requests.get(base_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = downloaded / total_size * 100
                            speed = downloaded / 1024 / 1024
                            self._notify(DownloadProgress(
                                model_id=model_config.id,
                                downloaded=downloaded,
                                total=total_size,
                                speed=f"{speed:.1f} MB/s",
                                status="downloading"
                            ))
            
            final_path = self.model_dir / filename
            shutil.move(str(temp_path), str(final_path))
            
            self._notify(DownloadProgress(
                model_id=model_config.id,
                downloaded=total_size,
                total=total_size,
                speed="",
                status="completed"
            ))
            
            return final_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def list_installed(self) -> List[Dict]:
        """列出已安装的模型"""
        installed = []
        if not self.model_dir.exists():
            return installed
        
        for model_path in self.model_dir.glob("*.gguf"):
            # 尝试匹配已知模型
            matched = None
            for config in ModelCatalog.MODELS.values():
                if config.filename == model_path.name:
                    matched = config
                    break
            
            stat = model_path.stat()
            info = {
                "filename": model_path.name,
                "path": str(model_path),
                "size_gb": round(stat.st_size / 1024**3, 2),
                "installed": True,
                "config": matched.to_dict() if matched else None
            }
            installed.append(info)
        
        return installed
    
    def delete(self, model_id: str) -> bool:
        """删除已安装的模型"""
        config = ModelCatalog.get_model(model_id)
        if not config:
            return False
        
        # 查找匹配的文件
        for model_path in self.model_dir.glob(f"*{config.filename.split('-')[0]}*.gguf"):
            if config.filename in str(model_path) or model_path.stat().st_size < 10 * 1024**3:
                model_path.unlink()
                print(f"已删除: {model_path}")
                return True
        
        return False

# ============================================
# 辅助函数
# ============================================

def list_popular_models() -> Dict[str, ModelConfig]:
    """获取热门模型列表（用于WebUI）"""
    return ModelCatalog.get_all_models()

def get_model_info(model_id: str) -> Optional[Dict]:
    """获取模型详细信息（用于WebUI）"""
    config = ModelCatalog.get_model(model_id)
    if config:
        return config.to_dict()
    return None

def get_system_info() -> Dict:
    """获取系统信息"""
    import psutil
    
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 1),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu": []
    }
    
    # 尝试检测 GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"].append({
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "available": True
            })
    except ImportError:
        pass
    
    return info

# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="LightLLM 模型管理工具")
    parser.add_argument("action", choices=["list", "install", "remove", "info", "recommend"], 
                        help="操作: list(列表) / install(安装) / remove(删除) / info(详情) / recommend(推荐)")
    parser.add_argument("model", nargs="?", help="模型ID")
    parser.add_argument("--all", action="store_true", help="显示所有模型")
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope",
                        help="下载来源")
    
    args = parser.parse_args()
    downloader = ModelDownloader()
    
    if args.action == "list":
        models = ModelCatalog.get_all_models()
        print("\n📦 可用模型列表:")
        print("-" * 70)
        for m in models.values():
            src = "🤗 HF" if m.source == ModelSource.HUGGINGFACE else "📦 MS"
            gpu = "🖥️" if m.requires_gpu else "💻"
            print(f"{src} {gpu} {m.id:20} | {m.name:20} | {m.size_gb:5.1f} GB | 需要 {m.memory_mb} MB")
    
    elif args.action == "install":
        if not args.model:
            print("错误: 请指定要安装的模型ID")
            return 1
        
        config = ModelCatalog.get_model(args.model)
        if not config:
            print(f"错误: 未找到模型 '{args.model}'")
            return 1
        
        print(f"开始安装: {config.name}")
        path = downloader.download(config)
        print(f"✅ 安装完成: {path}")
    
    elif args.action == "remove":
        if not args.model:
            print("错误: 请指定要删除的模型ID")
            return 1
        
        if downloader.delete(args.model):
            print(f"✅ 已删除模型: {args.model}")
        else:
            print(f"错误: 未找到已安装的模型 '{args.model}'")
    
    elif args.action == "info":
        if not args.model:
            print("错误: 请指定模型ID")
            return 1
        
        config = ModelCatalog.get_model(args.model)
        if not config:
            print(f"错误: 未找到模型 '{args.model}'")
            return 1
        
        print(f"\n📋 {config.name}")
        print("-" * 40)
        print(f"ID: {config.id}")
        print(f"大小: {config.size_gb} GB")
        print(f"内存需求: {config.memory_mb} MB")
        print(f"来源: {config.source.value}")
        print(f"仓库: {config.repo_id}")
        print(f"描述: {config.description}")
        print(f"场景: {', '.join(config.recommended_scenarios)}")
    
    elif args.action == "recommend":
        info = get_system_info()
        available_mb = int(info["memory_available_gb"] * 1024)
        recommended = ModelCatalog.recommend_for_memory(available_mb)
        
        print(f"\n💡 根据您的系统 ({info['memory_available_gb']} GB 可用内存) 推荐:")
        print("-" * 70)
        for m in recommended[:5]:
            print(f"⭐ {m.name} ({m.size_gb} GB)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())