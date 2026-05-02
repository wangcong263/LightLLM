#!/usr/bin/env python3
"""
LightLLM 模型管理器 v2.0
支持多种来源下载和管理本地大模型
"""

import os
import sys
import json
import shutil
import subprocess
import requests
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Callable
from enum import Enum
import argparse

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "lightllm" / "models"

DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ModelSource(Enum):
    """模型来源"""
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    CIVITAI = "civitai"
    OLLAMA = "ollama"
    PIXART = "pixart"
    LOCAL = "local"


class QuantizeLevel(Enum):
    """量化等级"""
    Q2_K = "Q2_K"      # 最高压缩，质量较低
    Q3_K_M = "Q3_K_M"  # 中等压缩
    Q4_0 = "Q4_0"      # 标准量化
    Q4_K_M = "Q4_K_M"  # 平衡压缩（推荐）
    Q5_0 = "Q5_0"      # 高质量
    Q5_K_M = "Q5_K_M"  # 高质量压缩
    Q6_K = "Q6_K"      # 接近原生
    Q8_0 = "Q8_0"      # 几乎无损
    FP16 = "FP16"      # 半精度
    FP32 = "FP32"      # 全精度


@dataclass
class ModelConfig:
    """模型配置"""
    id: str
    name: str
    source: ModelSource
    repo_id: str = ""           # HuggingFace repo_id
    filename: str = ""           # 下载文件名
    size_mb: int = 0
    default_quantize: str = "Q4_K_M"
    min_memory_mb: int = 4000    # 最低内存需求
    recommended_memory_mb: int = 8000
    description: str = ""
    tags: List[str] = field(default_factory=list)
    image_model: bool = False    # 是否是图像模型
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        if isinstance(data.get('source'), str):
            data['source'] = ModelSource(data['source'])
        if isinstance(data.get('tags'), list):
            data['tags'] = data['tags']
        return cls(**data)


@dataclass
class DownloadProgress:
    """下载进度"""
    model_id: str
    downloaded_mb: float = 0
    total_mb: float = 0
    speed_mbps: float = 0
    progress_percent: float = 0
    status: str = "pending"  # pending, downloading, completed, failed
    
    def percent(self) -> float:
        if self.total_mb > 0:
            return round(self.downloaded_mb / self.total_mb * 100, 1)
        return 0


class ModelCatalog:
    """模型目录"""
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelConfig]:
        """获取所有模型"""
        models = {}
        models.update(cls.get_llm_models())
        models.update(cls.get_image_models())
        return models
    
    @classmethod
    def get_llm_models(cls) -> Dict[str, ModelConfig]:
        """获取 LLM 模型"""
        return {
            # === HuggingFace 模型 ===
            "llama-3.1-8b-q4": ModelConfig(
                id="llama-3.1-8b-q4",
                name="Llama 3.1 8B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="NousResearch/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="*Q4_K_M*",
                size_mb=4870,
                default_quantize="Q4_K_M",
                min_memory_mb=6000,
                recommended_memory_mb=10000,
                description="Meta 最新开源大模型，8B 参数",
                tags=["llama", "meta", "instruct"]
            ),
            "llama-3.2-3b-q4": ModelConfig(
                id="llama-3.2-3b-q4",
                name="Llama 3.2 3B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="NousResearch/Llama-3.2-3B-Instruct-GGUF",
                filename="*Q4_K_M*",
                size_mb=2010,
                default_quantize="Q4_K_M",
                min_memory_mb=4000,
                recommended_memory_mb=6000,
                description="轻量级 Llama 模型，3B 参数",
                tags=["llama", "meta", "轻量"]
            ),
            "qwen-2.5-7b-q4": ModelConfig(
                id="qwen-2.5-7b-q4",
                name="Qwen 2.5 7B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                filename="*Q4_K_M*",
                size_mb=4470,
                default_quantize="Q4_K_M",
                min_memory_mb=6000,
                recommended_memory_mb=8000,
                description="阿里通义千问 2.5，7B 参数",
                tags=["qwen", "阿里", "中文"]
            ),
            "qwen-2.5-3b-q4": ModelConfig(
                id="qwen-2.5-3b-q4",
                name="Qwen 2.5 3B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="*Q4_K_M*",
                size_mb=2010,
                default_quantize="Q4_K_M",
                min_memory_mb=4000,
                recommended_memory_mb=6000,
                description="轻量级千问模型，3B 参数",
                tags=["qwen", "阿里", "轻量"]
            ),
            "qwen-2.5-1.5b-q4": ModelConfig(
                id="qwen-2.5-1.5b-q4",
                name="Qwen 2.5 1.5B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="*Q4_K_M*",
                size_mb=1080,
                default_quantize="Q4_K_M",
                min_memory_mb=2000,
                recommended_memory_mb=4000,
                description="超轻量千问模型，1.5B 参数",
                tags=["qwen", "阿里", "超轻量"]
            ),
            "mistral-7b-q4": ModelConfig(
                id="mistral-7b-q4",
                name="Mistral 7B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                filename="*Q4_K_M*",
                size_mb=4470,
                default_quantize="Q4_K_M",
                min_memory_mb=6000,
                recommended_memory_mb=8000,
                description="Mistral 7B 指令微调版",
                tags=["mistral", "instruct"]
            ),
            "phi-3.5-3b-q4": ModelConfig(
                id="phi-3.5-3b-q4",
                name="Phi-3.5 3B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="microsoft/Phi-3.5-mini-instruct-gguf",
                filename="*Q4_K_M*",
                size_mb=2200,
                default_quantize="Q4_K_M",
                min_memory_mb=4000,
                recommended_memory_mb=6000,
                description="微软 Phi-3.5 mini，3B 参数",
                tags=["phi", "microsoft", "轻量"]
            ),
            "deepseek-7b-q4": ModelConfig(
                id="deepseek-7b-q4",
                name="DeepSeek 7B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="TheBloke/deepseek-llm-7b-base-GGUF",
                filename="*Q4_K_M*",
                size_mb=4470,
                default_quantize="Q4_K_M",
                min_memory_mb=6000,
                recommended_memory_mb=8000,
                description="深度求索 7B 基座模型",
                tags=["deepseek", "基座"]
            ),
            "yi-1.5-6b-q4": ModelConfig(
                id="yi-1.5-6b-q4",
                name="Yi 1.5 6B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="TheBloke/Yi-1.5-6B-Chat-GGUF",
                filename="*Q4_K_M*",
                size_mb=3600,
                default_quantize="Q4_K_M",
                min_memory_mb=5000,
                recommended_memory_mb=8000,
                description="零一万物 Yi 1.5 6B",
                tags=["yi", "零一", "中文"]
            ),
            "gemma-2b-q4": ModelConfig(
                id="gemma-2b-q4",
                name="Gemma 2B Q4",
                source=ModelSource.HUGGINGFACE,
                repo_id="TheBloke/gemma-2b-it-GGUF",
                filename="*Q4_K_M*",
                size_mb=1610,
                default_quantize="Q4_K_M",
                min_memory_mb=3000,
                recommended_memory_mb=5000,
                description="Google Gemma 2B 指令版",
                tags=["gemma", "google", "轻量"]
            ),
            # === Civitai 模型 (SD 图像模型) ===
            "stable-diffusion-xl": ModelConfig(
                id="stable-diffusion-xl",
                name="Stable Diffusion XL",
                source=ModelSource.CIVITAI,
                repo_id="101680",
                size_mb=6600000,
                default_quantize="FP16",
                min_memory_mb=8000,
                recommended_memory_mb=16000,
                description="SDXL 1.0 图像生成模型",
                tags=["stable-diffusion", "图像", "SDXL"],
                image_model=True
            ),
            "sd-turbo": ModelConfig(
                id="sd-turbo",
                name="SD Turbo",
                source=ModelSource.CIVITAI,
                repo_id="2374712",
                size_mb=5200000,
                default_quantize="FP16",
                min_memory_mb=6000,
                recommended_memory_mb=12000,
                description="快速图像生成模型",
                tags=["stable-diffusion", "快速", "Turbo"],
                image_model=True
            ),
            # === Ollama 模型 ===
            "llama3:latest": ModelConfig(
                id="llama3:latest",
                name="Llama 3 (Ollama)",
                source=ModelSource.OLLAMA,
                repo_id="llama3",
                size_mb=4660,
                default_quantize="Q4_K_M",
                min_memory_mb=6000,
                recommended_memory_mb=8000,
                description="Ollama 官方 Llama 3 模型",
                tags=["llama", "ollama", "官方"]
            ),
            "qwen2.5:latest": ModelConfig(
                id="qwen2.5:latest",
                name="Qwen 2.5 (Ollama)",
                source=ModelSource.OLLAMA,
                repo_id="qwen2.5",
                size_mb=4500,
                default_quantize="Q4_K_M",
                min_memory_mb=6000,
                recommended_memory_mb=8000,
                description="Ollama 官方 Qwen 2.5 模型",
                tags=["qwen", "ollama", "中文"]
            ),
            "mistral:latest": ModelConfig(
                id="mistral:latest",
                name="Mistral (Ollama)",
                source=ModelSource.OLLAMA,
                repo_id="mistral",
                size_mb=4100,
                default_quantize="Q4_K_M",
                min_memory_mb=5000,
                recommended_memory_mb=8000,
                description="Ollama 官方 Mistral 模型",
                tags=["mistral", "ollama", "官方"]
            ),
            "phi3:latest": ModelConfig(
                id="phi3:latest",
                name="Phi-3 (Ollama)",
                source=ModelSource.OLLAMA,
                repo_id="phi3",
                size_mb=2300,
                default_quantize="Q4_K_M",
                min_memory_mb=4000,
                recommended_memory_mb=6000,
                description="Ollama 官方 Phi-3 模型",
                tags=["phi", "ollama", "轻量"]
            ),
            "codellama:latest": ModelConfig(
                id="codellama:latest",
                name="Code Llama (Ollama)",
                source=ModelSource.OLLAMA,
                repo_id="codellama",
                size_mb=3800,
                default_quantize="Q4_K_M",
                min_memory_mb=5000,
                recommended_memory_mb=8000,
                description="Ollama 官方代码模型",
                tags=["llama", "ollama", "代码"]
            ),
            # === Pixart 图像模型 ===
            "pixart-alpha": ModelConfig(
                id="pixart-alpha",
                name="Pixart-α",
                source=ModelSource.PIXART,
                repo_id="PixArt-alpha/PixArt-alpha",
                filename="PixArt-XL-1024-MS-生图的话，需要另外下载 tokenizer",
                size_mb=2400000,
                default_quantize="FP16",
                min_memory_mb=6000,
                recommended_memory_mb=12000,
                description="腾讯 Pixart-α 高质量图像生成",
                tags=["pixart", "图像", "腾讯", "高质量"],
                image_model=True
            ),
            # === 本地模型占位 ===
            "custom-local": ModelConfig(
                id="custom-local",
                name="自定义本地模型",
                source=ModelSource.LOCAL,
                size_mb=0,
                description="添加您本地的自定义模型",
                tags=["自定义", "本地"]
            ),
        }
    
    @classmethod
    def get_image_models(cls) -> Dict[str, ModelConfig]:
        """获取图像生成模型"""
        return {k: v for k, v in cls.get_llm_models().items() if v.image_model}
    
    @classmethod
    def get_by_source(cls, source: ModelSource) -> Dict[str, ModelConfig]:
        """按来源获取模型"""
        return {k: v for k, v in cls.get_all_models().items() if v.source == source}
    
    @classmethod
    def search(cls, query: str) -> List[ModelConfig]:
        """搜索模型"""
        query = query.lower()
        results = []
        for model in cls.get_all_models().values():
            if (query in model.name.lower() or 
                query in model.description.lower() or
                any(query in tag.lower() for tag in model.tags)):
                results.append(model)
        return results


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """设置进度回调"""
        self.progress_callback = callback
    
    def _report_progress(self, model_id: str, downloaded: float, total: float, speed: float = 0):
        """报告进度"""
        if self.progress_callback:
            progress = DownloadProgress(
                model_id=model_id,
                downloaded_mb=downloaded,
                total_mb=total,
                speed_mbps=speed,
                status="downloading"
            )
            self.progress_callback(progress)
    
    def download(self, model_config: ModelConfig) -> Path:
        """下载模型"""
        if model_config.source == ModelSource.HUGGINGFACE:
            return self._download_huggingface(model_config)
        elif model_config.source == ModelSource.MODELSCOPE:
            return self._download_modelscope(model_config)
        elif model_config.source == ModelSource.CIVITAI:
            return self._download_civitai(model_config)
        elif model_config.source == ModelSource.OLLAMA:
            return self._download_ollama(model_config)
        elif model_config.source == ModelSource.PIXART:
            return self._download_huggingface(model_config)
        elif model_config.source == ModelSource.LOCAL:
            raise ValueError("本地模型不需要下载")
        else:
            raise ValueError(f"不支持的来源: {model_config.source}")
    
    def _download_huggingface(self, model: ModelConfig) -> Path:
        """从 HuggingFace 下载"""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            print("需要安装 huggingface_hub: pip install huggingface_hub")
            raise
        
        print(f"从 HuggingFace 下载: {model.name}")
        print(f"仓库: {model.repo_id}")
        
        # 获取文件列表
        files = list_repo_files(model.repo_id)
        print(f"仓库文件: {files[:5]}...")
        
        # 查找匹配的文件
        target_file = None
        for f in files:
            if model.filename and model.filename.replace("*", "") in f:
                if any(q in f for q in ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "FP16"]):
                    target_file = f
                    break
        
        if not target_file:
            # 使用默认的 Q4_K_M 文件
            for f in files:
                if "Q4_K_M" in f:
                    target_file = f
                    break
        
        if not target_file:
            raise ValueError(f"未找到合适的 GGUF 文件: {model.repo_id}")
        
        print(f"下载文件: {target_file}")
        
        output_path = hf_hub_download(
            repo_id=model.repo_id,
            filename=target_file,
            local_dir=self.model_dir / model.id,
            local_dir_use_symlinks=False
        )
        
        return Path(output_path)
    
    def _download_modelscope(self, model: ModelConfig) -> Path:
        """从 ModelScope 下载"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            print("需要安装 modelscope: pip install modelscope")
            raise
        
        print(f"从 ModelScope 下载: {model.name}")
        
        cache_dir = str(self.model_dir / model.id)
        output_path = snapshot_download(
            model_id=model.repo_id,
            cache_dir=cache_dir
        )
        
        return Path(output_path)
    
    def _download_civitai(self, model: ModelConfig) -> Path:
        """从 Civitai 下载"""
        # Civitai 需要 API key 或者直接下载
        print(f"从 Civitai 下载: {model.name}")
        print(f"模型ID: {model.repo_id}")
        print("提示: Civitai 模型较大，建议使用 HuggingFace 镜像版本")
        
        # 提示用户手动下载
        print(f"请访问 https://civitai.com/models/{model.repo_id} 下载")
        raise NotImplementedError("Civitai 下载需要手动操作，请使用 HuggingFace 版本")
    
    def _download_ollama(self, model: ModelConfig) -> Path:
        """通过 Ollama 下载"""
        print(f"通过 Ollama 下载: {model.name}")
        
        # 检查 Ollama 是否安装
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Ollama 未安装，请先安装: https://ollama.com/download")
            raise
        
        # 提取模型名称（去掉 :latest）
        ollama_model = model.repo_id.replace(":latest", "")
        
        print(f"执行: ollama pull {ollama_model}")
        result = subprocess.run(
            ["ollama", "pull", ollama_model],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Ollama 下载失败: {result.stderr}")
        
        # 返回 Ollama 模型路径
        return Path(f"ollama://{ollama_model}")
    
    def list_installed(self) -> List[Dict]:
        """列出已安装的模型"""
        installed = []
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                files = list(model_dir.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                # 尝试匹配模型配置
                model_config = ModelCatalog.get_all_models().get(model_dir.name)
                
                installed.append({
                    "id": model_dir.name,
                    "path": str(model_dir),
                    "size_mb": round(total_size / 1024 / 1024, 2),
                    "files": len(files),
                    "config": model_config.to_dict() if model_config else None
                })
        return installed
    
    def remove(self, model_id: str) -> bool:
        """删除已安装的模型"""
        model_path = self.model_dir / model_id
        if model_path.exists():
            shutil.rmtree(model_path)
            print(f"已删除: {model_id}")
            return True
        return False


def list_popular_models() -> List[Dict]:
    """列出热门模型（用于 API）"""
    models = ModelCatalog.get_all_models()
    return [
        {
            **m.to_dict(),
            "source_name": m.source.value,
            "size_gb": round(m.size_mb / 1024, 2)
        }
        for m in list(models.values())[:20]  # 返回前20个
    ]


def get_model_info(model_id: str) -> Optional[Dict]:
    """获取模型信息"""
    model = ModelCatalog.get_all_models().get(model_id)
    if model:
        return {
            **model.to_dict(),
            "source_name": model.source.value,
            "size_gb": round(model.size_mb / 1024, 2)
        }
    return None


def get_system_info() -> Dict:
    """获取系统信息"""
    import psutil
    
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('C:\\' if os.name == 'nt' else '/')
    
    return {
        "memory_total_gb": round(memory.total / 1024**3, 1),
        "memory_available_gb": round(memory.available / 1024**3, 1),
        "memory_used_percent": memory.percent,
        "disk_total_gb": round(disk.total / 1024**3, 1),
        "disk_available_gb": round(disk.free / 1024**3, 1),
        "cpu_count": psutil.cpu_count(),
    }


def get_recommended_models(sys_info: Dict) -> List[Dict]:
    """根据系统配置推荐模型"""
    available_mem = sys_info.get("memory_available_gb", 8)
    recommended = []
    
    for model in ModelCatalog.get_all_models().values():
        if not model.image_model:  # 只推荐 LLM
            mem_required = model.min_memory_mb / 1024
            if available_mem >= mem_required:
                recommended.append({
                    "model": model,
                    "reason": f"需要 {mem_required:.0f}GB 内存，可用 {available_mem:.1f}GB"
                })
    
    # 按内存需求排序
    recommended.sort(key=lambda x: x["model"].min_memory_mb)
    return recommended


# ============= CLI =============
def main():
    parser = argparse.ArgumentParser(description="LightLLM 模型管理工具")
    parser.add_argument("action", choices=["list", "install", "remove", "info", "search", "installed", "recommend"],
                        help="操作: list(列表) / install(安装) / remove(删除) / info(详情) / search(搜索) / installed(已安装) / recommend(推荐)")
    parser.add_argument("--model", "-m", help="模型ID")
    parser.add_argument("--source", "-s", help="模型来源 (hf/ms/cs/ollama)")
    parser.add_argument("--list-sources", action="store_true", help="列出支持的来源")
    parser.add_argument("--list-all", action="store_true", help="列出所有模型(不含搜索)")
    
    args = parser.parse_args()
    
    if args.list_sources:
        print("\n支持的模型来源:")
        for source in ModelSource:
            print(f"  {source.value}: {source.name}")
        return 0
    
    if args.action == "list":
        if args.list_all:
            # 列出所有模型
            print("\n📦 所有可用模型:\n")
            models = ModelCatalog.get_all_models()
            for m in models.values():
                source_icon = {
                    ModelSource.HUGGINGFACE: "🤗",
                    ModelSource.MODELSCOPE: "📦",
                    ModelSource.CIVITAI: "🎨",
                    ModelSource.OLLAMA: "🦙",
                    ModelSource.PIXART: "🖼️",
                    ModelSource.LOCAL: "💾",
                }.get(m.source, "📦")
                print(f"{source_icon} {m.name} ({m.id})")
                print(f"   描述: {m.description}")
                print(f"   大小: {m.size_mb/1024:.1f} GB | 最低内存: {m.min_memory_mb//1024}GB")
                print()
        else:
            # 按来源列出
            source_filter = None
            if args.source:
                source_map = {"hf": ModelSource.HUGGINGFACE, "ms": ModelSource.MODELSCOPE,
                              "cs": ModelSource.CIVITAI, "ollama": ModelSource.OLLAMA}
                source_filter = source_map.get(args.source)
            
            print(f"\n📦 模型目录{' (来源: ' + args.source + ')' if args.source else ''}:\n")
            models = ModelCatalog.get_all_models() if not source_filter else ModelCatalog.get_by_source(source_filter)
            for m in models.values():
                source_icon = {
                    ModelSource.HUGGINGFACE: "🤗",
                    ModelSource.MODELSCOPE: "📦",
                    ModelSource.CIVITAI: "🎨",
                    ModelSource.OLLAMA: "🦙",
                    ModelSource.PIXART: "🖼️",
                    ModelSource.LOCAL: "💾",
                }.get(m.source, "📦")
                print(f"{source_icon} {m.name}")
                print(f"   ID: {m.id} | 大小: {m.size_mb/1024:.1f} GB")
                print()
    
    elif args.action == "search":
        query = args.model or input("输入搜索关键词: ")
        results = ModelCatalog.search(query)
        print(f"\n🔍 搜索 '{query}' 结果 ({len(results)} 个):\n")
        for m in results:
            print(f"  • {m.name} ({m.id})")
            print(f"    {m.description}\n")
    
    elif args.action == "info":
        if not args.model:
            print("请指定模型ID: --model <id>")
            return 1
        info = get_model_info(args.model)
        if info:
            print(f"\n📋 模型详情: {info['name']}")
            print(f"   ID: {info['id']}")
            print(f"   来源: {info['source_name']}")
            print(f"   大小: {info['size_gb']} GB")
            print(f"   最低内存: {info['min_memory_mb']//1024} GB")
            print(f"   推荐内存: {info['recommended_memory_mb']//1024} GB")
            print(f"   描述: {info['description']}")
            print(f"   标签: {', '.join(info['tags'])}")
        else:
            print(f"未找到模型: {args.model}")
    
    elif args.action == "installed":
        downloader = ModelDownloader()
        installed = downloader.list_installed()
        print(f"\n📁 已安装模型 ({len(installed)} 个):\n")
        if not installed:
            print("  暂无已安装的模型")
        for m in installed:
            size_gb = m['size_mb'] / 1024
            print(f"  • {m['id']} ({size_gb:.2f} GB)")
            print(f"    路径: {m['path']}")
            print()
    
    elif args.action == "install":
        if not args.model:
            print("请指定模型ID: --model <id>")
            return 1
        
        models = ModelCatalog.get_all_models()
        if args.model not in models:
            print(f"未找到模型: {args.model}")
            print("使用 'python -m src.model_manager list --list-all' 查看所有模型")
            return 1
        
        model_config = models[args.model]
        print(f"\n🚀 开始安装: {model_config.name}")
        print(f"   来源: {model_config.source.value}")
        print(f"   大小: {model_config.size_mb/1024:.1f} GB\n")
        
        downloader = ModelDownloader()
        try:
            path = downloader.download(model_config)
            print(f"\n✅ 安装完成: {path}")
        except Exception as e:
            print(f"\n❌ 安装失败: {e}")
            return 1
    
    elif args.action == "remove":
        if not args.model:
            print("请指定模型ID: --model <id>")
            return 1
        
        confirm = input(f"确认删除模型 '{args.model}'? (y/N): ")
        if confirm.lower() == 'y':
            downloader = ModelDownloader()
            if downloader.remove(args.model):
                print("✅ 删除成功")
            else:
                print("❌ 删除失败，模型可能不存在")
    
    elif args.action == "recommend":
        sys_info = get_system_info()
        print(f"\n💻 系统配置:")
        print(f"   内存: {sys_info['memory_available_gb']:.1f} GB / {sys_info['memory_total_gb']:.1f} GB")
        print(f"   CPU: {sys_info['cpu_count']} 核心\n")
        
        recommendations = get_recommended_models(sys_info)
        print("📋 推荐的模型:\n")
        for i, rec in enumerate(recommendations[:5], 1):
            m = rec["model"]
            print(f"  {i}. {m.name}")
            print(f"     {rec['reason']}")
            print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())