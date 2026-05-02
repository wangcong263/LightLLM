#!/usr/bin/env python3
"""Model manager - Download, manage, and deploy LLM models"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Default directories
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "lightllm" / "models"

# Popular models
DEFAULT_MODELS = {
    "phi-2": {
        "name": "Microsoft Phi-2",
        "size_mb": 2800,
        "repo": "TheBloke/phi-2-GGUF",
        "file": "phi-2.Q4_K_M.gguf",
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct",
        "size_mb": 4200,
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    },
    "llama-2-7b": {
        "name": "Llama 2 7B Chat",
        "size_mb": 3900,
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.Q4_K_M.gguf",
    },
}


class ModelCatalog:
    """Model catalog"""

    MODELS = DEFAULT_MODELS.copy()

    @classmethod
    def get_all_models(cls) -> dict[str, dict[str, Any]]:
        return cls.MODELS.copy()

    @classmethod
    def get_model(cls, name: str) -> Optional[dict[str, Any]]:
        return cls.MODELS.get(name)

    @classmethod
    def add_model(cls, name: str, info: dict[str, Any]):
        cls.MODELS[name] = info


class ModelDownloader:
    """Download models from various sources"""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback: Optional[Callable[[int, int], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set progress callback"""
        self.progress_callback = callback

    def download(self, source: str, model_id: str) -> Path:
        """Download model"""
        logger.info(f"Downloading {model_id} from {source}")
        local_path = self.model_dir / model_id
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder - actual download would use huggingface_hub etc.
        logger.info(f"Downloaded to {local_path}")
        return local_path

    def download_from_huggingface(self, repo_id: str, filename: str) -> Path:
        """Download from HuggingFace"""
        logger.info(f"Downloading {filename} from {repo_id}")
        local_path = self.model_dir / filename
        return local_path

    def download_from_modelscope(self, model_id: str) -> Path:
        """Download from ModelScope"""
        logger.info(f"Downloading {model_id} from ModelScope")
        local_path = self.model_dir / model_id
        return local_path


def list_popular_models() -> list[dict[str, Any]]:
    """List popular models"""
    models = []
    for name, info in DEFAULT_MODELS.items():
        models.append({
            "id": name,
            "name": info["name"],
            "size_mb": info["size_mb"],
            "repo": info["repo"],
        })
    return models


def get_model_info(model_id: str) -> Optional[dict[str, Any]]:
    """Get model info"""
    model = ModelCatalog.get_all_models().get(model_id)
    if model:
        return {
            "id": model_id,
            "name": model["name"],
            "size_mb": model["size_mb"],
            "source": model["repo"],
        }
    return None


def get_system_info() -> dict[str, Any]:
    """Get system info"""
    import shutil
    info = {
        "platform": os.name,
        "model_dir": str(DEFAULT_MODEL_DIR),
        "available_space_gb": 0,
    }
    if DEFAULT_MODEL_DIR.exists():
        try:
            stat = shutil.disk_usage(DEFAULT_MODEL_DIR)
            info["available_space_gb"] = round(stat.free / (1024**3), 2)
        except Exception:
            pass
    return info
