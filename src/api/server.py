#!/usr/bin/env python3
"""LightLLM WebUI API Server - REST API for model management and deployment"""
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports
LLMEngine = None
ModelManager = None
ModelConverter = None


def get_engine():
    global LLMEngine
    if LLMEngine is None:
        from src.core.engine import LLMEngine
    return LLMEngine


def get_manager():
    global ModelManager
    if ModelManager is None:
        from src.core.engine import ModelManager
    return ModelManager


def get_converter():
    global ModelConverter
    if ModelConverter is None:
        from src.model_converter import ModelConverter
    return ModelConverter


app = FastAPI(title="LightLLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelStatus(str, Enum):
    """Model status"""
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    UNLOADED = "unloaded"


@app.get("/")
async def root():
    return {"message": "LightLLM API Server", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/models")
async def list_models():
    """List available models"""
    manager_cls = get_manager()
    manager = manager_cls()
    return {"models": manager.list_models()}


@app.get("/api/models/{name}")
async def get_model(name: str):
    """Get model info"""
    manager_cls = get_manager()
    manager = manager_cls()
    config = manager.get(name)
    if not config:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"config": config}


@app.post("/api/models")
async def register_model(config: dict[str, Any]):
    """Register a model"""
    manager_cls = get_manager()
    from src.core.engine import BackendType, ModelConfig
    backend = BackendType(config.get("backend", "llama.cpp"))
    model_config = ModelConfig(
        name=config["name"],
        path=config["path"],
        backend=backend,
        n_ctx=config.get("n_ctx", 2048),
    )
    manager = manager_cls()
    manager.register(model_config)
    return {"status": "registered", "name": config["name"]}


@app.delete("/api/models/{name}")
async def unregister_model(name: str):
    """Unregister a model"""
    manager_cls = get_manager()
    manager = manager_cls()
    if manager.unregister(name):
        return {"status": "unregistered", "name": name}
    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/api/formats")
async def list_formats():
    """List supported model formats"""
    converter_cls = get_converter()
    return {"formats": converter_cls.list_supported_formats()}


@app.post("/api/convert")
async def convert_model(source: str, target_format: str, output: str):
    """Convert model format"""
    converter_cls = get_converter()
    converter = converter_cls()
    result = converter.convert(source, target_format, output)
    return {"status": "converted", "result": result}


@app.get("/api/system")
async def system_info():
    """Get system info"""
    converter_cls = get_converter()
    return {"info": converter_cls.get_system_info()}


def main():
    """Run API server"""
    host = os.getenv("LIGHTLLM_HOST", "0.0.0.0")
    port = int(os.getenv("LIGHTLLM_PORT", "8000"))
    logger.info(f"Starting LightLLM API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
