#!/usr/bin/env python3
"""
LightLLM API 服务器
提供 OpenAI 兼容的 REST API
"""
import asyncio
import json
import time
import logging
import argparse
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# 导入 LightLLM 核心
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.engine import LLMEngine, ModelManager, GenerationConfig, BackendType
from config import MODEL_CACHE_DIR, PRESET_MODELS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== API Models ==============

class Message(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """Chat Completion 请求"""
    model: str = Field(default="default", description="模型名称")
    messages: List[Message] = Field(..., description="消息列表")
    temperature: float = Field(default=0.7, ge=0, le=2, description="温度参数")
    top_p: float = Field(default=0.9, ge=0, le=1, description="Top-p 参数")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="最大生成长度")
    stream: bool = Field(default=True, description="是否流式输出")
    stop: Optional[List[str]] = Field(default=None, description="停止词")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "phi-2",
                "messages": [
                    {"role": "system", "content": "你是智能助手"},
                    {"role": "user", "content": "你好"}
                ],
                "stream": True
            }
        }


class CompletionRequest(BaseModel):
    """Completion 请求"""
    model: str = Field(default="default")
    prompt: str = Field(..., description="提示词")
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = Field(default=True)
    stop: Optional[List[str]] = None


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    owned_by: str = "lightllm"
    permission: List = field(default_factory=list)


# ============== 全局状态 ==============

model_manager: Optional[ModelManager] = None
current_engine: Optional[LLMEngine] = None
app_state = {
    "loaded": False,
    "model_name": None,
    "model_path": None,
}


# ============== FastAPI 应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("LightLLM API Server starting...")
    yield
    logger.info("LightLLM API Server shutting down...")


app = FastAPI(
    title="LightLLM API",
    description="轻量化本地 LLM 推理 API - OpenAI 兼容",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== API 路由 ==============

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": "LightLLM API",
        "version": "1.0.0",
        "status": "running" if app_state["loaded"] else "no_model_loaded",
        "model": app_state["model_name"],
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    models = []
    for name, info in PRESET_MODELS.items():
        models.append(ModelInfo(
            id=name,
            owned_by=info.get("repo", "unknown"),
        ))
    
    # 添加已加载的自定义模型
    if current_engine and current_engine.model_path:
        models.append(ModelInfo(
            id="current",
            owned_by="custom",
        ))
    
    return {
        "object": "list",
        "data": [m.model_dump() for m in models]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat Completion API - OpenAI 兼容"""
    global current_engine
    
    if not app_state["loaded"] or not current_engine:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load first.")
    
    # 提取 system 和 user 消息
    system_prompt = None
    user_prompt = None
    
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_prompt = msg.content
    
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")
    
    generation_config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop or ["</s>", "<|endoftext|>"],
        stream=request.stream,
    )
    
    if request.stream:
        return StreamingResponse(
            _stream_chat_response(current_engine, user_prompt, system_prompt, generation_config, request.model),
            media_type="text/event-stream",
        )
    else:
        # 非流式
        full_response = ""
        async for result in current_engine.generate(user_prompt, system_prompt, generation_config):
            if result.content:
                full_response += result.content
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(full_response) // 4,
                "total_tokens": 0,
            }
        }


async def _stream_chat_response(
    engine: LLMEngine,
    prompt: str,
    system: Optional[str],
    config: GenerationConfig,
    model_id: str
) -> AsyncIterator[str]:
    """流式聊天响应"""
    response_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    first_chunk = True
    
    yield f"event: chunk\n"
    
    async for result in engine.generate(prompt, system, config):
        if first_chunk:
            # 首块包含完整的响应头
            chunk_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": result.content},
                    "finish_reason": None,
                }]
            }
            first_chunk = False
        else:
            chunk_data = {
                "id": response_id,
                "object": "chat.completion.chunk", 
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": result.content},
                    "finish_reason": None,
                }]
            }
        
        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        
        if result.done:
            # 发送结束信号
            end_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }]
            }
            yield f"data: {json.dumps(end_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            break


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Completion API - OpenAI 兼容"""
    global current_engine
    
    if not app_state["loaded"] or not current_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    generation_config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        stream=request.stream,
    )
    
    if request.stream:
        return StreamingResponse(
            _stream_completion_response(current_engine, request.prompt, generation_config),
            media_type="text/event-stream",
        )
    else:
        full_response = ""
        async for result in current_engine.generate(request.prompt, None, generation_config):
            if result.content:
                full_response += result.content
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": full_response,
                "index": 0,
                "finish_reason": "stop",
            }],
        }


async def _stream_completion_response(
    engine: LLMEngine,
    prompt: str,
    config: GenerationConfig
) -> AsyncIterator[str]:
    """流式补全响应"""
    async for result in engine.generate(prompt, None, config):
        if result.done:
            yield "data: [DONE]\n\n"
            break
        
        chunk = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [{
                "text": result.content,
                "index": 0,
                "finish_reason": None,
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"


@app.post("/load")
async def load_model(
    model: str = "phi-2",
    backend: str = "llama_cpp"
):
    """加载模型"""
    global model_manager, current_engine
    
    logger.info(f"Loading model: {model} (backend: {backend})")
    
    # 确定模型路径
    model_path = None
    
    # 检查预设模型
    if model in PRESET_MODELS:
        preset = PRESET_MODELS[model]
        model_dir = MODEL_CACHE_DIR / model
        from glob import glob
        matches = glob(str(model_dir / "*.gguf"))
        if matches:
            model_path = matches[0]
    
    # 检查自定义路径
    if not model_path and Path(model).exists():
        model_path = model
    
    if not model_path:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model}. Use /download to download first."
        )
    
    # 创建引擎
    backend_type = BackendType(backend)
    from config import create_backend_config
    
    config = create_backend_config(backend_type)
    
    engine = LLMEngine(
        model_path=model_path,
        backend=backend_type,
        config=config,
    )
    
    # 加载
    success = await engine.load()
    
    if success:
        current_engine = engine
        app_state["loaded"] = True
        app_state["model_name"] = model
        app_state["model_path"] = model_path
        
        return {
            "status": "success",
            "model": model,
            "path": model_path,
            "backend": backend,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.post("/unload")
async def unload_model():
    """卸载模型"""
    global current_engine
    
    if current_engine:
        await current_engine.unload()
        current_engine = None
        app_state["loaded"] = False
        app_state["model_name"] = None
        app_state["model_path"] = None
    
    return {"status": "success", "message": "Model unloaded"}


@app.get("/status")
async def status():
    """获取状态"""
    return {
        "loaded": app_state["loaded"],
        "model": app_state["model_name"],
        "path": app_state["model_path"],
    }


@app.post("/download")
async def download_model(model: str = "phi-2"):
    """下载模型"""
    if model not in PRESET_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    
    preset = PRESET_MODELS[model]
    
    # 异步下载
    asyncio.create_task(_download_model_async(model, preset))
    
    return {
        "status": "downloading",
        "model": model,
        "message": f"Downloading {preset['name']} (~{preset['size_mb']}MB)",
    }


async def _download_model_async(model: str, preset: Dict):
    """异步下载模型"""
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    
    logger.info(f"Starting download: {preset['name']}")
    
    try:
        model_dir = MODEL_CACHE_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = hf_hub_download(
            repo_id=preset["repo"],
            filename=preset["file"],
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        
        logger.info(f"Download complete: {local_path}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")


@app.get("/presets")
async def list_presets():
    """列出预设模型"""
    return {
        "models": PRESET_MODELS
    }


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="LightLLM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model", default=None, help="Model to load on startup")
    parser.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "vllm", "ctransformers"])
    
    args = parser.parse_args()
    
    # 预加载模型
    if args.model:
        asyncio.run(load_model(args.model, args.backend))
    
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()