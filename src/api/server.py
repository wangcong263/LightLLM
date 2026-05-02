"""lightllm API Server - OpenAI 兼容 API"""
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request
from flask_cors import CORS
from sse_starlette.sse import EventSourceResponse

# 尝试导入核心模块
try:
    from src.core.engine import BackendType, LLMEngine
    from src.model_manager import ModelDownloader, get_system_info
except ImportError as e:
    print(f"Warning: {e}")
    BackendType = None
    LLMEngine = None
    ModelDownloader = None


app = Flask(__name__)
CORS(app)

# 全局引擎实例
llm_engine = None
model_manager = ModelDownloader() if ModelDownloader else None


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False
    stop: Optional[List[str]] = None


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    """列出可用模型"""
    if model_manager is None:
        return jsonify({"error": "Model manager not available"}), 500

    installed = model_manager.list_installed()
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": m.get("installed_at", 0),
                "owned_by": "local",
            }
            for m in installed
        ]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Chat Completion API - OpenAI 兼容"""
    try:
        data = request.json
        req = ChatCompletionRequest(
            model=data.get("model", "default"),
            messages=data.get("messages", []),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            max_tokens=data.get("max_tokens", 2048),
            stream=data.get("stream", False),
            stop=data.get("stop"),
        )

        if req.stream:
            return EventSourceResponse(
                generate_stream_response(req),
                media_type="text/event-stream"
            )

        return jsonify(generate_response(req))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_response(req: ChatCompletionRequest) -> Dict[str, Any]:
    """生成完整响应"""
    content = f"Echo: {req.messages[-1]['content'] if req.messages else ''}"

    return {
        "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def generate_stream_response(req: ChatCompletionRequest):
    """生成流式响应"""
    content = f"Echo: {req.messages[-1]['content'] if req.messages else ''}"

    for i, char in enumerate(content):
        chunk = {
            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None,
                }
            ],
        }
        yield {"event": "message", "data": chunk}

    # 发送最后一个 chunk
    final_chunk = {
        "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield {"event": "message", "data": final_chunk}


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """运行 API 服务器"""
    print(f"lightllm API Server starting on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
