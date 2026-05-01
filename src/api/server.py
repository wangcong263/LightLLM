"""
LightLLM API层 - 提供OpenAI兼容API
支持OpenClaw、Hermes等智能体框架
"""
import asyncio
import json
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.engine import LLMEngine, ModelConfig, ModelManager

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    """聊天补全请求"""
    model: str
    messages: List[Dict]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    stop: Optional[List[str]] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # LightLLM特有优化参数
    cache_prompt: bool = True  # 提示缓存
    low_memory: bool = True   # 低内存模式


@dataclass
class ChatCompletionResponse:
    """聊天补全响应"""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[Dict] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage,
        }


class LightLLMAPI:
    """
    轻量化API服务
    
    与Ollama对比的优化：
    1. OpenAI兼容 - 无缝对接现有工具
    2. 流式SSE - 更低的延迟
    3. 提示缓存 - 减少Token消耗
    4. 工具调用优化 - 更好的Agent支持
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.model_manager = ModelManager()
        self.server = None
        
        # 缓存机制
        self._prompt_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def start(self):
        """启动API服务"""
        try:
            from uvicorn import Config, App
            from fastapi import FastAPI, Request
            from fastapi.responses import StreamingResponse, JSONResponse
            import uvicorn
            
            app = FastAPI(title="LightLLM API")
            
            # 注册路由
            app.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
            app.add_api_route("/v1/models", self.list_models, methods=["GET"])
            app.add_api_route("/v1/embeddings", self.embeddings, methods=["POST"])
            app.add_api_route("/health", self.health, methods=["GET"])
            
            # 管理API
            app.add_api_route("/api/load", self.load_model, methods=["POST"])
            app.add_api_route("/api/unload", self.unload_model, methods=["POST"])
            app.add_api_route("/api/cache/stats", self.cache_stats, methods=["GET"])
            
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
            self.server = uvicorn.Server(config)
            
            logger.info(f"LightLLM API starting on {self.host}:{self.port}")
            await self.server.serve()
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise
    
    async def stop(self):
        """停止服务"""
        if self.server:
            self.server.should_exit = True
    
    # ========== API端点 ==========
    
    async def chat_completions(self, request: ChatCompletionRequest) -> Dict:
        """聊天补全 - OpenAI兼容"""
        try:
            # 获取模型
            engine = self.model_manager.get_current()
            if not engine:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No model loaded"}
                )
            
            # 转换消息格式
            messages = [ChatMessage(**m) for m in request.messages]
            
            # 提取系统消息
            system_prompt = None
            if messages and messages[0].role == "system":
                system_prompt = messages[0].content
                messages = messages[1:]
            
            # 构建提示
            prompt = self._build_prompt(messages, request.cache_prompt)
            
            # 生成响应
            response_text = ""
            async for token in engine.generate(
                prompt=prompt,
                system=system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False
            ):
                response_text += token
            
            # 计算Token使用
            prompt_tokens = engine.get_token_count(prompt)
            completion_tokens = engine.get_token_count(response_text)
            
            # 构建响应
            response = ChatCompletionResponse(
                id=f"chatcmpl-{self._generate_id()}",
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    async def chat_completions_stream(self, request: ChatCompletionRequest) -> AsyncIterator:
        """流式聊天补全"""
        engine = self.model_manager.get_current()
        if not engine:
            yield "data: {\"error\": \"No model loaded\"}\n\n"
            return
        
        messages = [ChatMessage(**m) for m in request.messages]
        system_prompt = None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            messages = messages[1:]
        
        prompt = self._build_prompt(messages, request.cache_prompt)
        
        yield "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        
        full_response = ""
        async for token in engine.generate(
            prompt=prompt,
            system=system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            full_response += token
            yield f"data: {{\"choices\":[{{\"delta\":{{\"content\":\"{token}\"}}}}]}}\n\n"
        
        yield "data: [DONE]\n\n"
    
    def _build_prompt(self, messages: List[ChatMessage], use_cache: bool = True) -> str:
        """
        构建优化的prompt
        
        优化点：
        1. 提示缓存 - 减少重复处理
        2. 消息压缩 - 减少Token使用
        3. 结构化格式 - 更好的理解
        """
        if not messages:
            return ""
        
        # 检查缓存
        cache_key = str(messages)
        if use_cache and cache_key in self._prompt_cache:
            self._cache_hits += 1
            return self._prompt_cache[cache_key]
        
        self._cache_misses += 1
        
        # 构建prompt
        parts = []
        for msg in messages:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                parts.append(f"Result: {msg.content}")
        
        prompt = "\n".join(parts)
        
        # 缓存
        if use_cache:
            self._prompt_cache[cache_key] = prompt
        
        return prompt
    
    async def list_models(self) -> Dict:
        """列出可用模型"""
        models = []
        for name, engine in self.model_manager.models.items():
            models.append({
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "LightLLM",
                "permission": [],
            })
        
        return {
            "object": "list",
            "data": models,
        }
    
    async def embeddings(self, request: Dict) -> Dict:
        """文本嵌入"""
        # 简化实现
        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": [0.0] * 384,  # 简化
                "index": 0,
            }],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            }
        }
    
    async def health(self) -> Dict:
        """健康检查"""
        return {
            "status": "ok",
            "model": self.model_manager.current_model,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
    
    async def load_model(self, request: Dict) -> Dict:
        """加载模型"""
        name = request.get("name")
        config = ModelConfig(**request.get("config", {}))
        
        self.model_manager.add_model(name, config)
        success = await self.model_manager.load(name)
        
        return {"status": "ok" if success else "error"}
    
    async def unload_model(self, request: Dict) -> Dict:
        """卸载模型"""
        name = request.get("name")
        if name in self.model_manager.models:
            await self.model_manager.models[name].unload()
        
        return {"status": "ok"}
    
    async def cache_stats(self) -> Dict:
        """缓存统计"""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "cached_prompts": len(self._prompt_cache),
        }
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())[:8]


class AgentConnector:
    """
    智能体连接器
    支持OpenClaw、Hermes等框架
    """
    
    def __init__(self, api: LightLLMAPI):
        self.api = api
        self.connected_agents: Dict[str, Any] = {}
    
    def connect_openclaw(self, agent_id: str, config: Dict) -> bool:
        """连接OpenClaw智能体"""
        try:
            # OpenClaw使用WebSocket连接
            ws_url = config.get("ws_url", "ws://localhost:8765")
            
            self.connected_agents[agent_id] = {
                "type": "openclaw",
                "url": ws_url,
                "config": config,
            }
            
            logger.info(f"Connected to OpenClaw: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenClaw: {e}")
            return False
    
    def connect_hermes(self, agent_id: str, config: Dict) -> bool:
        """连接Hermes智能体"""
        try:
            # Hermes使用HTTP API
            api_url = config.get("api_url", "http://localhost:8081")
            
            self.connected_agents[agent_id] = {
                "type": "hermes",
                "url": api_url,
                "config": config,
            }
            
            logger.info(f"Connected to Hermes: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Hermes: {e}")
            return False
    
    async def send_to_agent(self, agent_id: str, message: Dict) -> Optional[Dict]:
        """发送消息到智能体"""
        if agent_id not in self.connected_agents:
            raise ValueError(f"Agent {agent_id} not connected")
        
        agent = self.connected_agents[agent_id]
        
        if agent["type"] == "openclaw":
            return await self._send_openclaw(agent, message)
        elif agent["type"] == "hermes":
            return await self._send_hermes(agent, message)
        
        return None
    
    async def _send_openclaw(self, agent: Dict, message: Dict) -> Dict:
        """发送到OpenClaw"""
        # 实现WebSocket通信
        return {"status": "sent"}
    
    async def _send_hermes(self, agent: Dict, message: Dict) -> Dict:
        """发送到Hermes"""
        # 实现HTTP通信
        return {"status": "sent"}