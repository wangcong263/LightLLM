"""
Agent Bridge - 智能体连接器
OpenClaw、Hermes等智能体框架的集成
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentProtocol(Enum):
    """支持的智能体协议"""
    OPENCLAW = "openclaw"
    HERMES = "hermes"
    CLAUDE = "claude"
    OPENAI = "openai"


@dataclass
class AgentConfig:
    """智能体配置"""
    protocol: AgentProtocol
    host: str
    port: int
    api_key: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    headers: Dict[str, str] = field(default_factory=dict)


class AgentBridge:
    """
    智能体连接器
    支持OpenClaw、Hermes等多种智能体框架
    """

    def __init__(self):
        self.connections: Dict[str, aiohttp.ClientSession] = {}
        self.agents: Dict[str, AgentConfig] = {}
        self._running = False
        self._message_handlers: Dict[str, List[Callable]] = {}

    def register_agent(self, name: str, config: AgentConfig) -> None:
        """
        注册智能体

        Args:
            name: 智能体名称
            config: 智能体配置
        """
        self.agents[name] = config
        logger.info(f"Registered agent: {name} ({config.protocol.value})")

    def register_message_handler(self, protocol: str, handler: Callable) -> None:
        """
        注册消息处理器

        Args:
            protocol: 协议名称
            handler: 回调函数
        """
        if protocol not in self._message_handlers:
            self._message_handlers[protocol] = []
        self._message_handlers[protocol].append(handler)

    async def connect(self, agent_name: str) -> bool:
        """
        连接到智能体

        Args:
            agent_name: 智能体名称

        Returns:
            连接是否成功
        """
        if agent_name not in self.agents:
            logger.error(f"Agent not found: {agent_name}")
            return False

        config = self.agents[agent_name]

        try:
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            headers.update(config.headers)

            timeout = aiohttp.ClientTimeout(total=config.timeout)
            session = aiohttp.ClientSession(headers=headers, timeout=timeout)

            # 测试连接
            url = f"http://{config.host}:{config.port}/health"
            async with session.get(url) as resp:
                if resp.status == 200:
                    self.connections[agent_name] = session
                    logger.info(f"Connected to agent: {agent_name}")
                    return True

        except Exception as e:
            logger.error(f"Failed to connect to {agent_name}: {e}")
            return False

        return False

    async def disconnect(self, agent_name: str) -> None:
        """断开与智能体的连接"""
        if agent_name in self.connections:
            await self.connections[agent_name].close()
            del self.connections[agent_name]
            logger.info(f"Disconnected from agent: {agent_name}")

    async def send_message(
        self,
        agent_name: str,
        message: Dict[str, Any],
        endpoint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        发送消息到智能体

        Args:
            agent_name: 智能体名称
            message: 消息内容
            endpoint: 可选的API端点

        Returns:
            响应内容
        """
        if agent_name not in self.connections:
            logger.error(f"Not connected to agent: {agent_name}")
            return None

        config = self.agents[agent_name]
        session = self.connections[agent_name]

        base_url = f"http://{config.host}:{config.port}"
        url = f"{base_url}/{endpoint}" if endpoint else f"{base_url}/chat"

        for attempt in range(config.retry_count):
            try:
                async with session.post(url, json=message) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"Request failed with status {resp.status}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))

        return None

    async def stream_message(
        self,
        agent_name: str,
        message: Dict[str, Any],
        on_chunk: Callable[[str], None]
    ) -> Optional[Dict[str, Any]]:
        """
        流式发送消息到智能体

        Args:
            agent_name: 智能体名称
            message: 消息内容
            on_chunk: 收到数据块时的回调

        Returns:
            最终响应
        """
        if agent_name not in self.connections:
            logger.error(f"Not connected to agent: {agent_name}")
            return None

        config = self.agents[agent_name]
        session = self.connections[agent_name]

        base_url = f"http://{config.host}:{config.port}"
        url = f"{base_url}/chat/stream"

        try:
            async with session.post(url, json=message) as resp:
                if resp.status == 200:
                    full_response = ""
                    async for line in resp.content:
                        if line:
                            chunk = line.decode('utf-8').strip()
                            if chunk.startswith('data:'):
                                data = json.loads(chunk[5:].strip())
                                content = data.get('content', '')
                                full_response += content
                                on_chunk(content)
                    return {"content": full_response}

        except Exception as e:
            logger.error(f"Stream failed: {e}")

        return None

    async def call_skill(
        self,
        agent_name: str,
        skill_name: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        调用智能体的Skill

        Args:
            agent_name: 智能体名称
            skill_name: Skill名称
            params: 参数

        Returns:
            执行结果
        """
        message = {
            "type": "skill_call",
            "skill": skill_name,
            "params": params
        }
        return await self.send_message(agent_name, message, endpoint="skills/execute")

    async def get_skills(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        获取智能体可用的Skills列表

        Args:
            agent_name: 智能体名称

        Returns:
            Skills列表
        """
        if agent_name not in self.connections:
            return []

        config = self.agents[agent_name]
        session = self.connections[agent_name]

        url = f"http://{config.host}:{config.port}/skills"

        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('skills', [])
        except Exception as e:
            logger.error(f"Failed to get skills: {e}")

        return []

    async def health_check(self, agent_name: str) -> bool:
        """
        检查智能体健康状态

        Args:
            agent_name: 智能体名称

        Returns:
            是否健康
        """
        if agent_name not in self.connections:
            return False

        config = self.agents[agent_name]
        session = self.connections[agent_name]

        url = f"http://{config.host}:{config.port}/health"

        try:
            async with session.get(url) as resp:
                return resp.status == 200
        except:
            return False

    async def close_all(self) -> None:
        """关闭所有连接"""
        for agent_name in list(self.connections.keys()):
            await self.disconnect(agent_name)


# 便捷函数
def create_openclaw_bridge(host: str = "localhost", port: int = 8080) -> AgentBridge:
    """创建OpenClaw连接"""
    bridge = AgentBridge()
    bridge.register_agent("openclaw", AgentConfig(
        protocol=AgentProtocol.OPENCLAW,
        host=host,
        port=port
    ))
    return bridge


def create_hermes_bridge(host: str = "localhost", port: int = 8081, api_key: str = None) -> AgentBridge:
    """创建Hermes连接"""
    bridge = AgentBridge()
    bridge.register_agent("hermes", AgentConfig(
        protocol=AgentProtocol.HERMES,
        host=host,
        port=port,
        api_key=api_key
    ))
    return bridge
