"""Agent Bridge - Agent communication protocol"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class AgentProtocol(Enum):
    """Agent communication protocol types"""
    OPENCLAW = auto()
    JSON_RPC = auto()
    WEBSOCKET = auto()
    HTTP_STREAM = auto()


@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    protocol: AgentProtocol
    endpoint: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


class AgentBridge:
    """Bridge for agent communication"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._connected = False

    def connect(self) -> bool:
        """Connect to agent"""
        try:
            self._connected = True
            return True
        except Exception:
            return False

    def disconnect(self):
        """Disconnect from agent"""
        self._connected = False

    def send_message(self, message: str) -> Optional[str]:
        """Send message to agent"""
        if not self._connected:
            return None
        return f"Response to: {message}"

    def receive_message(self) -> Optional[str]:
        """Receive message from agent"""
        if not self._connected:
            return None
        return "Agent message"

    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected


def create_openclaw_bridge(
    endpoint: str,
    api_key: Optional[str] = None,
) -> AgentBridge:
    """Create OpenClaw agent bridge"""
    config = AgentConfig(
        name="OpenClaw",
        protocol=AgentProtocol.OPENCLAW,
        endpoint=endpoint,
        api_key=api_key,
    )
    return AgentBridge(config)