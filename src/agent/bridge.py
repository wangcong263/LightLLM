"""Agent Bridge - Agent communication protocol"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


class AgentProtocol(Enum):
    """Agent communication protocol types"""
    OPENCLAW = auto()
    NATIVE = auto()
    REST = auto()
    WEBSOCKET = auto()


@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    protocol: AgentProtocol = AgentProtocol.OPENCLAW
    endpoint: Optional[str] = None
    timeout: int = 30
    metadata: Optional[dict[str, str]] = None


class AgentBridge:
    """Agent communication bridge"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config
        self.protocol = config.protocol if config else AgentProtocol.OPENCLAW
        self.connected = False

    def connect(self) -> bool:
        """Connect to agent"""
        self.connected = True
        return True

    def disconnect(self):
        """Disconnect from agent"""
        self.connected = False

    def send_message(self, message: str) -> dict[str, Any]:
        """Send message to agent"""
        return {"status": "sent", "message": message}

    def receive_message(self) -> Optional[dict[str, Any]]:
        """Receive message from agent"""
        return None


def create_openclaw_bridge(
    name: str,
    endpoint: str,
    timeout: int = 30
) -> AgentBridge:
    """Create OpenClaw protocol bridge"""
    config = AgentConfig(
        name=name,
        protocol=AgentProtocol.OPENCLAW,
        endpoint=endpoint,
        timeout=timeout
    )
    return AgentBridge(config=config)
