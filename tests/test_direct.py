#!/usr/bin/env python3
"""
LightLLM Direct Function Test - Using pytest
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_skills_optimizer():
    """Test Skills optimizer"""
    from optimizer.skills_optimizer import SkillsOptimizer, Skill, SkillType

    optimizer = SkillsOptimizer()

    # Test register skill with correct params
    skill = Skill(
        name="test_skill",
        type=SkillType.TOOL,
        description="A test skill"
    )
    optimizer.register(skill)

    # Test stats
    stats = optimizer.get_statistics()
    assert "total_calls" in stats

    # Test GitHub optimizer
    from optimizer.skills_optimizer import GitHubSkillsOptimizer
    gh = GitHubSkillsOptimizer(optimizer)
    assert gh is not None


def test_context_compressor():
    """Test context compression"""
    from optimizer.context_compressor import ContextCompressor

    compressor = ContextCompressor()
    assert compressor is not None

    # Test token budgeting
    from optimizer.skills_optimizer import TokenBudget
    budget = TokenBudget(max_tokens=1000)
    budget.allocate(100)
    assert budget.get_available() == 900


def test_llm_engine():
    """Test LLM engine"""
    from core.engine import LLMEngine, ModelConfig

    # Test config
    config = ModelConfig(name="test", path="test.gguf")
    assert config.name == "test"
    assert config.path == "test.gguf"

    # Test engine creation (without loading)
    assert LLMEngine is not None


def test_agent_bridge():
    """Test agent bridge"""
    from agent.bridge import AgentBridge, AgentConfig, AgentProtocol, create_openclaw_bridge

    # Test bridge creation with config
    config = AgentConfig(
        name="test_agent",
        protocol=AgentProtocol.OPENCLAW,
        endpoint="http://localhost:8080"
    )
    bridge = AgentBridge(config=config)
    assert bridge is not None

    # Test convenience function
    openclaw = create_openclaw_bridge("openclaw_agent", "http://localhost:8080")
    assert openclaw is not None


def test_model_manager():
    """Test model manager"""
    from core.engine import ModelManager

    manager = ModelManager()
    assert manager is not None

    # Check methods exist
    methods = [m for m in dir(manager) if not m.startswith('_')]
    assert len(methods) > 0


def test_code_structure():
    """Test code structure"""
    base = Path(__file__).parent.parent / "src"

    # Check all modules exist
    modules = [
        "core/engine.py",
        "api/server.py",
        "agent/bridge.py",
        "optimizer/skills_optimizer.py",
        "optimizer/context_compressor.py",
        "cli.py"
    ]

    for mod in modules:
        path = base / mod
        assert path.exists(), f"{mod} should exist"


def test_project_files():
    """Test project files"""
    base = Path(__file__).parent.parent

    # Check project files
    files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        ".gitignore"
    ]

    for f in files:
        path = base / f
        assert path.exists(), f"{f} should exist"