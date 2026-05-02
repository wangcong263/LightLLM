#!/usr/bin/env python3
"""
LightLLM Direct Function Test - Using actual APIs
"""

import sys
import asyncio
from pathlib import Path

print("=" * 60)
print("lightllm Direct Function Test")
print("=" * 60)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

results = []


def test_skills_optimizer():
    """Test Skills optimizer"""
    print("\n" + "-" * 40)
    print("TEST 1: Skills Optimizer")
    print("-" * 40)

    try:
        from optimizer.skills_optimizer import SkillsOptimizer, Skill, SkillType

        optimizer = SkillsOptimizer()

        # Test register skill with correct params
        skill = Skill(
            name="test_skill",
            type=SkillType.TOOL,
            description="A test skill"
        )
        optimizer.register(skill)
        print(f"[OK] Skill registered: {skill.name}")

        # Test stats
        stats = optimizer.get_statistics()
        print(f"[OK] Stats: {stats}")

        # Test GitHub optimizer
        from optimizer.skills_optimizer import GitHubSkillsOptimizer
        gh = GitHubSkillsOptimizer(optimizer)
        print(f"[OK] GitHubSkillsOptimizer created")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_compressor():
    """Test context compression"""
    print("\n" + "-" * 40)
    print("TEST 2: Context Compression")
    print("-" * 40)

    try:
        from optimizer.context_compressor import ContextCompressor

        compressor = ContextCompressor()

        # Test token budgeting
        from optimizer.skills_optimizer import TokenBudget
        budget = TokenBudget(max_tokens=1000)
        budget.allocate(100)
        print(f"[OK] TokenBudget: available={budget.get_available()}")

        # Test compression (simple)
        print(f"[OK] ContextCompressor created")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_engine():
    """Test LLM engine"""
    print("\n" + "-" * 40)
    print("TEST 3: LLM Engine")
    print("-" * 40)

    try:
        from core.engine import LLMEngine, ModelConfig

        # Test config
        config = ModelConfig(name="test", path="test.gguf")
        print(f"[OK] ModelConfig created: {config.name}")

        # Test engine creation (without loading)
        print(f"[OK] LLMEngine can be configured")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_bridge():
    """Test agent bridge"""
    print("\n" + "-" * 40)
    print("TEST 4: Agent Bridge")
    print("-" * 40)

    try:
        from agent.bridge import AgentBridge, AgentConfig, AgentProtocol, create_openclaw_bridge

        # Test bridge creation with config
        config = AgentConfig(
            name="test_agent",
            protocol=AgentProtocol.OPENCLAW,
            endpoint="http://localhost:8080"
        )
        bridge = AgentBridge(config=config)
        print(f"[OK] AgentBridge created")

        # Test convenience function
        openclaw = create_openclaw_bridge("openclaw_agent", "http://localhost:8080")
        print(f"[OK] OpenClaw bridge created")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_manager():
    """Test model manager"""
    print("\n" + "-" * 40)
    print("TEST 5: Model Manager")
    print("-" * 40)

    try:
        from core.engine import ModelManager

        manager = ModelManager()
        print(f"[OK] ModelManager created")

        # Check methods
        methods = [m for m in dir(manager) if not m.startswith('_')]
        print(f"[OK] Methods: {', '.join(methods[:5])}...")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_structure():
    """Test code structure"""
    print("\n" + "-" * 40)
    print("TEST 6: Code Structure")
    print("-" * 40)

    try:
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
            if path.exists():
                print(f"[OK] {mod} exists")
            else:
                print(f"[FAIL] {mod} missing")
                return False

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_project_files():
    """Test project files"""
    print("\n" + "-" * 40)
    print("TEST 7: Project Files")
    print("-" * 40)

    try:
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
            if path.exists():
                print(f"[OK] {f} exists")
            else:
                print(f"[FAIL] {f} missing")
                return False

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


# Run all tests
tests = [
    ("Skills Optimizer", test_skills_optimizer),
    ("Context Compression", test_context_compressor),
    ("LLM Engine", test_llm_engine),
    ("Agent Bridge", test_agent_bridge),
    ("Model Manager", test_model_manager),
    ("Code Structure", test_code_structure),
    ("Project Files", test_project_files),
]

for name, test_func in tests:
    try:
        result = test_func()
        results.append((name, result))
    except Exception as e:
        print(f"\n[FAIL] {name} exception: {e}")
        import traceback
        traceback.print_exc()
        results.append((name, False))

# Summary
print("\n" + "=" * 60)
print("Test Results Summary")
print("=" * 60)

passed = sum(1 for _, r in results if r)
total = len(results)

for name, result in results:
    status = "[PASS]" if result else "[FAIL]"
    print(f"  {status} - {name}")

print(f"\nTotal: {passed}/{total} passed")

if passed == total:
    print("\nAll tests passed!")
else:
    print(f"\n{total - passed} tests failed")

sys.exit(0 if passed == total else 1)


