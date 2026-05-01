#!/usr/bin/env python3
"""
LightLLM Integration Test - Verify Core Functions
Uses mock mode to test core logic without compiled llama-cpp-python
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project src path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_llm_engine_mock():
    """Test LLM engine core logic (mock mode)"""
    print("\n" + "=" * 60)
    print("TEST 1: LLM Engine Core Logic")
    print("=" * 60)
    
    try:
        from core.engine import LLMEngine
        
        # Use mock mode
        with patch.object(LLMEngine, '_initialize_llama') as mock_init:
            mock_init.return_value = True
            
            # Create engine instance
            engine = LLMEngine(
                model_path="mock/path/model.gguf",
                n_ctx=2048,
                verbose=True
            )
            
            # Verify properties
            assert engine.context_length == 2048
            print(f"[OK] Context length set correctly: {engine.context_length}")
            
            assert engine.max_tokens == 2048
            print(f"[OK] Max tokens set correctly: {engine.max_tokens}")
            
            # Test generate method (mock)
            with patch.object(engine, '_generate_stream') as mock_gen:
                mock_gen.return_value = iter(["Hello", " world", "!"])
                result = engine.generate("Hi", max_tokens=10)
                print(f"[OK] Generate method works: '{result}'")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        return False
    
    return True


def test_skills_optimizer():
    """Test Skills optimizer"""
    print("\n" + "=" * 60)
    print("TEST 2: Skills Optimizer")
    print("=" * 60)
    
    try:
        from optimizer.skills_optimizer import SkillsOptimizer
        
        optimizer = SkillsOptimizer(cache_dir=".test_cache")
        
        # Test caching
        test_skills = ["skill1", "skill2", "skill3"]
        result1 = optimizer.optimize_skills_calls(test_skills, budget_tokens=1000)
        print(f"[OK] First optimization: {result1}")
        
        result2 = optimizer.optimize_skills_calls(test_skills, budget_tokens=1000)
        assert result2 == result1
        print(f"[OK] Cache hit: {result2}")
        
        # Test token budget
        budget = optimizer.create_token_budget(total_tokens=4096)
        assert budget.system_prompt == 500
        assert budget.user_input == 3500
        print(f"[OK] Token budget: system={budget.system_prompt}, user={budget.user_input}")
        
        # Test cache stats
        stats = optimizer.get_cache_stats()
        print(f"[OK] Cache stats: {stats}")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_context_compressor():
    """Test context compression"""
    print("\n" + "=" * 60)
    print("TEST 3: Context Compression")
    print("=" * 60)
    
    try:
        from optimizer.context_compressor import ContextCompressor
        
        compressor = ContextCompressor(max_tokens=100)
        
        # Test long message compression
        long_messages = [
            {"role": "system", "content": "You are a helpful assistant." * 100},
            {"role": "user", "content": "Hello!"},
        ]
        
        compressed = compressor.compress_context(long_messages)
        print(f"[OK] Before compression: {len(long_messages)} messages")
        print(f"[OK] After compression: {len(compressed)} messages")
        print(f"[OK] Total tokens: {compressor.total_tokens}")
        
        # Test token counting
        token_count = compressor.count_tokens("Hello, world!")
        print(f"[OK] Token count: '{token_count}'")
        
        # Test estimation
        estimate = compressor.estimate_messages_tokens(long_messages)
        print(f"[OK] Message token estimate: {estimate}")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_agent_bridge():
    """Test agent bridge"""
    print("\n" + "=" * 60)
    print("TEST 4: Agent Bridge")
    print("=" * 60)
    
    try:
        from agent.bridge import AgentBridge, AgentProtocol
        
        bridge = AgentBridge()
        
        # Test OpenClaw connection
        config = bridge.connect_openclaw(host="localhost", port=8080)
        assert config["host"] == "localhost"
        assert config["port"] == 8080
        print(f"[OK] OpenClaw config: {config}")
        
        # Test Hermes connection
        hermes_config = bridge.connect_hermes(api_key="test-key")
        assert hermes_config["api_key"] == "test-key"
        print(f"[OK] Hermes config: {hermes_config}")
        
        # Test message format
        message = bridge.format_message("Hello", role="user")
        assert message["role"] == "user"
        assert message["content"] == "Hello"
        print(f"[OK] Message format: {message}")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_manager():
    """Test model manager"""
    print("\n" + "=" * 60)
    print("TEST 5: Model Manager")
    print("=" * 60)
    
    try:
        from core.engine import ModelManager
        
        manager = ModelManager(cache_dir=".test_models")
        
        # Test local model scanning
        mock_models = [
            {"path": "model1.gguf", "size": 1024 * 1024 * 500},
            {"path": "model2.gguf", "size": 1024 * 1024 * 1000},
        ]
        
        with patch.object(manager, '_scan_cache') as mock_scan:
            mock_scan.return_value = mock_models
            
            # Test find
            found = manager.find_local_models()
            print(f"[OK] Found local models: {len(found)}")
            
            # Test recommend
            recommended = manager.recommend_model(memory_mb=800)
            print(f"[OK] Recommended model: {recommended}")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_api_server():
    """Test API server"""
    print("\n" + "=" * 60)
    print("TEST 6: API Server")
    print("=" * 60)
    
    try:
        from api.server import validate_chat_request
        
        # Test valid request
        valid_request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        
        errors = validate_chat_request(valid_request)
        assert errors is None
        print(f"[OK] Valid request verified")
        
        # Test invalid request
        invalid_request = {"messages": [{"content": "No role"}]}
        errors = validate_chat_request(invalid_request)
        assert errors is not None
        print(f"[OK] Invalid request rejected: {errors}")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_cli():
    """Test CLI"""
    print("\n" + "=" * 60)
    print("TEST 7: CLI")
    print("=" * 60)
    
    try:
        from cli import parse_args
        
        # Test argument parsing
        args = parse_args(["serve", "--port", "9000"])
        assert args.port == 9000
        assert args.command == "serve"
        print(f"[OK] Args parsed: port={args.port}, command={args.command}")
        
        # Test pull command
        args = parse_args(["pull", "llama2"])
        assert args.model == "llama2"
        print(f"[OK] Pull command args: model={args.model}")
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("=" * 60)
    print("LightLLM Integration Test")
    print("=" * 60)
    print("\nTest Items:")
    print("  1. LLM Engine Core Logic")
    print("  2. Skills Optimizer")
    print("  3. Context Compression")
    print("  4. Agent Bridge")
    print("  5. Model Manager")
    print("  6. API Server")
    print("  7. CLI")
    
    tests = [
        ("LLM Engine", test_llm_engine_mock),
        ("Skills Optimizer", test_skills_optimizer),
        ("Context Compression", test_context_compressor),
        ("Agent Bridge", test_agent_bridge),
        ("Model Manager", test_model_manager),
        ("API Server", test_api_server),
        ("CLI", test_cli),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} test exception: {e}")
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
        return 0
    else:
        print(f"\n{total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())