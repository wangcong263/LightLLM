#!/usr/bin/env python3
"""
使用ModelScope下载GGUF模型（国内镜像，速度更快）
"""

import os
import sys
from pathlib import Path

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("❌ 需要安装modelscope: pip install modelscope")
    sys.exit(1)

# 模型存储目录
MODELS_DIR = Path.home() / ".cache" / "lightllm" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_model(model_id: str, file_name: str):
    """下载模型"""
    print(f"📥 正在下载: {model_id}/{file_name}")
    print(f"📁 保存位置: {MODELS_DIR}")
    
    try:
        # 设置缓存目录
        cache_dir = str(MODELS_DIR)
        
        # 下载模型
        local_dir = snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir,
            revision='master'
        )
        
        # 查找下载的文件
        model_path = Path(local_dir) / file_name
        
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ 下载完成!")
            print(f"📊 文件大小: {file_size:.2f} MB")
            print(f"📍 路径: {model_path}")
            return str(model_path)
        else:
            # 列出目录内容
            print(f"⚠️ 指定的文件不存在: {file_name}")
            print(f"📂 目录内容:")
            for f in Path(local_dir).iterdir():
                if f.suffix == '.gguf':
                    file_size = f.stat().st_size / (1024 * 1024)
                    print(f"   {f.name} ({file_size:.2f} MB)")
                    return str(f)
            return None
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 可用模型列表
    models = [
        {
            "name": "Qwen2-0.5B-Instruct Q4 (推荐测试)",
            "model_id": "qwen/Qwen2-0.5B-Instruct-GGUF",
            "file_name": "qwen2-0.5b-instruct-q4_k_m.gguf",
            "expected_mb": 350,
            "desc": "通义千问2 0.5B，Q4量化 ~350MB"
        },
        {
            "name": "Qwen2-1.5B-Instruct Q4",
            "model_id": "qwen/Qwen2-1.5B-Instruct-GGUF",
            "file_name": "qwen2-1.5b-instruct-q4_k_m.gguf",
            "expected_mb": 900,
            "desc": "通义千问2 1.5B，Q4量化 ~900MB"
        },
        {
            "name": "Phi-3-mini-instruct Q4",
            "model_id": "LLM-Research/Phi-3-mini-instruct-gguf",
            "file_name": "phi3-mini-instruct-q4_k_m.gguf",
            "expected_mb": 2300,
            "desc": "微软Phi-3 mini，Q4量化 ~2.3GB"
        },
        {
            "name": "TinyLlama-1.1B Q4",
            "model_id": "AI-ModelScope/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "file_name": "TinyLlama-1.1B-Chat-v1.0-q4_k_m.gguf",
            "expected_mb": 650,
            "desc": "TinyLlama 1.1B，Q4量化 ~650MB"
        },
    ]
    
    print("=" * 60)
    print("🤖 LightLLM 模型下载工具 (ModelScope镜像)")
    print("=" * 60)
    print("\n可用模型:\n")
    
    for i, m in enumerate(models):
        print(f"[{i+1}] {m['name']}")
        print(f"    {m['desc']}")
        print()
    
    print("-" * 60)
    
    # 默认选择Qwen2-0.5B（最小）
    choice = 1
    
    if len(sys.argv) > 1:
        try:
            choice = int(sys.argv[1])
        except:
            pass
    
    if choice < 1 or choice > len(models):
        choice = 1
    
    model = models[choice - 1]
    
    print(f"\n🎯 选择: {model['name']}")
    print(f"⏳ 预计大小: {model['expected_mb']} MB")
    
    local_path = download_model(model["model_id"], model["file_name"])
    
    if local_path:
        print("\n" + "=" * 60)
        print("🎉 下载成功！")
        print("=" * 60)
        print(f"\n📌 下一步:")
        print(f"   1. 安装llama-cpp-python:")
        print(f"      pip install llama-cpp-python")
        print(f"\n   2. 测试模型:")
        print(f"      cd LightLLM")
        print(f"      python -c \"from lightllm import LLMEngine; e = LLMEngine('{local_path}'); print(e.generate('Hello'))\"")
        return 0
    
    return 1

if __name__ == "__main__":
    sys.exit(main())