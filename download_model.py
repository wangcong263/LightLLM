#!/usr/bin/env python3
"""
下载GGUF模型进行测试
使用HuggingFace下载TheBloke量化模型
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# 模型存储目录
MODELS_DIR = Path.home() / ".cache" / "lightllm" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_model(repo_id: str, filename: str, expected_size_mb: float = None):
    """下载GGUF模型"""
    print(f"📥 正在下载: {repo_id}/{filename}")
    print(f"📁 保存位置: {MODELS_DIR}")
    
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR / repo_id.split("/")[-1],
            local_dir_use_symlinks=False
        )
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"✅ 下载完成!")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"📍 路径: {local_path}")
        
        if expected_size_mb:
            expected_bytes = expected_size_mb * 1024 * 1024
            if abs(os.path.getsize(local_path) - expected_bytes) / expected_bytes > 0.1:
                print(f"⚠️ 警告: 文件大小与预期差异较大")
        
        return local_path
    
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def main():
    # 可用模型列表（从小到大排序）
    models = [
        {
            "name": "Phi-2 Q4 (推荐测试用)",
            "repo_id": "TheBloke/phi-2-GGUF",
            "filename": "phi-2.Q4_K_M.gguf",
            "expected_mb": 1800,
            "desc": "微软Phi-2模型，27亿参数，Q4量化 ~1.8GB"
        },
        {
            "name": "TinyLlama Q4",
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "expected_mb": 650,
            "desc": "TinyLlama 1.1B，Q4量化 ~650MB（最小）"
        },
        {
            "name": "Llama-2 7B Q4",
            "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
            "filename": "llama-2-7b-chat.Q4_K_M.gguf",
            "expected_mb": 3800,
            "desc": "Llama-2 7B，Q4量化 ~3.8GB"
        },
        {
            "name": "Mistral 7B Q4",
            "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "expected_mb": 4100,
            "desc": "Mistral 7B，Q4量化 ~4.1GB"
        },
    ]
    
    print("=" * 60)
    print("🤖 LightLLM 模型下载工具")
    print("=" * 60)
    print("\n可用模型:\n")
    
    for i, m in enumerate(models):
        print(f"[{i+1}] {m['name']}")
        print(f"    {m['desc']}")
        print()
    
    print("-" * 60)
    
    # 默认选择Phi-2（平衡大小和性能）
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
    
    local_path = download_model(model["repo_id"], model["filename"], model["expected_mb"])
    
    if local_path:
        print("\n" + "=" * 60)
        print("🎉 下载成功！")
        print("=" * 60)
        print(f"\n📌 使用方法:")
        print(f"   lightllm load \"{local_path}\"")
        print(f"\n或直接在Python中:")
        print(f"   from lightllm import LLMEngine")
        print(f"   engine = LLMEngine(model_path=\"{local_path}\")")
        return 0
    
    return 1

if __name__ == "__main__":
    sys.exit(main())