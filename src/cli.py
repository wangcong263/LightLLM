#!/usr/bin/env python3
"""
LightLLM 命令行界面
交互式聊天或单次生成
"""
import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import LLMEngine, BackendType, GenerationConfig
from config import MODEL_CACHE_DIR, PRESET_MODELS, create_backend_config, get_model_path


class CLI:
    """命令行界面"""
    
    def __init__(self, model: str, backend: str = "llama_cpp", stream: bool = True):
        self.model = model
        self.backend = BackendType(backend)
        self.stream = stream
        self.engine: Optional[LLMEngine] = None
    
    async def init(self):
        """初始化"""
        print(f"🚀 LightLLM CLI")
        print(f"   Model: {self.model}")
        print(f"   Backend: {self.backend.value}")
        print()
        
        # 查找模型
        model_path = get_model_path(self.model)
        
        if not model_path:
            # 检查自定义路径
            if Path(self.model).exists():
                model_path = self.model
            else:
                print(f"❌ Model '{self.model}' not found!")
                print(f"\n📥 Available models:")
                for name, info in PRESET_MODELS.items():
                    status = "✓" if get_model_path(name) else "○"
                    print(f"   {status} {name}: {info['name']} ({info['size_mb']}MB)")
                print(f"\n💡 Download with:")
                print(f"   python -m src.cli --download {self.model}")
                return False
        
        print(f"📂 Model: {model_path}")
        print(f"⏳ Loading model...")
        
        # 创建引擎
        config = create_backend_config(self.backend)
        self.engine = LLMEngine(model_path, self.backend, config)
        
        # 加载
        success = await self.engine.load()
        
        if success:
            print(f"✅ Model loaded!")
            return True
        else:
            print(f"❌ Failed to load model!")
            return False
    
    async def chat(self, system: Optional[str] = None):
        """交互式聊天"""
        print("\n" + "=" * 50)
        print("💬 Chat Mode (Ctrl+C to exit)")
        print("=" * 50)
        
        if system:
            print(f"📝 System: {system}")
        
        while True:
            try:
                # 获取输入
                print("\n👤 You:", end=" ")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["/exit", "/quit", "quit"]:
                    break
                
                if user_input.lower() == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                
                # 生成响应
                print("\n🤖 LightLLM:", end=" ")
                sys.stdout.flush()
                
                response = ""
                config = GenerationConfig(stream=False)
                
                async for result in self.engine.generate(user_input, system, config):
                    if result.content and not result.done:
                        print(result.content, end="", flush=True)
                        response += result.content
                
                print()
                
                if result.usage:
                    print(f"   [Tokens: {result.usage.get('completion_tokens', '?')}]")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    async def complete(self, prompt: str, max_tokens: int = 512):
        """单次生成"""
        if not self.engine:
            if not await self.init():
                return 1
        
        print(f"📝 Prompt: {prompt[:100]}...")
        print(f"⏳ Generating...")
        
        config = GenerationConfig(max_tokens=max_tokens)
        result_text = ""
        
        async for result in self.engine.generate(prompt, None, config):
            if result.content and not result.done:
                print(result.content, end="", flush=True)
                result_text += result.content
        
        print("\n✅ Done!")
        return 0
    
    async def download_model(self, model_name: str):
        """下载模型"""
        if model_name not in PRESET_MODELS:
            print(f"❌ Unknown model: {model_name}")
            print(f"\n📥 Available:")
            for name in PRESET_MODELS:
                print(f"   - {name}")
            return 1
        
        preset = PRESET_MODELS[model_name]
        
        print(f"📥 Downloading {preset['name']}")
        print(f"   Size: ~{preset['size_mb']}MB")
        print(f"   Repo: {preset['repo']}")
        print()
        
        try:
            from huggingface_hub import hf_hub_download
            
            model_dir = MODEL_CACHE_DIR / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            local_path = hf_hub_download(
                repo_id=preset["repo"],
                filename=preset["file"],
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            
            print(f"\n✅ Downloaded to: {local_path}")
            return 0
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return 1
    
    async def list_models(self):
        """列出可用模型"""
        print("\n📦 Available Models:")
        print("-" * 50)
        
        for name, info in PRESET_MODELS.items():
            status = "✓" if get_model_path(name) else "○"
            size = f"{info['size_mb']}MB"
            desc = info.get("description", "")
            
            print(f"  {status} {name}")
            print(f"     {info['name']} ({size})")
            print(f"     {desc}")
            print()
        
        return 0


async def async_main():
    parser = argparse.ArgumentParser(
        description="LightLLM - 轻量化本地 LLM 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 交互式聊天
  python -m src.cli chat phi-2
  
  # 单次生成
  python -m src.cli complete "Hello, how are you?"
  
  # 下载模型
  python -m src.cli download phi-2
  
  # 列出可用模型
  python -m src.cli list
  
  # 指定自定义模型
  python -m src.cli chat /path/to/model.gguf
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Chat
    chat_parser = subparsers.add_parser("chat", help="交互式聊天")
    chat_parser.add_argument("model", nargs="?", default="phi-2", help="模型名称或路径")
    chat_parser.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "vllm", "ctransformers"])
    chat_parser.add_argument("--system", help="系统提示词")
    
    # Complete
    complete_parser = subparsers.add_parser("complete", help="单次生成")
    complete_parser.add_argument("prompt", help="提示词")
    complete_parser.add_argument("--model", default="phi-2")
    complete_parser.add_argument("--max-tokens", type=int, default=512)
    
    # Download
    download_parser = subparsers.add_parser("download", help="下载模型")
    download_parser.add_argument("model", help="模型名称")
    
    # List
    list_parser = subparsers.add_parser("list", help="列出可用模型")
    
    # Default: interactive chat
    args = parser.parse_args()
    
    if not args.command:
        # 默认进入交互式聊天
        args.command = "chat"
        args.model = "phi-2"
        args.backend = "llama_cpp"
        args.system = None
    
    cli = CLI(args.model, getattr(args, "backend", "llama_cpp"))
    
    if args.command == "chat":
        await cli.init()
        await cli.chat(getattr(args, "system", None))
        
    elif args.command == "complete":
        await cli.complete(args.prompt, args.max_tokens)
        
    elif args.command == "download":
        await cli.download_model(args.model)
        
    elif args.command == "list":
        await cli.list_models()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted!")
        sys.exit(0)


if __name__ == "__main__":
    main()