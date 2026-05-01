"""
LightLLM CLI - 命令行接口
"""
import asyncio
import sys
import argparse
import json
from typing import Optional

from .core.engine import ModelConfig, ModelManager
from .api.server import LightLLMAPI, AgentConnector


def main():
    parser = argparse.ArgumentParser(description="LightLLM - 轻量化本地LLM运行工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 运行模型
    run_parser = subparsers.add_parser("run", help="运行模型")
    run_parser.add_argument("--model", "-m", required=True, help="模型名称")
    run_parser.add_argument("--path", "-p", required=True, help="模型路径")
    run_parser.add_argument("--context", "-c", type=int, default=4096, help="上下文长度")
    run_parser.add_argument("--threads", "-t", type=int, default=4, help="线程数")
    run_parser.add_argument("--gpu-layers", "-g", type=int, default=0, help="GPU层数")
    
    # 启动API服务
    serve_parser = subparsers.add_parser("serve", help="启动API服务")
    serve_parser.add_argument("--host", default="localhost", help="主机")
    serve_parser.add_argument("--port", "-p", type=int, default=8080, help="端口")
    
    # 交互模式
    chat_parser = subparsers.add_parser("chat", help="交互模式")
    chat_parser.add_argument("--model", "-m", required=True, help="模型名称")
    chat_parser.add_argument("--path", "-p", required=True, help="模型路径")
    
    # 连接智能体
    connect_parser = subparsers.add_parser("connect", help="连接智能体")
    connect_parser.add_argument("--agent", required=True, help="智能体类型 (openclaw/hermes)")
    connect_parser.add_argument("--url", required=True, help="连接URL")
    
    args = parser.parse_args()
    
    if args.command == "run":
        asyncio.run(run_model(args))
    elif args.command == "serve":
        asyncio.run(serve(args))
    elif args.command == "chat":
        asyncio.run(chat(args))
    elif args.command == "connect":
        asyncio.run(connect(args))
    else:
        parser.print_help()


async def run_model(args):
    """运行模型"""
    config = ModelConfig(
        name=args.model,
        path=args.path,
        context_length=args.context,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
    )
    
    manager = ModelManager()
    manager.add_model(args.model, config)
    
    print(f"Loading model: {args.model}...")
    await manager.load(args.model)
    print("Model loaded! Enter your prompts (Ctrl+C to exit):\n")
    
    engine = manager.get_current()
    
    while True:
        try:
            prompt = input("> ")
            if not prompt.strip():
                continue
            
            print("\nAssistant: ", end="", flush=True)
            
            async for token in engine.generate(prompt):
                print(token, end="", flush=True)
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            break


async def serve(args):
    """启动API服务"""
    api = LightLLMAPI(host=args.host, port=args.port)
    print(f"Starting LightLLM API on {args.host}:{args.port}")
    await api.start()


async def chat(args):
    """交互聊天"""
    config = ModelConfig(
        name=args.model,
        path=args.path,
    )
    
    manager = ModelManager()
    manager.add_model(args.model, config)
    await manager.load(args.model)
    
    engine = manager.get_current()
    
    print("Chat mode. Type 'exit' to quit.\n")
    
    messages = []
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            
            messages.append({"role": "user", "content": user_input})
            
            prompt = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in messages
            )
            
            response = ""
            async for token in engine.generate(prompt):
                response += token
            
            messages.append({"role": "assistant", "content": response})
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            break


async def connect(args):
    """连接智能体"""
    api = LightLLMAPI()
    connector = AgentConnector(api)
    
    if args.agent == "openclaw":
        success = connector.connect_openclaw("default", {"ws_url": args.url})
    elif args.agent == "hermes":
        success = connector.connect_hermes("default", {"api_url": args.url})
    else:
        print(f"Unknown agent type: {args.agent}")
        return
    
    if success:
        print(f"Connected to {args.agent} at {args.url}")
    else:
        print("Connection failed")


if __name__ == "__main__":
    main()