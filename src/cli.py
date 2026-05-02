#!/usr/bin/env python3
"""LightLLM CLI - Interactive chat or single generation"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import BackendType, LLMEngine


class CLI:
    """Command-line interface"""

    def __init__(self, model: str, backend: str = "llama_cpp", stream: bool = True):
        self.model = model
        self.backend = BackendType(backend)
        self.stream = stream
        self.engine: Optional[LLMEngine] = None

    async def init(self):
        """Initialize"""
        print("LightLLM CLI")
        print(f"   Model: {self.model}")
        print(f"   Backend: {self.backend.value}")
        print()
        print("Model initialization skipped (demo mode)")
        return True

    async def chat(self, system: Optional[str] = None):
        """Interactive chat"""
        print("\n" + "=" * 50)
        print("Chat Mode (Ctrl+C to exit)")
        print("=" * 50)

        if system:
            print(f"System: {system}")

        while True:
            try:
                print("\nYou:", end=" ")
                user_input = input().strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/exit", "/quit", "quit"]:
                    break

                if user_input.lower() == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue

                print("\nLightLLM:", end=" ")
                print(f"[Demo response to: {user_input}]")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    async def complete(self, prompt: str, max_tokens: int = 512):
        """Single generation"""
        print(f"Prompt: {prompt[:100]}...")
        print("Generating...")
        print(f"[Demo completion for: {prompt}]")
        print("\nDone!")
        return 0

    async def list_models(self):
        """List available models"""
        print("\nAvailable Models:")
        print("-" * 50)
        print("  phi-2: Microsoft Phi-2 (2.7B parameters)")
        print("  mistral-7b: Mistral 7B Instruct")
        print("  llama-2-7b: Llama 2 7B Chat")
        return 0


async def async_main():
    parser = argparse.ArgumentParser(description="lightllm - Lightweight local LLM inference")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("model", nargs="?", default="phi-2", help="Model name or path")
    chat_parser.add_argument("--backend", default="llama_cpp")
    chat_parser.add_argument("--system", help="System prompt")

    # Complete
    complete_parser = subparsers.add_parser("complete", help="Single generation")
    complete_parser.add_argument("prompt", help="Prompt")
    complete_parser.add_argument("--model", default="phi-2")
    complete_parser.add_argument("--max-tokens", type=int, default=512)

    # List
    subparsers.add_parser("list", help="List available models")

    args = parser.parse_args()

    if not args.command:
        args.command = "chat"
        args.model = "phi-2"
        args.backend = "llama_cpp"

    cli = CLI(args.model, getattr(args, "backend", "llama_cpp"))

    if args.command == "chat":
        await cli.init()
        await cli.chat(getattr(args, "system", None))
    elif args.command == "complete":
        await cli.complete(args.prompt, args.max_tokens)
    elif args.command == "list":
        await cli.list_models()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        sys.exit(0)


if __name__ == "__main__":
    main()
