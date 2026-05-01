#!/usr/bin/env python3
"""
LightLLM 一键部署工具
自动检测硬件环境，推荐并部署最适合的模型
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class HardwareProfile:
    """硬件配置"""
    os: str
    cpu: str
    memory_gb: int
    gpu_name: str
    gpu_memory_gb: int
    gpu_vendor: str  # 'nvidia', 'amd', 'apple', 'intel', 'none'
    
    def recommend_model_size(self) -> str:
        """根据硬件推荐模型大小"""
        if self.gpu_memory_gb >= 24:
            return "xlarge"  # 70B+
        elif self.gpu_memory_gb >= 16:
            return "large"   # 13B-70B
        elif self.gpu_memory_gb >= 10:
            return "medium"  # 7B-13B
        elif self.gpu_memory_gb >= 6:
            return "small"   # 3B-7B
        elif self.gpu_memory_gb >= 4:
            return "tiny"    # 1B-3B
        elif self.memory_gb >= 8:
            return "tiny"
        else:
            return "micro"   # <1B


def get_hardware_info() -> HardwareProfile:
    """获取硬件信息"""
    system = platform.system()
    cpu = platform.processor() or "Unknown CPU"
    
    # 获取内存信息
    memory_gb = 8  # 默认值
    try:
        if system == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", c_ulong),
                    ("dwMemoryLoad", c_ulong),
                    ("dwTotalPhys", c_ulong),
                    ("dwAvailPhys", c_ulong),
                    ("dwTotalPageFile", c_ulong),
                    ("dwAvailPageFile", c_ulong),
                    ("dwTotalVirtual", c_ulong),
                    ("dwAvailVirtual", c_ulong),
                ]
            memstatus = MEMORYSTATUS()
            memstatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memstatus))
            memory_gb = memstatus.dwTotalPhys / (1024**3)
        else:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                memory_gb = int(result.stdout.strip()) / (1024**3)
    except:
        pass
    
    # 获取 GPU 信息
    gpu_name = "未检测到"
    gpu_memory_gb = 0
    gpu_vendor = "none"
    
    try:
        if system == "Windows":
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        gpu_info = lines[0].split(',')
                        gpu_name = gpu_info[0].strip()
                        gpu_memory_gb = int(gpu_info[1].strip().replace('MiB', '')) / 1024
                        gpu_vendor = "nvidia"
            except:
                try:
                    result = subprocess.run(
                        ['powershell', '-Command', 'Get-CimInstance Win32_VideoController | Select-Object -First 1 Name'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        gpu_name = result.stdout.strip()
                        gpu_vendor = "unknown"
                        gpu_memory_gb = 4  # 假设有 4GB
                except:
                    pass
        elif system == "Darwin":
            import subprocess
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Chipset Model' in line or 'VRAM' in line:
                        gpu_name = line.split(':')[1].strip() if ':' in line else gpu_name
                # Apple Silicon 共享内存
                gpu_memory_gb = memory_gb / 4 if memory_gb else 8
                gpu_vendor = "apple"
    except:
        pass
    
    return HardwareProfile(
        os=system,
        cpu=cpu,
        memory_gb=int(memory_gb),
        gpu_name=gpu_name,
        gpu_memory_gb=int(gpu_memory_gb),
        gpu_vendor=gpu_vendor
    )


def print_banner():
    """打印横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   ⚡ LightLLM - 本地大模型一键部署工具 ⚡             ║
    ║                                                      ║
    ║   让每个人都能轻松运行本地大模型                      ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_hardware_info(info: HardwareProfile):
    """打印硬件信息"""
    print("\n📋 检测到您的硬件配置:")
    print(f"   💻 操作系统: {info.os}")
    print(f"   🔲 CPU: {info.cpu}")
    print(f"   🧠 内存: {info.memory_gb} GB")
    print(f"   🎮 GPU: {info.gpu_name}")
    if info.gpu_memory_gb > 0:
        print(f"   📦 显存: {info.gpu_memory_gb} GB")


def get_recommended_models(info: HardwareProfile) -> List[Dict]:
    """获取推荐的模型列表"""
    model_size = info.recommend_model_size()
    
    all_models = [
        # 微型模型 (<1B)
        {"id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "name": "Qwen2.5 0.5B", "params": "0.5B", 
         "size": "~400MB", "memory": "1GB", "backend": "llama.cpp", "description": "阿里通义千问微型版，适合尝鲜", "size_category": "micro"},
        {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF", "name": "TinyLlama 1.1B", "params": "1.1B", 
         "size": "~650MB", "memory": "2GB", "backend": "llama.cpp", "description": "轻量级开源模型", "size_category": "tiny"},
        
        # 小型模型 (1-3B)
        {"id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "name": "Qwen2.5 1.5B", "params": "1.5B", 
         "size": "~900MB", "memory": "3GB", "backend": "llama.cpp", "description": "阿里通义千问轻量版", "size_category": "tiny", "recommended": True},
        {"id": "microsoft/Phi-2-GGUF", "name": "Phi-2", "params": "2.7B", 
         "size": "~1.6GB", "memory": "4GB", "backend": "llama.cpp", "description": "微软小模型，性能出色", "size_category": "small"},
        
        # 中型模型 (3-7B)
        {"id": "Qwen/Qwen2.5-3B-Instruct-GGUF", "name": "Qwen2.5 3B", "params": "3B", 
         "size": "~1.8GB", "memory": "6GB", "backend": "llama.cpp", "description": "性价比最高的中文模型", "size_category": "small", "recommended": True},
        {"id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "name": "Qwen2.5 7B", "params": "7B", 
         "size": "~4.4GB", "memory": "12GB", "backend": "llama.cpp", "description": "主流开源模型，效果优秀", "size_category": "medium", "recommended": True},
        
        # 大型模型 (7B+)
        {"id": "meta-llama/Llama-3.1-8B-Instruct-GGUF", "name": "Llama 3.1 8B", "params": "8B", 
         "size": "~4.9GB", "memory": "16GB", "backend": "llama.cpp", "description": "Meta 最新开源模型", "size_category": "medium"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3-GGUF", "name": "Mistral 7B", "params": "7B", 
         "size": "~4.1GB", "memory": "12GB", "backend": "llama.cpp", "description": "欧洲最强开源模型", "size_category": "medium"},
    ]
    
    # 根据硬件筛选推荐
    size_priority = {
        "micro": ["micro", "tiny", "small", "medium", "large"],
        "tiny": ["tiny", "small", "medium", "large"],
        "small": ["small", "medium", "large"],
        "medium": ["medium", "large", "small"],
        "large": ["large", "medium", "xlarge"],
        "xlarge": ["xlarge", "large", "medium"],
    }
    
    priority = size_priority.get(model_size, ["medium"])
    
    # 排序
    def sort_key(m):
        try:
            return priority.index(m["size_category"])
        except:
            return 999
    
    return sorted(all_models, key=sort_key)[:6]


def print_model_recommendations(models: List[Dict]):
    """打印模型推荐列表"""
    print("\n🚀 根据您的硬件，推荐以下模型:")
    print("-" * 60)
    
    for i, model in enumerate(models, 1):
        rec_tag = " ⭐推荐" if model.get("recommended") else ""
        print(f"\n   {i}. {model['name']} ({model['params']}){rec_tag}")
        print(f"      📦 大小: {model['size']} | 💾 内存: {model['memory']}")
        print(f"      📝 {model['description']}")
    
    print("\n" + "-" * 60)


def install_dependencies():
    """安装依赖"""
    print("\n📦 正在检查/安装依赖...")
    
    deps = {
        "flask": "flask>=2.0",
        "flask-cors": "flask-cors>=3.0",
        "requests": "requests>=2.25",
        "huggingface_hub": "huggingface-hub>=0.19",
    }
    
    for name, package in deps.items():
        try:
            __import__(name.replace("-", "_"))
            print(f"   ✓ {name} 已安装")
        except ImportError:
            print(f"   📥 安装 {name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True)
            print(f"   ✓ {name} 安装完成")


def deploy_model(model_id: str, model_name: str):
    """部署指定模型"""
    print(f"\n🔄 开始部署: {model_name}")
    print("=" * 60)
    
    # 导入模型管理器
    from src.model_manager import ModelManager
    
    mm = ModelManager()
    
    # 下载模型
    print(f"\n📥 正在下载模型...")
    print(f"   源: HuggingFace / ModelScope")
    
    try:
        mm.download_model(model_id)
        print(f"\n✅ {model_name} 下载完成!")
        
        # 启动服务
        print(f"\n🚀 启动模型服务...")
        result = mm.start_model_service(model_id, 8000)
        
        print(f"\n" + "=" * 60)
        print(f"🎉 部署成功!")
        print(f"=" * 60)
        print(f"\n   📍 API 地址: http://localhost:8000")
        print(f"   📖 文档地址: http://localhost:8000/docs")
        print(f"\n   模型文件位置: {mm.models_dir}")
        print(f"\n   按 Ctrl+C 停止服务\n")
        
        # 保持运行
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 正在停止服务...")
            mm.stop_model_service(model_id)
            print("✅ 已停止\n")
            
    except Exception as e:
        print(f"\n❌ 部署失败: {e}")
        raise


def start_webui():
    """启动 WebUI"""
    from src.webui_api import run_webui
    run_webui(host='0.0.0.0', port=7860)


def main():
    """主函数"""
    print_banner()
    
    # 获取硬件信息
    info = get_hardware_info()
    print_hardware_info(info)
    
    # 交互式选择
    print("\n请选择操作:")
    print("   1. 🚀 智能推荐部署（自动选择最适合的模型）")
    print("   2. 📋 查看推荐模型列表")
    print("   3. 🌐 启动 WebUI 管理界面")
    print("   4. 📦 安装指定模型")
    print("   0. ❌ 退出")
    
    choice = input("\n请输入选项 [1-4, 0]: ").strip()
    
    if choice == "1":
        # 智能部署
        models = get_recommended_models(info)
        if models:
            default_model = next((m for m in models if m.get("recommended")), models[0])
            print(f"\n自动选择推荐模型: {default_model['name']}")
            
            install_dependencies()
            deploy_model(default_model["id"], default_model["name"])
        else:
            print("❌ 未找到适合的模型")
            
    elif choice == "2":
        # 查看推荐列表
        models = get_recommended_models(info)
        print_model_recommendations(models)
        
    elif choice == "3":
        # 启动 WebUI
        install_dependencies()
        print("\n🌐 正在启动 WebUI...")
        start_webui()
        
    elif choice == "4":
        # 安装指定模型
        models = get_recommended_models(info)
        print_model_recommendations(models)
        
        model_num = input("\n请输入模型编号 [1-6]: ").strip()
        try:
            idx = int(model_num) - 1
            if 0 <= idx < len(models):
                install_dependencies()
                model = models[idx]
                deploy_model(model["id"], model["name"])
            else:
                print("❌ 无效的编号")
        except ValueError:
            print("❌ 请输入有效数字")
            
    elif choice == "0":
        print("\n👋 再见!")
        return 0
        
    else:
        print("\n❌ 无效选项")
        
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
        sys.exit(0)