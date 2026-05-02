#!/usr/bin/env python3
"""
LightLLM 模型转换器
支持多种模型格式之间的转换
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import argparse

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_CACHE_DIR = Path.home() / ".cache" / "lightllm" / "models"


class ModelFormat(Enum):
    """支持的模型格式"""
    GGUF = "gguf"           # llama.cpp 格式
    SAFETENSORS = "safetensors"  # HuggingFace 安全格式
    PTH = "pth"             # PyTorch 格式
    ONNX = "onnx"           # ONNX 格式
    GPTQ = "gptq"           # GPTQ 量化格式
    AWQ = "awq"             # AWQ 量化格式
    EXL2 = "exl2"           # EXL2 量化格式
    MLX = "mlx"             # Apple Silicon 格式


@dataclass
class ConversionJob:
    """转换任务"""
    input_path: Path
    output_path: Path
    input_format: ModelFormat
    output_format: ModelFormat
    quantization: Optional[str] = None  # 量化参数如 Q4_K_M
    extra_args: Dict = None
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = {}


class ModelConverter:
    """模型转换器"""
    
    def __init__(self):
        self.supported_conversions = {
            # (源格式, 目标格式): self.方法
        }
    
    def convert(self, job: ConversionJob, progress_callback=None) -> Path:
        """执行转换"""
        key = (job.input_format, job.output_format)
        
        if key not in self.supported_conversions:
            raise ValueError(f"不支持的转换: {job.input_format.value} -> {job.output_format.value}")
        
        converter = self.supported_conversions[key]
        return converter(job, progress_callback)
    
    def convert_safetensors_to_gguf(self, job: ConversionJob, callback=None) -> Path:
        """Safetensors -> GGUF"""
        print(f"转换 Safetensors 到 GGUF...")
        print(f"输入: {job.input_path}")
        print(f"输出: {job.output_path}")
        
        # 检查 llama.cpp 工具
        convert_script = self._get_llama_cpp_convert_script()
        if not convert_script:
            raise RuntimeError("llama.cpp 未安装，请参考: https://github.com/ggerganov/llama.cpp")
        
        # 执行转换
        cmd = [
            sys.executable, convert_script,
            str(job.input_path),
            "--outfile", str(job.output_path),
        ]
        
        if job.quantization:
            cmd.extend(["--outtype", job.quantization.lower()])
        
        print(f"执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"转换失败: {result.stderr}")
        
        return job.output_path
    
    def convert_pth_to_gguf(self, job: ConversionJob, callback=None) -> Path:
        """PyTorch -> GGUF"""
        return self.convert_safetensors_to_gguf(job, callback)
    
    def quantize_gguf(self, job: ConversionJob, callback=None) -> Path:
        """GGUF 量化"""
        print(f"量化 GGUF 模型...")
        print(f"输入: {job.input_path}")
        print(f"输出: {job.output_path}")
        print(f"量化等级: {job.quantization}")
        
        # 查找 llama.cpp 的量化工具
        quantize_binary = self._get_quantize_binary()
        if not quantize_binary:
            raise RuntimeError("llama.cpp 量化工具未找到")
        
        cmd = [
            str(quantize_binary),
            str(job.input_path),
            str(job.output_path),
            job.quantization
        ]
        
        print(f"执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"量化失败: {result.stderr}")
        
        return job.output_path
    
    def convert_to_onnx(self, job: ConversionJob, callback=None) -> Path:
        """转换为 ONNX 格式"""
        print(f"转换到 ONNX 格式...")
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            print("需要安装: pip install torch transformers")
            raise
        
        print(f"加载模型: {job.input_path}")
        model = AutoModel.from_pretrained(str(job.input_path))
        tokenizer = AutoTokenizer.from_pretrained(str(job.input_path))
        
        # 导出为 ONNX
        print(f"导出 ONNX...")
        dummy_input = tokenizer("Hello", return_tensors="pt")
        
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input.get("attention_mask")),
            str(job.output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"}
            },
            opset_version=14
        )
        
        return job.output_path
    
    def convert_to_mlx(self, job: ConversionJob, callback=None) -> Path:
        """转换为 MLX 格式 (Apple Silicon)"""
        print(f"转换到 MLX 格式...")
        
        try:
            from mlx_lm import convert
        except ImportError:
            print("需要安装: pip install mlx-lm")
            raise
        
        print(f"输入: {job.input_path}")
        print(f"输出: {job.output_path}")
        
        # mlx_lm 转换
        convert(
            repo=str(job.input_path),
            output_path=str(job.output_path),
            quantize=True if job.quantization else False
        )
        
        return job.output_path
    
    def _get_llama_cpp_convert_script(self) -> Optional[Path]:
        """获取 llama.cpp 转换脚本"""
        possible_paths = [
            # llama.cpp 本地安装
            Path.home() / ".local" / "share" / "llama.cpp" / "convert.py",
            Path.home() / ".cache" / "llama.cpp" / "convert.py",
            # 系统安装
            Path("/usr/local/share/llama.cpp/convert.py"),
            Path("/opt/llama.cpp/convert.py"),
        ]
        
        for p in possible_paths:
            if p.exists():
                return p
        
        # 检查是否在 PATH 中
        if shutil.which("llama-cli"):
            # 尝试查找转换脚本
            llama_dir = Path(shutil.which("llama-cli")).parent.parent
            convert_script = llama_dir / "convert.py"
            if convert_script.exists():
                return convert_script
        
        return None
    
    def _get_quantize_binary(self) -> Optional[Path]:
        """获取 llama.cpp 量化工具"""
        possible_paths = [
            Path.home() / ".local" / "bin" / "llama-quantize",
            Path.home() / ".cache" / "llama.cpp" / "llama-quantize",
            Path("/usr/local/bin/llama-quantize"),
            Path("llama-quantize"),  # PATH 中
        ]
        
        for p in possible_paths:
            if p.exists() or shutil.which(str(p)) if not p.exists() else True:
                # 如果在 PATH 中
                if not p.exists() and shutil.which(str(p)):
                    return Path(shutil.which(str(p)))
                elif p.exists():
                    return p
        
        return None
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return [f.value for f in ModelFormat]
    
    def get_conversion_paths(self) -> List[Dict]:
        """获取支持的转换路径"""
        return [
            {"from": "safetensors", "to": "gguf", "description": "HuggingFace -> llama.cpp", "needs_quantize": True},
            {"from": "pth", "to": "gguf", "description": "PyTorch -> llama.cpp", "needs_quantize": True},
            {"from": "gguf", "to": "gguf", "description": "GGUF 量化", "needs_quantize": True},
            {"from": "safetensors", "to": "onnx", "description": "HuggingFace -> ONNX", "needs_quantize": False},
            {"from": "safetensors", "to": "mlx", "description": "Apple Silicon 优化", "needs_quantize": True},
            {"from": "safetensors", "to": "gptq", "description": "GPTQ 量化", "needs_quantize": True},
            {"from": "safetensors", "to": "awq", "description": "AWQ 量化", "needs_quantize": True},
        ]


def list_formats():
    """列出支持的格式"""
    converter = ModelConverter()
    print("\n📦 支持的模型格式:")
    for fmt in ModelFormat:
        print(f"  • {fmt.value}")
    
    print("\n🔄 支持的转换路径:")
    for path in converter.get_conversion_paths():
        q_note = " (可量化)" if path["needs_quantize"] else ""
        print(f"  {path['from']} -> {path['to']}{q_note}")
        print(f"     {path['description']}")


def convert_model(input_path: str, output_format: str, output_path: str = None,
                  quantization: str = None, verbose: bool = True):
    """转换模型"""
    input_p = Path(input_path)
    
    if not input_p.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 检测输入格式
    input_format = _detect_format(input_p)
    output_fmt = ModelFormat(output_format)
    
    if verbose:
        print(f"\n🔄 转换模型")
        print(f"   输入: {input_p}")
        print(f"   格式: {input_format.value} -> {output_fmt.value}")
        if quantization:
            print(f"   量化: {quantization}")
    
    # 生成输出路径
    if not output_path:
        output_path = input_p.parent / f"{input_p.stem}_{output_format}.bin"
    output_p = Path(output_path)
    
    # 创建转换任务
    job = ConversionJob(
        input_path=input_p,
        output_path=output_p,
        input_format=input_format,
        output_format=output_fmt,
        quantization=quantization
    )
    
    # 执行转换
    converter = ModelConverter()
    result = converter.convert(job)
    
    print(f"\n✅ 转换完成: {result}")
    return result


def _detect_format(path: Path) -> ModelFormat:
    """检测模型格式"""
    # 检查扩展名
    ext = path.suffix.lower()
    
    if ext == ".gguf" or ext == ".bin" and "gguf" in str(path):
        return ModelFormat.GGUF
    elif ext in [".safetensors", ".bin"] and path.suffix != ".bin":
        return ModelFormat.SAFETENSORS
    elif ext in [".pt", ".pth", ".ckpt"]:
        return ModelFormat.PTH
    elif ext == ".onnx":
        return ModelFormat.ONNX
    
    # 检查目录（HuggingFace 模型通常是目录）
    if path.is_dir():
        files = list(path.iterdir())
        file_names = [f.name.lower() for f in files]
        
        if any("model.safetensors" in f for f in file_names):
            return ModelFormat.SAFETENSORS
        elif any(f.endswith(".gguf") for f in file_names):
            return ModelFormat.GGUF
        elif any(f.endswith(".onnx") for f in file_names):
            return ModelFormat.ONNX
        elif any(f.endswith(".bin") for f in file_names):
            return ModelFormat.SAFETENSORS  # 默认假设
    
    return ModelFormat.SAFETENSORS  # 默认格式


def quantize_model(input_path: str, output_path: str, level: str = "Q4_K_M"):
    """量化模型"""
    input_p = Path(input_path)
    
    if not input_p.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    print(f"\n⚡ 量化模型")
    print(f"   输入: {input_p}")
    print(f"   输出: {output_path}")
    print(f"   量化: {level}")
    
    job = ConversionJob(
        input_path=input_p,
        output_path=Path(output_path),
        input_format=ModelFormat.GGUF,
        output_format=ModelFormat.GGUF,
        quantization=level
    )
    
    converter = ModelConverter()
    result = converter.quantize_gguf(job)
    
    print(f"\n✅ 量化完成: {result}")
    return result


def list_quantization_levels():
    """列出量化等级"""
    levels = [
        ("Q2_K", "最高压缩", "~2.5bit/参数", "极低"),
        ("Q3_K_M", "高压缩", "~3bit/参数", "较低"),
        ("Q4_0", "标准量化", "~4bit/参数", "中等"),
        ("Q4_K_M", "平衡压缩 (推荐)", "~4.5bit/参数", "良好"),
        ("Q5_0", "高质量", "~5bit/参数", "很好"),
        ("Q5_K_M", "高质量压缩", "~5.5bit/参数", "优秀"),
        ("Q6_K", "接近原生", "~6bit/参数", "极佳"),
        ("Q8_0", "几乎无损", "~8bit/参数", "最好"),
        ("FP16", "半精度", "16bit/参数", "标准"),
        ("FP32", "全精度", "32bit/参数", "原始"),
    ]
    
    print("\n📊 量化等级说明:")
    print(f"  {'等级':<10} {'描述':<20} {'压缩率':<15} {'质量':<8}")
    print("  " + "-" * 60)
    for level, desc, compress, quality in levels:
        print(f"  {level:<10} {desc:<20} {compress:<15} {quality:<8}")


def install_llama_cpp():
    """安装 llama.cpp"""
    print("\n🔧 安装 llama.cpp...")
    
    # 检查 git
    if not shutil.which("git"):
        print("需要安装 git")
        return False
    
    target_dir = Path.home() / ".local" / "share" / "llama.cpp"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if target_dir.exists():
        print(f"llama.cpp 已存在于: {target_dir}")
        print("更新中...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=target_dir,
            capture_output=True
        )
    else:
        print(f"克隆 llama.cpp 到: {target_dir}")
        result = subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(target_dir)],
            capture_output=True
        )
    
    if result.returncode != 0:
        print(f"安装失败: {result.stderr.decode() if result.stderr else 'Unknown error'}")
        return False
    
    # 编译
    print("编译中...")
    build_dir = target_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    result = subprocess.run(
        ["cmake", ".."],
        cwd=build_dir,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"CMake 失败: {result.stderr.decode() if result.stderr else 'Unknown error'}")
        return False
    
    result = subprocess.run(
        ["cmake", "--build", ".", "--config", "Release"],
        cwd=build_dir,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"编译失败: {result.stderr.decode() if result.stderr else 'Unknown error'}")
        return False
    
    print("\n✅ llama.cpp 安装完成!")
    print(f"   转换脚本: {target_dir}/convert.py")
    print(f"   量化工具: {build_dir}/bin/Release/llama-quantize")
    
    return True


# ============= CLI =============
def main():
    parser = argparse.ArgumentParser(description="LightLLM 模型转换工具")
    parser.add_argument("action", choices=["convert", "quantize", "formats", "levels", "install-llama"],
                        help="操作: convert(转换) / quantize(量化) / formats(格式) / levels(量化等级) / install-llama(安装llama.cpp)")
    parser.add_argument("--input", "-i", help="输入文件/目录")
    parser.add_argument("--output", "-o", help="输出文件/目录")
    parser.add_argument("--format", "-f", choices=["gguf", "onnx", "mlx", "gptq", "awq"],
                        help="目标格式")
    parser.add_argument("--quantize", "-q", choices=["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0", "FP16"],
                        help="量化等级")
    
    args = parser.parse_args()
    
    if args.action == "formats":
        list_formats()
    
    elif args.action == "levels":
        list_quantization_levels()
    
    elif args.action == "install-llama":
        success = install_llama_cpp()
        return 0 if success else 1
    
    elif args.action == "convert":
        if not args.input:
            print("请指定输入文件: --input <path>")
            return 1
        if not args.format:
            print("请指定目标格式: --format <gguf|onnx|mlx|gptq|awq>")
            return 1
        
        try:
            convert_model(
                args.input,
                args.format,
                args.output,
                args.quantize
            )
        except Exception as e:
            print(f"\n❌ 转换失败: {e}")
            return 1
    
    elif args.action == "quantize":
        if not args.input:
            print("请指定输入文件: --input <path>")
            return 1
        if not args.output:
            print("请指定输出文件: --output <path>")
            return 1
        
        level = args.quantize or "Q4_K_M"
        
        try:
            quantize_model(args.input, args.output, level)
        except Exception as e:
            print(f"\n❌ 量化失败: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())