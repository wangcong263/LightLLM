# LightLLM - 本地大模型运行工具

🤖 **一个简洁高效的本地大模型运行框架，支持 llama.cpp/vLLM/CTransformers 后端**

---

## ✨ 特性

- 🖥️ **多后端支持**: llama.cpp (CPU/GPU)、vLLM (NVIDIA GPU)、CTransformers
- 📥 **一键下载**: 支持 HuggingFace/ModelScope 镜像下载模型
- 🚀 **智能推荐**: 根据你的硬件自动推荐最适合的模型
- 🔧 **零配置**: 开箱即用，自动检测环境
- 🌐 **OpenAI 兼容 API**: 完美兼容 OpenAI 接口
- 📱 **多平台**: Windows、Linux、macOS 全支持

---

## 🚀 快速开始

### 方式一：一键部署（推荐）

```bash
# Windows
lightllm.bat

# 或直接运行
python deploy.py
```

部署工具会自动：
1. 检测你的硬件配置
2. 推荐最适合的模型
3. 下载并配置模型
4. 启动 API 服务

### 方式二：命令行安装

```bash
# 1. 列出可用模型
python -m src.model_manager list --all

# 2. 安装模型（如 Phi-2）
python -m src.model_manager install phi-2

# 3. 启动 API
python -m src.api.server --model "C:\Users\你\.cache\lightllm\models\phi-2\phi-2.Q4_K_M.gguf"
```

### 方式三：直接使用

```python
from src.core.engine import LLMEngine, BackendType

# 加载模型
engine = LLMEngine(
    model_path="path/to/model.gguf",
    backend=BackendType.LLAMA_CPP
)

# 生成回复
response = engine.generate(
    prompt="写一个快速排序算法",
    system="你是一个专业的程序员"
)
print(response.content)
```

---

## 📋 可用模型

### 按大小分类

| 类别 | 大小 | 模型 | 内存需求 | 说明 |
|------|------|------|----------|------|
| 🔸 微型 | <1GB | TinyLlama 1.1B | 1GB | CPU 可运行 |
| 🔸 微型 | ~350MB | Qwen2.5 0.5B | 1GB | 中文优化 |
| 🔹 小型 | ~650MB | TinyLlama | 2GB | 最小聊天模型 |
| 🔹 小型 | ~1.8GB | Phi-2 2.7B | 4GB | 优秀推理能力 |
| 🔹 小型 | ~2GB | Qwen2.5 3B | 4GB | 中文优化极佳 |
| 🟡 中型 | ~4GB | Llama-2 7B | 12GB | 主流选择 |
| 🟡 中型 | ~4.1GB | Mistral 7B | 12GB | 优质开源模型 |
| 🔴 大型 | ~4.9GB | Llama 3.1 8B | 16GB | 最新最强 |
| 🔴 大型 | ~9GB | Qwen2.5 14B | 24GB | 超强能力 |

### 推荐配置

| 硬件 | 推荐模型 |
|------|----------|
| 4GB 内存 | TinyLlama, Qwen2.5 0.5B |
| 8GB 内存 | Phi-2, Qwen2.5 1.5B |
| 16GB 内存 | Qwen2.5 7B, Llama-2 7B |
| RTX 3060 12GB | Mistral 7B, Llama-2 7B |
| RTX 4090 24GB | Qwen2.5 14B, Llama 3.1 8B |

---

## 📖 使用示例

### API 服务

```bash
# 启动 API 服务
python -m src.api.server --model phi-2 --port 8000
```

```bash
# 调用 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "用 Python 写一个快速排序"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### 命令行聊天

```bash
# 交互式聊天
python -m src.cli chat --model phi-2

# 单次生成
python -m src.cli complete "解释什么是量子计算"
```

### Python API

```python
from src.core.engine import LLMEngine, BackendType

# 同步使用
engine = LLMEngine("path/to/model.gguf")

result = engine.generate(
    prompt="写一首关于春天的诗",
    system="你是一个诗人",
    temperature=0.8,
    max_tokens=500
)
print(result.content)

# 异步使用
import asyncio
from src.core.engine import AsyncLLMEngine

async def main():
    engine = AsyncLLMEngine("path/to/model.gguf")
    async for chunk in engine.generate("解释机器学习"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

---

## 🔧 配置

### 配置文件

创建 `config.json`:

```json
{
  "model_path": "C:\\Users\\你\\.cache\\lightllm\\models\\phi-2",
  "context_length": 4096,
  "gpu_layers": 32,
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 2048
  }
}
```

### 环境变量

```bash
# 设置模型缓存目录
export LIGHTLLM_MODEL_DIR="D:\\models\\lightllm"

# 设置日志级别
export LIGHTLLM_LOG_LEVEL=DEBUG
```

---

## 📦 依赖

### 必需

- Python 3.8+
- llama-cpp-python
- huggingface-hub
- psutil

### 可选

- torch + CUDA (使用 vLLM 后端)
- fastapi + uvicorn (API 服务)
- colorama (彩色输出)

### 安装依赖

```bash
pip install llama-cpp-python huggingface-hub psutil colorama
```

---

## 📁 项目结构

```
LightLLM/
├── config.py              # 配置文件
├── deploy.py              # 一键部署脚本
├── download_model.py      # 模型下载工具
├── lightllm.bat           # Windows 启动脚本
├── lightllm.sh           # Linux/macOS 启动脚本
├── src/
│   ├── __init__.py
│   ├── cli.py             # 命令行工具
│   ├── api/
│   │   └── server.py      # API 服务
│   └── core/
│       └── engine.py      # 核心引擎
└── models/               # 模型存放目录
```

---

## 🆘 常见问题

### Q: 下载模型太慢？

A: 尝试使用 ModelScope 镜像（国内加速）：
```python
from src.model_manager import ModelCatalog, ModelSource, ModelDownloader

# 使用 ModelScope
model_config = ModelCatalog.get_model("qwen2.5-0.5b-cn")
downloader = ModelDownloader()
downloader.download(model_config)
```

### Q: 显存不够？

A: 使用更小的量化版本或更小的模型：
```bash
# 安装 TinyLlama（最小）
python -m src.model_manager install tinyllama
```

### Q: 如何查看已安装的模型？

```bash
python -m src.model_manager list
```

### Q: 如何删除模型？

```bash
python -m src.model_manager remove phi-2
```

---

## 📝 License

MIT License

---

## 🙏 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 高效的 GGML/GGUF 推理
- [vLLM](https://github.com/vllm-project/vllm) - 快速 PagedAttention 推理
- [HuggingFace](https://huggingface.co/) - 模型托管
- [TheBloke](https://huggingface.co/TheBloke) - 量化模型贡献