# LightLLM 开发文档

## 项目结构

```
LightLLM/
├── src/
│   ├── __init__.py
│   ├── cli.py              # 命令行入口
│   ├── core/
│   │   └── engine.py       # LLM引擎核心
│   ├── api/
│   │   └── server.py       # API服务
│   └── optimizer/
│       ├── skills_optimizer.py  # Skills优化
│       └── context_compressor.py # 上下文压缩
├── tests/
│   └── test_lightllm.py    # 测试用例
├── docs/                   # 文档
├── pyproject.toml          # 项目配置
└── README.md              # 说明文档
```

## 开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/
ruff check src/
```

## 添加新的推理后端

1. 在 `engine.py` 中添加 `_load_<backend>` 方法
2. 在 `_detect_backend` 中添加检测逻辑
3. 在 `_stream_generate` 中添加流式生成实现

## 添加新的智能体连接

1. 在 `AgentConnector` 中添加 `connect_<agent>` 方法
2. 实现 `_send_<agent>` 方法处理通信

## 发布到GitHub

```bash
# 1. 确保版本号更新
# 2. 创建tag
git tag v1.0.0
git push origin v1.0.0

# 3. GitHub Actions会自动发布到PyPI
```