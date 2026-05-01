#!/usr/bin/env python3
"""
LightLLM WebUI API Server
提供模型管理和部署的 REST API
"""

import os
import sys

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 导入模型管理模块
from src.model_manager import (
    ModelDownloader, list_popular_models, get_model_info,
    ModelCatalog, get_system_info, DEFAULT_MODEL_DIR
)

# ============================================
# Flask 应用
# ============================================

app = Flask(__name__, static_folder='webui')
CORS(app)

# 全局状态
download_tasks = {}
running_models = {}

# ============================================
# 静态文件
# ============================================

@app.route('/')
def index():
    return send_from_directory('webui', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('webui', filename)

# ============================================
# 系统信息
# ============================================

@app.route('/api/system')
def get_system_info_api():
    """获取系统信息"""
    try:
        info = get_system_info()
        return jsonify({
            "success": True,
            "data": info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ============================================
# 模型列表
# ============================================

@app.route('/api/models')
def get_models():
    """获取所有可用模型"""
    try:
        models = list_popular_models()
        return jsonify({
            "success": True,
            "data": {k: v.to_dict() for k, v in models.items()}
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/models/<model_id>')
def get_model_detail(model_id):
    """获取模型详情"""
    try:
        info = get_model_info(model_id)
        if info:
            return jsonify({"success": True, "data": info})
        return jsonify({"success": False, "error": "Model not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/models/recommend')
def get_recommend_models():
    """获取推荐模型"""
    try:
        info = get_system_info()
        available_mb = int(info["memory_available_gb"] * 1024)
        recommended = ModelCatalog.recommend_for_memory(available_mb)
        return jsonify({
            "success": True,
            "data": [m.to_dict() for m in recommended]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ============================================
# 已安装模型
# ============================================

@app.route('/api/installed')
def get_installed():
    """获取已安装模型"""
    try:
        downloader = ModelDownloader()
        installed = downloader.list_installed()
        return jsonify({
            "success": True,
            "data": installed
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ============================================
# 下载模型
# ============================================

@app.route('/api/download', methods=['POST'])
def download_model():
    """下载模型"""
    data = request.json
    model_id = data.get('model_id')
    quantize = data.get('quantize')
    
    if not model_id:
        return jsonify({"success": False, "error": "model_id is required"})
    
    config = ModelCatalog.get_model(model_id)
    if not config:
        return jsonify({"success": False, "error": "Model not found"})
    
    # 检查是否已在下载中
    if model_id in download_tasks:
        return jsonify({"success": False, "error": "Model is already downloading"})
    
    def progress_callback(progress):
        download_tasks[model_id] = {
            "downloaded": progress.downloaded,
            "total": progress.total,
            "percent": progress.percent(),
            "speed": progress.speed,
            "status": progress.status
        }
    
    def download_thread():
        try:
            downloader = ModelDownloader(progress_callback=progress_callback)
            path = downloader.download(config, quantize=quantize)
            download_tasks[model_id] = {
                "status": "completed",
                "path": str(path)
            }
        except Exception as e:
            download_tasks[model_id] = {
                "status": "failed",
                "error": str(e)
            }
    
    thread = threading.Thread(target=download_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "开始下载 " + config.name,
        "task_id": model_id
    })

@app.route('/api/download/<model_id>/status')
def get_download_status(model_id):
    """获取下载状态"""
    if model_id in download_tasks:
        return jsonify({
            "success": True,
            "data": download_tasks[model_id]
        })
    return jsonify({
        "success": False,
        "error": "No download task found"
    })

# ============================================
# 删除模型
# ============================================

@app.route('/api/delete', methods=['POST'])
def delete_model():
    """删除模型"""
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({"success": False, "error": "model_id is required"})
    
    try:
        downloader = ModelDownloader()
        success = downloader.delete(model_id)
        if success:
            return jsonify({
                "success": True,
                "message": "已删除模型 " + model_id
            })
        return jsonify({
            "success": False,
            "error": "Model not found or already deleted"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ============================================
# 启动/停止服务
# ============================================

@app.route('/api/start', methods=['POST'])
def start_model():
    """启动模型服务"""
    data = request.json
    model_id = data.get('model_id')
    port = data.get('port', DEFAULT_MODEL_DIR)
    
    if not model_id:
        return jsonify({"success": False, "error": "model_id is required"})
    
    if model_id in running_models:
        return jsonify({"success": False, "error": "Model is already running"})
    
    config = ModelCatalog.get_model(model_id)
    if not config:
        return jsonify({"success": False, "error": "Model not found"})
    
    # 检查模型是否已安装
    downloader = ModelDownloader()
    installed = downloader.list_installed()
    
    model_path = None
    for m in installed:
        if m["config"] and m["config"]["id"] == model_id:
            model_path = m["path"]
            break
    
    if not model_path:
        return jsonify({"success": False, "error": "Model not installed"})
    
    # 启动服务
    def run_thread():
        try:
            # 使用 engine 启动
            from src.core.engine import LLMEngine
            engine = LLMEngine(
                model_path=model_path,
                backend="llama.cpp",
                port=port
            )
            running_models[model_id] = engine
            engine.run()
        except Exception as e:
            if model_id in running_models:
                del running_models[model_id]
            print("Error running model: " + str(e))
    
    thread = threading.Thread(target=run_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "已启动 " + config.name,
        "port": port
    })

@app.route('/api/stop', methods=['POST'])
def stop_model():
    """停止模型服务"""
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({"success": False, "error": "model_id is required"})
    
    if model_id not in running_models:
        return jsonify({"success": False, "error": "Model is not running"})
    
    try:
        engine = running_models[model_id]
        engine.stop()
        del running_models[model_id]
        return jsonify({
            "success": True,
            "message": "已停止模型服务"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/running')
def get_running_models():
    """获取运行中的模型"""
    return jsonify({
        "success": True,
        "data": list(running_models.keys())
    })

# ============================================
# 配置
# ============================================

@app.route('/api/config')
def get_config():
    """获取配置"""
    return jsonify({
        "success": True,
        "data": {
            "model_dir": str(DEFAULT_MODEL_DIR),
            "default_port": DEFAULT_MODEL_DIR
        }
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """更新配置"""
    # TODO: 实现配置保存
    return jsonify({"success": True, "message": "Config saved"})

# ============================================
# 健康检查
# ============================================

@app.route('/api/health')
def health():
    """健康检查"""
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

# ============================================
# 启动
# ============================================

def main():
    port = 7860
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   LightLLM WebUI 已启动                                  ║
║                                                          ║
║   访问地址: http://localhost:{0}                          ║
║                                                          ║
║   按 Ctrl+C 停止服务                                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """.format(port))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    main()