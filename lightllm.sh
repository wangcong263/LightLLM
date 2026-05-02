#!/bin/bash
# LightLLM Linux/macOS 启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_menu() {
    clear
    echo -e "${BLUE}"
    echo "================================================"
    echo "      LightLLM - 本地大模型运行工具"
    echo "================================================${NC}"
    echo ""
    echo -e "  ${GREEN}[1]${NC}  列出可用模型"
    echo -e "  ${GREEN}[2]${NC}  安装模型"
    echo -e "  ${GREEN}[3]${NC}  删除模型"
    echo -e "  ${GREEN}[4]${NC}  一键部署 (推荐)"
    echo -e "  ${GREEN}[5]${NC}  启动 API 服务"
    echo -e "  ${GREEN}[6]${NC}  命令行聊天"
    echo -e "  ${GREEN}[7]${NC}  退出"
    echo ""
    echo "================================================"
    echo ""
}

list_models() {
    echo ""
    python -m src.model_manager list
    echo ""
    read -p "按回车继续..."
}

install_model() {
    echo ""
    python -m src.model_manager list --all
    echo ""
    read -p "请输入要安装的模型ID: " model_id
    python -m src.model_manager install "$model_id"
    echo ""
    read -p "按回车继续..."
}

remove_model() {
    echo ""
    python -m src.model_manager list
    echo ""
    read -p "请输入要删除的模型ID: " model_id
    python -m src.model_manager remove "$model_id"
    echo ""
    read -p "按回车继续..."
}

deploy() {
    python deploy.py
}

start_api() {
    echo ""
    read -p "请输入模型路径 (直接回车使用配置): " model_path
    if [ -z "$model_path" ]; then
        python -m src.api.server
    else
        python -m src.api.server --model "$model_path"
    fi
}

start_chat() {
    echo ""
    read -p "请输入模型路径 (直接回车使用配置): " model_path
    if [ -z "$model_path" ]; then
        python -m src.cli chat
    else
        python -m src.cli chat --model "$model_path"
    fi
}

# 主循环
while true; do
    show_menu
    read -p "请输入选项 [1-7]: " choice
    
    case $choice in
        1) list_models ;;
        2) install_model ;;
        3) remove_model ;;
        4) deploy ;;
        5) start_api ;;
        6) start_chat ;;
        7) echo ""; echo "再见!"; echo ""; exit 0 ;;
        *) echo ""; echo "无效选项"; sleep 1 ;;
    esac
done