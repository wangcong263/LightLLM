@echo off
chcp 65001 >nul
title LightLLM 模型管理器

echo.
echo ================================================
echo    LightLLM - 本地大模型运行工具
echo ================================================
echo.
echo  [1] 列出可用模型
echo  [2] 安装模型
echo  [3] 删除模型
echo  [4] 一键部署 (推荐)
echo  [5] 启动 API 服务
echo  [6] 命令行聊天
echo  [7] 退出
echo.
echo ================================================
echo.

set /p choice=请输入选项 [1-7]: 

if "%choice%"=="1" goto list
if "%choice%"=="2" goto install
if "%choice%"=="3" goto remove
if "%choice%"=="4" goto deploy
if "%choice%"=="5" goto api
if "%choice%"=="6" goto chat
if "%choice%"=="7" exit

:list
echo.
python -m src.model_manager list
echo.
pause
goto menu

:install
echo.
echo 可用模型:
python -m src.model_manager list --all
echo.
set /p model_id=请输入要安装的模型ID: 
python -m src.model_manager install %model_id%
echo.
pause
goto menu

:remove
echo.
python -m src.model_manager list
echo.
set /p model_id=请输入要删除的模型ID: 
python -m src.model_manager remove %model_id%
echo.
pause
goto menu

:deploy
python deploy.py
goto menu

:api
echo.
set /p model_path=请输入模型路径 (或直接回车使用配置): 
if "%model_path%"=="" (
    python -m src.api.server
) else (
    python -m src.api.server --model "%model_path%"
)
goto menu

:chat
echo.
set /p model_path=请输入模型路径 (或直接回车使用配置): 
if "%model_path%"=="" (
    python -m src.cli chat
) else (
    python -m src.cli chat --model "%model_path%"
)
goto menu

:menu
goto menu