@echo off
cd /d "%~dp0"
set PYTHONPATH=%cd%
echo Starting LightLLM WebUI...
python src\webui_api.py
pause