#!/usr/bin/env python3
"""修复Python文件格式问题"""
import os
from pathlib import Path

def fix_py_file(filepath):
    """修复单个Python文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 去除每行末尾的空白字符
    fixed_lines = [line.rstrip() + '\n' for line in lines]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"已修复: {filepath}")

def main():
    root = Path("C:/Users/18844/.hclaw/agents/agt_e491bd55/workspace/projects/LightLLM/src")
    
    for py_file in root.rglob("*.py"):
        try:
            fix_py_file(py_file)
        except Exception as e:
            print(f"错误: {py_file} - {e}")

if __name__ == "__main__":
    main()