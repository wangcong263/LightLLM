#!/usr/bin/env python3
import os

path = r'C:\Users\18844\.hclaw\agents\agt_e491bd55\workspace\projects\LightLLM\src\model_manager.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 在 def main(): 之前添加两个函数
new_funcs = '''
def list_popular_models() -> List[Dict]:
    """获取热门模型列表（用于WebUI）"""
    return ModelCatalog.get_all_models()

def get_model_info(model_id: str) -> Optional[Dict]:
    """获取模型详细信息（用于WebUI）"""
    config = ModelCatalog.get_model(model_id)
    if config:
        return config.to_dict()
    return None

'''

content = content.replace('\ndef main():', new_funcs + 'def main():')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('已添加函数')