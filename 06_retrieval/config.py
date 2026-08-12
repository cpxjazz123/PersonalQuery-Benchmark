"""
08_retrieval 配置加载模块
从 08_retrieval_config.json 读取路径配置
"""
import json
import os
from typing import Dict, Any

_CONFIG_CACHE: Dict[str, Any] = {}

def get_config() -> Dict[str, Any]:
    """获取完整配置（带缓存）"""
    if not _CONFIG_CACHE:
        config_path = os.path.join(os.path.dirname(__file__), '08_retrieval_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            _CONFIG_CACHE.update(json.load(f))
    return _CONFIG_CACHE

def resolve_path(path_template: str) -> str:
    """解析路径模板，将 {key} 替换为实际值"""
    config = get_config()
    result = path_template
    for key in ['test_result', 'result', 'workspace', 'root', 'amazon_reviews', 'scratch_result']:
        placeholder = f'{{{key}}}'
        if placeholder in result:
            result = result.replace(placeholder, config['base_paths'][key])
    return result

def get_category_config(category_name: str) -> Dict[str, Any]:
    """获取指定类别的配置，所有路径都会被解析"""
    config = get_config()
    cat_config = config['categories'].get(category_name, {})
    resolved = {}
    for key, value in cat_config.items():
        if isinstance(value, str):
            resolved[key] = resolve_path(value)
        else:
            resolved[key] = value
    return resolved

def get_global_paths() -> Dict[str, str]:
    """获取全局路径配置"""
    config = get_config()
    return {k: resolve_path(v) for k, v in config['global_paths'].items()}

def get_retriever_config() -> Dict[str, Any]:
    """获取检索器配置"""
    return get_config()['retriever_config']

def list_categories():
    """列出所有配置的类别"""
    return list(get_config()['categories'].keys())
