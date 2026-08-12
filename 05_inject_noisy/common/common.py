"""LambdaMART 噪声注入脚本通用函数模块（精简版）

仅保留 apply_lambdamart_userbased_noisy.py 依赖的工具函数。
"""
from common_utils import log  # 统一 log 函数
import json
import os
import re
from datetime import datetime


# ========================================
# 用户错误数据处理
# ========================================
def load_user_errors(error_file: str) -> dict:
    """加载用户错误数据 - writing_error.json 格式（支持读取不完整的 JSON）"""
    if not os.path.exists(error_file):
        raise FileNotFoundError(f"错误文件不存在: {error_file}")

    try:
        with open(error_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # 如果 JSON 不完整，尝试逐行解析
        data = []
        with open(error_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 尝试找到完整的 JSON 对象
            depth = 0
            start = 0
            for i, c in enumerate(content):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(content[start:i+1])
                            data.append(obj)
                        except json.JSONDecodeError:
                            pass
            if isinstance(data, list) and len(data) > 0 and 'user_results' in data[0]:
                data = data[0]['user_results']
            elif not isinstance(data, list):
                data = []

    if isinstance(data, list):
        users_list = data
    else:
        users_list = data.get('user_results', [])
    user_errors = {}
    for user in users_list:
        uid = user['user_id']
        if user['total_errors'] == 0 or not user.get('error_details'):
            continue
        seen = set()
        all_patterns = []
        for detail in user['error_details']:
            orig = detail.get('original', '')
            corr = detail.get('corrected', '')
            if not orig or not corr:
                continue
            key = (orig, corr)
            if key not in seen:
                seen.add(key)
                all_patterns.append({
                    'original': orig,
                    'corrected': corr,
                    'error_type': 'writing_error',
                })
        if all_patterns:
            user_errors[uid] = {'writing': all_patterns}
    log(f"加载了 {len(user_errors)} 个有错误的用户")
    return user_errors


# ========================================
# 查询记录处理
# ========================================
def load_query_records(file_path: str) -> list:
    """加载查询记录"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"查询文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get('records', [])


def build_query_tasks(records: list, user_errors: dict, completed_keys: set) -> list:
    """构建查询任务"""
    tasks = []
    for record in records:
        uid = record.get('user_id')
        asin = record.get('asin')

        # 提取 query 字符串
        expression_style_query = record.get('expression_style_query')

        clean_query = ''
        query_info = None

        if isinstance(expression_style_query, dict):
            clean_query = expression_style_query.get('query', '')
            query_info = expression_style_query
        else:
            clean_query = record.get('query', '')

        if not uid or not asin or not clean_query:
            continue

        record_key = (uid, asin)
        if record_key in completed_keys:
            continue

        errors_data = user_errors.get(uid, {})
        # load_user_errors 返回格式是 {'writing': [...]}
        if isinstance(errors_data, dict):
            errors = errors_data.get('writing', [])
        else:
            errors = errors_data if isinstance(errors_data, list) else []

        # 提取 attrs_used（如果有）
        attrs_used = None
        if isinstance(expression_style_query, dict):
            attrs_used = expression_style_query.get('attrs_used')
        elif query_info and isinstance(query_info, dict):
            attrs_used = query_info.get('attrs_used')

        tasks.append({
            'uid': uid,
            'asin': asin,
            'clean_query': clean_query,
            'query_info': query_info,
            'errors': errors,
            'attrs_used': attrs_used,
        })
    return tasks


# ========================================
# JSON 数组写入（标准格式）
# ========================================
def write_json_array(items: list, output_file: str, append: bool = True):
    """写入标准 JSON 数组格式（带缩进），支持追加模式"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_items = []
    if append and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            if content.startswith('['):
                try:
                    existing_items = json.loads(content)
                except json.JSONDecodeError:
                    existing_items = []
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            existing_items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    # 去重：基于 uid 和 asin
    existing_keys = {(item.get('uid'), item.get('asin')) for item in existing_items}
    new_items = [item for item in items if (item.get('uid'), item.get('asin')) not in existing_keys]

    all_items = existing_items + new_items

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
