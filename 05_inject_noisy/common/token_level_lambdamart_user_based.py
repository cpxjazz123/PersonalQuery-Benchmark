#!/usr/bin/env python3
"""基于用户真实错误的 LambdaMART Token Ranker

核心区别于旧版：
- 旧版：label 来自 by_token 规则选择结果
- 新版：label 来自用户真实错误中的 corrected token 位置

训练数据构建逻辑：
1. 从 writing_error.json 获取用户真实错误 (original → corrected)
2. 用 corrected 替换 span_text 中的 original，重建 clean span
3. 在 clean span 中，corrected token 位置 = 正样本 (label=1)
4. 其他 token = 负样本 (label=0)

这样训练出来的模型学习的是：
"给定用户错误画像和 token 特征，预测哪个 token 是用户最可能犯错的位置"
"""

import json
import os
import re
from collections import Counter
import numpy as np
import lightgbm as lgb


# ========================================
# 常量
# ========================================
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'im', 'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant',
    'couldnt', 'shouldnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt',
    'havent', 'hadnt', 'hes', 'shes', 'its', 'theyre', 'weve', 'ive',
    'youre', 'thats', 'whats', 'whos', 'wheres', 'whens', 'hows',
    'lets', 'thats', 'theres', 'heres', 'whys', 'cuz', 'cause', 'bc',
}

CATEGORY_ATTR_KEYWORDS = {
    'Baby_Products': {
        'colors': {'pink', 'blue', 'white', 'black', 'green', 'purple', 'yellow', 'brown', 'gray', 'grey', 'red', 'orange'},
        'sizes': {'small', 'medium', 'large', 'mini', 'miniature', 'compact', 'tiny', 'big', 'xl', 'xxl', 'xs', 'newborn'},
        'brands': set(),
        'materials': {'plastic', 'metal', 'fabric', 'cotton', 'silicon', 'bamboo', 'glass', 'stainless'},
    },
    'Grocery_and_Gourmet_Food': {
        'flavors': {'chocolate', 'vanilla', 'coffee', 'caramel', 'strawberry', 'mint', 'almond', 'cinnamon', 'spicy', 'sweet', 'salty', 'sour'},
        'roasts': {'light', 'medium', 'dark', 'blonde'},
        'forms': {'whole', 'ground', 'instant', 'whole_bean', 'powder', 'liquid', 'frozen', 'dried'},
        'brands': set(),
    },
    'Pet_Supplies': {
        'species': {'dog', 'cat', 'bird', 'fish', 'hamster', 'rabbit', 'reptile', 'small_pet'},
        'breeds': set(),
        'materials': {'plastic', 'metal', 'fabric', 'wood', 'leather', 'rubber'},
        'brands': set(),
    },
}

PHONETIC_PATTERNS = ['ei', 'ie', 'ough', 'ight', 'tle', 'ple', 'ble', 'dle', 'kel', 'cel']
VISUAL_SIMILAR_PAIRS = [
    ('m', 'n'), ('b', 'd'), ('p', 'q'), ('i', 'l'), ('o', 'e'),
    ('ae', 'e'), ('ve', 'y'), ('our', 'or'), ('ere', 'are'),
]


# ========================================
# 工具函数
# ========================================
def tokenize_query(query: str) -> list:
    """将 query 分词"""
    query_fixed = re.sub(r"(\w)'(\w)", r"\1'\2", query)
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)*", query_fixed)
    return tokens


def remove_stop_words(tokens: list) -> list:
    """删除停用词"""
    result = []
    for token in tokens:
        token_lower = token.lower()
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        if token_lower not in STOP_WORDS:
            result.append(token)
    return result


def char_similarity(s1: str, s2: str) -> float:
    """计算字符集相似度（Jaccard）"""
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def classify_error_type(original: str, corrected: str) -> str:
    """分类错误类型: phonetic, visual, spelling"""
    orig_lower = original.lower()
    corr_lower = corrected.lower()
    for pattern in PHONETIC_PATTERNS:
        if pattern in orig_lower or pattern in corr_lower:
            return 'phonetic'
    for a, b in VISUAL_SIMILAR_PAIRS:
        if (a in orig_lower and b in corr_lower) or (b in orig_lower and a in corr_lower):
            return 'visual'
    return 'spelling'


def get_user_error_words(user_errors: list) -> tuple:
    """从用户错误中提取信息"""
    corrected_words = set()
    error_words = set()
    char_freq = Counter()
    bigram_freq = Counter()
    error_types = {'phonetic': 0, 'visual': 0, 'spelling': 0}

    for err in user_errors:
        original = err.get('original', '').lower()
        corrected = err.get('corrected', '').lower()
        if not original or not corrected:
            continue
        if ' ' not in original.strip() and ' ' not in corrected.strip():
            corrected_words.add(corrected)
            error_words.add(original)
        for c in original:
            if c.isalpha():
                char_freq[c] += 1
        for i in range(len(original) - 1):
            bigram = original[i:i+2]
            bigram_freq[bigram] += 1
        err_type = classify_error_type(original, corrected)
        error_types[err_type] += 1

    return corrected_words, error_words, char_freq, bigram_freq, error_types


def top_k_similarity(query_token: str, error_words: set, k: int = 5) -> tuple:
    """计算 top-k 相似度"""
    if not error_words:
        return 0.0, []
    similarities = [(err_word, char_similarity(query_token, err_word)) for err_word in error_words]
    similarities.sort(key=lambda x: x[1], reverse=True)
    if not similarities:
        return 0.0, []
    return similarities[0][1], similarities[:k]


def contains_error_chars(token: str, char_freq: Counter) -> float:
    """计算常错字符分数"""
    if not char_freq or not token:
        return 0.0
    total_chars = sum(char_freq.values())
    if total_chars == 0:
        return 0.0
    token_chars = Counter(token.lower())
    error_char_count = sum(token_chars.get(c, 0) for c in char_freq)
    total_token_chars = len([c for c in token if c.isalpha()])
    if total_token_chars == 0:
        return 0.0
    error_ratio = error_char_count / total_token_chars
    max_char_freq = max(char_freq.values())
    char_boost = sum(
        (char_freq[c] / max_char_freq) * (token_chars.get(c, 0) / total_token_chars)
        for c in token.lower() if c in char_freq
    )
    return min(error_ratio + char_boost * 0.5, 1.0)


def contains_error_bigrams(token: str, bigram_freq: Counter) -> float:
    """计算常错 bigram 分数"""
    if not bigram_freq or len(token) < 2:
        return 0.0
    total_bigrams = sum(bigram_freq.values())
    if total_bigrams == 0:
        return 0.0
    token_bigrams = [token[i:i+2].lower() for i in range(len(token) - 1)]
    if not token_bigrams:
        return 0.0
    max_bigram_freq = max(bigram_freq.values())
    error_bigram_score = 0.0
    for bg in token_bigrams:
        if bg in bigram_freq:
            freq_ratio = bigram_freq[bg] / max_bigram_freq
            error_bigram_score += freq_ratio
    return min(error_bigram_score / len(token_bigrams), 1.0)


def surface_difficulty(token: str) -> float:
    """计算表层难度"""
    if not token:
        return 0.0
    score = 0.0
    length = len(token)
    score += min(length * 0.05, 0.3)
    if any(c.isdigit() for c in token):
        score += 0.2
    if any(c.isupper() for c in token):
        score += 0.1
    for i in range(len(token) - 1):
        if token[i] == token[i + 1]:
            score += 0.1
            break
    rare_chars = set('wxqz')
    if any(c.lower() in rare_chars for c in token):
        score += 0.1
    vowels = set('aeiouAEIOU')
    has_alternation = False
    for i in range(len(token) - 1):
        c1_is_vowel = token[i] in vowels
        c2_is_vowel = token[i + 1] in vowels
        if c1_is_vowel != c2_is_vowel:
            has_alternation = True
            break
    if not has_alternation and length > 3:
        score += 0.1
    return min(score, 1.0)


def attribute_type_risk(token: str) -> float:
    """计算属性类型风险"""
    if not token:
        return 0.0
    score = 0.0
    if token[0].isupper() and any(c.islower() for c in token):
        score += 0.15
    if any(c.isdigit() for c in token):
        score += 0.2
    colors = {'red', 'blue', 'green', 'white', 'black', 'pink', 'purple',
              'orange', 'yellow', 'brown', 'gray', 'grey', 'gold', 'silver'}
    if token.lower() in colors:
        score += 0.15
    sizes = {'small', 'medium', 'large', 'mini', 'miniature', 'compact',
             'tiny', 'big', 'xl', 'xxl', 'xs'}
    if token.lower() in sizes:
        score += 0.1
    return min(score, 1.0)


def compute_token_features(
    token: str,
    corrected_words: set,
    error_words: set,
    char_freq: Counter,
    bigram_freq: Counter,
    error_types: dict,
    category: str = None,
    user_error_count: int = 0
) -> dict:
    """计算 token 特征"""
    max_sim, _ = top_k_similarity(token, error_words, k=5)
    char_score = contains_error_chars(token, char_freq)
    bigram_score = contains_error_bigrams(token, bigram_freq)
    diff_score = surface_difficulty(token)
    attr_score = attribute_type_risk(token)

    total_errors = sum(error_types.values()) if error_types else 1
    phonetic_ratio = error_types.get('phonetic', 0) / total_errors
    visual_ratio = error_types.get('visual', 0) / total_errors
    spelling_ratio = error_types.get('spelling', 0) / total_errors

    is_color = 0.0
    is_size = 0.0
    is_brand = 0.0
    is_flavor = 0.0
    is_material = 0.0
    is_species = 0.0

    if category and category in CATEGORY_ATTR_KEYWORDS:
        cat_attrs = CATEGORY_ATTR_KEYWORDS[category]
        token_lower = token.lower()
        if token_lower in cat_attrs.get('colors', set()):
            is_color = 1.0
        if token_lower in cat_attrs.get('sizes', set()):
            is_size = 1.0
        if token_lower in cat_attrs.get('flavors', set()):
            is_flavor = 1.0
        if token_lower in cat_attrs.get('materials', set()):
            is_material = 1.0
        if token_lower in cat_attrs.get('species', set()):
            is_species = 1.0

    if token[0].isupper() and any(c.islower() for c in token) and len(token) > 2:
        is_brand = 1.0

    user_error_rate = min(user_error_count / 10.0, 1.0) if user_error_count else 0.0

    token_len = len(token)
    has_upper = float(any(c.isupper() for c in token))
    has_digit = float(any(c.isdigit() for c in token))
    has_rare_char = float(any(c.lower() in 'wxqz' for c in token))

    has_double_letter = 0.0
    for i in range(len(token) - 1):
        if token[i] == token[i + 1]:
            has_double_letter = 1.0
            break

    vowels = set('aeiouAEIOU')
    vowel_ratio = sum(1 for c in token if c.lower() in vowels) / max(len(token), 1)

    return {
        'topk_sim': max_sim,
        'char_score': char_score,
        'bigram_score': bigram_score,
        'surface_difficulty': diff_score,
        'attr_risk': attr_score,
        'phonetic_ratio': phonetic_ratio,
        'visual_ratio': visual_ratio,
        'spelling_ratio': spelling_ratio,
        'is_color': is_color,
        'is_size': is_size,
        'is_brand': is_brand,
        'is_flavor': is_flavor,
        'is_material': is_material,
        'is_species': is_species,
        'user_error_rate': user_error_rate,
        'token_len': token_len,
        'has_upper': has_upper,
        'has_digit': has_digit,
        'has_rare_char': has_rare_char,
        'has_double_letter': has_double_letter,
        'vowel_ratio': vowel_ratio,
    }


# ========================================
# 工具：JSON 加载（支持不完整的 JSON）
# ========================================
def _load_json_with_fallback(file_path: str) -> list:
    """加载 JSON 文件，支持不完整的 JSON（逐对象解析）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # 逐对象解析
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
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
        return data


# ========================================
# 核心：构建基于用户真实错误的训练数据
# ========================================
def build_training_data_from_writing_errors(
    writing_error_file: str,
    query_file: str,
    category: str
) -> tuple:
    """从 writing_error.json 构建训练数据

    训练数据构建逻辑：
    1. 对于每条错误记录 (original → corrected)
    2. 用 corrected 替换 span_text 中的 original，重建 clean span
    3. Tokenize clean span
    4. Corrected token = 正样本 (label=1)
    5. 其他 token = 负样本 (label=0)
    6. 特征 = token 特征 + 用户画像特征

    Returns:
        (X, y, query_ids, token_texts)
    """
    # 加载用户错误（支持处理不完整的 JSON）
    errors_data = _load_json_with_fallback(writing_error_file)

    # 建立用户画像
    user_profiles = {}
    for user in errors_data:
        uid = user['user_id']
        error_details = user.get('error_details', [])
        if not error_details:
            continue
        user_profiles[uid] = {
            'errors': error_details,
            'error_count': len(error_details)
        }

    # 加载 query 文件
    with open(query_file, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    if isinstance(query_data, dict):
        query_data = query_data.get('records', [])

    # 建立 user_id -> queries 的映射
    user_queries = {}
    for rec in query_data:
        uid = rec.get('user_id')
        if not uid:
            continue
        syntax_q = rec.get('syntax_depth_query', {})
        if isinstance(syntax_q, dict):
            query_text = syntax_q.get('query', '')
        else:
            query_text = ''
        if uid not in user_queries:
            user_queries[uid] = []
        user_queries[uid].append(query_text)

    X = []
    y = []
    query_ids = []
    token_texts = []

    # 对于每个用户的每条错误，构建训练样本
    for uid, profile in user_profiles.items():
        errors = profile['errors']
        user_error_count = profile['error_count']

        # 从错误中提取用户画像特征
        corrected_words, error_words, char_freq, bigram_freq, error_types = get_user_error_words(errors)

        for err in errors:
            original = err.get('original', '')
            corrected = err.get('corrected', '')
            span_text = err.get('span_text', '')

            if not original or not corrected or not span_text:
                continue

            # 用 corrected 替换 span_text 中的 original，构建 clean span
            # 使用 lambda 避免 regex 对 replacement 字符串的转义解释
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            clean_span = pattern.sub(lambda m: corrected, span_text, count=1)

            # Tokenize
            tokens = tokenize_query(clean_span)
            clean_tokens = remove_stop_words(tokens)

            if not clean_tokens:
                continue

            # 在 clean span 中找 corrected 的位置
            corrected_lower = corrected.lower()
            corrected_token_idx = -1
            for idx, token in enumerate(clean_tokens):
                if token.lower() == corrected_lower:
                    corrected_token_idx = idx
                    break

            if corrected_token_idx == -1:
                continue

            # 为每个 token 构建特征
            for idx, token in enumerate(clean_tokens):
                features = compute_token_features(
                    token,
                    corrected_words, error_words, char_freq, bigram_freq,
                    error_types,
                    category=category,
                    user_error_count=user_error_count
                )
                label = 1 if idx == corrected_token_idx else 0

                X.append(features)
                y.append(label)
                query_ids.append((uid, original))
                token_texts.append(token)

    return X, y, query_ids, token_texts


def train_lambdamart(X: list, y: list, query_ids: list) -> tuple:
    """训练 LambdaMART 模型"""
    if not X:
        raise ValueError("X is empty")

    feature_names = list(X[0].keys()) if isinstance(X[0], dict) else [f"f{i}" for i in range(len(X[0]))]
    if isinstance(X[0], dict):
        X_matrix = np.array([[x.get(f, 0.0) for f in feature_names] for x in X])
    else:
        X_matrix = np.array(X)

    # 按 query_id 分组
    query_start = {}
    query_len = {}
    for qid in query_ids:
        if qid not in query_start:
            query_start[qid] = len(query_len)
            query_len[qid] = 0
        query_len[qid] += 1

    group = [query_len[qid] for qid in sorted(query_start.keys())]
    y_array = np.array(y, dtype=np.float32)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3, 5],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
    }

    train_data = lgb.Dataset(
        X_matrix,
        label=y_array,
        group=group,
        feature_name=feature_names,
    )

    model = lgb.train(params, train_data, num_boost_round=100)
    return model, feature_names


# ========================================
# 预测
# ========================================
def predict_token_scores(model, tokens: list, user_profile: dict, category: str) -> list:
    """为每个 token 预测错误分数"""
    if not tokens:
        return []

    corrected_words, error_words, char_freq, bigram_freq, error_types = get_user_error_words(
        user_profile.get('errors', [])
    )
    user_error_count = user_profile.get('error_count', 0)

    X = []
    for token in tokens:
        features = compute_token_features(
            token, corrected_words, error_words, char_freq, bigram_freq,
            error_types, category=category, user_error_count=user_error_count
        )
        X.append([features.get(f, 0.0) for f in model.feature_name()])

    scores = model.predict(X)
    return list(zip(tokens, scores))


# ========================================
# 主流程
# ========================================
def main():
    import glob

    print("=" * 60)
    print("基于用户真实错误的 LambdaMART Token Ranker")
    print("=" * 60)

    output_dir = '/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy'
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    categories = ['Baby_Products', 'Grocery_and_Gourmet_Food', 'Pet_Supplies']

    for category in categories:
        print(f"\n处理类别: {category}")

        writing_error_file = f'/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{category}/writing_error.json'
        query_file_pattern = f'/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{category}/*train10_holdout10*.json'

        query_files = glob.glob(query_file_pattern)
        if not query_files:
            query_file_pattern = f'/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{category}/*.json'
            query_files = glob.glob(query_file_pattern)
        if not query_files:
            print("  跳过: 未找到 query 文件")
            continue
        query_file = query_files[0]

        if not os.path.exists(writing_error_file):
            print("  跳过: writing_error 文件不存在")
            continue

        print(f"  Writing error 文件: {writing_error_file}")
        print(f"  Query 文件: {query_file}")

        try:
            print("  构建训练数据...")
            X, y, query_ids, token_texts = build_training_data_from_writing_errors(
                writing_error_file, query_file, category
            )
            print(f"    样本数: {len(X)}, 正样本: {sum(y)}, 负样本: {len(y) - sum(y)}")

            if len(X) < 100:
                print(f"  跳过: 样本数太少 ({len(X)})")
                continue

            print("  训练 LambdaMART...")
            model, feature_names = train_lambdamart(X, y, query_ids)

            model_file = os.path.join(model_dir, f'lambdamart_{category}_user_based.json')
            model.save_model(model_file)
            print(f"    模型保存到: {model_file}")

            importance = dict(zip(feature_names, model.feature_importance()))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("    特征重要性 (Top 15):")
            for feat, imp in sorted_imp[:15]:
                print(f"      {feat}: {imp}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == '__main__':
    main()