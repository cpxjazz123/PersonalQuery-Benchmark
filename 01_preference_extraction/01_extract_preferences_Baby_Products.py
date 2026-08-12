#!/usr/bin/env python3
"""
Stage 1 v5: 5-Slot Product Attribute Extraction

提取 5 个标准槽位：
- A1: Category (产品类型)
- A2: Brand (品牌)
- A3: Price (价格)
- A4: Appearance (外观：颜色+风格)
- A5: Usage (使用场景：for X)

Input: meta_Baby_Products.jsonl.gz
Output: attributes_Baby_Products.json
"""

import os
import sys
import json
import gzip
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from multiprocessing import Pool, cpu_count


# ============ 匈牙利算法（Kuhn-Munkres）============


def build_user_product_graph(products: List[Dict]) -> Tuple[Dict[str, List[str]], Set[str]]:
    """构建用户-商品二分图"""
    user_to_products = defaultdict(list)
    all_users = set()
    for product in products:
        asin = product.get('asin')
        reviewed_by = product.get('reviewed_by_users', [])
        for user_id in reviewed_by:
            user_to_products[user_id].append(asin)
            all_users.add(user_id)
    return user_to_products, all_users


def hungarian_maximum_matching(
    user_to_products: Dict[str, List[str]],
    all_users: Set[str],
    products_per_user: int = 3,
) -> Dict[str, List[str]]:
    """
    使用 Kuhn 增广路径算法找到最大匹配（匈牙利算法的简化版）。

    两阶段策略：
    1. 第一阶段：优先保证每个用户获得至少1个商品（用户按候选商品数升序排列，优先处理选择少的用户）
    2. 第二阶段：对已获得1个商品的用户，尝试分配第2个商品

    每个商品只匹配给一个用户。

    返回: user_id -> [matched_asin_list]
    """
    user_match: Dict[str, List[str]] = {u: [] for u in all_users}
    product_match: Dict[str, str] = {}  # product -> user (商品只属于一个用户)

    def dfs(user: str, visited: Set[str]) -> bool:
        """尝试为用户分配一个未占用的商品"""
        if user in visited:
            return False
        visited.add(user)

        for product in user_to_products.get(user, []):
            if product in product_match:
                # 商品已被匹配，尝试把该商品当前的用户挤走
                current_user = product_match[product]
                if dfs(current_user, visited.copy()):
                    product_match[product] = user
                    user_match[user].append(product)
                    return True
            else:
                # 商品未被匹配，直接分配
                product_match[product] = user
                user_match[user].append(product)
                return True

        return False

    # ===== 第一阶段：优先保证每个用户获得至少1个商品 =====
    # 按候选商品数升序排列，优先处理选择少的用户（避免它们无法匹配）
    users_by_options = sorted(all_users, key=lambda u: len(user_to_products.get(u, [])))
    for user in users_by_options:
        if user_match[user]:
            continue  # 已有匹配，跳过
        visited = set()
        dfs(user, visited)

    # ===== 第二阶段：对已获得1个商品的用户，尝试分配第2个 =====
    if products_per_user >= 2:
        for user in sorted(all_users):
            if len(user_match[user]) >= 1:
                visited = set()
                dfs(user, visited)

    # ===== 第三阶段：对已获得2个商品的用户，尝试分配第3个 =====
    if products_per_user >= 3:
        for user in sorted(all_users):
            if len(user_match[user]) >= 2:
                visited = set()
                dfs(user, visited)

    return user_match


def run_hungarian_matching(all_results: List[Dict], products_per_user: int = 3) -> Tuple[List[Dict], Dict]:
    """
    对商品列表执行匈牙利算法，返回用户-商品一对多匹配结果

    Args:
        all_results: 商品列表
        products_per_user: 每个用户匹配的商品数量 (默认: 2)

    返回: (matched_results, matching_stats)
    """
    user_to_products, all_users = build_user_product_graph(all_results)

    user_match = hungarian_maximum_matching(user_to_products, all_users, products_per_user=products_per_user)

    matched_users = sum(1 for u in user_match if user_match[u])
    matched_products_set = set()
    for products in user_match.values():
        matched_products_set.update(products)
    unmatched_users = [u for u in sorted(all_users) if not user_match.get(u)]

    product_dict = {p.get('asin'): p for p in all_results}

    matched_results = []
    for user_id in sorted(all_users):
        matched_asins = user_match.get(user_id, [])
        for matched_asin in matched_asins:
            product = product_dict.get(matched_asin)
            if product:
                matched_results.append({'user_id': user_id, 'product': product})

    stats = {
        'total_users': len(all_users),
        'matched_users': matched_users,
        'matched_products': len(matched_products_set),
        'unmatched_users': len(unmatched_users),
        'unmatched_user_list': unmatched_users,
    }

    return matched_results, stats


def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def extract_price(price_str: Any) -> Optional[str]:
    """提取价格"""
    if not price_str:
        return None
    price_str = str(price_str).strip()
    match = re.search(r'\$?([\d,]+\.?\d*)', price_str)
    if match:
        price = match.group(1).replace(',', '')
        # 去掉末尾的句点
        price = price.rstrip('.')
        return price if price else None
    return None


def extract_price_from_text(text: str) -> Optional[str]:
    """从文本（title/description/feature）中提取价格"""
    if not text:
        return None
    # 匹配各种价格格式：$19.99, $1,299.00, 19.99 dollars, 1299 usd 等
    match = re.search(r'\$[\d,]+\.?\d*', text)
    if match:
        price = match.group(0).replace('$', '').replace(',', '')
        # 去掉末尾的句点
        price = price.rstrip('.')
        return price if price else None
    # 匹配纯数字+单位
    match = re.search(r'(\d+\.?\d*)\s*(dollars?|usd)', text, re.IGNORECASE)
    if match:
        price = match.group(1).rstrip('.')
        return price if price else None
    return None


def extract_product_type(category: List) -> Optional[str]:
    """提取产品类型 - 取倒数第一个（最具体的类别），但需要长度合理"""
    import re
    if isinstance(category, list) and len(category) >= 1:
        candidate = category[-1]
        # 过滤掉过长的（可能是描述文本）和过短的
        if candidate and 3 <= len(candidate) <= 50:
            # 如果包含 &，取 & 前后更长的那部分
            if ' & ' in candidate:
                parts = candidate.split(' & ')
                candidate = max(parts, key=len).strip()
            # 去掉末尾标点符号
            candidate = re.sub(r'[,，。.!?;：;]+$', '', candidate).strip()
            return candidate if candidate else None
        # 如果最后一个太长/太短，尝试倒数第二个
        if len(category) >= 2:
            candidate2 = category[-2]
            if candidate2 and 3 <= len(candidate2) <= 50:
                # 如果包含 &，取 & 前后更长的那部分
                if ' & ' in candidate2:
                    parts = candidate2.split(' & ')
                    candidate2 = max(parts, key=len).strip()
                # 去掉末尾标点符号
                candidate2 = re.sub(r'[,，。.!?;：;]+$', '', candidate2).strip()
                return candidate2 if candidate2 else None
    return None


def extract_use_case(title: str, description: str, feature: List) -> Optional[str]:
    """提取使用场景 - 使用扩充关键词列表（优化版）"""
    import re

    # 场景关键词列表（扩充版 - 同时包含带for和不带的版本）
    scene_keywords = [
        # 人物群体
        'for kids', 'for children', 'for beginners', 'for professionals',
        'for men', 'for women', 'for teens', 'for adults', 'for seniors',
        'for students', 'for teachers', 'for artists', 'for babies', 'for toddlers',
        'kids', 'children', 'beginners', 'professionals', 'men', 'women',
        'teens', 'adults', 'seniors', 'students', 'teachers', 'artists',
        # 婴儿/儿童相关
        'for baby', 'for infant', 'for newborn', 'for toddler',
        'baby', 'infant', 'newborn', 'toddler',
        # 场所/地点
        'for home', 'for office', 'for school', 'for classroom', 'for studio',
        'for workshop', 'for outdoor', 'for indoor', 'for garden', 'for bedroom',
        'for kitchen', 'for bathroom', 'for garage', 'for playroom',
        'home', 'office', 'school', 'classroom', 'studio', 'workshop',
        'outdoor', 'indoor', 'garden', 'bedroom', 'kitchen', 'bathroom',
        # 活动/用途
        'for travel', 'for camping', 'for hiking', 'for commuting',
        'for exercise', 'for fitness', 'for sport', 'for sports', 'for running',
        'for cycling', 'for swimming', 'for yoga', 'for meditation', 'for painting',
        'for drawing', 'for sewing', 'for crafting', 'for scrapbooking',
        'for jewelry', 'for knitting', 'for crocheting', 'for embroidery',
        'for quilting', 'for woodworking', 'for gardening', 'for cooking',
        'for baking', 'for photography', 'for writing', 'for journaling',
        'travel', 'camping', 'hiking', 'commuting', 'exercise', 'fitness',
        'sport', 'sports', 'running', 'cycling', 'swimming', 'yoga', 'meditation',
        'painting', 'drawing', 'sewing', 'crafting', 'scrapbooking', 'jewelry',
        'knitting', 'crocheting', 'embroidery', 'quilting', 'woodworking',
        'gardening', 'cooking', 'baking', 'photography', 'writing', 'journaling',
        # 场合/事件
        'for gift', 'for party', 'for wedding', 'for birthday', 'for christmas',
        'for holiday', 'for seasonal', 'for easter', 'for halloween',
        'for thanksgiving', 'for valentine', 'for anniversary',
        'gift', 'party', 'wedding', 'birthday', 'christmas', 'holiday',
        'seasonal', 'easter', 'halloween', 'thanksgiving', 'valentine',
        # 组织/存储
        'for organizing', 'for storage', 'for display', 'for presentation',
        'organizing', 'storage', 'display', 'presentation',
        # 婴儿用品常见用途
        'for feeding', 'for nursing', 'for bathing', 'for sleeping',
        'for diaper changing', 'for potty training', 'for teething',
        'feeding', 'nursing', 'bathing', 'sleeping', 'diaper', 'potty', 'teething',
        # 儿童玩具/教育
        'for playing', 'for learning', 'for development', 'for entertainment',
        'playing', 'learning', 'development', 'entertainment',
    ]

    # 场景关键词到输出标签的映射
    scene_map = {
        # 人物群体
        'for kids': 'Kids', 'for children': 'Children', 'for beginners': 'Beginners',
        'for professionals': 'Professional', 'for men': 'Men', 'for women': 'Women',
        'for teens': 'Teens', 'for adults': 'Adults', 'for seniors': 'Seniors',
        'for students': 'Students', 'for teachers': 'Teachers', 'for artists': 'Artists',
        'kids': 'Kids', 'children': 'Children', 'beginners': 'Beginners',
        'professionals': 'Professional', 'men': 'Men', 'women': 'Women',
        'teens': 'Teens', 'adults': 'Adults', 'seniors': 'Seniors',
        'students': 'Students', 'teachers': 'Teachers', 'artists': 'Artists',
        # 婴儿/儿童
        'for baby': 'Baby', 'for infant': 'Infant', 'for newborn': 'Newborn', 'for toddlers': 'Toddler',
        'baby': 'Baby', 'infant': 'Infant', 'newborn': 'Newborn', 'toddler': 'Toddler',
        # 场所/地点
        'for home': 'Home', 'for office': 'Office', 'for school': 'School',
        'for classroom': 'Classroom', 'for studio': 'Studio', 'for workshop': 'Workshop',
        'for outdoor': 'Outdoor', 'for indoor': 'Indoor', 'for garden': 'Garden',
        'for bedroom': 'Bedroom', 'for kitchen': 'Kitchen', 'for bathroom': 'Bathroom',
        'for garage': 'Garage', 'for playroom': 'Playroom',
        'home': 'Home', 'office': 'Office', 'school': 'School',
        'classroom': 'Classroom', 'studio': 'Studio', 'workshop': 'Workshop',
        'outdoor': 'Outdoor', 'indoor': 'Indoor', 'garden': 'Garden',
        'bedroom': 'Bedroom', 'kitchen': 'Kitchen', 'bathroom': 'Bathroom',
        'garage': 'Garage', 'playroom': 'Playroom',
        # 活动/用途
        'for travel': 'Travel', 'for camping': 'Camping', 'for hiking': 'Hiking',
        'for commuting': 'Commuting', 'for exercise': 'Exercise', 'for fitness': 'Fitness',
        'for sport': 'Sports', 'for sports': 'Sports', 'for running': 'Running',
        'for cycling': 'Cycling', 'for swimming': 'Swimming', 'for yoga': 'Yoga',
        'for meditation': 'Meditation', 'for painting': 'Painting', 'for drawing': 'Drawing',
        'for sewing': 'Sewing', 'for crafting': 'Crafting', 'for scrapbooking': 'Scrapbooking',
        'for jewelry': 'Jewelry', 'for knitting': 'Knitting', 'for crocheting': 'Crocheting',
        'for embroidery': 'Embroidery', 'for quilting': 'Quilting', 'for woodworking': 'Woodworking',
        'for gardening': 'Gardening', 'for cooking': 'Cooking', 'for baking': 'Baking',
        'for photography': 'Photography', 'for writing': 'Writing', 'for journaling': 'Journaling',
        'travel': 'Travel', 'camping': 'Camping', 'hiking': 'Hiking',
        'commuting': 'Commuting', 'exercise': 'Exercise', 'fitness': 'Fitness',
        'sport': 'Sports', 'sports': 'Sports', 'running': 'Running',
        'cycling': 'Cycling', 'swimming': 'Swimming', 'yoga': 'Yoga',
        'meditation': 'Meditation', 'painting': 'Painting', 'drawing': 'Drawing',
        'sewing': 'Sewing', 'crafting': 'Crafting', 'scrapbooking': 'Scrapbooking',
        'jewelry': 'Jewelry', 'knitting': 'Knitting', 'crocheting': 'Crocheting',
        'embroidery': 'Embroidery', 'quilting': 'Quilting', 'woodworking': 'Woodworking',
        'gardening': 'Gardening', 'cooking': 'Cooking', 'baking': 'Baking',
        'photography': 'Photography', 'writing': 'Writing', 'journaling': 'Journaling',
        # 场合/事件
        'for gift': 'Gift', 'for party': 'Party', 'for wedding': 'Wedding',
        'for birthday': 'Birthday', 'for christmas': 'Christmas', 'for holiday': 'Holiday',
        'for seasonal': 'Seasonal', 'for easter': 'Easter', 'for halloween': 'Halloween',
        'for thanksgiving': 'Thanksgiving', 'for valentine': 'Valentine', 'for anniversary': 'Anniversary',
        'gift': 'Gift', 'party': 'Party', 'wedding': 'Wedding', 'birthday': 'Birthday',
        'christmas': 'Christmas', 'holiday': 'Holiday', 'seasonal': 'Seasonal',
        'easter': 'Easter', 'halloween': 'Halloween', 'thanksgiving': 'Thanksgiving',
        'valentine': 'Valentine', 'anniversary': 'Anniversary',
        # 婴儿用品
        'for feeding': 'Feeding', 'for nursing': 'Nursing', 'for bathing': 'Bathing',
        'for sleeping': 'Sleeping', 'for diaper changing': 'Diaper Changing',
        'for potty training': 'Potty Training', 'for teething': 'Teething',
        'feeding': 'Feeding', 'nursing': 'Nursing', 'bathing': 'Bathing',
        'sleeping': 'Sleeping', 'diaper': 'Diaper', 'potty': 'Potty', 'teething': 'Teething',
        # 儿童玩具/教育
        'for playing': 'Playing', 'for learning': 'Learning', 'for development': 'Development',
        'for entertainment': 'Entertainment',
        'playing': 'Playing', 'learning': 'Learning', 'development': 'Development',
        'entertainment': 'Entertainment',
    }

    # 清理文本
    text = f"{title} {description}".lower()
    if isinstance(feature, list):
        text = f"{text} {' '.join(str(f) for f in feature)}"

    # 清理 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)

    # 快速字符串匹配 - 按长度降序排列优先匹配更长的词
    for keyword in sorted(scene_keywords, key=len, reverse=True):
        if keyword in text:
            result = keyword
            # 去掉 "for " 前缀（如果有）
            if result.startswith('for '):
                result = result[4:]
            return result.capitalize()

    # 匹配 "used for" 或 "be used for" 模式，提取后面的内容作为场景
    import re
    # 匹配 "used for XXX" 或 "be used for XXX"
    used_for_match = re.search(r'(?:be\s+)?used\s+for\s+([a-zA-Z][a-zA-Z\s]+?)(?:\s|,|\.|$)', text)
    if used_for_match:
        activity = used_for_match.group(1).strip()
        # 清理并标准化
        activity = re.sub(r'\s+', ' ', activity).strip()
        if activity and len(activity) >= 3 and len(activity) <= 30:
            # 标准化一些常见活动名称
            activity_map = {
                'doing': None, 'making': 'Making', 'creating': 'Creating',
                'decorating': 'Decorating', 'decoration': 'Decorating',
                'cutting': 'Cutting', 'shaping': 'Shaping', 'carving': 'Carving',
                'painting': 'Painting', 'drawing': 'Drawing', 'sewing': 'Sewing',
                'crafting': 'Crafting', 'scrapbooking': 'Scrapbooking',
                'candle': 'Candle Making', 'soap': 'Soap Making',
                'cake': 'Cake Decorating', 'clay': 'Clay Craft',
                'all kinds of': None, 'various': None,
            }
            if activity in activity_map:
                return activity_map[activity]
            # 返回首字母大写的结果
            return activity.capitalize()

    return None


def extract_features(title: str, feature: List, tech1: str) -> List[str]:
    """提取产品特性"""
    features = []

    if isinstance(feature, list):
        for f in feature:
            f_str = str(f).strip()
            if f_str and len(f_str) > 3 and len(f_str) < 200:
                # 清理 HTML 标签
                f_str = re.sub(r'<[^>]+>', '', f_str).strip()
                if f_str:
                    features.append(f_str)

    return features[:5]


# ============ 预编译的关键词（只保留A4需要的）============
import re as re_module

# 表面特性
SURFACE_SINGLE = {
    'smooth', 'polished', 'glossy', 'matte', 'textured', 'grooved',
    'non-slip', 'anti-slip', 'soft', 'shiny', 'bright', 'opaque',
    'sheen', 'glow', 'silky', 'luster', 'lustrous', 'rough', 'slick',
    'fleecy', 'fuzzy', 'sheer', 'gloss', 'dull', 'frosted',
}
SURFACE_MULTI = [re_module.compile(k) for k in ['non-stick', 'nonstick']]

# 颜色
COLOR_SINGLE = {
    'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
    'pink', 'brown', 'gray', 'grey', 'silver', 'gold', 'clear', 'transparent',
    'ivory', 'cream', 'tan', 'coral', 'lavender', 'mint', 'turquoise',
    'burgundy', 'maroon', 'olive', 'khaki', 'navy', 'jet', 'ruby', 'emerald',
    'crystal', 'pearl', 'amber', 'bronze', 'copper', 'brass',
}
COLOR_MULTI_RE = re_module.compile(r'\b(rose gold|antique silver|antiqued brass|antique bronze|antiqued silver|crystal clear|royal blue|light blue|antique gold|emerald green|charcoal grey|jet black|ruby red|aqua blue|silver tone|golden yellow|antique copper|silver plated|gold plated|navy blue|sky blue|sea green|forest green|lime green|hot pink|bright red|dark blue|pale pink|midnight blue|pastel blue|pastel pink|pastel green|neon yellow|neon pink|neon green|neon orange|neon blue|military green|army green|pearl white|pearl pink)\b')

# 形状/款式
STYLE_SINGLE = {
    'vintage', 'retro', 'antique', 'modern', 'classic', 'rustic',
    'bohemian', 'minimalist', 'elegant', 'glamorous', 'sleek', 'cute',
    'adorable', 'delicate', 'exquisite', 'handmade', 'craftsman',
    'long', 'short', 'mini', 'tiny', 'large', 'small', 'big', 'thick', 'thin',
    'round', 'square', 'oval', 'flat', 'curved', 'slim', 'compact', 'portable',
}
STYLE_MULTI_RE = re_module.compile(r'\b(heart-shaped|heart shape|star-shaped|star shape|round|l-shaped| u-shaped| o-shaped| t-shaped)\b')


def _extract_keyword_set(single_set: set, multi_re_list: list, text: str) -> List[str]:
    """从文本中提取关键词 - 单个词用集合匹配，多个词用正则匹配"""
    found = []
    # 单个词匹配（直接用集合操作，速度快）
    text_lower = text.lower()
    words = set(text_lower.split())
    for kw in single_set:
        if kw in words:
            found.append(kw.capitalize())
    # 多词匹配用正则
    for re_pat in multi_re_list:
        match = re_pat.search(text)
        if match:
            found.append(match.group(0).capitalize())
    return list(dict.fromkeys(found))


def extract_structured_features(feature: List, title: str = "") -> Dict[str, List[str]]:
    """从 features 中拆分出结构化属性（只提取A4：颜色+表面特性）"""
    # 合并所有 feature 文本
    all_text = ' '.join(str(f) for f in feature) if isinstance(feature, list) else str(feature)
    all_text = re_module.sub(r'<[^>]+>', ' ', all_text)

    # 加入 title 到提取来源
    if title:
        all_text = title + ' ' + all_text

    text_lower = all_text.lower()

    # A4_appearance：只保留真正的颜色词，表面特性和复合颜色
    appearance_vals = []

    # 无效词列表（不能作为外观属性值的词）
    # 注意：vintage, elegant, cute, delicate, exquisite 等现在是有效的款式关键词，不再过滤
    invalid_appearance_words = {
        'color', 'colour', 'colors', 'colours', 'mixed', 'assorted',
        'beautiful', 'simple', 'decorative', 'pretty', 'lovely', 'charming',
        'stylish', 'fancy', 'ornamental', 'random', 'picture', 'show',
        'none', 'all', 'various', 'different', 'multiple', 'set',
        'photo', 'per', 'see', 'like', 'exactly', 'actual', 'match', 'display',
    }

    # 排除包含这些模式的颜色值
    invalid_patterns = [
        'as photo', 'as the picture', 'as per', 'as shown', 'as your',
        'as i', 'not ', 'not-', '-not', 'please', 'review',
        'pack', 'pc ', ' pc', 'set of', 'one pack', 'each', 'approx',
        ':', ';', 'main material', 'thickness',
    ]

    # 1. 从 feature 的结构化 Color: 字段提取
    if isinstance(feature, list):
        for feat in feature:
            feat_str = str(feat)
            # 匹配 Color: 或 Colour: 开头的内容
            match = re_module.search(r'[Cc]olor:\s*([^<,]+)', feat_str)
            if match:
                color_str = match.group(1).strip()
                # 按 / 分割成多个颜色
                color_parts = [c.strip() for c in color_str.split('/')]
                for color in color_parts:
                    # 过滤掉不合适的内容
                    if color and len(color) >= 2 and len(color) < 50:
                        # 转小写检查是否包含无效词
                        color_lower = color.lower()
                        if not any(inv in color_lower for inv in invalid_appearance_words):
                            # 检查是否包含无效模式
                            if any(pat in color_lower for pat in invalid_patterns):
                                continue
                            # 去除括号及其内容，如 "Purple (amethyst)" -> "Purple"
                            color = re_module.sub(r'\s*\([^)]*\)', '', color).strip()
                            if not color:
                                continue
                            # 简化复合颜色词
                            color = color.replace('Antique ', 'Antique ').replace('Bright ', '').replace('Dull ', '')
                            appearance_vals.append(color.capitalize())

    # 2. 从关键词匹配颜色（用预编译正则）
    color_found = _extract_keyword_set(COLOR_SINGLE, [COLOR_MULTI_RE], text_lower)
    appearance_vals = list(dict.fromkeys(appearance_vals + color_found))

    # 3. 合并表面特性
    surface_found = _extract_keyword_set(SURFACE_SINGLE, SURFACE_MULTI, text_lower)
    appearance_vals = list(dict.fromkeys(appearance_vals + surface_found))

    # 4. 合并形状/款式关键词
    style_found = _extract_keyword_set(STYLE_SINGLE, [STYLE_MULTI_RE], text_lower)
    appearance_vals = list(dict.fromkeys(appearance_vals + style_found))

    # 4. 如果有复合颜色词包含基础颜色词（作为独立词），则过滤掉那个基础颜色词
    # 例如：已有 "Royal blue"，则过滤掉 "Blue"；已有 "Silver plated"，则过滤掉 "Silver"
    basic_colors = {
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
        'pink', 'brown', 'gray', 'grey', 'silver', 'gold', 'clear', 'transparent',
        'ivory', 'cream', 'tan', 'coral', 'lavender', 'mint',
        'turquoise', 'burgundy', 'maroon', 'olive', 'khaki', 'navy',
        'jet', 'ruby', 'emerald', 'sapphire', 'amethyst', 'pearl', 'brass', 'copper', 'bronze',
    }
    compound_words = [v for v in appearance_vals if ' ' in v]  # 复合词（包含空格）

    # 提取所有复合词中的单词
    compound_parts = set()
    for compound in compound_words:
        for word in compound.lower().split():
            compound_parts.add(word)

    # 如果 appearance_vals 中的词是某个复合词的组成部分（但不是那个复合词本身），则去掉它
    to_remove = set()
    for val in appearance_vals:
        val_lower = val.lower()
        # 如果这个词是复合词的组成部分但不是复合词本身，且是基础颜色词或金属词
        if val_lower in compound_parts and val_lower not in [cw.lower() for cw in compound_words]:
            if val_lower in basic_colors:
                to_remove.add(val)
        # 如果同时有 Bright 和 Bright red，只保留 Bright red（去掉 Bright）
        if val_lower == 'bright' and any('bright ' in cw.lower() for cw in compound_words):
            to_remove.add(val)

    for val in to_remove:
        if val in appearance_vals:
            appearance_vals.remove(val)

    # 5. 如果有多个属性值，只保留最长的那个
    if len(appearance_vals) > 1:
        appearance_vals = [max(appearance_vals, key=len)]

    # 6. 清理末尾标点符号
    appearance_vals = [re_module.sub(r'[,，。.!?;：;]+$', '', v).strip() for v in appearance_vals]
    appearance_vals = [v for v in appearance_vals if v]  # 过滤空值

    return {
        'A4_appearance': appearance_vals,  # 颜色+表面特性
    }


def extract_description_attributes(description) -> Dict[str, List[str]]:
    """从 description 中提取结构化属性"""
    import re

    # 如果 description 是列表，合并
    if isinstance(description, list):
        description = ' '.join(str(d) for d in description)

    # 清理 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', str(description))
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 20:
        return {
            'A18_quality': [],
        }

    # 按句子拆分（用句号、问号、感叹号分隔）
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    sentences = sentences[:5]  # 最多5个句子

    # 质量相关关键词
    quality_keywords = [
        'high quality', 'premium', 'professional', 'durable', 'sturdy',
        'excellent', 'superior', 'top quality', 'best', 'fine', 'perfect',
        'reliable', 'solid', 'heavy duty', 'industrial',
    ]

    # 外观相关（颜色+风格）
    appearance_keywords = [
        # 颜色
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
        'pink', 'brown', 'gray', 'grey', 'silver', 'gold', 'clear', 'transparent',
        'neon', 'pastel', 'vibrant', 'multicolor', 'assorted',
        # 风格
        'beautiful', 'elegant', 'stylish', 'modern', 'classic', 'vintage',
        'decorative', 'ornamental', 'cute', 'pretty', 'lovely', 'charming',
        'minimalist', 'simple', 'fancy', 'delicate', 'exquisite',
    ]

    # 提取函数
    def extract_keywords(keywords: List[str], text: str) -> List[str]:
        found = []
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(kw.capitalize())
        return list(dict.fromkeys(found))

    # 检测品牌提及（首字母大写的连续单词，可能是品牌）
    brand_mentions = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b', text)
    # 过滤掉常见非品牌词
    common_words = {'The', 'This', 'With', 'That', 'Your', 'Also', 'From', 'Have', 'Will', 'Would', 'Could', 'Should', 'There', 'Here', 'What', 'When', 'Where', 'Which', 'Who', 'Why', 'How', 'All', 'For', 'And', 'But', 'Not', 'You', 'Are', 'Can', 'May', 'Has', 'Was', ' Were', 'One', 'Two', 'Three', 'Four', 'Five'}
    brand_mentions = [b for b in brand_mentions if b not in common_words][:3]

    # 提取外观和风格
    appearance_found = extract_keywords(appearance_keywords, text)

    return {
        'A4_appearance': appearance_found,  # 包含颜色+风格
        'A18_quality': extract_keywords(quality_keywords, text),
    }


def extract_domain_specific_baby(item: Dict) -> Dict[str, Any]:
    """
    提取 Baby Products 领域特定的详细属性 (A6)
    包含: Target Species, Age Range, Size, Material, Color 等
    """
    details = item.get('details', {})
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            details = {}

    a6_fields = {}

    # Target Species - 目标物种
    target = details.get('Target Species', '')
    if target:
        a6_fields['target_species'] = str(target).strip()

    # Age Range - 年龄范围
    age = details.get('Age Range', '') or details.get('Age Range Description', '')
    if age:
        a6_fields['age_range'] = str(age).strip()

    # Size - 尺寸
    size = details.get('Size', '')
    if size:
        a6_fields['size'] = str(size).strip()

    # Material - 材料
    material = details.get('Material', '')
    if material:
        a6_fields['material'] = str(material).strip()

    # Color - 颜色
    color = details.get('Color', '')
    if color:
        a6_fields['color'] = str(color).strip()

    # Item Form - 商品形态
    item_form = details.get('Item Form', '')
    if item_form:
        a6_fields['item_form'] = str(item_form).strip()

    # Special Feature - 特殊功能
    special = details.get('Special Feature', '')
    if special:
        a6_fields['special_feature'] = str(special).strip()

    # Fit Type - 适合类型
    fit = details.get('Fit Type', '')
    if fit:
        a6_fields['fit_type'] = str(fit).strip()

    # Theme - 主题
    theme = details.get('Theme', '')
    if theme:
        a6_fields['theme'] = str(theme).strip()

    # Brand - 品牌（冗余但有用）
    brand = details.get('Brand', '')
    if brand:
        a6_fields['brand'] = str(brand).strip()

    return a6_fields


def extract_extended_attributes(item: Dict, description: str, feature: List) -> Dict[str, List[str]]:
    """
    提取 A7-A17 扩展属性槽位
    A7: Material, A8: Safety, A9: Durability, A10: Ease_of_use,
    A11: Temperature_resistance, A12: Surface, A13: Reusability,
    A14: Size, A15: Weight, A16: Compatibility, A17: Flavor
    """
    details = item.get('details', {})
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            details = {}

    result = {
        'A7_material': [],
        'A8_safety': [],
        'A9_durability': [],
        'A10_ease_of_use': [],
        'A11_temperature_resistance': [],
        'A12_surface': [],
        'A13_reusability': [],
        'A14_size': [],
        'A15_weight': [],
        'A16_compatibility': [],
        'A17_flavor': [],
    }

    # A7: Material - 从 details.Material 提取
    material = details.get('Material', '')
    if material:
        result['A7_material'] = [str(material).strip()]

    # A8: Safety - 从 description 提取安全相关关键词
    safety_keywords = ['safe', 'safety', 'non-toxic', 'nontoxic', 'hypoallergenic', 'bpa-free', 'bpa free',
                       'lead-free', 'lead free', 'phthalate-free', 'flame-retardant', 'eco-friendly',
                       'organic', 'natural', 'chemical-free', 'chemical free', 'Toxic free', 'toxin-free']
    desc_lower = description.lower() if description else ''
    found_safety = []
    for kw in safety_keywords:
        if kw in desc_lower:
            found_safety.append(kw.replace('-', ' ').title())
    if found_safety:
        result['A8_safety'] = list(dict.fromkeys(found_safety))

    # A9: Durability - 从 description 提取耐用性关键词
    durability_keywords = ['durable', 'durability', 'long-lasting', 'long lasting', 'heavy duty', 'sturdy',
                          'strong', 'reinforced', 'robust', 'solid', 'high quality', 'premium']
    found_durability = []
    for kw in durability_keywords:
        if kw in desc_lower:
            found_durability.append(kw.replace('-', ' ').title())
    if found_durability:
        result['A9_durability'] = list(dict.fromkeys(found_durability))

    # A10: Ease of use - 从 description 提取易用性关键词
    ease_keywords = ['easy', 'easy to use', 'simple', 'convenient', 'user-friendly', 'intuitive',
                    'effortless', 'quick', 'hassle-free', 'ready to use', 'pre-assembled', 'easy to clean']
    found_ease = []
    for kw in ease_keywords:
        if kw in desc_lower:
            found_ease.append(kw.replace('-', ' ').title())
    if found_ease:
        result['A10_ease_of_use'] = list(dict.fromkeys(found_ease))

    # A11: Temperature resistance - 从 description/details 提取耐温性
    temp_keywords = ['heat resistant', 'heat-resistant', 'cold resistant', 'cold-resistant', 'waterproof',
                    'weatherproof', 'uv resistant', 'fade-resistant', 'heat tolerance']
    found_temp = []
    for kw in temp_keywords:
        if kw in desc_lower:
            found_temp.append(kw.replace('-', ' ').title())
    if found_temp:
        result['A11_temperature_resistance'] = list(dict.fromkeys(found_temp))

    # A12: Surface - 表面特性（已部分在 A4_appearance，这里补充）
    surface_keywords = ['smooth', 'textured', 'anti-slip', 'non-slip', 'rubberized', 'padded', 'mesh']
    found_surface = []
    for kw in surface_keywords:
        if kw in desc_lower:
            found_surface.append(kw.replace('-', ' ').title())
    if found_surface:
        result['A12_surface'] = list(dict.fromkeys(found_surface))

    # A13: Reusability - 从 description 提取可重复使用性
    reuse_keywords = ['reusable', 'washable', 'disposable', 'one-time use', 'multi-use', 'refillable',
                      'recyclable', 'eco-friendly', 'sustainable', 'biodegradable']
    found_reuse = []
    for kw in reuse_keywords:
        if kw in desc_lower:
            found_reuse.append(kw.replace('-', ' ').title())
    if found_reuse:
        result['A13_reusability'] = list(dict.fromkeys(found_reuse))

    # A14: Size - 从 details.Size 等提取
    size_fields = ['Size', 'Item Dimensions LxWxH', 'Package Dimensions', 'Product Dimensions']
    found_size = []
    for field in size_fields:
        val = details.get(field, '')
        if val:
            found_size.append(str(val).strip())
    if found_size:
        result['A14_size'] = found_size[:3]

    # A15: Weight - 从 details.Item Weight 提取
    weight = details.get('Item Weight', '')
    if weight:
        result['A15_weight'] = [str(weight).strip()]

    # A16: Compatibility - 从 Target Species, Age Range, Target Audience 提取
    compat_fields = ['Target Species', 'Age Range', 'Age Range Description', 'Target Audience', 'Sport']
    found_compat = []
    for field in compat_fields:
        val = details.get(field, '')
        if val:
            found_compat.append(str(val).strip())
    if found_compat:
        result['A16_compatibility'] = found_compat[:3]

    # A17: Flavor - 从 details.Flavor 提取
    flavor = details.get('Flavor', '')
    if flavor:
        result['A17_flavor'] = [str(flavor).strip()]

    return result


def extract_attributes(item: Dict) -> Dict:
    """从商品元数据提取多个槽位"""
    # 支持 2018 格式 (asin, brand, category) 和 2023 格式 (parent_asin, details.Brand, categories)
    asin = item.get('asin') or item.get('parent_asin', '')
    title = item.get('title', '')
    # brand: 2018 直接取 brand，2023 从 details.Brand 获取
    brand = item.get('brand', '')
    if not brand:
        details = item.get('details', {})
        if isinstance(details, dict):
            brand = details.get('Brand', '')
        elif isinstance(details, str):
            try:
                details_dict = json.loads(details)
                brand = details_dict.get('Brand', '')
            except Exception:
                brand = ''
    price = item.get('price', '')
    description = item.get('description', '')
    if isinstance(description, list):
        description = ' '.join(str(d) for d in description)
    feature = item.get('feature', [])
    # category: 2018 是列表，2023 是普通字符串列表
    category = item.get('category', [])
    if not category:
        categories = item.get('categories', [])
        if categories:
            category = [categories[-1]] if categories else []
    tech1 = item.get('tech1', '')

    # 提取结构化特征
    structured = extract_structured_features(feature, title)

    # 从 description 中提取关键属性
    desc_structured = extract_description_attributes(description)

    # 合并 A4_appearance（避免 desc_structured 覆盖 structured 的值）
    # 但desc_structured中包含很多无效词，只使用structured的值
    # structured_a4 = structured.get('A4_appearance', [])
    # desc_a4 = desc_structured.get('A4_appearance', [])
    # if desc_a4:
    #     # 合并去重
    #     combined_a4 = list(dict.fromkeys(structured_a4 + desc_a4))
    #     structured['A4_appearance'] = combined_a4
    pass  # 不再从description合并A4，因为其中包含大量无效词

    # 合并 A18_quality（同样避免覆盖）
    structured_a18 = structured.get('A18_quality', [])
    desc_a18 = desc_structured.get('A18_quality', [])
    if desc_a18:
        combined_a18 = list(dict.fromkeys(structured_a18 + desc_a18))
        structured['A18_quality'] = combined_a18

    # 提取 Baby Products 领域特定属性
    domain_specific = extract_domain_specific_baby(item)

    # 提取 A7-A17 扩展属性
    extended = extract_extended_attributes(item, description, feature)

    return {
        'asin': asin,
        'A1_product_type': extract_product_type(category),
        'A2_brand': brand if brand and isinstance(brand, str) and len(str(brand).strip()) > 0 else None,
        'A3_price': extract_price(price),
        'A4_appearance': structured.get('A4_appearance', []),
        'A5_use_case': extract_use_case(title, description, feature),
        'A6_detailed': domain_specific,
        'A7_material': extended['A7_material'],
        'A8_safety': extended['A8_safety'],
        'A9_durability': extended['A9_durability'],
        'A10_ease_of_use': extended['A10_ease_of_use'],
        'A11_temperature_resistance': extended['A11_temperature_resistance'],
        'A12_surface': extended['A12_surface'],
        'A13_reusability': extended['A13_reusability'],
        'A14_size': extended['A14_size'],
        'A15_weight': extended['A15_weight'],
        'A16_compatibility': extended['A16_compatibility'],
        'A17_flavor': extended['A17_flavor'],
        'A18_quality': structured.get('A18_quality', []),
    }


def process_item(item: Dict) -> Optional[Dict]:
    """处理单个商品，返回属性字典或None（如果不满足条件）"""
    # 支持 2018 格式 (asin, brand, category) 和 2023 格式 (parent_asin, details.Brand, categories)
    asin = item.get('asin') or item.get('parent_asin', '')

    # brand: 2018 直接取 brand，2023 从 details.Brand 获取（details可能是dict或JSON字符串）
    brand = item.get('brand', '')
    if not brand:
        details = item.get('details', {})
        if isinstance(details, dict):
            brand = details.get('Brand', '')
        elif isinstance(details, str):
            try:
                details_dict = json.loads(details)
                brand = details_dict.get('Brand', '')
            except Exception:
                brand = ''

    # category: 2018 是列表，2023 是普通字符串列表，取最后一个作为最细粒度类别
    category = item.get('category', [])
    if not category:
        categories = item.get('categories', [])
        if categories:
            # categories 是 ['Baby Products', 'Baby Feeding', 'Bibs'] 这样的普通字符串列表
            # 取最后一个（最细粒度）
            category = [categories[-1]] if categories else []

    price = item.get('price', '')

    # 跳过 asin 或 brand 为空的商品
    if not asin or not brand:
        return None

    # 跳过 category 为空的商品
    if not category:
        return None

    # 过滤无效品牌
    invalid_brands = {'unknown', 'generic', 'n/a', 'na', 'none', 'null', ''}
    if brand.lower().strip() in invalid_brands:
        return None

    # 如果 price 为空，尝试从 title/description/features 中提取
    if not price:
        title = item.get('title', '')
        desc = ' '.join(str(d) for d in item.get('description', []))
        feature = ' '.join(str(f) for f in item.get('features', []))
        text_for_price = title + ' ' + desc + ' ' + feature
        price = extract_price_from_text(text_for_price)
        if price:
            item['price'] = price  # 把提取到的价格设置回 item，供 extract_attributes 使用

    # 如果 price 仍为空，跳过（需要价格信息）
    if not price:
        return None

    attrs = extract_attributes(item)

    # 过滤条件：A1-A18 中至少5个属性不为空
    non_empty_count = 0
    for key in attrs:
        if key.startswith('A') and len(key) >= 2 and key[1].isdigit():
            slot_str = key.split('_')[0][1:]
            if slot_str.isdigit():
                val = attrs[key]
                if val and (isinstance(val, (list, dict)) and len(val) > 0 if isinstance(val, (list, dict)) else bool(val)):
                    non_empty_count += 1

    if non_empty_count < 5:
        return None

    return attrs


def process_chunk(items: List) -> Tuple[List[Dict], Dict]:
    """处理一批数据（字典列表或字符串列表），返回结果列表和统计信息"""
    import ujson
    results = []
    stats = {
        'json_error': 0,
        'missing_asin': 0,
        'missing_brand': 0,
        'missing_category': 0,
        'missing_price': 0,
        'filtered_no_a4_a5': 0,
    }
    for slot in ['A1', 'A2', 'A3', 'A4', 'A5']:
        stats[f'has_{slot}'] = 0

    for item_or_line in items:
        # 支持字典列表（缓存）或字符串列表（原始文件）
        if isinstance(item_or_line, dict):
            item = item_or_line
        else:
            line = item_or_line.strip()
            if not line:
                continue
            try:
                item = ujson.loads(line)
            except Exception:
                stats['json_error'] += 1
                continue

        attrs = process_item(item)
        if attrs is None:
            # 统计被过滤的原因
            asin = item.get('asin', '')
            brand = item.get('brand', '')
            category = item.get('category', [])
            price = item.get('price', '')
            if not asin:
                stats['missing_asin'] += 1
            if not brand:
                stats['missing_brand'] += 1
            if not category:
                stats['missing_category'] = stats.get('missing_category', 0) + 1
            if not price:
                stats['missing_price'] = stats.get('missing_price', 0) + 1
            if asin and brand and category and price:
                stats['filtered_no_a4_a5'] += 1
            continue

        results.append(attrs)

        # 统计每个槽位的非空数量
        if attrs.get('A1_product_type'):
            stats['has_A1'] += 1
        if attrs.get('A2_brand'):
            stats['has_A2'] += 1
        if attrs.get('A3_price'):
            stats['has_A3'] += 1
        if attrs.get('A4_appearance'):
            stats['has_A4'] += 1
        if attrs.get('A5_use_case'):
            stats['has_A5'] += 1

    return results, stats


def main():
    INPUT_FILE = "/fs04/ar57/wenyu/data/Amazon-Reviews-2023/meta_Baby_Products.jsonl.gz"
    OUTPUT_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/Baby_Products/attributes_Baby_Products.json"
    STAGE0_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/Baby_Products"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 多进程配置 - 使用更多worker和更大chunk
    num_workers = 16
    chunk_size = 50000  # 减小块大小增加并行度

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1 v5: 5-Slot Product Attribute Extraction (多进程版)")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"输入文件: {INPUT_FILE}")
    log_with_timestamp(f"输出文件: {OUTPUT_FILE}")
    log_with_timestamp(f"使用进程数: {num_workers}")
    log_with_timestamp(f"分块大小: {chunk_size}")
    log_with_timestamp("")
    log_with_timestamp("槽位定义:")
    log_with_timestamp("  A1: Category (产品类型)")
    log_with_timestamp("  A2: Brand (品牌)")
    log_with_timestamp("  A3: Price (价格)")
    log_with_timestamp("  A4: Appearance (外观：颜色+风格)")
    log_with_timestamp("  A5: Usage (使用场景：for X)")
    log_with_timestamp("  A7: Material (材料)")
    log_with_timestamp("  A8: Safety (安全/环保)")
    log_with_timestamp("  A9: Durability (耐用性)")
    log_with_timestamp("  A10: Ease_of_use (易用性)")
    log_with_timestamp("  A11: Temperature_resistance (耐温性)")
    log_with_timestamp("  A12: Surface (表面特性)")
    log_with_timestamp("  A13: Reusability (可重复使用)")
    log_with_timestamp("  A14: Size (尺寸规格)")
    log_with_timestamp("  A16: Compatibility (兼容性)")
    log_with_timestamp("  A18: Quality (质量描述)")
    log_with_timestamp("")

    log_with_timestamp("📂 开始读取商品数据...")

    # 读取所有行
    with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    log_with_timestamp(f"总行数: {total_lines}")

    # 分块
    chunks = []
    for i in range(0, total_lines, chunk_size):
        chunks.append(all_lines[i:i+chunk_size])

    log_with_timestamp(f"分为 {len(chunks)} 个块进行处理...")
    log_with_timestamp("")

    # ========== 先建立 child->parent 映射（用于过滤匹配）==========
    log_with_timestamp("🔍 建立 child_asin -> parent_asin 映射...")
    child_to_parent = {}
    parent_to_children = {}
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        parent_asin = item.get('parent_asin', '')
        asin = item.get('asin', '')
        if asin and parent_asin and asin != parent_asin:
            child_to_parent[asin] = parent_asin
            if parent_asin not in parent_to_children:
                parent_to_children[parent_asin] = set()
            parent_to_children[parent_asin].add(asin)
        elif parent_asin:
            if parent_asin not in parent_to_children:
                parent_to_children[parent_asin] = set()
    log_with_timestamp(f"  映射: {len(child_to_parent)} 个 child->parent, {len(parent_to_children)} 个 parent")

    # 多进程处理
    all_results = []
    all_stats = []

    # 使用 imap_unordered 提高并行效率
    log_with_timestamp("🚀 开始多进程处理...")
    import sys
    sys.stdout.flush()
    with Pool(processes=num_workers) as pool:
        results_iter = pool.imap_unordered(process_chunk, chunks, chunksize=1)
        processed = 0
        for chunk_results, chunk_stats in results_iter:
            all_results.extend(chunk_results)
            all_stats.append(chunk_stats)
            processed += 1
            if processed % 2 == 0 or processed == len(chunks):
                log_with_timestamp(f"  已处理 {processed}/{len(chunks)} 块, 结果数: {len(all_results)}")
                sys.stdout.flush()
    log_with_timestamp(f"  处理完成, 结果数: {len(all_results)}")

    # ========== 新增：根据 Stage 0 用户评论过滤 ==========
    # STAGE0_DIR 已在上方定义

    log_with_timestamp("")
    log_with_timestamp("🔍 读取 Stage 0 用户评论数据...")

    # 读取单一用户评论文件
    reviewed_asins = set()
    asin_to_users = {}  # asin -> list of user_ids
    all_users_file = os.path.join(STAGE0_DIR, 'all_users_reviews.json')

    with open(all_users_file, 'r', encoding='utf-8') as f:
        all_users_data = json.load(f)
        for user_data in all_users_data.get('users', []):
            user_id = user_data['user_id']
            for product in user_data.get('results', []):
                asin = product.get('asin', '')
                if asin:
                    reviewed_asins.add(asin)
                    if asin not in asin_to_users:
                        asin_to_users[asin] = []
                    asin_to_users[asin].append(user_id)

    log_with_timestamp(f"  Stage 0 用户评论过的商品数: {len(reviewed_asins)}")

    # 找出被评论过的 parent_asins
    reviewed_parents = set()
    for asin in reviewed_asins:
        if asin in child_to_parent:
            reviewed_parents.add(child_to_parent[asin])
        elif asin in parent_to_children:
            reviewed_parents.add(asin)

    log_with_timestamp(f"  Stage 0 评论过的父商品数: {len(reviewed_parents)}")

    # 过滤：只保留评论过的商品（直接匹配 或 子商品被评论过）
    original_count = len(all_results)

    def is_reviewed_product(result_asin):
        if result_asin in reviewed_asins:
            return True
        if result_asin in reviewed_parents:
            return True
        return False

    all_results = [r for r in all_results if is_reviewed_product(r.get('asin'))]

    # 为每个商品添加评论用户信息
    for r in all_results:
        asin = r.get('asin', '')
        users = set()
        if asin in asin_to_users:
            users.update(asin_to_users[asin])
        if asin in parent_to_children:
            for child_asin in parent_to_children[asin]:
                if child_asin in asin_to_users:
                    users.update(asin_to_users[child_asin])
        r['reviewed_by_users'] = list(users)

    filtered_count = len(all_results)

    log_with_timestamp(f"  Stage 0 过滤后商品数: {filtered_count} (过滤掉 {original_count - filtered_count} 个)")

    # 找出被评论过但不在结果中的商品
    result_asins = set(r.get('asin') for r in all_results)
    missing_count = 0
    for asin in reviewed_asins:
        if asin in child_to_parent:
            parent = child_to_parent[asin]
            if parent not in result_asins:
                missing_count += 1
        elif asin not in result_asins:
            missing_count += 1
    if missing_count > 0:
        log_with_timestamp(f"  ⚠️ 评论过但未在结果中的商品数: {missing_count}")
    # ========== 过滤结束 ==========

    # 合并统计
    stats = {
        'total': len(all_results),
        'json_error': sum(s.get('json_error', 0) for s in all_stats),
        'missing_asin': sum(s.get('missing_asin', 0) for s in all_stats),
        'missing_brand': sum(s.get('missing_brand', 0) for s in all_stats),
        'missing_category': sum(s.get('missing_category', 0) for s in all_stats),
        'missing_price': sum(s.get('missing_price', 0) for s in all_stats),
        'filtered_no_a4_a5': sum(s.get('filtered_no_a4_a5', 0) for s in all_stats),
    }

    # 重新计算每个槽位的非空数量（基于过滤后的结果）
    for slot in ['A1', 'A2', 'A3', 'A4', 'A5']:
        slot_key = f'{slot}_product_type' if slot in ['A1'] else f'{slot}_' + {
            'A2': 'brand', 'A3': 'price', 'A4': 'appearance', 'A5': 'use_case'
        }[slot]
        stats[f'has_{slot}'] = sum(1 for r in all_results if r.get(slot_key))

    log_with_timestamp(f"✅ 处理完成，总计 {stats['total']} 个有效商品")

    # 保存结果
    log_with_timestamp(f"💾 保存结果到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source': 'meta_Baby_Products.jsonl.gz',
                'total_products': stats['total'],
                'extraction_time': datetime.now().isoformat(),
                'slots': {
                    'A1': 'Category - 产品类型',
                    'A2': 'Brand - 品牌',
                    'A3': 'Price - 价格',
                    'A4': 'Appearance - 外观（颜色+风格+表面）',
                    'A5': 'Usage - 使用场景',
                    'A6': 'Detailed - Baby领域特定属性（AgeRange/TargetSpecies/Size/Material等）',
                    'A7': 'Material - 材料',
                    'A8': 'Safety - 安全/环保',
                    'A9': 'Durability - 耐用性',
                    'A10': 'Ease_of_use - 易用性',
                    'A11': 'Temperature_resistance - 耐温性',
                    'A13': 'Reusability - 可重复使用',
                    'A14': 'Size - 尺寸规格',
                    'A16': 'Compatibility - 兼容性',
                    'A18': 'Quality - 质量描述',
                }
            },
            'stats': stats,
            'products': all_results
        }, f, indent=2, ensure_ascii=False)

    # 打印统计
    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("📊 属性提取统计")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"总有效商品数: {stats['total']}")
    log_with_timestamp(f"JSON 错误: {stats.get('json_error', 0)}")
    log_with_timestamp(f"缺失 ASIN: {stats.get('missing_asin', 0)}")
    log_with_timestamp(f"缺失 Brand: {stats.get('missing_brand', 0)}")
    log_with_timestamp(f"缺失 Category: {stats.get('missing_category', 0)}")
    log_with_timestamp(f"缺失 Price: {stats.get('missing_price', 0)}")
    log_with_timestamp(f"无 A4/A5 (过滤): {stats.get('filtered_no_a4_a5', 0)}")
    log_with_timestamp(f"无 A3 (过滤): {stats.get('filtered_no_a3', 0)}")
    log_with_timestamp("")
    log_with_timestamp("各槽位非空数量:")
    for slot in ['A1', 'A2', 'A3', 'A4', 'A5']:
        count = stats.get(f'has_{slot}', 0)
        pct = 100 * count / max(1, stats['total'])
        log_with_timestamp(f"  {slot}: {count} ({pct:.1f}%)")
    log_with_timestamp("=" * 80)

    # 显示示例
    log_with_timestamp("")
    log_with_timestamp("📋 示例输出 (前3个):")
    for i, p in enumerate(all_results[:3]):
        a6 = p.get('A6_detailed', {})
        a6_str = str(a6) if a6 else '{}'
        print(f"  {i+1}. asin={p['asin']}, A1={p['A1_product_type']}, A2={p['A2_brand']}, A3={p['A3_price']}, A5={p['A5_use_case']}, A6_detailed={a6_str}")

    # ========== 匈牙利算法最大匹配 ==========
    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("🔄 执行匈牙利算法最大匹配...")
    matched_results, matching_stats = run_hungarian_matching(all_results)

    log_with_timestamp(f"  总用户数: {matching_stats['total_users']}")
    log_with_timestamp(f"  成功匹配: {matching_stats['matched_users']}/{matching_stats['total_users']}")
    log_with_timestamp(f"  使用独特商品数: {matching_stats['matched_products']}")
    log_with_timestamp(f"  未匹配用户数: {matching_stats['unmatched_users']}")

    # 输出直接覆盖原文件，格式为用户-商品一对一
    log_with_timestamp(f"💾 保存结果到: {OUTPUT_FILE} (匈牙利匹配后)")
    output_results = []
    for item in matched_results:
        p = item['product']
        p_copy = p.copy()
        p_copy['user_id'] = item['user_id']
        output_results.append(p_copy)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source': 'meta_Baby_Products.jsonl.gz',
                'total_products': len(output_results),
                'total_users': matching_stats['total_users'],
                'matched_users': matching_stats['matched_users'],
                'unmatched_users': matching_stats['unmatched_users'],
                'extraction_time': datetime.now().isoformat(),
                'slots': {
                    'A1': 'Category - 产品类型',
                    'A2': 'Brand - 品牌',
                    'A3': 'Price - 价格',
                    'A4': 'Appearance - 外观（颜色+风格+表面）',
                    'A5': 'Usage - 使用场景',
                    'A6': 'Detailed - Baby领域特定属性（AgeRange/TargetSpecies/Size/Material等）',
                    'A7': 'Material - 材料',
                    'A8': 'Safety - 安全/环保',
                    'A9': 'Durability - 耐用性',
                    'A10': 'Ease_of_use - 易用性',
                    'A11': 'Temperature_resistance - 耐温性',
                    'A13': 'Reusability - 可重复使用',
                    'A14': 'Size - 尺寸规格',
                    'A16': 'Compatibility - 兼容性',
                    'A18': 'Quality - 质量描述',
                }
            },
            'stats': {
                'total': len(output_results),
                'total_users': matching_stats['total_users'],
                'matched_users': matching_stats['matched_users'],
                'unmatched_users': matching_stats['unmatched_users'],
                'unmatched_user_list': matching_stats['unmatched_user_list'],
            },
            'products': output_results
        }, f, indent=2, ensure_ascii=False)

    log_with_timestamp("")
    log_with_timestamp("📋 匈牙利匹配示例 (前3个):")
    for i, item in enumerate(matched_results[:3]):
        p = item['product']
        log_with_timestamp(f"  {i+1}. user={item['user_id']}, asin={p.get('asin')}, brand={p.get('A2_brand')}, type={p.get('A1_product_type')}")

    # ========== 生成匹配用户的过滤后评论文件 ==========
    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("🔄 生成匹配用户的过滤后评论文件...")

    # 获取成功匹配的用户ID集合
    matched_user_ids = set(item['user_id'] for item in matched_results)

    # 只保留匹配用户，保留所有原始商品（不过滤商品）
    filtered_users = []
    for user_data in all_users_data.get('users', []):
        user_id = user_data['user_id']
        if user_id not in matched_user_ids:
            continue  # 跳过未匹配用户

        filtered_users.append(user_data)  # 保留用户的所有原始数据

    # 保存过滤后的用户文件
    STAGE1_FILTERED_FILE = os.path.join(os.path.dirname(OUTPUT_FILE), 'stage1_filtered_users_reviews.json')
    log_with_timestamp(f"💾 保存过滤后用户文件到: {STAGE1_FILTERED_FILE}")
    with open(STAGE1_FILTERED_FILE, 'w', encoding='utf-8') as f:
        json.dump({'users': filtered_users}, f, ensure_ascii=False, indent=2)
    log_with_timestamp(f"  匹配用户数: {len(filtered_users)}")

    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("✅ Stage 1 全部完成")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
