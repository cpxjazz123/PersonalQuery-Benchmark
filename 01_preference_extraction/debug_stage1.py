#!/usr/bin/env python3
"""
Debug script for Stage 1 to identify why no results are being returned.
"""
import gzip
import json
import sys
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

# ========== Copy all necessary functions from the main script ==========

def extract_price(price_str) -> Optional[str]:
    if not price_str:
        return None
    price_str = str(price_str).replace('$', '').replace(',', '').strip()
    try:
        return str(float(price_str))
    except:
        return None

def extract_price_from_text(text: str) -> Optional[str]:
    import re
    patterns = [
        r'\$\s*(\d+(?:\.\d{2})?)',
        r'(\d+(?:\.\d{2})?)\s*(?:dollar|cent)',
        r'(?:price|cost)[:\s]*\$?\s*(\d+(?:\.\d{2})?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def extract_product_type(category: List) -> Optional[str]:
    if not category:
        return None
    return category[0] if isinstance(category, list) else category

def extract_use_case(title: str, description: str, feature: List) -> Optional[str]:
    import re
    text = title + ' ' + ' '.join(str(f) for f in feature) + ' ' + str(description)
    patterns = [
        r'for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
        r'perfect for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
        r'ideal for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
        r'use for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
        r'great for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
    ]
    use_cases = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        use_cases.extend(matches)
    return ' '.join(list(dict.fromkeys(use_cases))[:5]) if use_cases else None

def extract_structured_features(feature: List, title: str = "") -> Dict[str, List[str]]:
    import re
    combined = ' '.join(str(f) for f in feature) + ' ' + str(title)
    combined_lower = combined.lower()

    colors = ['red', 'blue', 'green', 'black', 'white', 'pink', 'purple', 'orange',
              'yellow', 'brown', 'gray', 'grey', 'gold', 'silver', 'clear', 'natural',
              'beige', 'navy', 'ivory', 'coral', 'turquoise', 'burgundy', 'lavender']
    found_colors = []
    for color in colors:
        if color in combined_lower:
            found_colors.append(color)
    found_colors = list(dict.fromkeys(found_colors))

    styles = ['vintage', 'modern', 'classic', 'rustic', 'minimalist', 'bohemian',
              'shabby', 'elegant', 'cute', 'simple', 'fancy', 'professional',
              'handmade', 'artisan', 'traditional', 'contemporary', 'retro', 'glam']
    found_styles = []
    for style in styles:
        if style in combined_lower:
            found_styles.append(style)

    materials = ['plastic', 'metal', 'wood', 'ceramic', 'glass', 'paper', 'fabric',
                 'leather', 'silk', 'cotton', 'polyester', 'rubber', 'foam', 'acrylic',
                 'resin', 'aluminum', 'steel', 'copper', 'brass', 'silver', ' bamboo']
    found_materials = []
    for mat in materials:
        if mat in combined_lower:
            found_materials.append(mat)

    return {
        'A4_appearance': found_colors + found_styles,
        'A7_material': found_materials,
    }

def extract_attributes(item: Dict) -> Dict:
    asin = item.get('asin', '')
    title = item.get('title', '')
    brand = item.get('brand', '')
    price = item.get('price', '')
    description = item.get('description', '')
    if isinstance(description, list):
        description = ' '.join(str(d) for d in description)
    feature = item.get('feature', [])
    category = item.get('category', [])
    tech1 = item.get('tech1', '')

    structured = extract_structured_features(feature, title)
    desc_structured = {}  # Skip description extraction for now

    return {
        'asin': asin,
        'A1_product_type': extract_product_type(category),
        'A2_brand': brand if brand and isinstance(brand, str) and len(str(brand).strip()) > 0 else None,
        'A3_price': extract_price(price),
        'A4_appearance': structured.get('A4_appearance', []),
        'A5_use_case': extract_use_case(title, description, feature),
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
            # categories 是 ['Pet Supplies', 'Dogs', 'Collars'] 这样的普通字符串列表
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

    # 只保留 A1_product_type、A4_appearance 和 A5_use_case 都不为空的商品
    if not attrs.get('A1_product_type') or not attrs.get('A4_appearance') or not attrs.get('A5_use_case'):
        return None

    return attrs

def debug_item(item: Dict, index: int) -> Dict:
    """Debug a single item and return detailed info about why it passes/fails"""
    result = {
        'index': index,
        'item_keys': list(item.keys()),
        'raw_asin': item.get('asin'),
        'raw_brand': item.get('brand'),
        'raw_category': item.get('category'),
        'raw_price': item.get('price'),
        'raw_parent_asin': item.get('parent_asin'),
    }

    # Check details
    details = item.get('details', {})
    result['details_type'] = type(details).__name__
    if isinstance(details, str):
        try:
            details_dict = json.loads(details)
            result['details_parsed'] = True
            result['details_brand'] = details_dict.get('Brand', '')
        except:
            result['details_parsed'] = False
            result['details_brand'] = ''
    elif isinstance(details, dict):
        result['details_parsed'] = True
        result['details_brand'] = details.get('Brand', '')

    # Check categories
    categories = item.get('categories', [])
    result['categories'] = categories
    if categories:
        result['category_fallback'] = [categories[-1]]
    else:
        result['category_fallback'] = []

    # Process item result
    asin = item.get('asin') or item.get('parent_asin', '')
    brand = item.get('brand', '')
    if not brand and isinstance(details, dict):
        brand = details.get('Brand', '')
    elif not brand and isinstance(details, str):
        try:
            details_dict = json.loads(details)
            brand = details_dict.get('Brand', '')
        except:
            brand = ''

    category = item.get('category', [])
    if not category:
        category = [categories[-1]] if categories else []

    result['computed_asin'] = asin
    result['computed_brand'] = brand
    result['computed_category'] = category
    result['computed_price'] = item.get('price', '')

    # Reasons for failure
    reasons = []
    if not asin:
        reasons.append('no_asin')
    if not brand:
        reasons.append('no_brand')
    if not category:
        reasons.append('no_category')
    if not item.get('price', ''):
        reasons.append('no_price')

    result['failure_reasons'] = reasons

    # Try process_item with detailed failure tracking
    proc_result = process_item(item)
    result['process_item_result'] = 'PASS' if proc_result else 'FAIL'

    if proc_result:
        result['attrs'] = {
            'A1': proc_result.get('A1_product_type'),
            'A4': proc_result.get('A4_appearance'),
            'A5': proc_result.get('A5_use_case'),
        }
    else:
        # Detailed failure tracking
        asin = item.get('asin') or item.get('parent_asin', '')
        brand = item.get('brand', '')
        if not brand:
            details = item.get('details', {})
            if isinstance(details, dict):
                brand = details.get('Brand', '')
            elif isinstance(details, str):
                try:
                    details_dict = json.loads(details)
                    brand = details_dict.get('Brand', '')
                except:
                    brand = ''

        category = item.get('category', [])
        if not category:
            categories = item.get('categories', [])
            if categories:
                category = [categories[-1]] if categories else []

        price = item.get('price', '')
        title = item.get('title', '')
        desc = ' '.join(str(d) for d in item.get('description', []))
        feature = ' '.join(str(f) for f in item.get('features', []))
        text_for_price = title + ' ' + desc + ' ' + feature
        extracted_price = extract_price_from_text(text_for_price)

        # Step by step checks
        checks = {
            'has_asin': bool(asin),
            'has_brand': bool(brand),
            'has_category': bool(category),
            'has_price_direct': bool(price),
            'has_price_extracted': bool(extracted_price),
        }

        invalid_brands = {'unknown', 'generic', 'n/a', 'na', 'none', 'null', ''}
        checks['brand_invalid'] = brand.lower().strip() in invalid_brands if brand else False

        # Get A4/A5
        structured = extract_structured_features(item.get('features', []), title)
        use_case = extract_use_case(title, desc, item.get('features', []))

        checks['has_a4'] = bool(structured.get('A4_appearance', []))
        checks['has_a5'] = bool(use_case)

        result['checks'] = checks
        result['title_sample'] = title[:80] if title else None
        result['price_direct'] = price
        result['price_extracted'] = extracted_price

    return result

def main():
    INPUT_FILE = "/workspace/PersonalQuery/data/Amazon-Reviews-2023/raw/meta_categories/meta_Arts_Crafts_and_Sewing.jsonl.gz"

    print(f"Reading data from {INPUT_FILE}...")

    # Read first 100 items
    items = []
    with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            items.append(json.loads(line))

    print(f"Loaded {len(items)} items")
    print()

    # Debug each item
    debug_results = []
    for i, item in enumerate(items):
        debug_results.append(debug_item(item, i))

    # Summary
    pass_count = sum(1 for r in debug_results if r['process_item_result'] == 'PASS')
    fail_count = len(debug_results) - pass_count

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total items: {len(debug_results)}")
    print(f"PASS: {pass_count} ({100*pass_count/len(debug_results):.1f}%)")
    print(f"FAIL: {fail_count} ({100*fail_count/len(debug_results):.1f}%)")
    print()

    # Failure reasons breakdown
    reason_counts = {}
    for r in debug_results:
        for reason in r['failure_reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("Failure reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    print()

    # Show first 5 failures with details
    failures = [r for r in debug_results if r['process_item_result'] == 'FAIL']
    if failures:
        print("=" * 80)
        print("FIRST 5 FAILURES (detailed)")
        print("=" * 80)
        for r in failures[:5]:
            print(f"\nItem {r['index']}:")
            print(f"  Keys: {r['item_keys']}")
            print(f"  raw_asin: {r['raw_asin']}")
            print(f"  raw_brand: {r['raw_brand']}")
            print(f"  raw_parent_asin: {r['raw_parent_asin']}")
            print(f"  details_type: {r['details_type']}")
            print(f"  details_brand: {r.get('details_brand', 'N/A')}")
            print(f"  computed_asin: {r['computed_asin']}")
            print(f"  computed_brand: {r['computed_brand'][:30] if r['computed_brand'] else None}")
            print(f"  computed_category: {r['computed_category']}")
            print(f"  failure_reasons: {r['failure_reasons']}")
            if 'checks' in r:
                print(f"  Detailed checks:")
                for k, v in r['checks'].items():
                    print(f"    {k}: {v}")
                print(f"  title_sample: {r.get('title_sample')}")
                print(f"  price_direct: {r.get('price_direct')}")
                print(f"  price_extracted: {r.get('price_extracted')}")

    # Show first 5 passes
    passes = [r for r in debug_results if r['process_item_result'] == 'PASS']
    if passes:
        print()
        print("=" * 80)
        print("FIRST 5 PASSES (detailed)")
        print("=" * 80)
        for r in passes[:5]:
            print(f"\nItem {r['index']}:")
            print(f"  computed_asin: {r['computed_asin']}")
            print(f"  computed_brand: {r['computed_brand'][:30] if r['computed_brand'] else None}")
            print(f"  computed_category: {r['computed_category']}")
            print(f"  attrs: {r.get('attrs')}")
    else:
        # Find items that LOOK like they should pass
        print()
        print("=" * 80)
        print("ITEMS THAT LOOK LIKE THEY SHOULD PASS")
        print("=" * 80)
        for r in debug_results[:10]:
            if not r['failure_reasons']:  # No basic failure reasons
                print(f"\nItem {r['index']}:")
                print(f"  computed_asin: {r['computed_asin']}")
                print(f"  computed_brand: {r['computed_brand'][:30] if r['computed_brand'] else None}")
                print(f"  computed_category: {r['computed_category']}")
                print(f"  failure_reasons: {r['failure_reasons']}")
                if 'checks' in r:
                    print(f"  Detailed checks:")
                    for k, v in r['checks'].items():
                        print(f"    {k}: {v}")
                print(f"  title_sample: {r.get('title_sample')}")
                print(f"  price_direct: {r.get('price_direct')}")
                print(f"  price_extracted: {r.get('price_extracted')}")

if __name__ == '__main__':
    main()
