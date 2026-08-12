#!/usr/bin/env python3
"""
Stage 13: Retrieval Evaluation - Shared Utilities (STaRK-style)

STaRK 风格的文档构建，包含:
- 结构化 YAML-like 输出格式
- Reviews 按 vote 排序
- Q&A 数据支持
- dimensions/weight 提取
- relations (also_buy, also_view)
"""

import json
import os
import gzip
import re
import sys
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional

# Suppress BeautifulSoup warning about file-like strings
warnings.filterwarnings("ignore", message=".*looks more like a filename.*")

import numpy as np
import pandas as pd

_BS4_PARSER = None

def _get_bs4_parser():
    global _BS4_PARSER
    if _BS4_PARSER is None:
        from bs4 import BeautifulSoup
        _BS4_PARSER = BeautifulSoup
    return _BS4_PARSER


def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_progress(message):
    """Log message with timestamp to stderr (for detailed progress logs)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True, file=sys.stderr)


def compact_text(text: str) -> str:
    """STaRK-style: Compact text by removing unnecessary spaces and punctuation issues"""
    if not text:
        return text
    text = text.replace("\n", ". ").replace("\r", "")
    text = text.replace("- ", "")
    text = text.replace(": .", ":").replace("ios:", ":")
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(".. ", ". ")
    return text.strip()


def clean_data(item) -> str:
    """STaRK-style: Clean text data - HTML removal + whitespace normalization"""
    if item is None:
        return ''
    if isinstance(item, str):
        parser = _get_bs4_parser()
        # Explicitly pass 'lxml' to avoid BeautifulSoup warnings about URL-like strings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*MarkupResemblesLocatorWarning.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*looks more like a filename.*")
            item = ' '.join(parser(item, "lxml").text.split())
    elif isinstance(item, list):
        item = ' '.join(str(x) for x in item if x)
    else:
        item = str(item)
    return item.strip()


def process_brand(brand: str) -> str:
    """STaRK-style: Clean and normalize brand names"""
    if not brand:
        return ''
    import string
    brand = brand.strip(" \" .*+,-_!@#$%^&*();\/|<>'\t\n\r\\")
    if brand.startswith('by '):
        brand = brand[3:]
    if brand.endswith('.com'):
        brand = brand[:-4]
    if brand.startswith('www.'):
        brand = brand[4:]
    if len(brand) > 100:
        brand = brand.split(' ')[0]
    return brand.strip()


def decode_escapes(s: str) -> str:
    """STaRK-style: Decode escape sequences in a string"""
    if not s:
        return s
    import codecs
    ESCAPE_SEQUENCE_RE = re.compile(r'''
        ( \\U........      # 8-digit hex escapes
        | \\u....          # 4-digit hex escapes
        | \\x..            # 2-digit hex escapes
        | \\[0-7]{1,3}     # Octal escapes
        | \\N\{[^}]+\}     # Unicode characters by name
        | \\[\\'"abfnrtv]  # Single-character escapes
        )''', re.UNICODE | re.VERBOSE)

    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


def decode_html_entities(text: str) -> str:
    """Decode HTML entities like &amp; -> &, &lt; -> <, etc."""
    if not text:
        return text
    import html
    return html.unescape(text)


# ============================================================================
# STaRK-style Document Building Functions
# ============================================================================

def get_chunk_info(product: Dict, attribute: str, max_entries: int = 25) -> str:
    """STaRK-style chunk info extraction for specific attributes."""
    node_attr = product.get(attribute)
    if not node_attr:
        if attribute == 'review':
            node_attr = product.get('reviews')
    if not node_attr:
        return ''
    
    # Handle features
    if 'feature' in attribute:
        features = [f for f in node_attr if f and 'asin' not in f.lower()]
        chunk = ' '.join(features)
        return chunk
    
    # Handle reviews (STaRK style: "The review "{summary}" states that "{reviewText}".")
    elif 'review' in attribute:
        chunk = ''
        if isinstance(node_attr, list) and node_attr:
            scores = []
            for review in node_attr:
                vote_val = review.get('vote')
                if pd.isnull(vote_val) or vote_val is None:
                    scores.append(0)
                else:
                    try:
                        scores.append(int(str(vote_val).replace(",", "")))
                    except (ValueError, AttributeError):
                        scores.append(0)
            
            ranks = np.argsort(-np.array(scores))
            
            for idx, review_idx in enumerate(ranks):
                review = node_attr[review_idx]
                summary = decode_html_entities(review.get('summary', ''))
                review_text = decode_html_entities(review.get('text', review.get('reviewText', '')))
                chunk += f'The review "{summary}" states that "{review_text}". '
                if idx >= max_entries:
                    break
        return chunk
    
    # Handle Q&A (STaRK style: "The question is "{question}", and the answer is "{answer}".")
    elif 'qa' in attribute:
        chunk = ''
        if isinstance(node_attr, list) and node_attr:
            for idx, qa in enumerate(node_attr):
                question = decode_html_entities(qa.get('question', ''))
                answer = decode_html_entities(qa.get('answer', ''))
                chunk += f'The question is "{question}", and the answer is "{answer}". '
                if idx >= max_entries:
                    break
        return chunk
    
    # Handle description
    elif 'description' in attribute and node_attr:
        if isinstance(node_attr, list):
            chunk = " ".join(str(x) for x in node_attr if x)
        else:
            chunk = str(node_attr)
        return chunk
    
    else:
        if isinstance(node_attr, list):
            return ' '.join(str(x) for x in node_attr if x)
        return str(node_attr) if node_attr else ''


def extract_dimensions_weight(product: Dict) -> Tuple[str, str]:
    """
    Extract dimensions and weight from product details/rank field.
    STaRK style: from details.product_dimensions.split(' ; ')
    
    Returns:
        (dimensions, weight) tuple
    """
    dimensions = ''
    weight = ''
    
    # Try rank field first (Amazon format: "Product Dimensions: X ; Y lbs")
    rank_str = product.get('rank', '')
    if rank_str and isinstance(rank_str, str):
        # Look for pattern like "Product Dimensions: 10 x 5 x 2 inches ; 1.2 pounds"
        dim_match = re.search(r'Product Dimensions:\s*([^;]+)', rank_str)
        if dim_match:
            dim_str = dim_match.group(1).strip()
            # Split by ' ; ' like STaRK
            parts = dim_str.split(' ; ')
            if len(parts) >= 2:
                dimensions = parts[0].strip()
                weight = parts[1].strip()
            elif len(parts) == 1:
                dimensions = parts[0].strip()
    
    # Also try to extract from description or other fields
    if not dimensions or not weight:
        description = product.get('description', '')
        if isinstance(description, list):
            description = ' '.join(description)
        
        # Look for dimension patterns
        if not dimensions:
            dim_patterns = [
                r'(\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*(?:inches|in|cm|mm))',
                r'Dimensions:\s*([\d.x\s]+(?:inches|in|cm)?)',
            ]
            for pattern in dim_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    dimensions = match.group(1).strip()
                    break
        
        # Look for weight patterns
        if not weight:
            weight_patterns = [
                r'(\d+(?:\.\d+)?\s*(?:pounds|lbs|oz|grams|kg))',
                r'Weight:\s*([\d.]+\s*(?:pounds|lbs|oz|grams|kg)?)',
            ]
            for pattern in weight_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    weight = match.group(1).strip()
                    break
    
    return dimensions, weight


def get_rel_info(product: Dict, all_metadata: Dict[str, Dict], n_rel: int = 3) -> str:
    """
    STaRK-style relation information extraction.
    
    Args:
        product: Product dictionary
        all_metadata: All products metadata for lookup
        n_rel: Number of related products to include
        
    Returns:
        Formatted relations string
    """
    doc = ''
    
    # Get also_buy products
    also_buy_asins = product.get('also_buy', [])
    if isinstance(also_buy_asins, list):
        str_also_buy = []
        for idx, asin in enumerate(also_buy_asins[:n_rel]):
            if asin in all_metadata:
                title = all_metadata[asin].get('title', 'Unknown')
                str_also_buy.append(f"#{idx + 1}: {title}\n")
        if str_also_buy:
            doc += f'  products also purchased: \n{"".join(str_also_buy)}'
    
    # Get also_view products
    also_view_asins = product.get('also_view', [])
    if isinstance(also_view_asins, list):
        str_also_view = []
        for idx, asin in enumerate(also_view_asins[:n_rel]):
            if asin in all_metadata:
                title = all_metadata[asin].get('title', 'Unknown')
                str_also_view.append(f"#{idx + 1}: {title}\n")
        if str_also_view:
            doc += f'  products also viewed: \n{"".join(str_also_view)}'
    
    # Add brand relation
    brand = product.get('brand', '')
    if brand:
        doc += f'  brand: {brand}\n'
    
    if doc:
        doc = '- relations:\n' + doc
    return doc


def build_stark_document(
    product: Dict, 
    all_metadata: Dict[str, Dict] = None,
    add_rel: bool = True,
    compact: bool = False,
    max_entries: int = 25
) -> str:
    """
    STaRK-style document builder with structured YAML-like format.
    
    Output format:
    - product: {title}
    - brand: {brand}
    - dimensions: {dimensions}
    - weight: {weight}
    - description: {description}
    - features: 
    #1: {feature1}
    #2: {feature2}
    - reviews: The review "{summary}" states that "{reviewText}". ...
    - Q&A: The question is "{question}", and the answer is "{answer}". ...
    - relations:
      products also purchased: 
      #1: {also_buy_title}
      products also viewed:
      #1: {also_view_title}
      brand: {brand_name}
    
    Args:
        product: Product dictionary
        all_metadata: All products metadata for relation lookup
        add_rel: Whether to include relations section
        compact: Whether to compact the text
        max_entries: Maximum number of reviews/Q&A entries
        
    Returns:
        Formatted document string
    """
    # Get title
    title = product.get('title', 'Unnamed Product')
    
    # Start building document
    doc = f'- product: {title}\n'
    
    # Add brand
    brand = product.get('brand', '')
    if brand:
        doc += f'- brand: {brand}\n'
    
    # Add dimensions and weight (STaRK style)
    dimensions, weight = extract_dimensions_weight(product)
    if dimensions:
        doc += f'- dimensions: {dimensions}\n'
    if weight:
        doc += f'- weight: {weight}\n'
    
    # Add description
    description_attr = product.get('description')
    if description_attr:
        description = get_chunk_info(product, 'description')
        if description:
            doc += f'- description: {description}\n'
    
    # Add features (STaRK style with numbering)
    feature_attr = product.get('feature', [])
    if feature_attr:
        feature_text = '- features: \n'
        feature_idx = 0
        for feature in feature_attr:
            if feature and 'asin' not in str(feature).lower():
                feature_idx += 1
                feature_text += f'#{feature_idx}: {clean_data(feature)}\n'
        if feature_idx > 0:
            doc += feature_text
    
    # Add reviews (STaRK style with vote sorting)
    if max_entries > 0:
        review_text = get_chunk_info(product, 'review', max_entries=max_entries)
        if review_text:
            doc += f'- reviews: {review_text}\n'
        
        # Add Q&A (STaRK style)
        qa_text = get_chunk_info(product, 'qa', max_entries=max_entries)
        if qa_text:
            doc += f'- Q&A: {qa_text}\n'
    
    # Add relations
    if add_rel and all_metadata:
        rel_info = get_rel_info(product, all_metadata)
        doc += rel_info
    
    # Compact if requested
    if compact:
        doc = compact_text(doc)
    
    return doc


# ============================================================================
# Legacy function for backward compatibility
# ============================================================================

def build_document_text(product: Dict, all_metadata: Dict[str, Dict] = None) -> str:
    """
    Legacy document builder - now uses STaRK style internally.
    Kept for backward compatibility with existing code.
    """
    return build_stark_document(product, all_metadata, add_rel=True, compact=False)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_product_metadata(meta_file: str, asins: Set[str], max_docs: int = None) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load product metadata from file with STaRK-style cleaning.

    Supports both 2018 format (asin field) and 2023 format (parent_asin field).
    If max_docs is set, stop after loading that many documents to limit memory usage.
    """
    target_products = {}
    all_metadata = {}

    log_with_timestamp(f"Loading product metadata from {meta_file}...")

    open_func = gzip.open if meta_file.endswith('.gz') else open

    try:
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_docs is not None and idx >= max_docs:
                    break
                try:
                    item = json.loads(line.strip())
                    # 2023 版使用 parent_asin，2018 版使用 asin
                    asin = item.get('asin') or item.get('parent_asin')
                    if not asin:
                        continue

                    title = clean_data(item.get('title', ''))
                    if len(title) > 300:
                        title = title[:300]

                    # 处理 brand：2023 版在 details 里
                    if isinstance(item.get('details'), dict):
                        brand = process_brand(item.get('details', {}).get('brand', ''))
                    else:
                        brand = process_brand(item.get('brand', ''))

                    # 处理 categories：2023 版是嵌套列表，2018 版是列表
                    categories = item.get('categories', [])
                    if categories and isinstance(categories[0], list):
                        # 2023 格式：[["Cat1", "SubCat"], ...]
                        categories = [c for cat_list in categories for c in cat_list]
                    category = categories

                    # 处理 features：2023 版是 features（复数），2018 版是 feature
                    features = item.get('features', []) or item.get('feature', [])
                    if features and isinstance(features[0], dict):
                        # 2023 格式：[{"value": ["feature1", "feature2"]}, ...]
                        feature = []
                        for f_dict in features:
                            if isinstance(f_dict, dict) and "value" in f_dict:
                                feature.extend(f_dict["value"])
                            elif isinstance(f_dict, dict):
                                for v in f_dict.values():
                                    if isinstance(v, list):
                                        feature.extend(v)
                                    elif isinstance(v, str):
                                        feature.append(v)
                    else:
                        feature = features
                    # 限制 feature 数量以减少内存占用（取前50条）
                    if len(feature) > 50:
                        feature = feature[:50]

                    # 处理 description：2023 版可能是列表
                    description = item.get('description', [])
                    if isinstance(description, list):
                        description = ' '.join(str(d) for d in description if d)
                    # 截断超长字段以减少内存占用
                    if len(description) > 500:
                        description = description[:500]

                    rank = ''  # 2023 版没有 rank

                    all_metadata[asin] = {
                        'asin': asin,
                        'title': title,
                        'brand': brand,
                        'category': category,
                        'feature': feature,
                        'description': description,
                        'rank': rank,
                        'price': item.get('price', ''),
                        'also_buy': item.get('bought_together', []) or item.get('also_buy', [])[:20],
                        'also_view': item.get('also_view', [])[:20],
                    }

                    if asins is None or asin in asins:
                        target_products[asin] = {
                            'asin': asin,
                            'title': title,
                            'brand': brand,
                            'category': category,
                            'feature': feature,
                            'description': description,
                            'rank': rank,
                            'price': item.get('price', ''),
                            'also_buy': item.get('bought_together', []) or item.get('also_buy', []),
                            'also_view': item.get('also_view', []),
                            'reviews': [],
                            'qa': [],
                        }

                        if len(target_products) % 50000 == 0:
                            log_with_timestamp(f"  Loaded {len(target_products)} target products...")
                except Exception as e:
                    if idx < 5:
                        log_with_timestamp(f"Error parsing line {idx}: {e}")
                    continue
    except Exception as e:
        log_with_timestamp(f"Error loading metadata: {e}")

    log_with_timestamp(f"Loaded {len(target_products)} target products")
    log_with_timestamp(f"Loaded {len(all_metadata)} total products (for related product lookup)")

    return target_products, all_metadata


def load_reviews_for_products(
    review_file: str, 
    products: Dict[str, Dict], 
    max_reviews_per_product: int = 25,
    min_review_words: int = 0  # STaRK doesn't filter by word count
) -> Dict[str, Dict]:
    """
    Load reviews for products with vote field for STaRK-style sorting.
    
    Args:
        review_file: Path to review JSON file
        products: Products dictionary to update
        max_reviews_per_product: Maximum reviews per product (STaRK default: 25)
        min_review_words: Minimum words filter (0 = no filter, STaRK style)
        
    Returns:
        Updated products dictionary
    """
    log_with_timestamp(f"Loading reviews from {review_file}...")

    open_func = gzip.open if review_file.endswith('.gz') else open

    asins_to_load = set(products.keys())
    reviews_by_asin = {asin: [] for asin in asins_to_load}
    total_review_count = 0

    try:
        with open_func(review_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    asin = review.get('asin')

                    if asin in asins_to_load:
                        total_review_count += 1

                        review_text = review.get('reviewText', '')
                        review_summary = review.get('summary', '')
                        vote = review.get('vote', None)

                        # Optional: filter by word count
                        if min_review_words > 0:
                            word_count = len(review_text.split())
                            if word_count < min_review_words:
                                continue

                        reviews_by_asin[asin].append({
                            'text': review_text,
                            'reviewText': review_text,  # Keep both for compatibility
                            'summary': review_summary,
                            'vote': vote,
                            'overall': review.get('overall'),
                            'verified': review.get('verified', False),
                        })
                except Exception as e:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error loading reviews: {e}")

    # Add reviews to products (will be sorted by vote in get_chunk_info)
    for asin, reviews in reviews_by_asin.items():
        if asin in products:
            products[asin]['reviews'] = reviews

    total_reviews = sum(len(r) for r in reviews_by_asin.values())
    log_with_timestamp(f"Total reviews processed: {total_review_count}")
    log_with_timestamp(f"Loaded {total_reviews} reviews for {len(products)} products")

    return products


def load_qa_for_products(
    qa_file: str, 
    products: Dict[str, Dict],
    max_qa_per_product: int = 25
) -> Dict[str, Dict]:
    """Load Q&A data for products (STaRK style)."""
    log_with_timestamp(f"Loading Q&A from {qa_file}...")

    open_func = gzip.open if qa_file.endswith('.gz') else open

    asins_to_load = set(products.keys())
    qa_by_asin = {asin: [] for asin in asins_to_load}
    total_qa_count = 0

    def parse_qa_line(line):
        """Parse Q&A line - handles both Python dict (single quotes) and JSON (double quotes)"""
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            try:
                return eval(line)
            except:
                return None

    try:
        with open_func(qa_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    qa = parse_qa_line(line)
                    if qa is None:
                        continue
                    asin = qa.get('asin')

                    if asin in asins_to_load:
                        total_qa_count += 1

                        qa_entry = {
                            'questionType': qa.get('questionType', ''),
                            'question': qa.get('question', ''),
                            'answer': qa.get('answer', ''),
                            'answerType': qa.get('answerType', ''),
                            'answerTime': qa.get('answerTime', ''),
                        }

                        if qa_entry['question'] and qa_entry['answer']:
                            qa_by_asin[asin].append(qa_entry)
                except Exception as e:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error loading Q&A: {e}")

    # Add Q&A to products
    for asin, qa_list in qa_by_asin.items():
        if asin in products:
            products[asin]['qa'] = qa_list[:max_qa_per_product]

    total_qa = sum(len(q) for q in qa_by_asin.values())
    log_with_timestamp(f"Total Q&A processed: {total_qa_count}")
    log_with_timestamp(f"Loaded {total_qa} Q&A entries for {len(products)} products")

    return products


# ============================================================================
# Enhanced Evaluation Functions
# ============================================================================

def compute_enhanced_metrics(
    retrieved: List[str], 
    relevant: Set[str], 
    k: int = 10
) -> Dict:
    """
    Compute comprehensive retrieval metrics including diagnostic indicators
    
    Returns dict with:
    - Base metrics: precision, recall, MAP, NDCG, MRR
    - Diagnostic: Hit@k, F1-score, AvgRank
    - Ranking metrics: DCG, CG, ERR, RBP
    - Specialized: R-Precision, Bpref, Novelty
    """
    retrieved_k = retrieved[:k]
    num_relevant = len([r for r in retrieved_k if r in relevant])
    
    precision = num_relevant / k if k > 0 else 0
    recall = num_relevant / len(relevant) if len(relevant) > 0 else 0
    
    ap = 0
    num_found = 0
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            num_found += 1
            ap += num_found / (i + 1)
    ap = ap / len(relevant) if len(relevant) > 0 else 0
    
    dcg = 0
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    mrr = 0
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            mrr = 1 / (i + 1)
            break
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    hit = 1 if num_relevant > 0 else 0
    
    relevant_ranks = []
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            relevant_ranks.append(i + 1)
    avg_rank = np.mean(relevant_ranks) if relevant_ranks else k + 1
    
    cg = float(num_relevant)
    err = compute_err(retrieved, relevant, k)
    rbp = compute_rbp(retrieved, relevant, k)
    r_precision = compute_r_precision(retrieved, relevant)
    bpref = compute_bpref(retrieved, relevant, k)
    novelty = compute_novelty(retrieved, relevant, k)
    
    return {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'ap': ap,
        'ndcg': ndcg,
        'mrr': mrr,
        'f1_at_k': f1,
        'hit_at_k': hit,
        'avg_rank': avg_rank,
        'dcg': dcg,
        'cg': cg,
        'err': err,
        'rbp': rbp,
        'r_precision': r_precision,
        'bpref': bpref,
        'novelty': novelty,
    }


def compute_noise_robustness(
    clean_metrics: Dict,
    noisy_metrics: Dict,
    key: str = 'ndcg'
) -> Dict:
    """
    Compute noise robustness metrics comparing clean vs noisy performance
    
    Args:
        clean_metrics: Metrics from clean queries
        noisy_metrics: Metrics from noisy queries
        key: Which metric to analyze (default: 'ndcg')
    
    Returns dict with:
    - Absolute difference
    - Relative change (%)
    - Robustness score [0-1]
    """
    clean_val = clean_metrics.get(key, 0)
    noisy_val = noisy_metrics.get(key, 0)
    
    delta = noisy_val - clean_val  # Can be positive or negative
    
    if clean_val > 0:
        rel_change = (delta / clean_val) * 100  # Percentage change
        # Robustness: how well it resists noise (penalize large negative changes)
        robustness = max(0, 1 - abs(delta) / clean_val) if clean_val > 0 else 0
    else:
        rel_change = 0
        robustness = 1 if delta >= 0 else 0
    
    return {
        'delta': round(delta, 4),
        'rel_change_pct': round(rel_change, 2),
        'robustness': round(robustness, 4),
    }


def compute_dcg(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain"""
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def compute_cg(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Compute Cumulative Gain (without discount)"""
    retrieved_k = retrieved[:k]
    cg = sum(1 for r in retrieved_k if r in relevant)
    return float(cg)


def compute_err(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Compute Expected Reciprocal Rank - probability-based metric"""
    retrieved_k = retrieved[:k]
    err = 0.0
    rel_so_far = 1.0
    
    for i, r in enumerate(retrieved_k):
        is_relevant = 1.0 if r in relevant else 0.0
        err += rel_so_far * is_relevant / (i + 1)
        rel_so_far *= (1.0 - is_relevant)
        
        if rel_so_far == 0:
            break
    
    return err


def compute_rbp(retrieved: List[str], relevant: Set[str], k: int = 10, p: float = 0.5) -> float:
    """Compute Rank-Biased Precision with persistence parameter p"""
    retrieved_k = retrieved[:k]
    rbp = 0.0
    
    for i, r in enumerate(retrieved_k):
        is_relevant = 1.0 if r in relevant else 0.0
        rbp += ((1.0 - p) * (p ** i)) * is_relevant
    
    return rbp


def compute_r_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute R-Precision: precision at R (total relevant count)"""
    if not relevant:
        return 0.0
    
    r = len(relevant)
    retrieved_r = retrieved[:r]
    num_relevant = sum(1 for r_item in retrieved_r if r_item in relevant)
    
    return num_relevant / r if r > 0 else 0.0


def compute_bpref(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Compute Binary Preference metric"""
    if not relevant:
        return 0.0
    
    retrieved_k = retrieved[:k]
    bpref = 0.0
    
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            non_rel_ranked_higher = sum(1 for j in range(i) if retrieved_k[j] not in relevant)
            bpref += 1.0 - (non_rel_ranked_higher / len(relevant))
    
    return bpref / len(relevant) if relevant else 0.0


def compute_novelty(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Compute Novelty: avoid duplicate/repeated relevant results"""
    retrieved_k = retrieved[:k]
    novelty = 0.0
    seen_relevant = {}
    
    for r in retrieved_k:
        if r in relevant:
            count = seen_relevant.get(r, 0)
            seen_relevant[r] = count + 1
            novelty += 1.0 / (1.0 + count)
    
    return novelty / k if k > 0 else 0.0


def compute_percentile_stats(
    metric_values: List[float]
) -> Dict:
    """
    Compute percentile statistics for a set of metric values
    """
    if not metric_values:
        return {}
    
    return {
        'mean': round(np.mean(metric_values), 4),
        'std': round(np.std(metric_values), 4),
        'min': round(np.min(metric_values), 4),
        'p25': round(np.percentile(metric_values, 25), 4),
        'median': round(np.percentile(metric_values, 50), 4),
        'p75': round(np.percentile(metric_values, 75), 4),
        'p90': round(np.percentile(metric_values, 90), 4),
        'p95': round(np.percentile(metric_values, 95), 4),
        'max': round(np.max(metric_values), 4),
    }


def compute_aggregate_metrics(
    all_metrics: Dict[int, List[Dict]],
    k_values: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
) -> Dict:
    """
    Compute aggregated metrics over all queries for each k value
    """
    aggregated = {}
    
    for k in k_values:
        if not all_metrics.get(k):
            continue
        
        metrics_list = all_metrics[k]
        
        aggregated[f'P@{k}'] = round(np.mean([m['precision_at_k'] for m in metrics_list]), 4)
        aggregated[f'R@{k}'] = round(np.mean([m['recall_at_k'] for m in metrics_list]), 4)
        aggregated[f'MAP@{k}'] = round(np.mean([m['ap'] for m in metrics_list]), 4)
        aggregated[f'NDCG@{k}'] = round(np.mean([m['ndcg'] for m in metrics_list]), 4)
        aggregated[f'MRR@{k}'] = round(np.mean([m['mrr'] for m in metrics_list]), 4)
        
        aggregated[f'F1@{k}'] = round(np.mean([m['f1_at_k'] for m in metrics_list]), 4)
        aggregated[f'Hit@{k}'] = round(np.mean([m['hit_at_k'] for m in metrics_list]), 4)
        aggregated[f'AvgRank@{k}'] = round(np.mean([m['avg_rank'] for m in metrics_list]), 2)
        
        aggregated[f'DCG@{k}'] = round(np.mean([m['dcg'] for m in metrics_list]), 4)
        aggregated[f'CG@{k}'] = round(np.mean([m['cg'] for m in metrics_list]), 4)
        aggregated[f'ERR@{k}'] = round(np.mean([m['err'] for m in metrics_list]), 4)
        aggregated[f'RBP@{k}'] = round(np.mean([m['rbp'] for m in metrics_list]), 4)
        
        if k == 10:
            aggregated['R-Precision'] = round(np.mean([m['r_precision'] for m in metrics_list]), 4)
            aggregated['Bpref@10'] = round(np.mean([m['bpref'] for m in metrics_list]), 4)
            aggregated['Novelty@10'] = round(np.mean([m['novelty'] for m in metrics_list]), 4)
        
        ndcg_values = [m['ndcg'] for m in metrics_list]
        ndcg_stats = compute_percentile_stats(ndcg_values)
        aggregated[f'NDCG@{k}_stats'] = ndcg_stats
        
        high_perf = sum(1 for m in metrics_list if m['ndcg'] >= 0.30) / len(metrics_list)
        med_perf = sum(1 for m in metrics_list if 0.15 <= m['ndcg'] < 0.30) / len(metrics_list)
        low_perf = sum(1 for m in metrics_list if m['ndcg'] < 0.15) / len(metrics_list)
        
        aggregated[f'Performance_Distribution@{k}'] = {
            'high': round(high_perf * 100, 1),
            'medium': round(med_perf * 100, 1),
            'low': round(low_perf * 100, 1),
        }
    
    return aggregated


# ============================================================================
# Original Evaluation Functions
# ============================================================================

def compute_metrics(retrieved: List[str], relevant: Set[str], k: int = 10) -> Dict:
    """Compute retrieval metrics"""
    retrieved_k = retrieved[:k]
    
    # Precision@K
    num_relevant = len([r for r in retrieved_k if r in relevant])
    precision = num_relevant / k if k > 0 else 0
    
    # Recall@K
    recall = num_relevant / len(relevant) if len(relevant) > 0 else 0
    
    # Average Precision
    ap = 0
    num_found = 0
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            num_found += 1
            ap += num_found / (i + 1)
    ap = ap / len(relevant) if len(relevant) > 0 else 0
    
    # NDCG
    dcg = 0
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            dcg += 1 / np.log2(i + 2)
    
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # MRR
    for i, r in enumerate(retrieved_k):
        if r in relevant:
            mrr = 1 / (i + 1)
            break
    else:
        mrr = 0
    
    return {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'ap': ap,
        'ndcg': ndcg,
        'mrr': mrr
    }


def evaluate_retriever(
    retriever, 
    queries: List[Dict], 
    all_asins: List[str], 
    k_values: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], 
    save_candidates_path: str = None,
    mode: str = None,
    return_query_results: bool = False
) -> Dict:
    """Evaluate a retriever on queries
    
    Args:
        retriever: The retriever instance
        queries: List of query dictionaries
        all_asins: List of all ASIN identifiers
        k_values: K values for evaluation metrics
        save_candidates_path: Path to save candidates (optional)
        mode: Query mode tag (clean/noisy)
        return_query_results: If True, return (metrics, query_results) tuple instead of just metrics
    
    Returns:
        If return_query_results=False: aggregated metrics dict
        If return_query_results=True: (aggregated metrics dict, query_results list) tuple
    """
    import time
    
    retriever_type = type(retriever).__name__
    log_progress(f"[EVAL_RETRIEVER_START] Starting evaluation with {retriever_type} ({len(queries)} queries)")
    
    all_metrics = {k: [] for k in k_values}
    all_candidates = []
    query_results = []  # 新增：收集每个查询的结果
    
    search_times = []
    
    for idx, q in enumerate(queries):
        asin = q.get('asin', '')
        query_text = q.get('query', '')
        
        if not query_text:
            pq = q.get('personalized_query', {})
            query_text = pq.get('original', '') or pq.get('noisy', '')
        
        if not query_text:
            continue
        
        search_start = time.time()
        results = retriever.search(query_text, top_k=max(k_values))
        search_time = time.time() - search_start
        search_times.append(search_time)
        
        retrieved_asins = [r[0] for r in results]
        
        mode_tag = f"[{mode}]" if mode else ""
        
        if save_candidates_path:
            all_candidates.append({
                'query': query_text,
                'asin': asin,
                'candidates': results
            })
        
        # 新增：保存top10结果
        if return_query_results:
            top10_results = [
                {
                    'rank': rank + 1,
                    'asin': result[0],
                    'score': float(result[1]) if isinstance(result[1], (int, float, np.number)) else float(result[1]) if hasattr(result[1], '__float__') else result[1]
                }
                for rank, result in enumerate(results[:10])
            ]
            
            query_results.append({
                'query_idx': idx,
                'asin': asin,
                'query_text': query_text,
                'search_time_seconds': search_time,
                'top10_results': top10_results
            })
        
        relevant = {asin}
        for k in k_values:
            metrics = compute_enhanced_metrics(retrieved_asins, relevant, k)
            all_metrics[k].append(metrics)
    
    if save_candidates_path and all_candidates:
        with open(save_candidates_path, 'w') as f:
            json.dump({'candidates': all_candidates, 'retriever': 'bm25'}, f)

    aggregated = compute_aggregate_metrics(all_metrics, k_values)
    
    # 返回结果
    if return_query_results:
        return aggregated, query_results
    else:
        return aggregated


def load_cached_candidates(cache_file: str) -> List[Dict]:
    """Load cached candidates from file"""
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, 'r') as f:
        data = json.load(f)
    return data.get('candidates', [])


# ============================================================================
# Convenience function to load all data
# ============================================================================

def load_all_product_data(
    meta_file: str,
    review_file: str,
    qa_file: str,
    asins: Set[str],
    max_reviews_per_product: int = 25,
    max_qa_per_product: int = 25,
    min_review_words: int = 0
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Load all product data including metadata, reviews, and Q&A.
    
    Args:
        meta_file: Path to metadata JSON file
        review_file: Path to review JSON file
        qa_file: Path to Q&A JSON file
        asins: Set of ASINs to load
        max_reviews_per_product: Maximum reviews per product
        max_qa_per_product: Maximum Q&A per product
        min_review_words: Minimum words for reviews
        
    Returns:
        (target_products, all_metadata) tuple
    """
    # Load metadata
    products, all_metadata = load_product_metadata(meta_file, asins)
    
    # Load reviews
    products = load_reviews_for_products(
        review_file, products, 
        max_reviews_per_product=max_reviews_per_product,
        min_review_words=min_review_words
    )
    
    # Load Q&A
    products = load_qa_for_products(
        qa_file, products,
        max_qa_per_product=max_qa_per_product
    )
    
    return products, all_metadata


def load_preprocessed_products(
    cache_dir: str,
    category: str = "Arts_Crafts_and_Sewing",
    target_asins: Set[str] = None
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load preprocessed data (auto-detect format)."""
    products_file = os.path.join(cache_dir, f"products_{category}.pkl")
    if os.path.exists(products_file):
        return load_preprocessed_products_v2(cache_dir, category, target_asins)
    else:
        return load_preprocessed_products_v1(cache_dir, category, target_asins)


def load_preprocessed_products_v1(
    cache_dir: str,
    category: str = "Arts_Crafts_and_Sewing",
    target_asins: Set[str] = None
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load from old dict-based format."""
    import pickle
    
    products_file = os.path.join(cache_dir, f"cleaned_products_{category}.pkl")
    metadata_file = os.path.join(cache_dir, f"cleaned_metadata_{category}.pkl")
    
    log_with_timestamp(f"Loading preprocessed data (v1) from {cache_dir}...")
    
    with open(products_file, 'rb') as f:
        all_products = pickle.load(f)
    
    with open(metadata_file, 'rb') as f:
        all_metadata = pickle.load(f)
    
    if target_asins is None:
        target_products = all_products
    else:
        target_products = {asin: all_products[asin] for asin in target_asins if asin in all_products}
    
    log_with_timestamp(f"Loaded {len(target_products)} target products")
    log_with_timestamp(f"Loaded {len(all_metadata)} total products")
    
    return target_products, all_metadata


def load_preprocessed_products_v2(
    cache_dir: str,
    category: str = "Arts_Crafts_and_Sewing",
    target_asins: Set[str] = None
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load from pandas format - optimized: only load target products."""
    import pandas as pd
    
    products_file = os.path.join(cache_dir, f"products_{category}.pkl")
    reviews_file = os.path.join(cache_dir, f"reviews_{category}.pkl")
    qa_file = os.path.join(cache_dir, f"qa_{category}.pkl")
    
    log_with_timestamp(f"Loading preprocessed data (v2 optimized) from {cache_dir}...")
    
    if target_asins:
        # Only load target products - FAST!
        df_products = pd.read_pickle(products_file)
        df_products = df_products[df_products['asin'].isin(target_asins)]
        log_with_timestamp(f"  Loaded {len(df_products)} target products")
        
        # Load reviews only for target products
        df_reviews = pd.read_pickle(reviews_file)
        df_reviews = df_reviews[df_reviews['asin'].isin(target_asins)]
        
        # Load Q&A only for target products
        if os.path.exists(qa_file):
            df_qa = pd.read_pickle(qa_file)
            df_qa = df_qa[df_qa['asin'].isin(target_asins)]
        else:
            df_qa = pd.DataFrame()
    else:
        # Load all (original behavior)
        df_products = pd.read_pickle(products_file)
        df_reviews = pd.read_pickle(reviews_file)
        df_qa = pd.read_pickle(qa_file) if os.path.exists(qa_file) else pd.DataFrame()
    
    log_with_timestamp(f"Converting to dict...")
    
    # Convert to dict
    products = {}
    all_metadata = {}
    
    for p in df_products.to_dict('records'):
        asin = p['asin']
        product = {
            'asin': asin,
            'title': p['title'],
            'brand': p['brand'],
            'category': p['category'],
            'feature': p['feature'],
            'description': p['description'],
            'rank': p['rank'],
            'also_buy': p['also_buy'],
            'also_view': p['also_view'],
            'reviews': [],
            'qa': [],
        }
        products[asin] = product
        all_metadata[asin] = product.copy()
    
    # Add reviews
    for asin, group in df_reviews.groupby('asin'):
        reviews = [{
            'summary': r['summary'],
            'text': r['reviewText'],
            'reviewText': r['reviewText'],
            'vote': r['vote'],
            'overall': r['overall'],
        } for r in group.to_dict('records')]
        
        if asin in products:
            products[asin]['reviews'] = reviews
        if asin in all_metadata:
            all_metadata[asin]['reviews'] = reviews
    
    # Add Q&A
    if len(df_qa) > 0:
        for asin, group in df_qa.groupby('asin'):
            qa_list = [{
                'question': q['question'],
                'answer': q['answer'],
            } for q in group.to_dict('records')]
            
            if asin in products:
                products[asin]['qa'] = qa_list
            if asin in all_metadata:
                all_metadata[asin]['qa'] = qa_list
    
    log_with_timestamp(f"Loaded {len(products)} target products")
    log_with_timestamp(f"Loaded {len(all_metadata)} total products")
    
    return products, all_metadata

