#!/usr/bin/env python3
"""
直接从文本提取5维复杂度特征的轻量工具
不依赖LingConv API，本地计算
"""
import re
import sys
import spacy

# 加载spaCy模型
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("下载 en_core_web_sm 模型...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


def extract_5d_features(text: str) -> dict:
    """
    从文本提取5维复杂度特征

    返回:
    {
        'clause_depth': 从句嵌套深度 (0-1范围，归一化),
        'dep_distance': 依存距离 (0-1范围),
        'modifier_density': 修饰语密度 (0-1范围),
        'coord_chain': 并列链深度 (0-1范围),
        'negation_scope': 否定作用域 (0-1范围),
        'avg_sentence_length': 平均句子长度 (词数)
    }
    """
    doc = nlp(text)
    sentences = list(doc.sents)

    if not sentences:
        return {
            'clause_depth': 0.0,
            'dep_distance': 0.0,
            'modifier_density': 0.0,
            'coord_chain': 0.0,
            'negation_scope': 0.0,
            'avg_sentence_length': 0.0
        }

    features_per_sent = []
    for sent in sentences:
        features = _compute_sentence_features(sent)
        features_per_sent.append(features)

    # 平均各维度
    n = len(features_per_sent)
    result = {
        'clause_depth': sum(f['clause_depth'] for f in features_per_sent) / n,
        'dep_distance': sum(f['dep_distance'] for f in features_per_sent) / n,
        'modifier_density': sum(f['modifier_density'] for f in features_per_sent) / n,
        'coord_chain': sum(f['coord_chain'] for f in features_per_sent) / n,
        'negation_scope': sum(f['negation_scope'] for f in features_per_sent) / n,
        'avg_sentence_length': sum(f['word_count'] for f in features_per_sent) / n,
    }

    return result


def _compute_sentence_features(sent) -> dict:
    """计算单个句子的复杂度特征"""
    tokens = [t for t in sent if not t.is_space and not t.is_punct]
    n = len(tokens)
    if n == 0:
        return {
            'clause_depth': 0.0,
            'dep_distance': 0.0,
            'modifier_density': 0.0,
            'coord_chain': 0.0,
            'negation_scope': 0.0,
            'word_count': 0,
        }

    # 1. 从句深度 - 通过检测从句关系
    SUBORDINATE_DEPS = {'acl', 'acl:relcl', 'relcl', 'advcl', 'ccomp', 'xcomp', 'csubj', 'csubjpass'}
    clause_depth = 0
    for token in tokens:
        if token.dep_ in SUBORDINATE_DEPS:
            clause_depth += 1
    clause_depth = min(clause_depth / max(n, 1), 1.0)

    # 2. 依存距离
    total_distance = 0
    for token in tokens:
        if token.head != token and token.head.i < len(tokens):
            total_distance += abs(token.i - token.head.i)
    max_possible_distance = n * (n - 1) / 2 if n > 1 else 1
    dep_distance = total_distance / max_possible_distance if max_possible_distance > 0 else 0.0

    # 3. 修饰语密度
    MODIFIER_DEPS = {'amod', 'advmod', 'compound', 'nn', 'nummod', 'poss'}
    modifier_count = sum(1 for t in tokens if t.dep_ in MODIFIER_DEPS)
    modifier_density = modifier_count / n

    # 4. 并列链深度
    conj_count = sum(1 for t in tokens if t.dep_ == 'conj')
    coord_chain = min(conj_count, 5) / 5.0  # 归一化到0-1

    # 5. 否定作用域
    negation_scope = 0
    for token in tokens:
        if token.dep_ == 'neg':
            # 计算neg节点下的子树大小
            subtree_size = len(list(token.subtree))
            negation_scope = max(negation_scope, subtree_size)
    negation_scope = min(negation_scope / n, 1.0) if n > 0 else 0.0

    return {
        'clause_depth': clause_depth,
        'dep_distance': round(dep_distance, 4),
        'modifier_density': round(modifier_density, 4),
        'coord_chain': round(coord_chain, 4),
        'negation_scope': round(negation_scope, 4),
        'word_count': n,
    }


def complexity_distance(c: dict, target: dict) -> float:
    """计算候选与目标的复杂度距离"""
    import numpy as np
    stds = {
        'clause_depth': 0.3,
        'dep_distance': 0.15,
        'modifier_density': 0.25,
        'coord_chain': 0.2,
        'negation_scope': 0.2,
    }
    dist = 0.0
    for dim in stds:
        diff = abs(c.get(dim, 0) - target.get(dim, 0))
        dist += (diff / stds[dim]) ** 2
    return round(np.sqrt(dist), 4)


if __name__ == '__main__':
    test_sentences = [
        "Beautiful Crystal Rhinestones which are made by ThreadNanny, not expensive at 25.95, for Embroidery.",
        "Shimmering Crystal Rhinestones, which boast unmatched brilliance, are crafted by ThreadNanny and priced affordably at 25.95, making them perfect for Embroidery projects that demand elegant sparkle.",
        "The Rhinestones and Crystal Embroidery kit on ThreadNanny, priced at 25.95, offers a brilliant sparkle that is not expensive, featuring high-quality, vivid threads.",
    ]

    print("=" * 70)
    print("5维复杂度特征提取工具 (本地spaCy)")
    print("=" * 70)

    for i, sent in enumerate(test_sentences, 1):
        print(f"\n句子 {i}: {sent[:80]}...")
        features = extract_5d_features(sent)
        print(f"  clause_depth:      {features['clause_depth']:.4f}")
        print(f"  dep_distance:      {features['dep_distance']:.4f}")
        print(f"  modifier_density:  {features['modifier_density']:.4f}")
        print(f"  coord_chain:       {features['coord_chain']:.4f}")
        print(f"  negation_scope:    {features['negation_scope']:.4f}")
        print(f"  avg_sentence_length: {features['avg_sentence_length']:.1f}")

    # 如果有命令行参数，当作输入文本处理
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        print(f"\n输入文本: {text[:100]}...")
        features = extract_5d_features(text)
        print(f"\n5维复杂度:")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")