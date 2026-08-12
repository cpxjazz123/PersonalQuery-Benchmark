#!/usr/bin/env python3
"""基于用户真实错误训练的 LambdaMART 模型生成噪声查询"""

import sys
import re
from pathlib import Path


_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import (
    load_query_records, build_query_tasks,
    write_json_array, log, load_user_errors
)
from token_level_lambdamart_user_based import (
    tokenize_query, remove_stop_words
)


# BPE-aware error injection (iter #178)
# When BPE_SIMILARITY_THRESHOLD is set, only inject errors that cause
# BPE tokenization divergence between the original and the (misspelled)
# corrected form — this simulates E5's subword sensitivity (paper §3.2
# E5 Δ=11.16 subword effect).
_BPE_TOKENIZER_NAME = 'intfloat/e5-base-v2'
_BPE_SIMILARITY_THRESHOLD = 0.8
_BPE_TOKENIZER = None  # lazy-loaded


def _get_bpe_tokenizer():
    """Lazy-load E5 BPE tokenizer for subword impact scoring (iter #178).

    Per loop.md §1 Rule 7: raises on failure, no fallback. Stage 05 caller
    must have `transformers` installed.
    """
    global _BPE_TOKENIZER
    if _BPE_TOKENIZER is None:
        from transformers import AutoTokenizer  # local import keeps top-of-file lean
        _BPE_TOKENIZER = AutoTokenizer.from_pretrained(_BPE_TOKENIZER_NAME)
    return _BPE_TOKENIZER


def compute_bpe_token_diff(original_word: str, corrected_word: str) -> float:
    """Compute BPE-tokenization Jaccard similarity between two word forms (iter #178).

    Returns float in [0, 1]:
        1.0 → identical BPE token sequences (no subword impact)
        0.0 → disjoint BPE token sequences (max subword divergence)

    Used by Stage 05 to gate which user errors actually impact subword
    retrievers (E5 / MiniLM / STAR / ANCE / ColBERTv2). When
    compute_bpe_token_diff < BPE_SIMILARITY_THRESHOLD the error is
    considered subword-impactful and eligible for injection.
    """
    tok = _get_bpe_tokenizer()
    orig_tokens = tok.tokenize(original_word)
    corr_tokens = tok.tokenize(corrected_word)
    if not orig_tokens or not corr_tokens:
        # Per loop.md §1 Rule 7: empty input → explicit result, no fallback
        return 0.0
    orig_set = set(orig_tokens)
    corr_set = set(corr_tokens)
    intersection = len(orig_set & corr_set)
    union = len(orig_set | corr_set)
    return intersection / union if union > 0 else 0.0


def build_config(category: str) -> dict:
    """根据类别构建配置"""
    base = '/home/wlia0047/ar57/wenyu'
    return {
        'category': category,
        'query_file': f'{base}/result/personal_query/06_query/{category}/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json',
        'user_error_file': f'{base}/result/personal_query/04_writing_analysis/{category}/writing_error.json',
        'model_file': f'{base}/result/personal_query/07_inject_noisy/models/lambdamart_{category}_user_based.json',
        'output_file': f'{base}/result/personal_query/07_inject_noisy/{category}/noisy_query.json',
    }


# ========================================
# 辅助函数：扩展匹配规则
# ========================================
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


# 视觉相似字符对
VISUAL_SIMILAR_PAIRS = [
    ('m', 'n'), ('b', 'd'), ('p', 'q'), ('i', 'l'), ('o', 'e'),
    ('a', 'e'), ('u', 'v'), ('g', 'q'), ('w', 'vv'),
]

# 键盘相邻键映射
KEYBOARD_ADJACENT = {
    'q': {'w', 'a'},
    'w': {'q', 'e', 'a', 's'},
    'e': {'w', 'r', 'd', 's'},
    'r': {'e', 't', 'f', 'd'},
    't': {'r', 'y', 'g', 'f'},
    'y': {'t', 'u', 'h', 'g'},
    'u': {'y', 'i', 'j', 'h'},
    'i': {'u', 'o', 'k', 'j'},
    'o': {'i', 'p', 'l', 'k'},
    'p': {'o', 'l'},
    'a': {'q', 'w', 's', 'z'},
    's': {'w', 'e', 'd', 'x', 'z', 'a'},
    'd': {'e', 'r', 'f', 'c', 'x', 's'},
    'f': {'r', 't', 'g', 'v', 'c', 'd'},
    'g': {'t', 'y', 'h', 'b', 'v', 'f'},
    'h': {'y', 'u', 'j', 'n', 'b', 'g'},
    'j': {'u', 'i', 'k', 'm', 'n', 'h'},
    'k': {'i', 'o', 'l', 'm', 'j'},
    'l': {'o', 'p', 'k'},
    'z': {'a', 's', 'x'},
    'x': {'s', 'd', 'c', 'z'},
    'c': {'d', 'f', 'v', 'x'},
    'v': {'f', 'g', 'b', 'c'},
    'b': {'g', 'h', 'n', 'v'},
    'n': {'h', 'j', 'm', 'b'},
    'm': {'j', 'k', 'n'},
}

# 音近替换模式
PHONETIC_PATTERNS = [
    ('ei', 'ie'),  # receive/recieve
    ('ph', 'f'),   # phone/fone
    ('kn', 'n'),   # knife/nife
    ('gh', ''),    # night/nite
    ('ck', 'k'),   # back/bak
    ('wr', 'r'),   # write/rite
    ('mb', 'm'),   # thumb/thum
    ('ps', 's'),   # psychology/psychology
    ('gh', 'g'),   # ghost/gost
]

# 元音替换对
VOWEL_REPLACEMENTS = [
    ('a', 'e'),  # Many vowel substitutions in English
    ('e', 'i'),
    ('i', 'y'),
    ('o', 'u'),
]

# 重复字母模式
DOUBLE_LETTERS = {'ss', 'tt', 'll', 'rr', 'pp', 'mm', 'nn', 'ff', 'gg', 'bb', 'dd', 'kk', 'cc'}


def is_transposition(s1: str, s2: str) -> bool:
    """检查是否只有一个字符换位（ab vs ba）"""
    if len(s1) != len(s2):
        return False
    if s1 == s2:
        return False
    diff_idx = []
    for i, (c1, c2) in enumerate(zip(s1.lower(), s2.lower())):
        if c1 != c2:
            diff_idx.append((i, c1, c2))
    if len(diff_idx) == 2:
        (i, c1, c2), (j, c3, c4) = diff_idx
        # 两个字符互换位置
        return i + 1 == j and c1 == c4 and c2 == c3
    return False


def is_phonetic_similar(s1: str, s2: str) -> bool:
    """检查是否只有音近替换差异"""
    if len(s1) != len(s2):
        return False
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    diff_count = 0
    for i in range(len(s1_lower)):
        c1, c2 = s1_lower[i], s2_lower[i]
        if c1 != c2:
            # 检查是否是音近模式
            is_phonetic = False
            for p1, p2 in PHONETIC_PATTERNS:
                if (p1 in s1_lower and p2 in s2_lower) or (p2 in s1_lower and p1 in s2_lower):
                    is_phonetic = True
                    break
                # 单字符音近
                if (c1 == 'f' and c2 == 'p') or (c1 == 'p' and c2 == 'f'):
                    is_phonetic = True
                    break
                if (c1 == 'v' and c2 == 'f') or (c1 == 'f' and c2 == 'v'):
                    is_phonetic = True
                    break
            if not is_phonetic:
                return False
            diff_count += 1
    return diff_count > 0


def is_vowel_substitution(s1: str, s2: str) -> bool:
    """检查是否只有元音替换（且长度相同）"""
    if len(s1) != len(s2):
        return False
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    diff_count = 0
    vowels = set('aeiou')
    for i in range(len(s1_lower)):
        c1, c2 = s1_lower[i], s2_lower[i]
        if c1 != c2:
            # 必须一个是元音
            if (c1 in vowels and c2 in vowels) or (c1 not in vowels and c2 not in vowels):
                return False
            diff_count += 1
    return diff_count > 0


def is_double_letter_error(s1: str, s2: str) -> bool:
    """检查是否只是重复字母数量的差异（short/shot,满/慢）"""
    if abs(len(s1) - len(s2)) > 2:
        return False

    # 找连续相同字母
    def get_double_letters(s):
        result = set()
        for i in range(len(s) - 1):
            if s[i] == s[i+1]:
                result.add(s[i:i+2].lower())
        return result

    doubles1 = get_double_letters(s1.lower())
    doubles2 = get_double_letters(s2.lower())

    # 如果都有重复字母但不一样
    if doubles1 != doubles2:
        return True

    # 检查长度差异是否是重复字母引起
    if len(s1) != len(s2):
        # 计算非重复部分的长度
        def remove_doubles(s):
            result = []
            i = 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == s[i+1]:
                    result.append(s[i])
                    i += 2
                else:
                    result.append(s[i])
                    i += 1
            return ''.join(result)
        return remove_doubles(s1.lower()) == remove_doubles(s2.lower())

    return False


def is_visually_similar(s1: str, s2: str) -> bool:
    """检查两个字符串是否只有视觉相似字符差异"""
    if len(s1) != len(s2):
        return False
    diff_count = 0
    for c1, c2 in zip(s1.lower(), s2.lower()):
        if c1 != c2:
            # 检查是否是视觉相似对
            is_sim = any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in VISUAL_SIMILAR_PAIRS)
            if not is_sim:
                return False
            diff_count += 1
    return diff_count > 0


def is_keyboard_adjacent(s1: str, s2: str) -> bool:
    """检查两个字符串是否只有键盘相邻键差异（最多一个字符不同）"""
    if len(s1) != len(s2):
        return False
    diff_positions = []
    for i, (c1, c2) in enumerate(zip(s1.lower(), s2.lower())):
        if c1 != c2:
            diff_positions.append((i, c1, c2))

    if len(diff_positions) == 0:
        return False
    if len(diff_positions) > 2:  # 太多差异
        return False

    for pos, c1, c2 in diff_positions:
        # 检查两个字符是否在键盘上相邻
        adj1 = KEYBOARD_ADJACENT.get(c1, set())
        adj2 = KEYBOARD_ADJACENT.get(c2, set())
        if c2 not in adj1 and c1 not in adj2:
            return False
    return True


def find_similar_error(token: str, user_errors: list) -> dict:
    """找到与 token 匹配的用户历史错误

    用户错误模式：original（用户写错的形式）→ corrected（正确形式）

    匹配规则（按优先级）：
    1. 首尾字母相同（与 original 比较，跳过撇号）
    2. 编辑距离为1（与 original 比较，跳过撇号）
    3. 字符换位（transposition）
    4. 视觉相似字符差异
    5. 键盘相邻键差异
    """
    if not user_errors:
        return None

    token_lower = token.lower()

    for err in user_errors:
        original = err.get('original', '')
        corrected = err.get('corrected', '')
        if not original or not corrected:
            continue

        original_lower = original.lower()

        # 清理撇号后的版本（用于比较）
        def remove_apostrophe(s):
            return s.replace("'", "").replace("'", "")

        token_clean = remove_apostrophe(token_lower)
        original_clean = remove_apostrophe(original_lower)

        # 规则1: 首尾字母与 original 相同（使用清理后的版本）+ 长度相近
        if token_clean and original_clean:
            if len(token_clean) >= 2 and len(original_clean) >= 2:
                # 长度差距不能太大（不超过2）
                if abs(len(token_clean) - len(original_clean)) <= 2:
                    if token_clean[0] == original_clean[0] and token_clean[-1] == original_clean[-1]:
                        return err

        # 规则2: 编辑距离为1（使用清理后的版本）
        if abs(len(token_clean) - len(original_clean)) <= 1:
            if edit_distance(token_clean, original_clean) == 1:
                return err

        # 规则3: 字符换位（ab vs ba）
        if len(token_lower) == len(original_lower) and len(token_lower) >= 2:
            if is_transposition(token_lower, original_lower):
                return err

        # 规则4: 视觉相似字符差异
        if len(token_lower) == len(original_lower) and len(token_lower) >= 2:
            if is_visually_similar(token_lower, original_lower):
                return err

        # 规则5: 键盘相邻键差异
        if len(token_lower) == len(original_lower) and len(token_lower) >= 2:
            if is_keyboard_adjacent(token_lower, original_lower):
                return err

    return None


def apply_error_to_query(query: str, error_case: dict, token: str) -> str:
    """将 token 替换为 error_case 的 original（用户错误形式）"""
    original = error_case.get('original', '')
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    noisy_query = pattern.sub(original, query, count=1)
    return noisy_query


def process_batch(model, tasks: list, category: str, bpe_aware: bool = False):
    """批量处理任务

    Args:
        model: LambdaMART model (currently unused; rule-based find_similar_error
            drives injection — see iter #47 audit).
        tasks: list of dicts with keys {uid, asin, clean_query, errors}.
        category: domain name.
        bpe_aware (iter #178): if True, gate injection by BPE-tokenization
            divergence between original and corrected forms. Only errors with
            compute_bpe_token_diff < _BPE_SIMILARITY_THRESHOLD are eligible.
            This simulates E5's subword sensitivity (paper §3.2 E5 Δ=11.16).
    """
    results = []
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        uid = task['uid']
        asin = task['asin']
        clean_query = task['clean_query']
        user_errors = task['errors']

        try:
            tokens = tokenize_query(clean_query)
            clean_tokens = remove_stop_words(tokens)

            if not clean_tokens or not user_errors:
                results.append({
                    'uid': uid,
                    'asin': asin,
                    'clean_query': clean_query,
                    'noisy_query': clean_query,
                    'query_rewritten': False,
                    'selected_token': None,
                    'score': 0.0,
                    'applied_error': None,
                    'status': 'skipped',
                    'reason': 'no_tokens_or_errors',
                })
                continue

            # 直接遍历所有 token，用规则匹配
            best_token = None
            error_case = None
            bpe_similarity = None
            bpe_impact = None
            for token in clean_tokens:
                err = find_similar_error(token, user_errors)
                if err:
                    # iter #178: BPE-aware gating
                    if bpe_aware:
                        orig = err.get('original') or err.get('corrected') or token
                        corr = err.get('corrected') or err.get('original') or token
                        bpe_similarity = compute_bpe_token_diff(orig, corr)
                        bpe_impact = bpe_similarity < _BPE_SIMILARITY_THRESHOLD
                        if not bpe_impact:
                            # Skip this error: not subword-impactful
                            continue
                    best_token = token
                    error_case = err
                    break

            # 应用错误
            noisy_query = clean_query
            applied_error = None
            if error_case:
                noisy_query = apply_error_to_query(clean_query, error_case, best_token)
                applied_error = {
                    'original': error_case.get('original'),
                    'corrected': error_case.get('corrected'),
                    'error_type': error_case.get('error_type', 'writing_error'),
                }
                if bpe_aware:
                    applied_error['bpe_similarity'] = bpe_similarity
                    applied_error['bpe_impact'] = bpe_impact

            output_item = {
                'uid': uid,
                'asin': asin,
                'clean_query': clean_query,
                'noisy_query': noisy_query,
                'query_rewritten': noisy_query != clean_query,
                'selected_token': best_token,
                'score': 1.0 if error_case else 0.0,
                'applied_error': applied_error,
                'status': 'success',
            }

            results.append(output_item)

            if idx % 500 == 0 or idx == total:
                log(f"[{idx}/{total}] 处理中: uid={uid[:12]}, token={best_token}")

        except Exception as e:
            log(f"[{idx}/{total}] 错误: uid={uid[:12]}, error={e}")
            results.append({
                'uid': uid,
                'asin': asin,
                'clean_query': clean_query,
                'noisy_query': clean_query,
                'query_rewritten': False,
                'selected_token': None,
                'score': 0.0,
                'applied_error': None,
                'status': 'error',
                'error': str(e),
            })

    return results


def main(category: str, bpe_aware: bool = False):
    config = build_config(category)

    print("=" * 60)
    if bpe_aware:
        print(f"规则匹配 + BPE-aware 噪声注入 ({category})  threshold={_BPE_SIMILARITY_THRESHOLD}")
    else:
        print(f"规则匹配 噪声注入 ({category})")
    print("=" * 60)

    query_file = config['query_file']
    user_error_file = config['user_error_file']
    model_file = config['model_file']
    output_file = config['output_file']

    print(f"Query 文件: {query_file}")
    print(f"用户错误文件: {user_error_file}")
    print(f"模型文件(未使用): {model_file}")
    print(f"输出文件: {output_file}")

    # 加载用户错误
    print("\n加载用户错误数据...")
    user_errors = load_user_errors(user_error_file)
    print(f"有错误数据的用户: {len(user_errors)}")

    # 加载查询
    print("\n加载查询记录...")
    query_records = load_query_records(query_file)
    print(f"查询记录总数: {len(query_records)}")

    # 构建任务
    print("\n构建查询任务...")
    completed_keys = set()
    tasks = build_query_tasks(query_records, user_errors, completed_keys)
    print(f"待处理任务: {len(tasks)}")

    if not tasks:
        print("没有待处理的任务，退出")
        return

    # 处理（model参数已不使用，传入None保持兼容）
    print("\n开始处理...")
    results = process_batch(None, tasks, category, bpe_aware=bpe_aware)
    print(f"处理完成: {len(results)} 个结果")

    # 提取注入成功的记录
    injected_results = [r for r in results if r.get('query_rewritten')]

    # 写入
    if injected_results:
        print(f"\n写入结果到: {output_file}")
        write_json_array(injected_results, output_file, append=False)
        print(f"写入完成: {len(injected_results)} 条记录")

    # 统计
    success_count = sum(1 for r in results if r.get('status') == 'success')
    rewritten_count = len(injected_results)
    total_count = len(results)
    print("\n统计:")
    print(f"  总记录: {total_count}")
    print(f"  成功: {success_count}")
    print(f"  注入错误: {rewritten_count}")
    print(f"  注入率: {rewritten_count/total_count*100:.1f}%")

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stage 05 user-based LambdaMART noise injection')
    parser.add_argument('category', nargs='?', default='Baby_Products',
                        choices=['Baby_Products', 'Grocery_and_Gourmet_Food', 'Pet_Supplies'])
    parser.add_argument('--bpe-aware', action='store_true',
                        help='iter #178: gate injection by BPE tokenization '
                             'divergence (only inject errors with BPE similarity '
                             f'< {_BPE_SIMILARITY_THRESHOLD}). Models E5 subword sensitivity.')
    args = parser.parse_args()
    main(args.category, bpe_aware=args.bpe_aware)