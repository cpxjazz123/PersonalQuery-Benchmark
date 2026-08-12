#!/usr/bin/env python3
"""Common functions for extracting and filtering writing errors from reviews."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import sys
# 添加 PersoanlQuery 到 sys.path
_PERSOANL_QUERY_ROOT = Path(__file__).resolve().parents[2]
if str(_PERSOANL_QUERY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERSOANL_QUERY_ROOT))
from llm_client import QwenLocalClient
from filelock import FileLock


# ============================================================================
# 常量
# ============================================================================

WRITING_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/04_writing_analysis")
LOCK_SUFFIX = ".lock"
STAGE1_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/01_preference_extraction")
PROGRESS_INTERVAL_USERS = 100

# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']

# Prompt 文件路径
PROMPT_FILE = Path(__file__).resolve().parent / "extract_prompts.json"


# ============================================================================
# 工具函数
# ============================================================================

from common_utils import log  # 统一 log 函数



def is_simple_error(original: str, corrected: str) -> Tuple[bool, str]:
    """判断是否是简单错误（应被过滤）。"""
    orig = original.lower().strip()
    corr = corrected.lower().strip()

    # 空值或完全相同 -> 不算错误
    if not orig or not corr:
        return False, "empty"
    if orig == corr:
        return False, "identity"

    # 过滤仅空格差异（如 alot -> a lot）——放在 multi_word 之前，避免被拦截
    if orig.replace(" ", "") == corr.replace(" ", "") and orig != corr:
        return True, "space_diff"

    # 多词替换 -> 不在单字级别判断
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "multi_word"

    # 过滤仅相差标点的错误（如 it's <-> its, It's <-> It）
    # 同时处理弯引号（U+2019, U+2018）和直引号
    APOSTROPHES = ("'", "\u2019", "\u2018")
    # 去掉撇号后的原始词和修正词
    orig_no_apos = ''.join(c for c in orig if c not in APOSTROPHES)
    corr_no_apos = ''.join(c for c in corr if c not in APOSTROPHES)
    if orig_no_apos == corr_no_apos and orig != corr:
        # 完全相等但原文不同（仅撇号差异）
        return True, "punct_only"
    # 过滤撇号后剩余字母以 corrected 开头的（如 "It's" -> "It"，"it's" -> "it"）
    # 去掉撇号后，如果 orig_no_apos 以 corr_no_apos 开头但 orig != corr
    if (orig_no_apos.startswith(corr_no_apos) and orig != corr and
            len(corr_no_apos) >= 1 and len(orig_no_apos) > len(corr_no_apos)):
        return True, "punct_only"

    # 过滤动词形式变化
    COMMON_VERB_VARIATIONS = {
        ('was', 'were'), ('is', 'are'), ('are', 'is'),
        ('do', 'did'), ('does', 'did'),
    }
    if (orig, corr) in COMMON_VERB_VARIATIONS or (corr, orig) in COMMON_VERB_VARIATIONS:
        return True, "verb_variation"

    # 过滤人称代词变化
    COMMON_PRONOUN_VARIATIONS = {
        ('i', 'me'), ('me', 'i'),
        ('he', 'him'), ('him', 'he'),
        ('she', 'her'), ('her', 'she'),
        ('we', 'us'), ('us', 'we'),
        ('they', 'them'), ('them', 'they'),
    }
    if (orig, corr) in COMMON_PRONOUN_VARIATIONS or (corr, orig) in COMMON_PRONOUN_VARIATIONS:
        return True, "pronoun_variation"

    # 过滤词缀变化
    COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']
    if len(orig) > len(corr):
        orig, corr = corr, orig
    if len(corr) - len(orig) >= 1 and len(orig) >= 3:
        for affix in COMMON_AFFIXES:
            if corr == orig + affix:
                return True, "affix_variation"

    return False, ""


def format_elapsed(start_time: float) -> str:
    """格式化耗时"""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m{seconds}s"


# ============================================================================
# Batch Error Extraction
# ============================================================================

class BatchErrorExtractor:
    """批量提取错误"""
    
    def __init__(self, llm_client: QwenLocalClient, base_system: str, user_template: str,
                 max_tokens: int = 256, max_retries: int = 3):
        self.llm_client = llm_client
        self.base_system = base_system
        self.user_template = user_template
        self.max_tokens = max_tokens
        self.max_retries = max_retries
    
    def create_batch_prompt(self, reviews: List[str]) -> str:
        """创建批量 prompt，将 system 和 user 内容合并"""
        # 构建评论文本
        reviews_text = "\n".join([f"[Review {i}] {review}" for i, review in enumerate(reviews)])
        user_content = self.user_template.replace("{reviews_text}", reviews_text)
        # 合并 system 和 user content
        full_prompt = f"{self.base_system}\n\n{user_content}"
        return full_prompt
    
    def extract_batch_errors(self, reviews: List[str]) -> Dict:
        """批量提取错误"""
        print(f"[EXTRACT_BATCH] 方法被调用, reviews数量={len(reviews)}", flush=True)
        # 构建评论文本
        reviews_text = "\n".join([f"[Review {i}] {review}" for i, review in enumerate(reviews)])
        user_content = self.user_template.replace("{reviews_text}", reviews_text)
        
        print(f"[DEBUG_CALL] 开始调用, base_system长度={len(self.base_system)}, user_content长度={len(user_content)}", flush=True)
        for attempt in range(self.max_retries):
            try:
                response, cache_info = self.llm_client.call_with_cache(
                    system_base=self.base_system,
                    user_content=user_content,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                print(f"[LLM_RESULT] response长度={len(response)}, cache_info={cache_info}", flush=True)
                if not response:
                    print("[LLM_ERROR] response为空!", flush=True)
                
                # 打印 token 统计
                input_tokens = cache_info.get('input_tokens', 0)
                output_tokens = cache_info.get('output_tokens', 0)
                cache_read = cache_info.get('cache_read_input_tokens', 0)
                cache_create = cache_info.get('cache_creation_input_tokens', 0)
                log(f"[TOKEN] input={input_tokens}, output={output_tokens}, cache_read={cache_read}, cache_create={cache_create}")
                
                # 解析 JSON 响应
                # 去掉 markdown 代码块标记
                clean_response = response.strip()
                if clean_response.startswith('```'):
                    lines = clean_response.split('\n')
                    start_idx = 1 if lines and lines[0].strip().startswith('```') else 0
                    end_idx = len(lines) - 1 if lines and lines[-1].strip() == '```' else len(lines)
                    clean_response = '\n'.join(lines[start_idx:end_idx])
                
                try:
                    result_data = json.loads(clean_response.strip())
                    if result_data:
                        log(f"[DEBUG] 回复内容: {response[:500]}")
                        return {
                            "status": "success",
                            "result": result_data,
                            "reviews": reviews
                        }
                    else:
                        log(f"[ERROR] 回复为空: {response}")
                except json.JSONDecodeError as e:
                    log(f"[ERROR] JSON解析失败: {e}, response={response[:200]}")
                    if attempt < self.max_retries - 1:
                        continue
                    return {
                        "status": "error",
                        "error": f"JSON parse failed: {e}",
                        "reviews": reviews
                    }
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    continue
                return {
                    "status": "error",
                    "error": str(e),
                    "reviews": reviews
                }
        
        return {
            "status": "error",
            "error": "Max retries exceeded",
            "reviews": reviews
        }
# ============================================================================
# 主处理函数
# ============================================================================

def extract_and_filter_errors(category: str, config: Dict = None) -> None:
    """提取并过滤错误"""
    start_time = time.time()
    
    # 加载配置
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    
    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        prompts_config = json.load(f)
    
    # 合并配置
    if config is None:
        config = {}
    
    max_users = config.get("max_users", prompts_config.get("max_users", None))
    max_reviews_per_user = config.get("max_reviews_per_user", prompts_config.get("max_reviews_per_user", 10))
    max_workers = config.get("max_workers", prompts_config.get("max_workers", 2))
    max_tokens = config.get("max_tokens", prompts_config.get("max_tokens", 256))
    max_retries = config.get("max_retries", prompts_config.get("max_retries", 3))
    base_system = config.get("base_system", prompts_config.get("base_system", ""))
    user_template = config.get("user_content_template", prompts_config.get("user_content_template", ""))
    use_minimaxio = config.get("use_minimaxio", prompts_config.get("use_minimaxio", False))
    
    # 文件路径
    input_file = STAGE1_ROOT / category / "stage1_filtered_users_reviews.json"
    output_file = WRITING_ANALYSIS_ROOT / category / "writing_error.json"
    
    log(f"=== Processing category: {category} ===")
    log(f"Reading from: {input_file}")
    log(f"Config: max_users={max_users}, max_reviews_per_user={max_reviews_per_user}, max_workers={max_workers}")
    
    if not input_file.exists():
        log(f"Error: File not found: {input_file}")
        return
    
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    users = data.get("users", [])
    original_total_users = len(users)
    log(f"原始用户总数: {original_total_users}")
    
    # 断点续传：读取已完成的用户
    completed_user_ids = set()
    lock_file = Path(str(output_file) + LOCK_SUFFIX)
    with FileLock(lock_file, timeout=30):
        if output_file.exists():
            try:
                with output_file.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                for row in existing_data:
                    if "user_id" in row:
                        completed_user_ids.add(row["user_id"])
                log(f"Found {len(completed_user_ids)} already processed users, will skip them")
            except Exception as e:
                log(f"Warning: Could not read existing output file: {e}")
                existing_data = []
        else:
            existing_data = []
    
    # 过滤掉已完成的用户，再限制数量
    users_to_process = [u for u in users if u.get("user_id") not in completed_user_ids]
    if max_users:
        users_to_process = users_to_process[:max_users]
        log(f"Limited to {max_users} users (after skipping {len(completed_user_ids)} completed)")
    
    log(f"任务信息: 全局原始用户总数={original_total_users}, 本次max_users={max_users}, 本次待处理={len(users_to_process)}, 本次已完成={len(completed_user_ids)}")
    log(f"断点续传: 全局已完成={len(completed_user_ids)}, 全局剩余未完成={original_total_users - len(completed_user_ids)}")
    
    # 初始化
    log("Initializing LLM client...")
    # 项目已下线 MiniMax 远程后端，统一使用本地 Qwen-7B/8B 推理；
    # use_minimaxio 配置项保留但不再生效，仅记录一条说明。
    llm_client = QwenLocalClient()
    log("使用本地 Qwen 推理客户端")
    
    # 预热缓存：先调用一次建立 ephemeral cache
    log("预热 LLM 缓存...")
    llm_client.call_with_cache(
        system_base=base_system,
        user_content="[Review 0] warmup",
        max_tokens=64,
        stream=True
    )
    log("缓存预热完成")
    
    extractor = BatchErrorExtractor(
        llm_client,
        base_system=base_system,
        user_template=user_template,
        max_tokens=max_tokens,
        max_retries=max_retries
    )
    
    total_errors = 0
    users_written = 0
    total_filtered = 0
    simple_counts_total: Counter = Counter()
    
    def process_user(user: Dict) -> Dict:
        user_id = user.get("user_id", "")
        results = user.get("results", [])
        log(f"[PROCESS] 开始处理 user_id={user_id}")
        # 收集所有评论
        all_reviews = []
        for result in results:
            reviews = result.get("target_reviews", [])
            if max_reviews_per_user:
                reviews = reviews[:max_reviews_per_user]
            all_reviews.extend(reviews)
        # 限制总评论数
        if max_reviews_per_user:
            all_reviews = all_reviews[:max_reviews_per_user]
        

        print("[DEBUG] user_id=" + user_id + " results=" + str(len(results)) + " reviews_collected=" + str(len(all_reviews)), flush=True)
        if not all_reviews:
            return {
                "user_id": user_id,
                "errors": [],
                "simple_counts": Counter(),
            }
        
        # 批量提取
        batch_result = extractor.extract_batch_errors(all_reviews)
        
        user_errors = []
        user_simple_counts = Counter()
        
        if batch_result.get("status") == "success":
            result_data = batch_result.get("result", {})
            # 支持两种格式：1) {"errors": [...]} 或 2) {"0": {"errors": [...]}, "1": {"errors": [...]}, ...}
            all_errors = []
            if isinstance(result_data, dict):
                if "errors" in result_data:
                    # 格式1: {"errors": [...]}
                    all_errors = result_data["errors"] if isinstance(result_data["errors"], list) else []
                else:
                    # 格式2: {"0": {"errors": [...]}, "1": {"errors": [...]}, ...}
                    # 格式3: {"0": [{...error...}, ...], "1": [{...}, ...], ...}
                    for key, value in result_data.items():
                        if isinstance(value, dict) and "errors" in value:
                            errors_list = value["errors"]
                            if isinstance(errors_list, list):
                                all_errors.extend(errors_list)
                        elif isinstance(value, list):
                            # 格式3: 值直接是错误列表
                            all_errors.extend(value)
            

            # 去重：根据 original + corrected 组合去重
            seen_errors = set()
            for err in all_errors:
                original = err.get("original", "")
                corrected = err.get("corrected", "")
                if original and corrected and original != corrected:
                    # 去重
                    error_key = (original.lower(), corrected.lower())
                    if error_key in seen_errors:
                        continue
                    seen_errors.add(error_key)
                    
                    is_simple, reason = is_simple_error(original, corrected)
                    if is_simple:
                        user_simple_counts[reason] += 1
                    else:
                        user_errors.append({
                            "original": original,
                            "corrected": corrected,
                            "span_text": err.get("span_text", ""),
                            "confidence": err.get("confidence", 1.0),
                        })
        
        return {
            "user_id": user_id,
            "errors": user_errors,
            "simple_counts": user_simple_counts,
        }
    
    # 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_user, user): user for user in users_to_process}
        
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            user_id = result["user_id"]
            user_errors = result["errors"]
            user_simple_counts = result["simple_counts"]
            
            total_errors += len(user_errors) + sum(user_simple_counts.values())
            total_filtered += sum(user_simple_counts.values())
            simple_counts_total.update(user_simple_counts)
            
            log(f"[{idx}/{len(users_to_process)}] user_id={user_id}, errors={len(user_errors)}, filtered={sum(user_simple_counts.values())}")
            
            # 只有有有效错误时才记录
            if user_errors:
                user_record = {
                    "user_id": user_id,
                    "category": category,
                    "status": "success",
                    "total_errors": len(user_errors) + sum(user_simple_counts.values()),
                    "filtered_errors": len(user_errors),
                    "simple_error_counts": dict(user_simple_counts),
                    "error_details": user_errors,
                }
                
                # 立即写入 JSON（加锁防止并发冲突）
                lock_file = Path(str(output_file) + LOCK_SUFFIX)
                with FileLock(lock_file, timeout=30):
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    if output_file.exists() and output_file.stat().st_size > 0:
                        with open(output_file, "r", encoding="utf-8") as f:
                            all_rows = json.load(f)
                    else:
                        all_rows = []
                    # 去重：移除旧记录（如果存在）
                    all_rows = [r for r in all_rows if r["user_id"] != user_id]
                    all_rows.append(user_record)
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(all_rows, f, ensure_ascii=False, indent=2)
                    
                    users_written += 1
            
            if idx % PROGRESS_INTERVAL_USERS == 0 or idx == len(users_to_process):
                log(
                    f"[{category}] Processed {idx}/{len(users_to_process)} users; "
                    f"total_errors={total_errors}; filtered={total_filtered}; "
                    f"elapsed={format_elapsed(start_time)}"
                )
    log(f"=== Finished: {category}; elapsed={format_elapsed(start_time)} ===")
    log(f"Users written: {users_written}/{len(users_to_process)}")
    log(f"Total: {len(users_to_process)} users, {total_errors} errors, {total_filtered} filtered")
    log(f"Simple error breakdown: {dict(simple_counts_total)}")
