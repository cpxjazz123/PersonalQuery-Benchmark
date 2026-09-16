"""PCFG dependency-rule → sentence embedding (encoder only).

本脚本只负责编码: parse sentences → 建 sparse rule-count matrix → 训监督 encoder,
产出 32d supervised z 缓存 (full pool + no-leakage 50/50 split)。

Gaussian fitting / attribution / ablation 一律外迁:
  - 04_gaussian/run_adaptive_encoder.py    3-way profile/val/test split + 自适应 encoder
  - 04_gaussian/fit_per_user_gaussian.py   per-user Gaussian + ASIN cohort gates
                                            (canonical production Gaussian)
  - analysis/attribution_ablation.py       7 种 attribution + 高/低方差分层 lift
  - analysis/variance_factor_*.py          σ²_u 多因子回归
  - analysis/covariance_reproducibility.py Σ_u profile/test 跨半稳定性

一体化 pipeline (per 项目 Rule 18, 03_spacy_encode 只允许一个主脚本)。
主入口 main_pipeline() 一键串行运行所有 stage, 不再依赖 MODE 切换;
individual stage_*() 函数保留用于单 stage smoke / debug。

Canonical pipeline 链路 (仅编码, 2 步):
  cache        parse sentences + 建 sparse rule-count matrix (per-user + per-sent)
  strict       训监督 encoder V→256→32 (rule dropout 0.1, L2 1e-2, profile-only) +
                 supervised_embeddings.npy (full pool z, 给 04_gaussian 消费) +
                 strict_embeddings.npz (no-leakage profile/test 50/50 split, 给 analysis 消费)

输入:
  result/02_user_review_sentence_extract/uid_to_sentences.pkl

中间缓存:
  /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/
    counts.npz, sent_vectors.npz, uid_list.json, user_n_sents.json, vocab.json, meta.json
    supervised_embeddings.npy (strict)
    strict_encoder.pt, strict_embeddings.npz (strict)

输出 (result/03_spacy_encode/):
  syntax_pcfg_strict_attr.json        (strict)

历史 lift 记录 (100 users, LOPO):
  pcfg_attr (discrete 7858d log-P):          7.97× chance
  super (z=32, rule dropout 0.3, L2 1e-2):   7.09× chance  ← 连续 z 的 SOTA
  super (z=64):                              6.83×
  super (z=32, weak reg):                    5.88×
  vae  (z=32, β=1e-3, collapse):             1.70×
  ae   (z=32, β=0):                          1.20×
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz, save_npz

# ============================================================================
# 常量 (硬编码, per Rule 3)
# ============================================================================

# 路径
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract/uid_to_sentences.pkl"
OUT_DIR = REPO_ROOT / "result/03_spacy_encode"
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")

# 解析
SPACY_MODEL = "en_core_web_sm"     # CPU; 4-process parallel via nlp.pipe n_process=4
PARSE_BATCH_SIZE = 512       # benchmarked peak for n_process=4 (larger batches slower)
PARSE_N_PROCESS = 4          # spaCy nlp.pipe n_process (parallel CPU workers)
CHUNK_USERS = 2000

# 通用
SEED = 42
HASH_SALT = "pcfg_lopo_v1"

# Internal: effective training limit set by smoke mode or N_USERS_LIMIT
# None = train on all cache users; set by main_pipeline() before stage_strict3()
_EFFECTIVE_TRAIN_LIMIT = None

# Stage: cache
N_USERS_LIMIT = None        # None = 全部 eligible 用户 (raw 3.39M → h≥30 = 64,996)
MIN_SENTS_PER_USER = 30
MIN_DOC_FREQ = 2              # vocab filter: rule must appear in ≥ N docs

# Smoke mode: 200 users, 5 epochs — 验证 16d encoder 训练能收敛
# smoke=True 时 N_USERS_LIMIT=200, SUP_EPOCHS=5, STRICT_PATIENCE=3
SMOKE = False                # 2026-09-15: 80/20 full run (smoke 验证已通过)
N_USERS_LIMIT_SMOKE = 200
SMOKE_EPOCHS = 5
SMOKE_PATIENCE = 3

# Stage: strict (SUP_* 共享: supervised encoder V→256→z_dim)
SUP_EPOCHS = 80
SUP_BATCH_SIZE = 2048
SUP_LR = 5e-4
SUP_Z_DIM = 16                # 2026-09-14: 32→16 per Phase L8.7b (16d wins all generalization metrics)
SUP_HIDDEN = (256,)
SUP_DROPOUT = 0.5
SUP_RULE_DROPOUT = 0.1     # 关键: 训练时随机 mask rule columns (profile-only 减半, 0.3 太狠 → 不收敛)
SUP_WEIGHT_DECAY = 1e-2
SUP_LABEL_SMOOTHING = 0.1
SUP_LOG_EVERY = 5
SUP_TRAIN_FRAC = 0.8       # per-user train/val split (hash-based)

# Stage: strict (no-leakage supervised encoder + 早停)
STRICT_VAL_FRAC = 0.15      # LEGACY: 仅 stage_strict() 从 profile 内切 15% 早停 val
STRICT_SEED = 42
STRICT_PATIENCE = 10        # val_acc plateau N epochs → early stop
# (2026-09-15: stage_strict3 改为 2-way 80/20, val(20%) 直接作为早停 + held-out,
#  不再从 profile 内切 15%; z_test/test_idx 字段保留为 val 别名以维持下游契约)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


COHORT_FINGERPRINT_VERSION = "sha256-cohort-v1"


def _hash_length_prefixed_text(hasher: "hashlib._Hash", text: str) -> None:
    """把字符串以长度前缀写入 hash，避免边界歧义。"""
    if not isinstance(text, str):
        raise TypeError(f"cohort fingerprint requires str, got {type(text)!r}")
    raw = text.encode("utf-8")
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _cohort_fingerprint(uid_list: list[str], uid_to_sents: dict) -> str:
    """对 UID 顺序、每用户句数及清洗后句子内容做稳定指纹。"""
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, COHORT_FINGERPRINT_VERSION)
    for uid in uid_list:
        if uid not in uid_to_sents:
            raise KeyError(f"selected uid missing from sentence cache: {uid}")
        sents = uid_to_sents[uid]
        if not isinstance(sents, list):
            raise TypeError(f"sentences for uid {uid} must be list")
        _hash_length_prefixed_text(hasher, uid)
        hasher.update(len(sents).to_bytes(8, "big"))
        for sentence in sents:
            _hash_length_prefixed_text(hasher, sentence)
    return hasher.hexdigest()


def _uid_layout_fingerprint(uid_list: list[str], user_n_sents: list[int]) -> str:
    """对 UID 顺序及句子布局做轻量指纹，供下游 contract 校验。"""
    if len(uid_list) != len(user_n_sents):
        raise ValueError("uid_list and user_n_sents length mismatch")
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, "uid-layout-v1")
    for uid, n_sents in zip(uid_list, user_n_sents):
        _hash_length_prefixed_text(hasher, uid)
        if not isinstance(n_sents, int) or n_sents < 0:
            raise ValueError(f"invalid sentence count for uid {uid}: {n_sents!r}")
        hasher.update(n_sents.to_bytes(8, "big"))
    return hasher.hexdigest()


def _array_fingerprint(array: np.ndarray) -> str:
    """对数组 dtype、shape 和字节内容做指纹，防止同 shape stale embedding。"""
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, str(contiguous.dtype))
    _hash_length_prefixed_text(hasher, repr(tuple(contiguous.shape)))
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _file_fingerprint(path: Path) -> str:
    """对大型 artifact 分块计算 SHA256，确认多个文件来自同一次提交。"""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _sparse_fingerprint(matrix) -> str:
    """对 sparse matrix 的结构和值做指纹，不把其展开成 dense。"""
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, str(matrix.dtype))
    _hash_length_prefixed_text(hasher, repr(tuple(matrix.shape)))
    for part in (matrix.indptr, matrix.indices, matrix.data):
        contiguous = np.ascontiguousarray(part)
        _hash_length_prefixed_text(hasher, str(contiguous.dtype))
        _hash_length_prefixed_text(hasher, repr(tuple(contiguous.shape)))
        hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _string_list_fingerprint(values: list[str], tag: str) -> str:
    """对有序字符串列表做稳定指纹。"""
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, tag)
    for value in values:
        _hash_length_prefixed_text(hasher, value)
    return hasher.hexdigest()


def _atomic_json_dump(payload: dict, path: Path) -> None:
    """以临时文件+rename 写 JSON，避免留下半个 manifest。"""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_pickle_dump(payload: dict, path: Path) -> None:
    """以临时文件+rename 写分块规则。"""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, matrix) -> None:
    """原子保存 sparse NPZ。"""
    tmp = Path(str(path) + ".tmp.npz")
    save_npz(tmp, matrix)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """原子保存 dense NPY。"""
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _atomic_save_npz_arrays(path: Path, **arrays) -> None:
    """原子保存包含多个数组的 NPZ。"""
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_stage02_cohort(preloaded: dict | None = None) -> tuple[list[str], list[int], dict]:
    """加载 Stage 02 并重建确定性的当前 cohort manifest。"""
    if preloaded is None:
        with open(SENT_CACHE, "rb") as f:
            uid_to_sents = pickle.load(f)
    else:
        uid_to_sents = preloaded
    if not isinstance(uid_to_sents, dict):
        raise TypeError(f"Stage 02 cache must be dict, got {type(uid_to_sents)!r}")
    counts = {}
    for uid, sents in uid_to_sents.items():
        if not isinstance(uid, str):
            raise TypeError(f"Stage 02 UID must be str, got {type(uid)!r}")
        if not isinstance(sents, list):
            raise TypeError(f"Stage 02 sentences for {uid} must be list")
        if any(not isinstance(sentence, str) for sentence in sents):
            raise TypeError(f"Stage 02 sentences for {uid} must all be str")
        counts[uid] = len(sents)
    selected = sorted([uid for uid, count in counts.items()
                       if count >= MIN_SENTS_PER_USER])
    rng = np.random.default_rng(SEED)
    rng.shuffle(selected)
    if N_USERS_LIMIT is not None:
        selected = selected[:N_USERS_LIMIT]
    user_n_sents = [counts[uid] for uid in selected]
    n_total = int(sum(user_n_sents))
    manifest = {
        "cache_schema_version": 2,
        "n_users": len(selected),
        "n_total_sents": n_total,
        "min_sents_per_user": MIN_SENTS_PER_USER,
        "n_users_limit": N_USERS_LIMIT,
        "seed": SEED,
        "uid_layout_fingerprint": _uid_layout_fingerprint(
            selected, user_n_sents),
        "sentence_source_fingerprint": _cohort_fingerprint(
            selected, uid_to_sents),
    }
    return selected, user_n_sents, manifest


def _validate_rules_cache(saved: dict, expected: dict,
                          expected_n_users: int,
                          expected_n_sents: int) -> tuple[list, int]:
    """严格验证可恢复的 PCFG partial/full cache。"""
    if not isinstance(saved, dict):
        raise ValueError("PCFG rules cache payload must be an object")
    for key, expected_value in expected.items():
        if saved.get(key) != expected_value:
            raise ValueError(
                f"PCFG rules cache cohort mismatch at {key}: "
                f"cached={saved.get(key)!r}, current={expected_value!r}; "
                f"delete the stale rules cache and rerun")
    cached_rules = saved.get("rules")
    cached_offset = saved.get("offset")
    cached_next_user = saved.get("next_user_index")
    if not isinstance(cached_rules, list):
        raise ValueError("PCFG rules cache field 'rules' must be a list")
    if not isinstance(cached_offset, int) or not isinstance(cached_next_user, int):
        raise ValueError("PCFG rules cache offset metadata must be integers")
    if cached_offset != len(cached_rules):
        raise ValueError(
            f"PCFG rules cache offset mismatch: offset={cached_offset}, "
            f"rules={len(cached_rules)}; delete cache and rerun")
    if not (0 <= cached_next_user <= expected_n_users):
        raise ValueError(
            f"PCFG rules cache next_user_index out of range: "
            f"{cached_next_user} not in [0,{expected_n_users}]")
    if cached_offset < 0 or cached_offset > expected_n_sents:
        raise ValueError(
            f"PCFG rules cache offset out of range: {cached_offset} "
            f"not in [0,{expected_n_sents}]")
    return cached_rules, cached_next_user


def _verify_cached_embeddings_cohort(emb_path: Path,
                                      expected_n_sents: int,
                                      z_dim: int,
                                      stage_name: str) -> None:
    """验证 cached embeddings 是否匹配当前 cache cohort。"""
    if not emb_path.exists():
        return  # 已在 caller 检查过 exists, 这里只验 shape
    arr = np.load(emb_path)
    if arr.ndim != 2 or arr.shape != (expected_n_sents, z_dim):
        raise ValueError(
            f"{stage_name} cache cohort mismatch: cached {emb_path.name} "
            f"shape={arr.shape} vs current cache expected "
            f"({expected_n_sents}, {z_dim}); delete {emb_path} and rerun")
    if not np.isfinite(arr).all():
        raise ValueError(f"{stage_name} cache contains NaN/Inf: {emb_path}")


def _npz_text(npz, key: str) -> str:
    """读取 NPZ 中的标量文本字段并拒绝缺失/非标量值。"""
    if key not in npz:
        raise ValueError(f"strict3 artifact missing metadata field: {key}")
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"strict3 metadata field {key} must be scalar")
    return str(value.item())


def _validate_strict3_artifact(npz, cache: dict) -> dict:
    """严格验证 strict3 的 UID、分区、shape、finite 和来源 manifest (2-way 80/20)。

    2026-09-15: strict3 改为 2-way 80/20, z_test/test_idx 是 z_val/val_idx 的别名,
    partition 验证只看 profile+val, test 字段要求存在但形状需与 val 对齐。
    """
    uid_list = cache["uid_list"]
    user_n_sents = cache["user_n_sents"]
    meta = cache["meta"]
    n_total = int(sum(user_n_sents))
    if "uid_list" not in npz:
        raise ValueError("strict3 artifact missing uid_list")
    artifact_uids = [str(uid) for uid in np.asarray(npz["uid_list"]).tolist()]
    if artifact_uids != uid_list:
        raise ValueError("strict3 artifact UID order/content differs from cache")
    for key in ("z_profile", "z_val"):
        if key not in npz:
            raise ValueError(f"strict3 artifact missing {key}")
        z = np.asarray(npz[key])
        if z.ndim != 2 or z.shape[1] != SUP_Z_DIM:
            raise ValueError(f"strict3 {key} shape invalid: {z.shape}")
        if not np.isfinite(z).all():
            raise ValueError(f"strict3 {key} contains NaN/Inf")
    # z_test 是 z_val 别名, 校验形状对齐
    if "z_test" not in npz or "test_idx" not in npz:
        raise ValueError("strict3 artifact missing z_test/test_idx alias fields")
    z_test = np.asarray(npz["z_test"])
    if (z_test.ndim != 2 or z_test.shape[1] != SUP_Z_DIM
            or z_test.shape[0] != np.asarray(npz["z_val"]).shape[0]):
        raise ValueError(f"strict3 z_test must equal z_val shape, "
                         f"got z_test {z_test.shape} vs z_val "
                         f"{np.asarray(npz['z_val']).shape}")
    if not np.array_equal(np.asarray(npz["test_idx"], dtype=np.int64),
                          np.asarray(npz["val_idx"], dtype=np.int64)):
        raise ValueError("strict3 test_idx must equal val_idx (alias)")
    if not np.isfinite(z_test).all():
        raise ValueError("strict3 z_test contains NaN/Inf")
    _validate_split_indices(npz, uid_list, user_n_sents)
    for z_key, idx_key in (("z_profile", "profile_idx"),
                           ("z_val", "val_idx")):
        if np.asarray(npz[z_key]).shape[0] != np.asarray(npz[idx_key]).size:
            raise ValueError(f"strict3 {z_key}/{idx_key} row count mismatch")
    if _npz_text(npz, "cohort_fingerprint") != meta["sentence_source_fingerprint"]:
        raise ValueError("strict3 cohort fingerprint mismatch")
    if _npz_text(npz, "uid_layout_fingerprint") != meta["uid_layout_fingerprint"]:
        raise ValueError("strict3 UID layout fingerprint mismatch")
    if _npz_text(npz, "vocab_fingerprint") != meta["vocab_fingerprint"]:
        raise ValueError("strict3 vocabulary fingerprint mismatch")
    expected_train, expected_val = _two_way_split(uid_list, SENT_CACHE)
    for key, expected_dict in (("profile_idx", expected_train),
                               ("val_idx", expected_val)):
        expected_idx = np.asarray(sorted(i for values in expected_dict.values()
                                         for i in values), dtype=np.int64)
        actual_idx = np.asarray(npz[key], dtype=np.int64)
        if not np.array_equal(actual_idx, expected_idx):
            raise ValueError(f"strict3 {key} differs from current sentence split")
    return {
        "n_total_sents": n_total,
        "n_users": len(uid_list),
        "cohort_fingerprint": meta["sentence_source_fingerprint"],
        "uid_layout_fingerprint": meta["uid_layout_fingerprint"],
        "vocab_fingerprint": meta["vocab_fingerprint"],
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """原子保存 PyTorch checkpoint。"""
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ============================================================================
# 共享工具
# ============================================================================

def extract_struct_rules(doc):
    """从 spaCy Doc 提取 dependency + POS rules (无词项)。

    D4|祖父POS|父POS|dep|子POS
    D3|父POS|dep|子POS
    P3|POS[i]|POS[i+1]|POS[i+2]

    优化: precompute pos/head/dep lists once per doc, avoid repeated
    property access on Cython Token (~3x speedup).
    """
    n = len(doc)
    if n < 3:
        return []
    pos = [t.pos_ for t in doc]
    heads_abs = [t.head.i for t in doc]
    deps = [t.dep_ for t in doc]
    rs = set()
    for i in range(n):
        h = heads_abs[i]
        if h == i:
            continue
        gh = heads_abs[h]
        gp_pos = pos[gh] if gh != h else "ROOT"
        rs.add(f"D4|{gp_pos}|{pos[h]}|{deps[i]}|{pos[i]}")
        rs.add(f"D3|{pos[h]}|{deps[i]}|{pos[i]}")
    for i in range(n - 2):
        rs.add(f"P3|{pos[i]}|{pos[i+1]}|{pos[i+2]}")
    return list(rs)


def hash_bucket(text: str, mod: int = 2, salt: str = None) -> int:
    """SHA1-based deterministic bucket, 同文本永远同 bucket。

    mod=2: 50/50 split; mod=10: 80/20 split (b<8 train)。
    salt: 默认用 HASH_SALT; 传 salt 改 hash (multi_seed 用, 让 split 跨 seed 不同)。
    """
    s = HASH_SALT if salt is None else salt
    h = int(hashlib.sha1(
        (s + "|" + text.strip().lower()).encode()
    ).hexdigest(), 16)
    return h % mod


def normalize_counts(sent_csr):
    """x_norm = count / (1 + count) ∈ [0, 1) for BCE decoder。"""
    data = sent_csr.data.astype(np.float32)
    norm = data / (1.0 + data)
    return csr_matrix(
        (norm, sent_csr.indices, sent_csr.indptr),
        shape=sent_csr.shape,
    )


def _validate_cache_manifest(meta: dict, expected_manifest: dict) -> None:
    """校验 sparse cache 是否来自当前 Stage 02 cohort。"""
    if not isinstance(meta, dict):
        raise ValueError("pcfg cache meta must be an object")
    required = {
        **expected_manifest,
        "cache_schema_version": 2,
        "spacy_model": SPACY_MODEL,
    }
    for key, expected_value in required.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"pcfg cache manifest mismatch at {key}: "
                f"cached={meta.get(key)!r}, current={expected_value!r}; "
                f"rerun stage_cache()")
    if meta.get("vocab_size", 0) <= 0:
        raise ValueError("pcfg cache vocabulary is empty")


def _validate_split_indices(npz: dict, uid_list: list[str],
                            user_n_sents: list[int]) -> None:
    """验证 strict3 索引非负、无重复且 profile+val 完整覆盖句子。
    2026-09-15: strict3 改为 2-way 80/20, test_idx 是 val_idx 别名,
    partition 验证只看 profile+val (互斥且穷尽), test 字段不参与。
    """
    n_total = int(sum(user_n_sents))
    arrays = []
    for key in ("profile_idx", "val_idx"):
        if key not in npz:
            raise ValueError(f"strict3 artifact missing {key}")
        index = np.asarray(npz[key], dtype=np.int64)
        if index.ndim != 1:
            raise ValueError(f"strict3 {key} must be 1-D, got {index.shape}")
        if len(index) and (int(index.min()) < 0 or int(index.max()) >= n_total):
            raise ValueError(f"strict3 {key} contains out-of-range index")
        if len(np.unique(index)) != len(index):
            raise ValueError(f"strict3 {key} contains duplicate indices")
        arrays.append(index)
    merged = np.concatenate(arrays)
    if len(np.unique(merged)) != n_total or not np.array_equal(
            np.sort(merged), np.arange(n_total, dtype=np.int64)):
        raise ValueError("strict3 split indices do not partition all sentences")
    if len(uid_list) != len(user_n_sents):
        raise ValueError("strict3 uid_list/user_n_sents length mismatch")


def _validate_cache_arrays(cache: dict, expected_manifest: dict) -> None:
    """验证 sparse matrix、UID 顺序、句子布局及 finite 值。"""
    sent_csr = cache["sent_csr"]
    counts = cache["counts"]
    uid_list = cache["uid_list"]
    user_n_sents = cache["user_n_sents"]
    meta = cache["meta"]
    _validate_cache_manifest(meta, expected_manifest)
    if not isinstance(uid_list, list) or uid_list != cache["expected_uid_list"]:
        raise ValueError("pcfg cache UID order/content differs from Stage 02")
    if not isinstance(user_n_sents, list) or user_n_sents != cache["expected_user_n_sents"]:
        raise ValueError("pcfg cache user_n_sents differs from Stage 02")
    expected_shape = (expected_manifest["n_total_sents"], meta["vocab_size"])
    if sent_csr.shape != expected_shape:
        raise ValueError(f"sent_vectors shape mismatch: {sent_csr.shape} != {expected_shape}")
    if counts.shape != (expected_manifest["n_users"], meta["vocab_size"]):
        raise ValueError(f"counts shape mismatch: {counts.shape}")
    if sent_csr.ndim != 2 or counts.ndim != 2:
        raise ValueError("pcfg sparse matrices must be 2-D")
    if not np.isfinite(sent_csr.data).all() or not np.isfinite(counts.data).all():
        raise ValueError("pcfg sparse cache contains NaN/Inf")
    if _string_list_fingerprint(cache["vocab"], "vocab-v1") != meta["vocab_fingerprint"]:
        raise ValueError("pcfg vocabulary fingerprint mismatch")
    if _sparse_fingerprint(sent_csr) != meta["sent_vectors_fingerprint"]:
        raise ValueError("sent_vectors fingerprint mismatch")
    if _sparse_fingerprint(counts) != meta["counts_fingerprint"]:
        raise ValueError("counts fingerprint mismatch")


def load_cache():
    required_paths = [
        CACHE_DIR / "sent_vectors.npz", CACHE_DIR / "counts.npz",
        CACHE_DIR / "uid_list.json", CACHE_DIR / "user_n_sents.json",
        CACHE_DIR / "vocab.json", CACHE_DIR / "meta.json",
        CACHE_DIR / "cache_ready.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"pcfg cache incomplete; missing {missing}; run stage_cache()")
    expected_uid_list, expected_user_n_sents, expected_manifest = (
        _load_stage02_cohort())
    sent_csr = load_npz(CACHE_DIR / "sent_vectors.npz")
    counts = load_npz(CACHE_DIR / "counts.npz")
    with open(CACHE_DIR / "uid_list.json") as f:
        uid_list = json.load(f)
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    with open(CACHE_DIR / "vocab.json") as f:
        vocab = json.load(f)
    with open(CACHE_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(CACHE_DIR / "cache_ready.json") as f:
        ready = json.load(f)
    if ready != meta:
        raise ValueError("pcfg cache_ready manifest differs from meta.json")
    cache = {
        "sent_csr": sent_csr, "counts": counts,
        "uid_list": uid_list, "user_n_sents": user_n_sents,
        "vocab": vocab, "meta": meta,
        "expected_uid_list": expected_uid_list,
        "expected_user_n_sents": expected_user_n_sents,
    }
    _validate_cache_arrays(cache, expected_manifest)
    return cache


# ============================================================================
# Stage: cache (parse + sparse matrix persist)
# ============================================================================

def _load_rules_chunk(path: Path, expected_manifest: dict,
                      start_user: int, end_user: int,
                      start_sent: int, end_sent: int) -> list[list[str]]:
    """读取并严格校验一个不可变的解析分片。"""
    with open(path, "rb") as f:
        saved = pickle.load(f)
    if not isinstance(saved, dict):
        raise ValueError(f"rules chunk is not an object: {path}")
    for key, expected_value in expected_manifest.items():
        if saved.get(key) != expected_value:
            raise ValueError(
                f"rules chunk cohort mismatch at {key}: {path}; "
                f"cached={saved.get(key)!r}, current={expected_value!r}")
    expected_ranges = {
        "start_user": start_user, "end_user": end_user,
        "start_sent": start_sent, "end_sent": end_sent,
    }
    for key, expected_value in expected_ranges.items():
        if saved.get(key) != expected_value:
            raise ValueError(
                f"rules chunk range mismatch at {key}: {path}; "
                f"cached={saved.get(key)!r}, current={expected_value!r}")
    rules = saved.get("rules")
    if not isinstance(rules, list) or len(rules) != end_sent - start_sent:
        raise ValueError(
            f"rules chunk length mismatch: {path}; "
            f"got={len(rules) if isinstance(rules, list) else type(rules)!r}, "
            f"expected={end_sent - start_sent}")
    if any(not isinstance(rule_list, list) for rule_list in rules):
        raise ValueError(f"rules chunk contains a non-list rule row: {path}")
    return rules


def _iter_rules_chunks(rules_dir: Path, selected: list[str],
                       user_offsets: np.ndarray,
                       manifest: dict):
    """按用户顺序读取所有已校验的规则分片。"""
    for start_user in range(0, len(selected), CHUNK_USERS):
        end_user = min(start_user + CHUNK_USERS, len(selected))
        start_sent = int(user_offsets[start_user])
        end_sent = int(user_offsets[end_user])
        path = rules_dir / f"chunk_{start_user}_{end_user}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing completed rules chunk: {path}")
        rules = _load_rules_chunk(
            path, manifest, start_user, end_user, start_sent, end_sent)
        yield start_user, end_user, start_sent, end_sent, rules


def stage_cache():
    """en_core_web_sm parse sentences → sparse rule-count matrix。"""
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # torch.set_default_tensor_type deprecated in PyTorch 2.1+ and conflicts
    # with spaCy CPU/GPU routing; let downstream code manage device explicitly.
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    log(f"loaded {SPACY_MODEL}")

    with open(SENT_CACHE, "rb") as f:
        uid_to_sents = pickle.load(f)
    selected, user_n_sents, cohort_manifest = _load_stage02_cohort(
        preloaded=uid_to_sents)
    user_offsets = np.concatenate(([0], np.cumsum(user_n_sents, dtype=np.int64)))
    n_total = int(user_offsets[-1])
    log(f"users={len(selected)}, min_sents={MIN_SENTS_PER_USER}")
    log(f"total sents: {n_total}")

    # 每个用户块单独原子落盘，避免把已解析规则反复整体 pickle 到内存和磁盘。
    rules_dir = CACHE_DIR / "rules_chunks"
    rules_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rules_dir / "manifest.json"
    chunk_paths = sorted(rules_dir.glob("chunk_*.pkl"))
    if manifest_path.exists():
        with open(manifest_path) as f:
            saved_manifest = json.load(f)
        if saved_manifest != cohort_manifest:
            raise ValueError(
                f"rules chunk manifest mismatch: cached={saved_manifest!r}, "
                f"current={cohort_manifest!r}; delete {rules_dir} and rerun")
    elif chunk_paths:
        raise ValueError(
            f"rules chunks exist without manifest at {rules_dir}; "
            f"delete the incomplete cache before rerunning")
    else:
        _atomic_json_dump(cohort_manifest, manifest_path)

    expected_names = {
        f"chunk_{start}_{min(start + CHUNK_USERS, len(selected))}.pkl"
        for start in range(0, len(selected), CHUNK_USERS)
    }
    unexpected = [p.name for p in chunk_paths if p.name not in expected_names]
    if unexpected:
        raise ValueError(f"unexpected rules chunk files: {unexpected[:5]}")

    t_parse = time.time()
    for start_user in range(0, len(selected), CHUNK_USERS):
        end_user = min(start_user + CHUNK_USERS, len(selected))
        start_sent = int(user_offsets[start_user])
        end_sent = int(user_offsets[end_user])
        path = rules_dir / f"chunk_{start_user}_{end_user}.pkl"
        if path.exists():
            _load_rules_chunk(path, cohort_manifest, start_user, end_user,
                               start_sent, end_sent)
            log(f"  chunk {start_user}-{end_user}: cache hit "
                f"({end_sent - start_sent} sents)")
            continue
        chunk_sents = []
        for uid in selected[start_user:end_user]:
            chunk_sents.extend(uid_to_sents[uid])
        if len(chunk_sents) != end_sent - start_sent:
            raise ValueError(
                f"chunk sentence layout mismatch: {start_user}-{end_user}; "
                f"got={len(chunk_sents)}, expected={end_sent - start_sent}")
        log(f"  chunk {start_user}-{end_user}: {len(chunk_sents)} sents ...")
        rules_flat = [extract_struct_rules(d)
                      for d in nlp.pipe(chunk_sents,
                                        batch_size=PARSE_BATCH_SIZE,
                                        n_process=PARSE_N_PROCESS)]
        if len(rules_flat) != end_sent - start_sent:
            raise RuntimeError(
                f"spaCy returned {len(rules_flat)} rows for "
                f"{end_sent - start_sent} sentences")
        payload = dict(cohort_manifest)
        payload.update({"start_user": start_user, "end_user": end_user,
                        "start_sent": start_sent, "end_sent": end_sent,
                        "rules": rules_flat})
        _atomic_pickle_dump(payload, path)
        del chunk_sents, rules_flat
        log(f"    {time.time()-t_parse:.0f}s elapsed, "
            f"{end_sent}/{n_total}")
    log(f"parse done ({time.time()-t_parse:.0f}s)")

    log("building vocab (with min_doc_freq filter) ...")
    doc_counts: Counter = Counter()
    for _, _, _, _, rules in _iter_rules_chunks(
            rules_dir, selected, user_offsets, cohort_manifest):
        for rs in rules:
            doc_counts.update(set(rs))
    vocab_full = sorted(doc_counts.keys())
    vocab = [r for r in vocab_full if doc_counts[r] >= MIN_DOC_FREQ]
    V = len(vocab)
    if V <= 0:
        raise RuntimeError("vocabulary is empty after min_doc_freq filtering")
    rule_to_id = {r: i for i, r in enumerate(vocab)}
    log(f"vocab: {V} (filtered from {len(vocab_full)}, "
        f"min_doc_freq={MIN_DOC_FREQ})")

    log("building per-sentence CSR and per-user counters ...")
    row_nnz = np.zeros(n_total, dtype=np.int64)
    user_counters: list[Counter] = [Counter() for _ in selected]
    for start_user, end_user, start_sent, _, rules in _iter_rules_chunks(
            rules_dir, selected, user_offsets, cohort_manifest):
        for local_i, rs in enumerate(rules):
            filtered = [(r, cnt) for r, cnt in Counter(rs).items()
                        if r in rule_to_id]
            row_nnz[start_sent + local_i] = len(filtered)
            global_i = start_sent + local_i
            owner = int(np.searchsorted(user_offsets[1:], global_i,
                                        side="right"))
            user_counters[owner].update(dict(filtered))
    indptr = np.concatenate(([0], np.cumsum(row_nnz, dtype=np.int64)))
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=np.int16)
    for _, _, start_sent, _, rules in _iter_rules_chunks(
            rules_dir, selected, user_offsets, cohort_manifest):
        for local_i, rs in enumerate(rules):
            row = start_sent + local_i
            cursor = int(indptr[row])
            for rule, cnt in Counter(rs).items():
                rule_id = rule_to_id.get(rule)
                if rule_id is None:
                    continue
                if cnt > np.iinfo(np.int16).max:
                    raise ValueError(f"sentence rule count exceeds int16: {cnt}")
                indices[cursor] = rule_id
                data[cursor] = cnt
                cursor += 1
            if cursor != int(indptr[row + 1]):
                raise RuntimeError(f"CSR row fill mismatch at sentence {row}")
    sent_sparse = csr_matrix((data, indices, indptr),
                             shape=(n_total, V), dtype=np.int16)
    sent_sparse.sort_indices()
    log(f"per-sent CSR: {sent_sparse.shape}, density "
        f"{sent_sparse.nnz/(n_total*V)*100:.3f}%")

    rows, cols, user_data = [], [], []
    for ui, cnt in enumerate(user_counters):
        for rule, value in cnt.items():
            rule_id = rule_to_id.get(rule)
            if rule_id is None:
                continue
            rows.append(ui)
            cols.append(rule_id)
            user_data.append(value)
    counts_sparse = csr_matrix(
        (np.asarray(user_data, dtype=np.int32),
         (np.asarray(rows, dtype=np.int32),
          np.asarray(cols, dtype=np.int32))),
        shape=(len(selected), V), dtype=np.int32,
    )
    counts_sparse.sort_indices()
    log(f"per-user CSR: {counts_sparse.shape}, density "
        f"{counts_sparse.nnz/(len(selected)*V)*100:.3f}%")

    vocab_fingerprint = _string_list_fingerprint(vocab, "vocab-v1")
    meta = {
        **cohort_manifest,
        "mode": ("LIMITED" if N_USERS_LIMIT else "FULL"),
        "spacy_model": SPACY_MODEL,
        "vocab_size": V,
        "vocab_fingerprint": vocab_fingerprint,
        "user_csr_nnz": int(counts_sparse.nnz),
        "user_csr_density_pct": round(
            counts_sparse.nnz / (len(selected) * V) * 100, 4),
        "sent_csr_nnz": int(sent_sparse.nnz),
        "sent_csr_density_pct": round(
            sent_sparse.nnz / (n_total * V) * 100, 4),
        "sent_vectors_fingerprint": _sparse_fingerprint(sent_sparse),
        "counts_fingerprint": _sparse_fingerprint(counts_sparse),
        "rules_cache": str(rules_dir),
    }
    _atomic_save_npz(CACHE_DIR / "counts.npz", counts_sparse)
    _atomic_save_npz(CACHE_DIR / "sent_vectors.npz", sent_sparse)
    _atomic_json_dump(selected, CACHE_DIR / "uid_list.json")
    _atomic_json_dump(user_n_sents, CACHE_DIR / "user_n_sents.json")
    _atomic_json_dump(vocab, CACHE_DIR / "vocab.json")
    _atomic_json_dump(meta, CACHE_DIR / "meta.json")
    # ready manifest 最后写入；下游只有看到它才能消费本轮完整 cache。
    _atomic_json_dump(meta, CACHE_DIR / "cache_ready.json")
    log(f"wrote → {CACHE_DIR}/")
    log(f"=== Total: {time.time()-t0:.0f}s ===")



# ============================================================================
# Supervised encoder (V → 256 → 32) — used by stage_strict, 产出
# supervised_embeddings.npy 给 stage_gauss + stage_ablation 消费
# ============================================================================

class _SupEncoder(nn.Module):
    def __init__(self, vocab_size, z_dim, hidden, n_users, dropout):
        super().__init__()
        h1 = hidden[0]
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, z_dim),
        )
        self.classifier = nn.Linear(z_dim, n_users)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.classifier(z)



# ============================================================================
# Helper used by stage_strict: SHA1 50/50 profile/test split
# ============================================================================

def _profile_test_split(uid_list, cache, sent_cache_path):
    """SHA1 hash 50/50 split → profile_idx/test_idx per uid。

    与历史 stage_attr 用同一 salt, 保证 cosine / Gaussian / Mahalanobis 三种
    attribution 跑在同一 train/test 划分上 (公平对比, 即使 attribution 已外迁)。
    """
    with open(sent_cache_path, "rb") as f:
        uid_to_sents = pickle.load(f)
    profile_idx = {u: [] for u in uid_list}
    test_idx = {u: [] for u in uid_list}
    sent_off = 0
    for ui, uid in enumerate(uid_list):
        sents = list(uid_to_sents[uid])
        for si, s in enumerate(sents):
            if hash_bucket(s, mod=2) == 0:
                profile_idx[uid].append(sent_off + si)
            else:
                test_idx[uid].append(sent_off + si)
        sent_off += len(sents)
    return profile_idx, test_idx


def _two_way_split(uid_list, sent_cache_path,
                    train_cut=8, mod=10, salt: str = None):
    """SHA1 hash 2-way split: train (b<8) / val (b≥8)。

    默认 80/20 (train/val)。两个桶互斥且穷尽,encoder 只见过 train,
    val 完全未参与训练并作为早停 + 下游 held-out eval (per user 2026-09-15:
    不需要 test, val 直接作 held-out)。
    """
    with open(sent_cache_path, "rb") as f:
        uid_to_sents = pickle.load(f)
    train_dict = {u: [] for u in uid_list}
    val_dict = {u: [] for u in uid_list}
    sent_off = 0
    for ui, uid in enumerate(uid_list):
        sents = list(uid_to_sents[uid])
        for si, s in enumerate(sents):
            b = hash_bucket(s, mod=mod, salt=salt)
            if b < train_cut:
                train_dict[uid].append(sent_off + si)
            else:
                val_dict[uid].append(sent_off + si)
        sent_off += len(sents)
    return train_dict, val_dict


# ============================================================================
# Stage: strict (no-leakage: profile-only train, test held-out for attr)
# ============================================================================

def stage_strict(out_path: Path = None):
    """严格无泄漏 supervised encoder + Gaussian attribution。

    协议 (与 stage_gauss_attr 完全对齐, 唯一变化是 encoder 训练只看到 profile):
      1. SHA1 hash 50/50 split → profile_idx / test_idx (与 _profile_test_split 同盐)
      2. encoder 只用 profile_idx 句子 + user_id labels 训练
         - 内部 15% hash 再分 profile_train / profile_val 做早停
      3. freeze encoder, 分别编码 profile / test → z_profile, z_test
      4. fit N(μ_u, σ²_u_diag) on z_profile (per-user Gaussian)
      5. test z_q → argmax_u {cos(z_q,μ_u), log N(z_q|μ_u,σ²_u),
                              -mahal_pooled(z_q,μ_u)}

    输出:
      result/03_spacy_encode/syntax_pcfg_strict_attr.json
        { config: {n_profile, n_test, encoder_config},
          no_leakage: True,
          lifts: {cosine, log_p_diag, mahalanobis_pooled},
          per_method: {...},
          encoder_diagnostics: {best_val_acc, history[-1], ...} }

    关键: encoder 从未看过 test 句子 (及其 user_id label), 任何 test 上 attribution
    提升都是真实泛化能力。
    """
    t0 = time.time()
    if out_path is None:
        out_path = OUT_DIR / "syntax_pcfg_strict_attr.json"

    cached = [
        out_path,
        CACHE_DIR / "strict_encoder.pt",
        CACHE_DIR / "strict_embeddings.npz",
        CACHE_DIR / "supervised_embeddings.npy",
    ]
    missing = [p for p in cached if not p.exists()]
    if not missing:
        cache_meta = json.loads((CACHE_DIR / "meta.json").read_text())
        npz = np.load(CACHE_DIR / "strict_embeddings.npz")
        n_profile = int(npz["z_profile"].shape[0])
        n_test = int(npz["z_test"].shape[0])
        n_users_cached = int(len(npz["uid_list"]))
        if (n_profile + n_test != cache_meta["n_total_sents"]
                or n_users_cached != cache_meta["n_users"]):
            raise ValueError(
                f"stage_strict cache cohort mismatch: cached profile="
                f"{n_profile} test={n_test} users={n_users_cached} vs "
                f"current n_sents={cache_meta['n_total_sents']} "
                f"n_users={cache_meta['n_users']}; delete "
                f"{CACHE_DIR / 'strict_embeddings.npz'} and rerun")
        _verify_cached_embeddings_cohort(
            CACHE_DIR / "supervised_embeddings.npy",
            cache_meta["n_total_sents"], SUP_Z_DIM, "stage_strict")
        log(f"skip stage_strict: {len(cached)} cached output(s) exist, "
            f"cohort match (n_sents={cache_meta['n_total_sents']}, "
            f"n_users={cache_meta['n_users']})")
        return

    cache = load_cache()
    sent_csr = cache["sent_csr"]
    V = sent_csr.shape[1]
    n_sents = sent_csr.shape[0]
    user_n_sents = cache["user_n_sents"]
    uid_list = cache["uid_list"]
    n_users = len(uid_list)
    log(f"strict: V={V}, n_sents={n_sents}, n_users={n_users}")

    sent_norm = normalize_counts(sent_csr)

    def to_dense_batch(indices: np.ndarray) -> torch.Tensor:
        rows = sent_norm[indices]
        return torch.tensor(rows.toarray(), dtype=torch.float32,
                            device=device)

    # === Step 1: profile/test split (same as _profile_test_split) ===
    profile_idx_dict, test_idx_dict = _profile_test_split(
        uid_list, cache, SENT_CACHE)
    profile_idx = np.asarray(
        sorted(i for v in profile_idx_dict.values() for i in v),
        dtype=np.int64)
    test_idx = np.asarray(
        sorted(i for v in test_idx_dict.values() for i in v),
        dtype=np.int64)
    log(f"profile={len(profile_idx)}, test={len(test_idx)} (50/50 hash, encoder "
        f"只见过 profile 的 {len(profile_idx)} 句 + user_id labels)")

    # per-sentence user labels
    user_labels = np.zeros(n_sents, dtype=np.int64)
    off = 0
    for ui, n in enumerate(user_n_sents):
        user_labels[off:off + n] = ui
        off += n

    # === Step 2: encoder 只用 profile 训练 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(STRICT_SEED)
    np.random.seed(STRICT_SEED)

    # profile_train / profile_val (15% val)
    rng_local = np.random.default_rng(STRICT_SEED)
    perm = rng_local.permutation(len(profile_idx))
    n_val = int(len(profile_idx) * STRICT_VAL_FRAC)
    val_idx_global = profile_idx[perm[:n_val]]
    train_idx_global = profile_idx[perm[n_val:]]
    log(f"profile_train={len(train_idx_global)}, "
        f"profile_val={len(val_idx_global)} (内部早停用)")

    y_train = torch.tensor(user_labels[train_idx_global],
                           dtype=torch.long, device=device)
    y_val = torch.tensor(user_labels[val_idx_global],
                         dtype=torch.long, device=device)

    model = _SupEncoder(V, SUP_Z_DIM, SUP_HIDDEN, n_users,
                        SUP_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"encoder: V={V} → {SUP_HIDDEN} → z={SUP_Z_DIM}, "
        f"params={n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR,
                           weight_decay=SUP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SUP_EPOCHS)

    best_val = 0.0
    best_state = None
    epochs_since_best = 0
    history = []
    n_train = len(train_idx_global)
    for epoch in range(1, SUP_EPOCHS + 1):
        model.train()
        perm_epoch = np.random.permutation(train_idx_global)
        ep_loss = ep_correct = 0
        n_batches = 0
        for i in range(0, n_train, SUP_BATCH_SIZE):
            batch_idx = perm_epoch[i:i + SUP_BATCH_SIZE]
            xb = to_dense_batch(batch_idx)
            yb = torch.tensor(user_labels[batch_idx],
                              dtype=torch.long, device=device)
            if SUP_RULE_DROPOUT > 0:
                mask = (torch.rand(xb.shape[1], device=device)
                        > SUP_RULE_DROPOUT).float()
                xb = xb * mask.unsqueeze(0)
            opt.zero_grad()
            _, logits = model(xb)
            loss = F.cross_entropy(logits, yb,
                                   label_smoothing=SUP_LABEL_SMOOTHING)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            ep_correct += int((logits.argmax(dim=1) == yb).sum().detach())
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v_correct = 0
            for i in range(0, len(val_idx_global), SUP_BATCH_SIZE):
                batch_idx = val_idx_global[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(batch_idx)
                _, logits = model(xb)
                yb = torch.tensor(user_labels[batch_idx],
                                  dtype=torch.long, device=device)
                v_correct += int(
                    (logits.argmax(dim=1) == yb).sum().detach())
        v_acc = v_correct / max(len(val_idx_global), 1)
        t_acc = ep_correct / max(n_train, 1)
        if n_batches <= 0:
            raise RuntimeError("encoder produced zero training batches")
        avg_loss = ep_loss / n_batches
        history.append({"epoch": epoch, "loss": avg_loss,
                        "train_acc": t_acc, "val_acc": v_acc})
        if v_acc > best_val:
            best_val = v_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if (epoch == 1 or epoch % SUP_LOG_EVERY == 0
                or epoch == SUP_EPOCHS):
            log(f"  epoch {epoch:>3}/{SUP_EPOCHS}  loss={avg_loss:.4f}  "
                f"train={t_acc*100:.1f}%  val={v_acc*100:.1f}%  "
                f"plateau={epochs_since_best}/{STRICT_PATIENCE}")
        if epochs_since_best >= STRICT_PATIENCE:
            log(f"  early stop at epoch {epoch}: val_acc plateau "
                f"{STRICT_PATIENCE} epochs (best={best_val*100:.2f}%)")
            break

    log(f"best val_acc: {best_val*100:.1f}%")
    model.load_state_dict(best_state)
    model.eval()

    # === Step 3: freeze encoder, encode profile + test ===
    def encode_all(indices: np.ndarray) -> np.ndarray:
        out = np.zeros((len(indices), SUP_Z_DIM), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(indices), SUP_BATCH_SIZE):
                batch_idx = indices[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(batch_idx)
                z, _ = model(xb)
                out[i:i + xb.shape[0]] = z.cpu().numpy()
        return out

    z_profile = encode_all(profile_idx)
    z_test = encode_all(test_idx)
    log(f"encoded: z_profile {z_profile.shape}, z_test {z_test.shape}")

    # === Step 3b: 用 strict encoder 再编码全部 n_sents 句子 (供 gauss/attr/gauss_attr
    #     消费, 与 strict_embeddings.npz 共享同一 encoder 但不泄露 test 标签,
    #     只是 encoder 本身训练时只看了 profile,test 句子本身仍可见) ===
    all_idx = np.arange(n_sents, dtype=np.int64)
    z_all = encode_all(all_idx)
    np.save(CACHE_DIR / "supervised_embeddings.npy", z_all)
    log(f"encoded all (供 gauss/attr/gauss_attr 消费): "
        f"z_all {z_all.shape} → supervised_embeddings.npy")

    # === Step 4: per-user Gaussian on profile ===
    mu = np.zeros((n_users, SUP_Z_DIM), dtype=np.float64)
    var = np.zeros((n_users, SUP_Z_DIM), dtype=np.float64)
    n_prof_per = np.zeros(n_users, dtype=np.int64)
    for ui, uid in enumerate(uid_list):
        # 重建该 user 的 profile 索引 (因为 profile_idx 排序过)
        idxs = profile_idx_dict[uid]
        n_prof_per[ui] = len(idxs)
        if not idxs:
            continue
        z = z_profile[np.searchsorted(profile_idx, idxs)]
        mu[ui] = z.mean(axis=0)
        if len(idxs) >= 2:
            var[ui] = z.var(axis=0, ddof=1)
    var_reg = var + GAUSS_EPS
    pooled_var = var.mean(axis=0) + GAUSS_EPS
    inv_pooled = 1.0 / pooled_var

    log(f"per-user Gaussian: μ norm med="
        f"{np.median(np.linalg.norm(mu,axis=1)):.3f}, "
        f"σ² mean={var.mean(axis=1).mean():.4f}, "
        f"profile sents/user avg={n_prof_per.mean():.1f}")

    # === Step 5: test attribution (三种方法) ===
    chance = 1.0 / n_users
    method_results = {}

    # --- cosine ---
    t1 = time.time()
    mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    correct = 0
    for ui, uid in enumerate(uid_list):
        idxs = test_idx_dict[uid]
        for global_i in idxs:
            local_i = np.searchsorted(test_idx, global_i)
            z_q = z_test[local_i]
            z_q_norm = z_q / (np.linalg.norm(z_q) + 1e-8)
            sim = mu_norm @ z_q_norm
            if int(np.argmax(sim)) == ui:
                correct += 1
    total = len(test_idx)
    acc = correct / max(total, 1)
    method_results["cosine"] = {"acc": acc, "chance": chance,
                                "lift": acc / chance,
                                "n_correct": correct, "n_total": total,
                                "elapsed_s": time.time() - t1}
    log(f"  cosine: {correct}/{total} = {acc*100:.2f}% "
        f"lift {acc/chance:.2f}x")

    # --- Gaussian log-P (diag Σ) ---
    t1 = time.time()
    half_log_det = 0.5 * np.log(var_reg).sum(axis=1)
    correct = 0
    for ui, uid in enumerate(uid_list):
        idxs = test_idx_dict[uid]
        for global_i in idxs:
            local_i = np.searchsorted(test_idx, global_i)
            z_q = z_test[local_i].astype(np.float64)
            diff = z_q - mu
            mahal = (diff ** 2 / var_reg).sum(axis=1)
            log_p = -0.5 * mahal - half_log_det
            if int(np.argmax(log_p)) == ui:
                correct += 1
    acc = correct / max(total, 1)
    method_results["log_p_diag"] = {"acc": acc, "chance": chance,
                                    "lift": acc / chance,
                                    "n_correct": correct, "n_total": total,
                                    "elapsed_s": time.time() - t1}
    log(f"  log_p_diag: {correct}/{total} = {acc*100:.2f}% "
        f"lift {acc/chance:.2f}x")

    # --- Mahalanobis pooled ---
    t1 = time.time()
    correct = 0
    for ui, uid in enumerate(uid_list):
        idxs = test_idx_dict[uid]
        for global_i in idxs:
            local_i = np.searchsorted(test_idx, global_i)
            z_q = z_test[local_i].astype(np.float64)
            diff = z_q - mu
            mahal = ((diff ** 2) * inv_pooled).sum(axis=1)
            if int(np.argmin(mahal)) == ui:
                correct += 1
    acc = correct / max(total, 1)
    method_results["mahalanobis_pooled"] = {"acc": acc, "chance": chance,
                                            "lift": acc / chance,
                                            "n_correct": correct,
                                            "n_total": total,
                                            "elapsed_s": time.time() - t1}
    log(f"  mahalanobis_pooled: {correct}/{total} = {acc*100:.2f}% "
        f"lift {acc/chance:.2f}x")

    # === Output ===
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "z_dim": SUP_Z_DIM, "n_users": n_users,
            "hidden": list(SUP_HIDDEN), "dropout": SUP_DROPOUT,
            "rule_dropout": SUP_RULE_DROPOUT,
            "weight_decay": SUP_WEIGHT_DECAY,
            "label_smoothing": SUP_LABEL_SMOOTHING,
            "epochs": SUP_EPOCHS, "lr": SUP_LR,
            "n_profile": int(len(profile_idx)),
            "n_profile_train": int(len(train_idx_global)),
            "n_profile_val": int(len(val_idx_global)),
            "n_test": int(len(test_idx)),
            "split": "sha1_hash_50_50",
            "gauss_eps": GAUSS_EPS,
            "seed": STRICT_SEED,
        },
        "no_leakage": True,
        "encoder_seen_test": False,
        "chance": chance,
        "lifts": {m: method_results[m]["lift"] for m in method_results},
        "per_method": method_results,
        "encoder_diagnostics": {
            "best_val_acc": best_val,
            "history_last": history[-1],
            "n_params_M": n_params / 1e6,
            "mu_norm_median": float(np.median(np.linalg.norm(mu, axis=1))),
            "sigma2_median_mean": float(np.median(var.mean(axis=1))),
            "pooled_sigma2_median": float(np.median(pooled_var)),
            "n_profile_per_user_avg": float(n_prof_per.mean()),
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 同时把 strict 训练的 encoder + profile/test embeddings 持久化
    # 便于后续 stage 复用 (避免重训)
    torch.save({"model_state": best_state,
                "config": {"vocab_size": V, "z_dim": SUP_Z_DIM,
                           "hidden": list(SUP_HIDDEN), "dropout": SUP_DROPOUT,
                           "n_users": n_users}},
               CACHE_DIR / "strict_encoder.pt")
    np.savez(CACHE_DIR / "strict_embeddings.npz",
             z_profile=z_profile, z_test=z_test,
             profile_idx=profile_idx, test_idx=test_idx,
             uid_list=np.asarray(uid_list))
    log(f"wrote → {out_path} + strict_encoder.pt + strict_embeddings.npz")

    log(f"\n=== STRICT no-leakage SUMMARY (z={SUP_Z_DIM}, N={n_users}) ===")
    log(f"  PCFG discrete (历史):     ~7.97× chance")
    log(f"  cosine to μ_u:            {method_results['cosine']['lift']:.2f}x "
        f"({method_results['cosine']['acc']*100:.2f}%)")
    log(f"  Gaussian log-P (diag):    {method_results['log_p_diag']['lift']:.2f}x "
        f"({method_results['log_p_diag']['acc']*100:.2f}%)")
    log(f"  Mahalanobis (pooled Σ):   {method_results['mahalanobis_pooled']['lift']:.2f}x "
        f"({method_results['mahalanobis_pooled']['acc']*100:.2f}%)")
    log(f"=== Total: {time.time()-t0:.1f}s ===")




# ============================================================================
# Stage: strict3 (3-way 50/20/30 split — unified encoder, 单次训练)
# ============================================================================

# 3-way split 桶配置 (per Rule 3 硬编码)
STRICT3_PROFILE_FRAC = 0.50
STRICT3_VAL_FRAC = 0.20
STRICT3_TEST_FRAC = 0.30


def stage_strict3(out_path: Path = None):
    """统一 2-way strict encoder (80/20, no separate test) — Stage 03/04 合并后唯一 encoder。

    协议 (per user 2026-09-15: 80% 拟合 / 20% val, 不需要 test):
      1. SHA1 hash mod=10 两段 split → profile (b<8, 80%) / val (b≥8, 20%)
      2. encoder 用 profile 训练, val(20%) 直接作早停 + 下游 held-out
         (不再从 profile 内切 15% inner val)
      3. freeze encoder, 编码 profile + val (no separate test)
      4. 用 strict3 encoder 再编码全部 n_sents 句子 → z_all (供 04_gaussian 消费)
      5. 输出 strict3_embeddings.npz + strict3_encoder.pt
         (z_test / test_idx 字段保留为 val 别名, 以维持下游 04_gaussian/analysis
          既有契约不破; 这些脚本读到的 test 实质就是 val held-out)

    输出:
      pcfg_cache/strict3_embeddings.npz
        z_profile (n_p, z_dim) + z_val (n_v, z_dim) + z_test (=z_val 别名)
        profile_idx / val_idx / test_idx (=val_idx 别名)
        uid_list (cohort uids)
      pcfg_cache/strict3_encoder.pt (best ckpt)
      pcfg_cache/supervised_embeddings.npy (full z_all, 兼容旧 consumer)

    下游:
      04_gaussian/fit_per_user_gaussian.py 读 strict3_embeddings.npz
        → fit raw full Σ_u + ASIN cohort gates → user_gaussian_stats.json
      analysis/syntax_pcfg_adaptive_eval.py 读 strict3_embeddings.npz
        → τ sweep + val held-out eval → syntax_pcfg_adaptive.json
    """
    t0 = time.time()
    canonical_npz = CACHE_DIR / "strict3_embeddings.npz"
    canonical_pt = CACHE_DIR / "strict3_encoder.pt"
    strict3_manifest_path = CACHE_DIR / "strict3_manifest.json"
    legacy_npz = CACHE_DIR / "strict_embeddings.npz"
    legacy_pt = CACHE_DIR / "strict_encoder.pt"
    # legacy 2-way artifacts 不得阻塞 canonical strict3；它们不会被任何下游消费。
    if legacy_npz.exists() or legacy_pt.exists():
        log("  legacy strict encoder artifact(s) detected; ignoring them")
    summary_path = out_path or OUT_DIR / "syntax_pcfg_strict_attr.json"
    cached = [canonical_npz, canonical_pt,
              CACHE_DIR / "supervised_embeddings.npy", strict3_manifest_path]
    missing = [p for p in cached if not p.exists()]
    cache = load_cache()
    if not missing:
        npz = np.load(canonical_npz, allow_pickle=False)
        manifest = json.loads(strict3_manifest_path.read_text())
        source_keys = ("cohort_fingerprint", "uid_layout_fingerprint",
                       "vocab_fingerprint")
        source_matches = all(
            manifest.get(key) == cache["meta"][
                {"cohort_fingerprint": "sentence_source_fingerprint",
                 "uid_layout_fingerprint": "uid_layout_fingerprint",
                 "vocab_fingerprint": "vocab_fingerprint"}[key]]
            for key in source_keys)
        npz_source_matches = all(
            key in npz and _npz_text(npz, key) == manifest.get(key)
            for key in source_keys)
        if source_matches and npz_source_matches:
            try:
                _validate_strict3_artifact(npz, cache)
            except ValueError as ve:
                log(f"  strict3 cached artifact validation failed "
                    f"(contract drift, likely 50/20/30 → 80/20): {ve}")
                log("  forcing rebuild for current cohort contract")
            else:
                z_all = np.load(CACHE_DIR / "supervised_embeddings.npy",
                                allow_pickle=False)
                _verify_cached_embeddings_cohort(
                    CACHE_DIR / "supervised_embeddings.npy",
                    cache["meta"]["n_total_sents"], SUP_Z_DIM, "stage_strict3")
                if _array_fingerprint(z_all) != manifest.get("z_all_fingerprint"):
                    raise ValueError("strict3 supervised embedding fingerprint mismatch")
                checkpoint = torch.load(canonical_pt, map_location="cpu")
                if not isinstance(checkpoint, dict) or not isinstance(
                        checkpoint.get("config"), dict):
                    raise ValueError("strict3 checkpoint missing config")
                config = checkpoint["config"]
                expected_config = {
                    "vocab_size": int(cache["sent_csr"].shape[1]),
                    "z_dim": SUP_Z_DIM,
                    "hidden": list(SUP_HIDDEN),
                    "dropout": SUP_DROPOUT,
                    "n_users": len(cache["uid_list"]),
                }
                for key, expected_value in expected_config.items():
                    if config.get(key) != expected_value:
                        raise ValueError(
                            f"strict3 checkpoint mismatch at {key}: "
                            f"cached={config.get(key)!r}, current={expected_value!r}")
                for key in source_keys:
                    if checkpoint.get(key) != manifest.get(key):
                        raise ValueError(
                            f"strict3 checkpoint source mismatch at {key}")
                log(f"skip stage_strict3: {len(cached)} cached output(s) exist, "
                    f"validated cohort={cache['meta']['sentence_source_fingerprint'][:12]}...")
                return
        log("  stale strict3 artifacts detected; rebuilding for current cohort")

    if missing:
        log(f"  strict3 missing outputs: {[p.name for p in missing]}")
    sent_csr = cache["sent_csr"]
    V = sent_csr.shape[1]
    n_sents = sent_csr.shape[0]
    user_n_sents = list(cache["user_n_sents"])
    uid_list = list(cache["uid_list"])
    if _EFFECTIVE_TRAIN_LIMIT is not None:
        uid_list = uid_list[:_EFFECTIVE_TRAIN_LIMIT]
        user_n_sents = user_n_sents[:_EFFECTIVE_TRAIN_LIMIT]
        n_smoke_sents = sum(user_n_sents)
        sent_csr = sent_csr[:n_smoke_sents]
        n_sents = n_smoke_sents
        log(f"  [SMOKE] limited to {len(uid_list)} users, {n_sents} sents")
    n_users = len(uid_list)
    log(f"strict3: V={V}, n_sents={n_sents}, n_users={n_users}")

    sent_norm = normalize_counts(sent_csr)

    def to_dense_batch(indices: np.ndarray) -> torch.Tensor:
        rows = sent_norm[indices]
        return torch.tensor(rows.toarray(), dtype=torch.float32,
                            device=device)

    # === Step 1: 2-way split (80% train / 20% val, no separate test) ===
    train_dict, val_dict = _two_way_split(uid_list, SENT_CACHE)
    profile_idx = np.asarray(
        sorted(i for v in train_dict.values() for i in v),
        dtype=np.int64)
    val_idx = np.asarray(
        sorted(i for v in val_dict.values() for i in v), dtype=np.int64)
    # test 字段作为 val 别名保留, 下游契约不变
    test_idx = val_idx
    log(f"profile={len(profile_idx)} (80%), val={len(val_idx)} (20%), "
        f"test=val_alias (no separate test split)")

    # per-sentence user labels
    user_labels = np.zeros(n_sents, dtype=np.int64)
    off = 0
    for ui, n in enumerate(user_n_sents):
        user_labels[off:off + n] = ui
        off += n

    # === Step 2: encoder 训练只用 profile (内部 15% 早停) ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(STRICT_SEED)
    np.random.seed(STRICT_SEED)

    rng_local = np.random.default_rng(STRICT_SEED)
    # 2026-09-15: 2-way 80/20, profile 直接作 encoder 训练数据,
    # val (20%) 直接作早停 + held-out, 不再从 profile 内部切 15% inner val
    profile_train_inner = profile_idx
    profile_val_inner = val_idx
    log(f"encoder train: {len(profile_train_inner)} train (80%) + "
        f"{len(profile_val_inner)} held-out val (20%)")

    model = _SupEncoder(V, SUP_Z_DIM, SUP_HIDDEN, n_users,
                        SUP_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"encoder: V={V} → {SUP_HIDDEN} → z={SUP_Z_DIM}, "
        f"params={n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR,
                           weight_decay=SUP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SUP_EPOCHS)

    best_val = -1.0
    best_state = None
    epochs_since_best = 0
    history = []
    n_train = len(profile_train_inner)
    if n_train <= 0 or len(profile_val_inner) <= 0:
        raise ValueError("strict3 requires non-empty train and validation splits")
    for epoch in range(1, SUP_EPOCHS + 1):
        model.train()
        perm_epoch = np.random.permutation(profile_train_inner)
        ep_loss = ep_correct = 0
        n_batches = 0
        for i in range(0, n_train, SUP_BATCH_SIZE):
            batch_idx = perm_epoch[i:i + SUP_BATCH_SIZE]
            xb = to_dense_batch(batch_idx)
            yb = torch.tensor(user_labels[batch_idx],
                              dtype=torch.long, device=device)
            if SUP_RULE_DROPOUT > 0:
                mask = (torch.rand(xb.shape[1], device=device)
                        > SUP_RULE_DROPOUT).float()
                xb = xb * mask.unsqueeze(0)
            opt.zero_grad()
            _, logits = model(xb)
            loss = F.cross_entropy(logits, yb,
                                   label_smoothing=SUP_LABEL_SMOOTHING)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            ep_correct += int((logits.argmax(dim=1) == yb).sum().detach())
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v_correct = 0
            for i in range(0, len(profile_val_inner), SUP_BATCH_SIZE):
                batch_idx = profile_val_inner[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(batch_idx)
                _, logits = model(xb)
                yb = torch.tensor(user_labels[batch_idx],
                                  dtype=torch.long, device=device)
                v_correct += int(
                    (logits.argmax(dim=1) == yb).sum().detach())
        v_acc = v_correct / max(len(profile_val_inner), 1)
        t_acc = ep_correct / max(n_train, 1)
        if n_batches <= 0:
            raise RuntimeError("encoder produced zero training batches")
        avg_loss = ep_loss / n_batches
        history.append({"epoch": epoch, "loss": avg_loss,
                        "train_acc": t_acc, "val_acc": v_acc})
        if v_acc > best_val:
            best_val = v_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if (epoch == 1 or epoch % SUP_LOG_EVERY == 0
                or epoch == SUP_EPOCHS):
            log(f"  epoch {epoch:>3}/{SUP_EPOCHS}  loss={avg_loss:.4f}  "
                f"train={t_acc*100:.1f}%  val={v_acc*100:.1f}%  "
                f"plateau={epochs_since_best}/{STRICT_PATIENCE}")
        if epochs_since_best >= STRICT_PATIENCE:
            log(f"  early stop at epoch {epoch}: val_acc plateau "
                f"{STRICT_PATIENCE} epochs (best={best_val*100:.2f}%)")
            break

    log(f"best val_acc: {best_val*100:.1f}%")
    model.load_state_dict(best_state)
    model.eval()

    # === Step 3: 编码三段 + 全量 ===
    def encode_all(indices: np.ndarray) -> np.ndarray:
        out = np.zeros((len(indices), SUP_Z_DIM), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(indices), SUP_BATCH_SIZE):
                batch_idx = indices[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(batch_idx)
                z, _ = model(xb)
                out[i:i + xb.shape[0]] = z.cpu().numpy()
        return out

    z_profile = encode_all(profile_idx)
    z_val = encode_all(val_idx)
    # test = val 别名, 复用同一嵌入以维持下游契约
    z_test = z_val
    log(f"encoded: z_profile {z_profile.shape}, z_val {z_val.shape}, "
        f"z_test = z_val (alias, no separate test split)")

    # === Step 4: 用同一 encoder 编码全部 n_sents → z_all (给 04_gaussian 消费) ===
    all_idx = np.arange(n_sents, dtype=np.int64)
    z_all = encode_all(all_idx)
    # === Step 5: 持久化（所有来源字段写入同一 artifact，最后才原子提交） ===
    cohort_fingerprint = cache["meta"]["sentence_source_fingerprint"]
    uid_layout_fingerprint = cache["meta"]["uid_layout_fingerprint"]
    vocab_fingerprint = cache["meta"]["vocab_fingerprint"]
    _atomic_save_npy(CACHE_DIR / "supervised_embeddings.npy", z_all)
    log(f"encoded all (供 04_gaussian 消费): z_all {z_all.shape} → "
        f"supervised_embeddings.npy")
    _atomic_torch_save({"model_state": best_state,
                        "config": {"vocab_size": V, "z_dim": SUP_Z_DIM,
                                   "hidden": list(SUP_HIDDEN), "dropout": SUP_DROPOUT,
                                   "n_users": n_users},
                        "cohort_fingerprint": cohort_fingerprint,
                        "uid_layout_fingerprint": uid_layout_fingerprint,
                        "vocab_fingerprint": vocab_fingerprint},
                       canonical_pt)
    _atomic_save_npz_arrays(
        canonical_npz,
        z_profile=z_profile, z_val=z_val, z_test=z_test,
        profile_idx=profile_idx, val_idx=val_idx, test_idx=test_idx,
        uid_list=np.asarray(uid_list),
        cohort_fingerprint=np.asarray(cohort_fingerprint),
        uid_layout_fingerprint=np.asarray(uid_layout_fingerprint),
        vocab_fingerprint=np.asarray(vocab_fingerprint))
    strict3_manifest = {
        "schema_version": 1,
        "cohort_fingerprint": cohort_fingerprint,
        "uid_layout_fingerprint": uid_layout_fingerprint,
        "vocab_fingerprint": vocab_fingerprint,
        "n_users": n_users,
        "n_total_sents": n_sents,
        "z_dim": SUP_Z_DIM,
        "vocab_size": V,
        "z_all_fingerprint": _array_fingerprint(z_all),
        "strict3_npz": str(canonical_npz),
        "strict3_checkpoint": str(canonical_pt),
    }
    _atomic_json_dump(strict3_manifest, strict3_manifest_path)
    log(f"wrote → {canonical_npz.name} + {canonical_pt.name} + "
        f"{strict3_manifest_path.name}")

    log(f"\n=== STRICT3 (2-way 80/20, test=val alias) SUMMARY "
        f"(z={SUP_Z_DIM}, N={n_users}) ===")
    log(f"  profile (train) sents: {len(profile_idx)} (80%)")
    log(f"  val sents:              {len(val_idx)} (20%, held-out)")
    log(f"  test sents:             {len(test_idx)} (=val alias)")
    log(f"  z_all sents:            {n_sents}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")




# ============================================================================
# Main pipeline (一键串行运行所有 stage, 不依赖 MODE 调度)
# ============================================================================

def main_pipeline():
    """Canonical pipeline: 串行执行全部 stage (统一 encoder, 单次训练)。

    链路: cache → strict3 (2-way 80/20 split + CE encoder, 唯一一次训练, 无独立 test)
    Gaussian fitting / adaptive eval 一律外迁:
      - 04_gaussian/fit_per_user_gaussian.py   per-user raw full Σ + ASIN cohort gates
      - analysis/syntax_pcfg_adaptive_eval.py  τ sweep + held-out eval (读 strict3)

    假设 cache 已有产物则可跳过 stage_cache() (resume);
    任意 stage 抛错会立即终止, 不做 fallback (per Rule 7)。
    """
    global N_USERS_LIMIT, SUP_EPOCHS, STRICT_PATIENCE, _EFFECTIVE_TRAIN_LIMIT
    t_total = time.time()
    log(f"=== main_pipeline: canonical chain start ===")

    # --- Smoke overrides: only affect training stage, NOT cache ---
    if SMOKE:
        _EFFECTIVE_TRAIN_LIMIT = N_USERS_LIMIT_SMOKE
        log(f"  [SMOKE MODE] training will use: "
            f"train_limit={_EFFECTIVE_TRAIN_LIMIT}, "
            f"SUP_EPOCHS={SMOKE_EPOCHS}, STRICT_PATIENCE={SMOKE_PATIENCE}")
        orig_epochs = SUP_EPOCHS
        orig_patience = STRICT_PATIENCE
        SUP_EPOCHS = SMOKE_EPOCHS
        STRICT_PATIENCE = SMOKE_PATIENCE
        # N_USERS_LIMIT still None → cache builds full cohort (resume-friendly)

    stage_cache()
    stage_strict3()  # 统一 3-way encoder,产出 strict3_embeddings.npz + supervised_embeddings.npy

    if SMOKE:
        SUP_EPOCHS = orig_epochs
        STRICT_PATIENCE = orig_patience
        _EFFECTIVE_TRAIN_LIMIT = None
        log(f"  [SMOKE MODE] restored training globals")

    log(f"\n=== ALL DONE ({time.time()-t_total:.0f}s) ===")


if __name__ == "__main__":
    main_pipeline()