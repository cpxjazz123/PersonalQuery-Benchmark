#!/usr/bin/env python3
"""Stage 03 — unified syntax encoder (supervised + cohort3).

Two tasks in one script (per Rule 18, single main script per functional dir):

TASK = "supervised"  (default; canonical Stage 03a)
    Pipeline (cache → strict3 80/20 supervised encoder):
      cache        parse sentences → sparse rule-count matrix
      strict3      2-way 80/20 supervised encoder (V → 256 → SUP_Z_DIM),
                    train on profile (80%), early-stop on val (20%),
                    encode all sentences → supervised_embeddings.npy +
                    strict3_embeddings.npz + strict3_encoder.pt

TASK = "cohort3"    (Stage 03b; (user, ASIN) multi-positive InfoNCE)
    Pipeline (real parent_asin sentence ownership → cohort3 MLP):
      - load SVD-z cache (already built by TASK=supervised)
      - retain (user, ASIN) pairs with >= 50 real sentences
      - retain ASINs with exactly 3 qualified users
      - split each pair's 50 sentences → 30 train / 10 val / 10 test
      - each train anchor uses 6 positives + 12 negatives (2 cohorts × 6)
      - multi-positive InfoNCE; no fallback sampling
      - save cohort3_mlp32_30_30ep.pt + cohort3_trained_uids.json +
        real_cohort3_training_summary.json

Usage (per Rule 3, no args):
  TASK=supervised  cd /home/wlia0047/ar57/wenyu/PersoanlQuery && $PY 03_spacy_encode/syntax_encoder.py
  TASK=cohort3     cd /home/wlia0047/ar57/wenyu/PersoanlQuery && $PY 03_spacy_encode/syntax_encoder.py

2026-09-23: 改为串行跑 3 个 category (Baby / Musical_Instruments / Video_Games).
  每个 category 写产物到 result/03_spacy_encode/<baby|musical|video_games>/.
  默认指向 Baby 的常量 (SENT_CACHE / RAW_DATA / OUT_DIR) 保持 Baby 路径, 不破坏
  Stage 04/05/08/09 等下游只读 Baby 路径的兼容。

History (canonical supervised 32d → 16d per Phase L8.7b; cohort3 was a
separate script pre-2026-09-18 and has been merged here).
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz, save_npz

# ============================================================================
# Constants (hardcoded per Rule 3)
# ============================================================================

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
# 用户指令 2026-09-23: data 目录从 REPO_ROOT/data 迁移到 hj82 同名 data 目录.
DATA_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")
# 用户指令 2026-09-23: 3 个 category 各自一份 Stage 02 产物 (Baby / Musical / Video_Games).
# 用户指令 2026-09-23: 改串行运行 3 个 category (默认仍走 Baby, 完整产物在 <out>/<category>/ 下).
CATEGORY_INPUTS = [
    # (category_key, raw_data_path, uid_to_sentences_pkl, output_subdir)
    ("Baby",                DATA_DIR / "Baby_Products_2023.jsonl",
                            REPO_ROOT / "result/02_user_review_sentence_extract/uid_to_sentences_baby.pkl",
                            "baby"),
    ("Musical_Instruments", DATA_DIR / "Musical_Instruments.jsonl",
                            REPO_ROOT / "result/02_user_review_sentence_extract/uid_to_sentences_musical.pkl",
                            "musical"),
    ("Video_Games",         DATA_DIR / "Video_Games.jsonl",
                            REPO_ROOT / "result/02_user_review_sentence_extract/uid_to_sentences_video_games.pkl",
                            "video_games"),
]
# 默认 SENT_CACHE / RAW_DATA / OUT_DIR 指向 Baby (向后兼容, Stage 04/05/08/09 只看 Baby).
SENT_CACHE = CATEGORY_INPUTS[0][2]
OUT_DIR = REPO_ROOT / "result/03_spacy_encode"
# 用户指令 2026-09-23: cache 拆 per-category (Baby / Musical / Video_Games 各一份),
# 默认指向 Baby. dispatcher 循环中重绑.
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache_baby")
RAW_DATA = CATEGORY_INPUTS[0][1]

# === Supervised (TASK=supervised) ===
# 用户指令 2026-09-23: 改用 en_core_web_trf 实测 GPU 反而比 sm 慢 7x (BERT-large + torch 2.14 + spacy-transformers 不跑满 GPU).
# 实测: sm n_process=12 = 2400 sent/sec vs trf GPU batch=512 = 333 sent/sec.
# 回退到 sm + n_process=12 (最优 baseline).
SPACY_MODEL = "en_core_web_sm"
PARSE_BATCH_SIZE = 1024
PARSE_N_PROCESS = 12     # sm CPU 多 process, A40 单卡情况下 12 process 反而比 trf GPU 快
CHUNK_USERS = 2000
SEED = 42
HASH_SALT = "pcfg_lopo_v1"

N_USERS_LIMIT = None
MIN_SENTS_PER_USER = 30
MIN_DOC_FREQ = 2

SMOKE = False
N_USERS_LIMIT_SMOKE = 200
SMOKE_EPOCHS = 5
SMOKE_PATIENCE = 3

SUP_EPOCHS = 80
SUP_BATCH_SIZE = 2048
SUP_LR = 5e-4
SUP_Z_DIM = 16
SUP_HIDDEN = (256,)
SUP_DROPOUT = 0.5
SUP_RULE_DROPOUT = 0.1
SUP_WEIGHT_DECAY = 1e-2
SUP_LABEL_SMOOTHING = 0.1
SUP_LOG_EVERY = 5
SUP_TRAIN_FRAC = 0.8

STRICT_VAL_FRAC = 0.15
STRICT_SEED = 42
STRICT_PATIENCE = 10

_EFFECTIVE_TRAIN_LIMIT = None

COHORT_FINGERPRINT_VERSION = "sha256-cohort-v1"

# === Cohort3 (TASK=cohort3) ===
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
COHORT3_N_PAIR_SENTENCES = 30
COHORT3_N_TRAIN = 18
COHORT3_N_VAL = 6
COHORT3_N_TEST = 6
COHORT3_COHORT_USERS = 2
COHORT3_N_POS = 4
COHORT3_K_PER_NEG_USER = COHORT3_N_POS
COHORT3_N_EPOCHS = 30
COHORT3_BATCH_SIZE = 128
COHORT3_N_NEG = (COHORT3_COHORT_USERS - 1) * COHORT3_K_PER_NEG_USER
COHORT3_TEMPERATURE = 0.1
COHORT3_LR = 3e-3
COHORT3_SVD_DIM = 256
COHORT3_MLP_HIDDEN = 128
COHORT3_MLP_OUT = 16
COHORT3_SMOKE = False
COHORT3_SMOKE_N_ASINS = 5

_COHORT3_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_COHORT3_HTML_TAG_RE = re.compile(r"<br\s*/?>")
_COHORT3_HTML_OTHER_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_COHORT3_MULTI_SPACE_RE = re.compile(r" {2,}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Shared helpers (used by both tasks)
# ============================================================================

def _hash_length_prefixed_text(hasher, text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"cohort fingerprint requires str, got {type(text)!r}")
    raw = text.encode("utf-8")
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _cohort_fingerprint(uid_list, uid_to_sents):
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


def _uid_layout_fingerprint(uid_list, user_n_sents):
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


def _array_fingerprint(array):
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, str(contiguous.dtype))
    _hash_length_prefixed_text(hasher, repr(tuple(contiguous.shape)))
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _sparse_fingerprint(matrix):
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, str(matrix.dtype))
    _hash_length_prefixed_text(hasher, repr(tuple(matrix.shape)))
    for part in (matrix.indptr, matrix.indices, matrix.data):
        contiguous = np.ascontiguousarray(part)
        _hash_length_prefixed_text(hasher, str(contiguous.dtype))
        _hash_length_prefixed_text(hasher, repr(tuple(contiguous.shape)))
        hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _string_list_fingerprint(values, tag):
    hasher = hashlib.sha256()
    _hash_length_prefixed_text(hasher, tag)
    for value in values:
        _hash_length_prefixed_text(hasher, value)
    return hasher.hexdigest()


def _atomic_json_dump(payload, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_pickle_dump(payload, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_save_npz(path, matrix):
    tmp = Path(str(path) + ".tmp.npz")
    save_npz(tmp, matrix)
    os.replace(tmp, path)


def _atomic_save_npy(path, array):
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _atomic_save_npz_arrays(path, **arrays):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_torch_save(payload, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def extract_struct_rules(doc):
    """D4/D3/P3 dependency rules from a spaCy Doc (no word terms)."""
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


def hash_bucket(text, mod=2, salt=None):
    s = HASH_SALT if salt is None else salt
    h = int(hashlib.sha1(
        (s + "|" + text.strip().lower()).encode()
    ).hexdigest(), 16)
    return h % mod


def normalize_counts(sent_csr):
    data = sent_csr.data.astype(np.float32)
    norm = data / (1.0 + data)
    return csr_matrix(
        (norm, sent_csr.indices, sent_csr.indptr),
        shape=sent_csr.shape,
    )


# ============================================================================
# TASK=supervised: Stage 03a
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


def _load_stage02_cohort(preloaded=None):
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
        "uid_layout_fingerprint": _uid_layout_fingerprint(selected, user_n_sents),
        "sentence_source_fingerprint": _cohort_fingerprint(selected, uid_to_sents),
    }
    return selected, user_n_sents, manifest


def _validate_cache_manifest(meta, expected_manifest):
    if not isinstance(meta, dict):
        raise ValueError("pcfg cache meta must be an object")
    required = {**expected_manifest, "cache_schema_version": 2,
                "spacy_model": SPACY_MODEL}
    for key, expected_value in required.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"pcfg cache manifest mismatch at {key}: "
                f"cached={meta.get(key)!r}, current={expected_value!r}; "
                f"rerun stage_cache()")
    if meta.get("vocab_size", 0) <= 0:
        raise ValueError("pcfg cache vocabulary is empty")


def _validate_cache_arrays(cache, expected_manifest):
    sent_csr = cache["sent_csr"]
    counts = cache["counts"]
    uid_list = cache["uid_list"]
    user_n_sents = cache["user_n_sents"]
    meta = cache["meta"]
    _validate_cache_manifest(meta, expected_manifest)
    if uid_list != cache["expected_uid_list"]:
        raise ValueError("pcfg cache UID order/content differs from Stage 02")
    if user_n_sents != cache["expected_user_n_sents"]:
        raise ValueError("pcfg cache user_n_sents differs from Stage 02")
    expected_shape = (expected_manifest["n_total_sents"], meta["vocab_size"])
    if sent_csr.shape != expected_shape:
        raise ValueError(f"sent_vectors shape mismatch: {sent_csr.shape} != {expected_shape}")
    if counts.shape != (expected_manifest["n_users"], meta["vocab_size"]):
        raise ValueError(f"counts shape mismatch: {counts.shape}")
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
    missing = [str(p) for p in required_paths if not p.exists()]
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


def _validate_rules_cache(saved, expected, expected_n_users, expected_n_sents):
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


def _verify_cached_embeddings_cohort(emb_path, expected_n_sents, z_dim, stage_name):
    if not emb_path.exists():
        return
    arr = np.load(emb_path)
    if arr.ndim != 2 or arr.shape != (expected_n_sents, z_dim):
        raise ValueError(
            f"{stage_name} cache cohort mismatch: cached {emb_path.name} "
            f"shape={arr.shape} vs current cache expected "
            f"({expected_n_sents}, {z_dim}); delete {emb_path} and rerun")
    if not np.isfinite(arr).all():
        raise ValueError(f"{stage_name} cache contains NaN/Inf: {emb_path}")


def _npz_text(npz, key):
    if key not in npz:
        raise ValueError(f"strict3 artifact missing metadata field: {key}")
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"strict3 metadata field {key} must be scalar")
    return str(value.item())


def _validate_strict3_artifact(npz, cache):
    """Verify strict3 contract: 2-way 80/20 split, test=val alias."""
    uid_list = cache["uid_list"]
    user_n_sents = cache["user_n_sents"]
    meta = cache["meta"]
    n_total = int(sum(user_n_sents))
    if "uid_list" not in npz:
        raise ValueError("strict3 artifact missing uid_list")
    artifact_uids = [str(u) for u in np.asarray(npz["uid_list"]).tolist()]
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


def _validate_split_indices(npz, uid_list, user_n_sents):
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


def _load_rules_chunk(path, expected_manifest, start_user, end_user,
                      start_sent, end_sent):
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


def _iter_rules_chunks(rules_dir, selected, user_offsets, manifest):
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
    """Parse sentences → sparse rule-count matrix.

    Skip fast-path: if cache files (sent_vectors.npz, counts.npz,
    vocab.json, meta.json, cache_ready.json) all exist AND the manifest
    rebuilt from Stage 02 matches the cached meta.json, skip parse/vocab/CSR
    entirely. Falls back to full rebuild only on manifest drift.
    """
    import spacy
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    required_paths = [
        CACHE_DIR / "sent_vectors.npz", CACHE_DIR / "counts.npz",
        CACHE_DIR / "uid_list.json", CACHE_DIR / "user_n_sents.json",
        CACHE_DIR / "vocab.json", CACHE_DIR / "meta.json",
        CACHE_DIR / "cache_ready.json",
    ]
    if all(p.exists() for p in required_paths):
        _, _, expected_manifest = _load_stage02_cohort()
        cached_meta = json.loads((CACHE_DIR / "meta.json").read_text())
        # Compare the fields that drive cache validity. The Stage 02 cohort
        # manifest and the cached meta.json share the same fingerprints and
        # counts when the inputs are unchanged.
        match = (
            cached_meta.get("cache_schema_version") == 2
            and cached_meta.get("sentence_source_fingerprint")
                == expected_manifest["sentence_source_fingerprint"]
            and cached_meta.get("uid_layout_fingerprint")
                == expected_manifest["uid_layout_fingerprint"]
            and cached_meta.get("n_users") == expected_manifest["n_users"]
            and cached_meta.get("n_total_sents") == expected_manifest["n_total_sents"]
            and cached_meta.get("min_sents_per_user")
                == expected_manifest["min_sents_per_user"]
            and cached_meta.get("seed") == expected_manifest["seed"]
        )
        if match:
            log(f"skip stage_cache: 7 cache files present, "
                f"manifest match (n_users={cached_meta['n_users']}, "
                f"n_sents={cached_meta['n_total_sents']}, "
                f"vocab_size={cached_meta.get('vocab_size')})")
            return
        log("  stale stage_cache manifest; rebuilding")
        # 用户指令 2026-09-23: 跨 category 跑时, 每个 category 的 cache 在自己的子目录
        # (pcfg_cache_<subdir>/), 如果该子目录已存在但 cohort 不同 → 自动删 stale rules_chunks
        # + 所有 cache 文件, 重新跑 spaCy parse. 不 raise 让上层重试.
        import shutil
        for stale in required_paths + [CACHE_DIR / "rules_chunks"]:
            if stale.exists():
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 用户指令 2026-09-23: trf 实测 GPU 比 sm n_process=12 慢 7x, 回退 sm.
    # sm 默认 tok2vec+tagger+parser. 禁用 ner/textcat/lemmatizer (只留 dep/pos/head).
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

    rules_dir = CACHE_DIR / "rules_chunks"
    rules_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rules_dir / "manifest.json"
    chunk_paths = sorted(rules_dir.glob("chunk_*.pkl"))
    # 用户指令 2026-09-23: 跨 category 跑时, stale rules_chunks 已在上面自动删除,
    # 这里不再 raise, 直接 dump 新 manifest.
    if not manifest_path.exists():
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
    doc_counts = Counter()
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
    user_counters = [Counter() for _ in selected]
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
    _atomic_json_dump(meta, CACHE_DIR / "cache_ready.json")
    log(f"wrote → {CACHE_DIR}/")
    log(f"=== Total: {time.time()-t0:.0f}s ===")


def _two_way_split(uid_list, sent_cache_path, train_cut=8, mod=10, salt=None):
    """SHA1 hash 2-way split: train (b<8) / val (b>=8)."""
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


def stage_strict3(out_path=None):
    """2-way 80/20 supervised encoder (single train)."""
    t0 = time.time()
    canonical_npz = CACHE_DIR / "strict3_embeddings.npz"
    canonical_pt = CACHE_DIR / "strict3_encoder.pt"
    strict3_manifest_path = CACHE_DIR / "strict3_manifest.json"
    legacy_npz = CACHE_DIR / "strict_embeddings.npz"
    legacy_pt = CACHE_DIR / "strict_encoder.pt"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(STRICT_SEED)
    np.random.seed(STRICT_SEED)

    def to_dense_batch(indices):
        rows = sent_norm[indices]
        return torch.tensor(rows.toarray(), dtype=torch.float32, device=device)

    train_dict, val_dict = _two_way_split(uid_list, SENT_CACHE)
    profile_idx = np.asarray(
        sorted(i for v in train_dict.values() for i in v), dtype=np.int64)
    val_idx = np.asarray(
        sorted(i for v in val_dict.values() for i in v), dtype=np.int64)
    test_idx = val_idx
    log(f"profile={len(profile_idx)} (80%), val={len(val_idx)} (20%), "
        f"test=val_alias (no separate test split)")

    user_labels = np.zeros(n_sents, dtype=np.int64)
    off = 0
    for ui, n in enumerate(user_n_sents):
        user_labels[off:off + n] = ui
        off += n

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
    n_train = len(profile_idx)
    if n_train <= 0 or len(val_idx) <= 0:
        raise ValueError("strict3 requires non-empty train and validation splits")
    for epoch in range(1, SUP_EPOCHS + 1):
        model.train()
        perm_epoch = np.random.permutation(profile_idx)
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
            for i in range(0, len(val_idx), SUP_BATCH_SIZE):
                batch_idx = val_idx[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(batch_idx)
                _, logits = model(xb)
                yb = torch.tensor(user_labels[batch_idx],
                                  dtype=torch.long, device=device)
                v_correct += int((logits.argmax(dim=1) == yb).sum().detach())
        v_acc = v_correct / max(len(val_idx), 1)
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

    def encode_all(indices):
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
    z_test = z_val
    log(f"encoded: z_profile {z_profile.shape}, z_val {z_val.shape}, "
        f"z_test = z_val (alias, no separate test split)")

    all_idx = np.arange(n_sents, dtype=np.int64)
    z_all = encode_all(all_idx)
    cohort_fingerprint = cache["meta"]["sentence_source_fingerprint"]
    uid_layout_fingerprint = cache["meta"]["uid_layout_fingerprint"]
    vocab_fingerprint = cache["meta"]["vocab_fingerprint"]
    _atomic_save_npy(CACHE_DIR / "supervised_embeddings.npy", z_all)
    log(f"encoded all: z_all {z_all.shape} → supervised_embeddings.npy")
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


def main_supervised():
    global N_USERS_LIMIT, SUP_EPOCHS, STRICT_PATIENCE, _EFFECTIVE_TRAIN_LIMIT
    t_total = time.time()
    log("=== TASK=supervised: cache + strict3 (2-way 80/20) ===")
    if SMOKE:
        _EFFECTIVE_TRAIN_LIMIT = N_USERS_LIMIT_SMOKE
        log(f"  [SMOKE MODE] training will use: "
            f"train_limit={_EFFECTIVE_TRAIN_LIMIT}, "
            f"SUP_EPOCHS={SMOKE_EPOCHS}, STRICT_PATIENCE={SMOKE_PATIENCE}")
        orig_epochs = SUP_EPOCHS
        orig_patience = STRICT_PATIENCE
        SUP_EPOCHS = SMOKE_EPOCHS
        STRICT_PATIENCE = SMOKE_PATIENCE

    stage_cache()
    stage_strict3()

    if SMOKE:
        SUP_EPOCHS = orig_epochs
        STRICT_PATIENCE = orig_patience
        _EFFECTIVE_TRAIN_LIMIT = None
        log(f"  [SMOKE MODE] restored training globals")

    log(f"\n=== ALL DONE ({time.time()-t_total:.0f}s) ===")


# ============================================================================
# TASK=cohort3: Stage 03b (real parent_asin cohort3 MLP)
# ============================================================================

class StyleMLP(nn.Module):
    """Public StyleMLP for downstream Stage 04/05/07/08 reuse."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COHORT3_SVD_DIM, COHORT3_MLP_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(COHORT3_MLP_HIDDEN, COHORT3_MLP_OUT),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def _cohort3_clean_html(text):
    if not text:
        return text
    previous = None
    while text != previous:
        previous = text
        text = html.unescape(text)
    text = _COHORT3_HTML_TAG_RE.sub(" ", text)
    text = _COHORT3_HTML_OTHER_RE.sub(" ", text)
    text = _COHORT3_MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _cohort3_multi_positive_nce(anchor, positives, negatives):
    pos_logits = torch.einsum("bd,bpd->bp", anchor, positives) / COHORT3_TEMPERATURE
    neg_logits = torch.einsum("bd,bnd->bn", anchor, negatives) / COHORT3_TEMPERATURE
    all_logits = torch.cat([pos_logits, neg_logits], dim=1)
    return -(torch.logsumexp(pos_logits, dim=1)
             - torch.logsumexp(all_logits, dim=1)).mean()


def _cohort3_load_real_pair_rows(uid_list, user_n_sents):
    uid_to_idx = {uid: i for i, uid in enumerate(uid_list)}
    row_starts = np.zeros(len(user_n_sents) + 1, dtype=np.int64)
    row_starts[1:] = np.cumsum(user_n_sents)
    pair_local = defaultdict(list)
    observed_counts = np.zeros(len(uid_list), dtype=np.int64)
    n_reviews = 0
    n_sentences = 0
    skipped_external_reviews = 0
    external_users = set()
    t0 = time.time()
    with open(RAW_DATA, "r") as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            asin = rec.get("parent_asin")
            if not asin:
                raise ValueError("Required parent_asin missing")
            u = uid_to_idx.get(uid)
            if u is None:
                skipped_external_reviews += 1
                external_users.add(uid)
                continue
            text = _cohort3_clean_html(rec.get("text", ""))
            sentences = [s.strip() for s in _COHORT3_SENT_SPLIT.split(text) if s.strip()]
            start = int(observed_counts[u])
            rows = np.arange(row_starts[u] + start,
                             row_starts[u] + start + len(sentences), dtype=np.int64)
            pair_local[(u, asin)].extend(rows.tolist())
            observed_counts[u] += len(sentences)
            n_reviews += 1
            n_sentences += len(sentences)
            if n_reviews % 500000 == 0:
                log(f"  raw reviews={n_reviews} pairs={len(pair_local)} elapsed={time.time()-t0:.1f}s")
    expected = np.asarray(user_n_sents, dtype=np.int64)
    if not np.array_equal(observed_counts, expected):
        bad = np.flatnonzero(observed_counts != expected)
        raise ValueError(f"Sentence row alignment failed for {len(bad)} users")
    log(f"  raw aligned: reviews={n_reviews} sentences={n_sentences} pairs={len(pair_local)} "
        f"skipped_external_reviews={skipped_external_reviews} external_users={len(external_users)}")
    return pair_local, row_starts


def _cohort3_build_cohorts(pair_local):
    qualified_by_asin = defaultdict(list)
    for (u, asin), rows in pair_local.items():
        if len(rows) >= COHORT3_N_PAIR_SENTENCES:
            qualified_by_asin[asin].append(u)
    qualified_by_asin = {
        asin: sorted(set(users))
        for asin, users in qualified_by_asin.items()
        if len(set(users)) >= COHORT3_COHORT_USERS
    }
    selected = {
        asin: users[:COHORT3_COHORT_USERS]
        for asin, users in sorted(qualified_by_asin.items())
    }
    if not selected:
        raise ValueError("No real ASIN has three qualified users")
    return selected


def _cohort3_make_split(pair_local, cohorts):
    rng = np.random.default_rng(SEED)
    train_rows, val_rows, test_rows = [], [], []
    uid_train, uid_val, uid_test = [], [], []
    asin_train, asin_val, asin_test = [], [], []
    pair_train_rows, pair_val_rows, pair_test_rows = {}, {}, {}
    for asin, users in cohorts.items():
        for u in users:
            rows = np.asarray(pair_local[(u, asin)], dtype=np.int64)
            chosen = rng.choice(rows, size=COHORT3_N_PAIR_SENTENCES, replace=False)
            tr = chosen[:COHORT3_N_TRAIN]
            va = chosen[COHORT3_N_TRAIN:COHORT3_N_TRAIN + COHORT3_N_VAL]
            te = chosen[COHORT3_N_TRAIN + COHORT3_N_VAL:]
            pair_train_rows[(u, asin)] = tr
            pair_val_rows[(u, asin)] = va
            pair_test_rows[(u, asin)] = te
            train_rows.extend(tr.tolist()); uid_train.extend([u] * COHORT3_N_TRAIN); asin_train.extend([asin] * COHORT3_N_TRAIN)
            val_rows.extend(va.tolist()); uid_val.extend([u] * COHORT3_N_VAL); asin_val.extend([asin] * COHORT3_N_VAL)
            test_rows.extend(te.tolist()); uid_test.extend([u] * COHORT3_N_TEST); asin_test.extend([asin] * COHORT3_N_TEST)
    return (
        np.asarray(train_rows, dtype=np.int64),
        np.asarray(val_rows, dtype=np.int64),
        np.asarray(test_rows, dtype=np.int64),
        np.asarray(uid_train, dtype=np.int64),
        np.asarray(uid_val, dtype=np.int64),
        np.asarray(uid_test, dtype=np.int64),
        np.asarray(asin_train, dtype=object),
        np.asarray(asin_val, dtype=object),
        np.asarray(asin_test, dtype=object),
        pair_train_rows, pair_val_rows, pair_test_rows,
    )


def _cohort3_make_schedules(train_rows, uid_train, asin_train,
                            pair_train_rows, cohorts, n_epochs):
    row_to_local = {int(r): i for i, r in enumerate(train_rows)}
    pair_local = {
        (u, a): np.asarray([row_to_local[int(r)] for r in rs], dtype=np.int64)
        for (u, a), rs in pair_train_rows.items()
    }
    neg_local = np.empty((len(train_rows), COHORT3_N_NEG), dtype=np.int64)
    pos_local = np.empty((n_epochs, len(train_rows), COHORT3_N_POS), dtype=np.int64)
    rng = np.random.default_rng(SEED + 1)
    asin_to_users = {a: users for a, users in cohorts.items()}
    for i in range(len(train_rows)):
        u = int(uid_train[i])
        a = str(asin_train[i])
        own = pair_local[(u, a)]
        other_users = [v for v in asin_to_users[a] if v != u]
        if len(other_users) != COHORT3_COHORT_USERS - 1:
            raise ValueError(f"ASIN {a} cohort is not exactly {COHORT3_COHORT_USERS}")
        neg_parts = []
        for v in other_users:
            rows = pair_local[(v, a)]
            neg_parts.append(rng.choice(rows, size=COHORT3_K_PER_NEG_USER, replace=False))
        neg_local[i] = np.concatenate(neg_parts)
        for ep in range(n_epochs):
            candidates = own[own != i]
            if len(candidates) < COHORT3_N_POS:
                raise ValueError(f"Pair {(u, a)} has insufficient positive candidates")
            pos_local[ep, i] = rng.choice(candidates, size=COHORT3_N_POS, replace=False)
    return pos_local, neg_local


@torch.no_grad()
def _cohort3_eval_metrics(enc, z_all_t, test_rows, uid_test, asin_test,
                          pair_train_rows, cohorts):
    z_test = enc(z_all_t[test_rows]).cpu().numpy()
    pair_proto = {}
    for pair, rows in pair_train_rows.items():
        emb = enc(z_all_t[np.asarray(rows, dtype=np.int64)]).cpu().numpy()
        m = emb.mean(axis=0)
        pair_proto[pair] = m / np.linalg.norm(m)
    ranks = []
    asin_to_users = cohorts
    for i, (u, a_obj) in enumerate(zip(uid_test, asin_test)):
        a = str(a_obj)
        u = int(u)
        users = asin_to_users[a]
        peers = [v for v in users if v != u]
        score_true = float(z_test[i] @ pair_proto[(u, a)])
        peer_scores = [float(z_test[i] @ pair_proto[(v, a)]) for v in peers]
        ranks.append(sum(s >= score_true for s in peer_scores))
    ranks = np.asarray(ranks)
    return {
        "n_evaluated": int(len(ranks)),
        "mean_rank": float(ranks.mean()),
        "top1_acc": float(np.mean(ranks == 0)),
        "top3_acc": float(np.mean(ranks < 3)),
        "random_baseline_top1": 1.0 / COHORT3_COHORT_USERS,
    }


def main_cohort3():
    """Train multi-positive InfoNCE on real (user, ASIN) cohorts."""
    log(f"device={DEVICE}, real cohorts exact={COHORT3_COHORT_USERS}, "
        f"k={COHORT3_K_PER_NEG_USER}, "
        f"split={COHORT3_N_TRAIN}/{COHORT3_N_VAL}/{COHORT3_N_TEST}")
    with np.load(CACHE_DIR / "svd_components.npz") as npz:
        Vt = np.asarray(npz["Vt"], dtype=np.float32)
    d = np.load(CACHE_DIR / "sent_vectors.npz", allow_pickle=True)
    X = sp.csr_matrix(
        (d["data"].astype(np.float32), d["indices"].astype(np.int32),
         d["indptr"].astype(np.int32)),
        shape=tuple(d["shape"]),
    )
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    with open(CACHE_DIR / "uid_list.json") as f:
        uid_list = json.load(f)

    pair_local, _ = _cohort3_load_real_pair_rows(uid_list, user_n_sents)
    cohorts = _cohort3_build_cohorts(pair_local)
    if COHORT3_SMOKE:
        cohorts = dict(list(cohorts.items())[:COHORT3_SMOKE_N_ASINS])
        log(f"COHORT3_SMOKE=True → ASINs={len(cohorts)}")
    log(f"  selected exact-3 cohorts: ASINs={len(cohorts)} "
        f"users={len(set(u for us in cohorts.values() for u in us))}")
    split = _cohort3_make_split(pair_local, cohorts)
    (train_rows, val_rows, test_rows, uid_train, uid_val, uid_test,
     asin_train, asin_val, asin_test, pair_train, pair_val, pair_test) = split
    log(f"  split rows: train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)}")
    pos_local, neg_local = _cohort3_make_schedules(
        train_rows, uid_train, asin_train, pair_train, cohorts, COHORT3_N_EPOCHS)

    z_all = (X @ Vt.T).astype(np.float32)
    del X
    z_all_t = torch.from_numpy(z_all).to(DEVICE)
    z_train = z_all_t[train_rows]
    enc = StyleMLP().to(DEVICE)
    enc = torch.compile(enc)
    opt = torch.optim.Adam(enc.parameters(), lr=COHORT3_LR)
    for ep in range(COHORT3_N_EPOCHS):
        perm = np.random.default_rng(SEED + ep).permutation(len(train_rows))
        total = 0.0
        for s in range(0, len(train_rows), COHORT3_BATCH_SIZE):
            idx = perm[s:s + COHORT3_BATCH_SIZE]
            anchor = enc(z_train[idx])
            pos = enc(z_train[pos_local[ep, idx].reshape(-1)]).view(
                len(idx), COHORT3_N_POS, COHORT3_MLP_OUT)
            neg = enc(z_train[neg_local[idx].reshape(-1)]).view(
                len(idx), COHORT3_N_NEG, COHORT3_MLP_OUT)
            loss = _cohort3_multi_positive_nce(anchor, pos, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        log(f"  epoch {ep+1}/{COHORT3_N_EPOCHS} "
            f"loss={total / ((len(train_rows) + COHORT3_BATCH_SIZE - 1) // COHORT3_BATCH_SIZE):.4f}")

    result = _cohort3_eval_metrics(enc, z_all_t, test_rows, uid_test, asin_test,
                                   pair_train, cohorts)
    log(f"  test n={result['n_evaluated']} top1={result['top1_acc']*100:.2f}% "
        f"mean_rank={result['mean_rank']:.3f}")
    out = OUT_DIR / "real_cohort3_training_summary.json"
    with open(out, "w") as f:
        json.dump({
            "config": {
                "cohort_users": COHORT3_COHORT_USERS,
                "k_per_negative_user": COHORT3_K_PER_NEG_USER,
                "pair_sentences": COHORT3_N_PAIR_SENTENCES,
                "split": f"{COHORT3_N_TRAIN}/{COHORT3_N_VAL}/{COHORT3_N_TEST}",
                "real_parent_asin": True,
            },
            "selected_asins": len(cohorts),
            "selected_users": len(set(u for us in cohorts.values() for u in us)),
            "result": result,
        }, f, indent=2)
    log(f"DONE wrote {out}")

    underlying = enc._orig_mod if hasattr(enc, "_orig_mod") else enc
    underlying.eval()
    ckpt = OUT_DIR / (
        f"cohort{COHORT3_COHORT_USERS}_mlp{COHORT3_MLP_OUT}_"
        f"{COHORT3_N_PAIR_SENTENCES}_{COHORT3_N_EPOCHS}ep.pt"
    )
    torch.save(underlying.state_dict(), ckpt)
    log(f"DONE wrote encoder weights {ckpt}")

    trained_uids_path = OUT_DIR / "cohort3_trained_uids.json"
    with open(trained_uids_path, "w") as f:
        json.dump(sorted({uid_list[int(u)] for u in uid_train}), f)
    log(f"DONE wrote trained uid whitelist {trained_uids_path} "
        f"({len({uid_list[int(u)] for u in uid_train})} unique uids)")


# ============================================================================
# Entry point
# ============================================================================

def main():
    """用户指令 2026-09-23: 串行运行 3 个 category (Baby / Musical_Instruments / Video_Games).

    每个 category 重新绑定全局 SENT_CACHE / RAW_DATA / OUT_DIR 为 <REPO_ROOT>/result/03_spacy_encode/<subdir>/,
    然后调原有 main_supervised() 或 main_cohort3()。所有 cache / encoder / strict3 产物按 category
    写到子目录, 不互相覆盖。

    Task 切换仍走 TASK 环境变量 (supervised | cohort3)。
    """
    task = os.environ.get("TASK", "supervised").lower()
    if task not in ("cohort3", "supervised"):
        raise ValueError(f"unknown TASK={task!r}; expected 'supervised' or 'cohort3'")

    # 备份默认 (Baby) 路径, 循环结束后恢复.
    global SENT_CACHE, RAW_DATA, OUT_DIR, CACHE_DIR
    saved = (SENT_CACHE, RAW_DATA, OUT_DIR, CACHE_DIR)
    base_out = REPO_ROOT / "result/03_spacy_encode"
    base_cache = Path("/home/wlia0047/hj82_scratch2/wenyu")

    for category, raw_data_path, sent_cache_path, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        SENT_CACHE = sent_cache_path
        RAW_DATA = raw_data_path
        OUT_DIR = base_out / subdir
        # 用户指令 2026-09-23: cache per-category, 每个 category 独立 pcfg_cache_<subdir>.
        CACHE_DIR = base_cache / f"pcfg_cache_{subdir}"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if task == "cohort3":
                main_cohort3()
            else:
                main_supervised()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise

    # 恢复默认 (为 import 后的 Stage 04/05/08/09 兼容, 它们只读 Baby 路径).
    SENT_CACHE, RAW_DATA, OUT_DIR, CACHE_DIR = saved


if __name__ == "__main__":
    main()
