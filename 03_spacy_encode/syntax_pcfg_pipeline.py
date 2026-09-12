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

# Stage: cache
N_USERS_LIMIT = None        # None = 全部 eligible 用户 (raw 3.39M → h≥30 = 64,996)
MIN_SENTS_PER_USER = 30
MIN_DOC_FREQ = 2              # vocab filter: rule must appear in ≥ N docs

# Stage: strict (SUP_* 共享: supervised encoder V→256→32)
SUP_EPOCHS = 80
SUP_BATCH_SIZE = 2048
SUP_LR = 5e-4
SUP_Z_DIM = 32
SUP_HIDDEN = (256,)
SUP_DROPOUT = 0.5
SUP_RULE_DROPOUT = 0.1     # 关键: 训练时随机 mask rule columns (profile-only 减半, 0.3 太狠 → 不收敛)
SUP_WEIGHT_DECAY = 1e-2
SUP_LABEL_SMOOTHING = 0.1
SUP_LOG_EVERY = 5
SUP_TRAIN_FRAC = 0.8       # per-user train/val split (hash-based)

# Stage: strict (no-leakage supervised encoder + 早停)
STRICT_VAL_FRAC = 0.15      # 从 profile 中再留 15% 做早停 validation
STRICT_SEED = 42
STRICT_PATIENCE = 10        # val_acc plateau N epochs → early stop


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _verify_cached_embeddings_cohort(emb_path: Path,
                                      expected_n_sents: int,
                                      z_dim: int,
                                      stage_name: str) -> None:
    """验证 cached embeddings 是否匹配当前 cache cohort。

    Per Rule 7: 不静默。stale cache(不同 n_sents / z_dim)直接 raise,
    让用户显式决定删除旧 cache 还是放弃当前 run。
    """
    if not emb_path.exists():
        return  # 已在 caller 检查过 exists, 这里只验 shape
    arr = np.load(emb_path)
    if arr.ndim != 2 or arr.shape != (expected_n_sents, z_dim):
        raise ValueError(
            f"{stage_name} cache cohort mismatch: cached {emb_path.name} "
            f"shape={arr.shape} vs current cache expected "
            f"({expected_n_sents}, {z_dim}); delete {emb_path} and rerun")


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


def load_cache():
    if not (CACHE_DIR / "sent_vectors.npz").exists():
        raise FileNotFoundError(
            f"cache not found at {CACHE_DIR}, run stage_cache() or main_pipeline() first")
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
    return {
        "sent_csr": sent_csr, "counts": counts,
        "uid_list": uid_list, "user_n_sents": user_n_sents,
        "vocab": vocab, "meta": meta,
    }


# ============================================================================
# Stage: cache (parse + sparse matrix persist)
# ============================================================================

def stage_cache():
    """en_core_web_sm parse sentences → sparse rule-count matrix。"""
    t0 = time.time()
    # torch.set_default_tensor_type deprecated in PyTorch 2.1+ and conflicts
    # with spaCy CPU/GPU routing; let downstream code manage device explicitly.
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    log(f"loaded {SPACY_MODEL}")

    with open(SENT_CACHE, "rb") as f:
        uid_to_sents = pickle.load(f)
    counts = {u: len(s) for u, s in uid_to_sents.items()}
    eligible = sorted([u for u, c in counts.items()
                       if c >= MIN_SENTS_PER_USER])
    rng = np.random.default_rng(SEED)
    rng.shuffle(eligible)
    if N_USERS_LIMIT is not None:
        eligible = eligible[:N_USERS_LIMIT]
    selected = eligible
    log(f"users={len(selected)}, min_sents={MIN_SENTS_PER_USER}")

    all_sents = []
    user_n_sents = []
    for u in selected:
        sents = list(uid_to_sents[u])
        all_sents.extend(sents)
        user_n_sents.append(len(sents))
    n_total = len(all_sents)
    log(f"total sents: {n_total}")

    t_parse = time.time()
    all_doc_rules: list[list[str]] = []
    sent_offset = 0
    rules_cache = CACHE_DIR / f"all_doc_rules_{len(selected)}.pkl"
    # Resume from partial cache if exists
    need_parse = True
    if rules_cache.exists():
        cache_t = time.time()
        with open(rules_cache, "rb") as f:
            saved = pickle.load(f)
        cached_rules = saved["rules"]
        cached_offset = saved["offset"]
        if len(cached_rules) >= n_total:
            # Cache 已经覆盖当前 cohort 全量, 跳过 parse loop 避免 30s+ 冗余 pickle 写盘
            all_doc_rules = cached_rules[:n_total]
            log(f"  full cache hit: {n_total} sents ready "
                f"({time.time()-cache_t:.1f}s), skip parse loop")
            need_parse = False
        else:
            all_doc_rules = cached_rules
            sent_offset = cached_offset
            log(f"  loaded partial parse cache: {sent_offset}/{n_total} sents "
                f"({time.time()-cache_t:.1f}s)")
    for ci in range(0, len(selected), CHUNK_USERS):
        if not need_parse:
            break
        chunk_users = selected[ci:ci+CHUNK_USERS]
        chunk_n = sum(user_n_sents[ci + j] for j in range(len(chunk_users)))
        chunk_sents = all_sents[sent_offset:sent_offset + chunk_n]
        if chunk_n == 0:
            continue
        log(f"  chunk {ci}-{ci+len(chunk_users)}: {chunk_n} sents ...")
        rules_flat = [extract_struct_rules(d)
                      for d in nlp.pipe(chunk_sents,
                                        batch_size=PARSE_BATCH_SIZE,
                                        n_process=PARSE_N_PROCESS)]
        all_doc_rules.extend(rules_flat)
        sent_offset += chunk_n
        del rules_flat
        # Save partial cache every chunk (resume-safe)
        with open(rules_cache, "wb") as f:
            pickle.dump({"rules": all_doc_rules, "offset": sent_offset}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        log(f"    {time.time()-t_parse:.0f}s elapsed, "
            f"{sent_offset}/{n_total}")
    log(f"parse done ({time.time()-t_parse:.0f}s)")

    log("building vocab (with min_doc_freq filter) ...")
    doc_counts: Counter = Counter()
    for rs in all_doc_rules:
        for r in set(rs):  # per-doc count, not per-occurrence
            doc_counts[r] += 1
    vocab_full = sorted(doc_counts.keys())
    vocab = [r for r in vocab_full if doc_counts[r] >= MIN_DOC_FREQ]
    V = len(vocab)
    rule_to_id = {r: i for i, r in enumerate(vocab)}
    log(f"vocab: {V} (filtered from {len(vocab_full)}, "
        f"min_doc_freq={MIN_DOC_FREQ})")

    log("building per-sentence CSR (filter rare rules) ...")
    sent_rows, sent_cols, sent_data = [], [], []
    for si, rs in enumerate(all_doc_rules):
        c = Counter(rs)
        for r, cnt in c.items():
            if r not in rule_to_id:
                continue
            sent_rows.append(si)
            sent_cols.append(rule_to_id[r])
            sent_data.append(cnt)
    sent_sparse = csr_matrix(
        (np.asarray(sent_data, dtype=np.int16),
         (np.asarray(sent_rows, dtype=np.int32),
          np.asarray(sent_cols, dtype=np.int32))),
        shape=(len(all_doc_rules), V),
    )
    log(f"per-sent CSR: {sent_sparse.shape}, density "
        f"{sent_sparse.nnz/(len(all_doc_rules)*V)*100:.3f}%")

    log("building per-user CSR (filter rare rules) ...")
    user_counters: list[Counter] = [Counter() for _ in selected]
    sent_idx = 0
    for ui, n in enumerate(user_n_sents):
        for rs in all_doc_rules[sent_idx:sent_idx + n]:
            filtered = [r for r in rs if r in rule_to_id]
            user_counters[ui].update(filtered)
        sent_idx += n
    del all_doc_rules

    rows, cols, data = [], [], []
    for ui, cnt in enumerate(user_counters):
        for r, c in cnt.items():
            if r not in rule_to_id:
                continue
            rows.append(ui)
            cols.append(rule_to_id[r])
            data.append(c)
    counts_sparse = csr_matrix(
        (np.asarray(data, dtype=np.int32),
         (np.asarray(rows, dtype=np.int32),
          np.asarray(cols, dtype=np.int32))),
        shape=(len(selected), V),
    )
    log(f"per-user CSR: {counts_sparse.shape}, density "
        f"{counts_sparse.nnz/(len(selected)*V)*100:.3f}%")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(CACHE_DIR / "counts.npz", counts_sparse)
    save_npz(CACHE_DIR / "sent_vectors.npz", sent_sparse)
    with open(CACHE_DIR / "uid_list.json", "w") as f:
        json.dump(selected, f)
    with open(CACHE_DIR / "user_n_sents.json", "w") as f:
        json.dump(user_n_sents, f)
    with open(CACHE_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f)
    meta = {
        "mode": ("LIMITED" if N_USERS_LIMIT else "FULL"),
        "n_users": len(selected),
        "n_users_limit": N_USERS_LIMIT,
        "n_total_sents": int(sent_sparse.shape[0]),
        "vocab_size": V,
        "min_sents_per_user": MIN_SENTS_PER_USER,
        "seed": SEED,
        "spacy_model": SPACY_MODEL,
        "user_csr_nnz": int(counts_sparse.nnz),
        "user_csr_density_pct": round(
            counts_sparse.nnz / (len(selected) * V) * 100, 4),
        "sent_csr_nnz": int(sent_sparse.nnz),
        "sent_csr_density_pct": round(
            sent_sparse.nnz / (sent_sparse.shape[0] * V) * 100, 4),
    }
    with open(CACHE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
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


def _three_way_split(uid_list, sent_cache_path,
                     profile_cut=5, val_cut=7, mod=10, salt: str = None):
    """SHA1 hash 3-way split: profile (b<5) / val (5≤b<7) / test (7≤b<10)。

    默认 50/20/30 (profile/val/test)。三个桶互斥且穷尽,encoder 只见过
    profile (内部 15% 早停),val 完全未参与训练,test 完全未参与 encoder 训
    练或阈值选择。
    """
    with open(sent_cache_path, "rb") as f:
        uid_to_sents = pickle.load(f)
    profile_dict = {u: [] for u in uid_list}
    val_dict = {u: [] for u in uid_list}
    test_dict = {u: [] for u in uid_list}
    sent_off = 0
    for ui, uid in enumerate(uid_list):
        sents = list(uid_to_sents[uid])
        for si, s in enumerate(sents):
            b = hash_bucket(s, mod=mod, salt=salt)
            if b < profile_cut:
                profile_dict[uid].append(sent_off + si)
            elif b < val_cut:
                val_dict[uid].append(sent_off + si)
            else:
                test_dict[uid].append(sent_off + si)
        sent_off += len(sents)
    return profile_dict, val_dict, test_dict


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
    """统一 3-way strict encoder — Stage 03/04 合并后唯一 encoder。

    协议 (与 stage_strict 完全同 encoder 架构, 唯一变化是 split 维度):
      1. SHA1 hash mod=10 三段 split → profile (b<5) / val (5≤b<7) / test (b≥7)
         = 50% / 20% / 30%
      2. encoder 只用 profile 训练 (内部 15% hash 做早停)
      3. freeze encoder, 编码 profile + val + test
      4. 用 strict3 encoder 再编码全部 n_sents 句子 → z_all (供 04_gaussian 消费)
      5. 输出 strict3_embeddings.npz + strict3_encoder.pt

    输出:
      pcfg_cache/strict3_embeddings.npz
        z_profile (n_p, 32) + z_val (n_v, 32) + z_test (n_t, 32)
        profile_idx / val_idx / test_idx (global sent indices)
        uid_list (cohort uids)
      pcfg_cache/strict3_encoder.pt (best ckpt)
      pcfg_cache/supervised_embeddings.npy (full z_all, 兼容旧 consumer)

    下游:
      04_gaussian/fit_per_user_gaussian.py 读 strict3_embeddings.npz
        → fit raw full Σ_u + ASIN cohort gates → user_gaussian_stats.json
      analysis/syntax_pcfg_adaptive_eval.py 读 strict3_embeddings.npz
        → τ sweep + held-out eval → syntax_pcfg_adaptive.json
    """
    t0 = time.time()
    canonical_npz = CACHE_DIR / "strict3_embeddings.npz"
    canonical_pt = CACHE_DIR / "strict3_encoder.pt"
    legacy_npz = CACHE_DIR / "strict_embeddings.npz"
    legacy_pt = CACHE_DIR / "strict_encoder.pt"
    # 兼容 legacy strict_embeddings.npz 已有 → 直接报错, 强制用户走 strict3
    if legacy_npz.exists() and not canonical_npz.exists():
        raise FileExistsError(
            f"found legacy {legacy_npz.name} but no strict3_embeddings.npz; "
            f"per unified-encoder migration (2026-09-11), strict3 is canonical. "
            f"Delete {legacy_npz.name} and rerun, or rename it if you "
            f"intentionally want to keep the 2-way split.")
    cached = [canonical_npz, canonical_pt,
              CACHE_DIR / "supervised_embeddings.npy"]
    missing = [p for p in cached if not p.exists()]
    if not missing:
        cache_meta = json.loads((CACHE_DIR / "meta.json").read_text())
        npz = np.load(canonical_npz)
        n_p = int(npz["z_profile"].shape[0])
        n_v = int(npz["z_val"].shape[0])
        n_t = int(npz["z_test"].shape[0])
        n_users_cached = int(len(npz["uid_list"]))
        if (n_p + n_v + n_t != cache_meta["n_total_sents"]
                or n_users_cached != cache_meta["n_users"]):
            raise ValueError(
                f"stage_strict3 cache cohort mismatch: cached "
                f"profile={n_p} val={n_v} test={n_t} users="
                f"{n_users_cached} vs current n_sents="
                f"{cache_meta['n_total_sents']} n_users="
                f"{cache_meta['n_users']}; delete {canonical_npz} and rerun")
        _verify_cached_embeddings_cohort(
            CACHE_DIR / "supervised_embeddings.npy",
            cache_meta["n_total_sents"], SUP_Z_DIM, "stage_strict3")
        log(f"skip stage_strict3: {len(cached)} cached output(s) exist, "
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
    log(f"strict3: V={V}, n_sents={n_sents}, n_users={n_users}")

    sent_norm = normalize_counts(sent_csr)

    def to_dense_batch(indices: np.ndarray) -> torch.Tensor:
        rows = sent_norm[indices]
        return torch.tensor(rows.toarray(), dtype=torch.float32,
                            device=device)

    # === Step 1: 3-way split (50/20/30) ===
    profile_dict, val_dict, test_dict = _three_way_split(
        uid_list, SENT_CACHE)
    profile_idx = np.asarray(
        sorted(i for v in profile_dict.values() for i in v),
        dtype=np.int64)
    val_idx = np.asarray(
        sorted(i for v in val_dict.values() for i in v), dtype=np.int64)
    test_idx = np.asarray(
        sorted(i for v in test_dict.values() for i in v), dtype=np.int64)
    log(f"profile={len(profile_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)} (50/20/30 hash)")

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
    perm = rng_local.permutation(len(profile_idx))
    n_val_inner = int(len(profile_idx) * STRICT_VAL_FRAC)
    profile_val_inner = profile_idx[perm[:n_val_inner]]
    profile_train_inner = profile_idx[perm[n_val_inner:]]
    log(f"encoder train: {len(profile_train_inner)} train + "
        f"{len(profile_val_inner)} inner val (来自 profile)")

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
    n_train = len(profile_train_inner)
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
    z_test = encode_all(test_idx)
    log(f"encoded: z_profile {z_profile.shape}, z_val {z_val.shape}, "
        f"z_test {z_test.shape}")

    # === Step 4: 用同一 encoder 编码全部 n_sents → z_all (给 04_gaussian 消费) ===
    all_idx = np.arange(n_sents, dtype=np.int64)
    z_all = encode_all(all_idx)
    np.save(CACHE_DIR / "supervised_embeddings.npy", z_all)
    log(f"encoded all (供 04_gaussian 消费): z_all {z_all.shape} → "
        f"supervised_embeddings.npy")

    # === Step 5: 持久化 ===
    torch.save({"model_state": best_state,
                "config": {"vocab_size": V, "z_dim": SUP_Z_DIM,
                           "hidden": list(SUP_HIDDEN), "dropout": SUP_DROPOUT,
                           "n_users": n_users}},
               canonical_pt)
    np.savez(canonical_npz,
             z_profile=z_profile, z_val=z_val, z_test=z_test,
             profile_idx=profile_idx, val_idx=val_idx, test_idx=test_idx,
             uid_list=np.asarray(uid_list))
    log(f"wrote → {canonical_npz.name} + {canonical_pt.name}")

    log(f"\n=== STRICT3 (3-way 50/20/30) SUMMARY "
        f"(z={SUP_Z_DIM}, N={n_users}) ===")
    log(f"  profile sents: {len(profile_idx)}")
    log(f"  val sents:     {len(val_idx)}")
    log(f"  test sents:    {len(test_idx)}")
    log(f"  z_all sents:   {n_sents}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")




# ============================================================================
# Main pipeline (一键串行运行所有 stage, 不依赖 MODE 调度)
# ============================================================================

def main_pipeline():
    """Canonical pipeline: 串行执行全部 stage (统一 encoder, 单次训练)。

    链路: cache → strict3 (3-way 50/20/30 split + CE encoder, 唯一一次训练)
    Gaussian fitting / adaptive eval 一律外迁:
      - 04_gaussian/fit_per_user_gaussian.py   per-user raw full Σ + ASIN cohort gates
      - analysis/syntax_pcfg_adaptive_eval.py  τ sweep + held-out eval (读 strict3)

    假设 cache 已有产物则可跳过 stage_cache() (resume);
    任意 stage 抛错会立即终止, 不做 fallback (per Rule 7)。
    """
    t_total = time.time()
    log(f"=== main_pipeline: canonical chain start ===")

    stage_cache()
    stage_strict3()  # 统一 3-way encoder,产出 strict3_embeddings.npz + supervised_embeddings.npy

    log(f"\n=== ALL DONE ({time.time()-t_total:.0f}s) ===")


if __name__ == "__main__":
    main_pipeline()