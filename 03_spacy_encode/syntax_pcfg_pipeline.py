"""PCFG dependency-rule → sentence embedding → user Gaussian → LOPO attribution.

一体化 pipeline (per 项目 Rule 18, 03_spacy_encode 只允许一个主脚本)。
主入口 main_pipeline() 一键串行运行所有 stage, 不再依赖 MODE 切换;
individual stage_*() 函数保留用于单 stage smoke / debug。

Canonical pipeline 链路:
  cache        parse sentences + 建 sparse rule-count matrix (per-user + per-sent)
  pcfg_attr    discrete PCFG log-likelihood baseline (独立 baseline, 不需 cache)
  strict       训监督 encoder V→256→32 (rule dropout 0.3, L2 1e-2, profile-only) →
                 supervised_embeddings.npy (给 gauss/attr/gauss_attr 消费) +
                 strict_embeddings.npz (no-leakage LOPO)
  gauss        fit per-user N(μ_u, Σ_u) on supervised z + signal_to_noise 诊断
  attr         LOPO attribution: argmax cos(z_q, μ_u)
  gauss_attr   3 种 attribution (cos / log N(z|μ,σ²) / Maha-pooled) 同 split 对比
  ablation     7 种 attribution + 用户方差分层, 验证 Σ 是否真捕获表达范围
  adaptive     3-way profile/val/test split, val 上扫 τ, test 一次性评估

(vae 作为 super 的备选 unsupervised encoder, 不在 canonical pipeline 中, 保留
stage_vae() 用于 ablation 对比; stage_super() 与 stage_multi_seed() 已删除)

输入:
  result/02_user_review_sentence_extract/uid_to_sentences.pkl

中间缓存:
  /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/
    counts.npz, sent_vectors.npz, uid_list.json, user_n_sents.json, vocab.json, meta.json
    sentence_embeddings.npy (vae), supervised_embeddings.npy (super)
    syntax_vae_model.pt, supervised_encoder.pt
    strict_encoder.pt, strict_embeddings.npz (strict)
    adaptive_encoder.pt, adaptive_embeddings.npz (adaptive)
    syntax_vae_train.json, supervised_train.json

输出 (result/03_spacy_encode/):
  pcfg_attribution_lopo.json          (pcfg_attr)
  syntax_pcfg_user_gaussian.json      (gauss)
  syntax_pcfg_attribution_supervised.json (attr)
  syntax_pcfg_gauss_attr.json         (gauss_attr)
  syntax_pcfg_strict_attr.json        (strict)
  syntax_pcfg_ablation.json           (ablation)
  syntax_pcfg_adaptive.json           (adaptive)

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
SPACY_MODEL = "en_core_web_sm"
PARSE_BATCH_SIZE = 256
CHUNK_USERS = 2000

# 通用
SEED = 42
HASH_SALT = "pcfg_lopo_v1"

# Stage: cache
N_USERS_LIMIT = 5000        # None = 全部 eligible
MIN_SENTS_PER_USER = 80
MIN_DOC_FREQ = 2              # vocab filter: rule must appear in ≥ N docs

# Stage: pcfg_attr (discrete PCFG)
N_TRAIN = 50; N_TEST = 30   # per user (random shuffle, 与 LOPO 不同)

# Stage: vae
VAE_EPOCHS = 80
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3
VAE_BETA = 0.0              # 0 = 纯 AE; >0 = β-VAE
VAE_Z_DIM = 32
VAE_HIDDEN = (512, 128)
VAE_DROPOUT = 0.1
VAE_WEIGHT_DECAY = 1e-5
VAE_LOG_EVERY = 5

# Stage: strict + adaptive (SUP_* 共享: supervised encoder V→256→32)
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

# Attribution hash split (与 Stage pcfg_attr 一致)
ATTR_TRAIN_FRAC = 0.5      # LOPO profile/test 各 50%

# Stage: gauss_attr
GAUSS_EPS = 1e-3           # diag σ² 正则化 (防止 0 维度 blow up log)
GAUSS_USE_POOLED_COV = True  # Mahalanobis 用 pooled Σ across all users

# Stage: strict (no-leakage supervised encoder + Gaussian attribution)
STRICT_VAL_FRAC = 0.15      # 从 profile 中再留 15% 做早停 validation
STRICT_SEED = 42
STRICT_PATIENCE = 10        # val_acc plateau N epochs → early stop

# Stage: ablation (distribution-vs-point)
ABLATION_EPS = 1e-3         # Σ 正则化 (diag & full)
ABLATION_FULL_SHRINK = 0.1  # full Σ 用 (1-α)Σ + α*trace(Σ)/D * I 收缩

# Stage: adaptive (3-way profile/val/test split, τ on val, eval on test)
ADAPTIVE_PROFILE_FRAC = 0.50   # encoder 训练 + Gaussian fit
ADAPTIVE_VAL_FRAC = 0.20       # threshold τ 选择 (encoder 没见过)
ADAPTIVE_TEST_FRAC = 0.30      # 最终 eval (encoder + threshold 都没见过)
# hash mod=10 分桶: b<5 profile, 5<=b<7 val, 7<=b<10 test
ADAPTIVE_SOFT_SCALE = 0.5      # soft mixture sigmoid 温度


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
    """en_core_web_trf parse sentences → sparse rule-count matrix。"""
    t0 = time.time()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    log(f"loaded {SPACY_MODEL}")

    with open(SENT_CACHE, "rb") as f:
        uid_to_sents = pickle.load(f)
    counts = {u: len([x for x in s if 5 < len(x.split()) < 50])
              for u, s in uid_to_sents.items()}
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
        sents = [s for s in uid_to_sents[u] if 5 < len(s.split()) < 50]
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
                      for d in nlp.pipe(chunk_sents, batch_size=PARSE_BATCH_SIZE)]
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
# Stage: pcfg_attr (discrete PCFG log-likelihood attribution, no embedding)
# ============================================================================

def stage_pcfg_attr():
    """原始 PCFG attribution: 用训练规则 log P 直接 argmax,不需要 embedding。

    100 用户随机 split: 50 train + 30 test per user。
    输出: result/03_spacy_encode/pcfg_attribution_lopo.json
    """
    t0 = time.time()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    nlp = spacy.load(SPACY_MODEL)

    with open(SENT_CACHE, "rb") as f:
        uid_to_sents = pickle.load(f)
    counts = {u: len([x for x in s if 5 < len(x.split()) < 50])
              for u, s in uid_to_sents.items()}
    eligible = sorted([u for u, c in counts.items()
                       if c >= MIN_SENTS_PER_USER])
    rng = np.random.default_rng(SEED)
    rng.shuffle(eligible)
    if N_USERS_LIMIT is not None:
        eligible = eligible[:N_USERS_LIMIT]
    selected = eligible[:100]
    log(f"pcfg_attr: {len(selected)} users, "
        f"random split {N_TRAIN} train + {N_TEST} test per user")

    def parse_batch(sents):
        return [extract_struct_rules(d)
                for d in nlp.pipe(sents, batch_size=PARSE_BATCH_SIZE)]

    np.random.seed(SEED)
    train_data = {u: [] for u in selected}
    test_data = {u: [] for u in selected}
    for u in selected:
        sents = [s for s in uid_to_sents[u] if 5 < len(s.split()) < 50]
        rng_local = np.random.default_rng(SEED)
        rng_local.shuffle(sents)
        train_data[u] = sents[:N_TRAIN]
        test_data[u] = sents[N_TRAIN:N_TRAIN + N_TEST]

    all_train = [s for u in selected for s in train_data[u]]
    all_test = [s for u in selected for s in test_data[u]]
    log(f"train={len(all_train)}, test={len(all_test)}")

    log("parsing train ...")
    train_rules_flat = parse_batch(all_train)
    log("parsing test ...")
    test_rules_flat = parse_batch(all_test)

    idx = 0
    train_rules = {}
    for u in selected:
        train_rules[u] = train_rules_flat[idx:idx + N_TRAIN]
        idx += N_TRAIN
    idx = 0
    test_rules = {}
    for u in selected:
        test_rules[u] = test_rules_flat[idx:idx + N_TEST]
        idx += N_TEST

    all_rules = set()
    for u in selected:
        for rs in train_rules[u]:
            all_rules.update(rs)
    V = len(all_rules)
    log(f"vocab: {V}")

    user_lp = {}
    for u in selected:
        cnt = Counter()
        for rs in train_rules[u]:
            cnt.update(rs)
        N = sum(cnt.values()) + V
        user_lp[u] = {r: np.log((cnt.get(r, 0) + 1) / N) for r in all_rules}

    correct = 0; total = 0
    results = {u: {"correct": 0, "total": 0} for u in selected}
    log_floor = np.log(1e-6)
    log_prior = -np.log(len(selected))
    for u in selected:
        for rs in test_rules[u]:
            if not rs:
                continue
            ll = {a: sum(user_lp[a].get(r, log_floor) for r in rs) + log_prior
                  for a in selected}
            pred = max(ll, key=ll.get)
            if pred == u:
                correct += 1; results[u]["correct"] += 1
            results[u]["total"] += 1; total += 1

    acc = correct / max(total, 1)
    chance = 1.0 / len(selected)
    log(f"\n=== PCFG discrete attribution N={len(selected)} ===")
    log(f"overall: {correct}/{total} = {acc*100:.2f}% "
        f"(chance {chance*100:.2f}%, lift {acc/chance:.2f}x)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pcfg_attribution_lopo.json", "w") as f:
        json.dump({
            "config": {"mode": "pcfg_attr", "n_users": len(selected),
                       "n_train": N_TRAIN, "n_test": N_TEST,
                       "split": "random_shuffle"},
            "no_leakage_text": False,
            "overall_acc": acc, "chance": chance,
            "lift": acc / chance, "vocab_size": V,
            "n_correct": correct, "n_total": total,
        }, f, indent=2)
    log(f"wrote → {OUT_DIR / 'pcfg_attribution_lopo.json'}")
    log(f"=== Total: {time.time()-t0:.0f}s ===")


# ============================================================================
# Stage: vae (AE/VAE 重建监督, β configurable)
# ============================================================================

class _SyntaxVAE(nn.Module):
    def __init__(self, vocab_size, z_dim, hidden, dropout):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(
            nn.Linear(vocab_size, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(h2, z_dim)
        self.fc_logvar = nn.Linear(h2, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, vocab_size),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def _vae_kl(mu, logvar):
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def stage_vae():
    """训练 AE (β=0) 或 β-VAE (β>0), 保存 encoder 输出到 embeddings.npy。"""
    t0 = time.time()
    cached = [
        CACHE_DIR / "sentence_embeddings.npy",
        CACHE_DIR / "syntax_vae_model.pt",
        CACHE_DIR / "syntax_vae_train.json",
    ]
    missing = [p for p in cached if not p.exists()]
    if not missing:
        cache_meta = json.loads((CACHE_DIR / "meta.json").read_text())
        _verify_cached_embeddings_cohort(
            CACHE_DIR / "sentence_embeddings.npy",
            cache_meta["n_total_sents"], VAE_Z_DIM, "stage_vae")
        log(f"skip stage_vae: {len(cached)} cached output(s) exist, "
            f"cohort match ({cache_meta['n_total_sents']} sents)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = load_cache()
    sent_csr = cache["sent_csr"]
    V = sent_csr.shape[1]
    n_sents = sent_csr.shape[0]
    log(f"vae: V={V}, n_sents={n_sents}, β={VAE_BETA}, z={VAE_Z_DIM}")

    sent_norm = normalize_counts(sent_csr)
    x_dense = torch.tensor(sent_norm.toarray(),
                           dtype=torch.float32, device=device)
    log(f"x_dense: {x_dense.shape}")

    model = _SyntaxVAE(V, VAE_Z_DIM, VAE_HIDDEN, VAE_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=VAE_LR,
                           weight_decay=VAE_WEIGHT_DECAY)

    log(f"training: epochs={VAE_EPOCHS}, batch={VAE_BATCH_SIZE}")
    history = []
    n = x_dense.shape[0]
    for epoch in range(1, VAE_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = ep_recon = ep_kl = 0.0
        n_batches = 0
        for i in range(0, n, VAE_BATCH_SIZE):
            idx = perm[i:i + VAE_BATCH_SIZE]
            xb = x_dense[idx]
            opt.zero_grad()
            logits, mu, logvar = model(xb)
            recon = F.binary_cross_entropy_with_logits(
                logits, xb, reduction="mean")
            kl = _vae_kl(mu, logvar) if VAE_BETA > 0 else torch.tensor(0.0)
            loss = recon + VAE_BETA * kl
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            ep_recon += float(recon.detach())
            ep_kl += float(kl.detach())
            n_batches += 1
        history.append({"epoch": epoch,
                        "loss": ep_loss / n_batches,
                        "recon": ep_recon / n_batches,
                        "kl": ep_kl / n_batches})
        if epoch == 1 or epoch % VAE_LOG_EVERY == 0 or epoch == VAE_EPOCHS:
            log(f"  epoch {epoch:>3}/{VAE_EPOCHS}  "
                f"loss={history[-1]['loss']:.4f}  "
                f"recon={history[-1]['recon']:.4f}  "
                f"kl={history[-1]['kl']:.4f}")

    model.eval()
    z_all = np.zeros((n_sents, VAE_Z_DIM), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_sents, VAE_BATCH_SIZE):
            xb = x_dense[i:i + VAE_BATCH_SIZE]
            mu, _ = model.encode(xb)
            z_all[i:i + xb.shape[0]] = mu.cpu().numpy()

    np.save(CACHE_DIR / "sentence_embeddings.npy", z_all)
    torch.save({"model_state": model.state_dict(),
                "config": {"vocab_size": V, "z_dim": VAE_Z_DIM,
                           "hidden": list(VAE_HIDDEN), "dropout": VAE_DROPOUT}},
               CACHE_DIR / "syntax_vae_model.pt")
    with open(CACHE_DIR / "syntax_vae_train.json", "w") as f:
        json.dump({"config": {"epochs": VAE_EPOCHS, "beta": VAE_BETA,
                              "z_dim": VAE_Z_DIM, "lr": VAE_LR,
                              "hidden": list(VAE_HIDDEN)},
                   "history": history,
                   "final_loss": history[-1]["loss"],
                   "n_params_M": n_params / 1e6}, f, indent=2)
    log(f"wrote → {CACHE_DIR}/sentence_embeddings.npy {z_all.shape}")
    log(f"=== Total: {time.time()-t0:.0f}s ===")


# ============================================================================
# Encoder class (shared by stage_strict + stage_adaptive; stage_super 已删除,
# stage_strict 兼任产出 supervised_embeddings.npy)
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
# Stage: gauss (per-user Gaussian fit on embeddings)
# ============================================================================

def stage_gauss(embed_path: Path = None,
                out_path: Path = None):
    """Per-user mean/var on embeddings → user_gaussian.json。"""
    t0 = time.time()
    if embed_path is None:
        # 默认优先 supervised (历史最佳)
        embed_path = (CACHE_DIR / "supervised_embeddings.npy"
                      if (CACHE_DIR / "supervised_embeddings.npy").exists()
                      else CACHE_DIR / "sentence_embeddings.npy")
    if out_path is None:
        out_path = OUT_DIR / "syntax_pcfg_user_gaussian.json"

    cache = load_cache()
    z_all = np.load(embed_path)
    z_dim = z_all.shape[1]
    user_n_sents = cache["user_n_sents"]
    uid_list = cache["uid_list"]
    n_users = len(uid_list)
    log(f"z_all: {z_all.shape}, users={n_users}, z_dim={z_dim}")

    out = {}
    offset = 0
    for ui, uid in enumerate(uid_list):
        n = user_n_sents[ui]
        z_u = z_all[offset:offset + n]
        offset += n
        mu = z_u.mean(axis=0)
        sigma2 = z_u.var(axis=0, ddof=1) if n >= 2 else np.zeros(z_dim)
        out[uid] = {
            "mu": mu.tolist(),
            "sigma2": sigma2.tolist(),
            "sigma_mean": float(sigma2.mean()),
            "sigma_min": float(sigma2.min()),
            "sigma_max": float(sigma2.max()),
            "n": n,
        }

    mus = np.array([out[u]["mu"] for u in uid_list])
    sigmas = np.array([out[u]["sigma2"] for u in uid_list])
    norms = np.linalg.norm(mus, axis=1)
    sample = np.random.default_rng(SEED).choice(
        uid_list, size=min(50, n_users), replace=False)
    idxs = [uid_list.index(u) for u in sample]
    diffs = [np.linalg.norm(mus[idxs[i]] - mus[idxs[j]])
             for i in range(len(idxs))
             for j in range(i + 1, len(idxs))]
    diffs = np.array(diffs)
    avg_sigma = np.sqrt(sigmas.mean(axis=1)).mean()
    log(f"mu norm median={np.median(norms):.3f}, "
        f"avg_pairwise_dist={diffs.mean():.3f}, "
        f"avg_sqrt(sigma2)={avg_sigma:.3f}, "
        f"SNR={diffs.mean()/avg_sigma:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"z_dim": z_dim, "n_users": n_users,
                       "embed_source": str(embed_path),
                       "total_sents": int(z_all.shape[0]),
                       "cache_meta": cache["meta"]},
            "diagnostics": {
                "mu_norm_median": float(np.median(norms)),
                "mu_norm_min": float(norms.min()),
                "mu_norm_max": float(norms.max()),
                "sigma2_median": float(np.median(sigmas)),
                "avg_pairwise_mu_dist": float(diffs.mean()),
                "avg_sqrt_sigma2": float(avg_sigma),
                "signal_to_noise": float(diffs.mean() / avg_sigma),
            },
            "users": out,
        }, f)
    log(f"wrote → {out_path}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


# ============================================================================
# Stage: attr (LOPO attribution: train profile mean → test argmax cosine)
# ============================================================================

def stage_attr(embed_path: Path = None,
               out_path: Path = None):
    """SHA1 hash 50/50 split, profile 算 mean(z), test argmax cos(z, μ_u)。"""
    t0 = time.time()
    if embed_path is None:
        embed_path = (CACHE_DIR / "supervised_embeddings.npy"
                      if (CACHE_DIR / "supervised_embeddings.npy").exists()
                      else CACHE_DIR / "sentence_embeddings.npy")
    if out_path is None:
        suffix = ("supervised" if "supervised_embeddings" in str(embed_path)
                  else "vae")
        out_path = OUT_DIR / f"syntax_pcfg_attribution_{suffix}.json"

    cache = load_cache()
    z_all = np.load(embed_path)
    user_n_sents = cache["user_n_sents"]
    uid_list = cache["uid_list"]
    n_users = len(uid_list)
    log(f"attr on {embed_path.name}: {z_all.shape}, users={n_users}")

    with open(SENT_CACHE, "rb") as f:
        uid_to_sents = pickle.load(f)

    profile_idx = {u: [] for u in uid_list}
    test_idx = {u: [] for u in uid_list}
    sent_off = 0
    for ui, uid in enumerate(uid_list):
        sents = [s for s in uid_to_sents[uid] if 5 < len(s.split()) < 50]
        for si, s in enumerate(sents):
            if hash_bucket(s, mod=2) == 0:
                profile_idx[uid].append(sent_off + si)
            else:
                test_idx[uid].append(sent_off + si)
        sent_off += len(sents)

    profile_mu = np.zeros((n_users, z_all.shape[1]), dtype=np.float64)
    for ui in range(n_users):
        idxs = profile_idx[uid_list[ui]]
        if idxs:
            profile_mu[ui] = z_all[idxs].mean(axis=0)
    profile_norm = profile_mu / (
        np.linalg.norm(profile_mu, axis=1, keepdims=True) + 1e-8)

    correct = 0; total = 0
    results = {uid: {"correct": 0, "total": 0} for uid in uid_list}
    for ui, uid in enumerate(uid_list):
        for idx in test_idx[uid]:
            z_q = z_all[idx]
            z_q_norm = z_q / (np.linalg.norm(z_q) + 1e-8)
            sim = profile_norm @ z_q_norm
            pred = int(np.argmax(sim))
            if pred == ui:
                correct += 1; results[uid]["correct"] += 1
            results[uid]["total"] += 1; total += 1

    acc = correct / max(total, 1)
    chance = 1.0 / n_users
    log(f"\n=== Embedding cosine attribution N={n_users} ({embed_path.name}) ===")
    log(f"overall: {correct}/{total} = {acc*100:.2f}% "
        f"(chance {chance*100:.2f}%, lift {acc/chance:.2f}x)")

    per_user_acc = sorted(
        [(results[u]["correct"] / max(results[u]["total"], 1), u)
         for u in uid_list if results[u]["total"] > 0])
    n = len(per_user_acc)
    log(f"per-user acc: median={per_user_acc[n//2][0]*100:.2f}% "
        f"p75={per_user_acc[3*n//4][0]*100:.2f}% "
        f"max={per_user_acc[-1][0]*100:.2f}%")
    n_above = sum(1 for a, _ in per_user_acc if a > chance)
    log(f"per-user acc > chance: {n_above}/{n_users}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"embed_source": str(embed_path),
                       "n_users": n_users, "z_dim": z_all.shape[1],
                       "split": "sha1_hash_50_50"},
            "no_leakage": True,
            "overall_acc": acc, "chance": chance, "lift": acc / chance,
            "n_correct": correct, "n_total": total,
            "n_above_chance": n_above,
            "per_user_median_acc": per_user_acc[n//2][0],
            "per_user_p75_acc": per_user_acc[3*n//4][0],
            "per_user_max_acc": per_user_acc[-1][0],
        }, f, indent=2)
    log(f"wrote → {out_path}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


# ============================================================================
# Stage: gauss_attr (Gaussian log-P + Mahalanobis on per-user N(μ, σ²))
# ============================================================================

def _profile_test_split(uid_list, cache, sent_cache_path):
    """SHA1 hash 50/50 split → profile_idx/test_idx per uid。

    与 stage_attr 用同一 salt, 保证 cosine / Gaussian / Mahalanobis 三种
    attribution 跑在同一 train/test 划分上 (公平对比)。
    """
    with open(sent_cache_path, "rb") as f:
        uid_to_sents = pickle.load(f)
    profile_idx = {u: [] for u in uid_list}
    test_idx = {u: [] for u in uid_list}
    sent_off = 0
    for ui, uid in enumerate(uid_list):
        sents = [s for s in uid_to_sents[uid] if 5 < len(s.split()) < 50]
        for si, s in enumerate(sents):
            if hash_bucket(s, mod=2) == 0:
                profile_idx[uid].append(sent_off + si)
            else:
                test_idx[uid].append(sent_off + si)
        sent_off += len(sents)
    return profile_idx, test_idx


def stage_gauss_attr(embed_path: Path = None,
                     out_path: Path = None):
    """LOPO attribution via per-user Gaussian on 32d supervised z。

    三种 attribution 在同一 50/50 hash split 上跑, 公平对比:
      1. cosine: argmax_u cos(z_q, μ_u)             (baseline)
      2. log_p_diag: argmax_u log N(z_q | μ_u, σ²_u_diag)  (核心)
      3. mahalanobis_pooled: argmax_u -(z_q-μ_u)^T Σ_pool^{-1} (z_q-μ_u)
                            Σ_pool = mean over users of diag(σ²) + eps I

    输出:
      result/03_spacy_encode/syntax_pcfg_gauss_attr.json
        { config, no_leakage, lifts: {cosine, log_p_diag, mahalanobis_pooled},
          per_method: { method: {acc, chance, lift, ...} } }
    """
    t0 = time.time()
    if embed_path is None:
        embed_path = (CACHE_DIR / "supervised_embeddings.npy"
                      if (CACHE_DIR / "supervised_embeddings.npy").exists()
                      else CACHE_DIR / "sentence_embeddings.npy")
    if out_path is None:
        out_path = OUT_DIR / "syntax_pcfg_gauss_attr.json"

    cache = load_cache()
    z_all = np.load(embed_path)
    z_dim = z_all.shape[1]
    n_sents = z_all.shape[0]
    user_n_sents = cache["user_n_sents"]
    uid_list = cache["uid_list"]
    n_users = len(uid_list)
    log(f"gauss_attr on {embed_path.name}: z={z_dim}, users={n_users}, "
        f"sents={n_sents}")

    profile_idx, test_idx = _profile_test_split(uid_list, cache, SENT_CACHE)
    profile_n = sum(len(v) for v in profile_idx.values())
    test_n = sum(len(v) for v in test_idx.values())
    log(f"profile={profile_n}, test={test_n} (50/50 hash split)")

    # ----- Per-user Gaussian on profile -----
    mu = np.zeros((n_users, z_dim), dtype=np.float64)   # (n_users, z_dim)
    var = np.zeros((n_users, z_dim), dtype=np.float64)  # diag
    n_profile = np.zeros(n_users, dtype=np.int64)
    for ui, uid in enumerate(uid_list):
        idxs = profile_idx[uid]
        n_profile[ui] = len(idxs)
        if not idxs:
            continue
        z = z_all[idxs]
        mu[ui] = z.mean(axis=0)
        if len(idxs) >= 2:
            var[ui] = z.var(axis=0, ddof=1)
    var_reg = var + GAUSS_EPS  # 正则化, 防止 σ²=0 → log=-inf
    log(f"per-user Gaussian: μ norm med={np.median(np.linalg.norm(mu,axis=1)):.3f}, "
        f"σ² med mean={np.median(var.mean(axis=1)):.4f}, "
        f"σ² min mean={var.mean(axis=1).min():.4f}")

    # ----- Pooled covariance (avg diag) -----
    pooled_var = var.mean(axis=0) + GAUSS_EPS
    inv_pooled = 1.0 / pooled_var
    log(f"pooled σ²: median={np.median(pooled_var):.4f}, "
        f"min={pooled_var.min():.4f}, max={pooled_var.max():.4f}")

    # ----- 准备 test set 评估 -----
    chance = 1.0 / n_users
    method_results = {}

    # === Method 1: cosine ===
    log("\n[1/3] cosine argmax(z_q, μ_u) ...")
    t1 = time.time()
    mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    correct = 0; total = 0
    for ui, uid in enumerate(uid_list):
        for idx in test_idx[uid]:
            z_q = z_all[idx]
            z_q_norm = z_q / (np.linalg.norm(z_q) + 1e-8)
            sim = mu_norm @ z_q_norm
            pred = int(np.argmax(sim))
            if pred == ui:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    method_results["cosine"] = {
        "acc": acc, "chance": chance, "lift": acc / chance,
        "n_correct": correct, "n_total": total,
        "elapsed_s": time.time() - t1,
    }
    log(f"  cosine: {correct}/{total} = {acc*100:.2f}% "
        f"lift {acc/chance:.2f}x ({time.time()-t1:.1f}s)")

    # === Method 2: Gaussian log-likelihood (diag Σ) ===
    log("\n[2/3] Gaussian log-P argmax log N(z_q|μ_u, σ²_u_diag) ...")
    t1 = time.time()
    log_sigma2 = np.log(var_reg)  # (n_users, z_dim)
    half_log_det = 0.5 * log_sigma2.sum(axis=1)  # (n_users,)
    correct = 0; total = 0
    for ui, uid in enumerate(uid_list):
        for idx in test_idx[uid]:
            z_q = z_all[idx].astype(np.float64)
            diff = z_q - mu  # (n_users, z_dim)
            mahal = (diff ** 2 / var_reg).sum(axis=1)
            log_p = -0.5 * mahal - half_log_det
            pred = int(np.argmax(log_p))
            if pred == ui:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    method_results["log_p_diag"] = {
        "acc": acc, "chance": chance, "lift": acc / chance,
        "n_correct": correct, "n_total": total,
        "elapsed_s": time.time() - t1,
    }
    log(f"  log_p_diag: {correct}/{total} = {acc*100:.2f}% "
        f"lift {acc/chance:.2f}x ({time.time()-t1:.1f}s)")

    # === Method 3: Mahalanobis with pooled Σ ===
    log("\n[3/3] Mahalanobis (pooled diag Σ) argmax -(z-μ)ᵀ Σ_p⁻¹ (z-μ) ...")
    t1 = time.time()
    correct = 0; total = 0
    for ui, uid in enumerate(uid_list):
        for idx in test_idx[uid]:
            z_q = z_all[idx].astype(np.float64)
            diff = z_q - mu  # (n_users, z_dim)
            mahal = ((diff ** 2) * inv_pooled).sum(axis=1)
            pred = int(np.argmin(mahal))
            if pred == ui:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    method_results["mahalanobis_pooled"] = {
        "acc": acc, "chance": chance, "lift": acc / chance,
        "n_correct": correct, "n_total": total,
        "elapsed_s": time.time() - t1,
    }
    log(f"  mahalanobis_pooled: {correct}/{total} = {acc*100:.2f}% "
        f"lift {acc/chance:.2f}x ({time.time()-t1:.1f}s)")

    # ----- Output -----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "embed_source": str(embed_path),
            "z_dim": z_dim, "n_users": n_users,
            "n_sents": n_sents,
            "split": "sha1_hash_50_50",
            "gauss_eps": GAUSS_EPS,
            "use_pooled_cov": GAUSS_USE_POOLED_COV,
        },
        "no_leakage": True,
        "chance": chance,
        "lifts": {m: method_results[m]["lift"] for m in method_results},
        "per_method": method_results,
        "diagnostics": {
            "mu_norm_median": float(np.median(np.linalg.norm(mu, axis=1))),
            "sigma2_median_mean": float(np.median(var.mean(axis=1))),
            "pooled_sigma2_median": float(np.median(pooled_var)),
            "n_profile_avg": float(n_profile.mean()),
            "n_test_avg": float(np.mean([len(v) for v in test_idx.values()])),
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n=== SUMMARY on {embed_path.name} (z={z_dim}, N={n_users}) ===")
    log(f"  PCFG discrete (历史):     ~7.97× chance")
    log(f"  cosine to μ_u:            {method_results['cosine']['lift']:.2f}x "
        f"({method_results['cosine']['acc']*100:.2f}%)")
    log(f"  Gaussian log-P (diag):    {method_results['log_p_diag']['lift']:.2f}x "
        f"({method_results['log_p_diag']['acc']*100:.2f}%)")
    log(f"  Mahalanobis (pooled Σ):   {method_results['mahalanobis_pooled']['lift']:.2f}x "
        f"({method_results['mahalanobis_pooled']['acc']*100:.2f}%)")
    log(f"wrote → {out_path}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


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
# Stage: ablation (distribution-vs-point on strict encoder)
# ============================================================================

def stage_ablation(strict_npz: Path = None,
                   out_path: Path = None):
    """Distribution-vs-point ablation: Gaussian Σ 是否真有额外价值?

    复用 cache/strict_embeddings.npz (strict encoder + 50/50 hash split),
    在 test 上对比 7 种 attribution:

      POINT 类 (只依赖 μ_u):
        M1 cosine_to_μ            argmax cos(z_q, μ_u)
        M2 euclidean_to_μ         argmin ||z_q - μ_u||²   (identity Σ)
        M3 cosine_to_unit_μ       argmax cos(z_q, μ_u/||μ_u||)

      SHARED-Σ 类 (μ_u 个体, Σ 全局共享):
        M4 mahalanobis_shared_diag
        M5 gaussian_logp_shared_diag (== M4 for argmax)

      PER-USER-Σ 类 (μ_u + Σ_u_diag):
        M6 mahalanobis_per_user_diag
        M7 gaussian_logp_per_user_diag  (核心: Σ_u 来自 profile)

      BONUS:
        M8 mahalanobis_per_user_full  (32×32 full Σ, shrunk toward trace/D·I)

    然后按 per-user profile variance (mean σ² across dims) median-split:
      high_var / low_var 两个组, 分别报告 7 种方法 lift
      假设: Gaussian 类方法 (M6/M7/M8) 在 high_var 组 > M1 cosine, 则 Σ 真实
      捕获了用户的"表达范围"

    输出:
      result/03_spacy_encode/syntax_pcfg_ablation.json
        { config, lifts, per_method, per_user_acc,
          user_variance: {median, mean, p25, p75, per_user: {uid: var}},
          stratified: {high_var: {users, lifts, per_method},
                       low_var:  {users, lifts, per_method} } }
    """
    t0 = time.time()
    if strict_npz is None:
        strict_npz = CACHE_DIR / "strict_embeddings.npz"
    if out_path is None:
        out_path = OUT_DIR / "syntax_pcfg_ablation.json"

    if not strict_npz.exists():
        raise FileNotFoundError(
            f"{strict_npz} not found, run stage_strict() or main_pipeline() first")

    log(f"loading strict embeddings: {strict_npz}")
    d = np.load(strict_npz, allow_pickle=True)
    z_profile = d["z_profile"]
    z_test = d["z_test"]
    profile_idx_global = d["profile_idx"]
    test_idx_global = d["test_idx"]
    uid_list = list(d["uid_list"])
    n_users = len(uid_list)
    z_dim = z_profile.shape[1]
    log(f"z_profile={z_profile.shape}, z_test={z_test.shape}, "
        f"users={n_users}, z_dim={z_dim}")

    cache = load_cache()
    user_n_sents = cache["user_n_sents"]

    # ---- per-user profile 索引重建 ----
    profile_idx_dict, test_idx_dict = _profile_test_split(
        uid_list, cache, SENT_CACHE)
    # map global index → local in z_profile / z_test
    global_to_profile_local = {g: l for l, g in enumerate(profile_idx_global)}
    global_to_test_local = {g: l for l, g in enumerate(test_idx_global)}

    # ---- fit per-user μ_u, σ²_u (diag), shared σ², full Σ_u ----
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    var_diag = np.zeros((n_users, z_dim), dtype=np.float64)
    sigma_full = np.zeros((n_users, z_dim, z_dim), dtype=np.float64)
    n_prof = np.zeros(n_users, dtype=np.int64)
    for ui, uid in enumerate(uid_list):
        idxs = profile_idx_dict[uid]
        n_prof[ui] = len(idxs)
        if not idxs:
            continue
        z = z_profile[[global_to_profile_local[i] for i in idxs]]
        mu[ui] = z.mean(axis=0)
        if len(idxs) >= 2:
            var_diag[ui] = z.var(axis=0, ddof=1)
            # full Σ with shrinkage toward trace/D · I (per-user sample too small)
            centered = z - mu[ui]
            cov = (centered.T @ centered) / max(len(idxs) - 1, 1)
            tr = np.trace(cov) / z_dim
            sigma_full[ui] = ((1 - ABLATION_FULL_SHRINK) * cov
                              + ABLATION_FULL_SHRINK * tr * np.eye(z_dim))

    var_reg = var_diag + ABLATION_EPS
    pooled_var = var_diag.mean(axis=0) + ABLATION_EPS
    inv_var_reg = 1.0 / var_reg
    inv_pooled = 1.0 / pooled_var
    log(f"per-user profile: μ norm med={np.median(np.linalg.norm(mu,axis=1)):.3f}, "
        f"σ² mean med={np.median(var_diag.mean(axis=1)):.4f}, "
        f"profile n/user avg={n_prof.mean():.1f}")

    # ---- full Σ pseudo-inverse + Cholesky for per-user full Mahalanobis ----
    sigma_full_reg = sigma_full + ABLATION_EPS * np.eye(z_dim)
    try:
        inv_sigma_full = np.linalg.inv(sigma_full_reg)
    except np.linalg.LinAlgError:
        inv_sigma_full = np.linalg.pinv(sigma_full_reg)
    sign_det_full, logdet_full = np.linalg.slogdet(sigma_full_reg)
    logdet_full = np.where(sign_det_full > 0, logdet_full,
                           np.full(n_users, np.log(ABLATION_EPS) * z_dim))
    log(f"per-user full Σ: det range "
        f"[{np.exp(logdet_full).min():.2e}, {np.exp(logdet_full).max():.2e}]")

    # ---- 测试集 evaluation ----
    chance = 1.0 / n_users
    methods = {}

    def evaluate(method_name: str, score_fn):
        """score_fn(z_q, ui) → (n_users,) score matrix; higher = better.

        Returns (correct, total, per_user_correct, per_user_total).
        """
        correct = 0; total = 0
        per_user_correct = np.zeros(n_users, dtype=np.int64)
        per_user_total = np.zeros(n_users, dtype=np.int64)
        for ui, uid in enumerate(uid_list):
            for global_i in test_idx_dict[uid]:
                local_i = global_to_test_local[global_i]
                z_q = z_test[local_i].astype(np.float64)
                scores = score_fn(z_q, ui)  # (n_users,)
                if int(np.argmax(scores)) == ui:
                    correct += 1
                    per_user_correct[ui] += 1
                per_user_total[ui] += 1
                total += 1
        return correct, total, per_user_correct, per_user_total

    # --- POINT 类 ---
    log("\n[1/7] M1 cosine_to_μ ...")
    t1 = time.time()
    mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)

    def m1_score(z_q, ui):
        z_q_n = z_q / (np.linalg.norm(z_q) + 1e-8)
        return mu_norm @ z_q_n
    c, t, puc, put = evaluate("M1", m1_score)
    methods["M1_cosine_to_μ"] = {"acc": c/t, "lift": (c/t)/chance,
                                  "n_correct": c, "n_total": t,
                                  "elapsed_s": time.time()-t1}
    log(f"  M1 cosine: {c}/{t} = {(c/t)*100:.2f}% lift {(c/t)/chance:.2f}x")

    log("\n[2/7] M2 euclidean_to_μ (identity Σ) ...")
    t1 = time.time()

    def m2_score(z_q, ui):
        return -((mu - z_q) ** 2).sum(axis=1)
    c, t, puc_m2, put = evaluate("M2", m2_score)
    methods["M2_euclidean_to_μ"] = {"acc": c/t, "lift": (c/t)/chance,
                                      "n_correct": c, "n_total": t,
                                      "elapsed_s": time.time()-t1}
    log(f"  M2 euclid: {c}/{t} = {(c/t)*100:.2f}% lift {(c/t)/chance:.2f}x")

    # --- SHARED Σ 类 ---
    log("\n[3/7] M4 mahalanobis_shared_diag ...")
    t1 = time.time()

    def m4_score(z_q, ui):
        diff = z_q - mu
        return -((diff ** 2) * inv_pooled).sum(axis=1)
    c, t, puc, put = evaluate("M4", m4_score)
    methods["M4_mahalanobis_shared_diag"] = {
        "acc": c/t, "lift": (c/t)/chance, "n_correct": c, "n_total": t,
        "elapsed_s": time.time()-t1}
    log(f"  M4 Maha shared: {c}/{t} = {(c/t)*100:.2f}% lift {(c/t)/chance:.2f}x")

    # --- PER-USER Σ 类 (核心) ---
    log("\n[4/7] M6 mahalanobis_per_user_diag ...")
    t1 = time.time()

    def m6_score(z_q, ui):
        diff = z_q - mu  # (n_users, z_dim)
        return -((diff ** 2) * inv_var_reg).sum(axis=1)
    c, t, puc_m6, put = evaluate("M6", m6_score)
    methods["M6_mahalanobis_per_user_diag"] = {
        "acc": c/t, "lift": (c/t)/chance, "n_correct": c, "n_total": t,
        "elapsed_s": time.time()-t1}
    log(f"  M6 Maha per-user diag: {c}/{t} = {(c/t)*100:.2f}% "
        f"lift {(c/t)/chance:.2f}x")

    log("\n[5/7] M7 gaussian_logp_per_user_diag ...")
    t1 = time.time()
    half_log_det = 0.5 * np.log(var_reg).sum(axis=1)

    def m7_score(z_q, ui):
        diff = z_q - mu
        mahal = ((diff ** 2) * inv_var_reg).sum(axis=1)
        return -0.5 * mahal - half_log_det
    c, t, puc_m7, put = evaluate("M7", m7_score)
    methods["M7_gaussian_logp_per_user_diag"] = {
        "acc": c/t, "lift": (c/t)/chance, "n_correct": c, "n_total": t,
        "elapsed_s": time.time()-t1}
    log(f"  M7 logP per-user diag: {c}/{t} = {(c/t)*100:.2f}% "
        f"lift {(c/t)/chance:.2f}x")

    log("\n[6/7] M8 mahalanobis_per_user_full (32x32 Σ shrunk) ...")
    t1 = time.time()

    def m8_score(z_q, ui):
        diff = z_q - mu  # (n_users, z_dim)
        # diff @ Σ_u^{-1} @ diff.T per user
        out = np.einsum("ud,ude,ue->u", diff, inv_sigma_full, diff)
        return -out
    c, t, puc_m8, put = evaluate("M8", m8_score)
    methods["M8_mahalanobis_per_user_full"] = {
        "acc": c/t, "lift": (c/t)/chance, "n_correct": c, "n_total": t,
        "elapsed_s": time.time()-t1}
    log(f"  M8 Maha per-user full: {c}/{t} = {(c/t)*100:.2f}% "
        f"lift {(c/t)/chance:.2f}x")

    log("\n[7/7] M3 cosine_to_unit_μ (sanity, identical to M1 by construction) ...")
    t1 = time.time()
    # 实际就是 M1, 跳过重复
    methods["M3_cosine_to_unit_μ"] = methods["M1_cosine_to_μ"]

    # ---- per-user variance stratification ----
    # variance proxy = mean trace / dim = mean variance across dims
    user_var = var_diag.mean(axis=1)
    median_var = float(np.median(user_var))
    high_idx = np.where(user_var >= median_var)[0]
    low_idx = np.where(user_var < median_var)[0]
    log(f"\n=== variance stratification ===")
    log(f"user σ² mean (across dims): median={median_var:.4f}, "
        f"min={user_var.min():.4f}, max={user_var.max():.4f}")
    log(f"high_var ({len(high_idx)} users, σ² ≥ median) vs "
        f"low_var ({len(low_idx)} users, σ² < median)")

    def per_group_lift(puc_array, put_array, idx, label):
        sub_c = int(puc_array[idx].sum())
        sub_t = int(put_array[idx].sum())
        sub_acc = sub_c / max(sub_t, 1)
        return {"users": int(len(idx)),
                "n_correct": sub_c, "n_total": sub_t,
                "acc": sub_acc, "lift": sub_acc / chance}

    # 重新做: 直接 reuse M1/M2/M6/M7/M8 attribution result 用 per-user correct
    # 上面 evaluate() 返回的 per_user_correct 是按方法来的, 这里为简化直接重新跑
    # (attr 总共 7×6800 ≈ 48K ops, < 1s)
    def per_user_correct_for(method_name):
        if method_name == "M1":
            fn = m1_score
        elif method_name == "M2":
            fn = m2_score
        elif method_name == "M4":
            fn = m4_score
        elif method_name == "M6":
            fn = m6_score
        elif method_name == "M7":
            fn = m7_score
        elif method_name == "M8":
            fn = m8_score
        else:
            raise ValueError(method_name)
        puc_m = np.zeros(n_users, dtype=np.int64)
        put_m = np.zeros(n_users, dtype=np.int64)
        for ui, uid in enumerate(uid_list):
            for global_i in test_idx_dict[uid]:
                local_i = global_to_test_local[global_i]
                z_q = z_test[local_i].astype(np.float64)
                scores = fn(z_q, ui)
                if int(np.argmax(scores)) == ui:
                    puc_m[ui] += 1
                put_m[ui] += 1
        return puc_m, put_m

    puc_per_method = {}
    for m_name in ["M1_cosine_to_μ", "M2_euclidean_to_μ",
                   "M4_mahalanobis_shared_diag",
                   "M6_mahalanobis_per_user_diag",
                   "M7_gaussian_logp_per_user_diag",
                   "M8_mahalanobis_per_user_full"]:
        puc_per_method[m_name] = per_user_correct_for(m_name.split("_")[0])

    log("\nper-group results:")
    stratified = {"high_var": {}, "low_var": {}}
    for group_label, idx in [("high_var", high_idx), ("low_var", low_idx)]:
        stratified[group_label] = {
            "n_users": int(len(idx)),
            "variance_threshold": (f">= median {median_var:.4f}"
                                   if group_label == "high_var"
                                   else f"< median {median_var:.4f}"),
            "users": [uid_list[i] for i in idx],
            "per_method": {},
        }
        for m_name, (puc, put) in puc_per_method.items():
            sub = per_group_lift(puc, put, idx, group_label)
            stratified[group_label]["per_method"][m_name] = sub
            log(f"  {group_label} {m_name}: "
                f"{sub['n_correct']}/{sub['n_total']} "
                f"= {sub['acc']*100:.2f}% lift {sub['lift']:.2f}x")

    # ---- 核心结论 ----
    log("\n=== ABLATION SUMMARY (strict encoder, no leakage) ===")
    log(f"PCFG discrete (历史):          ~7.97× chance")
    for m_name, r in methods.items():
        if m_name == "M3_cosine_to_unit_μ":
            continue
        log(f"  {m_name:>42s}: lift {r['lift']:.2f}x ({r['acc']*100:.2f}%)")
    log("\nHIGH-var vs LOW-var 组 lift 对比 (Gaussian 是否在高方差用户更好?):")
    for m_name in ["M1_cosine_to_μ", "M2_euclidean_to_μ",
                   "M4_mahalanobis_shared_diag",
                   "M6_mahalanobis_per_user_diag",
                   "M7_gaussian_logp_per_user_diag",
                   "M8_mahalanobis_per_user_full"]:
        h = stratified["high_var"]["per_method"][m_name]["lift"]
        l = stratified["low_var"]["per_method"][m_name]["lift"]
        delta = h - l
        marker = (" ✓ Σ_value" if (
            m_name in ["M6_mahalanobis_per_user_diag",
                       "M7_gaussian_logp_per_user_diag",
                       "M8_mahalanobis_per_user_full"]
            and h > stratified["high_var"]["per_method"]["M1_cosine_to_μ"]["lift"])
            else "")
        log(f"  {m_name:>42s}: high_var={h:.2f}x  low_var={l:.2f}x  "
            f"Δ(high-low)={delta:+.2f}{marker}")

    # ---- 输出 JSON ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "strict_npz": str(strict_npz),
            "z_dim": z_dim, "n_users": n_users,
            "n_profile_total": int(len(profile_idx_global)),
            "n_test_total": int(len(test_idx_global)),
            "gauss_eps": ABLATION_EPS,
            "full_shrink_alpha": ABLATION_FULL_SHRINK,
        },
        "no_leakage": True,
        "chance": chance,
        "methods": methods,
        "user_variance": {
            "median": median_var,
            "mean": float(user_var.mean()),
            "p25": float(np.percentile(user_var, 25)),
            "p75": float(np.percentile(user_var, 75)),
            "min": float(user_var.min()),
            "max": float(user_var.max()),
            "per_user": {uid_list[ui]: float(user_var[ui])
                         for ui in range(n_users)},
        },
        "stratified": stratified,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {out_path}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


# ============================================================================
# Stage: adaptive (3-way profile/val/test, τ on val, eval on test)
# ============================================================================

def _three_way_split(uid_list, cache, sent_cache_path,
                     profile_mod=10, val_mod=10,
                     profile_cut=5, val_cut=7, salt: str = None):
    """SHA1 hash 3-way split: profile (b<5) / val (5≤b<7) / test (7≤b<10)。

    50/20/30, 三个桶互斥且穷尽。encoder 只见过 profile (用内部 15% 做早停),
    validation 用 val 完全未参与训练, test 完全未参与 encoder 训练或阈值选择。
    salt: 改 hash 分配 (multi_seed 用)
    """
    with open(sent_cache_path, "rb") as f:
        uid_to_sents = pickle.load(f)
    profile_dict = {u: [] for u in uid_list}
    val_dict = {u: [] for u in uid_list}
    test_dict = {u: [] for u in uid_list}
    sent_off = 0
    for ui, uid in enumerate(uid_list):
        sents = [s for s in uid_to_sents[uid] if 5 < len(s.split()) < 50]
        for si, s in enumerate(sents):
            b = hash_bucket(s, mod=profile_mod, salt=salt)
            if b < profile_cut:
                profile_dict[uid].append(sent_off + si)
            elif b < val_cut:
                val_dict[uid].append(sent_off + si)
            else:
                test_dict[uid].append(sent_off + si)
        sent_off += len(sents)
    return profile_dict, val_dict, test_dict


def stage_adaptive(out_path: Path = None):
    """Adaptive scorer: variance-driven Cosine vs Maha-full switching。

    协议 (3-way split, encoder 完全无泄漏):
      1. SHA1 hash 50/20/30 → profile / val / test
      2. encoder 只用 profile 训练 (内部 15% hash 做早停)
      3. freeze encoder, 编码 profile + val + test
      4. profile 上 fit per-user μ_u, σ²_u_diag, Σ_u_full (with shrinkage)
      5. val 上 sweep τ in [V_min, V_max], 对每个 τ:
           per-user: V_u > τ → Maha-full, else Cosine
           算 val acc, 选 τ* = argmax
      6. test 上一次性报告 Cosine / Maha-full / Adaptive(τ*) / Soft-mixture

    Soft mixture: α(V) = sigmoid((τ - V)/scale)
      score = α * cos(z, μ) + (1-α) * (-mahal_full(z, μ, Σ))
      (cos 和 Maha 都 normalize 到 [0,1] 后再 mix)

    输出:
      result/03_spacy_encode/syntax_pcfg_adaptive.json
        { config: {n_profile, n_val, n_test, τ_sweep},
          tau_selected: float,
          val_curve: [{tau, acc, lift}, ...],
          test_results: {cosine, maha_full, adaptive_tau_star, soft_mixture,
                          per_method: {acc, lift, n_correct, n_total}},
          diagnostics: { per_user_variance, encoder_diagnostics } }
    """
    t0 = time.time()
    if out_path is None:
        out_path = OUT_DIR / "syntax_pcfg_adaptive.json"

    cached = [
        out_path,
        CACHE_DIR / "adaptive_encoder.pt",
        CACHE_DIR / "adaptive_embeddings.npz",
    ]
    missing = [p for p in cached if not p.exists()]
    if not missing:
        cache_meta = json.loads((CACHE_DIR / "meta.json").read_text())
        npz = np.load(CACHE_DIR / "adaptive_embeddings.npz")
        n_profile = int(npz["z_profile"].shape[0])
        n_val = int(npz["z_val"].shape[0])
        n_test = int(npz["z_test"].shape[0])
        n_users_cached = int(len(npz["uid_list"]))
        if (n_profile + n_val + n_test != cache_meta["n_total_sents"]
                or n_users_cached != cache_meta["n_users"]):
            raise ValueError(
                f"stage_adaptive cache cohort mismatch: cached profile="
                f"{n_profile} val={n_val} test={n_test} users="
                f"{n_users_cached} vs current n_sents="
                f"{cache_meta['n_total_sents']} n_users="
                f"{cache_meta['n_users']}; delete "
                f"{CACHE_DIR / 'adaptive_embeddings.npz'} and rerun")
        log(f"skip stage_adaptive: {len(cached)} cached output(s) exist, "
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
    log(f"adaptive: V={V}, n_sents={n_sents}, n_users={n_users}")

    sent_norm = normalize_counts(sent_csr)

    def to_dense_batch(indices: np.ndarray) -> torch.Tensor:
        rows = sent_norm[indices]
        return torch.tensor(rows.toarray(), dtype=torch.float32,
                            device=device)

    # === Step 1: 3-way split ===
    profile_dict, val_dict, test_dict = _three_way_split(
        uid_list, cache, SENT_CACHE)
    profile_idx = np.asarray(
        sorted(i for v in profile_dict.values() for i in v), dtype=np.int64)
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

    # === Step 2: train encoder on profile only ===
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

    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR,
                           weight_decay=SUP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SUP_EPOCHS)

    best_val = 0.0
    best_state = None
    n_train = len(profile_train_inner)
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 10
    for epoch in range(1, SUP_EPOCHS + 1):
        model.train()
        perm_epoch = np.random.permutation(profile_train_inner)
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
        sched.step()
        model.eval()
        with torch.no_grad():
            v_correct = 0
            for i in range(0, len(profile_val_inner), SUP_BATCH_SIZE):
                bi = profile_val_inner[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(bi)
                _, logits = model(xb)
                yb = torch.tensor(user_labels[bi],
                                  dtype=torch.long, device=device)
                v_correct += int(
                    (logits.argmax(dim=1) == yb).sum().detach())
        v_acc = v_correct / max(len(profile_val_inner), 1)
        if v_acc > best_val:
            best_val = v_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log(f"  early stop at epoch {epoch} "
                f"(no improve for {EARLY_STOP_PATIENCE} epochs)")
            break
        if (epoch == 1 or epoch % SUP_LOG_EVERY == 0
                or epoch == SUP_EPOCHS):
            log(f"  epoch {epoch:>3}/{SUP_EPOCHS} val={v_acc*100:.1f}%")
    log(f"encoder best val_acc (inner): {best_val*100:.1f}%")
    model.load_state_dict(best_state)
    model.eval()

    # === Step 3: encode all 3 sets ===
    def encode_all(indices: np.ndarray) -> np.ndarray:
        out = np.zeros((len(indices), SUP_Z_DIM), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(indices), SUP_BATCH_SIZE):
                bi = indices[i:i + SUP_BATCH_SIZE]
                xb = to_dense_batch(bi)
                z, _ = model(xb)
                out[i:i + xb.shape[0]] = z.cpu().numpy()
        return out

    z_profile = encode_all(profile_idx)
    z_val = encode_all(val_idx)
    z_test = encode_all(test_idx)
    log(f"encoded: profile {z_profile.shape}, val {z_val.shape}, "
        f"test {z_test.shape}")

    # global → local mapping
    p2l = {g: l for l, g in enumerate(profile_idx)}
    v2l = {g: l for l, g in enumerate(val_idx)}
    t2l = {g: l for l, g in enumerate(test_idx)}

    # === Step 4: fit per-user μ, σ², Σ_full on PROFILE only ===
    z_dim = SUP_Z_DIM
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    var_diag = np.zeros((n_users, z_dim), dtype=np.float64)
    sigma_full = np.zeros((n_users, z_dim, z_dim), dtype=np.float64)
    n_prof = np.zeros(n_users, dtype=np.int64)
    for ui, uid in enumerate(uid_list):
        idxs = profile_dict[uid]
        n_prof[ui] = len(idxs)
        if not idxs:
            continue
        z = z_profile[[p2l[i] for i in idxs]]
        mu[ui] = z.mean(axis=0)
        if len(idxs) >= 2:
            var_diag[ui] = z.var(axis=0, ddof=1)
            centered = z - mu[ui]
            cov = (centered.T @ centered) / max(len(idxs) - 1, 1)
            tr = np.trace(cov) / z_dim
            sigma_full[ui] = ((1 - ABLATION_FULL_SHRINK) * cov
                              + ABLATION_FULL_SHRINK * tr * np.eye(z_dim))

    sigma_full_reg = sigma_full + ABLATION_EPS * np.eye(z_dim)
    try:
        inv_sigma_full = np.linalg.inv(sigma_full_reg)
    except np.linalg.LinAlgError:
        inv_sigma_full = np.linalg.pinv(sigma_full_reg)

    user_var = var_diag.mean(axis=1)
    log(f"per-user profile σ² median={np.median(user_var):.4f}, "
        f"min={user_var.min():.4f}, max={user_var.max():.4f}, "
        f"profile n/user avg={n_prof.mean():.1f}")

    # === Precompute cosine/Maha score matrices ===
    # cosine: (n_users, z_dim) normalized
    mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    # mahalanobis_full: per-user (z_dim, z_dim)
    # 准备 cached score functions for fast evaluation
    def get_scores(z_q):
        """返回 cos_score (n_users,), maha_score (n_users,) 数组。
        cos_score ∈ [-1, 1] 越大越像; maha_score 越小越像 (实际返回 -mahal 让越大越像)
        """
        z_q_n = z_q / (np.linalg.norm(z_q) + 1e-8)
        cos_score = mu_norm @ z_q_n  # (n_users,)
        diff = z_q - mu  # (n_users, z_dim)
        maha = np.einsum("ud,ude,ue->u", diff, inv_sigma_full, diff)
        maha_score = -maha  # higher = closer
        return cos_score, maha_score

    # === Step 5: sweep τ on val ===
    # 候选 τ: 用户 variance 的 0, 5, 10, ..., 100 percentile
    var_sorted = np.sort(user_var)
    tau_candidates = np.concatenate([
        [var_sorted[0] - 1e-6],  # all use cosine
        np.quantile(var_sorted, np.linspace(0.05, 0.95, 19)),
        [var_sorted[-1] + 1e-6],  # all use maha
    ])
    tau_candidates = np.unique(np.round(tau_candidates, 6))

    def eval_on_split(split_dict, mapping, tau):
        """Eval adaptive on split; tau=None → all cosine; tau=+inf → all maha."""
        correct = 0; total = 0
        for ui, uid in enumerate(uid_list):
            for global_i in split_dict[uid]:
                li = mapping[global_i]
                z_q = (z_val if split_dict is val_dict else z_test).astype(np.float64)[li]
                cos_s, maha_s = get_scores(z_q)
                if tau is None:
                    pred = int(np.argmax(cos_s))
                elif tau == float("inf"):
                    pred = int(np.argmax(maha_s))
                else:
                    use_maha = user_var[ui] > tau
                    pred = int(np.argmax(maha_s if use_maha else cos_s))
                if pred == ui:
                    correct += 1
                total += 1
        return correct, total

    log(f"\nsweeping τ on val ({len(tau_candidates)} candidates) ...")

    # === Vectorized score matrix computation ===
    # precompute mu32, inv_sigma_full32, mu_inv_sigma_mu32 (per-user scalars)
    mu32 = mu.astype(np.float32)
    inv_sigma_full32 = inv_sigma_full.astype(np.float32)
    mu_norm32 = mu32 / (np.linalg.norm(mu32, axis=1, keepdims=True) + 1e-8)
    # mu_inv_sigma_mu[u] = sum_d sum_e mu[u,d] * inv_sigma[u,d,e] * mu[u,e]
    mu_inv_sigma_mu32 = np.einsum("ud,ude,ue->u", mu32, inv_sigma_full32, mu32)

    def compute_score_matrices(z_q_arr, chunk=1024, user_chunk=256):
        """返回 (cos_mat, maha_mat), 各 (n_q, n_users)。maha_mat = -D_M² (越大越像)。

        用 GPU torch matmul:per-user Σ⁻¹ × (cz - μ) 通过 batched matmul 算,
        避免 OpenBLAS 单核 einsum 性能问题。CUDA stream parallel + cuBLAS batching。
        """
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_q_arr = np.asarray(z_q_arr)
        if z_q_arr.ndim == 1:
            z_q_arr = z_q_arr.reshape(1, -1)
        if z_q_arr.ndim > 2:
            z_q_arr = z_q_arr.reshape(-1, z_q_arr.shape[-1])
        n = len(z_q_arr)
        cos_mat = np.zeros((n, n_users), dtype=np.float32)
        maha_mat = np.zeros((n, n_users), dtype=np.float32)
        # 把 Σ⁻¹ 和 μ 上 GPU (一次性,~20MB)
        inv_S_dev = torch.from_numpy(inv_sigma_full32).to(device)  # (n_users, z, z)
        mu_dev = torch.from_numpy(mu32).to(device)                  # (n_users, z)
        mu_inv_mu_dev = torch.from_numpy(mu_inv_sigma_mu32.copy()).to(device)  # (n_users,)
        for i in range(0, n, chunk):
            cz = z_q_arr[i:i + chunk].astype(np.float32)
            cz_n = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-8)
            cos_mat[i:i + chunk] = cz_n @ mu_norm32.T
            cz_dev = torch.from_numpy(cz).to(device)  # (chunk, z)
            for u_start in range(0, n_users, user_chunk):
                u_end = min(u_start + user_chunk, n_users)
                inv_S_uc = inv_S_dev[u_start:u_end]  # (uc, z, z)
                mu_uc = mu_dev[u_start:u_end]        # (uc, z)
                diff = cz_dev.unsqueeze(1) - mu_uc.unsqueeze(0)  # (chunk, uc, z)
                # D_M²[c, u] = Σ_d Σ_e diff[c,u,d] · inv_S[u,d,e] · diff[c,u,e]
                # bmm: diff @ inv_S_uc → (chunk, uc, z); then (· diff).sum(-1)
                # 用 bmm: batch = chunk*uc, matrix (1, z) @ (z, z) → (1, z) for each
                # Simpler: torch.einsum (PyTorch 会自动 fallback to matmul)
                mahal = torch.einsum("cud,ude,cue->cu", diff, inv_S_uc, diff)  # (chunk, uc)
                mahal = mahal + mu_inv_mu_dev[u_start:u_end].unsqueeze(0)  # (chunk, uc)
                maha_mat[i:i + chunk, u_start:u_end] = (-mahal).cpu().numpy().astype(np.float32)
        del inv_S_dev, mu_dev, mu_inv_mu_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cos_mat, maha_mat

    # q2u mapping for val and test
    q2u_val = np.zeros(len(val_idx), dtype=np.int32)
    for ui, uid in enumerate(uid_list):
        for gi in val_dict[uid]:
            if gi in v2l:
                q2u_val[v2l[gi]] = ui
    q2u_test = np.zeros(len(test_idx), dtype=np.int32)
    for ui, uid in enumerate(uid_list):
        for gi in test_dict[uid]:
            if gi in t2l:
                q2u_test[t2l[gi]] = ui

    log(f"  building val score matrices ...")
    t_sm = time.time()
    cos_val, maha_val = compute_score_matrices(z_val)
    log(f"    cos_val {cos_val.shape}, maha_val {maha_val.shape} "
        f"({time.time()-t_sm:.0f}s)")

    val_curve = []
    chance_val = 1.0 / n_users
    for tau in tau_candidates:
        n_high_maha = int((user_var > tau).sum())
        if n_high_maha == 0:
            pred = cos_val.argmax(axis=1)
        elif n_high_maha == n_users:
            pred = maha_val.argmax(axis=1)
        else:
            use_maha = user_var > tau  # (n_users,)
            adapt = np.where(use_maha[None, :], maha_val, cos_val)
            pred = adapt.argmax(axis=1)
        c = int((pred == q2u_val).sum())
        acc = c / max(len(pred), 1)
        val_curve.append({"tau": float(tau),
                          "n_high_maha": n_high_maha,
                          "acc": acc,
                          "lift": acc / chance if (chance := 1.0/n_users) else 0,
                          "n_correct": c, "n_total": len(q2u_val)})
    val_curve.sort(key=lambda r: -r["acc"])
    log(f"  best val τ: {val_curve[0]['tau']:.4f} "
        f"(acc={val_curve[0]['acc']*100:.2f}%, lift={val_curve[0]['lift']:.2f}x, "
        f"n_high_maha={val_curve[0]['n_high_maha']})")
    log(f"  worst val τ: {val_curve[-1]['tau']:.4f} "
        f"(acc={val_curve[-1]['acc']*100:.2f}%)")
    log(f"  all-cosine (τ=-inf): {val_curve[0] and [r for r in val_curve if r['n_high_maha']==n_users][0]}")
    # 重新找 all-cosine 和 all-maha baseline
    val_all_cos = next((r for r in val_curve if r["n_high_maha"] == 0), None)
    val_all_maha = next((r for r in val_curve if r["n_high_maha"] == n_users), None)
    log(f"  val all-cosine: lift={val_all_cos['lift']:.2f}x "
        f"({val_all_cos['acc']*100:.2f}%)")
    log(f"  val all-maha:   lift={val_all_maha['lift']:.2f}x "
        f"({val_all_maha['acc']*100:.2f}%)")

    # best τ*
    tau_star = val_curve[0]["tau"]

    # === Step 6: test held-out evaluation ===
    chance = 1.0 / n_users
    log(f"\n=== TEST evaluation (held-out, encoder & τ 都未见过) ===")

    log(f"  building test score matrices ...")
    t_sm = time.time()
    cos_test, maha_test = compute_score_matrices(z_test)
    log(f"    cos_test {cos_test.shape}, maha_test {maha_test.shape} "
        f"({time.time()-t_sm:.0f}s)")

    def test_eval(method: str):
        """Vectorized test eval on precomputed score matrices."""
        if method == "cosine":
            pred = cos_test.argmax(axis=1)
        elif method == "maha_full":
            pred = maha_test.argmax(axis=1)
        elif method == "adaptive":
            use_maha = user_var > tau_star  # (n_users,)
            adapt = np.where(use_maha[None, :], maha_test, cos_test)
            pred = adapt.argmax(axis=1)
        elif method == "soft_mixture":
            # alpha = sigmoid((τ - V) / scale), larger → more cosine
            alpha = 1.0 / (1.0 + np.exp(
                (user_var - tau_star) / ADAPTIVE_SOFT_SCALE))  # (n_users,)
            cos_n = (cos_test + 1.0) / 2.0  # [0,1]
            maha_n = np.exp(-np.maximum(maha_test, 0) / z_dim)
            score = alpha[None, :] * cos_n + (1.0 - alpha)[None, :] * maha_n
            pred = score.argmax(axis=1)
        else:
            raise ValueError(method)
        c = int((pred == q2u_test).sum())
        total = len(pred)
        acc = c / max(total, 1)
        return {"acc": acc, "lift": acc / chance,
                "n_correct": c, "n_total": total}

    test_results = {}
    for m in ["cosine", "maha_full", "adaptive", "soft_mixture"]:
        r = test_eval(m)
        test_results[m] = r
        log(f"  test {m:>15s}: {r['n_correct']}/{r['n_total']} "
            f"= {r['acc']*100:.2f}% lift {r['lift']:.2f}x")

    # adaptive vs best of (cosine, maha_full) gain
    base_best = max(test_results["cosine"]["lift"],
                    test_results["maha_full"]["lift"])
    adaptive_lift = test_results["adaptive"]["lift"]
    soft_lift = test_results["soft_mixture"]["lift"]
    log(f"\n  Δ(adaptive - best of cosine/maha_full): "
        f"{adaptive_lift - base_best:+.2f}x")
    log(f"  Δ(soft_mixture - best): {soft_lift - base_best:+.2f}x")

    # === Output ===
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "z_dim": SUP_Z_DIM, "n_users": n_users,
            "n_profile": int(len(profile_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "split": "sha1_hash_50_20_30",
            "profile_mod_cut": (ADAPTIVE_PROFILE_FRAC, ADAPTIVE_VAL_FRAC),
            "gauss_eps": ABLATION_EPS,
            "full_shrink_alpha": ABLATION_FULL_SHRINK,
            "soft_scale": ADAPTIVE_SOFT_SCALE,
        },
        "no_leakage": True,
        "encoder_seen_test": False,
        "tau_seen_test": False,
        "chance": chance,
        "encoder_diagnostics": {
            "best_inner_val_acc": best_val,
            "n_params_M": n_params / 1e6,
        },
        "val_curve_top10": val_curve[:10],
        "val_all_cosine": val_all_cos,
        "val_all_maha": val_all_maha,
        "tau_selected": float(tau_star),
        "test_results": test_results,
        "delta_vs_best": {
            "adaptive_minus_best_point": adaptive_lift - base_best,
            "soft_minus_best_point": soft_lift - base_best,
        },
        "user_variance": {
            "median": float(np.median(user_var)),
            "mean": float(user_var.mean()),
            "p25": float(np.percentile(user_var, 25)),
            "p75": float(np.percentile(user_var, 75)),
            "per_user": {uid_list[ui]: float(user_var[ui])
                         for ui in range(n_users)},
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 持久化 encoder + 三段 embeddings 便于后续复用
    torch.save({"model_state": best_state,
                "config": {"vocab_size": V, "z_dim": SUP_Z_DIM,
                           "hidden": list(SUP_HIDDEN), "dropout": SUP_DROPOUT,
                           "n_users": n_users}},
               CACHE_DIR / "adaptive_encoder.pt")
    np.savez(CACHE_DIR / "adaptive_embeddings.npz",
             z_profile=z_profile, z_val=z_val, z_test=z_test,
             profile_idx=profile_idx, val_idx=val_idx, test_idx=test_idx,
             uid_list=np.asarray(uid_list),
             user_var=user_var, tau_star=tau_star)
    log(f"wrote → {out_path} + adaptive_encoder.pt + adaptive_embeddings.npz")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


# ============================================================================
# Main pipeline (一键串行运行所有 stage, 不依赖 MODE 调度)
# ============================================================================

def main_pipeline():
    """Canonical pipeline: 串行执行全部 stage, 与历史 `full` MODE 等价。

    链路: cache → pcfg_attr → strict → gauss → attr → gauss_attr
          → ablation → adaptive
    (stage_super 与 stage_multi_seed 已删除, strict 兼任产出
     supervised_embeddings.npy 给 gauss/attr/gauss_attr 消费, 省 ~45min super + 4h multi_seed)

    假设 cache 已有产物则可跳过 stage_cache() (resume);
    任意 stage 抛错会立即终止, 不做 fallback (per Rule 7)。
    """
    t_total = time.time()
    log(f"=== main_pipeline: canonical chain start ===")

    stage_cache()
    stage_pcfg_attr()
    stage_strict()  # 同时产出 supervised_embeddings.npy + strict_embeddings.npz
    stage_gauss(embed_path=CACHE_DIR / "supervised_embeddings.npy")
    stage_attr(embed_path=CACHE_DIR / "supervised_embeddings.npy")
    stage_gauss_attr()
    stage_ablation()
    stage_adaptive()
    # stage_multi_seed() 已从主链路移除 (单 stage ~4h, 单独调)

    log(f"\n=== ALL DONE ({time.time()-t_total:.0f}s) ===")


if __name__ == "__main__":
    main_pipeline()