#!/usr/bin/env python3
"""Phase 4.C — PCA48-Nearest Exemplar Selection.

用户指令 2026-08-30:
Phase 4.B → STOP_SCALING_EXEMPLARS. 现在切换到 selection quality.
固定 N_EXEMPLARS=1 (Phase 4.B 最优点).
比较三种 selection:
  A. random    — Phase 4.B N=1 重现(对照)
  B. nearest   — 选该 user 历史中 Mahalanobis 距离 Gaussian 中心最近的句子
  C. farthest  — 选距离最大的句子 (negative control)

**Leakage control**:
  - Exemplar 候选池 = user 的历史句子, 但排除 user 在 target ASIN 上的句子
    (避免同商品 review 文本带来的 semantic leakage)
  - prompt 模板与 Phase 4.B 完全一致 (N_EXEMPLARS=1)

**新增诊断量** (per pair):
  - d_exemplar_self  (exemplar 到 user Gaussian 的 Mahalanobis)
  - d_query_self     (生成 query 到 user Gaussian 的 Mahalanobis, Stage 4 给的)
  - transfer_gain = d_query_self(random) - d_query_self(nearest)
  - ρ(d_exemplar, d_query) Spearman

**Decision**:
  GO           nearest 比 random 降 min_d_self ≥ 1.0, PASSED% 升, ρ>0
  PARTIAL-GO   nearest 降 F3/exclusivity 但 min_d_self <0.5
  NO-GO        nearest 与 random 在 min_d_self 上几乎一样
  STRONG NO-GO nearest 与 farthest 也几乎一样

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

# Import helpers from sweep / prototype regen (shared prompts + filters).
_COUNTSWEEP = REPO_ROOT / "gen_query" / "syntax_subspace_exemplar_count_sweep.py"
_spec = importlib.util.spec_from_file_location("count_sweep", _COUNTSWEEP)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["count_sweep"] = _mod
_spec.loader.exec_module(_mod)
batch_generate_vllm = _mod.batch_generate_vllm
build_user_content = _mod.build_user_content
log = _mod.log
get_top_n_attrs = _mod.get_top_n_attrs
count_attrs_covered = _mod.count_attrs_covered
has_invalid_punct = _mod.has_invalid_punct
has_first_person = _mod.has_first_person
has_emoji = _mod.has_emoji
has_self_talk = _mod.has_self_talk
is_query_too_long = _mod.is_query_too_long
n_tokens_simple = _mod.n_tokens_simple
make_exemplar_prompt = _mod.make_exemplar_prompt

PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
COHORT_ASINS = SCRATCH / "stage8_5_asins_strict34_intersect2174_ksweep_K200.json"
SENT_IN = SCRATCH / "sentences_for_rewrite_10k.jsonl"
GAUSS_PATH = SCRATCH / "stage8_5_user_gaussians.json"

# === Phase 4.C config (locked from Phase 4.B) ===
PILOT_N_ASIN = 50
N_SAMPLE_USERS = 20
RANDOM_SEED_ASIN = 42
RANDOM_SEED_USERS = 0
K_PER_PAIR = 4
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60
N_INPUT = 5
N_EXEMPLARS = 1  # LOCKED to 1 from Phase 4.B optimum
PCA_DIM = 48
R_95 = 8.073

SELECTION_CONDITIONS = ["random", "nearest", "farthest"]


# ----------------------------------------------------------------------------
# Feature extraction utilities (mirror Stage4 318d → F3 → scaler → PCA48)
# ----------------------------------------------------------------------------
import spacy
_NLP = None
def get_nlp():
    global _NLP
    if _NLP is None:
        nlp = spacy.load("en_core_web_sm")
        if "textcat" in nlp.pipe_names:
            nlp.remove_pipe("textcat")
        _NLP = nlp
    return _NLP


def extract_318d(sents: list[str], fnames: list[str]) -> np.ndarray:
    """Run 318d spaCy feature extraction on a batch of sentences.

    Returns array shape (n_sents, 318).
    """
    from syntactic_features import per_sentence_features_v2
    nlp = get_nlp()
    docs = list(nlp.pipe(sents, batch_size=512, n_process=8))
    feats = []
    for doc, sent in zip(docs, sents):
        try:
            d = per_sentence_features_v2(doc, sent)
            feats.append([float(d.get(fn, 0.0)) for fn in fnames])
        except Exception:
            feats.append([0.0] * len(fnames))
    return np.asarray(feats, dtype=np.float64)


def project_f3_pca48(feats_318: np.ndarray, gauss: dict) -> np.ndarray:
    """Take 318d features → F3 subset → StandardScaler → PCA48."""
    all_fnames = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx = [all_fnames.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gauss["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)  # (48, n_features_f3)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)              # (n_features_f3,)

    f3 = feats_318[:, col_idx]  # (n, len_f3)
    f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
    # PCA48: subtract pca_mean then project
    z = (f3_scaled - pca_mean) @ pca_components.T
    return z


def mahal_to_user(z_48: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance for batch z to (mu, sigma_diag).

    z: (n, 48); mu: (48,); sigma_diag: (48,)
    Returns (n,) distances.
    """
    diff = z_48 - mu
    return np.sqrt(np.maximum((diff ** 2 / np.maximum(sigma_diag, 1e-12)).sum(axis=1), 0.0))


# ----------------------------------------------------------------------------
# Cohort + sentences
# ----------------------------------------------------------------------------
def load_user_sentences() -> dict[str, list[dict]]:
    """Return user_id → list of {text, asin, review_index}."""
    by_user: dict[str, list[dict]] = {}
    with open(SENT_IN) as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("sentence_text", "").strip()
            wc = len(text.split())
            if wc < 5 or wc > 60:
                continue
            by_user.setdefault(rec["user_id"], []).append({
                "text": text,
                "asin": rec["asin"],
                "review_index": rec.get("review_index"),
            })
    return by_user


def get_pairs_and_attrs() -> list[dict]:
    """Return list of {user_id, asin, attrs}. Same logic as Phase 4.B."""
    pattrs = json.load(open(PATTRS_PATH))
    cohort = json.load(open(COHORT_ASINS))
    rng_asin = random.Random(RANDOM_SEED_ASIN)
    asin_data_sorted = sorted(cohort["asins"], key=lambda x: x["asin"])
    rng_asin.shuffle(asin_data_sorted)
    asin_data_pilot = asin_data_sorted[:PILOT_N_ASIN]
    pairs = []
    for entry in asin_data_pilot:
        users = entry["users_sampled"][:N_SAMPLE_USERS]
        rng = random.Random(RANDOM_SEED_USERS)
        rng.shuffle(users)
        for u in users:
            pa = pattrs.get(entry["asin"])
            if pa is None:
                continue
            top_n = get_top_n_attrs(pa, N_INPUT)
            if len(top_n) < N_INPUT:
                continue
            pairs.append({"user_id": u, "asin": entry["asin"], "attrs": top_n})
    return pairs


# ----------------------------------------------------------------------------
# Step 1: Compute per-(user, sentence) Mahalanobis distance to that user's Gaussian
# ----------------------------------------------------------------------------
def compute_user_sentence_distances(
    pairs: list[dict],
    sent_by_user: dict[str, list[dict]],
    gauss: dict,
) -> tuple[dict, dict]:
    """For every (user, sentence) candidate compute d_exemplar_self.

    Excludes sentences from the target ASIN (leakage control).

    Returns:
      pair_exemplars: pair_key → {text, asin, d_self}  (excludes target ASIN)
      user_distances: user_id → [(d_self, text, asin)] sorted
    """
    users_gauss = gauss["users"]
    # Per user: collect all candidate sentences (exclude target ASIN where applicable)
    user_sents = {}  # user_id → list of (text, asin)
    for pd in pairs:
        u = pd["user_id"]
        if u not in users_gauss:
            continue
        target_asin = pd["asin"]
        for sent in sent_by_user.get(u, []):
            if sent["asin"] == target_asin:
                continue  # leakage: skip target-ASIN sentences
            user_sents.setdefault(u, []).append((sent["text"], sent["asin"]))

    # Compute F3+PCA48 z per user
    user_distances = {}  # user_id → [(d_self, text, asin)]
    for u, sents in user_sents.items():
        g = users_gauss[u]
        mu = np.asarray(g["mu"], dtype=np.float64)
        sigma_diag = np.asarray(g["sigma_diag"], dtype=np.float64)
        texts = [s[0] for s in sents]
        feats = extract_318d(texts, gauss["feature_names_ordered"])
        z = project_f3_pca48(feats, gauss)
        ds = mahal_to_user(z, mu, sigma_diag)
        user_distances[u] = [(float(d), t, a) for d, (t, a) in zip(ds, sents)]
        user_distances[u].sort(key=lambda x: x[0])

    # Pair exemplar info: pick one per pair for each condition later
    pair_keys = [(pd["user_id"], pd["asin"]) for pd in pairs]
    return user_distances, {}


# ----------------------------------------------------------------------------
# Step 2: For each condition, build exemplars and run vLLM
# ----------------------------------------------------------------------------
def build_exemplar(pd: dict, condition: str, user_distances: dict, rng: random.Random) -> dict | None:
    """Pick 1 exemplar per condition, deterministic seed per pair+condition."""
    u = pd["user_id"]
    if u not in user_distances or not user_distances[u]:
        return None
    cands = user_distances[u]
    if condition == "random":
        return {"text": rng.choice(cands)[1], "source_asin": rng.choice(cands)[2]}
    if condition == "nearest":
        return {"text": cands[0][1], "source_asin": cands[0][2], "d_self": cands[0][0]}
    if condition == "farthest":
        return {"text": cands[-1][1], "source_asin": cands[-1][2], "d_self": cands[-1][0]}
    raise ValueError(condition)


def run_condition(
    condition: str,
    pairs: list[dict],
    user_distances: dict,
) -> dict:
    log(f"\n=== Condition: {condition} ===")
    pair_data = []
    n_no_sents = 0
    for pd in pairs:
        # Deterministic seed per (user, asin, condition)
        rng = random.Random(hash((pd["user_id"], pd["asin"], condition)) & 0xffffffff)
        ex = build_exemplar(pd, condition, user_distances, rng)
        if ex is None:
            n_no_sents += 1
            continue
        # For random, also compute d_self of the chosen one for transfer analysis
        d_self = None
        if "d_self" in ex:
            d_self = ex["d_self"]
        else:
            for d, t, a in user_distances[pd["user_id"]]:
                if t == ex["text"]:
                    d_self = d
                    break
        pair_data.append({**pd, "exemplar_text": ex["text"],
                          "exemplar_source_asin": ex["source_asin"],
                          "exemplar_d_self": d_self})

    log(f"  pairs with exemplar: {len(pair_data)} (filtered {n_no_sents})")

    # Build prompts
    all_prompts, pair_k = [], []
    for pd in pair_data:
        prompt = make_exemplar_prompt(pd["attrs"], [pd["exemplar_text"]], N_EXEMPLARS)
        for k in range(K_PER_PAIR):
            all_prompts.append(prompt)
            pair_k.append((pd["user_id"], pd["asin"], k))
    n_prompts = len(all_prompts)
    log(f"  total prompts: {n_prompts}")

    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts)
    log(f"  vLLM done in {time.time()-t0:.0f}s")

    # 5-layer strict filter
    pools = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full":0, "invalid":0, "first_p":0, "emoji":0, "self_talk":0, "too_long":0}
    pair_records = []
    for (u, a, k), out_text in zip(pair_k, outputs):
        attrs_dict = next(pd["attrs"] for pd in pair_data if pd["user_id"] == u and pd["asin"] == a)
        text = out_text.strip() if out_text else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs_dict)
        invalid = has_invalid_punct(text)
        first_p = has_first_person(text)
        emoji = has_emoji(text)
        self_talk = has_self_talk(text)
        too_long = is_query_too_long(text)
        strict = (n_cov == N_INPUT) and (not invalid) and first_p and (not emoji) and (not self_talk) and (not too_long)
        if n_cov == N_INPUT: filter_counts["cov_full"] += 1
        if invalid: filter_counts["invalid"] += 1
        if first_p: filter_counts["first_p"] += 1
        if emoji: filter_counts["emoji"] += 1
        if self_talk: filter_counts["self_talk"] += 1
        if too_long: filter_counts["too_long"] += 1
        if strict:
            n_total_strict += 1
        pools[a].append({"user_id":u, "k":k, "query":text, "strict":strict,
                         "attrs_covered":n_cov, "invalid":invalid,
                         "first_person":first_p, "emoji":emoji,
                         "self_talk":self_talk, "too_long":too_long,
                         "n_tok":n_tokens_simple(text)})
        # Per-pair record (one per k)
        pd_obj = next(pd for pd in pair_data if pd["user_id"] == u and pd["asin"] == a)
        pair_records.append({
            "user_id": u, "asin": a, "k": k,
            "exemplar_text": pd_obj["exemplar_text"],
            "exemplar_source_asin": pd_obj["exemplar_source_asin"],
            "exemplar_d_self": pd_obj["exemplar_d_self"],
            "query": text,
            "strict": strict,
        })

    log(f"  filter: {filter_counts}")
    log(f"  total strict: {n_total_strict}")

    out_path = SCRATCH / f"pool_pca48_selection_{condition}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"description": f"Phase 4.C — {condition}",
                       "N_EXEMPLARS": N_EXEMPLARS, "K_PER_PAIR": K_PER_PAIR},
            "filter_counts": filter_counts,
            "n_total_strict": n_total_strict,
            "pools": pools,
            "pair_records": pair_records,
        }, f, ensure_ascii=False)
    log(f"  wrote → {out_path.name}")
    return {"condition": condition, "n_pairs": len(pair_data), "n_strict": n_total_strict,
            "pool_path": str(out_path)}


def main():
    log("=== Phase 4.C — PCA48-nearest exemplar selection ===")
    pairs = get_pairs_and_attrs()
    log(f"  pilot pairs: {len(pairs)}")
    sent_by_user = load_user_sentences()
    log(f"  sentences map: {len(sent_by_user)} users")

    # Load user Gaussians (filter to cohort)
    gauss = json.load(open(GAUSS_PATH))
    cohort_users = {pd["user_id"] for pd in pairs}
    gauss["users"] = {u: g for u, g in gauss["users"].items() if u in cohort_users}
    log(f"  cohort Gaussians loaded: {len(gauss['users'])}")

    # Step 1: compute per-(user, sentence) distances
    log("\n--- Step 1: compute d_exemplar_self per (user, sentence) ---")
    t0 = time.time()
    user_distances, _ = compute_user_sentence_distances(pairs, sent_by_user, gauss)
    log(f"  done in {time.time()-t0:.1f}s")
    sample_count = sum(len(v) for v in user_distances.values())
    log(f"  total (user, sentence) candidates: {sample_count}")

    # Step 2: generate for each condition
    log("\n--- Step 2: generate queries per condition ---")
    cond_results = {}
    for cond in SELECTION_CONDITIONS:
        cond_results[cond] = run_condition(cond, pairs, user_distances)

    summary = {
        "config": {
            "PILOT_N_ASIN": PILOT_N_ASIN, "N_SAMPLE_USERS": N_SAMPLE_USERS,
            "K_PER_PAIR": K_PER_PAIR, "N_EXEMPLARS": N_EXEMPLARS, "TEMP": TEMP,
            "leakage_control": "exclude user sentences from target ASIN",
            "selection_conditions": SELECTION_CONDITIONS,
        },
        "results": cond_results,
        "next_step": "Phase 4.C-2: Stage 4 strict alignment on each pool, "
                     "compute per-pair transfer_gain, ρ(exemplar, query d_self)",
    }
    out = REPO_ROOT / "result/gen_query/pca48_selection_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {out}")


if __name__ == "__main__":
    main()