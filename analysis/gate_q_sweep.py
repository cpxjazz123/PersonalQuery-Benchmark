"""Gate Q sweep analysis: 对同一份编码数据，测试不同 FILTER_Q 阈值对 selection 数量的影响。

复用 syntax_select_mahalanobis_gate.py 的数据加载和编码逻辑，
仅修改 FILTER_Q 值，统计每个阈值下的 selection 数量。

用法 (无参数):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY analysis/gate_q_sweep.py

输出: result/08_select_query/gate_q_sweep.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.linalg
from collections import defaultdict
import spacy
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ATTRS_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/gate_q_sweep.json"

SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLE = ["ner", "textcat", "lemmatizer"]
ENCODE_DEVICE = "cuda:0"
ENCODE_CHUNK_SIZE = 1024
SPACY_BATCH_SIZE = 256
D2_BOUNDARY_RECHECK_TOL = 0.25

# Sweep Q values
GATE_Q_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(0)


_pcfg = None


def _load_pcfg():
    global _pcfg
    if _pcfg is None:
        spec = importlib.util.spec_from_file_location(
            "syntax_pcfg_pipeline",
            REPO_ROOT / "03_spacy_encode/syntax_pcfg_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pcfg = mod
    return _pcfg


def load_encoder():
    encoder_path = CACHE_DIR / "strict3_encoder.pt"
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    _SupEncoder = _load_pcfg()._SupEncoder
    model = _SupEncoder(
        cfg["vocab_size"], cfg["z_dim"], tuple(cfg["hidden"]),
        cfg["n_users"], cfg["dropout"]
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(torch.device(ENCODE_DEVICE))
    model.eval()
    return model


def encode_texts(texts, nlp, rule_to_id, vocab_size, encoder):
    extract_struct_rules = _load_pcfg().extract_struct_rules
    if not texts:
        return np.empty((0, 16), dtype=np.float32)
    chunks = []
    total = len(texts)
    for start in range(0, total, ENCODE_CHUNK_SIZE):
        chunk = texts[start:start + ENCODE_CHUNK_SIZE]
        counts = np.zeros((len(chunk), vocab_size), dtype=np.float32)
        for i, doc in enumerate(nlp.pipe(chunk, batch_size=SPACY_BATCH_SIZE)):
            for rule in extract_struct_rules(doc):
                j = rule_to_id.get(rule)
                if j is not None:
                    counts[i, j] = 1.0
        counts *= 0.5
        ct = torch.from_numpy(counts).to(torch.device(ENCODE_DEVICE), dtype=torch.float32)
        with torch.inference_mode():
            z, _ = encoder(ct)
        chunks.append(z.detach().cpu().numpy().astype(np.float32))
        del ct, counts, z
    return np.concatenate(chunks, axis=0)


def maha_d2_one(z, mu, inv_sigma):
    d = z - mu
    return float(d @ inv_sigma @ d)


def load_gauss_and_cohort(filter_q):
    """加载 Stage 04 Gaussian 和指定 Q 的 cohort gates。"""
    q_int = int(filter_q * 100)
    cohort_gates_path = REPO_ROOT / f"result/04_gaussian/cohort_gates_q{q_int:02d}.json"
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    with open(cohort_gates_path) as f:
        cohort_gates_raw = json.load(f)

    users_all = stage04.get("users", {})
    q_key = f"d2_q{q_int:02d}"

    # Build valid user set
    valid_uids = set()
    for uid, stats in users_all.items():
        if not isinstance(stats, dict) or "sigma_inv" not in stats:
            continue
        mu = stats.get("mu")
        si = stats.get("sigma_inv")
        gate_t = stats.get(q_key)
        if not isinstance(mu, list) or len(mu) not in (16, 32):
            continue
        if not isinstance(si, list) or len(si) not in (16, 32):
            continue
        if gate_t is None or not np.isfinite(float(gate_t)) or float(gate_t) < 0:
            continue
        flat = [float(x) for row in si for x in row]
        if not all(np.isfinite(x) for x in flat):
            continue
        if not all(np.isfinite(float(x)) for x in mu):
            continue
        valid_uids.add(uid)

    # Build gauss cache
    gauss_cache = {}
    for uid in valid_uids:
        stats = users_all[uid]
        mu = np.asarray(stats["mu"], dtype=np.float32)
        inv_sigma = np.asarray(stats["sigma_inv"], dtype=np.float32)
        gate_T = float(stats[q_key])
        gauss_cache[uid] = {
            "mu": mu,
            "inv_sigma": inv_sigma,
            "gate_T": gate_T,
            "d2_q50": float(stats.get("d2_q50", 0)),
        }

    # Build cohort gates per ASIN
    cohort_gates = {}
    for asin, source_cohort in cohort_gates_raw.items():
        if not isinstance(source_cohort, dict):
            continue
        fitted_uids = [uid for uid in source_cohort if uid in valid_uids]
        if len(fitted_uids) < 2:
            continue
        cohort_gates[asin] = {}
        for uid in fitted_uids:
            gate = source_cohort[uid]
            if not isinstance(gate, dict) or "gate_T" not in gate:
                continue
            cohort_gates[asin][uid] = float(gate["gate_T"])

    return gauss_cache, cohort_gates


def count_selections_for_q(Z_unique, asin_text_indices, pool_entries, gauss_cache, cohort_gates):
    """对给定 Q 的 gauss_cache 和 cohort_gates，统计 selection 数量。"""
    selections = 0
    asins_with_selection = set()
    users_with_selection = defaultdict(int)

    for entry in pool_entries:
        asin = entry["asin"]
        cohort = cohort_gates.get(asin)
        if cohort is None:
            continue
        cohort_uids = list(cohort.keys())
        if len(cohort_uids) < 2:
            continue

        Z_q = Z_unique[asin_text_indices[asin]]
        n_cand = Z_q.shape[0]

        # Precompute D² matrix
        mu = np.stack([gauss_cache[uid]["mu"] for uid in cohort_uids], axis=0)
        inv_sigma = np.stack([gauss_cache[uid]["inv_sigma"] for uid in cohort_uids], axis=0)
        gate_Ts = np.asarray([cohort[uid] for uid in cohort_uids], dtype=np.float64)

        diff = Z_q[:, None, :] - mu[None, :, :]
        left = np.einsum("cud,ude->cue", diff, inv_sigma)
        d2_matrix = np.sum(left * diff, axis=2, dtype=np.float32)

        # Recheck boundary
        near_gate = np.abs(d2_matrix - gate_Ts[None, :]) <= D2_BOUNDARY_RECHECK_TOL
        boundary_pairs = np.argwhere(near_gate)
        for candidate_i, cohort_i in boundary_pairs:
            ci = int(cohort_i)
            d2_matrix[int(candidate_i), ci] = maha_d2_one(
                Z_q[int(candidate_i)],
                gauss_cache[cohort_uids[ci]]["mu"],
                gauss_cache[cohort_uids[ci]]["inv_sigma"]
            )

        uid_to_idx = {uid: i for i, uid in enumerate(cohort_uids)}

        for uid in cohort_uids:
            target_i = uid_to_idx[uid]
            target_d2 = d2_matrix[:, target_i]
            gate_T = float(gate_Ts[target_i])
            target_inside = target_d2 <= gate_T
            competitor_inside = d2_matrix <= gate_Ts[None, :]
            competitor_inside[:, target_i] = False
            pass_unique = target_inside & (~competitor_inside.any(axis=1))
            if pass_unique.any():
                selections += 1
                asins_with_selection.add(asin)
                users_with_selection[uid] += 1

    return {
        "n_selections": selections,
        "n_asins_with_selection": len(asins_with_selection),
        "n_users_with_selection": len(users_with_selection),
        "total_asins_processed": len(cohort_gates),
    }


def main():
    t0 = time.time()
    log("=== Gate Q Sweep ===")
    require_cuda()

    # Load encoder + vocab once
    encoder = load_encoder()
    V = encoder.encoder[0].in_features
    with open(CACHE_DIR / "vocab.json") as f:
        vocab = json.load(f)
    rule_to_id = {r: i for i, r in enumerate(vocab)}
    nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLE)

    # Load pool
    with open(POOL_PATH) as f:
        pool_data = json.load(f)
    pool_queries = pool_data.get("pool", pool_data) if isinstance(pool_data, dict) else pool_data
    with open(ATTRS_PATH) as f:
        attrs_all = json.load(f)

    pool_entries = []
    for asin, queries in pool_queries.items():
        cands = [{"text": t} for t in queries]
        pool_entries.append({"asin": asin, "candidates": cands})

    # Global dedup + encode once
    unique_text_to_idx = {}
    unique_texts = []
    asin_text_indices = {}
    for entry in pool_entries:
        asin = entry["asin"]
        indices = []
        for c in entry["candidates"]:
            t = c["text"]
            if t not in unique_text_to_idx:
                unique_text_to_idx[t] = len(unique_texts)
                unique_texts.append(t)
            indices.append(unique_text_to_idx[t])
        asin_text_indices[asin] = np.asarray(indices, dtype=np.int64)

    log(f"  encoding {len(unique_texts)} unique texts ...")
    Z_unique = encode_texts(unique_texts, nlp, rule_to_id, V, encoder)
    z_dim = encoder.z_dim if hasattr(encoder, 'z_dim') else 16
    log(f"  encoded shape: {Z_unique.shape}")

    # Sweep each Q
    results = {}
    for q in GATE_Q_VALUES:
        log(f"  Sweeping Q={q:.2f} ...")
        gauss_cache, cohort_gates = load_gauss_and_cohort(q)
        stats = count_selections_for_q(
            Z_unique, asin_text_indices, pool_entries, gauss_cache, cohort_gates
        )
        results[f"Q_{q:.2f}"] = {
            "Q": q,
            "n_selections": stats["n_selections"],
            "n_asins_with_selection": stats["n_asins_with_selection"],
            "n_users_with_selection": stats["n_users_with_selection"],
            "total_asins_in_cohort": stats["total_asins_processed"],
        }
        log(f"    selections={stats['n_selections']}, "
            f"asins={stats['n_asins_with_selection']}, "
            f"users={stats['n_users_with_selection']}")

    # Write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"gate_q_values": GATE_Q_VALUES},
        "results": results,
    }
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.replace(OUT_PATH)
    log(f"  wrote → {OUT_PATH}")
    log(f"=== DONE in {time.time()-t0:.1f}s ===")

    # Print summary table
    print()
    print(f"{'Q':>6}  {'Selections':>12}  {'ASINs':>8}  {'Users':>8}  {'ASINs/Cohort':>14}")
    print("-" * 55)
    for q in GATE_Q_VALUES:
        r = results[f"Q_{q:.2f}"]
        pct = r["n_asins_with_selection"] / max(r["total_asins_in_cohort"], 1) * 100
        print(f"  {q:.2f}  {r['n_selections']:>12}  {r['n_asins_with_selection']:>8}  "
              f"{r['n_users_with_selection']:>8}  {pct:>13.1f}%")


if __name__ == "__main__":
    main()
