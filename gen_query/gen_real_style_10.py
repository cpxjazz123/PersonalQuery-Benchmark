#!/usr/bin/env python3
"""Build real user style vectors for 10 test users, then generate 10 queries.

Steps:
1. Compute per-user mean residual (layer 16/20/24/26) from residual_hidden_10k.npz
2. Build user_style_vectors_real.jsonl for 10 users
3. Generate K=4 candidates for 10 users via generate_strict_nohallu.py logic
4. Compare with stub-zero outputs

Reads:  residual_hidden_10k.npz + rewrites_10k.jsonl (from gaussian_vades/)
Writes: user_style_vectors_real_10u.jsonl + output_10u_strict.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
SCRATCH_G = SCRATCH / "gaussian_vades"
SCRATCH_U = SCRATCH / "user_style_steering"
SCRATCH_U.mkdir(parents=True, exist_ok=True)

REWRITES = SCRATCH_G / "rewrites_10k.jsonl"
NPZ_FILE = SCRATCH_G / "residual_hidden_10k.npz"
STYLE_OUT = SCRATCH_U / "user_style_vectors_real_10u.jsonl"
RECORDS_IN = REPO / "result/query_records_10k.json"
GEN_OUT = SCRATCH_U / "output_10u_strict.json"

LAYERS = [16, 20, 24, 26]
GEN_LAYER = 16  # injection layer for generation
GEN_ALPHA = 1.0  # REAL injection
GEN_K = 4
MAX_RECORDS = 10  # only 10 users


def build_style_vectors() -> dict[str, dict[int, list[float]]]:
    """Compute per-user mean residual and write style vector file."""
    print("[load] sentence→uid from rewrites_10k.jsonl (dedup)...")
    sent_to_uid: dict[str, str] = {}
    with open(REWRITES, "r") as f:
        for line in f:
            r = json.loads(line)
            sent = r.get("sentence_text", "")
            uid = r.get("user_id", "")
            if sent and sent not in sent_to_uid:
                sent_to_uid[sent] = uid
    print(f"  {len(sent_to_uid)} unique sentences, {len(set(sent_to_uid.values()))} users")

    print(f"[load] residuals from {NPZ_FILE} ...")
    npz = np.load(NPZ_FILE, allow_pickle=True)
    sentences = npz["sentences"].tolist()
    layer_resids = {L: npz[f"residual_layer_{L}"] for L in LAYERS}
    print(f"  shape per layer: {layer_resids[LAYERS[0]].shape}")

    print("[group] per-user mean residual per layer...")
    user_layer_vecs: dict[str, dict[int, list[list[float]]]] = {
        uid: {L: [] for L in LAYERS} for uid in set(sent_to_uid.values())
    }
    n_mapped = 0
    for sent_idx, sent in enumerate(sentences):
        uid = sent_to_uid.get(sent)
        if uid is None:
            continue
        for L in LAYERS:
            user_layer_vecs[uid][L].append(layer_resids[L][sent_idx])
        n_mapped += 1
    print(f"  mapped {n_mapped} / {len(sentences)} sentences")

    # Write only first 10 users
    user_ids = sorted(user_layer_vecs.keys())[:MAX_RECORDS]
    print(f"[write] {len(user_ids)} users → {STYLE_OUT}")
    with open(STYLE_OUT, "w", encoding="utf-8") as fh:
        for uid in user_ids:
            row = {"user_id": uid}
            for L in LAYERS:
                vecs = user_layer_vecs[uid][L]
                if not vecs:
                    mean_vec = [0.0] * 3584
                else:
                    arr = np.stack(vecs, axis=0).astype(np.float32)
                    mean_vec = arr.mean(axis=0).tolist()
                row[f"residual_mean_layer_{L}"] = mean_vec
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  written → {STYLE_OUT} ({STYLE_OUT.stat().st_size // 1024} KB)")

    # Return for immediate use
    result = {}
    for uid in user_ids:
        result[uid] = {L: np.array(user_layer_vecs[uid][L]).mean(axis=0).astype(np.float32) for L in LAYERS}
    return result


def generate(user_vecs: dict[str, dict[int, np.ndarray]]) -> list[dict]:
    """Generate queries for 10 users using real style injection."""
    print("[generate] loading records + Qwen...")
    records = json.load(open(RECORDS_IN))
    records = [r for r in records if r["user_id"] in user_vecs][:MAX_RECORDS]
    print(f"  {len(records)} records to generate")

    sys.path.insert(0, str(REPO))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    hidden_dim = client._hidden_backend.model.config.hidden_size
    print(f"  hidden_dim={hidden_dim}")

    # Load style vectors (reload from file to confirm format)
    profiles = {}
    with open(STYLE_OUT, "r") as f:
        for line in f:
            row = json.loads(line)
            uid = row["user_id"]
            profiles[uid] = {}
            for L in LAYERS:
                key = f"residual_mean_layer_{L}"
                if key in row:
                    profiles[uid][L] = row[key]

    print(f"  loaded {len(profiles)} user profiles (real style vectors)")

    # Build prompts and injections
    from gen_query.generate_strict_nohallu import make_prompt
    results = []
    for r in records:
        uid, asin = r["user_id"], r["asin"]
        attrs = r.get("attrs_used", {})
        prof = profiles.get(uid, {})
        bias = np.array(prof.get(GEN_LAYER, [0.0] * hidden_dim), dtype=np.float32)
        print(f"  uid={uid[:8]}... bias_norm={float((bias**2).sum()**0.5):.2f}")

        # K samples
        for k in range(GEN_K):
            # make_prompt already replaces {ATTRIBUTES} with formatted attrs
            prompt = make_prompt(attrs, variant_idx=k + 1)
            import torch
            bias_t = torch.as_tensor(bias, dtype=torch.float32)
            # Pass empty system so make_prompt's fully-formatted output goes into user role
            queries = client.generate_with_hidden_injection(
                system_text="You are a helpful assistant.",
                user_texts=[prompt],
                injection_per_row=[bias_t],
                injection_layers=[GEN_LAYER],
                injection_alpha=GEN_ALPHA,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                batch_size=1,
                max_input_length=256,
                mask_cjk=True,
            )
            results.append({
                "user_id": uid,
                "asin": asin,
                "attrs": attrs,
                "sample_idx": k,
                "query": queries[0],
                "bias_norm": float((bias**2).sum()**0.5),
            })
    return results


def main() -> None:
    t0 = time.time()
    user_vecs = build_style_vectors()
    results = generate(user_vecs)

    with open(GEN_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n[done] {len(results)} queries in {time.time()-t0:.1f}s → {GEN_OUT}")
    # Print summary
    for r in results:
        print(f"\nuid={r['user_id'][:8]} sample={r['sample_idx']} bias_norm={r['bias_norm']:.1f}")
        print(f"  attrs: {list(r['attrs'].values())[:3]}")
        print(f"  query: {r['query'][:200]}")


if __name__ == "__main__":
    main()
