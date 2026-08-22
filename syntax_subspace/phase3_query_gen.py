#!/usr/bin/env python3
"""Phase 3: query generation with syntax-only Δh injection.

复用 generate_strict_nohallu.py 的 K-sample + best-of-K + no-hallucination 框架,
但 bias 计算改成 syntax-only Δh (从 syntax_gaussian_layer_*.npz)。

Δh 计算:
  Δh_syntax = z_u @ U_keep   (n_keep=4-5, U_keep (n_keep, 3584))
  z_u 模式 (env SYNTAX_Z_MODE):
    - mean: z_u = mu_u (deterministic)
    - sample: z_u ~ N(mu_u, Sigma_u) per K-sample
  Layer 20 (n_keep=5) 默认;其它 layer env var 覆盖
  α: SYNTAX_ALPHA (mean 模式用 0.3-0.5;sample 模式用 0.05-0.15 因为 sampled norm 更大)

硬编码输出:
  result/query_records_with_query_inject_syntax_L20_a0.3.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result"

sys.path.insert(0, str(REPO_ROOT))

# === 硬编码路径 ===
RECORDS_IN = REPO_ROOT / "result/query_records.json"
SYNTAX_GAUSSIAN_NPZ_FMT = str(SCRATCH / "syntax_gaussian_layer_{layer}.npz")
OUT_SUFFIX = os.environ.get("SYNTAX_OUT_SUFFIX", "syntax_L20_a0.3")
RECORDS_OUT = OUT_DIR / f"query_records_with_query_inject_{OUT_SUFFIX}.json"

# === 实验参数 (env var 覆盖) ===
SYNTAX_LAYER = int(os.environ.get("SYNTAX_LAYER", "20"))
SYNTAX_Z_MODE = os.environ.get("SYNTAX_Z_MODE", "mean")  # mean | sample
SYNTAX_ALPHA = float(os.environ.get("SYNTAX_ALPHA", "0.3"))
K_SAMPLES = int(os.environ.get("SYNTAX_K", "4"))
GEN_BATCH = int(os.environ.get("SYNTAX_GEN_BATCH", "4"))
GEN_MAX_NEW = int(os.environ.get("SYNTAX_MAX_NEW", "128"))
GEN_TEMP = float(os.environ.get("SYNTAX_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("SYNTAX_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("SYNTAX_REP_PENALTY", "1.1"))
MAX_RECORDS = int(os.environ.get("SYNTAX_MAX_RECORDS", "10"))
MAX_INPUT_LENGTH = 384
MASK_CJK = os.environ.get("SYNTAX_MASK_CJK", "1") == "1"
SAMPLE_SEED = int(os.environ.get("SYNTAX_SAMPLE_SEED", "42"))


# === STRICT v8 prompt (与 generate_strict_nohallu.py 一致) ===
GEN_SYSTEM = (
    "You are an Amazon shopper writing a search query. Use EVERY attribute "
    "value below verbatim (mention duplicates only once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write one natural sentence (20-35 words) with varied syntax — clauses, "
    "coordination, prepositional phrases, relative clauses. Output ONLY the "
    "query, no preamble.\n\n"
    "Attributes:\n{ATTRIBUTES}"
)


# === Hallucination detection (与 generate_strict_nohallu.py 一致) ===
HALLUCINATION_WORDS = [
    "bottle", "bottles", "clothing", "clothes", "toy", "toys", "candle",
    "candles", "tumbler", "tumblers", "carrier", "carriers", "basket",
    "baskets", "socks", "shoes", "shirt", "shirts", "pants", "dress",
    "dresses", "blanket", "blankets", "pillow", "pillows", "diaper",
    "diapers", "wipes", "formula", "pacifier", "stroller", "highchair",
    "swaddle", "swaddles", "romper", "rompers", "onesie", "onesies",
    "mittens", "booties", "teether", "teethers", "rattle", "rattles",
    "mobile", "mobiles", "nightlight", "soap", "lotion", "shampoo",
    "brush", "comb", "drinkware", "beverage", "container", "holder",
    "vessel", "equipment", "gear", "appliance", "utensil", "dish",
    "sneaker", "sneakers", "sandal", "sandals", "boot", "boots",
    "headband", "headbands", "hairband", "hairbands", "cap", "hat",
    "jacket", "coats", "vest", "shorts", "skirt", "legging", "leggings",
    "scented", "scent", "fragrance", "aroma", "flavor", "flavour",
    "decorative", "decoration", "ornament", "ornamental",
    "friend", "friends", "sister", "sister's", "brother", "brother's",
    "wife", "wife's", "husband", "husband's", "mom", "mom's", "dad",
    "dad's", "mother", "father", "daughter", "son", "grandma",
    "grandpa", "aunt", "uncle", "cousin", "neighbor",
    "lightweight", "heavyweight", "premium", "luxury", "cheap",
    "professional-grade", "commercial", "industrial",
    "handmade", "handcrafted", "artisan",
    "accessory", "accessories", "tool", "tools", "decor", "gadget",
    "gizmo", "implement", "device", "apparatus",
]


def build_user_content(attrs: dict) -> str:
    lines = []
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(f"{k}: {s}")
    return "\n".join(lines)


def make_prompt(attrs: dict) -> str:
    return GEN_SYSTEM.replace("{ATTRIBUTES}", build_user_content(attrs))


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not attrs:
        return 0
    text_lower = text.lower()
    return sum(1 for v in attrs.values()
               if v and str(v).strip() and str(v).strip().lower() in text_lower)


def has_hallucination(text: str, attrs: dict) -> list[str]:
    text_lower = text.lower()
    attr_substrings = set()
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s:
            attr_substrings.add(s.lower())
            for tok in re.split(r"[\s,/]+", s.lower()):
                if len(tok) >= 4:
                    attr_substrings.add(tok)
    hits = []
    for w in HALLUCINATION_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text_lower) and w not in attr_substrings:
            hits.append(w)
    return hits


def has_invalid_punct(text: str) -> bool:
    return bool(re.search(r'["\(\)“”]', text))


def score_candidate(text: str, attrs: dict) -> tuple[int, int, int]:
    n_cov = count_attrs_covered(text, attrs)
    n_hallu = len(has_hallucination(text, attrs))
    invalid = 1 if has_invalid_punct(text) else 0
    return (n_cov, -n_hallu, -invalid)


class SyntaxGaussianProvider:
    """从 syntax_gaussian_layer_*.npz 加载, 返回 per-user Δh_syntax (3584,)。"""

    def __init__(self, layer: int, z_mode: str, sample_seed: int):
        import numpy as np
        self.z_mode = z_mode
        self.layer = layer
        npz = np.load(SYNTAX_GAUSSIAN_NPZ_FMT.format(layer=layer), allow_pickle=True)
        self.user_ids = [str(u) for u in npz["user_ids"]]
        self.mu = npz["mu"]  # (n_users, n_keep)
        self.sigma = npz["sigma"]  # (n_users, n_keep, n_keep)
        self.U_keep = npz["U_keep"]  # (n_keep, 3584)
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.rng = np.random.default_rng(sample_seed)
        print(f"[provider] layer={layer}, z_mode={z_mode}, n_users={len(self.user_ids)}, "
              f"n_keep={self.mu.shape[1]}, U_keep.shape={self.U_keep.shape}")

    def get_bias(self, user_id: str, layer_unused: int = None) -> "torch.Tensor | None":
        import torch
        idx = self.uid_to_idx.get(user_id)
        if idx is None:
            return None
        mu_u = self.mu[idx]
        if self.z_mode == "mean":
            z_u = mu_u
        else:  # sample
            z_u = self.rng.multivariate_normal(mu_u, self.sigma[idx])
        delta_h = z_u @ self.U_keep  # (3584,)
        return torch.as_tensor(delta_h, dtype=torch.float32)


def main() -> int:
    import torch
    t0 = time.time()

    # 1) Qwen
    print("[main] loading Qwen...", flush=True)
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    hidden_dim = client._hidden_backend.model.config.hidden_size
    print(f"[main] hidden_dim={hidden_dim}, layer={SYNTAX_LAYER}, "
          f"z_mode={SYNTAX_Z_MODE}, alpha={SYNTAX_ALPHA}, K={K_SAMPLES}")

    # 2) Provider
    provider = SyntaxGaussianProvider(SYNTAX_LAYER, SYNTAX_Z_MODE, SAMPLE_SEED)
    inject_layers = [SYNTAX_LAYER]

    # 3) Records
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[main] {len(records)} records", flush=True)

    # 4) 增量
    existing: dict[tuple[str, str], str] = {}
    if RECORDS_OUT.exists():
        prev = json.load(open(RECORDS_OUT, "r", encoding="utf-8"))
        for r in prev:
            if r.get("y_plus_query_inject") and "user_id" in r and "asin" in r:
                existing[(r["user_id"], r["asin"])] = r["y_plus_query_inject"]
        print(f"[main] {len(existing)} already-generated", flush=True)

    # 5) todo
    todo_records, todo_prompts, todo_injections, todo_sample_idx = [], [], [], []
    n_no_profile = 0
    for r in records:
        uid, asin = r["user_id"], r["asin"]
        if (uid, asin) in existing:
            continue
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        for k in range(K_SAMPLES):
            todo_records.append(r)
            todo_prompts.append(make_prompt(attrs))
            bias = provider.get_bias(uid, SYNTAX_LAYER)
            if bias is None:
                n_no_profile += 1
                bias = torch.zeros(hidden_dim, dtype=torch.float32)
            todo_injections.append(bias)
            todo_sample_idx.append(k)
    print(f"[main] todo={len(todo_records)} ({len(records) - len(existing)} records × K={K_SAMPLES}), "
          f"no_profile={n_no_profile}", flush=True)

    if not todo_records:
        print("[main] ✓ 无 todo, 退出")
        return 0

    # 6) 输出
    out_records = []
    for r in records:
        if (r["user_id"], r["asin"]) in existing:
            r2 = dict(r)
            r2["y_plus_query_inject"] = existing[(r["user_id"], r["asin"])]
            out_records.append(r2)
        else:
            out_records.append(dict(r))

    # 7) 分批生成
    candidates_by_key: dict[tuple[str, str], list[str]] = {}
    new_records_by_key: dict[tuple[str, str], dict] = {}
    bias_norms = []
    for i in range(0, len(todo_prompts), GEN_BATCH):
        chunk_p = todo_prompts[i:i + GEN_BATCH]
        chunk_r = todo_records[i:i + GEN_BATCH]
        chunk_inj = todo_injections[i:i + GEN_BATCH]
        bias_norms.extend(float(b.norm()) for b in chunk_inj)
        try:
            queries = client.generate_with_hidden_injection(
                system_text=GEN_SYSTEM,
                user_texts=chunk_p,
                injection_per_row=chunk_inj,
                injection_layers=inject_layers,
                injection_alpha=SYNTAX_ALPHA,
                max_new_tokens=GEN_MAX_NEW,
                temperature=GEN_TEMP,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REP_PENALTY,
                batch_size=GEN_BATCH,
                max_input_length=MAX_INPUT_LENGTH,
                mask_cjk=MASK_CJK,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise
        for r, q in zip(chunk_r, queries):
            key = (r["user_id"], r["asin"])
            candidates_by_key.setdefault(key, []).append(q)
            new_records_by_key[key] = r
        done = min(i + GEN_BATCH, len(todo_prompts))
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (len(todo_prompts) - done) / max(rate, 1e-6)
        print(f"[main] {done}/{len(todo_prompts)} ({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)

    print(f"[main] bias norm stats: min={min(bias_norms):.1f}, "
          f"max={max(bias_norms):.1f}, mean={sum(bias_norms)/len(bias_norms):.1f}")

    # 8) best-of-K
    n_with_inject = 0
    n_all_hallu = 0
    for key, cands in candidates_by_key.items():
        r = new_records_by_key[key]
        attrs = r.get("attrs_used", {})
        scored = [(score_candidate(c, attrs), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_q = scored[0]
        n_cov, neg_hallu, neg_inv = best_score
        n_hallu = -neg_hallu
        r["y_plus_query_inject"] = best_q
        r["n_candidates"] = len(cands)
        r["n_attrs_covered"] = n_cov
        r["n_hallucinations"] = n_hallu
        r["hallucination_words"] = has_hallucination(best_q, attrs)
        r["all_candidates"] = cands
        r["all_candidates_scores"] = [list(s) for s, _ in scored]
        r["syntax_layer"] = SYNTAX_LAYER
        r["syntax_z_mode"] = SYNTAX_Z_MODE
        r["syntax_alpha"] = SYNTAX_ALPHA
        r["syntax_n_keep"] = int(provider.mu.shape[1])
        n_with_inject += 1
        if n_hallu > 0:
            n_all_hallu += 1

    # 9) fill back
    for key, r in new_records_by_key.items():
        for orec in out_records:
            if orec["user_id"] == r["user_id"] and orec["asin"] == r["asin"]:
                orec.update(r)
                break

    # 10) 写出
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(RECORDS_OUT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[main] ✓ wrote {RECORDS_OUT}: {n_with_inject} records")
    print(f"[main]   records with hallucinations: {n_all_hallu}/{n_with_inject}")
    print(f"[main] total {time.time()-t0:.1f}s")

    # 11) per-record detail
    for key in sorted(candidates_by_key.keys()):
        r = new_records_by_key[key]
        attrs = r.get("attrs_used", {})
        best_q = r["y_plus_query_inject"]
        cands = r["all_candidates"]
        print(f"\n--- asin={r['asin']} (user={r['user_id'][:14]}) ---")
        print(f"  attrs ({len(attrs)}): {dict(list(attrs.items()))}")
        print(f"  BEST  (cov={r['n_attrs_covered']}/5 hallu={r['n_hallucinations']}): {best_q}")
        if r["hallucination_words"]:
            print(f"         hallucination: {r['hallucination_words']}")
        for i, c in enumerate(cands):
            cov = count_attrs_covered(c, attrs)
            hallu = has_hallucination(c, attrs)
            mark = " ★" if c == best_q else "  "
            print(f"  {mark} cand{i+1} (cov={cov}/5 hallu={len(hallu)}): {c}")
            if hallu:
                print(f"            hallucination: {hallu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
