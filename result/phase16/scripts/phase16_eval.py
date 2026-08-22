#!/usr/bin/env python3
"""Phase 16 Eval — 从 phase16_n30_gaussian_sweep.jsonl (2160 records, 9 conds)
   跑 style margin + semantic sim + paired bootstrap diff vs D_off & A_a1.0."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
JSONL_PATH = OUT_DIR / "phase16_n30_gaussian_sweep.jsonl"
OUT_EVAL = OUT_DIR / "phase16_n30_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase16_n30_per_pair.jsonl"
USER_EMB_NPZ = OUT_DIR / "phase10_user_embs_768d.npz"

BASELINES = ["D_off", "A_a1.0"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    log("=" * 70)
    log("Phase 16 EVAL — 9 conds × 30 pairs × K=8 = 2160 records")
    log("=" * 70)

    log("[1] Loading records ...")
    records: list[dict] = []
    with JSONL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log(f"  loaded: {len(records)} records")
    conds = sorted({r["condition"] for r in records})
    log(f"  conds ({len(conds)}): {conds}")

    log("[2] Loading 876 user 768d embs ...")
    user_data = np.load(str(USER_EMB_NPZ), allow_pickle=True)
    user_embs = user_data["embs"]
    user_ids = list(user_data["user_ids"])
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    user_embs_norm = l2_normalize(user_embs.astype(np.float32))
    log(f"  user_embs: {user_embs.shape}, {len(user_ids)} users")

    log("[3] Loading AnnaWegmann style-embedding model ...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")
    style_model = AutoModel.from_pretrained(
        "AnnaWegmann/Style-Embedding", trust_remote_code=True
    ).to("cuda:0")
    style_model.eval()

    cand_texts = [r["q_final_post"] for r in records]
    encoded: list[np.ndarray] = []
    bs = 64
    with torch.no_grad():
        for st in range(0, len(cand_texts), bs):
            chunk = cand_texts[st:st + bs]
            inp = tokenizer(
                chunk, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            inp = {k: v.to("cuda:0") for k, v in inp.items()}
            o = style_model(**inp)
            encoded.append(o.pooler_output.cpu().numpy())
    cand_embs = np.concatenate(encoded, axis=0)
    log(f"  cand_embs: {cand_embs.shape}")

    log("[4] Per-cand margin (cos_target - cos_off_mean) ...")
    cand_norm = l2_normalize(cand_embs.astype(np.float32))
    margin_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    skipped = 0
    for i, r in enumerate(records):
        uid = r["user_id"]
        if uid not in uid_to_idx:
            skipped += 1
            continue
        ti = uid_to_idx[uid]
        target_emb = user_embs_norm[ti]
        cos_target = float(np.dot(cand_norm[i], target_emb))
        off_idxs = np.array([j for j in range(len(user_ids)) if j != ti])
        cos_off_mean = float(np.mean(cand_norm[i].dot(user_embs_norm[off_idxs].T)))
        margin_by_key[(uid, r["condition"])].append(cos_target - cos_off_mean)
    if skipped:
        log(f"  skipped {skipped} cands (uid not in 876u set)")

    log("[5] Per-pair best-of-K margin ...")
    cond_to_bok: dict[str, list[float]] = defaultdict(list)
    cond_to_margins: dict[str, list[float]] = defaultdict(list)
    for (uid, cond_name), ms in margin_by_key.items():
        if not ms:
            continue
        cond_to_margins[cond_name].extend(ms)
        cond_to_bok[cond_name].append(max(ms))
    log(f"  per-cond best-of-K: " + ", ".join(
        f"{c}={len(v)}" for c, v in sorted(cond_to_bok.items())))

    log("[6] Loading sentence-bert for semantic ...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"
    )

    pair_uids = sorted({r["user_id"] for r in records})
    pair_attrs_text: dict[str, str] = {}
    for r in records:
        if r["user_id"] in pair_attrs_text:
            continue
        attrs = r["attrs"]
        pair_attrs_text[r["user_id"]] = ", ".join(
            f"{k}: {v}" for k, v in attrs.items() if v
        )
    attr_texts = [pair_attrs_text[u] for u in pair_uids]
    log(f"  encoding {len(pair_uids)} attr prompts ...")
    attr_embs = sbert.encode(
        attr_texts, convert_to_numpy=True, normalize_embeddings=True
    )
    cand_sbert = sbert.encode(
        cand_texts, convert_to_numpy=True, normalize_embeddings=True
    )
    uid_to_attr = {u: i for i, u in enumerate(pair_uids)}
    sem_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    cond_to_sem: dict[str, list[float]] = defaultdict(list)
    for i, r in enumerate(records):
        a = uid_to_attr[r["user_id"]]
        sem = float(np.dot(cand_sbert[i], attr_embs[a]))
        sem_by_key[(r["user_id"], r["condition"])].append(sem)
        cond_to_sem[r["condition"]].append(sem)

    log("[7] Per-pair dump ...")
    with OUT_PER_PAIR.open("w") as f:
        for uid in sorted(margin_by_key.keys() | sem_by_key.keys(), key=lambda x: (x[0], x[1])):
            u, c = uid
            ms = margin_by_key.get(uid, [])
            ss = sem_by_key.get(uid, [])
            row = {
                "user_id": u,
                "condition": c,
                "best_of_K_margin": max(ms) if ms else None,
                "mean_margin": float(np.mean(ms)) if ms else None,
                "mean_semantic_sim": float(np.mean(ss)) if ss else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log("[8] Aggregate per cond ...")
    eval_out: dict[str, dict] = {}
    for cond in conds:
        ms = cond_to_margins.get(cond, [])
        bks = cond_to_bok.get(cond, [])
        ss = cond_to_sem.get(cond, [])
        eval_out[cond] = {
            "n_candidates": len(ms),
            "n_pairs": len(bks),
            "mean_margin": float(np.mean(ms)) if ms else 0.0,
            "std_margin": float(np.std(ms)) if ms else 0.0,
            "mean_best_of_K_margin": float(np.mean(bks)) if bks else 0.0,
            "std_best_of_K_margin": float(np.std(bks)) if bks else 0.0,
            "mean_semantic_sim": float(np.mean(ss)) if ss else 0.0,
            "std_semantic_sim": float(np.std(ss)) if ss else 0.0,
        }

    log("[9] Bootstrap CI vs baselines (D_off, A_a1.0) ...")
    rng = np.random.default_rng(42)

    def _per_uid_best(d):
        out = {}
        for (uid, c), vs in d.items():
            if not vs:
                continue
            if c not in out or (out[c][uid] if uid in out[c] else None) is None or max(vs) > out[c].get(uid, -1e9):
                out.setdefault(c, {})[uid] = max(vs)
        return out

    base_bok_per_uid_per_cond: dict[str, dict[str, float]] = _per_uid_best(margin_by_key)
    base_sem_per_uid_per_cond: dict[str, dict[str, float]] = {
        c: {uid: float(np.mean(vs)) for (uid, cc), vs in sem_by_key.items() if cc == c and vs}
        for c in conds
    }

    for base in BASELINES:
        log(f"  vs {base}:")
        for cond in conds:
            if cond == base:
                continue
            a_bok = base_bok_per_uid_per_cond.get(cond, {})
            a_sem = base_sem_per_uid_per_cond.get(cond, {})
            b_bok = base_bok_per_uid_per_cond.get(base, {})
            b_sem = base_sem_per_uid_per_cond.get(base, {})

            bok_diffs = [a_bok[u] - b_bok[u] for u in a_bok if u in b_bok]
            sem_diffs = [a_sem[u] - b_sem[u] for u in a_sem if u in b_sem]

            eval_out[cond][f"vs_{base}"] = {}
            for k, v in [("bok_diff", bok_diffs), ("sem_diff", sem_diffs)]:
                if not v:
                    continue
                arr = np.array(v)
                mean = float(np.mean(arr))
                boots = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(2000)]
                lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                eval_out[cond][f"vs_{base}"][k] = {
                    "mean": mean, "ci_low": lo, "ci_high": hi,
                    "excludes_0": bool(lo > 0 or hi < 0),
                    "n_pairs": len(arr),
                }

    eval_out["meta"] = {
        "phase": "16",
        "n_total_records": len(records),
        "n_conds": len(conds),
        "n_pairs": len(pair_uids),
        "K": 8,
        "baselines": BASELINES,
    }

    OUT_EVAL.write_text(json.dumps(eval_out, indent=2, ensure_ascii=False))
    log(f"  saved: {OUT_EVAL}")

    print()
    log("=" * 70)
    log("Phase 16 eval summary")
    log("=" * 70)
    print(f"{'cond':<16} {'n_cand':>7} {'mean_marg':>10} {'BoK_marg':>10} {'sem_sim':>10}")
    print("-" * 60)
    for cond in conds:
        d = eval_out[cond]
        print(f"{cond:<16} {d['n_candidates']:>7} {d['mean_margin']:>10.4f} "
              f"{d['mean_best_of_K_margin']:>10.4f} {d['mean_semantic_sim']:>10.4f}")
    print()
    for base in BASELINES:
        log(f"--- vs {base} (CI excludes 0 = significant lift) ---")
        for cond in conds:
            if cond == base:
                continue
            vd = eval_out[cond].get(f"vs_{base}", {})
            for k in ("bok_diff", "sem_diff"):
                r = vd.get(k, {})
                if not r:
                    continue
                sig = "✓" if r.get("excludes_0") else "✗"
                print(f"  {cond:<14} {k:<14} diff={r['mean']:+.4f} "
                      f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {sig}")
    log("DONE")


if __name__ == "__main__":
    main()