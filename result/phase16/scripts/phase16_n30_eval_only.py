#!/usr/bin/env python3
"""Phase 15.D Eval (stand-alone): 从 jsonl 读 records 跑 style margin + semantic sim。

sweep.py 本身因为 evaluate_all() 里忘 import torch 崩了,
但 1680 records 已经成功生成并写入 phase16_n30_gaussian_sweep.jsonl。
本脚本独立重跑 eval 部分:
  - 768d AnnaWegmann style margin (cos_target - cos_off_mean)
  - best-of-K margin (per cond, per pair max across K=8)
  - sentence-bert all-MiniLM-L6-v2 semantic sim vs attrs text
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
JSONL_PATH = OUT_DIR / "phase16_n30_gaussian_sweep.jsonl"
OUT_EVAL = OUT_DIR / "phase16_n30_eval.json"
USER_EMB_NPZ = OUT_DIR / "phase10_user_embs_768d.npz"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    log("=" * 70)
    log("Phase 16 EVAL — read jsonl, compute style margin + semantic sim")
    log("=" * 70)

    # === [1] Load records ===
    log("[1] Loading 2160 records from jsonl ...")
    records: list[dict] = []
    with JSONL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log(f"  loaded: {len(records)} records (9 conds × 30 pairs × K=8)")
    conds = sorted({r["condition"] for r in records})
    log(f"  conds: {conds}")

    # === [2] Load user embs ===
    log("[2] Loading 876 user 768d embs ...")
    user_data = np.load(str(USER_EMB_NPZ), allow_pickle=True)
    user_embs = user_data["embs"]
    user_ids = list(user_data["user_ids"])
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    user_embs_norm = l2_normalize(user_embs.astype(np.float32))
    log(f"  user_embs: {user_embs.shape}, {len(user_ids)} users")

    # === [3] Encode all candidates with AnnaWegmann ===
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

    # === [4] Per-candidate margin (cos_target - cos_off_mean) ===
    log("[4] Computing per-cand margin (cos target - cos off mean) ...")
    cand_norm = l2_normalize(cand_embs.astype(np.float32))
    # margin_by_key: dict[(uid, cond)] → list[float] (K values per pair)
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

    # === [5] Per-pair best-of-K margin (use direct key grouping, no global index) ===
    log("[5] Computing per-pair best-of-K margin ...")
    cond_to_bok: dict[str, list[float]] = defaultdict(list)
    cond_to_margins: dict[str, list[float]] = defaultdict(list)
    for (uid, cond_name), ms in margin_by_key.items():
        if not ms:
            continue
        cond_to_margins[cond_name].extend(ms)
        cond_to_bok[cond_name].append(max(ms))
    log(f"  per-cond best-of-K: " + ", ".join(
        f"{c}={len(v)}" for c, v in sorted(cond_to_bok.items())))

    # === [6] Semantic sim via sentence-bert ===
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

    # === [7] Aggregate per cond ===
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

    # === [8] A vs D_off paired bootstrap diffs (per-pair BoK + mean sem) ===
    log("[7] Bootstrap CI for (A vs D_off) diffs ...")
    rng = np.random.default_rng(42)

    # Per-pair bok by uid (for each cond, max across K)
    base_bok_by_uid: dict[str, float] = {
        uid: max(ms) for (uid, c), ms in margin_by_key.items()
        if c == "D_off" and ms
    }
    base_sem_by_uid: dict[str, float] = {
        uid: float(np.mean(ss)) for (uid, c), ss in sem_by_key.items()
        if c == "D_off" and ss
    }

    for cond in conds:
        if cond == "D_off":
            continue
        a_bok_by_uid: dict[str, float] = {
            uid: max(ms) for (uid, c), ms in margin_by_key.items()
            if c == cond and ms
        }
        a_sem_by_uid: dict[str, float] = {
            uid: float(np.mean(ss)) for (uid, c), ss in sem_by_key.items()
            if c == cond and ss
        }
        bok_diffs = [
            a_bok_by_uid[uid] - base_bok_by_uid[uid]
            for uid in a_bok_by_uid if uid in base_bok_by_uid
        ]
        sem_diffs = [
            a_sem_by_uid[uid] - base_sem_by_uid[uid]
            for uid in a_sem_by_uid if uid in base_sem_by_uid
        ]

        eval_out[cond]["vs_D_off"] = {}
        for k, v in [("bok_diff", bok_diffs), ("sem_diff", sem_diffs)]:
            if len(v) == 0:
                continue
            arr = np.array(v)
            mean = float(np.mean(arr))
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(arr), len(arr))
                boots.append(float(np.mean(arr[idx])))
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            eval_out[cond]["vs_D_off"][k] = {
                "mean": mean, "ci_low": lo, "ci_high": hi,
                "excludes_0": bool(lo > 0 or hi < 0),
                "n_pairs": len(arr),
            }

    OUT_EVAL.write_text(json.dumps(eval_out, indent=2, ensure_ascii=False))
    log(f"  saved: {OUT_EVAL}")

    # === [9] Print verdict ===
    print()
    log("=" * 70)
    log("Phase 16 N=30 Gaussian steer sweep eval summary")
    log("=" * 70)
    print(f"{'cond':<16} {'n_cand':>7} {'mean_marg':>10} {'BoK_marg':>10} {'sem_sim':>10}")
    print("-" * 60)
    for cond in conds:
        d = eval_out[cond]
        print(f"{cond:<16} {d['n_candidates']:>7} {d['mean_margin']:>10.4f} "
              f"{d['mean_best_of_K_margin']:>10.4f} {d['mean_semantic_sim']:>10.4f}")
    print()
    log("A vs D_off (CI excludes 0 = significant lift):")
    for cond in conds:
        if cond == "D_off":
            continue
        vd = eval_out[cond].get("vs_D_off", {})
        for k in ("bok_diff", "margin_diff", "sem_diff"):
            r = vd.get(k, {})
            if not r:
                continue
            sig = "✓" if r.get("excludes_0") else "✗"
            print(f"  {cond:<14} {k:<14} diff={r['mean']:+.4f} CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {sig}")
    log("DONE")


if __name__ == "__main__":
    main()