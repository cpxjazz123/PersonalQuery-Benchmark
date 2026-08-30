#!/usr/bin/env python3
"""Phase 6.A.1 — Build (s_i, c_i, z_i) training dataset.

用户指令 2026-08-30: Phase 5 NO-GO 后转 Phase 6 — 两阶段可验证的
PCA48 Syntax Renderer。Phase 6.A.1 构造训练集:

  s_i = 真实历史句子 (Baby Products)
  c_i = 该句所在 ASIN 的 top-5 非数值属性 (canonical content)
  z_i = PCA48(s_i) — 沿用现有 user Gaussians 完全相同的 F3 103d →
        StandardScaler → PCA48 流水线 (48-dim, 与 Stage 4 strict
        alignment gate 用的 PCA48 同一坐标系)

为什么 c_i = ASIN attrs:
  1. 与下游 inference 对齐 — 推理时 c = 目标 ASIN 的 attrs
  2. attrs 是 syntax-neutral 的 (没有 clause/subordinator), 强迫模型用
     z_i 来决定 syntax 而不是从 c_i 推断
  3. 训练/推理分布一致

输出:
  scratch2/pca48_dataset/train.jsonl     每行: {user_id, asin, s_i, c_i,
                                              z_48, n_words}
  scratch2/pca48_dataset/z_stats.json    z 全局 mean/std (PC sweep 用)
  scratch2/pca48_dataset/z_all.npy       z (N, 48) NPY 供训练直读
  scratch2/pca48_dataset/attrs_index.json  attrs vocab 供 inference

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import spacy

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
SENT_IN = SCRATCH / "gaussian_vades" / "sentences_for_rewrite_10k.jsonl"
PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
OUT_DIR = SCRATCH / "pca48_dataset"
LOG_DIR = SCRATCH / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Phase 6.A.1 pilot 配置 (硬编码)
N_DATASET = 5000
N_INPUT = 5
MIN_WORDS = 5
MAX_WORDS = 60
PCA_DIM = 48
SEED = 42
SPACY_BATCH = 256
SPACY_N_PROCESS = 8

NUMERIC_RE = re.compile(r"[0-9]")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pca48_dataset] {msg}", flush=True)


def is_numeric_value(v) -> bool:
    """排除含数字字符的 attribute value (与 Phase 1 一致).

    兼容 list / None / int / float / bool 等非字符串类型:不是字符串或不含
    数字字符 → 视为 numeric (排除).
    """
    if not isinstance(v, str):
        return True
    return bool(NUMERIC_RE.search(v))


def get_top_n_attrs(asin: str, pattrs: dict, n: int) -> list[str]:
    """Top-N non-numeric attrs of an ASIN."""
    pa = pattrs.get(asin)
    if not pa:
        return []
    items = [(k, v) for k, v in pa.items() if not is_numeric_value(v)]
    if len(items) < n:
        return []
    return [v for _, v in items[:n]]


def main():
    log("=== Phase 6.A.1 — Build (s_i, c_i, z_i) dataset ===")

    log("loading attrs index ...")
    pattrs = json.load(open(PATTRS_PATH))
    log(f"  attrs index: {len(pattrs)} ASINs")

    log("loading user Gaussians (for F3 103d scaler + PCA48 components) ...")
    gauss = json.load(open(GAUSS_PATH))
    log(f"  Gaussians loaded: {gauss['n_users_with_gaussian']} users")
    log(f"  F3 features: {len(gauss['fnames_f3'])}; "
        f"PCA components: {len(gauss['pca_components'])} x {len(gauss['pca_components'][0])}")

    log("loading sentences ...")
    sentences = []
    n_total = 0
    with open(SENT_IN) as f:
        for line in f:
            rec = json.loads(line)
            s = rec.get("sentence_text", "").strip()
            wc = len(s.split())
            n_total += 1
            if MIN_WORDS <= wc <= MAX_WORDS:
                sentences.append({
                    "user_id": rec["user_id"],
                    "asin": rec["asin"],
                    "s_i": s,
                    "n_words": wc,
                })
    log(f"  raw: {n_total}, after word filter ({MIN_WORDS}-{MAX_WORDS}): {len(sentences)}")

    log("filtering sentences by attrs availability ...")
    valid = []
    n_no_attrs = 0
    for s in sentences:
        attrs = get_top_n_attrs(s["asin"], pattrs, N_INPUT)
        if len(attrs) == N_INPUT:
            s["c_i"] = attrs
            valid.append(s)
        else:
            n_no_attrs += 1
    log(f"  after attrs filter ({N_INPUT} non-numeric): {len(valid)} "
        f"(dropped {n_no_attrs})")

    # Subsample
    import random
    random.seed(SEED)
    random.shuffle(valid)
    valid = valid[:N_DATASET]
    log(f"final dataset size (after seed={SEED} subsample): {len(valid)}")

    log("loading spaCy ...")
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2

    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gauss["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)  # (48, 103)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)  # (103,)

    def project_pca48(feats_318: np.ndarray) -> np.ndarray:
        f3 = feats_318[:, col_idx_f3]
        f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
        return (f3_scaled - pca_mean) @ pca_components.T

    log("computing F3 318d + PCA48 for all sentences (spaCy batch) ...")
    t0 = time.time()
    all_z = []
    for st in range(0, len(valid), SPACY_BATCH):
        batch = valid[st:st + SPACY_BATCH]
        texts = [s["s_i"] for s in batch]
        docs = list(nlp.pipe(texts, batch_size=SPACY_BATCH, n_process=SPACY_N_PROCESS))
        feats = np.zeros((len(texts), len(fnames_all)), dtype=np.float64)
        for i, doc in enumerate(docs):
            try:
                # CRITICAL: per_sentence_features_v2 takes ONE arg (Doc/Span),
                # the existing extract_318d in pca48_nearest_selection.py
                # mistakenly passes (doc, sent) which silently fails the
                # try/except and yields all-zero features — that bug causes
                # z_all std ≈ 0 and breaks PCA48 control. We call correctly here.
                d = per_sentence_features_v2(doc)
                if d is None:
                    continue
                for j, fn in enumerate(fnames_all):
                    feats[i, j] = float(d.get(fn, 0.0))
            except Exception:
                pass  # already zeros
        z = project_pca48(feats)
        all_z.append(z)
        if (st // SPACY_BATCH) % 5 == 0:
            log(f"  batch {st // SPACY_BATCH + 1}/{(len(valid) - 1) // SPACY_BATCH + 1} "
                f"({min(st + SPACY_BATCH, len(valid))}/{len(valid)}) "
                f"elapsed {time.time() - t0:.0f}s")

    z_all = np.concatenate(all_z, axis=0).astype(np.float32)
    log(f"z_all shape: {z_all.shape}, dtype: {z_all.dtype}")

    # Stats for PC sweep (单位 std 用于 -2,-1,0,+1,+2 标定)
    z_mean = z_all.mean(axis=0)
    z_std = z_all.std(axis=0)
    log(f"z mean norm: {np.linalg.norm(z_mean):.3f}, std mean: {z_std.mean():.3f}")

    out_jsonl = OUT_DIR / "train.jsonl"
    with open(out_jsonl, "w") as f:
        for i, s in enumerate(valid):
            rec = {
                "user_id": s["user_id"],
                "asin": s["asin"],
                "s_i": s["s_i"],
                "c_i": s["c_i"],
                "z_48": [float(x) for x in z_all[i].tolist()],
                "n_words": s["n_words"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"wrote → {out_jsonl}  ({out_jsonl.stat().st_size / 1e6:.1f} MB)")

    np.save(OUT_DIR / "z_all.npy", z_all)
    log(f"wrote → {OUT_DIR / 'z_all.npy'}")

    z_stats = {
        "z_mean": z_mean.tolist(),
        "z_std": z_std.tolist(),
        "pca_explained_variance_ratio": gauss.get("pca_explained_variance_ratio", []),
        "n_samples": int(len(valid)),
    }
    with open(OUT_DIR / "z_stats.json", "w") as f:
        json.dump(z_stats, f, ensure_ascii=False, indent=2)
    log(f"wrote → {OUT_DIR / 'z_stats.json'}")

    attrs_index = {}
    for s in valid:
        attrs_index[s["asin"]] = s["c_i"]
    with open(OUT_DIR / "attrs_index.json", "w") as f:
        json.dump(attrs_index, f, ensure_ascii=False)
    log(f"wrote → {OUT_DIR / 'attrs_index.json'}  ({len(attrs_index)} unique ASINs)")

    log("\n=== Summary ===")
    log(f"  N samples: {len(valid)}")
    log(f"  z_all mean: {z_mean.tolist()[:5]}... (truncated)")
    log(f"  z_all std:  {z_std.tolist()[:5]}... (truncated)")
    log(f"  attrs per sample: {N_INPUT}")
    log(f"  pilot ready. Next: gen_query/syntax_subspace_pca48_train.py")


if __name__ == "__main__":
    main()