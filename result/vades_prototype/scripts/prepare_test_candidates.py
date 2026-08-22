#!/usr/bin/env python3
"""准备测试 candidates: 从 3000 个 v6 candidates 抽 30 个用户的真实 candidates + 真实 attrs (从 meta).

为 Phase 9.6 三组生成对比做基础准备:
  - candidates_from_v6.jsonl: 30 用户 × 10 候选 (现成)
  - attrs_by_asin.json: 5 个属性 (Brand/Item Weight/Product Dimensions/Color/Material)
  - synthetic_target_user_query.jsonl: 用 user 的 train 句作为"目标用户风格"基线

这样三组对比是:
  A) 无风格 (用现有 candidates, 无 user_mu conditioning)
  B) 随机用户风格 (v6 query 已经有 latent space Gaussian score, ≈ random selection)
  C) 目标用户风格 (用 user 的真实训练句作为风格基线 — 真实 raw 20d 风格)

跑 scorer 后比较三组的 maha_target, rank, semantic_sim, attr_pass。
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
CANDIDATE_FILE = VADES_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
N_TEST_USERS = 30  # 测试用户数
N_CANDIDATES_PER_USER = 5  # 每个用户 5 个候选 (无风格基线)


def main():
    log = lambda m: print(f"[prep-test] {m}", flush=True)
    log("=" * 70)
    log(f"准备 Raw-space VADES 测试 candidates (n_users={N_TEST_USERS})")
    log("=" * 70)

    # === 1. 加载 v6 candidates, 按 user 分组取前 N_CANDIDATES_PER_USER ===
    user_to_candidates: dict[str, list[dict]] = {}
    with CANDIDATE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            u = r["user_id"]
            if not r.get("query"):
                continue
            user_to_candidates.setdefault(u, []).append(r)

    test_user_ids = list(user_to_candidates.keys())[:N_TEST_USERS]
    log(f"  selected users: {len(test_user_ids)}")

    # === 2. 加载 user_profiles + sentences (取 target_user 真实训练句作为 "目标风格" 基线) ===
    user_profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            user_profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(user_profiles)}

    sentences_by_user: dict[str, list[str]] = {}
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_holdout"):
                continue  # 只用 train 句
            u = r["user_id"]
            q = r.get("sentence_text", "").strip()
            if q and len(q.split()) >= 4:  # 太短的不要
                sentences_by_user.setdefault(u, []).append(q)

    log(f"  sentences_by_user: {sum(len(v) for v in sentences_by_user.values())} train sentences total")

    # === 3. 从 meta 取 attrs (只关心 candidates 用到的 291 个 asin) ===
    log(f"读取 meta 找有完整 5 attrs 的 asins ...")
    meta_path = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"

    # 收集所有 candidates 用到的 asin
    candidates_all_asins = set()
    for u, cands in user_to_candidates.items():
        for c in cands:
            candidates_all_asins.add(c["asin"])
    log(f"  total candidates asins: {len(candidates_all_asins)}")

    # 构建 user -> [(asin, query)] 索引
    user_to_asin_query: dict[str, list[tuple[str, str]]] = {}
    for u in test_user_ids:
        seen_asin = set()
        for c in user_to_candidates[u]:
            if c["asin"] in seen_asin:
                continue
            seen_asin.add(c["asin"])
            user_to_asin_query.setdefault(u, []).append((c["asin"], c["query"]))

    # 强制扫描完整 meta 文件
    attrs_by_asin: dict[str, dict] = {}
    n_meta_scanned = 0
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            n_meta_scanned += 1
            r = json.loads(line)
            pa = r.get("parent_asin", "")
            if pa not in candidates_all_asins:
                continue  # 不在 candidates 里, 跳过 details 解析
            if pa in attrs_by_asin:
                continue
            details = r.get("details", {})
            if not isinstance(details, dict):
                continue
            attrs = {}
            ok = True
            for k in ATTR_FIELDS:
                v = details.get(k, "")
                if not v:
                    ok = False
                    break
                attrs[k] = str(v).strip()[:64]
            if ok:
                attrs_by_asin[pa] = attrs
            if len(attrs_by_asin) >= len(candidates_all_asins):  # 全部 candidates asin 都找到 5 attrs
                break
    log(f"  meta scanned: {n_meta_scanned} records, {len(attrs_by_asin)}/{len(candidates_all_asins)} asins have 5 attrs")

    # 筛掉 candidates 中没有 5 attrs 的 asin
    filtered_user_to_asin_query = {}
    for u, pairs in user_to_asin_query.items():
        valid = [(a, q) for a, q in pairs if a in attrs_by_asin]
        if valid:
            filtered_user_to_asin_query[u] = valid
    log(f"  users with at least one 5-attr asin: {len(filtered_user_to_asin_query)}/{len(test_user_ids)}")

    test_user_ids = list(filtered_user_to_asin_query.keys())[:N_TEST_USERS]
    log(f"  final test users: {len(test_user_ids)}")

    # === 4. 输出 ===
    # (A) no_style_candidates.jsonl: 每个用户 5 个候选 (无 user style conditioning)
    no_style_out = OUT_DIR / "no_style_candidates.jsonl"
    n_written = 0
    with no_style_out.open("w") as fout:
        for u in test_user_ids:
            pairs = filtered_user_to_asin_query[u][:N_CANDIDATES_PER_USER]
            for ci, (asin, q) in enumerate(pairs):
                attrs = attrs_by_asin[asin]
                row = {
                    "user_id": u,
                    "asin": asin,
                    "candidate_index": ci,
                    "candidate_query": q,
                    "attrs": attrs,
                    "reference_query": q,
                    "group": "A_no_style",
                }
                fout.write(json.dumps(row) + "\n")
                n_written += 1
    log(f"  A) 无风格候选: {n_written} ({no_style_out})")

    # (C) target_user_style: 用 user 的真实训练句作为"目标风格"基线
    target_style_out = OUT_DIR / "target_style_candidates.jsonl"
    n_written = 0
    with target_style_out.open("w") as fout:
        for u in test_user_ids:
            sentences = sentences_by_user.get(u, [])
            if not sentences:
                continue
            first_asin = filtered_user_to_asin_query[u][0][0]
            attrs = attrs_by_asin[first_asin]
            for si, q in enumerate(sentences[:N_CANDIDATES_PER_USER]):
                row = {
                    "user_id": u,
                    "asin": first_asin,
                    "candidate_index": si,
                    "candidate_query": q,
                    "attrs": attrs,
                    "reference_query": q,
                    "group": "C_target_style",
                }
                fout.write(json.dumps(row) + "\n")
                n_written += 1
    log(f"  C) 目标风格候选: {n_written} ({target_style_out})")

    # (B) random_style: 用其他用户的训练句 — 模拟 random user_mu condition
    rng = __import__("random")
    rng.seed(42)
    random_style_out = OUT_DIR / "random_style_candidates.jsonl"
    n_written = 0
    with random_style_out.open("w") as fout:
        for u in test_user_ids:
            other_users = [ou for ou in sentences_by_user.keys() if ou != u]
            first_asin = filtered_user_to_asin_query[u][0][0]
            attrs = attrs_by_asin[first_asin]
            for si in range(N_CANDIDATES_PER_USER):
                ru = rng.choice(other_users)
                if not sentences_by_user[ru]:
                    continue
                q = rng.choice(sentences_by_user[ru])
                row = {
                    "user_id": u,
                    "asin": first_asin,
                    "candidate_index": si,
                    "candidate_query": q,
                    "attrs": attrs,
                    "reference_query": q,
                    "group": "B_random_style",
                    "source_user": ru,
                }
                fout.write(json.dumps(row) + "\n")
                n_written += 1
    log(f"  B) 随机风格候选: {n_written} ({random_style_out})")

    # === 5. 输出 test_user_ids 列表 ===
    with (OUT_DIR / "test_user_ids.json").open("w") as f:
        json.dump(test_user_ids, f)
    log(f"  test_user_ids: {OUT_DIR / 'test_user_ids.json'}")

    log("=" * 70)
    log("完成 test candidates 准备")
    log("=" * 70)


if __name__ == "__main__":
    main()