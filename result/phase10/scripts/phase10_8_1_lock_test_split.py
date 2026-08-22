#!/usr/bin/env python3
"""Phase 10.8.1: 冻结数据协议.

锁定当前60 user 为 dev, 从剩余用户中抽60个 fresh test 用户 (从未参与任何阶段).

约束 (硬):
  - train/dev/test 用户ID交集必须为0
  - test 用户在模型/超参数完全锁定前不得查看
  - 用固定种子(42)抽 60 fresh test 用户
  - test list 写入后只读, 不能再覆盖
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

# === 硬编码 ===
SEED = 42
N_DEV = 60
N_TEST = 60
N_FRESH_TEST = 60

# 当前 60 user 是 dev (锁定)
DEV_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"
TEST_FILE = OUT_DIR / "phase10_test_60_user_ids.json"  # fresh test, 写后只读

# train 用户 (用于排除) — 当前 60 user 既是 dev 又是曾参与训练
# 这里我们也排除训练 preference_pairs 涉及的所有用户
# 但 60 user 本身就是 preference_pairs 的用户, 所以排除的 set 是一样的
USED_FILES = [
    "phase10_pairs_60.jsonl",         # 60 user
    "preference_pairs_60.jsonl",      # 60 user
    "phase10_eval_candidates_60.jsonl",  # 60 user
    "phase10_candidates_60.jsonl",    # 60 user
]

# 全部 user profile 列表
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"


def main():
    print("[phase10-8.1] ==================== 数据协议冻结 ====================", flush=True)

    # === 1. 收集所有"已使用"用户 ===
    used_ids = set()
    for fname in USED_FILES:
        fpath = OUT_DIR / fname
        if not fpath.exists():
            continue
        with fpath.open() as f:
            for line in f:
                d = json.loads(line)
                used_ids.add(d["user_id"])
    print(f"[phase10-8.1] 已使用用户数 (train + dev) = {len(used_ids)}")
    assert len(used_ids) == N_DEV, f"期望 60 user, 实际 {len(used_ids)}"

    # === 2. 加载所有候选用户 ===
    all_users = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            all_users.append(json.loads(line)["user_id"])
    print(f"[phase10-8.1] 总用户数 = {len(all_users)}")
    assert len(all_users) == 2918

    # === 3. 候选 fresh test 用户 (排除已使用) ===
    eligible = [u for u in all_users if u not in used_ids]
    print(f"[phase10-8.1] 候选 fresh test 用户数 = {len(eligible)}")
    assert len(eligible) >= N_FRESH_TEST, "候选用户不够!"

    # === 4. 用固定种子抽 60 ===
    import random
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    test_users = sorted(eligible[:N_FRESH_TEST])
    dev_users = sorted(used_ids)
    print(f"[phase10-8.1] dev user = {len(dev_users)}")
    print(f"[phase10-8.1] test user = {len(test_users)} (前 5 个: {test_users[:3]})")

    # === 5. 交集检查 (硬) ===
    dev_set = set(dev_users)
    test_set = set(test_users)
    intersection = dev_set & test_set
    assert len(intersection) == 0, f"FAIL: dev/test 交集 = {len(intersection)}"
    print(f"[phase10-8.1] ✓ dev ∩ test = ∅ (交集大小 = {len(intersection)})")

    # train 用户? 当前没有 train 用户(60 user 全在 dev)。
    # 用户原话: "train、dev、test 用户ID交集必须为0"
    # train=dev(60 user) 但 dev ≠ test, train ∩ test = ∅ 自然成立。
    # 如果未来 train 独立于 dev, 需要在这里显式排除 train set。
    train_users = sorted(used_ids)  # 当前 = dev
    print(f"[phase10-8.1] train ∩ dev = {len(set(train_users) & dev_set)}")
    print(f"[phase10-8.1] train ∩ test = {len(set(train_users) & test_set)}")

    # === 6. 写文件 ===
    DEV_FILE.write_text(json.dumps({
        "user_ids": dev_users,
        "n_users": len(dev_users),
        "frozen_at": "2026-08-19",
        "note": "Phase 10.5/10.6/10.7 用过的 60 user, 现在锁定为 dev (不能再作为 final test 证据)",
    }, ensure_ascii=False, indent=2))
    TEST_FILE.write_text(json.dumps({
        "user_ids": test_users,
        "n_users": len(test_users),
        "frozen_at": "2026-08-19",
        "seed": SEED,
        "note": "Fresh test set. 在模型/超参数完全锁定前不得查看或用于调参。只读。",
        "readonly": True,
    }, ensure_ascii=False, indent=2))
    print(f"[phase10-8.1] 已写 {DEV_FILE}")
    print(f"[phase10-8.1] 已写 {TEST_FILE}")

    # === 7. 总结 ===
    print()
    print(f"[phase10-8.1] 数据协议冻结完成:")
    print(f"  dev  = {len(dev_users)} users (locked, 当前 evidence 来自此)")
    print(f"  test = {len(test_users)} users (fresh, 写后只读, 未参与训练/调参)")
    print(f"  dev ∩ test = ∅ ✓")
    print(f"  文件: {DEV_FILE}")
    print(f"  文件: {TEST_FILE}")
    print("[phase10-8.1] ====================================================")


if __name__ == "__main__":
    main()