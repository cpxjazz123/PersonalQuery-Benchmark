#!/usr/bin/env python3
"""迭代式风格优化 (reward-guided refinement).

== 流程 ==
1. 从初始 query (来自 candidates.jsonl, 或新生成) 开始
2. 提取 20-d 句法特征 → 计算到 user_mu 的 Mahalanobis 距离
3. 如果距离 ≤ threshold 且属性完整 → 终止
4. 否则: 生成结构化反馈 (top-K 特征差异, 方向 + 幅度)
5. 用反馈 prompt LLM 重写 query
6. 重复 max_steps 轮, 保留 best (最小距离)
7. 输出 refined_query + 每轮 trace

== 评分 ==
R(q) = -λ1 * D_style(q, u) + λ2 * S_semantic(q, source) + λ3 * S_attr(q)
- D_style: Mahalanobis 距离 (VADES disentangle user_mu + user_logvar)
- S_semantic: cosine similarity (TF-IDF) vs source_query
- S_attr: 属性完整率 (5 attrs 都在)

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_refined.jsonl
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_trace.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/syntactic_analysis")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian.gaussian_vades import (
    SentenceEncoder,
    UserDistributionTableDisentangled,
)
from llm_client import QwenLocalClient

EXPERIMENT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
TEST_CASES = EXPERIMENT_DIR / "test_cases.jsonl"
CANDIDATES = EXPERIMENT_DIR / "candidates.jsonl"

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
ENCODER_CKPT = VADES_DIR / "vades_encoder.pt"
USER_TABLE_CKPT = VADES_DIR / "vades_user_table.pt"
SENTENCE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_sentences.jsonl"

OUT_REFINED = EXPERIMENT_DIR / "iterative_refined_v2.jsonl"
OUT_TRACE = EXPERIMENT_DIR / "iterative_trace_v2.json"
OUT_SUMMARY = EXPERIMENT_DIR / "iterative_summary_v2.json"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]

# 中文 feature display
FEATURE_DISPLAY = {
    "max_dependency_depth": "最大依存树深度",
    "mean_dependency_depth": "平均依存深度",
    "dependency_tree_height": "依存树高度",
    "depth_variance": "深度方差",
    "acl_count": "形容词性从句数",
    "relcl_count": "关系从句数",
    "ccomp_count": "补语从句数",
    "xcomp_count": "开放补语从句数",
    "advcl_count": "状语从句数",
    "clause_nesting_depth": "从句嵌套深度",
    "mean_dependency_distance": "平均依存距离",
    "max_dependency_distance": "最大依存距离",
    "long_dependency_ratio": "长依存比例",
    "amod_count": "形容词修饰数",
    "advmod_count": "副词修饰数",
    "nmod_count": "名词修饰数",
    "compound_count": "复合词数",
    "modifier_density": "修饰密度",
    "coordination_count": "并列结构数",
    "max_branching_factor": "最大分支因子",
}

MAX_STEPS = 10
DISTANCE_THRESHOLD = 5.0
TOP_K_FEEDBACK = 3  # V2: 只取 top-3 差距最大的特征
FEEDBACK_DIFF_THRESHOLD = 0.05  # V2: 只显示 |diff| > threshold 的特征 (负向 only)
BEAM_WIDTH = 2  # V2: beam search 保留 top-K 候选
BEAM_BRANCHING = 2  # V2: 每轮每个候选生成多少个扩展
LAMBDA_STYLE = 1.0
LAMBDA_SEMANTIC = 0.3
LAMBDA_ATTR = 1.0
TEMPERATURE = 0.3  # V2: 降低 → 更稳定
MAX_TOKENS = 200


# ============================================================
# Data classes
# ============================================================
@dataclass
class FeedbackItem:
    feature: str
    display: str
    direction: str  # "increase" / "decrease" / "ok"
    cand_value: float
    user_value: float
    diff: float


@dataclass
class StepTrace:
    step: int
    query: str
    features: dict[str, float]
    mu_query: list[float]
    mahalanobis: float
    feedback: list[FeedbackItem]
    semantic_sim: float
    attr_completeness: float
    reward: float


# ============================================================
# StyleRefiner
# ============================================================
class StyleRefiner:
    def __init__(self, encoder: SentenceEncoder, user_table: UserDistributionTableDisentangled,
                 scaler, nlp, source_queries: dict[tuple[str, str], str] | None = None):
        self.encoder = encoder
        self.user_table = user_table
        self.scaler = scaler
        self.nlp = nlp
        self.user_id_to_idx: dict[str, int] = {}
        self.user_raw_features: dict[int, np.ndarray] = {}  # V2: user mean raw features (20-d)
        self.source_queries = source_queries or {}

    def load_user_profiles(self, profile_file: Path) -> None:
        """Load user_id → idx mapping from user_profiles.jsonl."""
        with profile_file.open() as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                self.user_id_to_idx[row["user_id"]] = i

    def load_user_raw_features(self, sentence_file: Path) -> None:
        """V2: 从训练句子算每个 user 的 mean raw features (20-d).
        用于 raw 句法特征 feedback (LLM 更易理解).
        """
        log = lambda m: print(f"[refine] {m}", flush=True)
        per_user: dict[str, list[np.ndarray]] = {}
        with sentence_file.open() as f:
            for line in f:
                row = json.loads(line)
                uid = row.get("user_id")
                if uid is None:
                    continue
                feat = np.asarray(
                    [float(row["features"][n]) for n in FEATURE_NAMES], dtype=np.float64,
                )
                per_user.setdefault(uid, []).append(feat)
        # 映射 user_id → idx → mean
        n_loaded = 0
        for uid, feats in per_user.items():
            idx = self.user_id_to_idx.get(uid)
            if idx is None:
                continue
            self.user_raw_features[idx] = np.mean(feats, axis=0)
            n_loaded += 1
        log(f"  loaded user raw features for {n_loaded}/{len(self.user_id_to_idx)} users")

    def get_user_distribution(self, user_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """从 disentangle user_table 拿 (user_mu, user_logvar)."""
        with torch.no_grad():
            mu_full, logvar_full = self.user_table(
                torch.arange(self.user_table.num_users, device=DEVICE)
            )
        return mu_full[user_idx].cpu().numpy(), logvar_full[user_idx].cpu().numpy()

    def extract_features(self, query: str) -> np.ndarray:
        """对 query 提取 20-d 句法特征 (未标准化)."""
        from extract_clause_features_single_query import extract_clause_features_from_doc
        try:
            doc = self.nlp(query)
            f = extract_clause_features_from_doc(doc, query)
            return np.asarray([float(f[name]) for name in FEATURE_NAMES], dtype=np.float64)
        except Exception:
            return np.zeros(INPUT_DIM, dtype=np.float64)

    def encode_to_mu(self, raw_features: np.ndarray) -> np.ndarray:
        """encoder forward → 20-d latent mu_query."""
        scaled = self.scaler.transform(raw_features.reshape(1, -1))
        with torch.no_grad():
            mu, _logvar, _recon = self.encoder(
                torch.as_tensor(scaled, dtype=torch.float32, device=DEVICE)
            )
            return mu.detach().cpu().numpy().flatten()

    def compute_mahalanobis(self, mu_query: np.ndarray, user_mu: np.ndarray,
                            user_logvar: np.ndarray) -> float:
        """Mahalanobis 距离平方 (对角协方差)."""
        var = np.exp(user_logvar).clip(min=1e-6)
        diff = mu_query - user_mu
        return float(np.sum(diff ** 2 / var))

    def compute_feature_feedback(self, cand_features: np.ndarray, user_mu: np.ndarray,
                                 user_idx: int, top_k: int = TOP_K_FEEDBACK) -> list[FeedbackItem]:
        """V2: 用 raw 20-d 句法特征 vs user mean raw features, 取 top-K 差距 (负向 only)."""
        user_raw = self.user_raw_features.get(user_idx)
        if user_raw is None:
            # Fallback to latent (保持兼容)
            mu_query = self.encode_to_mu(cand_features)
            diff_latent = user_mu - mu_query
            abs_diff = np.abs(diff_latent)
            top_idxs = np.argsort(-abs_diff)[:top_k]
            items: list[FeedbackItem] = []
            for i in top_idxs:
                d = float(diff_latent[i])
                items.append(FeedbackItem(
                    feature=f"latent_dim_{i}",
                    display=f"latent 维度 {i}",
                    direction="increase" if d > 0 else "decrease",
                    cand_value=float(mu_query[i]),
                    user_value=float(user_mu[i]),
                    diff=d,
                ))
            return items
        # 正常路径: raw feature 反馈
        diff_raw = user_raw - cand_features  # 用户 - 当前 = 需要调整的方向
        abs_diff = np.abs(diff_raw)
        # 负向 only: 只显示 |diff| > threshold 的特征
        mask = abs_diff > FEEDBACK_DIFF_THRESHOLD
        if mask.sum() == 0:
            mask = abs_diff > 0.0  # fallback: 至少取一个
        # 按 abs_diff 排序取 top-k
        idxs = np.where(mask)[0]
        idxs = idxs[np.argsort(-abs_diff[idxs])][:top_k]
        items: list[FeedbackItem] = []
        for i in idxs:
            d = float(diff_raw[i])
            items.append(FeedbackItem(
                feature=FEATURE_NAMES[i],
                display=FEATURE_DISPLAY.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
                direction="increase" if d > 0 else "decrease",
                cand_value=float(cand_features[i]),
                user_value=float(user_raw[i]),
                diff=d,
            ))
        return items

    def semantic_similarity(self, q1: str, q2: str) -> float:
        """TF-IDF cosine similarity."""
        if not q1.strip() or not q2.strip():
            return 0.0
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000)
            X = vec.fit_transform([q1, q2])
            return float(cosine_similarity(X[0:1], X[1:]).flatten()[0])
        except Exception:
            return 0.0

    def check_attrs(self, query: str, attrs: dict[str, str]) -> dict:
        """检查 5 个属性是否都出现在 query 中."""
        q_lower = query.lower()
        results = {}
        for key, val in attrs.items():
            # 只匹配属性值的关键 token
            tokens = [t for t in re.split(r"[\s,.\-/]+", val) if len(t) >= 3]
            hit = any(tok.lower() in q_lower for tok in tokens)
            results[key] = hit
        return results

    def build_feedback_prompt(self, current_query: str, feedback_items: list[FeedbackItem],
                              attrs: dict[str, str], user_idx: int, step: int,
                              best_dist: float) -> str:
        """构造反馈 prompt 给 LLM."""
        attrs_text = "\n".join(f"- {k}: {v}" for k, v in sorted(attrs.items()))
        feedback_text_lines = []
        for it in feedback_items:
            arrow = "↑" if it.direction == "increase" else "↓"
            feedback_text_lines.append(
                f"- 维度 {it.feature.split('_')[-1]} {arrow}: "
                f"当前 {it.cand_value:+.3f} vs 目标 {it.user_value:+.3f} "
                f"(差距 {it.diff:+.3f})"
            )
        feedback_text = "\n".join(feedback_text_lines)
        return (
            f"你需要重写一段购物 query, 让它的句法风格更接近目标用户 #{user_idx}。\n\n"
            f"【商品属性】(必须保留, 不要修改)\n{attrs_text}\n\n"
            f"【当前 Query】\n{current_query}\n\n"
            f"【句法反馈】(维度数值差异, 共 {len(feedback_items)} 条)\n{feedback_text}\n\n"
            f"【说明】\n"
            f"- 上方 ↑ 表示该维度需提升, ↓ 表示需降低\n"
            f"- 调整幅度请参考差距数值 (差距越大, 调整越大)\n"
            f"- 只调整句法结构 (嵌套深度、从句数、修饰密度等), 不要改商品属性\n"
            f"- 保持 query 自然, 不要堆砌从句\n\n"
            f"请直接输出改写后的 query, 不要解释, 不要带引号, 单行:"
        )

    def extract_refined_query(self, raw_response: str, original_query: str) -> str:
        """从 LLM 输出中提取 query (去引号/解释)."""
        text = raw_response.strip()
        # 去掉可能的引号包裹
        for q in ['"', "'", "「", "」", "『", "』"]:
            if text.startswith(q) and text.endswith(q) and len(text) > 1:
                text = text[1:-1].strip()
        # 多行取第一段非空
        for line in text.split("\n"):
            line = line.strip()
            # 跳过纯解释行 (以"以下是", "改写后", ":" 开头)
            if line and not re.match(r"^(以下是|改写后|输出|结果|Answer|Note|注意)", line, re.I):
                return line
        return text or original_query

    def run_single(self, case_key: tuple[str, str], initial_query: str,
                   attrs: dict[str, str], user_idx: int,
                   max_steps: int = MAX_STEPS,
                   llm_client: QwenLocalClient | None = None) -> dict:
        """V2: beam search 迭代精化.
        - 每轮保留 top-K (BEAM_WIDTH) 候选
        - 每个候选生成 BEAM_BRANCHING 个新候选
        - 返回全局 best (最小 Mahalanobis 距离)
        """
        log = lambda m: print(f"[refine] {m}", flush=True)
        trace: list[StepTrace] = []
        user_mu, user_logvar = self.get_user_distribution(user_idx)
        source_q = self.source_queries.get(case_key, initial_query)
        n_attrs = len(attrs)

        # Beam: list of (dist, query, features, mu_query, sem_sim, attr_complete, feedback_items)
        # 初始化 beam: 只有 initial_query
        def evaluate_query(q: str) -> dict:
            cand_features = self.extract_features(q)
            mu_q = self.encode_to_mu(cand_features)
            d = self.compute_mahalanobis(mu_q, user_mu, user_logvar)
            sem = self.semantic_similarity(q, source_q)
            hits = self.check_attrs(q, attrs)
            attr_comp = sum(hits.values()) / max(1, n_attrs)
            fb = self.compute_feature_feedback(cand_features, user_mu, user_idx)
            return {
                "query": q, "features": cand_features, "mu_query": mu_q,
                "dist": d, "sem": sem, "attr_complete": attr_comp,
                "feedback_items": fb,
            }

        # Step 0: 评估初始 query, 初始化 beam
        beam: list[dict] = [evaluate_query(initial_query)]
        best_overall = min(beam, key=lambda x: x["dist"])
        best_q_global = best_overall["query"]
        best_dist_global = best_overall["dist"]
        log(f"  initial dist={best_overall['dist']:.3f} sem={best_overall['sem']:.3f} "
            f"attr={best_overall['attr_complete']:.2f}")

        for step in range(max_steps):
            # Termination check on top of beam
            top = beam[0] if beam else best_overall
            if top["dist"] <= DISTANCE_THRESHOLD and top["attr_complete"] >= 1.0:
                log(f"  ✓ 收敛 (step {step}, dist={top['dist']:.3f})")
                break

            new_candidates: list[dict] = []
            for cand in beam:
                if cand["dist"] <= DISTANCE_THRESHOLD and cand["attr_complete"] >= 1.0:
                    continue  # 已收敛, 不再扩展
                for _ in range(BEAM_BRANCHING):
                    if llm_client is None:
                        continue
                    fb_prompt = self.build_feedback_prompt(
                        cand["query"], cand["feedback_items"], attrs, user_idx, step,
                        best_dist_global,
                    )
                    try:
                        raw = llm_client.call(fb_prompt, max_tokens=MAX_TOKENS,
                                             temperature=TEMPERATURE)
                        new_q = self.extract_refined_query(raw, cand["query"])
                    except Exception as e:
                        log(f"  LLM call failed at step {step}: {e!r}")
                        new_q = None
                    if not new_q or new_q == cand["query"]:
                        continue
                    new_candidates.append(evaluate_query(new_q))

            if not new_candidates:
                log(f"  step {step}: 无新候选, 终止")
                break

            # 合并 beam + new_candidates, 取 top-K
            pool = beam + new_candidates
            pool.sort(key=lambda x: x["dist"])
            beam = pool[:BEAM_WIDTH]
            # 更新全局 best
            step_best = beam[0]
            if step_best["dist"] < best_dist_global:
                best_dist_global = step_best["dist"]
                best_q_global = step_best["query"]

            # 记录 trace (取本轮 best)
            top = beam[0]
            trace.append(StepTrace(
                step=step,
                query=top["query"],
                features={n: float(top["features"][i]) for i, n in enumerate(FEATURE_NAMES)},
                mu_query=top["mu_query"].tolist(),
                mahalanobis=top["dist"],
                feedback=[asdict(it) for it in top["feedback_items"]],
                semantic_sim=top["sem"],
                attr_completeness=top["attr_complete"],
                reward=(
                    -LAMBDA_STYLE * top["dist"]
                    + LAMBDA_SEMANTIC * top["sem"] * 10.0
                    + LAMBDA_ATTR * top["attr_complete"] * 10.0
                ),
            ))
            log(f"  step {step}: best_dist={top['dist']:.3f} (beam_width={len(beam)}, "
                f"new={len(new_candidates)})")

        return {
            "case_key": list(case_key),
            "initial_query": initial_query,
            "final_query": best_q_global,
            "best_dist": best_dist_global,
            "n_steps": len(trace),
            "trace": [asdict(t) for t in trace],
        }


# ============================================================
# Pipeline
# ============================================================
def fit_scaler(sentence_file: Path) -> tuple:
    from sklearn.preprocessing import StandardScaler
    rows = []
    with sentence_file.open() as f:
        for line in f:
            rows.append(json.loads(line))
    feat = np.asarray(
        [[float(row["features"][n]) for n in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    scaler = StandardScaler()
    scaler.fit(feat)
    return scaler, len(rows)


def build_models() -> tuple[SentenceEncoder, UserDistributionTableDisentangled]:
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_offsets"].shape[0])
    latent_dim = int(ut_sd["style_centers"].shape[1])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, latent_dim)
    user_table = UserDistributionTableDisentangled(
        num_users=num_users, latent_dim=latent_dim,
        user_cluster_ids=placeholder_cluster,
        style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    user_table.to(DEVICE).eval()
    return encoder, user_table


def load_initial_queries() -> dict[tuple[str, str], str]:
    """从 candidates.jsonl 每个 (user, asin) 取第 1 条 query 作为 initial."""
    log = lambda m: print(f"[refine] {m}", flush=True)
    log(f"读 candidates.jsonl → initial queries")
    initial: dict[tuple[str, str], str] = {}
    with CANDIDATES.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["user_id"], row["asin"])
            if key not in initial and row.get("candidates"):
                initial[key] = row["candidates"][0]
    log(f"  initial queries: {len(initial)}")
    return initial


def load_test_cases() -> dict[tuple[str, str], dict]:
    cases = {}
    with TEST_CASES.open() as f:
        for line in f:
            row = json.loads(line)
            cases[(row["user_id"], row["asin"])] = row
    return cases


def build_attrs_from_case(case: dict) -> dict[str, str]:
    """仿照 gen_candidates.py::build_attrs, 从 raw product fields 构造 5 个 attrs."""
    title = case.get("title", "")
    main_category = case.get("main_category", "")
    store = case.get("store", "")
    categories = case.get("categories", []) or []
    description = case.get("description", "")
    cat0 = (categories[0] if len(categories) > 0 else main_category) or "Baby product"
    cat1 = (categories[1] if len(categories) > 1 else "everyday use") or "everyday use"
    desc = (description or "").strip().split(".")[0]
    if not desc:
        desc = "everyday baby essential"
    return {
        "A1 (product_type)": cat0,
        "A2 (brand)": store or "Generic",
        "A3 (use_case)": cat1,
        "A4 (appearance)": title[:80],
        "A5 (detailed)": desc[:120],
    }


def main() -> None:
    log = lambda m: print(f"[refine] {m}", flush=True)
    log("=" * 70)
    log("迭代式风格优化 V2 (beam search + raw feedback + 稳定化)")
    log(f"MAX_STEPS={MAX_STEPS}, BEAM_WIDTH={BEAM_WIDTH}, BEAM_BRANCHING={BEAM_BRANCHING}, "
        f"TOP_K_FEEDBACK={TOP_K_FEEDBACK}, FEEDBACK_DIFF_THR={FEEDBACK_DIFF_THRESHOLD}, "
        f"TEMPERATURE={TEMPERATURE}")
    log("=" * 70)

    log("加载 spaCy ...")
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    log("拟合 StandardScaler ...")
    scaler, n_train = fit_scaler(SENTENCE_FILE)
    log(f"  n_train={n_train}")

    log("构建 encoder + user_table ...")
    encoder, user_table = build_models()

    log("加载 user_id → idx ...")
    refiner = StyleRefiner(encoder, user_table, scaler, nlp)
    refiner.load_user_profiles(VADES_DIR / "vades_disentangled_v2_train10_holdout10_user_profiles.jsonl")
    refiner.load_user_raw_features(SENTENCE_FILE)  # V2: raw feedback
    log(f"  {len(refiner.user_id_to_idx)} users")

    log("加载 initial queries (candidates.jsonl) ...")
    initial = load_initial_queries()

    log("加载 test_cases ...")
    cases = load_test_cases()

    log("初始化 QwenLocalClient (with_vllm=True) ...")
    llm = QwenLocalClient(with_vllm=True)

    log("开始迭代精化 ...")
    OUT_REFINED.parent.mkdir(parents=True, exist_ok=True)
    OUT_REFINED.unlink(missing_ok=True)

    all_results = []
    for case_idx, (key, initial_q) in enumerate(initial.items()):
        uid, asin = key
        case = cases.get(key, {})
        attrs = case.get("attrs_used") or build_attrs_from_case(case)
        if not attrs:
            log(f"  case {case_idx}: no attrs, skip")
            continue
        user_idx = refiner.user_id_to_idx.get(uid)
        if user_idx is None:
            log(f"  case {case_idx}: user {uid[:8]}... not in profiles, skip")
            continue
        log(f"=== Case {case_idx+1}: user={uid[:8]} asin={asin} initial='{initial_q[:60]}...' ===")
        result = refiner.run_single(key, initial_q, attrs, user_idx, llm_client=llm)
        result["user_id"] = uid
        result["asin"] = asin
        result["attrs"] = attrs
        all_results.append(result)
        with OUT_REFINED.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        log(f"  → best_dist={result['best_dist']:.3f}, n_steps={result['n_steps']}, "
            f"final='{result['final_query'][:80]}...'")

    # 写入 trace
    OUT_TRACE.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "trace"} | {"trace": r["trace"]}
                   for r in all_results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Summary
    summary = {
        "n_cases": len(all_results),
        "max_steps": MAX_STEPS,
        "distance_threshold": DISTANCE_THRESHOLD,
        "top_k_feedback": TOP_K_FEEDBACK,
        "lambda": {"style": LAMBDA_STYLE, "semantic": LAMBDA_SEMANTIC, "attr": LAMBDA_ATTR},
    }
    if all_results:
        summary["mean_initial_dist"] = float(np.mean([r["trace"][0]["mahalanobis"] for r in all_results]))
        summary["mean_final_dist"] = float(np.mean([r["best_dist"] for r in all_results]))
        summary["mean_n_steps"] = float(np.mean([r["n_steps"] for r in all_results]))
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== Summary ===")
    log(f"  cases: {summary['n_cases']}")
    if all_results:
        log(f"  mean_initial_dist: {summary['mean_initial_dist']:.3f}")
        log(f"  mean_final_dist:   {summary['mean_final_dist']:.3f}")
        log(f"  improvement:       {summary['mean_initial_dist'] - summary['mean_final_dist']:+.3f}")
        log(f"  mean_n_steps:      {summary['mean_n_steps']:.1f}")
    log(f"已写入 {OUT_REFINED}, {OUT_TRACE}, {OUT_SUMMARY}")


if __name__ == "__main__":
    main()