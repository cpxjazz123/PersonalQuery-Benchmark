#!/usr/bin/env python3
"""SErCL user-profile: per-user syntax-conditioned error co-occurrence.

设计 (用户 2026-09-06 提案):
  用户真实 review x → GEC 改正 → x' → ERRANT edit alignment → UD parse(before/after)
  → SErCL error type (op + UPOS + DEP + morph change) → D3 context (parent.dep.child)
  → 累加 C_u(r, e) matrix → P_u(e|r) + personalization lift L_u(r, e) = log P_u/P_global

与现有 Gaussian profile 正交: μ_u + Σ_u (写作风格) + P_u(e|r) (写作错误规律)

输入: result/02_user_review_sentence_extract/uid_to_sentences.pkl
输出: result/09_sercl_user_profile/user_sercl_profile.json
       (包含 user_profiles + user_word_edits: per-uid 词级 incorrect→corrected pair)
       result/09_sercl_user_profile/cohort_summary.json

用法 (Rule 3: 无参数):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 09_sercl_user_profile/sercl_user_profile.py
"""
from __future__ import annotations

import collections
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/09_sercl_user_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Hardcoded hyperparams (Rule 3) -----
SERCL_N_USERS = None                # 2026-09-06 full: None = 全量符合条件的 selected uids
SERCL_MIN_SENTS = 30                # 每个用户至少这么多句子 (Rule 20: smoke 小)
SERCL_MAX_SENTS_PER_USER = 50    # 2026-09-06: 每用户最多 50 句 (None=不限); tradeoff: stats stability vs compute
SERCL_SEED = 42
# 2026-09-06: vLLM BART → GECToR-2024 RoBERTa-large via subprocess to gector_env
# (BART 把品牌名/产品词当语法错改, GECToR 的 token-level confidence gate 保留 review 域特异性)
SERCL_GEC_MODEL = "gector-2024-roberta-large"  # 355M RoBERTa-large, Write&Improve+CoNLL14+JFLEG
SERCL_GEC_SUBPROCESS_PY = "/home/wlia0047/hj82_scratch2/wenyu/venvs/gector_env/bin/python"
SERCL_GEC_SUBPROCESS_SCRIPT = str(REPO_ROOT / "09_sercl_user_profile/gector_subprocess.py")
SERCL_GEC_IN_JSONL = Path("/home/wlia0047/hj82_scratch2/wenyu/tmp/gec_in.jsonl")
SERCL_GEC_OUT_JSONL = Path("/home/wlia0047/hj82_scratch2/wenyu/tmp/gec_out.jsonl")
SERCL_ALPHA = 0.1                   # Laplace 平滑
SERCL_D3_TUPLE = ("dep", "pos")     # (child.dep, child.pos, parent.pos) — 3-tuple string
# ERRANT error types
SERCL_OPS = ("R:", "M:", "U:")      # replacement / missing / unnecessary
# 高频 error type 白名单 (后续可扩展)
SERCL_TOP_ERROR_TYPES = None        # None = 不限; v1 不限, 看分布

UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract/uid_to_sentences.json"
SELECTED_QUERIES = REPO_ROOT / "result/08_select_query/selected_queries.json"  # 2026-09-06: cohort 必须从此选


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Stage 1: Load cohort (2026-09-06: 必须来自 selected_queries.json)
# ===========================================================================
def load_cohort() -> Tuple[List[str], Dict[str, List[str]]]:
    """Cohort = uids with ≥MIN_SENTS sentences AND appearing in selected_queries.json.

    按 uid 字典序排序保证可复现, 取前 N_USERS 作为 smoke cohort。
    """
    log(f"=== Stage 1: load cohort from {SELECTED_QUERIES.name} ∩ {UID_TO_SENTS.name} ===")
    # Step 1: extract uids from selected_queries.json
    with open(SELECTED_QUERIES, encoding="utf-8") as f:
        sel_data = json.load(f)
    selected_uids = set()
    for task in sel_data.get("selections", []):
        for u in task.get("users", []):
            uid = u.get("uid")
            if uid:
                selected_uids.add(uid)
    log(f"  unique uids in selected_queries.json: {len(selected_uids)}")

    # Step 2: load sentences (JSON dict: {uid: [sent, ...]})
    with open(UID_TO_SENTS, encoding="utf-8") as f:
        all_uids = json.load(f)
    log(f"  total uids in uid_to_sentences: {len(all_uids)}")

    # Step 3: intersect selected ∩ eligible (≥MIN_SENTS)
    eligible = []
    for uid in selected_uids:
        s = all_uids.get(uid)
        if isinstance(s, list) and len(s) >= SERCL_MIN_SENTS:
            eligible.append((uid, s))
    log(f"  eligible selected uids (≥{SERCL_MIN_SENTS} sents): {len(eligible)}")

    # Step 4: sort by uid (deterministic) → take first N_USERS (None = all)
    eligible.sort(key=lambda x: x[0])
    if SERCL_N_USERS is None:
        cohort = dict(eligible)
        log(f"  selected cohort: {len(cohort)} uids (no cap, all eligible)")
    else:
        cohort = dict(eligible[:SERCL_N_USERS])
        log(f"  selected cohort: {len(cohort)} uids (capped at SERCL_N_USERS={SERCL_N_USERS})")
    return list(cohort.keys()), cohort


# ===========================================================================
# Stage 2: GEC correction via GECToR-2024 RoBERTa-large (subprocess to gector_env)
# ===========================================================================
# pq_env = Python 3.10 + transformers 4.43.2; GECToR requires transformers >=4.49 + torch >=2.5.
# 必须从独立 gector_env (Python 3.11) 子进程调用, 不能 in-process import.
# Subprocess 通过 JSONL 文件 IPC, 单次启动加载模型一次, 跑完所有句子.


def gec_correct_batch(sents: List[str]) -> List[str]:
    """Write sents → JSONL → spawn gector_env subprocess → read corrected JSONL."""
    if not sents:
        return []
    SERCL_GEC_IN_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Write input JSONL
    t_w = time.time()
    with open(SERCL_GEC_IN_JSONL, "w", encoding="utf-8") as f:
        for i, s in enumerate(sents):
            f.write(json.dumps({"i": i, "text": s}, ensure_ascii=False) + "\n")
    log(f"  wrote input JSONL: {SERCL_GEC_IN_JSONL} (n={len(sents)}, {time.time()-t_w:.1f}s)")

    # Spawn subprocess (use -u for unbuffered stdout, so iteration logs appear in real-time)
    log(f"  spawning: {SERCL_GEC_SUBPROCESS_PY} -u {SERCL_GEC_SUBPROCESS_SCRIPT}")
    t0 = time.time()
    proc = subprocess.run(
        [SERCL_GEC_SUBPROCESS_PY, "-u", SERCL_GEC_SUBPROCESS_SCRIPT],
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"GECToR subprocess failed with exit code {proc.returncode}. "
            f"See stderr above (tail logs in /home/wlia0047/hj82_scratch2/wenyu/logs/)."
        )

    # Read output JSONL, sort by index
    if not SERCL_GEC_OUT_JSONL.exists():
        raise RuntimeError(f"GECToR subprocess did not produce: {SERCL_GEC_OUT_JSONL}")
    out = [None] * len(sents)
    with open(SERCL_GEC_OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            i = obj["i"]
            if i < 0 or i >= len(sents):
                raise RuntimeError(f"GECToR output index out of range: {i}")
            out[i] = obj["text"]
    if any(x is None for x in out):
        missing = [i for i, x in enumerate(out) if x is None]
        raise RuntimeError(f"GECToR output missing indices: {missing[:10]}..."
                           f" (have {sum(x is not None for x in out)}/{len(sents)})")
    rate = len(sents) / (time.time() - t0 + 1e-9)
    log(f"  GECToR subprocess done in {time.time()-t0:.1f}s (incl. model load), "
        f"n={len(out)}, rate={rate:.1f}/s")
    return out


# ===========================================================================
# Stage 3: ERRANT alignment + SErCL error type extraction
# ===========================================================================
_SPACY_NLP = None
_ERRANT_ANN = None


def _load_spacy_errant():
    global _SPACY_NLP, _ERRANT_ANN
    if _SPACY_NLP is not None:
        return _SPACY_NLP, _ERRANT_ANN
    import spacy
    import errant
    log(f"  loading spaCy en_core_web_sm + ERRANT ann")
    _SPACY_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    _ERRANT_ANN = errant.load("en", _SPACY_NLP)
    return _SPACY_NLP, _ERRANT_ANN


def _d3_context(token) -> str:
    """D3 = (child.dep, child.pos, parent.pos)."""
    parent = token.head
    return f"{token.dep_}|{token.pos_}|{parent.pos_}"


def _morph_diff(t_orig, t_cor) -> str:
    """Compare morph features; return 'f1=v1→f2=v2' for diffs else ''."""
    m_orig = {m for m in str(t_orig.morph).split("|") if m} if t_orig is not None else set()
    m_cor = {m for m in str(t_cor.morph).split("|") if m} if t_cor is not None else set()
    diffs = []
    for f in sorted(m_orig | m_cor):
        v_o = next((x.split("=")[1] for x in m_orig if x.startswith(f + "=")), "_")
        v_c = next((x.split("=")[1] for x in m_cor if x.startswith(f + "=")), "_")
        if v_o != v_c:
            diffs.append(f"{f}:{v_o}→{v_c}")
    return ";".join(diffs[:3])  # 限制 morph diff 长度


def extract_error_records(orig_sents: List[str], cor_sents: List[str]
                          ) -> List[Tuple[str, str, int, Dict[str, str]]]:
    """Return [(error_type, d3_context, sent_idx, word_pair)] per edit.

    word_pair = {"op": "R:"|"M:"|"U:", "incorrect_word": str, "corrected_word": str}
    """
    nlp, ann = _load_spacy_errant()
    records = []
    for si, (x, x_prime) in enumerate(zip(orig_sents, cor_sents)):
        if x.strip() == x_prime.strip():
            continue  # no edits
        try:
            orig = ann.parse(x)
            cor = ann.parse(x_prime)
            edits = ann.annotate(orig, cor)
        except Exception:
            continue
        # Build token arrays for morph lookup
        orig_tokens = [t for t in orig]
        cor_tokens = [t for t in cor]
        for e in edits:
            op = e.type[0] + ":"  # 'R:', 'M:', 'U:'
            if op not in SERCL_OPS:
                continue
            if op == "R:":
                t_o = orig_tokens[e.o_start] if e.o_start < len(orig_tokens) else None
                t_c = cor_tokens[e.c_start] if e.c_start < len(cor_tokens) else None
                upos_b = t_o.pos_ if t_o else "_"
                upos_a = t_c.pos_ if t_c else "_"
                dep_b = t_o.dep_ if t_o else "_"
                dep_a = t_c.dep_ if t_c else "_"
                morph_diff = _morph_diff(t_o, t_c)
                err_type = f"R:{upos_b}→{upos_a}|{dep_b}→{dep_a}"
                if morph_diff:
                    err_type += f"|{morph_diff}"
                # D3 from the orig-token (which carries the dep relations in orig parse)
                ctx = _d3_context(t_o) if t_o is not None else "_|_|_"
                incorrect_word = t_o.text if t_o else ""
                corrected_word = t_c.text if t_c else ""
            elif op == "M:":
                # Missing token: use the position right after edit in cor
                anchor_idx = min(e.c_start, len(cor_tokens) - 1)
                t_c = cor_tokens[anchor_idx]
                upos_a = t_c.pos_
                dep_a = t_c.dep_
                err_type = f"M:{upos_a}|{dep_a}"
                ctx = _d3_context(t_c)
                incorrect_word = ""
                corrected_word = t_c.text
            else:  # U:
                anchor_idx = min(e.o_start, len(orig_tokens) - 1)
                t_o = orig_tokens[anchor_idx]
                upos_b = t_o.pos_
                dep_b = t_o.dep_
                err_type = f"U:{upos_b}|{dep_b}"
                ctx = _d3_context(t_o)
                incorrect_word = t_o.text
                corrected_word = ""
            word_pair = {
                "op": op,
                "incorrect_word": incorrect_word,
                "corrected_word": corrected_word,
            }
            records.append((err_type, ctx, si, word_pair))
    return records


# ===========================================================================
# Stage 4: Co-occurrence + P_u(e|r) + L_u(r,e)
# ===========================================================================
def build_profiles(uid_to_records: Dict[str, List[Tuple[str, str]]],
                   alpha: float = SERCL_ALPHA):
    """Per-user C_u(r,e) → P_u(e|r), plus L_u(r,e) = log P_u/P_global."""
    # Global counts (for P_global)
    C_global: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    total_edits_global = 0
    for recs in uid_to_records.values():
        for et, ctx, _ in recs:
            C_global[ctx][et] += 1
            total_edits_global += 1
    n_e = len({et for recs in uid_to_records.values() for et, _, _ in recs})
    log(f"  global: n_contexts={len(C_global)}, n_error_types={n_e}, "
        f"total_edits={total_edits_global}")

    P_global = {}
    for ctx, ec in C_global.items():
        denom = sum(ec.values()) + alpha * n_e
        P_global[ctx] = {et: (c + alpha) / denom for et, c in ec.items()}

    user_profiles = {}
    for uid, recs in uid_to_records.items():
        C_u: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for et, ctx, _ in recs:
            C_u[ctx][et] += 1
        P_u = {}
        L_u = {}
        for ctx, ec in C_u.items():
            denom = sum(ec.values()) + alpha * n_e
            P_u[ctx] = {et: (c + alpha) / denom for et, c in ec.items()}
            L_u[ctx] = {}
            if ctx in P_global:
                for et, p in P_u[ctx].items():
                    pg = P_global[ctx].get(et, alpha / (denom + alpha * n_e))
                    L_u[ctx][et] = float(__import__("math").log(p / pg)) if pg > 0 else 0.0
        user_profiles[uid] = {
            "n_edits": len(recs),
            "n_contexts": len(C_u),
            "C_u": {ctx: dict(c) for ctx, c in C_u.items()},
            "P_u": {ctx: {et: float(v) for et, v in d.items()} for ctx, d in P_u.items()},
            "L_u": {ctx: {et: float(v) for et, v in d.items()} for ctx, d in L_u.items()},
        }
    return user_profiles, P_global, n_e


# ===========================================================================
# Main
# ===========================================================================
def main_pipeline():
    t_start = time.time()
    log("=== SErCL user-profile full run ===")
    log(f"  N_USERS={SERCL_N_USERS}  MIN_SENTS={SERCL_MIN_SENTS}  "
        f"MAX_PER_USER={SERCL_MAX_SENTS_PER_USER}  ALPHA={SERCL_ALPHA}")

    # Stage 1
    uids, cohort = load_cohort()

    # Stage 2: GEC correction (GECToR-2024 RoBERTa-large via gector_env subprocess)
    log("\n=== Stage 2: GEC correction (GECToR-2024 via gector_env subprocess) ===")
    all_orig, all_idx = [], []
    for uid in uids:
        sents = cohort[uid] if SERCL_MAX_SENTS_PER_USER is None \
                else cohort[uid][:SERCL_MAX_SENTS_PER_USER]
        for s in sents:
            all_orig.append(s.strip())
            all_idx.append(uid)
    log(f"  total sents to correct: {len(all_orig)}")
    corrected = gec_correct_batch(all_orig)
    assert len(corrected) == len(all_orig)

    # Stage 3: ERRANT alignment + UD parse + extract (err_type, D3)
    log("\n=== Stage 3: ERRANT align + UD parse + SErCL extract ===")
    # Group by uid for batch processing
    uid_to_records: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    user_word_edits: Dict[str, List[Dict]] = {uid: [] for uid in uids}
    n_no_edit = 0
    for uid, x, x_prime in zip(all_idx, all_orig, corrected):
        recs = extract_error_records([x], [x_prime])
        if not recs:
            n_no_edit += 1
        sent_edits: List[Dict[str, str]] = []
        for et, ctx, _si, wp in recs:
            uid_to_records[uid].append((et, ctx, uid))
            sent_edits.append(wp)
        if sent_edits:
            user_word_edits[uid].append({
                "orig_sent": x,
                "cor_sent": x_prime,
                "edits": sent_edits,
            })
    total_edits = sum(len(v) for v in uid_to_records.values())
    n_users_with_edits = sum(1 for v in uid_to_records.values() if v)
    log(f"  total edits: {total_edits}  (no-edit sents: {n_no_edit}/{len(all_orig)})")
    log(f"  users with ≥1 edit: {n_users_with_edits}/{len(uids)}")

    # Stage 4: build profiles
    log("\n=== Stage 4: build P_u(e|r) + L_u(r,e) ===")
    user_profiles, P_global, n_e = build_profiles(uid_to_records)

    # Save
    out_path = OUT_DIR / "user_sercl_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_users": SERCL_N_USERS,
                "min_sents": SERCL_MIN_SENTS,
                "max_per_user": SERCL_MAX_SENTS_PER_USER,
                "alpha": SERCL_ALPHA,
                "gec_model": SERCL_GEC_MODEL,
                "seed": SERCL_SEED,
                "d3_tuple": SERCL_D3_TUPLE,
            },
            "n_error_types": n_e,
            "P_global": P_global,
            "user_profiles": user_profiles,
            "user_word_edits": user_word_edits,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # Cohort summary
    summary = {
        "n_users": len(uids),
        "n_sents_total": len(all_orig),
        "n_sents_no_edit": n_no_edit,
        "n_edits_total": total_edits,
        "n_users_with_edits": n_users_with_edits,
        "n_error_types_global": n_e,
        "n_contexts_global": len(P_global),
        "edit_rate_per_user": {
            uid: len(uid_to_records[uid]) for uid in uids
        },
    }
    sum_path = OUT_DIR / "cohort_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"  wrote → {sum_path}")

    # Sanity: top error types
    log("\n=== Top 10 global error types ===")
    type_counts = collections.Counter()
    for recs in uid_to_records.values():
        for et, _, _ in recs:
            type_counts[et] += 1
    for et, c in type_counts.most_common(10):
        log(f"  {c:>5}  {et}")

    # Sanity: user-specific signal (off-diag cosine sim of L_u vectors)
    log("\n=== Sanity: user-specific signal ===")
    all_cells = sorted({(ctx, et) for ctx in P_global for et in P_global[ctx]})
    import numpy as np
    vecs = {}
    for uid, p in user_profiles.items():
        v = np.zeros(len(all_cells))
        for i, (ctx, et) in enumerate(all_cells):
            v[i] = p["L_u"].get(ctx, {}).get(et, 0.0)
        vecs[uid] = v
    uids = list(vecs.keys())
    sim = np.zeros((len(uids), len(uids)))
    for i, u1 in enumerate(uids):
        for j, u2 in enumerate(uids):
            v1, v2 = vecs[u1], vecs[u2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            sim[i, j] = (v1 @ v2) / (n1 * n2 + 1e-12)
    mask = ~np.eye(len(uids), dtype=bool)
    log(f"  off-diag L_u cosine sim: mean={sim[mask].mean():.4f}  "
        f"median={np.median(sim[mask]):.4f}  std={sim[mask].std():.4f}")
    log(f"  (low → user-specific signal present; target < 0.5)")

    log(f"\n=== SErCL run complete ({time.time()-t_start:.1f}s) ===")


if __name__ == "__main__":
    main_pipeline()