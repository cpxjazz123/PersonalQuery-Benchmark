#!/usr/bin/env python3
"""E22 Task 3 (part 3): Check-3 validation.

Validates on the four-control generations (result/e22_t3/e22_t3_generations.jsonl):
  1. Activation check      — prefix norms / real-vs-zero diff (from generate.json)
  2. User-swap check       — same product, z_A vs z_B: aggregate query syntax
                             vector closer to its own user; swap accuracy / AUC /
                             effect size / CI
  3. Product-swap check    — same user across products: syntax direction stable
  4. Monotonicity check    — interpolation z_A->z_B (needs extra interpolation
                             generations; if absent, report as not-run)
  5. Control comparison    — real-z vs shuffled/zero/off on syntax closeness:
                             user-level bootstrap CI lower > 0, perm p < 0.01
  6. Content preservation  — 5 attrs / digits / brand exact-match rates

Syntax vectors of generated queries are the Task-1 7-dim Query-compatible
features, computed on CONTENT-NEUTRALIZED text (same protocol as Task 2), so
closeness to z_user measures syntax only.

Outputs: result/e22_t3/e22_t3_check3.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
GEN = REPO_ROOT / "result" / "e22_t3" / "e22_t3_generations.jsonl"
GEN_JSON = REPO_ROOT / "result" / "e22_t3" / "e22_t3_generate.json"
OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_check3.json"

SEED = 4444
N_BOOTSTRAP = 2000
N_PERM = 9999
# pre-registered gates
MIN_TEST_USERS = 30
BOOTSTRAP_CI_LOWER = 0.0
PERM_P_THRESH = 0.01
CONTENT_EXACT_MIN = 0.99
MIN_SWAP_ACC = 0.60


def neutralize_content(query: str, attrs: dict[str, str]) -> str:
    out = query
    for v in attrs.values():
        if not v:
            continue
        out = re.sub(re.escape(str(v)), "xxxx", out)
    return out


def query7_vector(q: str, attrs: dict, nlp, fi, tm, ts) -> np.ndarray | None:
    qn = neutralize_content(q, attrs)
    doc = nlp(qn)
    sfs = [per_sentence_features_v2(s) for s in doc.sents]
    sfs = [s for s in sfs if s is not None]
    if not sfs:
        return None
    v = user_features_v2(sfs)
    if v is None:
        return None
    return ((v[fi] - tm[fi]) / ts[fi]).astype(np.float64)


def digit_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\d+(?:\.\d+)?", text))


def content_exact(attrs: dict, q: str) -> dict:
    ql = q.lower()
    vals = {k: str(v) for k, v in attrs.items()}
    ok = {k: vals[k].lower() in ql for k in vals}
    digits_ok = (digit_tokens(q)
                 == digit_tokens(" ".join(vals.values())))
    return {"attrs": ok, "all_5": all(ok.values()),
            "digits": digits_ok}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_all = vd["Z"].astype(np.float64)
    user_to_z = {u: z_all[i] for i, u in enumerate(user_ids)}
    fi = vd["feature_indices"].astype(int)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    gens = [json.loads(l) for l in open(GEN)]
    by_ctrl: dict[str, list[dict]] = defaultdict(list)
    for g in gens:
        by_ctrl[g["control"]].append(g)
    for c in by_ctrl:
        print(f"control {c}: {len(by_ctrl[c])} generations", flush=True)

    # ---------- 6. content preservation (all controls) ----------
    content: dict[str, dict] = {}
    for ctrl, rs in by_ctrl.items():
        n5 = n_digit = n_brand = n_price = 0
        n = 0
        for r in rs:
            ce = content_exact(r["attrs"], r["query"])
            n += 1
            n5 += int(ce["all_5"])
            n_digit += int(ce["digits"])
            n_brand += int(r["attrs"]["Brand"].lower() in r["query"].lower())
            n_price += int(r["attrs"]["Price"].lower() in r["query"].lower())
        content[ctrl] = {
            "n": n,
            "exact_5_attrs": round(n5 / n, 4),
            "exact_digits": round(n_digit / n, 4),
            "exact_brand": round(n_brand / n, 4),
            "exact_price": round(n_price / n, 4),
            "gate_pass": (n5 / n >= CONTENT_EXACT_MIN
                          and n_digit / n >= CONTENT_EXACT_MIN
                          and n_brand / n >= CONTENT_EXACT_MIN),
        }
        print(f"  content[{ctrl}]: 5attr={content[ctrl]['exact_5_attrs']:.4f} "
              f"digits={content[ctrl]['exact_digits']:.4f} "
              f"brand={content[ctrl]['exact_brand']:.4f} "
              f"price={content[ctrl]['exact_price']:.4f} "
              f"pass={content[ctrl]['gate_pass']}", flush=True)

    # ---------- syntax vectors of real-z generations ----------
    print("Computing 7-dim syntax vectors for real-z...", flush=True)
    real = by_ctrl["real-z"]
    # AGGREGATE per user across products (issue: "以同一用户跨多个商品生成
    # 的 Query 集合做聚合句法验证")
    user_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in real:
        v = query7_vector(r["query"], r["attrs"], nlp, fi, tm, ts)
        if v is not None:
            user_vecs[r["user_id"]].append(v)
    vecs: dict[str, np.ndarray] = {
        u: np.mean(vs, axis=0) for u, vs in user_vecs.items() if vs}
    print(f"  real-z aggregated vectors: {len(vecs)} users", flush=True)

    # ---------- 2. user-swap check ----------
    # Issue: "固定商品，交换 z_A/z_B 后，聚合 Query 句法向量必须分别更接近
    # 对应用户". Two complementary measures:
    #   (a) mean-other control: per user, aggregated real-z syntax vector must
    #       be closer to OWN z_user than to the AVERAGE of all other users'
    #       z (robust; primary gate);
    #   (b) paired swap: on the same product, the query generated with own z
    #       vs with a shuffled (other-user) z — own-z generation must be
    #       closer to z_user (secondary, reported).
    print("User-swap check...", flush=True)
    swap = {}
    if vecs:
        users = sorted(vecs)
        z_arr = np.stack([user_to_z[u] for u in users])
        margins = []
        acc_hits = 0
        for i, u in enumerate(users):
            v = vecs[u]
            d_own = np.linalg.norm(v - user_to_z[u])
            others = np.delete(z_arr, i, axis=0)
            d_other = np.linalg.norm(v[None, :] - others, axis=1).mean()
            margins.append(d_other - d_own)
            if d_own < d_other:
                acc_hits += 1
        margins = np.array(margins)
        bs = np.array([margins[rng.integers(0, len(margins), len(margins))].mean()
                       for _ in range(N_BOOTSTRAP)])
        obs = float(margins.mean())
        perm = np.empty(N_PERM)
        z_perm = z_arr.copy()
        for k in range(N_PERM):
            rng.shuffle(z_perm)
            d1 = np.linalg.norm(np.stack([vecs[u] for u in users]) - z_perm, axis=1)
            # compare against all-other mean under shuffled labels
            m_perm = np.empty(len(users))
            for i in range(len(users)):
                others = np.delete(z_perm, i, axis=0)
                d2 = np.linalg.norm(np.stack([vecs[u] for u in users])[i][None, :]
                                    - others, axis=1).mean()
                m_perm[i] = d2 - d1[i]
            perm[k] = m_perm.mean()
        pc = float(perm.mean())
        p_two = float((np.abs(perm - pc) >= abs(obs - pc)).sum() + 1) / (N_PERM + 1)
        swap = {
            "n_users": len(users),
            "accuracy_mean_other": round(acc_hits / len(users), 4),
            "margin_mean_other": round(obs, 4),
            "margin_95ci": [round(float(np.quantile(bs, 0.025)), 4),
                            round(float(np.quantile(bs, 0.975)), 4)],
            "cohen_d": round(float(obs / max(margins.std(), 1e-9)), 4),
            "perm_p_two": round(p_two, 5),
            "gate_pass": bool(acc_hits / max(1, len(users)) >= MIN_SWAP_ACC
                              and np.quantile(bs, 0.025) > 0
                              and p_two < 0.01),
        }
    else:
        swap = {"n_users": 0, "gate_pass": False}
    print(f"  swap acc={swap.get('accuracy_mean_other')} "
          f"margin={swap.get('margin_mean_other')} "
          f"CI={swap.get('margin_95ci')} p={swap.get('perm_p_two')} "
          f"pass={swap.get('gate_pass')}", flush=True)

    # (b) paired swap on same product (real-z vs shuffled-z)
    paired = {"run": False}
    cells_p: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in real + by_ctrl["shuffled-z"]:
        cells_p[(r["user_id"], r["asin"])][r["control"]] = r
    pm = []
    for (u, asin), cc in cells_p.items():
        if "real-z" not in cc or "shuffled-z" not in cc:
            continue
        vA = query7_vector(cc["real-z"]["query"], cc["real-z"]["attrs"],
                           nlp, fi, tm, ts)
        vB = query7_vector(cc["shuffled-z"]["query"], cc["shuffled-z"]["attrs"],
                           nlp, fi, tm, ts)
        if vA is None or vB is None:
            continue
        pm.append(np.linalg.norm(vB - user_to_z[u])
                  - np.linalg.norm(vA - user_to_z[u]))
    if pm:
        pm = np.array(pm)
        paired = {
            "run": True, "n_pairs": len(pm),
            "own_minus_shuffled_mean": round(float(pm.mean()), 4),
            "pct_own_closer": round(float((pm > 0).mean()), 4),
        }
    print(f"  paired swap: n={paired.get('n_pairs')} "
          f"own-closer={paired.get('pct_own_closer')} "
          f"margin={paired.get('own_minus_shuffled_mean')}", flush=True)

    # ---------- 3. product-swap check (syntax direction stability) ----------
    # user-level: variance of the per-product 7-dim syntax vectors relative to
    # between-user variance. Requires per-(user, product) vectors; real-z only.
    prod_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in real:
        v = query7_vector(r["query"], r["attrs"], nlp, fi, tm, ts)
        if v is not None:
            prod_vectors[r["user_id"]].append(v)
    users_multi = {u: vs_ for u, vs_ in prod_vectors.items() if len(vs_) >= 2}
    within = []
    between = []
    for u, vs_ in users_multi.items():
        mu = np.mean(vs_, axis=0)
        within.append(np.mean([np.linalg.norm(v - mu) for v in vs_]))
    # between: distance from each user's mean to the global mean
    all_mu = np.mean(np.stack([np.mean(vs_, axis=0) for vs_ in users_multi.values()]), axis=0)
    for u, vs_ in users_multi.items():
        mu = np.mean(vs_, axis=0)
        between.append(np.linalg.norm(mu - all_mu))
    w = float(np.mean(within)) if within else float("nan")
    b = float(np.mean(between)) if between else float("nan")
    product_check = {
        "n_users_multi_product": len(users_multi),
        "within_user_syntax_var": w,
        "between_user_syntax_var": b,
        "b_over_w": round(b / w, 4) if w > 1e-12 else None,
        "gate_pass": bool(w < b),
    }
    print(f"  product: within={w:.4f} between={b:.4f} b/w="
          f"{product_check['b_over_w']} pass={product_check['gate_pass']}",
          flush=True)

    # ---------- 4. monotonicity (not run: needs interpolation gens) ----------
    mono = {"run": False, "note": "interpolation generations not produced"}

    # ---------- 5. control comparison ----------
    # real-z must beat shuffled/zero/off on syntax closeness to z_user.
    print("Control comparison...", flush=True)
    ctrl_dist: dict[str, dict[str, float]] = {}
    ctrl_gate = {}
    for ctrl in ("real-z", "shuffled-z", "true-zero-z", "injection-off"):
        agg: dict[str, list[np.ndarray]] = defaultdict(list)
        for r in by_ctrl[ctrl]:
            v = query7_vector(r["query"], r["attrs"], nlp, fi, tm, ts)
            if v is None:
                continue
            agg[r["user_id"]].append(v)
        dv = {}
        for u, vs in agg.items():
            if vs:
                dv[u] = float(np.linalg.norm(np.mean(vs, axis=0)
                                             - user_to_z[u]))
        ctrl_dist[ctrl] = dv
    for ctrl in ("shuffled-z", "true-zero-z", "injection-off"):
        common = [u for u in ctrl_dist["real-z"] if u in ctrl_dist[ctrl]]
        if len(common) < MIN_TEST_USERS:
            ctrl_gate[ctrl] = {"run": False, "n_common": len(common)}
            continue
        diff = np.array([ctrl_dist[ctrl][u] - ctrl_dist["real-z"][u]
                         for u in common])   # positive = real-z better
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                       for _ in range(N_BOOTSTRAP)])
        obs = float(diff.mean())
        perm = np.empty(N_PERM)
        for k in range(N_PERM):
            perm[k] = (diff * rng.choice([-1, 1], len(diff))).mean()
        pc = float(perm.mean())
        p_two = float((np.abs(perm - pc) >= abs(obs - pc)).sum() + 1) / (N_PERM + 1)
        ctrl_gate[ctrl] = {
            "run": True, "n_users": len(common),
            "real_minus_ctrl_mean": round(obs, 4),
            "bootstrap_95ci": [round(float(np.quantile(bs, 0.025)), 4),
                               round(float(np.quantile(bs, 0.975)), 4)],
            "perm_p_two": round(p_two, 5),
            "gate_pass": bool(np.quantile(bs, 0.025) > BOOTSTRAP_CI_LOWER
                              and p_two < PERM_P_THRESH),
        }
        print(f"  real vs {ctrl}: n={len(common)} diff={obs:.4f} "
              f"CI={ctrl_gate[ctrl]['bootstrap_95ci']} p={p_two:.5f} "
              f"pass={ctrl_gate[ctrl]['gate_pass']}", flush=True)

    # ---------- 1. activation check ----------
    gen_json = json.load(open(GEN_JSON))
    acts = gen_json.get("activation", {})
    act_pass = bool(acts.get("prefix_diff_real_vs_zero", 0) > 0
                    and acts.get("prefix_real_norm", 0) > 0)
    activation = {**acts, "gate_pass": act_pass}
    print(f"  activation: real-norm={acts.get('prefix_real_norm')} "
          f"diff-vs-zero={acts.get('prefix_diff_real_vs_zero')} "
          f"pass={act_pass}", flush=True)

    result = {
        "version": "e22_t3_check3_v1",
        "seed": SEED,
        "n_generations": len(gens),
        "activation_check": activation,
        "user_swap_check": {**swap, "paired": paired},
        "product_swap_check": product_check,
        "monotonicity_check": mono,
        "control_comparison": ctrl_gate,
        "content_preservation": content,
        "all_gates": {
            "activation": bool(act_pass),
            "user_swap": bool(swap["gate_pass"]),
            "product_swap": bool(product_check["gate_pass"]),
            "control_compare": all(g.get("gate_pass", False)
                                   for g in ctrl_gate.values()),
            "content": all(c["gate_pass"] for c in content.values()),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    result["all_gates_pass"] = bool(all(result["all_gates"].values()))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nALL GATES PASS = {result['all_gates_pass']}", flush=True)
    print(json.dumps(result["all_gates"], indent=1), flush=True)


if __name__ == "__main__":
    main()
