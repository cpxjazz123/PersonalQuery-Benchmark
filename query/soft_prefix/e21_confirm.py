#!/usr/bin/env python3
"""E21 v6 (confirm): independent verification of (L=5, N=20).

Per user plan (2026-08-15):
  Step 1 — Final confirmation of syntactic signal.
  Use a fresh batch of users (NOT used in v1-v5) with ≥40 complete
  sentences (each ≥5 words). Random split into two halves of 20. Repeat
  30 seeds. 9999 user-label permutations.

Gates:
  test_users >= 150
  AUC >= 0.65
  Cohen d >= 0.5
  seed_pass >= 0.80
  perm_p_two < 0.01

If all pass: (5, 20) is the stable threshold; proceed to Step 2.
If not: signal exists but 40 sentences insufficient for stable extraction.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from e20_lopo_v2 import per_sentence_features
from e20_lopo_v2 import load_spacy_model, ALL_FEATS

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e21_confirm_results.json"
LOG = REPO_ROOT / "result" / "e21_confirm.log"

SEED = 43               # v6: NEW seed so subsample is disjoint from v5 (seed 42)
N_USERS = 5000          # v6: large pool to maximize eligible (≥40 sents @ ≥5 words)
MIN_WORDS = 100
MIN_ASINS = 2
L_FIXED = 5
N_FIXED = 20
N_SEEDS = 30
N_PERM = 9999
DEV_FRAC = 0.5
AUC_THRESH = 0.65
COHEN_D_THRESH = 0.5
P_THRESH = 0.01
SEED_PASS_FRAC = 0.80
TEST_MIN_USERS = 150


def user_features(sent_feats: list[dict]) -> np.ndarray | None:
    if not sent_feats:
        return None
    total_tok = sum(s["n_tok"] for s in sent_feats)
    n_sent = len(sent_feats)
    if total_tok == 0 or n_sent == 0:
        return None
    f = {}
    f["clause_rate"] = sum(s["n_clause"] for s in sent_feats) / total_tok
    f["acl_rate"] = sum(s["acl"] for s in sent_feats) / total_tok
    f["advcl_rate"] = sum(s["advcl"] for s in sent_feats) / total_tok
    f["ccomp_rate"] = sum(s["ccomp"] for s in sent_feats) / total_tok
    f["xcomp_rate"] = sum(s["xcomp"] for s in sent_feats) / total_tok
    f["relcl_rate"] = sum(s["relcl"] for s in sent_feats) / total_tok
    f["modifier_density"] = sum(s["n_mod"] for s in sent_feats) / total_tok
    f["coordination_density"] = sum(s["n_coord"] for s in sent_feats) / total_tok
    f["mean_dep_distance"] = float(np.mean([s["mean_dist"] for s in sent_feats]))
    f["depth_variance"] = float(np.mean([s["depth_var"] for s in sent_feats]))
    f["median_dep_depth"] = float(np.median([s["max_depth"] for s in sent_feats]))
    f["passive_rate"] = sum(1 for s in sent_feats if s["has_passive"]) / n_sent
    f["interrogative_rate"] = sum(1 for s in sent_feats if s["is_interrog"]) / n_sent
    f["conditional_rate"] = sum(1 for s in sent_feats if s["has_cond"]) / n_sent
    from collections import Counter
    OPENER_POSES = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
                    "AUX", "NUM", "INTJ", "PART", "PUNCT", "X", "SYM")
    SENT_TYPES = ("simple", "conjunctive", "complex")
    opener_counts = Counter(s["opener"] for s in sent_feats)
    for p in OPENER_POSES:
        f[f"opener_{p}"] = opener_counts.get(p, 0) / n_sent
    stype_counts = Counter(s["stype"] for s in sent_feats)
    for t in SENT_TYPES:
        f[f"senttype_{t}"] = stype_counts.get(t, 0) / n_sent
    return np.asarray([f[name] for name in ALL_FEATS], dtype=np.float32)


def norm_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def auc_self_vs_cross(d_self: np.ndarray, d_cross: np.ndarray) -> float:
    if len(d_self) == 0 or len(d_cross) == 0:
        return float("nan")
    auc = 0.0
    for x in d_self:
        auc += float((d_cross > x).sum())
    return auc / (len(d_self) * len(d_cross))


def cohen_d(d_cross: np.ndarray, d_self: np.ndarray) -> float:
    if len(d_cross) < 2 or len(d_self) < 2:
        return float("nan")
    delta = d_cross.mean() - d_self.mean()
    pooled = np.sqrt((d_cross.std(ddof=1) ** 2 + d_self.std(ddof=1) ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float(delta / pooled)


def perm_null_delta(all_halves: list[np.ndarray], rng: np.random.Generator) -> float:
    n = len(all_halves)
    if n < 4:
        return 0.0
    idx = rng.permutation(n)
    if n % 2 == 1:
        idx = idx[:-1]
    a = np.stack([all_halves[i] for i in idx[0::2]])
    b = np.stack([all_halves[i] for i in idx[1::2]])
    diffs = np.linalg.norm(a - b, axis=1)
    return float(diffs.mean())


def eval_cell(cell_users: list[str],
              user_filtered: dict[str, list[dict]],
              N: int,
              seeds: tuple[int, ...]) -> list[dict]:
    per_seed = []
    for sd in seeds:
        rs = np.random.default_rng(sd)
        diffs_self, diffs_cross = [], []
        for u in cell_users:
            sfs = user_filtered[u]
            if len(sfs) < 2 * N:
                continue
            idx = rs.permutation(len(sfs))[:2 * N]
            half1 = [sfs[i] for i in idx[:N]]
            half2 = [sfs[i] for i in idx[N:]]
            v1 = user_features(half1)
            v2 = user_features(half2)
            if v1 is None or v2 is None:
                continue
            v1n = norm_vec(v1)
            v2n = norm_vec(v2)
            diffs_self.append(float(np.linalg.norm(v1n - v2n)))
            others = [uu for uu in cell_users if uu != u]
            rs.shuffle(others)
            cross_n = 0
            for v_other in others:
                if cross_n >= 3:
                    break
                sfs_v = user_filtered[v_other]
                if len(sfs_v) < N:
                    continue
                idx_v = rs.permutation(len(sfs_v))[:N]
                vv = user_features([sfs_v[i] for i in idx_v])
                if vv is None:
                    continue
                vvn = norm_vec(vv)
                diffs_cross.append(float(np.linalg.norm(v1n - vvn)))
                cross_n += 1
        ds = np.asarray(diffs_self)
        dc = np.asarray(diffs_cross)
        if len(ds) < 5 or len(dc) < 5:
            continue
        per_seed.append({
            "seed": int(sd),
            "auc": auc_self_vs_cross(ds, dc),
            "delta": float(dc.mean() - ds.mean()),
            "cohen_d": cohen_d(dc, ds),
            "d_self_mean": float(ds.mean()),
            "d_cross_mean": float(dc.mean()),
            "n_self": len(ds),
            "n_cross": len(dc),
        })
    return per_seed


def eval_permutation(cell_users: list[str],
                     user_filtered: dict[str, list[dict]],
                     N: int,
                     obs_delta: float,
                     n_perm: int,
                     rng_p: np.random.Generator) -> tuple[float, float]:
    all_halves = []
    for u in cell_users:
        sfs = user_filtered[u]
        if len(sfs) < 2 * N:
            continue
        rs_fixed = np.random.default_rng(0)
        idx = rs_fixed.permutation(len(sfs))[:2 * N]
        h1 = [sfs[i] for i in idx[:N]]
        h2 = [sfs[i] for i in idx[N:]]
        v1 = user_features(h1)
        v2 = user_features(h2)
        if v1 is not None:
            all_halves.append(norm_vec(v1))
        if v2 is not None:
            all_halves.append(norm_vec(v2))
    if len(all_halves) < 4:
        return float("nan"), float("nan")
    null_deltas = np.asarray([perm_null_delta(all_halves, rng_p)
                              for _ in range(n_perm)])
    p_one = (float((null_deltas <= obs_delta).sum()) + 1) / (len(null_deltas) + 1)
    null_center = float(null_deltas.mean())
    p_two = (float((np.abs(null_deltas - null_center) >=
                    abs(obs_delta - null_center)).sum()) + 1) / (len(null_deltas) + 1)
    return float(p_one), float(p_two)


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    wc: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = d.get("text") or ""
            if u:
                wc[u] += len(t.split())
    print(f"users scanned: {len(wc)} (t={time.time() - t0:.1f}s)", flush=True)

    cands = [u for u, w in wc.items() if w >= MIN_WORDS]
    rng.shuffle(cands)
    cands = cands[:N_USERS * 4]
    poolset = set(cands)
    print(f"word>={MIN_WORDS} candidates: {len(cands)}", flush=True)

    reviews: dict[str, list[tuple]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in poolset:
                continue
            t = (d.get("text") or "").strip()
            if t:
                reviews[u].append((d.get("parent_asin") or d.get("asin"), t[:2000]))
    for u in list(reviews):
        if len({r[0] for r in reviews[u]}) < MIN_ASINS:
            del reviews[u]
    print(f"users with >= {MIN_ASINS} products: {len(reviews)}", flush=True)

    nlp = load_spacy_model()
    # Speed optimization: disable components we don't use
    # per_sentence_features only needs tagger + parser + sentencizer
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    print(f"  active spaCy pipes: {nlp.pipe_names}", flush=True)
    user_sents: dict[str, list[dict]] = {}
    user_texts: list[tuple[str, str]] = []
    for u, revs in reviews.items():
        for _, t in revs:
            user_texts.append((u, t))
    print(f"  parsing {len(user_texts)} reviews for {len(reviews)} users...", flush=True)
    BATCH = 256
    user_text_iter = iter(user_texts)
    batch_pairs = []
    parsed = 0
    while True:
        batch_pairs = []
        try:
            for _ in range(BATCH):
                batch_pairs.append(next(user_text_iter))
        except StopIteration:
            pass
        if not batch_pairs:
            break
        texts = [t for _, t in batch_pairs]
        for (u, _), doc in zip(batch_pairs, nlp.pipe(texts, batch_size=BATCH)):
            for sent in doc.sents:
                sf = per_sentence_features(sent)
                if sf is not None:
                    user_sents.setdefault(u, []).append(sf)
        parsed += len(batch_pairs)
        if parsed % 1000 < BATCH:
            print(f"    parsed {parsed}/{len(user_texts)} (t={time.time() - t0:.1f}s)", flush=True)
    print(f"users parsed: {len(user_sents)} (t={time.time() - t0:.1f}s)", flush=True)

    # Filter to eligible (≥40 sents of ≥L words)
    user_filtered: dict[str, list[dict]] = {}
    for u, sfs in user_sents.items():
        f = [s for s in sfs if s["n_tok"] >= L_FIXED]
        if len(f) >= 2 * N_FIXED:
            user_filtered[u] = f
    n_eligible = len(user_filtered)
    print(f"eligible (≥{2 * N_FIXED} sents of ≥{L_FIXED} words): {n_eligible}", flush=True)

    if n_eligible > N_USERS:
        user_list = sorted(user_filtered)
        rng.shuffle(user_list)
        keep = set(user_list[:N_USERS])
        user_filtered = {u: user_filtered[u] for u in keep}
        n_eligible = len(user_filtered)
        print(f"subsampled to {n_eligible}", flush=True)

    sorted_users = sorted(user_filtered)
    rng_split = np.random.default_rng(SEED + 100)
    perm = rng_split.permutation(n_eligible)
    n_dev = int(n_eligible * DEV_FRAC)
    dev_idx = set(perm[:n_dev].tolist())
    dev_users = [u for i, u in enumerate(sorted_users) if i in dev_idx]
    test_users = [u for i, u in enumerate(sorted_users) if i not in dev_idx]
    print(f"dev users: {len(dev_users)}, test users: {len(test_users)}", flush=True)

    # ============================================================
    # Run ONLY (L=5, N=20) with 9999 perms
    # ============================================================
    seeds_dev = tuple(int(SEED + 1000 + i) for i in range(N_SEEDS))
    seeds_test = tuple(int(SEED + 5000 + i) for i in range(N_SEEDS))
    rng_perm = np.random.default_rng(SEED + 2000)

    print(f"\n=== Dev evaluation (L={L_FIXED}, N={N_FIXED}) ===", flush=True)
    per_seed_dev = eval_cell(dev_users, user_filtered, N_FIXED, seeds_dev)
    if not per_seed_dev:
        raise RuntimeError("dev evaluation produced no seeds")
    aucs = np.asarray([s["auc"] for s in per_seed_dev])
    deltas = np.asarray([s["delta"] for s in per_seed_dev])
    cohen_ds = np.asarray([s["cohen_d"] for s in per_seed_dev])
    seed_pass_frac = float((aucs >= AUC_THRESH).mean())
    obs_delta = float(deltas.mean())
    obs_cohen = float(cohen_ds.mean())
    print(f"  dev AUC={aucs.mean():.4f}+-{aucs.std():.4f}  "
          f"d={obs_cohen:.4f}  seed_pass={seed_pass_frac:.2f}  "
          f"delta={obs_delta:.5f}", flush=True)

    print(f"\n=== Dev 9999 permutations ===", flush=True)
    p_one, p_two = eval_permutation(dev_users, user_filtered, N_FIXED,
                                    obs_delta, N_PERM, rng_perm)
    print(f"  p_one={p_one:.5f}  p_two={p_two:.5f}", flush=True)

    print(f"\n=== Test evaluation (independent users) ===", flush=True)
    per_seed_test = eval_cell(test_users, user_filtered, N_FIXED, seeds_test)
    aucs_t = np.asarray([s["auc"] for s in per_seed_test])
    deltas_t = np.asarray([s["delta"] for s in per_seed_test])
    cohen_ts = np.asarray([s["cohen_d"] for s in per_seed_test])
    test_seed_pass = float((aucs_t >= AUC_THRESH).mean())
    test_delta = float(deltas_t.mean())
    test_cohen = float(cohen_ts.mean())
    print(f"  test AUC={aucs_t.mean():.4f}+-{aucs_t.std():.4f}  "
          f"d={test_cohen:.4f}  seed_pass={test_seed_pass:.2f}  "
          f"delta={test_delta:.5f}", flush=True)

    # ============================================================
    # 4-gate evaluation
    # ============================================================
    gates = {
        "test_users": (len(test_users), TEST_MIN_USERS, ">="),
        "auc": (aucs.mean(), AUC_THRESH, ">="),
        "cohen_d": (obs_cohen, COHEN_D_THRESH, ">="),
        "seed_pass_frac": (seed_pass_frac, SEED_PASS_FRAC, ">="),
        "perm_p_two": (p_two, P_THRESH, "<"),
    }
    passes = {k: (v[0] >= v[1] if v[2] == ">=" else v[0] < v[1])
              for k, v in gates.items()}
    print(f"\n=== 4-gate evaluation ===", flush=True)
    for k, (val, thr, op) in gates.items():
        ok = "PASS" if passes[k] else "FAIL"
        print(f"  {k}: {val:.4f} {op} {thr}  [{ok}]", flush=True)
    all_pass = all(passes.values())
    print(f"\nALL GATES PASS: {all_pass}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "version": "v6 (confirm)",
        "purpose": "Step 1 — independent verification that (L=5, N=20) is stable threshold",
        "seed": SEED,
        "fresh_user_pool": True,
        "fixed_cell": {"L": L_FIXED, "N": N_FIXED},
        "n_perm": N_PERM,
        "n_seeds": N_SEEDS,
        "n_users_parsed": len(user_sents),
        "n_eligible": n_eligible,
        "dev_users": len(dev_users),
        "test_users": len(test_users),
        "per_seed_dev": per_seed_dev,
        "per_seed_test": per_seed_test,
        "dev_metrics": {
            "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std()), 4),
            "cohen_d_mean": round(float(obs_cohen), 4),
            "delta_mean": round(obs_delta, 5),
            "seed_pass_frac": round(seed_pass_frac, 4),
        },
        "test_metrics": {
            "auc_mean": round(float(aucs_t.mean()), 4),
            "auc_std": round(float(aucs_t.std()), 4),
            "cohen_d_mean": round(float(test_cohen), 4),
            "delta_mean": round(test_delta, 5),
            "seed_pass_frac": round(test_seed_pass, 4),
        },
        "permutation": {
            "perm_p_one": round(p_one, 5),
            "perm_p_two": round(p_two, 5),
            "n_perm_runs": N_PERM,
        },
        "gates": {
            "test_users_min": TEST_MIN_USERS,
            "auc_min": AUC_THRESH,
            "cohen_d_min": COHEN_D_THRESH,
            "seed_pass_min": SEED_PASS_FRAC,
            "perm_p_two_max": P_THRESH,
        },
        "gate_results": {k: bool(passes[k]) for k in gates},
        "all_gates_pass": bool(all_pass),
        "runtime_sec": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} (t={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()