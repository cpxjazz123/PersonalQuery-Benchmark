#!/usr/bin/env python3
"""[Reviewer-pilot] Real-LLM-judge eval on iter #75 synthetic annotations.

Iter #75 produced synthetic annotations (50 queries × 3 humans + 1 noisy-LLM)
but the LLM labels were synthetic, not from a real model. This iter replaces
the noisy-LLM labels with REAL Qwen2.5-7B-Instruct calls (already running on
vLLM at http://localhost:8000/v1 from iter #54) so we can measure:

  - Real-LLM agreement vs (a) ground-truth label (from the synthetic data),
    (b) each of the 3 synthetic humans, (c) the majority human label.

Inputs:
  result/personal_query/02_writing_analysis/llm_human_eval/llm_human_annotations.json
  (output of iter #75)

Outputs:
  result/personal_query/02_writing_analysis/llm_human_eval/llm_human_annotations_real.json
  result/personal_query/02_writing_analysis/llm_human_eval/real_llm_judge_summary.json

Run: python3 02_writing_analysis/llm_human_eval_real_llm_judge.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agreement_metrics import cohens_kappa, fleiss_kappa  # noqa: E402

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
EVAL_DIR = REPO_ROOT / "result" / "personal_query" / "02_writing_analysis" / "llm_human_eval"

VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "Qwen2.5-7B-Instruct"

# Synthetic query+product pairs (50). Pre-existing iter #75 rows are referenced
# only via query_id; here we attach text for the LLM to actually rate.
# Domain: Baby_Products. Each pair generated to be IRREL/PARTIAL/REL by design.
SYNTHETIC_QUERY_PRODUCTS = [
    ("q_000", "best baby bottle for newborn",  "Philips Avent Natural Baby Bottle, BPA-free, 9oz, slow flow nipple", "REL"),
    ("q_001", "baby formula for sensitive stomach", "Enfamil NeuroPro Gentlease Baby Formula, easy-to-digest, 29.4 oz", "REL"),
    ("q_002", "stroller for jogging",        "BOB Revolution Flex 3.0 Jogging Stroller, suspension system, 75 lb capacity", "REL"),
    ("q_003", "high chair easy to clean",    "IKEA Antilop High Chair with wipeable surface", "REL"),
    ("q_004", "baby monitor with camera",    "Infant Optics DXR-8 Video Baby Monitor with 3 cameras", "REL"),
    ("q_005", "diaper cream for rash",       "Boudreaux's Butt Paste Diaper Rash Ointment, 4 oz", "REL"),
    ("q_006", "teething toy for infant",     "Sophie la Girafe Baby Teether, 100% natural rubber", "REL"),
    ("q_007", "baby carrier wrap",           "Ergobaby Omni 360 Baby Carrier, ergonomic, all positions", "REL"),
    ("q_008", "swaddle blanket newborn",     "Aden + Anais Muslin Swaddle Blanket, 4-pack 47x47", "REL"),
    ("q_009", "car seat rear-facing infant", "Graco SnugRide SnugLock 35 Infant Car Seat, rear-facing", "REL"),
    ("q_010", "baby bathtub",                "The First Years Sure Comfort Newborn to Toddler Tub", "REL"),
    ("q_011", "best baby bottle for newborn", "Coleman 4-Person Tent, weatherproof, camping outdoor", "IRREL"),
    ("q_012", "baby formula for sensitive stomach", "Bose QuietComfort 45 Bluetooth Headphones", "IRREL"),
    ("q_013", "stroller for jogging",        "Munchkin Snack Catcher, green, BPA-free", "PARTIAL"),
    ("q_014", "high chair easy to clean",    "Delta Children Folding Baby Crib, espresso", "PARTIAL"),
    ("q_015", "baby monitor with camera",    "VTech VM819 Wi-Fi Baby Monitor, audio only", "PARTIAL"),
    ("q_016", "diaper cream for rash",       "Huggies Natural Care Wipes, sensitive, 56-count", "PARTIAL"),
    ("q_017", "teething toy for infant",     "Gerber 1st Foods Baby Food Pack, 2 oz", "PARTIAL"),
    ("q_018", "baby carrier wrap",           "Halo SleepSack Wearable Blanket, swaddle transition", "PARTIAL"),
    ("q_019", "swaddle blanket newborn",     "Honest Company Diaper Subscription Box", "PARTIAL"),
    ("q_020", "car seat rear-facing infant", "Evenflo Chase SelectFit Booster Seat, forward-facing", "PARTIAL"),
    ("q_021", "baby bathtub",                "Summer Infant Contoured Bath Sponge, soft cushion", "PARTIAL"),
    ("q_022", "infant car seat travel system", "Britax Marathon ClickTight Convertible Car Seat", "REL"),
    ("q_023", "natural baby shampoo",        "California Baby Super Sensitive Shampoo & Body Wash", "REL"),
    ("q_024", "pacifier for newborn",        "MAM Newborn Pacifiers, 0-2 months, 2-pack", "REL"),
    ("q_025", "baby crib mattress organic",  "Naturepedic Organic Cotton Crib Mattress, lightweight", "REL"),
    ("q_026", "baby shoes 6 month",          "Stride Rite SM Baby Shoes, soft sole, 12-month size", "REL"),
    ("q_027", "baby sunscreen sensitive",    "Badger Baby Sunscreen Cream, SPF 30, mineral", "REL"),
    ("q_028", "infant sleep sack",           "Nested Bean Zen Sleep Sack, lightly weighted, 0-6m", "REL"),
    ("q_029", "baby nail clipper set",       "Piyo Piyo Baby Nail Clipper Set, electric", "REL"),
    ("q_030", "stroller cup holder",         "Universal Stroller Cup Holder, attaches to handlebar", "REL"),
    ("q_031", "best baby bottle for newborn", "Yeti Rambler 30 oz Tumbler, stainless steel", "IRREL"),
    ("q_032", "stroller for jogging",        "Black+Decker Cordless Drill 20V MAX, 1/2-inch chuck", "IRREL"),
    ("q_033", "high chair easy to clean",    "GoPro HERO11 Black Action Camera, waterproof", "IRREL"),
    ("q_034", "baby monitor with camera",    "Cuisinart 14-Cup Food Processor, stainless", "IRREL"),
    ("q_035", "diaper cream for rash",       "Logitech MX Master 3 Wireless Mouse, graphite", "IRREL"),
    ("q_036", "teething toy for infant",     "DeWalt 20V MAX XR Impact Driver Kit", "IRREL"),
    ("q_037", "baby carrier wrap",           "Samsung 65-inch QLED 4K Smart TV, Quantum HDR", "IRREL"),
    ("q_038", "swaddle blanket newborn",     "WagWell 30-inch Steel Pet Gate for dogs", "IRREL"),
    ("q_039", "car seat rear-facing infant", "Dyson V11 Torque Drive Cordless Vacuum", "IRREL"),
    ("q_040", "baby bathtub",                "Adidas Men's Ultraboost 22 Running Shoes", "IRREL"),
    ("q_041", "baby monitor with camera",    "Motorola MBP36XL Video Baby Monitor with parent unit", "PARTIAL"),
    ("q_042", "stroller for jogging",        "Evenflo Folio Travel System with Stroller + Car Seat", "PARTIAL"),
    ("q_043", "diaper cream for rash",       "A+D Original Ointment, skin protectant, 16 oz", "PARTIAL"),
    ("q_044", "high chair easy to clean",    "Chicco Polly Magic High Chair, foldable, dishwasher tray", "PARTIAL"),
    ("q_045", "swaddle blanket newborn",     "HALO Bassinest Swivel Sleeper, bedside sleeper", "PARTIAL"),
    ("q_046", "baby carrier wrap",           "Tula Explore Baby Carrier, ergonomic, multiple positions", "PARTIAL"),
    ("q_047", "teething toy for infant",     "Comotomo Baby Silicone Pacifier, 0-3 months, 2-pack", "PARTIAL"),
    ("q_048", "stroller for jogging",        "Baby Jogger City Mini GT Travel System with car seat", "PARTIAL"),
    ("q_049", "baby bathtub",                "Puj Tub, newborn bath tub, fits in sink", "PARTIAL"),
]


def _label_to_int(s: str) -> int:
    return ["IRREL", "PARTIAL", "REL"].index(s)


def _pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")

    def _rank(values):
        sorted_pairs = sorted(enumerate(values), key=lambda iv: iv[1])
        ranks = [0] * len(values)
        for r, (idx, _) in enumerate(sorted_pairs, start=1):
            ranks[idx] = r
        return ranks

    return _pearson(_rank(xs), _rank(ys))


def _call_vllm(prompt: str, model: str = VLLM_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a relevance judge. Rate the relevance of the retrieved product to the user query on a 3-class scale: "
                    "2 = REL (product directly satisfies the query); "
                    "1 = PARTIAL (product partially satisfies / same category but different specifics); "
                    "0 = IRREL (product is unrelated). "
                    "Reply with ONLY the integer 0, 1, or 2."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 4,
    }
    req = urllib.request.Request(
        VLLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def _parse_label(reply: str) -> int:
    m = re.search(r"[012]", reply)
    if m is None:
        raise ValueError(f"could not parse int from LLM reply: {reply!r}")
    return int(m.group(0))


def main():
    annotations_path = EVAL_DIR / "llm_human_annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"missing: {annotations_path} (run iter #75 first)")
    with open(annotations_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != len(SYNTHETIC_QUERY_PRODUCTS):
        raise ValueError(
            f"annotation rows {len(rows)} vs synthetic_query_products {len(SYNTHETIC_QUERY_PRODUCTS)} mismatch"
        )

    text_by_qid = {qid: (q, p, gt_label) for qid, q, p, gt_label in SYNTHETIC_QUERY_PRODUCTS}

    real_llm_labels: list[int] = []
    failures: list[tuple[str, str]] = []
    for r in rows:
        qid = r["query_id"]
        query_text, product_text, _gt = text_by_qid[qid]
        prompt = f"Query: {query_text}\n\nRetrieved product: {product_text}\n\nRelevance (0=IRREL, 1=PARTIAL, 2=REL)?"
        reply = _call_vllm(prompt)
        try:
            lbl = _parse_label(reply)
        except ValueError as e:
            failures.append((qid, str(e)))
            lbl = -1
        real_llm_labels.append(lbl)

    print(f"Real-LLM calls completed: {len(real_llm_labels)} (failures: {len(failures)})")
    if failures:
        print("  (these rows will be excluded from agreement metrics)")

    # Augment rows in-place with real_llm_label.
    for r, lbl in zip(rows, real_llm_labels):
        r["real_llm_label"] = ["IRREL", "PARTIAL", "REL"][lbl] if lbl >= 0 else "ERROR"
        r["real_llm_label_int"] = lbl

    out_annotations = EVAL_DIR / "llm_human_annotations_real.json"
    with open(out_annotations, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {out_annotations}")

    # Compute metrics on the success subset.
    valid = [(r, lbl) for r, lbl in zip(rows, real_llm_labels) if lbl >= 0]
    valid_rows, valid_llm = zip(*valid)
    n = len(valid)

    real_llm_int = list(valid_llm)
    gt_int = [_label_to_int(text_by_qid[r["query_id"]][2]) for r in valid_rows]
    fleiss_matrix = [
        [_label_to_int(r["human_labels"][h]) for h in range(3)] for r in valid_rows
    ]
    fleiss_kappa_value = fleiss_kappa(fleiss_matrix, n_categories=3)
    cohens_gt = cohens_kappa(real_llm_int, gt_int)
    per_human_cohens = []
    for h in range(3):
        h_int = [_label_to_int(r["human_labels"][h]) for r in valid_rows]
        k = cohens_kappa(real_llm_int, h_int)
        per_human_cohens.append({"human_id": f"H{h}", "cohens_kappa_vs_real_llm": k})
    majority_human = []
    for r in valid_rows:
        maj = Counter(r["human_labels"]).most_common(1)[0][0]
        majority_human.append(_label_to_int(maj))
    cohens_maj = cohens_kappa(real_llm_int, majority_human)
    # Spearman on the continuous scores (real LLM does not produce scores here; use
    # category-int as ordinal proxy).
    pearson_real = _pearson(real_llm_int, gt_int)
    spearman_real = _spearman(real_llm_int, gt_int)

    summary = {
        "n_queries_total": len(rows),
        "n_valid_real_llm_calls": n,
        "failures": failures,
        "fleiss_kappa_among_3_humans": fleiss_kappa_value,
        "cohens_kappa_real_llm_vs_synthetic_gt": cohens_gt,
        "cohens_kappa_real_llm_vs_each_human": per_human_cohens,
        "cohens_kappa_real_llm_vs_majority_human": cohens_maj,
        "pearson_real_llm_label_vs_gt_label": pearson_real,
        "spearman_real_llm_label_vs_gt_label": spearman_real,
        "label_distribution_real_llm": dict(Counter(["IRREL", "PARTIAL", "REL"][i] for i in real_llm_int)),
        "label_distribution_synthetic_gt": dict(Counter(text_by_qid[r["query_id"]][2] for r in valid_rows)),
    }

    summary_path = EVAL_DIR / "real_llm_judge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {summary_path}")

    print("\n=== Real LLM judge vs synthetic annotations ===")
    print(f"  Valid calls: {n}/{len(rows)}")
    print(f"  Real-LLM label distribution: {summary['label_distribution_real_llm']}")
    print(f"  Synthetic GT label distribution: {summary['label_distribution_synthetic_gt']}")
    print(f"  Fleiss κ (3 humans):           {fleiss_kappa_value:.4f}")
    print(f"  Cohen  κ (real-LLM vs synth-GT): {cohens_gt:.4f}")
    print(f"  Cohen  κ (real-LLM vs maj-H):  {cohens_maj:.4f}")
    for entry in per_human_cohens:
        print(f"  Cohen  κ (real-LLM vs {entry['human_id']}):        {entry['cohens_kappa_vs_real_llm']:.4f}")
    print(f"  Pearson  (real-LLM vs GT):     {pearson_real:.4f}")
    print(f"  Spearman (real-LLM vs GT):     {spearman_real:.4f}")


if __name__ == "__main__":
    main()
