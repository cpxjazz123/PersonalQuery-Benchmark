#!/usr/bin/env python3
"""[Reviewer-pilot] iter #76 extension: cross-domain (3 × 50 queries) LLM-as-judge κ.

iter #76 ran only on Baby_Products 50 queries. This iter #80 extends the
real-LLM-as-judge (Qwen2.5-7B-Instruct via vLLM) to Grocery_and_Gourmet_Food
and Pet_Supplies — yielding a 3-domain table of Fleiss κ (inter-human),
Cohen κ (real-LLM vs synthetic GT / per-human / majority-human), Pearson
and Spearman (real-LLM vs GT ordinal).

Each domain has 50 (query, retrieved product) pairs with the same 3-class
{REL / PARTIAL / IRREL} structure and the same 20% noise on the synthetic
human + LLM layers from iter #75 / iter #76.

Inputs (auto-generated if missing by calling `pilot_review_vs_query_correlation.py`
-style helpers):
  - iter #75: 50 synthetic (query, product, GT, synthetic humans) per domain
  - iter #76: real-LLM label per query

Outputs:
  result/personal_query/02_writing_analysis/llm_human_eval/llm_human_eval_cross_domain.json
  stdout cross-domain table

Run: python3 02_writing_analysis/llm_human_eval_cross_domain.py
"""

from __future__ import annotations

import json
import random
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

RELEVANCE_LABELS = ["IRREL", "PARTIAL", "REL"]
N_QUERIES_PER_DOMAIN = 50
SEED = 42
LLM_NOISE = 0.20
HUMAN_NOISE = 0.20

# Per-domain 50 (query, product, GT) tuples. Different domains use different
# query subjects and product categories so the LLM judge sees distinct content.
DOMAINS = {
    "Baby_Products": [
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
    ],
    "Grocery_and_Gourmet_Food": [
        ("q_000", "best olive oil for cooking", "California Olive Ranch Extra Virgin Olive Oil, 750ml", "REL"),
        ("q_001", "organic coffee beans", "Kicking Horse Coffee Whole Bean Organic, 1lb", "REL"),
        ("q_002", "gluten-free pasta", "Barilla Gluten-Free Penne, 12oz", "REL"),
        ("q_003", "ketchup no high fructose", "Heinz Simply Tomato Ketchup, 31oz, no HFCS", "REL"),
        ("q_004", "cereal high fiber low sugar", "Fiber One Original Cereal, 18oz, 14g fiber", "REL"),
        ("q_005", "chocolate bar dark", "Lindt Excellence 85% Cocoa Dark Chocolate Bar, 3.5oz", "REL"),
        ("q_006", "almond milk unsweetened", "Silk Almond Milk Unsweetened Vanilla, half gallon", "REL"),
        ("q_007", "snack bars protein", "RXBAR Chocolate Sea Salt Protein Bar, 1.83oz × 12", "REL"),
        ("q_008", "tea green loose leaf", "Harney & Sons Japanese Sencha Loose Green Tea, 4oz", "REL"),
        ("q_009", "honey raw", "Nature Nate's 100% Pure Raw & Unfiltered Honey, 32oz", "REL"),
        ("q_010", "mustard dijon", "Maille Classic Dijon Mustard, 8.5oz jar", "REL"),
        ("q_011", "best olive oil for cooking", "DeWalt 20V MAX Lithium Battery Charger", "IRREL"),
        ("q_012", "organic coffee beans", "Gibson Les Paul Standard Electric Guitar", "IRREL"),
        ("q_013", "gluten-free pasta", "Barilla Classic Spaghetti, 16oz (contains wheat)", "PARTIAL"),
        ("q_014", "ketchup no high fructose", "Heinz Tomato Ketchup, 32oz with HFCS", "PARTIAL"),
        ("q_015", "cereal high fiber low sugar", "Kellogg's Froot Loops Cereal, 12oz, 9g sugar", "PARTIAL"),
        ("q_016", "chocolate bar dark", "Nestle Crunch Milk Chocolate Bar, 1.55oz", "PARTIAL"),
        ("q_017", "almond milk unsweetened", "Silk Vanilla Almond Milk Creamer, 32oz, sweetened", "PARTIAL"),
        ("q_018", "snack bars protein", "Clif Bar Original Trail Mix Bar, 2.4oz", "PARTIAL"),
        ("q_019", "tea green loose leaf", "Lipton Yellow Label Black Tea Bags, 100ct", "PARTIAL"),
        ("q_020", "honey raw", "SueBee Clover Honey, 12oz squeeze bottle", "PARTIAL"),
        ("q_021", "mustard dijon", "Grey Poupon Dijon Mustard, 8oz", "PARTIAL"),
        ("q_022", "peanut butter natural", "Smucker’s Natural Creamy Peanut Butter, 16oz", "REL"),
        ("q_023", "salsa medium", "Tostitos Medium Salsa, 16oz jar", "REL"),
        ("q_024", "salmon canned wild", "Wild Planet Wild Sockeye Salmon, 5oz", "REL"),
        ("q_025", "rice jasmine", "Royal Basmati Rice, 10lb", "REL"),
        ("q_026", "tortillas flour", "Mission Super Size Flour Tortillas, 18ct", "REL"),
        ("q_027", "cheese cheddar sharp", "Tillamook Sharp Cheddar Cheese Block, 2lb", "REL"),
        ("q_028", "yogurt greek plain", "Fage Total 0% Greek Yogurt, 17.6oz", "REL"),
        ("q_029", "olive tapenade", "Reese Tapenade Black Olive Spread, 7.5oz", "REL"),
        ("q_030", "pesto basil", "Barilla Basilico Pesto Sauce, 6.5oz", "REL"),
        ("q_031", "best olive oil for cooking", "Bose Noise Cancelling Headphones 700", "IRREL"),
        ("q_032", "organic coffee beans", "Apple iPhone 14 Pro, 256GB, deep purple", "IRREL"),
        ("q_033", "gluten-free pasta", "Samsung 32-inch Curved Monitor, 1080p", "IRREL"),
        ("q_034", "ketchup no high fructose", "Yeti Roadie 24 Cooler, charcoal", "IRREL"),
        ("q_035", "cereal high fiber low sugar", "Logitech G502 Wireless Gaming Mouse", "IRREL"),
        ("q_036", "chocolate bar dark", "Dyson V15 Detect Cordless Vacuum", "IRREL"),
        ("q_037", "almond milk unsweetened", "Adobe Photoshop Elements 2024 Software", "IRREL"),
        ("q_038", "snack bars protein", "Bosch 18-inch Dishwasher 300 Series", "IRREL"),
        ("q_039", "tea green loose leaf", "GoPro HERO12 Black Action Camera", "IRREL"),
        ("q_040", "honey raw", "Trek Roscoe 7 Mountain Bike, 29-inch wheels", "IRREL"),
        ("q_041", "mustard dijon", "French's Classic Yellow Mustard, 12oz", "PARTIAL"),
        ("q_042", "peanut butter natural", "Jif Creamy Peanut Butter, 18oz with added sugar", "PARTIAL"),
        ("q_043", "salsa medium", "Pace Medium Picante Sauce, 16oz", "PARTIAL"),
        ("q_044", "salmon canned wild", "Bumble Bee Chunk Light Tuna, 5oz (not salmon)", "PARTIAL"),
        ("q_045", "rice jasmine", "Mahatma Extra Long Grain Rice, 5lb (jasmine sub)", "PARTIAL"),
        ("q_046", "tortillas flour", "Mission Corn Tortillas Soft Taco, 30ct", "PARTIAL"),
        ("q_047", "cheese cheddar sharp", "Sargento Sharp Cheddar Slices, 12ct", "PARTIAL"),
        ("q_048", "yogurt greek plain", "Chobani Plain Greek Yogurt, 32oz", "PARTIAL"),
        ("q_049", "olive tapenade", "Mezzetta Italian Castelvetrano Olives, 9oz", "PARTIAL"),
    ],
    "Pet_Supplies": [
        ("q_000", "dog food grain-free", "Blue Buffalo Wilderness Grain-Free Dry Dog Food, 24lb", "REL"),
        ("q_001", "cat litter clumping", "Arm & Hammer Clump & Seal Litter, 40lb", "REL"),
        ("q_002", "dog collar large", "PetSafe Stay & Play Wireless Fence, large collar", "REL"),
        ("q_003", "cat toys interactive", "Petstages Tower of Tracks Cat Toy, 21 levels", "REL"),
        ("q_004", "dog bed orthopedic large", "Big Barker 7-inch Orthopedic Dog Bed, large", "REL"),
        ("q_005", "flea treatment for cats", "Frontline Plus Flea Treatment for Cats, 6 doses", "REL"),
        ("q_006", "dog leash retractable", "Flexi New Classic Retractable Dog Leash, 26ft", "REL"),
        ("q_007", "automatic cat feeder", "PetSafe Automatic Cat Feeder, 5-meal capacity", "REL"),
        ("q_008", "dog shampoo sensitive skin", "Veterinary Formula Clinical Care Sensitive Dog Shampoo, 16oz", "REL"),
        ("q_009", "cat scratching post tall", "SmartCat Pioneer Pet Lounger Scratching Post, 32-inch", "REL"),
        ("q_010", "dog crate 36 inch", "MidWest iCrate Dog Crate, 36-inch, double door", "REL"),
        ("q_011", "dog food grain-free", "Apple iPad Pro 12.9-inch, 2TB Wi-Fi", "IRREL"),
        ("q_012", "cat litter clumping", "Nike Air Max 270 Running Shoes", "IRREL"),
        ("q_013", "dog collar large", "KONG Classic Dog Toy, durable rubber (toy not collar)", "PARTIAL"),
        ("q_014", "cat toys interactive", "Friskies Cat Treats, Party Mix, 2.1oz (treat not toy)", "PARTIAL"),
        ("q_015", "dog bed orthopedic large", "PetFusion Ultimate Dog Bed, large, 36x27", "PARTIAL"),
        ("q_016", "flea treatment for cats", "Advantage II Flea Treatment for Dogs, 6 doses", "PARTIAL"),
        ("q_017", "dog leash retractable", "Mighty Paw Tactical Dog Leash, 6ft (not retractable)", "PARTIAL"),
        ("q_018", "automatic cat feeder", "PetSafe Drinkwell Pet Fountain, 100oz (waterer not feeder)", "PARTIAL"),
        ("q_019", "dog shampoo sensitive skin", "Earthbath Oatmeal Dog Shampoo, 16oz", "PARTIAL"),
        ("q_020", "cat scratching post tall", "Cat Tree Condo with Hammock, 47-inch", "PARTIAL"),
        ("q_021", "dog crate 36 inch", "Petmate Ultra Vari Kennel, 36-inch heavy-duty crate", "PARTIAL"),
        ("q_022", "dog dental chews", "Greenies Original Natural Dental Dog Treats, 36oz", "REL"),
        ("q_023", "catnip toys", "Yeowww Organic Catnip Toy, 4-inch rainbow", "REL"),
        ("q_024", "dog raincoat", "Ruffwear Sun Shower Dog Rain Jacket", "REL"),
        ("q_025", "aquarium fish food", "TetraMin Tropical Flake Fish Food, 7.06oz", "REL"),
        ("q_026", "small pet hamster cage", "Prevue Pet Small Animal Cage, multi-level", "REL"),
        ("q_027", "dog paw balm", "Musher's Secret Dog Paw Wax, 200g", "REL"),
        ("q_028", "cat calming spray", "Feliway Classic Calming Spray for Cats, 60ml", "REL"),
        ("q_029", "dog waste bags", "Earth Rated Dog Waste Bags, 600ct", "REL"),
        ("q_030", "pet first aid kit", "Kurgo First Aid Kit for Dogs", "REL"),
        ("q_031", "dog food grain-free", "Microsoft Surface Pro 9, i7, 16GB", "IRREL"),
        ("q_032", "cat litter clumping", "Levi's 501 Original Fit Jeans, 32W", "IRREL"),
        ("q_033", "dog collar large", "Yeti Rambler 20oz Tumbler", "IRREL"),
        ("q_034", "cat toys interactive", "Logitech MX Keys Wireless Keyboard", "IRREL"),
        ("q_035", "dog bed orthopedic large", "Samsung Galaxy Watch 5, 40mm", "IRREL"),
        ("q_036", "flea treatment for cats", "Cuisinart 11-Cup Food Processor", "IRREL"),
        ("q_037", "dog leash retractable", "Bose SoundLink Flex Bluetooth Speaker", "IRREL"),
        ("q_038", "automatic cat feeder", "Garmin Forerunner 265 GPS Watch", "IRREL"),
        ("q_039", "dog shampoo sensitive skin", "Trek Marlin 5 Mountain Bike", "IRREL"),
        ("q_040", "cat scratching post tall", "Anker USB-C 100W Charger", "IRREL"),
        ("q_041", "dog crate 36 inch", "MidWest iCrate Dog Crate, 42-inch", "PARTIAL"),
        ("q_042", "dog dental chews", "Pedigree Dentastix, 28 treats", "PARTIAL"),
        ("q_043", "catnip toys", "Catnip Mad Mice Catnip Toy, 3-pack", "PARTIAL"),
        ("q_044", "dog raincoat", "Kurgo Step-N-Strobe Rain Jacket, large", "PARTIAL"),
        ("q_045", "aquarium fish food", "API Tropical Fish Flakes, 5.7oz", "PARTIAL"),
        ("q_046", "small pet hamster cage", "Kaytee CritterTrail Habitat, multi-level", "PARTIAL"),
        ("q_047", "dog paw balm", "Bag Balm Skin Ointment, 17oz (livestock)", "PARTIAL"),
        ("q_048", "cat calming spray", "Comfort Zone Calming Spray for Cats", "PARTIAL"),
        ("q_049", "dog waste bags", "Pogi Poop Bags, 300ct", "PARTIAL"),
    ],
}


def _synthesize_humans(rng: random.Random, gt: str) -> list[str]:
    gt_idx = RELEVANCE_LABELS.index(gt)
    out = []
    for _ in range(3):
        if rng.random() < HUMAN_NOISE:
            delta = rng.choice([-1, 1])
            lab = max(0, min(2, gt_idx + delta))
        else:
            lab = gt_idx
        out.append(RELEVANCE_LABELS[lab])
    return out


def _call_vllm(query_text: str, product_text: str) -> str:
    prompt = f"Query: {query_text}\n\nRetrieved product: {product_text}\n\nRelevance (0=IRREL, 1=PARTIAL, 2=REL)?"
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a relevance judge. Rate the relevance of the retrieved product to the user query on a 3-class scale: "
                "2 = REL (product directly satisfies the query); "
                "1 = PARTIAL (product partially satisfies / same category but different specifics); "
                "0 = IRREL (product is unrelated). "
                "Reply with ONLY the integer 0, 1, or 2."
            )},
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


def _evaluate_domain(domain: str, queries: list) -> dict:
    rng = random.Random(SEED)
    rows = []
    for qid, q_text, p_text, gt in queries:
        humans = _synthesize_humans(rng, gt)
        try:
            reply = _call_vllm(q_text, p_text)
            real_label_int = _parse_label(reply)
        except (ValueError, Exception) as e:
            real_label_int = -1
            print(f"  [{domain}/{qid}] LLM call failed: {e}")
        rows.append({
            "query_id": qid,
            "ground_truth_label": gt,
            "ground_truth_int": RELEVANCE_LABELS.index(gt),
            "human_labels": humans,
            "real_llm_label_int": real_label_int,
            "real_llm_label": RELEVANCE_LABELS[real_label_int] if real_label_int >= 0 else "ERROR",
        })

    valid_rows = [r for r in rows if r["real_llm_label_int"] >= 0]
    fleiss_matrix = [[RELEVANCE_LABELS.index(r["human_labels"][h]) for h in range(3)] for r in valid_rows]
    fleiss_kappa_value = fleiss_kappa(fleiss_matrix, n_categories=3)
    real_llm_int = [r["real_llm_label_int"] for r in valid_rows]
    gt_int = [r["ground_truth_int"] for r in valid_rows]
    cohens_gt = cohens_kappa(real_llm_int, gt_int)
    per_human_cohens = []
    for h in range(3):
        h_int = [RELEVANCE_LABELS.index(r["human_labels"][h]) for r in valid_rows]
        k = cohens_kappa(real_llm_int, h_int)
        per_human_cohens.append({"human_id": f"H{h}", "cohens_kappa_vs_real_llm": k})
    majority_human = []
    for r in valid_rows:
        maj = Counter(r["human_labels"]).most_common(1)[0][0]
        majority_human.append(RELEVANCE_LABELS.index(maj))
    cohens_maj = cohens_kappa(real_llm_int, majority_human)
    pearson_score = _pearson(real_llm_int, gt_int)
    spearman_score = _spearman(real_llm_int, gt_int)
    return {
        "domain": domain,
        "n_queries_total": len(rows),
        "n_valid": len(valid_rows),
        "label_distribution_real_llm": dict(Counter(r["real_llm_label"] for r in rows if r["real_llm_label"] != "ERROR")),
        "label_distribution_gt": dict(Counter(r["ground_truth_label"] for r in rows)),
        "fleiss_kappa_among_3_humans": fleiss_kappa_value,
        "cohens_kappa_real_llm_vs_synthetic_gt": cohens_gt,
        "cohens_kappa_real_llm_vs_each_human": per_human_cohens,
        "cohens_kappa_real_llm_vs_majority_human": cohens_maj,
        "pearson_real_llm_label_vs_gt_label": pearson_score,
        "spearman_real_llm_label_vs_gt_label": spearman_score,
        "rows": rows,
    }


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for domain, queries in DOMAINS.items():
        if len(queries) != N_QUERIES_PER_DOMAIN:
            raise ValueError(f"{domain} has {len(queries)} queries, expected {N_QUERIES_PER_DOMAIN}")
        print(f"\n=== {domain} ({len(queries)} queries) ===")
        result = _evaluate_domain(domain, queries)
        all_results.append(result)
        print(f"  Valid real-LLM calls: {result['n_valid']}/{result['n_queries_total']}")
        print(f"  Real-LLM vs synthetic humans: Fleiss κ={result['fleiss_kappa_among_3_humans']:.4f}, "
              f"Cohen κ vs maj-H={result['cohens_kappa_real_llm_vs_majority_human']:.4f}")
        print(f"  Real-LLM vs GT:               Cohen κ={result['cohens_kappa_real_llm_vs_synthetic_gt']:.4f}, "
              f"Pearson={result['pearson_real_llm_label_vs_gt_label']:.4f}, "
              f"Spearman={result['spearman_real_llm_label_vs_gt_label']:.4f}")
    out_path = EVAL_DIR / "llm_human_eval_cross_domain.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")
    print("\n=== Cross-domain summary (3 × 50 queries) ===")
    header = (f"{'Domain':<26} {'n':>4} {'Fleiss κ':>10} {'Cohen κ vs maj-H':>18} "
              f"{'Cohen κ vs GT':>14} {'Pearson':>10} {'Spearman':>10}")
    print(header)
    print("-" * 96)
    for r in all_results:
        print(f"{r['domain']:<26} {r['n_valid']:>4} "
              f"{r['fleiss_kappa_among_3_humans']:>10.4f} "
              f"{r['cohens_kappa_real_llm_vs_majority_human']:>18.4f} "
              f"{r['cohens_kappa_real_llm_vs_synthetic_gt']:>14.4f} "
              f"{r['pearson_real_llm_label_vs_gt_label']:>10.4f} "
              f"{r['spearman_real_llm_label_vs_gt_label']:>10.4f}")


if __name__ == "__main__":
    main()
