"""Phase 14.G prep: build per-asin × N_attrs attr pools from raw metadata.

For each (asin, N_attrs), precompute a dict of N attributes:
- 3-5 attrs: random.sample(real_5_keys, N), seed = N
- 6-10 attrs: real_5_keys + (N-5) random.sample(metadata keys not in real_5), seed = N

Output: phase14_g_attrs_pool.json
"""
import json
import gzip
import random
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
PAIRS_FILE = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase14_b_pairs.jsonl")
META_PATH = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
OUT_POOL = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase14_g_attrs_pool.json")

REAL_5_KEYS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
N_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
N_PAIRS = 30
SEED = 42


def load_pairs():
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def load_meta_by_asin(target_asins):
    """Return {asin: {key: value}} from raw metadata for the target asins."""
    target_set = set(target_asins)
    out = {}
    with gzip.open(META_PATH, "rt") as f:
        for line in f:
            r = json.loads(line)
            pa = r.get("parent_asin", "")
            if pa in target_set:
                details = r.get("details", {})
                if isinstance(details, dict):
                    clean = {k: str(v).strip()[:64] for k, v in details.items() if v}
                    out[pa] = clean
                    if len(out) >= len(target_set):
                        break
    return out


def attrs_for_asin_n(real_5: dict, meta_full: dict, n: int) -> dict:
    """Build attrs dict for given N from real 5 + raw metadata.

    Args:
        real_5: dict of 5 fixed keys (Brand, Color, etc.)
        meta_full: dict of full raw metadata {key: value}
        n: target number of attrs (3..10)
    """
    rng = random.Random(SEED + n)
    base = list(real_5.keys())
    if n <= 5:
        selected = sorted(rng.sample(base, n))
        return {k: real_5[k] for k in selected}
    # 6-10: all 5 + (N-5) from metadata keys not in real_5
    extra_keys = [k for k in meta_full.keys() if k not in REAL_5_KEYS]
    if len(extra_keys) < (n - 5):
        # fallback: repeat if not enough
        extra_keys = extra_keys * 5
    extras = rng.sample(extra_keys, n - 5)
    out = dict(real_5)
    for k in extras:
        out[k] = meta_full[k]
    return out


def main():
    print("=== Phase 14.G prep: per-asin × N_attrs pool ===")
    pairs = load_pairs()
    print(f"  pairs: {len(pairs)}")
    target_asins = [p["asin"] for p in pairs]
    print(f"  loading meta for {len(target_asins)} asin ...")
    meta_by_asin = load_meta_by_asin(target_asins)
    print(f"  meta hits: {len(meta_by_asin)}/{len(target_asins)}")

    pool = {}
    for pair in pairs:
        asin = pair["asin"]
        meta = meta_by_asin.get(asin, {})
        if not meta:
            print(f"  WARN: meta missing for {asin}")
            continue
        real_5 = {k: pair["attrs"][k] for k in REAL_5_KEYS}
        pool[asin] = {
            "real_5": real_5,
            "meta_keys_full": sorted(meta.keys()),
            "meta_keys_count": len(meta),
            "by_n": {},
        }
        for n in N_VALUES:
            attrs = attrs_for_asin_n(real_5, meta, n)
            pool[asin]["by_n"][str(n)] = attrs

    # Stats summary
    print(f"\n  pool asin: {len(pool)}")
    for n in N_VALUES:
        avg_attrs = sum(len(pool[a]["by_n"][str(n)]) for a in pool) / len(pool)
        print(f"  N={n}: avg attrs per asin = {avg_attrs:.1f}")

    OUT_POOL.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"\n  saved → {OUT_POOL}")

    # Print one sample for sanity
    sample_asin = list(pool.keys())[0]
    print(f"\n  sample asin {sample_asin}:")
    for n in N_VALUES:
        print(f"    N={n}: {pool[sample_asin]['by_n'][str(n)]}")


if __name__ == "__main__":
    main()
