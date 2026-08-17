"""E30 — build n-attribute product specs for n ∈ {4,5,6,7,8,9,10}.

For each picked product, accumulate attributes by:
  - A1 = brand (always)
  - A2 = title (always)
  - A3 = first award-style feature (always)
  - A4 = first interchangeable-lens-style feature (always)
  - A5..A10 = subsequent distinct features from features / description / title

Returns a JSON dict: {products: [{asin, attrs_n: {n: 4, ...}}, ...]}
"""
import gzip
import json
import re
import sys
import time
from pathlib import Path

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
PICKED_4ATTR = Path(f'{PAPER}/e30_picked_products.json')
META_GZ = '/fs04/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl.gz'
OUT_NATTR = Path(f'{PAPER}/e30_picked_products_nattrs.json')

STOP = {'a', 'the', 'of', 'in', 'and', 'or', 'to', 'is', 'for', 'with', 'on',
        'at', 'by', 'an', 'as', 'be', 'this', 'that', 'it', 'its'}


def shared_words(a, b):
    wa = {w.lower() for w in re.findall(r'[A-Za-z]+', a) if w.lower() not in STOP}
    wb = {w.lower() for w in re.findall(r'[A-Za-z]+', b) if w.lower() not in STOP}
    return len(wa & wb)


def extract_title_features(title):
    """Pull short feature phrases from a product title."""
    if not title:
        return []
    parts = re.split(r',\s*|\s+and\s+', title)
    phrases = []
    for p in parts:
        p = p.strip()
        if 2 <= len(p.split()) <= 5 and 4 <= len(p) <= 40:
            phrases.append(p)
    return phrases


def candidate_feature_sentences(meta_record, max_chars=80):
    """Yield short sentences from features + description."""
    seen = set()
    sources = []
    for f in meta_record.get('features', []) or []:
        for sent in re.split(r'[.!?]', f):
            sent = sent.strip()
            if 5 <= len(sent) <= max_chars:
                sources.append(sent)
    desc = meta_record.get('description', [])
    if isinstance(desc, list):
        for d in desc:
            for sent in re.split(r'[.!?]', d):
                sent = sent.strip()
                if 5 <= len(sent) <= max_chars:
                    sources.append(sent)
    return sources


def build_n_attrs(meta_record, current_4attrs, n_max=10):
    """Build A1..A_{n_max} for a product. Returns attrs dict (may have < n_max
    distinct values if meta is sparse)."""
    attrs = {}
    attrs['A1'] = current_4attrs.get('A1', '')  # brand
    attrs['A2'] = current_4attrs.get('A2', '')  # title
    attrs['A3'] = current_4attrs.get('A3', '')  # award
    attrs['A4'] = current_4attrs.get('A4', '')  # feature

    if n_max <= 4:
        return {k: attrs[k] for k in sorted(attrs.keys())[:n_max]}

    # already-used
    used = list(attrs.values())
    # title-derived phrases (avoid duplicates with A2)
    title_phrases = extract_title_features(meta_record.get('title', ''))
    # candidate sentences from features/description
    cand_sentences = candidate_feature_sentences(meta_record)

    # merge unique candidates with title phrases
    all_candidates = []
    for tp in title_phrases:
        all_candidates.append(tp)
    for sent in cand_sentences:
        all_candidates.append(sent)

    # pick candidates with low overlap to existing attrs
    next_idx = 5
    for cand in all_candidates:
        if next_idx > n_max:
            break
        if not cand or not cand.strip():
            continue
        # overlap check
        overlap = max(shared_words(cand, u) for u in used)
        if overlap < 2:
            attrs[f'A{next_idx}'] = cand
            used.append(cand)
            next_idx += 1

    # fill remaining slots with shorter variations
    while next_idx <= n_max:
        # use a title phrase not yet used
        for tp in title_phrases:
            if tp not in used and shared_words(tp, attrs.get('A2', '')) < 2:
                attrs[f'A{next_idx}'] = tp
                used.append(tp)
                break
        else:
            # fallback: synthetic placeholder
            attrs[f'A{next_idx}'] = f'(detail {next_idx})'
        next_idx += 1

    return attrs


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"loading {PICKED_4ATTR}...")
    with open(PICKED_4ATTR) as f:
        picked = json.load(f)
    products = picked['products']
    log(f"  {len(products)} products")

    target_asins = set(p['asin'] for p in products)
    log(f"loading meta for {len(target_asins)} products...")
    meta = {}
    t0 = time.time()
    with gzip.open(META_GZ, 'rt') as f:
        for line in f:
            r = json.loads(line)
            pa = r.get('parent_asin') or r.get('asin')
            if pa in target_asins:
                meta[pa] = r
    log(f"  loaded {len(meta)} meta records in {time.time()-t0:.0f}s")

    # build n-attrs
    new_products = []
    log("building 10 attrs per product...")
    for p in products:
        asin = p['asin']
        m = meta.get(asin)
        if m is None:
            continue
        a10 = build_n_attrs(m, p['attrs'], n_max=10)
        np = dict(p)
        np['attrs_n'] = a10
        new_products.append(np)

    # summarize for one example
    log("\n--- example B00ECHYTBI 10 attrs ---")
    for np in new_products[:1]:
        for k, v in sorted(np['attrs_n'].items()):
            log(f"  {k}: {v[:90]!r}")

    # count how many products have full 10 distinct attrs
    n_full = sum(1 for np in new_products
                 if len(set(v for v in np['attrs_n'].values() if v)) >= 10)
    n_at_least_8 = sum(1 for np in new_products
                       if len(set(v for v in np['attrs_n'].values() if v)) >= 8)
    log(f"\n{len(new_products)} products extended; "
        f"{n_full} have 10 distinct attrs, "
        f"{n_at_least_8} have ≥8 distinct attrs")

    with open(OUT_NATTR, 'w') as f:
        json.dump({'products': new_products,
                   'attr_field': 'attrs_n',
                   'note': 'A1=brand A2=title A3=award A4=feat A5..A10=additional distinct features'},
                  f, ensure_ascii=False, indent=2)
    log(f"wrote {OUT_NATTR}")


if __name__ == '__main__':
    main()