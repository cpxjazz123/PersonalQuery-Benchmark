"""E30 — extend source-query attributes (4→6 attrs) to grow query length,
then test if TinyStyler K=8/α=2 produces longer + more-style + more-attribute
queries.

Approach:
1. Build e30_picked_products_6attrs.json with 6 attrs per product (split title
   features + award + first description sentence).
2. Run copy-aware Qwen with 6 attrs (longer prompt) → longer source query.
3. Run TinyStyler K=8/α=2 with MAX_NEW_TOKENS=64.
4. Measure: query length, n_unique, n_attrs_in_styled, cos_user.

Hardcoded inputs:
- PICKED_4ATTR = e30_picked_products.json (current 4-attr)
- META_GZ = /fs04/.../meta_Baby_Products_2023.jsonl.gz
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
OUT_6ATTR = Path(f'{PAPER}/e30_picked_products_6attrs.json')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_title_features(title):
    """From a long product title (e.g. 'Infant Optics DXR-8 ... FHSS Connection,
    Interchangeable Lenses, Pan Tilt Zoom, LED Sound Bar, Night Vision,
    Two-way Talk'), pull out 4-5 short feature phrases.

    Strategy: split on commas + ' and '. Drop leading brand words. Keep
    2-5 word phrases, 4-40 chars.
    """
    if not title:
        return []
    # drop the very first segment (brand prefix like 'Infant Optics DXR-8 ...')
    parts = re.split(r',\s*|\s+and\s+', title)
    phrases = []
    for p in parts:
        p = p.strip()
        if 2 <= len(p.split()) <= 5 and 4 <= len(p) <= 40:
            phrases.append(p)
    return phrases


def build_attrs_for(meta_record, current_4attrs):
    """Given a meta record and the current 4 attrs (with A1=brand, A2=title,
    A3=award, A4=feature), build 6 attrs by adding 2 more features.

    We try to add:
      A5 = one feature from `features` (different from A3/A4 in substance)
      A6 = one title-derived feature phrase (different from A2)

    Returns attrs dict with 6 entries (some may fallback to last option if
    no distinct attribute is found).
    """
    attrs = {}
    attrs['A1'] = current_4attrs.get('A1', '')
    attrs['A2'] = current_4attrs.get('A2', '')
    attrs['A3'] = current_4attrs.get('A3', '')
    attrs['A4'] = current_4attrs.get('A4', '')

    # helper: check overlap by counting shared words (excluding stop words)
    STOP = {'a', 'the', 'of', 'in', 'and', 'or', 'to', 'is', 'for', 'with', 'on', 'at', 'by', 'an'}
    def shared_words(a, b):
        wa = {w.lower() for w in re.findall(r'[A-Za-z]+', a) if w.lower() not in STOP}
        wb = {w.lower() for w in re.findall(r'[A-Za-z]+', b) if w.lower() not in STOP}
        return len(wa & wb)

    existing = [attrs['A1'], attrs['A2'], attrs['A3'], attrs['A4']]

    # A5: feature phrase from meta_record['features']
    features = meta_record.get('features', []) or []
    a5 = None
    for f in features:
        # take first short sentence (split on '.' '!' '?')
        sent = re.split(r'[.!?]', f)[0].strip()
        # truncate to first 60 chars
        if len(sent) > 60:
            sent = sent[:60].rsplit(' ', 1)[0]
        if 5 <= len(sent):
            # check overlap with all existing attrs
            overlap = max(shared_words(sent, e) for e in existing) if existing else 0
            if overlap < 2:  # less than 2 shared words = distinct
                a5 = sent
                break
    if a5 is None:
        # fallback: try description first sentence
        desc = meta_record.get('description', [])
        if isinstance(desc, list) and desc:
            for d in desc:
                sent = re.split(r'[.!?]', d)[0].strip()
                if len(sent) > 60:
                    sent = sent[:60].rsplit(' ', 1)[0]
                if 5 <= len(sent):
                    overlap = max(shared_words(sent, e) for e in existing)
                    if overlap < 2:
                        a5 = sent
                        break
    if a5 is None:
        a5 = attrs['A4']  # last resort
    attrs['A5'] = a5

    # A6: title-derived feature phrase
    title = meta_record.get('title', '')
    title_phrases = extract_title_features(title)
    a6 = None
    for tp in title_phrases:
        overlap = max(shared_words(tp, e) for e in existing + [a5])
        if overlap < 2 and len(tp.split()) >= 2:
            a6 = tp
            break
    if a6 is None:
        # pick last title phrase as fallback
        a6 = title_phrases[-1] if title_phrases else ''
    attrs['A6'] = a6

    return attrs


def main():
    log(f"loading {PICKED_4ATTR}...")
    with open(PICKED_4ATTR) as f:
        picked = json.load(f)
    products = picked['products']
    log(f"  {len(products)} products")

    # build meta lookup
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

    # extend attrs
    new_products = []
    n_extended = 0
    n_distinct = 0
    for p in products:
        asin = p['asin']
        m = meta.get(asin)
        if m is None:
            log(f"  WARN no meta for {asin}, skipping")
            continue
        old_attrs = p['attrs']
        new_attrs = build_attrs_for(m, old_attrs)
        unique_vals = set(v for v in new_attrs.values() if v)
        if len(unique_vals) >= 6:
            n_extended += 1
        else:
            log(f"  WARN {asin}: only {len(unique_vals)} distinct attrs: "
                f"{new_attrs}")
        np = dict(p)
        np['attrs_6'] = new_attrs
        new_products.append(np)

    log(f"\n{len(new_products)} products extended to 6 attrs")
    log(f"  {n_extended} have 6 fully-distinct values")

    # show 3 examples
    log("\n--- example 6-attr products ---")
    for np in new_products[:3]:
        log(f"  {np['asin']}:")
        for k, v in np['attrs_6'].items():
            log(f"    {k}: {v[:80]!r}")

    # write
    out = {'products': new_products,
           'attr_count': 6,
           'note': 'A1=brand, A2=title, A3=award, A4=feature, A5=feature2, A6=title-feature'}
    with open(OUT_6ATTR, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\nwrote {OUT_6ATTR}")


if __name__ == '__main__':
    main()