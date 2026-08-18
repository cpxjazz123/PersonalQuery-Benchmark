"""E30 Step 1 — pick K_PRODUCTS shared products in X=8 / Y=40 cell.

A product is "shared" in this cell if it has ≥5 distinct eligible users (each
with ≥Y=40 sents of ≥X=8 words). For each picked product we extract 4 STRUCTURED
attribute VALUES (no key prefixes) from `meta_Baby_Products_2023.jsonl.gz`:

  A1 = brand (store)
  A2 = title (truncated to 60 chars at word boundary)
  A3 = details[some_key] VALUE only (no "key:" prefix)
  A4 = details[other_key] VALUE only (no "key:" prefix)

"Structured" means: complete, well-defined values — NOT raw fragments with
unclosed quotes or incomplete sentences (which was the case when we sliced
features[0] to 50 chars and got "Awarded \"Best Baby Monitor Overall, "). We
use the `details` dict to get clean, complete attribute values like
"19.3 x 13.4 x 6.5 inches" or "Plastic". The "key:" prefix is metadata, not
part of the attribute value — we strip it so the model can naturally embed the
value into a shopping query.

Writes /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_picked_products.json
"""
import gzip
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

USER_SENTS_PKL = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache2.pkl')
META_GZ = '/fs04/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl.gz'
PRODUCT_ATTRS_CACHE = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_product_attrs.pkl')
OUT_JSON = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_picked_products.json')

X = 8           # min words per sentence
Y = 40          # min sents per user
MIN_PRODUCTS_PER_USER = 1  # user must have ≥1 product with ≥Y qualifying sents
MIN_USERS_PER_PRODUCT = 5  # product must have ≥5 such users (shared-product constraint)
K_PRODUCTS = 100
SEED = 42

WORD_RE = re.compile(r'\s+')
ATTR_BLACKLIST = re.compile(r'[\[\]\(\)\{\}]')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def word_count(s):
    return len(WORD_RE.split(s)) - 1


def main():
    log("loading user_sents_cache2.pkl...")
    t0 = time.time()
    with open(USER_SENTS_PKL, 'rb') as f:
        user_sents = pickle.load(f)
    log(f"loaded {len(user_sents):,} users in {time.time()-t0:.0f}s")

    # Step A: count per-user sX (X=8 only) once
    log("counting per-user sents with ≥8 words...")
    t0 = time.time()
    user_eligible_sents = {}
    for u, sents in user_sents.items():
        qual = [(s, wc, a) for s, wc, a in sents if wc >= X]
        if len(qual) >= Y:
            user_eligible_sents[u] = qual
    log(f"eligible users: {len(user_eligible_sents):,} in {time.time()-t0:.0f}s")

    # Step B: user → set of products they reviewed (any product they have ≥1
    # qualifying sent on). The X=8/Y=40 filter is at USER level; products
    # don't need Y sents per user (impossible with single-parent-asin dedup).
    log("mapping users → products...")
    t0 = time.time()
    user_to_products = {}
    for u, qual in user_eligible_sents.items():
        prods = {a for s, wc, a in qual if a}
        if len(prods) >= MIN_PRODUCTS_PER_USER:
            user_to_products[u] = prods
    log(f"users with ≥{MIN_PRODUCTS_PER_USER} qualifying products: {len(user_to_products):,} in {time.time()-t0:.0f}s")

    # Step C: product → set of users (inverse map)
    product_to_users = defaultdict(set)
    for u, prods in user_to_products.items():
        for p in prods:
            product_to_users[p].add(u)
    log(f"distinct products in shared pool: {len(product_to_users):,}")

    # Step D: filter products to those with ≥MIN_USERS_PER_PRODUCT users, sort by user count desc
    shared = [(p, users) for p, users in product_to_users.items()
              if len(users) >= MIN_USERS_PER_PRODUCT]
    shared.sort(key=lambda x: -len(x[1]))
    log(f"products with ≥{MIN_USERS_PER_PRODUCT} shared users: {len(shared):,}")

    # Take top K_PRODUCTS
    top = shared[:K_PRODUCTS]
    top_asins = {p for p, _ in top}
    log(f"selected top {len(top)} products")

    # Step E: load product metadata for the top asins (with cache)
    log("loading product metadata (with cache)...")
    t0 = time.time()
    if PRODUCT_ATTRS_CACHE.exists():
        with open(PRODUCT_ATTRS_CACHE, 'rb') as f:
            all_attrs = pickle.load(f)
        log(f"loaded cached attrs for {len(all_attrs):,} products")
    else:
        all_attrs = {}
    need = top_asins - set(all_attrs.keys())
    log(f"need to parse {len(need)} products from meta gz")
    if need:
        def short(s, maxlen=60):
            s = (s or '').strip()
            s = ' '.join(s.split())
            if len(s) > maxlen:
                s = s[:maxlen].rsplit(' ', 1)[0]
            return s
        def pick_details_values(details):
            """Pick up to 4 short, complete values from details dict (VALUE only, no key prefix).

            Prefers values WITHOUT digits (less likely to be paraphrased / dropped by
            generation models). Falls back to numeric values if needed.

            A "good" value:
              - is str (not dict/list/number), 3 ≤ len(value) ≤ 40 chars
              - does not end with a comma/colon/'See more'
              - key is meaningful (skip shipping / discontinued / rankings / model numbers)
            Returns list of value strings (NOT "key: value", just value), with
            no-number values first.
            """
            bad_keys = {
                'Domestic Shipping', 'International Shipping', 'Best Sellers Rank',
                'Is Discontinued By Manufacturer', 'Date First Available',
                'Package Dimensions', 'Item model number', 'Country/Region of origin',
                'ASIN', 'Item Weight', 'Brand', 'Model Name', 'Number Of Items',
                'Batteries required', 'Batteries', 'Number of channels',
                'Standing screen display size', 'Battery life', 'Item model number',
                'Best Sellers Rank',
            }
            no_num = []
            with_num = []
            for k, v in (details or {}).items():
                if k in bad_keys:
                    continue
                if not isinstance(v, str):
                    continue
                v = v.strip()
                if not (3 <= len(v) <= 40):
                    continue
                if v.endswith((',', ':', ';')) or 'See more' in v:
                    continue
                if v.lower().startswith('item can') or v.lower().startswith('this item'):
                    continue
                if any(c.isdigit() for c in v):
                    with_num.append(v)
                else:
                    no_num.append(v)
            return no_num + with_num
        with gzip.open(META_GZ, 'rt') as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                pa = o.get('parent_asin') or o.get('asin')
                if pa in need:
                    brand = short(o.get('store') or '', 30)
                    title = short(o.get('title') or '', 60)
                    vals = pick_details_values(o.get('details') or {})
                    all_attrs[pa] = {
                        'title': title, 'store': brand,
                        'kvs': vals,  # values only now
                    }
        with open(PRODUCT_ATTRS_CACHE, 'wb') as f:
            pickle.dump(all_attrs, f, protocol=4)
        log(f"saved cache for {len(all_attrs):,} products in {time.time()-t0:.0f}s")

    # Step F: build final picked products list with attrs (A1=brand, A2=title, A3/A4=details VALUE)
    log("building picked products list with attrs...")
    picked = []
    skipped = 0
    skipped_no_kv = 0
    for asin, users in top:
        meta = all_attrs.get(asin)
        if meta is None:
            skipped += 1
            continue
        attrs = {}
        if meta['store']:
            attrs['A1'] = meta['store']
        if meta['title']:
            attrs['A2'] = meta['title']
        vals = meta.get('kvs', [])
        short_vals = [v for v in vals if len(v) <= 50]
        if len(short_vals) < 2:
            skipped_no_kv += 1
            continue
        attrs['A3'] = short_vals[0]
        attrs['A4'] = short_vals[1]
        picked.append({
            'asin': asin,
            'attrs': attrs,
            'title': meta['title'],
            'store': meta['store'],
            'n_eligible_users': len(users),
            'candidate_users': sorted(users),
        })
    log(f"skipped {skipped_no_kv} products with <2 short KV values")

    log(f"final picked: {len(picked)} (skipped {skipped})")

    total_eligible_users = set()
    for p in picked:
        total_eligible_users.update(p['candidate_users'])
    total_eligible_pairs = sum(len(p['candidate_users']) for p in picked)
    out = {
        'X': X, 'Y': Y,
        'min_products_per_user': MIN_PRODUCTS_PER_USER,
        'min_users_per_product': MIN_USERS_PER_PRODUCT,
        'k_requested': K_PRODUCTS,
        'products': picked,
        'total_eligible_users': len(total_eligible_users),
        'total_eligible_pairs': total_eligible_pairs,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUT_JSON}")


if __name__ == '__main__':
    main()