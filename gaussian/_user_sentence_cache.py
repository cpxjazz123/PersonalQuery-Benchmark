"""User-sentence cache for 8.A / 8.A.2 / 8.A.3 cohort (B00A4B34IA, 12 users).

Caches the scan of `data/Baby_Products_2023.jsonl.gz` for the 12-user cohort
to a single gzipped JSONL. Avoids re-scanning the 6M-line source file
on every run (~37s saved per run).

Cache key:
  cohort_hash = sha1("-".join(sorted(cohort)).hexdigest()[:12]
  file path   = scratch2/gaussian_vades/phase8a_user_sents_{hash}.jsonl.gz

Each line is JSON: {"uid": "...", "sentence": "..."}
(dedup per-user; same sentence text kept once per user).

Usage in any 8.A.x script (replaces the scan loop):

    from _user_sentence_cache import load_user_sents
    user_sents = load_user_sents(COHORT, REPO)
    # user_sents[uid] = list[str]   deduped, 8-60 word filter NOT applied

This module is local to the gaussian/ directory. Do NOT move to common/.
"""
import collections, gzip, hashlib, json, os, time

CACHE_DIR = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades"


def _cohort_hash(cohort):
    return hashlib.sha1("-".join(sorted(cohort)).encode()).hexdigest()[:12]


def _cache_path(cohort):
    return os.path.join(CACHE_DIR, f"phase8a_user_sents_{_cohort_hash(cohort)}.jsonl.gz")


def build_cache(cohort, repo, force=False):
    """Scan source reviews once and cache per-user sentences.

    Filters: 8 <= len(s.split()) <= 60  (matches the per-sentence feature
    gate used by the 8.A family).
    Returns dict[uid] -> list[str] (deduped, lowercase-dedup).
    """
    cache_file = _cache_path(cohort)
    if os.path.exists(cache_file) and not force:
        print(f"[cache] hit  {cache_file}")
        return load_cache(cohort)

    import re
    split_sents = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]

    print(f"[cache] miss, scanning 6M-line Baby_Products_2023.jsonl.gz...")
    t0 = time.time()
    cohort_set = set(cohort)
    user_sents = collections.defaultdict(set)
    n_scanned = 0
    with gzip.open(f"{repo}/data/Baby_Products_2023.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            if uid in cohort_set and r.get("text"):
                for s in split_sents(r["text"]):
                    if 8 <= len(s.split()) <= 60:
                        user_sents[uid].add(s.lower())
            n_scanned += 1
            if n_scanned % 5_000_000 == 0:
                print(f"  {n_scanned/1e6:.1f}M scanned, t={time.time()-t0:.0f}s")
    print(f"  done: scanned {n_scanned}, t={time.time()-t0:.0f}s")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
        for uid, sents in user_sents.items():
            for s in sents:
                f.write(json.dumps({"uid": uid, "sentence": s}) + "\n")
    n_lines = sum(len(v) for v in user_sents.values())
    print(f"[cache] wrote {n_lines} sents → {cache_file}")
    return {uid: list(s) for uid, s in user_sents.items()}


def load_cache(cohort):
    """Load cached user sentences. Returns dict[uid] -> list[str] (lowercase)."""
    cache_file = _cache_path(cohort)
    if not os.path.exists(cache_file):
        return None
    user_sents = collections.defaultdict(list)
    with gzip.open(cache_file, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user_sents[r["uid"]].append(r["sentence"])
    return dict(user_sents)


def load_user_sents(cohort, repo):
    """Public API: build cache if missing, else load cache."""
    cached = load_cache(cohort)
    if cached is None:
        return build_cache(cohort, repo)
    print(f"[cache] hit  {_cache_path(cohort)}")
    return cached


def main():
    """Build the cache for the canonical 12-user cohort (idempotent)."""
    COHORT = [
        "AHM56WA4FAB2KZITXPRW2WUPZIMQ", "AHLG5OEROOZJVHMD6TNKUKOL6GFA",
        "AHACBVNWG7USY3FSLG2TPVWOQZZA", "AEVYSHLGGB64TL6SSODHFRP67VGQ",
        "AGAORBCX76OT4GQU3SJZF3TNICWA", "AFIROMS23ORMKGET6XC6ZEDCTEXQ",
        "AGKSBZKWZHGJKCNMBMM6AYTX3WHQ", "AEPMPLRZWHU7VAPDGCFF2JPQDJYQ",
        "AECSNCOIZWGHJCONYOWQE5IIARCQ", "AG2ADGIK63GFBE4DLKJXCJ2DS3MA",
        "AH2EEKGHPEZTYJLJX7DESQ3VY4GQ", "AERM7FIHHMIQH5FSEATT4WBBKF3Q",
    ]
    REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
    sents = build_cache(COHORT, REPO)
    for uid, ss in sorted(sents.items()):
        print(f"  {uid[:30]} : {len(ss):>5} unique sentences")


if __name__ == "__main__":
    main()
