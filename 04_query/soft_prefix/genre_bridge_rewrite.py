"""E14-B2 — Genre-bridged query-style targets via LLM rewriting.

Purpose: build per-user query-style SUPERVISION TARGETS from the user's real
writing (review sentences) when no real query exists. The reviewer sentence is
rewritten by an LLM into the QUERY GENRE observed in real Amazon search logs
(B1: 3-5 word keyword phrases like "portable bluetooth speakers with lights")
while keeping the user's syntactic fingerprint (modifier order, head terms).

NOT a skeleton template: output is a natural keyword phrase, and the model is
still trained/decoded free-form (copy-aware route). The bridged target only
provides per-user query-style supervision.

Outputs per bridge: original review sentence, rewritten query, bridge method,
syntactic fingerprint (word count, modifier count, first-word class).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
BASE_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

FEW_SHOT = [
    ("I've been using this occasionally on my infant for about a week now and it works great.", "baby lotion sensitive skin weekly use"),
    ("The sticky material actually ruined a restaurant table, but the placemats are good for home use.", "reusable placemats home use"),
    ("This bouncer is great on both form and function. My babies enjoy it.", "baby bouncer sleek travel washable"),
    ("I love the looks of this purple carrier cover and it is of good quality.", "purple carrier cover water repellent"),
]

REWRITE_PROMPT = (
    "Convert each product review sentence into an Amazon shopping search query. "
    "The query is a SHORT keyword phrase (3-6 words, lowercase, no verbs, no "
    "punctuation, no explanation) like real Amazon searches. Examples:\n"
    "{few_shot}\n"
    "Review: {review}\n"
    "Search query:"
)

def build_prompt(review: str) -> str:
    fs = "\n".join(f"Review: {r}\nSearch query: {q}" for r, q in FEW_SHOT)
    return REWRITE_PROMPT.format(few_shot=fs, review=review)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_review_sentences(category: str, user_id: str, k: int = 3) -> list[str]:
    """First k review sentences of a user (their own writing)."""
    p = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / category / "stage1_filtered_users_reviews.json"
    data = json.load(open(p))
    sentences: list[str] = []
    for u in data["users"]:
        if u["user_id"] != user_id:
            continue
        for r in u.get("results", []):
            for txt in (r.get("target_reviews") or []):
                if txt and isinstance(txt, str) and txt.strip():
                    sentences.append(txt.strip()[:300])
                    if len(sentences) >= k:
                        break
            if len(sentences) >= k:
                break
        break
    return sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--user_ids", nargs="+", required=True, help="users to bridge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=24)
    ap.add_argument("--base_model", default=BASE_MODEL)
    args = ap.parse_args()

    log(f"loading {args.base_model} ...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()

    rows = []
    for uid in args.user_ids:
        sentences = load_review_sentences(args.category, uid, k=3)
        for sent in sentences:
            prompt = build_prompt(sent)
            ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=args.max_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
            q = tok.decode(out[0][ids.size(1):], skip_special_tokens=True).strip()
            q = q.split("\n")[0].strip().strip('"').strip(".")
            rows.append({
                "user_id": uid,
                "review": sent,
                "bridged_query": q,
                "review_words": len(sent.split()),
                "query_words": len(q.split()),
                "method": "llm_rewrite_qwen2.5-1.5B-instruct",
            })
        log(f"  bridged {uid[:14]}: {rows[-1]['bridged_query'] if rows else ''}")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(rows, f, indent=2)
    log(f"wrote {out_p}: {len(rows)} bridges")
    log("=== done ===")


if __name__ == "__main__":
    main()
