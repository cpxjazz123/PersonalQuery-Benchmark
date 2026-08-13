#!/usr/bin/env python3
"""E11/E12 — VADES soft-prefix conditioning: generation.

Loads a checkpoint produced by soft_prefix_query_train.py (projector.pt +
tokenizer/ + config.json + split/vector manifests) and generates one query per
(user_id, asin) record using the SAME preprocessing as training:
- identical prompt (placeholder->value mapping) and Qwen chat template;
- identical UserVectorProvider (mode, seed, permutation) so shuffled/zero
  controls are exactly reproducible;
- identical soft-prefix injection (K tokens, gate alpha, position ids,
  attention mask).

E12 deterministic pipeline:
1. LLM generates a query TEMPLATE with placeholders <A1>..<A5>.
2. parse_template validates exactly one occurrence of each required
   placeholder (glued/duplicated/extra forms rejected).
3. replace_placeholders_with_attrs substitutes original values verbatim
   (no inflection/case/number formatting).
4. validate_query_uses_exactly_five_attrs on the final query.
5. up to ``--max_retries`` retries; on persistent failure the final query
   falls back to the deterministic B0 teacher (VADES-retained query or
   nearest content-valid candidate), never emitting corrupted attributes.

Fair-experiment support (issue #12):
- ``--vector_mode`` overrides the conditioning vector AT INFERENCE on the
  SAME checkpoint: correct (vades) / shuffled / zero / global_mean / none.
- identical decoding config for every vector mode; repetition_penalty is
  an independent ablation only (default 1.0 = off).

Outputs: <out_dir>/<category>_vec=<vector_mode>_K=<K>_generated.json
Each result records raw_template, final_query, retry_count, fallback,
failure_reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from user_style_vectors import (  # noqa: E402
    UserVectorProvider,
    load_vades_profiles,
)
from projector import SoftPrefixProjector  # noqa: E402
from content_validation import content_valid  # noqa: E402
from template_placeholder import (  # noqa: E402
    PLACEHOLDER_BY_ATTR,
    parse_template,
    replace_placeholders_with_attrs,
)
from soft_prefix_query_train import build_messages  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class SoftPrefixGenerator(nn.Module):
    """Frozen Qwen + projector; single-sequence generation (no padding)."""

    def __init__(self, base_model, projector, device: str):
        super().__init__()
        self.base = base_model
        self.projector = projector
        self.device = device
        self.num_tokens = projector.num_tokens
        self.base.eval()
        self.projector.eval()

    def build_inputs(self, prompt_ids, user_vec: Optional[np.ndarray]):
        text_embeds = self.base.get_input_embeddings()(prompt_ids)
        target_dtype = next(self.base.parameters()).dtype
        text_embeds = text_embeds.to(target_dtype)
        if self.num_tokens > 0 and user_vec is not None:
            z = torch.as_tensor(user_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            prefix = self.projector(z.to(target_dtype))  # [1, K, d_model]
            full = torch.cat([prefix, text_embeds], dim=1)
            attention_mask = torch.ones(full.shape[:2], dtype=torch.long, device=self.device)
            prefix_pos = torch.arange(self.num_tokens, device=self.device).unsqueeze(0)
            text_pos = torch.arange(
                prompt_ids.size(1), device=self.device
            ).unsqueeze(0) + self.num_tokens
            position_ids = torch.cat([prefix_pos, text_pos], dim=1)
        else:
            full = text_embeds
            attention_mask = torch.ones(full.shape[:2], dtype=torch.long, device=self.device)
            position_ids = torch.arange(
                prompt_ids.size(1), device=self.device
            ).unsqueeze(0)
        return full, attention_mask, position_ids

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        user_vec: Optional[np.ndarray],
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: int,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Manual greedy decoding.

        transformers.generate() with inputs_embeds + soft prefix desyncs its
        internal cache_position (RoPE), producing garbage after the first
        token. Decoding step-by-step keeps prefix embeddings and position ids
        fully under control and yields correct output.
        """
        full, attention_mask, position_ids = self.build_inputs(prompt_ids, user_vec)
        past = None
        generated = []
        for _ in range(max_new_tokens):
            out = self.base(
                inputs_embeds=full if past is None else None,
                input_ids=None if past is None else next_ids,
                attention_mask=attention_mask,
                position_ids=position_ids if past is None else None,
                use_cache=True,
                past_key_values=past,
            )
            past = out.past_key_values
            next_logits = out.logits[0, -1]
            if repetition_penalty != 1.0:
                for tid in set(generated):
                    next_logits[tid] /= repetition_penalty
            next_id = int(torch.argmax(next_logits))
            generated.append(next_id)
            if next_id == eos_token_id:
                break
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)], dim=1
            )
            if past is None:
                position_ids = torch.cat(
                    [position_ids, torch.full((1, 1), position_ids.size(1), dtype=position_ids.dtype, device=self.device)],
                    dim=1,
                )
        return generated

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids_list: List[torch.Tensor],
        user_vecs_list: List[Optional[np.ndarray]],
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: int,
        repetition_penalty: float = 1.0,
    ) -> List[List[int]]:
        """Batched manual greedy decoding.

        All samples share one forward pass per step (right-padded to the
        longest prompt; per-sample position ids + attention masks; per-sample
        prefix). Samples that hit EOS keep emitting EOS for the remaining
        steps so the batch shape stays fixed (correctness unaffected:
        callers truncate at the first EOS). Falls back to the single-sample
        path for a batch of one.
        """
        bsz = len(prompt_ids_list)
        if bsz == 1:
            return [self.generate(
                prompt_ids_list[0], user_vecs_list[0], max_new_tokens,
                pad_token_id, eos_token_id, repetition_penalty,
            )]
        if any(v is None for v in user_vecs_list) and self.num_tokens > 0:
            raise ValueError("batch with prefix requires user_vec for every sample")
        target_dtype = next(self.base.parameters()).dtype
        max_len = max(t.view(-1).size(0) for t in prompt_ids_list)
        emb = self.base.get_input_embeddings()

        text_embeds = []
        masks = []
        positions = []
        for i, (ids, vec) in enumerate(zip(prompt_ids_list, user_vecs_list)):
            ids = ids.view(-1)  # caller passes [1, L]; drop the batch dim
            pad_n = max_len - ids.size(0)
            row_emb = emb(ids).to(target_dtype)
            if pad_n > 0:
                pad_emb = emb(torch.full((pad_n,), pad_token_id, device=self.device)).to(target_dtype)
                row_emb = torch.cat([row_emb, pad_emb], dim=0)
            if self.num_tokens > 0:
                z = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                prefix = self.projector(z.to(target_dtype)).squeeze(0)  # [K, D]
                full_row = torch.cat([prefix, row_emb], dim=0)
                mask = [1] * (self.num_tokens + ids.size(0)) + [0] * pad_n
                pos = list(range(self.num_tokens)) + list(range(self.num_tokens, self.num_tokens + ids.size(0))) + [0] * pad_n
            else:
                full_row = row_emb
                mask = [1] * ids.size(0) + [0] * pad_n
                pos = list(range(ids.size(0))) + [0] * pad_n
            masks.append(mask)
            positions.append(pos)
            text_embeds.append(full_row)

        # full batch embeddings: per-sample prefix/text are already built;
        # pad rows to the longest (they are already equal length).
        full = torch.stack(text_embeds, dim=0)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device)

        past = None
        generated: List[List[int]] = [[] for _ in range(bsz)]
        done = [False] * bsz
        next_token_emb = full  # first step uses the full embedding sequence
        for _ in range(max_new_tokens):
            out = self.base(
                inputs_embeds=next_token_emb if past is None else None,
                input_ids=None if past is None else next_ids,
                attention_mask=attention_mask,
                position_ids=position_ids if past is None else None,
                use_cache=True,
                past_key_values=past,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1]  # [B, V]
            if repetition_penalty != 1.0:
                for b in range(bsz):
                    for tid in set(generated[b]):
                        next_logits[b, tid] /= repetition_penalty
            next_ids = torch.argmax(next_logits, dim=-1)  # [B]
            if all(done):
                break
            for b in range(bsz):
                if done[b]:
                    next_ids[b] = eos_token_id
                    continue
                tid = int(next_ids[b])
                generated[b].append(tid)
                if tid == eos_token_id:
                    done[b] = True
            next_token_emb = emb(next_ids.unsqueeze(1)).to(target_dtype)
            next_ids = next_ids.unsqueeze(1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((bsz, 1), dtype=attention_mask.dtype, device=self.device)], dim=1
            )
        return generated


def load_checkpoint(ckpt_dir: Path, device: str):
    config_p = ckpt_dir / "config.json"
    with open(config_p) as f:
        config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    if len(tokenizer) != base.config.vocab_size:
        # checkpoint tokenizer has extra placeholder special tokens; resize
        # BEFORE loading the LoRA adapter so shapes match, and restore the
        # exact embedding rows trained with (random resize rows would corrupt
        # generation).
        base.resize_token_embeddings(len(tokenizer))
        log(f"resized vocab {base.config.vocab_size} -> {len(tokenizer)}")
        ste_p = ckpt_dir / "special_token_embeddings.pt"
        if ste_p.exists():
            ste = torch.load(ste_p, map_location="cpu")
            with torch.no_grad():
                emb = base.get_input_embeddings()
                for tid, row in zip(ste["token_ids"], ste["embeddings"]):
                    emb.weight[int(tid)].copy_(row)
            log(f"restored {len(ste['token_ids'])} special-token embedding rows")
    lora_dir = ckpt_dir / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(lora_dir))
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    projector = SoftPrefixProjector(
        user_dim=config["user_dim"],
        hidden_dim=config["hidden_dim"],
        num_tokens=config["num_tokens"],
        model_dim=config["model_dim"],
        dtype=torch.bfloat16,
    ).to(device)
    state = torch.load(ckpt_dir / "projector.pt", map_location=device)
    projector.load_state_dict(state)
    model = SoftPrefixGenerator(base, projector, device=device).to(device)
    return model, tokenizer, config


def teacher_query_for(rec: dict, retained_by_key, nearest_by_key, attrs_by_key):
    """Deterministic B0 fallback: VADES-retained query first, else nearest
    content-valid candidate, else None."""
    key = (rec["user_id"], rec["asin"])
    q = retained_by_key.get(key)
    if q and content_valid(q, attrs_by_key.get(key, {})):
        return q
    q = nearest_by_key.get(key)
    if q and content_valid(q, attrs_by_key.get(key, {})):
        return q
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0=all records")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vector_mode", action="append", default=None,
                    help="override conditioning at inference on the same checkpoint: "
                         "vades(correct)/shuffled/zero/global_mean/none; "
                         "repeatable: pass multiple --vector_mode to run all modes "
                         "in one process (model loaded once)")
    ap.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="independent ablation only; must be identical across vector modes")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8, help="batched decoding size")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config = load_checkpoint(ckpt_dir, args.device)
    train_mode = config["mode"]
    vector_modes = args.vector_mode or [train_mode]
    if isinstance(vector_modes, str):
        vector_modes = [vector_modes]
    log(f"checkpoint: mode={train_mode} K={config['num_tokens']} | inference vector_modes={vector_modes}")

    records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    log(f"records={len(records)} (04_query 10-candidate source)")

    with open(ckpt_dir / "vector_manifest.json") as f:
        vector_manifest = json.load(f)
    profiles = load_vades_profiles(args.category)

    # B0 teacher lookup (VADES-retained / nearest content-valid candidate).
    retained_by_key: Dict[tuple, Optional[str]] = {}
    retained_p = REPO_ROOT / "result" / "personal_query" / "06_query" / args.category / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
    if retained_p.exists():
        with open(retained_p) as f:
            for rec in json.load(f):
                retained_by_key[(rec["user_id"], rec["asin"])] = rec.get("syntax_depth_query", {}).get("query")
    nearest_by_key: Dict[tuple, Optional[str]] = {}
    attrs_by_key: Dict[tuple, Dict[str, str]] = {}
    for rec in records:
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        candidates = rec.get("syntax_depth_queries", [])
        if not attrs:
            for cand in candidates:
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if attrs:
            attrs_by_key[(rec["user_id"], rec["asin"])] = attrs
        for cand in candidates:
            if cand.get("query") and content_valid(cand.get("query", ""), attrs or {}):
                nearest_by_key[(rec["user_id"], rec["asin"])] = cand["query"]
                break

    results = []
    providers = {}
    for vector_mode in vector_modes:
        providers[vector_mode] = UserVectorProvider(
            mode=vector_mode if vector_mode != "correct" else train_mode,
            profiles=profiles,
            user_ids=[r["user_id"] for r in records],
            seed=vector_manifest["seed"],
        )
        log(f"provider[{vector_mode}]: {providers[vector_mode].manifest()}")

    for vector_mode in vector_modes:
        jobs = []  # (rec, attrs, prompt_ids, user_vec, has_vector)
        no_attrs = []
        for i, rec in enumerate(records):
            uid, asin = rec["user_id"], rec["asin"]
            attrs = attrs_by_key.get((uid, asin))
            if not attrs:
                no_attrs.append({**rec, "vector_mode": vector_mode, "final_query": None, "error": "no attrs"})
                continue
            prompt_str = tokenizer.apply_chat_template(
                build_messages(attrs), tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(args.device)
            vec, has_vector = providers[vector_mode].get(uid)
            jobs.append((rec, attrs, prompt_ids, vec, has_vector))
        results.extend(no_attrs)

        mode_out = []
        pending = jobs
        round_no = 0
        while pending and round_no <= args.max_retries:
            round_no += 1
            next_pending = []
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start:start + args.batch_size]
                outs = model.generate_batch(
                    [j[2] for j in chunk],
                    [j[3] for j in chunk],
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                )
                for j, out in zip(chunk, outs):
                    rec, attrs, _, vec, has_vector = j
                    # placeholder special tokens (<A1>..) must NOT be skipped in decode
                    raw_template = tokenizer.decode(out, skip_special_tokens=False).strip()
                    required = {PLACEHOLDER_BY_ATTR[k] for k in attrs if k in PLACEHOLDER_BY_ATTR}
                    parsed = parse_template(raw_template, required=required)
                    failure_reason = None
                    final_query = None
                    if not parsed["valid"]:
                        failure_reason = (
                            f"template invalid: missing={parsed['missing']} "
                            f"dup={parsed['duplicated']} extra={parsed['extra']}"
                        )
                    else:
                        final_query, r_err = replace_placeholders_with_attrs(raw_template, attrs)
                        if final_query is None:
                            failure_reason = r_err
                        elif not content_valid(final_query, attrs):
                            failure_reason = "final query failed five-attr validation"
                            final_query = None
                    if final_query is None:
                        next_pending.append((j[0], j[1], j[2], j[3], j[4], raw_template, failure_reason))
                        continue
                    mode_out.append(
                        {
                            "user_id": rec["user_id"],
                            "asin": rec["asin"],
                            "final_query": final_query,
                            "raw_template": raw_template,
                            "retry_count": round_no - 1,
                            "fallback": False,
                            "failure_reason": None,
                            "vector_mode": vector_mode,
                            "num_tokens": config["num_tokens"],
                            "has_vector": bool(has_vector),
                            "repetition_penalty": args.repetition_penalty,
                        }
                    )
                    if len(mode_out) % 20 == 0:
                        log(f"  [{vector_mode}] {len(mode_out)}/{len(jobs)} done")
            pending = [(j[0], j[1], j[2], j[3], j[4]) for j in next_pending]
            if pending:
                log(f"  [{vector_mode}] round {round_no}: {len(pending)} pending retry")

        # fallback for records that never produced a valid template
        for j in pending:
            rec, attrs, _, vec, has_vector = j[:5]
            failure_reason = j[5] if len(j) > 5 else "no valid template"
            raw_template = j[6] if len(j) > 6 else None
            final_query = teacher_query_for(rec, retained_by_key, nearest_by_key, attrs_by_key)
            fallback = True
            if final_query is None:
                failure_reason = (failure_reason or "no valid template") + "; no teacher fallback"
            elif not content_valid(final_query, attrs):
                failure_reason = f"{failure_reason or ''}; teacher fallback invalid"
                final_query = None
            mode_out.append(
                {
                    "user_id": rec["user_id"],
                    "asin": rec["asin"],
                    "final_query": final_query,
                    "raw_template": raw_template,
                    "retry_count": args.max_retries,
                    "fallback": fallback,
                    "failure_reason": failure_reason,
                    "vector_mode": vector_mode,
                    "num_tokens": config["num_tokens"],
                    "has_vector": bool(has_vector),
                    "repetition_penalty": args.repetition_penalty,
                }
            )
        results.extend(mode_out)
        log(f"  [{vector_mode}] done: {sum(1 for r in mode_out if not r['fallback'])}/{len(mode_out)} valid templates")

    for vector_mode in vector_modes:
        mode_results = [r for r in results if r.get("vector_mode") == vector_mode]
        out_p = out_dir / f"{args.category}_vec={vector_mode}_K={config['num_tokens']}_rp={args.repetition_penalty}.json"
        payload = {
            "config": {
                **config,
                "vector_mode_override": vector_mode,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "max_retries": args.max_retries,
                "batch_size": args.batch_size,
            },
            "results": mode_results,
        }
        with open(out_p, "w") as f:
            json.dump(payload, f, indent=2)
        n_valid = sum(1 for r in mode_results if r["final_query"])
        n_fallback = sum(1 for r in mode_results if r["fallback"])
        log(f"wrote {out_p} | valid final={n_valid}/{len(mode_results)} fallback={n_fallback}")
    log("=== done ===")


if __name__ == "__main__":
    main()
