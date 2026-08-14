#!/usr/bin/env python3
"""E11 — VADES soft-prefix conditioning: training (SFT).

Main route (B5): z_user = L2-normalized VADES 20-dim user_mu -> MLP projector ->
K soft-prefix tokens prepended to the Qwen2-7B input embeddings. Qwen is
frozen; only the projector is trained (--lora enables shared LoRA for B6).

Conditioning modes (experiment matrix):
  none        (B1)  no user conditioning at all
  zero        (B3)  zero vector conditioning
  global_mean (B2)  global mean user vector
  shuffled    (B3)  fixed seeded permutation of user vectors
  vades       (B5)  VADES user_mu soft prefix (main route)
  e5_mean     (B4)  E5 review mean soft prefix (control)

Guarantees (issue #11 acceptance):
- trainable-parameter assertion: projector must be trainable; when LoRA is
  enabled LoRA params must also be trainable (no double-freeze);
- supervision target is never the fixed first candidate: y+ is the
  content-valid candidate nearest to the user's VADES center (or the
  VADES-retained teacher query when content-valid);
- prompt contains the five product attributes explicitly;
- attention mask never treats padding EOS tokens as valid (explicit 0/1);
- train/generate share the exact same prompt + chat-template + padding code
  (this module and soft_prefix_query_generate.py import the same helpers);
- missing user vectors are explicit zero + has_vector flag, never random;
- user split (test users unseen at training) with a seeded manifest;
- non-inferiority margin for content/retrieval metrics is fixed in the run
  config BEFORE evaluation.

Outputs:
  checkpoint/  projector.pt + tokenizer/ + config.json + manifests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from user_style_vectors import (  # noqa: E402
    UserVectorProvider,
    load_vades_profiles,
    load_feature_scaler,
    load_candidate_feature_rows,
    VADES_TAG,
)
from projector import SoftPrefixProjector  # noqa: E402
from content_validation import (  # noqa: E402
    StyleDatasetBuilder,
    build_attr_prompt,
)
from template_placeholder import (  # noqa: E402
    build_attr_mapping_prompt,
    parse_template,
)

QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
HIDDEN_DIM = 3584  # Qwen2-7B hidden dim
SYSTEM_PROMPT = (
    "You are a shopping query writer. You must produce a query TEMPLATE using "
    "every placeholder listed in the product attributes exactly once."
)
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_messages(attrs_used: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_attr_mapping_prompt(attrs_used)},
    ]


def encode_sft_example(
    tokenizer, attrs_used: Dict[str, str], target_query: str,
    max_query_len: int = 80,
) -> Dict[str, list]:
    """Encode (prompt, target) with the shared chat template.

    Returns input_ids/labels/attention_mask/position_ids where only the
    assistant response tokens are supervised (labels=-100 elsewhere) and the
    mask is explicit (0 for padding, 1 for real tokens). Right-padded; the
    caller prepends the K soft-prefix embedding rows afterwards.
    """
    prompt_str = tokenizer.apply_chat_template(
        build_messages(attrs_used), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    target_ids = tokenizer.encode(target_query, add_special_tokens=False)[:max_query_len]
    target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "position_ids": list(range(len(input_ids))),
    }


def collate(batch: List[dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad; padding tokens get mask 0 and label -100."""
    max_len = max(len(s["input_ids"]) for s in batch)
    fields = ["input_ids", "labels", "attention_mask", "position_ids"]
    padded: Dict[str, list] = {k: [] for k in fields}
    for s in batch:
        pad_n = max_len - len(s["input_ids"])
        padded["input_ids"].append(s["input_ids"] + [pad_token_id] * pad_n)
        padded["labels"].append(s["labels"] + [-100] * pad_n)
        padded["attention_mask"].append(s["attention_mask"] + [0] * pad_n)
        padded["position_ids"].append(s["position_ids"] + list(range(max_len - pad_n, max_len)))
    out = {
        "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
        "labels": torch.tensor(padded["labels"], dtype=torch.long),
        "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
        "position_ids": torch.tensor(padded["position_ids"], dtype=torch.long),
    }
    return out


class SoftPrefixModel(nn.Module):
    """Frozen Qwen2-7B + trainable SoftPrefixProjector.

    Forward: [prefix K tokens; text tokens] as inputs_embeds with an explicit
    attention mask (prefix + real tokens = 1) and position ids.
    """

    def __init__(
        self,
        base_model,
        projector: SoftPrefixProjector,
        device: str,
    ):
        super().__init__()
        self.base = base_model
        self.projector = projector
        self.device = device
        self.num_tokens = projector.num_tokens
        # Freeze the base model WITHOUT double-freezing LoRA adapters.
        # get_peft_model() marks LoRA params trainable; freezing via
        # self.base.parameters() after PEFT wrapping would re-freeze them
        # (the exact bug fixed in issue #11 Phase 0).
        if hasattr(base_model, "active_peft_config") or "peft" in type(base_model).__module__:
            for name, p in base_model.named_parameters():
                if "lora_" in name:
                    continue
                if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
                    # E12: placeholder rows were explicitly unfrozen with a
                    # grad mask; keep them trainable (not a double-freeze).
                    continue
                if p.requires_grad:
                    p.requires_grad = False
        else:
            for p in base_model.parameters():
                p.requires_grad = False
        self.base.eval()
        if getattr(self.base, "config", None) is not None and hasattr(self.base.config, "use_cache"):
            self.base.config.use_cache = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        user_vecs: torch.Tensor,
    ):
        text_embeds = self.base.get_input_embeddings()(input_ids)
        bsz = text_embeds.size(0)
        if self.num_tokens > 0:
            proj_dtype = next(self.projector.parameters()).dtype
            prefix = self.projector(user_vecs.to(proj_dtype))  # [B, K, d_model]
            full = torch.cat([prefix, text_embeds], dim=1)
            prefix_mask = torch.ones((bsz, self.num_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            prefix_labels = torch.full(
                (bsz, self.num_tokens), -100, dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
            prefix_pos = torch.arange(
                self.num_tokens, dtype=position_ids.dtype, device=position_ids.device
            ).unsqueeze(0).expand(bsz, -1)
            text_pos = position_ids + self.num_tokens
            position_ids = torch.cat([prefix_pos, text_pos], dim=1)
        else:
            full = text_embeds
        out = self.base(
            inputs_embeds=full,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
        )
        return out


class StyleQueryDataset(Dataset):
    def __init__(self, rows: List[dict], provider: UserVectorProvider):
        self.rows = rows
        self.provider = provider

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        vec, has_vector = self.provider.get(row["user_id"])
        # attrs_used key sets vary per record; serialize to avoid default
        # collate trying to merge dicts with mismatched keys.
        return {
            "user_id": row["user_id"],
            "attrs_json": json.dumps(row["attrs_used"], ensure_ascii=False),
            "y_plus_query": row["y_plus_query"],
            "user_vec": np.zeros(self.provider.vector_dim, dtype=np.float32)
            if vec is None else vec,
            "has_vector": has_vector,
            "weight": float(row.get("weight", 1.0)),
        }


def make_user_split(user_ids: List[str], test_frac: float, seed: int) -> Dict[str, List[str]]:
    """Seeded user split: test users never appear in training."""
    rng = np.random.default_rng(seed)
    ids = sorted(set(user_ids))
    n_test = max(1, int(round(len(ids) * test_frac)))
    idx = rng.permutation(len(ids))
    test_ids = [ids[int(i)] for i in idx[:n_test]]
    train_ids = [ids[int(i)] for i in idx[n_test:]]
    return {"train_users": train_ids, "test_users": test_ids}


def train(
    category: str,
    out_dir: Path,
    mode: str = "vades",
    num_tokens: int = 4,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-3,
    seed: int = 42,
    test_frac: float = 0.15,
    max_records: int = 0,
    hidden_dim: int = 128,
    lora: bool = False,
    lora_r: int = 8,
    gate_init: float = 1e-3,
    max_query_len: int = 80,
    device: str = "cuda:0",
    base_model_path: str = QWEN_PATH,
    template_targets: bool = True,
    add_placeholder_tokens: bool = True,
    supervision: str = "primary",
    topk_temperature: float = 1.0,
    counterfactual: bool = False,
    cf_margin: float = 0.2,
    cf_lambda: float = 1.0,
) -> Dict:
    log(f"=== E11 train {category} mode={mode} K={num_tokens} ===")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if mode == "none":
        num_tokens = 0  # B1: no user conditioning means no prefix tokens at all
        log("mode=none -> num_tokens forced to 0 (no prefix)")

    profiles = load_vades_profiles(category)
    scaler = load_feature_scaler(category)
    candidate_rows = load_candidate_feature_rows(category)
    log(f"vades profiles={len(profiles)} candidates={len(candidate_rows)}")

    records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    retained_p = REPO_ROOT / "result" / "personal_query" / "06_query" / category / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
    retained_by_key: Dict[tuple, Optional[str]] = {}
    if retained_p.exists():
        with open(retained_p) as f:
            for rec in json.load(f):
                retained_by_key[(rec["user_id"], rec["asin"])] = rec.get("syntax_depth_query", {}).get("query")
    if not candidate_rows:
        log("candidate feature file missing -> building from 04_query records...")
        from build_candidate_features import build_candidate_features_file
        candidate_rows = build_candidate_features_file(category, records)
    if max_records:
        records = records[:max_records]
    log(f"records={len(records)}")

    builder = StyleDatasetBuilder(category, profiles, scaler, seed=seed)
    rows, skipped = builder.build_all(
        records, candidate_rows, retained_by_key,
        supervision=supervision, template_targets=template_targets,
        topk_temperature=topk_temperature,
    )
    skipped_reasons = sorted({s["reason"] for s in skipped})
    log(f"data rows={len(rows)} skipped={len(skipped)} reasons={skipped_reasons} "
        f"supervision={supervision}")
    if supervision == "primary":
        # V-B1: exactly one primary positive per (user_id, asin).
        keys = [(r["user_id"], r["asin"]) for r in rows]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            raise ValueError(f"primary supervision produced duplicate (user,asin): {list(dup)[:5]}")
        log("V-B1 ok: exactly one primary positive per (user_id, asin)")

    split = make_user_split([r["user_id"] for r in rows], test_frac=test_frac, seed=seed)
    train_rows = [r for r in rows if r["user_id"] in set(split["train_users"])]
    test_rows = [r for r in rows if r["user_id"] in set(split["test_users"])]
    log(f"train rows={len(train_rows)} test rows={len(test_rows)} "
        f"train users={len(split['train_users'])} test users={len(split['test_users'])}")

    e5_vecs = None
    if mode == "e5_mean":
        e5_vecs = load_e5_user_vecs(records, category)

    provider = UserVectorProvider(mode, profiles, [r["user_id"] for r in rows], seed=seed, e5_vecs=e5_vecs)
    log(f"vector provider: {provider.manifest()}")
    cf_provider = None
    if counterfactual:
        # E13-C1: seeded shuffled vectors for the counterfactual ranking loss.
        cf_provider = UserVectorProvider("shuffled", profiles, [r["user_id"] for r in rows], seed=seed + 1, e5_vecs=None)
        log(f"counterfactual shuffled provider: {cf_provider.manifest()}")

    log(f"loading Qwen2-7B from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if template_targets and add_placeholder_tokens:
        # E12: register <A1>..<A5> (and the full A1..A18 set used by the data)
        # as special tokens so each placeholder is ONE token id, stable across
        # contexts (raw "<A1>" splits into <A/1/> and even merges into ">."
        # tokens, which a frozen model cannot reliably emit).
        from template_placeholder import PLACEHOLDER_BY_ATTR
        existing = set(tokenizer.get_vocab().keys())
        new_tokens = sorted(
            {ph for ph in PLACEHOLDER_BY_ATTR.values()} - existing
        )
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        log(f"added {len(new_tokens)} placeholder special tokens: {new_tokens[:5]}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    if template_targets and add_placeholder_tokens:
        base_model.resize_token_embeddings(len(tokenizer))
        # New embedding rows initialized at the vocab mean (sane prior); the
        # rows are PERSISTED so inference can restore identical embeddings.
        new_ids = sorted(tokenizer.convert_tokens_to_ids(new_tokens))
        with torch.no_grad():
            emb = base_model.get_input_embeddings()
            # mean must be computed in float32: bf16 accumulation over ~150K
            # rows collapses to near-zero (norm ~0.1), corrupting generation.
            mean = emb.weight[:new_ids[0]].float().mean(dim=0)
            for tid in new_ids:
                emb.weight[tid].copy_(mean.to(emb.weight.dtype))
        log(f"placeholder special-token embeddings initialized at vocab mean ({len(new_ids)} rows)")

    lora_adapter = None
    if lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=2 * lora_r,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_cfg)
        lora_adapter = base_model
    else:
        for p in base_model.parameters():
            p.requires_grad = False

    projector = SoftPrefixProjector(
        user_dim=provider.vector_dim,
        hidden_dim=hidden_dim,
        num_tokens=num_tokens,
        model_dim=HIDDEN_DIM,
        dtype=torch.bfloat16,
        gate_init=gate_init,
    ).to(device)
    model = SoftPrefixModel(base_model, projector, device=device).to(device)

    if template_targets and add_placeholder_tokens:
        # E12: the LM-head rows for the new tokens are zero after resize and
        # would stay zero (frozen) -> the model could never EMIT a placeholder
        # at generation (logits ~0 regardless of hidden state). Unfreeze ONLY
        # the new rows of the input embedding and the LM head via a gradient
        # mask. IMPORTANT: this must happen AFTER get_peft_model(), which
        # re-freezes all non-LoRA parameters.
        emb_mod = base_model.get_input_embeddings()
        head_mod = base_model.get_output_embeddings()
        emb_mod.weight.requires_grad = True
        head_mod.weight.requires_grad = True
        mask = torch.zeros(emb_mod.weight.size(0), dtype=torch.bool, device=emb_mod.weight.device)
        mask[new_ids] = True
        for mod in (emb_mod, head_mod):
            m = mask.to(mod.weight.device).unsqueeze(1)
            mod.weight.register_hook(lambda grad, m=m: grad * m.to(grad.dtype))
        log(f"unfroze {int(mask.sum())} new-row params in input embedding + LM head (grad-masked, after PEFT wrap)")

    # Trainable-parameter assertion (issue #11 acceptance): projector must be
    # trainable; LoRA params (if enabled) must also be trainable.
    n_proj = sum(1 for p in model.projector.parameters())
    assert any(p.requires_grad for p in model.projector.parameters()), "projector must be trainable"
    n_base_train = sum(1 for p in model.base.parameters() if p.requires_grad)
    if lora:
        assert n_base_train > 0, "LoRA enabled but no base/LoRA params trainable (double-freeze bug)"
    else:
        # without LoRA only the E12 placeholder rows (embed + lm_head) may be
        # trainable; the rest of the base must stay frozen.
        assert n_base_train <= 2, f"expected only E12 placeholder rows trainable, got {n_base_train}"
    n_proj_params = model.projector.num_parameters()
    log(f"assert OK: projector trainable params={n_proj_params} ({n_proj} tensors), "
        f"base trainable={n_base_train}{' (LoRA)' if lora else ''}")

    ds = StyleQueryDataset(train_rows, provider)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_ds = StyleQueryDataset(test_rows, provider)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def encode_batch(batch) -> Dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            # default collate deep-collates nested dicts; unpack per-sample.
            attrs_list = [json.loads(a) for a in batch["attrs_json"]]
            y_plus_list = batch["y_plus_query"]
            user_vecs = torch.stack(
                [torch.as_tensor(v, dtype=torch.float32) for v in batch["user_vec"]]
            )
            weights = torch.tensor(
                [float(w) for w in batch["weight"]], dtype=torch.float32
            )
        else:
            attrs_list = [json.loads(r["attrs_json"]) for r in batch]
            y_plus_list = [r["y_plus_query"] for r in batch]
            user_vecs = torch.stack(
                [torch.as_tensor(r["user_vec"], dtype=torch.float32) for r in batch]
            )
            weights = torch.tensor([float(r["weight"]) for r in batch], dtype=torch.float32)
        enc = collate(
            [
                encode_sft_example(tokenizer, attrs, target, max_query_len)
                for attrs, target in zip(attrs_list, y_plus_list)
            ],
            tokenizer.pad_token_id,
        )
        return {**enc, "user_vecs": user_vecs, "weights": weights}

    def cf_loss(batch, batch_t, out) -> torch.Tensor:
        """E13-C1 counterfactual ranking: the correct vector's NLL must be
        lower than the shuffled vector's NLL (pre-registered margin)."""
        cf_vecs = []
        for r in batch["user_id"]:
            v, _ = cf_provider.get(r)
            cf_vecs.append(v if v is not None else np.zeros(provider.vector_dim, dtype=np.float32))
        cf_t = torch.tensor(np.stack(cf_vecs), dtype=torch.float32, device=device)
        out_cf = model(
            input_ids=batch_t["input_ids"],
            labels=batch_t["labels"],
            attention_mask=batch_t["attention_mask"],
            position_ids=batch_t["position_ids"],
            user_vecs=cf_t,
        )
        loss_correct = compute_loss(batch_t, out)
        loss_shuffled = compute_loss(batch_t, out_cf)
        hinge = torch.relu(loss_correct - loss_shuffled + cf_margin)
        return loss_correct + cf_lambda * hinge, loss_correct, loss_shuffled, hinge

    def compute_loss(batch_t, out) -> torch.Tensor:
        """Token-level CE with per-sample weights (E13-B). The model prepends
        K soft-prefix tokens, so the effective labels include K leading -100s;
        align logits and labels to the FULL (prefix+text) sequence."""
        labels = batch_t["labels"]
        if model.num_tokens > 0:
            prefix_labels = torch.full(
                (labels.size(0), model.num_tokens), -100,
                dtype=labels.dtype, device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
        logits = out.logits  # [B, K+T, V]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        V = shift_logits.size(-1)
        ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1),
            reduction="none",
        )
        mask = (shift_labels.view(-1) != -100).float()
        B = labels.size(0)
        per_sample = (ce.view(B, -1) * mask.view(B, -1)).sum(dim=1) / mask.view(B, -1).sum(dim=1).clamp_min(1.0)
        w = batch_t["weights"].to(per_sample.device)
        return (per_sample * w).sum() / w.sum().clamp_min(1.0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = len(dl) * epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(20, total_steps // 10), num_training_steps=total_steps)
    log(f"trainable params: {sum(p.numel() for p in trainable_params)} | total_steps={total_steps}")

    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dl:
            batch_t = {k: v.to(device) for k, v in encode_batch(batch).items()}
            out = model(
                input_ids=batch_t["input_ids"],
                labels=batch_t["labels"],
                attention_mask=batch_t["attention_mask"],
                position_ids=batch_t["position_ids"],
                user_vecs=batch_t["user_vecs"],
            )
            if counterfactual:
                loss, loss_c, loss_s, hinge = cf_loss(batch, batch_t, out)
            else:
                loss = compute_loss(batch_t, out)
            optimizer.zero_grad()
            if trainable_params and model.num_tokens > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
            epoch_loss += float(loss.detach())
            step += 1
            if step % 20 == 0:
                extra = ""
                if counterfactual:
                    extra = f" | correct={float(loss_c.detach()):.4f} shuffled={float(loss_s.detach()):.4f} hinge={float(hinge.detach()):.4f}"
                log(f"  step {step}/{total_steps} loss={float(loss.detach()):.4f}{extra}")
        log(f"  epoch {epoch + 1}/{epochs} avg_loss={epoch_loss / max(len(dl), 1):.4f}")

    # Test-set evaluation (SFT loss on held-out users) at the end.
    model.eval()
    test_loss_sum = 0.0
    test_n = 0
    with torch.no_grad():
        for batch in test_dl:
            batch_t = {k: v.to(device) for k, v in encode_batch(batch).items()}
            out = model(
                input_ids=batch_t["input_ids"],
                labels=batch_t["labels"],
                attention_mask=batch_t["attention_mask"],
                position_ids=batch_t["position_ids"],
                user_vecs=batch_t["user_vecs"],
            )
            test_loss_sum += float(out.loss.detach()) * batch_t["labels"].size(0)
            test_n += batch_t["labels"].size(0)
    test_loss = test_loss_sum / max(test_n, 1)
    log(f"held-out user SFT loss={test_loss:.4f}")

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), ckpt / "projector.pt")
    if template_targets and add_placeholder_tokens:
        # E12: save the FINAL (post-training) special-token embedding rows so
        # inference restores exactly what the model was trained with.
        new_ids = sorted(tokenizer.convert_tokens_to_ids(new_tokens))
        with torch.no_grad():
            emb_final = model.base.get_input_embeddings().weight
            torch.save(
                {"token_ids": new_ids, "embeddings": emb_final[new_ids].detach().cpu()},
                ckpt / "special_token_embeddings.pt",
            )
        log(f"saved post-training special-token embeddings ({len(new_ids)} rows)")
    tokenizer.save_pretrained(ckpt / "tokenizer")
    if lora_adapter is not None:
        lora_adapter.save_pretrained(ckpt / "lora_adapter")
    config = {
        "category": category,
        "mode": mode,
        "num_tokens": num_tokens,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "test_frac": test_frac,
        "max_records": max_records,
        "hidden_dim": hidden_dim,
        "lora": lora,
        "lora_r": lora_r,
        "max_query_len": max_query_len,
        "device": device,
        "base_model_path": base_model_path,
        "user_dim": provider.vector_dim,
        "model_dim": HIDDEN_DIM,
        "vector_encoder": "VADES-lite diagonal 20d user_mu (train10/holdout5 sentences, "
                          f"tag={VADES_TAG})",
        "tokenizer": "Qwen2-7B-Instruct chat template",
        "padding_side": "right",
        "supervision": "y+ = content-valid candidate nearest user VADES center "
                       "(or VADES-retained teacher query when content-valid); never queries[0]",
        "non_inferiority_margin_content_attr": 0.15,
        "non_inferiority_margin_retrieval": 0.05,
        "train_loss": float(epoch_loss / max(len(dl), 1)),
        "heldout_user_sft_loss": float(test_loss),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "n_skipped": len(skipped),
        "skipped_reasons": skipped_reasons,
        "template_targets": bool(template_targets),
        "supervision": supervision,
        "topk_temperature": topk_temperature,
        "counterfactual": bool(counterfactual),
        "cf_margin": cf_margin,
        "cf_lambda": cf_lambda,
        "gate": {
            "type": "scalar_alpha",
            "init": gate_init,
            "final": float(model.projector.alpha.detach().item()),
        },
        "created_at": datetime.now().isoformat(),
    }
    with open(ckpt / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(ckpt / "split_manifest.json", "w") as f:
        json.dump(split, f, indent=2)
    with open(ckpt / "vector_manifest.json", "w") as f:
        json.dump(provider.manifest(), f, indent=2)
    with open(ckpt / "data_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(ckpt / "skipped_records.json", "w") as f:
        json.dump(skipped, f, indent=2)
    log(f"checkpoint saved to {ckpt}")
    log("=== done ===")
    return config


def load_e5_user_vecs(records: List[dict], category: str) -> Dict[str, np.ndarray]:
    """E5 mean of user review sentence embeddings (control route B4 only)."""
    import sentence_transformers
    log("loading E5 for control route...")
    reviews_p = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / category / "stage1_filtered_users_reviews.json"
    with open(reviews_p) as f:
        data = json.load(f)
    user_reviews: Dict[str, List[str]] = {}
    for rec in data["users"]:
        uid = rec.get("user_id")
        for r in rec.get("results", []):
            for txt in r.get("target_reviews", []):
                if txt and isinstance(txt, str) and txt.strip():
                    user_reviews.setdefault(uid, []).append(txt[:1000])
    e5 = sentence_transformers.SentenceTransformer("intfloat/e5-large-v2", device="cuda:0")
    vecs: Dict[str, np.ndarray] = {}
    for uid in {r["user_id"] for r in records}:
        revs = user_reviews.get(uid, [])
        if not revs:
            continue
        embs = e5.encode(revs[:32], batch_size=32, normalize_embeddings=True)
        vecs[uid] = embs.mean(axis=0).astype(np.float32)
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=sorted(UserVectorProvider.MODES), default="vades")
    ap.add_argument("--num_tokens", type=int, default=4, help="K soft-prefix tokens")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--max_records", type=int, default=0, help="0=all")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lora", action="store_true", help="B6: shared LoRA + projector")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--gate_init", type=float, default=1e-3)
    ap.add_argument("--no_template", action="store_true", help="disable E12 placeholder-template targets")
    ap.add_argument("--placeholder_tokens", action="store_true", default=True,
                    help="register <A1>..<A18> as special tokens (E12; default on)")
    ap.add_argument("--supervision", choices=["primary", "all_candidates", "topk_weighted"],
                    default="primary", help="E13-B SFT supervision strategy")
    ap.add_argument("--topk_temperature", type=float, default=1.0)

    ap.add_argument("--counterfactual", action="store_true", help="E13-C1 counterfactual ranking loss")
    ap.add_argument("--cf_margin", type=float, default=0.2)
    ap.add_argument("--cf_lambda", type=float, default=1.0)
    args = ap.parse_args()
    args_dict = vars(args)
    args_dict["template_targets"] = not args_dict.pop("no_template")
    args_dict["add_placeholder_tokens"] = args_dict.pop("placeholder_tokens", True)
    train(**args_dict)


if __name__ == "__main__":
    main()
