#!/usr/bin/env python3
"""E17 Step 3b training: copy-aware content + user-vector syntax alignment.

Losses (all on the generated target part):
  lm + copy_lambda*ptr        content fidelity (inherited)
  syn = MSE(SyntaxHead(h_pool), y_z)      syntax-head supervised on training
                                          query features (z-scored)
  align = MSE(SyntaxHead(h_pool), z_u_z)  align generation to the user's
                                          profile syntax contour (z-scored)
  cf = margin(align_correct - align_shuffled + m)  counterfactual: correct
                                          vector beats a shuffled one
h_pool = mean-pooled hidden of valid target positions.

Dev rule (preregistered): pick the epoch with the largest fraction of dev
samples where the correct-vector syntax distance < shuffled-vector distance
(must exceed 0.5 baseline); report at each epoch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from copy_aware import CopyAwareHead, build_attr_prompt_lines, mixed_logits  # noqa: E402
from projector import SoftPrefixProjector  # noqa: E402
from user_stat_vector import FEATURES20, OPENER_CLASSES, opener_class_of  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features  # noqa: E402

BASE = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E17 = REPO_ROOT / "result" / "personal_query" / "e17"
SPLIT = json.load(open(E17 / "e17_train_dev_split.json"))
TRAIN_VECS = json.load(open(E17 / "e17_train_vectors.json"))
CKPT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e17_ckpt")

SEED = 42
EPOCHS = 8
BATCH = 8
LR = 3e-4
LORA_R = 8
NUM_TOKENS = 4
GATE_INIT = 0.05
COPY_LAMBDA = 0.5
LAMBDA_SYN = 1.0
LAMBDA_ALIGN = 1.0
LAMBDA_CF = 0.5
CF_MARGIN = 0.05
MAX_QUERY_LEN = 80
MAX_PROMPT_LEN = 256

FEATS = FEATURES20  # 20 features; head predicts 20 + 10 opener = 30
LOG = sys.stdout


def log(*a):
    print(*a, flush=True)


class SyntaxHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = len(FEATS) + len(OPENER_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Linear(256, out_dim))

    def forward(self, h):
        return self.net(h)


def query_features_batch(texts: list[str]) -> np.ndarray:
    nlp = load_spacy_model()
    F = []
    for t in texts:
        try:
            f = extract_clause_features(t)
            row = [float(f.get(k, 0.0)) for k in FEATS]
        except Exception:
            row = [0.0] * len(FEATS)
        try:
            toks = [tk for tk in nlp(t) if not tk.is_punct and not tk.is_space]
            o = OPENER_CLASSES.index(opener_class_of(toks[0].text)) if toks else 0
        except Exception:
            o = 0
        oh = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
        oh[o] = 1.0
        F.append(row + oh.tolist())
    return np.asarray(F, dtype=np.float32)


def zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / (sd + 1e-9)


class E17Dataset(Dataset):
    def __init__(self, rows, tokenizer, feat_z, vecs, z_mu, z_sd, max_query_len=MAX_QUERY_LEN):
        self.rows = rows
        self.tok = tokenizer
        self.feat_z = feat_z
        self.vecs = vecs
        self.z_mu = z_mu
        self.z_sd = z_sd
        self.max_query_len = max_query_len
        self.enc = []
        for r in rows:
            prompt = build_attr_prompt_lines(r.get("attrs") or r.get("attrs_used"))
            pid = tokenizer(prompt, add_special_tokens=False).input_ids[: MAX_PROMPT_LEN]
            tids = tokenizer(r.get("query") or r.get("y_plus_query"), add_special_tokens=False).input_ids[: max_query_len]
            self.enc.append((r, pid, tids))

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        r, pid, tids = self.enc[i]
        uv = self.vecs.get(r["user_id"])
        z30 = None
        if uv is not None:
            z30 = zscore(np.asarray(uv, dtype=np.float32), self.z_mu, self.z_sd)
        return {"user_id": r["user_id"], "prompt_ids": torch.tensor(pid, dtype=torch.long),
                "target_ids": torch.tensor(tids, dtype=torch.long),
                "z30": torch.tensor(z30, dtype=torch.float32) if z30 is not None else None,
                "feat_z": torch.tensor(self.feat_z[i], dtype=torch.float32),
                "attrs": r.get("attrs") or r.get("attrs_used")}


def collate(batch, pad_id):
    max_p = max(b["prompt_ids"].size(0) for b in batch)
    max_t = max(b["target_ids"].size(0) for b in batch)
    pids, tids, feats, zs, uids = [], [], [], [], []
    D = len(FEATS) + len(OPENER_CLASSES)
    for b in batch:
        pids.append(F.pad(b["prompt_ids"], (0, max_p - b["prompt_ids"].size(0)), value=pad_id))
        tids.append(F.pad(b["target_ids"], (0, max_t - b["target_ids"].size(0)), value=-100))
        feats.append(b["feat_z"])
        zs.append(b["z30"] if b["z30"] is not None else torch.zeros(D))
        uids.append(b["user_id"])
    return {"prompt_ids": torch.stack(pids), "target_ids": torch.stack(tids),
            "feat_z": torch.stack(feats), "z30": torch.stack(zs), "user_ids": uids}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    train_rows = SPLIT["train"]["rows"]
    dev_rows = SPLIT["dev"]["rows"]
    log(f"rows: train={len(train_rows)} dev={len(dev_rows)}")

    # syntax features of training/dev queries (spaCy, batched)
    tq = [r.get("query") or r.get("y_plus_query") for r in train_rows]
    dq = [r.get("query") or r.get("y_plus_query") for r in dev_rows]
    tf = query_features_batch(tq)
    df = query_features_batch(dq)
    mu_y = tf.mean(axis=0)
    sd_y = tf.std(axis=0) + 1e-9
    tf_z = zscore(tf, mu_y, sd_y)
    df_z = zscore(df, mu_y, sd_y)
    log(f"query feature stats: mu_y={np.round(mu_y, 3).tolist()}")

    # user vector z-scoring over TRAINING users only (dev/test users excluded
    # from the contour statistics; test users are unseen at this point)
    train_uvec = np.asarray([TRAIN_VECS["vectors"][u]
                             for u in SPLIT["train"]["users"] if u in TRAIN_VECS["vectors"]], dtype=np.float32)
    z_mu = train_uvec.mean(axis=0)
    z_sd = train_uvec.std(axis=0) + 1e-9
    log(f"user contour stats over {len(train_uvec)} train users")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                device_map="cuda:0", trust_remote_code=True)
    H = base.config.hidden_size
    from peft import LoraConfig, get_peft_model
    base = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=LORA_R, lora_alpha=2 * LORA_R,
                                           target_modules=["q_proj", "v_proj"], lora_dropout=0.05))
    base.train()
    projector = SoftPrefixProjector(user_dim=30, hidden_dim=128, num_tokens=NUM_TOKENS,
                                    model_dim=H, dtype=torch.bfloat16, gate_init=GATE_INIT).to("cuda:0")
    copy_head = CopyAwareHead(H, base.get_output_embeddings().weight.size(0), dtype=torch.bfloat16).to("cuda:0")
    syntax_head = SyntaxHead(H).to("cuda:0").to(torch.bfloat16)

    trainable = [p for p in base.parameters() if p.requires_grad] + \
                list(projector.parameters()) + list(copy_head.parameters()) + list(syntax_head.parameters())
    opt = torch.optim.AdamW(trainable, lr=LR)
    log(f"trainable params={sum(p.numel() for p in trainable)}")

    ds = E17Dataset(train_rows, tokenizer, tf_z, TRAIN_VECS["vectors"], z_mu, z_sd)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                    num_workers=2, persistent_workers=False)
    dev_ds = E17Dataset(dev_rows, tokenizer, df_z, TRAIN_VECS["vectors"], z_mu, z_sd)
    dev_dl = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    emb = base.get_input_embeddings()
    metrics = {"epochs": []}
    best_dev, best_epoch = -1, 0

    for epoch in range(EPOCHS):
        base.train()
        ep = {"lm": 0.0, "ptr": 0.0, "syn": 0.0, "align": 0.0, "cf": 0.0}
        for step, batch in enumerate(dl):
            pids, tids = batch["prompt_ids"].cuda(), batch["target_ids"].cuda()
            z20 = batch["z30"].cuda().to(torch.bfloat16)
            feat_z = batch["feat_z"].cuda().to(torch.bfloat16)
            Bn = pids.size(0)
            prefix = projector(z20).view(Bn, NUM_TOKENS, H)
            text_emb = emb(pids)
            target_emb = emb(tids.clamp_min(0))
            full = torch.cat([prefix, text_emb, target_emb], dim=1)
            am = torch.cat([torch.ones((Bn, NUM_TOKENS), dtype=torch.long, device="cuda:0"),
                            torch.ones((Bn, pids.size(1) + tids.size(1)), dtype=torch.long, device="cuda:0")], dim=1)
            src_mask = torch.zeros((Bn, pids.size(1)), dtype=torch.bool, device="cuda:0")
            out = base(inputs_embeds=full, attention_mask=am, use_cache=False, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            src_len, T = pids.size(1), tids.size(1)
            src_hidden = hidden[:, NUM_TOKENS: NUM_TOKENS + src_len]
            gen_hidden = hidden[:, NUM_TOKENS + src_len: NUM_TOKENS + src_len + T]
            p_copy, copy_logits = copy_head(gen_hidden, src_hidden, src_mask, pids)
            mixed = mixed_logits(out.logits[:, NUM_TOKENS + src_len: NUM_TOKENS + src_len + T], p_copy, copy_logits)
            shift_mixed = mixed[:, :-1].reshape(-1, mixed.size(-1))
            shift_labels = tids[:, 1:].reshape(-1)
            lm_loss = F.cross_entropy(shift_mixed, shift_labels, ignore_index=-100)

            gate = (tids != -100).float()
            valid = gate.sum(dim=1) > 1
            h_pool = ((gen_hidden * gate.unsqueeze(-1)).sum(dim=1) / gate.sum(dim=1, keepdim=True).clamp_min(1)).to(torch.bfloat16)
            pred = syntax_head(h_pool[valid]) if valid.sum() > 0 else syntax_head(h_pool[:0])
            syn_loss = F.mse_loss(pred, feat_z[valid]) if valid.sum() > 0 else torch.tensor(0.0, device="cuda:0")
            align_loss = F.mse_loss(pred, z20[valid]) if valid.sum() > 0 else torch.tensor(0.0, device="cuda:0")

            perm = torch.randperm(Bn, device="cuda:0")
            z20_shuf = z20[perm]
            prefix_s = projector(z20_shuf).view(Bn, NUM_TOKENS, H)
            full_s = torch.cat([prefix_s, text_emb, target_emb], dim=1)
            out_s = base(inputs_embeds=full_s, attention_mask=am, use_cache=False, output_hidden_states=True)
            hidden_s = out_s.hidden_states[-1]
            gen_hidden_s = hidden_s[:, NUM_TOKENS + src_len: NUM_TOKENS + src_len + T]
            h_pool_s = ((gen_hidden_s * gate.unsqueeze(-1)).sum(dim=1) / gate.sum(dim=1, keepdim=True).clamp_min(1)).to(torch.bfloat16)
            pred_s = syntax_head(h_pool_s[valid]) if valid.sum() > 0 else syntax_head(h_pool_s[:0])
            align_s = F.mse_loss(pred_s, z20[valid]) if valid.sum() > 0 else torch.tensor(0.0, device="cuda:0")
            cf_loss = F.relu(align_loss - align_s + CF_MARGIN)

            ptr_loss = torch.tensor(0.0, device="cuda:0")
            loss = lm_loss + COPY_LAMBDA * ptr_loss + LAMBDA_SYN * syn_loss + \
                LAMBDA_ALIGN * align_loss + LAMBDA_CF * cf_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep["lm"] += float(lm_loss.detach())
            ep["syn"] += float(syn_loss.detach())
            ep["align"] += float(align_loss.detach())
            ep["cf"] += float(cf_loss.detach())

        # dev: fraction of samples where correct-vector syntax distance <
        # shuffled-vector distance (preregistered)
        base.eval()
        hit, tot = 0, 0
        with torch.no_grad():
            for batch in dev_dl:
                pids, tids = batch["prompt_ids"].cuda(), batch["target_ids"].cuda()
                z20 = batch["z30"].cuda().to(torch.bfloat16)
                Bn = pids.size(0)
                for b in range(Bn):
                    if (tids[b] != -100).sum() < 3:
                        continue
                    zp = projector(z20[b:b + 1]).view(1, NUM_TOKENS, H)
                    full = torch.cat([zp, emb(pids[b:b + 1]), emb(tids[b:b + 1].clamp_min(0))], dim=1)
                    am = torch.ones(full.shape[:2], dtype=torch.long, device="cuda:0")
                    o = base(inputs_embeds=full, attention_mask=am, use_cache=False, output_hidden_states=True)
                    gh = o.hidden_states[-1][0, NUM_TOKENS + pids.size(1):]
                    nv = (tids[b] != -100).float()
                    hpool = ((gh * nv.unsqueeze(-1)).sum(dim=0) / nv.sum().clamp_min(1)).to(torch.bfloat16)
                    pred = syntax_head(hpool[:0]) if False else syntax_head(hpool.unsqueeze(0))[0]
                    d_correct = float(F.mse_loss(pred, z20[b]))
                    j = rng.choice([k for k in range(Bn) if k != b])
                    zj = projector(z20[j:j + 1]).view(1, NUM_TOKENS, H)
                    full_j = torch.cat([zj, emb(pids[b:b + 1]), emb(tids[b:b + 1].clamp_min(0))], dim=1)
                    o_j = base(inputs_embeds=full_j, attention_mask=am, use_cache=False, output_hidden_states=True)
                    gh_j = o_j.hidden_states[-1][0, NUM_TOKENS + pids.size(1):]
                    hpool_j = ((gh_j * nv.unsqueeze(-1)).sum(dim=0) / nv.sum().clamp_min(1)).to(torch.bfloat16)
                    pred_j = syntax_head(hpool_j.unsqueeze(0))[0]
                    d_shuf = float(F.mse_loss(pred_j, z20[b]))
                    if d_correct < d_shuf:
                        hit += 1
                    tot += 1
        frac = hit / max(tot, 1)
        log(f"epoch {epoch + 1}/{EPOCHS} lm={ep['lm'] / max(len(dl), 1):.4f} "
            f"syn={ep['syn'] / max(len(dl), 1):.4f} align={ep['align'] / max(len(dl), 1):.4f} "
            f"cf={ep['cf'] / max(len(dl), 1):.4f} | dev hit={frac:.3f} ({hit}/{tot})")
        metrics["epochs"].append({"epoch": epoch + 1, "lm": ep["lm"] / max(len(dl), 1),
                                  "syn": ep["syn"] / max(len(dl), 1),
                                  "align": ep["align"] / max(len(dl), 1),
                                  "cf": ep["cf"] / max(len(dl), 1), "dev_hit_frac": frac})
        if frac > best_dev:
            best_dev, best_epoch = frac, epoch + 1
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(projector.state_dict(), CKPT_DIR / "projector.pt")
            torch.save(copy_head.state_dict(), CKPT_DIR / "copy_head.pt")
            torch.save(syntax_head.state_dict(), CKPT_DIR / "syntax_head.pt")
            base.save_pretrained(CKPT_DIR / "lora_adapter")
            tokenizer.save_pretrained(CKPT_DIR / "tokenizer")
            json.dump({"epoch": epoch + 1, "dev_hit_frac": frac}, open(CKPT_DIR / "best_epoch.json", "w"))

    metrics["best_epoch"] = best_epoch
    metrics["best_dev_hit_frac"] = best_dev
    metrics["meta"] = {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "lora_r": LORA_R,
                       "num_tokens": NUM_TOKENS, "gate_init": GATE_INIT,
                       "copy_lambda": COPY_LAMBDA, "lambda_syn": LAMBDA_SYN,
                       "lambda_align": LAMBDA_ALIGN, "lambda_cf": LAMBDA_CF, "cf_margin": CF_MARGIN,
                       "base": BASE, "ckpt_dir": str(CKPT_DIR),
                       "dev_rule": "fraction of dev samples with correct-vector syntax distance < shuffled-vector distance (>0.5 baseline)"}
    json.dump(metrics, open(E17 / "e17_train_manifest.json", "w"), indent=1)
    log(f"done. best epoch={best_epoch} dev_hit={best_dev:.3f}")


if __name__ == "__main__":
    main()
