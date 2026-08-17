# E30 — Style-Controlled Query Generation

**GO**: copy-aware Qwen × TinyStyler with K=8 prefix + α=2 steering.

## Final config

| Param | Value |
|-------|-------|
| TinyStyler steering | K_prefix=8, α=2.0 |
| MAX_NEW_TOKENS | 64 (T5 EOS naturally stops at ~25) |
| BATCH_SIZE | 4 (fp16, fits 34GB cgroup) |
| Source query model | copy-aware Qwen2-7B-Instruct +4 soft prefix tokens + copy head |

## Headline metrics (10 users, single product B00ECHYTBI)

| Metric | Value |
|--------|-------|
| n_unique | **9/10** |
| styled spread (cos range) | **0.80** |
| user-style spread (reference) | 0.72 |
| SELF−CROSS style alignment | **+0.109** |
| baseline (zero-style) SELF−CROSS | 0.000 |
| attribute preservation (4 attr) | 2.0/4 (50%) |
| attribute preservation (8 attr, sweet spot) | 3.2/8 (40%) |

## Artifacts (all in /home/wlia0047/hj82_scratch2/wenyu/e29_paper/)

### Pipeline
- `e30_pick_products.py` — Step 1: pick products with X=8/Y=40 eligible users
- `e30_style_emb.py` — Step 2: build user 768-dim AnnaWegmann embeddings
- `e30_gen_source.py` — Step 3: copy-aware Qwen → 100 source queries
- `e30_gen_styled.py` — Step 4: TinyStyler K=8/α=2 → styled queries (default config)

### Steering experiments
- `_steering_diag.py` — diagnostic: user embs discriminative (cos mean 0.77)
- `_steering_gen.py` — α-scaling sweep at K=1
- `_steering_layer.py` — middle-layer activation add (FAILED, washed out)
- `_steering_multi_prefix.py` — K × α sweep, K=8/α=2 wins
- `_steering_length.py` — MAX_NEW_TOKENS × steering, 48 tokens sweet spot
- `_sweep_n_attrs.py` — n_attrs ∈ {5..10}, n=8 best joint score
- `_qwen_rewriter.py` — Qwen few-shot prompting (broken output formats)
- `_qwen_style_steering.py` — Qwen + AnnaWegmann untrained projector (1/10 unique)
- `_style_vec_of_styled.py` — AnnaWegmann re-encoding, spread/alignment validation

### Outputs
- `e30_picked_products.json` — 100 products × 4 attrs
- `e30_picked_products_6attrs.json`, `e30_picked_products_nattrs.json` — extended
- `e30_style_embs.npz` — 10,877 user 768-dim style vectors
- `e30_source_queries.jsonl` — 100 source queries
- `e30_styled_queries.jsonl` — 10 (product, user) styled query records
- `e30_n_attrs_sweep.json` — n_attrs sweep metrics
- `e30_style_vec_of_styled.json` — cos sim matrices for validation

### Logs
- `_steering_multi_prefix.log`, `_steering_length.log`, `_sweep_n_attrs.log`
- `_qwen_rewriter.log`, `_qwen_style_steering.log`
- `_style_vec_of_styled.log`
- `e30_gen_styled_kprefix.log`

## Key conclusions

1. **Multi-token style prefix (K=8)** × **α-scaling (α=2)** is the effective
   TinyStyler steering recipe. Beats single-token α-scaling (8/10 → 9/10 unique)
   and middle-layer activation add (4/10 unique, washed out by T5 layer-norm).
2. **Output length ≥48 tokens** required for style signal to surface.
   cos(styled, user) jumps from +0.37 (16 tok) to +0.66 (64 tok).
3. **n_attrs = 8** is the sweet spot for joint (unique + attr-preserved) score.
4. **Replacing TinyStyler with Qwen is impractical without retraining** the
   projector (1-2 day effort). Stick with TinyStyler.
5. **Style transfer is semantic, not cosmetic**: AnnaWegmann re-encoding of the
   10 styled queries shows spread matching user-style spread (0.80 vs 0.72)
   AND SELF−CROSS gap +0.11 (baseline 0.00). Each query points toward its
   own user's style in embedding space.

## GitLab issue

Issue #25 — full report with all 9 experiments, configs, and example outputs.
