# E30 — Style-Controlled Query Generation

End-to-end pipeline: copy-aware Qwen (content) × TinyStyler (style).

## Pipeline (run in order)

```bash
# Step 1: pick products with X=8/Y=40 eligible users (100 products)
python pick_products.py

# Step 2: build per-user 768-dim AnnaWegmann style embeddings
python style_emb.py

# Step 3: copy-aware Qwen → source queries (100 products)
python gen_source.py

# Step 4: TinyStyler K=8/α=2 → styled queries (10 users/product)
python gen_styled.py
```

## Sweep / exploratory scripts

These explore the TinyStyler steering space and validate the final config:

| Script | Purpose | Key finding |
|--------|---------|-------------|
| `steering_diag.py` | Check if user style embs are discriminative | cos mean=0.77, range=0.25-0.97 ✓ |
| `steering_gen.py` | α-scaling sweep at K=1 | α=2 → 8/10 unique (saturates) |
| `steering_layer.py` | Middle-layer activation steering | FAILED, T5 layer-norm washes out |
| `steering_multi_prefix.py` | K ∈ {1,2,4,8,16} × α sweep | **K=8, α=2 → 9/10 unique** |
| `steering_length.py` | MAX_NEW_TOKENS sweep | 48 tokens = sweet spot |
| `sweep_n_attrs.py` | n_attrs ∈ {5..10} | n=8 best joint (unique + attrs) |
| `qwen_rewriter.py` | Qwen few-shot prompting | broken output formats |
| `qwen_style_steering.py` | Qwen + untrained AnnaWegmann projector | 1/10 unique (zero prefix) |
| `style_vec_of_styled.py` | Re-encode styled queries, check spread | +0.11 SELF−CROSS gap ✓ |

## Steering config (production)

`gen_styled.py` defaults:
- `STYLE_K_PREFIX = 8` (multi-token prefix length)
- `STYLE_ALPHA = 2.0` (α-scaling factor)
- `MAX_NEW_TOKENS = 64` (T5 EOS naturally stops at ~25)
- `BATCH_SIZE = 4` (fp16, fits 34GB cgroup)
- `NUM_BEAMS = 2`

## Artifacts

All outputs go to `/home/wlia0047/hj82_scratch2/wenyu/e29_paper/` (per
CLAUDE.md Rule 10 — outputs to scratch2, not /fs04).

| File | What |
|------|------|
| `e30_picked_products.json` | 100 products × 4 attrs (X=8/Y=40 cell) |
| `e30_picked_products_6attrs.json`, `e30_picked_products_nattrs.json` | Extended (5-10 attrs) |
| `e30_style_embs.npz` | 10,877 user 768-dim AnnaWegmann embeddings |
| `e30_source_queries.jsonl` | 100 source queries from copy-aware Qwen |
| `e30_styled_queries.jsonl` | 10 (product, user) styled query records |
| `e30_n_attrs_sweep.json` | n_attrs sweep metrics |
| `e30_style_vec_of_styled.json` | cos sim matrices for spread/alignment validation |
| `*.log` | Per-step run logs |

## Key Results (GO)

| Metric | Value |
|--------|-------|
| n_unique (10 users) | **9/10** |
| styled spread (cos range) | **0.80** |
| user-style spread (reference) | 0.72 |
| SELF−CROSS style alignment | **+0.109** |
| baseline (zero-style) SELF−CROSS | 0.000 |
| attribute preservation (4 attr) | 2.0/4 (50%) |
| attribute preservation (8 attr, sweet spot) | 3.2/8 (40%) |

See `result/e30_summary.md` for the full summary and `issue #25` on GitLab
for the complete report.
