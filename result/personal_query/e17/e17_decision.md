# E17 Issue 17 Decision

## Per-Check Results

| Check | GO/NO-GO | Notes |
|---|---|---|
| Check-1 | ✅ GO | n_users=400; vector_hash=f46e2b505331677fa64318c74d967a06567734bc317e573cbdd0e52911530c92; boot95_CI=[0.2192, 0.321]; permutation_p=0.001; leakage_guard={'train_users_excluded': True, 'n_train_users': |
| Check-2 | ✅ GO | spec_match=True; mechanism_active=True; output_effect=True |
| Check-3 | ✅ GO | baseline_hit=1.0; shuffled_hit=0.6025641025641025; alpha_zero_hit=0.0 |
| Check-4 | ❌ NO-GO | n_test_users=100; n_pairs_complete=299; auc=0.4796196868008949; swap_rate=1.0; monotonicity_dims_ok=4 |
| Check-5 | ❌ NO-GO | per_condition_five_attr={'correct': 0.23411371237458195, 'shuffled': 0.17725752508361203, 'global_mean': 0.5551839464882943, 'zero': 0.5551839464882943}; consistency_rate=0.09364548494983277 |
| Check-6 | ✅ GO | artifacts_present=13; artifacts_missing=0; preregister_present=True |

## Final Decision

**NO-GO**: Not all six checks pass. See the per-check evidence above for blockers.

## Artifacts

- `e17_style_vector_contract.json`: `3c8c9b8988c2e646...` (1557 bytes)
- `e17_split_manifest.json`: `4c919dc7eadb6a6b...` (44196 bytes)
- `e17_style_vectors.json`: `4bb41d22bca4dfff...` (219202 bytes)
- `e17_train_vectors.json`: `383edd4d0b680afe...` (114362 bytes)
- `e17_train_dev_split.json`: `d2a452f00205f844...` (667439 bytes)
- `e17_injection_contract.json`: `63b03cc451e56172...` (1942 bytes)
- `e17_train_manifest.json`: `f70f07ffacc7ec88...` (1962 bytes)
- `e17_ablations.json`: `245ac1a36cbe58aa...` (527 bytes)
- `e17_test_samples.jsonl`: `f1d00fbd3bcc1616...` (519279 bytes)
- `e17_step4_meta.json`: `cc3be62a95563c13...` (491 bytes)
- `e17_style_eval.json`: `3183c72a31428806...` (2309 bytes)
- `e17_content_eval.json`: `5459e2f5bf30dca3...` (6598 bytes)
- `e17_hash_manifest.json`: `b24f7943bf59a58f...` (1796 bytes)
