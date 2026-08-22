#!/usr/bin/env python3
"""Phase 10.19.B smoke test: 5 pairs × 2 rounds × 4 candidates."""
import sys
sys.path.insert(0, "/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
import phase10_19_b_contrastive_rag as m
m.N_PAIRS = 5
m.N_ROUNDS = 2
m.N_CANDIDATES_PER_ROUND = 4
m.OUT_LOG = m.OUT_DIR / "phase10_19_iter_smoke.jsonl"
m.main()