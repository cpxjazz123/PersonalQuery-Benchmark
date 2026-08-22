#!/usr/bin/env python3
"""Phase 10.18.B smoke test: 30 pairs × 5 rounds × 8 candidates."""
import sys
sys.path.insert(0, "/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
import phase10_18_b_iter_search as m
m.N_PAIRS = 30
m.N_ROUNDS = 5
m.N_CANDIDATES_PER_ROUND = 8
m.OUT_LOG = m.OUT_DIR / "phase10_18_iter_smoke.jsonl"
m.SUMMARY_OUT = m.OUT_DIR / "phase10_18_iter_smoke_summary.json"
m.main()
