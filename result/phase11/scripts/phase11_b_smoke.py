#!/usr/bin/env python3
"""Phase 11.B smoke: 3 pairs × 4 conds × 3 candidates = 36."""
import sys
sys.path.insert(0, "/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
import phase11_b_gaussian_sampling as m
m.N_PAIRS = 3
m.N_CANDIDATES_PER_COND = 3
m.OUT_GENERATIONS = m.OUT_DIR / "phase11_b_gaussian_samples_smoke.jsonl"
m.OUT_REPORT = m.OUT_DIR / "phase11_b_diversity_report_smoke.json"
m.main()