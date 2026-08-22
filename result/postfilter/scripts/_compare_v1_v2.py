#!/usr/bin/env python3
"""Per-case 对比 V1 vs V2 迭代精化结果."""
import json

v1 = {}
with open('/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_refined.jsonl') as f:
    for line in f:
        r = json.loads(line)
        key = (r['user_id'], r['asin'])
        v1[key] = r

v2 = {}
with open('/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_refined_v2.jsonl') as f:
    for line in f:
        r = json.loads(line)
        key = (r['user_id'], r['asin'])
        v2[key] = r

common = set(v1.keys()) & set(v2.keys())
print(f'V1 cases: {len(v1)}, V2 cases: {len(v2)}, Common: {len(common)}')
print()
print(f"{'case':<10} {'V1_init':>8} {'V1_fin':>8} {'V1_Δ%':>6} {'V2_init':>8} {'V2_fin':>8} {'V2_Δ%':>6} {'Δ_pp':>6}")
v1_init_avg, v2_init_avg = [], []
v1_fin_avg, v2_fin_avg = [], []
for key in sorted(common):
    r1 = v1[key]
    r2 = v2[key]
    init1 = r1['trace'][0]['mahalanobis']
    init2 = r2['trace'][0]['mahalanobis']
    fin1 = r1['best_dist']
    fin2 = r2['best_dist']
    pct1 = (init1-fin1)/init1*100
    pct2 = (init2-fin2)/init2*100
    pp = pct2 - pct1  # V2 比 V1 多改善多少个百分点
    marker = " ◀" if fin2 < fin1 else (" " if fin2 == fin1 else "")
    print(f"{key[0][:8]:<10} {init1:>8.1f} {fin1:>8.1f} {pct1:>5.1f}% {init2:>8.1f} {fin2:>8.1f} {pct2:>5.1f}% {pp:>+5.1f}{marker}")
    v1_init_avg.append(init1)
    v2_init_avg.append(init2)
    v1_fin_avg.append(fin1)
    v2_fin_avg.append(fin2)

print()
print(f'V1 mean initial: {sum(v1_init_avg)/len(v1_init_avg):.1f}, V1 mean final: {sum(v1_fin_avg)/len(v1_fin_avg):.1f}, V1 mean Δ%: {(sum(v1_init_avg)-sum(v1_fin_avg))/sum(v1_init_avg)*100:.1f}%')
print(f'V2 mean initial: {sum(v2_init_avg)/len(v2_init_avg):.1f}, V2 mean final: {sum(v2_fin_avg)/len(v2_fin_avg):.1f}, V2 mean Δ%: {(sum(v2_init_avg)-sum(v2_fin_avg))/sum(v2_init_avg)*100:.1f}%')
print()
print(f'V2 < V1 (better): {sum(1 for a, b in zip(v1_fin_avg, v2_fin_avg) if b < a)}/{len(common)}')
print(f'V2 == V1: {sum(1 for a, b in zip(v1_fin_avg, v2_fin_avg) if b == a)}/{len(common)}')
print(f'V2 > V1 (worse): {sum(1 for a, b in zip(v1_fin_avg, v2_fin_avg) if b > a)}/{len(common)}')