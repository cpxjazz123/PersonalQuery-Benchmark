#!/usr/bin/env python3
"""V1 vs V2 vs V3 迭代精化对比（固定预算 40 calls/case 对齐）.

V1: 单路径 5 steps, ~5 calls/case
V2: beam 2x2, T=0.3, 10 steps, 0~40 calls/case (无 early stop / restart)
V3: beam 2x2, T schedule (0.7<3步, 0.3>=3), early stop (patience=3),
    restart (interval=4), 预算 40 calls/case
"""
import json

V1_FILE = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_refined.jsonl'
V2_FILE = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_refined_v2.jsonl'
V3_FILE = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_refined_v3.jsonl'
V1_SUMMARY = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_summary.json'
V2_SUMMARY = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_summary_v2.json'
V3_SUMMARY = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/iterative_summary_v3.json'
OUT = '/home/wlia0047/hj82_scratch2/wenyu/postfilter/v1_v2_v3_comparison.json'


def load(path: str) -> dict[tuple[str, str], dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r['user_id'], r['asin'])] = r
    return out


def get_initial(r: dict) -> float:
    """优先用 r['initial_dist'] (V3), 否则用 trace[0]['mahalanobis'] (V1/V2)."""
    if 'initial_dist' in r:
        return r['initial_dist']
    return r['trace'][0]['mahalanobis']


def main():
    v1 = load(V1_FILE)
    v2 = load(V2_FILE)
    v3 = load(V3_FILE)
    common = set(v1.keys()) & set(v2.keys()) & set(v3.keys())
    print(f'V1={len(v1)}, V2={len(v2)}, V3={len(v3)}, Common={len(common)}')
    print()

    # 表头
    print(f"{'case':<10} "
          f"{'V1_init':>8} {'V1_fin':>8} {'V1_Δ%':>7} {'V1_calls':>9}  "
          f"{'V2_init':>8} {'V2_fin':>8} {'V2_Δ%':>7} {'V2_calls':>9}  "
          f"{'V3_init':>8} {'V3_fin':>8} {'V3_Δ%':>7} {'V3_calls':>9}  "
          f"{'best':>5}")
    rows = []
    for key in sorted(common):
        r1, r2, r3 = v1[key], v2[key], v3[key]
        ini1, fin1 = get_initial(r1), r1['best_dist']
        ini2, fin2 = get_initial(r2), r2['best_dist']
        ini3, fin3 = get_initial(r3), r3['best_dist']
        n1 = r1['trace'][-1]['step'] + 1  # V1 用 step 计数
        n2 = r2['trace'][-1]['step'] + 1
        n3 = r3.get('n_llm_calls', r3['trace'][-1]['step'] * 4 + 1)
        p1 = (ini1 - fin1) / ini1 * 100
        p2 = (ini2 - fin2) / ini2 * 100
        p3 = (ini3 - fin3) / ini3 * 100
        best = min([('V1', fin1), ('V2', fin2), ('V3', fin3)], key=lambda x: x[1])[0]
        print(f"{key[0][:8]:<10} "
              f"{ini1:>8.1f} {fin1:>8.1f} {p1:>6.1f}% {n1:>9}  "
              f"{ini2:>8.1f} {fin2:>8.1f} {p2:>6.1f}% {n2:>9}  "
              f"{ini3:>8.1f} {fin3:>8.1f} {p3:>6.1f}% {n3:>9}  "
              f"{best:>5}")
        rows.append({
            'key': key, 'init': (ini1 + ini2 + ini3) / 3,
            'v1': fin1, 'v2': fin2, 'v3': fin3,
            'v1_pct': p1, 'v2_pct': p2, 'v3_pct': p3,
            'v1_calls': n1, 'v2_calls': n2, 'v3_calls': n3,
            'best': best,
        })

    # 汇总
    print()
    print("=" * 100)
    print("汇总 (均值 over common cases)")
    print("=" * 100)
    mean_init = sum(r['init'] for r in rows) / len(rows)
    mean_v1 = sum(r['v1'] for r in rows) / len(rows)
    mean_v2 = sum(r['v2'] for r in rows) / len(rows)
    mean_v3 = sum(r['v3'] for r in rows) / len(rows)
    pct_v1 = (mean_init - mean_v1) / mean_init * 100
    pct_v2 = (mean_init - mean_v2) / mean_init * 100
    pct_v3 = (mean_init - mean_v3) / mean_init * 100
    print(f"  mean initial: {mean_init:.1f}")
    print(f"  V1 mean final: {mean_v1:.1f}  Δ% = {pct_v1:+.2f}%")
    print(f"  V2 mean final: {mean_v2:.1f}  Δ% = {pct_v2:+.2f}%")
    print(f"  V3 mean final: {mean_v3:.1f}  Δ% = {pct_v3:+.2f}%")
    print()
    print(f"  per-case winner:")
    from collections import Counter
    cnt = Counter(r['best'] for r in rows)
    for name, n in sorted(cnt.items()):
        print(f"    {name}: {n}/{len(rows)}")
    print()
    print(f"  Per-method LLM calls (avg):")
    print(f"    V1: {sum(r['v1_calls'] for r in rows)/len(rows):.1f}")
    print(f"    V2: {sum(r['v2_calls'] for r in rows)/len(rows):.1f}")
    print(f"    V3: {sum(r['v3_calls'] for r in rows)/len(rows):.1f}")
    print()
    print(f"  V3 < V2 (better final): {sum(1 for r in rows if r['v3'] < r['v2'])}/{len(rows)}")
    print(f"  V3 == V2: {sum(1 for r in rows if r['v3'] == r['v2'])}/{len(rows)}")
    print(f"  V3 > V2 (worse final):  {sum(1 for r in rows if r['v3'] > r['v2'])}/{len(rows)}")
    print()
    print(f"  V3 < V1 (better final): {sum(1 for r in rows if r['v3'] < r['v1'])}/{len(rows)}")
    print(f"  V3 == V1: {sum(1 for r in rows if r['v3'] == r['v1'])}/{len(rows)}")
    print(f"  V3 > V1 (worse final):  {sum(1 for r in rows if r['v3'] > r['v1'])}/{len(rows)}")

    # 写 JSON
    summary = {
        'n_common': len(rows),
        'mean_initial_dist': mean_init,
        'v1_mean_final_dist': mean_v1,
        'v2_mean_final_dist': mean_v2,
        'v3_mean_final_dist': mean_v3,
        'v1_improvement_pct': pct_v1,
        'v2_improvement_pct': pct_v2,
        'v3_improvement_pct': pct_v3,
        'v1_mean_calls': sum(r['v1_calls'] for r in rows) / len(rows),
        'v2_mean_calls': sum(r['v2_calls'] for r in rows) / len(rows),
        'v3_mean_calls': sum(r['v3_calls'] for r in rows) / len(rows),
        'per_case_winner': dict(cnt),
        'v3_vs_v2': {
            'better': sum(1 for r in rows if r['v3'] < r['v2']),
            'equal': sum(1 for r in rows if r['v3'] == r['v2']),
            'worse': sum(1 for r in rows if r['v3'] > r['v2']),
        },
        'v3_vs_v1': {
            'better': sum(1 for r in rows if r['v3'] < r['v1']),
            'equal': sum(1 for r in rows if r['v3'] == r['v1']),
            'worse': sum(1 for r in rows if r['v3'] > r['v1']),
        },
    }
    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
