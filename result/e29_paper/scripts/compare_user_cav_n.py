"""Compare N=100 vs N=1000 vs N=10000 user-level CAV gap."""
import json
from pathlib import Path

OUT_DIR = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper')
r100 = json.load(open(OUT_DIR / 'user_level_cav.json'))
r1000 = json.load(open(OUT_DIR / 'user_level_cav_1000.json'))
r10000 = json.load(open(OUT_DIR / 'user_level_cav_10000.json'))


def row(key, d):
    X, Y = key.split('_')
    X, Y = int(X), int(Y)
    row = {'X': X, 'Y': Y}
    for src, r in (('n100', r100), ('n1000', r1000), ('n10000', r10000)):
        v = r.get(key)
        if v and v.get('n_sampled'):
            row[f'{src}_gap'] = v['gap']
            row[f'{src}_auc'] = v['auc']
            row[f'{src}_n'] = v['n_sampled']
            row[f'{src}_self'] = v['self_sim']
            row[f'{src}_cross'] = v['cross_sim']
        else:
            row[f'{src}_gap'] = None
    return row


# Use only cells that appear in all 3 (n10000 has 8 cells)
common = []
for key in r10000:
    if r10000[key].get('n_sampled'):
        common.append(row(key, None))

# Sort by X then Y
common.sort(key=lambda r: (r['X'], r['Y']))

# Print as table
print(f"{'X':>3} {'Y':>3} | {'N=100':^22} | {'N=1000':^22} | {'N=10000':^22} | Δ(10k-100):>9")
print(f"{'':>3} {'':>3} | {'gap':>6} {'auc':>6} {'self':>5} {'cross':>5} | {'gap':>6} {'auc':>6} {'self':>5} {'cross':>5} | {'gap':>6} {'auc':>6} {'self':>5} {'cross':>5} |")
print('-' * 105)
for r in common:
    def fmt(r, suffix):
        g = r[f'{suffix}_gap']
        a = r[f'{suffix}_auc']
        s = r[f'{suffix}_self']
        c = r[f'{suffix}_cross']
        if g is None:
            return '   --   --   --   --'
        return f'{g:+.3f} {a:.3f} {s:.3f} {c:.3f}'
    line = f"{r['X']:>3} {r['Y']:>3} | {fmt(r,'n100')} | {fmt(r,'n1000')} | {fmt(r,'n10000')} |"
    delta = (r['n10000_gap'] or 0) - (r['n100_gap'] or 0)
    line += f" {delta:+.3f}"
    print(line)

# Save as JSON
with open(OUT_DIR / 'compare_user_cav_n.json', 'w') as f:
    json.dump(common, f, indent=2)
print(f"\nsaved {OUT_DIR / 'compare_user_cav_n.json'}")