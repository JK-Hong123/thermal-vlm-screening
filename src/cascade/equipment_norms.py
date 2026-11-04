# -*- coding: utf-8 -*-
"""equipment_norms.py — per-equipment NORMAL-frame distributions from the train split (feat_cache).
Gives every case a grounded 'vs normal population of the same equipment' fact (percentile of t_max / dT_scene / dT_sib).
Output: results/equipment_norms.json"""
import numpy as np, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import FEAT, FEAT_NAMES
Z = np.load(FEAT, allow_pickle=True); X, Y, FAC, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['GRP']
FN = list(Z['FEAT_NAMES']); tr = GRP == 'train'
Q = [5, 25, 50, 75, 90, 95, 99]
norms = {}
for f in sorted(set(FAC[tr])):
    m = tr & (FAC == f); mn = m & (Y == 0); md = m & (Y == 1)
    d = {'n_normal': int(mn.sum()), 'n_danger': int(md.sum())}
    for k in ('t_max', 'dT_scene_max', 'dT_sib_max', 't_mean'):
        v = X[mn, FN.index(k)]; vd = X[md, FN.index(k)]
        d[k] = {'normal_q': {str(q): round(float(np.percentile(v, q)), 1) for q in Q}, 'normal_sorted': None,
                'danger_q': {str(q): round(float(np.percentile(vd, q)), 1) for q in Q} if md.sum() > 5 else None}
    norms[f] = d
# store sorted normal t_max / dT arrays compactly (for percentile lookup at case time) — subsample to <=2000
for f in norms:
    m = tr & (FAC == f) & (Y == 0)
    for k in ('t_max', 'dT_scene_max', 'dT_sib_max'):
        v = np.sort(X[m, FN.index(k)]); step = max(1, len(v)//2000)
        norms[f][k]['normal_sorted'] = [round(float(x), 2) for x in v[::step]]
json.dump(norms, open(r"<PROJECT_ROOT>\results\equipment_norms.json", 'w', encoding='utf-8'), ensure_ascii=False)
print(f"{'equipment':24s} {'nN':>6s} {'nD':>5s} | t_max normal p50/p95/p99 | danger p50 || dT_scene normal p50/p95/p99 | danger p50")
for f, d in norms.items():
    t = d['t_max']; s = d['dT_scene_max']
    print(f"{f:24s} {d['n_normal']:6d} {d['n_danger']:5d} | {t['normal_q']['50']:5.1f}/{t['normal_q']['95']:5.1f}/{t['normal_q']['99']:5.1f} | {t['danger_q']['50'] if t['danger_q'] else float('nan'):5.1f} || {s['normal_q']['50']:4.1f}/{s['normal_q']['95']:4.1f}/{s['normal_q']['99']:4.1f} | {s['danger_q']['50'] if s['danger_q'] else float('nan'):4.1f}")
