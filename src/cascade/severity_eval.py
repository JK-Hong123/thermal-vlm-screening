# -*- coding: utf-8 -*-
"""severity_eval.py — E8: validity of the CODE-OWNED severity layer on the full held-out bbox set (feat_cache test + gen_test),
without any VLM. For each bbox: gate (RF prob), S/A/P axes -> final level; then per equipment: danger rate by level (monotonic?),
L>=2 vs danger precision/recall, axis usage, conflicts. Also the review-zone recovery of RF false negatives."""
import numpy as np, json, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import FEAT, load_rf, onehot
from protocol_v1 import ELEC_CONN, BULK, level_sibling, level_scene, level_population
NORMS = json.load(open(r"<PROJECT_ROOT>\results\equipment_norms.json", encoding='utf-8'))
Z = np.load(FEAT, allow_pickle=True); X, Y, FAC, SES, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['SES'], Z['GRP']
FN = list(Z['FEAT_NAMES']); i_ts, i_tm, i_sib, i_sm = FN.index('dT_scene_max'), FN.index('t_max'), FN.index('dT_sib_max'), FN.index('scene_med')
rf, vocab = load_rf()
def pct_of(sv, v): a = np.asarray(sv); return 100*float((a < v).mean())
for grp in ('test', 'gen_test'):
    m = GRP == grp; Xg = X[m]; Yg = Y[m]; Fg = FAC[m]
    P = rf.predict_proba(np.hstack([Xg, onehot(Fg, vocab)]))[:, 1]
    levels = np.zeros(len(Yg), int); axis = []; zone = []
    for k in range(len(Yg)):
        eq = Fg[k]; dts = Xg[k, i_ts]; dsib = Xg[k, i_sib]; tm = Xg[k, i_tm]
        # dT_sib_max == dT_scene when no sibling (sib_ref=scene median) -> treat equality as 'no sibling'
        has_sib = abs(dsib - dts) > 1e-6
        scene_lvl = level_scene(dts)
        if eq in BULK and eq in NORMS:
            pct = pct_of(NORMS[eq]['t_max']['normal_sorted'], tm); dp = pct_of(NORMS[eq]['dT_scene_max']['normal_sorted'], dts)
            base = max(level_population(pct, dp), scene_lvl - 1); ax = 'P'
        elif eq in ELEC_CONN and has_sib: base = level_sibling(dsib); ax = 'S'
        else: base = scene_lvl; ax = 'A'
        z = 'flag' if P[k] >= 0.5 else ('review' if (P[k] >= 0.35 and dts > 5.5) else 'pass')
        levels[k] = max(1, base) if z == 'flag' else 0; axis.append(ax); zone.append(z)
    axis = np.array(axis); zone = np.array(zone)
    print(f"\n===== {grp}: bboxes {len(Yg)}, danger {Yg.sum()} =====")
    print("  zone sizes:", dict(collections.Counter(zone)), "| danger recall by zone:", {z: f"{Yg[zone==z].sum()}/{Yg.sum()}" for z in ('flag','review','pass')})
    print("  axis usage:", dict(collections.Counter(axis)))
    print("  danger rate by final level:", {int(l): f"{Yg[levels==l].mean():.2f} (n={int((levels==l).sum())})" for l in sorted(set(levels))})
    pred = levels >= 2; tp = (pred & (Yg == 1)).sum(); print(f"  L>=2 vs danger: precision {tp/max(pred.sum(),1):.2f}, recall {tp/max(Yg.sum(),1):.2f} | flagged(L>=1) precision {Yg[levels>=1].mean():.2f} recall {Yg[levels>=1].sum()/Yg.sum():.2f}")
    print("  per-equipment danger rate by level (n>=20 equipment):")
    for eq in sorted(set(Fg)):
        mm = Fg == eq
        if mm.sum() < 20: continue
        rates = []
        for l in range(5):
            sel = mm & (levels == l); rates.append(f"L{l}:{Yg[sel].mean():.2f}({sel.sum()})" if sel.sum() >= 5 else f"L{l}:-({sel.sum()})")
        vals = [Yg[mm & (levels == l)].mean() for l in range(5) if (mm & (levels == l)).sum() >= 5]
        mono = all(vals[i] <= vals[i+1] + 0.05 for i in range(len(vals)-1)) if len(vals) >= 2 else None
        print(f"    {eq:22s} axis={collections.Counter(axis[mm]).most_common(1)[0][0]} {' '.join(rates)} monotone={mono}")
