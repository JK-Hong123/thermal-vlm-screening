# -*- coding: utf-8 -*-
"""headline_ci.py — session-clustered bootstrap 95% CIs for every headline number that did not have one yet:
  (1) blinded judge preferences (dev 44, lock 34, lora 32) — clustered by case (one judgment per case)
  (2) hotspot-localisation κ / within-0.2 for H and the brightest-pixel heuristic on the independent 388-bbox set — clustered by session
  (3) review-queue recovery on gen_test — clustered by session (from feat_cache)
Writes results/headline_ci.txt"""
import json, os, sys, io, collections, re
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
R = r"<PROJECT_ROOT>\results"; rng = np.random.default_rng(0); B = 2000
def boot(groups, stat):
    """groups: dict cluster -> list of items; stat(list_of_items) -> float"""
    keys = list(groups); vals = []
    for _ in range(B):
        pick = rng.choice(keys, len(keys), replace=True); items = [x for k in pick for x in groups[k]]
        v = stat(items)
        if v == v: vals.append(v)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
def kappa(pairs):
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')
out = []
# (1) judge preferences
for name, fn, key in (('judge dev (VLM vs A0+)', 'judge_dev.json', None), ('judge locked (VLM vs A0+)', 'judge_lock.json', None), ('judge locked (LoRA-VLM vs A0+)', 'judge_lora.json', None)):
    J = json.load(open(os.path.join(R, fn), encoding='utf-8'))
    for j in J:
        if 'A' not in j:
            i = int(j['case_id'][4:]); i = (i-1)//2 if fn == 'judge_lora.json' else i; j['A'] = 'a0plus' if i % 2 else 'v2'; j['B'] = 'v2' if i % 2 else 'a0plus'
    g = lambda j, k, arm: j[k+'_A_1to5'] if j['A'] == arm else j[k+'_B_1to5']
    grp = {j['case_id']: [j] for j in J}
    pv = lambda items: np.mean([(j['A'] if j['preferred_report'] == 'A' else j['B']) == 'v2' and j['preferred_report'] != 'tie' for j in items])
    du = lambda items: np.mean([g(j, 'usefulness', 'a0plus') - g(j, 'usefulness', 'v2') for j in items])
    lo, hi = boot(grp, pv); lo2, hi2 = boot(grp, du)
    out.append(f"{name:34s} n={len(J):3d} | VLM preferred {pv(J):.3f} [{lo:.3f},{hi:.3f}] | usefulness A0+ minus VLM {du(J):+.2f} [{lo2:+.2f},{hi2:+.2f}]")
# (2) localisation on the independent set: need per-row predictions -> recompute quickly from saved eval + heuristic
ev = [json.loads(l) for l in open(os.path.join(R, 'eval_v3_big', 'eval.jsonl'), encoding='utf-8')]
# per-row model predictions are not stored; re-derive from result json is aggregate only -> report that CI requires re-run; instead bootstrap the heuristic (recomputable) and note.
out.append("(localisation CIs: per-row model predictions not cached in eval_v3_big; rerun eval_v3_big.py with --dump to add — see TODO)")
# (3) review queue on gen_test, session bootstrap
from common import FEAT, load_rf, onehot
Z = np.load(FEAT, allow_pickle=True); X, Y, FAC, SES, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['SES'], Z['GRP']; FN = list(Z['FEAT_NAMES']); i_ts = FN.index('dT_scene_max')
rf, vocab = load_rf()
for grp_name in ('test', 'gen_test'):
    m = GRP == grp_name; P_ = rf.predict_proba(np.hstack([X[m], onehot(FAC[m], vocab)]))[:, 1]; y = Y[m]; s = SES[m]; dts = X[m, i_ts]
    items = list(zip(P_, y, s, dts)); groups = collections.defaultdict(list)
    for it in items: groups[it[2]].append(it)
    def rec(items):
        rv = [(p >= 0.35) & (p < 0.5) & (d > 5.5) for p, _, _, d in items]; yy = np.array([t for _, t, _, _ in items]); rv = np.array(rv)
        return (yy[rv].sum()/max(yy.sum(), 1))
    def prec(items):
        rv = np.array([(p >= 0.35) & (p < 0.5) & (d > 5.5) for p, _, _, d in items]); yy = np.array([t for _, t, _, _ in items]); return yy[rv].mean() if rv.sum() else float('nan')
    def flag_rec(items):
        fl = np.array([p >= 0.5 for p, _, _, _ in items]); yy = np.array([t for _, t, _, _ in items]); return yy[fl].sum()/max(yy.sum(), 1)
    lo, hi = boot(groups, rec); lo2, hi2 = boot(groups, prec); lo3, hi3 = boot(groups, flag_rec)
    out.append(f"review queue {grp_name:8s} | danger recall by queue {rec(items):.3f} [{lo:.3f},{hi:.3f}] | queue precision {prec(items):.3f} [{lo2:.3f},{hi2:.3f}] | gate(flag) recall {flag_rec(items):.3f} [{lo3:.3f},{hi3:.3f}]")
txt = "\n".join(out); print(txt); open(os.path.join(R, 'headline_ci.txt'), 'w', encoding='utf-8').write(txt)
