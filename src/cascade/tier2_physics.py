# -*- coding: utf-8 -*-
"""tier2_physics.py — physics-side analyses requested by the simulated review (all from feat_cache + manifest, no GPU):
 1. per-equipment threshold rule (one scene-rise threshold per equipment class, chosen on training sessions) and a
    per-equipment percentile rule, as auditable comparators to the random forest;
 2. leave-one-equipment-out (LOEO): for each of the 16 classes, RF trained on the other classes' training rows and
    evaluated on every row of the held-out class, against the global rule on the same rows;
 3. prevalence and operating point (sensitivity / specificity / precision / flag rate) for the rule at 5.5 C, the RF at
    posterior 0.5 and the review zone, plus alarms per session;
 4. camera/scene leakage audit: session-id prefix shared between training and held-out sessions;
 5. scene median versus the camera's entered atmospheric temperature, per site;
 6. session-level evidence for subtle faults (faulty boxes < 4 C above the scene median).
Writes results/tier2_physics.txt and results/tier2_loeo.json.
"""
import numpy as np, json, sys, io, collections, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from sklearn.ensemble import RandomForestClassifier
from common import FEAT, MAN, onehot
R = r"<PROJECT_ROOT>\results"
Z = np.load(FEAT, allow_pickle=True); X, Y, FAC, SES, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['SES'], Z['GRP']
FN = list(Z['FEAT_NAMES']); print("features:", FN)
i_dT = FN.index('dT_scene_max'); i_tmax = FN.index('t_max') if 't_max' in FN else None
tr, te, ge = GRP == 'train', GRP == 'test', GRP == 'gen_test'; vocab = sorted(set(FAC[tr]))
dT = X[:, i_dT]
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); o = s.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(s)+1)
    p = y.sum(); n = len(y)-p; return float('nan') if p == 0 or n == 0 else float((r[y == 1].sum()-p*(p+1)/2)/(p*n))
def ci(s, y, ses, B=600):
    uq = np.unique(ses); rng = np.random.default_rng(0); idxby = {u: np.where(ses == u)[0] for u in uq}; v = []
    for _ in range(B):
        idx = np.concatenate([idxby[u] for u in rng.choice(uq, len(uq), replace=True)]); a = auroc(s[idx], y[idx])
        if a == a: v.append(a)
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (float('nan'), float('nan'))
def fmt(s, y, ses): a = auroc(s, y); lo, hi = ci(s, y, ses); return f"{a:.3f} [{lo:.3f},{hi:.3f}]"
out = []
# ---------- 1. per-equipment rules ----------
thr = {}
for e in vocab:
    m = tr & (FAC == e); y = Y[m]; d = dT[m]
    if y.sum() == 0 or (1-y).sum() == 0: thr[e] = 5.5; continue
    cands = np.unique(np.round(d, 1)); best = (-1, 5.5)
    for c in cands:
        pred = d > c; tpr = (pred & (y == 1)).sum()/y.sum(); tnr = (~pred & (y == 0)).sum()/(1-y).sum(); j = tpr + tnr - 1
        if j > best[0]: best = (j, float(c))
    thr[e] = best[1]
tau = np.array([thr.get(e, 5.5) for e in FAC]); s_eq = dT - tau            # per-equipment threshold rule score
# per-equipment percentile rule: percentile of dT within the training NORMAL distribution of that equipment
pct = np.zeros(len(dT), float)
for e in vocab:
    ref = np.sort(dT[tr & (FAC == e) & (Y == 0)])
    m = FAC == e
    pct[m] = np.searchsorted(ref, dT[m], side='right') / max(len(ref), 1)
pct[~np.isin(FAC, vocab)] = np.nan
out.append("=== 1. auditable comparators (AUROC, session-clustered bootstrap 95% CI) ===")
out.append(f"global rule dT_scene > 5.5        within-test {fmt(dT[te], Y[te], SES[te])} | unseen equip {fmt(dT[ge], Y[ge], SES[ge])}")
out.append(f"per-equipment threshold rule       within-test {fmt(s_eq[te], Y[te], SES[te])} | unseen equip (falls back to 5.5) {fmt(s_eq[ge], Y[ge], SES[ge])}")
out.append(f"per-equipment normal-percentile    within-test {fmt(pct[te], Y[te], SES[te])} | unseen equip: n/a (no equipment history)")
out.append("per-equipment thresholds (C): " + ", ".join(f"{e}={thr[e]:.1f}" for e in vocab))
# ---------- 2. LOEO ----------
out.append(""); out.append("=== 2. leave-one-equipment-out: RF trained on the other classes' training rows, evaluated on every row of the held-out class ===")
loeo = {}
for e in sorted(set(FAC)):
    m_eval = FAC == e; m_train = tr & (FAC != e)
    if Y[m_eval].sum() == 0 or (1-Y[m_eval]).sum() == 0: out.append(f"  {e}: single-class, skipped"); continue
    voc = sorted(set(FAC[m_train]))
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight='balanced', n_jobs=-1, random_state=0)
    rf.fit(np.hstack([X[m_train], onehot(FAC[m_train], voc)]), Y[m_train])
    p = rf.predict_proba(np.hstack([X[m_eval], onehot(FAC[m_eval], voc)]))[:, 1]
    a_rf, a_rule = fmt(p, Y[m_eval], SES[m_eval]), fmt(dT[m_eval], Y[m_eval], SES[m_eval])
    loeo[e] = dict(n=int(m_eval.sum()), sessions=int(len(np.unique(SES[m_eval]))), prevalence=float(Y[m_eval].mean()), rf=a_rf, rule=a_rule, rf_point=auroc(p, Y[m_eval]), rule_point=auroc(dT[m_eval], Y[m_eval]))
    out.append(f"  {e:28s} n={m_eval.sum():5d} sess={loeo[e]['sessions']:3d} prev={Y[m_eval].mean():.2f} | RF {a_rf} | rule {a_rule}")
    print(out[-1])
wins = sum(1 for v in loeo.values() if v['rf_point'] > v['rule_point']); out.append(f"  RF above rule in {wins}/{len(loeo)} held-out classes; mean AUROC RF {np.mean([v['rf_point'] for v in loeo.values()]):.3f} vs rule {np.mean([v['rule_point'] for v in loeo.values()]):.3f}")
json.dump(loeo, open(os.path.join(R, 'tier2_loeo.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
# ---------- 3. prevalence & operating point ----------
import joblib
from common import RF_PATH
rfm = joblib.load(RF_PATH); rf0, voc0 = rfm['rf'], rfm['vocab']
out.append(""); out.append("=== 3. prevalence and operating point ===")
for name, m in (('within-test', te), ('unseen equip', ge)):
    y = Y[m]; p = rf0.predict_proba(np.hstack([X[m], onehot(FAC[m], voc0)]))[:, 1]; rule = dT[m] > 5.5
    def op(pred):
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
        return f"sens {tp/max(tp+fn,1):.2f} spec {tn/max(tn+fp,1):.2f} prec {tp/max(tp+fp,1):.2f} flag-rate {pred.mean():.2f}"
    flag = p >= 0.5; review = (p >= 0.35) & (p < 0.5) & rule
    nses = len(np.unique(SES[m]))
    out.append(f"{name}: boxes {m.sum()} sessions {nses} prevalence {y.mean():.3f} | rule@5.5: {op(rule)} | RF@0.5: {op(flag)} | review zone {review.mean():.3f} of boxes | flagged boxes per session {flag.sum()/nses:.1f}, review per session {review.sum()/nses:.1f}")
# ---------- 4. camera / scene leakage ----------
out.append(""); out.append("=== 4. session-id prefix (camera/scene code) shared between training and held-out sessions ===")
pref = lambda s: s.split('(')[0]
ptr = set(pref(s) for s in np.unique(SES[tr])); pte = [pref(s) for s in np.unique(SES[te])]; pge = [pref(s) for s in np.unique(SES[ge])]
out.append(f"distinct prefixes: train {len(ptr)} | test sessions sharing a training prefix {sum(p in ptr for p in pte)}/{len(pte)} | unseen-equipment sessions sharing one {sum(p in ptr for p in pge)}/{len(pge)}")
m_seen = te & np.isin(np.array([pref(s) for s in SES]), list(ptr)); m_unseen = te & ~m_seen
p_te = rf0.predict_proba(np.hstack([X[te], onehot(FAC[te], voc0)]))[:, 1]; P = np.zeros(len(Y)); P[te] = p_te
for lab, mm in (('test rows with a training prefix', m_seen), ('test rows with a new prefix', m_unseen)):
    if mm.sum() and Y[mm].sum() and (1-Y[mm]).sum(): out.append(f"  {lab}: n={mm.sum()} sessions={len(np.unique(SES[mm]))} | RF {fmt(P[mm], Y[mm], SES[mm])} | rule {fmt(dT[mm], Y[mm], SES[mm])}")
    else: out.append(f"  {lab}: n={mm.sum()} (no AUROC: single class or empty)")
# ---------- 5. scene median vs entered atmospheric temperature ----------
out.append(""); out.append("=== 5. scene median minus the camera's entered atmospheric temperature, per site (session means) ===")
atm = {}; site = {}
for l in open(MAN, encoding='utf-8'):
    r = json.loads(l); c = r.get('calib') or {}; atm.setdefault(r['session'], c.get('atmospheric_temp')); site.setdefault(r['session'], r.get('region'))
if i_tmax is not None:
    med = X[:, i_tmax] - dT; bys = collections.defaultdict(list)
    for s, mv in zip(SES, med): bys[s].append(mv)
    rows = collections.defaultdict(list)
    for s, v in bys.items():
        a = atm.get(s)
        if a is not None: rows[site.get(s)].append(np.mean(v) - a)
    for st_, v in rows.items():
        v = np.array(v); out.append(f"  {st_:10s} sessions={len(v):3d} | scene median - atmospheric setting: median {np.median(v):+.1f} C, IQR [{np.percentile(v,25):+.1f}, {np.percentile(v,75):+.1f}], range [{v.min():+.1f}, {v.max():+.1f}]")
else: out.append("  t_max feature not found; skipped")
# ---------- 6. session-level evidence for subtle faults ----------
out.append(""); out.append("=== 6. subtle faults (label faulty, dT_scene < 4 C) in held-out splits: does any other box of the same equipment in the same session exceed the rule or the gate? ===")
for name, m in (('within-test', te), ('unseen equip', ge)):
    sub = m & (Y == 1) & (dT < 4); n = sub.sum(); rec_rule = rec_rf = 0
    p_all = np.zeros(len(Y)); p_all[m] = rf0.predict_proba(np.hstack([X[m], onehot(FAC[m], voc0)]))[:, 1]
    for i in np.where(sub)[0]:
        same = m & (SES == SES[i]) & (FAC == FAC[i]); same[i] = False
        if same.any() and (dT[same] > 5.5).any(): rec_rule += 1
        if same.any() and (p_all[same] >= 0.5).any(): rec_rf += 1
    out.append(f"  {name}: subtle faulty boxes {n} | session contains another same-equipment box with rule>5.5: {rec_rule} | with RF>=0.5: {rec_rf}")
txt = "\n".join(out); open(os.path.join(R, 'tier2_physics.txt'), 'w', encoding='utf-8').write(txt); print(txt)
