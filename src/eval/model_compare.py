# -*- coding: utf-8 -*-
"""
model_compare.py — core experiment. Does a learned equipment-conditioned model beat the ΔT rule?
  baselines: ΔT rule (single feature dT_scene_max)
  learned  : RandomForest on physics features (+/- equipment one-hot)
  eval     : within-test (seen equipment, new sessions) + gen_test (unseen equipment/zero-shot)
             session-grouped bootstrap 95% CI; hard-case breakdown; per-equipment AUROC.
"""
import numpy as np, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.ensemble import RandomForestClassifier

Z = np.load(r"<PROJECT_ROOT>\results\feat_cache.npz", allow_pickle=True)
X, Y, FAC, SES, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['SES'], Z['GRP']
FN = list(Z['FEAT_NAMES']); dT_i = FN.index('dT_scene_max')
def m(g): return GRP == g
tr, te, ge = m('train'), m('test'), m('gen_test')
print(f"bboxes: train {tr.sum()}, test {te.sum()}, gen_test {ge.sum()}")
print(f"danger: train {Y[tr].sum()}, test {Y[te].sum()}, gen {Y[ge].sum()}")

def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); o = s.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(s)+1)
    p = y.sum(); n = len(y) - p
    return float('nan') if p == 0 or n == 0 else float((r[y==1].sum() - p*(p+1)/2)/(p*n))
def ci(scores, y, ses, n=1000):
    ses = np.asarray(ses); uniq = np.unique(ses); rng = np.random.default_rng(0); v = []
    idxby = {s: np.where(ses==s)[0] for s in uniq}
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idxby[s] for s in pick])
        a = auroc(np.asarray(scores)[idx], np.asarray(y)[idx])
        if a == a: v.append(a)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

# equipment one-hot (train vocab)
vocab = sorted(set(FAC[tr])); vi = {f: i for i, f in enumerate(vocab)}
def onehot(fac):
    O = np.zeros((len(fac), len(vocab)), np.float32)
    for k, f in enumerate(fac):
        if f in vi: O[k, vi[f]] = 1
    return O

def evalset(name, mask):
    print(f"\n===== {name} (bboxes {mask.sum()}, sessions {len(set(SES[mask]))}) =====")
    y = Y[mask]; ses = SES[mask]
    # 1) rule
    r = X[mask, dT_i]; a = auroc(r, y); lo, hi = ci(r, y, ses)
    print(f"  [rule ΔT-scene]         AUROC {a:.3f}  [{lo:.3f}, {hi:.3f}]")
    # 2) RF physics-only
    a2 = auroc(rf_phys.predict_proba(X[mask])[:,1], y); lo2, hi2 = ci(rf_phys.predict_proba(X[mask])[:,1], y, ses)
    print(f"  [RF physics]            AUROC {a2:.3f}  [{lo2:.3f}, {hi2:.3f}]")
    # 3) RF physics + equipment
    Xe = np.hstack([X[mask], onehot(FAC[mask])])
    a3 = auroc(rf_eq.predict_proba(Xe)[:,1], y); lo3, hi3 = ci(rf_eq.predict_proba(Xe)[:,1], y, ses)
    print(f"  [RF physics+equip]      AUROC {a3:.3f}  [{lo3:.3f}, {hi3:.3f}]")
    return r, rf_phys.predict_proba(X[mask])[:,1]

# train models
print("\ntraining RandomForests...")
rf_phys = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
                                 class_weight='balanced', n_jobs=-1, random_state=0).fit(X[tr], Y[tr])
rf_eq = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
                               class_weight='balanced', n_jobs=-1, random_state=0).fit(np.hstack([X[tr], onehot(FAC[tr])]), Y[tr])

rule_te, rf_te = evalset("WITHIN-TEST (seen equipment)", te)
evalset("GEN-TEST (unseen equipment, zero-shot)", ge)

# hard-case breakdown on within-test
print("\n===== HARD CASES (within-test) — where the ΔT rule fails =====")
dT = X[te, dT_i]; y = Y[te]
hn = (y==0) & (dT > 6)   # hot-but-normal (rule false-positive)
sf = (y==1) & (dT < 4)   # subtle-fault (rule false-negative)
print(f"  hot-but-normal (normal, ΔT>6): {hn.sum()}  | subtle-fault (danger, ΔT<4): {sf.sum()}")
# on these, does RF assign the correct side better than a ΔT threshold(5.5)?
for name, mm, want in [('hot-but-normal→normal', hn, 0), ('subtle-fault→danger', sf, 1)]:
    if mm.sum() == 0: continue
    rule_pred = (dT[mm] > 5.5).astype(int)
    rf_pred = (rf_te[mm] > 0.5).astype(int)
    print(f"    {name}: rule 정답률 {100*(rule_pred==want).mean():.0f}% | RF 정답률 {100*(rf_pred==want).mean():.0f}%  (n={mm.sum()})")

# per-equipment AUROC on test
print("\n===== per-equipment AUROC (within-test): rule vs RF =====")
for f in sorted(set(FAC[te])):
    mm = te & (FAC == f); y = Y[mm]
    if y.sum() == 0 or (y==0).sum() == 0: print(f"    {f}: (danger 없음, 생략)"); continue
    print(f"    {f:22s} n={mm.sum():4d} d={int(y.sum()):3d} | rule {auroc(X[mm,dT_i],y):.3f} | RF {auroc(rf_phys.predict_proba(X[mm])[:,1],y):.3f}")
