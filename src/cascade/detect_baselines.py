# -*- coding: utf-8 -*-
"""detect_baselines.py — stronger detection baselines on the same session-level split and features (feat_cache):
rule ΔT / logistic regression / random forest (ours) / HistGradientBoosting / (optional) LightGBM|XGBoost / MLP,
each with equipment one-hot, evaluated on within-test and gen_test with session-clustered bootstrap 95% CIs.
Writes results/detect_baselines.txt"""
import numpy as np, sys, io, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import FEAT, onehot
Z = np.load(FEAT, allow_pickle=True); X, Y, FAC, SES, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['SES'], Z['GRP']
FN = list(Z['FEAT_NAMES']); dT_i = FN.index('dT_scene_max')
tr, te, ge = GRP == 'train', GRP == 'test', GRP == 'gen_test'; vocab = sorted(set(FAC[tr]))
Xe = np.hstack([X, onehot(FAC, vocab)])
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); o = s.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(s)+1)
    p = y.sum(); n = len(y)-p; return float('nan') if p == 0 or n == 0 else float((r[y == 1].sum()-p*(p+1)/2)/(p*n))
def ci(s, y, ses, B=1000):
    uq = np.unique(ses); rng = np.random.default_rng(0); idxby = {u: np.where(ses == u)[0] for u in uq}; v = []
    for _ in range(B):
        idx = np.concatenate([idxby[u] for u in rng.choice(uq, len(uq), replace=True)]); a = auroc(s[idx], y[idx])
        if a == a: v.append(a)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)
models = {
 'rule dT_scene': None,
 'logistic (std)': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
 'RF (ours)': RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20, class_weight='balanced', n_jobs=-1, random_state=0),
 'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, class_weight='balanced', random_state=0),
 'MLP (std, 2x128)': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0)),
}
try:
    import lightgbm as lgb; models['LightGBM'] = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=0, verbose=-1)
except Exception: pass
try:
    import xgboost as xgb; models['XGBoost'] = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=float((Y[tr]==0).sum()/max((Y[tr]==1).sum(),1)), random_state=0, n_jobs=8, verbosity=0)
except Exception: pass
out = [f"detection baselines | features 26 radiometric+spatial + equipment one-hot | train {tr.sum()} / test {te.sum()} (167 sess) / gen_test {ge.sum()} (86 sess) | AUROC [session-bootstrap 95% CI]",
       f"{'model':22s} {'within-test':>24s} {'gen_test (unseen equip)':>26s}"]
for name, m in models.items():
    if m is None: s_te, s_ge = X[te, dT_i], X[ge, dT_i]
    else:
        m.fit(Xe[tr], Y[tr]); s_te = m.predict_proba(Xe[te])[:, 1]; s_ge = m.predict_proba(Xe[ge])[:, 1]
    a1, (l1, h1) = auroc(s_te, Y[te]), ci(s_te, Y[te], SES[te]); a2, (l2, h2) = auroc(s_ge, Y[ge]), ci(s_ge, Y[ge], SES[ge])
    out.append(f"{name:22s} {a1:.3f} [{l1:.3f},{h1:.3f}]      {a2:.3f} [{l2:.3f},{h2:.3f}]"); print(out[-1])
open(r"<PROJECT_ROOT>\results\detect_baselines.txt", 'w', encoding='utf-8').write("\n".join(out)); print("saved results/detect_baselines.txt")
