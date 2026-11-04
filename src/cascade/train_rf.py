# -*- coding: utf-8 -*-
"""train_rf.py — fit the gate (RF physics+equipment) on train split and persist it (same hyperparams as model_compare)."""
import numpy as np, joblib, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from common import FEAT, RF_PATH, onehot
Z = np.load(FEAT, allow_pickle=True)
X, Y, FAC, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['GRP']
tr = GRP == 'train'; vocab = sorted(set(FAC[tr]))
rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20, class_weight='balanced', n_jobs=-1, random_state=0)
rf.fit(np.hstack([X[tr], onehot(FAC[tr], vocab)]), Y[tr])
joblib.dump({'rf': rf, 'vocab': vocab, 'feat_names': list(Z['FEAT_NAMES'])}, RF_PATH)
for g in ('test', 'gen_test'):
    m = GRP == g; p = rf.predict_proba(np.hstack([X[m], onehot(FAC[m], vocab)]))[:, 1]
    print(f"{g}: AUROC {roc_auc_score(Y[m], p):.3f}  (flag rate @0.5: {(p>=0.5).mean():.3f}, danger prev {Y[m].mean():.3f})")
print("saved", RF_PATH, "vocab", len(vocab))
