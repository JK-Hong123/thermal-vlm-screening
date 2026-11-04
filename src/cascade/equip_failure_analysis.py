# -*- coding: utf-8 -*-
"""equip_failure_analysis.py — why do pump (AUROC 0.29), PT (0.62), disconnector/DS (0.68) look weak?
Per-equipment AUROC with session-bootstrap 95% CI (to see if 0.29 is even distinguishable from chance
given n_sessions=5), feature importance restricted to each equipment's own training rows, and a dump
of the worst-ranked misclassified boxes with their dT_scene/dT_sibling/rule_verdict to cross-check
against the label-vs-standard divergence story (Section 3.3 of the manuscript).
Writes results/equip_failure_analysis.txt"""
import numpy as np, joblib, sys, io, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from sklearn.metrics import roc_auc_score
from common import FEAT, RF_PATH, onehot

Z = np.load(FEAT, allow_pickle=True)
X, Y, FAC, SES, GRP = Z['X'], Z['Y'].astype(int), Z['FAC'], Z['SES'], Z['GRP']
FN = list(Z['FEAT_NAMES'])
d = joblib.load(RF_PATH); rf, vocab = d['rf'], d['vocab']
dT_i = FN.index('dT_scene_max')

tr, te, ge = GRP == 'train', GRP == 'test', GRP == 'gen_test'
Xe_all = np.hstack([X, onehot(FAC, vocab)])
p_all = rf.predict_proba(Xe_all)[:, 1]
rule_all = X[:, dT_i] > 5.5

def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    o = s.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    pp = y.sum(); nn = len(y) - pp
    return float('nan') if pp == 0 or nn == 0 else float((r[y == 1].sum() - pp * (pp + 1) / 2) / (pp * nn))

def ci(s, y, ses, B=2000):
    uq = np.unique(ses); rng = np.random.default_rng(0); idxby = {u: np.where(ses == u)[0] for u in uq}; v = []
    for _ in range(B):
        if len(uq) < 2: break
        idx = np.concatenate([idxby[u] for u in rng.choice(uq, len(uq), replace=True)])
        a = auroc(s[idx], y[idx])
        if a == a: v.append(a)
    if not v: return float('nan'), float('nan')
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

TARGETS = ['펌프', '계기용변압기(PT)', '단로기(DS)', '고압전동기', '변압기 접속부']
out = ["설비별 실패 원인 분석 | RF(설비조건부) vs 규칙(dT_scene>5.5) | test 분할 | 세션 부트스트랩 95% CI", ""]
for eqname in TARGETS:
    m = (FAC == eqname) & te
    n = m.sum(); nses = len(set(SES[m].tolist())); y = Y[m]; prev = y.mean() if n else float('nan')
    a_rf = auroc(p_all[m], y); lo_rf, hi_rf = ci(p_all[m], y, SES[m])
    a_rule = auroc(X[m, dT_i], y); lo_ru, hi_ru = ci(X[m, dT_i], y, SES[m])
    out.append(f"[{eqname}] n={n} sessions={nses} danger_prev={prev:.2f}")
    out.append(f"  RF   AUROC {a_rf:.3f} [{lo_rf:.3f},{hi_rf:.3f}]  (CI includes 0.5: {lo_rf<=0.5<=hi_rf})")
    out.append(f"  rule AUROC {a_rule:.3f} [{lo_ru:.3f},{hi_ru:.3f}]  (CI includes 0.5: {lo_ru<=0.5<=hi_ru})")
    # cross-check against label-vs-standard divergence: among RF's errors, how many are "rule says X, label says not-X"?
    idx = np.where(m)[0]
    rule_flag = rule_all[idx]; rf_flag = p_all[idx] >= 0.5; yy = Y[idx]
    rf_wrong = rf_flag != yy
    n_wrong = rf_wrong.sum()
    among_wrong_rule_agrees_label = int((rule_flag[rf_wrong] == yy[rf_wrong]).sum()) if n_wrong else 0
    out.append(f"  RF 오분류 {n_wrong}/{n}건 중 규칙(dT_scene)은 라벨과 일치했던 비율: "
                f"{among_wrong_rule_agrees_label}/{n_wrong if n_wrong else 1} "
                f"({'규칙이 RF보다 라벨에 더 가까움을 시사' if n_wrong and among_wrong_rule_agrees_label/n_wrong>0.5 else '규칙도 같이 틀림 - 라벨 자체 문제 가능성'})")
    out.append("")

out.append("=== 특징 중요도 (전체 RF, 상위 12) ===")
imp = rf.feature_importances_
names_full = FN + [f"equip={v}" for v in vocab]
order = np.argsort(imp)[::-1][:12]
for i in order:
    out.append(f"  {names_full[i]:28s} {imp[i]:.4f}")

out.append("")
out.append("=== 펌프 상세: 오분류 사례 표본 (최대 8건) ===")
m = (FAC == '펌프') & te
idx = np.where(m)[0]
rf_flag = p_all[idx] >= 0.5; yy = Y[idx]
wrong_idx = idx[rf_flag != yy]
rng = np.random.default_rng(0)
sample = wrong_idx[rng.permutation(len(wrong_idx))[:8]] if len(wrong_idx) else []
for i in sample:
    out.append(f"  세션={SES[i]} 라벨={'danger' if Y[i] else 'normal'} RF_p={p_all[i]:.2f} "
                f"dT_scene={X[i,dT_i]:.1f} t_max={X[i, FN.index('t_max')]:.1f} "
                f"(규칙판정={'flag' if rule_all[i] else 'pass'})" if 't_max' in FN else
                f"  세션={SES[i]} 라벨={'danger' if Y[i] else 'normal'} RF_p={p_all[i]:.2f} dT_scene={X[i,dT_i]:.1f}")

txt = "\n".join(out)
open(r"<PROJECT_ROOT>\results\equip_failure_analysis.txt", 'w', encoding='utf-8').write(txt)
print(txt)
