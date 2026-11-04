# -*- coding: utf-8 -*-
"""heuristic_vs_model.py — is the trained VLM better than the trivial 'brightest token inside the box' heuristic
on the SAME independent eval set? Scores the heuristic at token grids K=24 (=672px Qwen), K=48 and pixel level,
and additionally reports agreement between heuristic and model predictions where the model's cached outputs exist."""
import json, os, sys, io, numpy as np, cv2, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from scipy.spatial import cKDTree
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
D = r'<PROJECT_ROOT>\results\eval_v3_big'
rows = [json.loads(l) for l in open(os.path.join(D, 'eval.jsonl'), encoding='utf-8')]
lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_INFERNO).reshape(256, 3).astype(np.float32)
tree = cKDTree(lut)
def inv(img):
    _, idx = tree.query(img.reshape(-1, 3).astype(np.float32), k=1); return idx.reshape(img.shape[:2]).astype(np.float32)
def kappa(pairs):
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe)
def cell(x, y): return GRID_KO[min(2, max(0, int(3*y)))*3 + min(2, max(0, int(3*x)))]
res = collections.defaultdict(list)
for r in rows:
    img = cv2.imread(r['image']); S = img.shape[0]
    m = (img.min(2) > 245); ys, xs = np.where(m)
    if len(xs) < 50: continue
    x1, x2, y1, y2 = xs.min()+3, xs.max()-3, ys.min()+3, ys.max()-3
    tm = inv(img); tm[m] = 0
    for K in (24, 48, 'pixel'):
        if K == 'pixel':
            sub = tm[y1:y2, x1:x2]; py, px = np.unravel_index(int(np.argmax(sub)), sub.shape); x = px/max(sub.shape[1]-1, 1); y = py/max(sub.shape[0]-1, 1)
        else:
            pooled = cv2.resize(tm, (K, K), interpolation=cv2.INTER_AREA); tx1, ty1 = int(x1/S*K), int(y1/S*K)
            tx2, ty2 = max(tx1+1, int(np.ceil(x2/S*K))), max(ty1+1, int(np.ceil(y2/S*K)))
            sub = pooled[ty1:ty2, tx1:tx2]; py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
            cx = (tx1+px+0.5)/K*S; cy = (ty1+py+0.5)/K*S; x = (cx-x1)/max(x2-x1, 1); y = (cy-y1)/max(y2-y1, 1)
        res[K].append((cell(x, y), r['cell'], float(np.hypot(x-r['x_norm'], y-r['y_norm'])), r['group'], r['equipment']))
L = [f"brightest-token heuristic on the SAME independent eval set (n={len(res[24])}) | model H: all κ 0.650 dist 0.176 <0.2 0.727 / test κ 0.761 / gen_test κ 0.564 / PV κ 0.501"]
for K, v in res.items():
    for g in ('all', 'test', 'gen_test', 'PV'):
        vv = [t for t in v if g == 'all' or (g == 'PV' and t[4] == '신재생 설비') or t[3] == g]
        if not vv: continue
        L.append(f"  K={K!s:6s} {g:9s} n={len(vv):3d} acc {np.mean([a==b for a,b,_,_,_ in vv]):.3f}  κ {kappa([(a,b) for a,b,_,_,_ in vv]):.3f}  dist {np.mean([d for _,_,d,_,_ in vv]):.3f}  <0.2 {np.mean([d<0.2 for _,_,d,_,_ in vv]):.3f}")
txt = "\n".join(L); print(txt); open(os.path.join(D, 'heuristic_vs_model.txt'), 'w', encoding='utf-8').write(txt)
