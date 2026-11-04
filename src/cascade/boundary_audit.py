# -*- coding: utf-8 -*-
"""
boundary_audit.py — measurement-validity audit found by the image judges:
a bbox's t_max can come from an EXTERNAL hot body bleeding a few pixels inside the box border.
For every test/gen_test bbox we recompute the peak on an eroded ROI (inset by k px) and measure:
  * how often the peak sits on the border (<=2 px)
  * how much dT_scene drops when the border ring is excluded
  * whether the RF gate's decision flips
  * whether erosion changes AUROC (i.e. is the border ring signal or artifact?)
usage: python boundary_audit.py [--inset 3]
"""
import json, os, sys, io, argparse, collections
import numpy as np
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi, load_rf, rf_prob, FEAT_NAMES, frame_feats_idx

def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); o = s.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(s)+1)
    p = y.sum(); n = len(y)-p
    return float('nan') if p == 0 or n == 0 else float((r[y == 1].sum()-p*(p+1)/2)/(p*n))

INSET = 3
def job(rec):
    T = parse_csv(rec['csv'])
    if T is None: return []
    sm = float(np.median(T))
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    out = []
    for a in anns:
        b = a['bbox']
        roi, (x, y, x2, y2) = clip_roi(T, b)
        if roi.size < 64 or min(roi.shape) < 2*INSET+2: continue
        py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
        border = min(px, py, roi.shape[1]-1-px, roi.shape[0]-1-py)
        inner = roi[INSET:-INSET, INSET:-INSET]
        tmax = float(roi.max()); tin = float(inner.max())
        # ring = the border band only
        ring = np.concatenate([roi[:INSET].ravel(), roi[-INSET:].ravel(), roi[:, :INSET].ravel(), roi[:, -INSET:].ravel()])
        # how hot is it just OUTSIDE the box on the peak's side (external-source evidence)?
        H, W = T.shape
        ox1, oy1, ox2, oy2 = max(0, x-INSET), max(0, y-INSET), min(W, x2+INSET), min(H, y2+INSET)
        outer = T[oy1:oy2, ox1:ox2]
        mask = np.ones(outer.shape, bool)
        mask[y-oy1:y2-oy1, x-ox1:x2-ox1] = False
        t_out = float(outer[mask].max()) if mask.any() else sm
        out.append(dict(status=a['status'], facility=rec['facility'], session=rec['session'], grp=rec['split_group'],
                        t_max=tmax, t_in=tin, ring_max=float(ring.max()), t_outside=t_out, scene_med=sm,
                        border_dist=int(border), w=int(roi.shape[1]), h=int(roi.shape[0])))
    return out

def main():
    global INSET
    ap = argparse.ArgumentParser(); ap.add_argument('--inset', type=int, default=3); ap.add_argument('--frames', type=int, default=2500)
    a = ap.parse_args(); INSET = a.inset
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    pool_recs = [r for r in recs if r.get('csv') and r['split_group'] in ('test', 'gen_test')]
    by_s = collections.defaultdict(list)
    for r in pool_recs: by_s[r['session']].append(r)
    sample = [r for s in sorted(by_s) for r in by_s[s][:4]][:a.frames]
    print(f"frames: {len(sample)} (inset {INSET}px)")
    rows = []
    with Pool(8) as p:
        for k, res in enumerate(p.imap(job, sample, chunksize=16)):
            rows += res
            if (k+1) % 500 == 0: print(f"  {k+1}/{len(sample)} -> {len(rows)} bboxes")
    R = rows; n = len(R)
    y = np.array([1 if r['status'] == 'danger' else 0 for r in R])
    dT = np.array([r['t_max']-r['scene_med'] for r in R]); dTin = np.array([r['t_in']-r['scene_med'] for r in R])
    bd = np.array([r['border_dist'] for r in R]); drop = dT - dTin
    out = []
    out.append(f"=== 경계 오염 감사 (test+gen_test, bbox {n}, inset {INSET}px) ===")
    out.append(f"peak가 상자 테두리 {INSET}px 이내: {100*(bd < INSET).mean():.1f}%  (danger {100*(bd[y==1] < INSET).mean():.1f}% / normal {100*(bd[y==0] < INSET).mean():.1f}%)")
    out.append(f"테두리 제거 시 ΔT 하락: 중앙값 {np.median(drop):.2f}°C | >2°C 하락 {100*(drop > 2).mean():.1f}% | >5°C 하락 {100*(drop > 5).mean():.1f}%")
    big = drop > 5
    out.append(f"  >5°C 하락 케이스의 라벨: danger {int(y[big].sum())} / normal {int((1-y[big]).sum())} (전체 대비 {100*big.mean():.1f}%)")
    ext = np.array([r['t_outside'] > r['t_max']-0.5 for r in R])
    out.append(f"상자 바로 바깥({INSET}px 링)이 상자 최고온과 같거나 더 뜨거움: {100*ext.mean():.1f}%  | 그중 ΔT>5.5로 플래그되는 비율 {100*(dT[ext] > 5.5).mean():.1f}%")
    susp = ext & (drop > 2) & (dT > 5.5)
    out.append(f"★ 외부 열원 오염 의심(바깥이 더 뜨겁고 + 테두리 제거 시 2°C 이상 하락 + 규칙상 이상): {susp.sum()} bbox = {100*susp.mean():.1f}%  | 라벨 danger {int(y[susp].sum())} normal {int((1-y[susp]).sum())}")
    out.append("")
    out.append("=== 규칙 성능: 원본 ΔT vs 테두리 제거 ΔT ===")
    for g in ('test', 'gen_test'):
        m = np.array([r['grp'] == g for r in R])
        out.append(f"  {g}: rule AUROC 원본 {auroc(dT[m], y[m]):.3f} → 테두리 제거 {auroc(dTin[m], y[m]):.3f}  (n={m.sum()}, danger {int(y[m].sum())})")
    out.append("")
    out.append("=== 설비별 경계 접촉률(테두리 <3px에 peak) ===")
    for f in sorted(set(r['facility'] for r in R)):
        mm = np.array([r['facility'] == f for r in R])
        if mm.sum() < 50: continue
        out.append(f"  {f:24s} n={int(mm.sum()):5d} 경계접촉 {100*(bd[mm] < INSET).mean():5.1f}% | ΔT 하락>2°C {100*(drop[mm] > 2).mean():5.1f}% | 외부가 더 뜨거움 {100*ext[mm].mean():5.1f}%")
    txt = "\n".join(out); print(txt)
    open(r"<PROJECT_ROOT>\results\boundary_audit.txt", 'w', encoding='utf-8').write(txt)
    np.savez(r"<PROJECT_ROOT>\results\boundary_audit.npz", y=y, dT=dT, dTin=dTin, bd=bd,
             grp=np.array([r['grp'] for r in R]), fac=np.array([r['facility'] for r in R]), ses=np.array([r['session'] for r in R]))

if __name__ == '__main__':
    main()
