# -*- coding: utf-8 -*-
"""
extract_features.py — parse each frame's radiometric CSV once (multiprocessing) and cache per-bbox
features + metadata (label, facility, session, split_group). Enables fast model iteration.
Output: <PROJECT_ROOT>\results\feat_cache.npz
"""
import json, os, re, sys, io, time
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

MAN = r"<PROJECT_ROOT>\data\manifests\manifest_e_split.jsonl"
OUT = r"<PROJECT_ROOT>\results\feat_cache.npz"
FEAT_NAMES = ['t_max','t_mean','t_p95','t_std','dT_scene_max','dT_scene_mean','dT_sib_max',
              'frac_hot','grad_max','grad_mean','hot_conc','w','h','aspect','scene_med','scene_p95',
              # spatial-summary (hotspot shape/concentration/location/multiplicity)
              'hot_frac','n_hot','big_frac','hot_cx','hot_cy','compact','peak_z','skew','kurt','center_edge']

def parse_csv(cf):
    try: lines = open(cf, encoding='utf-8').read().splitlines()
    except: return None
    rows = []
    for l in lines[5:]:
        try: r = [float(p) for p in l.split(';') if p.strip() != '']
        except: continue
        if len(r) >= 600: rows.append(r[:640])
    return np.array(rows, np.float32) if len(rows) >= 400 else None

def frame_feats(rec):
    T = parse_csv(rec.get('csv'))
    if T is None: return []
    H, W = T.shape; sm = float(np.median(T)); sp95 = float(np.percentile(T, 95))
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    info = []
    for a in anns:
        b = a['bbox']; x = max(0, b['x']); y = max(0, b['y']); x2 = min(W, x + b['width']); y2 = min(H, y + b['height'])
        roi = T[y:y2, x:x2]
        info.append((a.get('class'), float(roi.max()) if roi.size else sm, roi) if roi.size >= 16 else None)
    out = []
    for i, a in enumerate(anns):
        if info[i] is None: continue
        cls, tmax, roi = info[i]
        sib = [info[j][1] for j in range(len(anns)) if j != i and info[j] and info[j][0] == cls]
        sib_ref = float(np.median(sib)) if sib else sm
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, 3)
        g = np.sqrt(gx * gx + gy * gy)
        thr = sm + 0.7 * (tmax - sm)                    # hotspot concentration: area near the peak
        feat = [tmax, float(roi.mean()), float(np.percentile(roi, 95)), float(roi.std()),
                tmax - sm, float(roi.mean()) - sm, tmax - sib_ref,
                float((roi > sp95).mean()), float(g.max()), float(g.mean()),
                float((roi > thr).mean()), float(roi.shape[1]), float(roi.shape[0]),
                float(roi.shape[0] / max(roi.shape[1], 1)), sm, sp95]
        # --- spatial-summary features (hotspot shape / concentration / location / multiplicity) ---
        rv = roi.ravel().astype(np.float64); mu = rv.mean(); sd = rv.std() + 1e-6
        skew = float(((rv-mu)**3).mean()/sd**3); kurt = float(((rv-mu)**4).mean()/sd**4 - 3)
        peak_z = float((tmax - mu)/sd)
        hi = sm + 0.6*(tmax - sm); hot = (roi > hi).astype(np.uint8)
        hot_frac = float(hot.mean())
        ncc, lbl = cv2.connectedComponents(hot)
        n_hot = ncc - 1
        big_frac = hcx = hcy = compact = 0.0
        if n_hot >= 1:
            sizes = np.bincount(lbl.ravel())[1:]; big = 1 + int(np.argmax(sizes)); bm = (lbl == big)
            big_frac = float(bm.sum())/hot.size
            ys, xs = np.where(bm); hcx = float(xs.mean())/roi.shape[1]; hcy = float(ys.mean())/roi.shape[0]
            cnts, _ = cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            peri = cv2.arcLength(cnts[0], True) if cnts else 0
            compact = float(4*np.pi*bm.sum()/(peri*peri)) if peri > 1 else 0.0
        h_, w_ = roi.shape; c = roi[h_//3:2*h_//3, w_//3:2*w_//3]
        center = float(c.mean()) if c.size else mu
        edge = float((roi.sum() - c.sum())/max(roi.size - c.size, 1))
        feat += [hot_frac, float(n_hot), big_frac, hcx, hcy, compact, peak_z, skew, kurt, center - edge]
        out.append((feat, 1 if a['status'] == 'danger' else 0, rec['facility'], rec['session'], rec['split_group']))
    return out

def main():
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    recs = [r for r in recs if r.get('csv')]
    print(f"frames to parse: {len(recs)}")
    t0 = time.time(); X = []; Y = []; FAC = []; SES = []; GRP = []
    with Pool(8) as p:
        for k, res in enumerate(p.imap(frame_feats, recs, chunksize=40)):
            for feat, y, fac, ses, grp in res:
                X.append(feat); Y.append(y); FAC.append(fac); SES.append(ses); GRP.append(grp)
            if (k + 1) % 5000 == 0: print(f"  {k+1}/{len(recs)} frames ({time.time()-t0:.0f}s), {len(Y)} bboxes")
    X = np.array(X, np.float32); Y = np.array(Y, np.int8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, X=X, Y=Y, FAC=np.array(FAC), SES=np.array(SES), GRP=np.array(GRP), FEAT_NAMES=np.array(FEAT_NAMES))
    print(f"\nsaved {OUT}: {len(Y):,} bboxes, {X.shape[1]} feats")
    import collections
    print("by group:", dict(collections.Counter(GRP)))
    print("danger by group:", {g: int(Y[np.array(GRP)==g].sum()) for g in set(GRP)})

if __name__ == '__main__':
    main()
