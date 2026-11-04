# -*- coding: utf-8 -*-
"""
extract_crops.py — build 3-channel radiometric context crops per bbox (multiprocessing) and cache.
  ch0 = absolute temp (fixed scale)   ch1 = temp relative to scene median   ch2 = thermal gradient
Output: <PROJECT_ROOT>\results\crops_cache.npz  (X uint8 Nx96x96x3, Y, FAC, SES, GRP, DT)
"""
import json, os, sys, io, time
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
MAN = r"<PROJECT_ROOT>\data\manifests\manifest_e_split.jsonl"
OUT = r"<PROJECT_ROOT>\results\crops_cache.npz"
SZ = 96; LO, HI = 5.0, 85.0

def parse_csv(cf):
    try: lines = open(cf, encoding='utf-8').read().splitlines()
    except: return None
    rows = []
    for l in lines[5:]:
        try: r = [float(p) for p in l.split(';') if p.strip() != '']
        except: continue
        if len(r) >= 600: rows.append(r[:640])
    return np.array(rows, np.float32) if len(rows) >= 400 else None

def make(rec):
    T = parse_csv(rec.get('csv'))
    if T is None: return []
    H, W = T.shape; sm = float(np.median(T))
    out = []
    for a in rec['annotations']:
        b = a.get('bbox'); st = a.get('status')
        if not b or st not in ('danger', 'normal'): continue
        bw, bh = b['width'], b['height']
        if bw < 4 or bh < 4: continue
        cx = b['x'] + bw / 2; cy = b['y'] + bh / 2; side = max(bw, bh) * 2.0
        x = int(max(0, cx - side/2)); y = int(max(0, cy - side/2)); x2 = int(min(W, cx + side/2)); y2 = int(min(H, cy + side/2))
        roi = T[y:y2, x:x2]
        if roi.size < 16: continue
        ch0 = np.clip((roi - LO) / (HI - LO), 0, 1)
        ch1 = np.clip((roi - sm + 15) / 40.0, 0, 1)
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, 3)
        ch2 = np.clip(np.sqrt(gx*gx + gy*gy) / 50.0, 0, 1)
        img = cv2.resize(np.stack([ch0, ch1, ch2], -1), (SZ, SZ))
        bx = max(0, b['x']); by = max(0, b['y']); bx2 = min(W, bx+bw); by2 = min(H, by+bh)
        dT = float(T[by:by2, bx:bx2].max()) - sm
        out.append(((img*255).astype(np.uint8), 1 if st == 'danger' else 0, rec['facility'], rec['session'], rec['split_group'], dT))
    return out

def main():
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8') if 'csv' in json.loads(l)]
    print(f"frames: {len(recs)}")
    t0 = time.time(); X=[]; Y=[]; FAC=[]; SES=[]; GRP=[]; DT=[]
    with Pool(8) as p:
        for k, res in enumerate(p.imap(make, recs, chunksize=40)):
            for im, y, fac, ses, grp, dt in res:
                X.append(im); Y.append(y); FAC.append(fac); SES.append(ses); GRP.append(grp); DT.append(dt)
            if (k+1) % 5000 == 0: print(f"  {k+1}/{len(recs)} ({time.time()-t0:.0f}s), {len(Y)} crops")
    X = np.array(X, np.uint8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, X=X, Y=np.array(Y, np.int8), FAC=np.array(FAC), SES=np.array(SES), GRP=np.array(GRP), DT=np.array(DT, np.float32))
    print(f"saved {OUT}: {X.shape} ({X.nbytes/1e9:.1f}GB) in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
