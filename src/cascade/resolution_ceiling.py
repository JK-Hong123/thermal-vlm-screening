# -*- coding: utf-8 -*-
"""
resolution_ceiling.py — is the hotspot cell even recoverable at the VLM's visual-token resolution?
For each val crop PNG (what the model actually sees) we take the brightest pixel INSIDE the white box
(inferno luminance ~ temperature) and map it to a 3x3 cell:
  * at full 336px pixel resolution          -> information ceiling of the rendering
  * after average-pooling to a KxK token grid (K=12 = Qwen 336px after 2x2 merge; 24 = 672px; 6 = worst case)
    -> what a model limited to K x K spatial tokens could at best recover
Also repeats for a TIGHT crop (ctx 1.4) re-rendered from CSV at 672px.
"""
import json, os, sys, io, collections
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
DATA = r"<PROJECT_ROOT>\results\lora_data_v2"
# inverse inferno: luminance of the rendered colour is monotone-ish in temperature; use a LUT from the colormap itself
_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_INFERNO).reshape(256, 3).astype(np.float32)
def inv_inferno(img):
    """map each BGR pixel to the colormap index (0..255) of its nearest LUT colour -> proxy temperature."""
    h, w = img.shape[:2]; px = img.reshape(-1, 3).astype(np.float32)
    # nearest neighbour in LUT (256 entries) — vectorised in chunks
    out = np.empty(px.shape[0], np.float32)
    for i in range(0, px.shape[0], 20000):
        d = ((px[i:i+20000, None, :] - _lut[None, :, :])**2).sum(-1); out[i:i+20000] = d.argmin(1)
    return out.reshape(h, w)

def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs)
    po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats)
    return (po-pe)/(1-pe) if pe < 1 else float('nan')

def box_from_render(size, ctx, b):
    """where the bbox lands inside the resized crop (same geometry as render_stretch)."""
    bw, bh = b['width'], b['height']; s = max(bw, bh)*ctx
    scale = size/s; x0 = (s-bw)/2*scale; y0 = (s-bh)/2*scale
    return int(x0), int(y0), int(x0+bw*scale), int(y0+bh*scale)

def cell_at_K(temp_map, box, K):
    """pool the proxy-temperature map to a KxK token grid over the WHOLE image, then find the max token inside the box."""
    H, W = temp_map.shape
    pooled = cv2.resize(temp_map, (K, K), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(x1, W-1)), max(0, min(y1, H-1)); x2, y2 = max(x1+1, min(x2, W)), max(y1+1, min(y2, H))
    tx1, ty1 = min(K-1, int(x1/W*K)), min(K-1, int(y1/H*K)); tx2, ty2 = min(K, max(tx1+1, int(np.ceil(x2/W*K)))), min(K, max(ty1+1, int(np.ceil(y2/H*K))))
    sub = pooled[ty1:ty2, tx1:tx2]; py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
    # token centre -> pixel -> cell within box
    cx = (tx1+px+0.5)/K*W; cy = (ty1+py+0.5)/K*H
    gx = min(2, max(0, int(3*(cx-x1)/max(x2-x1, 1)))); gy = min(2, max(0, int(3*(cy-y1)/max(y2-y1, 1))))
    return GRID_KO[gy*3+gx]

def job(r):
    recs = job.recs; rec = recs[r['frame_key']]; T = parse_csv(rec['csv'])
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    b = anns[r['ann']]['bbox']; truth = GRID_KO[r['cell_smooth']]; out = {'truth': truth, 'truth_argmax': GRID_KO[r['cell_argmax']]}
    for name, size, ctx, top in (('p99.5 ctx2.4@336', 336, 2.4, 99.5), ('p100 ctx2.4@336', 336, 2.4, 100.0), ('p100 ctx1.4@672', 672, 1.4, 100.0), ('p100 ctx1.4@1008', 1008, 1.4, 100.0)):
        # re-render stretch at this geometry directly from the CSV
        H, W = T.shape; bw, bh = b['width'], b['height']; cxp = b['x']+bw/2; cyp = b['y']+bh/2; s = max(bw, bh)*ctx
        x = int(max(0, cxp-s/2)); y = int(max(0, cyp-s/2)); x2 = int(min(W, cxp+s/2)); y2 = int(min(H, cyp+s/2))
        sub = T[y:y2, x:x2]
        if sub.size == 0: return None
        lo, hi = np.percentile(sub, 1), np.percentile(sub, top); hi = max(hi, lo+0.5)
        norm = np.clip((sub-lo)/(hi-lo), 0, 1); col = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        col = cv2.resize(col, (size, size)); tm = inv_inferno(col)
        bx = (int((b['x']-x)*size/(x2-x)), int((b['y']-y)*size/(y2-y)), int((b['x']-x+bw)*size/(x2-x)), int((b['y']-y+bh)*size/(y2-y)))
        out[name+'/pixel'] = cell_at_K(tm, bx, size)         # K=size -> pixel-level ceiling
        for K in (12, 24, 36, 48):
            out[f'{name}/K{K}'] = cell_at_K(tm, bx, K)
    return out

def init(recs): job.recs = recs

def main():
    rows = [json.loads(l) for l in open(os.path.join(DATA, 'val.jsonl'), encoding='utf-8')][:400]
    recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    with Pool(8, initializer=init, initargs=(recs,)) as p: res = [r for r in p.map(job, rows, chunksize=8) if r]
    keys = [k for k in res[0] if k != 'truth']
    print(f"val bboxes {len(res)} | label = smoothed-argmax cell | chance acc 0.111\n")
    print(f"{'geometry / resolution':22s} {'acc':>6s} {'kappa':>6s}")
    for k in keys:
        pairs = [(r[k], r['truth']) for r in res]
        print(f"{k:22s} {sum(p==t for p,t in pairs)/len(pairs):6.3f} {kappa(pairs):6.3f}")
    print("\nreading: 'pixel' = information present in the rendered PNG; K12 = Qwen2.5-VL at 336px (12x12 merged tokens); K24 = 672px.")

if __name__ == '__main__':
    main()
