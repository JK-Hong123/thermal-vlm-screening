# -*- coding: utf-8 -*-
"""
build_lora_data_v2.py — larger UNMARKED-only dataset for "find the hotspot from the image alone" (CSV replacement).
Per bbox we store TWO renderings and TWO labels so the ablations share identical samples:
  render 'fixed'   : global 10-80°C inferno (what the cascade uses)          -> img_fixed
  render 'stretch' : per-crop percentile stretch p1..p99 (hotspot = brightest by construction) -> img_stretch
  label 'argmax'   : 3x3 cell of the raw hottest pixel
  label 'smooth'   : 3x3 cell of the hottest pixel after 5x5 Gaussian smoothing (robust to single-pixel outliers)
  plus normalized peak coordinates (x,y in [0,1] within the box) for coordinate-regression variants.
Train sessions only; val = 10% of train sessions held out. Locked-set crops are ALSO rendered in 'stretch' for evaluation.
usage: python build_lora_data_v2.py [--n 12000] [--val 800] [--fps 6]
"""
import json, os, sys, io, random, argparse, collections
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi, colorize
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
OUT = r"<PROJECT_ROOT>\results\lora_data_v2"
LOCK = r"<PROJECT_ROOT>\results\cases_lock"

def cell_of(b, px, py):
    gx = min(2, int(3*px/max(b['width'], 1))); gy = min(2, int(3*py/max(b['height'], 1)))
    return gy*3 + gx

def crop_box(T, b, ctx=2.4):
    H, W = T.shape; bw, bh = b['width'], b['height']; cx = b['x']+bw/2; cy = b['y']+bh/2; s = max(bw, bh)*ctx
    return int(max(0, cx-s/2)), int(max(0, cy-s/2)), int(min(W, cx+s/2)), int(min(H, cy+s/2))

def render_fixed(T, b, size=336):
    col = colorize(T); x, y, x2, y2 = crop_box(T, b)
    cv2.rectangle(col, (b['x'], b['y']), (b['x']+b['width'], b['y']+b['height']), (255, 255, 255), 2)
    c = col[y:y2, x:x2]; return cv2.resize(c, (size, size)) if c.size else None

def render_stretch(T, b, size=336):
    """percentile stretch computed on the CROP (so the hottest thing in view is always the brightest)."""
    x, y, x2, y2 = crop_box(T, b); sub = T[y:y2, x:x2]
    if sub.size == 0: return None
    lo, hi = np.percentile(sub, 1), np.percentile(sub, 99.5)
    if hi - lo < 0.5: hi = lo + 0.5
    norm = np.clip((sub - lo)/(hi - lo), 0, 1)
    col = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.rectangle(col, (b['x']-x, b['y']-y), (b['x']-x+b['width'], b['y']-y+b['height']), (255, 255, 255), 2)
    return cv2.resize(col, (size, size))

def labels_for(T, b):
    roi, _ = clip_roi(T, b)
    py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
    sm = cv2.GaussianBlur(roi, (5, 5), 0)
    sy, sx = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return dict(cell_argmax=cell_of(b, px, py), cell_smooth=cell_of(b, sx, sy),
                x_norm=round(float(sx)/max(b['width']-1, 1), 3), y_norm=round(float(sy)/max(b['height']-1, 1), 3),
                dT_scene=round(float(roi.max()-np.median(T)), 1))

def job(args):
    rec, outdir = args
    T = parse_csv(rec['csv'])
    if T is None: return []
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    out = []
    for i, a in enumerate(anns):
        b = a['bbox']
        if min(b['width'], b['height']) < 24: continue
        roi, _ = clip_roi(T, b)
        if roi.size < 400: continue
        f = render_fixed(T, b); s = render_stretch(T, b)
        if f is None or s is None: continue
        stem = f"{rec['frame_key']}_{i}".replace('(', '_').replace(')', '_')
        pf = os.path.join(outdir, stem + '_fixed.png'); ps = os.path.join(outdir, stem + '_stretch.png')
        cv2.imwrite(pf, f); cv2.imwrite(ps, s)
        lab = labels_for(T, b)
        out.append(dict(img_fixed=pf, img_stretch=ps, equipment=rec['facility'], session=rec['session'],
                        frame_key=rec['frame_key'], ann=i, status=a['status'], **lab))
    return out

def render_lock_stretch():
    recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    idx = [json.loads(l) for l in open(os.path.join(LOCK, 'index.jsonl'), encoding='utf-8')]
    n = 0
    for it in idx:
        d = os.path.join(LOCK, it['case_id']); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
        T = parse_csv(recs[f['frame_key']]['csv']); s = render_stretch(T, f['bbox'])
        if s is not None: cv2.imwrite(os.path.join(d, 'thermal_crop_v2_stretch.png'), s); n += 1
        lab = labels_for(T, f['bbox']); f['hotspot_labels_v2'] = lab
        json.dump(f, open(os.path.join(d, 'facts.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"locked set: {n} stretch renders + v2 labels written")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=12000); ap.add_argument('--val', type=int, default=800)
    ap.add_argument('--fps', type=int, default=6); a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'img'), exist_ok=True)
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    tr = [r for r in recs if r.get('csv') and r['split_group'] == 'train']
    by_s = collections.defaultdict(list)
    for r in tr: by_s[r['session']].append(r)
    sess = sorted(by_s); random.seed(0); random.shuffle(sess)
    val_sess = set(sess[:max(1, len(sess)//10)])
    # spread frames evenly inside each session (frames are near-duplicates; take every k-th)
    sample = []
    for s in sess:
        fr = by_s[s]; step = max(1, len(fr)//a.fps); sample += fr[::step][:a.fps]
    random.shuffle(sample)
    print(f"frames to parse: {len(sample)} from {len(sess)} sessions ({len(val_sess)} val sessions)")
    rows = []
    with Pool(8) as p:
        for k, res in enumerate(p.imap(job, [(r, os.path.join(OUT, 'img')) for r in sample], chunksize=8)):
            rows += res
            if (k+1) % 500 == 0: print(f"  {k+1}/{len(sample)} frames, {len(rows)} bboxes")
    train = [r for r in rows if r['session'] not in val_sess]; val = [r for r in rows if r['session'] in val_sess]
    random.shuffle(train); random.shuffle(val); train = train[:a.n]; val = val[:a.val]
    for name, data in (('train', train), ('val', val)):
        with open(os.path.join(OUT, f'{name}.jsonl'), 'w', encoding='utf-8') as fo:
            for r in data: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
        c = collections.Counter(r['cell_smooth'] for r in data); agree = np.mean([r['cell_smooth'] == r['cell_argmax'] for r in data])
        print(f"{name}: {len(data)} bboxes | smooth-cell dist {dict(sorted(c.items()))} | argmax==smooth {agree:.2f}")
        print(f"   equipment: {dict(collections.Counter(r['equipment'] for r in data).most_common(8))}")
    render_lock_stretch()

if __name__ == '__main__':
    main()
