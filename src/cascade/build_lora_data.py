# -*- coding: utf-8 -*-
"""
build_lora_data.py — free supervision from the radiometric CSV: thermal crop -> which 3x3 cell holds the hotspot.
Two task variants per bbox:
  marked   : cyan circle+cross drawn at the CSV argmax  (tests marker reading)
  unmarked : no marker, model must find the hottest point itself (the useful capability)
Only TRAIN-split sessions are used. Writes 336px pngs + a jsonl of {image, task, answer, case meta}.
usage: python build_lora_data.py [--n 6000] [--val 600] [--out DIR]
"""
import json, os, sys, io, random, argparse, collections
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi, colorize
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
CYAN = (255, 255, 0)

def cell_of(b, px, py):
    gx = min(2, int(3*px/max(b['width'], 1))); gy = min(2, int(3*py/max(b['height'], 1)))
    return gy*3 + gx

def render(T, b, mark, size=336, ctx=2.4):
    H, W = T.shape; col = colorize(T)
    bw, bh = b['width'], b['height']; cx = b['x']+bw/2; cy = b['y']+bh/2; s = max(bw, bh)*ctx
    x = int(max(0, cx-s/2)); y = int(max(0, cy-s/2)); x2 = int(min(W, cx+s/2)); y2 = int(min(H, cy+s/2))
    cv2.rectangle(col, (b['x'], b['y']), (b['x']+bw, b['y']+bh), (255, 255, 255), 2)
    if mark:
        cv2.circle(col, mark, 9, CYAN, 2); cv2.drawMarker(col, mark, CYAN, cv2.MARKER_CROSS, 22, 2)
    c = col[y:y2, x:x2]
    return cv2.resize(c, (size, size)) if c.size else None

def job(args):
    rec, outdir, tag = args
    T = parse_csv(rec['csv'])
    if T is None: return []
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    out = []
    for i, a in enumerate(anns):
        b = a['bbox']
        if min(b['width'], b['height']) < 24: continue
        roi, (x, y, x2, y2) = clip_roi(T, b)
        if roi.size < 400: continue
        py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
        ci = cell_of(b, px, py)
        for task, mark in (('marked', (x+px, y+py)), ('unmarked', None)):
            img = render(T, b, mark)
            if img is None: continue
            fn = f"{tag}_{rec['frame_key']}_{i}_{task}.png".replace('(', '_').replace(')', '_')
            fp = os.path.join(outdir, fn)
            cv2.imwrite(fp, img)
            out.append({'image': fp, 'task': task, 'answer': GRID_KO[ci], 'cell': ci,
                        'equipment': rec['facility'], 'session': rec['session'], 'frame_key': rec['frame_key'], 'ann': i})
        if len(out) >= 6: break   # cap per frame
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=6000); ap.add_argument('--val', type=int, default=600)
    ap.add_argument('--out', default=r"<PROJECT_ROOT>\results\lora_data"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, 'img'), exist_ok=True)
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    tr = [r for r in recs if r.get('csv') and r['split_group'] == 'train']
    by_s = collections.defaultdict(list)
    for r in tr: by_s[r['session']].append(r)
    sess = sorted(by_s); random.seed(0); random.shuffle(sess)
    val_sess = set(sess[:max(1, len(sess)//10)])
    # sample up to 2 frames per session, interleaved so we cover many sessions first
    sample = [r for s in sess for r in by_s[s][:2]]
    random.shuffle(sample)
    need = (a.n + a.val)//2 + 200   # 2 examples per bbox, ~1-3 bboxes per frame
    sample = sample[:need]
    print(f"frames to parse: {len(sample)} ({len(sess)} train sessions, {len(val_sess)} held out for val)")
    rows = []
    with Pool(8) as p:
        for k, res in enumerate(p.imap(job, [(r, os.path.join(a.out, 'img'), 'tr') for r in sample], chunksize=8)):
            rows += res
            if (k+1) % 300 == 0: print(f"  {k+1}/{len(sample)} frames, {len(rows)} examples")
    train = [r for r in rows if r['session'] not in val_sess][:a.n]
    val = [r for r in rows if r['session'] in val_sess][:a.val]
    for name, data in (('train', train), ('val', val)):
        with open(os.path.join(a.out, f'{name}.jsonl'), 'w', encoding='utf-8') as fo:
            for r in data: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
        c = collections.Counter(r['answer'] for r in data)
        print(f"{name}: {len(data)} examples | cell distribution {dict(c)}")
        print(f"   equipment: {dict(collections.Counter(r['equipment'] for r in data).most_common(6))}")

if __name__ == '__main__':
    main()
