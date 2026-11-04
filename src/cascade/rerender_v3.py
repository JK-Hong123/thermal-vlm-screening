# -*- coding: utf-8 -*-
"""rerender_v3.py — high-resolution, tight-crop, no-top-clip stretch renders for the resolution experiment (E).
Reads existing lora_data_v2 rows (train subset + val) and the locked set; writes <stem>_v3.png (size px, ctx) with
percentile stretch lo=p1, hi=MAX (the hottest pixel is never saturated into a tie). Adds 'img_v3' to the jsonl rows.
usage: python rerender_v3.py [--size 672] [--ctx 1.4] [--n_train 3000]"""
import json, os, sys, io, argparse
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv
DATA = r"<PROJECT_ROOT>\results\lora_data_v2"; LOCK = r"<PROJECT_ROOT>\results\cases_lock"

def render_v3(T, b, size, ctx):
    H, W = T.shape; bw, bh = b['width'], b['height']; cx = b['x']+bw/2; cy = b['y']+bh/2; s = max(bw, bh)*ctx
    x = int(max(0, cx-s/2)); y = int(max(0, cy-s/2)); x2 = int(min(W, cx+s/2)); y2 = int(min(H, cy+s/2))
    sub = T[y:y2, x:x2]
    if sub.size == 0: return None
    lo, hi = float(np.percentile(sub, 1)), float(sub.max()); hi = max(hi, lo+0.5)
    norm = np.clip((sub-lo)/(hi-lo), 0, 1); col = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    col = cv2.resize(col, (size, size), interpolation=cv2.INTER_CUBIC)
    sx, sy = size/(x2-x), size/(y2-y)
    cv2.rectangle(col, (int((b['x']-x)*sx), int((b['y']-y)*sy)), (int((b['x']-x+bw)*sx), int((b['y']-y+bh)*sy)), (255, 255, 255), 3)
    return col

def job(args):
    r, size, ctx, out = args
    rec = job.recs[r['frame_key']]; T = parse_csv(rec['csv'])
    if T is None: return None
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    img = render_v3(T, anns[r['ann']]['bbox'] if 'ann' in r else r['bbox'], size, ctx)
    if img is None: return None
    cv2.imwrite(out, img); return out
def init(recs): job.recs = recs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--size', type=int, default=672); ap.add_argument('--ctx', type=float, default=1.4); ap.add_argument('--n_train', type=int, default=3000)
    a = ap.parse_args(); recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    jobs = []; rows_all = {}
    for name, lim in (('train', a.n_train), ('val', None)):
        rows = [json.loads(l) for l in open(os.path.join(DATA, f'{name}.jsonl'), encoding='utf-8')]
        rows_all[name] = rows
        for i, r in enumerate(rows[:lim] if lim else rows):
            out = r['img_stretch'].replace('_stretch.png', f'_v3_{a.size}.png'); r['img_v3'] = out; jobs.append((r, a.size, a.ctx, out))
    idx = [json.loads(l) for l in open(os.path.join(LOCK, 'index.jsonl'), encoding='utf-8')]
    for it in idx:
        d = os.path.join(LOCK, it['case_id']); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
        jobs.append(({'frame_key': f['frame_key'], 'bbox': f['bbox']}, a.size, a.ctx, os.path.join(d, f'thermal_crop_v3_{a.size}.png')))
    print(f"rendering {len(jobs)} images at {a.size}px ctx {a.ctx}")
    with Pool(8, initializer=init, initargs=(recs,)) as p:
        done = sum(1 for _ in p.imap_unordered(job, jobs, chunksize=16) if _)
    for name, rows in rows_all.items():
        with open(os.path.join(DATA, f'{name}.jsonl'), 'w', encoding='utf-8') as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"done {done}/{len(jobs)}; jsonl updated with img_v3")

if __name__ == '__main__':
    main()
