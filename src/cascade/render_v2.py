# -*- coding: utf-8 -*-
"""render_v2.py — GTD-v2 renders per case: thermal_crop_v2.png (cyan circle+cross at argmax peak),
thermal_crop_v2_pert.png (marker deliberately drawn in a different 3x3 cell — marker-following control),
thermal_crop_v2_nomark.png (no marker — does the VLM find the hotspot itself?), rgb_crop_v2.png (no bbox; alignment unverified).
usage: python render_v2.py [--cases_dir DIR]"""
import json, os, sys, io, argparse
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import *
CYAN = (255, 255, 0)  # BGR

def crop_box(T, b, ctx=2.4):
    H, W = T.shape; bw, bh = b['width'], b['height']; cx = b['x']+bw/2; cy = b['y']+bh/2; s = max(bw, bh)*ctx
    return int(max(0, cx-s/2)), int(max(0, cy-s/2)), int(min(W, cx+s/2)), int(min(H, cy+s/2))

def draw_marker(col, pt):
    cv2.circle(col, pt, 9, CYAN, 2); cv2.drawMarker(col, pt, CYAN, cv2.MARKER_CROSS, 22, 2)

def render(T, b, mark, out, size=336):
    col = colorize(T); x, y, x2, y2 = crop_box(T, b)
    cv2.rectangle(col, (b['x'], b['y']), (b['x']+b['width'], b['y']+b['height']), (255, 255, 255), 2)
    if mark: draw_marker(col, mark)
    c = col[y:y2, x:x2]
    if c.size == 0: return None
    cv2.imwrite(out, cv2.resize(c, (size, size))); return (x, y, x2, y2)

def perturbed_point(b, px, py):
    """move the marker to a different 3x3 cell (opposite cell; center if peak is at a corner-ish edge)."""
    w, h = b['width'], b['height']; gx = min(2, int(3*px/max(w, 1))); gy = min(2, int(3*py/max(h, 1)))
    nx, ny = 2-gx, 2-gy
    if (nx, ny) == (gx, gy): nx = (gx+1) % 3
    return int(b['x'] + (nx+0.5)*w/3), int(b['y'] + (ny+0.5)*h/3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cases_dir', default=CASES); a = ap.parse_args()
    idx = [json.loads(l) for l in open(os.path.join(a.cases_dir, 'index.jsonl'), encoding='utf-8')]
    recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    for it in idx:
        d = os.path.join(a.cases_dir, it['case_id']); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
        rec = recs[f['frame_key']]; T = parse_csv(rec['csv']); b = f['bbox']
        roi, (x, y, x2, y2) = clip_roi(T, b); py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
        pk = (x+px, y+py)
        render(T, b, pk, os.path.join(d, 'thermal_crop_v2.png'))
        render(T, b, perturbed_point(b, px, py), os.path.join(d, 'thermal_crop_v2_pert.png'))
        render(T, b, None, os.path.join(d, 'thermal_crop_v2_nomark.png'))
        rgb = imread_u(rec.get('rgb')); box = crop_box(T, b)
        if rgb is not None and rgb.shape[0] >= box[3] and rgb.shape[1] >= box[2]:
            cv2.imwrite(os.path.join(d, 'rgb_crop_v2.png'), cv2.resize(rgb[box[1]:box[3], box[0]:box[2]], (336, 336)))
        f['images'].update({'thermal_crop_v2': 'thermal_crop_v2.png', 'thermal_crop_v2_pert': 'thermal_crop_v2_pert.png',
                            'thermal_crop_v2_nomark': 'thermal_crop_v2_nomark.png', 'rgb_crop_v2': 'rgb_crop_v2.png' if rgb is not None else None})
        json.dump(f, open(os.path.join(d, 'facts.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"rendered v2 images for {len(idx)} cases in {a.cases_dir}")

if __name__ == '__main__':
    main()
