# -*- coding: utf-8 -*-
"""add_boundary_facts.py — add a 'boundary' block to each case's facts.json (judge-driven fix):
peak distance to the box border, eroded-ROI t_max/dT, and the hottest temperature just outside the box.
usage: python add_boundary_facts.py [--cases_dir DIR] [--inset 3]"""
import json, os, sys, io, argparse
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES, MAN, parse_csv, clip_roi

ap = argparse.ArgumentParser(); ap.add_argument('--cases_dir', default=CASES); ap.add_argument('--inset', type=int, default=3)
a = ap.parse_args(); INSET = a.inset
recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
idx = [json.loads(l) for l in open(os.path.join(a.cases_dir, 'index.jsonl'), encoding='utf-8')]
n_susp = 0
for it in idx:
    fp = os.path.join(a.cases_dir, it['case_id'], 'facts.json'); f = json.load(open(fp, encoding='utf-8'))
    T = parse_csv(recs[f['frame_key']]['csv']); b = f['bbox']
    roi, (x, y, x2, y2) = clip_roi(T, b); H, W = T.shape
    py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
    border = int(min(px, py, roi.shape[1]-1-px, roi.shape[0]-1-py))
    if min(roi.shape) >= 2*INSET+2:
        inner = roi[INSET:-INSET, INSET:-INSET]; t_in = float(inner.max())
    else: t_in = float(roi.max())
    ox1, oy1, ox2, oy2 = max(0, x-INSET), max(0, y-INSET), min(W, x2+INSET), min(H, y2+INSET)
    outer = T[oy1:oy2, ox1:ox2]; mask = np.ones(outer.shape, bool)
    mask[y-oy1:y2-oy1, x-ox1:x2-ox1] = False
    t_out = float(outer[mask].max()) if mask.any() else float(np.median(T))
    sm = f['thermal']['scene_median']
    susp = bool(border < INSET and t_out >= float(roi.max())-0.5 and (float(roi.max())-t_in) > 2.0)
    n_susp += susp
    f['boundary'] = {'inset_px': INSET, 'peak_border_distance_px': border,
                     't_max_eroded': round(t_in, 1), 'dT_scene_eroded': round(t_in-sm, 1),
                     't_max_just_outside': round(t_out, 1),
                     'dT_drop_when_border_removed': round(float(roi.max())-t_in, 1),
                     'external_source_suspected': susp}
    json.dump(f, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"{len(idx)} cases updated in {a.cases_dir} | external-source suspected: {n_susp}")
