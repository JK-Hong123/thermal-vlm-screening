# -*- coding: utf-8 -*-
"""
radiometric_baseline.py — Phase 1 motivation + first baseline.
Question: is 'danger' separable by ABSOLUTE temperature (a threshold), or is it CONTEXTUAL
(relative to siblings / scene)? Answered via per-feature AUROC on radiometric CSV temperatures.
This both (a) motivates a learned/context model over a threshold, and (b) is the first baseline.
"""
import json, random, sys, io, collections
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

MANIFEST = r"<PROJECT_ROOT>\data\manifests\manifest.jsonl"
random.seed(0)
N_PER = 300  # images per status to sample

def parse_csv(cf):
    try: lines = open(cf, encoding='utf-8').read().splitlines()
    except: return None
    rows = []
    for l in lines[5:]:
        try: row = [float(p) for p in l.split(';') if p.strip() != '']
        except: continue
        if len(row) >= 600: rows.append(row[:640])
    return np.array(rows) if len(rows) >= 400 else None

def auroc(scores, labels):
    s = np.array(scores); y = np.array(labels)
    order = s.argsort(); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(s)+1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0: return float('nan')
    return (ranks[y == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

# load manifest, stratify by status
rows = [json.loads(l) for l in open(MANIFEST, encoding='utf-8')]
by_status = {'danger': [], 'normal': []}
for r in rows:
    if r.get('csv') and r['status'] in by_status: by_status[r['status']].append(r)
samp = random.sample(by_status['danger'], N_PER) + random.sample(by_status['normal'], N_PER)
random.shuffle(samp)
print(f"sampling {len(samp)} images ({N_PER} danger + {N_PER} normal)...")

feats = collections.defaultdict(list); labels = []
n_read = 0
for r in samp:
    T = parse_csv(r['csv'])
    if T is None: continue
    n_read += 1
    H, W = T.shape; scene_med = np.median(T)
    anns = r['annotations']
    # sibling context: mean temp of each box
    box_means = []
    for a in anns:
        b = a['bbox'];
        if not b: box_means.append(scene_med); continue
        x=max(0,b['x']); y=max(0,b['y']); x2=min(W,x+b['width']); y2=min(H,y+b['height'])
        roi = T[y:y2, x:x2]
        box_means.append(float(roi.mean()) if roi.size else scene_med)
    for i, a in enumerate(anns):
        st = a.get('status');
        if st not in ('danger','normal'): continue
        b = a['bbox']
        if not b: continue
        x=max(0,b['x']); y=max(0,b['y']); x2=min(W,x+b['width']); y2=min(H,y+b['height'])
        roi = T[y:y2, x:x2]
        if roi.size == 0: continue
        mx = float(roi.max()); mean = float(roi.mean())
        others = [box_means[j] for j in range(len(anns)) if j != i]
        sib_ref = np.median(others) if others else scene_med
        feats['abs_max'].append(mx)
        feats['abs_mean'].append(mean)
        feats['rise_over_scene'].append(mx - scene_med)
        feats['contrast_vs_siblings'].append(mx - sib_ref)   # relative/contextual cue
        labels.append(1 if st == 'danger' else 0)
    if n_read % 100 == 0: print(f"  ...{n_read} csv read")

labels = np.array(labels)
print(f"\ncsv read: {n_read}   bbox instances: {len(labels)}  (danger {int(labels.sum())}, normal {int((labels==0).sum())})")
print("\n=== per-feature AUROC (danger vs normal, bbox-level) ===")
for k in ['abs_max','abs_mean','rise_over_scene','contrast_vs_siblings']:
    print(f"  {k:22s} AUROC = {auroc(feats[k], labels):.3f}")
print("\nInterpretation: AUROC~0.5 = that cue does NOT separate danger.  Higher = stronger cue.")
