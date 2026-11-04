# -*- coding: utf-8 -*-
"""marker_follow_recompute.py — recompute, from SAVED raw pass-1 outputs and with the corrected (longest-first) cell parser,
the blind-observation metrics reported in the paper for every image-based run:
  exact / kappa vs the radiometric truth cell (condition v2), fraction of answers that follow the displaced marker and
  fraction unchanged after displacement (v2 vs v2_markerpert), share of the two most frequent answers.
usage: python marker_follow_recompute.py   -> results/marker_follow_recompute.txt
"""
import json, os, sys, io, re, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np
from common import MAN, parse_csv, clip_roi
import protocol_v2 as P
R = r"<PROJECT_ROOT>\results"
RUNS = [('v2lock', 'cases_lock', 'Qwen2.5-VL-3B zero-shot, locked set'), ('v2_7b', 'cases', 'Qwen2.5-VL-7B 4-bit, development set'),
        ('v2lora', 'cases_lock', 'Qwen2.5-VL-3B + marker-reading adapter, locked set'), ('v2', 'cases', 'Qwen2.5-VL-3B zero-shot, development set'),
        ('frontier', 'cases_lock', 'Frontier (Claude Fable 5.1), locked set — SEPT RUN, agents had shell/image tools (superseded)'),
        ('frontier_clean', 'cases_lock', 'Frontier (Claude Fable 5.1), locked set — tool-free re-run, displaced condition in separate sessions'),
        ('subject_gemini_3_5_flash_lite', 'cases_lock', 'Gemini 3.5 Flash-Lite, locked set')]
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for c in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(c)
        except Exception: pass
    return None
def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')
def perturbed_point(b, px, py):
    w, h = b['width'], b['height']; gx = min(2, int(3*px/max(w, 1))); gy = min(2, int(3*py/max(h, 1))); nx, ny = 2-gx, 2-gy
    if (nx, ny) == (gx, gy): nx = (gx+1) % 3
    return int(b['x'] + (nx+0.5)*w/3), int(b['y'] + (ny+0.5)*h/3)
def cell_of(b, x, y):
    gx = min(2, int(3*(x-b['x'])/max(b['width'], 1))); gy = min(2, int(3*(y-b['y'])/max(b['height'], 1))); return P.GRID_KO[gy*3+gx]
def raw_of(fp):
    j = json.load(open(fp, encoding='utf-8')); r = j.get('raw', '')
    return r.get('pass1', '') if isinstance(r, dict) else r
recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
out = []
for tag, cdir, label in RUNS:
    root = os.path.join(R, 'vlm_out', tag); L = os.path.join(R, cdir)
    if not os.path.isdir(os.path.join(root, 'v2')): out.append(f"[{tag}] no v2 outputs"); continue
    idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]
    rows = []
    for it in idx:
        cid = it['case_id']; f = json.load(open(os.path.join(L, cid, 'facts.json'), encoding='utf-8')); b = f['bbox']; truth = P.grid_ko(f['thermal']['hotspot_position_in_target'])
        p1 = os.path.join(root, 'v2', cid + '.json'); p2 = os.path.join(root, 'v2_markerpert', cid + '.json')
        if not os.path.exists(p1): continue
        T = parse_csv(recs[f['frame_key']]['csv']); roi, _ = clip_roi(T, b); py, px = np.unravel_index(int(np.argmax(roi)), roi.shape); pert = cell_of(b, *perturbed_point(b, px, py))
        o1, _ = P.validate_pass1(parse_json(raw_of(p1)), f); g1 = o1['p_grid']
        g2 = P.validate_pass1(parse_json(raw_of(p2)), f)[0]['p_grid'] if os.path.exists(p2) else None
        rows.append(dict(truth=truth, pert=pert, g1=g1, g2=g2))
    n = len(rows); ok = [r for r in rows if r['g1'] in P.GRID_XY]
    exact = np.mean([r['g1'] == r['truth'] for r in rows]); k = kappa([(r['g1'], r['truth']) for r in ok])
    cnt = collections.Counter(r['g1'] for r in rows); top2 = sum(c for _, c in cnt.most_common(2)) / n
    line = f"[{tag}] {label}: n={n} | exact {exact:.2f} | kappa vs truth {k:.3f} | top-2 answers share {top2:.2f} {cnt.most_common(2)}"
    pr = [r for r in rows if r['g2'] is not None]
    if pr:
        okp = [r for r in pr if r['g2'] in P.GRID_XY]
        line += f" | displaced: n={len(pr)} follows moved marker {np.mean([r['g2'] == r['pert'] for r in pr]):.2f} (kappa vs moved {kappa([(r['g2'], r['pert']) for r in okp]):.3f}) | unchanged after move {np.mean([r['g1'] == r['g2'] for r in pr]):.2f}"
        three = collections.Counter('follows displaced marker' if r['g2'] == r['pert'] else ('reports true hot cell' if r['g2'] == r['truth'] else 'other') for r in pr)
        line += f" | three-way: {dict(three)}"
    out.append(line)
txt = "\n".join(out); print(txt); open(os.path.join(R, 'marker_follow_recompute.txt'), 'w', encoding='utf-8').write(txt + "\n\n(parser: longest-first cell matching, 2026-10-29; earlier files v2lock_marker_follow.txt / marker_follow_3b_vs_7b.txt used the substring-order parser)")
