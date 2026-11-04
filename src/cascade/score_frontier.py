# -*- coding: utf-8 -*-
"""score_frontier.py — score the frontier-VLM blind pass-1 outputs with the SAME validator/cross-check used for the
3B/7B runs: p_grid κ vs CSV truth, marker-following under perturbation, pattern agreement, visual-flag distribution."""
import json, os, sys, io, re, collections
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi
import protocol_v2 as P
def perturbed_point(b, px, py):
    w, h = b['width'], b['height']; gx = min(2, int(3*px/max(w, 1))); gy = min(2, int(3*py/max(h, 1)))
    nx, ny = 2-gx, 2-gy
    if (nx, ny) == (gx, gy): nx = (gx+1) % 3
    return int(b['x'] + (nx+0.5)*w/3), int(b['y'] + (ny+0.5)*h/3)
L = r"<PROJECT_ROOT>\results\cases_lock"; F = r"<PROJECT_ROOT>\results\vlm_out\frontier"
def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s)):
        try: return json.loads(cand)
        except Exception: pass
    return None
def cell_of(b, x, y):
    gx = min(2, int(3*(x-b['x'])/max(b['width'], 1))); gy = min(2, int(3*(y-b['y'])/max(b['height'], 1))); return P.GRID_KO[gy*3+gx]
recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]
rows = {'v2': [], 'v2_markerpert': []}
for it in idx:
    f = json.load(open(os.path.join(L, it['case_id'], 'facts.json'), encoding='utf-8')); b = f['bbox']
    truth = P.grid_ko(f['thermal']['hotspot_position_in_target'])
    T = parse_csv(recs[f['frame_key']]['csv']); roi, (x, y, x2, y2) = clip_roi(T, b); py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
    pert = cell_of(b, *perturbed_point(b, px, py))
    for cond in rows:
        fp = os.path.join(F, cond, it['case_id'] + '.json')
        if not os.path.exists(fp): continue
        raw = json.load(open(fp, encoding='utf-8')).get('raw', ''); obs, viol = P.validate_pass1(parse_json(raw), f); cc = P.cross_check(obs, f)
        rows[cond].append(dict(case=it['case_id'], stratum=it['stratum'], pg=obs['p_grid'], truth=truth, pert=pert, sp=cc['spatial_check'], pat=cc['pattern_check'],
                               hp=obs['heat_pattern'], vf=obs['visual_flags'], part=obs['part'], eqp=obs['equipment_plausible'], digits=bool(re.search(r'\d', raw)), viol=viol))
out = []
for cond, R in rows.items():
    if not R: out.append(f"{cond}: no outputs yet"); continue
    n = len(R); sp = collections.Counter(r['sp'] for r in R); pairs = [(r['pg'], r['truth']) for r in R if r['pg'] in P.GRID_XY]
    line = f"[frontier/{cond}] n={n} | p_grid exact {sp['exact']/n:.2f} adj+ {(sp['exact']+sp['adjacent'])/n:.2f} mismatch {sp['mismatch']/n:.2f} | κ vs truth {kappa(pairs):.3f}"
    if cond == 'v2_markerpert':
        line += f" | follows moved marker {np.mean([r['pg']==r['pert'] for r in R]):.2f} (κ vs moved {kappa([(r['pg'], r['pert']) for r in R if r['pg'] in P.GRID_XY]):.3f})"
    line += f" | pattern match {np.mean([r['pat']=='match' for r in R]):.2f} | digits {np.mean([r['digits'] for r in R]):.2f} | eq_plausible {dict(collections.Counter(r['eqp'] for r in R))}"
    out.append(line); out.append("   visual flags: " + str(dict(collections.Counter(v for r in R for v in r['vf']))))
    out.append("   answers: " + str(collections.Counter(r['pg'] for r in R).most_common(6)))
ref = "reference (same locked set): Qwen2.5-VL-3B exact 0.11 κ 0.09 follows-marker 0.10 | 7B κ 0.09 follows 0.04 | 3B+LoRA(marked task) κ 0.73"
out.append(ref); txt = "\n".join(out); print(txt); open(os.path.join(F, 'frontier_scores.txt'), 'w', encoding='utf-8').write(txt)
