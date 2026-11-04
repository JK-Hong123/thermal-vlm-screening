# -*- coding: utf-8 -*-
"""score_judge_grounding.py — for a judge output directory (forced-grounding protocol), score the judge's OWN stage-1 marker
reading (my_marker_cell) against the radiometric truth cell of each locked case, and summarise preference / ungrounded
claims by arm. usage: python score_judge_grounding.py --dir results/judge_crossfamily_gemini_full2 [--key _key_ext80.json]
"""
import json, os, sys, io, re, argparse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import protocol_v2 as P
R = r"<PROJECT_ROOT>\results"; L = os.path.join(R, 'cases_lock')
ROW = {'상단': 0, '중단': 1, '하단': 2}; COL = {'좌': 0, '중': 1, '우': 2}
def norm_cell(s):
    s = str(s or '').replace(' ', '').replace('-', '').replace('측', '').replace('앙', '')
    if '중' in s and len(s) <= 2: return P.GRID_KO[4]          # '중앙'/'중'
    r = next((v for k, v in ROW.items() if s.startswith(k)), None); c = next((v for k, v in COL.items() if k in s[2:]), None)
    return P.GRID_KO[r*3 + c] if r is not None and c is not None else None
def kappa(pairs):
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--key', default=None); a = ap.parse_args()
d = a.dir if os.path.isabs(a.dir) else os.path.join(R, os.path.basename(a.dir))
keyf = a.key or (os.path.join(R, 'judge_frontier', '_key_ext80.json') if len([f for f in os.listdir(d) if f.startswith('case')]) > 40 else os.path.join(R, 'judge_frontier', '_key.json'))
key = json.load(open(keyf, encoding='utf-8')); pairs = []; adj = 0; unparsed = 0; pref = collections.Counter(); ung = collections.defaultdict(list)
for cid in sorted(key):
    fp = os.path.join(d, cid + '.json')
    if not os.path.exists(fp): continue
    j = json.load(open(fp, encoding='utf-8')); f = json.load(open(os.path.join(L, cid, 'facts.json'), encoding='utf-8'))
    truth = P.grid_ko(f['thermal']['hotspot_position_in_target']); mine = norm_cell(j.get('my_marker_cell'))
    if mine is None: unparsed += 1
    else:
        pairs.append((mine, truth)); (x1, y1), (x2, y2) = P.GRID_XY[mine], P.GRID_XY[truth]; adj += max(abs(x1-x2), abs(y1-y2)) <= 1
    p = j.get('preferred_report'); arm = 'tie' if p == 'tie' else key[cid].get(p, '?'); pref[arm] += 1
    for letter in 'AB': ung[key[cid][letter]].append(int(j.get(f'ungrounded_claims_{letter}', 0) or 0))
n = len(pairs); ex = sum(a == b for a, b in pairs)
print(f"[{os.path.basename(d)}] n={n+unparsed} | judge's own marker cell vs truth: exact {ex}/{n} = {ex/max(n,1):.2f}, adjacent-or-exact {adj/max(n,1):.2f}, kappa {kappa(pairs):.3f} (unparsed {unparsed})")
print(f"   preference: {dict(pref)} | ungrounded claims per report: " + ', '.join(f"{k} {sum(v)/len(v):.2f}" for k, v in ung.items()))
