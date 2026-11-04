# -*- coding: utf-8 -*-
"""score_fact_control.py — balanced scoring of the FACT-FALSIFICATION control.
Two arms per model, same image (marker at the true radiometric peak), same prompt:
  factpert : the 'measured record' names the WRONG cell (diagonally opposite)  -> a grounded model should REJECT it
  facttrue : the 'measured record' names the TRUE cell                          -> a grounded model should ACCEPT it
A model that rejects everything (or accepts everything) is not grounded; the 2x2 separates grounding from a blanket policy.
usage: python score_fact_control.py  -> results/fact_control_balanced.txt
"""
import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import protocol_v2 as P
F = r"<PROJECT_ROOT>\results\vlm_out\frontier"
ARMS = [('Frontier (Claude Fable 5.1), tool-free agents', 'factpert', 'facttrue'),
        ('Gemini 3.5 Flash-Lite (REST)', 'factpert_gemini_3_5_flash_lite', 'facttrue_gemini_3_5_flash_lite')]
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    for c in (m.group(0), m.group(0).replace('\n', ' ')):
        try: return json.loads(c)
        except Exception: pass
    return None
def norm_cell(s):
    s = str(s or '')
    for g in sorted(P.GRID_KO, key=len, reverse=True) + ['확인 불가']:
        if g in s: return g
    return '확인 불가'
def rows_of(d, key):
    out = []
    for cid, t in key.items():
        fp = os.path.join(d, cid + '.json')
        if not os.path.exists(fp): continue
        j = parse_json(json.load(open(fp, encoding='utf-8')).get('raw', '')) or {}
        c = str(j.get('consistent', '')); my = norm_cell(j.get('my_cell'))
        out.append(dict(cid=cid, accept=('예' in c and '아니' not in c), reject=('아니' in c), unsure=(not ('예' in c or '아니' in c)), my=my, truth=t['truth'], claimed=t['claimed']))
    return out
lines = []
kp = json.load(open(os.path.join(F, '_key_factpert.json'), encoding='utf-8')); kt = json.load(open(os.path.join(F, '_key_facttrue.json'), encoding='utf-8'))
assert all(v['truth'] == v['claimed'] for v in kt.values()) and all(v['truth'] != v['claimed'] for v in kp.values())
for label, dp, dt in ARMS:
    rp = rows_of(os.path.join(F, dp), kp); rt = rows_of(os.path.join(F, dt), kt)
    n1, n2 = len(rp), len(rt)
    a = lambda rows, k: sum(r[k] for r in rows)
    lines.append(f"[{label}]")
    lines.append(f"  wrong record (n={n1}): accepted {a(rp,'accept')}  rejected {a(rp,'reject')}  unsure {a(rp,'unsure')} | own cell = true {sum(r['my']==r['truth'] for r in rp)}  = planted {sum(r['my']==r['claimed'] for r in rp)}")
    lines.append(f"  true record  (n={n2}): accepted {a(rt,'accept')}  rejected {a(rt,'reject')}  unsure {a(rt,'unsure')} | own cell = true {sum(r['my']==r['truth'] for r in rt)}")
    tpr = a(rt, 'accept') / max(n2, 1); fpr = a(rp, 'accept') / max(n1, 1)
    lines.append(f"  acceptance rate: true record {tpr:.2f} vs wrong record {fpr:.2f}  (difference = discrimination {tpr-fpr:+.2f}; a blanket policy gives 0)")
    wrong_rej_cases = [r['cid'] for r in rt if r['reject']]
    if wrong_rej_cases: lines.append(f"  true records wrongly rejected: {wrong_rej_cases}")
txt = "\n".join(lines) + "\n\n(same image and prompt in both arms; the record text is the only difference; frontier agents audited tool-free, 2026-10-29/30)"
print(txt); open(r"<PROJECT_ROOT>\results\fact_control_balanced.txt", 'w', encoding='utf-8').write(txt + "\n")
