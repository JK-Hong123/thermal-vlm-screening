# -*- coding: utf-8 -*-
"""assemble_frontier.py — build full GTD-v2 reports for the frontier-VLM condition from the saved raw pass-1/pass-2
outputs (same validators/assembly as the 3B runs), write results/vlm_out/v2frontier/v2/<case>.{json,txt}."""
import json, os, sys, io, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import protocol_v2 as P
_keep = sys.stdout
from render_report import render
ap = argparse.ArgumentParser(); ap.add_argument('--src', default='frontier', help="raw-output directory under results/vlm_out (frontier = September run; frontier_clean = tool-free re-run)")
ap.add_argument('--out', default='v2frontier', help="assembled-report directory under results/vlm_out"); a = ap.parse_args()
L = r"<PROJECT_ROOT>\results\cases_lock"; F = os.path.join(r"<PROJECT_ROOT>\results\vlm_out", a.src)
O = os.path.join(r"<PROJECT_ROOT>\results\vlm_out", a.out, 'v2'); os.makedirs(O, exist_ok=True)
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for c in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(c)
        except Exception: pass
    return None
idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]; n = 0; miss = 0; nviol = 0
for it in idx:
    cid = it['case_id']; f = json.load(open(os.path.join(L, cid, 'facts.json'), encoding='utf-8'))
    p1 = os.path.join(F, 'v2', cid + '.json'); p2 = os.path.join(F, 'pass2', cid + '.json')
    if not (os.path.exists(p1) and os.path.exists(p2)): miss += 1; continue
    sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf); viol = []
    raw1 = json.load(open(p1, encoding='utf-8'))['raw']; raw2 = json.load(open(p2, encoding='utf-8'))['raw']
    obs, v1 = P.validate_pass1(parse_json(raw1), f); viol += v1; cc = P.cross_check(obs, f)
    p2f, v2 = P.validate_pass2(parse_json(raw2), f, sev, ct, obs, cc); viol += v2
    rep = P.assemble(f, sev, ct, conf, qf, obs, cc, p2f, viol, {'pass1': raw1, 'pass2': raw2})
    json.dump({'case_id': cid, 'cond': 'v2frontier', 'raw': {'pass1': raw1, 'pass2': raw2}, 'report': rep}, open(os.path.join(O, cid + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(O, cid + '.txt'), 'w', encoding='utf-8').write(render(rep)); n += 1; nviol += len(rep['validation']['violations'])
print(f"assembled {n} frontier reports (missing {miss}); mean validation violations {nviol/max(n,1):.2f}")
