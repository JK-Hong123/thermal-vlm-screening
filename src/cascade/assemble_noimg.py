# -*- coding: utf-8 -*-
"""assemble_noimg.py — assemble full reports for the NO-IMAGE frontier control arm from results/vlm_out/frontier/noimg/<case>.json
(raw model JSON with pass-1-style descriptive fields + pass-2 fields), using the SAME validators and assembly as every
other arm; writes results/vlm_out/v2frontier/noimg/<case>.{json,txt}."""
import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import protocol_v2 as P
from render_report import render
L = r"<PROJECT_ROOT>\results\cases_lock"; F = r"<PROJECT_ROOT>\results\vlm_out\frontier\noimg"
O = r"<PROJECT_ROOT>\results\vlm_out\v2frontier\noimg"; os.makedirs(O, exist_ok=True)
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
    p = os.path.join(F, cid + '.json')
    if not os.path.exists(p): miss += 1; continue
    sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf); viol = []
    raw = json.load(open(p, encoding='utf-8'))['raw']; j = parse_json(raw) or {}
    # pass-1-style fields: no image was shown, so the marker cell comes from the measurement, visual flags are fixed
    obs_in = {'equipment_plausible': '확인 불가', 'part': j.get('part', ''), 'p_grid': P.grid_ko(f['thermal']['hotspot_position_in_target']),
              'heat_pattern': j.get('heat_pattern', ''), 'visual_flags': ['가시광 없음'], 'image_quality': '보통', 'part_location_ko': j.get('part_location_ko', '')}
    obs, v1 = P.validate_pass1(obs_in, f); viol += v1; cc = P.cross_check(obs, f)
    p2f, v2 = P.validate_pass2(j, f, sev, ct, obs, cc); viol += v2
    rep = P.assemble(f, sev, ct, conf, qf, obs, cc, p2f, viol, {'noimg': raw})
    json.dump({'case_id': cid, 'cond': 'noimg', 'raw': {'noimg': raw}, 'report': rep}, open(os.path.join(O, cid + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(O, cid + '.txt'), 'w', encoding='utf-8').write(render(rep)); n += 1; nviol += len(rep['validation']['violations'])
print(f"assembled {n} no-image reports (missing {miss}); mean validation violations {nviol/max(n,1):.2f}")
