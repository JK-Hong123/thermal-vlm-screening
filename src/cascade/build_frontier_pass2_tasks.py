# -*- coding: utf-8 -*-
"""build_frontier_pass2_tasks.py — pass-2 tasks for the frontier SUBJECT from the TOOL-FREE pass-1 observations
(results/vlm_out/frontier_clean/v2), so that the judged frontier reports no longer rest on the superseded September
observations. Same template as every other arm (prompts/pass2_reason_v2.txt): band-worded facts + the model's own
validated observation; text only. Agents read tasks_pass2.json and write {"raw": "<answer>", "model": ...} to 'out'.
Then: python assemble_frontier.py --src frontier_clean --out v2frontier_clean
"""
import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import protocol_v2 as P
L = r"<PROJECT_ROOT>\results\cases_lock"; F = r"<PROJECT_ROOT>\results\vlm_out\frontier_clean"
P2T = r"<PROJECT_ROOT>\prompts\pass2_reason_v2.txt"; OUT = os.path.join(F, 'pass2'); os.makedirs(OUT, exist_ok=True)
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for c in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(c)
        except Exception: pass
    return None
t2 = open(P2T, encoding='utf-8').read(); idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]; tasks = []
for it in idx:
    cid = it['case_id']; f = json.load(open(os.path.join(L, cid, 'facts.json'), encoding='utf-8'))
    raw1 = json.load(open(os.path.join(F, 'v2', cid + '.json'), encoding='utf-8'))['raw']
    sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf)
    obs, _ = P.validate_pass1(parse_json(raw1), f)
    fv = P.facts_vlm(f, sev, ct, conf, obs); causes = ' / '.join(P.CAUSES_BULK if f['equipment'] in P.BULK else P.CAUSES_ELEC)
    tasks.append({'case_id': cid, 'prompt': t2.format(facts=fv, causes=causes, checks=' / '.join(P.CHECKS)), 'out': os.path.join(OUT, cid + '.json')})
json.dump(tasks, open(os.path.join(F, 'tasks_pass2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"wrote {len(tasks)} pass-2 tasks -> {os.path.join(F, 'tasks_pass2.json')}")
