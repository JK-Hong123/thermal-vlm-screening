# -*- coding: utf-8 -*-
"""gemini_pass2.py — pass 2 (number-free reasoning from band-worded facts + the model's OWN validated pass-1 observation)
for the Gemini SUBJECT, text only, same template as every other arm (prompts/pass2_reason_v2.txt); then assemble full
GTD-v2 reports with the same validators as the 3B/frontier arms.
  raw  -> results/vlm_out/subject_gemini_3_5_flash_lite/pass2/<case>.json
  rep  -> results/vlm_out/v2gemini/v2/<case>.{json,txt}
usage: GOOGLE_API_KEY=... python gemini_pass2.py [--model gemini-3.5-flash-lite] [--assemble-only]
"""
import json, os, sys, re, time, argparse
import probe_api as PA          # probe_api and render_report each re-wrap sys.stdout as UTF-8; keep every wrapper referenced,
_keep1 = sys.stdout             # otherwise the replaced wrapper is garbage-collected and closes the shared underlying buffer
import protocol_v2 as P
from render_report import render
_keep2 = sys.stdout
L = r"<PROJECT_ROOT>\results\cases_lock"; P2T = r"<PROJECT_ROOT>\prompts\pass2_reason_v2.txt"
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for c in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(c)
        except Exception: pass
    return None
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='gemini-3.5-flash-lite'); ap.add_argument('--assemble-only', action='store_true'); a = ap.parse_args()
    S = os.path.join(r"<PROJECT_ROOT>\results\vlm_out", 'subject_' + a.model.replace('.', '_').replace('-', '_'))
    P2 = os.path.join(S, 'pass2'); os.makedirs(P2, exist_ok=True)
    O = os.path.join(r"<PROJECT_ROOT>\results\vlm_out", 'v2gemini', 'v2'); os.makedirs(O, exist_ok=True)
    t2 = open(P2T, encoding='utf-8').read(); key = os.environ.get('GOOGLE_API_KEY')
    idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]; ok = fail = n = 0; nviol = 0
    for it in idx:
        cid = it['case_id']; f = json.load(open(os.path.join(L, cid, 'facts.json'), encoding='utf-8'))
        p1 = os.path.join(S, 'v2', cid + '.json')
        if not os.path.exists(p1): print(cid, 'no pass-1'); continue
        raw1 = json.load(open(p1, encoding='utf-8'))['raw']
        sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf); viol = []
        obs, v1 = P.validate_pass1(parse_json(raw1), f); viol += v1; cc = P.cross_check(obs, f)
        fp2 = os.path.join(P2, cid + '.json')
        if not os.path.exists(fp2):
            if a.assemble_only: continue
            fv = P.facts_vlm(f, sev, ct, conf, obs)
            causes = ' / '.join(P.CAUSES_BULK if f['equipment'] in P.BULK else P.CAUSES_ELEC)
            prompt = t2.format(facts=fv, causes=causes, checks=' / '.join(P.CHECKS))
            for attempt in range(4):
                try:
                    raw2 = PA.call(a.model, key, prompt, []); json.dump({'raw': raw2, 'model': a.model, 'prompt': prompt}, open(fp2, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); ok += 1; print(cid, 'ok'); break
                except Exception as e:
                    msg = repr(e); print(cid, 'retry' if attempt < 3 else 'FAILED', msg[:100]); time.sleep(20 if '429' in msg else 5)
            else: fail += 1; continue
            time.sleep(4.2)
        raw2 = json.load(open(fp2, encoding='utf-8'))['raw']
        p2f, v2 = P.validate_pass2(parse_json(raw2), f, sev, ct, obs, cc); viol += v2
        rep = P.assemble(f, sev, ct, conf, qf, obs, cc, p2f, viol, {'pass1': raw1, 'pass2': raw2})
        json.dump({'case_id': cid, 'cond': 'v2gemini', 'raw': {'pass1': raw1, 'pass2': raw2}, 'report': rep}, open(os.path.join(O, cid + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        open(os.path.join(O, cid + '.txt'), 'w', encoding='utf-8').write(render(rep)); n += 1; nviol += len(rep['validation']['violations'])
    print(f"pass-2 calls: {ok} ok, {fail} failed | assembled {n} Gemini reports -> {O} | mean validation violations {nviol/max(n,1):.2f}")
if __name__ == '__main__': main()
