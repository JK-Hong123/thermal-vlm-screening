# -*- coding: utf-8 -*-
"""revalidate_v2.py — re-run GTD-v2 validation/assembly from SAVED raw VLM outputs (no GPU).
usage: python revalidate_v2.py --tag v2 [--cases_dir DIR] [--cond v2]"""
import json, os, sys, io, argparse, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
import protocol_v2 as P
OUT = r"<PROJECT_ROOT>\results\vlm_out"

def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(cand)
        except Exception: pass
    return None

EMPTY_OBS = {'equipment_plausible': '확인 불가', 'part': '확인 불가', 'p_grid': '확인 불가', 'heat_pattern': '불명', 'visual_flags': ['특이 없음'], 'image_quality': '보통', 'part_location_ko': ''}
EMPTY_P2 = {'cause_hypothesis': ['확인 불가'], 'cause_basis': [], 'field_checks': [], 'operator_note_ko': ''}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v2'); ap.add_argument('--cases_dir', default=CASES); ap.add_argument('--cond', default=None)
    a = ap.parse_args(); root = os.path.join(OUT, a.tag); n = 0
    for cond in sorted(os.listdir(root)):
        cd = os.path.join(root, cond)
        if not os.path.isdir(cd) or cond in ('template', 'a0plus'): continue
        if a.cond and cond != a.cond: continue
        for fn in sorted(os.listdir(cd)):
            if not fn.endswith('.json'): continue
            fp = os.path.join(cd, fn); r = json.load(open(fp, encoding='utf-8')); raw = r.get('raw', {})
            f = json.load(open(os.path.join(a.cases_dir, r['case_id'], 'facts.json'), encoding='utf-8'))
            sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf); viol = []
            if raw.get('pass1') is not None: obs, v1 = P.validate_pass1(parse_json(raw['pass1']), f); viol += v1
            else: obs = dict(EMPTY_OBS)
            cc = P.cross_check(obs, f)
            if raw.get('pass2') is not None: p2f, v2 = P.validate_pass2(parse_json(raw['pass2']), f, sev, ct, obs, cc); viol += v2
            else: p2f = dict(EMPTY_P2)
            r['report'] = P.assemble(f, sev, ct, conf, qf, obs, cc, p2f, viol, raw)
            json.dump(r, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); n += 1
    print('revalidated', n, 'records')

if __name__ == '__main__':
    main()
