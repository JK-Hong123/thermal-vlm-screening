# -*- coding: utf-8 -*-
"""grade_v1.py — automatic grading for protocol-v1 outputs.
Code-owned fields are consistent by construction; what we measure is the VLM's behaviour:
  json_ok, raw_digit_rate (VLM emitted digits before validation), raw_cjk_rate, fallback_rate (fields replaced by template),
  vocab_ok (heat_pattern/cause within closed vocab), consistency_answer distribution, latency,
  and 'informative' = fraction of VLM fields that survived validation AND are not '확인 불가'.
Per stratum too. usage: python grade_v1.py --tag v1"""
import json, os, sys, io, argparse, collections, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
from protocol_v1 import VLM_FIELDS, CJK
OUT = r"<PROJECT_ROOT>\results\vlm_out"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v1'); a = ap.parse_args()
    idx = {i['case_id']: i for i in (json.loads(l) for l in open(os.path.join(CASES, 'index.jsonl'), encoding='utf-8'))}
    root = os.path.join(OUT, a.tag); summ = {}
    for cond in sorted(os.listdir(root)):
        cd = os.path.join(root, cond)
        if not os.path.isdir(cd): continue
        rows = []
        for fn in sorted(os.listdir(cd)):
            if not fn.endswith('.json'): continue
            r = json.load(open(os.path.join(cd, fn), encoding='utf-8')); rep = r['report']; p = r.get('vlm_parsed') or {}
            raw_txt = json.dumps(p, ensure_ascii=False) if p else (r.get('raw') or '')
            fb = [v for v in rep['vlm_violations'] if not v.endswith('truncated')]
            inf = sum(1 for k in VLM_FIELDS if k in ('component_part', 'heat_description', 'visual_context', 'cause_hypothesis', 'operator_note', 'consistency_reason')
                      and not any(v.startswith(k + ':') for v in fb) and '확인 불가' not in str(rep['vlm'][k]) and '기본 문구' not in str(rep['vlm'][k]))
            rows.append(dict(case_id=r['case_id'], stratum=idx[r['case_id']]['stratum'], json_ok=bool(p) or cond == 'template',
                             raw_digits=bool(re.search(r'\d', raw_txt)) if cond != 'template' else False,
                             raw_cjk=bool(CJK.search(raw_txt)) if cond != 'template' else False,
                             n_fallback=len(fb), fallback_any=len(fb) > 0, informative=inf/6,
                             consistent=rep['vlm']['image_consistent_with_facts'], sec=r['sec'],
                             level=rep['severity_level'], verdict=rep['verdict'], ctype=rep['case_type'], conf=rep['confidence'], label=idx[r['case_id']]['label']))
        if not rows: continue
        n = len(rows)
        s = dict(n=n, json_ok=sum(x['json_ok'] for x in rows)/n, raw_digits=sum(x['raw_digits'] for x in rows)/n, raw_cjk=sum(x['raw_cjk'] for x in rows)/n,
                 fallback_any=sum(x['fallback_any'] for x in rows)/n, mean_fallback_fields=sum(x['n_fallback'] for x in rows)/n,
                 informative=sum(x['informative'] for x in rows)/n, mean_sec=sum(x['sec'] for x in rows)/n,
                 consistent=dict(collections.Counter(x['consistent'] for x in rows)))
        summ[cond] = {'overall': s, 'by_stratum': {}}
        for st in sorted(set(x['stratum'] for x in rows)):
            rs = [x for x in rows if x['stratum'] == st]; m = len(rs)
            summ[cond]['by_stratum'][st] = dict(n=m, raw_digits=sum(x['raw_digits'] for x in rs)/m, fallback_any=sum(x['fallback_any'] for x in rs)/m,
                                                informative=sum(x['informative'] for x in rs)/m, consistent_yes=sum(x['consistent']=='예' for x in rs)/m)
        json.dump(rows, open(os.path.join(root, f'grades_{cond}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(summ, open(os.path.join(root, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\n===== {a.tag}: protocol-v1 automatic grades =====")
    print(f"{'cond':11s} {'n':>3s} {'json':>5s} {'rawDig':>7s} {'rawCJK':>7s} {'fbAny':>6s} {'fbFld':>6s} {'inform':>7s} {'sec':>5s}  consistent")
    for c, v in summ.items():
        o = v['overall']; print(f"{c:11s} {o['n']:3d} {o['json_ok']:5.2f} {o['raw_digits']:7.2f} {o['raw_cjk']:7.2f} {o['fallback_any']:6.2f} {o['mean_fallback_fields']:6.2f} {o['informative']:7.2f} {o['mean_sec']:5.1f}  {o['consistent']}")
    for c, v in summ.items():
        if c == 'template': continue
        print(f"\n  -- {c} by stratum (rawDig / fbAny / inform / consistent=예) --")
        for st, w in v['by_stratum'].items(): print(f"    {st:18s} n={w['n']:2d}  {w['raw_digits']:.2f} {w['fallback_any']:.2f} {w['informative']:.2f} {w['consistent_yes']:.2f}")
    # code-owned layer summary (same for all conds): verdict/severity/case_type/confidence distribution vs hidden label
    rows = json.load(open(os.path.join(root, 'grades_template.json'), encoding='utf-8')) if os.path.exists(os.path.join(root, 'grades_template.json')) else []
    if rows:
        print("\n  -- code-owned layer (verdict=gate by construction): severity level by stratum, confidence, case_type --")
        for st in sorted(set(x['stratum'] for x in rows)):
            rs = [x for x in rows if x['stratum'] == st]
            print(f"    {st:18s} levels={dict(collections.Counter(x['level'] for x in rs))} conf={dict(collections.Counter(x['conf'] for x in rs))} type={dict(collections.Counter(x['ctype'] for x in rs))}")

if __name__ == '__main__':
    main()
