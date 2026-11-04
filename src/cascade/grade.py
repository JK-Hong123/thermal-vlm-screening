# -*- coding: utf-8 -*-
"""
grade.py — automatic grounding/consistency grading of Stage-2 outputs against facts.json.
Metrics per condition (and per stratum):
  json_ok          structured output parsed
  num_halluc_rate  fraction of numeric tokens in output NOT present in supplied facts (tolerance 0.15; ints/percent/prob handled)
  any_halluc       case has >=1 hallucinated number
  verdict_gate     output verdict agrees with RF gate (flag->이상, pass->정상)   [protocol says VLM must not override gate]
  verdict_label    output verdict agrees with hidden ground truth (info only)
  sev_ok           severity level == band from ΔT table (or L0 when gate passes)
  comp_ok          component names the equipment (string match, KO)
  no_overclaim     for gate-pass cases: output does not assert 과열/이상 as fact
usage: python grade.py --tag v0 --bands ..\..\prompts\severity_bands_v0.json
"""
import json, os, re, sys, io, argparse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
OUT = r"<PROJECT_ROOT>\results\vlm_out"
NUM = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.]*[a-zA-Z_])')

def band_of(dT, bands):
    for b in bands['bands']:
        if b['dT_min'] <= dT < b['dT_max']: return b['level']
    return bands['bands'][-1]['level']

def allowed_numbers(f):
    vals = set()
    def add(v):
        if isinstance(v, bool) or v is None: return
        if isinstance(v, (int, float)): vals.add(round(float(v), 2))
        elif isinstance(v, dict): [add(x) for x in v.values()]
        elif isinstance(v, list): [add(x) for x in v]
    add(f['thermal']); add(f['gate']); add(f.get('calib')); add(f['n_components_in_frame']); add(f['bbox'])
    # derived numbers a faithful report may legitimately state
    t = f['thermal']
    vals |= {round(t['t_max']-t['t_mean'], 1), round(t['scene_max']-t['t_max'], 1), round(f['gate']['rf_prob']*100, 1), round(f['gate']['rf_prob']*100)}
    vals |= {1.0, 2.0, 3.0, 4.0, 10.0, 15.0, 40.0, 0.0, 5.5}  # band edges / list indices / rule threshold are template facts
    return vals

def nums_in(text):
    out = []
    for m in NUM.finditer(text):
        try: out.append(float(m.group(1)))
        except: pass
    return out

def flatten(p):
    if p is None: return ''
    return json.dumps(p, ensure_ascii=False)

def strip_severity_tokens(s):
    return re.sub(r'L[0-4]', '', s)

def grade_one(f, rec, bands):
    p = rec['parsed']; txt = flatten(p) if p else (rec['raw'] or '')
    allowed = allowed_numbers(f)
    nums = nums_in(strip_severity_tokens(txt))
    halluc = [n for n in nums if not any(abs(n-a) <= 0.15 for a in allowed)]
    g = f['gate']; t = f['thermal']
    gate_flag = g['rf_verdict'] == 'flag'
    verdict = (p or {}).get('verdict', '') or ''
    v_abn = ('이상' in verdict) and ('정상' not in verdict) and ('보류' not in verdict)
    v_norm = ('정상' in verdict) and ('이상' not in verdict)
    v_hold = '보류' in verdict
    sev_raw = str((p or {}).get("severity", "")); mm = re.search(r"L[0-4]", sev_raw); sev = mm.group(0) if mm else sev_raw.strip()
    want_sev = band_of(t['dT_scene_max'], bands) if gate_flag else 'L0'
    comp = str((p or {}).get('component', ''))
    eq = f['equipment']; eq_core = re.sub(r'\(.*?\)', '', eq).split()[0]
    body = ' '.join(str((p or {}).get(k, '')) for k in ('thermal_finding', 'probable_cause', 'recommended_action'))
    overclaim = (not gate_flag) and bool(re.search(r'(과열|이상 (있|발생|확인)|고장|위험)', body)) and not re.search(r'(아니|없|정상|않)', body)
    return dict(json_ok=p is not None, n_nums=len(nums), n_halluc=len(halluc), halluc_vals=halluc[:6],
                any_halluc=len(halluc) > 0,
                verdict_gate=(v_abn if gate_flag else (v_norm or v_hold)),
                verdict_label=(v_abn if f['hidden']['label'] == 'danger' else v_norm),
                sev_ok=(sev == want_sev), sev_out=sev, sev_want=want_sev,
                comp_ok=(eq_core in comp) or (eq in comp), no_overclaim=(not overclaim))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--bands', required=True)
    a = ap.parse_args(); bands = json.load(open(a.bands, encoding='utf-8'))
    idx = {i['case_id']: i for i in (json.loads(l) for l in open(os.path.join(CASES, 'index.jsonl'), encoding='utf-8'))}
    root = os.path.join(OUT, a.tag); conds = [c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))]
    summary = {}
    for cond in conds:
        rows = []
        for fn in sorted(os.listdir(os.path.join(root, cond))):
            if not fn.endswith('.json'): continue
            rec = json.load(open(os.path.join(root, cond, fn), encoding='utf-8')); cid = rec['case_id']
            f = json.load(open(os.path.join(CASES, cid, 'facts.json'), encoding='utf-8'))
            gr = grade_one(f, rec, bands); gr['case_id'] = cid; gr['stratum'] = idx[cid]['stratum']; gr['sec'] = rec.get('sec', 0); rows.append(gr)
        if not rows: continue
        keys = ['json_ok', 'any_halluc', 'verdict_gate', 'verdict_label', 'sev_ok', 'comp_ok', 'no_overclaim']
        agg = {k: sum(r[k] for r in rows)/len(rows) for k in keys}
        agg['num_halluc_rate'] = sum(r['n_halluc'] for r in rows)/max(sum(r['n_nums'] for r in rows), 1)
        agg['mean_sec'] = sum(r['sec'] for r in rows)/len(rows); agg['n'] = len(rows)
        summary[cond] = {'overall': agg, 'by_stratum': {}}
        for s in sorted(set(r['stratum'] for r in rows)):
            rs = [r for r in rows if r['stratum'] == s]
            summary[cond]['by_stratum'][s] = {k: round(sum(r[k] for r in rs)/len(rs), 2) for k in keys} | {'n': len(rs)}
        json.dump(rows, open(os.path.join(root, f'grades_{cond}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(summary, open(os.path.join(root, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\n===== {a.tag}: automatic grounding grades =====")
    print(f"{'cond':12s} {'n':>3s} {'json':>5s} {'anyHal':>7s} {'numHal%':>8s} {'v=gate':>7s} {'v=label':>8s} {'sev':>5s} {'comp':>5s} {'noOver':>7s} {'sec':>5s}")
    for cond, s in summary.items():
        o = s['overall']
        print(f"{cond:12s} {o['n']:3d} {o['json_ok']:5.2f} {o['any_halluc']:7.2f} {100*o['num_halluc_rate']:8.1f} {o['verdict_gate']:7.2f} {o['verdict_label']:8.2f} {o['sev_ok']:5.2f} {o['comp_ok']:5.2f} {o['no_overclaim']:7.2f} {o['mean_sec']:5.1f}")
    for cond, s in summary.items():
        if cond == 'template': continue
        print(f"\n  -- {cond} by stratum (json/anyHal/v=gate/v=label/sev/noOver) --")
        for st, v in s['by_stratum'].items():
            print(f"    {st:18s} n={v['n']:2d}  {v['json_ok']:.2f} {v['any_halluc']:.2f} {v['verdict_gate']:.2f} {v['verdict_label']:.2f} {v['sev_ok']:.2f} {v['no_overclaim']:.2f}")

if __name__ == '__main__':
    main()
