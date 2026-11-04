# -*- coding: utf-8 -*-
"""grade_v2.py — automatic GTD-v2 metrics (E1, E3, E4, E5, E7-partial, E13) per condition and stratum.
usage: python grade_v2.py --tag v2 [--cases_dir DIR]"""
import json, os, sys, io, argparse, collections, re
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
from protocol_v2 import CJK, GRID_XY
OUT = r"<PROJECT_ROOT>\results\vlm_out"

def kappa(pairs):
    """Cohen's kappa for list of (pred, truth) categorical pairs."""
    if not pairs: return float('nan')
    cats = sorted(set([p for p, _ in pairs] + [t for _, t in pairs])); n = len(pairs)
    po = sum(p == t for p, t in pairs)/n
    pe = sum((sum(p == c for p, _ in pairs)/n)*(sum(t == c for _, t in pairs)/n) for c in cats)
    return (po-pe)/(1-pe) if pe < 1 else float('nan')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v2'); ap.add_argument('--cases_dir', default=CASES); a = ap.parse_args()
    idx = {i['case_id']: i for i in (json.loads(l) for l in open(os.path.join(a.cases_dir, 'index.jsonl'), encoding='utf-8'))}
    root = os.path.join(OUT, a.tag); summ = {}
    for cond in sorted(os.listdir(root)):
        cd = os.path.join(root, cond)
        if not os.path.isdir(cd): continue
        rows = []
        for fn in sorted(os.listdir(cd)):
            if not fn.endswith('.json'): continue
            r = json.load(open(os.path.join(cd, fn), encoding='utf-8')); rep = r['report']; raw = r.get('raw', {})
            f = json.load(open(os.path.join(a.cases_dir, r['case_id'], 'facts.json'), encoding='utf-8'))
            r1 = raw.get('pass1', '') or ''; r2 = raw.get('pass2', '') or ''
            viol = rep['validation']['violations']
            obs = rep['vlm_pass1']; p2 = rep['vlm_pass2']
            # marker-perturbation truth: where the marker was drawn (for v2_markerpert) -> compute perturbed cell
            b = f['bbox']; t = f['thermal']
            rows.append(dict(case_id=r['case_id'], stratum=idx[r['case_id']]['stratum'], label=idx[r['case_id']]['label'], equipment=f['equipment'],
                json1=bool(r1) and rep['vlm_pass1']['p_grid'] != '확인 불가' or ('pass1:missing' not in viol and bool(r1)),
                json2=bool(r2) and 'pass2:missing' not in viol,
                dig1=bool(re.search(r'\d', r1)), dig2=bool(re.search(r'\d', r2)), cjk=bool(CJK.search(r1 + r2)),
                oov=sum(1 for v in viol if v.endswith(':oov')), n_viol=len(viol), fallback=len(rep['validation']['fallback_fields']),
                spatial=rep['spatial_check'], p_grid=obs['p_grid'], truth=rep['vlm_pass1'] and None, truth_grid=None,
                pattern=rep['pattern_check'], hp=obs['heat_pattern'], pc=None,
                part_unknown=obs['part'] == '확인 불가', eq_plaus=obs['equipment_plausible'],
                cause_unknown=p2['cause_hypothesis'] == ['확인 불가'], n_checks=len(p2['field_checks']), note_ok=bool(p2['operator_note_ko']) and not any(v.startswith('operator_note_ko') for v in viol),
                polarity=any('polarity' in v for v in viol), basis_viol=any(v.startswith('basis') for v in viol), compat_viol=any('incompatible' in v for v in viol),
                sec=sum(r.get('secs', {}).values()), level=rep['severity_level'], verdict=rep['verdict'], ctype=rep['case_type'], conf=rep['confidence'],
                visual=obs['visual_flags']))
        if not rows: continue
        n = len(rows); has1 = any(x['json1'] for x in rows) or cond.startswith('v2') and cond not in ('v2_pass2only',)
        sp = collections.Counter(x['spatial'] for x in rows); pt = collections.Counter(x['pattern'] for x in rows)
        # kappa for p_grid vs truth: read truth from facts
        pairs = []
        for x in rows:
            f = json.load(open(os.path.join(a.cases_dir, x['case_id'], 'facts.json'), encoding='utf-8'))
            from protocol_v2 import grid_ko
            tr = grid_ko(f['thermal']['hotspot_position_in_target']); x['truth_grid'] = tr
            if x['p_grid'] in GRID_XY: pairs.append((x['p_grid'], tr))
        s = dict(n=n, json1=np.mean([x['json1'] for x in rows]), json2=np.mean([x['json2'] for x in rows]),
                 dig1=np.mean([x['dig1'] for x in rows]), dig2=np.mean([x['dig2'] for x in rows]), cjk=np.mean([x['cjk'] for x in rows]),
                 oov_per_case=np.mean([x['oov'] for x in rows]), fallback_fields=np.mean([x['fallback'] for x in rows]),
                 spatial_exact=sp['exact']/n, spatial_adj=(sp['exact']+sp['adjacent'])/n, spatial_mismatch=sp['mismatch']/n, spatial_unknown=sp['unknown']/n,
                 p_grid_kappa=kappa(pairs), n_pairs=len(pairs),
                 pattern_match=pt['match']/n, pattern_differ=pt['differ']/n,
                 part_unknown=np.mean([x['part_unknown'] for x in rows]), eq_plaus=dict(collections.Counter(x['eq_plaus'] for x in rows)),
                 cause_unknown=np.mean([x['cause_unknown'] for x in rows]), note_ok=np.mean([x['note_ok'] for x in rows]),
                 polarity=np.mean([x['polarity'] for x in rows]), basis_viol=np.mean([x['basis_viol'] for x in rows]), compat_viol=np.mean([x['compat_viol'] for x in rows]),
                 mean_sec=np.mean([x['sec'] for x in rows]), visual=dict(collections.Counter(v for x in rows for v in x['visual'])))
        summ[cond] = {'overall': {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}, 'by_stratum': {}}
        for st in sorted(set(x['stratum'] for x in rows)):
            rs = [x for x in rows if x['stratum'] == st]; m = len(rs); spc = collections.Counter(x['spatial'] for x in rs)
            summ[cond]['by_stratum'][st] = dict(n=m, spatial_exact=round(spc['exact']/m, 2), spatial_adj=round((spc['exact']+spc['adjacent'])/m, 2),
                                                pattern_match=round(sum(x['pattern']=='match' for x in rs)/m, 2), dig=round(np.mean([x['dig1'] or x['dig2'] for x in rs]), 2),
                                                fallback=round(np.mean([x['fallback'] for x in rs]), 2), cause_unknown=round(np.mean([x['cause_unknown'] for x in rs]), 2), polarity=round(np.mean([x['polarity'] for x in rs]), 2))
        json.dump(rows, open(os.path.join(root, f'grades_{cond}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    json.dump(summ, open(os.path.join(root, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f"\n===== {a.tag}: GTD-v2 automatic grades =====")
    print(f"{'cond':14s} {'n':>3s} {'js1':>4s} {'js2':>4s} {'dig1':>5s} {'dig2':>5s} {'cjk':>4s} {'oov':>4s} {'fb':>4s} | {'gridEx':>6s} {'gridAdj':>7s} {'gridMis':>7s} {'κ':>5s} | {'patM':>5s} {'patD':>5s} | {'part?':>5s} {'cause?':>6s} {'noteOK':>6s} {'polar':>5s} {'basisV':>6s} {'compV':>5s} {'sec':>5s}")
    for c, v in summ.items():
        o = v['overall']
        print(f"{c:14s} {o['n']:3d} {o['json1']:4.2f} {o['json2']:4.2f} {o['dig1']:5.2f} {o['dig2']:5.2f} {o['cjk']:4.2f} {o['oov_per_case']:4.2f} {o['fallback_fields']:4.2f} | {o['spatial_exact']:6.2f} {o['spatial_adj']:7.2f} {o['spatial_mismatch']:7.2f} {o['p_grid_kappa']:5.2f} | {o['pattern_match']:5.2f} {o['pattern_differ']:5.2f} | {o['part_unknown']:5.2f} {o['cause_unknown']:6.2f} {o['note_ok']:6.2f} {o['polarity']:5.2f} {o['basis_viol']:6.2f} {o['compat_viol']:5.2f} {o['mean_sec']:5.1f}")
        print(f"{'':14s} eq_plausible={o['eq_plaus']} visual={o['visual']}")
    for c, v in summ.items():
        if c == 'template': continue
        print(f"\n  -- {c} by stratum (gridEx / gridAdj / patM / dig / fallback / cause? / polarity) --")
        for st, w in v['by_stratum'].items(): print(f"    {st:18s} n={w['n']:2d}  {w['spatial_exact']:.2f} {w['spatial_adj']:.2f} {w['pattern_match']:.2f} {w['dig']:.2f} {w['fallback']:.2f} {w['cause_unknown']:.2f} {w['polarity']:.2f}")

if __name__ == '__main__':
    main()
